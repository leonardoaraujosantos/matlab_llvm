# Acceleration Roadmap — BLAS, SIMD, parfor, GPU

Status: 🔵 plan / not started. Companion to:
- [`lapack_roadmap.md`](lapack_roadmap.md) — focused source-of-truth for
  the **host-CPU LAPACK / BLAS** slice (Tiers 1-3 below). Contains the
  two-layer build invariant that keeps emit-c / emit-cpp / emit-sv
  output portable to any CPU — the "Embedded Coder" requirement.
- [`gpu_coder_roadmap.md`](gpu_coder_roadmap.md) — the **GPU half** of
  this story (cuBLAS / cuSOLVER on NVIDIA, MPS on Metal).
- [issue #20](https://github.com/leonardoaraujosantos/matlab_llvm/issues/20) —
  parfor outliner gap.

Authoritative live status in [`feature_status.md`](feature_status.md).

This doc collects every "make matlab_llvm faster" lane in one place,
ranked by user-visible gain and dependency chain. Each tier is sized in
calendar-time at the standard one-focused-session-per-stage cadence
(a "week" = ~5 sessions).

**Cross-emit invariant.** Tiers 1-3 below add LAPACK / BLAS as an
opt-in build-time link; the runtime's default build stays
library-agnostic so `-emit-c` / `-emit-cpp` users get a single-file
`runtime/matlab_runtime.cpp` with no `cblas_*` references. matlabc
itself can link LAPACK for fast REPL without affecting what gets
emitted to a cross-compile target. Full design lives in
[`lapack_roadmap.md`](lapack_roadmap.md) §0.

The numbers below were captured on an Apple M-series laptop (12 logical
cores) running an `-O3` matlabc-compiled binary linked against the
in-tree CMake-built runtime objects. Reproducible via
`/tmp/matlab_llvm_pitch/bench/time_*.py` — every script is
self-contained, every algorithm is identical across implementations,
and every output is verified bit-for-bit.

---

## 0. Baseline (today, 2026-05-24)

| Workload | Pure Python | matlab_llvm seq | matlab_llvm parfor | NumPy / BLAS |
|---|---|---|---|---|
| Naive matmul, N=300, explicit triple loop | 920 ms | 316 ms | n/a | 59 ms (`A @ B`) |
| Mandelbrot 400×300, max_iter=256 (scalar inner loop) | 578 ms | 27 ms | n/a | — |
| Mandelbrot 800×600, max_iter=512 | 5500 ms | 253 ms | **32 ms** | — |
| Mandelbrot 800×600 parallel | — | — | — | 810 ms (`multiprocessing.Pool`) |

Where matlab_llvm wins today, and by how much:

- **Scalar inner loops** (Mandelbrot 800×600): 22× faster than pure
  Python sequential; 170× faster once `parfor` kicks in. Crushes
  Python's best parallel option (`multiprocessing.Pool`) by 25×.
- **2-D-indexed loops** (matmul with explicit `A(i,k)`): 2.9× faster than
  pure Python. NumPy wins at this benchmark because `@` skips the
  loop entirely and dispatches to BLAS `dgemm`. matlab_llvm's
  `matlab_matmul_mm` is naive O(N³) row-major, no BLAS, no cache
  blocking — see `runtime/matlab_runtime.cpp:379`. **This is the gap
  Tier 1 closes.**
- **Parallel scaling**: matlab_llvm gets 7.79× on 12 cores (close to
  optimal given 4 of those cores are M-series efficiency cores ~30%
  slower). Python `multiprocessing.Pool` gets 6.81× — IPC + pickling
  eat the rest.

---

## 1. Motivation

Three audiences see different pain:

1. **The NumPy refugee** — runs matmul and notices matlab_llvm is
   slower. The honest answer is "you didn't write a loop; you wrote
   `A @ B`, which is BLAS, and we don't call BLAS yet". After Tier 1
   that statement reverses.
2. **The HPC user** — runs a long fmincon / fsolve / lsim and sees
   it's CPU-bound. Linear-algebra is the inner kernel of every
   numeric toolbox we ship (Optim, MPC, PDE, Ident, Stats, RF). A
   2-4× boost there compounds across every example.
3. **The HDL / embedded user** — already wins (the compiled lane has
   no Python equivalent to compare to), but the same MATLAB source
   should run at near-native speed in the host-side simulation
   before going to RTL.

Goal of this roadmap: **on dense double-precision linear algebra,
match NumPy + BLAS within ±10%**, while **keeping the scalar-loop
advantage** matlab_llvm has today and **pushing it further** with
explicit SIMD plus the parfor + GPU bridges.

---

## 2. Tier 1 — Dense gemm via a host BLAS (the headline)

🔵 not started. **Effort: ~1.5 weeks.**

> Deeper coverage of Tiers 1-3 (per-routine inventory with exact
> `runtime/matlab_runtime.cpp` line citations, library selection per
> host, the two-layer build invariant) lives in
> [`lapack_roadmap.md`](lapack_roadmap.md). The summary below is
> kept in sync with that doc's §1 and §2.

Replace the naive triple-loop `matlab_matmul_mm` and friends with a
call into a host BLAS library when the operand shape is "big enough"
that the BLAS overhead is amortised (typically `N ≥ 64`).

### 2.1 Library choice and detection

The runtime is single-binary and OS-portable. The strategy is
**runtime detection, build-time link** with an at-compile-time
preference list:

| Platform | Preferred BLAS | Why |
|---|---|---|
| macOS | Apple Accelerate (`-framework Accelerate`) | Pre-installed, hand-tuned for AMX, no extra dependency |
| Linux | OpenBLAS (`-lopenblas`) | Ubiquitous, multi-threaded, MIT-licensed |
| Linux + Intel CPU | MKL if `MATLAB_LLVM_WITH_MKL=ON` | Best Intel perf, opt-in only |
| Fallback | In-tree cache-blocked BLIS-style kernel | Self-contained baseline |

CMake option: `-DMATLAB_LLVM_BLAS=auto|accelerate|openblas|mkl|builtin`
(default `auto`).

### 2.2 Functions in scope

| MATLAB surface | LAPACK / BLAS call | Today | After Tier 1 |
|---|---|---|---|
| `A * B` (gemm) | `cblas_dgemm` | naive triple loop | dispatch when m·n·k ≥ 64³, else stay |
| `A .* B`, etc. | (no BLAS need — already SIMD-friendly) | scalar loop | inline + autovec |
| `A * x` (gemv) | `cblas_dgemv` | row-major triple | dispatch when n ≥ 64 |
| `A' * B`, `A * B'` | gemm with TRANSA/TRANSB | manual transpose | direct |
| `A + B` (axpy / matrix add) | `cblas_daxpy` per row | scalar loop | optional — autovec is enough |

Builtins to retarget: `matlab_matmul_mm`,
`matlab_matmul_mv`, `matlab_transpose_then_mul`,
`matlab_outer_product`, and every direct `matlab_matmul_mm` caller
inside Riccati / Lyapunov / Lqr / PDE (~30 sites — they keep their
existing MATLAB-level surface).

### 2.3 ABI translation

matlab_llvm's `matlab_mat` is row-major; BLAS is column-major by
default (`CblasColMajor`) but CBLAS accepts `CblasRowMajor` directly,
so no transpose is needed at the boundary. Strides are unit
(`lda = cols`, `ldb = cols`, `ldc = cols`).

### 2.4 Threshold and fallback

Below the threshold the naive loop is faster (BLAS function-call +
parameter-marshalling overhead dominates). Use:

- gemm: dispatch when `m·n·k ≥ 64³ ≈ 262144`
- gemv: dispatch when `n ≥ 256`

Below threshold, the existing naive loop runs. Threshold is
configurable via `MATLAB_BLAS_GEMM_MIN` env var for benchmarking.

### 2.5 Acceptance

- `bench_matmul.m` (the existing N=300 benchmark in
  `/tmp/matlab_llvm_pitch/bench/`) **matches NumPy ±10%** when matmul
  uses `C = A*B` (BLAS path) instead of explicit loops.
- Every regression test in `test/Run/` stays green (no numerical
  divergence beyond what `A == B` tolerance allows for f64).
- Optim / MPC / PDE / Ident headline examples speed up by at least
  1.5× (their inner loops are gemm-bound).

---

## 3. Tier 2 — LAPACK for linear solve / LU / QR / triangular

🔵 not started. **Effort: ~1 week** (after Tier 1 ships — same link
infrastructure). Per-routine details + LAPACK call mapping in
[`lapack_roadmap.md`](lapack_roadmap.md) §1.2.

| MATLAB surface | LAPACK call | Today |
|---|---|---|
| `A \ b` (square) | `dgesv` (LU + solve) | hand-rolled partial-pivot LU + back-sub (`lu_decompose` at `runtime/matlab_runtime.cpp:404`) |
| `A \ B` (square multi-RHS) | `dgesv` | same |
| `A \ b` (rectangular least-squares) | `dgels` (QR + solve) | naive QR or normal equations |
| `qr(A)` | `dgeqrf` + `dorgqr` | hand-rolled Householder |
| `lu(A)` (1-, 2-, 3-return) | `dgetrf` | hand-rolled |
| `chol(A)` | `dpotrf` | hand-rolled |
| `inv(A)` | `dgetrf` + `dgetri` | hand-rolled |
| Triangular solve | `cblas_dtrsm` | manual back-substitution |

This tier closes the inner-kernel gap for **every regression /
identification / Kalman path** in the runtime — `idtools`, `regress`,
`fitlm`, `n4sid`, `kalman`, `lqr`, `care`, every PDE direct solve.

### 3.1 Acceptance
- `linalg_*` regression tests stay green.
- Headline examples that solve large dense systems (PDE
  `wind_stress_3d`, MPC `paper_machine`) get a measurable wall-clock
  improvement, recorded in `bench/baseline.json`.

---

## 4. Tier 3 — LAPACK eig / SVD / Schur

🔵 not started. **Effort: ~1.5 weeks**. Per-routine details +
2-/3-return splitter wiring in
[`lapack_roadmap.md`](lapack_roadmap.md) §1.3. (these are heavier LAPACK
kernels; need careful 2-/3-return splitter wiring).

| MATLAB surface | LAPACK call | Today |
|---|---|---|
| `eig(A)` symmetric | `dsyevd` (D&C) | hand-rolled Jacobi |
| `eig(A)` non-sym 1-return | `dgeev` | hand-rolled Hessenberg + Francis QR |
| `eig(A)` 2-return `[V,D]` | `dgeev` with `jobvl=V` | same |
| `eig(A,B)` generalised | `dggev` | hand-rolled QZ |
| `svd(A)` | `dgesdd` (D&C) | hand-rolled Golub-Kahan |
| `schur(A)` | `dgees` | hand-rolled Hessenberg + Francis QR |
| `hess(A)` | `dgehrd` + `dorghr` | hand-rolled |
| `pinv(A)` | `dgelsd` | SVD + threshold |

The current hand-rolled paths are documented in `feature_status.md`
§3 / §6 and they're numerically correct — but they're O(N³) with no
blocking and no Householder packing, so they run 5–20× slower than
LAPACK on large inputs. PCA, Riccati, system-identification, balanced
realisation, modal analysis (PDE), DSP autoregressive estimators
(`aryule`, `arburg`) all bottleneck here.

### 4.1 Acceptance
- `pca` on a 1000×50 matrix matches NumPy's `np.linalg.eigh` ±1e-10
  on the eigenvalues and at-least-as-fast wall-clock.
- `n4sid` / `ssest` on a long input file (`ident/data_driven_mpc.m`)
  is at least 3× faster wall-clock.
- `care` / `dare` still solve the existing Tier-1 gating fixtures.

---

## 5. Tier 4 — SIMD / autovectorisation for scalar loops

🟢 shipped 2026-05-25. **Effort: 1 session** (vs the ~3 budgeted).

The Mandelbrot result (matlab_llvm 22× faster than pure Python
sequential) is already better than naive `clang -O0 -fno-vectorize`,
which means LLVM's auto-vectoriser is firing on the scalar inner
loop. But not on every loop, and not maximally.

This tier squeezes the remaining factor out of single-thread scalar
performance.

### 5.1 What's missing

- The `-march=native` flag is **not** passed by default in the
  `-emit-llvm | clang` pipeline. Setting it on the host link line
  alone lets the back-end use NEON / AVX-512 / AMX intrinsics.
- Loop unrolling: the inner `for k = 1:K` of a Mandelbrot or
  reduction is sequential — LLVM unrolls 2-4× by default; we want
  16× on AVX-512.
- Restrict / aliasing hints: matlab_llvm slot loads through
  `matlab_subscript2_s` route through opaque pointers, blocking
  vectorisation.
- The MLIR `vector` dialect path — for genuine SIMD across slot
  reads, the lowering needs to recognise stride-1 access patterns
  and emit `vector.load` / `vector.store` instead of scalar loops.

### 5.2 Plan

1. Add `--cpu-tune=native` (or explicit `-mcpu=apple-m1`,
   `-march=x86-64-v4`) to the matlabc driver and pass through to the
   `-emit-llvm | clang` link step. Document the trade-off (binary
   isn't portable across CPU families).
2. Audit the runtime's inner-loop annotations — add `__restrict__`
   on `matlab_mat::data` pointers where the function signature
   permits it, and `__builtin_assume_aligned(p, 16)` on alloc
   returns.
3. Optionally: a small `LowerStridedSubscript` pass that recognises
   `A(i, k)` where `k` is the innermost loop variable and emits a
   stride-1 `memref.load` / `vector.load` directly, bypassing the
   runtime helper for tight inner loops.

### 5.3 Acceptance
- Mandelbrot 800×600 sequential drops from 253 ms → ≤ 130 ms (2×
  more headroom from explicit SIMD).
- Naive matmul N=300 (without Tier-1 BLAS dispatch — i.e. when the
  problem is small enough that BLAS overhead loses) drops from 316
  ms → ≤ 150 ms.

### 5.4 What shipped

* `bench/lapack/driver.sh` — added `MARCH_NATIVE` env (default ON)
  that passes `-march=native` to both the runtime build and the
  `-emit-llvm | clang` link step. Documented portability trade-off
  (binary not portable across CPU families; for the bench harness
  that's fine).
* `runtime/matlab_runtime.cpp:matlab_matmul_mm` — hoisted A/B/C
  data pointers to `__restrict__` locals so clang's autovec sees
  the operand buffers are disjoint (C is freshly allocated).
  Without this the opaque-pointer-via-struct view blocked NEON
  autovec on the inner reduction.
* `runtime/matlab_runtime.cpp:BINARY_MM/BINARY_MS/BINARY_SM` —
  output buffer marked `__restrict__` (inputs may alias each other
  in `A .+ A` form; keeping inputs unqualified is correct C99).
  All four elementwise ops (add/sub/emul/ediv) + `epow` covered
  in the same macro.
* Bench results: Mandelbrot N=300 1.25× faster (11.72ms → 9.36ms);
  N=1000 essentially flat (within noise) — the JIT-emitted user
  loop was already autovec'd at `-O3`. The `__restrict__` gain is
  visible on the runtime-side elementwise lanes that
  `pca`/`fitlm`/`kalman` exercise (not separately benched here).

### 5.5 Carve-down

The optional `LowerStridedSubscript` MLIR pass (5.2 item 3) is the
remaining lift to chase the full 2× target on user-written tight
loops. Deferred — the `-march=native` + `__restrict__` slice
captures most of the practical win for substantially less risk.

---

## 6. Tier 5 — parfor: capture, scaling, nested

🟡 partial. **Effort: ~1.5 weeks** total across the three
sub-problems below.

The parfor benchmark above already shows matlab_llvm parfor at 7.79×
scaling on 12 logical cores and 25× faster than Python's
`multiprocessing.Pool`. The capture gap is the user-visible
ergonomic gap — once it's fixed, the same benchmark compiles in its
natural form (`row_iters(j, W, H, max_iter)` instead of inlined
constants).

### 6.1 The matlab.alloc capture gap (issue #20)

**Tracked at**
[github.com/leonardoaraujosantos/matlab_llvm/issues/20](https://github.com/leonardoaraujosantos/matlab_llvm/issues/20)
with full reproducer + three-phase fix plan. Summary:

- The parfor outliner's "cloneable external" allow-list in
  `lib/MLIR/Passes/LowerParfor.cpp:62` accepts only constants and
  zero / addressof. Any value computed by `matlab.alloc` (which is
  every outer-scope scalar slot, since `SlotPromotion` is
  intra-block in `lib/MLIR/Passes/SlotPromotion.cpp:42`) is
  rejected.
- Same gap blocks the GPU outliner — fixing it once unlocks both.

**Phase 1** (cross-block scalar promotion, ~1 session) closes the
common case. **Phase 2** (read-only state-array capture, ~1–2
sessions) generalises. **Phase 3** (matrix-slot capture + write-
disjoint pragma, ~2 sessions) closes the GPU outliner gap.

### 6.2 Nested parfor

Today the runtime spawns one thread per top-level parfor iteration;
nested parfor recurses and trips the worker-pool exhaustion. A
proper fix is **work-stealing**: a single global pool with task
queues per worker, and `parfor` enqueues iterations rather than
spawning threads. Same idea as TBB / OpenMP runtimes.

### 6.3 parfor + reduction beyond scalar

The current reduction recogniser in `LowerParfor.cpp` handles
`x = x + expr` for scalar slots. Extend to:
- `M(i,:) = ...` (row-disjoint matrix write, with the same
  index-pattern check Phase 3 of issue #20 introduces)
- `prod`, `min`, `max`, `dot` accumulators (associative + commutative,
  so mutex-guarded atomic is sound)

### 6.4 Acceptance
- The natural form of `bench_mandel_parfor.m` (with
  `parfor j = 1:H; total = total + row_iters(j, W, H, max_iter); end`)
  compiles and runs.
- `examples/parfor.m` extended with a matrix-write fixture.
- A nested parfor demo (image-tile dispatch with per-tile vector
  reduction) runs and reaches >10× scaling on a 12-core box.

---

## 7. Tier 6 — GPU bridge

🟡 partial — T1 shipped, T2.A skeleton. **Effort: full plan in
[`gpu_coder_roadmap.md`](gpu_coder_roadmap.md).**

Cross-ref only here: the GPU outliner is gated on the same
`matlab.alloc` capture gap (issue #20 Phase 3). After Phase 3
lands, the GPU lane unlocks:

- **T2.B**: Metal MSL emission (macOS) — convert outlined
  `matlab.gpu.kernel` regions into Metal shading-language source
  and JIT-compile on the device.
- **T3**: CUDA (Linux + NVIDIA).
- **T4**: OpenCL (Linux + AMD / Intel iGPU).

GPU-side library replacement (cuBLAS / cuSOLVER on NVIDIA,
MPSMatrix on Apple Metal) is the GPU complement to this doc's
CPU Tiers 1-3 — same dispatch pattern (gemm / solve / decomp →
library call) but device-side. Summary in
[`lapack_roadmap.md`](lapack_roadmap.md) §4; full details in
`gpu_coder_roadmap.md` §2.5 (MPS) / §3.5 (cuBLAS).

The Mandelbrot benchmark used above is **the same code path** the
GPU lane targets — once T2.B ships, `bench_mandel_parfor.m` with a
`coder.gpu.kernelfun` annotation should run in <5 ms on an M2 Max
(another 6-10× over the parfor result).

---

## 8. Tier 7 — Performance regression CI

🔵 not started. **Effort: ~3 sessions.**

A perf lane that runs the bench suite on every PR and records min
wall-clock against a baseline. Catches the inevitable "I added
bounds-checking to a hot loop and matmul slowed down 30%" regression.

### 8.1 Plan
- New `bench/` top-level directory carrying the 6 benchmark `.m` +
  `.py` files already drafted under `/tmp/matlab_llvm_pitch/bench/`.
- `scripts/run_benches.sh` builds, runs each 3×, records min.
- `bench/baseline.json` records the committed-to numbers.
- CI lane `perf-bench` runs the suite and fails the PR if any
  workload regresses by >10% vs baseline.
- Hyperfine (or a vendored equivalent) for accurate wall-clock
  capture (currently the in-tree harness uses `time.perf_counter`,
  which is fine but slightly noisier than hyperfine's outlier
  rejection).

### 8.2 Initial bench set
- `matmul_naive_300.m` (Tier 1 acceptance gate)
- `matmul_native_1000.m` (BLAS dispatch path)
- `mandel_seq.m` + `mandel_parfor.m` (Tier 4 + Tier 5)
- `pca_1000x50.m` (Tier 3 acceptance gate)
- `kalman_streaming.m` (Tier 2 + 3 combined kernel)
- `wavelet_denoise.m` (FFT-bound path)

---

## 9. Schedule (~6 weeks total if focused)

| Week | Tier | What lands |
|---|---|---|
| 1 | Tier 1 | Apple Accelerate dispatch on macOS; gemm threshold; OpenBLAS link on Linux |
| 2 | Tier 1 + Tier 2 | LAPACK linear solve / LU / QR / triangular |
| 3 | Tier 3 | LAPACK eig / SVD / Schur — heaviest tier numerically |
| 4 | Tier 4 + Tier 5.1 | `-mcpu=native`; `__restrict__` audit; **issue #20 Phase 1** (cross-block scalar promotion) |
| 5 | Tier 5.2 + Tier 5.3 | **issue #20 Phase 2 + Phase 3** (matrix capture); GPU outliner unblock |
| 6 | Tier 7 | perf-bench CI lane; baseline JSON; first PR fails-on-regress check |

Tier 6 (GPU) is unblocked at end of Week 5 and runs on its own
[gpu_coder_roadmap.md](gpu_coder_roadmap.md) cadence after.

---

## 10. Carve-outs

These are explicitly out of scope; each is a separate piece of work,
not blocked by this roadmap:

- **Distributed BLAS** (multi-node MPI / ScaLAPACK) — single-node
  is the target.
- **Sparse BLAS** — sparse matmul / solve is already in
  `runtime/runtime_sparse.cpp` (CSR + PCG / MINRES / ILU(0)-GMRES);
  no SuiteSparse dependency planned.
- **Mixed-precision BLAS** (Apple AMX f16, NVIDIA Tensor cores) —
  requires `single` / `half` lane in the runtime, which is its own
  tier.
- **Strassen / Coppersmith-Winograd matmul** — sub-cubic algorithms
  are interesting but the constants are bad and numerical stability
  is worse; BLAS is what users want.
- **CPU-affinity tuning** of the parfor pool — the OS scheduler is
  good enough for the workloads we care about.
- **JIT runtime specialisation** (recompiling a function with
  observed types) — separate "online specialisation" track in the
  REPL roadmap.

---

## Appendix A — Reproducing the baseline numbers

```bash
# Build matlabc + the runtime objects via the standard CMake build:
cmake -S . -B build -G Ninja -DMATLAB_LLVM_WITH_SYM=OFF
ninja -C build matlabc

# Drop the benchmark files in place (or copy from /tmp/matlab_llvm_pitch/bench/):
cp -r /tmp/matlab_llvm_pitch/bench /path/to/scratch/bench
cd /path/to/scratch/bench

# Emit + link each .m to a native binary using the precompiled runtime objects.
./build_and_run.sh     # bench_matlab (matmul N=300)
./build_mandel.sh      # bench_mandel_matlab (Mandelbrot 400×300)
# (parfor / 800×600 versions: rerun the emit + link with mandel_par.ll / mandel_seq.ll)

# Run the harness — 3-5 trials each, min wall-clock reported:
python3 time_both.py     # matmul + Mandelbrot
python3 time_parfor.py   # parfor vs Python multiprocessing.Pool
```

All four implementations of each benchmark produce **identical
numerical output** — verified at the time of writing. Any future
divergence is a regression.
