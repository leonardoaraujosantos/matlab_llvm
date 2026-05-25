# LAPACK Acceleration Roadmap — CPU + GPU complement

Status: 🔵 plan / not started. Companion to
[`acceleration_roadmap.md`](acceleration_roadmap.md) (the broader
gemm + SIMD + parfor story) and
[`gpu_coder_roadmap.md`](gpu_coder_roadmap.md) (the GPU half —
cuBLAS / cuSOLVER on NVIDIA, MPS on Metal). This doc is the focused
source-of-truth for the **host-CPU dense linear-algebra acceleration**
slice: which `runtime/matlab_runtime.cpp` entry points dispatch to
LAPACK / BLAS, what the dispatch looks like, and — most importantly —
**how the two-layer build keeps emit-c / emit-cpp / emit-systemverilog
output portable to any CPU** (the "Embedded Coder" invariant).

The dependency is opt-in. By default `runtime/matlab_runtime.cpp` is
exactly what it is today — one self-contained C++ TU, no
BLAS / LAPACK includes, no external symbol references (the comment at
`runtime/matlab_runtime.cpp:359` is enforceable on every emit lane).

---

## 0. The two-layer build invariant

The non-negotiable design constraint: enabling LAPACK on the
developer's matlabc build **must not leak into the output the developer
ships to embedded / cross-compile targets**. Two independent build
decisions:

| Layer | Knob | What it affects |
|---|---|---|
| **matlabc itself** (the compiler binary) | `-DMATLAB_LLVM_BLAS=auto\|accelerate\|openblas\|mkl\|none` (default `auto`) | REPL + the AOT executor's own matrix workload; any Sema-time numeric folding the compiler does internally |
| **user's compiled output** (runtime + emitted code shipped to a target) | `-DMATLAB_LLVM_WITH_BLAS=ON\|OFF` (default **OFF**) | Whether the user's binary calls LAPACK or stays in the naive `O(N³)` fallback |

These layers don't talk to each other. The matlabc binary on a dev
machine can be linked against Apple Accelerate for fast REPL while
emitting C source that cross-compiles to a Cortex-M7 against the
naive-only runtime build. The emitted source is **bit-identical** in
both cases; only the runtime's compiled object changes.

### 0.1 Why this works architecturally

The emit-c / emit-cpp pipeline emits calls into the public
`matlab_*` ABI (`matlab_matmul_mm`, `matlab_solve`, `matlab_svd`, …).
Those public entries are defined in `runtime/matlab_runtime.cpp` and
internally choose naive-or-LAPACK based on the **runtime's** compile
flags. matlabc's MLIR → C translation never inserts `cblas_*`
references; it only inserts the public ABI symbol.

Even when matlabc constant-folds (e.g. a compile-time-known
100×100 `A*B` whose operands are literals), the EMITTED C contains the
*result* as data literals, not the LAPACK call that produced it.

### 0.2 Implementation shape

Two options, both preserve the invariant:

* **(a) `#ifdef`-guarded inline dispatch** — the LAPACK calls live in
  `runtime/matlab_runtime.cpp` wrapped in
  `#ifdef MATLAB_LLVM_WITH_BLAS`. When the macro is off the
  preprocessor strips them; the resulting object references no
  `cblas_*` symbol.
* **(b) Separate file with weak symbols** — `runtime/runtime_blas.cpp`
  hosts the LAPACK shim; `matlab_runtime.cpp` calls weak forward
  decls. The library-agnostic build simply doesn't link
  `runtime_blas.cpp`. The "single-file runtime" reading of
  `matlab_runtime.cpp` stays cleaner.

Default to (b) — keeps the central file unchanged and gives the user
a one-line "drop this file from the build" escape for cross-compile.

### 0.3 Verification

CI lane `test/Examples/run_sweep.sh` already builds the runtime
without LAPACK and links every example against it. Adding a second
sweep with `-DMATLAB_LLVM_WITH_BLAS=ON` lets us diff outputs
bit-for-bit (numerics) and check no new symbol dependencies on the
OFF path (`nm runtime/matlab_runtime.o | grep cblas_` returns empty).

---

## 1. What gets accelerated

### 1.1 Tier 1 — BLAS dense (headline gains)

`runtime/matlab_runtime.cpp` lines 379–542. The current naive
implementations are documented as "fine for teaching-scale inputs"
(line 363). LAPACK / BLAS dispatch closes the gap to NumPy + MATLAB.

| MATLAB | Runtime entry | BLAS call | Threshold | Today |
|---|---|---|---|---|
| `A * B` | `matlab_matmul_mm` (379) | `cblas_dgemm` | `m·n·k ≥ 64³` | naive triple loop |
| `A * x` | `matlab_matmul_mv` | `cblas_dgemv` | `n ≥ 256` | row-major triple loop |
| `A' * B` / `A * B'` / `A' * B'` | manual transpose + matmul | gemm with TRANSA/TRANSB | as gemm | manual transpose copy + gemm |
| `A + B` (matrix add) | `matlab_add_mm` | `cblas_daxpy` per row | autovec wins | scalar loop (already vectorisable) |
| Outer product `x * y'` | `matlab_outer_product` | `cblas_dger` | `n·m ≥ 256` | scalar loop |

Below the threshold the naive loop runs (the BLAS call/parameter
overhead outweighs the SIMD gain). Configurable via
`MATLAB_BLAS_GEMM_MIN` env var for benchmarking.

### 1.2 Tier 2 — LAPACK solve / factorization

`runtime/matlab_runtime.cpp` lines 468–542 + 13020–13191. Closes the
inner-kernel for every regression / Kalman / Riccati / direct-solve
path.

| MATLAB | Runtime entry | LAPACK call | Today |
|---|---|---|---|
| `A \ b`, `A \ B` (square) | `matlab_mldivide_mm` (488), `matlab_inv` (468) | `dgesv` | partial-pivot LU + back-sub (`lu_decompose` at 404) |
| `A \ b` (rectangular least-sq) | hand-rolled normal-eq path | `dgels` / `dgelsd` | normal equations |
| `lu(A)` 1/2/3-return | `matlab_lu_L`/`_U` (13126/13139) | `dgetrf` | hand-rolled |
| `qr(A)` 1/2-return | `matlab_qr_Q`/`_R` (13180/13191) | `dgeqrf` + `dorgqr` | hand-rolled Householder |
| `chol(A)` | `matlab_chol` (13020) | `dpotrf` | hand-rolled |
| `inv(A)` | `matlab_inv` (468) | `dgetrf` + `dgetri` | hand-rolled |
| Triangular solve | manual back-sub | `cblas_dtrsm` | manual |
| `B / A` (rdivide) | `matlab_mrdivide_mm` (509) | transpose-rule + `dgesv` | transpose + LU solve |

### 1.3 Tier 3 — LAPACK eig / SVD / Schur

`runtime/matlab_runtime.cpp` lines 543–3098 + 13240–13340. The
hand-rolled paths are numerically correct (and gated by
`linalg_eig_*` / `linalg_svd_*` regression tests) but they're
unblocked O(N³) without Householder packing or cache blocking —
5–20× slower than LAPACK for `N ≥ 200`.

| MATLAB | Runtime entry | LAPACK call | Today |
|---|---|---|---|
| `eig(A)` symmetric | `matlab_eig` (865) symmetric path | `dsyevd` (divide & conquer) | Jacobi |
| `eig(A)` non-sym 1-return | `matlab_eig` (865) non-sym path | `dgeev` | Hessenberg + Francis QR |
| `eig(A)` 2-return `[V,D]` | `matlab_eig_V`/`_D` (1115/1148) | `dgeev` with `jobvl=V` | same hand-rolled |
| `eig(A,B)` generalised | `matlab_eig_gen` (2990) | `dggev` | hand-rolled QZ |
| `svd(A)` | `matlab_svd` (543) | `dgesdd` | hand-rolled Golub-Kahan |
| `schur(A)` | `matlab_schur*` (2782/2793/2795) | `dgees` | hand-rolled Hessenberg + Francis QR |
| `hess(A)` | `matlab_hess*` (2679/2742/2744) | `dgehrd` + `dorghr` | hand-rolled |
| `pinv(A)` | `matlab_pinv` (13056) | `dgelsd` | SVD + threshold |
| `null(A)` / `orth(A)` | `matlab_null` (13240), `matlab_orth` (13273) | via `dgesdd` | hand-rolled |
| `rank(A)` | `matlab_rank` | via `dgesdd` (count singular values) | hand-rolled |
| `cond(A)` | `matlab_cond` | via `dgesdd` (σ₁/σₙ) | hand-rolled |

### 1.4 Stays hand-coded (LAPACK doesn't ship these directly)

These benefit indirectly because their inner gemm / Schur calls now
hit Tier 1/3, but the high-level algorithm stays in `matlab_runtime.cpp`:

* `expm(A)` — Padé + scaling-and-squaring (`matlab_expm` line 2394).
  Uses gemm internally → Tier 1 win.
* `lyap(A,Q)` / `sylvester(A,B,C)` (lines 3099 / 3148) —
  Bartels-Stewart. Schur step → Tier 3 win; inner triangular Sylvester
  could call `dtrsyl` for a smaller incremental gain.
* `care(A,B,Q,R)` / `dare` / `_5` variants (3242 / 3342 / 3403) —
  Riccati. SLICOT has `SB02OD`; LAPACK alone needs the Hamiltonian
  Schur factorisation (uses Tier 3).
* Anything in `runtime_sparse.cpp` — SuiteSparse if at all, not LAPACK.

---

## 2. Library selection per host

The runtime is single-binary and OS-portable. Strategy:
**runtime detection, build-time link**, with a compile-time preference
list per host.

| Host | Preferred BLAS / LAPACK | Why |
|---|---|---|
| macOS (Apple Silicon + Intel) | Apple Accelerate (`-framework Accelerate`) | Pre-installed, hand-tuned for AMX matrix coprocessor on Apple Silicon, no extra dependency |
| Linux | OpenBLAS (`-lopenblas -llapack`) | Ubiquitous, multi-threaded, MIT-licensed, ARM + x86 |
| Linux + Intel CPU (opt-in) | MKL (`-DMATLAB_LLVM_WITH_MKL=ON`) | Best Intel perf, AVX-512 tuned |
| Windows | OpenBLAS via vcpkg / MKL | Same shape as Linux |
| Cross-compile / embedded | **none** (default) | Single-file runtime, no link deps |
| Fallback when the preferred lib is absent | In-tree cache-blocked BLIS-style kernel | Mid-tier perf, no system deps |

CMake top-level: `-DMATLAB_LLVM_BLAS=auto|accelerate|openblas|mkl|builtin|none`
(default `auto` — picks the preferred lib for the host, falls back to
`builtin` if none detected).

The matlabc CMake build defaults to `auto` so REPL gets the speedup
out of the box. The user's redistributable runtime build defaults to
`none`.

---

## 3. Cross-target portability matrix

The two-layer split delivers this:

| User scenario | matlabc build | runtime build | Output / behaviour |
|---|---|---|---|
| Power-user REPL on M2 | `BLAS=auto` (→ Accelerate) | (built into matlabc) | Accelerate-fast interactive `>>` |
| Standalone AOT for Linux server | `BLAS=openblas` | `WITH_BLAS=ON` (→ openblas) | OpenBLAS-fast `./a.out` |
| Embedded Coder–style `.c` for Cortex-M | (any) | OFF | Single-file `runtime.cpp` + emitted `.c`, no deps, naive O(N³) |
| HDL synthesis | (any) | (not linked) | SystemVerilog, no runtime at all |
| User wants the source, picks performance later | `BLAS=auto` | OFF | Clean emit-c, ships anywhere; user opts in to BLAS for their build |
| User on a hermetic build farm | `BLAS=none` | OFF | Bit-reproducible runs across hosts, no LAPACK variance |

Last row matters for some users (numerical reproducibility,
regulatory contexts): LAPACK implementations differ in their last
ULPs across vendors. The `none` mode gives a single source of truth.

---

## 4. GPU complement — cuBLAS / cuSOLVER / MPS

Already covered in [`gpu_coder_roadmap.md`](gpu_coder_roadmap.md);
duplicating only the essentials here so the LAPACK story is
self-contained.

The GPU side answers a different question — "what if I have a
gpuArray, where does the matmul go?" — using the same library-call
pattern but against the device-side libraries:

| MATLAB | Apple Metal (`-emit-metal`) | NVIDIA CUDA (`-emit-cuda`) | Today |
|---|---|---|---|
| `gpuArray(A) * gpuArray(B)` | `MPSMatrixMultiplication` | `cublasDgemm` | host fallback (Tier 1 on CPU) |
| `gpuArray(A) \ gpuArray(B)` | `MPSMatrixDecompositionLU` + `MPSMatrixSolveLU` | `cusolverDnXgesv` | host fallback (Tier 2) |
| `inv(gpuArray(A))` | `MPSMatrixInverse` | `cusolverDnXgetri` | host fallback |
| `svd(gpuArray(A))` | (no direct MPS — Metal MLX has `linalg.svd`) | `cusolverDnXgesvd` | host fallback (Tier 3) |
| `qr(gpuArray(A))` | (Metal MLX `linalg.qr`) | `cusolverDnXgeqrf` | host fallback |
| `chol(gpuArray(A))` | `MPSMatrixDecompositionCholesky` | `cusolverDnXpotrf` | host fallback |
| `eig(gpuArray(A))` symmetric | `MPSMatrixDecompositionSymmetric` | `cusolverDnXsyevd` | host fallback |
| `fft(gpuArray(x))` | `MPSGraph` FFT node | `cufftPlan` + `cufftExecD2Z` | host fallback (`matlab_fft_c`) |

GPU-side build flags (mirroring the gpu_coder_roadmap.md proposal):

* macOS: `-DMATLAB_LLVM_WITH_METAL=ON` — links `-framework Metal
  -framework MetalPerformanceShaders -framework Foundation`. Default
  on when building on macOS.
* Linux + NVIDIA: `-DMATLAB_LLVM_WITH_CUDA=ON` — links
  `-lcublas -lcusolver -lcufft`. Default off; opt-in.
* The library-replacement dispatch lives inside the GPU outliner
  (`coder.gpu.kernelfun` body lowering) — when both operands are
  `gpuArray` and shape clears the threshold, emit the
  library-replacement op instead of an MSL/CUDA kernel.

**Cross-emit invariant for GPU lanes**: the emit-c lane never emits
`cublasDgemm` either. GPU library replacement only fires for
`-emit-cuda` / `-emit-metal` outputs, which the user explicitly
opted into. Emit-c users always get the host `matlab_*` ABI.

### 4.1 Apple Accelerate vs. Metal — two different things

The user's question conflates them slightly; for the record:

* **Apple Accelerate framework** (`<Accelerate/Accelerate.h>`) — the
  **CPU** library, BLAS + LAPACK + vDSP + vImage. Lives on every Mac.
  Uses the AMX matrix coprocessor on Apple Silicon. **Covers Tiers
  1-3 of this roadmap on macOS.**
* **Metal Performance Shaders** (`<MetalPerformanceShaders/MPS.h>`) —
  the **GPU** library, on top of Metal. Covers the GPU complement
  story above. Lives in `gpu_coder_roadmap.md` §2.5.

Both ship in macOS by default; both are "free" dependencies on Apple
hardware.

---

## 5. Phasing

The acceleration_roadmap.md sizing carries over verbatim — same
tiering, same effort estimates, same acceptance.

### 5.1 Phase 1 — BLAS gemm (Tier 1) — ~1.5 weeks

Touches `matlab_matmul_mm` + ~5 sibling entries. Headline gain on
every numeric example.

**Acceptance**: `bench_matmul.m` (N=300) matches NumPy ±10% when the
`A*B` form (BLAS path) is used. `test/Run/` lane stays 535/535. Every
emit-c / emit-cpp / emit-systemverilog / emit-cocotb fixture
unchanged. `nm runtime/matlab_runtime.o | grep cblas_` empty on the
`WITH_BLAS=OFF` build.

### 5.2 Phase 2 — LAPACK solve / LU / QR / chol (Tier 2) — ~1 week

After Phase 1 — reuses the same link infrastructure. Closes the
regression / Kalman / Riccati inner kernel.

**Acceptance**: `linalg_*` stays green. PDE `wind_stress_3d.m` and
MPC `paper_machine.m` get measurable wall-clock improvements
recorded in `bench/baseline.json`. Embedded-target invariant
verified the same way.

### 5.3 Phase 3 — LAPACK eig / SVD / Schur (Tier 3) — ~1.5 weeks

Heavier kernels; needs careful 2-/3-return splitter wiring at the
`matlab_eig_V` / `matlab_svd_U` / etc. ABI surface.

**Acceptance**: `pca` on 1000×50 matches `np.linalg.eigh` to ±1e-10;
`n4sid` on `ident/data_driven_mpc.m` runs ≥3× faster.

### 5.4 Phase 4 — GPU library replacement — see `gpu_coder_roadmap.md`

Sized at ~2 weeks per platform in that doc. Builds on Phases 1-3
(host fallback when GPU library is below threshold or not built in).

---

## 6. Open questions

* **Static vs. dynamic linking on Linux.** OpenBLAS is most often
  installed dynamic. Static-link variants exist but the .a is
  ~30 MB. Default to dynamic on Linux distros; offer
  `-DMATLAB_LLVM_BLAS_STATIC=ON` for sealed-image builds.
* **Reproducibility across libraries.** Different LAPACK
  implementations agree to ~10 ULP on most routines but can diverge
  on near-singular inputs. Test policy: regression goldens that hash
  outputs use tolerance comparisons, not bit-equality, for the
  WITH_BLAS lane. The default OFF lane stays bit-exact.
* **Single vs. double precision.** All entries above are `double`.
  `single` matrix support exists in `matlab_runtime.cpp` for a few
  routines but is sparse. cuBLAS / MPS / Accelerate all ship `sgemm`
  / `sgesv`; the dispatch table doubles in size if we extend.
  Recommendation: ship double-only in Phases 1-3, add single in a
  follow-up once user demand surfaces.
* **`complex` matrices.** `runtime_complex.cpp` has separate entries
  (`matlab_matmul_mm_c`, `matlab_eig_c`, …). LAPACK ships `zgemm` /
  `zgesv` / `zgeev`. Add a parallel dispatch table in Phase 2; the
  matrix runtime already plumbs complex via `matlab_mat_c`.
* **Threading model.** OpenBLAS / MKL / Accelerate all spawn their
  own thread pools. parfor outlining (issue #20) spawns ours. The
  two pools interact poorly under contention. Recommendation: set
  `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` /
  `VECLIB_MAXIMUM_THREADS=1` when matlabc detects parfor in the
  program, and document it.

---

## 7. Related

* [`acceleration_roadmap.md`](acceleration_roadmap.md) — broader
  acceleration story (Tiers 1-3 here are sourced from there;
  Tier 4 SIMD autovec + Tier 5 parfor live there).
* [`gpu_coder_roadmap.md`](gpu_coder_roadmap.md) — the GPU half
  (cuBLAS, cuSOLVER, MPS).
* [`runtime.md`](runtime.md) — the runtime's ABI shape that the
  LAPACK shim must preserve.
* [`feature_status.md`](feature_status.md) — live status of the
  linalg surface.
* `runtime/matlab_runtime.cpp:359` — the "no BLAS / LAPACK"
  comment block that this roadmap relaxes (carefully).
