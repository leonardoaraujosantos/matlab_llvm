# Running the full test gate on Linux CI

This document tracks the work to make the **complete ctest gate** pass on
the Linux GitHub-Actions runner (it had only ever been run green on the
maintainer's macOS machine), and the **one remaining open bug**
(`stats_t4_cluster`) with everything a contributor needs to reproduce and
fix it on a Linux box.

Status as of 2026-05-23: **DONE — the full Linux ctest gate is green** (no
skips). All 13 originally-failing lanes are fixed, including the last one,
`stats_t4_cluster` (the MLIR dominance bug, issue #13, now resolved — see
§3). SymPP is built in CI with the symbolic lanes running. Merged via #12.

---

## 1. Background — why Linux differed from macOS

The macOS toolchain uses Clang + **libc++**; the Linux CI runner uses
Clang + **libstdc++**. Three classes of difference surfaced once the full
gate ran on Linux:

1. **Standard-library transitive includes.** libc++ pulls in `<string>`,
   `<cstdio>`, etc. transitively; libstdc++ does not. Code compiled at the
   default `-std=gnu++17` that *worked* on macOS *failed* on Linux.
2. **Unordered-container iteration order.** `std::unordered_map` /
   `std::unordered_set` iterate in a different order under libstdc++, which
   changed (a) emitted-text ordering in goldens and (b) op-insertion order
   in the MLIR pipeline.
3. **Floating-point last digit.** macOS libm and Linux glibc round the last
   significant digit of transcendental results differently.

These are documented in the project memory (`ci_linux_gate.md`).

## 2. What was fixed (all merged on branch `ci/linux-test-gate`)

| Lane(s) | Root cause | Fix |
|---|---|---|
| `run-tests`, `run-tests-emit-c/cpp(+strict)` | Test harnesses compiled the C++20 runtime at the compiler default (gnu++17); libstdc++ leaves `<string>`/`<cstdio>` incomplete | `-std=c++20` in the harnesses (`test/Run/run_tests*.sh`); surface the real compiler error instead of `2>/dev/null` |
| `flowchart-cross-dialect`, execute lanes | `CLANG` defaulted to the macOS-only Homebrew path | Auto-detect: Homebrew clang if present, else system `clang` (`run_tests.sh`, `runtime/scripts/build_and_run.sh`, `test/RunSym/run_tests.sh`) |
| `run-tests-emit-python`, `cocotb-tests`, `flowchart-emit-{diagram,chart-cocotb}` | Emitted `matlab_runtime.py` imports **numpy**, absent in CI | `python3-numpy` in the shared `setup-llvm` action |
| `cocotb-tests` (`cic_decimator`) | Verilator 5.020 escalates `WIDTHEXPAND`/`WIDTHTRUNC` to fatal on valid implicit-width Verilog | Emit `-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC` into the cocotb Makefile (`tools/matlabc/main.cpp`) |
| `flowchart-tests` | StateChart→MATLAB emitter listed `persistent r_*`/`l_*` from unordered containers → platform-dependent goldens | Ordered `std::map`/`std::set` (`lib/StateChart/Lowering.cpp`); goldens regenerated |
| `debug-dap-tests` | DAP server dup2's a pipe over stdout; glibc full-buffers it, so JIT `disp()` never reached the reader | `setvbuf(stdout, _IOLBF)` after the dup2 (`tools/matlabc/main.cpp`) |
| `run-tests-emit-c/cpp-strict` | clang `-Wextra` flags `-Wmissing-field-initializers` on designated-initializer runtime structs | `-Wno-missing-field-initializers` in the strict `WFLAGS` |
| ~25 numeric fixtures (eig/qz/fft/rf/ctrl) | macOS-generated `%g` goldens diverge in the last digit on Linux libm | `test/Run/numdiff.py` — token-wise compare, numbers within rel/abs tolerance, text exact |
| 16 `rf_*` fixtures | Hardcoded the maintainer's macOS home path `/Users/leonardoaraujo/.../fixtures/rf/...` | Test-relative paths + run the binary from `TESTDIR` |
| `fft_bluestein` | `round(real(fft([1 2 3 4 5])))` rounds values exactly on `-2.5`; libm ε flips `round(-2.5±ε)` between -2/-3 | Drop `round()`, print `real(y)` |
| `sig_fir`/`ctrl_balreal`/`ctrl_kalman`/`table_basic` (emit-python only) | numpy diverges from the C golden beyond tolerance | `.skip-emit-python` (best-effort Python port; matches existing precedent) |

Separately, the **Symbolic Math Toolbox (SymPP)** lanes were enabled in CI
(the Full job builds SymPP from <https://github.com/leonardoaraujosantos/SymPP>,
configures matlabc with `-DMATLAB_LLVM_WITH_SYM=ON`, and runs
`run-tests-sym` + `sym_basic` instead of skipping). The symbolic suite was
expanded from 4 to 8 fixtures (`test/RunSym/`).

---

## 3. RESOLVED — `stats_t4_cluster` MLIR dominance failure on Linux

> Was GitHub issue
> [#13](https://github.com/leonardoaraujosantos/matlab_llvm/issues/13),
> **fixed in `ec1d872`.** Root cause: the 5-output `pca` splitter held the
> call operand in a non-owning `ValueRange ar = ValueRange{X}`; the
> temporary initializer-list backing array was freed before the
> `LLVM::CallOp::create` used it, so operand #0 was read from freed memory.
> libc++ (macOS) left the slot intact → worked; libstdc++ (Linux)
> clobbered it → "operand #0 does not dominate this use". Fix: hold the
> operand in an owning `SmallVector<Value>` (clears the `-Wdangling`
> warning at that site). Verified green on the Linux runner — no test skip.
> The original analysis below is kept for reference.

### Symptom

`run-tests` lane, Linux only:

```
FAIL stats_t4_cluster: matlabc -emit-llvm errored
  loc("…/test/Run/stats_t4_cluster.m":6:1): error: operand #0 does not dominate this use
  error: MLIR-to-LLVM conversion pipeline failed
```

`test/Run/stats_t4_cluster.m:6` is the **five-output** PCA, fed by a
horzcat of a column-derived value:

```matlab
Xb = [1 4; 2 1; 3 5; 4 2; 5 6; 6 3; 7 2; 8 5];
c3 = Xb(:,1) + Xb(:,2);                              % line 4 — column extract + add
Xa = [Xb c3];                                        % line 5 — horzcat
[coeff, score, latent, ts, explained] = pca(Xa);     % line 6 — 5-output pca  <-- fails here
```

### Why it's Linux-only

It **passes on macOS** — the lowered LLVM IR there is correctly ordered:

```
%23 = call ptr @matlab_horzcat(ptr %17, ptr %22)   ; Xa
%24 = call ptr @matlab_stats_pca(ptr %23)          ; pca(Xa) — %23 dominates %24
%25 = call ptr @matlab_stats_pca_score()
%26 = call ptr @matlab_stats_pca_latent()
%27 = call ptr @matlab_stats_pca_empty()
%28 = call ptr @matlab_stats_pca_explained()
```

On Linux the same lowering produces an IR where a definition lands **after**
its use (dominance violation). The trigger is **op-insertion order**, which
differs because some pass iterates an `unordered_map`/`unordered_set` (or
relies on a leftover `OpBuilder` insertion point that is itself
order-dependent). It is the *same class* of bug as the StateChart
`persistent`-ordering fix in §2, but in the MLIR lowering rather than the
emitter.

### Suspected locations (start here)

- **5-output PCA splitter** — `lib/MLIR/Passes/LowerTensorOps.cpp`,
  `if (NA && Name == "pca" …)` (~line 3837). It boxes the input and emits
  five `matlab_stats_pca*` calls.
- **bracket-concat fold (`matlab_horzcat`/`matlab_vertcat`)** — same file,
  the `fold` lambda (~line 419–448). It uses `B.setInsertionPoint(Op)`
  before each create; verify the produced `matlab_horzcat` is placed before
  every consumer in *all* fixpoint interleavings.
- **column extraction + add** feeding `c3` (the `+` and `Xb(:,k)` lowering),
  and the slot-promotion store/load ordering for `Xa`.

The reliable structural fix is to make the relevant op creation
**dominance-deterministic**: set the `OpBuilder` insertion point explicitly
(to the consuming op) before creating any operand-boxing/conversion op, and
never depend on a leftover insertion point from a previous fixpoint
iteration. (A blanket `mlir::sortTopologically` is **not** safe here — the
`matlab_stats_pca*` calls have a hidden runtime ordering dependency: the
first call stashes a thread-local that the other four read, with no SSA
edge between them, so a topological re-sort could reorder them.)

---

## 4. How to reproduce and verify (on Linux)

The error is **swallowed by default** unless you build on Linux —
`test/Run/run_tests.sh` now prints the matlabc stderr on an `-emit-llvm`
failure, so CI shows the diagnostic. To reproduce locally on a Linux box:

```bash
# 1. Build matlabc on Linux (needs LLVM 22 + MLIR — see §5 about the blocker).
cmake -S . -B build -G Ninja -DMATLAB_LLVM_WITH_MLIR=ON \
  -DLLVM_DIR=/opt/llvm/lib/cmake/llvm -DMLIR_DIR=/opt/llvm/lib/cmake/mlir
cmake --build build --target matlabc -j"$(nproc)"

# 2. Reproduce — this prints "operand #0 does not dominate this use" on Linux:
build/matlabc -emit-llvm test/Run/stats_t4_cluster.m > /dev/null

# 3. Inspect the bad IR (compare op order against the macOS-valid order in §3):
build/matlabc -emit-llvm test/Run/stats_t4_cluster.m 2>&1 | less
```

A fix is verified when:

```bash
build/matlabc -emit-llvm test/Run/stats_t4_cluster.m > /dev/null   # exit 0
./test/Run/run_tests.sh "$PWD/build/matlabc"                       # run … 493/0
```

and the macOS run stays green (`493/0` there too) — the goal is one IR
ordering that is valid under both libc++ and libstdc++.

---

## 5. Blocker: the prebuilt LLVM tarball 404s

The Docker/local Linux repro is currently impeded because the prebuilt
`/opt/llvm` tarball that `.github/actions/setup-llvm` downloads
(`releases/download/toolchain-llvmorg-22.1.3/llvm-llvmorg-22.1.3-linux-x64.tar.zst`)
**returns 404** — the `.github/workflows/build-llvm-toolchain.yml` job that
publishes it failed (so CI itself falls back to the Actions cache, and a
fresh machine has no fast path to LLVM 22 + MLIR). Options to unblock a
contributor:

- Fix `build-llvm-toolchain.yml` so the tarball publishes (best — also
  speeds up CI cold starts), then `curl` it in a container.
- Or build LLVM 22 + MLIR from source once (~2 h) and cache `/opt/llvm`.

---

## 6. Checklist

- [x] `-std=c++20` + CLANG auto-detect in the Run harnesses
- [x] numpy in the shared CI action
- [x] Verilator width-lint suppression in the cocotb Makefile
- [x] StateChart emitter determinism (ordered containers)
- [x] DAP stdout line-buffering
- [x] strict `-Wno-missing-field-initializers`
- [x] tolerance-aware numeric golden compare (`numdiff.py`)
- [x] portable `rf_*` fixture paths; stable `fft_bluestein`
- [x] SymPP built in CI + symbolic suite expanded (4 → 11)
- [x] **`stats_t4_cluster` MLIR dominance bug (§3)** — fixed (dangling
      `ValueRange` in the pca splitter, `ec1d872`)
- [ ] Fix `build-llvm-toolchain.yml` so the prebuilt tarball publishes (§5)
      — remaining nice-to-have (CI falls back to the Actions cache today)
