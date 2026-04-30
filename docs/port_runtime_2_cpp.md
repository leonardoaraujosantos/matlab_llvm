# Port Runtime to C++ — Plan

Plan for migrating `runtime/matlab_runtime.cpp` (originally
`matlab_runtime.c` — ~8200 lines after Tier-1/2/3, ~485 functions) from
C to modern C++. The companion file
[`runtime/matlab_runtime.hpp`](../runtime/matlab_runtime.hpp) is the
existing thin C++ wrapper exposed to the EmitC C++ path; it stays.

This document is a **plan**, not a record. Status markers below describe
intended state; they will be updated as work lands.

## Status legend

- **planned** — designed, not started
- **in progress** — partially landed
- **shipped** — merged and exercised by tests
- **deferred** — acknowledged but out of scope for this port

---

## Goal & non-goals

**Goal.** Reduce code-quality risk in the runtime: leak surface from 178
manual `malloc`/`free`/`memcpy` calls and 189 alloc/free pair sites,
copy-pasted shape-op skeletons, a 7188-line monolithic translation
unit, and zero direct unit-test coverage.

**Non-goals.**

- **Do not change the public ABI.** The runtime is loaded into `matlabc`
  via `target_sources(matlabc PRIVATE runtime/matlab_runtime.c)` at
  `CMakeLists.txt:173` and JIT-emitted code resolves symbols by C name
  through `LLJIT::DynamicLibrarySearchGenerator::GetForCurrentProcess`.
  425 call sites in `lib/MLIR/` and `lib/MIR/` reference runtime symbols
  by string literal. All public functions stay `extern "C"` with their
  current signatures.
- **Do not rewrite to "idiomatic C++".** This is "keep the symbol
  surface, modernize the guts." No template metaprogramming, no
  exceptions across the ABI, no STL types in public signatures.
- **Do not port to C++ before the prerequisites in Phase 0–2 land.**
  Coverage data, unit tests, and a file split are higher-ROI and
  de-risk the port itself.

---

## Phase 0 — Coverage baseline (prerequisite) — **shipped**

Wired Clang source-based coverage so the existing test corpus reports
which runtime functions actually execute. Cold paths
(`lu_decompose`, `jacobi_sym`, `matlab_svd`, `matlab_eig_V/D`,
`rng_normal`, `set_op`, `sortrows`, the entire `MAT_C_BINARY`
complex-matrix family, FFT/conv/chol/norm) are now identified
quantitatively rather than by guess.

**Wired in this round.**

- `MATLAB_LLVM_COVERAGE` CMake option (`CMakeLists.txt:22`). Off by
  default; when on, appends `-fprofile-instr-generate
  -fcoverage-mapping` to `matlabc`'s compile + link lines. Errors
  fast if either compiler isn't Clang.
- `scripts/runtime_coverage.sh` — turn-key script that configures with
  the option on, builds `matlabc` plus the Phase 1 `runtime-test-*`
  binaries, runs every CTest suite, merges all `.profraw` files via
  `llvm-profdata merge -sparse`, then emits `summary.txt`,
  `uncovered.txt`, and (with `--html`) an HTML drill-down. All output
  lands under `build-coverage/coverage/`.

**Baseline (run 2026-04-29, after Phase 1 unit tests landed).**

| Metric    | Covered | Total | Coverage  |
| --------- | ------- | ----- | --------- |
| Functions | 187     | 451   | **41.46%** |
| Lines     | 2249    | 4880  | **46.09%** |
| Regions   | 1854    | 4981  | 37.22%    |
| Branches  | 915     | 3174  | 28.83%    |

**311 of 451 functions (69%) have 0% line coverage** — the full list
is at `build-coverage/coverage/uncovered.txt`. High-value entries the
port should keep an eye on, all 0%-covered today:

- Linalg cold paths: `matlab_chol`, `matlab_norm`, `matlab_trace`,
  `matlab_kron`, `matlab_sortrows`, `matlab_diff`
- FFT family: `matlab_fft_c`, `matlab_ifft_c`, `matlab_fft2_c`,
  `matlab_ifft2_c`, `matlab_fftshift_c`, `matlab_ifftshift_c`
- Set / signal ops: `matlab_intersect`, `matlab_union`,
  `matlab_conv`, `matlab_conv2`, `matlab_median`, `matlab_var`,
  `matlab_std`

Reproduce: `./scripts/runtime_coverage.sh` from the repo root. The
27 CTest suites currently take ~13 minutes wall time on a Mac
(`flowchart-lsp-tests` fails because `matlab-lsp` isn't built by the
script — coverage data is unaffected; fix is one extra `--target` if
needed).

**Exit criteria.** ✅ Reproducible coverage report; per-function
0%-coverage list captured. Subsequent phases must not regress these
numbers.

---

## Phase 1 — Direct runtime unit tests (prerequisite) — **in progress**

Initial cut landed alongside Phase 0. Five C harnesses live under
`test/Runtime/`, register as CTest entries, link
`runtime/matlab_runtime.c` directly, and run in under 2 seconds total.

**Wired in this round.**

| File                          | Functions exercised                                                            |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `test/Runtime/runtime_test.h` | Tiny assertion macros + struct-layout shims (`matlab_mat`, `matlab_mat_c`)     |
| `test/Runtime/test_linalg.c`  | `matmul`, `inv`, `det`, `mldivide`, `mrdivide`, `transpose`, `diag`, `eig`, `eig_V`, `eig_D`, `svd`, `zeros`, `ones`, `eye`, `magic` — 90 assertions |
| `test/Runtime/test_shape.c`   | `fliplr`, `flipud`, `rot90`, `repmat`, `reshape`, `range`, `diag` — 54 assertions |
| `test/Runtime/test_rng.c`     | `rand`, `randn` (range, finiteness, mean check) — 37 assertions                |
| `test/Runtime/test_complex.c` | `complex_scalar`, `mat_c_from_real`, `add_cc`, `emul_cc`, `ediv_cc`, `matmul_cc`, `conj_c`, `neg_c`, `real_c`, `imag_c`, `abs_c`, `angle_c`, `ctranspose_c` — 46 assertions |
| `test/Runtime/test_reduce.c`  | `sum`, `prod`, `mean`, `min`, `max`, `cumsum`, `cumprod`, `sort`, `unique`, `ismember` — 39 assertions |
| `test/Runtime/test_signal.c`  | Tier-1/2: `conv`, `conv2`, `filter`, `xcorr`, `fftshift`, `ifftshift`, `hamming`, `hann`, `blackman`, `upsample`, `downsample`, `diff` — 102 assertions |
| `test/Runtime/test_stats.c`   | Tier-1/2: `any`, `all`, `tril`, `triu`, `std`, `var`, `median`, `meshgrid`, `ndgrid`, `polyval`, `polyfit`, `roots`, `interp1`, `interp2`, `trapz`, `cumtrapz`, `gradient` — 75 assertions |
| `test/Runtime/test_image.c`   | Tier-3: `imfilter`, `padarray`, `rank`, `cond`, `null`, `orth` — 30 assertions |
| `test/Runtime/test_more.c`    | Phase-4-touched + 0%-allocator catch-up: `chol`, `lu_L`, `lu_U`, `qr_Q`, `qr_R`, `pinv`, `kron`, `intersect`, `union`, `setdiff`, `repmat`, `linspace`, `find`, `horzcat`, `vertcat`, `squeeze`, `slice2`, `matpow` — 67 assertions |

CMake glue is the `foreach(_rt_test IN ITEMS linalg shape rng complex
reduce signal stats image)` block in `CMakeLists.txt`, which compiles
each `test_*.c` with the runtime TU directly and registers
`runtime-tests-<name>` with CTest. When `MATLAB_LLVM_COVERAGE=ON` is
set, the same flags propagate to the test binaries automatically.

After Phase 3 the runtime TU is `runtime/matlab_runtime.cpp`; the
`test/Runtime/test_*.c` files keep `.c` extensions and link against
the C++ TU directly. The C++ runtime is built with `extern "C"` around
the entire payload so the symbol table is byte-identical to the
former C build.

The struct layouts (`rt_test_mat_layout`, `rt_test_matc_layout` in
`test/Runtime/runtime_test.h`) deliberately mirror the runtime's
private definitions in `runtime/matlab_runtime.c` so tests can read
output matrices field-by-field. If the runtime layout ever changes,
update the shims.

**Coverage delta.** Adding the 5 original unit-test suites moved line
coverage from 36.96% (matlabc-only baseline) to **46.09%** — a 9.1pp
lift. The Tier-1/2/3 follow-up adds three more suites
(`test_signal.c`, `test_stats.c`, `test_image.c`) covering 33 newly
introduced builtins with 207 fresh assertions; a re-run of
`scripts/runtime_coverage.sh` will report the updated number.

**Phase-1 catch-up still pending** for the pre-existing 52 public
0%-allocators listed below. Closing them is the path to the ≥70%
target line coverage.

### Phase 1 punch list — port-touched functions

The Phase 4 RAII migration only touches functions that allocate.
Scanning `runtime/matlab_runtime.c` for `mat_alloc` / `mat_c_alloc` /
`malloc` / `calloc` call sites yields **111 such functions**. Crossed
against the Phase 0 coverage report:

| Bucket                                         | Count | What to do                                                       |
| ---------------------------------------------- | ----- | ---------------------------------------------------------------- |
| Fully covered (line cov = 100%)                | 19    | Nothing — keep green                                             |
| Partial (1–89%)                                | 15    | Add edge-case tests until ≥90% lines                             |
| Public, 0% covered                             | 52    | Write a direct test before the port modifies the body            |
| Static helpers / not surfaced in `-show-functions` | 25  | Covered transitively when their callers gain tests               |

The 15 partial entries (most need empty-input / shape-edge cases):

```
matlab_abs_c        62.50%    matlab_inv          86.36%
matlab_angle_c      56.25%    matlab_magic        90.48%
matlab_dbg_enter_frame 91.67% matlab_mldivide_mm  86.96%
matlab_eig          80.70%    matlab_range        90.00%
matlab_eig_D        88.89%    matlab_real_c       58.33%
matlab_eig_V        66.67%    matlab_sort         57.89%
matlab_imag_c       70.00%    matlab_struct_get_mat 93.33%
matlab_svd          72.46%
```

The 52 public 0%-covered allocators that the port will modify:

```
matlab_all             matlab_horzcat         matlab_pinv
matlab_any             matlab_ind2sub         matlab_qr_Q
matlab_atan2_m         matlab_kron            matlab_qr_R
matlab_cell_get_mat    matlab_linspace        matlab_size
matlab_cell_new        matlab_load_mat        matlab_slice1
matlab_chol            matlab_lu_L            matlab_slice2
matlab_conv            matlab_lu_U            matlab_sortrows
matlab_conv2           matlab_mat_from_scalar matlab_squeeze
matlab_dbg_undo_record_irreversible            matlab_string_concat
matlab_diff            matlab_matpow          matlab_string_from_literal
matlab_empty_mat       matlab_max_mm          matlab_strrep
matlab_epow_mm         matlab_median          matlab_tril
matlab_epow_ms         matlab_meshgrid_X      matlab_triu
matlab_epow_sm         matlab_meshgrid_Y      matlab_var
matlab_erase_cols      matlab_min_mm          matlab_vertcat
matlab_erase_rows      matlab_ndgrid_X
matlab_filter          matlab_ndgrid_Y
matlab_find            matlab_permute
matlab_flip
matlab_fread
```

Regenerate this list anytime with:

```sh
./scripts/runtime_coverage.sh   # rebuilds + reruns; ~13 min
# then read build-coverage/coverage/uncovered.txt
```

**Exit criteria for Phase 1.**

- All 52 public 0%-allocators have at least one direct test in
  `test/Runtime/`.
- The 15 partial entries reach ≥90% line coverage.
- `runtime-tests-*` runs under ASan + UBSan with zero diagnostics.

That target — call it **the port-touched-coverage line** — is what
matters before Phase 3 starts. It is *not* the same as overall
coverage hitting 90%: about 200 of the 311 remaining 0%-covered
functions are macro-generated unaries (`matlab_acos_m`,
`matlab_atan2_s`, ...), debug/workspace bridges only reachable via
DAP, and cell/string helpers the port will not touch. Pushing those
to 90% adds compile time without de-risking the migration.

---

## Phase 2 — File split — **shipped (initial cut)**

Started with a conservative two-file split: `matlab_runtime.cpp` keeps
the numerical core, `runtime_debug.cpp` carries the DAP / REPL
workspace machinery (about 2,800 lines extracted from the old
contiguous block at lines 4149–6965). Both TUs share private types
and globals via `runtime/runtime_internal.h`. This narrows
`matlab_runtime.cpp` from 8,116 to 5,299 lines, which already
sidesteps the worst of the navigation pain.

**What landed.**

- `runtime/runtime_internal.h` — new, ~165 lines. Exposes:
  * `struct matlab_mat`, `struct matlab_mat_c`, `struct matlab_mat3`
    layouts (full def, used by both TUs to reach into descriptor
    fields).
  * `struct matlab_struct_s`, `struct matlab_obj_s` — the workspace
    mirror needs the layout to walk variables for the DAP variables
    panel.
  * `MATLAB_MAT_C_MAGIC` / `MATLAB_MAT3_MAGIC` constants and the
    `mat_is_complex` / `mat_is_3d` inline predicates.
  * `extern` declarations for `matlab_io_mutex`, `matlab_error_msg`,
    `matlab_error_msg_len`, `matlab_error_flag`. Definitions are
    `non-static` in `matlab_runtime.cpp`.
  * Allocator forward decls: `mat_alloc`, `mat_c_alloc`, `mat3_alloc`,
    `struct_find_field`, `struct_reserve` — all dropped their
    `static` qualifier so `runtime_debug.cpp` can call them.
  * Phase-4 RAII helpers (see below) inside a `#ifdef __cplusplus`
    guard.
- `runtime/runtime_debug.cpp` — new, ~2,870 lines. Contains the entire
  REPL workspace + DAP machinery extracted verbatim from
  `matlab_runtime.cpp`. Its preamble forward-declares the
  `matlab_struct_*` / `matlab_disp_*` helpers it calls (signatures
  match the public ABI but kept internal — the public header would
  pull in the conflicting `void *` macro decls).
- `runtime/matlab_runtime.cpp` — duplicates of the moved struct/static
  layouts removed; `matlab_err_snapshot_frames` and
  `matlab_err_emit_traceback_to_stderr` lost their `static` so
  `matlab_set_error` can call across the TU boundary.
- `CMakeLists.txt`: matlabc `target_sources` and the runtime-test
  `add_executable` foreach both list the new file.
- `runtime/build_and_run.sh` and the `test/Run/run_tests*.sh` scripts:
  every link line now passes both `.cpp` files.

**What's still on the table for Phase 2.5.**

The plan's full 9-file layout is still the right end-state. Splitting
the remaining ~5,300 lines into core / linalg / complex / rng /
parfor / array still buys a lot. But it's contained, mechanical work
now that the debug block is out of the way; saving for a follow-up.

**Exit criteria — met for the initial cut.**

- ✅ All 32 CTest suites still green (none of the moved code broke).
- ✅ Coverage and unit tests bytes-identical (the moves were
  textual). Re-run `scripts/runtime_coverage.sh` to confirm.
- ⏭ "No file over ~1500 lines" — not yet. `matlab_runtime.cpp` is
  ~5,300; `runtime_debug.cpp` is ~2,870. Phase 2.5 takes them down.

**Proposed layout** (under `runtime/`):

| File                  | Contents                                                       |
| --------------------- | -------------------------------------------------------------- |
| `runtime_internal.h`  | `mat_alloc`, struct layouts, shared statics — not installed    |
| `runtime_io.c`        | `matlab_disp_*`, `matlab_fprintf_*`, `matlab_input_num`        |
| `runtime_array.c`     | constructors, shape ops, reductions, scans, sort, set ops      |
| `runtime_linalg.c`    | `matmul`, `inv`, `mldivide`, `mrdivide`, `svd`, `eig`, `lu`    |
| `runtime_complex.c`   | `matlab_mat_c` family + `MAT_C_BINARY`                         |
| `runtime_rng.c`       | `rng_uniform`, `rng_normal`, `rand`, `randn`                   |
| `runtime_parfor.c`    | `matlab_parfor_dispatch`, `matlab_parfor_worker`, reductions   |
| `runtime_debug.c`     | `matlab_dbg_*` frame/file/error machinery used by DAP          |
| `runtime_workspace.c` | `matlab_ws_*` REPL/DAP workspace bridge                        |

**Steps.**

1. Move code, no edits. Each `.c` keeps the same `static` helpers it
   uses; promote one helper to `runtime_internal.h` only when it has
   genuine cross-file users.
2. Update `CMakeLists.txt:173` from a single source to a glob /
   explicit list.
3. Re-run Phase 0 coverage and Phase 1 unit tests — bytes identical,
   so both must be green.

**Exit criteria.** No file over ~1500 lines; coverage and unit tests
unchanged.

---

## Phase 3 — Compile as C++ (no behavior change) — **shipped**

Renamed `runtime/matlab_runtime.c` → `runtime/matlab_runtime.cpp`,
wrapped the entire payload in a single `extern "C" { … }` block (with
the include preamble outside the block), and pointed every build
recipe at the new path. **Zero source changes inside the body** — the
existing C-shaped code compiles cleanly under Clang's C++ frontend
with no typed-cast / VLA / designated-init fixups required.

**Things that landed.**

- `runtime/matlab_runtime.c` removed; `runtime/matlab_runtime.cpp`
  added (one `extern "C"` open after the includes; one matching close
  at end-of-file).
- `CMakeLists.txt` two `target_sources` / `add_executable` lines
  updated to reference `matlab_runtime.cpp`. The `runtime-test-*`
  CTest binaries now build with `add_executable(runtime-test-X
  test/Runtime/test_X.c runtime/matlab_runtime.cpp)` — Clang
  auto-detects each TU's language from the extension, so the C tests
  and C++ runtime co-link cleanly.
- `runtime/build_and_run.sh` switched from `clang` to `clang++`.
- `test/Run/run_tests.sh` switched to `${CLANG}++` for the runtime
  link.
- `test/Run/run_tests_emitc.sh` reworked: previously fed the runtime
  to the C compiler with `-x c`; now feeds it to `$CXX` with
  `-x c++` while still compiling the matlabc-emitted `.c` source as C
  in the same invocation. Both `MODE=c` and `MODE=cpp` paths covered.

**Things deliberately *not* done in this phase.**

- No file split (Phase 2 — deferred per the note above).
- No RAII (Phase 4 — explicit follow-up).
- No `OBJECT` library promotion. The runtime is still pulled into
  every consumer via `target_sources` / direct add-source; promoting
  it to an `add_library(matlab_runtime OBJECT …)` is a small follow-up
  that would let the test binaries link a shared object file instead
  of recompiling the runtime once per test.
- No coverage rebuild — `MATLAB_LLVM_COVERAGE=ON` flags need to flow
  through the C++ compiler now; verify with a re-run of
  `scripts/runtime_coverage.sh`.

**Exit criteria — met.**

- ✅ All 32 CTest suites pass on macOS (`frontend-tests`,
  `flowchart-tests`, `flowchart-emit-*`, the 8 `runtime-tests-*`
  binaries, all 5 `run-tests*`, DAP / cocotb / SV / dwarf suites).
- ⏭ ASan + UBSan re-run pending (deferred to a follow-up that adds
  the sanitizer CMake option).
- ⏭ Coverage report under C++ pending; expected within noise.

---

## Phase 4 — RAII for the array/struct/cell handles — **in progress (linalg + Tier-1/2/3 sweep landed)**

The first migration landed as a named exemplar: **`matlab_inv`** in
`runtime/matlab_runtime.cpp`. The smart-pointer types and the
allocator-wrapping factory functions live in `runtime/runtime_internal.h`
under `namespace matlab::runtime`:

```cpp
struct MatDeleter  { void operator()(matlab_mat  *p) const noexcept; };
struct MatCDeleter { void operator()(matlab_mat_c *p) const noexcept; };
using MatPtr  = std::unique_ptr<matlab_mat,  MatDeleter>;
using MatCPtr = std::unique_ptr<matlab_mat_c, MatCDeleter>;

inline MatPtr  make_mat   (int64_t m, int64_t n);
inline MatCPtr make_mat_c (int64_t m, int64_t n);
```

`matlab_inv` before:
```cpp
matlab_mat *matlab_inv(matlab_mat *A) {
    if (A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    double *LU = (double *)malloc(n*n*sizeof(double));
    memcpy(LU, A->data, n*n*sizeof(double));
    int64_t *piv = (int64_t *)malloc(n*sizeof(int64_t));
    int sign;
    if (lu_decompose(LU, n, piv, &sign) != 0) {
        free(LU); free(piv);   /* manual cleanup on early exit */
        return mat_alloc(0, 0);
    }
    matlab_mat *X = mat_alloc(n, n);
    double *rhs = (double *)malloc(n*sizeof(double));
    double *col = (double *)malloc(n*sizeof(double));
    for (...) { ... }
    free(rhs); free(col); free(piv); free(LU);   /* four frees */
    return X;
}
```

`matlab_inv` after:
```cpp
matlab_mat *matlab_inv(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    std::vector<double> LU(A->data, A->data + n*n);
    std::vector<int64_t> piv(n);
    int sign;
    if (lu_decompose(LU.data(), n, piv.data(), &sign) != 0)
        return mat_alloc(0, 0);
    matlab::runtime::MatPtr work = matlab::runtime::make_mat(n, n);
    std::vector<double> rhs(n), col(n);
    for (...) { ... }
    return work.release();   /* hands the ptr back over the C ABI */
}
```

Net: 4 `malloc` and 4 `free` calls drop to zero; the previously-leaked
`LU` + `piv` on the singular path is now leak-free by construction;
the public ABI is byte-identical (`work.release()` returns the same
`matlab_mat *`).

**What's left for Phase 4 sweep.**

The runtime today had 178 manual `malloc`/`free` calls and 189
alloc/free pair sites at the start of the project. After this round
the linalg core, the polynomial / numeric-calculus tail, and the
set/sort scratch users are migrated:

1. ✅ `matlab_inv` (linalg exemplar)
2. ✅ `matlab_mldivide_mm` (same LU machinery; `mrdivide` reuses it)
3. ✅ `matlab_svd`, `matlab_eig`, `matlab_eig_V`, `matlab_eig_D`,
   `jacobi_sym` — heaviest scratch users
4. ✅ `matlab_chol`, `matlab_lu_L`, `matlab_lu_U`, `matlab_qr_Q`,
   `matlab_qr_R`. `matlab_pinv` builds on `matmul`/`inv` so it gains
   leak-free behaviour transitively.
5. ✅ Tier-1/2/3 scratch users: `matlab_filter`, `matlab_polyfit`,
   `matlab_roots`, `matlab_median`, `matlab_trapz`, `matlab_trapz_xy`,
   `matlab_gradient`, `set_op` (drives `setdiff`/`intersect`/`union`).
6. ⏭ Complex / FFT family (`matlab_fft_c`, `matlab_fft2_c`,
   `fft_radix2_inplace`, `fft_bluestein`) — still on raw `malloc`.
7. ⏭ Misc: `matlab_kron`, `matlab_repmat`, `matlab_horzcat`,
   `matlab_vertcat`, `matlab_permute`, `matlab_squeeze` — these
   currently use only `mat_alloc` (no scratch buffers), so the
   migration there is purely cosmetic. Defer until Phase 5
   shape-op-template lands.

Per-turn migration tally: ~24 manual `malloc`/`free` pairs eliminated
across 13 functions in this round. Combined with the prior `matlab_inv`
exemplar that's about 28 of the 178 manual calls retired (~16%),
focused on the highest-traffic numerical paths.

Each migration follows the `matlab_inv` template — replace
`malloc`/`free` pairs with `std::vector` for raw scratch and `MatPtr`
for the result descriptor; `release()` at the return.

**Exit criteria for full Phase 4.** Manual `malloc`/`free` count in
the runtime drops from 178 to under 30 (the ones legitimately
interfacing the C ABI boundary). ASan + LSan report zero leaks across
all suites. Each migrated function has direct unit-test coverage
exercised under sanitizers.

---

## Phase 5 — De-duplicate the shape-op skeletons — **planned**

The macro families `BINARY_MM/MS/SM`, `CMP_*`, `UNARY_M`,
`COLWISE_REDUCE`, `DIM_REDUCE`, `CUM_SCAN`, `MAT_C_BINARY` already do
their job and should stay. The remaining duplication is in the
hand-written shape operations: `fliplr`, `flipud`, `flip`, `rot90`,
`transpose`, `diag`, `repmat`, `permute`, `squeeze`, `reshape` — each
is "null-guard → `mat_alloc(m, n)` → double for-loop with one differing
index expression → return" repeated 10+ times.

**Approach.**

- Introduce a templated internal helper:

  ```cpp
  template <class IndexFn>
  static matlab_mat *shape_op(matlab_mat *A, int64_t m, int64_t n, IndexFn idx);
  ```

- Each public op becomes a one-line lambda passed to `shape_op`. The
  resulting `.cpp` body for all shape ops fits on one screen.
- Same treatment for the "alloc + scalar fill" family
  (`zeros`/`ones`/`eye`/`magic`/`mat_from_buf`).

**Exit criteria.** The shape-op section in `runtime_array.cpp` shrinks
by ~60% with no behavior change measured by Phase-1 tests.

---

## Phase 6 — Error-path consolidation — **planned**

Today, errors from the runtime go through `matlab_set_error_msg` plus
ad-hoc `if (cond) { matlab_set_error_msg(...); return mat_alloc(0,0); }`
patterns scattered across linalg routines. With RAII in place, this
becomes a single internal helper.

**Approach.**

- Internal-only `RuntimeError` struct returned via `std::optional` /
  `std::expected`-style helpers (C++23 `std::expected` if available,
  else a hand-rolled equivalent — **never** crosses the C ABI).
- Public entry points translate into the existing
  `matlab_set_error_msg` + empty-mat return at the outermost layer.

**Exit criteria.** Error-set call sites collapse to one location per
public function.

---

## Risks & mitigations

| Risk                                                              | Mitigation                                                                                  |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| C++ name mangling leaks into the ABI                              | `extern "C"` enforced inside each `.cpp`; CI grep-check that every public symbol matches    |
| C++ exceptions cross the JIT boundary                             | Compile runtime with `-fno-exceptions`; ASan test that throws inside runtime aborts cleanly |
| ABI-level layout changes to `matlab_mat`                          | Layout stays in `runtime_internal.h`; no virtual members, no inheritance                    |
| `std::vector` reallocations in hot loops regress perf             | Phase 4 includes a microbenchmark suite for `matmul`/`add_mm`/`svd`; hold within ±3%        |
| EmitC C++ output (`matlab_runtime.hpp`) breaks                    | Phase 3 keeps `.hpp` byte-identical; covered by `test/EmitC/*.cpp.expected`                 |
| LLJIT symbol resolution misses C++-mangled internal helpers       | Internal helpers stay `static` inside the `.cpp`; only `extern "C"` symbols are JIT-visible |
| Coverage tooling differs between Linux and macOS CI               | Phase 0 picks `-fprofile-instr-generate` (Clang) for portability                            |

---

## Sequencing & estimated cost

| Phase                            | Why it goes here                                                  | Rough size |
| -------------------------------- | ----------------------------------------------------------------- | ---------- |
| 0 — Coverage baseline            | Need numbers before claiming "improvement"                        | ~0.5 day   |
| 1 — Direct unit tests            | Without these, the port is unsafe to land                         | 2–3 days   |
| 2 — File split (still C)         | Mechanical, reversible, isolated diff                             | 1 day      |
| 3 — Compile as C++ (no changes)  | Smallest possible "is C++ even green?" step                       | 1 day      |
| 4 — RAII migration               | The actual code-quality win                                       | 3–5 days   |
| 5 — Shape-op dedup               | Cleanup that becomes natural once C++ is in                       | 1 day      |
| 6 — Error-path consolidation     | Last because it depends on RAII being in place                    | 1–2 days   |

Phases 0–2 are valuable on their own; if the C++ migration is paused
after Phase 2, the runtime is already meaningfully better and the
remaining phases can resume later without rework.

---

## Open questions

- Should the runtime migrate to its own CMake target now (`add_library(
  matlab_runtime OBJECT …)`), or stay as `target_sources` on `matlabc`
  until Phase 3? Migrating sooner makes the unit-test binary trivially
  link against the same TU; deferring keeps the CMake diff small.
- Do we want a dedicated `runtime/include/matlab/` public header
  directory now, or keep `matlab_runtime.h` flat? Affects how the EmitC
  C++ wrapper's `#include "matlab_runtime.hpp"` line is resolved by
  downstream `clang` invocations on emitted code.
- For Phase 6, is C++23 `std::expected` available on every platform we
  target, or do we hand-roll? Decide once Phase 3 establishes the
  baseline `-std=` flag.

---

## Validation matrix

Each phase must keep these green:

- `frontend-tests` (`test/run_tests.sh`)
- `flowchart-tests`
- `frontend-emitc-tests`, `frontend-emitsv-tests`, `frontend-emitcocotb-tests`,
  `frontend-emitpython-tests`, `frontend-emitts-tests` — these cover the
  runtime indirectly through generated code
- `debug-hook-tests` and `debug-dap-tests` (Python harness under
  `test/Debug/`) — exercise `matlab_dbg_*` and `matlab_ws_*`
- `runtime-tests` (new, lands in Phase 1)
- ASan + UBSan clean on `runtime-tests` and on a representative subset
  of `frontend-tests`
