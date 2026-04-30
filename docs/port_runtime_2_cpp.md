# Port Runtime to C++ — Plan

Plan for migrating `runtime/matlab_runtime.c` (7188 lines, 346 functions,
99 statics) from C to C++. The companion file
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

## Phase 0 — Coverage baseline (prerequisite) — **planned**

Before changing anything, measure what the existing `.m` integration
suite (`test/Run/`, 144 files) actually exercises in the runtime. Cold
paths likely include `lu_decompose`, `jacobi_sym`, `matlab_svd`,
`matlab_eig_V/D`, `rng_normal`, `set_op`, `sortrows`, and the entire
`MAT_C_BINARY` complex-matrix family.

**Steps.**

1. Add a `MATLAB_LLVM_COVERAGE` CMake option that appends
   `--coverage` (or `-fprofile-instr-generate -fcoverage-mapping` on
   Clang) to the runtime translation unit and `matlabc` link line.
2. Wire a `coverage` CTest target that runs the existing suites and
   merges `.gcda`/`.profraw` into an `lcov` HTML report under
   `build/coverage/`.
3. Commit the resulting per-function coverage table into this doc as
   the **baseline** the port must not regress.

**Exit criteria.** A reproducible coverage report; the list of
runtime functions with 0% line coverage is known and recorded.

---

## Phase 1 — Direct runtime unit tests (prerequisite) — **planned**

The runtime currently has **no direct tests**. `grep -rln
"matlab_runtime\|matlab_zeros\|matlab_inv\|matlab_svd" test/` returns
zero matches against runtime symbols; everything is exercised via
codegen → JIT, which couples test failures to the lowering layer and
leaves cold runtime paths untested.

**Steps.**

1. Create `test/Runtime/` with a minimal C harness (`runtime_smoke.c`,
   `runtime_linalg.c`, `runtime_complex.c`, `runtime_rng.c`,
   `runtime_shape.c`).
2. Link directly against `runtime/matlab_runtime.c` — no JIT, no
   MATLAB frontend. Each test allocates inputs, calls the runtime
   function, asserts the result, calls `matlab_mat_free`.
3. Prioritize functions Phase 0 reports as 0% covered, plus the
   functions about to be touched by the port (everything that calls
   `mat_alloc` / `malloc`).
4. Register the binary as a CTest entry alongside `frontend-tests` /
   `flowchart-tests` in `CMakeLists.txt`.
5. Run under ASan + UBSan in CI to catch leaks the port might
   introduce or fix.

**Exit criteria.** Every function the port touches in Phase 3+ has at
least one direct test; ASan reports zero leaks on the existing C
runtime before any C++ work begins.

---

## Phase 2 — File split (still C) — **planned**

Splitting the monolith is mechanical, reversible, and worth more than
the language change. Done in C first so the diff is purely about
boundaries, not language.

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

## Phase 3 — Compile as C++ (no behavior change) — **planned**

Rename the `.c` files to `.cpp`, fix the compile, **stop**. This phase
introduces zero new C++ features; it just gets the code through a C++
compiler so later phases can adopt RAII incrementally.

**Concrete fixes expected.**

- `void *` → typed pointer casts (C++ requires explicit casts for
  `malloc` results); already partly done where `matlab_mat *` is used,
  but `mat_alloc` returns `void *` in places.
- Designated initializers — already C99/C++20, fine on modern Clang
  but may need `-std=c++20`.
- Variable-length arrays (search `runtime/matlab_runtime.c` for `[n]`
  on the stack); replace with `std::vector` or heap buffers as needed.
- C-style implicit `enum`-to-`int` conversions and signed/unsigned
  comparisons cleaned up.
- Wrap public declarations in `extern "C" { … }` directly inside the
  `.cpp` files (not just the header) so the compiler enforces the ABI
  locally.

**Build changes.**

- `CMakeLists.txt`: switch the runtime sources from `.c` to `.cpp`,
  ensure `target_compile_features(matlab_runtime PUBLIC cxx_std_20)`
  applies only to the runtime target if it becomes its own object
  library.
- Promote the runtime to its own `OBJECT` library (`add_library(
  matlab_runtime OBJECT …)`) so `matlabc` and any future test binary
  link the same TU. Avoids drift.

**Exit criteria.** All four CTest suites green
(`frontend-tests`, `flowchart-tests`, the new `runtime-tests`, and the
DAP `test/Debug/` Python suite) with the runtime built by the C++
compiler. ASan + UBSan still clean. Coverage report unchanged within
noise.

---

## Phase 4 — RAII for the array/struct/cell handles — **planned**

This is where the actual code-quality win lands. Replace 189 manual
alloc/free pairs with destructors.

**Design.**

- Keep `matlab_mat`, `matlab_mat_c`, `matlab_struct`, `matlab_cell` as
  the opaque types in `runtime/matlab_runtime.h` — they are returned
  across the C ABI and JIT-emitted code stores them as `i8*`.
- Inside the `.cpp` TUs, define internal smart-pointer aliases:

  ```cpp
  struct MatDeleter { void operator()(matlab_mat* p) const noexcept; };
  using MatPtr = std::unique_ptr<matlab_mat, MatDeleter>;
  ```

- Rewrite internal helpers (`set_op`, `jacobi_sym`, `lu_decompose`,
  the temporary-buffer-heavy linalg routines) to hold intermediates
  in `MatPtr` / `std::vector<double>`. The public functions still
  take and return raw `matlab_mat *`; they `release()` at the return.
- Public entry points become a thin pattern:

  ```cpp
  extern "C" matlab_mat *matlab_inv(matlab_mat *A) {
      if (!A) return mat_alloc(0, 0);
      MatPtr work = clone(A);
      // … RAII-managed scratch …
      return work.release();
  }
  ```

**Steps.**

1. Land `MatPtr`, `MatCPtr`, `StructPtr`, `CellPtr` plus their
   deleters in `runtime_internal.h`.
2. Migrate **one** linalg routine end-to-end (`matlab_inv`) as the
   exemplar; review.
3. Sweep the remaining linalg + complex routines.
4. Sweep array/shape ops.
5. Each migrated function gets a Phase-1 unit test if it does not
   already have one, exercised under ASan.

**Exit criteria.** Manual `malloc`/`free` count in the runtime drops
from 178 to under 30 (the ones legitimately interfacing the C ABI
boundary). ASan + LSan report zero leaks across all suites.

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
