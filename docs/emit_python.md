# Python Emission — Plan

Forward-looking design doc for adding a `-emit-python` backend alongside
the existing `-emit-c` / `-emit-cpp` pipeline. Nothing in this document
has shipped yet; it lays out the concrete steps, file-by-file, to bring
the backend up.

## Scope

One new driver flag:

| Flag | Output |
|---|---|
| `-emit-python` | Self-contained `.py` source |

The generated file imports a small `matlab_runtime.py` shim (NumPy-backed)
and runs on CPython 3.10+. No LLVM or C toolchain required at runtime:

```bash
matlabc -emit-python foo.m > foo.py
python3 foo.py
```

## Why this is cheap

The fork point already exists. After `runLowerIO`, the module is a
closed set of ~12 op kinds (see `emit_c_cpp.md` for the full list) and
the `emitC()` walker is a 1166-line single-file translator we can clone.
Python is a *softer* target than C: dynamic typing eliminates most of
the casts, `scf.if` / `scf.while` map to native Python control flow
without the "mutable local + break" trick, and the runtime can be a
thin wrapper over NumPy.

## Architecture reuse

Everything up to `runLowerIO` is shared untouched. The pipeline
diverges at the same fork point as `-emit-c`:

```
AST ──► MLIR ──► [SlotPromotion, ..., LowerIO]
                       │
                       ├──► [scf→cf, arith→llvm, ...] ──► .ll
                       ├──► emitC()     ──► .c / .c++
                       └──► emitPython() ──► .py     ← new
```

## Step-by-step plan

### Step 1 — CLI flag (30 min)

`tools/matlabc/main.cpp`:
- Add `EmitPython` to the `Mode` enum (line 35).
- Add `-emit-python` parsing (line 56 block).
- Extend the `WantFullPipeline` check (line 143–156) to include the new mode.
- After `runLowerIO`, add a branch mirroring the `EmitC` branch at line 285 that calls `emitPython(M)`.

`include/matlab/MLIR/Passes/Passes.h`:
- Declare `LogicalResult emitPython(ModuleOp M, raw_ostream &OS);` next to `emitC()` (line 123).

### Step 2 — Skeleton emitter (~1 day)

`lib/MLIR/Passes/EmitPython.cpp`:
- Clone `EmitC.cpp` as a starting point. Keep:
  - The `Emitter` class shape
  - `DenseMap<Value, std::string>` SSA naming
  - `matlab.name` attribute propagation for readable locals
  - Single-use inlining analysis (applies identically to Python)
  - Pre-emit `mlir::verify()` call
  - Fail-fast on unknown ops
- Remove:
  - `extern "C"` block — replaced with `import matlab_runtime as rt` + `import numpy as np`
  - Per-string `unsigned char[]` fallback — Python strings are native UTF-8
  - C type prefixes (`double`, `int64_t`, `void*`) — Python is untyped
  - Function forward declarations — Python doesn't need them

`CMakeLists.txt`:
- Register `EmitPython.cpp` in the `MatlabMLIR` target alongside `EmitC.cpp`.

### Step 3 — Op-by-op mapping (~2 days)

| MLIR | Python output |
|---|---|
| `%c = arith.constant 1.0 : f64` | `v0 = 1.0` |
| `%c = arith.constant 1 : i64` | `v0 = 1` |
| `%s = arith.addf %a, %b` | `v1 = v0 + v2` |
| `arith.mulf / divf / subf` | `*` / `/` / `-` |
| `arith.cmpi / cmpf` | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| `arith.andi / ori / xori` | `&`, `\|`, `^` (on `int`); `and`/`or` for `i1` |
| `arith.select %c, %a, %b` | `v0 = a if c else b` |
| casts (`sitofp`, `fptosi`, etc.) | `float(x)` / `int(x)` / no-op |
| `scf.if %c -> T { yield %a } else { yield %b }` | `v0 = a if c else b` (single-result), else `if/else` block assigning to `v0` |
| `scf.while { cond } do { body }` | `while True: <before>; if not cond: break; <after>` — **or** the simpler `while cond: <body>` when the before-region has no side effects |
| `func.func @foo(%a, %b) -> T` | `def foo(a, b): ...` with `return` on `func.return` |
| `func.call @foo(%a)` | `v0 = foo(a)` |
| `llvm.call @matlab_matmul_mm(%a, %b)` | `v0 = rt.matmul(a, b)` (remap each `matlab_*` symbol) |
| `llvm.call %fp(%a)` (indirect) | `v0 = fp(a)` |
| `llvm.alloca` (scalar slot) | skip entirely — emit the stored value as a plain assignment on `store` |
| `llvm.alloca` (array slot) | `v0 = np.zeros(N)` |
| `llvm.store %v, %p` | if `%p` traces to a scalar slot: `slot_name = v`; if array: `v0[idx] = v` |
| `llvm.load %p` | `slot_name` or `v0[idx]` |
| `llvm.getelementptr %base[%i]` | tracked as `(base, i)` pair; materialized only when load/store consumes it |
| `llvm.mlir.global @s = "..."` | `s = "..."` at module scope |

**Key simplification**: `llvm.alloca` + `store` + `load` for scalar slots
collapses to plain Python variables. Track each alloca's identity; when
the only thing that ever happens to the pointer is `store V` / `load`,
replace with direct assignment. The C emitter keeps the slot because C
needs taking-the-address semantics for `matlab_*` runtime calls —
Python's runtime takes values, not pointers, so the slot is pure
overhead.

**`scf.while` simplification**: if the before-region is a single
`arith.cmp*` feeding `scf.condition`, emit `while <cmp>:` instead of
`while True: ... if not cond: break`. Falls back to the break-form when
the before-region has stores or calls.

### Step 4 — Runtime shim (`runtime/matlab_runtime.py`, ~1 day)

One Python module, one function per `matlab_*` symbol the C runtime
exposes. Majority are NumPy one-liners:

```python
import numpy as np

def zeros(n, m=None):      return np.zeros((n, m) if m else (n,))
def ones(n, m=None):       return np.ones((n, m) if m else (n,))
def matmul(a, b):          return a @ b
def transpose(a):          return a.T
def inv(a):                return np.linalg.inv(a)
def disp(x):               print(x)
def fprintf(fmt, *args):   print(fmt % args, end="")
def length(a):             return max(a.shape) if a.ndim else 1
def size(a, dim=None):     return a.shape[dim-1] if dim else a.shape
```

Harder cases that need actual logic (not one-liners):
- **`matlab_struct_*`** — back with a `dict` subclass or `types.SimpleNamespace`.
- **`matlab_cell_*`** — back with a Python `list`.
- **`matlab_parfor_dispatch`** — `concurrent.futures.ThreadPoolExecutor`, or sequential for v1.
- **Error flag / try-catch** — the C runtime threads an error pointer through every call; in Python this becomes exceptions. The emitter should recognize the error-flag dance (`LowerIO` output) and translate it back to `try` / `except` around the `scf.if` that checks the flag.

Generate the list of runtime symbols to implement by grepping
`llvm.call @matlab_*` across `test/Run/*.m.mlir-post-lowerio` dumps —
anything referenced there needs a Python equivalent.

### Step 5 — Tests (~1 day)

`test/Run/run_tests_emitpython.sh` mirroring `run_tests_emitc.sh`:
- For each `test/Run/*.m`, run `matlabc -emit-python` → `python3 out.py`.
- Diff stdout against the existing `.stdout` golden (byte-for-byte).
- Honor `.sorted` markers (parfor nondeterminism) identically.

`CMakeLists.txt`:
- Add `run-tests-emit-python` CTest target.

`justfile`:
- Add `emit-python FILE` and `compile-python FILE` recipes.
- Extend `test-emitc` or add `test-emitpython`.

## Open questions

1. **Integer type width.** MATLAB's default numeric type is `double`;
   `arith.addi` rarely appears unless from explicit `int32()` casts. For
   the common case we emit Python `int` / `float` and don't worry about
   overflow semantics. If `i32` / `i64` semantics matter for a specific
   test, wrap in `np.int32(...)` / `np.int64(...)` — decide per-test
   rather than globally.

2. **Floating-point formatting.** MATLAB's `disp` of a scalar prints
   `3.1416` where Python's default prints `3.1415926535897931`. The
   runtime's `disp` needs to match MATLAB formatting (5 significant
   digits by default) — implement in `matlab_runtime.py`, not in the
   emitter.

3. **Globals / persistents.** `llvm.mlir.global` with internal linkage
   becomes a module-level Python variable. For `persistent` vars the C
   runtime uses a `matlab_persistent_*` call; the Python runtime can
   back it with a function-attribute (`fn.x = x`) or a module-level
   dict keyed by function name.

## Effort estimate

| Phase | Effort |
|---|--:|
| CLI flag + skeleton compiling | 0.5 day |
| Op mapping (scalars, control flow) | 1 day |
| Slot collapse + single-use inlining port | 0.5 day |
| Runtime shim (common subset) | 1 day |
| Tests passing on `examples/` | 1 day |
| Tests passing on `test/Run/*.m` | 1–2 days |
| **Total** | **~1 week** |

## Non-goals

- Idiomatic Python. The output is generated code, not hand-written
  Python. We optimize for correctness and `-emit-c` parity, not
  PEP 8 style. Single-use inlining gives acceptable readability.
- Performance. NumPy-backed runtime is the ceiling; no JIT, no
  Cython, no type hints. If the output is slow, use `-emit-c` instead.
- Standalone (no-runtime) output. The runtime is ~200 LoC of NumPy
  wrappers; shipping it alongside the `.py` file is fine.

## What changes if we want "direct NumPy" instead of a runtime shim

Alternative: instead of `rt.matmul(a, b)`, emit `a @ b` at the call
site. Produces more readable Python, but:
- Emitter needs a per-runtime-symbol mapping from `matlab_*` name to
  inline Python expression template. ~40 entries, each 1–3 lines.
- Some runtime functions have no clean inline form (struct/cell
  helpers, parfor, formatted I/O). Those still need the shim.
- Doubles the emitter work vs. the shim approach.

Recommendation: ship the shim first (gets tests green), then selectively
inline the ~10 most common runtime calls (`matmul`, `transpose`, elementwise
arithmetic, `zeros`, `ones`) as a readability pass. Hybrid is fine when
the two layers don't overlap.
