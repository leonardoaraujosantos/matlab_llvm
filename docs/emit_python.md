# Python Emission

`-emit-python` is an implemented backend that lowers the shared MLIR
pipeline into runnable Python source. It is the most experimental codegen
path in the repository, but it is no longer a plan-only design.

## Scope

Driver flag:

| Flag | Output |
|---|---|
| `-emit-python` | Self-contained `.py` source |

The generated file imports [`runtime/matlab_runtime.py`](../runtime/matlab_runtime.py),
which is a NumPy-backed runtime shim. No LLVM toolchain or C compiler is
needed at runtime:

```bash
build/matlabc -emit-python foo.m > foo.py
PYTHONPATH=runtime python3 foo.py
```

You can also use:

```bash
just emit-python examples/hello.m
just compile-python examples/hello.m
just test-emitpython
```

## Status

What exists today:
- CLI support in `matlabc`
- MLIR-to-Python emitter in [`lib/MLIR/Passes/EmitPython.cpp`](../lib/MLIR/Passes/EmitPython.cpp)
- Python runtime shim in [`runtime/matlab_runtime.py`](../runtime/matlab_runtime.py)
- CMake and `just` integration
- dedicated execution tests in [`test/Run/run_tests_emitpython.sh`](../test/Run/run_tests_emitpython.sh)

Current positioning:
- useful for inspection, portability, and rapid execution without a C toolchain
- not as mature as the LLVM / C / C++ paths
- some numerically sensitive programs are intentionally skipped in the
  Python test lane

## Architecture reuse

Everything up to `runLowerIO` is shared with the other backends. The
pipeline diverges at the same fork point as `-emit-c` and `-emit-cpp`:

```
AST ──► MLIR ──► [SlotPromotion, ..., LowerIO]
                       │
                       ├──► [scf→cf, arith→llvm, ...] ──► .ll
                       ├──► emitC()       ──► .c / .c++
                       └──► emitPython()  ──► .py
```

The emitter targets a small closed MLIR subset and remaps runtime calls
to `import matlab_runtime as rt`.

## Behavior

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

Important emitter behavior:
- scalar `alloca` + `store` + `load` patterns collapse into plain Python locals
- many runtime calls become `rt.<name>(...)`
- top-level script bodies are hoisted to module scope
- break and continue flags introduced by lowering are reconstructed into
  native Python `break` and `continue`
- preserved comments and line markers are supported, similar to the C/C++
  emitter

## Runtime Model

The backend relies on a Python runtime shim rather than trying to inline
all semantics directly into emitted code.

Examples from [`runtime/matlab_runtime.py`](../runtime/matlab_runtime.py):

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

The runtime covers:
- matrix creation and arithmetic through NumPy
- display and formatting compatibility helpers
- structs and cells via Python containers
- file and string utilities
- portions of `parfor`, handles, and runtime error plumbing

## Testing

The Python backend has its own execution lane:

```bash
just test-emitpython
```

That runner:
- emits Python from every `test/Run/*.m`
- runs the result with `python3`
- diffs stdout against the same golden files used by other backends
- honors `.sorted` markers for nondeterministic `parfor` output
- honors `.skip-emit-python` markers for known divergences

## Limitations

This backend should be treated as experimental.

Known constraints:
- some numerically sensitive cases are skipped in the Python lane
- parity is strongest on common scalar, matrix, control-flow, and
  runtime-library programs, not on every edge case
- the generated Python is intended to be readable enough to inspect, not
  idiomatic hand-written Python
- if you need the most mature code generation path, prefer `-emit-c`,
  `-emit-cpp`, or `-emit-llvm`
