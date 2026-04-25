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
| `%c = arith.constant 1.0 : f64` | `1.0` (shortest round-tripping repr; `0.05`, not `0.050000000000000003`) |
| `%c = arith.constant 1 : i64` | `1` |
| `%s = arith.addf %a, %b` | `v1 = a + b` |
| `arith.mulf / divf / subf` | `*` / `/` / `-` |
| `arith.cmpi / cmpf` | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| `arith.andi / ori / xori` | `&`, `\|`, `^` (on `int`); `and` / `or` / `!=` for `i1` |
| `arith.select %c, %a, %b` | `v0 = a if c else b` |
| casts (`sitofp`, `fptosi`, etc.) | `float(x)` / `int(x)` / no-op |
| `scf.if %c -> T { yield %a } else { yield %b }` | `if c: v0 = a` / `else: v0 = b` (no `v0 = 0` predeclaration; the yields cover both paths) |
| `scf.if` with statically-known `c` | only the live branch is emitted; e.g. polymorphic dispatch's `if 2 == 2` collapses |
| `scf.while` matching `for i = a:s:b` | `for i in range(a, b+1, s):` (or `rt.frange(a, b, s)` for non-integer bounds) |
| `scf.while` (general) | `while cond: <body>` — or `while True: ...; if not cond: break; ...` when the before-region has work |
| `func.func @foo(%a, %b) -> T` | `def foo(a, b):` (param names taken from `matlab.name`); `pass` for empty bodies |
| `func.call @foo(%a)` | `foo(a)` (inlined into the use site when single-use) |
| `llvm.call @matlab_matmul_mm(%a, %b)` | `rt.matmul_mm(a, b)` (every `matlab_*` symbol drops its prefix) |
| `llvm.call @matlab_disp_str(%p, %len)` | `rt.disp_str(...)` — the `(ptr, length)` C ABI tail is dropped, since Python strings carry their own length |
| `llvm.call %fp(%a)` (indirect) | `v0 = fp(a)` |
| `llvm.alloca` (scalar slot) | skipped — load/store emit as plain Python `name = ...` |
| `llvm.alloca` (matrix-literal array) | `slot = [v0, v1, ...]` (the GEP+store pairs that filled it are folded in) |
| `llvm.store %v, %p` | scalar slot: `name = v`; array element: `name[idx] = v` |
| `llvm.load %p` | scalar slot: bare `name`; array element: `name[idx]` |
| `llvm.getelementptr %base[%i]` | tracked as `(base, i)` pair; materialized only at the load/store consumer |
| `llvm.mlir.global @s = "..."` | inlined as a Python string literal at every `addressof` use site (no top-of-file declaration) |

Emitter behavior worth knowing:
- **Strings inline**: `rt.disp_str("Hello")` (or `print("Hello")`) directly,
  instead of a top-of-file `__matlab_str0 = "Hello"` declaration referenced
  by `rt.disp_str(__matlab_str0, 13)`. When the argument is a string literal,
  the `rt.disp_str` call collapses further to `print(...)` — byte-equivalent
  output, but reads as plain Python.
- **Scalar `disp` substitution**: `rt.disp_f64(x)` collapses to
  `print(f'{x:g}')`. Output matches MATLAB's `%g` format for every finite
  value; the runtime path stays for matrices (`disp_mat`) where MATLAB's
  multi-column alignment rules don't have a clean f-string equivalent.
- **Conditional imports**: the `import matlab_runtime as rt` and
  `import numpy as np` lines are added only when the emitted body
  actually references those modules. A pure-arithmetic program like
  `examples/factorial.m` emits zero `import` lines.
- **Matrix `disp` substitution**: `rt.disp_mat(M)` collapses to plain
  `print(M)`. Numpy's bracket / dotted-float matrix repr diverges from
  MATLAB's right-aligned `%7g` columns, so the Python lane uses
  per-test `<name>.stdout-python` overrides for tests that print
  matrices; the C / C++ goldens stay on the shared `<name>.stdout`.
  Programs that only do matrix display now drop the `import
  matlab_runtime as rt` line entirely (e.g. `examples/matrix_mult.py`).
- **Numpy rewrite for matrix builtins**: the matrix subset of the
  runtime ABI is emitted as inline numpy / Python-operator expressions
  rather than `rt.<helper>` calls. Specifically:

  | Runtime call | Emitted as |
  |---|---|
  | `matlab_matmul_mm(A, B)`  | `A @ B`           |
  | `matlab_add_mm/ms/sm`     | `A + B`           |
  | `matlab_sub_mm/ms/sm`     | `A - B`           |
  | `matlab_emul_mm/ms/sm`    | `A * B`           |
  | `matlab_ediv_mm/ms/sm`    | `A / B`           |
  | `matlab_transpose(A)`     | `A.T`             |
  | `matlab_inv(A)`           | `np.linalg.inv(A)`     |
  | `matlab_det(A)`           | `np.linalg.det(A)`     |
  | `matlab_mldivide_mm(A, B)`| `np.linalg.solve(A, B)`|
  | `matlab_norm(A)`          | `np.linalg.norm(A)`    |
  | `matlab_trace(A)`         | `np.trace(A)`          |
  | `matlab_zeros(m[, n])`    | `np.zeros((m, n))`     |
  | `matlab_ones(m[, n])`     | `np.ones((m, n))`      |
  | `matlab_eye(n[, m])`      | `np.eye(n)` / `np.eye(n, m)` |
  | `matlab_mat_from_buf(buf, m, n)` | `np.array(buf).reshape(m, n)` |
  | `matlab_sqrt_m / exp_m / sin_m / …` | `np.sqrt(A)` / `np.exp(A)` / `np.sin(A)` / … |

  MATLAB-semantics helpers (`slice1` / `slice2`, MATLAB-style
  column-major reductions like `sum` / `mean` on a 2-D matrix,
  `disp_mat`'s right-aligned `%7g` formatting, `eig` / `lu` / `qr` /
  `svd` whose return shapes don't map 1:1 to numpy) stay on the
  runtime path — they don't have a clean one-line numpy equivalent.
- **For loops**: `scf.while` ops produced by `LowerSeqLoops::lowerForOp` collapse
  to native `for i in range(...):`. When init/end/step are integer literals,
  Python's `range` is used; otherwise the `rt.frange(start, end, step)` generator
  preserves MATLAB's inclusive `start:step:end` semantics.
- **`elif` chaining**: an `scf.if` whose else-region is a single nested `scf.if`
  (the shape MATLAB `elseif` lowers to) emits as `elif <cond>:`, walking the
  cascade down to a final `else:` block. Result-bearing `scf.if` chains have
  their result SSA values aliased so each branch writes to the shared local.
- **Static-condition folding**: `scf.if` whose condition is a constant-vs-constant
  comparison (e.g. the `if nargin == 2` baked in by polymorphic dispatch) emits
  only the live branch.
- **`select(c, 1.0, 0.0)` fold**: emitted as `float(c)` rather than the literal
  ternary `1.0 if c else 0.0` (covers MATLAB's logical-to-double coercion).
- **Pure-read inlining**: a single-use `rt.obj_get_f64` (or another helper on
  the known-pure allowlist: `size`, `numel`, `length`, `ndims`, `isempty`, …)
  inlines past peer pure-read calls. `Savings.interest` collapses to a single
  return line: `return rt.obj_get_f64(obj, "Balance") * rt.obj_get_f64(obj, "Rate")`.
- **Scalar slot collapse**: alloca + store + load patterns become plain Python
  variables. The `slot = 0` pre-declaration is dropped when the slot is
  unconditionally written before any read, including via an `scf.if` whose
  both branches store into the slot (covers MATLAB return-slot patterns).
- **Trailing `return`**: dropped when it sits at the end of a void function;
  Python's implicit return covers it.
- **String-length operands dropped**: runtime helpers whose C ABI takes
  `(ptr, i64-length)` (`disp_str`, `fprintf_str`, `fprintf_f64*`, `input_num`)
  or `(name_ptr, name_len)` (`obj_get_f64`, `obj_set_f64`) lose the length
  operand at the call site. The Python runtime stubs make the length parameter
  optional so hand-written callers still work.
- **break / continue un-lowering**: the frontend's `did_break` / `did_continue`
  flag slots are re-lowered to native Python `break` / `continue`, with the
  guarding `& !did_break` conjunct stripped from the loop condition.
- **Per-function name scope**: each `def` gets a fresh identifier table so
  parameters keep their natural names (`x`, `i`) instead of accumulating
  `_2`, `_3` suffixes from earlier functions.
- **Top-level script body** is hoisted to module scope (mirrors the
  `LowerIO` rename of `@script` to `@main`).
- **Comments and blank lines** from the MATLAB source propagate to the
  emitted Python, similar to the C/C++ emitter.

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

## Sample output

```matlab
% factorial.m
function y = fact(n)
  if n <= 1
    y = 1;
  else
    y = n * fact(n - 1);
  end
end
disp("fact(1..6):");
for i = 1:6
  disp(fact(i));
end
```

emits (note no `import` lines — this body needs neither the runtime
shim nor numpy):

```python
def fact(n):
    if n <= 1.0:
        y = 1.0
    else:
        y = n * fact(n - 1.0)
    return y

print("fact(1..6):")
for i in range(1, 7):
    print(f'{fact(i):g}')
```

A second example showing `elseif` chaining and `select(c, 1.0, 0.0)` folding,
from `examples/traffic_action.m`:

```python
def traffic_action(color, is_emergency):
    if is_emergency != 0.0:
        a = 9.0
    elif color < 1.0:
        a = 0.0
    elif color > 3.0:
        a = 0.0
    elif color == 1.0:
        a = 1.0
    elif color == 2.0:
        a = 2.0
    else:
        a = 3.0 if color == 3.0 else 0.0
    return a
```

A third example, `examples/matrix_mult.m`, showing the numpy rewrite —
no `matlab_runtime` import is needed at all because matrix display also
collapses to plain `print(M)`:

```python
import numpy as np

slot = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
v0 = np.array(slot).reshape(3, 3)

slot_2 = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
v1 = np.array(slot_2).reshape(3, 3)

print("A * B =")
print(v0 @ v1)

print("A .* B =")
print(v0 * v1)

print("A' =")
print(v0.T)
```

And from `examples/bank_account.m`, showing pure-read inlining and the
`select(c, 1.0, 0.0)` fold collapsing the body of `eq` to one line:

```python
def BankAccount__get_Overdrawn(obj):
    return float(rt.obj_get_f64(obj, "Balance") < 0.0)

def BankAccount__eq(a, b):
    return float(rt.obj_get_f64(a, "Id") == rt.obj_get_f64(b, "Id"))

def Savings__interest(obj):
    return rt.obj_get_f64(obj, "Balance") * rt.obj_get_f64(obj, "Rate")
```

## Limitations

This backend should be treated as experimental.

Known constraints:
- some numerically sensitive cases are skipped in the Python lane
- parity is strongest on common scalar, matrix, control-flow, and
  runtime-library programs, not on every edge case
- the generated Python aims to read like the natural translation of the
  MATLAB source — not perfectly idiomatic hand-written Python, but close
  enough to inspect and modify
- if you need the most mature code generation path, prefer `-emit-c`,
  `-emit-cpp`, or `-emit-llvm`
