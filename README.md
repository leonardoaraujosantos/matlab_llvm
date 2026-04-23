# matlab_llvm

A compiler from a practical subset of MATLAB to native executables,
with three backends — **LLVM IR**, **portable C**, and **portable C++**
— all producing identical stdout. Built end-to-end: lexer → parser →
AST → semantic analysis → in-house SSA IR → MLIR (`matlab` + `func` +
`scf` + `arith` + `llvm` dialects) → codegen.

No MathWorks source, no Octave dependency, no numerics library
dependency. Just C++20, MLIR 22.x, and a ~1700-line C runtime shim
(libc + pthreads + a heap-allocated `matlab_mat` descriptor, plus
`matlab_struct` / `matlab_cell`, plus matmul / inverse / solve / SVD /
eig implemented inline) that compiles stand-alone.

## Runs today

```matlab
x = 0;
parfor i = 1:10
    x = x + i;
end
disp(x);        % 55 — parallel sum reduction, mutex-guarded atomic add
```

```matlab
% Linear algebra in pure C — no BLAS, no LAPACK.
A = [4 3; 6 3];  b = [7; 9];
disp(A \ b);                         % [1; 1]  — LU with partial pivoting
disp(det(A));    disp(inv(A));       % -6, Gauss-Jordan via LU
[V, D] = eig([2 -1 0; -1 2 -1; 0 -1 2]);
disp(V * D * V');                    % reconstructs A (Jacobi, symmetric)
```

```matlab
% Anonymous functions, user-function handles, builtin handles.
k = 5;
f = @(x) x + k;     % scalar by-value capture at @-time
g = @sq;            % user-function handle
h = @sin;           % builtin handle
disp(f(3));  disp(g(6));  disp(h(0));     % 8, 36, 0
function y = sq(x), y = x * x; end
```

More in [`examples/`](examples/) — every file there compiles and runs
under the current compiler end-to-end.

## Building

Prerequisites: LLVM 22.x + MLIR (tested with Homebrew `llvm@22.1.3` on
macOS arm64), CMake ≥ 3.20, Ninja, C++20 compiler.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Or via [just](https://github.com/casey/just) (recipes in `justfile`):

```bash
just build                 # configure + ninja
just test                  # all ctest suites
just compile FILE OUT      # .m → native executable (LLVM path)
just compile-c FILE        # .m → native executable (C path)
just examples              # build and run every examples/*.m
just --list                # full recipe list
```

Frontend-only build (no MLIR — just Lexer/Parser/AST/Sema/MIR):

```bash
cmake -S . -B build -G Ninja -DMATLAB_LLVM_WITH_MLIR=OFF
```

## Architecture

```mermaid
flowchart LR
  src["foo.m"] --> FE["Frontend<br/>Lexer · Parser ·<br/>AST · Sema"]
  FE --> MLIR["MLIR module<br/>matlab + func + scf +<br/>arith + llvm dialects"]
  MLIR --> Passes["MLIR passes<br/>slot prom · scalar→arith ·<br/>parfor · user calls ·<br/>anon calls · tensor ops ·<br/>I/O · scalar slots"]
  Passes --> LL[LLVM IR]
  Passes --> CSrc[C / C++ source]
  LL --> Exe1["executable<br/>clang + runtime"]
  CSrc --> Exe2["executable<br/>cc / c++ + runtime"]
  RT[runtime/matlab_runtime.c] -.-> Exe1
  RT -.-> Exe2
```

The frontend has no external dependencies. The in-house MIR
(`lib/MIR/`) is kept as a reference/diagnostic IR (`-emit-mir`) —
production codegen flows through MLIR. The runtime is single-file C,
library-agnostic by design: every matrix op has an in-tree
implementation so the whole stack stays transpilable. The tradeoff is
performance (naive O(N³) matmul vs. OpenBLAS), not correctness.

`parfor` compiles to a `pthread`-per-iteration fan-out with a
mutex-guarded atomic-add entry for reductions, so `x = x + i` across
10 threads deterministically prints 55.

Design docs:
- [`docs/emit_c_cpp.md`](docs/emit_c_cpp.md) — the C / C++ backend: op-to-C mapping, runtime ABI bridge, design alternatives considered.
- [`docs/feature_status.md`](docs/feature_status.md) — complete feature inventory and gap analysis for full MATLAB compatibility.
- [`docs/emit_python.md`](docs/emit_python.md) — planned Python backend.
- [`docs/emit_systemc.md`](docs/emit_systemc.md) — planned SystemC (synthesizable) backend.
- [`docs/repl.md`](docs/repl.md) — planned interactive interpreter / REPL.

## CLI

One driver, many stages:

| Flag | Produces |
|---|---|
| `-dump-tokens` | Flat token stream |
| `-dump-ast` | Pretty-printed AST |
| `-emit-sema` | AST annotated with resolved bindings and inferred types |
| `-emit-mir` | In-house SSA IR (MLIR-shaped, no external deps) |
| `-emit-mlir` | Real MLIR module |
| `-emit-mlir -opt` | After slot promotion + scalar-to-arith |
| `-emit-llvm` | LLVM IR text |
| `-emit-c` | Self-contained C source (links with `runtime/matlab_runtime.c`) |
| `-emit-cpp` | Self-contained C++ source (same runtime via `extern "C"`) |

Build and run, via any of the three backends:

```bash
# LLVM path (one-shot helper)
runtime/build_and_run.sh path/to/foo.m           # → ./foo

# Or manually via LLVM IR
build/matlabc -emit-llvm foo.m > foo.ll
clang foo.ll runtime/matlab_runtime.c -o foo

# Or via the C path (no LLVM needed at compile time)
build/matlabc -emit-c foo.m > foo.c
cc foo.c runtime/matlab_runtime.c -o foo -lm -lpthread

# Or via the C++ path
build/matlabc -emit-cpp foo.m > foo.cpp
c++ -x c++ foo.cpp -x c runtime/matlab_runtime.c -o foo -lm -lpthread
```

All three backends produce stdout that matches byte-for-byte on the
98-program test corpus.

## Features

See [`docs/feature_status.md`](docs/feature_status.md) for the
authoritative inventory. Short version:

**Supported:** numeric scalars and 2-D dense matrices (f64); 3-D
arrays via `zeros(m,n,p)` / `ones(m,n,p)` with scalar `A(i,j,k)`
read/write; integer cast builtins (`int8` / `int16` / `int32` /
`int64` / `uint8` / `uint16` / `uint32` / `uint64` / `single` /
`double` / `logical`) with MATLAB-style truncate + saturate
semantics; all standard arithmetic / comparison / logical /
element-wise operators; control flow (`if` / `elseif` / `else` /
`for` / `while` / `switch` / `try` / `break` / `continue` /
`return`); `parfor` with atomic-add reductions; user-defined
functions with multi-return and recursion; polymorphic call
monomorphization (per-callsite `nargin` / `nargout`); `varargin` with
call-site cell packing; anonymous functions with captures; function
handles (`@sin`, `@myFunc`, `@(x) x+k`); structs (nested fields,
dynamic `s.(name)`, `isstruct` / `isfield` / `rmfield`); 1-D cell
arrays; real string type (`"..."`, `+`, `disp`, `strlen`,
`isstring`); `global` / `persistent`; error flag + `catch ME;
ME.message`; implicit display; command syntax; minimum `classdef`
(handle-shaped objects, `properties`, `methods`, constructor with
`nargin`, property read/write, dot-method dispatch).

Runtime built-ins include: linear algebra (`*`, `\`, `/`, `inv`,
`det`, `svd`-values, `eig` for symmetric matrices); constructors
(`zeros`, `ones`, `eye`, `magic`, `rand`, `randn`, `linspace`); shape
ops (`transpose`, `diag`, `reshape`, `repmat`); reductions (`sum`);
element-wise math (`exp`, `log`, `sin`, `cos`, `tan`, `sqrt`, `abs`);
predicates (`isempty`, `isequal`, `find`); I/O (`disp`, `fprintf`
up to 4 args, `input`, `error`, `warning`).

**Not yet:** `classdef` / OOP, struct arrays, 2-D cells,
`varargout`, complex numbers, 3-D vector slicing (only scalar
`A(i,j,k)` today), 4-D+ arrays, sparse matrices, non-symmetric
`eig`, full `[U, S, V] = svd(A)`, `fft` family, string functions
(`sprintf`, `regexp`, `num2str`, etc.), file I/O, REPL, debugger.

**Not planned:** plotting, Simulink, toolboxes, GPU arrays, live
scripts (`.mlx`), MathWorks bit-exact numerics.

## Testing

Multiple CTest suites, ~560 goldens. The end-to-end `Run` lane
compiles 98 `.m` programs through all three backends (LLVM, C, C++,
plus `-Wall -Wextra -Werror` strict lanes for the C/C++ paths) and
diffs stdout against `.stdout` goldens:

```bash
ctest --test-dir build
```

To regenerate goldens after an intentional change:

```bash
UPDATE=1 test/run_tests.sh build/matlabc
```

## Repo layout

```
include/matlab/
  Basic/  Lex/  Parse/  AST/  Sema/  MIR/  MLIR/
lib/                           implementations mirror include/
tools/matlabc/                 CLI driver (all flags wired in main.cpp)
runtime/                       matlab_runtime.c + build_and_run.sh
test/                          goldens + run scripts (per-suite subdirs)
examples/                      end-to-end example programs
docs/                          design docs (see Architecture section)
justfile                       task runner
```
