# MATLAB Compatibility — Feature Status

Inventory of what this project supports today and what's missing for full
MATLAB language compatibility. Derived from the lexer, parser, AST, Sema,
MLIR passes, runtime, and test corpus as of the current branch.

The project's target is the **practical scalar-and-dense-matrix subset**
of MATLAB: enough to run numeric programs, linear algebra, and
moderately complex control flow. OOP, toolboxes, and GUI features are
out of scope.

---

## Legend

- ✅ Implemented and tested end-to-end (all three backends: LLVM / C / C++)
- 🟡 Partial — parsed and/or modelled in Sema but runtime/lowering incomplete
- ❌ Not supported

---

## 1. Language surface

### Lexical (`include/matlab/Lex/TokenKinds.def` — 100 tokens)

| Feature | Status | Notes |
|---|:-:|---|
| Integer / float / imaginary (`1`, `3.14`, `2i`) literals | ✅ | |
| String literals (`"..."`) and char arrays (`'...'`) | ✅ | Context-sensitive apostrophe handling |
| Escape sequences (`\n \t \r \\ \' \" \0`) | ✅ | |
| Line continuation (`...`) | ✅ | |
| Line and block comments (`%`, `%{ ... %}`) | ✅ | |
| Arithmetic ops (`+ - * / \ ^`) + element-wise (`.* ./ .\ .^`) | ✅ | |
| Comparison, logical, short-circuit (`== ~= < <= > >= & | && || ~`) | ✅ | |
| Transpose and conjugate-transpose (`'`, `.'`) | ✅ | |
| Function handle operator (`@`) | ✅ | |
| OOP keywords (`classdef properties methods events enumeration`) | 🟡 | Tokenized; no parser/sema |
| `spmd`, `import` | 🟡 | Tokenized; `import` parses but is ignored |

### Parser — expressions (`include/matlab/AST/`)

| Feature | Status | Notes |
|---|:-:|---|
| Number / string / char / imaginary literals | ✅ | |
| Identifier, `end` (in index), bare `:` (colon) | ✅ | |
| Binary / unary / postfix operators | ✅ | |
| Range (`a:b`, `a:s:b`) | ✅ | Folds to concrete length at compile time |
| Matrix literal (`[1 2; 3 4]`) with whitespace separators | ✅ | |
| Cell literal (`{a, b}`) | ✅ | 1-D only |
| Call / index (parser-level `CallOrIndex`, resolved by Sema) | ✅ | |
| Cell index (`C{i}`), field access (`s.x`), dynamic field (`s.(name)`) | ✅ | |
| Anonymous function (`@(x) x+1`) with captures | ✅ | Scalar and matrix captures tested |
| Function handle (`@sin`, `@myFunc`) | ✅ | |
| Complex literal arithmetic | 🟡 | Literals parse; arithmetic returns NaN |

### Parser — statements

| Feature | Status | Notes |
|---|:-:|---|
| Expression statement, assignment (incl. multi-LHS `[u,v] = f(x)`) | ✅ | |
| `if / elseif / else / end` | ✅ | |
| `for ... end` (range + step, negative step) | ✅ | |
| `while ... end`, `break`, `continue`, `return` | ✅ | |
| `switch / case / otherwise / end` | ✅ | |
| `try / catch / end` with error binding | ✅ | `catch ME; disp(ME.message)` works |
| `global`, `persistent` | ✅ | Scalar f64 only; 128-entry table |
| `parfor ... end` | ✅ | pthread fan-out + reduction mutex |
| `function` declaration (incl. nested functions, multi-return) | ✅ | |
| Script-mode top-level (no leading `function`) | ✅ | |
| Command syntax (`clear x`) | ✅ | Parser-level sugar to `clear('x')` |
| `import` statement | 🟡 | Parses, not executed |
| `classdef` / OOP | 🟡 | `properties` + `methods`, constructor, inheritance (`< Parent`), static methods, operator overloading (`plus`, `minus`, `mtimes`, `eq`, etc.), `Dependent` properties with `get.Prop` / `set.Prop`, `enumeration` blocks. Missing: value-class copy semantics (all objects handle-shaped), events / listeners, property validators (parsed but not enforced), `handle` destructors. |
| `spmd` | ❌ | |

---

## 2. Semantic analysis (`lib/Sema/`)

| Feature | Status | Notes |
|---|:-:|---|
| Hierarchical scope resolution with 8 binding kinds | ✅ | `Var`, `Param`, `Output`, `Global`, `Persistent`, `Function`, `Builtin`, `Import` |
| `CallOrIndex` disambiguation via binding lookup | ✅ | |
| Forward references across TU-level functions | ✅ | |
| Type inference (fixpoint with CF merges) | ✅ | |
| Shape propagation through slicing, broadcast | ✅ | |
| `nargin` / `nargout` dispatch (multi-return selection) | ✅ | |
| Polymorphic call monomorphization | ✅ | |
| Integer dtype tracking (`int8..int64`, `uint8..uint64`) | 🟡 | Tracked in type lattice; runtime is f64-only |
| Complex dtype tracking | 🟡 | Tracked; no runtime arithmetic |
| N-dim (>2D) rank tracking | 🟡 | Tracked; runtime assumes ≤2D |

---

## 3. Numeric types & values

| Type | Status | Runtime backing |
|---|:-:|---|
| `double` (2-D dense matrix) | ✅ | `matlab_mat { data:f64*, rows, cols }` |
| `logical` | ✅ | Stored as f64 0/1 |
| `char` array (single-quoted) | ✅ | UTF-8 byte array; display supported |
| `string` scalar (double-quoted) | ✅ | |
| `single` | 🟡 | Cast builtin routes to f64 (truncate only) |
| `int8..int64`, `uint8..uint64` | 🟡 | Cast builtins truncate + saturate; storage stays f64 |
| `complex` | ❌ | Imaginary literals lex/parse; arithmetic missing |
| N-D arrays (3-D) | 🟡 | `zeros(m,n,p)` / `ones(m,n,p)` + scalar `A(i,j,k)` read/write, `size(A, 3)`, `numel`, `ndims` |
| N-D arrays (>3D) | ❌ | |
| Sparse matrices | ❌ | |
| `categorical`, `datetime`, `duration`, `table`, `timetable` | ❌ | |

---

## 4. Built-in functions (runtime: `runtime/matlab_runtime.c`)

### Creation & shape

| Function | Status |
|---|:-:|
| `zeros`, `ones`, `eye`, `rand`, `randn`, `magic` | ✅ |
| `diag`, `reshape`, `repmat`, `linspace` | ✅ |
| `size`, `length`, `numel`, `ndims` | ✅ |
| `cat`, `horzcat`, `vertcat` (beyond `[A B]` literal) | ❌ |
| `permute`, `squeeze`, `flip`, `rot90` | ❌ |

### Element-wise math

| Function | Status |
|---|:-:|
| `+ - * / .* ./ .^` on matrix/matrix, matrix/scalar, scalar/matrix | ✅ |
| `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan` | ✅ |
| `floor`, `ceil`, `round`, `fix`, `mod`, `rem` | ✅ |
| `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `log2`, `log10` | ❌ |
| `sign`, `conj`, `real`, `imag`, `angle` | ❌ |

### Reductions

| Function | Status |
|---|:-:|
| `sum` (all elements) | ✅ |
| `min`, `max`, `mean`, `prod` | 🟡 | Registered as builtins; not all runtime-implemented |
| `std`, `var`, `median`, `mode`, `cumsum`, `cumprod` | ❌ |
| Dimension-aware reductions (`sum(A, 2)`) | ❌ |

### Linear algebra

| Function | Status | Notes |
|---|:-:|---|
| `*` (matmul), `mldivide` (`A\b`), `mrdivide` (`A/b`) | ✅ | Pure-C triple-loop + LU |
| `inv`, `det`, `transpose`, `ctranspose` | ✅ | |
| `eig` (symmetric) | ✅ | Jacobi; non-symmetric is symmetrized (approximate) |
| `eig` (non-symmetric, correct) | ❌ | |
| `svd` (singular values only) | 🟡 | `U`, `V` not returned |
| `pinv`, `rank` | 🟡 | Registered; not all wired |
| `qr`, `lu`, `chol`, `schur`, `hess`, `null`, `orth` | ❌ |
| `norm`, `trace`, `kron`, `cross`, `dot` | ❌ |

### Indexing / search

| Operation | Status |
|---|:-:|
| Scalar indexing (`A(i)`, `A(i,j)`) with 1-based, OOB→0 | ✅ |
| Slicing (`A(1:3,:)`, logical mask) | ✅ |
| Indexed store (`A(i)=v`, `A(:,j)=v`) | ✅ |
| `end` in index expressions | ✅ |
| `find`, `isempty`, `isequal` | ✅ |
| Row/column deletion (`A(i,:)=[]`) | 🟡 | Runtime entries exist; frontend pattern not wired |
| `sort`, `sortrows`, `unique`, `ismember` | ❌ |
| Linear vs. subscript indexing edge cases (`sub2ind`, `ind2sub`) | ❌ |

### Heterogeneous data

| Feature | Status | Notes |
|---|:-:|---|
| Struct: scalar, nested (`s.a.b`), `isstruct`, `isfield`, `rmfield` | ✅ | |
| Struct: dynamic field (`s.(name)`) | ✅ | |
| Struct: field-as-matrix (transparent 1×1 boxing) | ✅ | |
| Struct arrays (`s(i).x`) | ❌ | Scalar struct only |
| `fieldnames(s)` | 🟡 | Needs char-matrix dtype |
| Cell: 1-D literal, read/write, `numel`, `iscell` | ✅ | Auto-grows on OOB write |
| Cell: 2-D | ❌ | |
| Cell: concatenation (`{C{:}, x}`) | ❌ | |
| `cellfun`, `arrayfun` (beyond trivial cases) | 🟡 | Registered; not all wired |
| Containers.Map | ❌ | |

### I/O

| Feature | Status |
|---|:-:|
| `disp` (string, scalar, vector, matrix) | ✅ |
| `fprintf` (up to 4 numeric args) with escape sequences | ✅ |
| `sprintf` | 🟡 | Registered; runtime entry missing |
| `input` (numeric) | ✅ |
| `error`, `warning` with message text | ✅ |
| File I/O: `fopen`, `fclose`, `fprintf(fid, ...)`, `fgetl`, `feof`, `fread`, `fwrite`, `save`, `load` | 🟡 | Text + binary single-matrix round-trip work. `save`/`load` use a custom `MLB1` header format, **not** MATLAB's `.mat` format. |
| `readtable`, `writetable`, `readmatrix`, `xlsread` | ❌ |

### Control / system

| Feature | Status |
|---|:-:|
| `error` flag mechanism, try/catch with `ME.message` | ✅ |
| `global`, `persistent` (scalar f64) | ✅ |
| `clear` | ✅ |
| `parfor` with reduction mutex | ✅ |
| `keyboard`, `pause`, `tic`, `toc` | 🟡 | Registered; implementation varies |
| `assert` | ❌ |
| `eval`, `evalin`, `assignin` | ❌ |
| `feval` | 🟡 | Via function handles |

### Strings

| Feature | Status |
|---|:-:|
| String literal creation, `strlen`, `isstring` | ✅ |
| Concatenation (`[s1 s2]`, `strcat`) | 🟡 | `[]` concat works; `strcat` not wired |
| `sprintf`, `strsplit`, `strjoin`, `strtrim`, `strrep`, `regexp`, `regexprep` | ❌ |
| `num2str`, `str2num`, `str2double` | ❌ |
| `upper`, `lower`, `startsWith`, `endsWith`, `contains` | ❌ |

---

## 5. Compilation pipeline

| Stage | Status | Tool |
|---|:-:|---|
| Lexer (context-sensitive) | ✅ | `-dump-tokens` |
| Parser (Pratt + recursive descent) | ✅ | `-dump-ast` |
| Sema (Resolver + type inference) | ✅ | `-emit-sema` |
| Reference IR (in-house, zero-dep) | ✅ | `-emit-mir` |
| MLIR lowering (`matlab`, `func`, `scf`, `arith`) | ✅ | `-emit-mlir` |
| Optimization passes (slot promotion, scalar→arith) | ✅ | `-emit-mlir -opt` |
| LLVM IR emission | ✅ | `-emit-llvm` |
| C emission (self-contained) | ✅ | `-emit-c` |
| C++ emission | ✅ | `-emit-cpp` |
| Python emission | ❌ | See `docs/emit_python.md` |
| SystemC (synthesizable) emission | ❌ | See `docs/emit_systemc.md` |
| JIT / REPL | 🟡 | `matlabc -repl` with MLIR ExecutionEngine; state persists via a runtime workspace. No line editing / JIT cache / live user-function definitions yet. See `docs/repl.md`. |

### MLIR passes (`lib/MLIR/Passes/`)

`SlotPromotion` → `LowerScalarsToArith` → `OutlineParfor` →
`LowerSeqLoops` → `LowerAnonCalls` → `LowerUserCalls` (fixpoint) →
`LowerTensorOps` → `LowerScalarSlots` → `LowerIO`.

All implemented; see `docs/emit_c_cpp.md` for pipeline diagram.

---

## 6. Test corpus

| Suite | Count | Status |
|---|--:|:-:|
| `frontend-tests` (Lexer, Parser, Sema, MIR, MLIR, Opt, Programs, Errors) | 77 | ✅ 77/77 |
| `run-tests` (`-emit-llvm` + clang) | 98 | ✅ |
| `run-tests-emit-c` (`-emit-c` + cc) | 98 | ✅ |
| `run-tests-emit-cpp` (`-emit-cpp` + c++) | 98 | ✅ |
| `run-tests-emit-c-strict` / `-cpp-strict` (-Wall -Wextra -Werror) | 98 | ✅ |
| `emitc-fail-tests` (diagnostic contract) | 1+ | ✅ |

Examples gallery: 14 programs under `examples/` exercise matrix ops,
recursion, anonymous functions, function handles, parfor, linear
algebra, logical masks, struct/cell usage.

---

## 7. Tooling

| Feature | Status |
|---|:-:|
| Compiler CLI (`matlabc`) with 9 emit modes + `-repl` | ✅ |
| CMake + `just` build system | ✅ |
| CTest integration (7 lanes) | ✅ |
| Diagnostics with source-location | ✅ |
| `#line` directives in emitted C / C++ | ✅ |
| REPL / interactive interpreter | 🟡 | JIT via MLIR ExecutionEngine, persistent workspace. `matlabc -repl`. |
| Debugger (DAP) | ❌ | Aids shipped: `dbg(x)` source-located print, `who`/`whos`/`clear` workspace commands, `#line` in emitted C/C++. Full breakpoint/step debugging is blocked on a JIT-level instrumentation pass or a tree-walking interpreter — see `docs/debug.md`. |
| Language Server (LSP) | 🟡 | `matlab-lsp` binary with initialize / shutdown, didOpen / didChange / didClose, publishDiagnostics, definition, documentSymbol. No completion / hover / rename / workspace-symbol yet. See `docs/lsp.md`. |
| Unit-test framework (MATLAB `matlab.unittest`) | ❌ |
| Live Scripts (`.mlx`) | ❌ |
| MEX interop (loading `.mex` files) | ❌ |
| Formatter / linter | ❌ |

---

## 8. What's missing for full MATLAB compatibility

Grouped by category and rough scope. "Full" means matching MathWorks'
MATLAB semantics on a representative program corpus. Some of these are
deliberate non-goals; see "Out of scope."

### Language core (substantial work)

| Missing | Scope | Notes |
|---|---|---|
| **OOP** — `classdef`, properties, methods, events, inheritance, operator overloading | Large | ~6–8 weeks. New AST nodes, new Sema (method dispatch, inheritance), runtime object layout, `dot`-call vs field-access ambiguity |
| **N-dim arrays (>3D)** | Medium | ~2–3 weeks. Runtime descriptor generalization from `(rows, cols, depth)` to `(ndims, shape[])`; update all per-op lowering. 3-D already supported via `matlab_mat3` for `zeros/ones` + scalar indexing |
| **Integer runtime** (`int8..int64`, `uint8..uint64`) | Medium | ~2 weeks. Cast builtins already truncate + saturate against f64 storage; dedicated typed runtime (`matlab_mat_i32`, etc.) still needed for memory-layout fidelity |
| **Complex numbers** | Medium | ~2 weeks. Runtime `matlab_mat_c64`; complex-aware versions of every elementwise op + linalg |
| **Struct arrays** (`s(i).x`) | Medium | ~1 week. Runtime struct-array descriptor; slicing over struct fields |
| **Sparse matrices** | Large | ~3–4 weeks. Sparse representation + sparse-aware linalg; or lean on SuiteSparse |
| **`varargout`** | Small | ~2–3 days. `varargin` ships; `varargout` needs multi-return unpacking at call site |
| **`classdef` dependent types** (`table`, `datetime`, `categorical`) | Large | Built on OOP; add after that |
| **`eval`, `evalin`, `assignin`** | Small | ~2–3 days. Requires REPL / interpreter path — see `docs/repl.md` |

### Built-in library breadth (incremental — adds per function)

Each of these is ~0.5–2 days of runtime work plus test coverage:

- **Trig/exp tail**: `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `log2`, `log10`, `sign`, `conj`, `real`, `imag`, `angle`
- **Reductions**: full `min`/`max`/`mean`/`prod`/`std`/`var`/`median`/`mode`, plus `cumsum`/`cumprod`, plus dimension arg
- **Reshape/layout**: `cat`, `horzcat`, `vertcat`, `permute`, `squeeze`, `flip`, `rot90`
- **Linalg tail**: correct non-symmetric `eig`, full `svd` with `U`/`V`, `pinv`, `rank`, `qr`, `lu`, `chol`, `norm`, `trace`, `kron`
- **Search / sort**: `sort`, `sortrows`, `unique`, `ismember`, `setdiff`, `intersect`, `union`
- **Indexing helpers**: `sub2ind`, `ind2sub`, `A(i,:)=[]` wiring
- **Strings**: `sprintf`, `num2str`, `str2double`, `strsplit`, `strjoin`, `strtrim`, `strrep`, `regexp`, `upper`, `lower`, `startsWith`, `endsWith`, `contains`
- **I/O**: `fopen`/`fread`/`fwrite`/`fclose`, `load`/`save` (.mat format), `readtable`/`writetable` (needs `table` type)

Rough cumulative estimate for "covers 95% of everyday MATLAB code":
**~6–8 weeks of runtime work**, parallelizable.

### Tooling (each standalone)

| Missing | Scope | Reference |
|---|---|---|
| REPL / interpreter | 4 weeks v1 | `docs/repl.md` |
| Debugger (DAP server) | 2–3 weeks | Hooks on top of REPL |
| Language Server (LSP) | 2–3 weeks | Reuses Lexer/Parser/Sema |
| Formatter | 1 week | AST pretty-printer already close |
| Package manager / path | 1 week | `addpath`, `+pkg` directories |

### Out of scope (deliberate non-goals)

- **Plotting / figures / UI** — no graphics backend planned. Reject cleanly.
- **Simulink and toolboxes** (Signal Processing, Image Processing, Control Systems, Statistics, Symbolic Math, etc.) — each is a separate MathWorks product; would require reimplementing thousands of functions.
- **MEX interop** — loading compiled `.mex` files; deep binary-ABI lock-in with MathWorks.
- **Live Scripts** (`.mlx`) — proprietary format; use Jupyter or a documentation toolchain instead.
- **GPU arrays** (`gpuArray`) — would require a CUDA/ROCm backend; out of scope unless specifically prioritized.
- **Code generation toolbox features** (`coder.config`, etc.) — this project *is* a code generator; MATLAB Coder compatibility is a different product.
- **Bit-exact MATLAB numerics** — LAPACK vs. pure-C linear algebra will disagree in the last few ULPs. Correct to tolerance, not to bit.

---

## 9. Rough "fully compatible MATLAB-subset" roadmap

If the goal is to run a majority of general-purpose MATLAB programs
(not toolboxes, not OOP-heavy code, not GUI), the order that gives
the most leverage:

| Priority | Item | Effort | Unlocks |
|:-:|---|--:|---|
| 1 | Dimension-aware reductions + full trig/exp tail | 1 week | Everyday numeric scripts |
| 2 | `varargin`/`varargout` + call-site polish | 1 week | Library-style functions |
| 3 | Struct arrays | 1 week | Data-in-records patterns |
| 4 | Integer runtime (i32/i64 minimum) | 1.5 weeks | Image processing pixel code |
| 5 | N-dim arrays (3D common case) | 2 weeks | Volumetric data, batch dims |
| 6 | `sort`, `unique`, `find` extensions, linalg tail | 1–2 weeks | Data manipulation patterns |
| 7 | String/regex built-ins (`sprintf`, `regexp`, etc.) | 1–2 weeks | Text processing |
| 8 | REPL (see `docs/repl.md`) | 4 weeks | Interactive use |
| 9 | File I/O (`load`/`save` .mat, `fopen` family) | 2 weeks | Real data pipelines |
| 10 | Complex arithmetic | 2 weeks | DSP programs |
| 11 | OOP (`classdef`) | 6–8 weeks | Modern MATLAB code |
| 12 | Sparse matrices | 3–4 weeks | Scientific computing |

Items 1–7 are roughly a **quarter of focused work** and would cover a
large majority of non-OOP MATLAB programs. Items 8–10 round out the
interactive and I/O surface. Items 11–12 are the big remaining land
masses; either is a multi-month project on its own.

---

## 10. Summary

**Where we are:** a production-quality compiler for the scalar-and-
dense-matrix subset of MATLAB, with three backends (LLVM, C, C++), full
control flow, anonymous functions with captures, structs, 1-D cells,
`parfor`, error handling, and a ~200-program test corpus that passes
across all three backends.

**Biggest gaps to a "general-purpose MATLAB replacement":** integer /
complex / N-D runtime, struct arrays, the long tail of built-in
functions, and interactive tooling. Each is tractable; none is
blocking; priorities depend on the target audience.

**Biggest architectural asks:** OOP and sparse matrices. Both are
multi-month projects and neither has started.
