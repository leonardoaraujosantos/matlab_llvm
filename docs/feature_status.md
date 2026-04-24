# MATLAB Compatibility — Feature Status

Inventory of what this project supports today and what's missing for full
MATLAB language compatibility. Derived from the lexer, parser, AST, Sema,
MLIR passes, runtime, and test corpus as of the current branch.

The project's target is the **practical scalar / matrix / classdef
subset** of MATLAB: enough to run numeric programs, linear algebra,
control flow, text processing, file I/O, user-defined functions and
user-defined classes (with inheritance and operator overloading),
and to surface all of that through three compiled backends (LLVM,
C, C++), a JIT-backed REPL, an editor-facing Language Server, and
a source formatter. Toolboxes, plotting, and GUI features are
explicitly out of scope.

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
| `horzcat`, `vertcat` (as builtins + `[A B]` / `[A;B]` literal forms) | ✅ |
| `permute` (2-D identity / transpose), `squeeze` (2-D no-op), `flip` / `fliplr` / `flipud`, `rot90` | ✅ |
| `cat` (N-dim), `permute` (>2D) | ❌ |

### Element-wise math

| Function | Status |
|---|:-:|
| `+ - * / .* ./ .^` on matrix/matrix, matrix/scalar, scalar/matrix | ✅ |
| `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan` | ✅ |
| `floor`, `ceil`, `round`, `fix`, `mod`, `rem` | ✅ |
| `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `log2`, `log10`, `sign` | ✅ |
| `conj`, `real`, `imag`, `angle` | ❌ | Gated behind complex-number runtime |

### Reductions

| Function | Status |
|---|:-:|
| `sum` (all elements, column-wise, or `sum(A, dim)`) | ✅ |
| `min`, `max`, `mean`, `prod` (same 3 forms as `sum`) | ✅ |
| `cumsum`, `cumprod` (single-arg + `(A, dim)`) | ✅ |
| Dimension-aware reductions (`sum(A, 2)`, `mean(A, 1)`, ...) | ✅ |
| `std`, `var`, `median`, `mode` | ❌ |

### Linear algebra

| Function | Status | Notes |
|---|:-:|---|
| `*` (matmul), `mldivide` (`A\b`), `mrdivide` (`A/b`) | ✅ | Pure-C triple-loop + LU |
| `inv`, `det`, `transpose`, `ctranspose` | ✅ | |
| `eig` (symmetric, 1- or 2-return `[V, D] = eig(A)`) | ✅ | Jacobi; non-symmetric is symmetrized (approximate) |
| `lu` (partial pivoting, 2-return `[L, U] = lu(A)`) | ✅ | |
| `qr` (Gram-Schmidt, 2-return `[Q, R] = qr(A)`) | ✅ | m ≥ n |
| `chol` (upper-triangular R with R'R = A) | ✅ | SPD-only; error flag on non-SPD input |
| `pinv` (via normal equations) | ✅ | Full-rank square / tall / wide |
| `norm` (Frobenius), `trace`, `kron` | ✅ | |
| `eig` (non-symmetric, correct) | ❌ | |
| `svd` (singular values only) | 🟡 | `U`, `V` not returned |
| `rank`, `schur`, `hess`, `null`, `orth`, `cross`, `dot` | ❌ |

### Indexing / search

| Operation | Status |
|---|:-:|
| Scalar indexing (`A(i)`, `A(i,j)`) with 1-based, OOB→0 | ✅ |
| Slicing (`A(1:3,:)`, logical mask) | ✅ |
| Indexed store (`A(i)=v`, `A(:,j)=v`) | ✅ |
| `end` in index expressions | ✅ |
| `find`, `isempty`, `isequal` | ✅ |
| `sort` (column-wise + vector), `sortrows` (stable lex) | ✅ |
| `unique`, `ismember` | ✅ |
| `setdiff`, `intersect`, `union` | ✅ |
| `sub2ind`, `ind2sub` (column-major, matching MATLAB's user-visible convention) | ✅ |
| Row/column deletion (`A(i,:)=[]`) | 🟡 | Runtime entries exist; frontend pattern not wired |

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
| `sprintf` (literal + single-f64 form) | ✅ | Result is a `matlab_string` |
| `input` (numeric) | ✅ |
| `error`, `warning` with message text | ✅ |
| File I/O: `fopen`, `fclose`, `fprintf(fid, ...)`, `fgetl`, `feof`, `fread`, `fwrite`, `save`, `load` | 🟡 | Text + binary single-matrix round-trip work. `save`/`load` use a custom `MLB1` header format, **not** MATLAB's `.mat` format. |
| `readtable`, `writetable`, `readmatrix`, `xlsread` | ❌ |

### Control / system

| Feature | Status |
|---|:-:|
| `error` flag mechanism, try/catch with `ME.message` | ✅ |
| `global`, `persistent` (scalar f64) | ✅ |
| `clear` (all or named; function + command syntax; REPL-aware) | ✅ |
| `who`, `whos` (REPL workspace introspection) | ✅ |
| `dbg(x)` / `dbg(x, 'label')` — source-located debug print | ✅ |
| `assert(cond)` / `assert(cond, msg)` | ✅ | Sets the runtime error flag |
| `parfor` with reduction mutex | ✅ |
| `keyboard`, `pause`, `tic`, `toc` | 🟡 | Registered; implementation varies |
| `eval`, `evalin`, `assignin` | ❌ |
| `feval` | 🟡 | Via function handles |

### Strings

| Feature | Status |
|---|:-:|
| String literal creation, `strlen`, `isstring` | ✅ |
| Concatenation: `[s1 s2]`, `strcat(a, b)`, `s1 + s2` | ✅ |
| `sprintf` (literal + single-f64 form), `num2str`, `str2double` | ✅ |
| `strtrim`, `strrep` | ✅ |
| `upper`, `lower`, `startsWith`, `endsWith`, `contains` | ✅ |
| `strsplit`, `strjoin`, `regexp`, `regexprep`, `str2num` | ❌ |

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
| C++ emission (classes + inheritance preserved) | ✅ | `-emit-cpp` |
| Source formatter (AST pretty-printer) | ✅ | `-format` |
| JIT / REPL | 🟡 | `matlabc -repl` with MLIR ExecutionEngine; state persists via a runtime workspace. No line editing / JIT cache / live user-function definitions yet. See `docs/repl.md`. |
| Python emission | ❌ | See `docs/emit_python.md` |
| SystemC (synthesizable) emission | ❌ | See `docs/emit_systemc.md` |

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
| `run-tests` (`-emit-llvm` + clang) | 118 | ✅ |
| `run-tests-emit-c` (`-emit-c` + cc) | 118 | ✅ |
| `run-tests-emit-cpp` (`-emit-cpp` + c++) | 118 | ✅ |
| `run-tests-emit-c-strict` / `-cpp-strict` (-Wall -Wextra -Werror) | 118 | ✅ |
| `emitc-fail-tests` (diagnostic contract) | 1+ | ✅ |

Examples gallery: 15 programs under `examples/` exercise matrix ops,
recursion, anonymous functions, function handles, parfor, linear
algebra, logical masks, struct/cell usage, and OOP (`bank_account.m`
— classdef with inheritance, `Dependent` properties, operator
overloading).

---

## 7. Tooling

| Feature | Status |
|---|:-:|
| Compiler CLI (`matlabc`) with 9 emit modes + `-format` + `-repl` | ✅ |
| CMake + `just` build system | ✅ |
| CTest integration (7 lanes) | ✅ |
| Diagnostics with source-location | ✅ |
| `#line` directives in emitted C / C++ | ✅ |
| Formatter (AST pretty-printer, idempotent) | ✅ | `matlabc -format` / `just format`. Drops comments (not in AST). |
| REPL / interactive interpreter | 🟡 | JIT via MLIR ExecutionEngine, persistent workspace, implicit display, `who`/`whos`/`clear`. `matlabc -repl`. See `docs/repl.md`. |
| Language Server (LSP) | 🟡 | `matlab-lsp` binary: initialize/shutdown, didOpen/didChange/didClose, publishDiagnostics, definition, documentSymbol. No completion / hover / rename / workspace-symbol yet. See `docs/lsp.md`. |
| Debugger (DAP) | 🟡 | `matlabc -dap FILE.m` speaks the full Debug Adapter Protocol over stdio: breakpoints (`setBreakpoints`), step (`next`/`stepIn`/`stepOut`), stack trace, `Locals` scope via the workspace snapshot, stdout forwarded as `output` events, clean `disconnect`. Plus the lightweight aids: `dbg(x)` prints to stderr, `who`/`whos`/`clear` list and purge the workspace, `#line` directives in emitted C / C++ so gdb/lldb step `.m` source. Deferred: pushing a stack frame on user-function entry (single `<script>` frame for now), `setVariable`, `evaluate`, conditional breakpoints. See `docs/debug.md`. |
| Unit-test framework (MATLAB `matlab.unittest`) | ❌ |
| Live Scripts (`.mlx`) | ❌ |
| MEX interop (loading `.mex` files) | ❌ |
| Linter (style / unused-var warnings) | ❌ |

---

## 8. What's missing for full MATLAB compatibility

Grouped by category and rough scope. "Full" means matching MathWorks'
MATLAB semantics on a representative program corpus. Some of these are
deliberate non-goals; see "Out of scope."

### Language core (substantial work still open)

| Missing | Scope | Notes |
|---|---|---|
| **OOP value-class copy semantics** | Medium | ~1–2 weeks. Every object is handle-shaped today. True value semantics needs copy-on-assign / copy-on-modify plumbing at every `obj.prop = ...` and every call-site pass. |
| **OOP events / listeners** | Medium | ~1 week. `notify` / `addlistener` / callback machinery. |
| **OOP property validators** (`{mustBeNumeric}`, size specs) | Small | ~2–3 days. Syntax parses today; need runtime checks at each assignment. |
| **N-dim arrays (>3D)** | Medium | ~2–3 weeks. Runtime descriptor generalization from `(rows, cols, depth)` to `(ndims, shape[])`; update all per-op lowering. 3-D already supported via `matlab_mat3` for `zeros/ones` + scalar indexing. |
| **3-D slicing** (`A(:,:,k)`) | Small | ~2–3 days. 3-D exists for scalar `A(i,j,k)`; vector / slice forms not wired. |
| **Integer runtime** (`int8..int64`, `uint8..uint64`) | Medium | ~2 weeks. Cast builtins already truncate + saturate against f64 storage; dedicated typed runtime still needed for memory-layout fidelity. |
| **Complex numbers** | Medium | ~2 weeks. Runtime `matlab_mat_c64`; complex-aware versions of every elementwise op + linalg. |
| **Struct arrays** (`s(i).x`) | Medium | ~1 week. Runtime struct-array descriptor; slicing over struct fields. |
| **Sparse matrices** | Large | ~3–4 weeks. Sparse representation + sparse-aware linalg; or lean on SuiteSparse. |
| **`varargout`** | Small | ~2–3 days. `varargin` ships; `varargout` needs multi-return unpacking at call site. |
| **`classdef` dependent types** (`table`, `datetime`, `categorical`) | Large | Built on OOP; add after value semantics land. |
| **`eval`, `evalin`, `assignin`** | Small | ~2–3 days. Evaluator already exists in `-repl`; hook it. |

### Built-in library breadth (incremental, each ~0.5–2 days)

- **Reductions tail**: `std`, `var`, `median`, `mode`.
- **Reshape tail**: N-dim `cat`, N-dim `permute`.
- **Linalg tail**: correct non-symmetric `eig`, full `[U, S, V] = svd(A)`, `rank`, `qr` (m<n), `schur`, `hess`, `null`, `orth`, `cross`, `dot`.
- **Strings tail**: `strsplit`, `strjoin`, `regexp`, `regexprep`, `str2num`.
- **Search / indexing tail**: `A(i,:)=[]` frontend wiring (runtime exists).
- **I/O tail**: MATLAB `.mat` v5 format for `save`/`load`, `readtable`/`writetable` (needs `table` type).

### Tooling (each standalone)

| Missing | Scope | Reference |
|---|---|---|
| User-function frames in DAP stack trace | 0.5 week | Inject `matlab_dbg_enter_frame` / `_leave_frame` at function entry / return in the MLIR lowerer. Runtime and DAP server already call `stackTrace` from the frame list. See `docs/debug.md`. |
| LSP completion / hover / rename | 2 weeks | Extends the current skeleton. See `docs/lsp.md`. |
| Package manager / path | 1 week | `addpath`, `+pkg` directories. |
| Linter (style + unused-var) | 1 week | AST pass; formatter infrastructure already reusable. |
| Live-editor integration (Jupyter kernel) | 2 weeks | REPL already acts as a one-shot evaluator; a Jupyter adapter would mediate. |

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

The path from today's state to running a majority of general-purpose
MATLAB programs (not toolboxes, not GUI). Items 1–7 from the earlier
version of this doc — dim-aware reductions, varargin / call polish,
sort / linalg tail, strings, REPL, file I/O, basic OOP, tooling —
**have all shipped**. The remaining runway:

| Priority | Item | Effort | Unlocks |
|:-:|---|--:|---|
| 1 | Struct arrays (`s(i).x`) | 1 week | Data-in-records patterns |
| 2 | Integer runtime (typed `matlab_mat_i32` / `_u8` / …) | 1.5 weeks | Image processing pixel code |
| 3 | `varargout` + 3-D vector slicing (`A(:,:,k)`) | 1 week | Library-style + volumetric code |
| 4 | Complex-number runtime | 2 weeks | DSP programs |
| 5 | OOP value-class copy semantics + property validators | 2 weeks | Modern MATLAB code |
| 6 | DAP user-function frames + `evaluate` | 1 week | Stepping into user functions shows their frames; watch expressions |
| 7 | `regexp` / `regexprep` + string tail | 1–2 weeks | Text-processing scripts |
| 8 | Full non-symmetric `eig` + `[U, S, V] = svd` | 1 week | Scientific computing |
| 9 | MATLAB `.mat` file-format parser | 2 weeks | Real data pipelines |
| 10 | N-dim arrays (>3D, full indexing) | 2–3 weeks | Batch dims, tensor code |
| 11 | OOP events / listeners | 1 week | Callback-heavy code |
| 12 | Sparse matrices | 3–4 weeks | Large-scale linalg |
| 13 | `classdef` table / datetime / categorical | 3–4 weeks | Data-analysis idioms |

Items 1–3 are the immediate-leverage path for generic MATLAB
compatibility. Items 4–9 round out the "serious numeric work"
surface. Items 10+ are larger investments whose shape depends on
which direction the project pushes next.

---

## 10. Summary

**Where we are:** a production-quality compiler + tooling stack
covering the scalar / dense-matrix / classdef subset of MATLAB.

- **Three compiled backends** (LLVM IR, portable C, portable C++)
  producing byte-identical stdout on a 118-program run-test corpus.
- **JIT-backed REPL** (`matlabc -repl`) with a persistent workspace,
  implicit display, operator-overloading / indexing / transpose all
  auto-showing, plus `who` / `whos` / `clear`.
- **Language Server** (`matlab-lsp`): diagnostics, goto-definition,
  document outline. Works with Neovim, VS Code, Helix out of the
  box.
- **Source formatter** (`matlabc -format`) with attribute-aware
  classdef output and idempotent round-trip.
- **Debug aids**: `dbg(x)` source-located print, workspace
  inspection, `#line` directives in emitted C/C++ so gdb / lldb
  step the original `.m`.
- **OOP**: `classdef` with single inheritance, static methods,
  operator overloading, `Dependent` properties (`get.Prop` /
  `set.Prop`), enumerations.
- **File I/O**: text (`fopen` / `fgetl` / `fprintf`), binary
  (`fread` / `fwrite`), plus a custom single-matrix `save` /
  `load` format.
- **Linear algebra**: LU, QR, Cholesky, pseudo-inverse, norm,
  trace, kron, symmetric eig, SVD singular values — all pure-C,
  no BLAS / LAPACK dependency.
- **~3100-line single-file C runtime** that compiles stand-alone.

**Biggest gaps to a "general-purpose MATLAB replacement":** struct
arrays, typed integer runtime, complex numbers, 3-D vector slicing,
full DAP, and MATLAB `.mat`-format compatibility. Each is tractable
(Section 9 lays out the order); none is blocking any of the above.

**Biggest architectural asks:** value-class copy semantics for
OOP, sparse matrices, and true N-D (>3D) arrays. Each is multi-week
work and their priority depends on which direction the project
pushes next.
