# MATLAB Compatibility — Feature Status

This document is the high-signal inventory of what the codebase supports
today and what is still partial or missing. It is derived from the
frontend, MLIR passes, runtimes, CLI modes, and the in-tree test corpus.

The target is a **practical MATLAB subset** for numeric and compiler
workflows:
- dense numeric programs
- linear algebra
- control flow and functions
- handles, anonymous functions, structs, strings, and basic cells
- `classdef` with inheritance and operator overloading
- multiple output backends plus REPL and editor tooling

Out of scope:
- toolboxes
- plotting and GUI APIs
- full MATLAB compatibility
- `.mat` file compatibility

## Reading Guide

- Use this file to answer "is feature X implemented?"
- Use the README for the overall project view.
- Use backend-specific docs when you need lowering or runtime details.

---

## Legend

- ✅ Implemented and tested end-to-end on the main shipped paths
  (LLVM / C / C++ unless noted otherwise)
- 🟡 Partial — parsed and/or modelled in Sema, but runtime, lowering, or
  backend coverage is incomplete
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
| OOP keywords (`classdef properties methods events enumeration`) | ✅ | `classdef`, `properties`, `methods`, `enumeration` all parse and lower end-to-end (see §3 / §8 OOP rows). `events` parses but is ignored at runtime. |
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
| Complex literal arithmetic | ✅ | `2i`, `3+4i`, mixed real+complex binops route through `matlab_mat_c` (separate re/im planes, magic-tagged for polymorphic dispatch) |

### Parser — statements

| Feature | Status | Notes |
|---|:-:|---|
| Expression statement, assignment (incl. multi-LHS `[u,v] = f(x)`) | ✅ | |
| `if / elseif / else / end` | ✅ | |
| `for ... end` (range + step, negative step) | ✅ | |
| `while ... end`, `break`, `continue`, `return` | ✅ | |
| `switch / case / otherwise / end` | ✅ | |
| `try / catch / end` with error binding | ✅ | `catch ME; disp(ME.message)` works |
| `global`, `persistent` | ✅ | Scalar f64 only; 128-entry table. C/C++/Python/TS emit recognize the canonical `if isempty(x); x = init; end` first-call-init pattern and use `init` as the static-decl initializer (no runtime isempty check). |
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
| Integer dtype tracking (`int8..int64`, `uint8..uint64`) | 🟡 | Tracked in type lattice; `int32` and `uint8` matrix lanes have a typed runtime (Phase 1.1); narrower / wider int lanes still f64-shadowed |
| Complex dtype tracking | ✅ | Lowers to `!llvm.ptr` (matlab_mat_c*); runtime arithmetic shipped |
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
| `int8..int64`, `uint8..uint64` | 🟡 | **`int32` + `uint8` matrix lanes shipped (Phase 1.1)**: dedicated `matlab_mat_i32` / `matlab_mat_u8` runtime descriptors with saturating arithmetic (add/sub/.*/./), round-half-away-from-zero division, `int*N + double → int*N` MATLAB rule (double scalar saturating-cast at the binop), comparisons (return logical f64 0/1), cross-lane casts (`int32(uint8_mat)` etc.), typed disp formatting, REPL cross-input typed display + binops (registry-tagged workspace slots, `MxN int32`/`MxN uint8` in DAP variable view), and Python (`mat_i32_*` numpy int32 / int64-acc) + TypeScript (`mat_i32_*` NDArray) runtime parity. Gating tests: `test/Run/int_matrix_binops.m`, `int_image_filter.m`, `int_pixel_math.m`. **Still f64-shadowed**: `int8`, `int16`, `int64`, `uint16`, `uint32`, `uint64` matrix lanes, scalar-int+matrix interaction tail, fi/typed-int interplay. Scalar typed-int casts use the f64 runtime + saturating cast on assignment. |
| `complex` | ✅ | Imaginary literals (`2i`, `3j`), scalar + matrix arithmetic (add/sub/mul/div/matmul), mixed real+complex binops. Separate re/im planes; scalars auto-boxed to 1×1 — see [`docs/complex.md`](complex.md). |
| `fi` (Fixed-Point Designer) | 🟡 | Phases 1–5 shipped: scalar `fi(value, signed, WL, FL)` and `fi(value, T)` / `fi(value, T, F)` constructors with literal-fold, `+ - *`, `(:)` type-preserving assignment, `Saturate`/`Wrap` overflow, all five rounding modes (`Floor`/`Nearest`/`Zero`/`Ceiling`/`Convergent`), sub-native WL (e.g. WL=12 in i16 lane), implicit `fi + double` promotion, `int(n)` / `storedInteger(n)` / `double(n)`, `bin/hex/dec` display, **fi arrays** (`fi(zeros(1,N),...)`, indexing, slicing, vector concat, `sum`/`mean`), **persistent storage** of fi arrays, `numerictype` / `fimath` first-class objects, `setfimath` / `removefimath`, `reinterpretcast`, `-emit-fixed-point-report` driver flag. Gating test: FIR filter in `test/Run/fi_filter.m`. Storage = native `int8/16/32/64`. **Still open:** function-internal fi typing across user calls (`function y = f(x)` doesn't propagate the spec), 2-D fi matrices (1-D shipped), reductions tail (`prod`/`min`/`max`/`cumsum` on fi), `fi` parfor reductions, `fipref` display preferences, slope/bias scaling, complex `fi`, 3-D fi arrays. emit-typescript: FIR test skipped (BigInt-vs-number coercion). See [`docs/emit_fixed_point.md`](emit_fixed_point.md) §10.1. |
| N-D arrays (3-D) | 🟡 | `zeros(m,n,p)` / `ones(m,n,p)` + scalar `A(i,j,k)` read/write, `size(A, 3)`, `numel`, `ndims` |
| N-D arrays (>3D) | ❌ | |
| Sparse matrices | ❌ | |
| `datetime` / `duration` | ✅ shipped (Phase 5.1) | Scalar `datetime` (Unix-epoch seconds) and `duration` descriptors with constructors (`datetime(y,m,d)`, `datetime(y,m,d,h,mn,s)`, `datetime("now")`, `seconds/minutes/hours/days/years(n)`), MATLAB-default display formatting, and arithmetic (`dt + dur → dt`, `dt - dt → dur`, `dur ± dur → dur`). UTC; civil-date math via Howard Hinnant's algorithm. C / Python / TypeScript runtimes byte-identical. Gating test: `test/Run/datetime_basic.m`. **Open follow-up**: vector / array forms, calendar arithmetic (months, years), zoned datetimes, `between`/`caldays`/`calmonths`/`calyears`. |
| `categorical` | ✅ shipped (Phase 5.2) | 1-D categorical built from a string-array literal (`categorical(["a","b","a"])`). Auto-deduplicates and alphabetically sorts category names; per-element codes are 1-based with 0 = `<undefined>`. Surfaces: `length(c)`, `numel(c)`, `categories(c)` (returns a cell of category strings), `iscategory(c, "name")`, `disp(c)`. C / Python / TypeScript runtimes byte-identical. Gating test: `test/Run/categorical_basic.m`. **Open follow-up**: `categorical(values, valueset, catnames)` full constructor, `addcats`/`removecats`/`mergecats`/`renamecats`, ordinal categoricals, comparison ops beyond `==`. |
| `table` | ✅ shipped (Phase 5.3) | Column-major record with named variables; constructors `table(c1, c2, ...)` (auto-named Var1..VarN) and `table(c1, c2, ..., 'VariableNames', {n1, n2})`. Surfaces: `T.<name>` column read / write (with dynamic column add), `height(T)`, `width(T)`, `numel(T)`, `size(T, dim)`, `disp(T)` (right-aligned column body with header + underline). Each column stored as a `matlab_mat *`. C / Python / TypeScript runtimes byte-identical on the C/TS lanes; Python ships a `.stdout-python` override (numpy 2-D array repr for column print). Gating test: `test/Run/table_basic.m`. **Open follow-up**: heterogeneous columns (mixed numeric / string / categorical), row indexing `T(i,:)`, sub-table extraction, `readtable`/`writetable`. |
| `timetable` | ❌ | Builds on `table` + `datetime` row index. |

### Symbolic Math Toolbox (`sym` / `syms`)

Opt-in via `-DMATLAB_LLVM_WITH_SYM=ON` — requires [SymPP](https://github.com/leonardoaraujosantos/SymPP).
See [`docs/sym.md`](sym.md) for the full surface.

| Function | Status | Notes |
|---|:-:|---|
| `syms x y z`, `sym('expr')`, `str2sym` | ✅ | Workspace kind=7; cross-input REPL persistence |
| `+ - * / ^ ==` arithmetic dispatch | ✅ | Pure sym + mixed-mode (sym op double) |
| `diff(f, x, [n])`, `int(f, x, [a, b])` | ✅ | |
| `simplify`, `expand`, `factor`, `subs` | ✅ | `simplify` chains `refine()` so registered assumptions propagate (Phase 6.2): after `assume(y,'positive')`, `simplify(sqrt(y*y))` → `y` |
| `solve(eq, x)` | ✅ | Single eq, single var; multi-eq via `sym_solve_sys` (variadic) below |
| `taylor(f, x, a, n)`, `limit(f, x, target)` | ✅ | |
| `vpa(s, dps)`, `double(s)` | ✅ | |
| `sym('pi')`, `sym('exp1')`, `sym('I')` | ✅ | Phase 6.2 — recognises `pi`/`Pi`/`exp1`/`EulerGamma`/`Catalan`/`I`/`true`/`false` as SymPP singletons; `vpa(sym('pi'),32)` returns the digits of π |
| `dsolve(eq, y, yp, x)` (1st-order) | ✅ | SymPP's plain-symbol convention; no AppliedFunction lifting |
| `dsolve(eq, y, yp, ypp, x)` (2nd-order) | ✅ | Auto-classifies const-coeff vs Cauchy-Euler |
| `dsolve(A, x)` (linear system) | ✅ | `sym_dsolve_system(A, x)` — explicit symmat constructor |
| `dsolve_ivp(eq, y, yp, x, x0, y0)`, `apply_ivp(...)` | ✅ | Phase 6.2 — single-condition AND multi-condition forms (parallel sym vectors `[x0, x1, …]`, `[y0, y1, …]`) both wired |
| `checkodesol(eq, sol, y, yp, x)` | ✅ | Returns residual sym |
| `pdsolve(a, b, c, x, y)`, `pdsolve_heat`, `pdsolve_wave` | ✅ | First-order linear, heat, wave |
| `laplace`/`ilaplace`, `fourier`/`ifourier`, `ztrans`/`iztrans` | ✅ | |
| `assume(x, 'prop')`, `assumeAlso`, `clearAssumptions` | ✅ | 10 properties (real, integer, positive, …); rebinds the variable |
| `latex`, `pretty`, `ccode` | ✅ | Returns char* via `matlab_sym_*` |
| `matlabFunction(...)` | 🟡 | SymPP emits Octave source; not wrapped into a function handle |
| Symbolic matrices: `[a 1; 2 b]` literal syntax (Phase 6.2), `sym_matrix`, `sym_eye`, `sym_zeros`, `sym_det`, `sym_inv`, `sym_transpose`, `sym_trace`, `sym_rank`, `sym_linsolve`, `sym_dsolve_system` | ✅ | Distinct opaque type (kind=8) with cross-input REPL persistence + DAP rendering. The standard MATLAB `[a 1; 2 b]` matrix literal detects sym entries and routes through `matlab_symmat_*`; `sym_matrix(R, C, …)` stays as an explicit constructor |
| Multi-eq `sym_solve_sys([eq…], [var…])`, `sym_solve_2x2`, `sym_solve_3x3` | ✅ | Variadic + fixed-arity (Phase 6.2). Returns symmat with one row per joint solution |
| `nsolve`, `vpasolve` | ✅ | Newton's method in MPFR |
| Elementary functions on sym (`sin(sym)`, `exp(sym)`, …) | ✅ | Auto-dispatch when the operand is sym |

Backend matrix: `-emit-cpp` / `-emit-llvm` / REPL JIT / DAP all support sym.
`-emit-c` emits valid C (compile as C++ to link SymPP). `-emit-python`,
`-emit-typescript`, and `-emit-systemverilog` diagnose unsupported sym usage
at emit time with a clear error.

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
| `conj`, `real`, `imag`, `angle` | ✅ | Polymorphic — accept either real or complex input |
| `fft`, `ifft`, `fft2`, `ifft2` | ✅ | Pure-C Cooley-Tukey radix-2 + Bluestein for general N. See [`docs/complex.md`](complex.md). |

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

### Numerical solvers — ODE / IVP

See [`docs/ode.md`](ode.md) for the full surface, ABI notes, and call shapes.

| Function | Status | Notes |
|---|:-:|---|
| `ode45` (Dormand–Prince 5(4)) | ✅ | Adaptive FSAL; cubic-Hermite dense output (Refine = 4 default). |
| `ode23` (Bogacki–Shampine 3(2)) | ✅ | Same shape as `ode45`; Refine = 1 default. |
| `ode23s` (Rosenbrock 2(3), Shampine — **stiff solver**) | ✅ | One numerical-FD Jacobian per accepted step + three linear solves; the `(I − h·d·J)` factor absorbs stiff modes. Solves Robertson kinetics in ~9 steps where `ode45` would diverge. Scalar and vector `y`; same odeset surface; same call shapes. |
| Scalar `y` — `[t, y] = ode45(@(t,y) -2*y + sin(t), [0 10], 1)` | ✅ | |
| Vector `y` — `[t, y] = ode45(@(t,y) [-y(2); y(1)], [0 2*pi], [1; 0])` | ✅ | Anon-handle path only (the `LowerAnonCalls` pre-pass retypes the `y` block arg from f64 to ptr when the call site has matrix `y0`). Named-function handles (`@oscillator`) still blocked by the LowerUserCalls signature gate. |
| Backward integration (`tspan = [t1 t0]` with `t1 > t0`) | ✅ | |
| User-time grid (`tspan = [t0 t1 … tN]`, N > 2) | ✅ | Output at exactly the supplied times via Hermite; `Refine` ignored in this mode. |
| 3-return form `[t, y, stats] = ode45(...)` | ✅ | `stats` is a struct with `nsteps` / `nfailed` / `nfevals`. |
| `odeset` fields: `RelTol`, `AbsTol`, `MaxStep`, `InitialStep`, `Refine`, `Stats` | ✅ | `Stats = 1` numeric flag (deviates from MATLAB's `'on'` string — see [`ode.md`](ode.md)). |
| Event detection — `[t, y, te, ye, ie] = ode_events(@f, tspan, y0, @evt)` | ✅ | Bracket-then-bisect over each accepted DP45 step. `evt` returns 3×1 `[value; isterminal; direction]`; `isterminal = 1` halts integration at the event. Non-MATLAB call shape — wired as a dedicated builtin since the function-handle-in-struct ABI for `opts.Events` is still TBD. |
| `odeset` fields: `Events`, `OutputFcn`, `Jacobian`, `Mass`, `NonNegative`, `NormControl` | ❌ | Silently ignored. `Events` ships separately as the `ode_events` builtin (above). |
| Higher-order stiff (`ode15s`, `ode23t`, `ode23tb`, `ode15i`) | ❌ | `ode15s` (variable-order BDF + Newton) is the natural next step on top of the shipped `ode23s` infrastructure. |
| Non-stiff multistep (`ode113`) and high-order (`ode78`, `ode89`) | ❌ | |
| BVP (`bvp4c`, `bvp5c`), DDE (`dde23`) | ❌ | |
| `pdepe` — 1-D parabolic-elliptic PDE via method-of-lines | ✅ | Cartesian / cylindrical / spherical (`m = 0, 1, 2`); Dirichlet, Neumann, Robin BCs; non-uniform mesh; scalar PDE. Wraps `ode23s_v` for stiff time integration. Output `sol` is N_t × N_x. See [`ode.md`](ode.md). |
| `pdepe` extensions — multi-component systems (`npde > 1`); axis-of-symmetry `xmesh(1) = 0` for `m > 0`; `odeset` plumbed through | ❌ | Tracked in roadmap. |
| 2-D / 3-D FEM (`createpde`, mesh generation, `solvepde`) | ❌ | Multi-month scope. |
| Symbolic `pdsolve` family (closed-form heat / wave / 1st-order linear) | ✅ | In symbolic toolbox — see [`sym.md`](sym.md). |

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
| Struct arrays (`s(i).x`) | ✅ shipped (Phase 2) | New `matlab_struct_arr` runtime descriptor (vector of `matlab_struct *`); `s(i).x = v` auto-promotes the binding, `s(i).x` reads the i-th element, `length(s)` / `numel(s)` / `size(s, dim)` all dispatch correctly. Scalar fields work fully; matrix-valued fields (`s(i).vec = [1 2 3]`) carry the same pre-existing tensor->ptr conversion gap as the scalar struct path (`s.vec = [1 2 3]`) and are out of scope for this slice. Python and TypeScript runtimes ship parity. Gating test: `test/Run/struct_arr_basic.m`. |
| `fieldnames(s)` | 🟡 | Needs char-matrix dtype |
| Cell: 1-D literal, read/write, `numel`, `iscell` | ✅ | Auto-grows on OOB write |
| Cell: 2-D literals + `C{r, k}` indexing | ✅ shipped (Phase 1.3) | `{a, b; c, d}` -> `matlab_cell_new_2d` + per-cell `matlab_cell_set_<f64\|mat>_2d`; `C{r, k}` reads / writes via the matching get / set entries. `size(C, dim)` routes to `matlab_cell_size_dim`. Python (`cell_*_2d`) and TypeScript (`Cell2D` wrapper) runtimes ship with byte-identical output. |
| Cell: concatenation (`[A, B]`, `[A; B]`) | ✅ shipped (Phase 1.3) | Bracket-concat of all-cell elements chains `matlab_cell_concat_row` / `_col`; assignment auto-tags the LHS as a cell binding so `size` / `iscell` keep dispatching through the cell runtime. Spread-into-cell (`{C{:}, x}`) still missing — needs `varargin`-style unpacking at the literal site. |
| `cellfun`, `arrayfun` (beyond trivial cases) | 🟡 | Registered; not all wired |
| `containers.Map` / `dictionary` | ✅ shipped (Phase 4) | New `matlab_dict` runtime descriptor: a flat key/value table where keys may be f64 scalars or matlab_string * and values may be f64 scalars or matlab_mat *. Surfaces: `containers.Map()`, `dictionary()`, `dictionary(k1,v1,k2,v2,...)`, indexed read / write `m(k) / m(k) = v` (with CharLiteral keys auto-coerced to matlab_string), `length(m)`, `isKey(m,k)`, `remove(m,k)`. Python (list-of-tuples) and TypeScript (typed pairs array) runtimes ship parity. Lookup is O(N) — fine for the small dictionaries typical MATLAB programs build. Gating test: `test/Run/dict_basic.m`. |

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
| C emission (self-contained) | ✅ | `-emit-c`. Multi-return uses out-pointer params (`void f(args, T0 *out_0, ...)`); persistent + isempty-init pattern lowers to `static T x = <init>;`. `matlab.eq/ne/lt/le/gt/ge/short_or/short_and` and other unregistered MATLAB ops handled. |
| C++ emission (classes + inheritance preserved) | ✅ | `-emit-cpp`. Same scope as `-emit-c` plus `std::tuple<...>` return for multi-return; `std::tie(a, b) = f(...)` at call sites. |
| Source formatter (AST pretty-printer) | ✅ | `-format` |
| Flowchart (`.mflow`) frontend | ✅ | `-dump-flow` loads + validates a MatForge IDE flowchart JSON file. `-emit-matlab` (alias `-emit-m`) and any `-emit-*` lower a `.mflow` through the existing pipeline by synthesizing an AST. `-emit-mflow` goes the **reverse direction** — emit a `.mflow` from any `.m` (or round-trip a `.mflow`) in IDE-canonical JSON format with deterministic ids, alphabetical keys, and auto-layout; `--preserve-layout REF` merges `ui.position` from a reference file so IDE-set positions survive re-emits. The format is idempotent: `.m → .mflow → .m → .mflow` produces a byte-identical second `.mflow` from iteration 2 onward. Covers linear chains, structured control flow (`if`/`else`, `for`, `while`, `break`, `continue`, `return`, `switch`/`case`/`otherwise`, `try`/`catch`, arbitrary nesting), sub-flows lifted to top-level `Function`s, `function_definition` / `subflow_call` blocks, and `custom` blocks with three provenance modes (`source` inline / `path` `.m` file / `library_id` resolved against `--block-path` / `MATFORGE_BLOCK_PATH` / DAP+LSP `initializationOptions.blockPath`). Function-insertion dedup + optional arity validation. `matlab-lsp` accepts `.mflow` URIs and surfaces loader/builder diagnostics inline. `matlabc -dap` accepts `.mflow` programs — breakpoints set on `.mflow` JSON lines fire correctly because every synthesized statement carries the originating block's `.mflow` byte offset as its `Range.Begin`; stack frames point at the `.mflow` source and carry `[block:<id>]` in the frame name so the IDE highlights the active block on the canvas; step-over collapses multi-statement blocks to one logical step. Cross-backend round-trip lane (12 fixtures × 4 backends — C / C++ / Python / TS) confirms structural equivalence with text-source MATLAB. See `docs/flowchart_frontend.md`. |
| JIT / REPL | 🟡 | `matlabc -repl` with MLIR ExecutionEngine; state persists via a runtime workspace. No line editing / JIT cache / live user-function definitions yet. See `docs/repl.md`. |
| Python emission | ✅ | `-emit-python`. NumPy-backed runtime in `runtime/matlab_runtime.py`; see `docs/emit_python.md`. Matrix display uses numpy's bracket repr (`.stdout-python` per-test goldens for the test lane). Multi-return uses native tuple unpacking (`a, b = f(...)`); persistent + isempty-init lowers to `<fn>.<name> = <init>` at module scope. |
| TypeScript emission | 🟡 | `-emit-typescript`. Same scope as Python; runtime in `runtime/matlab_runtime.ts`. Multi-return uses array destructuring (`const [a, b] = f(...)`); persistent + isempty lowers to `let <fn>_<name>: number = <init>;`. |
| SystemVerilog (ASIC, synthesizable) emission | ✅ | `-emit-systemverilog`. Vendor-neutral, synthesizable RTL targeting ASIC flows. Tier-1 closure shipped: scalar combinational + FSMs + fixed-point pipeline + persistent fi-arrays + readability polish + bit-slicing `x(hi:lo)` (any width 1..64) + runtime-indexed persistent fi-arrays (auto-decoded regfile pattern) + hierarchical multi-module emission (`func.call` → SV instance with auto-wired clk/rst_n). 76 golden fixtures lint clean under Verilator (incl. `aes_round`, `cic_decimator`, `cordic_pipe`, `crc32`, `fir_asic_pipelined`, `i2c_bit_bang`, `regfile_dyn`, `spi_master`, `uart_rx`, `vector_processor`, plus `hier_combinational` / `hier_sequential` for multi-module). 7 fi-spec ↔ SV declaration regression tests in `test/EmitSVPorts/`, 2 boolean-port lint-hint tests in `test/EmitSVHint/`, 10 synthesizability-gate diagnostic tests in `test/EmitSVFail/`. Open: 2-D fi matrices, RAM inference, CORDIC for transcendentals. See `docs/sv_supported_subset.md` (supported-subset reference) and `docs/emit_systemverilog.md` (backend architecture). |

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
| `run-tests` (`-emit-llvm` + clang) | 172 | ✅ |
| `run-tests-emit-c` (`-emit-c` + cc) | 172 | ✅ |
| `run-tests-emit-cpp` (`-emit-cpp` + c++) | 172 | ✅ (`string_concat_mixed` + table / typed-int matrix issues fixed in Phase 6.2 emit-cpp pass) |
| `run-tests-emit-c-strict` / `-cpp-strict` (-Wall -Wextra -Werror) | 172 | ✅ |
| `run-tests-emit-python` (`-emit-python` + python3) | 172 | ✅ (some `.stdout-python` overrides for numpy repr) |
| `run-tests-emit-typescript` (`-emit-typescript` + bun) | 172 | ✅ (`string_concat_mixed` fixed in Phase 6.2; ~20 skipped for BigInt-vs-number coercion) |
| `run-tests-sym` (`-emit-cpp` + SymPP, opt-in via `-DMATLAB_LLVM_WITH_SYM=ON`) | 4 | ✅ — Phase 6.2 sym_phase_a/b/b1/b2 fixtures; skip-if-missing-SymPP via rc=77 |
| `emit-sv` golden tests + Verilator lint + Yosys synth | 76 | ✅ 76/76 |
| `emit-sv-fail` synthesizability gate diagnostics | 10 | ✅ 10/10 |
| `emit-sv-ports` fi-spec ↔ SV declaration regression | 7 | ✅ 7/7 |
| `emit-sv-hint` boolean-port lint hints | 2 | ✅ 2/2 |
| `emitc-fail-tests` (diagnostic contract) | 1+ | ✅ |
| `flowchart-tests` (`.mflow` loader: schema, validation, error paths) | 9 | ✅ 9/9 |
| `flowchart-emit-matlab-tests` (linear / control / sub-flows / custom blocks) | 17 | ✅ 17/17 |
| `flowchart-cross-backend-tests` (`.mflow` ≡ round-tripped `.m` across C / C++ / Python / TS) | 12 × 4 | ✅ 48/48 |
| `flowchart-lsp-tests` (`matlab-lsp` accepts `.mflow`, surfaces diagnostics) | 3 | ✅ 3/3 |
| `flowchart-dap-tests` (`matlabc -dap` on `.mflow`: bp verify, stop, frame source) | 3 | ✅ 3/3 |
| `flowchart-emit-mflow-tests` (`-emit-mflow` idempotency: `.m` → `.mflow` → `.m` → `.mflow` byte-identical) | 11 | ✅ 11/11 |

Examples gallery: 29 programs under `examples/` exercise matrix ops,
recursion, anonymous functions, function handles, parfor, linear
algebra, logical masks, struct/cell usage, OOP (`bank_account.m`
— classdef with inheritance, `Dependent` properties, operator
overloading), Symbolic Math Toolbox (`symbolic_demo.m` — full
sym surface incl. matrix literals, multi-eq solve, dsolve/pdsolve,
transforms), and ODE / PDE solvers (`ode_solver.m`). 39 synthesizable
HDL modules under `examples/hdl/` cover combinational primitives (ALU,
mux, priority encoder, leading-zero detector, popcount), FSMs (Mealy,
Moore, computed-state, UART RX, SPI master, I2C bit-banger), pipelined
DSP (FIR ASIC, CIC decimator, sequential processor, 4-stage CORDIC),
arithmetic engines (multi-cycle / Booth multiplier, AES round, CRC8 /
CRC32, FNV-1a, Galois LFSR, Hamming(7,4)), memory and dataflow
patterns (FIFO, async FIFO, register file with both static and runtime
indexing, AXI handshake, memory-mapped peripheral, sync 2-FF CDC,
Manchester encoder, barrel shifter), and the bit-slice / hierarchical-
module reference fixtures. 10 flowchart
programs under `examples/mflow/` showcase the `.mflow` JSON frontend;
each mirrors a text counterpart and produces byte-identical output
through every existing backend.

---

## 7. Tooling

| Feature | Status |
|---|:-:|
| Compiler CLI (`matlabc`) with 17 emit modes (incl. `-emit-matlab`, `-dump-flow`, `-emit-mflow`) + `-format` + `-repl` + `-dap` | ✅ |
| CMake + `just` build system | ✅ |
| CTest integration (22 lanes — frontend, run-tests × 4 backends, SV golden / port-spec / diagnostics, debug-hook, debug-DAP, debug-DWARF, plus 6 flowchart lanes) | ✅ |
| Diagnostics with source-location | ✅ |
| `#line` directives in emitted C / C++ | ✅ |
| Formatter (AST pretty-printer, idempotent) | ✅ | `matlabc -format` / `just format`. Drops comments (not in AST). |
| REPL / interactive interpreter | 🟡 | JIT via MLIR ExecutionEngine, persistent workspace, implicit display, `who`/`whos`/`clear`. `matlabc -repl`. See `docs/repl.md`. |
| Language Server (LSP) | 🟡 | `matlab-lsp` binary: initialize/shutdown, didOpen/didChange/didClose, publishDiagnostics, definition, documentSymbol. Accepts both `.m` and `.mflow` URIs (the latter routes through the flowchart loader + builder before Sema). No completion / hover / rename / workspace-symbol yet. See `docs/lsp.md`. |
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
| **OOP value-class copy semantics** | 🟡 partially shipped (Phase 3) | `b = a` on a value-class binding (any class without `< handle`) clones via `matlab_obj_clone` so writes to `b` don't leak into `a`. Handle classes (`< handle`) keep reference semantics. Three runtimes (C/Python/TypeScript) ship the clone helper with byte-identical behaviour. **Open follow-up**: full method-dispatch value semantics — calling `obj.foo()` on a value class should also operate on a fresh copy of `obj` inside the method. The existing in-tree class corpus (class_basic / class_dependent / class_operators / DAP scenarios) was written against the previous handle-style method dispatch, so the parameter-entry clone is intentionally NOT enabled today; flipping it on requires a corpus migration to either rebind the receiver (`obj = obj.foo()`) or annotate the class with `< handle`. Gating test: `test/Run/value_class_copy.m`. |
| **OOP events / listeners** | Medium | ~1 week. `notify` / `addlistener` / callback machinery. |
| **OOP property validators** (`{mustBeNumeric}`, size specs) | Small | ~2–3 days. Syntax parses today; need runtime checks at each assignment. |
| **N-dim arrays (>3D)** | Medium | ~2–3 weeks. Runtime descriptor generalization from `(rows, cols, depth)` to `(ndims, shape[])`; update all per-op lowering. 3-D already supported via `matlab_mat3` for `zeros/ones` + scalar indexing. |
| **3-D slicing** (`A(:,:,k)`) | Small | ~2–3 days. 3-D exists for scalar `A(i,j,k)`; vector / slice forms not wired. |
| **Integer runtime — narrower / wider lanes** (`int8`, `int16`, `int64`, `uint16`, `uint32`, `uint64`) | Medium | ~1 week. The `int32` + `uint8` lanes shipped in Phase 1.1 establish the descriptor / lowering / Python+TS / REPL+DAP shape; the remaining lanes drop in mechanically against the same template. |
| **Complex numbers — linalg tail** | Small | Scalars / matrix arithmetic / FFT shipped. Remaining: complex `inv` / `det` / `svd` / `eig` / `chol` / `qr`. |
| **Struct arrays** (`s(i).x`) | ✅ shipped (Phase 2) | Scalar fields work end-to-end; matrix-valued fields share the pre-existing tensor->ptr conversion gap with scalar structs and are deferred. |
| **Sparse matrices** | Large | ~3–4 weeks. Sparse representation + sparse-aware linalg; or lean on SuiteSparse. |
| **`varargout`** | ✅ shipped (Phase 1.2) | Pure (`function varargout = f(...)`) and mixed (`function [first, varargout] = f(...)`) forms; caller unpacks any LHS beyond the declared boundary from the matlab_cell* via `matlab_cell_get_mat`. Plain user-function multi-return (`[a, b] = swap(x, y)`) was also broken before this slice — both LHS got the same value — and is now wired through the same `matlab.call` (N results, `nargout` attr) shape the builtin path uses. Gating test: `test/Run/varargout_basic.m`. |
| **`classdef` dependent types** (`datetime`, `categorical`, `table`) | ✅ shipped (Phase 5.1–5.3) | datetime / duration / categorical / table all backed by dedicated runtime descriptors; see the per-type rows above. `timetable` (5.4) still missing. |
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
| ~~1~~ | ~~Struct arrays (`s(i).x`)~~ — **shipped Phase 2** | — | Data-in-records patterns |
| 2 | Integer runtime — `int32` + `uint8` matrix lanes complete (runtime, lowering, Python+TS, REPL+DAP). Remaining lanes (i8/i16/i64/u16/u32/u64 matrices) drop in against the same template. | ~1 week left | Image processing pixel code. (Note: 64-bit lanes already exist via fi-array work.) |
| 2b | Fixed-Point Designer (`fi`) — Phases 1–5 shipped. **Open**: function-internal fi typing (~1 week), 2-D fi matrices (~1.5 weeks), fi parfor reductions, reductions tail. See [`emit_fixed_point.md`](emit_fixed_point.md) §10.1. | 2 weeks total | DSP simulation, hardware-faithful integer math |
| 3 | ~~`varargout`~~ (shipped Phase 1.2) + 3-D vector slicing (`A(:,:,k)`) | ~3 days remaining | Library-style + volumetric code |
| 4 | Complex linalg tail (`inv` / `det` / `svd` / `eig`) | 1 week | Complete DSP / scientific code |
| 5 | OOP value-class copy semantics — **partially shipped (Phase 3)**: copy-on-assign works; method-dispatch value semantics still requires test-corpus migration. + property validators. | ~1 week left | Modern MATLAB code |
| 6 | DAP user-function frames + `evaluate` | 1 week | Stepping into user functions shows their frames; watch expressions |
| 7 | `regexp` / `regexprep` + string tail | 1–2 weeks | Text-processing scripts |
| 8 | Full non-symmetric `eig` + `[U, S, V] = svd` | 1 week | Scientific computing |
| 9 | MATLAB `.mat` file-format parser | 2 weeks | Real data pipelines |
| 10 | N-dim arrays (>3D, full indexing) | 2–3 weeks | Batch dims, tensor code |
| 11 | OOP events / listeners | 1 week | Callback-heavy code |
| 12 | Sparse matrices | 3–4 weeks | Large-scale linalg |
| ~~13~~ | ~~`classdef` table / datetime / categorical~~ — **shipped Phase 5.1–5.3** (timetable still pending) | ~1 week (timetable only) | Data-analysis idioms |
| 14 | `containers.Map` / `dictionary` — **shipped Phase 4** | — | Key-value patterns |
| 15 | 2-D cells and cell concatenation — **shipped Phase 1.3** | — | Heterogeneous data |

Items 1–3 are the immediate-leverage path for generic MATLAB
compatibility. Items 4–9 round out the "serious numeric work"
surface. Items 10+ are larger investments whose shape depends on
which direction the project pushes next.

---

## 10. Summary

**Where we are:** a production-quality compiler + tooling stack
covering the scalar / dense-matrix / classdef subset of MATLAB.

- **Three compiled backends** (LLVM IR, portable C, portable C++)
  producing byte-identical stdout on a 172-program run-test corpus,
  with Python and TypeScript ports tracking the same surface.
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
- **~6500-line C runtime** (split across `matlab_runtime.cpp`,
  `runtime_debug.cpp`, `runtime_complex.cpp`) that compiles
  stand-alone.
- **`containers.Map` / `dictionary`, 2-D cell arrays, struct arrays
  (`s(i).x`), datetime / duration, categorical, table** — heterogeneous
  data containers shipped Phases 1.3 / 2 / 4 / 5.1 / 5.2 / 5.3.
- **Typed `int32` / `uint8` matrix lanes** (Phase 1.1) with saturating
  arithmetic, REPL / DAP display, Python + TypeScript runtime parity.

**Biggest gaps to a "general-purpose MATLAB replacement":** narrower /
wider integer lanes (i8/i16/i64/u16/u32/u64 matrices — same template
as the shipped 1.1), `timetable` (5.4), 3-D vector slicing
(`A(:,:,k)`), full method-dispatch value-class semantics, complex
linalg tail (`inv` / `det` / `eig` / `svd` for complex), and MATLAB
`.mat`-format compatibility. Each is tractable; none is blocking any
of the above.

**Biggest architectural asks:** sparse matrices, true N-D (>3D)
arrays, and full method-dispatch value semantics for OOP. Each is
multi-week work and their priority depends on which direction the
project pushes next.
