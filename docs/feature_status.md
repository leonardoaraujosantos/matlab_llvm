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
- GUI APIs (interactive figures, App Designer, Live Editor inline plots,
  ginput, pan/zoom/rotate)
- full MATLAB compatibility
- `.mat` file compatibility

Now in scope (covered by dedicated docs):
- **Plotting**: headless Cairo-backed `plot` / `bar` / `surf` / etc. with
  PNG/SVG/PDF output. See [`plotting.md`](plotting.md).

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
| `eig` (non-symmetric, single-return correct path) | ✅ shipped | Tier-1.1 of CST roadmap. `matlab_eig` detects symmetry on entry; symmetric path stays on Jacobi (unchanged), non-symmetric path runs Hessenberg reduction → Francis double-shift implicit QR with deflation → eigenvalue extraction from real Schur form (1×1 real blocks, 2×2 complex-pair blocks). Returns `matlab_mat *` (real column) when all eigenvalues are real, `matlab_mat_c *` cast through the polymorphic descriptor when any conjugate pair exists. Ascending sort by real part, tie-break by imag. Verified: companion(3×3 / 4×4) → roots-of-poly; rotation matrix → ±i; damped oscillator → expected complex pole pair; lower-triangular → diagonal entries. The Python lane was likewise upgraded (the previous `.real.reshape` form silently dropped imaginary parts); TS lane keeps the existing stub and the test carries `.skip-emit-typescript`. The 2-return form `[V, D] = eig(A)` (eigenvectors) is **still on the symmetric path only** — fixing it needs eigenvector back-substitution from the real Schur form, follow-on slice. EmitC routing for `matlab_eig` no longer goes through the `m1("eig")` method-chain shortcut — the shortcut breaks the named-assign-then-multiply-use pattern the same way it bit `expm`/`hess`. Gating: `test/Run/linalg_eig_nonsym.m` (4 lanes; TS skipped) + 4 unit tests in `test/Runtime/test_linalg.c` (159 total passing). See [`docs/control_toolbox_roadmap.md`](control_toolbox_roadmap.md) §2.1. |
| `lu` (partial pivoting, 2-return `[L, U] = lu(A)`) | ✅ | |
| `qr` (Gram-Schmidt, 2-return `[Q, R] = qr(A)`) | ✅ | m ≥ n |
| `chol` (upper-triangular R with R'R = A) | ✅ | SPD-only; error flag on non-SPD input |
| `pinv` (via normal equations) | ✅ | Full-rank square / tall / wide |
| `norm` (Frobenius), `trace`, `kron` | ✅ | |
| `eig` (non-symmetric 2-return — eigenvectors) | ❌ | 1-return path shipped (see row above); 2-return form `[V, D] = eig(A)` for non-symmetric A still falls back to symmetrization. Needs eigenvector back-substitution from real Schur form. |
| `svd` (singular values only) | 🟡 | `U`, `V` not returned |
| `gram_c(A, B)` / `gram_o(A, C)` (gramians) | ✅ shipped | Tier-3.4 of CST roadmap. Three-line wrappers over `lyap` (Tier 1.4): `gram_c = lyap(A, B B')`, `gram_o = lyap(A', C' C)`. Used by the H₂ system norm `||G||₂ = √trace(B' Wo B) = √trace(C Wc C')` and balanced realisation. The model-object form `gram(sys, 'c')` is a Tier-2.1 follow-on once `ss` constructors land. Functional API names with `_c` / `_o` suffix to avoid string-arg dispatching. 2 unit tests + Run test `ctrl_step_gram.m` (5-lane byte-identical). |
| `[mag, phase] = bode_tf(b, a, w)` (TF frequency response) | ✅ shipped | Tier-2.4 follow-on. Polynomial coefficients in MATLAB convention (highest power first). Complex Horner evaluation `H(jω) = b(jω) / a(jω)` per frequency; ~30 lines. Bridges to SPT users who work in (b, a) form for analog filters. Verified that `bode_tf([1], [1, 1], w)` matches `bode_ss([-1], [1], [1], [0], w)` byte-identical (both representations of `H(s) = 1/(s+1)`). 2-return splitter mirrors bode_ss; 1-return form returns magnitude. 3 unit tests + Run test `ctrl_bode_tf.m` (5-lane byte-identical). |
| `lsim_ss(A, B, C, D, u, dt)` (state-space input simulation) | ✅ shipped | Tier-2.3 follow-on. Generalises `step_ss` to arbitrary input trajectory `u` (N×m matrix, one row per sample). ZOH discretisation between samples; relaxed initial state x[0] = 0. Verified that `lsim_ss(A, B, C, D, ones(N,1), dt)` matches `step_ss(A, B, C, D, dt, N)` exactly to 1e-12, and that zero input → zero output. 2 unit tests + Run test `ctrl_lsim_margin.m` (5-lane byte-identical). The MATLAB-faithful `lsim(sys, u, t)` form (with auto-time-grid) waits for Tier-2.1 model objects. |
| `gain_margin(A,B,C,D,w)` / `phase_margin(A,B,C,D,w)` | ✅ shipped | Tier-2.4 follow-on. Each scans the user-provided frequency grid `w`, finds the first crossover via linear interpolation: `gain_margin` looks for `phase = -180°` and returns `1/|L|` at that point; `phase_margin` looks for `|L| = 1` and returns `180° + phase(L)` at that point. Both return `+Inf` if no crossover is found on the grid (first-order plant has infinite gain margin; low-DC-gain plant has infinite phase margin). Verified Pm = 51.83° for L(s) = 4/(s(s+2)) — matches the closed-form `wc = √(2(√5−1)) ≈ 1.5723`, `phase(L(jwc)) = -90° - atan(wc/2) ≈ -128.16°`, `Pm = 180° - 128.16° ≈ 51.84°`. The MATLAB-faithful 4-return `[Gm, Pm, Wcg, Wcp] = margin(sys)` is a follow-on (would also need crossover-frequency entries; user can recover them today with `bode_ss` + scan). 2 unit tests + included in `ctrl_lsim_margin.m` Run test (5-lane byte-identical). |
| `[mag, phase] = bode_ss(A, B, C, D, w)` (SISO frequency response) | ✅ shipped | Tier-2.4 of CST roadmap. Per-frequency complex linear solve `(jω·I − A)·X = B` decomposed via the standard `[real, -imag; imag, real]` block trick into a real `2n × 2n` system — uses the existing `lu_decompose` + `lu_solve_column` helpers, no complex linalg needed. Returns linear magnitude (not dB) and phase in degrees, MATLAB convention. Splits via the eig_V/eig_D 2-return precedent (`matlab_bode_ss_mag` / `matlab_bode_ss_phase`); 1-return form returns magnitude. Verified on first-order lowpass `H(s) = 1/(s+1)` (closed-form `|H(1)| = 1/√2`, `phase(1) = -45°`), double integrator (`|H| = 1/w²`, `|phase| = 180°`), and second-order underdamped (`|H(wₙ)| = 1/(2ζ)`). 2 unit tests + Run test `ctrl_bode.m` (5-lane byte-identical — no overrides; `bode_ss` returns real outputs and never invokes the polymorphic eig path). SISO only; MIMO is a follow-on (build the per-frequency complex H matrix column-by-column). The plot-the-data `bode(sys)` form awaits Tier-2.1 model objects. **Open**: `bodemag`, `nyquist` (returns `(re, im, w)` instead of `(mag, phase, w)`), `nichols`, `sigma` for MIMO, transfer-function (b, a) form, `margin`/`allmargin`, `bandwidth`, `dcgain`. |
| `step_ss(A, B, C, D, dt, N)` (state-space step) | ✅ shipped | Tier-2.3 of CST roadmap. Discrete-time recurrence `x[k+1] = Ad x[k] + Bd u, y[k] = C x[k] + D u` from relaxed initial state `x[0] = 0`, with ZOH discretisation via `c2d` at sample `dt`. Returns `N × p` output matrix. Matches the closed-form first-order step `y(k·dt) = 1 - exp(-k·dt/τ)` to ~1e-10 and the steady-state value `y_ss = -C A⁻¹ B` to ~1e-3 after 200 samples. The MATLAB-faithful `[y, t] = step(sys)` form awaits Tier-2.1 model objects; current functional API uses positional arguments. 2 unit tests + Run test (5-lane byte-identical). |
| `[Ad, Bd] = c2d(A, B, Ts)` (zero-order hold) | ✅ shipped | Tier-2.2 of CST roadmap. Discretises continuous-time `xdot = A x + B u` to `x[k+1] = Ad x[k] + Bd u[k]` for ZOH on `u`, via the Van Loan augmented-matrix `expm` trick: build `M = [[A·Ts, B·Ts]; [0, 0]]`, compute `expm(M) = [[Ad, Bd]; [0, I_m]]`, slice out the top-left and top-right blocks. One `expm` call gives both. Verified on the diagonal closed form (`Ad = exp(A_diag·Ts)`, `Bd = (1 − exp(A_diag·Ts))/(−A_diag)·B_diag`) and the double integrator's exact ZOH (`Ad = [1, Ts; 0, 1]`, `Bd = [Ts²/2; Ts]`). Routes via `matlab_c2d_Ad` / `matlab_c2d_Bd` mirroring the eig_V/eig_D 2-return precedent; the `[Ad, Bd] = c2d(A, B, Ts)` shape goes through a dedicated `(p, p, f)` 2-return splitter in `LowerTensorOps.cpp` (same shape as `bilinear`). 2 unit tests + Run test `ctrl_c2d.m` (4-lane: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — eig stub for the post-discretisation stability check). **Open**: Tustin / `tustin` / `'tustin'` method (no `expm` needed; closed-form bilinear), foh / impulse / matched / least-squares methods, `d2c` (inverse direction). Currently positional `c2d(A, B, Ts)` form only — the model-object `c2d(sys, Ts)` form is a Tier-2.1 follow-on once `ss`/`tf` constructors land. |
| `[K, S, e] = lqr/dlqr(A, B, Q, R)` (3-return splitter) | ✅ shipped | Tier-4.1 of CST roadmap. Multi-return splitter mapping `[K, S, e] = lqr(A, B, Q, R)` to `matlab_lqr` (gain) + `matlab_care` (Riccati X) + `matlab_lqr_e` (closed-loop poles `eig(A − B·K)`); same shape for `dlqr`. The 2-return form `[K, S]` is also handled. 1-return form `K = lqr(...)` continues to use the existing direct dispatch. Verified on the canonical double integrator: `K = [1, √3]`, `S = [√3, 1; 1, √3]`, eig at `−√3/2 ± j·0.5`. 2 unit tests (continuous closed-form, discrete Schur stability) + Run test `ctrl_lqr_3ret.m`. |
| `[X, K, L] = care/dare(A, B, Q, R)` (3-return splitter) | ✅ shipped | Tier-1.5 of CST roadmap. Symmetric companion to the `lqr` 3-return splitter — one call returns the Riccati solution `X`, the LQ gain `K = R⁻¹B'X` (or `(R + B'XB)⁻¹B'XA` for discrete), and the closed-loop spectrum `L = eig(A − B·K)` (`eig(Ad − Bd·K)` for discrete). Routes to `matlab_care` / `matlab_dare` + `matlab_lqr` / `matlab_dlqr` + `matlab_lqr_e` / `matlab_dlqr_e`; the 2-return `[X, K]` form drops L. The 1-return `X = care(...)` form continues to use the direct dispatch. Verified on the double integrator (`X = [√3, 1; 1, √3]`, `K = [1, √3]`) and a Schur-stable diagonal discrete plant (per-axis closed-form positivity). Run test `ctrl_care_3ret.m` (5-lane byte-identical: LLVM / emit-c / emit-cpp / emit-python / emit-typescript). |
| **`Lowering.cpp` field-store dispatch — tensor RHS** | ✅ shipped (bug fix) | The struct/object field-store path at `lib/MLIR/Lowering.cpp:3539` previously dispatched only on `Rhs.getType() == PtrTy` for the `_mat`-vs-`_f64` runtime callee choice. Tensor-typed RHS (e.g. `obj.Numerator = num` where `num: tensor<2xf64>`) silently fell to `_f64`, downstream LowerTensorOps rejected the type mismatch, and the matlab.call_builtin survived to LLVM-translation as an unhandled op. Fix routes both `RankedTensorType` and `UnrankedTensorType` RHS to `_mat`. Same fix at the struct-array store site and the plain struct store. Surfaced while attempting CST §3.1 (`tf` classdef with matrix properties); needed independently of the §3.1 scope. |
| **CST stdlib prelude wiring (matlabc auto-prepend)** | ✅ shipped (infra) | `tools/matlabc/main.cpp` learned a `findCstPrelude` helper that walks `<bin>/../runtime/cst_classdefs.m` (and a few sibling install paths), and when present prepends the file to the user's `.m` source via the same multi-file concat path used by `--extra-input`. No-op if the file isn't found. The wiring is now conditional via `userMentionsCstClass`: a comment-stripped textual scan of the user input for `tf(`, `ss(`, etc. patterns (or one of these names on the LHS of a single `=`), so unused classdefs don't compile down to `func.func` bodies whose `none`-typed slots no downstream pass can resolve. |
| **§3.1 model objects + Tier-2/3/4 leftovers + model-object short-form surface** | ✅ shipped (full slice) | Closes §3.1 / §3.2 / §3.3 / §3.4 / §3.5 / §3.6 / §4.1 / §4.3 / §4.4 / §4.5 / §5.1 / §5.2 / §5.3. **Classdefs**: `tf` (full — constructor + property reads + tf-vs-tf operator overloads `+ − ∗ / −` + scalar mixing + `s = tf([1 0], 1)` polynomial composition + `tf('s')` / `tf('z')` char-literal sugar via constructor-call lowering rewrite + `disp(tf)` centred-fraction s-domain rendering via the `matlab_tf_disp` runtime helper through the existing `disp(obj)` class-method route). `ss` (full — constructor + property reads + operator overloads `+ − ∗ −` doing block-diagonal A assembly / series cascade / output negation via `horzcat` / `vertcat`). `zpk` (full — constructor + `∗ / −` via root concatenation + gain product). `pid` (full — coefficient-wise `+ − −`). `frd` (full — element-wise on `ResponseData` for `+ − ∗ −`; `mtimes` uses `.*` since H_ab(jω) = H_a(jω) · H_b(jω)). Per-class preludes auto-prepended only when mentioned in the source. **Value-returning model-object short forms**: `pole(sys)`, `step(sys [, dt, N])`, `impulse(sys [, dt, N])`, `initial(sys, x0 [, dt, N])`, `lsim(sys, u, dt)`, `bode(sys, w)`, `freqresp(sys, w)`, `nyquist(sys, w)`, `allmargin(sys, w)`, `dcgain(sys)`, `bandwidth(sys)`, `damp(sys)`, `isstable(sys)`, `ctrb(sys)`, `obsv(sys)`, `gram(sys, 'c'\|'o')`, `norm(sys)` / `norm(sys, 2)`, `hsvd(sys)`, `balreal_T(sys)`, `lqry(sys, Q, R)`. **Class-returning short forms** (Sema's `pinnedOfRhs` propagates the class pin through known short-form names): `c2d(sys, Ts)`, `feedback(sys1, sys2)`, `series(sys1, sys2)`, `parallel(sys1, sys2)`, `append(sys1, sys2)`, `blkdiag(sys1, sys2)`, `sminreal(sys)`, `modred(sys, elim, 'Truncate'\|'MatchDC')`. **New matrix-arg runtime entries**: generalised `eig(A, B)` via QZ + 2×2-block quadratic; 5-arg cross-term `lqr_5(A, B, Q, R, N)` / `dlqr_5`; output-weighted `lqry_ss(A, B, C, D, Q, R)`; `impulse_ss`, `initial_ss`, `freqresp_ss`/`_tf`, `nyquist_ss`/`_tf`, `allmargin_ss`, `logspace`; `pade(τ, n)` (closed-form [n/n] symmetric Padé), tf-form `minreal(num, den, tol)`, `sminreal_{A,B,C}` (boolean-graph reach/observability), `modred_{A,B,C}` (truncate or MatchDC Schur complement), `thiran(D, n)`. **Architectural enablement**: `pinnedOfRhs` extended in Resolver.cpp to recognise class-returning builtin short-form names (the call_builtin → fresh class-instance path); class-method monomorphisation gate skipping tensor-arg refinement on `matlab.class_name`-tagged functions; sibling-clone retargeting for `matlab.call` → `func.call` conversion; binary-op scalar-boxing wrapper restricted to CST classes (so Vec2-style classes with custom scalar-mixing methods stay unchanged); unary-op class-method dispatch; field-store dispatch tensor-RHS routing to `_set_mat`. **Gating tests**: `ctrl_tf_basic.m`, `ctrl_tf_disp.m`, `ctrl_model_objects.m`, `ctrl_zpk_ops.m`, `ctrl_ss_ops.m`, `ctrl_pid_ops.m`, `ctrl_frd_ops.m`, `ctrl_sys_short.m`, `ctrl_tier2_response.m`, `ctrl_tier3_design.m`, `ctrl_tier4_reduce.m`, `ctrl_tier4_assemble.m`, `ctrl_tier4_close.m`, `linalg_eig_gen.m`. LLVM lane only — the §3.1 tests carry `.skip-emit-{c,cpp,python,typescript}` markers because `EmitC.cpp::emitCStructTypedef` currently models classdef property layouts as all-`double` (matrix-typed properties don't fit the struct layout, and class instances flow through the emit pipeline by-value while runtime helpers expect `void *` — per-property type tracking in `CppClassDef::Properties` is the fix). **Still 🔵**: model-object multi-return (`[Ar, Br, Cr, hsv] = balred(sys, k)`); ss-form `minreal(sys)` (needs `ctrbf` / `obsvf` staircase); graph-style `connect` / `sumblk` / `lft`; H∞ norm; `stabsep` / `freqsep` / `loopsens` / `gangoffour` (need ordered Schur); internal-delay representation on classdefs; emit-lane parity for model-object tests. See `control_toolbox_roadmap.md` §3.1–§5.3 closure summary. |
| `[L, P] = kalman/kalmd(A, G, C, Qn, Rn)` (2-return splitter) | ✅ shipped | Tier-4.2 of CST roadmap. Multi-return splitter mapping `[L, P] = kalman(...)` to `matlab_kalman_L` (gain) + `matlab_kalman_P` (dual-care covariance `P = care(A', C', G·Qn·G', Rn)`); same shape for `kalmd` (uses `dare`). 1-return form defaults to L. Verified that `L = P · C' / Rn = P(:, 1)` for SISO with `Rn = 1` and `C = [1, 0]` (the Kalman gain identity). Run test `ctrl_kalman_2ret.m`. |
| `[Ar, Br, Cr] = balred(A, B, C, k)` (3-return splitter) | ✅ shipped | Tier-4 of CST roadmap. User-facing 3-return entry mapping to `matlab_balred_A` / `matlab_balred_B` / `matlab_balred_C`. 1-return form `Ar = balred(...)` defaults to balred_A. Same numerical underpinning as the original three single-return entries (rebuilds the full balanced realisation via balreal_T then truncates). Existing `ctrl_balred.m` Run-test rewritten to use the 3-return form; produces byte-identical output. |
| `c2d_tustin(A, B, Ts)` / `d2c_tustin(Ad, Bd, Ts)` (Tustin discretisation pair) | ✅ shipped | Tier-2.2 follow-on. Closed-form bilinear: `Ad = (I − αA)⁻¹(I + αA)`, `Bd = Ts·(I − αA)⁻¹·B` with `α = Ts/2`; inverse direction `A = (2/Ts)·(Ad − I)·(I + Ad)⁻¹`, `B = (2/Ts)·(I + Ad)⁻¹·Bd`. No `expm` needed. 2-return splitter for `[Ad, Bd]` / `[A, B]`. Round-trip through c2d_tustin → d2c_tustin reproduces original A, B to machine precision. Hurwitz → Schur preserved. 4 unit tests + Run tests `ctrl_c2d_tustin.m`, `ctrl_d2c_tustin.m`. **Open**: foh / impulse / matched-pole-zero. |
| `isstable_d(A)` / `norm_h2_d(A, B, C, D)` (discrete companions) | ✅ shipped | Tier-3 discrete companions. `isstable_d` returns 1.0 if all `\|eig(A)\| < 1` (Schur-stable); marginal eigenvalues on the unit circle fail per MATLAB convention. `norm_h2_d = sqrt(trace(D·D') + trace(C·Wc·C'))` with `Wc = dlyap(A, B·B')`; +Inf if A not Schur-stable. Closed-form spot check: 1st-order `a=0.5, b=c=1, D=0` gives `‖G‖_2 = 2/√3 ≈ 1.1547`. 6 unit tests + Run test `ctrl_norm_h2_d.m`. |
| `stepinfo(y, t)` (step-response metrics) | ✅ shipped | Tier-3 of CST roadmap. Returns 1×5 row `[RiseTime, SettlingTime, Overshoot, Peak, PeakTime]`. Final value = `y(end)`. Rise = first-90% minus first-10% crossing. Settling = last index where `\|y - Final\| > 2%·\|Final\|`. Overshoot = `(Peak − \|Final\|) / \|Final\| · 100` (percent, clipped to non-negative). 1st-order closed-form check: `τ·log(9) ≈ 1.0986`, `−τ·log(0.02) ≈ 1.957`. 2nd-order underdamped (`ζ = 0.3`) gives 37.21% overshoot — matches the closed form `100·exp(−π·ζ/√(1−ζ²))` exactly. 1 unit test + Run test `ctrl_stepinfo.m`. **Open**: full struct-return `[Rise, Settle, SettlingMin, SettlingMax, Overshoot, Undershoot, Peak, PeakTime, TransientTime]`. |
| `bandwidth_ss(A, B, C, D)` / `getPeakGain_ss(A, B, C, D)` (frequency-domain metrics) | ✅ shipped | Tier-3 of CST roadmap. `bandwidth_ss` scans a 200-point log-spaced grid (1e-3 → 1e6 rad/s) and returns the lowest `w` where `\|H(jw)\| < \|H(j0)\|/√2`; +Inf if no crossover (integrator, all-pass). `getPeakGain_ss` returns `max\|H(jω)\|` over the same grid — first approximation of the H∞ norm; misses sharp resonances (within ~10% for `ζ ≥ 0.05`). Verified on the 1st-order closed-form BW = 1.0 (exact) and 2nd-order `wn=10, ζ=0.7` BW ≈ 10.087 (closed form `wn·√(1 − 2ζ² + √((1 − 2ζ²)² + 1))`). 3 unit tests + Run tests `ctrl_bandwidth.m`, `ctrl_pole_peak.m`. **Open**: exact H∞ norm via Boyd-Balakrishnan-Kabamba γ-bisection on Hamiltonian eigenvalues. |
| `pole(A)` / `dcgain_ss(A, B, C, D)` (system characterization) | ✅ shipped | Tier-3 of CST roadmap. `pole(A)` is a name alias for `eig(A)` (closed-loop poles). `dcgain_ss = D − C · A⁻¹ · B`; returns 0×0 if A is singular (DC gain unbounded — integrator). Both wired through the standard dispatch. 3 unit tests for dcgain_ss + Run test `ctrl_dcgain.m`. |
| `feedback_ss / series_ss / parallel_ss / append_ss(A1, B1, C1, A2, B2, C2)` (matrix-arg interconnection) | ✅ shipped | Tier-2 of CST roadmap (matrix-arg form, strictly proper plants). All four are 3-return splitters routing `[Acl, Bcl, Ccl] = name(...)` to `matlab_<name>_{A,B,C}`. **Negative feedback**: `Acl = [A1, -B1·C2; B2·C1, A2]`, `Bcl = [B1; 0]`, `Ccl = [C1, 0]`. **Series cascade** (sys2 fed by sys1's output): `Acl = [A1, 0; B2·C1, A2]`, `Bcl = [B1; 0]`, `Ccl = [0, C2]`. **Parallel sum**: `Acl = blkdiag(A1, A2)`, `Bcl = [B1; B2]`, `Ccl = [C1, C2]`. **Append (MIMO blkdiag)**: `Acl = blkdiag(A1, A2)`, `Bcl = blkdiag(B1, B2)`, `Ccl = blkdiag(C1, C2)`. Generalised splitter dispatcher (one block recognises any of the four function names). 1-return forms default to Acl. 6 unit tests (block-layout assertions per primitive) + Run tests `ctrl_feedback.m`, `ctrl_interconnect.m` + example `interconnect_demo.m` showing series + parallel + feedback + append on a mass-spring-damper plant. **Open**: D ≠ 0 plants (current implementation assumes strictly proper); model-object form `feedback(sys1, sys2)` once Tier-2.1 ships. |
| **Compiler bug fix**: meshgrid / ndgrid multi-return type inference | ✅ shipped | `[xx, yy] = meshgrid(...)` previously typed both outputs as `Any`; downstream `exp(xx)` fell through to scalar Double, and `0.333 * exp_result` lowered as `arith.mulf(f64, !llvm.ptr)` and crashed the LLVM pipeline. Fix at two levels: `lib/Sema/TypeInference.cpp` (per-LHS `Array(Double, matrix)` type when LHS arity > 1 and RHS is meshgrid/ndgrid) + `lib/MLIR/Lowering.cpp` (multi-return MLIR result-type table). Plus added `.skip-emit-c` / `.skip-emit-cpp` support to `test/Run/run_tests_emitc.sh` mirroring the existing python/ts skip convention. Regression test `test/Run/lang_multiret_meshgrid.m` covers meshgrid + ndgrid + scalar*matrix chains; the original `surf_mesh.m` plotting example (which originally hit this bug) now compiles cleanly. |
| `lqr(A, B, Q, R)` (continuous LQR — 1-return form, gain only) | ✅ shipped | First Tier-2 user-facing wrapper of CST roadmap. Returns the optimal state-feedback gain `K = R⁻¹ B' X` where X solves the algebraic Riccati equation via `care` (Tier 1.5). Closed-loop dynamics `A − B K` are Hurwitz; user can compute closed-loop poles via `eig(A - B*K)` and the optimal cost from any initial condition via `x0' * X * x0` (X recovered via `care` directly). The 3-return MATLAB shape `[K, S, e] = lqr(A, B, Q, R)` (gain + Riccati + closed-loop poles) is a follow-on; same applies to `lqi` (integral action — augments A/Q with an integrator state) and `lqry` (output-weighted — computes Q_x = C'·Q·C from the SS realization). Verified on the canonical double integrator (closed-form `K = [1, √3]`, closed-loop poles at `−√3/2 ± j/2`) and a marginally-unstable plant `A = [1 1; 0 -2]` (LQR places real-negative closed-loop poles). 2 unit tests + Run test `test/Run/ctrl_lqr.m` (4-lane byte-identical: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped since eig stub returns zeros). Same auto-box allowlist entries (`lqr`) and EmitC plumbing (`matlab_lqr` in MatrixReturningFns + wrapperCovers) as the Tier-1 primitives. See [`docs/control_toolbox_roadmap.md`](control_toolbox_roadmap.md) §4.1. |
| `ctrb(A, B)` / `obsv(A, C)` (structural controllability/observability) | ✅ shipped | Tier-3 of CST roadmap. `ctrb(A, B) = [B, A B, A² B, …, A^{n-1} B]` (n × n·m); pair is controllable iff `rank(ctrb) = n`. `obsv(A, C) = [C; C A; C A²; …; C A^{n-1}]` (p·n × n); pair is observable iff `rank(obsv) = n`. Block matrices built by repeated matmul (one matmul per power). Structural-rank companions to the energy-based gramians `gram_c` / `gram_o` already shipped. 2 unit tests + Run test `test/Run/ctrl_place.m` (5-lane byte-identical: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — closing eig sanity check needs the eig stub fix). |
| `kalman_L(A, G, C, Qn, Rn)` / `kalmd_L(...)` (steady-state Kalman gain) | ✅ shipped | Tier-4.2 of CST roadmap. Continuous + discrete steady-state Kalman gains via the LQR/Kalman duality: `L = (lqr(A', C', G·Qn·G', Rn))'` (continuous) or `(dlqr(...))'` (discrete). Plant `xdot = A x + G w`, `y = C x + v` with `cov(w)=Qn`, `cov(v)=Rn`; the estimator `(A − L·C)` is Hurwitz (continuous) / Schur-stable (discrete). Verified on the 1×1 closed form `L = √2 − 1` (dual ARE `−2P − P² + 1 = 0` gives `P = √2 − 1`, and `L = P` for unit `Rn`), the open-loop unstable plant `[1 1; 0 -2]` (Kalman estimator places both poles at `-√3`), and Schur stability of the discrete estimator. End-to-end LQG demo in the Run-test exercises the separation principle: LQR closed-loop poles and Kalman estimator poles are computed independently. 3 unit tests + Run test `ctrl_kalman.m` (4-lane: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — care/dare degrade on the eig stub). The MATLAB-faithful 4-return `[kest, L, P] = kalman(sys, Q, R)` (estimator state-space + gain + Riccati) and the `lqgreg` / `lqg` / `lqgtrack` / `'current'` / `'delayed'` variants are follow-ons. |
| `norm_h2(A, B, C)` (H₂ system norm) | ✅ shipped | Tier-3 of CST roadmap. Continuous LTI strictly-proper H₂ norm `‖G‖_2 = sqrt(trace(C · Wc · C'))` where `Wc = lyap(A, B B')`. Returns `+Inf` if A is not Hurwitz (gramian unbounded) — checked via `isstable` before the Lyapunov solve. The two equivalent formulations (`sqrt(trace(C Wc C'))` and `sqrt(trace(B' Wo B))`) yield the same value; we use the C·Wc·C' form (one Lyapunov solve plus a small trace). The discrete `norm_h2_d` follow-on uses `dlyap` and the same trace formula. The H∞ system norm (Boyd-Balakrishnan-Kabamba bisection or Bruinsma-Steinbuch) is a separate Tier-3 follow-on. Verified on the 1st-order closed form `‖G‖_2 = 1/√2` for `G(s) = 1/(s+1)`, similarity-invariance under a 2×2 transform, and `+Inf` for an unstable plant. 3 unit tests + Run test `ctrl_norm_h2.m` (4-lane: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — eig stub makes isstable always return 0 so norm_h2 always returns +Inf). Sits cleanly on Tier-1.4 lyap. |
| `balred_A` / `balred_B` / `balred_C` (k-state balanced truncation) | ✅ shipped | Tier-4 of CST roadmap. `balred_A(A, B, C, k)` returns the `k × k` upper-left block of the balanced A; `balred_B` returns the first k rows of balanced B; `balred_C` returns the first k columns of balanced C. Each rebuilds the full balanced realization via `balreal_T` then truncates. The H∞ error bound is `‖G − G_k‖ ≤ 2 · sum(HSV[k+1:n])` so users can decide k from `hsvd(...)`. Verified on a 4-state plant where two of the four HSVs collapse to ~1e-7: balred(...,2) preserves the dominant block, the truncated realization is still Hurwitz, and `hsvd(Ar,Br,Cr)` exactly equals the top-2 HSVs of the original. The MATLAB-faithful 3-return shape `[Ar, Br, Cr] = balred(A, B, C, k)` is a follow-on (3-return splitter mirroring c2d / bode_ss precedent); users today call the three single-return entries. 2 unit tests + Run test `ctrl_balred.m` (4-lane: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — eig stub). |
| `balreal_T(A, B, C)` (balancing similarity transform) | ✅ shipped | Tier-4 of CST roadmap (model reduction). Returns the similarity transform `T` such that the realization `(T⁻¹ A T, T⁻¹ B, C T)` is internally balanced — its controllability and observability gramians become equal and diagonal, with diagonal = Hankel singular values (descending). Algorithm: Laub 1980 eigendecomposition variant (no Cholesky). Compute `Wc = gram_c(A,B)`, `Wo = gram_o(A,C)`; symmetric square root `X = Vc · sqrt(Dc) · Vc'` from sym-eig of `Wc`; sym-eig of `M = X · Wo · X = U · Σ² · U'`; reorder columns of U descending; then `T = X · U · Σ^{-1/2}`. The MATLAB-faithful 4-return form `[Ab, Bb, Cb, hsv] = balreal(A, B, C)` is a follow-on (multi-return splitter); users assemble the balanced realization today via `T = balreal_T(...); Ab = inv(T) * A * T; Bb = inv(T) * B; Cb = C * T`. The structural foundation that gates `balred` (model reduction by truncating small-HSV states). 2 unit tests (1×1 closed form `\|T\| = 1`; 2×2 post-balancing `Wcb = Wob = diag(HSV)` to 1e-7) + Run test `ctrl_balreal.m` (4-lane: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — eig stub returns zeros so balreal_T degrades to identity). |
| `isstable(A)` / `damp(A)` / `hsvd(A, B, C)` (model characterization triad) | ✅ shipped | Tier-3 of CST roadmap. **isstable** returns 1.0 if every `eig(A)` has strictly negative real part (continuous Hurwitz), else 0.0; marginally-stable poles (zero real part) are *not* stable in MATLAB convention. **damp** returns an `n × 2` table where row `k` is `[wn_k, zeta_k] = [|λ|, -real(λ)/|λ|]` for each pole — natural frequency and damping ratio; canonical 2-column form (the full 4-column `[pole, damping, freq, time-const]` shape is a follow-on once we have multi-return splitters for the 4-tuple). **hsvd** returns Hankel singular values `sqrt(eig(Wc · Wo))` sorted descending — the intrinsic I/O invariants of an LTI system, used as the diagnostic for balanced model reduction (`balred`/`balreal` follow-on). All three sit on top of the existing `eig` (Tier 1.1) and `gram_c`/`gram_o` (Tier 1.4 lyap) primitives. 7 unit tests (Hurwitz / unstable / marginal isstable; real-pole and underdamped damp closed-form; first-order hsvd closed form 1/2; similarity invariance of hsvd) + Run test `ctrl_charac.m` (4-lane: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — eig stub returns zeros so isstable always returns 0 and damp/hsvd produce empty rows). |
| `place(A, B, P)` (SISO pole placement via Ackermann) | ✅ shipped | Tier-3 of CST roadmap. Returns the state-feedback gain `K` such that `eig(A − B K) = P`. SISO only (B is n × 1); the multi-input Kautsky-Nichols-Van Dooren variant (which uses extra DOF for orthogonal-eigenvector conditioning) is a follow-on. Algorithm: Ackermann's formula `K = [0 0 … 1] · ctrb(A, B)⁻¹ · α(A)` where `α(s) = ∏(s − pᵢ)` is the desired closed-loop characteristic polynomial. α(A) is built by Horner on A; α coefficients are accumulated by complex polynomial multiplication then truncated to real (the imaginary part collapses to round-off for a valid conjugate-paired root set). Verified on the canonical double integrator (closed form `K = [2, 3]` for desired poles `{-1, -2}`) and a 3-state companion form (places at `{-1, -2, -3}` to ~1e-9). Accepts real or complex `P` (the runtime entry takes `void *P` and dispatches via the magic-tag). MATLAB `acker(A, B, P)` is the same Ackermann formula and would be a 1-line alias. 2 unit tests + Run test `ctrl_place.m`. |
| `dlqr(Ad, Bd, Q, R)` (discrete LQR — 1-return form, gain only) | ✅ shipped | Tier-2 discrete companion of `lqr`. Returns `K = (R + B' X B)⁻¹ B' X A` where X solves the discrete algebraic Riccati equation via `dare`. Closed-loop `Ad − Bd K` is Schur-stable (eigenvalues inside the unit disk). Verified on a Schur-stable diagonal 2×2 plant (per-axis quadratic closed-form), a c2d-discretised mass-spring-damper, and via DARE residual self-consistency to ~1e-9. 3 unit tests (1×1 closed-form residual, 2×2 residual, closed-loop Schur stability) + Run test `test/Run/ctrl_dlqr.m` (4-lane byte-identical: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped — eig stub for the closed-loop pole-magnitude check). 3-return shape `[K, S, e] = dlqr(...)` and `dlqi` / `dlqry` follow the same pattern as the continuous lqr counterparts. |
| `dare(Ad, Bd, Q, R)` (discrete algebraic Riccati) | ✅ shipped | Tier-1.5 follow-on of CST roadmap. Solves `A' X A − X − A' X B (R + B' X B)⁻¹ B' X A + Q = 0` for the unique stabilising `X = X' ⪰ 0`. Algorithm: Newton-Kleinman iteration (Hewer 1971) seeded from `X₀ = dlyap(Ad', Q)`. Each step computes `K_k = (R + B' X_k B)⁻¹ B' X_k A`, the closed-loop `A_cl = Ad − Bd K_k`, then `X_{k+1} = dlyap(A_cl', Q + K_k' R K_k)` — Newton iterations preserve closed-loop stability so once K_0 stabilises convergence is quadratic. **Limitation**: `K_0 = 0` stabilises only when `Ad` is already Schur-stable (the typical case after `c2d` of a stable continuous plant); for unstable Ad the user must pre-stabilise (continuous `lqr` then `c2d` the gain). The direct symplectic-pencil approach via QZ is the textbook large-scale algorithm and is deferred until QZ ships. Sits cleanly on Tier-1.4 `dlyap`. Gates `dlqr` (shipped above), `dlqi`, `kalmd`. |
| `care(A, B, Q, R)` (continuous algebraic Riccati) | ✅ shipped | Tier-1.5 of CST roadmap. Solves `A'X + XA - X B R⁻¹ B' X + Q = 0` for the unique stabilising solution `X = X' ⪰ 0`. Algorithm: matrix sign function via Newton iteration (Roberts 1980) on the Hamiltonian `H = [[A, -B R⁻¹ B']; [-Q, -A']]` — converges quadratically when no eigenvalue lies on the imaginary axis (the standard stabilisable + detectable LQR setup). Each iteration is one `inv` + one `(S + S⁻¹)/2`. After `S = sign(H)` converges, `P = (I - S)/2` projects onto the stable invariant subspace; `X = P_bot · P_top⁻¹`. Verified on the 1×1 closed form `x = -1 + √2`, the double-integrator closed form `[√3, 1; 1, √3]`, and a 2×2 stable plant via residual self-consistency to ~1e-9. The discrete `dare` is a follow-on (needs Cayley CARE↔DARE bridge or QZ pencil — neither shipped yet). 3 unit tests + Run test `linalg_care.m` (4-lane byte-identical: LLVM / emit-c / emit-cpp / emit-python with override; TS skipped because the TS-lane `eig` is still a stub). Gates `lqr` / `lqi` / `kalman` / `lqg` and the H₂ / H∞ system norms. The slice also added `inv` to the `LowerTensorOps.cpp` `AutoBoxNames` allowlist so 1×1 SISO `inv(R2)` calls auto-box to `matlab_mat *`, and removed the `m1("inv")` shortcut in `EmitC.cpp` (same emit-cpp method-chain bug that bit `expm`/`hess`/`eig`). |
| `lyap(A, Q)` / `dlyap(A, Q)` (Lyapunov / Stein) | ✅ shipped | Tier-1.4 of CST roadmap. Continuous Lyapunov `A X + X A' + Q = 0` and discrete `A X A' - X + Q = 0` — both implemented by vectorising the matrix equation (row-major `vec`) into an `n²·n²` dense linear system and solving with the existing pivoted-LU helper. `O(n^6)` cost; perfectly adequate for the small plants typical of CST workflows (n = 2..10). For larger plants the proper algorithm is Bartels-Stewart back-substitution on the Schur form (the Tier-1.2 follow-on `schur` is the gating piece — already shipped) — flagged as a follow-on optimisation in the source comments. The 1×1 `lyap` / `dlyap` shape (`lyap([-1], [1])`) is supported via the `AutoBoxNames` allowlist in `LowerTensorOps.cpp` so scalar-typed args get auto-boxed to `matlab_mat *`. Gates `gram` (controllability / observability gramians as Lyapunov solutions), the H₂ system norm, and balanced realisation. C / C++ / Python / TS lanes mirror the same algorithm; Python uses `np.kron` directly, TS builds the Kronecker product element-by-element. 5 unit tests + Run test `test/Run/linalg_lyap.m` (5-lane byte-identical). See [`docs/control_toolbox_roadmap.md`](control_toolbox_roadmap.md) §2.4. |
| `schur` (real, 1- and 2-return `[U, T] = schur(A)`) | ✅ shipped | Tier-1.2 follow-on of CST roadmap. Hessenberg reduction + Francis double-shift QR with the orthogonal accumulator U threaded through both passes. Returns `T = U' A U` (1-return) or `[U, T]` such that `A = U T U'`. The 2-return form routes via `matlab_schur_U` / `matlab_schur_T` mirroring the eig_V/eig_D precedent. Same numerical core as `matlab_eig`'s non-symmetric path; the schur entries skip the symmetry detection (Schur is meaningful for symmetric matrices too). 5 unit tests in `test_linalg.c` (1×1 trivial, upper-triangular fixed point, A = U T U' reconstruction on a 3×3, U' U = I orthogonality on a 4×4, trace and det preservation). C / C++ / Python lanes mirror the same algorithm bit-for-bit (Python lane reimplements Francis QR rather than deferring to scipy.linalg.schur — same macOS-Anaconda numpy/scipy ABI mismatch that affected expm). TS lane stub returns A as-is; non-canonical but the schur Run test happens to pass via the trace/det invariants. **Open**: ordered-Schur (the variant that gates `care`/`dare` Riccati). |
| `rank`, `null`, `orth`, `cross`, `dot` | ❌ | (`hess`, `schur` shipped — see rows above) |
| `expm` (matrix exponential) | ✅ shipped | Scaling-and-squaring with [13/13] Padé approximant (Higham 2005). Gates Tier-2 of the Control System Toolbox roadmap (`c2d` ZOH, `lsim` continuous, `initial`-condition response, closed-form Lyapunov / Riccati). Mirrored byte-identical across C / C++ / Python / TS lanes; the Python lane reimplements the same Padé rather than deferring to scipy.linalg.expm because the macOS Anaconda numpy/scipy combination commonly fails to import. Gating: `test/Run/linalg_expm.m` (5 lanes) + 5 direct C unit tests in `test/Runtime/test_linalg.c`. See [`docs/control_toolbox_roadmap.md`](control_toolbox_roadmap.md) §2.3. |
| `hess` (Hessenberg reduction, 1-return + 2-return `[H, P]`) | ✅ shipped | Householder reflections, in-place. Building block for Francis double-shift QR → real Schur form, which gates Tier-1.1 (non-symmetric `eig`), Tier-1.2 follow-on (`schur`), Tier-1.4 (`lyap`/`dlyap` Bartels-Stewart), and Tier-1.5 (`care`/`dare` ordered Schur on Hamiltonian pencil). 2-return shape `[H, P] = hess(A)` shipped via `matlab_hess_H` / `matlab_hess_P` mirroring the eig_V / eig_D precedent — H is upper Hessenberg, P is the orthogonal accumulator with P' A P = H. The reflection convention drives the subdiagonal element to `-sign(x_k)·‖x‖`, so already-Hessenberg-but-with-positive-subdiagonal inputs come back with the subdiagonal sign flipped (eigenvalues preserved). Mirrored byte-identical across C / C++ / Python / TS lanes; the same fix that landed `expm` (avoid the `m1("expm")` method-chain shortcut in EmitC) applies here so `hess` falls through to the default `matlab_hess(...)` form. Gating: `test/Run/linalg_hess.m` + `test/Run/linalg_hess_2ret.m` (LLVM lane; TS skipped) + 4 direct C unit tests in `test/Runtime/test_linalg.c`. See [`docs/control_toolbox_roadmap.md`](control_toolbox_roadmap.md) §2.2. |
| `logm` (matrix logarithm) | ✅ shipped | Tier-1.3 follow-on of CST roadmap §2.3. Schur-then-Parlett-recurrence (Higham 2008): real Schur T = U' A U, then F = log(T) computed via F[i,i] = log(T[i,i]) on the diagonal and Parlett's commutativity recurrence on the strict upper triangle. Reconstructs log(A) = U F U'. Returns 0×0 if the Schur form has 2×2 quasi-triangular blocks (complex eigenvalue pairs), non-positive diagonals, or coincident eigenvalues — the failure modes that need complex-arithmetic block log + Parlett's block recurrence + confluent-Taylor expansion respectively, all deferred. Gating tests: trivial 1×1 `log(4)`, diagonal positive 2×2, `logm(expm(A))` round-trip on an upper-triangular A, non-symmetric 2×2 with distinct positive eigenvalues, plus the two failure paths. The Python lane uses the `eig`-based diagonalisation path (V·diag(log(d_i))·V⁻¹) for the same preconditions; arithmetic order differs slightly from the C lane but the printed values agree to disp-level precision. TS skipped (the lane's `eig` is a stub). Run-test `linalg_logm.m`. |
| `lyapchol` (Cholesky factor of controllability gramian) | ✅ shipped | Tier-1.4 follow-on of CST roadmap §2.4. R = lyapchol(A, B) returns upper-triangular R with R'·R = Wc, where Wc solves A·Wc + Wc·A' + B·B' = 0. v1 implementation: round-trip via `gram_c` (which calls `lyap`) then `chol`. The square-root Hammarling solver (which avoids forming Wc explicitly) is the proper large-plant follow-on. Gates the balanced-realisation tail of Tier-4 model reduction. Gating tests: 1×1 closed form, R'·R = Wc identity on a stable 2×2, mass-spring-damper plant. Run-test `linalg_lyapchol.m`. |
| `sylvester` (3-arg Sylvester equation, surfaced as `lyap(A, B, C)`) | ✅ shipped | Tier-1.4 follow-on of CST roadmap §2.4. Solves A·X + X·B + C = 0 with A: n×n, B: m×m, C and X: n×m. v1: vectorise + dense LU on the (n·m)² Kronecker matrix M[(i·m+j), (k·m+l)] = A[i,k]·δ_{j,l} + δ_{i,k}·B[l,j]. The dispatch table in `LowerTensorOps.cpp` adds a second `lyap` entry with arity-3 signature routing to `matlab_sylvester` — same MATLAB convention (`lyap` is overloaded by arity for the 2-arg vs 3-arg shapes). Bartels-Stewart on Schur(A), Schur(B) is the large-plant follow-on. Gating tests: 1×1 closed form, 2×2-by-2×2 with diagonal A and B (per-element closed form), asymmetric 2×3 shape, residual self-consistency on a non-diagonal case. Run-test `linalg_sylvester.m`. |
| `qz` ([AA, BB, Q, Z] = qz(A, B), 4-return) | ✅ shipped (B-invertible path) | Tier-1.2 follow-on of CST roadmap §2.2. Generalised Schur of the matrix pencil A − λ·B: Q·A·Z = AA (real upper quasi-triangular), Q·B·Z = BB (real upper triangular), Q and Z orthogonal. v1 implementation layered on the existing `schur` and `qr` primitives: C = B⁻¹·A → real Schur U, T → QR of B·U gives O, R → set Q = O', Z = U, AA = R·T, BB = R. Generalised eigenvalues fall out as the diagonal pairs (AA[i,i], BB[i,i]). Returns 0×0 from each of the four entries when B is singular — that path needs proper Hessenberg-Triangular reduction + double-shift QZ iteration (Moler-Stewart 1973), the gating piece for `zero(sys)` on the Rosenbrock system matrix where B is rank-deficient by construction. Four entries (`matlab_qz_{AA,BB,Q,Z}`) follow the schur_U / schur_T precedent — each recomputes the full decomposition independently. 4-return splitter in `LowerTensorOps.cpp` mirrors the existing 3-return shapes (balred, feedback_ss, lqr-3ret). Gating tests: B = I gives standard Schur (AA quasi-triangular, BB = I, Q·A·Z = AA reconstruction holds), diagonal A and B (eigenvalues = a_ii / b_ii), singular-B failure path. Run-test `linalg_qz.m` (LLVM lane; TS skipped — the lane's `schur` is itself a stub). |
| `[V, D] = eig(A)` (2-return, non-symmetric) | ✅ shipped (real-eigenvalues path) | Tier-1.1 follow-on of CST roadmap §2.1. Symmetric A still goes through the Jacobi sweep with ascending-eigenvalue column ordering (unchanged). Non-symmetric A: real Schur U' A U = T → for each eigenvalue λ_i = T[i,i] solve (T − λ_i I)·y = 0 by upper-triangular back-substitution, recover v_i = U·y, normalise to unit 2-norm. Schur-diagonal column order so V's column k matches D's (k, k) diagonal entry. Returns 0×0 when any 2×2 quasi-triangular block (complex conjugate pair) is detected — proper complex back-substitution is the follow-on. Run-test `linalg_eig_2ret_nonsym.m`: diagonal A (canonical-basis eigenvectors), upper-triangular A (eigenvalues = diag, A·V = V·D check), non-symmetric A with distinct real eigenvalues, rotation matrix [[0 1]; [−1 0]] complex-pair failure path. The Python lane uses `np.linalg.eig` directly under the same precondition (real-eigvals only, else 0×0). |
| `icare` / `idare` (numerically-robust Riccati) | ✅ shipped (alias to care / dare) | Tier-1.5 follow-on of CST roadmap §2.5. v1 entries forward to the existing `care` / `dare` paths — same numerics for well-conditioned pencils. The proper Mehrmann-Voss structure-preserving QZ on the extended pencil (which avoids the matrix-sign squaring step that loses 1 bit per Newton iteration on ill-conditioned inputs) is the follow-on, gated on the singular-B QZ path. For the small CST-roadmap plants (n = 2..10) the numerical gap is below disp-precision; the rename gives user code on the modern API surface a working entry today. Run-test `linalg_icare_idare.m` validates that icare matches care on the double-integrator (X = [√3 1; 1 √3]) and idare matches dare on a Schur-stable diagonal plant. |
| 5-arg `care(A, B, Q, R, S)` / `dare(A, B, Q, R, S)` (state-input cross term) | ✅ shipped | Tier-1.5 follow-on of CST roadmap §2.5. Cost J = ∫(x'Qx + 2x'Su + u'Ru) dt admits a reduction to the standard 4-arg form via the change of basis A_hat = A − B·R⁻¹·S' and Q_hat = Q − S·R⁻¹·S' which preserves the stabilising solution (the cross-term is absorbed into the drift matrix and the state weighting). Same algebra for `dare` (Newton-Kleinman path) and for `icare` / `idare` (5-arg shape forwards through the same reduction). Dispatch table in `LowerTensorOps.cpp` adds a second `care` / `dare` / `icare` / `idare` entry with arity-5 signature routing to `matlab_care_5` / `matlab_dare_5` (same multi-arity pattern as `lyap`'s 2-arg-vs-3-arg entries). The 6-arg descriptor form `care(A, B, Q, R, S, E)` with generalised E·X·E' shape is the remaining follow-on (gated on the generalised-Riccati QZ). Run-test `linalg_care_5arg.m`: S = 0 → matches 4-arg care, non-zero S → residual `A'X + XA − (XB + S)R⁻¹(B'X + S') + Q ≈ 0` to ~1e-8, dare 5-arg matches 4-arg dare on a diagonal Schur-stable plant. |

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

### Signal Processing Toolbox (subset)

Per-toolbox roadmap in [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md).
Carved out: apps, GUI tools, deep-learning entries, Simulink Data Inspector,
MATLAB Coder integration, Python coexecution.

| Group | Function | Status | Notes |
|---|---|:-:|---|
| Convolution / correlation | `conv`, `conv2`, `xcorr` | ✅ | |
| Filter | `filter(b, a, x)` | ✅ | Direct-form II transposed; scalar IIR / FIR. |
| FFT / shift | `fft`, `ifft`, `fft2`, `ifft2`, `fftshift`, `ifftshift` | ✅ | See `complex.md`. |
| Multirate stubs | `upsample(x, n)`, `downsample(x, n)` | ✅ | Zero-stuff / decimate, no anti-aliasing filter (these remain the raw building blocks). The proper anti-aliased `resample`/`decimate`/`interp`/`upfirdn` ship separately — see the Tier-3 §4.1 row below. |
| Windows tail (Tier-1 §2.3) | `hamming`, `hann`, `blackman`, `rectwin`, `triang`, `bartlett`, `barthannwin`, `bohmanwin`, `parzenwin`, `nuttallwin`, `blackmanharris`, `flattopwin`, `kaiser`, `tukeywin`, `gausswin`, `chebwin`, `taylorwin` | ✅ shipped | All return an `n × 1` column. Two-arg parametric windows take their shape parameter as the second double; `taylorwin` takes `(n, nbar, sll)`. Symmetric (non-periodic) form. C / C++ / Python / TS runtimes byte-identical. Gating tests: `test/Run/sig_windows.m` (4-lane), `test/Runtime/test_signal.c` (15 reference-value checks: `rectwin` all-ones, `bartlett` triangular peak at midpoint, `kaiser(N, 0)` ≡ `rectwin(N)`, `tukeywin(N, 1)` ≡ `hann(N)`, etc.). |
| Filter design — IIR LP + HP + BP + BS (Tier-1 §2.1) — `butter(n, Wn[, 'high'])` / `butter(n, [W1 W2][, 'stop'])`, same shape for `cheby1(n, Rp, ...)` and `cheby2(n, Rs, ...)`, `freqz(b, a, N)` (single + 2-return forms) | ✅ shipped (LP+HP+BP+BS) | Refactored design pipeline: build analog LP prototype with Wn=1, apply analog frequency transformation (`lp2lp`/`lp2hp`/`lp2bp`/`lp2bs`), bilinear + gain-normalise at the type-specific frequency (DC for LP/BS, Nyquist for HP, ω₀=2·atan(W₀/2) for BP). The previous bandpass peak-normalisation bug was rooted in the prewarp-vs-bilinear T-convention mismatch; `bilinear_pole_` now uses (2+s)/(2-s) so all four filter types reproduce scipy / MATLAB to floating-point precision. Sema-level dispatch on call shape: scalar Wn / 2-element-vector Wn (via `matlab.concat_row` defining op) and the optional `'high'` / `'stop'` string literal. C / C++ / Python / TS runtimes byte-identical (TS `freqz` returns magnitude-only because NDArray has no native complex shape — same friction as `roots` / `fft_c`; gated with `.skip-emit-typescript`). Gating: `test/Run/sig_iir.m` + `sig_iir_more.m` + `sig_iir_bands.m` + 5 direct C unit tests. |
| Standalone analog↔digital + form conversions (Tier-1 §2.1) — `[bd, ad] = bilinear(b, a, fs)`, `H = freqs(b, a, w)`, `[z, p, k] = tf2zp(b, a)`, `[b, a] = zp2tf(z, p, k)`, `sos = tf2sos(b, a)`, `[b, a] = sos2tf(sos)`, `besself(n, Wo)` | ✅ shipped | Standalone `bilinear` reuses `matlab_roots` (Durand-Kerner) for analog factorisation and the `(2·fs+s)/(2·fs-s)` pole transform; preserves analog DC gain. `freqs` Horner-evaluates B(jw)/A(jw); returns `matlab_mat_c`. `tf2zp`/`zp2tf` round-trip through complex roots + `poly_from_complex_`. `tf2sos`/`sos2tf` build cascades of biquads via conjugate-pair grouping; `besself` uses the closed-form Bessel polynomial recurrence with MATLAB's `norm='phase'` scaling. All five lanes byte-identical via `test/Run/sig_iir_more2.m` + `sig_iir_sos.m` (the TS lane uses simpler "walk in pairs" pairing because TS NDArray drops the imaginary part of complex roots). |
| Order-selection helpers — `[n, Wn] = buttord(...)` / `cheb1ord(...)` / `cheb2ord(...)` | ✅ shipped | Lowpass scope; `buttord` uses `log10((10^(Rs/10)−1)/(10^(Rp/10)−1)) / (2 log10(Wsa/Wpa))`, Cheby variants use `acosh(...) / acosh(Wsa/Wpa)`. `cheb1ord` anchors at the passband edge (`Wn = Wp`); `cheb2ord` at the stopband edge (`Wn = Ws`). Multi-return splits via `matlab_<name>_n` / `_Wn` (eig precedent); single-LHS form returns `n`. |
| Filter design — IIR open: `ellip` / `ellipord` (need Jacobi elliptic functions), analog prototypes as standalone 3-return entries (`buttap` / `cheb1ap` / `cheb2ap` / `ellipap` / `besselap`), state-space conversions (`tf2ss` / `ss2tf` / `zp2sos`) | 🔵 | Follow-on. The big band-variants + standalone-bilinear/freqs + besself + form-conversions slice shipped — see the rows above. |
| Filter design — FIR (Tier-1 §2.2) — `fir1(n, Wn)` lowpass, `sgolay(k, f)`, `sgolayfilt(x, k, f)` | ✅ shipped (lowpass scope) | `fir1` is windowed-sinc with default Hamming, normalized to unit DC gain. `sgolay`/`sgolayfilt` use the standard polynomial-projection construction `B = V (V'V)^-1 V'`, with corresponding boundary rows applied at the first/last `(f-1)/2` samples. Lowpass / Savitzky-Golay scope only; `fir2`, `firls`, `firpm`, `firrcos`, `kaiserord` deferred. Gating: `test/Run/sig_fir.m` (3-lane: C/C++/LLVM + Python; TS skips for 1-ULP rounding-order divergence). |
| Polynomial helpers (Tier-1 §2.4) — `roots`, `poly`, `polyder`, `polyint`, `polyint(p, k)` | ✅ shipped | `roots` uses Durand-Kerner (Weierstrass) iteration on the leading-zero-stripped, trailing-zero-padded polynomial — bypasses the eig dependency entirely. `poly` builds via repeated convolution by `[1, -r_i]` in the complex plane; conjugate-symmetric inputs round-trip to a real coefficient vector. `polyder`/`polyint` are scalar arithmetic on coefficient vectors. C / C++ / Python / TS runtimes byte-identical (numpy bracket repr `.stdout-python` override). Gating tests: `test/Run/sig_poly.m` (4-lane) + 6 direct C unit tests in `test/Runtime/test_signal.c` (polyder of `x^3 + 2x^2 - x + 5`, polyint identity, `poly(roots)` round-trip, etc.). |
| Partial-fraction expansion — `[r, p, k] = residue(b, a)` | ✅ shipped (Tier-1 §2.4) | Distinct-pole scope: long-divides `b/a` for the direct term `k`, finds poles via `roots(a)`, computes residues by the cover-up rule `r_i = b'(p_i) / a'(p_i)`. Multi-return wired via three independent runtime entries `matlab_residue_{r,p,k}` (eig_V/eig_D precedent); the LowerTensorOps dispatch auto-boxes f64 scalar args via `matlab_mat_from_scalar` and defers tensor-typed args until the matrix-slot lowering converts them to ptr in a fixpoint iteration. C / C++ / Python / TS runtimes byte-identical. Gating: `test/Run/sig_residue.m` (4-lane, asserts symmetric functions of `r` and `p` so the test is solver-order-independent) + 2 direct C unit tests. **Out of scope for this slice**: repeated-pole multiplicity grouping (FP-tolerance choice is non-trivial; `a'(p_i) → 0` for repeated `p_i` so the cover-up rule degrades) — most DSP filter designs produce distinct poles by construction. `residuez` (z-domain) is a separate slice. |
| Filter implementation (Tier-1 §2.5) — `filtfilt(b, a, x)`, `sosfilt(sos, x)`, `impz(b, a, N)`, `stepz(b, a, N)`, `grpdelay(b, a, N)` | ✅ shipped | Internal `filter_flat_` helper (direct-form-II transposed) drives all five entries. `filtfilt` uses lfilter_zi-based steady-state initial conditions (scipy's `method='pad'` default, `padtype='odd'`): the IC vector solves `(I − A)·zi = B` for the canonical companion-form state-transition + each pass scales `zi` by the boundary value of the padded signal, so constant signals are now preserved exactly. `sosfilt` cascades L × 6 `[b0 b1 b2 a0 a1 a2]` rows. `impz`/`stepz` drive `filter_flat_` with `[1; 0; …]` / `ones(N, 1)`. `grpdelay` evaluates `H(e^{jω})` at the freqz grid and at `ω + dω` with one-step phase unwrapping in `(−π, π]`. Gating: `test/Run/sig_filt.m` (4-lane). |
| Filter implementation — open: strict 1996 Gustafsson method for `filtfilt` (scipy's `method='gust'` — explicit edge-elimination linear system instead of padding), `phasez`, `zerophase` | 🔵 | §2.5 follow-on. |
| Nonparametric spectral (Tier-2 §3.1) — `periodogram(x)`, `pwelch(x, win, noverlap)`, `cpsd(x, y, win, noverlap)`, `mscohere(x, y, win, noverlap)`, `tfestimate(x, y, win, noverlap)` | ✅ shipped | Single-output, default fs = 1. `cpsd` / `tfestimate` return matlab_mat_c (TS lane returns magnitude only — no native complex). Periodogram is `\|FFT\|² / N` single-sided; pwelch is the segment-and-average periodogram with window-energy normalisation. **Open**: dpss + pmtm (multitaper — needs Slepian eigendecomposition); 2-/3-return `[P, f, …]` forms. |
| Linear prediction + parametric PSD (Tier-2 §3.2) — `levinson(r, p)`, `lpc(x, p)`, `aryule(x, p)`, `arburg(x, p)`, `pyulear(x, p, N)`, `pburg(x, p, N)` | ✅ shipped | Levinson-Durbin recursion; LPC = biased-autocorr + Levinson; Burg uses forward+backward prediction-error recursion. Parametric PSD evaluates σ²·\|1/A(e^{jω})\|² on an N-point grid. **Open**: pcov / pmcov (covariance / modified covariance AR), pmusic / peig / rootmusic / rooteig (subspace methods), prony / stmcb (IIR design from impulse response). |
| Time-frequency (Tier-2 §3.3) — `spectrogram(x, win, noverlap)` | ✅ shipped (single-output) | `\|STFT\|²` per (freq, frame), output (M × K). **Open**: `stft` / `istft` (with COLA inversion), `pspectrum`, `instfreq`, `instbw`. |
| Other transforms (Tier-2 §3.4) — `dct`, `idct`, `fwht`, `hilbert`, `goertzel` | ✅ shipped | DCT-II / DCT-III via direct O(N²); fwht via in-place butterfly (natural Hadamard ordering, divided by N); hilbert via FFT zero-negative-half + IFFT (returns matlab_mat_c); goertzel single-bin (returns 1×1 complex). **Open**: czt (chirp Z-transform), dst / idst, cceps / rceps / icceps. |
| Multirate (Tier-3 §4.1) — `upfirdn(x, h, p, q)`, `decimate(x, r)`, `interp(x, r)`, `resample(x, p, q)` | ✅ shipped | Real anti-aliased multirate replacing the toy `upsample` / `downsample` stubs. `upfirdn` is the kernel (upsample-by-p → FIR-h → downsample-by-q); `decimate`/`interp`/`resample` are wrappers that build a default Hamming-windowed `fir1` lowpass. Output lengths: `decimate` = ceil(N/r), `interp` = N·r, `resample` = ceil(N·p/q). **Open**: `polyphase` decomposition; group-delay correction for FIR transient. |
| Waveform generators (Tier-3 §4.2) — `chirp(t, f0, t1, f1)`, `sawtooth(t, w)`, `square(t, duty)`, `gauspuls(t, fc, bw)`, `rectpuls(t, w)`, `tripuls(t, w)`, `sinc(x)` | ✅ shipped | All take a time-vector and return same-shape signal. chirp linear method only; default-arg shorthands (sawtooth(t) ⇒ width=1) deferred. **Open**: chirp quadratic / logarithmic / hyperbolic methods, `pulstran`, `diric`, `gmonopuls`, `vco`. |
| Alignment helpers (Tier-3 §4.4) — `xcov(x, y)`, `finddelay(x, y)`, `dtw(x, y)` | ✅ shipped | xcov is mean-removed cross-correlation; finddelay is argmax of \|xcorr\|; dtw is dynamic time warping (scalar distance). **Open**: `alignsignals` (multi-return), `gccphat`, `xcorr` scaling-option strings ('biased'/'unbiased'/'normalized'/'coeff'). |
| Pulse / waveform measurements (Tier-3 §4.3, full surface) — `findpeaks` (1- and 2-return), scalar reductions `rms` / `peak2peak` / `peak2rms` / `rssq`, signal cleanup `medfilt1` / `hampel` / `envelope`, pulse statistics `midcross` / `risetime` / `falltime` / `dutycycle`, plus `statelevels` / `slewrate` / `pulseperiod` / `pulsewidth` / `overshoot` / `undershoot` / `settlingtime` | ✅ shipped | `findpeaks` uses strict-monotonic local-maximum definition (no plateaus, no endpoints). `medfilt1` has zero-padded edges with odd-N coercion. `hampel` uses 3·1.4826·MAD threshold. `envelope` is peak-interpolation linear between local maxima. `statelevels` uses the histogram-based estimator (100 uniform bins, top-count bin in each half); the rest of the §4.3 tail (`slewrate` = 0.8·(hi−lo)/risetime, `overshoot`/`undershoot` in % of state range, `settlingtime(x, d)` with default `d = 0.02`) sits on top of `statelevels` + the existing `mean_transit_` / `matlab_midcross` scaffolding. Gating: `test/Run/sig_peaks.m`, `sig_pulse.m`, `sig_stat.m`, `sig_pulse_tail.m`. **Open**: name-value options for `findpeaks` (`MinPeakHeight` / `MinPeakDistance` / `MinPeakProminence` / `Threshold` / `SortStr`) — gated on Sema's name-value-arg parsing. |
| Wavelets / Wigner-Ville / synchrosqueezed (`cwt`, `dwt`, `wvd`, `fsst`) | 🔵 | Tier-4 §5.4. |
| `digitalFilter` / `designfilt` system object | 🔵 | Tier-4 §5.1 — needs Tier-1 IIR/FIR shipped first. |

### Communications Toolbox — Tier-1 base layer (function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §2. The base-layer prerequisites that gate every higher Comm tier — bit sources, RNG seed control, AWGN channel, BER/SER measurement. Function-form, numeric-tag dispatch; all entries live in `runtime/runtime_comm.cpp`. End-to-end demos under `examples/comm/` (`comm_tier1_smoke.m`, `source_bits.m`, `ber_awgn_uncoded.m`).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| §2.1 randi | `randi(imax)` (scalar), `randi(imax, n)` (n×n), `randi(imax, m, n)` (m×n) | ✅ shipped | Reuses the existing xorshift64 kernel + `matlab_rng_state` so seeding is shared with `rand` / `randn`. `randi(imax)` returns f64 scalar; multi-arg forms return matrix. The bracketed `randi([imin imax], m, n)` shape is exposed as `randi_range` for the function-form lane (callers can use scalar imin/imax directly). |
| §2.2 rng | `rng(seed)`, `rngDefault()`, `rngShuffle()`, `s = rngGet()`, `rngSet(s)` | ✅ shipped | The 'default' / 'shuffle' string variants are exposed as separate named entries to keep the numeric dispatch lane clean. Save/restore round-trips the xorshift state through an f64 (loses 11 LSBs but the mixer re-spreads entropy within 2 advances). |
| §2.3 randsrc / randerr | `randsrc(m, n, alphabet)` (uniform pick from a column-vector alphabet), `randsrcWeighted(m, n, alphabet, probs)` (with explicit probability vector), `randerr(m, n, errs)` (m×n binary matrix, exactly `errs` ones per row via Fisher-Yates partial shuffle) | ✅ shipped | |
| §2.4 bit / int conversion | `int2bit(ints, nbits)` (MSB-first), `bit2int(bits, nbits)` (MSB-first inverse), `de2bi(d, n)` (LSB-first legacy), `bi2de(b)` (LSB-first inverse) | ✅ shipped | Round-trip verified: `bit2int(int2bit(x, 4), 4) == x`, `bi2de(de2bi(x, 4)) == x`. nbits clamped to [1, 53] so the f64 lane never loses precision. |
| §2.5 awgn | `awgn(x, snr_dB)` ('measured' default — signal power read from x), `awgn(x, snr_dB, sigpower_dBW)` (explicit signal power) | ✅ shipped | Polymorphic on the descriptor magic — real `matlab_mat *` produces real noise (sigma² = noiseP); complex `matlab_mat_c *` produces complex noise with sigma²/2 per axis so the total variance matches signal_power / snr_lin. Shares the seeded PRNG with the rest of the runtime. |
| §2.6 biterr / symerr | `biterr(x, y)` → ratio, `biterrCount(x, y)` → integer count, `biterrK(x, y, k)` → ratio for k-bit symbols, `symerr(x, y)` → ratio, `symerrCount(x, y)` → integer count | ✅ shipped | Single-return-form convention: `biterr` returns the BER (second of MATLAB's `[nerr, ratio]` pair). Verified: `biterr([0;1;1;0;1;0;1;1], [0;0;1;0;1;1;1;0]) = 3/8`. The BPSK Monte-Carlo demo (`examples/comm/ber_awgn_uncoded.m`) tracks Q(sqrt(SNR_lin)) within ~5% from 4 dB onward at 50,000 bits per point. |
| Test gate (BPSK over AWGN) | `examples/comm/ber_awgn_uncoded.m` | ✅ closes Tier-1 test gate | Canonical "modulate → AWGN → demod → BER" loop with theoretical-curve overlay. At 50 k bits / point: SNR 4 dB → sim 0.058 vs theory 0.060; SNR 6 dB → sim 0.024 vs theory 0.024; SNR 10 dB → sim 0.00084 vs theory 0.00075. |

### Propagation Models (Communications + Antenna Toolboxes — function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §3. Function-form surface — no classdef System Objects required, so this track ships in parallel with the SO-gated Comm Tier-3+ / RF / Antenna arcs. All entries live in `runtime/runtime_prop.cpp`. End-to-end demos under `examples/rf/` (Barbados PtP + ITM coverage, sector-coverage SINR, Fresnel/diffraction, pattern sampling).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| Closed-form ITU-R / NIST path loss (PROP-Tier-1a §3.1.1) | `fspl(d, freq)`, `pathlossRain(d, f, R, pol)`, `pathlossGas(d, f, T, P, rho)`, `pathlossFog(d, f, M)`, `pathlossCloseIn(d, f, n, σ, d0)` | ✅ shipped | All numeric; numeric tag for polarisation (0=H, 1=V). Rain coefficients interpolated log-log in frequency from the ITU-R P.838-3 table. Free-space matches `20·log10(4π·d/λ)` exactly. |
| Cellular empirical (PROP-Tier-1a §3.1.2) | `pathlossHata(f, ht, hr, d, env)`, `pathlossCost231`, `pathlossEgli`, `pathlossEcc33`, `pathlossSui(..., terrain)`, `pathlossEricsson9999` | ✅ shipped | Numeric env tag: 1=urban-large, 2=urban-medium, 3=suburban, 4=open. SUI terrain tag: 1=A, 2=B, 3=C. Hata urban-large at 30 m / 1.5 m / 1 km / 2.4 GHz → 137.56 dB. |
| Fresnel-zone math (PROP-Tier-1a §3.1.3) | `fresnelZoneRadius(d1, d2, λ, n)`, `fresnelClearance(profile, h_tx, h_rx, d_total, λ, n)` | ✅ shipped | First-zone radius at 5 km midpoint / 5.8 GHz ≈ 5.6 m. Clearance returns 0–100 (% of the n-th Fresnel zone); >60% is the TIA-recommended bar. |
| Knife-edge diffraction (PROP-Tier-1a §3.1.4) | `diffractionKnifeEdge(h, d1, d2, λ)`, `diffractionBullington(profile, h_tx, h_rx, d_total, λ)`, `diffractionDeygout(...)` | ✅ shipped | Single-edge uses ITU-R P.526 closed-form `J(v) = 6.9 + 20·log10(√((v−0.1)²+1) + v − 0.1)`. Bullington picks the steepest-up TX slope vs steepest-up RX slope and reduces to one equivalent edge. Deygout recurses 3 deep, picking the dominant edge per sub-path. |
| Geographic helpers (PROP-Tier-1a §3.1.5) | `haversine(lat1, lon1, lat2, lon2)`, `bearing(...)`, `vincenty(...)`, `greatCircleDestLat(lat1, lon1, d_m, az)`, `greatCircleDestLon(...)` | ✅ shipped | Haversine on the mean-sphere (R = 6371.009 km); Vincenty on WGS-84 with 30-iteration fallback to Haversine on near-antipodal degeneracy. Verified: LHR→JFK = 5540 km Haversine vs 5555 km Vincenty (vs published 5555 km), bearing 288°. Sao Paulo→Tokyo = 18537 / 18534 km. |
| Longley-Rice / ITM (PROP-Tier-2a §3.2) | `itmPathloss(profile, freq, ht, hr, pol, climate, Ns, σ, εr, d_total, q_t, q_l, q_s)` | ✅ shipped (engineering port) | Closed-form regime blend (line-of-sight / smooth-Earth diffraction / tropospheric scatter) with smooth horizon transition; climate codes 1–7 per the NTIA convention; reliability triple drives a Gaussian-quantile correction on the median (σ_T ≈ 3.5 dB + freq-aware, σ_L ≈ 2 dB, σ_S = 1.5 dB). Free-space floor enforced. For NTIA byte-identical conformance, swap in the v7.0 reference port (carved out per roadmap §3.7). Verified: 15 km flat 5.8 GHz V-pol climate-5 median = 131.24 dB (= FSPL — link well within horizon, climate clamp at free-space); 60 km same setup ranges 146–151 dB across climates 1–7 (equatorial→desert); reliability sweep 50/50/50 → 95/99/99 adds ~20 dB. |
| Terrain + LOS + link budget + single-TX coverage (PROP-Tier-2b §3.3) | `terrainProfile(heightmap, lat/lon-box, lat1, lon1, lat2, lon2, n)`, `losObstruction(...)`, `losClear(...)`, `linkBudget(...)` (struct return), `coverageGrid(...)` | ✅ shipped | Heightmap is a `[NumLat × NumLon]` real matrix spanning a user-supplied bounding box; bilinear sample along the great-circle path. LOS check uses 4/3-Earth effective bulge. `linkBudget` returns a struct with `Distance`, `Azimuth`, `PathLoss`, `TxPower_dBm`, `ReceivedPower`, `NoiseFloor`, `Snr`, `LinkMargin`, `FresnelClearance`, `LosClear`, `Frequency`, `Model`, `Profile`. Auto-fetch of SRTM / DTED tiles is **carved out** (user supplies the heightmap directly). |
| Directional antenna patterns (PROP-Tier-3 §3.4.1) | `sectorPattern(az, el, bw_az, bw_el, gain, fb_dB)`, `cosinePattern(az, el, hb_az, hb_el, gain, n)`, `gaussianPattern(...)`, `isotropicPattern(...)` | ✅ shipped | 3GPP TR 36.942 sector (default 65° az / 10° el / 25 dB front-to-back); cosine-power (good for dishes — boresight 22 dBi / n=30 → 21.7 dBi at ±4°, 14 dBi at ±20°); Gaussian roll-off (no sidelobes). |
| Mount orientation (PROP-Tier-3 §3.4.2) | `applyMountOrientation(az_w, el_w, m_az, m_tilt)` (1×2 matrix), `applyMountAz(...)` / `applyMountEl(...)` (scalars) | ✅ shipped | Scalar siblings exposed because the matrix-return form requires indexing of the 1×2 result, which the dispatch can't always retype when feeding directly into another pattern call. |
| Multi-site coverage with directional antennas (PROP-Tier-3 §3.4.3) | `coverageGridMulti(sites, antennas, heightmap, bbox, NLat, NLon, rx_h, rx_g, model, agg, climate, q_t, q_l, q_s)` | ✅ shipped | `sites` is `[N × 6]` (lat, lon, h_m, P_W, f_Hz, n_ant). `antennas` is `[Σn_ant × 8]` (pattern code, peak gain, bw_az, bw_el, fb_or_n, mount_az, mount_tilt, _). `agg` ∈ {0 = best-server, 1 = sum-power, 2 = SINR (dB)}. Verified: two-site three-sector FSPL best-server gives 48×48 grid spanning −43 / −76 / −85 dBm; same scenario SINR yields ~41 / 7 / 0 dB. |
| Examples | `examples/rf/coverage_barbados.m` | ✅ | Mount Hillaby ↔ Bridgetown 13.8 km PtP link, dual 22 dBi cosine-pattern dishes, Longley-Rice (climate 3 maritime subtropical, 80/99/99 reliability) → 149.5 dB path loss, −68.5 dBm at RX, 35 dB margin. Plus a 48×48 best-server coverage map from Mount Hillaby. Companion demos: `pathloss_models.m`, `fresnel_diffraction.m`, `antenna_patterns.m`, `longley_rice_link.m`, `geo_helpers.m`, `coverage_three_sector.m`, `prop_smoke.m`. README in `examples/rf/`. |
| Carved out (deliberately deferred) | Site Viewer 3-D map, ray tracing through OSM buildings, auto-fetch SRTM/DTED/OSM tile servers, TIREM, MSI Planet I/O, animated live coverage, GPU acceleration | 🔴 | Per `comm_toolbox_roadmap.md` §3.7. |
| Classdef wrappers (PROP-Tier-1b §3.5) | `propagationModel(...)`, `txsite(...)`, `rxsite(...)`, `pathloss(prop, rx, tx)`, `link(...)`, `coverage(tx, prop)`, `los(...)` | 🔵 | Gated on the System-Object lowering fix (CST roadmap §12 / §11.1). Thin parameter-holders + method delegates to the function-form layer; +3 sessions once unblocked. |

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
| SystemVerilog (ASIC, synthesizable) emission | ✅ | `-emit-systemverilog`. Vendor-neutral, synthesizable RTL targeting ASIC flows. Tier-1 closure shipped: scalar combinational + FSMs + fixed-point pipeline + persistent fi-arrays + readability polish + bit-slicing `x(hi:lo)` (any width 1..64) + runtime-indexed persistent fi-arrays (auto-decoded regfile pattern) + hierarchical multi-module emission (`func.call` → SV instance with auto-wired clk/rst_n). 77 golden fixtures lint clean under Verilator (incl. `aes_round`, `cic_decimator`, `cordic_pipe`, `crc32`, `fir_asic_pipelined`, `i2c_bit_bang`, `regfile_dyn`, `spi_master`, `uart_rx`, `vector_processor`, plus `hier_combinational` / `hier_sequential` for multi-module). 7 fi-spec ↔ SV declaration regression tests in `test/EmitSVPorts/`, 2 boolean-port lint-hint tests in `test/EmitSVHint/`, 10 synthesizability-gate diagnostic tests in `test/EmitSVFail/`. Open: 2-D fi matrices, RAM inference, CORDIC for transcendentals. See `docs/sv_supported_subset.md` (supported-subset reference) and `docs/emit_systemverilog.md` (backend architecture). |

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

- **Interactive UI / GUIs** — no live windows, mouse picking, App Designer.
  Headless plotting (`plot`, `surf`, `bar`, ... → PNG/SVG/PDF) is shipped;
  see [`plotting.md`](plotting.md).
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
