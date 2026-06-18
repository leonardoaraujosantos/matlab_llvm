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
- GUI APIs (interactive figures, App Designer, Live Editor inline plots,
  ginput, pan/zoom/rotate)
- full MATLAB compatibility
- `.mat` file compatibility

In scope (subsets shipped, covered by dedicated docs):
- **Plotting**: headless Cairo-backed `plot` / `bar` / `surf` / etc. with
  PNG/SVG/PDF output, plus `getframe` + `VideoWriter` animation capture to
  MP4 (H.264) / AVI (MJPEG) via libav, on by default within a `WITH_PLOT`
  build (opt out with `-DMATLAB_LLVM_WITH_PLOT_FFMPEG=OFF`). See
  [`plotting.md`](plotting.md).
- **Toolboxes (thirteen shipped surfaces)** — earlier revisions of this
  doc listed toolboxes as out of scope; that scope has expanded. The
  runtime now ships practical subsets of:
  Signal Processing ([`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md)),
  Control System ([`control_toolbox_roadmap.md`](control_toolbox_roadmap.md)),
  Communications ([`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md)),
  RF ([`rf_toolbox_plan.md`](rf_toolbox_plan.md) + [`verilog_a_plan.md`](verilog_a_plan.md)),
  Antenna ([`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md)),
  Propagation Models ([`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md)),
  Optimization ([`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)),
  Model Predictive Control ([`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md)),
  System Identification ([`ident_toolbox_roadmap.md`](ident_toolbox_roadmap.md)),
  Partial Differential Equation ([`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md)),
  Symbolic Math via SymPP ([`sym.md`](sym.md) / [`symbolic_toolbox_roadmap.md`](symbolic_toolbox_roadmap.md)),
  Fixed-Point Designer (`fi`) ([`fixed_point_toolbox_roadmap.md`](fixed_point_toolbox_roadmap.md) / [`emit_fixed_point.md`](emit_fixed_point.md)),
  and Stateflow / mStateflow ([`mStateflow_roadmap.md`](mStateflow_roadmap.md)).
  A fourteenth — Global Optimization ([`global_optim_toolbox_roadmap.md`](global_optim_toolbox_roadmap.md))
  — is complete (all 6 tiers: `ga` / `particleswarm` / `simulannealbnd` +
  `MultiStart` / `GlobalSearch` / `patternsearch` / `surrogateopt` / `gamultiobj` / `paretosearch`
  + `optimoptions('ga')` options surface + integer-constrained `ga`).
  A fifteenth — Statistics and Machine Learning
  ([`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md)) — is
  complete (all 6 tier cores: descriptive + distributions +
  `makedist`/`fitdist`; hypothesis tests + ANOVA; regression; PCA +
  clustering; classification + ensembles; `bayesopt` + Markov models —
  the `iris_classify` headline is closed).
  A sixteenth — Image Processing
  ([`image_toolbox_roadmap.md`](image_toolbox_roadmap.md)) — is complete
  (all 6 tier cores: I/O incl. **real PNG + baseline-JPEG `imread`** and
  lossless PNG `imwrite`; filtering; geometric; morphology; segmentation +
  `regionprops`; transforms/quality/colour/deblur — the `rice_grains`
  headline is closed).
  A seventeenth — Curve Fitting
  ([`curve_fitting_toolbox_roadmap.md`](curve_fitting_toolbox_roadmap.md)) —
  is complete (all 6 tiers: `fit(x,y,'polyN')` → `cfit` + `feval`/`gof`;
  nonlinear `exp`/`power`/`gauss`/`sin`/`fourier` library via hand-coded
  Levenberg-Marquardt + `fitoptions` + robust IRLS; custom `fittype`
  equations + `confint`/`differentiate`/`integrate`; interpolant fits +
  `smooth`/`csaps`; polynomial surfaces → `sfit`; ppform `spline`/`pchip` +
  `fnval`/`fnder`/`fnint`).
  An eighteenth — Wavelet
  ([`wavelet_toolbox_roadmap.md`](wavelet_toolbox_roadmap.md)) — is complete
  (all 6 tiers, matrix lane over the shipped `conv`/`fft`: `wfilters`/`dwt`/
  `wavedec`/`waverec` with exact perfect reconstruction across the
  `haar`/`db`/`sym`/`coif` catalogue; denoising `wthresh`/`thselect`/
  `wnoisest`/`wdenoise`/`measerr`; FFT-domain `cwt`/`icwt`/`wcoherence`;
  undecimated `modwt`/`modwtmra` + 2-D `dwt2`/`wavedec2`; wavelet packets
  `wpdec`/`wprec`/`besttree`; special topics `emd`/`vmd`/`ewt`/
  `matchingPursuit` + `waveletScattering`→`fitcsvm` — the `denoise_signal`
  headline (SNR +21 dB) is closed).
  Apps / Live Editor / GUIs / Simulink integration for each are
  individually carved out — see the per-toolbox roadmap.
- **Core-compiler roadmap (not a toolbox):**
  [`any_shape_roadmap.md`](any_shape_roadmap.md) — arbitrary-shape (N-D)
  array support. 2-D + arbitrary-depth 3-D arrays ship today (RAM-bound;
  `300×200×4` works for `zeros`/indexing/`size`); the doc plans full
  MATLAB-faithful N-D (Tier A 3-D polish → Tier B reshape/permute → Tier C
  rank-N descriptor) and the diff-against-`main` regression strategy for
  proving nothing breaks.

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
| Comparison, logical, short-circuit (`== ~= < <= > >= & | && || ~`) | ✅ | Scalar `& \| && \|\| ~` lower to `arith.{andi,ori,xori}` in `LowerScalarsToArith` (operands truth-coerced to i1). `&&` / `\|\|` are eager — the frontend emits both operands, so there is no runtime short-circuit (an RHS guarded against an error by the LHS is not skipped). |
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
| Polymorphic call monomorphization | ✅ | Sema-time clone-per-signature pass (`lib/Sema/Monomorphize.cpp`, #38 / PR #39). Stamps concrete arg types on `Function::ParamTypeStamps` so AST→MLIR emits concrete `func.func` sigs (e.g. `@sq(double)` + `@sq__s1(ptr)` for `sq(5)` vs `sq([1 2 3])`). Default on; `MATLAB_LLVM_SEMA_MONO=0` falls back to the late MLIR mono. Matrix / arity-varying / `varargout` still flow through the late pass — tracked in #40. |
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
| `fi` (Fixed-Point Designer) | 🟡 | Phases 1–5 shipped: scalar `fi(value, signed, WL, FL)` and `fi(value, T)` / `fi(value, T, F)` constructors with literal-fold, `+ - *`, `(:)` type-preserving assignment, `Saturate`/`Wrap` overflow, all five rounding modes (`Floor`/`Nearest`/`Zero`/`Ceiling`/`Convergent`), sub-native WL (e.g. WL=12 in i16 lane), implicit `fi + double` promotion, `int(n)` / `storedInteger(n)` / `double(n)`, `bin/hex/dec` display, **fi arrays** (`fi(zeros(1,N),...)`, indexing, slicing, vector concat, `sum`/`mean`), **persistent storage** of fi arrays, `numerictype` / `fimath` first-class objects, `setfimath` / `removefimath`, `reinterpretcast`, `-emit-fixed-point-report` driver flag. Gating test: FIR filter in `test/Run/fi_filter.m`. Storage = native `int8/16/32/64`. **Still open** (Tier-6 follow-ons): function-internal fi typing across user calls (`function y = f(x)` doesn't propagate the spec — biggest UX gap), 2-D fi matrices (1-D shipped), reductions tail (`prod`/`min`/`max`/`cumsum` on fi), `fi` parfor reductions, `fipref` display preferences, slope/bias scaling, complex `fi`, 3-D fi arrays. emit-typescript: FIR test skipped (BigInt-vs-number coercion). See [`docs/fixed_point_toolbox_roadmap.md`](fixed_point_toolbox_roadmap.md) §7 (tiered compatibility plan) and [`docs/emit_fixed_point.md`](emit_fixed_point.md) §10.1 (implementation reference). |
| N-D arrays (3-D) | 🟡 | `zeros(m,n,p)` / `ones(m,n,p)` + scalar `A(i,j,k)` read/write, `size(A, 3)`, `numel`, `ndims` |
| N-D arrays (>3D) | ❌ | |
| Sparse matrices | ❌ | |
| `datetime` / `duration` | ✅ shipped (Phase 5.1) | Scalar `datetime` (Unix-epoch seconds) and `duration` descriptors with constructors (`datetime(y,m,d)`, `datetime(y,m,d,h,mn,s)`, `datetime("now")`, `seconds/minutes/hours/days/years(n)`), MATLAB-default display formatting, and arithmetic (`dt + dur → dt`, `dt - dt → dur`, `dur ± dur → dur`). UTC; civil-date math via Howard Hinnant's algorithm. C / Python / TypeScript runtimes byte-identical. Gating test: `test/Run/datetime_basic.m`. Vector `datetime` / `duration` shipped (Phase 5.4): `datetime(...) + days(0:N)` → datetime column, `dt_vec - dt → dur_vec`, length / numel / indexing / disp (`test/Run/datetime_vec.m`). **Open follow-up**: calendar arithmetic (months, years), zoned datetimes, `between`/`caldays`/`calmonths`/`calyears`. |
| `categorical` | ✅ shipped (Phase 5.2) | 1-D categorical built from a string-array literal (`categorical(["a","b","a"])`). Auto-deduplicates and alphabetically sorts category names; per-element codes are 1-based with 0 = `<undefined>`. Surfaces: `length(c)`, `numel(c)`, `categories(c)` (returns a cell of category strings), `iscategory(c, "name")`, `disp(c)`. C / Python / TypeScript runtimes byte-identical. Gating test: `test/Run/categorical_basic.m`. **Open follow-up**: `categorical(values, valueset, catnames)` full constructor, `addcats`/`removecats`/`mergecats`/`renamecats`, ordinal categoricals, comparison ops beyond `==`. |
| `table` | ✅ shipped (Phase 5.3) | Column-major record with named variables; constructors `table(c1, c2, ...)` (auto-named Var1..VarN) and `table(c1, c2, ..., 'VariableNames', {n1, n2})`. Surfaces: `T.<name>` column read / write (with dynamic column add), `height(T)`, `width(T)`, `numel(T)`, `size(T, dim)`, `disp(T)` (right-aligned column body with header + underline). Each column stored as a `matlab_mat *`. C / Python / TypeScript runtimes byte-identical on the C/TS lanes; Python ships a `.stdout-python` override (numpy 2-D array repr for column print). Gating test: `test/Run/table_basic.m`. **Open follow-up**: heterogeneous columns (mixed numeric / string / categorical), row indexing `T(i,:)`, sub-table extraction, `readtable`/`writetable`. |
| `timetable` | ✅ shipped (Phase 5.4) | `table`-style column store + a `datetime` RowTimes axis. Constructors `timetable(c1,...,'VariableNames',{...},'RowTimes',dt)` + `table2timetable`. Surfaces: `TT.Var` / `TT.Time` dot-read, `TT(:,'col')` + `TT(idx,:)` subscripts, `TT.Properties.Description=`, `timerange(t1,t2,'closed'|...)` + `TT(tr,:)`, `retime(TT,'weekly',method)` (6 aggregators), `synchronize`, bracket horz-cat `[TT1 TT2]`, `fillmissing`/`summary`/`head`, `movavg`/`macd`, `plot(TT.Time,TT.Var)`. Gating: `test/Run/timetable_*.m` + `using_timetables_in_finance.m`. **Open follow-up**: `fints` migration, `withtol`, `rowfun` handle ABI, multi-column retime. |

### Symbolic Math Toolbox (`sym` / `syms`)

Opt-in via `-DMATLAB_LLVM_WITH_SYM=ON` — requires [SymPP](https://github.com/leonardoaraujosantos/SymPP).
See [`docs/sym.md`](sym.md) for the full user-facing surface, and
[`docs/symbolic_toolbox_roadmap.md`](symbolic_toolbox_roadmap.md) for
the tiered compatibility plan (Tiers 1 → 4 ✅ closed; Tier-5
`matlabFunction` handle + AppliedFunction lifting + cell-array
array-arg lowering + extended assumption properties is the next
slice; Tier-6 `-emit-python` via SymPy is the multi-backend track).

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

## 4. Built-in functions (runtime: 12 C++ TUs under `runtime/`)

The runtime is now split across 12 `.cpp` translation units totalling
~52 kLOC and ~1,100 exported C-ABI entries. The architecture, ABI
conventions, and per-TU contents are documented in
[`runtime.md`](runtime.md). The matrix below is the per-feature
shipped / partial / missing inventory; per-toolbox tier plans live in
the companion roadmap docs (signal / control / comm / RF / antenna /
propagation / optim / MPC / PDE / symbolic / fixed-point / stateflow /
Verilog-A).

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
| `sind`, `cosd`, `tand`, `asind`, `acosd`, `atand`, `atan2d` | ✅ | Degree-argument trigonometry; scalar + matrix runtime entries (`matlab_sind_s` / `matlab_sind_m` / …). |
| `log1p`, `expm1`, `factorial`, `nextpow2`, `hypot`, `nthroot`, `gcd`, `lcm`, `isprime`, `nchoosek` | ✅ | Scalar forms (the common case); element-wise/vector forms (`primes`, `factor`, vector `isprime`) are follow-ups. |
| `conj`, `real`, `imag`, `angle` | ✅ | Polymorphic — accept either real or complex input |
| `fft`, `ifft`, `fft2`, `ifft2` | ✅ | Pure-C Cooley-Tukey radix-2 + Bluestein for general N. See [`docs/complex.md`](complex.md). |

### Reductions

| Function | Status |
|---|:-:|
| `sum` (all elements, column-wise, or `sum(A, dim)`) | ✅ |
| `min`, `max`, `mean`, `prod` (same 3 forms as `sum`) | ✅ | `min([])`/`max([])` return `[]` (0×0); `sum([])=0`, `prod([])=1`, `mean([])=NaN`. |
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
| `norm` (Frobenius / vector 2-norm), `norm(x, p)` (p = 1 / 2 / Inf; vector + induced matrix), `trace`, `kron` | ✅ | |
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
| Higher-order stiff (`ode15s`, `ode23t`, `ode23tb`, `ode15i`) | ❌ | `ode15s` (variable-order BDF + Newton) is the natural next step on top of the shipped `ode23s` infrastructure. Drafted as the DAE-solver core of [`dae_solver_roadmap.md`](dae_solver_roadmap.md) (Tiers 1–2) — mass-matrix `ode15s` + fully-implicit `ode15i` + `decic`, the foundation for the Verilog-A analog circuit simulator (MNA) in the same roadmap. |
| Non-stiff multistep (`ode113`) and high-order (`ode78`, `ode89`) | ❌ | |
| BVP (`bvp4c`, `bvp5c`), DDE (`dde23`) | ❌ | |
| `pdepe` — 1-D parabolic-elliptic PDE via method-of-lines | ✅ | Cartesian / cylindrical / spherical (`m = 0, 1, 2`); Dirichlet, Neumann, Robin BCs; non-uniform mesh; scalar PDE. Wraps `ode23s_v` for stiff time integration. Output `sol` is N_t × N_x. See [`ode.md`](ode.md). |
| `pdepe` extensions — multi-component systems (`npde > 1`); axis-of-symmetry `xmesh(1) = 0` for `m > 0`; `odeset` plumbed through | ❌ | Tracked in roadmap. |
| Model Predictive Control (`mpc`, `mpcstate`, `mpcmove`, `sim`, `nlmpc`, `nlmpcmove`, explicit / adaptive / time-varying / finite-control-set variants) | ✅ | **Tiers 1 → 6 shipped** via [`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md). Linear MPC on a hand-coded KWIK active-set QP; output + mixed input/output constraints + ECR soft slack + output-disturbance integrator + `mpcmoveopt` run-time overrides; adaptive (`mpcmoveAdaptive`) / time-varying (`mpcmoveTV`) / gain-scheduled / LPV + the mflow `MpcMove` block (emit-c/cpp/python/SV + cocotb SIL); explicit MPC via offline grid tessellation (`generateExplicitMPC` / `mpcmoveExplicit`) + standalone `mpcActiveSetSolver` + finite-control-set `mpcmoveFinite`; nonlinear MPC (`nlmpc` / `nlmpcmove` over `fmincon` with an RK4 prediction rollout, anonymous-handle StateFcn); Tier-6 carve-down sweep (continuous-plant auto-c2d, rate bounds, MV-tracking `Wu`/`u_target`, `setEstimator`/`getEstimator`/`review`, `mpcsimopt`, reference previewing). **25 MPC tests green.** Headlines `examples/mpc/{dc_servo_mpc,paper_machine,pendulum_nlmpc,twin_rotor_nlmpc}.m` + `examples/quadrotor/`. **Carve-outs**: MPC Designer GUI, Simulink MPC block, FORCESPRO/Embotech NLP, CUDA, data-driven / passivity / C-GMRES NMPC, economic MPC, `nlmpcMultistage`, `getCodeGenerationData`, IP QP solver. |
| System Identification (`iddata`, `idpoly`, `idss`, `idfrd`, `idgrey`, `idnlgrey`, `arx` / `ar` / `armax` / `oe` / `bj` / `iv4` / `tfest` / `n4sid` / `ssest` / `greyest` / `nlgreyest` / `etfe` / `spa` / `impulseest` / `forecast` / `pe` / `resid` / `delayest`, `extendedKalmanFilter` / `unscentedKalmanFilter`, `recursiveARX` / `recursiveLS`, `arxOptions` + `getcov` / `getpvec` / `setpvec`) | ✅ | **All 6 tiers shipped** via [`ident_toolbox_roadmap.md`](ident_toolbox_roadmap.md). Backed by [`runtime/toolbox/ident/runtime_ident.cpp`](../runtime/toolbox/ident/runtime_ident.cpp) + [`runtime/toolbox/ident/ident_classdefs.m`](../runtime/toolbox/ident/ident_classdefs.m). T1 `iddata` + `arx`/`ar` (QR-LS via normal equations) + `sim`/`predict`/`compare`(NRMSE)/`goodnessOfFit`/`fpe`/`aic` + `ss`/`tf`(idpoly). T2 PEM core: `armax`/`oe`/`bj` via `lsqnonlin` with one general predictor `e=(D/C)(A·y−B/F·u)`, `iv4` instrumental-variables, `pe`/`resid` whiteness, `delayest`. T3 subspace state-space: `n4sid`/`ssest` via Ho-Kalman/ERA (block-Hankel SVD through symmetric Gram-eig since `matlab_svd` returns only singular values), `tfest`, `idss`, state-space sim/compare, `ss(idss)`. T4 non-parametric: `etfe`/`spa` → `idfrd` (real magnitude/phase columns), `impulseest`, `forecast`, **linear grey-box** `greyest`/`idgrey` (function-handle structure fn → packed continuous `[A B; C D]` + ZOH `c2d` + `lsqnonlin`). T5 heavy: **EKF/UKF** `extendedKalmanFilter`/`unscentedKalmanFilter` (the project's first dynamic Kalman filtering loop — CST `kalman` is steady-state-gain only), forgetting-factor RLS `recursiveARX`/`recursiveLS`, nonlinear grey-box `nlgreyest`. T6 polish: regularized `arx(data,orders,arxOptions)` ridge `(ΦᵀΦ+λI)⁻¹Φᵀy` + `getcov`/`getpvec`/`setpvec` parameter introspection. **20 ident tests green; 435/435 full regression clean.** Seven headlines: `examples/ident/{arx_lab_process,armax_refine,data_driven_mpc,greybox_msd,ukf_state_estimation,recursive_arx_tracking,arx_regularization}.m`. The `data_driven_mpc.m` tracer-bullet runs `ssest(z,2) → ss(idsys) → mpc(P,10,3)` end-to-end. **Open carve-downs** (consolidated index in roadmap §8b): nonlinear black-box `nlarx`/`nlhw` + mapping objects (idSigmoidNetwork/idWaveletNetwork/idTreePartition), `particleFilter`, recursive-PEM (`recursiveARMAX`/`recursiveOE`/`recursiveBJ`), estimation `Report` struct, multi-return forms, `showConfidence` uncertainty bands, other `*Options` carriers (`armaxOptions`/`oeOptions`/`bjOptions`/`ssestOptions`), Fisher-information `getcov` for PEM-fit models, `merge` multi-experiment, ARIMA/seasonal, MIMO. **Carve-outs** (per the roadmap): System Identification App + Time Series Modeler app (Chapters 23–24), Simulink blocks (Ch.22), Neural State-Space + LSTM/cascade-correlation/`narxnet` (Deep Learning dependency), ML-NLARX (Stats & ML dependency), Reduced Order Modeling chapter, C-MEX grey-box, Diagnostics & Prognostics (Predictive Maintenance overlap). |
| Econometrics (`arima`, `garch` / `egarch` / `gjr`, `varm`, `ssm` / `dssm`, `bayeslm`, `dtmc`; `autocorr` / `parcorr` / `crosscorr`, `adftest` / `pptest` / `kpsstest` / `lmctest` / `vratiotest`, `lbqtest` / `archtest` / `aicbic` / `lratiotest` / `waldtest` / `lmtest` / `hac` / `fgls`, `price2ret` / `ret2price` / `hpfilter`, `egcitest` / `jcitest` / `jcontest`) | ✅ | **All 6 tiers shipped** via [`econometrics_toolbox_roadmap.md`](econometrics_toolbox_roadmap.md). Backed by [`runtime/toolbox/econ/runtime_econ.cpp`](../runtime/toolbox/econ/runtime_econ.cpp) (~2.3 kLOC) + [`runtime/toolbox/econ/econ_classdefs.m`](../runtime/toolbox/econ/econ_classdefs.m) — composed entirely over the shipped Optim (`lsqnonlin`/`fminunc` lineage, here a self-contained Nelder-Mead), Stats (test CDFs via local incomplete-gamma/beta), LAPACK, and System-Identification base (no statsmodels/R). **T1** data prep + the full diagnostic/unit-root/cointegration test surface (function-form via the LowerTensorOps Spec table; self-contained χ²/normal CDF + OLS + Jacobi-eig helpers). **T2** `arima(p,D,q)` — `estimate` (Hannan-Rissanen two-stage), `infer`, `forecast` (recursive MMSE + integration), `simulate`. **T3** `garch`/`egarch`/`gjr` (one `ModelKind`-discriminated kernel set) — Gaussian-MLE `estimate` over the conditional-variance recursion via Nelder-Mead, `infer`/`forecast`/`simulate`. **T4** `varm` (VAR) — equation-wise-OLS `estimate`, `forecast`, `simulate`, `irf` (orthogonalized via Cholesky); cointegration `egcitest` (Engle-Granger residual-ADF) + `jcitest`/`jcontest` (Johansen trace, symmetric-reduced eigenproblem). **T5** `ssm`/`dssm` — `estimate` (Kalman-filter ML over the free B/D loadings), `filter` (Kalman), `smooth` (RTS), `forecast` (mutate-in-place receiver to sidestep the matrix-param-ctor frontend limit). **T6** `bayeslm` (diffuse-prior posterior mean = OLS) `estimate`/`forecast`, `dtmc` Markov chains `asymptotics`/`simulate`. **Two general compiler fixes**: LowerTensorOps boxes a scalar assigned to a matrix classdef property (`matlab_mat_from_scalar`), and RefineSlotTypes refines a none slot whose stores are a consistent ranked tensor. **7 gating tests green** (`test/Run/econ_*.m`). Six headlines `examples/econ/{stationarity_workflow,arima_cpi_forecast,garch_volatility,var_macro,ssm_kalman,bayeslm_regression}.m`; `arima_cpi_forecast` runs the full Box-Jenkins arc (test → difference → identify → estimate → diagnose → forecast). **Carve-downs**: `regARIMA` (matrix-param ctor via the estimate fresh-ctor path), Markov-switching `msVAR` (HMM Baum-Welch EM), Time-Series-Regression example series. **Carve-outs**: Econometric Modeler GUI, DSGE, HMC/NUTS samplers, Bayesian/non-Gaussian/particle state-space, structural-VAR identification. |
| Global Optimization (`ga`, `particleswarm`, `simulannealbnd`, `MultiStart`, `GlobalSearch`, `createOptimProblem`, `patternsearch`, `surrogateopt`, `gamultiobj`, `paretosearch`, `optimoptions`) | ✅ | **All 6 tiers shipped** via [`global_optim_toolbox_roadmap.md`](global_optim_toolbox_roadmap.md). Backed by [`runtime/toolbox/gads/runtime_gads.cpp`](../runtime/toolbox/gads/runtime_gads.cpp) + [`gads_classdefs.m`](../runtime/toolbox/gads/gads_classdefs.m) — an *amplifier* of the shipped Optimization Toolbox: every solver runs over the shared seeded PRNG (`rng`-reproducible) and reuses the shipped `fmincon` / `mldivide` (no external dependency). **T1**: the three derivative-free global solvers — `ga` (real-coded GA), `particleswarm` (Clerc-Kennedy PSO), `simulannealbnd` (geometric-cooling SA), each with a `fmincon` hybrid-polish step. **T2**: the multi-start meta-solvers — `createOptimProblem('fmincon',…)` (name-value scan → thread-local problem context), `MultiStart` (k fmincon restarts), `GlobalSearch` (scatter-sample + fmincon). **T3**: `patternsearch` — deterministic GPS direct search (complete 2N-basis poll, no PRNG, no hybrid), robust on nonsmooth/discontinuous objectives. **T4**: `surrogateopt` — cubic-RBF surrogate (`mldivide` coeff solve) + merit-weighted adaptive sampling, sample-efficient for expensive objectives. **T5**: multiobjective — `gamultiobj` (NSGA-II: non-dominated sort + crowding) + `paretosearch` (non-dominated archive + GPS poll), returning the Pareto set (vector-out objective). **T6** (focused carve-down sweep): `optimoptions('ga', …)` options carrier (PopulationSize / MaxGenerations) + **integer-constrained `ga`** (`IntCon`) — `ga(fun,nvars,…,opts)` routes to `matlab_gads_ga_opts`, which rounds the `IntCon` variables to the nearest feasible integer each generation and auto-skips the `fmincon` hybrid for integer problems (a shared `gads_ga_core` keeps the Tier-1 path byte-identical). Objective is the shipped `double(@fun)(x)` / vector-out handle ABI; the MATLAB call forms are remapped in the Lowering dispatch. **9 gating tests green; full regression clean.** Headlines `examples/globaloptim/rastrigin_ga.m` (`fminunc` trapped at f=16.91 → `ga`/`particleswarm` recover the global f=0 on Rastrigin) + `sixhump_multistart.m` (single solve trapped at −0.2155 → `MultiStart`/`GlobalSearch` find the global −1.0316 on the camelback) + `nonsmooth_patternsearch.m` (`fminunc` stalls at f=125 on a discontinuous staircase → `patternsearch` finds the global f=0) + `branin_surrogate.m` (`surrogateopt` finds Branin's global f=0.3979) + `pareto_front.m` (`gamultiobj`/`paretosearch` recover the full Pareto trade-off curve) + `gear_train_intga.m` (mixed-integer Sandgren gear-train: `ga` with `IntCon=[1 2 3 4]` picks integer tooth counts giving a ratio error ≈ 2.3e-11). **Tier-6 follow-ons (🔵)**: `optimoptions` for the other solvers + `HybridFcn`/`FunctionTolerance` knobs, `exitflag`/`output` multi-return, `IntCon` for `surrogateopt`, nonlinear-constraint handles, problem-based `solve` routing, `patternsearch` `PollMethod`/NUPS, parallel, the dipole cross-toolbox demo. **Carve-outs**: Optimize Live Editor Task + apps, Simulink optimization, cluster-parallel, custom-data-type genomes, GPU. |
| Statistics and Machine Learning (descriptive + distributions + `makedist`/`fitdist`; `ttest*`/`vartest2`/`kstest`/`ranksum`/`anova1`; `regress`/`fitlm`/`fitglm`/`ridge`; `pca`/`kmeans`/`pdist2`/`silhouette`; `fitcknn`/`fitcnb`/`fitcdiscr`/`fitctree`/`fitcsvm`/`fitcecoc`/`fitcensemble`/`TreeBagger` + `predict`/`confusionmat`; `bayesopt`; `hmm*`) | ✅ | **All 6 tier cores shipped** via [`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md) — the biggest single-toolbox roadmap; **the `iris_classify` headline is CLOSED**. Backed by [`runtime/toolbox/stats/runtime_stats.cpp`](../runtime/toolbox/stats/runtime_stats.cpp) (~1.6 kLOC) + [`stats_classdefs.m`](../runtime/toolbox/stats/stats_classdefs.m); no external dependency. **T1**: descriptive battery, `cov`/`corr`/`corrcoef`, Normal/Exponential/Uniform pdf/cdf/inverse (normal CDF via libc `erf`, inverse normal via Acklam), RNGs `normrnd`/`unifrnd`/`exprnd`, distribution objects `makedist`/`fitdist` → `ProbDistUnivParam` (`pdf`/`cdf`/`icdf`/`random`). **T2**: hypothesis tests + one-way ANOVA with the MATLAB `[h,p,ci,stats]` multi-output; p-values on hand-coded Student-t / F / χ² CDFs (regularized incomplete gamma + beta). **T3**: regression — `regress`, `fitlm` (`LinearModel` coefficient table + R²/RMSE + `predict`), `fitglm` (logistic IRLS), `ridge`. **T4**: PCA (Jacobi eig of covariance, `[coeff,score,latent,~,explained]`), `kmeans` (Lloyd + k-means++), `pdist2`/`pdist`/`squareform`, `silhouette`. **T5**: classification — `fitcknn`, `fitcnb` (Gaussian), `fitcdiscr` (LDA), `fitctree` (hand-coded CART), `fitcsvm` (linear), `fitcecoc` (one-vs-one multiclass) + `predict` (runtime-dispatched on the model class) + `confusionmat`. **T6**: ensembles — `fitcensemble` (bagged CART) + `TreeBagger` (random forest = bootstrap + √p feature subset), `ClassificationModel` ModelType 7 (trees concatenated + `Offsets`, majority vote); `bayesopt(fun,lb,ub)` (GP surrogate + expected-improvement, functional form over the objective-handle ABI); Markov models `hmmgenerate`/`hmmviterbi`/`hmmdecode`/`hmmtrain` (Viterbi + scaled forward-backward + Baum-Welch). **General compiler fixes landed here**: (1) the shared `pde_table` matcher scans *all* same-name entries → multi-arity overloads (`normcdf(x)` vs `normcdf(x,mu,sigma)`); (2) bracket concat of matrix/column-vector operands (`[x1 x2]`, `[a; b]`) lowers via `matlab_horzcat`/`matlab_vertcat` (essential for ML design matrices). **12 gating tests green; full regression 456/456.** Headline `examples/stats_ml/iris_classify.m` (descriptive → `pca` → `kmeans` → `fitcecoc` → `confusionmat`, recovering real Fisher-iris behaviour: setosa clean, versicolor/virginica overlap, ~95% accuracy) + 8 more (`fit_normal`/`exploratory_analysis`/`distribution_fitting`/`hypothesis_testing`/`linear_regression`/`glm_logistic`/`hmm_markov`/`ensemble_classify`). **Follow-ons (🔵)**: boosting (`AdaBoost`/`LogitBoost`; bagging shipped), wider distributions (binomial/Poisson/gamma via the now-available incomplete gamma/beta), `grpstats`/`crosstab`, `anovan`/`multcompare`, Wilkinson-formula `fitlm`, `fitnlm`/`lasso`, RBF-kernel SVM, `crossval`/`cvpartition`/`perfcurve`, `gmdistribution`/`linkage`, `bayesopt` `OptimizeHyperparameters` integration. **Carve-outs**: Classification/Regression Learner + Distribution Fitter apps, DL-backed models, Simulink ML blocks, MATLAB-Coder API, tall/GPU, incremental learning, fairness/interpretability, survival analysis. |
| Image Processing (I/O + types + arithmetic; `fspecial`/`imgaussfilt`/`medfilt2`/`histeq`/`imnoise`; `imresize`/`imrotate`/`imwarp` + `affine2d`/`fitgeotform2d`; `graythresh`/`imbinarize`/`strel`/`imopen`/`imfill`/`edge`; `bwlabel`/`regionprops`/`bwareaopen`/`label2rgb`/`imsegkmeans`; `dct2`/`radon`/`hough`/`psnr`/`ssim`/`poly2mask`/`rgb2hsv`/`rgb2lab`/`im2col`/`deconvwnr`) | 🟡 | **ALL 6 tier cores shipped — the `rice_grains` headline is closed** via [`image_toolbox_roadmap.md`](image_toolbox_roadmap.md). Backed by [`runtime/toolbox/images/runtime_images.cpp`](../runtime/toolbox/images/runtime_images.cpp); no external dependency (no OpenCV/libpng/stb/libjpeg). Images are double matrices in [0,255] (uint8-class) / [0,1] (double-class) — grayscale M×N or slice-major M×N×3 truecolor — reusing the shipped double kernel (`conv2`/`imfilter`/`padarray`/`fft2`) and the uint8 saturation convention. **T1**: `imread` for **PGM/PPM/BMP + hand-coded PNG (full zlib inflate: stored/fixed/dynamic Huffman + LZ77, all 5 unfilter types, grayscale/RGB/palette/+alpha, 1/2/4/8/16-bit) + baseline JPEG (JFIF/SOF0: Huffman + dequant + 8×8 IDCT + YCbCr→RGB + 4:4:4/4:2:2/4:2:0 upsample)** — PNG decode is exact vs PIL, JPEG within lossy tolerance; `imwrite` for PGM/PPM/BMP + **lossless PNG (store-mode deflate, valid output PIL/libpng read)**; a `checkerboard` synthetic; type conversions (`im2double`/`im2single`/`im2uint8`/`rgb2gray`/`im2gray`/`mat2gray`); saturating image arithmetic (`imadd`/`imsubtract`/`immultiply`/`imdivide`/`imabsdiff`/`imcomplement`/`imlincomb`); intensity stats (`imhist`/`imadjust` auto+ranges+gamma/`stretchlim`/`mean2`/`std2`). **T2**: `fspecial` (gaussian/average/laplacian/log/sobel/prewitt/disk/motion); `imfilter` (shipped) + `imgaussfilt`/`imboxfilt`/`medfilt2`/`ordfilt2`/`stdfilt`/`rangefilt`; enhancement `histeq`/`adapthisteq` (tiled CLAHE)/`imsharpen`/`imhistmatch`/`imnoise` (gaussian/salt&pepper/speckle). **T3**: geometric transforms — `imresize` (nearest/bilinear/bicubic-conv), `imrotate` (crop/loose), `imcrop`, `imtranslate`, `imwarp` (affine + projective, auto bbox, bilinear inverse-resample) with `affine2d`/`projective2d`/`imref2d` classdefs (3×3 forward matrix `T` + `Kind`), and `fitgeotform2d` (LS affine/similarity from control points → class-pinned `affine2d`); all resamplers handle grayscale + per-channel RGB. **T4**: binarization + morphology — `graythresh` (Otsu, plateau-averaged)/`otsuthresh`/`imbinarize`/`im2bw`, `strel` (disk/square/rectangle/line mask), grayscale+binary `imerode`/`imdilate`/`imopen`/`imclose`/`imtophat`/`imbothat`, `imfill` ('holes'), `edge` (Sobel + Canny NMS/hysteresis), `bwareaopen`. **T5**: segmentation + region analysis — `bwlabel` (8-conn), `regionprops(L,prop)` returning the property as a matrix (`Area`/`Centroid`/`BoundingBox`/`Perimeter`/`EquivDiameter`/`Extent`/`MajorAxisLength`/`MinorAxisLength`/`Eccentricity`/`Orientation`), `bweuler`, `label2rgb`, `imsegkmeans` (reuses Stats `kmeans`). **Two general fixes landed here**: (1) the shared `pde_table` matcher materialises single-quoted string literals (`matlab.const_char`) into `matlab_string*`, so any builtin can take a literal filename/option string (`imread('f.pgm')`, `fspecial('gaussian',…)`, `imresize(I,0.5,'bilinear')`); (2) `size`/`numel`/`ndims`/`length` are now runtime **mat3-aware** (read the 3-D magic tag), so any RGB result answers `size(X,3)`/`ndims` correctly. **T6**: transforms `dct2`/`idct2` (separable DCT, exact round-trip)/`radon`/`hough`+`houghpeaks`; quality `immse`/`psnr`/`ssim` (8×8 windowed); ROI `poly2mask`/`roifilt2`; colour `rgb2hsv`/`hsv2rgb`/`rgb2ycbcr`/`ycbcr2rgb`/`rgb2lab`/`lab2rgb` (whole-image pipeline via `img_color_apply`); block `im2col`/`col2im`; deblur `deconvwnr` (Wiener via 2-D complex FFT)/`edgetaper`. **Third general fix — 3-D array indexing** (frontend, benefits all toolboxes): `A(:,:,k)` plane read/store, `A(i,j,k)` element read/store, and `cat(3,A,B,C)` / `cat(1|2,…)` now lower for `matlab_mat3`, so colour images split/process/merge per channel (`R = rgb(:,:,1)`) — not just pipeline-style; bindings from `cat(3,…)`/`zeros(m,n,3)`/colour conversions/`label2rgb` are tracked 3-D. **Fourth general capability — real image-format codecs** (`runtime_images.cpp`, no external lib): hand-coded PNG decode (puff-style inflate) + encode + baseline JPEG decode, so `imread('photo.png')` / `imread('photo.jpg')` work on real files. **10 gating tests green (incl. `array3d_indexing`, `image_png_roundtrip`); full regression 465/465.** Headlines `examples/images/basic_image.m` + `filtering.m` + `geometric.m` + **`rice_grains.m`** (background subtraction → Otsu → `bwlabel`/`regionprops`: counts 40 grains, mean area 45px — *the toolbox headline, closed*) + `transforms.m` (DCT round-trip 6e-12, colour round-trips, Hough line, Wiener restore 11.3→12.2 dB PSNR) + `channel_split.m` (`cat(3,…)` → `rgb(:,:,k)` split → boost → merge). **Follow-ons**: TIFF/GIF decode (PNG/JPEG/PGM/PPM/BMP ship), progressive JPEG, JPEG encode, `imfinfo`, RGB per-channel `imfilter`, `wiener2`, `imwarp` `'OutputView'`, `watershed`/`bwdist`, `iradon`/`houghlines`, `blockproc`/`nlfilter` (handle-per-block ABI), `regionprops` struct-array/`'table'`, `bwboundaries`, `superpixels`, `activecontour`, `deconvlucy`. **Carve-outs**: all apps + modular GUI tools, Deep Learning chapter, blocked/bigimage + MapReduce, GPU + Code Generation, intensity-based `imregister`, DICOM/HDR, Hyperspectral + Optical-System add-ons, Computer-Vision overlap. |
| Deep Learning + Deep Learning HDL (`dlarray` / `dlfeval` / `dlgradient` / `extractdata`; `dlnetwork` carrier + `trainnet` / `trainingOptions`; `adamupdate` / `sgdmupdate` / `rmspropupdate` functional solvers; `relu` / `sigmoid` / `tanh` / `softmax` / `gelu` / `swish` / `leakyrelu` / `elu` / `softplus`; `crossentropy` / `mse` / `l1loss` / `huber` / `l2loss`; `conv2` / `conv1d` (forward + im2col backward); `batchnorm` / `layernorm` / `groupnorm` / `instancenorm` / `rmsnorm`; `lstm` / `gru` / `bilstm` / `lstmp` + functional attention/MHA + `embed`; `dropout`; `imageDatastore` / `countEachLabel` / `splitEachLabel` / `augmentedImageDatastore`; `gradCAM` / `occlusionSensitivity` / `imageLIME` / `tsne`; `accuracy` / `precision` / `recall` / `fScore` / `rocmetrics` / `aucroc`; `dlquantize` / `dlqscale` / `dlqcalibrate` / `dlqclip`; `bayesopt`-driven HP search; `runExperiment` sweep harness; magnitude-based pruning; ONNX import/export (~56 ops); cocotb-verified `dlhdl_*` fi-typed SystemVerilog datapaths) | ✅ | **All 6 tiers + DL HDL H1–H5 shipped** via [`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md) — the catalogue's largest toolbox.  Backed by [`runtime/toolbox/dlnet/runtime_dlnet.cpp`](../runtime/toolbox/dlnet/runtime_dlnet.cpp) (autodiff tape, layer forward/backward, solvers, recurrent kernels, quantizer, datastores) + [`runtime/toolbox/dlnet/runtime_onnx.cpp`](../runtime/toolbox/dlnet/runtime_onnx.cpp) (hand-rolled Protobuf reader/writer, ~56 op handlers) + [`runtime/toolbox/dlnet/dlnet_classdefs.m`](../runtime/toolbox/dlnet/dlnet_classdefs.m); no external dependency (no PyTorch / TensorFlow / cuDNN / ONNX runtime).  **T1 inference**: `dlarray` value type carrying data + tape-node id, `dlnetwork` carrier (`addFC`/`addRelu`/`addSigmoid`/`addTanh`/`addSoftmax` + `predict`), full activation + loss surface (operator overloading: `W*X + b`, `+`, `.*`, `./`, `sqrt`, reductions, all with numpy-style broadcasting on row/col/scalar shapes), `imageDatastore`/`countEachLabel`/`splitEachLabel`.  **T2 autodiff (keystone)**: reverse-mode AD tape (Wengert list); `dlgradient` verified against central finite differences to **1.24e-10**; analytic pullbacks for `+`/`-`/`.*`/`*` (matmul) / `./` / `sum` / `mean(X, dim)` / `reshape` / `permute` / `exp` / `log` / `sqrt` / `tanh` / `max` / `relu` / `gelu` / `swish` / `softmax` / `crossentropy` / `mse` / `transpose` / `embed` (gather + scatter-add) / `lstm` / `gru` (one tape node each, BPTT inside); LayerNorm / GroupNorm / InstanceNorm / RMSNorm / BatchNorm (EMA-tracked at train, identity-with-stored-stats at infer); single-head Transformer encoder block (embed + scaled-dot-product attention + residual + LayerNorm + GELU FFN); MHA training (concatenated heads); dropout (Bernoulli mask outside tape × dlarray); VAE w/ reparameterization.  **T3 training**: `trainnet`/`trainingOptions` driver (Adam over `Learnables` table) **and** the custom training loop (`dlfeval` + `dlgradient` + `adamupdate` / `sgdmupdate` / `rmspropupdate` functional solvers); `augmentedImageDatastore` w/ random rotation/scale/translation + `ColorPreprocessing='gray2rgb'` + per-batch resize; headless training monitor; single-device GPU training dispatch (`matlab_gpu_gemm` routing on forward/backward/solver step).  **T4 sequence/recurrent/attention** (functional surface): `lstm`/`gru`/`bilstm`/`lstmp` (each one custom tape node + BPTT), functional scaled-dot-product attention via shipped `matmul` + `softmax` + `transpose`, `embed` (gather-forward + scatter-add-backward, repeated indices correctly accumulate), 1-D conv sequence path (`conv1d`).  **T5 architectures + transfer learning** (functional patterns): residual blocks via overloaded `plus` (skip-add records on tape, gradient flows both branches); transfer learning (frozen encoder as plain matrices + autodiff head, 96% accuracy); autoencoder; LSGAN-style GAN (`dl_gan.m`: 1-D generator → N(2, 0.5²)); Siamese twin embedding (shared weights, within-cluster ≈ 0 vs between-cluster ≈ 0.002); Neural ODE (T5.7, stepper-matrix form on the dense lane).  **T6 tuning / viz / metrics / quantization**: Grad-CAM-style attribution on MLPs (`dl_gradcam.m`) **and** image-domain `gradCAM` / `occlusionSensitivity` / `imageLIME` on the 4-D `SSCB` tensor, `tsne`, classification metrics (`accuracy`/`precision`/`recall`/`fScore`/`rocmetrics`/`aucroc` over the Stats `confusionmat` kernel), `dlqcalibrate` + `dlqclip` activation-quantization (companion to H1's weight quantizer), Bayesian HP search wrapping shipped `bayesopt`, gradient-based l∞ robustness check (dual-norm), **magnitude-based pruning** (rank `Learnables` by `|w|`, zero bottom-k%, re-fine-tune over the custom loop), **programmatic experiment sweep harness** (`runExperiment(@trialFn, gridSpec, opts)` over Cartesian or Bayesian sweep, emits per-trial result table — no GUI).  **DL HDL H1–H5**: H1 `dlquantize`/`dlqscale` symmetric per-tensor INT8 weight quantization (`dl_quantize_check.m`: T3 MLP — both double and INT8 at 100% accuracy, max logit drift ≈ 0.1); H2 fi-typed SystemVerilog emission (`dlhdl_quant_mlp.m` — Q16.8 2-2-1 MLP → ~15 lines synthesizable SV via shipped `EmitSV` lane, Verilator + Yosys clean); H3 cocotb bit-accuracy (`% hdl: precise_fi` opt-in pragma: Sema-mono on the HW lane so fi-grown widths thread through `matlab.matmul`/`matlab.add` result types, EmitSV emits 64-bit intermediates + FL-rescaled bias — 41/41 vectors bit-accurate vs Python reference; closes [issue #75](https://github.com/leonardoaraujosantos/matlab_llvm/issues/75)); H4 LSTM-on-FPGA (`dlhdl_rnn_cell` / `dlhdl_lstm_cell` / `dlhdl_lstm_step` — recurrent fi kernel, `persistent` h_state/c_state lowers to `always_comb` + `always_ff` shift-register; cocotb 44/44); **H5 minimal ONNX inference-graph importer**: parse ONNX Protobuf (Conv2D / Linear / ReLU / MaxPool / BatchNorm / Add / Concat / Softmax / Sigmoid / Tanh / Reshape / Transpose + 40 more — ~56 ops) → T1 layer DAG, initializers → `Learnables`; round-trip writer too (`dl_onnx_roundtrip.m` / `dl_onnx_ops_coverage.m`).  **Carve-down sweep landed alongside (A–H)**: A multi-head attention training; B 1-D conv (`dl_conv1d_train.m`); C `dlnetwork` carrier + `trainnet` driver; D emit-c 4-D `SSCB` lane (`builtin.unrealized_conversion_cast` handler); E image-domain attribution (`dl_gradcam_image` / `dl_occlusion_image` / `dl_lime_image`); F object-array carrier (`matlab_dlnet_oa_new`/`oa_append`); G handle-classdef `void*` typedef + value-receiver method bodies + LiveGlobals scan in `EmitC` so every dlarray-using test lifts cleanly through emit-c bit-exact to the LLVM lane; H classdef-array literal `[obj1; obj2; obj3]` (concat fold detects all-classdef-instance leaves and routes to obj-array carrier).  **39 `dl_*.m` gating tests in `test/Run/`** + **44 examples** in `examples/dlnet/`; full regression green; all `EmitSV` + cocotb DL-HDL fixtures green.  **Carve-outs** (per the roadmap §10): Deep Network Designer + Experiment Manager apps (GUIs; programmatic engines ship), all Simulink Deep-Learning blocks (the `mflowLink` lane is the answer), full external-framework import/export (`importNetworkFromPyTorch` / `importNetworkFromTensorFlow` / `exportNetworkToTensorFlow`; the *minimal* inference-graph ONNX importer ships as H5), real named pretrained weight blobs (AlexNet / ResNet / YOLO / BERT — 100s of MB), multi-GPU / cluster / cloud training, big-data datastores backed by disk, Reinforcement Learning / Computer Vision / Audio / Text Analytics toolbox deps, **DL HDL silicon** (bitstream + board deployment + LIBIIO + vendor synthesis — simulation surface only), GPU/CPU `'lib'` quantizer targets (TensorRT/MKL-DNN), legacy "Neural Network Toolbox" shallow-net surface (`patternnet` / `feedforwardnet` / `fitnet`).  Two semantic-divergence holdouts (`dl_mha_train` minor FP rounding diff vs LLVM lane; `dl_cnn_classifier` flatlines under emit-c at uniform softmax) skipped on emit-c only. |
| Reinforcement Learning (`rlNumericSpec` / `rlFiniteSetSpec` + `getObservationInfo` / `getActionInfo`; `rlPredefinedEnv` + `rlMDPEnv`; `rlTable` / `rlQValueFunction`; `rlQAgent` / `rlSARSAAgent` / `rlDQNAgent` / `rlPGAgent` / `rlDDPGAgent` / `rlTD3Agent` / `rlPPOAgent` / `rlSACAgent` / `rlGRPOAgent` / `rlTRPOAgent`; `rl*AgentOptions` / `rlOptimizerOptions` / `rlTrainingOptions` / `rlSimulationOptions`; `train` / `sim` / `getCritic` / `getLearnableParameters` / `getAction` / `getMaxQValue` / `getGreedyPolicy` → `rlMaxQPolicy`) | ✅ | **All 6 tiers shipped + a beyond-list GRPO agent** (PR #83) via [`reinforcement_learning_toolbox_roadmap.md`](reinforcement_learning_toolbox_roadmap.md). Backed by [`runtime/toolbox/rl/runtime_rl.cpp`](../runtime/toolbox/rl/runtime_rl.cpp) + [`runtime/toolbox/rl/rl_classdefs.m`](../runtime/toolbox/rl/rl_classdefs.m); no external dependency (no Gym / Stable-Baselines / RLlib). **Keystone**: the deep agents add *no* new numerical kernel — the RL runtime builds each actor/critic forward pass as `dlarray` shells (`matlab_obj_new`; a dlarray = matlab_obj with `Data` mat + `Id` tape-node) and drives the **shipped Deep Learning autodiff tape** (`matlab_dlnet_mtimes`/`plus`/`relu`/`tanh`/`exp`/`softplus`/`softmax`/`log`/`sum`/`mse` + `matlab_dlnet_grad`) directly from C++; only the Adam moment-update + replay/episode loop are RL-side. **T1 (tabular, autodiff-free)**: grid-world / deterministic-MDP envs, table Q critic, `rlQAgent` / `rlSARSAAgent` ε-greedy TD loop over the env tensors (`gridworld_qlearning.m` → 11.0; `mdp_qlearning.m` → 13.0). **T2 (control envs + policy use)**: `CartPole-Discrete` (Barto) + `Pendulum-Continuous` (swing-up) + greedy `sim`; `getAction` / `getMaxQValue` / `getGreedyPolicy`. **T3 (DQN)**: auto-MLP critic + replay + target net + ε-greedy + MSE-TD → `dlgradient` → Adam (`cartpole_dqn.m` 269 steps). **T4 (REINFORCE)**: `rlPGAgent` softmax actor, `−Σ logπ·Ĝ` (`cartpole_reinforce.m`). **T5 (DDPG)**: deterministic actor + Q-critic + target nets + OU noise, DPG step through the critic via tape `vertcat` (`pendulum_ddpg.m` ~−391; reward-scaling for convergence). **T6**: **TD3** (twin critics + delayed policy + target smoothing, `pendulum_td3.m` ~−371), **PPO** (on-policy, GAE(λ) + value baseline + clipped surrogate, `cartpole_ppo.m`), **SAC** (squashed-Gaussian reparam actor + twin critics + entropy, tanh-squash log-prob on the tape, `pendulum_sac.m` ~−370, fixed-coefficient variant), **TRPO** (on-policy natural-gradient: conjugate-gradient `x=F⁻¹g` + KL trust region + backtracking line search, with Fisher-vector products from the reverse-mode KL gradient, `cartpole_trpo.m` ~203 steps). **Beyond the MathWorks list — GRPO** (`rlGRPOAgent`, DeepSeek): critic-free group-relative advantage `(rᵢ−μ)/σ` + clipped surrogate + KL-to-reference, on a **Countdown arithmetic verifier env** (`rlPredefinedEnv("Countdown-Discrete")`); `countdown_grpo.m` solves 7/8 puzzles (untrained 0/8). **10 gating tests** (`rl_gridworld`/`rl_mdp`/`rl_dqn`/`rl_reinforce`/`rl_getaction`/`rl_ddpg`/`rl_td3`/`rl_ppo`/`rl_sac`/`rl_grpo`/`rl_trpo`) + 11 examples; the chaotic agents assert platform-stable threshold verdicts (libm-divergent exact values). The DDPG/training-scale memory blocker ([#82](https://github.com/leonardoaraujosantos/matlab_llvm/issues/82)) and the `rlFunctionEnv` frontend gaps ([#78](https://github.com/leonardoaraujosantos/matlab_llvm/issues/78)/[#79](https://github.com/leonardoaraujosantos/matlab_llvm/issues/79)/[#80](https://github.com/leonardoaraujosantos/matlab_llvm/issues/80)/[#81](https://github.com/leonardoaraujosantos/matlab_llvm/issues/81)) are all resolved; the fix for a general dlnet gemm-transpose-scratch leak (~20 GB → ~810 MB for DDPG) lifts the training-scale ceiling for *all* DL training. **Carve-outs**: SAC automatic-temperature tuning, `rlFunctionEnv` custom-env classdef, RL Designer app, Simulink/Simscape envs, parallel/GPU, multi-agent, recurrent actors, offline/evolutionary training, training-monitor/MBPO/deploy infra. |
| 2-D / 3-D FEM (`createpde`, `femodel`, `multicuboid`, mesh generation, `solvepde`, `solve`, `pdeplot3D`, `VonMisesStress`, …) | ✅ | **11 arcs shipped** via [`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md): full Tier-1 → Tier-4 surface.  Sparse CSR infra + ILU(0)+GMRES + MINRES + PCG.  Lanczos shift-invert with mode shapes.  Modal superposition + Rayleigh damping.  T10 quadratic tets with stress recovery.  STL/GLB import (surface + volumetric via voxelize).  pdeplot / pdegplot / pdemesh / pdeplot3D.  Geometry primitives (`multicuboid` / `multicylinder` / `multisphere`) + Bey red refinement (`refineMeshBey`).  `femodel` classdef façade + legacy aliases (`solvepde`, `specifyCoefficients`, `applyBoundaryCondition`).  AnalysisType dispatch: structuralStatic / structuralTransient / structuralModal / structuralFrequency (damped via 2N×2N real-bordered + complex Krylov) / structuralTransientModal / structuralStaticNL / structuralStaticTL / thermalSteadyState (with Picard nonconstant `k(T)`) / thermalTransient / electrostatic / magnetostatic / dcConduction / harmonicElectromagnetic.  Thermal-stress coupling (`cellLoad(Temperature=…)`).  Full Craig-Bampton ROM (`pde_reduce_craig_bampton`) + modal-truncation ROM (`reduce`/`reconstructSolution`).  N-component coupled PDEs (`pde_solve_multi_n`).  **33 PDE end-to-end tests green.**  Remaining (mostly polish): full Green-Lagrange B_NL + geometric K_σ for true large-rotation elasticity, hanging-node red-green propagation, real Delaunay/TetGen mesher (today's volumetric meshing uses Kuhn 6-tet), 3-D Gouraud shading (per-triangle flat today).  PDE Modeler 2-D app + STEP import + PINN/GNN/FNO + Battery P2D explicitly carved out. |
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
| Cell: `c{i} = matrix / string`, `cell(m,n)` preallocation | ✅ shipped (#292) | A dynamic element store records the element kind (`CellMatElems` / `CellStrElems`) so a later `c{k}` read picks `matlab_cell_get_mat` / `_str` instead of defaulting to `get_f64` (which returned 0 for a matrix); a char RHS is wrapped to a `matlab_string*` and stored via `matlab_cell_set_str`. `cell(n)` / `cell(m,n)` lower to `matlab_cell_new[_2d]`. Subscripting a brace-read result (`c{i}(k)`) already worked. Gating test: `test/Run/regress_cell_element_assign.m`. |
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

### Communications Toolbox — Tier-7 modern channel codes (function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §5.4 — flipped from "🔴 stretch carve-out" to "✅ shipped" via function-form implementations. The classdef `comm.LDPCEncoder` / `LDPCDecoder` / `TurboEncoder` / `TurboDecoder` / `PolarEncoder` / `PolarDecoder` System Objects stay gated on the SO lowering fix. Runtime extends `runtime/runtime_comm.cpp`. Demos under `examples/comm/` (`tier7_smoke.m`, `modern_codes_ber.m`).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| §5.4.A Polar | `polarEncode(u, N)`, `polarSCdecode(llr, frozen_mask, N)` | ✅ shipped | Arikan polar transform via the recursive butterfly (no bit-reversal); SC decoder via recursive f / g node operations on min-sum LLRs. Caller supplies a frozen-mask vector (0 = info, 1 = frozen); placement of info bits is caller-controlled — production polar codes use a reliability-sequence lookup (3GPP NR sequence is a follow-on table). Verified: zero-noise round-trip exact at N = 4, 8, 16, 32, 64, 128. At 5 dB SNR, (128, 64) recovers 64 info bits with 0 errors. |
| §5.4.B LDPC | `ldpcEncode(msg, P)`, `ldpcDecodeMS(llr, H, max_iter)` | ✅ shipped (function-form) | Systematic encoder from the parity portion `P` (k × (n−k)) of a systematic generator (G = [I_k | P], H = [P^T | I_{n−k}]). Decoder is flooding-schedule min-sum belief propagation on the Tanner graph of H. Caller supplies P / H — generic generators for irregular / 5G NR base matrices are a separate lookup-table slice. Verified: hand-rolled (6, 3) recovers a corrupted codeword in 20 iterations at SNR 5 dB. |
| §5.4.C Turbo | `turboEncode(msg, trellis, perm)`, `turboDecode(llr_sys, llr_p1, llr_p2, trellis, perm, max_iter)` | ✅ shipped | Parallel-concatenated convolutional codes (PCCC): two systematic-RSC encoder passes through an interleaver, emitting `[systematic; parity1; parity2]` of length 3 × k. Decoder is the canonical iterative max-log-MAP / BCJR with extrinsic LLR exchange across the interleaver. Verified: (7, 5)₈ K=3 RSC with a shift-by-11 permutation at k = 64 — 0 errors from SNR = 2 dB onwards. |
| Closure tests | `examples/comm/tier7_smoke.m`, `examples/comm/modern_codes_ber.m` | ✅ shipped | At SNR = 5 dB on a 64-bit message: uncoded BPSK 2 errors / Polar (128, 64) 0 / Turbo PCCC 0 / LDPC (6, 3) 0 (across 21 blocks). Compile, execute, and DWARF-debug lanes pass for both; REPL JIT runs every Tier-7 primitive correctly before tripping on the same `make_handle`-on-colon-slice REPL limitation documented across the rest of the comm surface. |

### Communications Toolbox — Tier-6 spreading + source coding (function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §8. Spreading sequences (PN / Gold / Walsh-Hadamard) and source coding (uniform quantiser, μ-law / A-law companding, DPCM, Lloyd-Max codebook optimisation). System-Object variants (`comm.PNSequence`, `comm.GoldSequence`, `comm.KasamiSequence`) stay gated on the SO lowering fix. Hybrid ARQ (`comm.HybridARQ`) and ray-tracing-driven propagation (`propagationModel('raytracing')`) are explicit roadmap carve-outs. End-to-end demos under `examples/comm/` (`tier6_smoke.m`, `cdma_walsh_demo.m`).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| §8.1 PN | `pnSequence(poly_int, init_int, length, output_mode)` | ✅ shipped | LFSR (Fibonacci) generator. `poly_int` is the feedback polynomial as an integer mask with the implicit leading 1 (e.g. x⁴+x+1 → 0b10011 = 19); polynomial degree is the highest set bit. `output_mode` 0 = `{0, 1}` bits / 1 = `{−1, +1}` bipolar. Verified: poly = 19, length 30 produces an exact 15-period repetition. |
| §8.1 Gold | `goldSequence(poly1, poly2, init1, init2, length, output_mode)` | ✅ shipped | XOR of two LFSR outputs. Caller supplies preferred-pair polynomials per the textbook (e.g. (19, 25) for degree-4). |
| §8.1 Hadamard / Walsh | `hadamard(n)` (n × n Sylvester-form matrix; n snapped to next power of 2); `walshCode(n, k)` (1-based row index) | ✅ shipped | Standard recursive construction `H(2n) = [[H(n), H(n)]; [H(n), −H(n)]]`. Walsh codes (rows) are mutually orthogonal; verified in `cdma_walsh_demo.m` via `||A+B||² − ||A−B||² = 0`. Non-power-of-2 Hadamard orders (n = 12, 20, …) are not in scope. |
| §8.1 Kasami | `comm.KasamiSequence` | 🔵 | Function-form follow-on; needs the m-sequence-decimation helper not yet shipped. |
| §8.2 quantiz | `quantiz(sig, partition, codebook)` → integer index column; `quantizApply(idx, codebook)` → look-up | ✅ shipped | Caller supplies partition (M-1 sorted thresholds) and codebook (M entries). The split form matches the MATLAB `[indx, quant, dist] = quantiz(...)` triple via two calls (the distortion is `norm(sig - quantizApply(indx, codebook))^2`). |
| §8.2 Lloyd-Max | `lloydsQuant(sig, init_codebook, max_iter, tol)` | ✅ shipped | Iterative codebook refinement: per iteration, midpoint-partition assignment + per-region mean update. Stops when `max(\|Δcodebook\|)` falls below `tol`. Verified: 4-level codebook on 2000 i.i.d. standard-normal samples shifts to the canonical optimal levels (≈ −1.51 / −0.45 / 0.45 / 1.51). |
| §8.2 μ-law | `compandMu(x, mu, V, dir)` | ✅ shipped | G.711 μ-law: `dir = 0` compress / `dir = 1` expand. `V` is the peak amplitude. Round-trip is exact (norm of error 4e-16 at machine precision). |
| §8.2 A-law | `compandA(x, A, V, dir)` | ✅ shipped | G.711 A-law with the canonical A = 87.6 default. |
| §8.2 DPCM | `dpcmEncode(sig, partition, codebook)` / `dpcmDecode(idx, codebook)` | ✅ shipped | First-order predictor; residual is quantised through (partition, codebook) — caller designs both. The MATLAB-faithful `dpcmopt(sig, max_order, n_levels)` codebook-design helper is a 1-session follow-on. |
| Closure tests | `examples/comm/cdma_walsh_demo.m` | ✅ shipped | Two-user Walsh-coded CDMA round-trip (length-8 chips, 15 dB AWGN). Walsh-code orthogonality verified via the norm identity; both users decode 0 symbol errors. |

### Communications Toolbox — Tier-5 OFDM / fading / MIMO (function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §7. Function-form OFDM mod / demod, Rayleigh / Rician fading channels with Jakes-style Doppler, Alamouti 2-Tx space-time block coding, simple ML detector. System-Object variants (`comm.OFDMModulator`, `comm.OFDMDemodulator`, `comm.RayleighChannel`, `comm.RicianChannel`, `comm.OSTBCEncoder`, `comm.OSTBCCombiner`, `comm.SphereDecoder`) stay gated on the SO lowering fix. Runtime extends `runtime/runtime_comm.cpp`. End-to-end demos under `examples/comm/` (`tier5_smoke.m`, `ofdm_awgn.m`, `alamouti_diversity.m`).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| §7.1 OFDM | `ofdmmod(data, fft_len, cp_len)` / `ofdmdemod(samples, fft_len, cp_len)` | ✅ shipped | data is Nfft × Nsym complex (subcarriers as rows, OFDM symbols as columns); time-domain output is `(Nfft + cp_len) · Nsym × 1`. Uses the existing `matlab_fft_c` / `matlab_ifft_c` runtime kernels (Cooley-Tukey + Bluestein). Pilots / guards / null-subcarriers are caller-side compositions: zero out the relevant rows of `data` before `ofdmmod`. OFDM-over-AWGN loopback at 15 dB SNR recovers all 64 QPSK subcarriers (0 errors). |
| §7.2 Rayleigh | `rayleighChannel(x, delays_samples, gains_dB, max_doppler_Hz, fs_Hz)` | ✅ shipped | Multi-path Rayleigh with per-path Jakes sum-of-sinusoids generators (M = 16 oscillators, modified Mosalavi 2002). Each path independently faded; output is `(Nin + max_delay) × 1`. Both `delays` and `gains_dB` must have ≥ 2 elements (the dispatch reads ptr args; single-element literals are typed as scalar f64). |
| §7.2 Rician | `ricianChannel(x, K_dB, delays, gains_dB, max_doppler_Hz, fs_Hz)` | ✅ shipped | Rician decomposition: LOS component is the unshifted input scaled by `√(K/(K+1))`; scatter component is `rayleighChannel(...) · √(1/(K+1))`. K_dB is the Rician K factor in dB. Verified at K = 10 dB: average channel-output power ≈ 1.09 (LOS-dominated, as expected). |
| §7.3 Alamouti | `ostbcEncode(x)` → N × 2 complex; `ostbcCombine(y, h1_re, h1_im, h2_re, h2_im)` → N × 1 | ✅ shipped | Encoder pairs symbols `(s_{2k}, s_{2k+1})` across two Tx antennas with the canonical (s0, s1 / −conj(s1), conj(s0)) pattern. Combiner is the maximum-ratio Alamouti receiver, normalising by `|h1|² + |h2|²`. Channel is flat-fading scalar `(h1, h2)`; for time-varying channels callers re-call per coherence-time chunk. |
| §7.3 ML detect | `mlDetect(y, alphabet)` | ✅ shipped | Per-symbol Euclidean-nearest decision against a complex alphabet column. Returns integer labels 0..M−1. |
| §7.3 Sphere decode | `comm.SphereDecoder` / `sphereDecode(y, H, alphabet)` | 🔵 | Deferred — needs lattice reduction + tree search. ML on the alphabet (above) is the fallback for small M; for richer ZF / MMSE multi-antenna detection, complex LU (a separate ~1 wk slice) is the prerequisite. |
| Closure tests | `examples/comm/ofdm_awgn.m`, `examples/comm/alamouti_diversity.m` | ✅ shipped | OFDM AWGN loopback at 15 dB SNR: 0 symbol errors on 64 QPSK subcarriers. Alamouti 2-Tx with a known scalar `(h1, h2)` channel + AWGN at 10 dB: 0 errors vs 0.27% for the single-Tx baseline (combiner's ~3 dB coherent-combine gain is visible). |

### Communications Toolbox — Tier-4 equalisation / sync / RF impairments (function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §6. Function-form adaptive equalisers + carrier / symbol / frame sync + the four canonical RF impairments + soft-decision Viterbi. The `comm.LinearEqualizer` / `DecisionFeedbackEqualizer` / `CarrierSynchronizer` / `SymbolSynchronizer` / `PreambleDetector` / `PhaseNoise` / `MemorylessNonlinearity` System Objects stay gated on the SO lowering fix. Runtime extends `runtime/runtime_comm.cpp`. End-to-end demos under `examples/comm/` (`tier4_smoke.m`, `ber_soft_vs_hard.m`, `impairment_demo.m`).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| §6.1 LMS | `lms(x, d, mu, ntaps)` | ✅ shipped | Widrow-Hoff LMS adaptive filter. `mu` is the step size (typical 1e-3 to 1e-1). Returns the equalised output stream. For channels with non-trivial delay the caller should supply `d` aligned to the centre-tap delay. Verified: 7-tap LMS converges to 0 BER on a 3-tap first-tap-dominant channel at SNR 30 dB. |
| §6.1 RLS | `rls(x, d, lambda, delta, ntaps)` | ✅ shipped | Recursive least-squares with forgetting factor `lambda` (0.95–0.999 typical) and initial-P diagonal `delta` (1e2–1e4 typical). Converges faster than LMS at higher per-sample cost. Verified at 0 BER under the same scenario. |
| §6.1 CMA | `cma(x, mu, ntaps, R2)` | ✅ shipped (real projection) | Constant-modulus (Godard) blind equaliser; `R2 = E[|s|^4] / E[|s|^2]` (= 1 for unit-circle PSK). Centre-tap initialised to 1.0. Complex-input CMA is a Tier-5 follow-on. |
| §6.1 DFE | `dfe(x, d, mu, n_ff, n_fb)` | ✅ shipped | LMS-trained decision-feedback equaliser; `n_ff` feed-forward + `n_fb` feedback taps. First half of `d` runs in training mode, then switches to decision-directed (BPSK threshold at 0). |
| §6.2 Costas PLL | `costasPll(x, M_psk, loop_bw, fs)` | ✅ shipped (BPSK / QPSK / Mₐ) | M=2 squarer discriminator, M=4 4-PSK error term, otherwise atan2. 2nd-order PLL with damping 1/√2; `loop_bw` normalised to `fs`. Returns the de-rotated complex stream. |
| §6.2 Mueller-Müller | `symbolSyncMM(x, sps, loop_bw)` | ✅ shipped | NCO-driven sample selector with Mueller-Müller TED on real BPSK-style input; 1st-order loop. Output length = floor(N/sps). Verified: 20/20 last-symbol match on a clean 4-sps input. |
| §6.2 Preamble | `preambleDetect(x, preamble)` | ✅ shipped | Argmax of the cross-correlation across all valid lags; returns the 1-based start index. |
| §6.3 Phase / freq offset | `phaseFreqOffset(x, df_Hz, fs_Hz)` | ✅ shipped | y[n] = x[n] · exp(j·2π·df·n/fs). Inverse round-trip is machine-precision exact. |
| §6.3 IQ imbalance | `iqimbal(x, amp_imb_dB, phase_imb_deg)` | ✅ shipped | Scales the Q axis by 10^(amp_dB/20) and rotates by `phase_imb_deg` before re-adding to I. Verified: 0.5 dB / 5° imbalance shifts the QPSK magnitude norm from 10.00 to 10.30. |
| §6.3 Memoryless PA | `memorylessNl(x, model_code, p1, p2, p3, p4)` | ✅ shipped | model_code 0 cubic clipper (`p1` = saturation amplitude); 1 Saleh AM/AM + AM/PM with the classical 4-parameter form; 2 Rapp (`p1` = smoothness, `p2` = `Asat`); 3 Ghorbani-style 4-parameter form. Verified: Rapp at 1.5× drive into `Asat = 1` compresses 94.87 → 62.36 (≈ 1.0 · √N, the saturated limit). |
| §6.3 Phase noise | `phaseNoise(x, level_dBcHz, fs_Hz)` | ✅ shipped | Random-walk phase noise with σ² = 10^(level/10) · fs / 2 per sample. Verified: `|x|` preserved to 4e-15 (unit-modulus rotation). |
| §6.x Soft Viterbi | `vitdecSoft(llr, trellis, tblen, opmode)` | ✅ shipped | Max-log-MAP path-metric Viterbi; branch metric is the sum of LLR-with-sign over the n-tuple per state transition. Convention: positive LLR favours bit=0 (matches `qamdemodLlr`). Verified: at Eb/N0 = 5 dB on (171,133)₈ K=7, soft-decision BER ≈ 5.1e-3 vs hard 0.120 — textbook ≈24× gain. |
| Closure test (`examples/comm/ber_soft_vs_hard.m`) | hard vs soft Viterbi BER curves under BPSK + AWGN | ✅ shipped | At 50 k bits per Eb/N0 point: hard 0.484 / soft 0.415 at 1 dB; hard 0.120 / soft 0.0051 at 5 dB. Soft curve sits ~3 dB to the left of the hard curve — canonical soft-decision coding gain. |

### Communications Toolbox — Tier-3 channel coding (function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §5. Function-form CRC + convolutional codes + Hamming + block interleavers — every entry that does *not* need the System-Object lowering fix. The classdef `comm.CRCGenerator` / `comm.CRCDetector` form stays gated on the SO fix (CST roadmap §12 / §11.1). Runtime extends `runtime/runtime_comm.cpp`. End-to-end demos under `examples/comm/` (`tier3_smoke.m`, `ber_coded_vs_uncoded.m`).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| §5.1 CRC (function-form) | `crcGenerate(bits, poly_int, nbits)` → bits with CRC appended; `crcCheck(bits, poly_int, nbits)` → 0 / 1 flag; `crcStrip(bits, nbits)` → payload only | ✅ shipped | The generator polynomial is passed as a decimal integer whose binary representation is the lower `nbits` bits with the leading-1 implicit. E.g. CRC-16-CCITT poly 0x11021 → `crcGenerate(bits, 4129, 16)`. Plain shift-register implementation; works to `nbits` ≤ 63. The classdef `comm.CRCGenerator` form is the SO-gated arc. |
| §5.2 convolutional codes | `poly2trellis(K, gens)` (returns struct), `convenc(msg, trellis)`, `vitdec(code, trellis, tblen, opmode, dectype)`, `oct2dec(octal_decimal)` | ✅ shipped (hard-decision Viterbi) | Trellis struct has `numInputSymbols` / `numOutputSymbols` / `numStates` / `K` / `n` / `nextStates` / `outputs`. `oct2dec(171) = 121` bridge for textbook octal generators. `vitdec` runs forward path-metric accumulation + traceback; `opmode` 0 truncated / 1 terminated (assumes end-state 0); `dectype` is hard for the MVP slice. Verified: (171,133)₈ K=7 rate-1/2 corrects single bit errors clean and beats uncoded BPSK by ~2× at Eb/N0 = 7 dB in `ber_coded_vs_uncoded.m`. |
| §5.3 Hamming codes | `hammgenParity(m)` → m×n parity-check matrix; `hammingEncode(msg, m)` / `hammingDecode(code, m)` for binary (n = 2^m − 1, k = n − m) | ✅ shipped (single-error correction) | Systematic Hamming: message bits at non-power-of-two positions, parity bits at positions 1, 2, 4, …, 2^(m-1). Syndrome-decode with 1-bit correction. Verified: Hamming(7, 4) corrects a flip at every position 1–7. Per-call shape is one codeword (`length(msg) == k`); batching is caller-side. |
| §5.5 block interleavers | `intrlv(data, perm)` / `deintrlv(data, perm)` | ✅ shipped | Permutation vector is a 1-based row/column index map. Inverse round-trip is exact (verified). Convolutional / matrix interleavers stay deferred. |
| §5.4 LDPC / Turbo / Polar | — | 🔴 | Carved out per the roadmap §5.4 — each is a multi-week iterative-decoder arc (Tier-7 stretch). |
| BCH / Reed-Solomon + `gf(2^m)` | — | 🔵 | Deferred — needs a new typed runtime descriptor (small `m` + primitive-polynomial pair), ~2 wk on its own. |
| Closure test (`examples/comm/ber_coded_vs_uncoded.m`) | uncoded BPSK vs Hamming(7, 4) vs (171, 133)₈ K=7 convolutional, all over AWGN | ✅ shipped | 30 k information bits per Eb/N0 point. At 2 dB: uncoded 0.10 / Hamming 0.16 / conv 0.44 (the rate-1/2 conv code pays its dB penalty without enough SNR margin to recover); at 7 dB: uncoded 0.0125 / Hamming 0.0174 / conv 0.0056 — conv crosses over and beats uncoded by ~2× as Eb/N0 climbs past ~5 dB, exactly the canonical hard-decision Viterbi curve. |

### Communications Toolbox — Tier-2 digital modulation MVP (function-form)

Per-toolbox roadmap in [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §4. The first user-visible Comm slice: source → modulate → AWGN → demodulate → BER, with a closed-form theory overlay. Function-form, numeric-tag dispatch; runtime extends `runtime/runtime_comm.cpp`. End-to-end demos under `examples/comm/` (`tier2_smoke.m`, `pulse_shape_demo.m`, `ber_qam_montecarlo.m`).

| Group | Function | Status | Notes |
|---|---|:-:|---|
| §4.1 PAM | `pammod(x, M, order)`, `pamdemod(y, M, order)` | ✅ shipped | order = 0 natural / 1 Gray. Constellation `2k − (M−1)` for k in [0, M−1]. Real-line output. |
| §4.3 PSK | `pskmod(x, M, ini_phase, order)`, `pskdemod(y, M, ini_phase, order)` | ✅ shipped | Phase = ini_phase + 2π·k/M with Gray decoding. Complex output. Hard demod via atan2 + nearest-phase. |
| §4.2 QAM | `qammod(x, M, order, unit_avg)` + `qamdemod` (hard / `qamdemodBit` / `qamdemodLlr`) | ✅ shipped | Square M ∈ {4, 16, 64, 256, 1024} via independent kx + ky bit splits per I/Q axis. Rectangular cross-QAM for M=8 (4×2) and M=32 (8×4). `unit_avg = 1` scales by `1/√(2(M−1)/3)` so mean symbol energy is 1. `qamdemodBit` emits `N·log2(M)` MSB-first bits; `qamdemodLlr` max-log LLR with a user-supplied noise variance. |
| §4.6 Generic | `genqammod(x, alphabet)`, `genqamdemod(y, alphabet)` | ✅ shipped | Alphabet is a `matlab_mat_c` column; demod is nearest-Euclidean-distance. |
| §4.7 Pulse shaping | `rcosdesign(beta, span, sps, shape)`, `gaussdesign(BT, span, sps)` | ✅ shipped | shape = 0 root-raised-cosine ('sqrt'), 1 = full raised-cosine ('normal'). Closed-form impulse response with L'Hôpital handling at `t=0` and `t = ±span/(4β)`. Unit-energy normalised. Gaussian uses GMSK `α = √(ln 2 / 2) / BT`, sum-normalised to 1. |
| §4.8 berawgn | `berawgn(EbN0_dB, M, mod_code)` | ✅ shipped | mod_code 0 PAM / 1 PSK / 2 QAM / 3 DPSK / 4 FSK-coh / 5 FSK-nc. Closed-form per the user-guide table; uses libc `erfc`. Verified: BPSK BER @ 10 dB Eb/N0 = 3.87e-6, 16-QAM = 1.75e-3 — matches textbook to printed precision. |
| §4.9 scatterplot | `scatterplot(x)` | ✅ shipped | Numeric form returning N×2 real matrix of (re, im) pairs. Cairo plotting on top is left to user code. |
| §4.5 FSK | `fskmod` / `fskdemod` | 🔵 | Deferred — not needed for the closure test; the oversampling + non-coherent energy-detect path is a separate ~1 wk slice. |
| Scalar runtime tail | `qfunc(x)`, `erfc(x)` | ✅ shipped | `qfunc(x) = 0.5·erfc(x/√2)`; thin wrappers over libc. |
| Closure test (`examples/comm/ber_qam_montecarlo.m`) | `randi → qammod → awgn → qamdemod → biterrK` vs `berawgn` overlay | ✅ shipped | 16-QAM at 20 k symbols / Eb/N0 point: sim 0.0578 / 0.0283 / 0.0094 / 0.0019 / 1.25e-4 vs theory 0.0586 / 0.0279 / 0.0092 / 0.0018 / 1.39e-4 at 4 / 6 / 8 / 10 / 12 dB. The 14 dB point is statistically noisy (~1 expected error per run). |

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

Per-toolbox roadmap in [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md) (promoted to a dedicated roadmap as of 2026-05-17; previously chapter §3 of [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md)). Function-form surface — no classdef System Objects required, so this track ships in parallel with the SO-gated Comm Tier-3+ / RF / Antenna arcs. All entries live in `runtime/runtime_prop.cpp`. End-to-end demos under `examples/rf/` (Barbados PtP + ITM coverage, sector-coverage SINR, Fresnel/diffraction, pattern sampling).

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
| Carved out (deliberately deferred) | Site Viewer 3-D map, ray tracing through OSM buildings, auto-fetch SRTM/DTED/OSM tile servers, TIREM, MSI Planet I/O, animated live coverage, GPU acceleration | 🔴 | Per [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md) §8 (Site Viewer track at [`siteviewer.md`](siteviewer.md)). |
| Classdef wrappers (PROP-Tier-1b §3.5) | `propagationModel(kind)`, `txsite(...)`, `rxsite(...)`, `pathloss(prop, rx, tx)`, `link(tx, rx)`, `coverage(tx, prop, ...)`, `los(tx, rx)`, `sigstrength(rx, tx, pm)`, `show(...)` | ✅ shipped | CamelCase classdefs (`PropagationModel`, `TxSite`, `RxSite`) with kwarg-sugar constructors — `TxSite('Latitude', 42.3, 'Longitude', -71.35, ...)` etc.  Methods dispatch through the function-form runtime.  Antenna gain is a scalar stub today (full directional patterns land with ANT-Tier-2). |

### RF Toolbox (subset)

Per-toolbox plan in [`rf_toolbox_plan.md`](rf_toolbox_plan.md).  Two-commit closure arc: `44198e5` (Tier-1 + Tier-1 polish) and `56e324c` (Tier-2 generalizations).  All entries live in `runtime/runtime_rf.cpp` (~6 100 lines).  387 / 387 Run/ tests pass.

| Group | Function | Status | Notes |
|---|---|:-:|---|
| Touchstone I/O (RF-Tier-1.3) | `touchstoneRead(filename)`, `touchstoneWrite(filename, data)`, `touchstoneWriteS2p(...)`, `tsSij(data, i, j)` | ✅ shipped | Auto-detects `.s1p` / `.s2p` / `.s3p` / `.s4p` / `.sNp` / `.ts` from the filename extension; tolerates MA / DB / RI data formats, Hz / kHz / MHz / GHz units, multi-line per-frequency rows.  v2 (`.ts`) parser recognises `[Number of Ports]` / `[Reference]` / `[Two-Port Order]` / `[Network Data]` / `[End]` keywords. |
| Network parameter classdefs (RF-Tier-1.1) | `RFSparameters` + `RFYparameters` / `RFZparameters` / `RFHparameters` / `RFGparameters` / `RFAbcdparameters` / `RFTparameters` | ✅ shipped | Property-holder skeletons paralleling `RFSparameters`.  Population from the corresponding `sparam*` runtime helper's struct return via direct field assignment. |
| 2-port conversions (RF-Tier-1.2) | `sparamS2y`, `sparamS2z`, `sparamS2h`, `sparamS2g`, `sparamS2abcd`, `sparamS2t` + inverses `sparamH2s`, `sparamG2s`, `sparamAbcd2s`, `sparamT2s` | ✅ shipped | Per-frequency closed-form 2×2 matrix algebra.  Round-trip S → X → S exact to machine precision for any non-singular 2-port. |
| N-port conversions (RF-Tier-1.2 generalization) | `sparamS2yN`, `sparamS2zN`, `sparamS2abcdN`, `sparamS2hN`, `newref(spar, z0_new)` | ✅ shipped | Native complex matrix algebra via the LU path below.  N-port ABCD / H use the Y-partition formula for even N.  `newref` re-references S to a new scalar reference impedance via the Γ_a-renormalization formula. |
| Mixed-mode 4-port (RF-Tier-1.2 follow-on) | `sparamS2smm(s11..s44, block_code)` | ✅ shipped | block_code ∈ {0=dd, 1=dc, 2=cd, 3=cc} selects the differential / common / mode-conversion 2×2 sub-block of the mixed-mode transform. |
| Port extraction (RF-Tier-1.2 follow-on) | `snp2smp(data, port_list, m)`, `snp2smpZ(data, port_list, z_term, m)` | ✅ shipped | Matched (`snp2smp`) and arbitrary-termination (`snp2smpZ` Schur complement) port extraction. |
| Closed-form 2-port analyses (RF-Tier-2.1) | `gammaIn`, `gammaOut`, `vswr`, `powerGain` (Gt/Ga/Gp via type code), `stabilityK` (Rollett), `stabilityMu` (Edwards-Sinsky mu1/mu2), `s2tf`, `s2tfPort`, `gammams`, `gammaml`, `groupdelay`, `gamma2z`, `z2gamma` | ✅ shipped | All per-frequency closed-form.  `groupdelay` uses centered finite differences over the supplied frequency vector with phase unwrap.  `gammams` / `gammaml` are the simultaneous-conjugate-match Γ values for max-available-gain design. |
| Network cascade (RF-Tier-2.2) | `cascadeSparams2`, `cascadeSparamsN`, `cascadeSparamsNFull`, `cascadeSparamsNFullK` | ✅ shipped | 2-port T-parameter chain; N-port diagonal approximation; full Redheffer star product (k = N/2 symmetric case + arbitrary k generalization with asymmetric outer port counts). |
| RF system analysis (RF-Tier-2.3) | `rfbudgetFriis(gains_dB, nfs_dB, ip3_dBm, p_in_dBm, bw_Hz)`, `rfbudgetTable(...)`, `stabCircleLoad`, `stabCircleSource` | ✅ shipped | Friis cascade returns `{Gain_dB, NF_dB, IP3_in_dBm, OutputPower_dBm, NoiseFloor_dBm, SNR_dB, ...}`.  `rfbudgetTable` adds per-stage cumulative columns.  Stability circles return per-frequency Center / Radius / Denom (sign-of-Denom encodes which side of the circle is unstable). |
| Vector Fitting (RF-Tier-3.1) | `rationalfit(freqs, h_re, h_im, nPoles, nIter)`, `rationalfitWeighted(..., weight, ...)`, `freqresp(mdl, freqs)`, `passivity(mdl, f_lo, f_hi)`, `rfPoles` / `rfResidues` / `rfD` / `rfOrder` / `rfFitError` typed getters | ✅ shipped | Gustavsen-Semlyen v2 with **both real and complex-conjugate pole pairs** — the relocation matrix M uses the real-arithmetic `[α β; -β α]` block form for complex pairs, eig auto-classifies eigenvalues, output stores complex Poles + complex Residues columns.  `freqresp` / `passivity` / `timeresp` consume complex poles via layout-magic-aware getters.  `rationalfitWeighted` scales each LS row by √(weight[k]). |
| Bulk delay + passivity (RF-Tier-3.1 follow-on) | `rfDelayEstimate(freqs, h_re, h_im)`, `rfApplyDelay(freqs, h_re, h_im, tau)`, `rfPassivityEnforce(mdl, f_lo, f_hi)` | ✅ shipped | `rfDelayEstimate` extracts transport delay τ from the top-25%-of-spectrum phase slope.  `rfApplyDelay` multiplies the data by exp(+jωτ) to de-delay before VF.  `rfPassivityEnforce` iteratively scales residues + D until max|H(jω)| ≤ 1. |
| Time-domain RF (RF-Tier-3.2) | `timeresp(mdl, u, ts)`, `s2tdr(S11, freqs, nPoles, ts, nSamples)`, `s2tdt(S21, freqs, nPoles, ts, nSamples)` | ✅ shipped | Per-pole ZOH discretization with complex state when poles are complex.  Step-response of `1/(s+1)` to a unit step at ts=1 matches `[0, 1−1/e, 1−1/e², 1−1/e³, 1−1/e⁴]` exactly. |
| Transmission line geometries (RF-Tier-3.3) | `rfckt_txline(Z0, εr, len, freqs, z0)`, `rfckt_coaxial(a, b, εr, len, ...)`, `rfckt_microstrip(w, h, εr, len, ...)`, `rfckt_cpw(w, s, εr, len, ...)`, `rfckt_parallelplate(...)`, `rfckt_twowire(r, D, εr, len, ...)` | ✅ shipped | Closed-form lossless.  Microstrip uses Hammerstad-Jensen.  CPW uses Hilberg's K(k')/K(k) approximation.  FR-4 microstrip w/h ≈ 1.91 → Z₀ = 50.42 Ω verified. |
| Matching networks (RF-Tier-4.1) | `matchingnetwork(zs_re, zs_im, zl_re, zl_im, freq)` (L), `matchingnetworkT(..., q_target)`, `matchingnetworkPi(..., q_target)` | ✅ shipped | L-section auto-synthesis + T/Pi via dual-L cascade at a virtual-impedance node.  100→50 Ω at 1 GHz → Q=1, L_series = 7.96 nH, C_shunt = 1.59 pF. |
| RF circuit hierarchy (RF-Tier-4.2) | `RFCktAmplifier`, `RFCktMixer`, `RFCktPassive` blocks + `RFCktCascade` / `RFCktParallel` / `RFCktSeries` / `RFCktShunt` combinators + `analyze(block, freqs)` method on each + function-form `rfAnalyzeAmplifier` / `rfAnalyzePassive` / `rfAnalyzeSeries` / `rfAnalyzeShunt` | ✅ shipped | Each rfckt classdef has an `analyze(obj, freqs)` method that dispatches to the corresponding `rfAnalyze*` runtime helper; result is a 2-port S-param struct identical in shape to `touchstoneRead`'s output. |
| LC filter circuits (RF-Tier-4.2 follow-on) | `rfckt_lcfilter(topology, comp1, comp2, freqs, z0)` (3-element Lowpass / Highpass Tee + Pi), `rfckt_lcfilter4(topology, L1, C1, L2, C2, freqs, z0)` (4-element Bandpass / Bandstop Tee + Pi) | ✅ shipped | Closed-form per-frequency S-params via T-parameter chain composition.  Bandpass-Tee at ω₀ = 1 GHz: \|S₂₁\| ≈ 1 in-band, ≈ 2×10⁻⁵ at 3 GHz (deep stopband). |
| Smith chart (RF-Tier-4.3) | `smithGrid(r_norm, n_pts)` + `smithRCircle(g)` / `smithUnitCircle(g)` typed getters, `stabCircleLoad(spar)` / `stabCircleSource(spar)` | ✅ shipped | Constant-r circle + unit-circle Γ-plane overlays; input + output stability circles with per-frequency center + radius + Denom-sign. |
| `RFRational` value classdef (rfmodel.rational) | `RFRational()` with A / C / D / Delay / Order / Error properties | ✅ shipped | MathWorks-API value wrapper around the rationalfit struct.  Population via the typed getters: `mdl = RFRational(); mdl.A = rfPoles(s); mdl.C = rfResidues(s); ...`. |
| MathWorks-faithful lowercase aliases | `s2y` / `s2z` / `s2h` / `s2g` / `s2abcd` / `s2t` + inverses + `rfbudget` / `rfwrite` / `sparameters` | ✅ shipped | Lowercase aliases registered in the dispatch table; tutorial-style MathWorks code copies verbatim. |
| Internal numerics | Native complex N×N LU decomposition with partial pivoting (Doolittle); fallback to 2N×2N real-equivalent on singular pivot | ✅ shipped | Transparently powers every matrix-inverse call in the RF runtime (~4× faster than the real-equivalent path for non-singular matrices).  Caps at N ≤ 9 (matches the multi-port field-name decoration). |
| Verilog-A export (Tiers 1 – 10) | RF / signal-integrity: `writeVerilogA`, `writeVerilogATF`, `writeVerilogAZPK`, `writeVerilogASS`.  Analog blocks: `writeVerilogASource`, `writeVerilogAComparator`, `writeVerilogASchmitt`, `writeVerilogAVCO`, `writeVerilogADAC`, `writeVerilogADiode`, `writeVerilogAOpAmp`, `writeVerilogARTD`, `writeVerilogAThermistor`.  Noise: `writeVerilogANoise`.  Lookup tables: `writeVerilogATable` (+ `.tbl` sidecar via `$table_model`).  **Composite RF / signal-chain blocks (Tier-7 follow-on)**: `writeVerilogAAmplifier` (gain × LPF × `tanh`), `writeVerilogAAM` (amplitude modulator), `writeVerilogAIQMod` (generic I/Q modulator — covers QAM-16 / QPSK / 8-PSK).  User reference: [`emit_verilog_a.md`](emit_verilog_a.md).  Tier-10 polish: `scripts/va_lint.sh` (OpenVAF / ADMS) + `scripts/va_cosim.sh` (ngspice+OpenVAF / Xyce+ADMS) plus two opt-in CTest lanes (`run-emit-va-admslint`, `run-emit-va-cosim`) gated on `MATLAB_LLVM_WITH_VA_LINT` / `MATLAB_LLVM_WITH_VA_COSIM`. | ✅ shipped | Per [`verilog_a_plan.md`](verilog_a_plan.md) §6 Tiers 1–10.  Tiers 1–3 cover rational models / tf / zpk / state-space (real-pole + complex-pair biquads + `ddt(x[i])` arrays + `absdelay` delay wrap).  Tier 4 ships analog sources (`$abstime`-driven sin/cos/square/exp), `cross()`-event comparators + Schmitt triggers, both with `transition()` smoothing.  Tier 5 emits VCOs via `idtmod` phase accumulation.  Tier 6 ships a pure-Verilog-A behavioral DAC (analog-coded input; bit-bus deferred to Tier-11 / Verilog-AMS).  Tier 7 covers compact components (Shockley diode, `tanh`-saturated op-amp) and sensor models with first-class `$temperature` (Pt-100 RTD, β-equation NTC thermistor).  Scalar-fold dispatch shims handle 1-element collapses.  27 `rf_writeva_*` run-tests, 24 examples under `examples/verilog_a/` (Tier-1 through Tier-9 primitives + Tier-7 follow-on composites + the canonical RF modulator suite — LP filter, amplifier, AM / FM / QAM-16 modulators).  Two infra fixes shipped alongside: `complex(re, im)` matrix-arg variants (closes the `1i * real_col` ergonomics gap) and classdef matrix-property storage / read-back (via `TypeName == "complex"` annotations + `mat_alloc(0,0)` zero-size fall-back in writeVerilogA, with a follow-on `rf_va_field_or` fix at commit `7b1f727` to read `rows`/`cols` at the correct offsets for `matlab_mat_c` descriptors). |
| Carved out (deliberately deferred) | Circuit envelope simulation, harmonic balance solver, RF Budget Analyzer / Smith Chart Tool apps (Qt), Modelithics commercial component library, IEEE P370 fixture characterization, AMP file format reader, Simulink RF Blockset | 🔴 | All require infrastructure outside the language layer (multi-tone time-stepping solvers, GUI stack, commercial licensing).  Per `rf_toolbox_plan.md`. |

### Strings

| Feature | Status |
|---|:-:|
| String literal creation, `strlen`, `isstring` | ✅ |
| Concatenation: `[s1 s2]`, `strcat(a, b)`, `s1 + s2` | ✅ |
| `sprintf` (literal + single-f64 form), `num2str`, `str2double` | ✅ |
| `strtrim`, `strrep`, `deblank`, `blanks` | ✅ |
| `upper`, `lower`, `startsWith`, `endsWith`, `contains` | ✅ |
| `strcmp`, `strcmpi`, `strncmp` | ✅ |
| `strfind`, `strjoin` | ✅ | `strjoin(C[, delim])` joins a cell of strings; `strfind` returns a 1×k index row vector. |
| `char(code)` (numeric scalar → 1-char string) | ✅ | Vector `char([codes])` is a follow-up. |
| char-array arithmetic / comparison on codes (`'hello' == 'l'`, `c + 1`) | ✅ | A char value in a numeric/comparison op evaluates on character codes (literals folded; variables materialized to an f64 code matrix — AOT i8-tensor and REPL `matlab_string*` lanes). Works **cross-turn** in the REPL via a dedicated char workspace kind (kind=18) + `Binding::IsChar`, so `disp`/concat stay string-semantic (#265, #289). |
| char-literal args (`'…'`) to the above string builtins | ✅ | Predicates/transforms/`str2double` materialise `const_char` literals, so both literal and string-variable args work. |
| `strsplit`, `regexp`, `regexprep`, `str2num` | ❌ | `strsplit` needs cell-result element-kind tracking; regex needs an engine. |

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
| Stateflow (`.mflow` `settings.kind = "state_chart"`) dialect | 🟡 | **Hierarchical state-chart authoring + live-debug** as a third `.mflow` dialect alongside `control_flow` and `signal_flow`. **Status: PARTIAL** — the compiler / debugger / DAP / REPL side and the integer-typed Moore / Mealy / AND-parallel examples are shipped end-to-end; broader Stateflow semantic parity is still a work in progress (matches `openspec/specs/stateflow` and the 🟡 in `README.md`). Schema additions: `FlowNode.parent`, `ui.size`, `data.onEventActions`, `Flow.symbols` (data + events + messages tables), `EdgeKind.transition` with `event[guard]{condAction}/transAction` label parsing. New node kinds: `state` + 5 junction variants + 3 chart-function call-sites. Compiler stack (`lib/StateChart/`): `StateChartIR` (Chart / ChartState / ChartJunction / Transition / ChartFunction), `Lowering` with two target forms — JIT-friendly persistent-scalar (drives `-emit-matlab`/`-emit-mir`/`-emit-llvm`/`-emit-c`/`-emit-cpp` cleanly through matlabc's MATLAB→LLVM lane) and synthesizable HDL form (drives `-emit-systemverilog`/`-check-synthesizable`/`-emit-hardware-report`/`-emit-cocotb` — per-variable `if isempty(X), X = intW(0); end` reset inits, integer-typed locals, single-pass tick). C++ chart interpreter (`Interpreter.{h,cpp}`) with backtracking junction resolution, history, super-transitions, temporal counters, symbol-change watchpoints, snapshot/restore — hosts the live DAP simulation. `runtime/runtime_mstateflow.cpp` (bounded FIFO event queue + snapshot ring with introspection + DAP event sinks) + `runtime/mstateflow_helpers.m` + `runtime/stateflow_classdefs.m` (`stateChart` REPL classdef). CLI: `-dump-chart`, `-emit-matlab`, `-simulate` (interpreter trace), `-simulate --sim-dap` (chart DAP server). DAP namespace `stateChart/*`: events stateEnter/stateExit/transitionFired/eventBroadcast/superStepBegin/End/maxIterations + `stopped` on BP hits; requests emit/setLocal/getActive/getLocals/stepSuperStep/stepTransition/set{State,Transition,Symbol}Breakpoints/save+restoreOperatingPoint; introspection list{States,Transitions,Junctions,Events,Symbols,Snapshots}. All six MathWorks §6.8 canonical fixtures shipped + Moore / Mealy / AND-parallel examples that emit verilator-clean SystemVerilog. See `docs/mStateflow_roadmap.md` and `examples/stateflow/`. |
| JIT / REPL | 🟡 | `matlabc -repl` with MLIR ExecutionEngine; state persists via a runtime workspace. No line editing / JIT cache / live user-function definitions yet. See `docs/repl.md`. |
| Python emission | ✅ | `-emit-python`. NumPy-backed runtime in `runtime/matlab_runtime.py`; see `docs/emit_python.md`. Matrix display uses numpy's bracket repr (`.stdout-python` per-test goldens for the test lane). Multi-return uses native tuple unpacking (`a, b = f(...)`); persistent + isempty-init lowers to `<fn>.<name> = <init>` at module scope. |
| TypeScript emission | 🟡 | `-emit-typescript`. Same scope as Python; runtime in `runtime/matlab_runtime.ts`. Multi-return uses array destructuring (`const [a, b] = f(...)`); persistent + isempty lowers to `let <fn>_<name>: number = <init>;`. |
| SystemVerilog (ASIC, synthesizable) emission | ✅ | `-emit-systemverilog`. Vendor-neutral, synthesizable RTL targeting ASIC flows. Tier-1 closure shipped: scalar combinational + FSMs + fixed-point pipeline + persistent fi-arrays + readability polish + bit-slicing `x(hi:lo)` (any width 1..64) + runtime-indexed persistent fi-arrays (auto-decoded regfile pattern) + hierarchical multi-module emission (`func.call` → SV instance with auto-wired clk/rst_n). 77 golden fixtures lint clean under Verilator (incl. `aes_round`, `cic_decimator`, `cordic_pipe`, `crc32`, `fir_asic_pipelined`, `i2c_bit_bang`, `regfile_dyn`, `spi_master`, `uart_rx`, `vector_processor`, plus `hier_combinational` / `hier_sequential` for multi-module). Also covers **mStateflow chart inputs**: a `.mflow` with `settings.kind = "state_chart"` flows through `lib/StateChart/Lowering.cpp`'s HDL target (one-pass tick, per-variable `if isempty(X), X = intW(0); end` resets, integer-typed locals + region codes, inlined `in()` predicates) before re-entering the standard SV pipeline. Moore / Mealy / AND-parallel charts in `examples/stateflow/` produce verilator-clean modules (`traffic_light_moore.sv` 122 lines, `vending_machine_mealy.sv` 106 lines, `model_air_temperature_controller.sv` 208 lines). 7 fi-spec ↔ SV declaration regression tests in `test/EmitSVPorts/`, 2 boolean-port lint-hint tests in `test/EmitSVHint/`, 10 synthesizability-gate diagnostic tests in `test/EmitSVFail/`. Open: 2-D fi matrices, RAM inference, CORDIC for transcendentals. See `docs/sv_supported_subset.md` (supported-subset reference), `docs/emit_systemverilog.md` (backend architecture), and `docs/mStateflow_roadmap.md` (chart → SV pipeline). |
| GPU kernel emission (CUDA / Metal / OpenCL) | ✅ | `-emit-cuda` / `-emit-metal` / `-emit-opencl`. Standalone GPU Coder bundles (kernel source + host driver + Makefile) from `coder.gpu.kernelfun` MATLAB. CUDA emits an **nvcc-free** NVRTC host driver; OpenCL an **SDK-free** ICD driver; both build + run with just the device driver. The runtime `gpuArray` path also has device backends behind opt-in flags (`-DMATLAB_LLVM_GPU_{CUDA,OPENCL,METAL}=ON`, default OFF): **CUDA** (cuBLAS `Dgemm` fp64 + NVRTC JIT, driver-API only) and **OpenCL** (fp64 GEMM kernel) — **validated end-to-end on NVIDIA hardware** (RTX 5060, sm_120; issue #25) — plus Apple **Metal** (MPS) on Apple silicon. HW-gated validation lanes `test/Run/run_gpu_{cuda,opencl}_validation.sh`. See `docs/gpu_coder_roadmap.md`. |

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
| `run-tests` (`-emit-llvm` + clang) | 387 | ✅ |
| `run-tests-emit-c` (`-emit-c` + cc) | 387 | ✅ (RF Toolbox tests skip emit-C — they exercise runtime classdef wrappers + Touchstone I/O that only lower through MLIR / JIT) |
| `run-tests-emit-cpp` (`-emit-cpp` + c++) | 387 | ✅ (RF Toolbox tests skip emit-C++ for the same reason) |
| `run-tests-emit-c-strict` / `-cpp-strict` (-Wall -Wextra -Werror) | 387 | ✅ |
| `run-tests-emit-python` (`-emit-python` + python3) | 387 | ✅ (RF Toolbox tests skip emit-Python; some `.stdout-python` overrides for numpy repr) |
| `run-tests-emit-typescript` (`-emit-typescript` + bun) | 387 | ✅ (RF Toolbox tests skip emit-TypeScript; `string_concat_mixed` fixed in Phase 6.2; ~20 skipped for BigInt-vs-number coercion) |
| `run-tests-sym` (`-emit-cpp` + SymPP, opt-in via `-DMATLAB_LLVM_WITH_SYM=ON`) | 4 | ✅ — Phase 6.2 sym_phase_a/b/b1/b2 fixtures; skip-if-missing-SymPP via rc=77 |
| `emit-sv` golden tests + Verilator lint + Yosys synth | 77 | ✅ 77/77 |
| `emit-sv-fail` synthesizability gate diagnostics | 10 | ✅ 10/10 |
| `emit-sv-ports` fi-spec ↔ SV declaration regression | 7 | ✅ 7/7 |
| `emit-sv-hint` boolean-port lint hints | 2 | ✅ 2/2 |
| `emitc-fail-tests` (diagnostic contract) | 1+ | ✅ |
| `flowchart-tests` (`.mflow` loader: schema, validation, error paths — control / signal / state-chart dialects all covered) | 55 | ✅ 55/55 (3 control-flow + 8 control-flow-error + 4 state-chart-error + 10 state-chart fixtures × 4 modes — flow / chart-IR / lowered-MATLAB / interpreter-trace) |
| `flowchart-emit-matlab-tests` (linear / control / sub-flows / custom blocks) | 17 | ✅ 17/17 |
| `flowchart-cross-backend-tests` (`.mflow` ≡ round-tripped `.m` across C / C++ / Python / TS) | 12 × 4 | ✅ 48/48 |
| `flowchart-lsp-tests` (`matlab-lsp` accepts `.mflow`, surfaces diagnostics) | 3 | ✅ 3/3 |
| `flowchart-dap-tests` (`matlabc -dap` on `.mflow`: bp verify, stop, frame source) | 3 | ✅ 3/3 |
| `flowchart-emit-mflow-tests` (`-emit-mflow` idempotency: `.m` → `.mflow` → `.m` → `.mflow` byte-identical) | 11 | ✅ 11/11 |

Examples gallery: 380 `.m` programs under `examples/` (31 top-level + dedicated subdirectories per toolbox: `examples/optim/`, `examples/comm/`, `examples/rf/`, `examples/control/`, `examples/signal/`, `examples/pde/`, `examples/plot/` — incl. `getframe`/`VideoWriter` animation demos, `examples/antenna/`, `examples/verilog_a/`, `examples/hdl/`, `examples/mflow/`, `examples/mflowlink/`, `examples/stateflow/`). They exercise matrix ops,
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
| REPL / interactive interpreter | 🟡 | JIT via MLIR ExecutionEngine, persistent workspace, implicit display, `who`/`whos`/`clear`. **Line editing** (history ↑/↓, Ctrl-A/E/U/K/L), **multi-line block input** (#290), **persistent history** (`$MATLABC_HISTFILE`) + **tab completion** (#291). `matlabc -repl`. See `docs/repl.md`. |
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
| **`classdef` dependent types** (`datetime`, `categorical`, `table`) | ✅ shipped (Phase 5.1–5.3) | datetime / duration / categorical / table / timetable all backed by dedicated runtime descriptors (timetable = Phase 5.4); see the per-type rows above. |
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
  Headless plotting (`plot`, `surf`, `bar`, ... → PNG/SVG/PDF, plus
  `getframe` / `VideoWriter` → MP4/AVI) is shipped; see
  [`plotting.md`](plotting.md).
- **Simulink** — full block-library + simulation engine + Coder + RT
  workshop is its own product. A practical subset ships under a
  different name as the **mflowLink** (signal-flow `.mflow`) dialect +
  Embedded Coder lane: see [`mflow_link_roadmap.md`](mflow_link_roadmap.md)
  and [`embedded_coder_roadmap.md`](embedded_coder_roadmap.md). Stateflow's
  charting / debug surface ships under **mStateflow**: see
  [`mStateflow_roadmap.md`](mStateflow_roadmap.md).
- **Toolboxes — apps, Live Editor tasks, deep-learning bridges**.
  Practical subsets of Signal Processing, Control System,
  Communications + RF + Antenna + Propagation, Optimization, PDE,
  Symbolic Math, and Fixed-Point Designer **do** ship in the runtime —
  see the in-scope list at the top of this doc. What stays out of
  scope: every toolbox's interactive app surface (e.g., Filter
  Designer, Control System Designer, PID Tuner, optimtool, Antenna
  Designer, Smith Chart Tool, RF Budget Analyzer, PDE Modeler, Live
  Editor tasks), MATLAB Coder UI integration, deep-learning bridges
  (DL Toolbox / RL Toolbox), and the full SO surface for the comm
  classes (gated on the System-Object lowering fix tracked in CST §12).
- **MEX interop** — loading compiled `.mex` files; deep binary-ABI lock-in with MathWorks.
- **Live Scripts** (`.mlx`) — proprietary format; use Jupyter or a documentation toolchain instead.
- **GPU arrays** (`gpuArray`) — **now shipped** (was previously out of scope). The `gpuArray` runtime surface + GPU Coder emit lanes (`-emit-{cuda,metal,opencl}`) have CUDA / Metal / OpenCL device backends; CUDA (cuBLAS + NVRTC) and OpenCL are validated on NVIDIA hardware (issue #25). See the GPU row in §5 and `docs/gpu_coder_roadmap.md`. (AMD ROCm / clBLAST remain follow-ons.)
- **Code generation toolbox features** (`coder.config`, etc.) — this project *is* a code generator; MATLAB Coder compatibility is a different product.
- **Bit-exact MATLAB numerics** — LAPACK vs. pure-C linear algebra will disagree in the last few ULPs. Correct to tolerance, not to bit.

---

## 9. Open runway

This section tracks what's still missing from the **language core** —
the bits of MATLAB that aren't toolbox-specific. The earlier
2024-era list (struct arrays, varargout, dim-aware reductions, sort
/ linalg tail, strings, REPL, file I/O, basic OOP, tooling,
`containers.Map`/`dictionary`, 2-D cells, datetime/duration/
categorical/table) has all shipped — those items were rolled into
the per-feature matrix above and into the
[`port_runtime_2_cpp.md`](port_runtime_2_cpp.md) shipped log. The
remaining language-core runway:

| Priority | Item | Effort | Unlocks |
|:-:|---|--:|---|
| 1 | Narrower / wider int matrix lanes (i8 / i16 / i64 / u16 / u32 / u64) on top of the shipped i32/u8 template | ~1 wk | Image-processing pixel code (note: 64-bit lanes already exist via the fi-array work) |
| 2 | 3-D vector slicing `A(:,:,k)` and the broader 3-D tensor surface (most elementwise / reduction ops reject 3-D today) | 2–3 wk | Batch dims, volumetric code, tensor code |
| 3 | Complex linalg tail — full complex `inv` / `det` / `svd` / `eig` (real paths shipped; complex partial via 2N×2N real-equivalent fallback) | 1 wk | Complete DSP / scientific code |
| 4 | OOP value-class method-dispatch semantics + property validators (copy-on-assign already ships) | ~1 wk | Modern MATLAB code |
| 5 | DAP user-function frames + watch expressions inside function bodies | 1 wk | Stepping into user functions shows their frames |
| 6 | `regexp` / `regexprep` + string tail | 1–2 wk | Text-processing scripts |
| 7 | `[U, S, V] = svd` (full SVD with U / V; today only singular values) | 1 wk | Scientific computing |
| 8 | MATLAB `.mat` v5 file-format parser | 2 wk | Real data pipelines |
| 9 | OOP events / listeners | 1 wk | Callback-heavy code |
| 10 | `timetable` — ✅ SHIPPED (Phase 5.4) | done | Time-series analysis (Financial Toolbox) |
| 11 | System-Object lowering fix — gates the `comm.*` / RF / Antenna / Propagation classdef wrappers on the SO surface | 1 wk | Closes Comm Tier-3+, the SO-bearing rows of [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §11.1 |

For per-toolbox follow-ons, see the dedicated roadmap docs — each
keeps its own "next slice" list:

| Toolbox | Roadmap |
|---|---|
| Signal Processing | [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) |
| Control System | [`control_toolbox_roadmap.md`](control_toolbox_roadmap.md) |
| Communications | [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) |
| RF | [`rf_toolbox_plan.md`](rf_toolbox_plan.md) |
| Antenna | [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md) |
| Propagation | [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md) |
| Optimization | [`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md) |
| Model Predictive Control | [`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md) |
| System Identification | [`ident_toolbox_roadmap.md`](ident_toolbox_roadmap.md) |
| Global Optimization (all 6 tiers shipped) | [`global_optim_toolbox_roadmap.md`](global_optim_toolbox_roadmap.md) |
| Statistics and Machine Learning (all 6 tier cores shipped) | [`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md) |
| Image Processing (all 6 tier cores shipped) | [`image_toolbox_roadmap.md`](image_toolbox_roadmap.md) |
| PDE | [`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md) |
| Symbolic Math | [`symbolic_toolbox_roadmap.md`](symbolic_toolbox_roadmap.md) |
| Fixed-Point Designer | [`fixed_point_toolbox_roadmap.md`](fixed_point_toolbox_roadmap.md) |
| Stateflow / mStateflow | [`mStateflow_roadmap.md`](mStateflow_roadmap.md) |
| Verilog-A export | [`verilog_a_plan.md`](verilog_a_plan.md) |
| Embedded Coder (mflowLink) | [`embedded_coder_roadmap.md`](embedded_coder_roadmap.md) |

---

## 10. Summary

**Where we are:** a production-quality MATLAB compiler + tooling stack
+ multi-toolbox numerical library covering the scalar / dense-matrix /
classdef / typed-int / fi / sym surface of MATLAB plus 13 shipped
toolbox subsets.

### Scale

- **~52,000-line C++ runtime** across **14 translation units**
  (`matlab_runtime.cpp` + 13 `runtime_*.cpp`), ~1,200 exported C-ABI
  entries, no BLAS / LAPACK dependency. Architecture documented in
  [`runtime.md`](runtime.md).
- **7 compiled / interpreted backends**:
  LLVM IR · portable C · portable C++ · Python (numpy shim) ·
  TypeScript (numpy_ts shim) · synthesizable SystemVerilog (ASIC,
  Verilator lint-clean) · Verilog-A (Tier-1 → Tier-10). Plus
  `-emit-matlab` and `-emit-mflow` source-to-source reverse-direction
  emitters, and **GPU kernel emit lanes** (`-emit-{cuda,metal,opencl}`)
  with CUDA / OpenCL device backends validated on NVIDIA hardware
  (issue #25) + Apple Metal (MPS).
- **435 `.m` execution tests** in `test/Run/`, each compiled and
  executed across **7 emit lanes** (~3,000 build-and-execute checks).
  **77 SystemVerilog golden fixtures** verilator-lint-clean. **39 HDL
  examples** verified bit-exact via cocotb. **55 Stateflow chart
  fixtures**.
- **25 direct C-ABI runtime tests** in `test/Runtime/` with **436
  test functions** covering every runtime TU. Regular build: **0.43 s
  wall**. Under `MATLAB_LLVM_RUNTIME_ASAN=ON` (AddressSanitizer +
  UndefinedBehaviorSanitizer): **2.82 s wall, 0 findings**.
- **0 compiler warnings** under the default `-Wall -Wextra
  -Wpedantic`. `-Wold-style-cast` + `-Werror=old-style-cast` enforced
  on the modernised toolbox TUs.

### Shipped capability

- **Three compiled backends with byte-identical stdout** (LLVM IR,
  portable C, portable C++) plus Python and TypeScript ports tracking
  the same surface across 435 fixtures.
- **JIT-backed REPL** (`matlabc -repl`) with persistent workspace,
  implicit display, operator-overloading / indexing / transpose
  auto-showing, `who` / `whos` / `clear`.
- **Language Server** (`matlab-lsp`): diagnostics, goto-definition,
  document outline. Accepts both `.m` and `.mflow` URIs.
- **DAP debugger** (`matlabc -dap`): line + conditional + log-point
  breakpoints, step-into-function with full frame stack, multi-frame
  variables inspection, `evaluate` against any frame, `setVariable`
  via the REPL JIT.
- **Source formatter** (`matlabc -format`): attribute-aware classdef
  output, idempotent round-trip.
- **OOP**: `classdef` with single inheritance, static methods,
  operator overloading, `Dependent` properties, enumerations,
  value-class copy-on-assign.
- **File I/O**: text + binary, plus a custom `save` / `load` format.
  Subset of `.mat` v5 is the next slice.
- **Linear algebra**: full LU / QR / Cholesky / pseudo-inverse / norm
  / trace / kron / non-symmetric `eig` (Hessenberg + Francis QR) /
  symmetric `eig` / SVD singular values / `expm` / `hess` / `schur` /
  Lyapunov / Riccati — all pure-C, no BLAS / LAPACK.
- **Heterogeneous data**: `containers.Map` / `dictionary`,
  2-D cell arrays, struct arrays (`s(i).x`), `datetime` / `duration`,
  `categorical`, `table`.
- **Typed integers + Fixed-Point Designer**: `int8`–`int64` /
  `uint8`–`uint64` matrix lanes with saturating arithmetic;
  `fi` Q-format scalar + 1-D array arithmetic with 5 rounding modes;
  `numerictype` + `fimath` first-class objects.

### Twenty-four shipped toolbox surfaces

Signal Processing · Control System · Communications · RF · Antenna ·
Propagation Models · Optimization · Model Predictive Control · **System
Identification** · Global Optimization · Statistics and Machine Learning ·
Image Processing · Curve Fitting · DSP System · Wavelet · Partial
Differential Equation · Symbolic Math (opt-in via SymPP) · Stateflow
(mStateflow) · Financial · Econometrics · Fixed-Point Designer · **Sensor
Fusion and Tracking** · **Robotics System** · **Navigation**. Plus headless
plotting (Cairo) and Verilog-A export.
See the per-toolbox roadmap docs in §9 for each toolbox's current
tier closure + open follow-ons.

### Biggest gaps to a "general-purpose MATLAB replacement"

- Narrower / wider int lanes (i8 / i16 / i64 / u16 / u32 / u64
  matrices — same template as the shipped i32 / u8 case)
- 3-D tensor surface (most elementwise / reduction ops reject 3-D
  inputs today; `A(:,:,k)` slicing follows from that)
- Complex linalg tail (full complex `inv` / `det` / `svd` / `eig`)
- `timetable` — ✅ shipped (Phase 5.4)
- Full method-dispatch value-class semantics
- MATLAB `.mat` v5 file-format compatibility
- `regexp` / `regexprep` and the string tail
- DAP user-function frames + watch-expression evaluation inside
  function bodies

### Biggest architectural asks

- **True N-D arrays** (>3D). Multi-week, structural rather than
  per-op work.
- **System-Object lowering fix**. Gates the `comm.*` / RF / Antenna /
  Propagation classdef wrappers on the SO surface; tracked in
  [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §11.1.
- **Sparse-matrix language surface**. Sparse CSR + Krylov solvers
  already exist in `runtime_sparse.cpp` for the PDE Toolbox; exposing
  them as a first-class `sparse(A)` MATLAB type is the open work.
- **Full N-D backend parity** across `-emit-python` / `-emit-typescript`
  (the C / C++ / LLVM lanes are byte-identical; Python is best-effort
  via numpy shim; TS least exercised).

Each is multi-week work and their priority depends on which direction
the project pushes next.
