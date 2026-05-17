# Symbolic Math Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Symbolic-Math-Toolbox programs.

Backed by [**SymPP**](https://github.com/leonardoaraujosantos/SymPP)
— a clean-room C++20 port of SymPy. The integration is opt-in:
build with `-DMATLAB_LLVM_WITH_SYM=ON`. Runtime entries live in
[`runtime/runtime_sym.cpp`](../runtime/runtime_sym.cpp) (~820 LOC,
92 exported entries) and dispatch through a distinct opaque
`matlab_sym` type (kind=7) + `matlab_symmat` (kind=8).

Source: *Symbolic Math Toolbox User's Guide* (R2026a), [`docs/sym.md`](sym.md)
(user reference), `test/RunSym/` (gating tests, 4 phases × 4 lanes).

Companion docs: [`feature_status.md`](feature_status.md) (compat
matrix), [`roadmap.md`](roadmap.md) (project-wide forward tracker),
[`ode.md`](ode.md) (numeric ODE solvers — symbolic `dsolve` is the
sibling family), [`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md)
(numeric PDE — symbolic `pdsolve_heat` / `pdsolve_wave` is the
analytic counterpart).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. Tier-1 is
  the foundational CAS: declare symbols, arithmetic dispatch, simplify
  / expand / factor / subs, single-eq `solve`, scalar `diff` / `int`,
  `vpa` / `double`. Tier-2 closes the calculus + transforms + ODE / PDE
  surface (`taylor` / `limit` / `dsolve` / `pdsolve` / `laplace` /
  `fourier` / `ztrans`). Tier-3 is symbolic linear algebra (sym
  matrices + det / inv / linsolve + multi-eq solvers). Tier-4 is
  assumption-driven refinement + numeric solvers (`nsolve` / `vpasolve`
  / IVP). Tier-5 wraps the runtime in MATLAB-API polish
  (`matlabFunction`, cell-array array-arg lowering, `even` / `odd` /
  `prime` assumption tail). Tier-6 is the multi-backend story
  (`-emit-python` via SymPy, `-emit-typescript` via mathjs / nerdamer).
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started ·
  🔴 deliberately deferred.
- **Tiers 1 / 2 / 3 / 4 are ✅ shipped** (commit arc `f57612e` → `8bcfae8`,
  4 phases). Tier-5 / 6 are the next slices — see §6 / §7.
- **REPL / Debug**: every sym builtin returns `matlab_sym *` or
  `matlab_symmat *`. The matlab_ws workspace stores them under
  kind=7 / kind=8 with cross-input persistence; the DAP variable
  renderer pretty-prints via SymPP's `to_string` (`b + 2*x*a`-style
  output). REPL JIT bridges the SymPP shared library via the opt-in
  CMake gate.
- **Compile/Execute path**: Sema registers a new builtin in
  `lib/Sema/Builtins.cpp` returning the sym kind; LowerTensorOps
  routes the call to `matlab_sym_*` or `matlab_symmat_*`; runtime
  unboxes through the `sympp::expression` value type. No MLIR op of
  its own — sym values flow as `llvm.ptr` through the existing scalar
  fast lane (`RefineSlotTypes::isScalarPrim` includes ptr so sym
  slots get Mem2Reg'd).

---

## 1. Already shipped (Tier-0 baseline, inherited from core)

These primitives sit underneath the symbolic surface:

| Group | Functions / capabilities | Notes |
|---|---|---|
| Scalar arithmetic dispatch | `+ - * / ^ == .* ./ .^` for sym op sym + sym op double | LowerTensorOps + LowerScalarsToArith path; mixed-mode uses `_d` variants without boxing the f64 literal |
| Workspace kind=7 / kind=8 | sym + symmat opaque types | Cross-input REPL persistence; DAP variable inspection via `matlab_dbg_render_*` |
| Char-array I/O for sym | `disp(s)`, `latex(s)`, `pretty(s)`, `ccode(s)` | Return char-array via `matlab_sym_*_string` accessors |
| MPFR / GMP | Arbitrary-precision arithmetic | Comes from SymPP transitively; `brew install gmp mpfr` on macOS |

---

## 2. Tier 1 — core CAS surface ✅ **shipped**

Foundational symbolic algebra: declare, manipulate, evaluate. Sized
to the *single-symbol scalar* case. Bigger than it looks because
this slice closes ~70% of typical MATLAB symbolic usage (textbook
calculus + algebra problems).

### 2.1 Symbol declaration ✅

```matlab
syms x y z                   % declare 3 fresh sym variables (workspace kind=7)
a = sym('a');                % single-symbol explicit constructor
e = sym('a^2 + 2*a + 1');    % parse-from-string
n = sym(42);                 % numeric → sym
n = sym(0.25);               % rational round-trip (1/4)
pi_sym = sym('pi');          % SymPP singleton (also: 'exp1', 'EulerGamma', 'Catalan', 'I')
e = str2sym('diff(sin(x),x)'); % bypass sym() escaping
```

Backed by `matlab_sym_named` / `matlab_sym_from_str` /
`matlab_sym_from_double` / `matlab_sym_from_i64` / `matlab_sym_str2sym`.

### 2.2 Arithmetic dispatch ✅

`+ - * / ^ == .* ./ .^` auto-route to `matlab_sym_*` when either
operand is sym. Mixed-mode `x^2`, `x + 1`, `2*x`, `x/3`, `1/x`,
`2^x` all go through `_d`-suffixed variants without boxing the
f64 literal (`matlab_sym_add_d`, `matlab_sym_d_sub`, etc.).

### 2.3 Algebra ✅

| Function | Runtime entry | Notes |
|---|---|---|
| `simplify(e)` | `matlab_sym_simplify` | Auto-chains `refine()` so assumptions propagate (Phase 6.2 ergonomics) |
| `expand(e)` | `matlab_sym_expand` | Polynomial expand |
| `factor(e, x)` | `matlab_sym_factor` | Univariate (var arg required) |
| `subs(e, old, new)` | `matlab_sym_subs` | Single substitution; cell-array substitution lists 🔵 (Tier-5) |

### 2.4 Single-equation `solve` ✅

```matlab
roots = solve(x^2 - 5*x + 6, x);    % [3, 2]
roots = solve(x^2 - 5*x + 6 == 0, x);
```

Multi-root output: `matlab_sym_solve(eq, var, &n_out)` returns
`matlab_sym **`; one `matlab_sym *` per root. The
multi-return ABI matches `[V, D] = eig`.

### 2.5 Numeric eval ✅

| Function | Runtime entry | Notes |
|---|---|---|
| `double(s)` | `matlab_sym_to_double` | Converts numeric sym → f64; throws on free symbols |
| `vpa(s, dps)` | `matlab_sym_vpa` | Arbitrary-precision via MPFR; default 32 digits |

### 2.6 Display / codegen ✅

| Function | Runtime entry | Output |
|---|---|---|
| `disp(s)` | `matlab_sym_disp` | SymPP `to_string` form (e.g. `b + 2*x*a`) |
| `latex(s)` | `matlab_sym_latex` | LaTeX source `\frac{...}{...}` |
| `pretty(s)` | `matlab_sym_pretty` | Multi-line ASCII art |
| `ccode(s)` | `matlab_sym_ccode` | C expression string |

### 2.7 Tier-1 closure summary

| Primitive | Effort | Status |
|---|---|---|
| `syms` / `sym(...)` / `str2sym` (2.1) | 0.5 wk | ✅ shipped |
| Arithmetic + comparison dispatch (2.2) | 0.5 wk | ✅ shipped |
| `simplify` / `expand` / `factor` / `subs` (2.3) | 0.5 wk | ✅ shipped |
| `solve` (single-eq, multi-root) (2.4) | 0.5 wk | ✅ shipped |
| `double` / `vpa` (2.5) | 0.5 wk | ✅ shipped |
| `disp` / `latex` / `pretty` / `ccode` (2.6) | 0.5 wk | ✅ shipped |
| **Closure test**: `test/RunSym/sym_phase_a.m` | — | ✅ shipped |

---

## 3. Tier 2 — calculus + transforms + ODE / PDE ✅ **shipped**

Tier-2 is what makes the toolbox useful beyond a CAS scratchpad —
real calculus on real functions, classical transform pairs, and
analytic ODE / PDE solving.

### 3.1 Calculus ✅

| Function | Runtime entry | Notes |
|---|---|---|
| `diff(f, x)`, `diff(f, x, n)` | `matlab_sym_diff` / `matlab_sym_diff_n` | n-th derivative |
| `int(f, x)` | `matlab_sym_int` | Indefinite |
| `int(f, x, a, b)` | `matlab_sym_int_def` | Definite |
| `taylor(f, x, a, n)` | `matlab_sym_taylor` | Series expansion around `a`, order `n` |
| `limit(f, x, target)` | `matlab_sym_limit` | Two-sided; ±∞ via `sym('inf')` |

### 3.2 Integral transforms ✅

| Pair | Runtime entries | Notes |
|---|---|---|
| Laplace | `matlab_sym_laplace` / `matlab_sym_ilaplace` | `f(t) ↔ F(s)` |
| Fourier | `matlab_sym_fourier` / `matlab_sym_ifourier` | `f(t) ↔ F(ω)` |
| Z-transform | `matlab_sym_ztrans` / `matlab_sym_iztrans` | `f[n] ↔ F(z)` |

### 3.3 Analytic ODE — `dsolve` ✅

```matlab
syms y yp x
disp(dsolve(yp + 2*y - x, y, yp, x))   % 1st-order linear
syms ypp
disp(dsolve(ypp + y, y, yp, ypp, x))   % 2nd-order (auto-classify)
```

| Form | Runtime entry | Notes |
|---|---|---|
| 1st-order `dsolve(eq, y, yp, x)` | `matlab_sym_dsolve` | Linear + separable |
| 2nd-order `dsolve(eq, y, yp, ypp, x)` | `matlab_sym_dsolve_2` | Auto-classifies const-coeff vs Cauchy-Euler |
| System `sym_dsolve_system(A, x)` | `matlab_symmat_dsolve_system` | First-order linear system `y' = A·y` via eigendecomp |

**Convention**: SymPP's `dsolve` takes the unknown function and
its derivatives as **plain symbols** (`y`, `yp`, `ypp`), not
MATLAB's AppliedFunction form `y(x)`. Lifting `diff(y(x), x)` →
the (y, yp, x) form is a Tier-5 ergonomics item (§6.2).

### 3.4 Analytic PDE — `pdsolve` ✅

| Form | Runtime entry | Notes |
|---|---|---|
| 1st-order linear `pdsolve(a, b, c, x, y)` | `matlab_sym_pdsolve` | `a·u_x + b·u_y = c` via method of characteristics |
| Heat eq `pdsolve_heat(k, lambda, x, t)` | `matlab_sym_pdsolve_heat` | `u_t = k·u_xx` separation-of-variables, single eigenmode |
| Wave eq `pdsolve_wave(c, x, t)` | `matlab_sym_pdsolve_wave` | `u_tt = c²·u_xx` d'Alembert form |

Multi-mode Fourier-series heat / wave is a Tier-5 item; numeric
PDE is its own toolbox ([`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md)).

### 3.5 Tier-2 closure summary

| Primitive | Effort | Status |
|---|---|---|
| `diff` / `int` indefinite + definite (3.1) | 0.5 wk | ✅ shipped |
| `taylor` / `limit` (3.1 tail) | 0.5 wk | ✅ shipped |
| Laplace / Fourier / Z + inverses (3.2) | 1 wk | ✅ shipped |
| `dsolve` 1st-order (3.3) | 0.5 wk | ✅ shipped |
| `dsolve` 2nd-order auto-classify (3.3) | 0.5 wk | ✅ shipped |
| `pdsolve` family (3.4) | 1 wk | ✅ shipped |
| **Closure test**: `test/RunSym/sym_phase_b.m` | — | ✅ shipped |

---

## 4. Tier 3 — symbolic linear algebra ✅ **shipped**

Sym matrices as a first-class workspace value: kind=8 opaque type
with cross-input REPL persistence, DAP rendering, and the full
arithmetic + linalg surface.

### 4.1 Sym matrix construction ✅

```matlab
syms a b
M = [a 1; 2 b];                      % standard literal — sym entries auto-detected
N = sym_matrix(2, 2, a, sym(1), sym(2), b);  % explicit row-major constructor
I = sym_eye(3);                      % 3x3 identity (sym)
Z = sym_zeros(2, 3);                 % zero matrix
```

**Phase 6.2 ergonomics**: standard `[a 1; 2 b]` matrix-literal syntax
detects sym entries at lowering time and routes through `matlab_symmat_*`
instead of the f64 matrix path. `sym_matrix(R, C, ...)` stays as
an explicit constructor for cases where the row-major flattening
needs to be resolved at compile time.

### 4.2 Sym matrix operations ✅

| Function | Runtime entry | Notes |
|---|---|---|
| `+` / `-` / `*` / scalar `*` | `matlab_symmat_add` / `_sub` / `_mul` / `_scalar_mul` | Matrix arithmetic |
| `'` (transpose) | `matlab_symmat_transpose` | Non-conjugate (sym has no complex notion natively) |
| `inv(M)` | `matlab_symmat_inverse` | Symbolic inverse via cofactor expansion |
| `det(M)` | `matlab_symmat_det` | Leibniz formula |
| `trace(M)` | `matlab_symmat_trace` | Sum of diagonal |
| `rank(M)` | `matlab_symmat_rank` | Symbolic rank |
| `eig(M)` | `matlab_symmat_eigenvals` | Returns variable-length `matlab_sym **`; multiplicity in solve order |
| `chol(M)` | `matlab_symmat_cholesky` | Lower-triangular factor; throws on non-PD |

### 4.3 Symbolic linsolve ✅

```matlab
A = [a 1; 0 b];
bv = [sym(1); sym(2)];
xs = sym_linsolve(A, bv);            % closed-form 2-vector
```

`matlab_symmat_linsolve(A, b)` returns a symmat with the symbolic
solution.

### 4.4 Multi-equation system solvers ✅

| Form | Runtime entry | Notes |
|---|---|---|
| Variadic `sym_solve_sys([eq1, eq2, …], [v1, v2, …])` | `matlab_sym_solve_sys` | Any-size N×N via LLVM stack-array ABI |
| Fixed `sym_solve_2x2(eq1, eq2, v1, v2)` | `matlab_sym_solve_2x2` | Shortcut |
| Fixed `sym_solve_3x3(eq1, eq2, eq3, v1, v2, v3)` | `matlab_sym_solve_3x3` | Shortcut |

All three return a symmat with **one row per joint solution**, so
`sols(1,:)` is one (x, y) pair, `sols(2,:)` is the next.

### 4.5 Tier-3 closure summary

| Primitive | Effort | Status |
|---|---|---|
| `[a 1; 2 b]` literal syntax detection (4.1) | 1 sess | ✅ shipped |
| `sym_matrix` / `sym_eye` / `sym_zeros` (4.1) | 2 sess | ✅ shipped |
| Matrix arithmetic + scalar mul + transpose (4.2) | 0.5 wk | ✅ shipped |
| `inv` / `det` / `trace` / `rank` (4.2) | 1 wk | ✅ shipped |
| `eig` / `chol` (4.2) | 3 sess | ✅ shipped |
| `linsolve` (4.3) | 3 sess | ✅ shipped |
| Variadic `sym_solve_sys` (4.4) | 3 sess | ✅ shipped (Phase 6.2 LLVM stack-array ABI) |
| Fixed-arity `_2x2` / `_3x3` shortcuts (4.4) | 1 sess | ✅ shipped |
| **Closure tests**: `sym_phase_b1.m` (matrices + multi-eq + IVP), `sym_phase_b2.m` (Phase 6.2 ergonomics) | — | ✅ shipped |

---

## 5. Tier 4 — assumptions + numeric solvers + IVP ✅ **shipped**

Refines the algebraic surface with **constraint-aware** rewriting
and adds Newton-style numeric solvers for cases where `solve`
returns no closed form.

### 5.1 Assumption framework ✅

```matlab
syms x
assume(x, 'positive');
disp(simplify(sqrt(x*x)));    % x   — Phase 6.2: simplify chains refine()
disp(simplify(sqrt((-x)*x))); % sqrt(-1)*x — sign preserved
```

10 properties supported in SymPP's mask:

```
real, rational, integer, positive, negative, zero,
nonzero, nonnegative, nonpositive, finite
```

Per-symbol; `assume` rebinds the variable to a fresh symbol carrying
the mask. `assumeAlso` ANDs into the existing mask. `clearAssumptions`
strips everything.

**MATLAB's `even`, `odd`, `prime`, `algebraic`, `complex` throw** —
not representable in SymPP's current mask (Tier-5 / SymPP-side phase).

### 5.2 Numeric solvers ✅

| Function | Runtime entry | Notes |
|---|---|---|
| `nsolve(eq, x, x0, dps)` | `matlab_sym_nsolve` | Newton's method in MPFR at `dps` digits |
| `vpasolve(eq, x, x0, dps)` | `matlab_sym_vpasolve` | MATLAB-API alias; same backend |

### 5.3 Initial-value ODE workflow ✅

```matlab
syms y yp x
general = dsolve(yp + 2*y - x, y, yp, x);
particular = apply_ivp(general, x, 0, 1);    % single-condition
checkodesol(yp + 2*y - x, particular, y, yp, x)  % residual

% Multi-condition (Phase 6.2):
syms y yp ypp
gen2 = dsolve(ypp + y, y, yp, ypp, x);
part2 = apply_ivp(gen2, x, [0, pi/2], [1, 0]);   % y(0)=1, y(pi/2)=0
```

| Function | Runtime entry | Notes |
|---|---|---|
| `dsolve_ivp(eq, y, yp, x, x0, y0)` | `matlab_sym_dsolve_ivp_1` (1-cond) / `matlab_sym_dsolve_ivp` (multi) | Compose `dsolve` + `apply_ivp` in one call |
| `apply_ivp(general, x, x0, y0)` | `matlab_sym_apply_ivp_1` / `matlab_sym_apply_ivp` | Plug initial conditions into a general solution |
| `checkodesol(eq, sol, y, yp, x)` | `matlab_sym_checkodesol` | Returns residual sym; should simplify to 0 |

### 5.4 Tier-4 closure summary

| Primitive | Effort | Status |
|---|---|---|
| `assume` / `assumeAlso` / `clearAssumptions` (5.1) | 0.5 wk | ✅ shipped (10 properties) |
| `simplify` chains `refine()` (5.1 ergonomics) | 1 sess | ✅ shipped (Phase 6.2) |
| `nsolve` / `vpasolve` (5.2) | 0.5 wk | ✅ shipped |
| `apply_ivp` single-condition (5.3) | 1 sess | ✅ shipped |
| `apply_ivp` multi-condition (5.3) | 1 sess | ✅ shipped (Phase 6.2 — parallel sym vectors) |
| `dsolve_ivp` composed (5.3) | 1 sess | ✅ shipped |
| `checkodesol` (5.3) | 1 sess | ✅ shipped |
| **Closure test**: `test/RunSym/sym_phase_b2.m` | — | ✅ shipped |

---

## 6. Tier 5 — MATLAB-API polish 🟡 **partial / next slice**

The functional surface is done; what's left is the ergonomics MATLAB
users expect — function handles back from `matlabFunction`, cell-array
substitution lists, the AppliedFunction lifting pass, etc.

### 6.1 `matlabFunction(f, vars)` returning a function handle 🟡

```matlab
syms x y
f = x^2 + 3*y;
h = matlabFunction(f, [x, y]);
disp(h(2, 4))    % 16
```

SymPP currently emits Octave source as a string; the matlab_llvm side
doesn't yet parse-and-bind that into a callable function handle.

**Plan**:
1. Add `matlab_sym_matlabFunction_octave(f, vars)` returning the Octave
   source string (already shipped — just exposes the SymPP builtin).
2. Wrap the string in a `function_handle` descriptor (kind=4): a
   light reparse via the existing MATLAB parser binds the body, and
   the handle stores both the source and a JIT'd `LowerAnonCalls`
   trampoline.
3. Numeric coercion of args via `matlab_sym_subs_d` + `matlab_sym_to_double`.

**Effort**: 1 wk (parser reuse is straightforward; the trampoline
caching is the bulk of the work).

**Status**: 🔵 Tier-5 next slice.

### 6.2 AppliedFunction lifting pass 🔵

MATLAB writes `dsolve(diff(y, x) + y == 0, y)`; SymPP wants
`dsolve(yp + y, y, yp, x)`. A Sema-level lifting pass would:

1. Recognise `diff(y, x)` / `diff(y, x, n)` in an `eq` arg of `dsolve`.
2. Auto-introduce shadow symbols `yp`, `ypp`, `…` for each derivative
   order found.
3. Rewrite the call to the matlab_llvm convention.

**Effort**: 0.5 wk. Status: 🔵 Tier-5.

### 6.3 Cell-array array-arg lowering 🔵

These runtime entries ship in `runtime_sym.cpp` but the MATLAB
cell-array language lowering isn't wired:

| Function | Runtime entry exists | Why blocked |
|---|---|---|
| `rsolve(coeffs, n)` recurrence | `matlab_sym_rsolve` | Variadic `const matlab_sym *const *` ABI; needs cell-array → C array lowering |
| `groebner([eq1, eq2, …], [v1, v2, …])` | `matlab_sym_groebner` | Same ABI |
| `pythagorean_triples(max_z)` | `matlab_sym_pythagorean_triples` | Returns variable-length `matlab_sym **`; needs multi-return for arrays |
| `linear_diophantine(a, b, c)` | (not in runtime yet) | SymPP supports it |

**Effort**: 0.5 wk for the lowering pattern, then ~1 session per
builtin. Status: 🔵 Tier-5.

### 6.4 Substitution + simplification tail 🔵

| Function | MATLAB ref | Notes |
|---|---|---|
| `subs(e, {old1, old2}, {new1, new2})` | cell-array form | Plug list of substitutions in one call (Sema lowers to a chain of single `subs`) |
| `combine(e)` | combine-like-terms | SymPP has `simplify`; MATLAB's `combine(e, 'log')` / `'exp'` / `'sincos'` directives need explicit category dispatch |
| `rewrite(e, target)` | rewrite in terms of `'sincos'` / `'exp'` / `'sqrt'` | Wraps SymPP's `Expression::rewrite` (if exposed) |
| `collect(e, x)` | collect powers of `x` | SymPP `collect` |
| `horner(e)` | Horner-form polynomial | Re-bracket for evaluation efficiency |
| `numden(e)` | `[num, den] = numden(e)` | Multi-return; numerator/denominator extraction |
| `partfrac(e, x)` | partial-fraction decomposition | Distinct-pole only matches existing `residue` shape |

**Effort**: 1 wk. Status: 🔵 Tier-5.

### 6.5 Extended assumption properties 🔵

MATLAB's `even`, `odd`, `prime`, `algebraic`, `complex` throw today —
not in SymPP's 10-property mask.

**Plan**: SymPP-side phase to extend the mask. Even / odd are simple
parity bits; prime needs lazy evaluation; algebraic vs. transcendental
is a tag rather than a check. Complex flips the assumption semantics
so `simplify(conj(x))` doesn't fold to `x`.

**Effort**: 1 wk (SymPP-side), 1 sess (matlab_llvm-side wiring).
Status: 🔵 Tier-5 / SymPP-side phase.

### 6.6 Tier-5 closure summary

| Primitive | Effort | Status |
|---|---|---|
| `matlabFunction` returning function handle (6.1) | 1 wk | 🔵 — biggest ergonomics gap |
| AppliedFunction lifting `diff(y, x)` → `(y, yp, x)` (6.2) | 0.5 wk | 🔵 |
| Cell-array array-arg lowering: `rsolve` / `groebner` / `pythagorean_triples` (6.3) | 1 wk | 🔵 — runtime present |
| Substitution + simplification tail (6.4) | 1 wk | 🔵 — `subs` cell-form, `combine`, `rewrite`, `collect`, `horner`, `numden`, `partfrac` |
| Extended assumption properties (6.5) | 1 wk SymPP + 1 sess wiring | 🔵 — gated on SymPP-side mask extension |
| Closure test: `sym_phase_c.m` (Tier-5 features) | — | 🔵 |

**Total Tier-5**: ~4.5 weeks of focused sessions.

---

## 7. Tier 6 — multi-backend emission 🔵 **not started**

Today `-emit-cpp` / `-emit-llvm` work end-to-end (SymPP links into
the emitted binary). `-emit-c` emits valid C that must be compiled
as C++ to pull SymPP. `-emit-python`, `-emit-typescript`,
`-emit-systemverilog` all diagnose unsupported sym usage at emit
time. Tier-6 is the cross-backend story.

### 7.1 `-emit-python` via SymPy 🔵

SymPP's design is a clean-room port of SymPy, so the call-shape
mapping is nearly 1:1:

```python
from sympy import Symbol, symbols, diff, integrate, simplify, expand
from sympy import solve, dsolve, Function, Eq, laplace_transform
```

| matlab_llvm sym entry | SymPy equivalent |
|---|---|
| `matlab_sym_diff(f, x)` | `diff(f, x)` |
| `matlab_sym_int(f, x)` | `integrate(f, x)` |
| `matlab_sym_int_def(f, x, a, b)` | `integrate(f, (x, a, b))` |
| `matlab_sym_simplify(e)` | `simplify(e)` |
| `matlab_sym_solve(eq, var)` | `solve(eq, var)` |
| `matlab_sym_dsolve(eq, y, yp, x)` | `dsolve(Eq(yp + ..., 0), Function('y')(x))` — needs AppliedFunction reconstruction |
| `matlab_sym_laplace(f, t, s)` | `laplace_transform(f, t, s)` |
| `matlab_symmat_det(M)` | `Matrix(M).det()` |

**Plan**:
1. Add a `lib/MIR/EmitPythonSym.cpp` pass that walks the AST and
   emits SymPy calls for the sym-typed expressions.
2. Re-route the dispatched arithmetic (`matlab_sym_add` etc.) into
   plain Python operators since SymPy's `Symbol` overloads `+ - * / **`.
3. AppliedFunction reconstruction: when emitting `dsolve`, regenerate
   `Function('y')(x)` from the (y, yp, x) shape — inverse of §6.2.
4. Numpy interop for `double(s)` → `float(s)`, `vpa` → `s.evalf(dps)`.

**Effort**: 2 wk. Status: 🔵 Tier-6.

### 7.2 `-emit-typescript` via mathjs or nerdamer 🔵

[mathjs](https://mathjs.org/) has a symbolic algebra surface
(`derivative`, `simplify`, `parse`, `evaluate`) but no `dsolve` /
`pdsolve`. [nerdamer](https://nerdamer.com/) has wider coverage
including `solve` / `dsolve` but is JS-only (no TS types).

**Plan**: target mathjs for the Tier-1+2 algebraic surface; emit a
clean diagnostic for `dsolve` / `pdsolve` / transforms that mathjs
doesn't cover. The TS lane is the least-exercised backend overall
so this is a lower priority.

**Effort**: 1.5 wk. Status: 🔵 Tier-6 (low priority).

### 7.3 `-emit-systemverilog` 🔴 **deliberately deferred**

Symbolic computation is not synthesizable to hardware. Emit a clear
diagnostic and direct users to `double(s)` / `matlabFunction` for
the numeric subset they need. Status: 🔴.

### 7.4 Tier-6 closure summary

| Backend | Effort | Status |
|---|---|---|
| `-emit-python` via SymPy (7.1) | 2 wk | 🔵 |
| `-emit-typescript` via mathjs (7.2) | 1.5 wk | 🔵 (lower priority) |
| `-emit-systemverilog` (7.3) | — | 🔴 deferred (not synthesizable) |

---

## 8. Tier 7+ — Long-tail features 🔵 **not started**

Less-common but documented MATLAB symbolic surface that hasn't
been needed yet:

| Function | What it does | Effort | Notes |
|---|---|---|---|
| `symfun` — symbolic functions `f(x, y) = x^2 + y^2` | Function-symbol with explicit args | 0.5 wk | Currently approximated with plain `sym` + `subs` |
| `argnames(f)` | Inspect symfun args | 1 sess | Trivial wrapper |
| `formula(f)` | Extract body of a symfun | 1 sess | Trivial wrapper |
| `piecewise(cond1, val1, cond2, val2, ...)` | Piecewise expression | 0.5 wk | SymPP has `Piecewise`; needs cell-array language wiring (Tier-5 §6.3 dependency) |
| `dirac(x)` / `heaviside(x)` | Distributions | 1 sess | SymPP has both; just wire them |
| `besselj(n, x)` / `besseli(n, x)` / `bessely(n, x)` / `besselk(n, x)` | Bessel functions of symbolic args | 0.5 wk | SymPP has them; reuse the elementary-functions dispatch |
| `gamma(x)` / `factorial(x)` symbolic | Special functions | 0.5 wk | SymPP `Gamma` / `factorial` |
| `zeta(x)` / `polylog(s, z)` / `lerchphi(z, s, a)` | Number-theory specials | 1 wk | Currently throw; less-common requests |
| `int(f, [x, a, b], 'PrincipalValue', true)` | Cauchy principal value | 3 sess | SymPP's `Integrate` has it; needs name-value pair lowering |
| `simplify(e, 'Steps', N)` | Bounded simplification depth | 3 sess | Name-value pair lowering |
| `chebyshevT(n, x)` / `legendreP(n, x)` / `hermiteH(n, x)` / etc. | Orthogonal polynomials | 0.5 wk | SymPP has them |
| `solve(eq, var, 'Real', true)` | Filter to real-valued roots | 1 sess | Use the assumption framework — assume var real, then solve |
| `vpa(symmat, dps)` | Element-wise vpa on a sym matrix | 1 sess | Pure dispatch |
| `[r, p] = numden(e)` | Numerator + denominator multi-return | 1 sess | Reuse the `[V, D] = eig` multi-return ABI |
| `coeffs(e, x)` | Polynomial coefficients (multi-return: `[c, t]`) | 0.5 wk | SymPP has `Poly.coeffs` |
| `degree(p, x)` / `poly2sym(c, x)` / `sym2poly(p)` | Polynomial bridge to numeric | 0.5 wk | Wrappers |
| `bernoulli(n)` / `euler(n)` / `harmonic(n)` | Number-theory functions | 3 sess | SymPP has them |

**Total Tier-7+**: ~6-8 weeks if all-in. Most slices are 1-3 sessions
each and don't form a natural arc — pick what real users ask for.

---

## 9. Carve-outs (deliberately not in scope)

- **Live Editor integration**: `symvar` browser, inline LaTeX
  rendering, Symbolic Math Live Editor task — all GUI / Live Editor
  surface, out of scope per the project's headless-only policy.
- **MuPAD compatibility**: `evalin(symengine, ...)`, MuPAD notebooks,
  the MuPAD source language — MathWorks deprecated MuPAD in R2019b,
  and SymPP is a SymPy port, not a MuPAD port. Migration is one-way.
- **`mupadmex` / pkg-config-style MuPAD packages**: same reason.
- **SymPy → Symbolic Math Live Editor co-execution**: depends on the
  Live Editor surface. Out of scope.
- **`fplot` / `fmesh` / `fsurf` symbolic plotting**: the plot runtime
  (Cairo backend, [`plotting.md`](plotting.md)) needs sym-arg
  dispatch — small wiring, but blocked on the existing plotting
  roadmap not symbolic. Likely Tier-7 follow-on once a user asks.
- **MATLAB-encrypted P-files containing symbolic functions**: P-files
  are out of scope project-wide.
- **`finverse(f, x)`** functional inverse: SymPP doesn't have it
  as a primitive; would need `solve(y == f(x), x)`-based wrapper.
  Trivial when needed.
- **GPU-accelerated symbolic**: not in any MATLAB toolbox version;
  out of scope.

---

## 10. Execution order — if user demand drives prioritization

Each row unblocks the next:

| Order | What | Effort | Status |
|---|---|---|---|
| 1–7 | **Tier-1 core CAS** (declare / arithmetic / simplify / solve / diff / int / vpa / display) | ~3 wk total | ✅ shipped |
| 8–14 | **Tier-2 calculus + transforms + ODE/PDE** (taylor / limit / Laplace / Fourier / Z / dsolve / pdsolve) | ~4 wk total | ✅ shipped |
| 15–22 | **Tier-3 sym matrices** (literal syntax + arithmetic + linalg + multi-eq solvers) | ~3 wk total | ✅ shipped |
| 23–28 | **Tier-4 assumptions + numeric solvers + IVP** | ~2.5 wk total | ✅ shipped |
| 29 | `matlabFunction(f, vars)` → function handle (Tier-5 §6.1) | 1 wk | 🔵 — biggest ergonomics win |
| 30 | AppliedFunction lifting pass (Tier-5 §6.2) | 0.5 wk | 🔵 — closes the `diff(y(x), x)` MATLAB syntax gap |
| 31 | Cell-array array-arg lowering (Tier-5 §6.3) | 1 wk | 🔵 — unblocks `rsolve` / `groebner` / `pythagorean_triples` |
| 32 | Substitution + simplification tail (Tier-5 §6.4) | 1 wk | 🔵 — `subs` cell-form, `combine`, `rewrite`, `collect`, `horner`, `numden`, `partfrac` |
| 33 | Extended assumption properties (Tier-5 §6.5) | 1 wk SymPP-side | 🔵 — `even` / `odd` / `prime` / `algebraic` / `complex` |
| 34 | `-emit-python` via SymPy (Tier-6 §7.1) | 2 wk | 🔵 — biggest backend gap |
| 35 | `symfun` + `piecewise` + special functions (Tier-7) | 2 wk | 🔵 — long-tail; pick as user demand surfaces |
| 36 | `-emit-typescript` via mathjs (Tier-6 §7.2) | 1.5 wk | 🔵 — low priority |

**Tier-5 closure**: ~4.5 weeks. Lights up `matlabFunction(f, vars)`
end-to-end and closes the AppliedFunction syntax gap, which makes
the symbolic toolbox feel native instead of "SymPP-flavored".

**Tier-6 closure**: ~3.5 weeks. Adds Python (high-value) and TS
(low-priority) emit paths so the SymPP dependency isn't a hard
build-time requirement for the end-user.

---

## 11. Gating tests

| Lane | Tests | What it gates |
|---|---|---|
| `run-tests-sym` (`-emit-cpp` + SymPP) | 4 | All four phase fixtures — Tier-1 / Tier-2 / Tier-3 / Tier-4 in one CI lane |
| `test/RunSym/sym_phase_a.m` | 1 | Tier-1 core CAS (declare / arithmetic / simplify / solve / diff / int / vpa / disp) |
| `test/RunSym/sym_phase_b.m` | 1 | Tier-2 calculus + transforms + ODE/PDE |
| `test/RunSym/sym_phase_b1.m` | 1 | Tier-3 sym matrices + multi-eq + IVP |
| `test/RunSym/sym_phase_b2.m` | 1 | Tier-4 assumptions + Phase 6.2 ergonomics (literal syntax + variadic + multi-cond IVP + simplify-refine + singletons) |

Opt-in via `-DMATLAB_LLVM_WITH_SYM=ON`; skips with rc=77 when SymPP
isn't found at the configured prefix.

A Tier-5 closure test `sym_phase_c.m` is the next slice — exercises
`matlabFunction` end-to-end, cell-array `subs`, `rsolve` / `groebner`
multi-return, and the AppliedFunction lifting pass.

---

## 12. Internal references

- Runtime: [`runtime/runtime_sym.cpp`](../runtime/runtime_sym.cpp)
  (~820 LOC, 92 exported entries), [`runtime/runtime_sym.h`](../runtime/runtime_sym.h)
- Frontend: builtins registered in `lib/Sema/Builtins.cpp` under
  the `sym_*` and `matlab_sym*` / `matlab_symmat*` groups
- Lowering: `lib/MLIR/LowerTensorOps.cpp` routes the dispatched
  builtins to the `matlab_sym_*` ABI
- Display: `runtime/runtime_debug.cpp` ships the DAP / REPL
  renderer for kind=7 / kind=8 workspace values
- User reference: [`docs/sym.md`](sym.md)
- Project-wide roadmap: [`docs/roadmap.md`](roadmap.md) §6 / §6.1 /
  §6.2 (shipped); §6.3 (next slice)
- Companion toolbox plans: [`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md)
  (numeric counterpart to `pdsolve`), [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md)
  (sym surfaces in §3.4 cepstrum / `freqz` symbolic — Tier-7+),
  [`control_toolbox_roadmap.md`](control_toolbox_roadmap.md)
  (sym in `tf` / `ss` operations — Tier-7+ follow-on)
