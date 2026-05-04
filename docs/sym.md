# Symbolic Math Toolbox

`matlab_llvm` ships a MATLAB-compatible Symbolic Math Toolbox front-end
backed by [SymPP](https://github.com/leonardoaraujosantos/SymPP) — a
clean-room C++20 port of SymPy. The integration is opt-in: build with
`-DMATLAB_LLVM_WITH_SYM=ON` to pull SymPP in.

## Quick start

```matlab
syms x
syms a b c
f = a*x^2 + b*x + c;
disp(f)                     % c + x*b + a*x**2
disp(diff(f, x))            % b + 2*x*a
disp(int(x^2, x))           % 1/3*x**3
disp(expand((x+1)^3))       % 3*x**2 + 3*x + x**3 + 1
disp(factor(x^2 - 1, x))    % (x - 1)*(x + 1)
disp(solve(x^2 - 5*x + 6, x))    % [3, 2]
disp(subs(x + 1, x, 2))     % 3
disp(double(subs(x+1, x, 2)))    % 3
```

## Build

```bash
# Build SymPP first.
git clone https://github.com/leonardoaraujosantos/SymPP
cd SymPP
cmake -S . -B build
cmake --build build
cmake --install build --prefix /tmp/sympp_install   # any prefix works

# Now build matlab_llvm with sym enabled.
cd /path/to/matlab_llvm
cmake -S . -B build-sym -DMATLAB_LLVM_WITH_SYM=ON
cmake --build build-sym
```

CMake auto-detects SymPP at `/tmp/sympp_install`, `/opt/homebrew/lib/cmake/SymPP`,
and `<SymPP>/build`. Set `-DSymPP_DIR=...` to point elsewhere.

GMP and MPFR (SymPP's deps) need to be on the system: `brew install gmp mpfr`
on macOS, `apt install libgmp-dev libmpfr-dev` on Debian.

## Function surface

Mirrors MATLAB's Symbolic Math Toolbox User's Guide (R2026a).

| Group | Functions |
|---|---|
| Declaration | `syms x y z`, `sym(name)`, `sym(expr_string)`, `sym(numeric)`, `str2sym('expr')` |
| Calculus | `diff(f, x)`, `diff(f, x, n)`, `int(f, x)`, `int(f, x, a, b)`, `taylor(f, x, a, n)`, `limit(f, x, target)` |
| Algebra | `simplify`, `expand`, `factor(e, x)`, `subs(e, old, new)` |
| Single-eq solvers | `solve(eq, x)`, `solve(f == 0, x)`, `vpasolve(eq, x, x0, dps)`, `nsolve(eq, x, x0, dps)` |
| Multi-eq solvers | `sym_solve_2x2(eq1, eq2, var1, var2)`, `sym_solve_3x3(eq1, eq2, eq3, var1, var2, var3)` — return symmat (one row per joint solution) |
| Numeric eval | `double(s)`, `vpa(s, dps)` |
| ODE / PDE | `dsolve(eq, y, yp, x)` (1st-order), `dsolve(eq, y, yp, ypp, x)` (2nd-order auto-classify), `dsolve_ivp(eq, y, yp, x, x0, y0)`, `apply_ivp(general, x, x0, y0)`, `checkodesol(eq, sol, y, yp, x)`, `pdsolve(a, b, c, x, y)`, `pdsolve_heat(k, lambda, x, t)`, `pdsolve_wave(c, x, t)` |
| Transforms | `laplace(f, t, s)`, `ilaplace(F, s, t)`, `fourier(f, t, w)`, `ifourier(F, w, t)`, `ztrans(f, n, z)`, `iztrans(F, z, n)` |
| Assumptions | `assume(x, 'positive')`, `assumeAlso(x, 'integer')`, `clearAssumptions(x)` |
| Symbolic matrices | Standard `[a 1; 2 b]` matrix literal (sym entries auto-detected at lowering time, routes through `matlab_symmat_*`), `sym_matrix(R, C, e11, e12, …, eRC)` (R, C must be integer literals), `sym_eye(n)`, `sym_zeros(R, C)`, `sym_det(M)`, `sym_inv(M)`, `sym_transpose(M)`, `sym_trace(M)`, `sym_rank(M)`, `sym_linsolve(A, b)`, `sym_dsolve_system(A, x)` |
| Multi-eq variadic solver | `sym_solve_sys([eq1, eq2, …], [v1, v2, …])` — N-equation, N-unknown, lowered through an `llvm.alloca`+GEP+store stack-array ABI |
| Display / codegen | `disp(s)`, `latex(s)`, `pretty(s)`, `ccode(s)`, `matlabFunction(...)` |
| Elementary | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs` (all dispatch to sym variants when the argument is sym) |

Operator dispatch: `+ - * / ^ == .* ./ .^` route to `matlab_sym_*`
when either operand is sym. Mixed-mode (`x^2`, `x + 1`, etc.) goes
through `_d` variants without boxing the f64 literal.

## Backend support

| Backend | Status |
|---|---|
| `-emit-cpp`, `-emit-llvm` | ✅ End-to-end (link against SymPP at compile time of the emitted source) |
| `-emit-c` | ✅ Emits valid C; the user must compile/link the emitted source as C++ to pull in SymPP |
| REPL JIT (`-repl`) | ✅ Full sym support including cross-input persistence (workspace kind=7) |
| DAP debugger (`-dap`) | ✅ Sym variables render via SymPP's pretty form in the variables panel |
| `-emit-python` | ❌ Diagnoses at emit time — would route through SymPy but the emitter doesn't translate the call shape yet |
| `-emit-typescript` | ❌ Diagnoses at emit time — no JS-side symbolic library |
| `-emit-systemverilog` | ❌ Diagnoses at emit time — symbolic computation is not synthesizable |

## Assumptions

`assume` registers in SymPP's side-table and rebinds the variable to a
fresh symbol carrying the mask. The mask is honoured by `refine`; the
default `simplify` does **not** auto-traverse assumptions (matching SymPy
behaviour). Supported property strings:

```
real, rational, integer, positive, negative, zero,
nonzero, nonnegative, nonpositive, finite
```

Anything else throws `std::runtime_error`. MATLAB's `even`, `odd`, `prime`,
`algebraic`, `complex` are not yet representable in SymPP's mask.

## ODE convention

SymPP's `dsolve` does not use the AppliedFunction `y(x)` shape MATLAB writes;
instead, the unknown function and its derivatives are passed as **plain
symbols**:

```matlab
syms y yp ypp x
% MATLAB's `y'' + y == 0`:
disp(dsolve(ypp + y, y, yp, ypp, x))
```

This matches the SymPP facade convention. A future phase may add a parser
pass that lifts MATLAB's `diff(y(x), x)` syntax into the (y, yp, x) form.

## Tests

`test/RunSym/` contains end-to-end fixtures driven by `test/RunSym/run_tests.sh`.
The CTest target `run-tests-sym` is gated on `MATLAB_LLVM_WITH_SYM=ON` and
skips with rc=77 when SymPP isn't found at the configured prefix.

## Symbolic matrix usage

```matlab
syms a b
M = sym_matrix(2, 2, a, sym(1), sym(2), b);   % row-major literal entries
disp(M)            % Matrix([[a, 1], [2, b]])
disp(sym_det(M))   % b*a - 2

A = sym_matrix(2, 2, sym(1), sym(2), sym(3), sym(4));
bv = sym_matrix(2, 1, sym(1), sym(2));
xs = sym_linsolve(A, bv);    % Matrix([[0], [1/2]])

% Multi-equation system:
syms x y
sols = sym_solve_2x2(x^2 + y^2 - 1, y - x, x, y);
% Returns a 2x2 symmat — each row is a joint solution (x, y).
```

The `sym_matrix(R, C, e11, e12, ..., eRC)` constructor takes integer
literal R, C (so the row-major flattening is resolved at compile time)
followed by R×C scalar sym entries. Standard `[a 1; 2 b]` matrix literal
syntax doesn't yet detect sym entries — extending matrix-literal lowering
to route sym entries through `matlab_symmat_*` is Phase 6.2.

## Phase 6.2 — what shipped

- **Standard `[a 1; 2 b]` matrix literal syntax** — sym entries detected at
  lowering time; routes through `matlab_symmat_*` instead of the f64 matrix
  path. `sym_matrix(R, C, ...)` is no longer required for the common case.
- **Variadic `sym_solve_sys`** — `sym_solve_sys([eq1, eq2, eq3], [x, y, z])`
  for systems of any size. Lowers to LLVM stack arrays of sym pointers
  passed to the existing variadic runtime entry. The fixed-arity
  `sym_solve_2x2` / `sym_solve_3x3` remain as shortcuts.
- **Multi-condition `dsolve_ivp` / `apply_ivp`** — pass parallel arrays:
  `apply_ivp(sol, x, [x0, x1, ...], [y0, y1, ...])`. Single-condition form
  `apply_ivp(sol, x, x0, y0)` still works as before.
- **`simplify` auto-honours assumptions** — chains `refine()` before the
  structural simplify so masked symbols' properties propagate. After
  `assume(y, 'positive')`, `simplify(sqrt(y*y))` → `y`.
- **`sym('pi')` → Pi singleton** — recognises `pi` / `Pi` / `exp1` /
  `EulerGamma` / `Catalan` / `I` / `true` / `false` as SymPP singletons.
  `vpa(sym('pi'), 32)` returns the digits of π.

## What's not in scope yet (Phase 6.3+)

- **`matlabFunction(f, vars)` returning a callable handle** — SymPP returns
  Octave source as a string; matlab_llvm doesn't yet parse-and-bind that
- **Assumption properties beyond the 10 in SymPP's mask** — `even`, `odd`,
  `prime`, `algebraic`, `complex` throw; SymPP-side phase
- **Array-arg builtins**: `rsolve`, `groebner`, `pythagorean_triples`,
  `linear_diophantine` ship in the runtime but the cell-array language
  lowering for them is not yet wired
- **SV emitter `state_display`-style output drop** (introduced in
  3622f10): the const-fold pass eliminates assignments to output-port
  slots when the value comes from a persistent register read. Pre-Phase
  6.2 attempts to fix it surface as a cocotb timing divergence between
  the Python and SV references — needs lockstep pipeline-equivalence
  work in a Phase 7 effort.

