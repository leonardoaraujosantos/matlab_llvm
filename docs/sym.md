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
| Solvers | `solve(eq, x)` (single eq, single var), `solve(f == 0, x)` |
| Numeric eval | `double(s)`, `vpa(s, dps)` |
| ODE / PDE | `dsolve(eq, y, yp, x)` (1st-order), `dsolve(eq, y, yp, ypp, x)` (2nd-order auto-classify), `pdsolve(a, b, c, x, y)`, `pdsolve_heat(k, lambda, x, t)`, `pdsolve_wave(c, x, t)` |
| Transforms | `laplace(f, t, s)`, `ilaplace(F, s, t)`, `fourier(f, t, w)`, `ifourier(F, w, t)`, `ztrans(f, n, z)`, `iztrans(F, z, n)` |
| Assumptions | `assume(x, 'positive')`, `assumeAlso(x, 'integer')`, `clearAssumptions(x)` |
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

## What's not in scope yet

- Symbolic matrices and `linsolve` of a symbolic matrix
- Multi-equation `solve([eq1, eq2], [x, y])` returning a real solution vector
  (currently flattens to a string-rendered single sym)
- `matlabFunction` returning a callable handle (the SymPP facade returns
  emitted Octave source as a string; matlab_llvm doesn't yet wrap that into
  a function handle)
- `simplify` honouring assumptions automatically (use `refine` explicitly
  on SymPP's C++ side)
- Symbolic constant resolution: `sym('pi')` creates a Symbol named "pi" rather
  than the Pi singleton — workaround: `sym(pi)` (boxes the f64 constant)
