# Symbolic Math Toolbox (SymPP) — Tutorial

The Symbolic Math lane is backed by **SymPP**, a clean-room C++20 port of SymPy.
It gives you a real CAS inside compiled MATLAB: declare symbols with `syms`,
differentiate, integrate, solve, simplify, take limits and transforms, and do
symbolic linear algebra. Symbolic values flow as opaque `matlab_sym` /
`matlab_symmat` pointers through the existing scalar lane, so they compose with
the rest of the language. The integration is **opt-in** at build time.

## Supported features

- **Declaration**: `syms`, `sym`, `str2sym`.
- **Arithmetic**: `+ - * / ^` on sym (pure and mixed-mode with `double` literals).
- **Calculus**: `diff` (1st and nth order), `int` (indefinite and definite),
  `taylor`, `limit`.
- **Algebra**: `simplify`, `expand`, `factor`, `subs`.
- **Solve**: `solve(eq, x)`, `solve(f == 0, x)` (relational form),
  `vpasolve` / `nsolve` (Newton at variable precision via MPFR).
- **Numeric eval**: `double(s)`, `vpa(s, dps)`.
- **ODE / PDE**: `dsolve` (1st + 2nd order), `dsolve_ivp`, `checkodesol`,
  `pdsolve_heat`, `pdsolve_wave`.
- **Transforms**: `laplace` / `ilaplace`, `fourier` / `ifourier`,
  `ztrans` / `iztrans`.
- **Assumptions**: `assume`, `assumeAlso`, `clearAssumptions`.
- **Symbolic matrices**: `sym_matrix`, `sym_eye`, `sym_det`, `sym_inv`,
  `sym_linsolve`, `sym_dsolve_system`, `sym_solve_2x2`.

## Build & run

The symbolic runtime lives behind an opt-in CMake gate — build with
`-DMATLAB_LLVM_WITH_SYM=ON`. The SymPP shared library (`libmatlab_sym.dylib`) is
`dlopen`'d lazily at runtime, so the link line just needs `-ldl`.

```bash
build/matlabc -emit-llvm examples/symbolic_demo.m > /tmp/symbolic_demo.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/symbolic_demo.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/symbolic_demo
/tmp/symbolic_demo
```

It also runs through `-emit-cpp` or the JIT REPL. In the REPL/DAP, sym values
pretty-print via SymPP's `to_string` (e.g. `b + 2*x*a`).

## Worked examples

All excerpts below are from the single tour file `examples/symbolic_demo.m`.

### Declaration & calculus

```matlab
syms x
syms a b c
f = a*x^2 + b*x + c;

disp(diff(f, x))           % 2*a*x + b   — first derivative wrt x
disp(diff(f, x, 2))        % 2*a         — second derivative
disp(int(x^3, x))          % x^4/4       — indefinite integral
disp(int(x^2, x, sym(0), sym(1)))   % 1/3 — definite integral on [0,1]
disp(taylor(sin(x), x, sym(0), 5))  % degree-5 Taylor about x = 0
disp(limit(sin(x)/x, x, sym(0)))    % 1   — the classic limit
```

`diff` takes an optional order argument; `int` overloads to definite when given
bounds (as `sym` values). Note bounds and substitution targets are wrapped in
`sym(...)` so they enter the CAS rather than being treated as plain doubles.

### Algebra & single-equation solve

```matlab
disp(simplify(sin(x)^2 + cos(x)^2))   % 1
disp(expand((x + 1)^4))               % x^4 + 4*x^3 + 6*x^2 + 4*x + 1
disp(factor(x^2 - 1, x))              % (x - 1)*(x + 1)
disp(subs(x + 1, x, sym(2)))          % 3

disp(solve(x^2 - 5*x + 6, x))         % quadratic roots {2, 3}
disp(solve(x^2 == 4, x))              % relational form -> {-2, 2}
disp(vpasolve(cos(x) - x, x, sym(1), 32))   % Dottie number to 32 digits
```

`solve` accepts both an expression (implicitly `== 0`) and an explicit relational
`==`. `vpasolve` runs Newton's method at MPFR variable precision.

### Numeric evaluation & VPA

```matlab
disp(double(subs(x + 1, x, sym(2))))   % 3.0   — sym -> f64
disp(vpa(sym(pi), 32))                 % pi to 32 decimal digits
```

`double` collapses a sym to an f64; `vpa(·, dps)` renders to arbitrary decimal
precision.

### ODEs and an IVP

SymPP's `dsolve` takes the unknown function and its derivatives as plain symbols
(no `y(x)` applied-function shape).

```matlab
syms y yp
disp(dsolve(yp + y, y, yp, x))         % first-order linear  y' + y = 0
syms ypp
disp(dsolve(ypp + y, y, yp, ypp, x))   % second-order  y'' + y = 0

% Apply an initial condition: y' + y = 0, y(0) = 1  ->  exp(-x)
sol = dsolve_ivp(yp + y, y, yp, x, sym(0), sym(1));
disp(sol)
disp(checkodesol(yp + y, sol, y, yp, x))   % residual, expect 0
```

`dsolve_ivp` applies the initial condition and `checkodesol` substitutes the
solution back to verify the residual is zero.

### Transforms

```matlab
syms s w n z t a
disp(laplace(exp(-a*t), t, s))     % 1/(s + a)
disp(fourier(exp(-t*t), t, w))     % sqrt(pi)*exp(-w^2/4)
disp(ztrans(sym(1), n, z))         % z/(z - 1)
```

Each transform takes `(expr, source_var, target_var)`; inverse forms
(`ilaplace`, `ifourier`, `iztrans`) round-trip.

### Symbolic matrices & linear systems

`sym_matrix(R, C, e11, e12, ...)` takes integer-literal dimensions followed by
the entries row-major (the standard `[a 1; 2 b]` literal doesn't yet detect sym
entries — see carve-outs).

```matlab
M = sym_matrix(2, 2, a, sym(1), sym(2), b);
disp(sym_det(M))                       % a*b - 2
disp(sym_inv(M))                       % symbolic inverse

A  = sym_matrix(2, 2, sym(1), sym(2), sym(3), sym(4));
bv = sym_matrix(2, 1, sym(1), sym(2));
disp(sym_linsolve(A, bv))              % solve A x = b  ->  [0; 1/2]

% Linear ODE system y'' = A*y for the rotation matrix
syms tx
A2 = sym_matrix(2, 2, sym(0), sym(1), sym(-1), sym(0));
disp(sym_dsolve_system(A2, tx))

% Multi-equation solve: circle intersect y = x
syms u v
disp(sym_solve_2x2(u^2 + v^2 - 1, v - u, u, v))
```

## Limitations & carve-outs

- **Assumptions are structural by default**: `assume(p, 'positive')` registers a
  mask, but `simplify` only auto-honours it via SymPP's `refine()` (chained in
  the Phase 6.2 ergonomics).
- **Matrix-literal syntax**: `[a 1; 2 b]` doesn't yet detect sym entries — use
  `sym_matrix(...)` with explicit row/col counts.
- **`-emit-systemverilog`** of symbolic expressions is deliberately deferred
  (the general sym surface isn't synthesizable).
- Out of scope: Live Editor / inline-LaTeX rendering, MuPAD compatibility
  (`evalin(symengine, …)`, MuPAD notebooks, `mupadmex`), symbolic plotting
  (`fplot`/`fmesh`/`fsurf`), `finverse`, and GPU-accelerated symbolic.

## See also

- User reference: [`../sym.md`](../sym.md)
- Roadmap / design: [`../symbolic_toolbox_roadmap.md`](../symbolic_toolbox_roadmap.md)
- Example: `examples/symbolic_demo.m`
