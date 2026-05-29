# Optimization Toolbox — Tutorial

The Optimization Toolbox runtime in `matlab_llvm` covers the full
solver-based surface (root finding, line/curve search, LP/QP/MILP/SOCP,
constrained nonlinear, least squares, minimax) plus the problem-based
`optimvar` / `optimproblem` / `eqnproblem` workflow. All five tiers are
shipped, and most examples run in both the compile-execute lane and the
experimental REPL.

## Supported features

- **Root finding & 1-D search**: `fzero` (Brent), `fminbnd` (golden
  section + parabolic interpolation).
- **Unconstrained N-D**: `fminunc` (BFGS quasi-Newton + Armijo line
  search), `fminsearch` (Nelder-Mead simplex, derivative-free).
- **Nonlinear systems**: `fsolve` (scalar Newton / N-D Levenberg-Marquardt).
- **Linear & quadratic programs**: `linprog` (2-phase simplex),
  `quadprog` (augmented-Lagrangian), `lsqnonneg` (Lawson-Hanson NNLS).
- **Constrained nonlinear**: `fmincon` (augmented-Lagrangian over
  bound-projected BFGS) with bounds, linear (in)equalities, and
  `nonlcon` handles.
- **Least squares**: `lsqnonlin`, `lsqcurvefit` (Levenberg-Marquardt).
- **Mixed-integer & cones**: `intlinprog` (branch-and-bound), `coneprog`
  (SOCP reformulated as a nonlinear inequality).
- **Multiobjective**: `fminimax`, `fgoalattain` (epigraph reformulation).
- **Problem-based**: `optimvar`, `optimintvar`, `optimproblem`,
  `eqnproblem`, `solve` (auto-dispatch LP / QP / MILP / nonlinear).

## Build & run

```bash
build/matlabc -emit-llvm examples/optim/fzero_root.m > /tmp/fzero_root.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/fzero_root.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/fzero_root
/tmp/fzero_root
```

## Worked examples

### Scalar root finding (`examples/optim/fzero_root.m`)

`fzero` brackets a sign change and drives it to a root with Brent's
method. Both the guess form and the `[a b]` bracket form are supported.

```matlab
% cos(x) = x  ->  the Dottie number ~ 0.739085
r1 = fzero(@(x) cos(x) - x, 0.5);
% a cubic root  x^3 - x - 2 = 0 ~ 1.521380
r2 = fzero(@(x) x*x*x - x - 2, 0);
% bracket form: fzero(@fn, [a b]) requires f(a)*f(b) <= 0
r3 = fzero(@(x) sin(x), [3, 4]);   % isolates pi
```

The objective is an anonymous-function handle; the runtime evaluates it
through the single-argument `double(@fun)(x)` ABI. Output prints
`0.739085`, `1.521380`, and `3.141593`.

### Unconstrained minimisation (`examples/optim/fminunc_rosenbrock.m`)

`fminunc` is BFGS quasi-Newton with a forward-difference gradient when
none is supplied; `fminsearch` (`examples/optim/fminsearch_neldermead.m`)
solves the same Rosenbrock problem derivative-free.

```matlab
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
           (1 - x(1))*(1 - x(1));
r = fminunc(ros, [-1.2; 1]);       % -> [1.0000, 1.0000]
```

Note the `x(1)*x(1)` form: scalar power (`x(1)^2`) inside an anonymous
function passed to a solver is a known frontend gap — use the product.

### Constrained nonlinear minimisation (`examples/optim/fmincon_disk.m`)

`fmincon` accepts the 4-, 8-, and 9-argument call shapes. Here Rosenbrock
is minimised inside the unit disk via a `nonlcon` handle returning
`c(x) <= 0`:

```matlab
ros  = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + (1 - x(1))*(1 - x(1));
disk = @(x) [x(1)*x(1) + x(2)*x(2) - 1];   % nonlcon: c(x) <= 0
r = fmincon(ros, [0; 0], [], [], [], [], [], [], disk);
% -> x ~= [0.7864, 0.6177], on the disk boundary
```

The same file also shows the bound-constrained 8-argument form and the
linear-inequality 4-argument form `fmincon(@fun, x0, A, b)`.

### Linear programming (`examples/optim/linprog_diet.m`)

`linprog` runs a dense 2-phase tableau simplex. Both `linprog(f, A, b)`
and the full 7-argument form are supported.

```matlab
f = [-1; -2];
A = [1, 1; 1, 3];   b = [4; 6];
x  = linprog(f, A, b);                    % 3-arg, lb defaults to 0
x2 = linprog(f, A, b, [], [], lb, ub);    % 7-arg with bounds
x3 = linprog(f, A, b, [1, 1], 3, lb, ub); % adds equality x1+x2=3
% optimum vertex (3, 1), objective -5
```

### Mixed-integer LP (`examples/optim/intlinprog_knapsack.m`)

`intlinprog` is depth-first branch-and-bound over the simplex. The
0/1 knapsack picks items 2 and 4 (value 15, weight 10):

```matlab
f      = [-8; -11; -6; -4];        % maximise -> minimise the negation
intcon = [1; 2; 3; 4];
A      = [5, 7, 4, 3];   b = 10;
lb = [0;0;0;0];   ub = [1;1;1;1];
x = intlinprog(f, intcon, A, b, [], [], lb, ub);   % -> [0 1 0 1]
```

The file's second part encodes a 3x3 assignment problem with `Aeq`/`beq`.

### Problem-based LP/QP/MILP (`examples/optim/problem_based_lp.m`)

The problem-based workflow builds the objective and constraints from
operator-overloaded `optimvar` variables and lets `solve` dispatch the
right solver. `solve` returns the solution as a column vector in
variable-creation order.

```matlab
x = optimvar();   y = optimvar();
prob = optimproblem();
prob.Objective = -x - 2*y;
prob.Constraints.c1 = x + y <= 4;
prob.Constraints.c2 = x + 3*y <= 6;
sol = solve(prob);                 % -> x=3, y=1
```

Use `optimintvar()` to request integer variables (MILP). The companion
`examples/optim/problem_based_eqn.m` shows `eqnproblem` with
`prob.Equations.e1 = lhs == rhs` for solving linear and nonlinear systems.

### Other examples

- `quadprog_portfolio.m` — convex QP (minimum-variance portfolio,
  equality + bound constraints).
- `lsqnonlin_curvefit.m` — `lsqnonlin` residual handle and `lsqcurvefit`
  model handle fitting `y = a*exp(-b*t)`.
- `lsqnonneg_fit.m` — non-negative least squares (Lawson-Hanson active set).
- `coneprog_socp.m` — second-order cone program (maximise on the unit disk).
- `fminimax_design.m` — `fminimax` over three paraboloids + `fgoalattain`.
- `fminbnd_minimum.m` — bounded 1-D minimisation.
- `blade_pitch_opt.m` — cross-toolbox headline: a 3-D PDE elasticity solve
  characterises a turbine blade, then `fmincon` picks the pitch that
  maximises power within the stress limit.

## Limitations & carve-outs

- Scalar power `x(i)^2` and `max(a,b)` inside a solver-bound anonymous
  function are frontend gaps — use `x(i)*x(i)` and rewrite accordingly.
- Anonymous objectives cannot capture workspace variables; inline the
  constants (see `blade_pitch_opt.m`, where the FEM coefficient is written
  as a literal and asserted against the live value).
- The `optimoptions` options surface and the `[x, fval, exitflag, output]`
  multi-return are Tier-1 carve-downs (1.7 / 1.8).
- Problem-based scope is scalar variables (`optimvar()` / `optimintvar()`),
  not the name/size form; `show` / `write` / `prob2struct` are deferred.
- Nonlinear equality constraints and complex-valued objectives are out of
  scope. The `blade_pitch_opt.m` example runs in the compile-execute lane
  only (a cross-turn scalar/matrix type-inference issue in the REPL).

## See also

- Roadmap: [`../optim_toolbox_roadmap.md`](../optim_toolbox_roadmap.md)
- Examples: `examples/optim/`
