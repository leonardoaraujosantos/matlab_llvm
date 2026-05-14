# examples/optim/

Self-contained programs that exercise the Optimization Toolbox
surface shipped by `matlab_llvm`. Each example synthesises its
problem inline and prints the result, so they double as a
reading-order tour of the toolbox.

See [`../../docs/optim_toolbox_roadmap.md`](../../docs/optim_toolbox_roadmap.md)
for the full tiered plan. All five tiers are shipped.

## Running an example

Compile + link against the runtime, then run the binary:

```sh
matlabc -emit-llvm examples/optim/fzero_root.m > /tmp/x.ll
clang++ /tmp/x.ll runtime/matlab_runtime.cpp runtime/runtime_debug.cpp \
    runtime/runtime_complex.cpp runtime/runtime_comm.cpp \
    runtime/runtime_prop.cpp runtime/runtime_rf.cpp runtime/runtime_pde.cpp \
    runtime/runtime_sparse.cpp runtime/runtime_optim.cpp \
    -Iruntime -o /tmp/x.out && /tmp/x.out
```

## Examples

### Solver-based — Tier-1 / Tier-2 / Tier-3

| File | Feature |
|---|---|
| `fzero_root.m` | scalar root finding (Brent's method) |
| `fminbnd_minimum.m` | bounded 1-D minimisation (golden section + parabolic) |
| `fminunc_rosenbrock.m` | unconstrained N-D minimisation (BFGS) |
| `fminsearch_neldermead.m` | derivative-free minimisation (Nelder-Mead simplex) |
| `linprog_diet.m` | linear programming (dense 2-phase simplex) |
| `lsqnonneg_fit.m` | non-negative least squares (Lawson-Hanson) |
| `fsolve_system.m` | nonlinear equations — scalar + N-D systems |
| `fmincon_disk.m` | general constrained nonlinear minimisation |
| `quadprog_portfolio.m` | convex quadratic programming |
| `lsqnonlin_curvefit.m` | nonlinear least squares + curve fitting (Levenberg-Marquardt) |
| `intlinprog_knapsack.m` | mixed-integer LP (branch-and-bound) — knapsack + assignment |
| `coneprog_socp.m` | second-order cone programming |
| `fminimax_design.m` | minimax + goal attainment (multiobjective) |

### Problem-based — Tier-4 / Tier-5

| File | Feature |
|---|---|
| `problem_based_lp.m` | `optimvar` / `optimproblem` / `solve` — LP, QP, MILP |
| `problem_based_eqn.m` | `eqnproblem` — problem-based equation solving |

### Cross-toolbox

| File | Feature |
|---|---|
| `blade_pitch_opt.m` | headline demo — `fmincon` driven by a 3-D PDE elasticity solve |

## REPL / debug support

Both lanes — file / compile-execute (`-emit-llvm`) and the experimental
REPL (`matlabc -repl`) — run the toolbox surface.

- **Solver-based examples** (`fzero` … `fminimax_design`) run in the
  REPL, including vector-objective anons (`@(x) x(1)*x(1) + x(2)*x(2)`)
  defined on one line and handed to a solver on a later line.
- **Problem-based examples** (`problem_based_lp`, `problem_based_eqn`)
  run in the REPL: `optimvar` / `optimproblem` / `eqnproblem` / `solve`
  build the expression DAG across turns and `solve` dispatches just as
  in the file lane.
- **`blade_pitch_opt`** runs via the file / compile-execute lane only.
  Its REPL gap is *not* in the optimisation path — it's a pre-existing
  cross-turn scalar/matrix type-inference issue: a `max(...)` reduction
  result stored to the workspace comes back matrix-typed, so a later
  turn's `if abs(k_stress - ...) < ...` builds a tensor-typed condition
  that `scf.if` rejects. The same program compiles and executes
  correctly through `-emit-llvm`.
