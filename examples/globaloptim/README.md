# Global Optimization Toolbox Examples

Programs that exercise the Global Optimization Toolbox runtime in
matlab_llvm.  See
[`docs/global_optim_toolbox_roadmap.md`](../../docs/global_optim_toolbox_roadmap.md)
for the full tiered roadmap.

This toolbox is an **amplifier of the shipped Optimization Toolbox** —
every solver is a derivative-free / stochastic global search that runs
over the shared seeded PRNG (`rng`-reproducible) and reuses the shipped
`fmincon` for its hybrid-polish step.  No external solver dependency.

## Tier-1 (shipped)

The three stochastic global solvers on a box-bounded objective, each with
a `fmincon` hybrid-polish step.

| Example | User's Guide | Notes |
|---|---|---|
| [`rastrigin_ga.m`](rastrigin_ga.m) | Genetic Algorithm chapter, *Minimize Rastrigin's Function* | **Tier-1 headline.**  Rastrigin (≈30 local minima on `[-5.12,5.12]²`): a local solver (`fminunc`) from `(3.1,2.9)` traps at f=16.91; `ga` + hybrid recovers the global **f=0** at the origin; `particleswarm` also f=0; `simulannealbnd` from `(4,4)` reaches f=0.995 (the near-global lattice ring). |

### Surface covered

- **`ga(fun, nvars, A, b, Aeq, beq, lb, ub)`** — real-coded genetic
  algorithm (tournament selection, BLX-α crossover, Gaussian mutation,
  elitism) + `fmincon` hybrid polish.
- **`particleswarm(fun, nvars, lb, ub)`** — Clerc-Kennedy constriction
  PSO with bound reflection.
- **`simulannealbnd(fun, x0, lb, ub)`** — bounded simulated annealing
  (geometric cooling + reannealing).
- All three honor `rng(seed)` for reproducible runs and reuse the shipped
  Optimization Toolbox `fmincon` for the hybrid polish.

### Tier-1 limitations (carve-downs)

The objective is the single-arg `double(@fun)(x)` handle ABI.  Note that
`x(i)^2` (scalar power inside an anonymous function) is a pre-existing
compiler gap shared with the shipped `fminunc` — use `x(i)*x(i)`.

## Tier-2 (shipped)

The multi-start meta-solvers that orchestrate many `fmincon` local solves
from scattered start points.

| Example | User's Guide | Notes |
|---|---|---|
| [`sixhump_multistart.m`](sixhump_multistart.m) | GlobalSearch & MultiStart chapter, *Find Global or Multiple Local Minima* | **Tier-2 headline.**  Six-hump camelback (six local minima, two global at f*=−1.0316): a single `fminunc` from `(1.6,−0.6)` traps at f=−0.2155; `MultiStart` (20 restarts) and `GlobalSearch` both recover the global −1.0316. |

### Surface covered

- **`createOptimProblem('fmincon', 'objective', @f, 'x0', x0, 'lb', lb,
  'ub', ub)`** — builds the problem the meta-solvers consume (the
  objective handle rides to `run` through a runtime thread-local).
- **`MultiStart()` + `run(ms, problem, k)`** — k `fmincon` restarts from
  x0 + random points in bounds; returns the best local solution.
- **`GlobalSearch()` + `run(gs, problem)`** — scatter-sample trial
  points + `fmincon` from the most promising.

### Tier-2 limitation

Tier-2 supports one active `createOptimProblem` at a time (the objective
handle rides through a single thread-local context).

## Tier-3 (shipped)

Deterministic direct search — derivative-free, robust where gradients are
undefined, discontinuous, or noisy.

| Example | User's Guide | Notes |
|---|---|---|
| [`nonsmooth_patternsearch.m`](nonsmooth_patternsearch.m) | Using Direct Search chapter | **Tier-3 headline.**  On a *discontinuous staircase bowl* the gradient solver `fminunc` stalls at f=125 (its FD gradient is ~0 on every flat step); `patternsearch` steps down to the global f=0 at (2,−3).  Plus a nonsmooth V-valley whose minimum sits at a kink. |

### Surface covered

- **`patternsearch(fun, x0, A, b, Aeq, beq, lb, ub)`** — Generalized
  Pattern Search: complete poll over the 2N positive basis {±e_i}, mesh
  expand on success / contract on failure.  Fully deterministic (no
  PRNG); no hybrid polish (the mesh refinement is the convergence).

## Tier-4 (shipped)

Surrogate optimization — the sample-efficient global solver for expensive
objectives.

| Example | User's Guide | Notes |
|---|---|---|
| [`branin_surrogate.m`](branin_surrogate.m) | Surrogate Optimization chapter | **Tier-4 headline.**  `surrogateopt` fits a cubic-RBF surrogate and adaptively samples it; it recovers Branin's global f*=0.3979 (the canonical surrogate-/Bayesian-optimization benchmark) and the six-hump camelback's global −1.0316. |

### Surface covered

- **`surrogateopt(fun, lb, ub)`** — cubic-RBF surrogate (φ(r)=r³ + linear
  tail, solved via the shipped `mldivide`) + merit-weighted adaptive
  sampling (surrogate value vs distance-to-samples, cycled weight) + a
  final `fmincon` polish.

## Tier-5 (shipped)

Multiobjective optimization — Pareto-front computation for conflicting
objectives.  The objective returns a *vector* of objective values.

| Example | User's Guide | Notes |
|---|---|---|
| [`pareto_front.m`](pareto_front.m) | Multiobjective Optimization chapter | **Tier-5 headline.**  Two conflicting objectives (`f1=(x-1)²`, `f2=(x+1)²`) make the Pareto-optimal set the whole interval `x∈[-1,1]`.  Both `gamultiobj` (NSGA-II) and `paretosearch` recover the *full trade-off curve* (x=-1 to x=+1) — not a single compromise. |

### Surface covered

- **`gamultiobj(fun, nvars, A, b, Aeq, beq, lb, ub)`** — NSGA-II: fast
  non-dominated sort + crowding-distance crowded-comparison tournament,
  BLX-α crossover, elitist (P∪Q) survival; returns the first front.
- **`paretosearch(fun, nvars, …)`** — non-dominated archive seeded by
  scatter sampling, refined by a GPS-style poll, crowding-pruned each
  iteration.
- Both take a *vector-returning* objective handle and return the Pareto
  set as a k×nvars matrix.

## Tier-6 (shipped — focused subset)

Configurability carve-down sweep.  The flagship `ga` gains an options
carrier and the mixed-integer capability.

| Example | User's Guide | Notes |
|---|---|---|
| [`gear_train_intga.m`](gear_train_intga.m) | Mixed-Integer ga Optimization chapter | **Tier-6 headline.**  Classic Sandgren gear-train design: four gears with *integer* tooth counts (12…60) chosen so `(z1·z2)/(z3·z4)` approximates `1/6.931`.  `optimoptions('ga', 'IntCon', [1 2 3 4])` forces every variable integer; `ga` reaches a ratio error ≈ 2.3e-11 — far below any rounded-continuous guess. |
| [`ackley_compare.m`](ackley_compare.m) | *Compare Global Solvers* | Multi-modal Ackley function (sharp global minimum at the origin in a ripple-covered plateau).  Drives `ga` through the **canonical full signature** `ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,options)` with an `optimoptions('ga', …)` carrier, then cross-checks `particleswarm` / `patternsearch` / `simulannealbnd`: `fminunc` from (3,3) traps at f≈6.56 while every global solver reaches f=0. |

### Surface covered

- **`optimoptions('ga', 'Name', val, …)`** — options carrier classdef.
  Tier-6 reads `PopulationSize`, `MaxGenerations`, and `IntCon`
  (sentinel `−1` = "use the solver default").
- **Integer-constrained `ga`** — `ga(fun, nvars, …, lb, ub, opts)` (or the
  5-arg `ga(fun, nvars, lb, ub, opts)` convenience form) routes to
  `matlab_gads_ga_opts`, which rounds the `IntCon` variables to the nearest
  feasible integer at init, after every crossover/mutation, and at the
  final result.  The `fmincon` hybrid polish is **auto-skipped** when any
  variable is integer.
- **Options-bearing `ga` call forms** — the Lowering dispatch detects the
  trailing options carrier in the 5-arg `ga(fun,nvars,lb,ub,opts)`, 9-arg
  `ga(fun,nvars,A,b,Aeq,beq,lb,ub,opts)`, and 10-arg
  `ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,opts)` (canonical full signature;
  `nonlcon` must be `[]` today — nonlinear constraints are a follow-on)
  forms.

### Carve-downs (Tier-6 follow-ons)

`optimoptions` for the other solvers (`SwarmSize`/`MaxIterations`/…) +
`HybridFcn`/`FunctionTolerance` knobs / `exitflag`/`output` +
`GlobalOptimSolution` multi-return / `IntCon` for `surrogateopt` /
hypervolume + spread metrics / `PollMethod`(GSS/MADS) / NUPS / `SearchFcn` /
nonlinear + multiobjective constraints / problem-based `solve` routing /
the dipole cross-toolbox demo / parallel.  See the roadmap.

> **Cross-toolbox note**: a `surrogateopt` dipole-VSWR demo (coupling to
> the shipped Antenna `antennaWireSolve`) is deferred — the antenna
> solver returns a struct, and `f(...).field` inside an anonymous-function
> objective body is a current compiler gap.  It unlocks once a
> scalar-returning antenna wrapper ships.
