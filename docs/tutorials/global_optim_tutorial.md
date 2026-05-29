# Global Optimization Toolbox — Tutorial

The Global Optimization Toolbox runtime is an amplifier of the shipped
Optimization Toolbox: every solver is a derivative-free or stochastic
global search that runs over the shared seeded PRNG (so results are
`rng`-reproducible) and reuses the shipped `fmincon` for its hybrid-polish
step. No external solver dependency. All six tiers are shipped.

## Supported features

- **Stochastic global solvers**: `ga` (real-coded genetic algorithm +
  `fmincon` hybrid polish), `particleswarm` (Clerc-Kennedy constriction
  PSO), `simulannealbnd` (bounded simulated annealing).
- **Multi-start meta-solvers**: `createOptimProblem`, `MultiStart` /
  `run(ms, problem, k)`, `GlobalSearch` / `run(gs, problem)`.
- **Direct search**: `patternsearch` (Generalized Pattern Search, fully
  deterministic, robust on nonsmooth/discontinuous objectives).
- **Surrogate optimization**: `surrogateopt` (cubic-RBF surrogate +
  merit-weighted adaptive sampling, for expensive objectives).
- **Multiobjective**: `gamultiobj` (NSGA-II), `paretosearch`
  (non-dominated archive + direct-search poll) — both return the full
  Pareto front from a vector-returning objective.
- **Options & mixed-integer**: `optimoptions('ga', ...)` carrier reading
  `PopulationSize` / `MaxGenerations` / `IntCon`; integer-constrained `ga`.

## Build & run

```bash
build/matlabc -emit-llvm examples/globaloptim/rastrigin_ga.m > /tmp/rastrigin_ga.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/rastrigin_ga.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/rastrigin_ga
/tmp/rastrigin_ga
```

## Worked examples

### Genetic algorithm vs. a trapped local solver (`examples/globaloptim/rastrigin_ga.m`)

Rastrigin's function is a paraboloid studded with ~30 local minima; a
local solver from a poor start traps, while `ga` explores globally and the
`fmincon` hybrid polishes to the exact optimum.

```matlab
rastrigin = @(x) 20 ...
    + (x(1)*x(1) - 10*cos(2*pi*x(1))) ...
    + (x(2)*x(2) - 10*cos(2*pi*x(2)));
lb = [-5.12; -5.12];   ub = [5.12; 5.12];

rng(0);
xlocal = fminunc(rastrigin, [3.1; 2.9]);   % trapped at f~16.91
rng(42);
xga = ga(rastrigin, 2, [], [], [], [], lb, ub);   % global f=0 at origin
rng(42);
xps = particleswarm(rastrigin, 2, lb, ub);        % also f=0
rng(42);
xsa = simulannealbnd(rastrigin, [4; 4], lb, ub);
```

Note the `x(i)*x(i)` form — scalar power in a solver-bound anonymous
function is a known compiler gap. All three solvers honor `rng(seed)`.

### Multi-start meta-solvers (`examples/globaloptim/sixhump_multistart.m`)

The six-hump camelback has six local minima (two global). `MultiStart`
runs many `fmincon` restarts; `GlobalSearch` scatter-samples then polishes.
The objective handle rides from `createOptimProblem` to `run` through a
runtime thread-local context.

```matlab
problem = createOptimProblem('fmincon', 'objective', camel, ...
                             'x0', [2; 1], 'lb', lb, 'ub', ub);
ms = MultiStart();
xms = run(ms, problem, 20);        % 20 fmincon restarts -> global -1.0316
gs = GlobalSearch();
xgs = run(gs, problem);            % scatter-search + fmincon -> global
```

### Direct search on a discontinuous objective (`examples/globaloptim/nonsmooth_patternsearch.m`)

`patternsearch` polls a positive-spanning basis on a mesh, never touching
a gradient — so it works where the FD gradient is meaningless. Here a
quantized "staircase" bowl stalls `fminunc` but `patternsearch` steps down
to the global minimum.

```matlab
% f = round( ((x1-2)^2 + (x2+3)^2) * 2 ) / 2  — global min 0 at (2,-3)
staircase = @(x) round( ((x(1)-2)*(x(1)-2) + (x(2)+3)*(x(2)+3)) * 2 ) / 2;
xu = fminunc(staircase, [7; 7]);                          % stalled
xp = patternsearch(staircase, [7;7], [], [], [], [], lb, ub);  % -> (2,-3)
```

The file's bonus shows a nonsmooth V-valley `3|x1-1| + 3|x2+2| + 0.5` with
its minimum at a kink.

### Surrogate optimization (`examples/globaloptim/branin_surrogate.m`)

`surrogateopt` fits a cubic-RBF surrogate (coefficients solved via the
shipped `mldivide`) and adaptively samples it — sample-efficient for
expensive objectives. It recovers Branin's global `f* = 0.3979`:

```matlab
branin = @(x) (x(2) - 5.1/(4*pi*pi)*x(1)*x(1) + 5/pi*x(1) - 6) ...
              * (x(2) - 5.1/(4*pi*pi)*x(1)*x(1) + 5/pi*x(1) - 6) ...
              + 10*(1 - 1/(8*pi))*cos(x(1)) + 10;
rng(7);
xb = surrogateopt(branin, [-5; 0], [10; 15]);   % f ~ 0.3979
```

Constants are inlined (an anon that captures variables and is also passed
to a solver is a known gap). The file also runs the six-hump camelback.

### Pareto fronts (`examples/globaloptim/pareto_front.m`)

When objectives conflict there is no single best point. The objective
returns a *vector* (same vector-out ABI as `lsqnonlin`), and both solvers
return the whole front as a `k x nvars` matrix.

```matlab
fun = @(x) [(x(1) - 1)*(x(1) - 1); (x(1) + 1)*(x(1) + 1)];
rng(1);
Xg = gamultiobj(fun, 1, [], [], [], [], -3, 3);   % NSGA-II
rng(1);
Xp = paretosearch(fun, 1, [], [], [], [], -3, 3);  % archive + poll
% both recover the full trade-off set x in [-1, 1]
```

### Mixed-integer ga (`examples/globaloptim/gear_train_intga.m`)

The Sandgren gear-train benchmark: four gears with *integer* tooth counts
chosen so `(z1*z2)/(z3*z4)` approximates `1/6.931`.
`optimoptions('ga', 'IntCon', [1 2 3 4])` forces every variable integer;
the `fmincon` hybrid polish is auto-skipped once any variable is integer.

```matlab
err = @(z) ((1/6.931) - (z(1)*z(2)) / (z(3)*z(4))) * ...
           ((1/6.931) - (z(1)*z(2)) / (z(3)*z(4)));
lb = [12;12;12;12];   ub = [60;60;60;60];
opts = optimoptions('ga', 'PopulationSize', 200, ...
                          'MaxGenerations', 200, ...
                          'IntCon', [1 2 3 4]);
z = ga(err, 4, [], [], [], [], lb, ub, opts);   % ratio error ~ 2.3e-11
```

`examples/globaloptim/ackley_compare.m` drives `ga` through the canonical
full 10-argument signature `ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon,
options)` with an `optimoptions` carrier and cross-checks every solver on
the multi-modal Ackley function.

## Limitations & carve-outs

- Scalar power `x(i)^2` inside a solver-bound anon is a frontend gap — use
  `x(i)*x(i)`. Solver-bound anons must be capture-free (inline constants).
- `optimoptions` reads only `PopulationSize` / `MaxGenerations` / `IntCon`
  today; in the full `ga` signature `nonlcon` must be `[]` (nonlinear
  constraints are a follow-on).
- Tier-2 supports one active `createOptimProblem` at a time.
- Carve-downs: `optimoptions` for the other solvers, `HybridFcn` /
  tolerance knobs, `[x, fval, exitflag, output]` / `GlobalOptimSolution`
  multi-return, `IntCon` for `surrogateopt`, hypervolume/spread metrics,
  `PollMethod` (GSS/MADS), `SearchFcn`, nonlinear + multiobjective
  constraints, problem-based `solve` routing, and parallel runs.

## See also

- Roadmap: [`../global_optim_toolbox_roadmap.md`](../global_optim_toolbox_roadmap.md)
- Examples: `examples/globaloptim/`
