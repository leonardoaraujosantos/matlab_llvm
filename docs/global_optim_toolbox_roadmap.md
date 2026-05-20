# Global Optimization Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Global-Optimization-Toolbox programs.

Source: *Global Optimization Toolbox User's Guide* (R2026a, 17 chapters:
Getting Started · Problem-Based Global Optimization · GlobalSearch &
MultiStart · Problem-Based Multiple Start · Direct Search · Problem-Based
Direct Search · Genetic Algorithm · Problem-Based GA · Particle Swarm ·
Surrogate Optimization · Problem-Based Surrogate · Simulated Annealing ·
Multiobjective Optimization · Problem-Based Multiobjective · Parallel
Processing · Options Reference · Functions).

This toolbox is an **amplifier of the already-shipped Optimization
Toolbox** ([`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md), all 5
tiers shipped 2026-05-14).  Every solver here is a derivative-free /
stochastic global search that, where it needs a local refinement step,
calls the shipped `fmincon` / `fminunc` / `lsqnonlin`.  The Optim roadmap
explicitly carved Global Optimization out as "separate toolbox, separate
roadmap" (§ carve-outs) — this is that roadmap.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/globaloptim/rastrigin_ga.m`](../examples/globaloptim/rastrigin_ga.m):
*minimise Rastrigin's function (the canonical multi-modal benchmark, ~30
local minima on a `[-5.12, 5.12]²` box) with `ga`, then polish the best
individual with a `fmincon` hybrid step* — the UG's own "Minimize
Rastrigin's Function" walkthrough.  Achieving it end-to-end is what
closes **GADS-Tier-1**.

Companion docs: [`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)
(the shipped local-solver base every row leans on),
[`feature_status.md`](feature_status.md), [`roadmap.md`](roadmap.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the smallest end-to-end loop: the three population/point stochastic
  solvers `ga` + `particleswarm` + `simulannealbnd` on a box-bounded
  objective. **Tier-2** is the multi-start meta-solvers `MultiStart` +
  `GlobalSearch` (thin orchestration over the shipped `fmincon`).
  **Tier-3** is deterministic direct search (`patternsearch` GPS/GSS +
  NUPS). **Tier-4** is `surrogateopt` (radial-basis-function surrogate +
  adaptive sampling). **Tier-5** is multiobjective (`gamultiobj` +
  `paretosearch` → Pareto fronts). **Tier-6** is the carve-down sweep —
  the shipped subset is the `optimoptions('ga', …)` options carrier +
  integer-constrained `ga` (`IntCon`); other-solver options, hybrid
  functions, and problem-based `solve` integration stay documented as
  follow-ons.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: T1 ~1.5 wk,
  T2 ~1 wk, T3 ~1 wk, T4 ~1.5 wk, T5 ~1.5 wk, T6 ~1 wk (~7.5 wk full).
  This is **cheap** for the user-visible payoff — the numeric base is
  entirely shipped.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **ALL 6 TIERS SHIPPED 2026-05-20** (`runtime/toolbox/gads/`) — Tier-1
  `ga` + `particleswarm` + `simulannealbnd` with `fmincon` hybrid
  polish; Tier-2 `MultiStart` + `GlobalSearch` + `createOptimProblem`;
  Tier-3 `patternsearch` (deterministic GPS direct search); Tier-4
  `surrogateopt` (cubic-RBF surrogate); Tier-5 `gamultiobj` +
  `paretosearch` (multiobjective Pareto fronts); **Tier-6** (focused
  carve-down sweep) `optimoptions('ga', …)` options carrier +
  integer-constrained `ga` (`IntCon`) — see §7 for the shipped subset
  vs. documented follow-ons.
- **Compile/Execute path** (identical pattern across rows): Sema
  registers each solver name in
  [`lib/Sema/Resolver.cpp::registerBuiltins()`](../lib/Sema/Resolver.cpp);
  per-builtin shape rules go in
  [`lib/Sema/TypeInference.cpp`](../lib/Sema/TypeInference.cpp);
  `matlab.call_builtin @name(...)` is rewritten to
  `llvm.call @matlab_gads_*(...)` in
  [`LowerTensorOps.cpp`](../lib/MLIR/Passes/LowerTensorOps.cpp) (the
  `pde_table` loose-match block — same precedent as Optim / Ident);
  runtime entries live in a new
  [`runtime/toolbox/gads/runtime_gads.cpp`](../runtime/toolbox/gads/runtime_gads.cpp).
  Add `gads` to both `kToolboxDirs` tables in
  [`tools/matlabc/main.cpp`](../tools/matlabc/main.cpp), to all three
  CMake source lists, and to `test/Run/run_tests.sh`.
- **Function-handle ABI**: every solver takes the objective as a
  single-arg `matlab_mat*(*)(matlab_mat*)` handle — the **exact ABI**
  already proven by `fminunc` / `lsqnonlin` / `nlmpc` / `greyest`.  The
  `LowerAnonCalls.cpp::retypeAnonsForVectorObjective` recognizer just
  needs `ga` / `particleswarm` / `simulannealbnd` / etc. added to its
  list (objective handle at operand 0).
- **Debug / REPL**: the solver outputs are plain matrices (`x`) and
  scalars (`fval`); the only new descriptor types are the option carriers
  (`optimoptions('ga', …)` — see Tier-6) and `GlobalSearch` /
  `MultiStart` solver objects, which render via the existing classdef
  property-bag path.
- **No external solver dependencies**: matching the project's hand-coded
  precedent. The stochastic solvers are pure hand-coded loops over the
  shipped PRNG (`matlab_rng_state` + `matlab_rand` / `matlab_randn`); the
  local-refinement / hybrid steps call the shipped Optim solvers. **No
  GA library, no NLopt, no external surrogate package.**

---

## 1. Reusable infrastructure (Tier-0 baseline — no GADS code yet)

Everything below already exists and **does not need to be re-built**.
This toolbox is unusually cheap precisely because its entire numeric base
is shipped.

| Group | Surface (already shipped) | Location | How GADS uses it |
|---|---|---|---|
| Local NLP solvers | `fmincon` (aug-Lagrangian), `fminunc` (BFGS), `lsqnonlin` (LM), `fminsearch` (Nelder-Mead) | `runtime/toolbox/optim/runtime_optim.cpp` | The **hybrid / refinement** step of `ga` / `particleswarm` / `simulannealbnd`; the per-start local solve of `MultiStart` / `GlobalSearch`. |
| Function-handle ABI | `void *fn_p` → `matlab_mat*(*)(matlab_mat*)`; `LowerAnonCalls` retyping | `runtime_optim.cpp`, `LowerAnonCalls.cpp` | Every GADS objective + nonlinear-constraint handle. |
| PRNG | `matlab_rng_state` (xorshift) + `matlab_rand` / `matlab_randn` + `rng(seed)` | `runtime/toolbox/comm/runtime_comm.cpp`, `matlab_runtime.cpp` | Population init, mutation, crossover, swarm velocity, SA proposals, surrogate sampling — all reproducible via `rng`. |
| Bound/constraint plumbing | `lb` / `ub` / `A·x≤b` / `Aeq·x=beq` / nonlcon already parsed + passed for `fmincon` | `runtime_optim.cpp` | Same arg shapes reused verbatim for every GADS solver. |
| Dense linear algebra | `mldivide`, `qr`, `chol`, `pinv`, `matlab_matmul_mm` | `runtime/matlab_runtime.cpp` | Surrogate RBF coefficient solve; pattern-search basis generation. |
| Problem-based API | `optimproblem` / `optimvar` / `solve` expression-DAG | `runtime/toolbox/optim/optim_classdefs.m` | The "Problem-Based" GADS chapters route through the existing `solve` dispatch with a `Solver` name — Tier-6 hook. |
| Classdef plumbing | `matlab_obj_new` / `_set_*`, kwarg-ctor sugar, class-pinned dispatch | `lib/MLIR/Lowering.cpp` | `GlobalSearch` / `MultiStart` solver objects + `optimoptions` carriers. |
| Multiobjective scaffolding | `fminimax` / `fgoalattain` / `paretosearch`-style weighting already in Optim Tier-3 | `runtime_optim.cpp` | Reference for `gamultiobj` non-dominated sorting. |

**Net assessment**: ~95% of the numeric machinery is shipped. The genuinely
new code is the **stochastic search loops** (GA operators, PSO velocity
update, SA cooling schedule, RBF surrogate, non-dominated sorting) — each
a self-contained few-hundred-line hand-coded routine over the existing
PRNG + Optim base.

---

## 2. Tier-1 — Stochastic global solvers (`ga` + `particleswarm` + `simulannealbnd`) ✅ shipped 2026-05-20

Goal: the three headline derivative-free global solvers running on a
box-bounded objective, each with a `fmincon` hybrid-polish step. Shipped
in [`runtime/toolbox/gads/runtime_gads.cpp`](../runtime/toolbox/gads/runtime_gads.cpp).

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 1.1 | `ga(fun, nvars, [], [], [], [], lb, ub)` | ✅ | Real-coded GA: tournament-of-2 selection, BLX-α crossover, Gaussian mutation (annealed scale), elitism. Pop `min(200, max(20,10·nvars))`, gens ≤ 400. | `matlab_gads_ga` |
| 1.2 | `particleswarm(fun, nvars, lb, ub)` | ✅ | Clerc-Kennedy constriction PSO: `v ← w·v + c₁r₁(p−x) + c₂r₂(g−x)` (w=0.729, c=1.49), bound reflection. Swarm `min(100, max(20,10·nvars))`. | `matlab_gads_particleswarm` |
| 1.3 | `simulannealbnd(fun, x0, lb, ub)` | ✅ | Bounded SA: temperature-scaled Gaussian proposals, Metropolis acceptance, **slow geometric cooling** (T←0.95·T every 50 steps) with reheat-from-best reannealing after 400 stalled steps, 6000 iters. | `matlab_gads_simulannealbnd` |
| 1.4 | hybrid-function polish | ✅ | After the stochastic phase, refine the best point with the shipped `fmincon` (kept only if it improves). **Tier-1 always polishes**; the options-controlled `HybridFcn` is Tier-6. | reuses `matlab_optim_fmincon` |
| 1.5 | `rng`-reproducibility | ✅ | All three run over the shared `matlab_rng_state` xorshift; `rng(seed)` makes every run byte-reproducible (the gating-test requirement). | shipped PRNG |
| 1.6 | multi-return `[x, fval]` | 🔵 | 1-return `x` ships; `[x,fval,exitflag,output]` is Tier-6. | — |

**🎯 Headline (shipped, closes Tier-1)**:
[`examples/globaloptim/rastrigin_ga.m`](../examples/globaloptim/rastrigin_ga.m)
— minimise Rastrigin (≈30 local minima on `[-5.12,5.12]²`).  A local
solver (`fminunc`) from `(3.1,2.9)` gets **trapped at f=16.91**; `ga` +
hybrid recovers the **global f=0 at the origin**; `particleswarm` also
f=0; `simulannealbnd` from `(4,4)` reaches f=0.995 (the near-global
lattice ring — SA is the most local-biased of the three, an honest
result).

**Gating tests** (LLVM lane, `.skip-emit-*`): `gads_t1_ga.m` (Rastrigin
→ f=0), `gads_t1_pso.m` (cosine-perturbed bowl → f=0 at (2,−3)),
`gads_t1_sa.m` (sin-perturbed objective from a bad start).  All seeded
via `rng` for determinism.

**Compile/Execute wiring (as built)**: objective handle = the shipped
`double(*)(matlab_mat*)` ABI; Lowering dispatch (in the builtin-call
block) remaps the MATLAB call forms (`ga`'s 8-arg `(fun,nvars,A,b,Aeq,
beq,lb,ub)`, the 4-arg `particleswarm`/`simulannealbnd`) onto the 5-arg
runtime entries and injects the hybrid flag; matrix args lowered via
plain `lowerExpr` (NOT `setType`-to-ptr) so inline `[…;…]` literals
keep their tensor type and the concat lowers, with the `pde_table`
loose-match coercing tensor→ptr at the call; `ga`/`particleswarm`/
`simulannealbnd` (+ runtime symbols) added to
`LowerAnonCalls::retypeAnonsForVectorObjective` (objective at operand 0).

**Carve-downs**: integer constraints (`IntCon`) ✅ **shipped in Tier-6**
and the `optimoptions('ga', …)` surface (PopulationSize / MaxGenerations)
✅ **shipped in Tier-6**. Still 🔵 (→ Tier-6 follow-ons):
nonlinear-constraint handles, custom population/data types, vectorized
objectives, the remaining option knobs (`HybridFcn`/`FunctionTolerance`),
`exitflag`/`output` multi-return.

---

## 3. Tier-2 — Multi-start meta-solvers (`GlobalSearch` + `MultiStart`) ✅ shipped 2026-05-20

Goal: the two solver objects that orchestrate many `fmincon` local solves
from scattered start points. Almost pure orchestration over the shipped
Optim core. Shipped in
[`runtime/toolbox/gads/runtime_gads.cpp`](../runtime/toolbox/gads/runtime_gads.cpp)
+ [`gads_classdefs.m`](../runtime/toolbox/gads/gads_classdefs.m).

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 2.1 | `createOptimProblem('fmincon', 'objective',@f,'x0',x0,'lb',lb,'ub',ub)` | ✅ | Lowering scans the name-value pairs and stashes the objective handle + x0/lb/ub into a runtime **thread-local** problem context (handles can't round-trip through the obj bag — the nlmpc precedent); returns a marker. | `matlab_gads_make_problem` |
| 2.2 | `MultiStart` + `run(ms, problem, k)` | ✅ | `fmincon` from x0 + (k−1) uniform-random starts in [lb,ub]; returns the lowest-objective local solution. Default k=20. | `matlab_gads_multistart` |
| 2.3 | `GlobalSearch` + `run(gs, problem)` | ✅ | Pragmatic OQNLP: scatter-sample 200 trial points, score, run `fmincon` from x0 + the 8 most promising; return the best. | `matlab_gads_globalsearch` |
| 2.4 | `MultiStart` with `lsqcurvefit`/`lsqnonlin` | 🔵 | The least-squares meta-start form — Tier-6 follow-on. | — |
| 2.5 | `GlobalOptimSolution` array | 🔵 | The multi-return solutions object (X/Fval/Exitflag/X0 per distinct local min) — Tier-6 follow-on; single-return `x` ships. | — |

**🎯 Headline (shipped)**:
[`examples/globaloptim/sixhump_multistart.m`](../examples/globaloptim/sixhump_multistart.m)
— the UG "Find Global or Multiple Local Minima" workflow on the six-hump
camelback (six local minima, two global at f*=−1.0316).  A single
`fminunc` from `(1.6,−0.6)` is **trapped at f=−0.2155**; both `MultiStart`
(20 restarts) and `GlobalSearch` recover the **global f=−1.0316**.

**Gating tests** (LLVM lane): `gads_t2_multistart.m` (two-basin function
→ global f=0), `gads_t2_globalsearch.m` (camelback → f=−1.0316).

**Compile/Execute wiring (as built)**: `createOptimProblem` intercepted
in the Lowering builtin-call block — scans `CharLiteral`/`StringLiteral`
keys (`objective`/`x0`/`lb`/`ub`), lowers the handle as ptr (retyped by
`LowerAnonCalls` at `matlab_gads_make_problem` operand 0), emits the
thread-local stash; `MultiStart()`/`GlobalSearch()` factories pinned in
`pinnedOfRhs`; `run(solver, problem[, k])` dispatched on the solver's
pinned class (`MultiStart` → `matlab_gads_multistart(k)`, `GlobalSearch`
→ `matlab_gads_globalsearch()`). `run` is not registered as a generic
builtin trigger (too common); the solver-object mentions pull the prelude.

**Tier-2 simplification**: one active problem at a time (the thread-local
holds the most-recent `createOptimProblem`); `CustomStartPointSet`,
distinct-minima clustering / `GlobalOptimSolution`, parallel restarts, and
output/plot functions are Tier-6 carve-downs.

---

## 4. Tier-3 — Deterministic direct search (`patternsearch`) ✅ shipped 2026-05-20

Goal: the derivative-free pattern-search family — robust on noisy /
nonsmooth / discontinuous objectives where gradients are meaningless.
Shipped in [`runtime/toolbox/gads/runtime_gads.cpp`](../runtime/toolbox/gads/runtime_gads.cpp).

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 3.1 | `patternsearch(fun, x0, A, b, …, lb, ub)` | ✅ | GPS: complete poll over the 2N positive basis {±e_i}, move on success + expand mesh (Δ←2Δ, capped 1e6), contract on failure (Δ←Δ/2) until Δ<1e-9. Bound-clamped. **Fully deterministic** (no PRNG). **No hybrid** — the mesh refinement is the convergence (a gradient polish is inappropriate on the nonsmooth objectives this targets). | `matlab_gads_patternsearch` |
| 3.2 | GSS + MADS poll methods | 🔵 | `PollMethod` option-dispatched variants — Tier-6. | — |
| 3.3 | NUPS algorithm | 🔵 | Nonuniform mesh refinement — Tier-6. | — |
| 3.4 | search-then-poll | 🔵 | `SearchFcn` (e.g. a `ga` search before each poll) — Tier-6. | — |
| 3.5 | nonlinear-constraint solver | 🔵 | Augmented-Lagrangian `nonlcon` wrapper — Tier-6. | — |

**🎯 Headline (shipped)**:
[`examples/globaloptim/nonsmooth_patternsearch.m`](../examples/globaloptim/nonsmooth_patternsearch.m)
— a **discontinuous staircase bowl** (a paraboloid quantized into flat
steps).  The gradient solver `fminunc` from `(7,7)` **stalls at f=125**
(its FD gradient is ~0 on every flat step); `patternsearch` steps down
the staircase to the **global f=0 at (2,−3)**.  Plus a nonsmooth V-valley
whose minimum sits at a kink — `patternsearch` lands the exact corner.

**Gating test** (LLVM lane): `gads_t3_patternsearch.m` (the staircase
contrast + the kinked V-valley).

**Compile/Execute wiring (as built)**: dedicated Lowering arm (x0 is the
matrix start at arg 1; 8-arg `(fun,x0,A,b,Aeq,beq,lb,ub)` / 4-arg
`(fun,x0,lb,ub)` / 2-arg forms; **no** hybrid arg → 4-arg runtime);
`patternsearch` + `matlab_gads_patternsearch` added to
`LowerAnonCalls::retypeAnonsForVectorObjective`; `pde_table`
`{PtrTy,PtrTy,PtrTy,PtrTy}`. No classdef / prelude (free function).

**Carve-downs** (→ Tier-6): `PollMethod` (GSS/MADS), NUPS, `SearchFcn`,
nonlinear constraints, `Cache`, vectorized/parallel poll.

---

## 5. Tier-4 — Surrogate optimization (`surrogateopt`) ✅ shipped 2026-05-20

Goal: the sample-efficient global solver for expensive objectives — fits
a radial-basis-function surrogate and adaptively samples it.  Shipped in
[`runtime/toolbox/gads/runtime_gads.cpp`](../runtime/toolbox/gads/runtime_gads.cpp).

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 4.1 | `surrogateopt(fun, lb, ub)` | ✅ | Cubic-RBF surrogate (φ(r)=r³ + linear polynomial tail) fit via the shipped `mldivide` on the (N+n+1)-square interpolation system; adaptive sampling alternates incumbent-perturbation and global-exploration candidates, scored by a merit that trades surrogate value against distance-to-samples with a **cycled weight** {0.3,0.5,0.7,0.95}; eval budget 60+30n; final `fmincon` polish. | `matlab_gads_surrogateopt` |
| 4.2 | integer constraints (`intcon`) | 🔵 | Mixed-integer surrogate — Tier-6. | — |
| 4.3 | nonlinear constraints | 🔵 | Merit-function constraint handling — Tier-6. | — |
| 4.4 | checkpoint / resume | 🔵 | `CheckpointFile` — Tier-6. | — |
| 4.5 | feasibility-only mode | 🔵 | No-objective feasibility — Tier-6. | — |

**🎯 Headline (shipped)**:
[`examples/globaloptim/branin_surrogate.m`](../examples/globaloptim/branin_surrogate.m)
— **Branin's function**, the canonical surrogate-/Bayesian-optimization
benchmark (three equal global minima, f*=0.3979): `surrogateopt` recovers
**f=0.3979**.  Also the six-hump camelback → **global f=−1.0316** (past
its four non-global local minima).  The RBF coefficient solves reuse the
shipped `mldivide`; the adaptive sampling runs over the shared seeded
PRNG.

**Gating test** (LLVM lane): `gads_t4_surrogateopt.m` (Branin 0.3979 +
camelback −1.0316).

**Compile/Execute wiring (as built)**: dedicated Lowering arm
(`surrogateopt(fun, lb, ub)` — no start point, lb/ub at args 1,2 + hybrid
flag injected); `surrogateopt` + `matlab_gads_surrogateopt` added to
`LowerAnonCalls::retypeAnonsForVectorObjective`; `pde_table`
`{PtrTy,PtrTy,PtrTy,F64}`.

**Cross-toolbox antenna demo — deferred.** The roadmap's Yagi-Uda idea
isn't feasible: multi-wire MoM is ANT-Tier-2b (not shipped — only the
closed-form single dipole is), and the shipped `antennaWireSolve` returns
a *struct* (`.VSWR`/`.Zin_re`/…), but struct-field-access on a call result
inside an anonymous-function body yields `none` (a compiler gap).  A
single-dipole VSWR-vs-length `surrogateopt` demo becomes possible once
either (a) a scalar-returning `antennaWireVSWR(L,a,n,f)` runtime wrapper
ships, or (b) anon bodies support `f(...).field`.  Tracked as a Tier-4
follow-on.

**Carve-downs** (→ Tier-6): merit-function tuning surface,
`MaxFunctionEvaluations` option, `intcon` / nonlinear constraints,
checkpoint/resume, parallel batch evaluation, the antenna cross-toolbox
demo above.

---

## 6. Tier-5 — Multiobjective optimization (`gamultiobj` + `paretosearch`) ✅ shipped 2026-05-20

Goal: Pareto-front computation for competing objectives.  The objective
returns a vector of `nobj` values (the vector-out handle ABI, like
`lsqnonlin`); both solvers return the Pareto set (non-dominated decision
points) as a k×nvars matrix.  Shipped in
[`runtime/toolbox/gads/runtime_gads.cpp`](../runtime/toolbox/gads/runtime_gads.cpp).

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 5.1 | `gamultiobj(fun, nvars, …)` | ✅ | **NSGA-II**: fast non-dominated sort + crowding-distance crowded-comparison tournament, BLX-α crossover + Gaussian mutation, elitist (P∪Q) survival; returns the first front. | `matlab_gads_gamultiobj` |
| 5.2 | `paretosearch(fun, nvars, …)` | ✅ | Non-dominated **archive** seeded by scatter sampling, refined by a GPS-style ±Δ poll around archive points; crowding-distance pruning each iteration (essential — the front is a continuum, so the archive must stay bounded). | `matlab_gads_paretosearch` |
| 5.3 | Pareto-front outputs | 🟡 | Single-return `x` (the k×nvars non-dominated set) ships; `[x,fval]` + hypervolume/spread → Tier-6. | — |
| 5.4 | constraint handling | 🔵 | Linear + nonlinear constraints on the multiobjective problem → Tier-6. | — |

**🎯 Headline (shipped)**:
[`examples/globaloptim/pareto_front.m`](../examples/globaloptim/pareto_front.m)
— two conflicting objectives `f1=(x−1)²`, `f2=(x+1)²` pull the design
variable toward opposite targets, so the Pareto-optimal set is the whole
interval `x∈[−1,1]`.  Both `gamultiobj` (NSGA-II) and `paretosearch`
recover the **full trade-off curve** (spanning x=−1 to x=+1) — not a
single compromise point.  Demonstrates that a multiobjective solver
returns a *set*, the defining difference from the single-objective tiers.

**Gating test** (LLVM lane): `gads_t5_multiobjective.m` (bi-objective
Pareto endpoints at ±1 for both solvers).

**Compile/Execute wiring (as built)**: dedicated Lowering arm (nvars at
arg 1; 8-arg `(fun,nvars,A,b,Aeq,beq,lb,ub)` / 4-arg forms; no hybrid);
the **vector-returning** objective handle is retyped by the same
`LowerAnonCalls::retypeAnonsForVectorObjective` recognizer (operand 0);
`pde_table` `{PtrTy,F64,PtrTy,PtrTy}`.  Shared `gads_dominates` /
non-dominated-sort / crowding helpers serve both solvers.

**Carve-downs** (→ Tier-6): `[x,fval]` multi-return + hypervolume/spread
metrics, constraint handling, 3-D Pareto plotting, custom output
functions.

---

## 7. Tier-6 — Carve-down sweep / polish ✅ (focused subset shipped)

Mirrors the established Tier-6 pattern: a focused pass over the highest-value
deferred items. The flagship-`ga` configurability slice shipped; the rest
stay documented as follow-ons (this is the same partial-sweep discipline as
the MPC / System ID Tier-6 tiers).

**Shipped:**

| # | Item | Status | Notes |
|---|------|--------|-------|
| 6.1 | `optimoptions('ga', 'Name', val, …)` | ✅ | Options carrier classdef (`gads_classdefs.m`). Lowering intercepts the name-value form in the constructor-call path (the leading solver-name string is skipped), allocates the zero-arg shell, and writes the named fields (`PopulationSize` / `MaxGenerations` via `matlab_obj_set_f64`; `IntCon` via `matlab_obj_set_mat`). Sentinel `−1` = "use the solver default". |
| 6.2 | integer constraints (`IntCon` for `ga`) | ✅ | The mixed-integer capability. `ga(fun,nvars,…,lb,ub,opts)` routes to `matlab_gads_ga_opts`, which reads `IntCon` and rounds those variables to the nearest feasible integer at init, after every crossover/mutation, and at the final result. The `fmincon` hybrid polish is **auto-skipped** when any variable is integer (a continuous refinement is meaningless). Runtime shares one `gads_ga_core(pop, gens, do_hybrid, isint)` between the default `matlab_gads_ga` and the options entry, so the Tier-1 path is byte-identical. |
| 6.3 | options-bearing `ga(…, opts)` call forms | ✅ | The Lowering dispatch detects the trailing options carrier in the 5-arg `ga(fun,nvars,lb,ub,opts)`, 9-arg `ga(fun,nvars,A,b,Aeq,beq,lb,ub,opts)`, and 10-arg `ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,opts)` (the canonical full signature; `nonlcon` must be `[]` — nonlinear constraints are a follow-on) forms → 6-arg `matlab_gads_ga_opts`. |

**Remaining follow-ons (still 🔵):**

- **`optimoptions` for the other solvers** — `SwarmSize`/`MaxIterations`
  (particleswarm), `MaxIterations` (simulannealbnd / patternsearch /
  surrogateopt). The carrier already declares the fields; only the
  per-solver `_opts` runtime entries + dispatch arms remain.
- **`HybridFcn` / `FunctionTolerance` / `Display`** option knobs (the
  carrier declares them; not yet read by the runtime).
- **`exitflag` / `output` multi-return** across all solvers
  (`[x, fval, exitflag, output] = ga(...)`).
- **`IntCon` for `surrogateopt`** (mixed-integer surrogate).
- **Nonlinear-constraint handles** uniformly across all solvers.
- **Problem-based integration**: `solve(prob, 'Solver', 'ga')` /
  `'particleswarm'` / `'surrogateopt'` routing through the shipped
  `optimproblem`/`solve` expression-DAG.
- **Hybrid functions** beyond `fmincon` (`patternsearch`/`fminsearch`).
- **Vectorized objectives** (`UseVectorized`).
- **Parallel evaluation** (`UseParallel` → pthread fan-out, the Optim
  Tier-5 precedent).
- **Output / plot functions** (`OutputFcn` / `PlotFcn` — non-GUI numeric
  forms; the live plots are carved).

**Headlines:** `examples/globaloptim/gear_train_intga.m` — the classic
Sandgren mixed-integer gear-train design: four integer tooth counts
(12…60) chosen so `(z1·z2)/(z3·z4)` approximates `1/6.931`. `ga` with
`IntCon = [1 2 3 4]` reaches a ratio error ≈ 2.3e-11 — far below any
rounded-continuous guess. Plus `examples/globaloptim/ackley_compare.m` —
the *Compare Global Solvers* walkthrough on the multi-modal Ackley
function, driving `ga` through the canonical 10-arg full signature with an
`optimoptions('ga', …)` carrier (`fminunc` traps at f≈6.56;
`ga`/`particleswarm`/`patternsearch`/`simulannealbnd` all reach f=0).

---

## 8. Carve-outs (explicitly out of scope)

Matching the established roadmap discipline:

- **Optimize Live Editor Task** + all interactive apps (the
  Optimization Explorer / Surrogate plots) — GUI surface.
- **Simulink model optimization** (`optimize Simulink model in
  parallel`, the Optimization Explorer over Simulink) — needs Simulink.
- **Parallel Computing across a cluster** — only single-process pthread
  fan-out is in scope (the Optim Tier-5 precedent); `parpool` clusters
  are out.
- **Custom data-type optimization** (`ga` over permutations / cell
  genomes — e.g. multiprocessor scheduling, TSP) — niche; the real-coded
  + integer forms cover the numeric mainstream.
- **GPU acceleration** of any solver.
- **Live plot functions** (`gaplotbestf`, `psoplotbestf`, …) as figures
  — the numeric `OutputFcn` callback form may ship in Tier-6, but the
  rendered plots are carved.

---

## 9. Dependency summary

```
Tier-1 (ga + particleswarm + simulannealbnd)  ── needs: PRNG, fmincon (hybrid)   ◀── HEADLINE: rastrigin_ga
   ├─ Tier-2 (MultiStart + GlobalSearch)       ── needs: fmincon, classdef plumbing
   ├─ Tier-3 (patternsearch)                    ── needs: PRNG, aug-Lagrangian (Optim)
   ├─ Tier-4 (surrogateopt)                     ── needs: mldivide (RBF), PRNG   ◀── cross-toolbox: Yagi-Uda antenna
   └─ Tier-5 (gamultiobj + paretosearch)        ── needs: Tier-1 GA core + non-dominated sort
        └─ Tier-6 ✅ (optimoptions('ga') + IntCon)  ◀── HEADLINE: gear_train_intga
              └─ follow-ons (exitflag / other-solver options / problem-based / parallel)
```

**Critical new build (not reusable from elsewhere)**: the stochastic
search loops themselves — GA operators (selection/crossover/mutation),
PSO velocity update, SA cooling, RBF surrogate, non-dominated sorting.
Everything else (local solves, PRNG, bounds, handle ABI, classdefs) is
shipped.  **No external dependency.**
