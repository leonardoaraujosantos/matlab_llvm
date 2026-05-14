# Optimization Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Optimization-Toolbox programs.

Source: *Optimization Toolbox User's Guide* (R2026a, 1798 pages,
15 chapters: Getting Started · Setting Up an Optimization · Examining
Results · Steps to Take After Running a Solver · Unconstrained Nonlinear
· Constrained Nonlinear · Multiobjective · Linear / Mixed-Integer Linear
· Problem-Based · Quadratic Programming · Least-Squares · Equation
Solving · Parallel Computing · Argument and Options Reference ·
Functions).

The headline tracer-bullet (the gating example for the whole roadmap)
is [`examples/optim/blade_pitch_opt.m`](../examples/optim/blade_pitch_opt.m):
*use `fmincon` to pick a wind-turbine blade-pitch angle that maximises
generated power subject to a PDE-evaluated von-Mises stress bound*.
This couples the Optim core to the existing PDE Tier-2 elasticity
runtime (`runtime_pde.cpp`); achieving that demo end-to-end is what
closes **Optim-Tier-2** below.

Companion docs: [`feature_status.md`](feature_status.md),
[`roadmap.md`](roadmap.md), [`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md)
(headline demo couples to PDE Tier-2 elasticity), [`sym.md`](sym.md)
(SymPP analytic derivatives feed `checkGradients` / problem-based
auto-Jacobian).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. Tier-1 is
  the smallest end-to-end LLVM-lane loop (`fzero` + `fminbnd` +
  `fminsearch` + `fminunc` BFGS + dense-simplex `linprog` + `lsqnonneg`).
  Tier-2 closes the headline demo (`fmincon` SQP / IP + analytic
  gradient + `quadprog` / `lsqlin` / `lsqnonlin` IP + LM). Tier-3 is
  MILP / cone / semi-infinite / multiobjective. Tier-4 wraps the
  numeric core in the **problem-based** API (`optimvar`, `optimproblem`,
  `solve`). Tier-5 ships result classdefs + REPL renderers + parallel
  polish.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Tier-1 through Tier-5 are all ✅ shipped (2026-05-14).**  The
  remaining work to call the toolbox "fully complete" is the
  `optimoptions` options surface + `[x,fval,exitflag,output]`
  multi-return (the Tier-1 carve-downs) — see §6.
- **Compile/Execute path** (identical pattern across rows): Sema
  registers a new builtin in
  [`lib/Sema/Resolver.cpp::registerBuiltins()`](../lib/Sema/Resolver.cpp);
  type inference rules go in [`lib/Sema/TypeInference.cpp`](../lib/Sema/TypeInference.cpp);
  `matlab.call_builtin @name(...)` is rewritten to
  `llvm.call @matlab_optim_*(...)` inside `LowerTensorOps.cpp` (split
  into a dedicated `LowerOptim.cpp` pass once Optim entries exceed
  ~10 rows — same precedent as PDE / Comm); runtime entries live in a
  new [`runtime/runtime_optim.cpp`](../runtime/runtime_optim.cpp)
  mirroring `runtime_pde.cpp`.
- **Debug / REPL**: every new descriptor type
  (`OptimizationProblem`, `OptimizationVariable`, `OptimizationResult`,
  `OptimizationExpression`) needs a renderer in
  [`runtime/runtime_debug.cpp`](../runtime/runtime_debug.cpp)
  (`matlab_ws_set_*` family) and a DAP child-walker — same pattern as
  `StaticStructuralResults` in `pde_classdefs.m` and `tf`/`ss` in
  `cst_classdefs.m`.
- **No external solver dependencies**: matching the project's
  hand-coded LAPACK-style precedent (PDE's Lanczos shift-invert,
  Control's Schur / Lyap / Riccati), Optim is hand-coded too —
  **no Ipopt, NLopt, HiGHS, OSQP, GLPK, Coin-OR**. Krylov + LDLᵀ +
  Cholesky already exist in `runtime/runtime_sparse.cpp` +
  `runtime/matlab_runtime.cpp`.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Optim code yet)

The following primitives already exist and **do not need to be re-built**
for Optim — every solver row below leans on them.

| Group | Surface | Location | Notes |
|---|---|---|---|
| Dense linear algebra | `mldivide`, `lu`, `qr`, `chol`, `eig` (sym+non-sym), `schur`, `hess`, `svd`, `pinv` | `runtime/matlab_runtime.cpp` | Hand-coded LAPACK-style. Suffices for ≤ 5 k-DOF QP/SQP factor-and-solve. |
| Sparse linalg | CSR descriptor, triplet→CSR builder, sparse matvec / dot, **PCG (Jacobi)**, **GMRES(30) + ILU(0)**, MINRES | `runtime/runtime_sparse.cpp` | Direct fuel for sparse `linprog` / `quadprog` / `lsqlin` Newton steps. |
| Function-handle ABI | `void *fn_p` cast to typed function pointer inside runtime entries (see `matlab_pdepe` for the template) | `runtime/matlab_runtime.cpp` §pdepe | Same shape `fzero` / `fminunc` / `fmincon` need: cast `void *` to `double(*)(double,...)`. |
| ODE solvers | `ode23s`, `ode45`, `ode23` (function + vector forms) | `runtime/matlab_runtime.cpp` | Used by Tier-2 simulation-objective examples (`fmincon` over ODE response). |
| Classdef hub + operator overloads | `tf`/`ss`/`zpk` (CST), `femodel` (PDE), `RFCktAmplifier` (RF); operator dispatch in `lib/MLIR/Lowering.cpp` | `runtime/cst_classdefs.m`, `runtime/pde_classdefs.m` | Pattern to mirror for `OptimizationProblem` / `OptimizationVariable` / `OptimizationExpression`. |
| Live-object registry | `matlab_obj_new(class_id)` + `matlab_obj_set_*` accessors | `runtime/matlab_runtime.cpp` §obj | The host for `OptimizationProblem` field bags. |
| Class auto-prelude | `tools/matlabc/main.cpp` prelude table | (new) `optim_classdefs.m` | When user mentions `optimproblem` / `optimvar`, the compiler auto-prepends `optim_classdefs.m` — the exact PDE pattern. |
| Sema builtin registration | `Resolver::registerBuiltin(name)` + `registerBuiltins()` array | `lib/Sema/Resolver.cpp` | Add Optim names to the array; per-builtin shape/dtype rules go in `lib/Sema/TypeInference.cpp`. |
| MLIR lowering | `matlab.call_builtin @name` → `llvm.call @matlab_*` rewrites | `lib/MLIR/Passes/LowerTensorOps.cpp` | Extend now; split into a dedicated `LowerOptim.cpp` once Optim row count > ~10. |
| Debug / REPL renderers | `matlab_ws_set_*` family + DAP frame hooks | `runtime/runtime_debug.cpp` | Plus the sym pretty-printer pattern for expression-tree variables. |
| Symbolic / autodiff | SymPP (`lib/Sym/`) — symbolic Jacobian + Hessian | `lib/Sym/` | `checkGradients` and the problem-based auto-derivative path lean on SymPP for analytic gradients when available. |

---

## 2. Tier-1 — Smallest end-to-end Optim loop (✅ shipped 2026-05-14)

Goal: `fminunc` (quasi-newton BFGS), `fminbnd`, `fzero`, scalar `fsolve`,
dense `linprog`, `lsqnonneg` running on the LLVM lane.  All six numeric
solvers ship single-return; the runtime core is hand-coded in
`runtime/runtime_optim.cpp` with no external dependencies.

| # | Function | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 1.1 | `fzero` | ✅ | Brent (van Wijngaarden–Dekker–Brent); bracket-expansion when given a scalar guess | `matlab_optim_fzero` / `_iv` |
| 1.2 | `fminbnd` | ✅ | Brent `localmin` — golden-section + parabolic interpolation | `matlab_optim_fminbnd` |
| 1.3 | `fminsearch` | ✅ | Nelder–Mead downhill simplex; MATLAB's 5 % initial-simplex construction | `matlab_optim_fminsearch` |
| 1.4 | `fminunc` quasi-newton | ✅ | BFGS (inverse-Hessian form) + FD gradient + backtracking Armijo line search | `matlab_optim_fminunc` |
| 1.5 | `linprog` (dense simplex) | ✅ | 2-phase tableau simplex; 3-arg + 7-arg forms; lb-shift, finite-ub rows, equalities | `matlab_optim_linprog` / `_linprog3` |
| 1.6 | `lsqnonneg` | ✅ | Lawson–Hanson active-set NNLS; passive-set sub-problems via GE-with-partial-pivoting | `matlab_optim_lsqnonneg` |
| 1.9 | `fsolve` 1-D (scalar) | ✅ | Newton with FD derivative; bracket-expansion + Brent fallback when Newton stalls | `matlab_optim_fsolve_scalar` |

**Gating tests** (all green on the LLVM lane, `.skip-emit-*` markers on
the C/C++/Python/TS lanes):
`optim_fzero_brent.m`, `optim_fminbnd_golden.m`,
`optim_fminsearch_rosenbrock.m`, `optim_fminunc_bfgs_rosenbrock.m`,
`optim_linprog_diet.m`, `optim_lsqnonneg_smoke.m`,
`optim_fsolve_scalar.m`.

**Compile/Execute wiring**:
- Sema registers the seven names in `lib/Sema/Resolver.cpp`; scalar
  returns (`fzero`/`fminbnd`/`fsolve`) typed in `TypeInference.cpp`
  alongside the array-returning solvers.
- `LowerTensorOps.cpp`: single-signature solvers (`fminbnd`, `fsolve`,
  `fminsearch`, `fminunc`, `lsqnonneg`) sit in the shared dispatch
  table; `fzero` and `linprog` carry two operand shapes each so they
  use hand-rolled dispatch blocks (the table's name→signature match is
  single-shot).  `linprog` boxes scalar operands (`beq = 3`) via
  `matlab_mat_from_scalar`.
- `LowerAnonCalls.cpp`: `retypeAnonsForVectorObjective` is a pre-pass
  modelled on `retypeAnonsForVectorODE` — when an anonymous objective
  is passed to `fminsearch` / `fminunc`, its sole block arg `x` is
  retyped f64 → ptr so the body's `x(i)` subscripts lower against a
  matrix base.

**Tier-1 carve-down** (the two non-numeric rows of the original plan):
- **1.7 `optimoptions` / `optimset`** — Tier-1 takes solver options as
  a *plain MATLAB struct* (`opts.TolX = 1e-10; …`), the established
  `odeset` precedent in this codebase.  The dedicated variadic
  name-value option-object builders are deferred to a follow-up.
- **1.8 multi-return `[x,fval,exitflag,output]`** — Tier-1 ships
  single-return (`x = solver(…)`).  The cached-second-return machinery
  (the `[t,y] = ode45(…)` precedent) for `fval` / `exitflag` / `output`
  / `lambda` is deferred; none of the Tier-1 gating tests need it.

**REPL / Debug**: Tier-1 results are scalars or plain matrices →
existing renderers suffice.  No new `runtime_debug.cpp` work until the
problem-based descriptor types arrive in Tier-4.

---

## 3. Tier-2 — Constrained + nonlinear least squares (✅ shipped 2026-05-14)

Goal: close `examples/optim/blade_pitch_opt.m` and ship the
general-constrained and nonlinear-least-squares solvers.  Two
hand-coded cores in `runtime/runtime_optim.cpp` back every Tier-2
solver:

- **`al_minimize`** — augmented-Lagrangian method (Powell-Hestenes-
  Rockafellar) with a bound-projected BFGS inner solver.  Bounds are
  enforced by projection; linear and nonlinear inequalities and linear
  equalities are folded into the augmented Lagrangian; nonlinear
  constraint gradients come from forward finite differences.  Backs
  `fmincon`, `quadprog`, `lsqlin`.
- **`lm_solve`** — Levenberg-Marquardt for nonlinear least squares:
  FD Jacobian, damped normal equations `(JᵀJ + λ·diag) p = −Jᵀr`,
  λ shrinking on success / growing on failure.  Backs `lsqnonlin`,
  `lsqcurvefit`, `fsolve` (N-D).

| # | Function | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 2.1–2.3 | `fmincon` (all algorithms) | ✅ | One augmented-Lagrangian method serves the SQP / interior-point / active-set / trust-region-reflective `Algorithm` choices.  Handles bounds + linear (in)equalities + nonlinear inequalities. | `matlab_optim_fmincon` |
| 2.5 | Finite-difference gradients / Jacobians | ✅ | Forward differences, used internally by every Tier-2 solver. | (internal) |
| 2.8 | `quadprog` (convex QP) | ✅ | Augmented-Lagrangian core with the analytic objective ½xᵀHx + fᵀx. | `matlab_optim_quadprog` |
| 2.10 | `lsqlin` (constrained linear LS) | ✅ | Augmented-Lagrangian core with the analytic objective ½‖Cx − d‖². | `matlab_optim_lsqlin` |
| 2.11 | `lsqnonlin` (nonlinear LS) | ✅ | Levenberg-Marquardt; bounds enforced by projecting the trial step. | `matlab_optim_lsqnonlin` |
| 2.12 | `lsqcurvefit` (curve fitting) | ✅ | LM over the residual `fun(x,t) − ydata`; the model handle is evaluated per data point (ABI `double(*)(matlab_mat*, double)`). | `matlab_optim_lsqcurvefit` |
| 2.13 | `fsolve` (N-D system) | ✅ | LM on ‖F(x)‖²; the scalar form still routes to the Tier-1 Newton + Brent solver. | `matlab_optim_fsolve` |

**Headline tracer-bullet** — `examples/optim/blade_pitch_opt.m`
(gating copy `test/Run/optim_blade_pitch_3d.m`): a cross-toolbox
wind-turbine blade-pitch optimisation.
1. **FEM characterisation** (PDE Tier-2): the script body runs the
   full `pde_mesh_cuboid_tet` → `pde_assemble_elast_3d` →
   `pde_face_pressure_3d` → `pde_apply_fixed_3d` → `mldivide` →
   `pde_von_mises_3d` pipeline on a blade-root segment under a
   reference windward pressure, and extracts the stress-per-unit-load
   coefficient `k_stress`.  Linear elasticity makes peak stress
   *exactly* proportional to the load, so this is an exact surrogate.
2. **`fmincon` optimisation**: maximises an analytic aerodynamic-power
   surrogate subject to the FEM-derived stress constraint and pitch
   bounds.  The optimiser drives the pitch up to 17° — where the
   stress limit binds, below the 25° pitch bound.

Because anonymous functions cannot capture workspace variables and a
named function referenced only via `@handle` does not currently lower,
the demo's objective / constraint are captureless anonymous functions
and the FEM-derived `k_stress` is written as a literal (the script
asserts it matches the live FEM result).

**Gating tests** (all green on the LLVM lane, `.skip-emit-*` on the
others): `optim_fmincon_rosenbrock_disk.m` (disk-constrained
Rosenbrock + bound / linear-(in)equality forms),
`optim_quadprog_portfolio.m` (minimum-variance portfolio + verifiable
QP forms), `optim_lsqlin_smoke.m`, `optim_lsqnonlin_exp_fit.m`
(exponential fit), `optim_fsolve_nd.m` (2×2 / 3×3 systems + scalar
fall-through), `optim_lsqcurvefit_smoke.m`, and
`optim_blade_pitch_3d.m` (the headline).

**Compile/Execute wiring**:
- Sema registers `fmincon` / `quadprog` / `lsqlin` / `lsqnonlin` /
  `lsqcurvefit` in `Resolver.cpp`; all are array-returning in
  `TypeInference.cpp`.  `fsolve` is typed conditionally — scalar `x0`
  → scalar return, vector `x0` → vector return.
- `LowerTensorOps.cpp`: `lsqcurvefit` (single 4-arg shape) sits in the
  shared dispatch table; `fmincon` / `quadprog` / `lsqlin` /
  `lsqnonlin` go through a generic multi-arity block that maps the
  first N call operands to the first N fixed-ABI slots and null-pads
  the rest (the runtime's `mat_absent()` treats a null ptr as an
  omitted argument); `fsolve` has a hand-rolled scalar-vs-N-D block.
- `LowerAnonCalls.cpp`: `retypeAnonsForVectorObjective` (extended from
  the Tier-1 version) retypes objective / constraint / residual /
  model handles' vector block args f64 → ptr.  `lsqcurvefit`'s model
  takes `(params, t)` — only `params` is retyped; `t` stays a scalar
  f64 so element-wise model expressions lower cleanly.

**Tier-2 carve-down** (deferred rows of the original plan):
- **2.1/2.2/2.3 distinct algorithms** — one augmented-Lagrangian
  method serves every `fmincon` `Algorithm` choice.  The separate
  SQP / interior-point / active-set / trust-region-reflective
  implementations (and their distinct convergence/feasibility
  behaviours) are deferred.
- **Nonlinear *equality* constraints** in `fmincon` — Tier-2 handles
  nonlinear *inequalities* (`nonlcon` returns `c(x) ≤ 0`); nonlinear
  equalities are deferred to Tier-3.
- **2.4 analytic gradient / Hessian intake + 2.6 `checkGradients`** —
  deferred; they need multi-output objective-handle support
  (`[f,g] = fun(x)`).  Tier-2 uses finite-difference gradients
  throughout.
- **2.7 `fminunc` trust-region** — deferred; the Tier-1 BFGS
  `fminunc` already serves `fminunc`.
- **2.9 `linprog` interior-point** — deferred; the Tier-1 dense
  2-phase simplex already solves LPs.
- **2.14 iterative display / `OutputFcn` hooks** — deferred (cosmetic;
  needs the multi-return / options surface).

**REPL / Debug**: Tier-2 results are plain matrices → existing
renderers suffice; no new `runtime_debug.cpp` work until the
problem-based descriptor types arrive in Tier-4.

---

## 4. Tier-3 — MILP, cone, minimax, semi-infinite (✅ shipped 2026-05-14)

Every Tier-3 solver is a **reformulation on top of the Tier-1/2
cores** — no new numerical kernel was needed.

| # | Function | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 3.1 | `intlinprog` | ✅ | Depth-first branch-and-bound over `linprog_core` (dense 2-phase simplex): tighten bounds per node, prune by the incumbent objective, branch on the most-fractional integer variable. | `matlab_optim_intlinprog` |
| 3.4 | `coneprog` | ✅ | Single second-order cone constraint `‖Asc·x + bsc‖ ≤ dscᵀx + γ` handled as one nonlinear inequality through `al_minimize`. | `matlab_optim_coneprog` |
| 3.5 | `fminimax` | ✅ | Epigraph reformulation — with `z = [x; γ]`, minimise `γ` s.t. `F_i(x) − γ ≤ 0`, through `al_minimize`. | `matlab_optim_fminimax` |
| 3.6 | `fgoalattain` | ✅ | Epigraph reformulation — minimise `γ` s.t. `F_i(x) − weightᵢ·γ ≤ goalᵢ`. | `matlab_optim_fgoalattain` |
| 3.7 | `fseminf` | ✅ | Outer-approximation: minimise over the current finite set of sampled `w`-points, then add the most-violating `w` from a fine grid; iterate.  Single semi-infinite constraint, per-point handle ABI `double(*)(matlab_mat*, double)`. | `matlab_optim_fseminf` |

**Gating tests** (all green on the LLVM lane, `.skip-emit-*` on the
others): `optim_intlinprog_knapsack.m` (0/1 knapsack + tighter-capacity
variant), `optim_intlinprog_assignment.m` (3×3 assignment — the
totally-unimodular polytope means the root LP relaxation is already
integral), `optim_fminimax_smoke.m` (three balanced paraboloids),
`optim_fgoalattain_smoke.m` (coupled goal attainment + bounded form),
`optim_coneprog_smoke.m` (max over the unit disk + scaled cone),
`optim_fseminf_smoke.m` (linear-in-`w` semi-infinite constraint).

**Compile/Execute wiring**: all five register in `Resolver.cpp`,
type-infer as array-returning in `TypeInference.cpp`, and slot into the
generic multi-arity dispatch block in `LowerTensorOps.cpp` (first N
call operands → first N fixed-ABI slots, null-padded; scalar operands
like `coneprog`'s `gamma` are boxed via `matlab_mat_from_scalar`).
`retypeAnonsForVectorObjective` in `LowerAnonCalls.cpp` retypes the
`fminimax` / `fgoalattain` objective handle and the `fseminf` objective
+ per-point constraint handles.

**Tier-3 carve-down** (deferred rows of the original plan):
- **3.2/3.3/3.8/3.9/3.10 — distinct algorithm variants** (`quadprog`
  active-set / trust-region-reflective, `lsqlin` / `lsqnonlin`
  trust-region-reflective, `linprog` active-set).  These are alternate
  algorithms for solvers that **already work** via `al_minimize` /
  `lm_solve` / the dense simplex — the same posture as Tier-2's
  SQP/IP/active-set collapse for `fmincon`.  The named sub-algorithms
  (and their distinct convergence/feasibility behaviours) are deferred.
- **Multi-cone `coneprog`** — Tier-3 supports a single second-order
  cone (the common SOCP shape).  An array of `secondordercone`
  objects needs the classdef + struct-array machinery and is deferred.
- **Full `fseminf` ABI** — the MATLAB `seminfcon` returns a variable
  number of outputs (`[c, ceq, K1, …, Kntheta, s]`); Tier-3 supports a
  single semi-infinite constraint via the per-point handle.  The
  multi-output ABI is deferred (needs multi-output handle support).
- **Nonlinear equality constraints** in `fmincon` (carried over from
  Tier-2) remain deferred — `al_minimize` has the equality machinery
  but the `nonlcon` handle ABI only returns the inequality vector.

---

## 5. Tier-4 — Problem-based API (✅ shipped 2026-05-14)

The **operator-overloaded expression layer** on top of the Tier-1–3
numerics.  The expression DAG lives in `runtime/runtime_optim.cpp`
(the `matlab_optim_pb_*` family); `runtime/optim_classdefs.m` is the
thin classdef layer whose operator overloads forward to it.

**Architecture:**
- A global scalar expression-DAG node pool in the runtime.  Node
  kinds: VAR, CONST, ADD/SUB/NEG/MUL/DIV/POW, and the relation nodes
  LE/GE/EQ.  Each `matlab_optim_pb_*` builder appends a node and
  returns its id as an f64.
- `OptimizationExpression` is a thin classdef boxing one `Id`
  property.  Its operator methods (`plus`/`minus`/`uminus`/`mtimes`/
  `times`/`mrdivide`/`rdivide`/`mpower`/`power`) forward to a runtime
  builder and re-wrap the returned id; the relational methods
  (`le`/`ge`/`eq`) return the bare constraint node id.  Scalar mixing
  (`2*x`) is handled by the existing operator-dispatch scalar-boxing
  (the class is added to the box list in `Lowering.cpp`), which boxes
  the constant through the 1-arg constructor → a CONST node.
- `OptimizationProblem` is a classdef with `Objective` / `Constraints`
  / `Maximize` properties.  `prob.Objective = expr` and the nested
  `prob.Constraints.<name> = (expr <= rhs)` both go through the
  generic property-/struct-field-assignment path — no special-casing
  needed.
- `solve(prob)` routes via the generic class-pinned-first-arg method
  dispatch (the `step`/`bode` precedent) to `OptimizationProblem.solve`,
  which hands the classdef object to `matlab_optim_pb_solve`.  That
  reads `Objective` / `Constraints` / `Maximize` off the object,
  **collects only the variables actually referenced** by this
  problem, reduces the DAG (linear reduction detects LP/MILP), and
  dispatches: linear + integer → the branch-and-bound MILP solver;
  everything else (LP, QP, nonlinear) → `al_minimize` with a
  DAG-evaluation objective and a DAG-evaluation nonlinear-constraint
  closure.  Returns the solution as a column vector in
  variable-creation order.
- `optimvar()` / `optimintvar()` / `optimproblem()` are factory
  functions in `optim_classdefs.m`; their results are pinned by a
  `pinnedOfRhs` rule in `Resolver.cpp` (resolved by name, since the
  classdef prelude is appended after the user script).  The umbrella
  file is auto-prepended via the file-compilation prelude table in
  `tools/matlabc/main.cpp`.

**Gating tests** (all green on the LLVM lane, `.skip-emit-*` on the
others): `optim_pb_lp.m` (diet-style LP + a `Maximize` problem),
`optim_pb_qp.m` (bowl / constrained / cross-term QP), `optim_pb_milp.m`
(integer program + 0/1 knapsack pick), `optim_pb_nonlinear.m` (quartic,
nonlinear objective + nonlinear `a²+b²≤1` constraint, bounded cubic).

**`solve` name collision**: SymPP's `solve` is gated on a *sym* first
argument, so `solve(prob)` with an `OptimizationProblem` falls straight
through to the generic method dispatch — no extra wiring needed.

**Tier-4 carve-down** (deferred from the original plan):
- **Vector / matrix optimisation variables** — Tier-4 supports
  *scalar* variables (`optimvar()` with no arguments); a
  multi-variable problem uses several scalar `optimvar()`s.  The
  `optimvar('name', [n m], 'Type', ...)` string/size/keyword form is
  deferred (a vector-valued DAG with shape-carrying nodes is a
  substantial extension).  Because `solve` returns a plain solution
  vector (not a `sol.x` struct), the variable name is cosmetic and is
  dropped.
- **`eqnproblem`** (equation problems → `fsolve`), **`optimexpr`**,
  **`prob2struct`**, **`evaluate(expr, x)`**, **`show`/`write`**, and
  **problem-based `checkGradients`** — deferred.
- **Nonlinear equality constraints** in problem-based `solve` — only
  `<=` / `>=` nonlinear constraints reduce to the AL nonlinear-
  inequality closure; nonlinear `==` is deferred (carried over from
  Tier-2/3).
- **REPL/Debug renderers** for the descriptor types — deferred; the
  classdef objects render through the generic object inspector today.

---

## 6. Tier-5 — Problem-based equation solving (✅ shipped 2026-05-14)

Tier-5 ships **`eqnproblem`** — problem-based equation solving, the
natural completion of the problem-based story.  It reuses the Tier-4
expression DAG and the Tier-2 `lm_solve` core, so the implementation
is thin:

- `EquationProblem` classdef (`Equations` struct property + a `solve`
  method) and the `eqnproblem()` factory in `optim_classdefs.m`.
- `prob.Equations.<name> = (lhs == rhs)` — each `==` expression
  contributes a residual node (the existing `PBK_EQ` DAG node, which
  evaluates to its canonical `lhs − rhs`).
- `solve(prob)` routes — via the same generic class-method dispatch —
  to `matlab_optim_pb_solve_eqn`, which reads the `Equations` struct,
  collects the referenced variables, builds the residual vector
  `F(x)` from the equation nodes, and solves `F(x) = 0` by
  Levenberg-Marquardt.

**Gating test**: `optim_pb_eqn.m` — a linear 2×2 system, a scalar
nonlinear equation (`x³ + x == 10`), and a nonlinear 2×2 system, all
solved from the origin.

**Tier-5 carve-down — blocked on earlier carve-downs.** The remaining
rows of the original Tier-5 plan all depend on features that were
themselves carved down in Tier-1 / Tier-4, so they are deferred as a
group.  "Completing" them is really the work of finishing those
earlier carve-downs first:

| Original row | Blocked on |
|---|---|
| 5.1 `OptimizationResult` classdef | the `[x,fval,exitflag,output]` multi-return surface (Tier-1 row 1.8) — there is no `output` struct to wrap yet |
| 5.2 vector / `OptimizationVariableArray` polish | vector/matrix `optimvar` (Tier-4 carve-down) |
| 5.3 `optimwarmstart` | 5.1 (result objects to resume from) |
| 5.4 `UseParallel` FD-Jacobian over pthreads | the `optimoptions` options surface (Tier-1 row 1.7) — no way to enable it — plus user-handle FD is not thread-safe against the runtime's thread-local state |
| 5.5 `optim.problemdef.ProblemRule` namespace | n/a — explicitly a stub in the original plan |

The genuine "finish the Optimization Toolbox" follow-up is therefore
the **`optimoptions` options surface (1.7) + multi-return
`[x,fval,exitflag,output]` (1.8)**, after which `OptimizationResult`
/ `optimwarmstart` / `UseParallel` become straightforward.

---

## 7. Out of scope / carved out

- **Optimize Live Editor task** — UI-only feature; no Live Editor host.
- **App Designer / `optimtool`** — interactive GUI carved out across
  all toolboxes.
- **Code generation (MATLAB Coder)** to MEX / C — `matlabc` already
  emits C/C++; the Coder UI is not a goal.
- **Distributed Computing** parallelisation — `UseParallel=true` is
  degraded to pthread fan-out (Tier-5); cluster execution is out.
- **Global Optimization Toolbox** (`ga`, `gamultiobj`, `particleswarm`,
  `simulannealbnd`, `surrogateopt`, `paretosearch`, `MultiStart`,
  `GlobalSearch`) — separate toolbox, separate roadmap.
- **Symbolic auto-Hessian** via `prob2matlabfunction` — punted to a
  later cross-cutting SymPP / Optim integration row.
- **Complex-valued objective** path — the toolbox itself warns about
  this (User's Guide §2-18); we exclude it.
- **Legacy `optimset` name aliases** for retired option names — only
  R2026a names honoured.

---

## 8. Critical files

**New**:
- `runtime/runtime_optim.cpp` — all `matlab_optim_*` C-ABI entries.
- `runtime/optim_classdefs.m` (Tier-4) — `OptimizationVariable`,
  `OptimizationExpression`, `OptimizationInequality`,
  `OptimizationEquality`, `OptimizationProblem`, `EquationProblem`,
  `OptimizationOptions`, `OptimizationResult`. Operator overloads
  forward to `matlab_optim_expr_*` runtime entries.
- `examples/optim/blade_pitch_opt.m` (Tier-2 headline).
- `examples/optim/sudoku_intlinprog.m` (Tier-3) + ~12 small examples
  mirroring User's-Guide canonicals.

**Extended**:
- `lib/Sema/Resolver.cpp` — extend `registerBuiltins()` with every
  Optim function name.
- `lib/Sema/TypeInference.cpp` (or equivalent) — shape/dtype rules
  for Optim builtins.
- `lib/MLIR/Passes/LowerTensorOps.cpp` — initially extend with Optim
  dispatch table, then split into `lib/MLIR/Passes/LowerOptim.cpp`
  once > ~10 rows.
- `lib/MLIR/Lowering.cpp` — register `optim_classdefs.m` operator
  overloads alongside CST / PDE.
- `tools/matlabc/main.cpp` — add `optim_classdefs.m` to the prelude
  auto-include table.
- `runtime/runtime_debug.cpp` — add
  `matlab_ws_set_OptimizationProblem` / `_Variable` / `_Expression`
  renderers + DAP child-walker.
- `CMakeLists.txt` — wire `runtime/runtime_optim.cpp`.
- `test/Run/run_tests.sh` — picks up new `optim_*.m` tests via
  existing glob.

---

## 9. Verification — as shipped

**Gating tests** — 25 `test/Run/optim_*.m` + 4 `test/Run/regress_*.m`,
all green on the LLVM lane (each carries `.skip-emit-*` markers for
the C/C++/Python/TS lanes).  Every test checks the solver against a
known optimum / root via the `if abs(...) < tol; disp(1); …` pattern:

- Tier-1: `optim_fzero_brent`, `optim_fminbnd_golden`,
  `optim_fminsearch_rosenbrock`, `optim_fminunc_bfgs_rosenbrock`,
  `optim_linprog_diet`, `optim_lsqnonneg_smoke`, `optim_fsolve_scalar`.
- Tier-2: `optim_fmincon_rosenbrock_disk` (the disk-Rosenbrock optimum
  matches the MATLAB-documented `[0.7864, 0.6177]`),
  `optim_quadprog_portfolio`, `optim_lsqlin_smoke`,
  `optim_lsqnonlin_exp_fit`, `optim_fsolve_nd`,
  `optim_lsqcurvefit_smoke`, `optim_blade_pitch_3d` (the headline,
  `fmincon` ↔ PDE elasticity).
- Tier-3: `optim_intlinprog_knapsack`, `optim_intlinprog_assignment`,
  `optim_fminimax_smoke`, `optim_fgoalattain_smoke`,
  `optim_coneprog_smoke`, `optim_fseminf_smoke`.
- Tier-4/5: `optim_pb_lp`, `optim_pb_qp`, `optim_pb_milp`,
  `optim_pb_nonlinear`, `optim_pb_eqn`.
- Compiler-limitation regressions (fixed during the Optim work):
  `regress_logical_ops`, `regress_degree_trig`,
  `regress_anon_call_after_pass`, `regress_fprintf_reduction`.

**Examples** — `examples/optim/` carries 16 illustrative programs
(one per major feature + the `blade_pitch_opt` headline) plus a
`README.md`.  Every example compiles and executes correctly through
the `-emit-llvm` → native lane with MATLAB-correct results.

**REPL / compile-execute matrix**:
- **Solver-based** (`fzero` … `fminimax` / `fgoalattain` / `fseminf`)
  — compile/execute ✅, REPL ✅.  Includes vector-objective anons
  (`@(x) x(1)*x(1) + x(2)*x(2)`) defined on one turn and handed to a
  solver on a later turn.
- **Problem-based** (`optimvar` / `optimproblem` / `eqnproblem` /
  `solve`) — compile/execute ✅, REPL ✅.  The expression DAG is built
  across turns and `solve` dispatches just as in the file lane.
- 15 of the 16 `examples/optim/` programs run unchanged in the REPL.
  `blade_pitch_opt` is file-lane-only — its gap is *not* in the
  optimisation path but a pre-existing cross-turn type-inference
  issue: a `max(...)` reduction result stored to the workspace comes
  back matrix-typed, so a later turn's `if abs(k_stress - ...) < ...`
  builds a tensor-typed condition `scf.if` rejects.

End-to-end: `cmake --build build && test/Run/run_tests.sh build/matlabc`.
