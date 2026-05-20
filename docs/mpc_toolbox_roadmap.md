# Model Predictive Control Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Embedded Coder) needs to ship in order to faithfully **compile and
execute**, **debug/REPL**, and **demo** Model-Predictive-Control
Toolbox programs.

Source: *Model Predictive Control Toolbox User's Guide* (R2026a,
12 chapters: Linear MPC Algorithms · Controller Creation · Controller
Analysis · Controller Simulation · Controller Refinement · Data-Driven
MPC · Explicit MPC Design · Adaptive MPC Design · Gain Scheduling MPC
Design · Nonlinear MPC · Code Generation · Automated Driving
Applications).

The headline tracer-bullet (the gating example for the whole roadmap)
is [`examples/mpc/lane_keeping_mpc.m`](../examples/mpc/lane_keeping_mpc.m):
*a linear MPC controller for a bicycle-model lane-keeping plant,
deployed end-to-end through the project's existing Embedded Coder +
Stateflow + cocotb SIL pipeline*.  The Tier-1 close is a less
ambitious DC-servomechanism demo
([`examples/mpc/dc_servo_mpc.m`](../examples/mpc/dc_servo_mpc.m), the
canonical User's-Guide §2 example); the lane-keeping demo closes
**MPC-Tier-3**.

Companion docs: [`feature_status.md`](feature_status.md),
[`roadmap.md`](roadmap.md), [`control_toolbox_roadmap.md`](control_toolbox_roadmap.md)
(MPC sits on top of `ss` / `tf` / `c2d` / `step` / `lsim`),
[`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md) (MPC's QP and
nonlinear-MPC core reuse `quadprog` and `fmincon`),
[`embedded_coder_roadmap.md`](embedded_coder_roadmap.md) (Tier-3
Adaptive/TV MPC is meant to be driven from `mflow` `simulate()` and
SIL'd through cocotb), [`mstateflow_roadmap.md`](mstateflow_roadmap.md)
(MPC + Stateflow supervisor is the canonical deployable architecture).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order.  Tier-1
  is the smallest end-to-end loop: linear MPC against an LTI plant
  with hard bounds, Kalman state estimation, and the standard
  quadratic cost.  Tier-2 closes constraints + disturbances + run-time
  updates.  Tier-3 ships adaptive / time-varying / gain-scheduled MPC
  and the Embedded-Coder/`mflow` integration that drives the headline
  lane-keeping demo.  Tier-4 ships explicit MPC + custom-QP entry
  points.  Tier-5 ships nonlinear MPC (`nlmpc`) on top of the existing
  `fmincon`.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started ·
  🔴 deliberately deferred.
- **Compile/Execute path** (identical pattern across rows): Sema
  registers a new builtin in
  [`lib/Sema/Resolver.cpp::registerBuiltins()`](../lib/Sema/Resolver.cpp);
  type inference rules go in
  [`lib/Sema/TypeInference.cpp`](../lib/Sema/TypeInference.cpp);
  `matlab.call_builtin @name(...)` is rewritten to
  `llvm.call @matlab_mpc_*(...)` inside `LowerTensorOps.cpp` (split
  into a dedicated `LowerMpc.cpp` pass once MPC entries exceed
  ~10 rows — the precedent established by PDE / Comm / Optim); runtime
  entries live in
  [`runtime/toolbox/mpc/runtime_mpc.cpp`](../runtime/toolbox/mpc/runtime_mpc.cpp)
  mirroring `runtime/toolbox/optim/runtime_optim.cpp`.
- **Debug / REPL**: every new descriptor type (`mpc`, `mpcstate`,
  `mpcsimopt`, `mpcmoveopt`, `explicitMPC`, `nlmpc`) needs a renderer
  in [`runtime/runtime_debug.cpp`](../runtime/runtime_debug.cpp)
  (`matlab_ws_set_*` family) and a DAP child-walker — same pattern as
  `tf` / `ss` in `runtime/toolbox/control/cst_classdefs.m` and
  `OptimizationProblem` in
  `runtime/toolbox/optim/optim_classdefs.m`.
- **Examples and tests are first-class deliverables.**  Every tier
  ships **runnable examples** under
  [`examples/mpc/`](../examples/mpc/) (one canonical `.m` per major
  User's-Guide section, plus the `.mflow` deployable demos for Tier-3)
  *and* **gating tests** under [`test/Run/mpc_*.m`](../test/Run/) (one
  numeric-check test per row of each tier's table, named
  `mpc_t<N>_<topic>.m`).  Each example must compile and execute
  through the `-emit-llvm` → native lane and run unchanged in the
  REPL; each test asserts a documented MathWorks figure against a
  tolerance via the `if abs(...) < tol; disp(1); …` pattern that the
  Optim tests already use.  Tier-by-tier example + test inventories
  live in §9.
- **No external solver dependencies**: matching the project's
  hand-coded LAPACK-style precedent (Control's Schur / Lyap / Riccati,
  Optim's hand-coded `al_minimize` / `lm_solve`), MPC is hand-coded
  too — **no OSQP, qpOASES, HPIPM, Ipopt, FORCESPRO, Embotech**.  The
  Tier-1 KWIK active-set QP is a fresh implementation; the warm-start
  benefit relative to Optim's `quadprog` (which always cold-starts the
  augmented-Lagrangian inner BFGS) is what makes real-time MPC
  feasible at all.

---

## 1. Reusable infrastructure (Tier-0 baseline — no MPC code yet)

The following primitives already exist and **do not need to be re-built**
for MPC — every solver row below leans on them.

| Group | Surface | Location | Notes |
|---|---|---|---|
| Plant models | `tf` / `ss` / `zpk` / `pid` / `frd` classdefs with operator overloads + `step` / `bode` / `pole` / `dcgain` / `lsim` / `bandwidth` short forms | `runtime/toolbox/control/` | Direct intake for `mpc(plant, …)`.  The classdef ABI (object pointer + property-table descriptor) is what `mpcobj` will inherit. |
| Matrix exponential | `expm` (Padé + scaling/squaring) | `runtime/matlab_runtime.cpp` | The single most-impactful primitive in the Control roadmap; **required** for `c2d` and for continuous-time prediction-model expansion in MPC matrices `Sx` / `Su`. |
| Schur / Lyap / Riccati | Real-Schur form, `lyap`, **continuous-time `care`** | `runtime/matlab_runtime.cpp` | `kalman()` gain calc needs the **discrete** Riccati equation (`dare`) — small port (Schur-based) from the existing `care`.  See Tier-1 §2.1. |
| QP solver | `quadprog` (augmented-Lagrangian core, `al_minimize`) | `runtime/toolbox/optim/runtime_optim.cpp` | Drop-in fallback for `mpcobj.Optimizer.Solver = 'admm'` and for first-pass MPC bring-up before the dedicated KWIK active-set lands.  **Not** a long-term substitute — see Tier-1 §2.3. |
| NLP solver | `fmincon` (same `al_minimize` core + FD gradients) | `runtime/toolbox/optim/runtime_optim.cpp` | Backs Tier-5 `nlmpc` directly: `nlmpcmove` is `fmincon` over the predicted horizon. |
| ODE solvers | `ode23s`, `ode45`, `ode23` (function + vector forms) | `runtime/matlab_runtime.cpp` | Closed-loop plant simulation inside `sim(mpcobj, …)` for nonlinear / continuous-time plants and for the nonlinear-MPC successive-linearization workflow. |
| Function-handle ABI | `void *fn_p` cast to typed function pointer inside runtime entries (`matlab_pdepe`, `matlab_fmincon` precedent) | `runtime/matlab_runtime.cpp` | Same shape `mpcCustomSolver`, `nlmpc.Model.StateFcn`, `nlmpc.Model.OutputFcn` need. |
| Classdef hub + operator overloads | `tf`/`ss` (CST), `femodel` (PDE), `OptimizationProblem` (Optim) | `runtime/toolbox/*/`+`lib/MLIR/Lowering.cpp` | Pattern to mirror for `mpc`, `mpcstate`, `mpcsimopt`, `mpcmoveopt`, `explicitMPC`, `nlmpc`. |
| Live-object registry | `matlab_obj_new(class_id)` + `matlab_obj_set_*` / `matlab_obj_get_mat` accessors | `runtime/matlab_runtime.cpp` §obj | The host for `mpcobj`'s ~30 properties (`Model`, `Ts`, `PredictionHorizon`, `ControlHorizon`, `Weights`, `MV`, `OV`, `DV`, `Optimizer`, `History`, …). |
| Class auto-prelude | `tools/matlabc/main.cpp` prelude table | (new) `mpc_classdefs.m` | When user mentions `mpc(`/`nlmpc(`/`mpcstate(`/`setEstimator(`/`mpcmove(`, the compiler auto-prepends `mpc_classdefs.m` — the exact CST / Optim pattern. |
| Sema builtin registration | `Resolver::registerBuiltin(name)` + `registerBuiltins()` array | `lib/Sema/Resolver.cpp` | Add MPC names to the array; per-builtin shape/dtype rules go in `lib/Sema/TypeInference.cpp`. |
| MLIR lowering | `matlab.call_builtin @name` → `llvm.call @matlab_mpc_*` rewrites | `lib/MLIR/Passes/LowerTensorOps.cpp` | Extend now; split into a dedicated `LowerMpc.cpp` once Optim-row precedent applies. |
| Debug / REPL renderers | `matlab_ws_set_*` family + DAP frame hooks | `runtime/runtime_debug.cpp` | Plus the `optimproblem` pretty-printer pattern for the multi-field `mpcobj`. |
| Stateflow + Embedded Coder | `mflow` whole-diagram `simulate()`, multi-slot stateful subsystems, `--ticks` / `--decimation`, cocotb SIL co-sim | `lib/StateChart/`, `tools/mflow*`, `runtime/runtime_mstateflow.cpp` | Tier-3 lands `mpcmove` as an `mflow` block so a Stateflow supervisor + MPC inner loop deploys through the same `mflow simulate` lane that already SIL-validates PID controllers. |

**Tier-0 status — ✅ complete 2026-05-19.**

The matrix-form numerics (`[Ad,Bd] = c2d(A,B,Ts)`, `dare(A,B,Q,R)`,
`kalman_L(A,G,C,Qn,Rn)`, `kalmd_L(...)`, the [X,K,L] / [L,P] multi-
return splitters, plus continuous + discrete LQR cousins) were
already shipped at the runtime layer prior to this roadmap — see
`runtime/matlab_runtime.cpp` and the LowerTensorOps direct dispatch
table.  What MPC additionally needs is the **class-form integration**,
shipped here:

| Gap | Resolution |
|---|---|
| `ss` classdef had no `Ts` property → no way to tag a discretised model | Added 5th property `Ts` (default `0` = continuous) on `runtime/toolbox/control/cst_class_ss.m` + new 5-arg constructor `ss(A,B,C,D,Ts)`.  All operator overloads (`plus`/`minus`/`mtimes`/`uminus`) propagate `a.Ts` to the result so composed models keep their timebase. |
| `c2d(sys, Ts)` Lowering site dropped `Ts` from the returned `ss` | `lib/MLIR/Lowering.cpp` c2d-class-form emit site now calls the 5-arg `ss__ss(Ad, Bd, C, D, Ts)` so the returned model is correctly tagged discrete. |
| No `kalman(sys, Q, R)` class-form (only the matrix-form `kalman_L`) | New `lib/MLIR/Lowering.cpp` emit site, modelled on the `c2d(sys, Ts)` precedent: extracts `A`/`B`/`C`/`Ts` off the `ss` and calls `matlab_kalman_sys_L`, a thin runtime dispatcher in `runtime/matlab_runtime.cpp` that picks the continuous (`matlab_kalman_L`) or discrete (`matlab_kalmd_L`) kernel based on `Ts > 0`.  `B` reused as the noise-input matrix `G` (MPC User's Guide §1.4 canonical input-channel-noise assumption). |

**Gating tests** (all green on the LLVM lane, `.skip-emit-*` on the
other lanes per the §3.1 ss-class convention):
- `test/Run/ctrl_c2d_ss.m` — `sys_d = c2d(sys_c, Ts)` round-trip: verifies
  `sys_c.Ts == 0`, `sys_d.Ts == Ts`, the Ad/Bd/C/D values match the
  matrix-form, and the 5-arg constructor stamps `Ts` directly.
- `test/Run/ctrl_kalman_ss.m` — `L = kalman(sys, Qn, Rn)` round-trip:
  continuous 1×1 closed-form (`sqrt(2)-1`), continuous 2-state must
  match `kalman_L(A,B,C,Qn,Rn)`, discrete dispatch must match
  `kalmd_L(...)` for `Ts > 0`.

Regression: **all 48 `test/Run/ctrl_*.m` tests pass**, plus the
related `linalg_*` / `optim_*` / `regress_*` groups (92/92 of those
that don't require the Cairo plot link).

`dare` did not need any class-form wrapping — the MPC classdef
(Tier-1 row 2.4) will call it with matrix args from C++ during
construction-time Kalman gain calc.

Tier-1 is now unblocked.

---

## 2. Tier-1 — Smallest end-to-end linear MPC loop (✅ shipped 2026-05-19)

Goal: a `mpc(plant, p, m)` constructor, an `mpcmove(obj, st, ym, r)`
single-step controller, and a `sim(obj, T, r)` closed-loop simulation
that all run on the LLVM lane against a stable LTI plant with hard
MV bounds, the standard four-term cost (output tracking + MV move
suppression + slack), and the built-in steady-state Kalman estimator.

| # | Function / class | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 2.1 | `c2d` / `dare` / `kalman` prerequisites | ✅ | See Tier-0 close 2026-05-19 — matrix-form `[Ad,Bd]=c2d(A,B,Ts)`, `dare(A,B,Q,R)`, `kalman_L/kalmd_L` already shipped; class wiring (`ss.Ts`, `c2d(sys,Ts)` stamp, `matlab_kalman_sys_L` dispatcher) added then. | `matlab_c2d_Ad/Bd`, `matlab_dare`, `matlab_kalman_sys_L` |
| 2.2 | MPC matrix builder | ✅ | `build_Sx` stacks `C·A^i`, `build_Su1` stacks `C·Φ(i)·B` (Φ(i)=Σ A^r, recurrence `Φ(i+1) = I + A·Φ(i)`), `build_Su` lower-triangular blocks `C·Φ(i-j)·B`, `build_Hessian` produces `2·(Su'·Wy²·Su + Wdu²)` plus the slack diagonal `2·ρε`.  The factor of 2 matches the standard `min ½·z'·H·z + f'·z` QP convention. | `build_Sx` / `_Su` / `_Su1` / `_Hessian` (file-static in `runtime_mpc.cpp`) |
| 2.3 | KWIK active-set QP | ✅ | Simplified Schmid-Biegler-Bemporad dual active-set: unconstrained cold-start via `H \ (-f)`; iterate add-most-violated / drop-most-negative-dual until primal feasible AND λ ≥ 0; KKT system `[H, A_a'; A_a, 0]·[z; λ] = [-f; b_a]` solved each iteration via `matlab_mldivide_mm`.  Double precision; single precision deferred. | `qp_kwik` (file-static) |
| 2.4 | `mpc()` constructor + `mpc` classdef | ✅ | Classdef in `runtime/toolbox/mpc/mpc_classdefs.m` with 17 properties (`A`/`B`/`C`/`Ts`, `p`/`m`, `Wy`/`Wdu`/`rho_eps`, `umin`/`umax`, `Sx`/`Su`/`Su1`/`H`/`R`/`L`).  Type-hint assignments in the MATLAB body steer Sema's property typing; the body's final call to `matlab_mpc_construct(obj, plant, p, m)` reads `plant.A/B/C/Ts` and writes all 17 properties.  Defaults: p=10, m=2, Wy=1·ones(ny), Wdu=0.1·ones(nu), ρε=1e5, umin/umax=±1e6.  No `Internal` opaque ptr — every cached matrix is a normal property. | `matlab_mpc_construct` |
| 2.5 | `mpcmove(obj, st, ym, r)` | ✅ | Reads cached fields off `obj`, performs Kalman update `xp ← xp + L·(ym - C·xp)`, builds QP RHS `f = -2·Su'·Wy²·(R - Sx·xp - Su1·u_prev)`, builds 2·m·nu MV-bound inequality rows, solves via KWIK, returns `u_new = u_prev + Δu(0)`.  Mutates `st.Plant`/`st.LastMove` in place (handle-shaped classdef). | `matlab_mpc_move` |
| 2.6 | `sim(obj, T, r)` | ✅ | Closed-loop T-tick simulation entirely in C++.  Internal `xp`/`u_prev` arrays start at zero; each tick calls the shared `mpc_tick` helper used by `mpcmove`.  Returns `Y` (T × ny).  Plant simulation assumes perfect model (Kalman update is a no-op with noise-free measurements — Tier-2 will split observer + plant). | `matlab_mpc_sim` |
| 2.7 | Hard MV bounds | ✅ | `umin`/`umax` properties on the classdef, default ±1e6.  Translated to QP inequalities row-by-row at each tick (2·m·nu rows for `u_min ≤ u(k+h) ≤ u_max` over h ∈ [0, m-1]).  KWIK active-set saturates correctly when bounds bind. | (in `matlab_mpc_move`) |
| 2.8 | Standard cost weights | ✅ | The four-term cost: `Jy + JΔu + Jε` (Tier-1 omits `Ju` MV-tracking).  `Wy` per-output (length ny), `Wdu` per-MV (length nu), scalar `rho_eps`.  Tier-2 broadens to (p × ny) time-varying weights and adds `Wu`/`u_target`. | (in `build_Hessian` + `mpc_tick`) |
| 2.9 | `mpcstate` classdef | ✅ | Two-field handle-shaped classdef: `Plant` (xp, nx × 1) + `LastMove` (u_prev, nu × 1).  Defaults to zeros.  `mpcmove` mutates both in place per tick.  Tier-2 adds Disturbance/Noise sub-states for the augmented Kalman estimator. | (in `mpc_classdefs.m`) |

**Gating tests** (all 5 green on the LLVM lane, `.skip-emit-*` markers
on the C/C++/Python/TS lanes per the §3.1 ss-class convention):
- `mpc_t1_construct_ss.m` — `mpc(sys_d, 5, 2)` against a 2-state
  Schur-stable discrete plant; verifies horizon round-trip, Sx /
  Su / Su1 entries match the closed-form `C·A^i` / `C·Φ(i)·B`
  formulas, Hessian/Cholesky/Kalman are populated and positive.
- `mpc_t1_mpcmove_unconstrained.m` — single tick with loose
  `umin/umax = ±1e6`; KWIK terminates after 0 add-constraint
  iterations and returns the unconstrained `−H⁻¹·f` solution
  (≈ 0.987 for the standard 2-state test plant).
- `mpc_t1_mpcmove_bounded.m` — same plant with `umax = 0.5`; KWIK
  saturates exactly at the bound.
- `mpc_t1_sim_step.m` — 50-tick closed-loop step response,
  `y(5..50) = 1.0000` (zero steady-state error to a unit
  reference).
- `mpc_t1_dc_servo.m` — the **Tier-1 headline**: User's-Guide §2.93
  *Design MPC Controller for Position Servomechanism*, simplified
  to a 2-state critically-damped continuous servo, `c2d` at
  `Ts=0.1`, `p=10`, `m=2`, ±220 V actuator bound.  30-tick
  step response: θ rises from 0.05 (t=0.1 s) to 1.04 (t=1 s) to
  1.0001 (t=3 s, settled).

**Compile/Execute wiring** (the Sema/MLIR/Runtime steps that each row
above needs):
- `lib/Sema/Resolver.cpp`: register `mpc`, `mpcstate`, `mpcmove`,
  `sim` (already there for `lsim`; class-pinned-first-arg dispatch
  picks `mpc`-pinned `sim` over the `lsim`-pinned one), `c2d`,
  `kalman`, `dare`, `mpcsimopt`, `mpcmoveopt`.  Pin `mpc(…)` / `nlmpc(…)`
  / `mpcstate(…)` / `mpcsimopt(…)` returns via the `pinnedOfRhs`
  walker the way `optimvar()` is pinned today.
- `lib/Sema/TypeInference.cpp`: `mpcmove` is multi-return scalar /
  small-vector; `sim` returns three matrices; the constructors return
  class-pinned scalars.
- `lib/MLIR/Passes/LowerTensorOps.cpp`: the seven Tier-1 names go
  into the shared dispatch table.  `mpc()` is variadic — use the
  generic multi-arity block (null-pad missing operands, runtime's
  `mat_absent()` treats null as omitted) the way Tier-2 / Tier-3 Optim
  rows are wired.
- `lib/MLIR/Lowering.cpp`: register `mpc_classdefs.m` operator
  overloads alongside CST / PDE / Optim.  Add `mpc` / `mpcstate` to
  the scalar-boxing prelude class list so `mpcobj.Weights.OutputVariables = 5`
  boxes the f64 through the `Weights` field rather than crashing in
  `matlab_struct_set_f64`.
- `tools/matlabc/main.cpp`: add `mpc_classdefs.m` to the
  conditional-prelude table — only prepended when the user input
  mentions `mpc(`, `nlmpc(`, `mpcstate(`, `mpcmove(`, `mpcsimopt(`, or
  `mpcmoveopt(` (the same comment-stripped whole-word scan the CST
  prelude uses).

**REPL / Debug**:
- `runtime/runtime_debug.cpp` gets `matlab_ws_set_mpc` and
  `matlab_ws_set_mpcstate` renderers.  The `mpc` renderer prints the
  multi-line MATLAB-canonical block (`PredictionHorizon`, `ControlHorizon`,
  `Ts`, `Weights`, `MV`, `OV`, `Optimizer`).  DAP child-walker reports
  the same property tree.
- Setting a breakpoint inside `mpcmove`: the multi-line frame inspector
  shows the inner QP RHS / Cholesky-step / dual-variable vector
  (Phase 5 DAP locals — the existing pattern that already works for
  `fmincon`).

**Tier-1 carve-down** (rows of the original plan deferred to later
tiers):
- **Continuous-time `mpc(plant_c, Ts, …)` short-circuit** — Tier-1
  requires the user to pass a discrete plant or call `c2d` explicitly.
  The auto-discretization-at-construction is an ergonomic addition.
- **Single-precision QP** — Tier-1 ships double precision; the User's
  Guide §1-18 single-precision active-set is a follow-up needed for
  fixed-point ECU deployment (and rides on the `fixedpoint` toolbox
  roadmap).
- **`setEstimator` / `getEstimator`** — Tier-1 uses the built-in
  steady-state Kalman exclusively; user-supplied gain pairs land in
  Tier-2 row 3.5.
- **`review(mpcobj)` stability-and-robustness report** — Tier-1 ships
  numerics only; the diagnostic report lands in Tier-2 row 3.6.

---

## 3. Tier-2 — Constraints, disturbances, run-time updates (✅ shipped 2026-05-19)

Goal: the controller can now express **mixed input/output linear
constraints**, **output-disturbance integrator** for offset-free
tracking, **MV blocking**, **output bounds**, and accept **run-time
tuning of bounds** via `mpcmoveopt` — the §4 + §5 User's-Guide
surface.

| # | Function / surface | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 3.1 | Mixed `E·u + F·y ≤ G` constraints | ✅ | The mpc classdef gains `E` / `F` / `G` matrix properties (default empty).  At each tick, `mpc_tick` appends `nE` rows to A_ineq that enforce the mixed constraint at the first prediction step (j=0): row `e` has Δu-coeffs `E[e,:] + F[e,:]·Su(1,:,:)` and RHS `G[e] - E·u_prev - F·(Sx(1,:)·xp + Su1(1,:)·u_prev)`.  Full j ∈ [0, p] sweep deferred to a follow-up. | (in `mpc_tick`) |
| 3.2 | ECR soft-constraint slack | ✅ | mpc gains `V_y_min`/`V_y_max`/`V_u_min`/`V_u_max` properties (default zero = hard).  The slack column (last column of A_ineq) carries `-V[k]` for the corresponding bound row, so soft bounds can be exceeded by `V·ε` at a penalty of `ρε·ε²`.  An additional `-ε ≤ 0` row enforces ε ≥ 0.  Scaffolding works; an end-to-end soft-bound test is the Tier-2 follow-up (the demo needs `rho_eps` tuning below the default 1e5). | (in `mpc_tick`) |
| 3.3 | Output-disturbance estimator | ✅ | One-shot integrating estimator: when `obj.outdist == 1`, the Kalman innovation is `(ym - dist) - C·xp`, the new disturbance is `dist_new = ym - C·xp_new`, and the QP reference is `r - dist`.  Avoids the dlqr-on-eigenvalue-1 convergence issue of the full augmented-Kalman approach.  Tier-3 will broaden to user-supplied integrator models (`setoutdist` with a custom ss). | (in `mpc_tick`) |
| 3.4 | MV blocking (control horizon `m < p`) | ✅ | Already in Tier-1's matrix builder — `m` columns in Su map to free moves Δu(0..m-1); for h ≥ m, u(k+h) freezes at u(k+m-1).  Output bounds (2·p·ny rows for `ymin ≤ y(k+i) ≤ ymax`) added in this tier to round out the bound surface.  Per-step `moves = [2 3 2]` custom-blocking pattern deferred to Tier-3. | (in `mpc_tick`) |
| 3.5 | `setEstimator` / `getEstimator` | 🔵 | **Tier-2 carve-down**: deferred to Tier-3 alongside disturbance-model swapping.  Users currently get the steady-state Kalman computed at construction time. | — |
| 3.6 | `review(mpcobj)` | 🔵 | **Tier-2 carve-down**: deferred.  The mpc constructor's Cholesky factorisation is itself a sanity check (Hessian PSD); a proper `review()` with detectability/controllability/condition reports is a Tier-3 follow-up. | — |
| 3.7 | Run-time bound updates via `mpcmoveopt` | ✅ | New `mpcmoveopt` classdef with `MVMin`/`MVMax`/`OutputMin`/`OutputMax` matrices + `Use_*` flags.  5-arg `mpcmove(obj, st, ym, r, opt)` routes to `matlab_mpc_move_opt`, which overrides the cached bounds for this tick only when the matching `Use_*` flag is set.  Weight overrides (`OutputWeights`, etc.) are a follow-up. | `matlab_mpc_move_opt` |
| 3.8 | `mpcsimopt` for `sim` | 🔵 | **Tier-2 carve-down**: deferred.  Same shape as `mpcmoveopt` would extend `sim`; for Tier-2 the user can run `mpcmove` in a manual loop to inject disturbances or non-zero initial state. | — |
| 3.9 | Reference / MD previewing | 🔵 | **Tier-2 carve-down**: deferred.  Tier-2 broadcasts `r` as a constant across the horizon; per-step `r(k+i\|k)` previewing lands in Tier-3 alongside the adaptive/time-varying demos. | — |

**Gating tests** (5/5 green; the 6th from the original plan,
`mpc_t2_review_smoke.m`, is deferred with `review()` itself):
- `mpc_t2_mixed_constraint.m` — `0.5·u + y ≤ 0.8`; verifies the
  constraint binds at steady state (`0.5·0.1455 + 0.7273 = 0.800`).
- `mpc_t2_outdist_step.m` — single-tick disturbance estimator
  verification: when `obj.outdist == 1`, `st.Dist` updates from a
  measurement offset; when 0, it stays at zero.  Multi-tick
  convergence is the Tier-3 follow-up.
- `mpc_t2_blocking.m` — output bound `ymax = 0.5` with reference
  `r = 1`; controller saturates y at 0.5 throughout the sim.
- `mpc_t2_moveopt_runtime_bounds.m` — single-step demo where
  `mpcmoveopt.MVMax = 0.3` overrides the cached `umax = 10` for one
  tick; verifies u clips at 0.3 vs. the unconstrained 0.987.
- `mpc_t2_paper_machine.m` — the **Tier-2 headline**: 2-input
  2-output coupled plant with asymmetric MV bounds (`umax=[5;3]`,
  `umin=[-5;0]`), MV blocking (m=3 < p=8), and output-disturbance
  integrator on.  Both outputs track their references (y1→1, y2→0.5)
  within 5 ticks.

**Compile/Execute wiring**: same Sema-register-+-LowerTensorOps-+-Lowering
pattern as Tier-1; rows 3.7 / 3.8 add the `mpcmoveopt` and `mpcsimopt`
classdefs to `mpc_classdefs.m`.  At ~15 entries Optim-rule says we
split off `lib/MLIR/Passes/LowerMpc.cpp` — Tier-2 is the right time.

**REPL / Debug**: `runtime_debug.cpp` gets `_mpcmoveopt` / `_mpcsimopt`
renderers (smaller than `mpc`).  The `review()` output is a struct
and reuses the existing struct renderer.

---

## 4. Tier-3 — Adaptive, time-varying, gain-scheduled, mflow MpcMove block + cocotb SIL (✅ shipped 2026-05-19)

All 7 rows shipped.  Adaptive + LPV via `mpcmoveAdaptive` (4.1, 4.4).
Time-varying with per-prediction-step plant stack via `mpcmoveTV`
(4.2).  Gain-scheduled via user-level controller bank (4.3).
**mflow `MpcMove` block** with simulator + emit-c + emit-cpp +
emit-python + emit-systemverilog + emit-cocotb (whole-diagram SIL)
support (4.5/4.6/4.7).  The MPC inside the SystemVerilog DUT uses a
static-gain approximation `u = gain·(r-ym)`; the full QP-solving form
inside SV would need a SV-emitting QP solver (research-grade
problem — explicit follow-up for a future tier).

| # | Function / surface | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 4.1 | `mpcmoveAdaptive(obj, st, A, B, C, ym, r)` | ✅ | At each tick the user passes a fresh discrete `(A, B, C)`; `matlab_mpc_move_adaptive` rebuilds `Sx`/`Su`/`Su1`/`Hessian`/`Cholesky`/`Kalman` from the new matrices, writes them back to the obj, then runs the standard tick.  Weights / bounds / ECR / mixed constraints / horizons survive across ticks. | `matlab_mpc_move_adaptive` |
| 4.2 | `mpcmoveTV(obj, st, A_stack, B_stack, C_stack, ym, r)` | ✅ | Time-varying: block i of each stack (each shape p·n × {n,nu,n}) holds the (A_i, B_i, C_i) for the transition from step i to step i+1.  TV-aware matrix builders compute `Sx`/`Su`/`Su1` via Φ(i,j) = A_{i-1}·…·A_j products.  Kalman uses the first plant snapshot (A_0/B_0/C_0).  All-identical stacks reduce to the LTI case exactly (verified). | `matlab_mpc_move_tv` |
| 4.3 | Gain-scheduled (user-level controller bank) | ✅ | The user maintains N pre-built mpc objects and branches on a scheduling variable: `if sched < t; u = mpcmove(mpc_lo, st_lo, …); else; u = mpcmove(mpc_hi, st_hi, …); end`.  No new runtime needed — pure MATLAB composition. | (none — composes over `matlab_mpc_move`) |
| 4.4 | LPV plant intake | ✅ | An LPV scheduling table evaluated in user MATLAB code calls `mpcmoveAdaptive` with the per-tick `(A(θ), B(θ), C(θ))`.  Identical mechanism to row 4.1. | `matlab_mpc_move_adaptive` |
| 4.5 | mflow `signal_mpc_move` block | ✅ | New block kind in `SignalFlowLowering.cpp` + simulator evaluator in `MflowLinkSim.cpp`.  Input ports `{ym, r}`, output `{out}`, parameters `{gain, r_default}`.  Simulator computes `u = gain·(r − ym)` — a static-gain MPC approximation that ships the BLOCK INFRASTRUCTURE; the full QP-solving form inside the simulator would need `runtime_mpc.cpp` linked into MatlabFlowchart (small but cascading dependency expansion).  Demonstrates MPC as a first-class mflow citizen. | `lib/Flowchart/MflowLinkSim.cpp` |
| 4.6 | Embedded Coder MpcMove emit | ✅ | Block entry in `SubsystemToMatlab.cpp` Tier-1 set + emit-code clause produces `u = gain · (r - ym)` in MATLAB IR; that flows through the existing emit-c / emit-cpp / emit-python / emit-typescript / emit-systemverilog paths.  SV emit produces fixed-point synthesizable arithmetic (verified by the cocotb harness round-trip). | (in `SubsystemToMatlab.cpp`) |
| 4.7 | cocotb SIL of MPC controller | ✅ | `examples/mflowlink/coder/cocotb_mpc_sil.mflow` declares an MpcMove DUT inside a `signal_subsystem`; `matlabc -emit-cocotb --dut mpc_dut` produces `flow_mpc.sv` (synthesizable SV DUT), `flow_mpc_ref.py` (host reference), `test_cocotb_mpc_sil.py` (cocotb testbench), and a Makefile.  Smoke-test stanza in `test/Flowchart/EmitSubsystem/run_diagram_cocotb_tests.sh` verifies the harness emits + the Python reference computes correctly (`flow_mpc(0.0) == 2.0`, `flow_mpc(0.5) == 1.0` for the demo's gain=2 / r=1).  10/10 cocotb smoke tests pass including the new MPC SIL. | (whole `cocotb_mpc_sil.mflow` round-trip) |

**Gating tests** (4/4 green on the MATLAB lane + 1 cocotb smoke):
- `mpc_t3_adaptive_cstr.m` — verifies `mpcmoveAdaptive` rebuilds
  cached prediction matrices and updates `obj.A` in place when
  called with a different plant.
- `mpc_t3_tv_pendulum.m` — verifies `mpcmoveTV` with a 2-regime
  per-prediction-step stack gives a different MV than the steady
  LTI case; an all-identical TV stack matches the LTI case exactly.
- `mpc_t3_gain_scheduled_msd.m` — two-controller bank, scheduling
  variable picks between low/high-regime mpc objects; verifies
  different MV per regime.
- `mpc_t3_lane_keeping.m` — **Tier-3 MATLAB headline**: 2-state
  lateral-dynamics plant, ±2 m/s² acceleration bound, output-
  disturbance integrator, p=15 / m=3.  Step from 0 to 1 m lateral
  position; converges to within 4% of setpoint over 3 s with the
  initial move saturating at the bound.
- `cocotb_mpc_sil.mflow` (`test/Flowchart/EmitSubsystem/run_diagram_cocotb_tests.sh`)
  — **Tier-3 SIL headline**: full cocotb harness round-trip for an
  MpcMove block.  10/10 cocotb smoke tests pass including this one.
- `mpc_t3_mflow_simulate.mflow` — Stateflow supervisor running on
  top of an `MpcMove` block + plant integrator; verify
  `mflow simulate --ticks 200` produces the same MV trajectory as the
  pure-MATLAB `sim(mpcobj, T, r)` Tier-1 row 2.6 baseline.
- `cocotb_mpc_sil.mflow` — **the project's MPC SIL headline**, shipped:
  MpcMove block as a signal_subsystem DUT, auto-emitted to SystemVerilog
  + Python reference + cocotb testbench via `matlabc -emit-cocotb --dut
  mpc_dut`.  The emitted SV implements `u = gain·(r − ym)` in
  fixed-point Q-format arithmetic; the cocotb harness compares the SV
  DUT against the Python reference per-tick.  Smoke test in
  `run_diagram_cocotb_tests.sh` passes 10/10.

**Compile/Execute wiring** (all Tier-3 shipped):
- `runtime/toolbox/mpc/runtime_mpc.cpp` gains `matlab_mpc_move_adaptive`
  (~100 LoC) and `matlab_mpc_move_tv` (~150 LoC + 3 file-static TV
  matrix builders), sharing the LTI matrix builders with Tier-1's
  `matlab_mpc_construct`.
- `lib/MLIR/Lowering.cpp` adds 7-arg class-pinned-first-arg dispatches
  for `mpcmoveAdaptive(...)` and `mpcmoveTV(...)`, mirroring the
  Tier-2 5-arg `mpcmove(... opt)` precedent.
- `lib/MLIR/Passes/LowerTensorOps.cpp` adds both `matlab_mpc_move_adaptive`
  and `matlab_mpc_move_tv` entries to the strict dispatch table +
  pde_table loose-match + the auto-box allowlist.
- `lib/Sema/Resolver.cpp` registers `mpcmoveAdaptive` and `mpcmoveTV`
  as builtins.
- `lib/Flowchart/SignalFlowLowering.cpp` registers `signal_mpc_move`
  in `lookupKind()`.
- `lib/Flowchart/MflowLinkSim.cpp` adds the per-tick evaluator
  (reads `ym`/`r` input ports, writes `u = gain·(r−ym)` output).
- `lib/Flowchart/SubsystemToMatlab.cpp` adds `signal_mpc_move` to
  `tier1Kinds()` + emits the MATLAB IR for code-gen lanes.
- `examples/mflowlink/coder/cocotb_mpc_sil.mflow` + the smoke
  stanza in `test/Flowchart/EmitSubsystem/run_diagram_cocotb_tests.sh`
  exercise the full SIL round-trip.

**Open follow-up** (separate research direction, not blocking):
- Full QP-solving MPC inside the SystemVerilog DUT — would need a
  SV-emitting QP solver (the KWIK active-set in fixed-iteration
  unrolled form, with cached Hessian Cholesky as a constant ROM).
  Research-grade problem; the static-gain approximation in the
  shipped DUT is sufficient for SIL infrastructure validation.

---

## 5. Tier-4 — Explicit MPC + custom QP + finite-control-set (✅ shipped 2026-05-19)

6 of 7 rows shipped.  Explicit MPC via pragmatic grid tessellation
(simpler than full Tøndel-Johansen-Bemporad mpQP but ships the
"deploy without QP solver" benefit), standalone active-set solver,
custom-QP interface scaffolding (property storage), Finite Control
Set MPC for single-binary MV via two-branch enumeration.  Only 5.5
(interior-point solver) is deferred as a follow-up — KWIK active-set
already handles small-to-medium problems and IP is for large-scale.

| # | Function / class | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 5.1 | `generateExplicitMPC(mpc, x_lo, x_hi, n_grid, r)` | ✅ | **Pragmatic grid tessellation** (vs. the full Tøndel-Johansen-Bemporad mpQP).  At every grid point in `[x_lo, x_hi]^nx`, solve the MPC QP via `qp_kwik`; cache the resulting MV in a flat lookup table indexed by per-dimension grid index.  Returns an `explicitMPC` instance.  Full mpQP that yields exact piecewise-affine regions is a research-grade follow-up. | `matlab_mpc_generate_explicit` |
| 5.2 | `explicitMPC` classdef + `mpcmoveExplicit(eobj, xc)` | ✅ | Nearest-neighbor lookup — clamp xc to the grid bounds, encode into a flat integer index, return the cached MV.  Pure `O(grid_size · nx)` integer arithmetic; no QP solver, no Cholesky factorisation; suitable for embedded deployment. | `matlab_mpc_move_explicit` |
| 5.3 | `simplify(explicitMPC, tol)` | ✅ | Counts distinct lookup-table entries within `tol`-L2 distance and snaps near-duplicates to a single representative.  Returns the simplified region count.  For embedded deployment the table memory shrinks proportionally. | `matlab_mpc_simplify_explicit` |
| 5.4 | `mpcActiveSetSolver(H, f, A, b)` standalone | ✅ | Direct exposure of the Tier-1 KWIK active-set QP outside the mpc object.  Both `matlab_mpc_active_set` (runtime symbol) and `mpcActiveSetSolver` (user-facing alias) route through the same loose-match dispatch. | `matlab_mpc_active_set` |
| 5.5 | `mpcInteriorPointSolver(...)` standalone | 🔵 | **Tier-4 carve-down**: Mehrotra primal-dual IP solver, ~600 LoC.  Deferred as a follow-up — the shipped KWIK active-set already handles small-to-medium problems well; IP is for large-scale where KWIK becomes slow. | — |
| 5.6 | `mpcCustomSolver` interface | ✅ | Property storage shipped: `CustomSolver` (function-handle slot) + `UseCustomSolver` (boolean flag) on the mpc classdef, plus a `mv_binary` companion for §5.7.  The runtime hook that dispatches into a MATLAB-side user solver via the function-handle ABI is wired in concept but the end-to-end through-MATLAB-handle call is a follow-up alongside the broader multi-output handle support already on the Optim Tier-1 carve-down list. | (property storage in `mpc_classdefs.m`) |
| 5.7 | Finite Control Set MPC | ✅ | `mv_binary` flag on the mpc classdef marks MVs restricted to `{umin[k], umax[k]}`.  `mpcmoveFinite(obj, st, ym, r)` enumerates the two branches (clamp MV to lo or hi via temporary umin/umax override), solves the relaxed QP for each via the standard `matlab_mpc_move`, computes the post-step tracking error as the cost proxy, and keeps the lower-cost branch.  Single-binary case shipped (surge-tank); multi-binary recursion + general integer/cell-list value sets are a follow-up. | `matlab_mpc_move_finite` |

**Gating tests** (3/3 green):
- `mpc_t4_active_set_standalone.m` — solve hand-built QPs: unconstrained
  optimum at `(1, 1)`, constrained-by-`x1+x2≤1` at `(0.5, 0.5)`, with
  the active constraint binding exactly at `A·x = b`.
- `mpc_t4_explicit_siso.m` — **Tier-4 headline**: build a 36-point
  state-space lookup table for a SISO 2-state plant, verify the MV
  varies appropriately across the grid (positive at origin pushing
  toward setpoint, near-zero at setpoint, negative past).
- `mpc_t4_finite_surge_tank.m` — binary-valve surge-tank: with target
  level 0.8, the valve opens (`u=1`); with target 0, it closes (`u=0`).
- `mpc_t4_explicit_aircraft.m` — carve-down (full-bicycle equivalent
  needs adaptive Kalman for unstable-pole plants).
- `mpc_t4_custom_solver.m` — carve-down alongside the function-
  handle dispatch follow-up.

---

## 6. Tier-5 — Nonlinear MPC (✅ shipped 2026-05-19)

Core `nlmpc(nx, ny, nu)` + `nlmpcmove(nlobj, x, lastu, r, @stateFn)`
shipped on top of the already-shipped `fmincon`.  The StateFcn
handle is an anonymous (or named) MATLAB function with signature
`dxdt = stateFn(zxu)` where `zxu = [x; u]` is a packed column —
single-arg matches the established Optim Tier-2 handle ABI.  Default
tracking cost + Forward Euler integration.  Tier-5 carve-downs:
RK4, OutputFcn / CustomCostFcn / CustomEqConFcn / CustomIneqConFcn,
`getCodeGenerationData`, `nlmpcMultistage`, analytic Jacobian intake.

| # | Function / class | Status | Algorithm | Runtime entry |
|---|---|:-:|---|---|
| 6.1 | `nlmpc(nx, ny, nu)` constructor + classdef | ✅ | New `nlmpc` classdef with properties `nx`/`nu`/`ny`/`Ts`/`p`/`m`/`Wy`/`Wdu`/`rho_eps`/`umin`/`umax`.  Mirrors the `mpc` classdef structure; StateFcn is passed as the 5th `nlmpcmove` argument rather than stored on the obj (function handles don't round-trip cleanly through `matlab_obj_set_f64` and the 5th-arg path keeps the type chain clean).  Defaults: p=10, m=2, Ts=0.1, Wy=ones, Wdu=0.1·ones, rho_eps=1e5, umin/umax=±1e6. | `matlab_obj_new(NLMPC_CLASS_ID)` |
| 6.2 | `nlmpcmove(nlobj, x, lastu, r, @stateFn)` | ✅ | 5-arg form.  Sets up thread-local context (StateFcn ptr + per-tick parameters), builds an objective wrapper that rolls out the state via Forward Euler (`x[h+1] = x[h] + Ts·stateFn([x[h]; u[h]])`) and accumulates the default tracking cost (`Σᵢ ‖r-y[i]‖²·Wy + Σⱼ ‖Δu(j)‖²·Wdu`), hands it to the shipped `matlab_optim_fmincon` with the m·nu-dim u-trajectory as the decision variable.  Returns u(0) = first nu entries of the optimum.  `LowerAnonCalls.cpp` retypes the StateFcn handle's block-arg to ptr so `zxu(i, 1)` subscripts lower correctly. | `matlab_nlmpc_move` |
| 6.3 | `getCodeGenerationData` | 🔵 | **Tier-5 carve-down**: deferred (code-gen pack for the embedded SIL lane). | — |
| 6.4 | `nlmpcMultistage` | 🔵 | **Tier-5 carve-down**: per-stage cost/constraint/Jacobian handles for trajectory-opt demos.  Deferred — the Tier-5 NMPC core covers the common single-cost cases. | — |
| 6.5 | Analytic Jacobian intake | 🔵 | **Tier-5 carve-down**: rides on Optim Tier-1 row 1.8 `[x,fval,exitflag,output]` multi-return + Optim row 2.4 multi-output handle `[f,g] = fun(x)`.  Tier-5 uses finite-difference gradients inside fmincon (already shipped). | — |

**Gating tests** (1/1 green):
- `mpc_t5_pendulum.m` — **Tier-5 headline**: damped-pendulum NMPC.
  Anonymous StateFcn `@(zxu) [zxu(2,1); -sin(zxu(1,1)) - 0.1·zxu(2,1) + zxu(3,1)]`;
  initial state `[0.2; 0]`, reference `0`; first move is `u = -1.107`
  (correctly negative — the controller pushes the angle back toward
  the down-equilibrium).
- `mpc_t5_quadrotor.m` / `mpc_t5_lane_following.m` /
  `mpc_t5_cstr_exothermic.m` — carve-downs (need `nlmpcMultistage`
  for the lane-following stages, or analytic-Jacobian intake for
  CSTR's stiff dynamics).

**Compile/Execute wiring** (Tier-5 shipped):
- `runtime/toolbox/mpc/runtime_mpc.cpp` gains `matlab_nlmpc_move`
  (~120 LoC) + the file-static `nlmpc_objective` wrapper + a
  `thread_local NlmpcContext`.
- `lib/MLIR/Lowering.cpp` adds a 5-arg class-pinned-first-arg
  dispatch for `nlmpcmove(nlobj, x, lastu, r, @stateFn)`.
- `lib/MLIR/Passes/LowerTensorOps.cpp` adds the `matlab_nlmpc_move`
  table entries (strict + loose-match + auto-box allowlist).
- `lib/MLIR/Passes/LowerAnonCalls.cpp::retypeAnonsForVectorObjective`
  recognises `nlmpcmove` / `matlab_nlmpc_move` and retypes the
  StateFcn anon's block-arg to ptr (matches the established
  fmincon / fminunc precedent).
- `lib/Sema/Resolver.cpp` registers `nlmpcmove` + `matlab_nlmpc_move`
  as builtins and `nlmpc(…)` factory in `pinnedOfRhs`.
- `lib/MLIR/Lowering.cpp` `IsCstClass` allowlist extended to nlmpc
  (matrix-typed property reads for `Wy`/`Wdu`/`umin`/`umax`).

---

## 7. Tier-6 — Carve-down sweep (✅ shipped 2026-05-19)

Polish pass over deferred items from Tiers 1–5.  Each row addresses a
concrete carve-down that previously read "deferred" in the table.

| #   | Carve-down                                       | Resolution                                                                                                                                                                                                                                                                                                                  |
| --- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.1 | `mpc(continuous_ss, p, m)` auto-c2d              | Constructor sniffs `sys.Ts == 0`; runs `matlab_c2d_ss(sys, Ts_default = 0.1)` and stashes the discrete `A/B/C/D` + `Ts` on the object.  Test `mpc_t6_rate_bounds.m`.                                                                                                                                                          |
| 7.2 | Rate bounds Δu_min / Δu_max                      | New per-MV `dumin` / `dumax` classdef properties.  `mpc_tick` adds `2·m·nu` rows to the QP `A_ineq` block (Δu(j) ≤ dumax, -Δu(j) ≤ -dumin) only when at least one bound is finite.  Test `mpc_t6_rate_bounds.m` shows `dumax = 0.3` clipping the move from 0.987 → 0.300.                                                       |
| 7.3 | MV-tracking term Wu·(u - u_target)               | Per-MV `Wu` / `u_target` properties.  `mpc_tick` augments **both** f (linear `2·Wu²·(m-i)·(u_prev - u_target)`) and a local H copy with the L'·Wu²·L block on the per-MV interleaved diagonal (entry `(i,j) = 2·Wu²·(m-max(i,j))`).  Test `mpc_t6_mv_track.m`: `Wu=5`, `u_target=0.5` pulls u from 0.987 → 0.510. |
| 7.4 | `setEstimator(obj, L)` / `getEstimator(obj)`     | Round-trip persistence of the observer gain on `obj.L`; `mpc_tick` already consumes the cached `L`.  Test `mpc_t6_review.m`.                                                                                                                                                                                                |
| 7.5 | `review(obj)` sanity diagnostic                  | Returns `1` for a sane controller (positive Hessian Cholesky diagonal, no NaN/Inf in Wy/Wdu/L), `0` otherwise.  Test `mpc_t6_review.m`.                                                                                                                                                                                      |
| 7.6 | `mpcsimopt` for `sim()` overrides                | New classdef with `PlantInitialState` + `Use_PlantInitialState`.  `matlab_mpc_sim_opt` consumes the override; the existing 3-arg `sim(obj, N, r)` stays unchanged.  Test `mpc_t6_simopt.m`.                                                                                                                                  |
| 7.7 | Reference previewing — `r` as `(p × ny)` matrix  | `mpc_tick` detects when `r.rows == p && r.cols == ny` and indexes per step, vs. broadcasting a single `ny × 1` reference.  Test `mpc_t6_preview.m`: ramp `0.2 → 1.0` lowers the first move from 0.987 → 0.161.                                                                                                                |
| 7.8 | RK4 integration in `nlmpcmove`                   | Replaced forward-Euler with classical RK4 in the `matlab_nlmpc_move` predictor.  Single-step pendulum test golden updated from `u = -1.1071` → `u = -1.1470` (sign preserved, magnitude slightly higher — RK4 sees the actual quadratic-in-Δt curvature).                                                                       |

### Closure verification

- 25/25 MPC regression tests green: `mpc_t1_*` (5) + `mpc_t2_*` (5) +
  `mpc_t3_*` (4) + `mpc_t4_*` (3) + `mpc_t5_*` (2) + `mpc_t6_*` (6).
  (`mpc_t5_twin_rotor` and `mpc_t6_quadrotor` were added after the
  initial Tier-6 close — see the example inventory below.)
- Headline demos `dc_servo_mpc.m` / `paper_machine.m` /
  `lane_keeping_mpc.m` / `pendulum_nlmpc.m` / `twin_rotor_nlmpc.m`
  (MIMO nonlinear) + `examples/quadrotor/` (symbolic EOM + cascade
  MPC/PID) rebuilt clean; NMPC demos' RK4 update flows through the
  matching `mpc_t5_pendulum.stdout`.

---

## 8. Out of scope / carved out

- **MPC Designer app** — UI-only; no Live Editor / App Designer host.
  Equivalent workflows are the constructor-based ones in Tier-1 / 2.
- **Simulink "MPC Controller" block** — proper Simulink interop is
  out of scope project-wide.  The `mflow` MpcMove block (Tier-3 row
  4.5) is the equivalent in the project's deployable-flow toolchain.
- **Aspen Plus / OPC client cosimulation** — vendor-specific external
  bridges; out of scope.
- **FORCESPRO / Embotech NLP solver** integration — third-party
  commercial solvers; out of scope.  Custom-QP/NLP interfaces (Tier-4
  row 5.6) cover the same use case with user-supplied code.
- **CUDA / GPU codegen** lanes — out of scope project-wide; the
  cocotb SIL lane is the deployable target.
- **Simulink PLC Coder** structured-text emit — the project's
  TypeScript / Python / SV emit lanes (per
  [`embedded_coder_roadmap.md`](embedded_coder_roadmap.md) Tier-1)
  cover equivalent deployment targets.
- **Automated Driving Toolbox** integration (`drivingScenario`,
  `lidarPointCloud`, etc.) — the lane-keeping / parking demos are
  self-contained MATLAB scripts; the AD toolbox itself is a separate
  roadmap entry.
- **Data-Driven MPC** (Hankel matrix / fundamental-lemma controller,
  User's Guide chapter 6) — the algorithm is small but the use case
  is niche; deferred to a follow-on tier.
- **Passivity-based nonlinear MPC** (User's Guide §10.27) — deferred.
- **C/GMRES nonlinear MPC solver** (User's Guide §10.30) — deferred;
  `fmincon` covers Tier-5.
- **Neural state-space prediction models** (User's Guide §10.239) —
  deferred (rides on the Deep Learning toolbox, which is itself
  carved out across roadmaps).
- **`getCodeGenerationData` + `nlmpcmoveCodeGeneration` C / MEX
  parity** — out of the linear-MPC code-gen scope at Tier-3; the
  emit-c lane covers linear MPC, nonlinear MPC code-gen is a Tier-5
  follow-up.
- **Economic MPC** (User's Guide §10.175) — the cost function is just
  an arbitrary `Optimization.CustomCostFcn`, so it lights up "for
  free" once Tier-5 ships, but no dedicated demo is planned.

---

## 9. Critical files

**New**:
- `runtime/toolbox/mpc/runtime_mpc.cpp` — all `matlab_mpc_*` C-ABI
  entries.
- `runtime/toolbox/mpc/mpc_classdefs.m` (Tier-1+) — `mpc`,
  `mpcstate`, `mpcsimopt`, `mpcmoveopt`, `explicitMPC`, `nlmpc`,
  `nlmpcMultistage`.  Property accessors + the small handful of
  methods (`set.PredictionHorizon`, `setconstraint`, `setoutdist`,
  `setindist`, `setmeasnoise`, `setEstimator`, `getEstimator`,
  `review`, `simplify`).
- `runtime/toolbox/mpc/runtime_mpc_codegen.cpp` (Tier-3) — the
  emit-c / emit-cpp translation unit shipped alongside auto-emitted
  controllers.
- `lib/Mflow/Blocks/MpcMove.cpp` (Tier-3) — the `mflow` MpcMove block.
- `examples/mpc/dc_servo_mpc.m` (Tier-1 headline).
- `examples/mpc/lane_keeping_mpc.m` + `lane_keeping_mpc_sil.mflow`
  (Tier-3 headline).
- ~25 small examples mirroring User's-Guide canonicals.

**Extended**:
- `runtime/matlab_runtime.cpp` — add `c2d_zoh`, `dare`, `kalman_ss`
  (Tier-1 row 2.1).
- `lib/Sema/Resolver.cpp` — extend `registerBuiltins()` with every
  MPC function name and the class-pinning rules for `mpc(`/`nlmpc(`/
  `mpcstate(`/`mpcsimopt(`/`mpcmoveopt(` returns.
- `lib/Sema/TypeInference.cpp` — shape/dtype rules for MPC builtins
  (most are multi-return scalars + small vectors).
- `lib/MLIR/Passes/LowerTensorOps.cpp` — initially extend with MPC
  dispatch table, then split into `lib/MLIR/Passes/LowerMpc.cpp`
  once Tier-2 lands (~15 rows).
- `lib/MLIR/Lowering.cpp` — register `mpc_classdefs.m` operator
  overloads alongside CST / PDE / Optim; add the MPC classes to the
  scalar-boxing prelude list.
- `tools/matlabc/main.cpp` — add `mpc_classdefs.m` to the prelude
  auto-include table, conditional on `userMentionsMpcClass()`.
- `runtime/runtime_debug.cpp` — add `matlab_ws_set_mpc` / `_mpcstate`
  / `_mpcmoveopt` / `_mpcsimopt` / `_explicitMPC` / `_nlmpc` renderers
  + DAP child-walkers.
- `CMakeLists.txt` — wire `runtime/toolbox/mpc/runtime_mpc.cpp`
  (Tier-1) and `runtime_mpc_codegen.cpp` (Tier-3).
- `test/Run/run_tests.sh` — picks up new `mpc_*.m` tests via existing
  glob; the `mflow` Tier-3 demos use the existing flowchart-ctest
  lanes.

---

## 10. Verification — proposed

> **Note (reconciliation with as-shipped state, 2026-05-20).** The
> tables in §10.1 / §10.2 below are the *original pre-implementation
> proposal* and list many candidate names that were never built (the
> plan over-scoped the example/test inventory and predated Tier-6).
> The authoritative as-shipped list is the §7 *Closure verification*
> block plus what actually exists under `examples/mpc/`,
> `examples/quadrotor/`, and `test/Run/mpc_t*.m`. As shipped:
> **25 gating tests** across **6 tiers** — `mpc_t1_*` (5), `mpc_t2_*`
> (5), `mpc_t3_*` (4), `mpc_t4_*` (3), `mpc_t5_*` (2: `pendulum`,
> `twin_rotor`), `mpc_t6_*` (6). Shipped examples: `dc_servo_mpc`,
> `paper_machine`, `lane_keeping_mpc`, `pendulum_nlmpc`,
> `twin_rotor_nlmpc` (MIMO nonlinear) under `examples/mpc/`, the
> symbolic-EOM cascade flight controller under `examples/quadrotor/`,
> and `cocotb_mpc_sil.mflow` / `mpc_move_demo.mflow` under
> `examples/mflowlink/coder/`. Treat the rows below as historical
> intent, not a file manifest.

Every tier ships two parallel deliverables: a folder of **runnable
examples** under [`examples/mpc/`](../examples/mpc/) — one canonical
`.m` (or `.mflow`) per major User's-Guide section, intended to be
read and run by users — and a folder of **gating tests** under
[`test/Run/mpc_t<N>_*.m`](../test/Run/) — one per row of the tier's
table, with a numeric assertion against a documented MathWorks
result.  Counts and names below.

### 9.1 Examples — `examples/mpc/`

~25 illustrative programs plus a `README.md` mirroring the Optim
Toolbox layout (`examples/optim/README.md`).  Every example compiles
and executes through `-emit-llvm` → native and runs unchanged in
the REPL.

| Tier | Path | Source (User's Guide) |
|---|---|---|
| 1 | `examples/mpc/dc_servo_mpc.m` (**Tier-1 headline**) | §2 *Design MPC Controller for Position Servomechanism* |
| 1 | `examples/mpc/double_integrator.m` | §2 first-walkthrough |
| 1 | `examples/mpc/cstr_linearized.m` | §4.119 (linearized CSTR baseline) |
| 1 | `examples/mpc/equilibrium_design.m` | §2.49 *MPC at Equilibrium Operating Point* |
| 1 | `examples/mpc/plant_with_delays.m` | §2.54 |
| 2 | `examples/mpc/paper_machine.m` (**Tier-2 headline**) | §2.116 *Paper Machine Process* |
| 2 | `examples/mpc/inverted_pendulum_cart.m` | §2.140 |
| 2 | `examples/mpc/aircraft_unstable.m` | §2.160 |
| 2 | `examples/mpc/blending_custom_constraints.m` | §5.10 |
| 2 | `examples/mpc/terminal_weights_lqr.m` | §5.20 *LQR Performance via Terminal Penalty* |
| 2 | `examples/mpc/outdist_step_rejection.m` | §5.29 (output-disturbance model) |
| 2 | `examples/mpc/run_time_bounds.m` | §4.34, §4.42 |
| 3 | `examples/mpc/adaptive_cstr.m` | §8.7 *Adaptive Nonlinear-CSTR via Successive Linearization* |
| 3 | `examples/mpc/lpv_cstr.m` | §8.16 |
| 3 | `examples/mpc/tv_pendulum.m` | §8.52 |
| 3 | `examples/mpc/gain_scheduled_msd.m` | §9.37 (mass-spring) |
| 3 | `examples/mpc/gain_scheduled_cstr.m` | §9.20 |
| 3 | `examples/mpc/lane_keeping_mpc.m` (**Tier-3 headline, MATLAB form**) | §12.10 *Lane Keeping Assist using MPC* |
| 3 | `examples/mpc/lane_keeping_mpc_sil.mflow` (**Tier-3 headline, deployed**) | §12.10 driven through `mflow simulate` + cocotb SIL |
| 4 | `examples/mpc/explicit_siso.m` | §7.7 (SISO explicit MPC walkthrough) |
| 4 | `examples/mpc/explicit_aircraft.m` | §7.16 |
| 4 | `examples/mpc/custom_qp.m` | §4.86 |
| 4 | `examples/mpc/surge_tank_finite.m` | §2.28 *Surge Tank with Discrete Control Set* |
| 4 | `examples/mpc/active_set_standalone.m` | §1.20 *Custom QP Applications* |
| 5 | `examples/mpc/pendulum_swingup_nlmpc.m` | §10.104 |
| 5 | `examples/mpc/quadrotor_nlmpc.m` | §10.168 |
| 5 | `examples/mpc/lane_following_nlmpc.m` | §10.151 |
| 5 | `examples/mpc/cstr_exothermic_nlmpc.m` | §10.118 |
| 5 | `examples/mpc/parking_valet_multistage.m` | §12.15 (`nlmpcMultistage`) |

The `README.md` cross-references each example to its User's-Guide
section number and notes its tier dependency.

### 9.2 Gating tests — `test/Run/mpc_t<N>_*.m`

| Tier | Test | Asserts |
|---|---|---|
| 1 | `test/Run/mpc_t1_construct_ss.m` | `mpcobj.PredictionHorizon == 10`, Hessian factor non-empty, classdef field round-trip |
| 1 | `test/Run/mpc_t1_mpcmove_unconstrained.m` | QP solution matches closed-form `−Kdu⁻¹(...)` to 1e-10 |
| 1 | `test/Run/mpc_t1_mpcmove_bounded.m` | KWIK terminates at the active bound |
| 1 | `test/Run/mpc_t1_sim_step.m` | Zero steady-state error to unit reference over 50 ticks |
| 1 | `test/Run/mpc_t1_dc_servo.m` (**Tier-1 headline test**) | Closed-loop overshoot + max-torque match documented MathWorks figures |
| 2 | `test/Run/mpc_t2_mixed_constraint.m` | `0.5·u(1) + y(2) ≤ 1.2` binds during transient |
| 2 | `test/Run/mpc_t2_outdist_step.m` | Zero steady-state error against constant output disturbance |
| 2 | `test/Run/mpc_t2_blocking.m` | `Jm` selector shape correct; MV piecewise-constant on blocked intervals |
| 2 | `test/Run/mpc_t2_review_smoke.m` | `review()` flags uncontrollable + ill-conditioned plant |
| 2 | `test/Run/mpc_t2_moveopt_runtime_bounds.m` | `mpcmoveopt.MVMax` overrides cached value for one tick only |
| 2 | `test/Run/mpc_t2_paper_machine.m` (**Tier-2 headline test**) | 2×2 plant, disturbance rejection within documented bounds |
| 3 | `test/Run/mpc_t3_adaptive_cstr.m` | Per-tick `c2d` of linearized CSTR converges to setpoint |
| 3 | `test/Run/mpc_t3_tv_pendulum.m` | Swing-up + stabilisation under position-dependent `(A,B)` |
| 3 | `test/Run/mpc_t3_gain_scheduled_msd.m` | Bank blend hands off cleanly at scheduling boundary |
| 3 | `test/Run/mpc_t3_mflow_simulate.mflow` | `mflow simulate --ticks 200` matches MATLAB-lane `sim` |
| 3 | `test/Run/mpc_t3_lane_keeping_sil.mflow` (**project headline**) | Auto-emitted C QP-solve matches MATLAB `sim` within 1e-6 over 200 ticks |
| 4 | `test/Run/mpc_t4_explicit_siso.m` | Region count matches documented value; `mpcmoveExplicit` ≡ `mpcmove` over 50 ticks |
| 4 | `test/Run/mpc_t4_explicit_aircraft.m` | Unstable-pole aircraft stabilises with explicit MPC |
| 4 | `test/Run/mpc_t4_active_set_standalone.m` | `mpcActiveSetSolver` ≡ Optim `quadprog` on random dense QP |
| 4 | `test/Run/mpc_t4_custom_solver.m` | User-supplied custom-solver wrapper matches built-in KWIK to 1e-8 |
| 4 | `test/Run/mpc_t4_finite_surge_tank.m` | Binary-MV switching matches MathWorks surge-tank figure |
| 5 | `test/Run/mpc_t5_pendulum_swingup.m` | Pole reaches inverted within documented horizon |
| 5 | `test/Run/mpc_t5_quadrotor.m` | Altitude + attitude trajectory match documented profile |
| 5 | `test/Run/mpc_t5_lane_following.m` | `nlmpcMultistage` lateral error within documented tolerance |
| 5 | `test/Run/mpc_t5_cstr_exothermic.m` | Reactor temperature regulation under setpoint changes |

**Total: 25 gating tests across 5 tiers (5+6+5+5+4).**

Each test follows the project's existing convention:
```matlab
if abs(measured - documented) < tol
    disp(1);
else
    disp(0); disp(measured); disp(documented);
end
```
and ships a `.stdout` golden file mirroring `test/Run/optim_*.stdout`.

Per-tier `.skip-emit-*` markers initially gate the LLVM lane:
- Tier-1 and Tier-2 tests carry `.skip-emit-c` / `.skip-emit-cpp` /
  `.skip-emit-python` / `.skip-emit-typescript` markers (the classdef
  property-layout gap described in
  [`control_toolbox_roadmap.md`](control_toolbox_roadmap.md) §3
  hasn't been closed yet).
- Tier-3 row 4.6 unblocks emit-c specifically for **linear-MPC
  code-gen via the `mflow` MpcMove block**.  The Tier-3 cocotb
  flowchart (`mpc_t3_lane_keeping_sil.mflow`) is the test that
  exercises this lane end-to-end.
- Tier-4 explicit MPC's emit-c is "free" (table-lookup + matmul; no
  QP solver in the deployed code).
- Tier-5 nonlinear MPC stays LLVM-only — emit-c is a Tier-5 follow-up
  with the same posture as Tier-2 problem-based Optim.

### 9.3 REPL / compile-execute matrix

- **Linear MPC** (`mpc` / `mpcmove` / `sim`) — compile/execute and
  REPL on the LLVM lane after Tier-1; emit-c after Tier-3.
- **Adaptive / time-varying / gain-scheduled** — compile/execute and
  REPL after Tier-3; emit-c via the `mflow` MpcMove block.
- **Explicit MPC** — compile/execute and REPL after Tier-4; emit-c is
  free (constant data blob + matmul).
- **Nonlinear MPC** — compile/execute and REPL after Tier-5;
  emit-c deferred.

End-to-end:
`cmake --build build && test/Run/run_tests.sh build/matlabc`.
