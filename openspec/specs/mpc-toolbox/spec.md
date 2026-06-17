# Model Predictive Control Toolbox Spec

## Purpose
Documents the shipped subset of the Model Predictive Control Toolbox in the matlab_llvm compiler: linear MPC with a KWIK active-set QP core, adaptive and time-varying MPC, explicit (offline-tessellated) MPC with a standalone QP solver, and nonlinear MPC (NMPC) backed by fmincon with RK4 rollout, plus the supporting `mpc`/`mpcstate`/`nlmpc`/`explicitMPC` objects. All six implementation tiers are shipped (2026-05-19). (doc: docs/mpc_toolbox_roadmap.md) (src: runtime/toolbox/mpc)

## Requirements

### Requirement: Linear MPC controller objects and single-step control
The system SHALL provide an `mpc` controller object with companion `mpcstate`, and single-step linear model-predictive control. (src: runtime/toolbox/mpc/mpc_classdefs.m) (src: runtime/toolbox/mpc/runtime_mpc.cpp)

#### Scenario: Construct and step a linear MPC controller
- **WHEN** a program constructs `mpc(plant, p, m)`, builds an `mpcstate`, and calls `mpcmove(obj, state, ym, r)`
- **THEN** the system SHALL build the prediction/QP machinery via `matlab_mpc_construct` and return the optimal move solved by the KWIK active-set QP via `matlab_mpc_move`, honoring MV bounds and the four-term cost (Wy, Wdu, slack)

### Requirement: Constraints, disturbances, and runtime overrides
The system SHALL support mixed/output/rate constraints, an output-disturbance integrator, and runtime bound overrides via `mpcmoveopt`. (doc: docs/mpc_toolbox_roadmap.md) (src: runtime/toolbox/mpc/runtime_mpc.cpp)

#### Scenario: Move with runtime bound overrides
- **WHEN** a program calls `mpcmove` with an `mpcmoveopt` carrying MV/output bound overrides
- **THEN** the system SHALL apply the overrides for that tick and return the constrained optimal move via `matlab_mpc_move_opt`

### Requirement: Adaptive and time-varying MPC
The system SHALL provide adaptive and time-varying MPC that rebuilds the prediction model from per-tick plant matrices. (doc: docs/mpc_toolbox_roadmap.md) (src: runtime/toolbox/mpc/runtime_mpc.cpp)

#### Scenario: Update the plant model each tick
- **WHEN** a program calls `mpcmoveAdaptive(obj, state, A, B, C, ym, r)` or supplies a time-varying plant stack
- **THEN** the system SHALL rebuild the cached prediction/Hessian/estimator matrices from the fresh `(A,B,C)` and return the move via `matlab_mpc_move_adaptive` or `matlab_mpc_move_tv`

### Requirement: Explicit MPC and standalone QP solver
The system SHALL provide explicit (offline-tessellated) MPC generation/evaluation and a standalone active-set QP solver. (doc: docs/mpc_toolbox_roadmap.md) (src: runtime/toolbox/mpc/mpc_classdefs.m) (src: runtime/toolbox/mpc/runtime_mpc.cpp)

#### Scenario: Generate and evaluate an explicit controller
- **WHEN** a program calls `generateExplicitMPC` to tessellate the parameter hypercube, evaluates with `mpcmoveExplicit`, simplifies with `simplify`, or invokes the standalone solver
- **THEN** the system SHALL build the lookup table via `matlab_mpc_generate_explicit`, return the lookup move via `matlab_mpc_move_explicit`, deduplicate via `matlab_mpc_simplify_explicit`, and solve standalone QPs via `matlab_mpc_active_set`

### Requirement: Nonlinear MPC (NMPC)
The system SHALL provide a nonlinear MPC controller object and single-step nonlinear control via an fmincon backend with RK4 state rollout. (doc: docs/mpc_toolbox_roadmap.md) (src: runtime/toolbox/mpc/mpc_classdefs.m) (src: runtime/toolbox/mpc/runtime_mpc.cpp)

#### Scenario: Solve a nonlinear MPC move
- **WHEN** a program constructs `nlmpc(nx, ny, nu)` and calls `nlmpcmove(nlobj, x, u_prev, r, @stateFcn)`
- **THEN** the system SHALL roll the state forward with RK4, accumulate the tracking cost, and return the optimal move solved by fmincon via `matlab_nlmpc_move`

### Requirement: Closed-loop simulation and controller review
The system SHALL provide closed-loop simulation (with initial-state override) and a controller sanity-review function. (doc: docs/mpc_toolbox_roadmap.md) (src: runtime/toolbox/mpc/runtime_mpc.cpp)

#### Scenario: Simulate and review a controller
- **WHEN** a program calls `sim(obj, T, r)` (optionally with an `mpcsimopt` initial-state override) or `review(obj)`
- **THEN** the system SHALL run the T-tick closed loop via `matlab_mpc_sim`/`matlab_mpc_sim_opt` and report stability/conditioning diagnostics via `matlab_mpc_review`
