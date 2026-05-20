# MPC Toolbox Examples

Programs that exercise the Model Predictive Control Toolbox runtime in
matlab_llvm.  See [`docs/mpc_toolbox_roadmap.md`](../../docs/mpc_toolbox_roadmap.md)
for the full tiered roadmap.

## Tier-1 (shipped)

Linear MPC against an LTI plant with hard MV bounds and the standard
four-term cost (output tracking + MV move suppression + slack).

| Example | User's Guide | Notes |
|---|---|---|
| [`dc_servo_mpc.m`](dc_servo_mpc.m) | §2.93 *Design MPC Controller for Position Servomechanism* | **Tier-1 headline.**  2-state critically-damped servo, ±220 V actuator bound, p=10 / m=2, unit-step position tracking. |

## Tier-2 (shipped)

Output bounds, mixed input/output constraints, ECR soft-slack scaffolding,
run-time bound overrides via `mpcmoveopt`, output-disturbance estimator
for offset-free tracking under model mismatch.

| Example | User's Guide | Notes |
|---|---|---|
| [`paper_machine.m`](paper_machine.m) | §2.116 *Paper Machine Process* | **Tier-2 headline.**  Simplified 2-input 2-output coupled plant with asymmetric MV bounds, MV blocking (m=3 < p=8), output-disturbance integrator.  Coordinates two MVs to track two outputs simultaneously through a cross-coupled plant. |

## Tier-3 (shipped)

Adaptive MPC (`mpcmoveAdaptive`), time-varying MPC (`mpcmoveTV` with
stacked per-prediction-step plants), gain-scheduled via user-level
controller bank, LPV intake (alias of adaptive).  Plus the **mflow
MpcMove block** — MPC as a first-class signal-flow citizen, with
simulator + emit-c + emit-cpp + emit-python + emit-systemverilog +
cocotb SIL all working through a single block kind.

| Example | User's Guide | Notes |
|---|---|---|
| [`lane_keeping_mpc.m`](lane_keeping_mpc.m) | §12.10 *Lane Keeping Assist using MPC* | **Tier-3 MATLAB headline.**  Simplified 2-state lateral-dynamics plant, ±2 m/s² lateral-acceleration bound, output-disturbance integrator. |
| [`../mflowlink/coder/cocotb_mpc_sil.mflow`](../mflowlink/coder/cocotb_mpc_sil.mflow) | §11 *Code Generation* + §12.10 (deployed form) | **Tier-3 SIL headline.**  MpcMove block as a `signal_subsystem` DUT, emit-cocotb produces SystemVerilog DUT + Python reference + cocotb testbench.  Static-gain MPC approximation in the SV; full QP-solving SV is an open follow-up. |

## Tier-4 (shipped)

Explicit MPC (offline grid tessellation — deploys as a pure lookup
table with zero QP solve at runtime), standalone active-set solver,
Finite Control Set MPC for binary-valued MVs.

| Example | User's Guide | Notes |
|---|---|---|
| `test/Run/mpc_t4_explicit_siso.m` | §7.7 *Explicit MPC for SISO Plant* | **Tier-4 headline.**  36-point grid over a 2-state state space, lookup table replaces the run-time QP solve. |
| `test/Run/mpc_t4_finite_surge_tank.m` | §2.28 *Surge Tank with Discrete Control Set* | Single-binary-MV surge tank, branch-and-bound enumeration over the two valve states. |

## Tier-5 (shipped)

Nonlinear MPC: `nlmpc` classdef + `nlmpcmove` that builds an RK4 state
rollout via the user's StateFcn handle and hands the cost to the shipped
`fmincon`.

| Example | User's Guide | Notes |
|---|---|---|
| [`pendulum_nlmpc.m`](pendulum_nlmpc.m) | §10.104 *Swing-Up of Pendulum using NMPC* (simplified) | **Tier-5 SISO headline.**  Damped pendulum, anonymous-handle StateFcn, default tracking cost, ±5 N·m saturation.  First move correctly drives toward the equilibrium. |
| [`twin_rotor_nlmpc.m`](twin_rotor_nlmpc.m) | 2-DOF helicopter / TRMS — the canonical MIMO nonlinear benchmark | **Tier-5 MIMO headline.**  `nlmpc(4,2,2)` coordinates two cross-coupled rotors to track pitch + yaw against a gravity nonlinearity.  Two-phase setpoint, RK4 plant, 4-panel plot.  A pair of SISO loops would fight each other through the off-axis torques; one MIMO NMPC handles both.  Steady inputs match the analytic trim (u1=1.04, u2=−0.31). |

## Coverage matrix

|               | SISO                               | MIMO                                              |
|---------------|------------------------------------|---------------------------------------------------|
| **Linear**    | `dc_servo_mpc`, `lane_keeping_mpc` | `paper_machine`, `../quadrotor/quadrotor_pid_mpc` |
| **Nonlinear** | `pendulum_nlmpc`                   | **`twin_rotor_nlmpc`**                            |

See the roadmap §9.1 for the full example inventory.
