# Model Predictive Control Toolbox — Tutorial

The MPC Toolbox runtime designs and simulates predictive controllers
against LTI and nonlinear plants. Linear MPC builds the prediction matrices
once at construction and solves a hard-bounded QP via a KWIK active-set
solver at each tick; nonlinear MPC rolls out the plant with RK4 and hands
the cost to the shipped `fmincon`. All five tiers are shipped, and the
controller plugs into the shipped Control System Toolbox `ss` / `c2d`
objects and the System Identification Toolbox identified models.

## Supported features

- **Linear MPC**: `mpc(plant, p, m)` (prediction / control horizons),
  hard MV bounds `obj.umax` / `obj.umin`, output / move weights `obj.Wy` /
  `obj.Wdu`, output-disturbance estimator `obj.outdist` for offset-free
  tracking, MV blocking (`m < p`).
- **Simulation**: `sim(obj, T, r)` closed-loop step; `mpcmove` /
  `mpcmoveopt` run-time moves and bound overrides.
- **Plant intake**: `ss(...)`, `c2d(...)` from the Control System Toolbox;
  identified models from `ssest` / `n4sid`.
- **Adaptive / time-varying (Tier-3)**: `mpcmoveAdaptive`, `mpcmoveTV`,
  gain-scheduled controller banks, plus the mflow `MpcMove` block (emit-c /
  cpp / python / systemverilog + cocotb SIL).
- **Explicit & finite-set (Tier-4)**: explicit MPC (offline grid
  tessellation -> lookup table), standalone active-set solver, Finite
  Control Set MPC for binary MVs.
- **Nonlinear MPC (Tier-5)**: `nlmpc(nx, ny, nu)` classdef + `nlmpcmove`
  with an anonymous `StateFcn` handle (packed `zxu = [x; u]` signature),
  RK4 rollout, `fmincon`-based NLP solve.

## Build & run

```bash
build/matlabc -emit-llvm examples/mpc/dc_servo_mpc.m > /tmp/dc_servo_mpc.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/dc_servo_mpc.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/dc_servo_mpc
/tmp/dc_servo_mpc
```

## Worked examples

### Linear MPC for a position servo (`examples/mpc/dc_servo_mpc.m`)

The Tier-1 headline: a 2-state critically-damped servo with a ±220 V
actuator bound. The continuous plant is discretised by `c2d` inside the
MPC, which builds the prediction matrices once and solves the bounded QP
each tick.

```matlab
A_c = [0, 1; 0, 0-4];   B_c = [0; 2];   C_c = [1, 0];   D_c = [0];
sys_c = ss(A_c, B_c, C_c, D_c);
sys_d = c2d(sys_c, 0.1);

obj = mpc(sys_d, 10, 2);           % p = 10 prediction, m = 2 control moves
obj.umax = [220];   obj.umin = [-220];

y = sim(obj, 30, [1]);             % closed-loop unit-step in theta
fprintf('t=1.0s: theta = %.4f\n', y(10, 1));
```

The step tracks within ~1.5 s with a small overshoot.

### Multivariable MPC (`examples/mpc/paper_machine.m`)

A 2-input 2-output paper-machine plant with cross-coupling, asymmetric MV
bounds, MV blocking (`m=3 < p=8`), and an output-disturbance integrator
(`obj.outdist = 1`) for zero steady-state error under model mismatch. One
controller coordinates both MVs against the coupling.

```matlab
A = [0.7, 0.0; 0.0, 0.5];   B = [0.5, 0.1; 0.1, 0.4];
C = eye(2);   D = zeros(2);
sys_d = ss(A, B, C, D, 0.5);

obj = mpc(sys_d, 8, 3);
obj.umax = [5; 3];   obj.umin = [-5; 0];   % asymmetric, per channel
obj.outdist = 1;
y = sim(obj, 40, [1.0; 0.5]);              % track both outputs
```

### Lane-keeping with disturbance rejection (`examples/mpc/lane_keeping_mpc.m`)

The Tier-3 MATLAB headline: a 2-state lateral-dynamics plant under a tight
±2 m/s² acceleration bound, with output/move weights and an integrating
output disturbance for offset-free steady state under wind/bank loads.

```matlab
sys_d = ss(A_d, B_d, C_d, D_d, 0.05);
obj = mpc(sys_d, 15, 3);
obj.umax = [2.0];   obj.umin = [-2.0];
obj.outdist = 1;
obj.Wy  = [5.0];                   % output tracking weight
obj.Wdu = [0.2];                   % move suppression weight
y = sim(obj, 60, [1.0]);
```

### Nonlinear MPC — SISO (`examples/mpc/pendulum_nlmpc.m`)

`nlmpc(nx, ny, nu)` builds an RK4 state rollout via the user's `StateFcn`
handle and hands the cost to `fmincon`. The handle uses the packed
single-argument `zxu = [x; u]` signature.

```matlab
nlobj = nlmpc(2, 1, 1);
nlobj.Ts = 0.1;   nlobj.p = 10;   nlobj.m = 3;
nlobj.umax = [5];   nlobj.umin = [-5];

state_fn = @(zxu) [zxu(2, 1); 0-sin(zxu(1, 1)) - 0.1*zxu(2, 1) + zxu(3, 1)];
u = nlmpcmove(nlobj, [0.2; 0], [0], [0], state_fn);
fprintf('first move: u = %.4f\n', u(1, 1));
```

### Nonlinear MPC — MIMO twin rotor (`examples/mpc/twin_rotor_nlmpc.m`)

A 2-DOF helicopter (`nlmpc(4, 2, 2)`) coordinates two cross-coupled rotors
to track pitch and yaw against a gravity nonlinearity — a pair of SISO
loops would fight each other through the off-axis torques. The example runs
a full closed loop with an RK4 plant integrator and a two-phase setpoint.

```matlab
nlobj = nlmpc(4, 2, 2);            % nx=4, ny=2, nu=2
nlobj.Ts = 0.1;   nlobj.p = 12;   nlobj.m = 3;
nlobj.umax = [4; 4];   nlobj.umin = [-4; -4];
nlobj.Wy  = [3; 3];    nlobj.Wdu = [0.1; 0.1];

state_fn = @(zxu) [zxu(3,1); zxu(4,1); ...
    (0.12*zxu(5,1) + 0.02*zxu(6,1) - 0.02*zxu(3,1) - 0.4*sin(zxu(1,1)))/0.05; ...
    (0.10*zxu(6,1) + 0.03*zxu(5,1) - 0.02*zxu(4,1))/0.04];

u = nlmpcmove(nlobj, S, u_prev, r, state_fn);
```

Inside the loop, copy the solver return into a fresh local vector before
extracting the scalars — indexing the same builtin return at multiple
positions otherwise confuses type inference (see the `uu` pattern in the
file). Steady inputs match the analytic trim (u1=1.04, u2=−0.31).

### Other examples

Explicit and finite-set MPC are exercised by the gating tests
`test/Run/mpc_t4_explicit_siso.m` (36-point grid lookup table, zero
run-time QP) and `test/Run/mpc_t4_finite_surge_tank.m` (branch-and-bound
over a binary valve). The mflow SIL form lives at
`examples/mflowlink/coder/cocotb_mpc_sil.mflow`.

## Limitations & carve-outs

- Linear MPC `StateFcn` / nonlinear handles cannot capture workspace
  variables — inline the coefficients as literals.
- Nonlinear MPC Tier-5 uses RK4 (pendulum example uses Forward Euler) and
  the default tracking cost; `CustomCostFcn` and multistage NMPC are
  carve-downs.
- The mflow `MpcMove` SIL path emits a static-gain MPC approximation in
  SystemVerilog; full QP-solving SV is an open follow-up (the
  `lane_keeping_mpc_sil.mflow` form needs matrix-state extensions to the
  signal-width inference pass).
- In MIMO `nlmpcmove` loops, re-index the solver return through a fresh
  local vector to keep type inference stable.

## See also

- Roadmap: [`../mpc_toolbox_roadmap.md`](../mpc_toolbox_roadmap.md)
- Examples: `examples/mpc/`
