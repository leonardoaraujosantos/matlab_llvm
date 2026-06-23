# Quadrotor flight control — mflowLink block diagrams

Drag-and-drop **signal-flow** versions of the quadrotor cascade controller from
`examples/quadrotor/` (the MATLAB `quadrotor_pid_mpc.m` study), built as
`.mflow` models that run on the mflowLink simulator. They mirror the editor
reference `quadrotor_cascade.mflow` shipped with the MatForge IDE.

| File | Outer (position) loop | Inner (attitude) loop | Altitude |
| --- | --- | --- | --- |
| `quadrotor_pid.mflow` | **cascade PID** — a position PID emits the tilt command | attitude PID | PID |
| `quadrotor_mpc.mflow` | **MPC** (`signal_mpc_move`, velocity-lead damping) | attitude PID | PID |

## The plant (hover linearization)

Same model the MATLAB derivation (`quadrotor_derive_eom.m`) and MPC use, with
`m = 1`, `g = 9.81`, `Ix = Iy = 0.01`:

```
  tilt cmd ─► [attitude PID] ─► 1/Ix=100 ─► ∫ ─► θ̇ ─► ∫ ─► θ ─┐
                    ▲                                          │
                    └────────────── θ feedback ───────────────┘
  θ ─► ×g(9.81) ─► ∫ ─► ẋ ─► ∫ ─► x        (x¨ = g·θ)
```

The horizontal axis is a **double-integrator-through-tilt**: the attitude loop
slaves `θ → θ_cmd` fast, and `x¨ = g·θ` turns tilt into translation. Altitude is
a direct `z¨ = (1/m)·u` double integrator. Both examples drive `x` and `z` to a
**1 m step** and log `θ`, `x`/`z` position, and the reference.

## Two outer-loop strategies

- **PID** (`quadrotor_pid.mflow`): the outer position error feeds a PID whose
  derivative term damps the double integrator (`θ_cmd = Kp·(x_ref−x) − Kd·ẋ`).
  Gains: outer `Kp=0.15, Kd=0.22`; attitude `Kp=6, Kd=0.45`; altitude
  `Kp=8, Ki=2, Kd=4`. Fast, with a small overshoot.
- **MPC** (`quadrotor_mpc.mflow`): `signal_mpc_move` computes the tilt command as
  `gain·(r − ym)` where `ym = x + 0.8·ẋ` carries a **velocity lead** for damping
  (the static-gain MPC approximation the simulator ships). Smoother, overshoot-free
  approach to the setpoint.

The vertical (altitude) loop is a plain PID in both — thrust acts directly on
`z`, so no inner attitude loop is needed there.

## Run

```sh
# interpreter
build/matlabc -simulate examples/quadrotor/mflowlink/quadrotor_pid.mflow
build/matlabc -simulate examples/quadrotor/mflowlink/quadrotor_mpc.mflow

# standalone compiled binary (byte-identical CSV to the interpreter)
build/matlabc -emit-mflowlink-cpp examples/quadrotor/mflowlink/quadrotor_mpc.mflow > /tmp/q.cpp
clang++ -std=c++17 -O2 -Iinclude /tmp/q.cpp build/libMatlab*.a -o /tmp/q && /tmp/q
```

Both run in **interpreted and compiled** mode with byte-identical output, and are
regression-gated in CI (`flowchart-simulate-run-tests` asserts `x→1` and `z→1`;
`flowchart-emit-mflowlink-cpp-tests` checks compiled-vs-interpreted parity).

> These are the single-horizontal-axis (`x`) + altitude (`z`) cores; the lateral
> `y` axis is symmetric to `x` with the roll sign flipped (`y¨ = −g·φ`), exactly
> as the 3-axis IDE `quadrotor_cascade.mflow` and the MATLAB `quadrotor_pid_mpc.m`
> figure-8 demo extend it.
