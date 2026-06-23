# Quadrotor flight control — mflowLink block diagrams

Drag-and-drop **signal-flow** versions of the quadrotor cascade controller from
`examples/quadrotor/` (the MATLAB `quadrotor_pid_mpc.m` study), built as
`.mflow` models that run on the mflowLink simulator. They mirror the editor
reference `quadrotor_cascade.mflow` shipped with the MatForge IDE.

| File | Outer (position) loop | Inner (attitude) loop | Altitude |
| --- | --- | --- | --- |
| `quadrotor_pid.mflow` | **cascade PID** — a position PID emits the tilt command | attitude PID | PID |
| `quadrotor_mpc.mflow` | **MPC** (`signal_mpc_move`, velocity-lead damping) | attitude PID | PID |
| `plot_trajectory.m` | companion script — renders the logged 3-D path (`plot3`) | | |

Both models control the **full 3-DOF** position: forward `x → 1.0 m`, lateral
`y → 1.5 m`, altitude `z → 1.0 m`.

## The plant (hover linearization)

Same model the MATLAB derivation (`quadrotor_derive_eom.m`) and MPC use, with
`m = 1`, `g = 9.81`, `Ix = Iy = 0.01`:

```
  tilt cmd ─► [attitude PID] ─► 1/Ix=100 ─► ∫ ─► θ̇ ─► ∫ ─► θ ─┐
                    ▲                                          │
                    └────────────── θ feedback ───────────────┘
  θ ─► ×g(9.81) ─► ∫ ─► ẋ ─► ∫ ─► x        (x¨ = g·θ)
```

Each horizontal axis is a **double-integrator-through-tilt**: the attitude loop
slaves the angle to its command fast, and `x¨ = g·θ` / `y¨ = −g·φ` turns tilt
into translation. The **lateral `y` axis** is the structural twin of `x` with the
roll convention `y¨ = −g·φ`: the PID model carries that sign with a `−1` gain on
the outer command plus `g = −9.81`, and the MPC model with the outer gain `−0.4`
(exactly as the 3-axis IDE `quadrotor_cascade.mflow` does). Altitude is a direct
`z¨ = (1/m)·u` double integrator driven by a plain PID — thrust acts on `z`
directly, so no inner attitude loop is needed there.

## Two outer-loop strategies

- **PID** (`quadrotor_pid.mflow`): the outer position error feeds a PID whose
  derivative term damps the double integrator. Gains: outer `Kp=0.15, Kd=0.22`;
  attitude `Kp=6, Kd=0.45`; altitude `Kp=8, Ki=2, Kd=4`. Fast, small overshoot.
- **MPC** (`quadrotor_mpc.mflow`): `signal_mpc_move` computes the tilt command as
  `gain·(r − ym)` where `ym = pos + 0.8·vel` carries a **velocity lead** for
  damping (the static-gain MPC approximation the simulator ships). Smoother,
  overshoot-free approach to the setpoint.

## Visualizing the 3-D path — `signal_scope3d`

Both models end in a **`signal_scope3d`** block: a scope with three inputs
(`x`, `y`, `z`) that logs the trajectory as a `traj[x] / traj[y] / traj[z]`
column group — the marker a 3-D path viewer (the IDE, or the companion script)
uses to render the flight in space.

```sh
# 1) simulate → CSV (columns: t, traj[x], traj[y], traj[z])
build/matlabc -simulate examples/quadrotor/mflowlink/quadrotor_mpc.mflow \
    > quadrotor_traj.csv

# 2) render the 3-D path to quadrotor_traj.png (needs a plot-enabled build)
build/matlabc -repl examples/quadrotor/mflowlink/plot_trajectory.m
```

## Run

```sh
# interpreter
build/matlabc -simulate examples/quadrotor/mflowlink/quadrotor_pid.mflow

# standalone compiled binary (byte-identical CSV to the interpreter)
build/matlabc -emit-mflowlink-cpp examples/quadrotor/mflowlink/quadrotor_mpc.mflow > /tmp/q.cpp
clang++ -std=c++17 -O2 -Iinclude /tmp/q.cpp build/libMatlab*.a -o /tmp/q && /tmp/q
```

Both run in **interpreted and compiled** mode with byte-identical output, and are
regression-gated in CI (`flowchart-simulate-run-tests` asserts `x→1`, `y→1.5`,
`z→1` and the 3-D scope header; `flowchart-emit-mflowlink-cpp-tests` checks
compiled-vs-interpreted parity).
