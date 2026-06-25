## Why

`examples/control/` ships forward-looking Control System Toolbox tours, but every
example is text-only 2-D math — the only inverted-pendulum content is a *linearized*
single pendulum used as an eigenvalue/pole-placement plant (`eig_poles_demo.m`,
`place_pole_assignment.m`). Meanwhile the 3-D animation surface (`sim3d.*`,
`examples/sim3d/`, `examples/quadrotor/quadrotor_pid_mpc_3d.m`) can render a
controller driving a body through a scene, but no example marries a *pendulum* to a
3-D scene, and there is no double inverted pendulum anywhere in the repo. The
canonical undergraduate/graduate controls demo — stabilizing an inverted pendulum on
a cart, watched in 3-D — is missing, even though every primitive needed to build it
already ships.

## What Changes

- **New example directory `examples/control/3d/`** containing a 3-D-animated
  inverted-pendulum example suite. Two plants, three controllers each (six runnable
  programs):
  - **Cart-pole** (single inverted pendulum on a cart): PID, LQR (state-space),
    pole-placement.
  - **Double inverted pendulum** on a cart: PID, LQR, pole-placement.
- Each program: builds a `sim3d.World` (ground `plane`, cart `box`, pole(s) as
  `cylinder` actors **parented to the cart** via `setParent`), integrates the
  **nonlinear** plant in the loop, applies the controller, records one keyframe per
  control step, and exports a self-contained Babylon.js HTML player via
  `sim3d.export`. Runs interpreted (`matlabc -repl`) and compiled with identical
  output.
- **Controllers reuse shipped CST builtins**: `lqr`/`dlqr` (state-space), `place`
  with `ctrb`/`obsv` (pole-placement), and `ss` for the linearized design model. PID
  is hand-tuned classical control.
- **Honest documentation of the underactuated PID limit**: a single-input double
  inverted pendulum is not robustly stabilizable by independent PID loops; that
  example uses a documented best-effort full-state PD (weighted sum of position +
  link-angle errors styled as PID gains) and says so in its header and the README.
- **New `examples/control/3d/README.md`** — a reading-order tour (plant equations,
  controller table, run instructions) mirroring the style of `examples/control/README.md`.
- **Cross-link** added to `examples/sim3d/README.md` pointing at the new control-3d
  suite as an applied counterpart to the orbit/vehicle demos.

No new runtime behavior, builtins, or language features — this change is additive
example and documentation content exercising already-shipped capabilities.

## Capabilities

### New Capabilities
- `control-3d-examples`: A documented, runnable example suite that demonstrates
  stabilizing controllers (PID, LQR, pole-placement) on inverted-pendulum plants
  (cart-pole and double inverted pendulum) with 3-D Babylon.js animation, built
  entirely on the shipped `sim3d` command-line API and Control System Toolbox builtins.

### Modified Capabilities
<!-- None. This change adds example/documentation content only; it does not alter the
     requirements of the existing `control-system-toolbox` capability or the sim3d API. -->

## Impact

- **Added files**: `examples/control/3d/cartpole_pid_3d.m`,
  `cartpole_lqr_3d.m`, `cartpole_place_3d.m`,
  `double_pendulum_pid_3d.m`, `double_pendulum_lqr_3d.m`,
  `double_pendulum_place_3d.m`, `examples/control/3d/README.md`.
- **Edited files**: `examples/sim3d/README.md` (cross-link), optionally
  `examples/control/README.md` (pointer to the 3-D subdir).
- **Depends on (already shipped, no changes)**: the `sim3d` command-line API
  (`sim3d.World`/`Actor`, `Translation`/`Rotation`/`Scale`/`setParent`,
  `add`/`open`/`run`/`close`, `sim3d.export`) and CST builtins (`lqr`, `dlqr`,
  `place`, `ctrb`, `obsv`, `ss`).
- **No impact** on runtime libraries, the compiler, or existing examples/tests.
