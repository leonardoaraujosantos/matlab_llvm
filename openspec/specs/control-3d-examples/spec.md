# control-3d-examples Specification

## Purpose
TBD - created by archiving change control-3d-pendulum-examples. Update Purpose after archive.
## Requirements
### Requirement: Cart-pole 3-D example suite

The repository SHALL ship, under `examples/control/3d/`, three runnable programs that
stabilize a single inverted pendulum on a cart (cart-pole) and visualize the result in
3-D: one using classical PID, one using state-space LQR, and one using pole-placement.
Each program SHALL integrate the **nonlinear** cart-pole plant in the control loop (not
only the linearized design model), SHALL build a `sim3d.World` whose pole is a
`cylinder` actor parented to the cart `box` via `setParent`, and SHALL export a
self-contained Babylon.js HTML player via `sim3d.export`.

#### Scenario: Cart-pole LQR example stabilizes the pole and exports a scene
- **WHEN** `examples/control/3d/cartpole_lqr_3d.m` is run interpreted (`matlabc -repl`)
  from an initial pole tilt
- **THEN** the program designs the gain via `lqr` (or `dlqr`) on the linearized
  `ss` model, drives the nonlinear plant so the pole angle converges toward upright,
  prints periodic diagnostics, and writes `cartpole_lqr_3d.html`

#### Scenario: Cart-pole pole-placement example uses ctrb/place
- **WHEN** `examples/control/3d/cartpole_place_3d.m` is run
- **THEN** the program verifies controllability with `ctrb`, designs the gain with
  `place` for a chosen stable pole set, stabilizes the nonlinear cart-pole, and exports
  `cartpole_place_3d.html`

#### Scenario: Cart-pole PID example stabilizes the pole
- **WHEN** `examples/control/3d/cartpole_pid_3d.m` is run
- **THEN** a classical PID law on the pole angle (with cart-position handling
  documented in the header) keeps the pole upright over the simulation and exports
  `cartpole_pid_3d.html`

### Requirement: Double inverted pendulum 3-D example suite

The repository SHALL ship, under `examples/control/3d/`, three runnable programs that
stabilize a double inverted pendulum on a cart and visualize the result in 3-D: one
using PID, one using LQR, and one using pole-placement. Each program SHALL integrate the
**nonlinear** two-link plant, SHALL render the cart plus two `cylinder` link actors via
the parented `sim3d` scene graph, and SHALL export a Babylon.js HTML player.

#### Scenario: Double-pendulum LQR example stabilizes both links
- **WHEN** `examples/control/3d/double_pendulum_lqr_3d.m` is run from a small initial
  perturbation
- **THEN** the program designs full-state feedback via `lqr` on the linearized
  6-state model, drives the nonlinear double pendulum so both link angles converge
  toward upright, and writes `double_pendulum_lqr_3d.html`

#### Scenario: Double-pendulum pole-placement example stabilizes both links
- **WHEN** `examples/control/3d/double_pendulum_place_3d.m` is run
- **THEN** the program confirms controllability with `ctrb`, designs the gain with
  `place`, stabilizes the nonlinear plant from a small perturbation, and exports
  `double_pendulum_place_3d.html`

#### Scenario: Double-pendulum PID example documents the underactuated limit
- **WHEN** `examples/control/3d/double_pendulum_pid_3d.m` is run
- **THEN** the program applies a documented best-effort full-state PD law (weighted
  sum of cart-position and both link-angle errors), its header and the README state
  honestly that independent PID loops cannot robustly stabilize a single-input double
  inverted pendulum, and it exports `double_pendulum_pid_3d.html`

### Requirement: 3-D control example documentation

The example suite SHALL include `examples/control/3d/README.md` that documents the plant
models, the controller-per-plant matrix, and run instructions in the reading-order style
of `examples/control/README.md`, and the `sim3d` README SHALL cross-link to it.

#### Scenario: README documents the suite
- **WHEN** a reader opens `examples/control/3d/README.md`
- **THEN** it lists all six programs with the plant and controller each demonstrates,
  states the nonlinear-plant / linearized-design split, gives the `matlabc -repl` run
  command and the emitted HTML filename, and notes the double-pendulum PID limitation

#### Scenario: sim3d README points at the applied suite
- **WHEN** a reader opens `examples/sim3d/README.md`
- **THEN** it contains a cross-link identifying `examples/control/3d/` as an applied
  control counterpart to the orbit/vehicle animation demos

### Requirement: Examples reuse only shipped primitives

The example suite SHALL depend only on already-shipped capabilities — the `sim3d`
command-line API and Control System Toolbox builtins (`lqr`, `dlqr`, `place`, `ctrb`,
`obsv`, `ss`) — and SHALL NOT introduce new runtime builtins, language features, or
viewer changes.

#### Scenario: No new runtime surface is added
- **WHEN** the change is implemented
- **THEN** only files under `examples/` are added or edited (plus this OpenSpec change),
  and no `runtime/`, compiler, or viewer source is modified

