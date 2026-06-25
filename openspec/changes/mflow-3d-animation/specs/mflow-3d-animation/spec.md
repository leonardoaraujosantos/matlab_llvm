## ADDED Requirements

### Requirement: 3-D world and kinematic actors

A mflowLink model SHALL be able to declare a single 3-D scene with one `signal_world3d`
block and any number of `signal_actor3d` blocks. An actor SHALL carry a primitive geometry
(`box`, `sphere`, `cylinder`, `cone`, `capsule`, or `plane`) and SHALL accept `translation`
(3), `rotation` (3, roll/pitch/yaw in radians), and `scale` (3) input ports driven by model
signals; unconnected transform ports default to identity (zero translation/rotation, unit
scale). Declaring a second `signal_world3d` SHALL be a sourced error. The coordinate frame
SHALL be right-handed, Z-up, in metres.

#### Scenario: Signal-driven actor transform
- **GIVEN** a `signal_actor3d` with `shape = box` whose `translation` port is driven by a
  signal evaluating to `[1, 2, 3]` at time `t`
- **WHEN** the model is simulated
- **THEN** the recorded transform timeline places that actor at translation `[1, 2, 3]` at `t`

#### Scenario: Two worlds rejected
- **WHEN** a model declares two `signal_world3d` blocks
- **THEN** lowering reports a sourced error

### Requirement: Recorded transform timeline

Simulation SHALL record, for each major step, every actor's translation, rotation, scale,
color, and visibility into a transform timeline keyed by simulation time, at the same cadence
as the signal log. The timeline produced by `-simulate` and the timeline embedded by
`-emit-mflowlink-babylon` SHALL be identical for the same model.

#### Scenario: Timeline length matches sim steps
- **WHEN** a scene model is simulated for N recorded major steps
- **THEN** the transform timeline has N keyframes (after delta-encoding, with unchanged actors
  carried forward)

### Requirement: Babylon HTML export

`-emit-mflowlink-babylon` SHALL write a single `.html` file that embeds the scene-graph, the
transform timeline, and all viewer logic inline, such that opening the file renders and plays
the animation. The Babylon/Havok engine SHALL be referenced from a pinned CDN host by default,
overridable with `--babylon-cdn <url>`; full asset vendoring (inlining the engine for a
network-free artifact) is a documented packaging follow-on. Output SHALL go to stdout or, when
`-o <file>` is given, to that file.

#### Scenario: Emitted artifact
- **WHEN** a scene model is emitted with `-emit-mflowlink-babylon -o scene.html`
- **THEN** `scene.html` is a valid standalone HTML document whose embedded scene lists every
  actor (and, in later tiers, light/camera) in the model and whose timeline length equals the
  simulated step count
- **AND** the scene-graph, keyframe timeline, and viewer logic are embedded inline (only the
  Babylon engine `<script>` is external, from the default or `--babylon-cdn` host)

### Requirement: Lights, cameras, hierarchy, and materials

The scene SHALL support `signal_light3d` (directional, point, or spot, with signal-drivable
position/direction/intensity), `signal_camera3d` (a static viewpoint or a follow-actor
camera), actor parent/child hierarchy (a child transform is composed relative to its parent),
and per-actor material color. A parent cycle SHALL be a sourced error.

#### Scenario: Child transform is relative to parent
- **GIVEN** actor `child` with `parent = "base"`, `base` at translation `[1,0,0]`, and `child`
  at local translation `[0,1,0]`
- **WHEN** the model is simulated
- **THEN** the recorded world translation of `child` is `[1,1,0]`

### Requirement: Mesh and URDF actors

A `signal_actor3d` SHALL accept a `mesh` path (glTF/GLB) embedded into the emitted scene and
driven by transforms exactly like a primitive, and a `urdf` path whose links are driven from a
`jointAngles` input port using the shipped robotics forward-kinematics path, so the visualized
pose matches the robotics-toolbox result.

#### Scenario: URDF pose matches forward kinematics
- **GIVEN** a `signal_actor3d` with a URDF arm and `jointAngles` driven by a known signal
- **WHEN** the model is simulated
- **THEN** the recorded end-effector transform equals `getTransform` of the same robot at those
  joint angles, and the emitted scene has one node per robot link

### Requirement: Viewer-side physics (visualization)

When `signal_world3d.physics` is true, a `signal_actor3d` with `physics = true` SHALL emit its
`mass`, `friction`, `restitution`, and `collisionShape` so the viewer seeds a rigid body that
integrates under the world gravity and resolves collisions for rendering. The viewer physics
backend SHALL be selectable via `engine = "havok" | "ammo"`. Viewer physics SHALL be
visualization-only: its result SHALL NOT re-enter the model and SHALL NOT affect any simulation
golden.

#### Scenario: Physics body emitted, not fed back
- **GIVEN** a `falling_stack` model with `physics = true` actors and no transform inputs
- **WHEN** it is emitted
- **THEN** the scene JSON marks each actor as a physics body with its mass/restitution/collision
  shape
- **AND** the simulation CSV/timeline for those actors is unchanged by toggling `engine` between
  `havok` and `ammo` (the engine affects only in-viewer rendering)

### Requirement: Lock-step co-simulation feedback

A `signal_actor3d` with `cosim = true` SHALL expose `pose` (6), `velocity` (6), and `contact`
(1) output ports computed by a deterministic fixed-step rigid-body/contact solver inside the
simulator, and a `signal_collision3d` block SHALL output a collision boolean and contact force
for a referenced actor pair. These signals SHALL be deterministic and golden-stable across
platforms, and SHALL be usable as inputs to the rest of the model.

#### Scenario: Controller reacts to a collision
- **GIVEN** a cart driven by a controller, a wall, and a `signal_collision3d` between them whose
  output feeds the controller
- **WHEN** the cart reaches the wall
- **THEN** the collision signal asserts and the recorded cart position stops at the wall within
  tolerance, identically across runs and platforms

#### Scenario: Co-sim bounce is a golden
- **WHEN** a sphere under `cosim` gravity bounces on the ground plane
- **THEN** the recorded `pose`/`contact` signals match the analytic bounce within tolerance and
  are byte-stable between `-simulate` and `-emit-mflowlink-cpp`

### Requirement: 3-D sensors as N-D signals

`signal_sensor3d` SHALL synthesise sensor data deterministically from the scene each step:
`kind = depth` as a `[rows, cols]` signal, `kind = semantic` as a `[rows, cols]` class-id
signal, `kind = lidar` as a `[numPoints, 3]` signal, and `kind = rgb` as a `[rows, cols, 3]`
signal, each carried as an N-D wire signal so it can feed image-processing and computer-vision
blocks.

#### Scenario: Depth sensor feeds an image block
- **GIVEN** a `signal_sensor3d` with `kind = depth`, `rows = R`, `cols = C` aimed at a primitive
- **WHEN** the model is simulated
- **THEN** its output is a rank-2 `[R, C]` signal whose values are the distances to the primitive
  surface (and background/far elsewhere), accepted as input by an image-processing block

#### Scenario: Lidar point-cloud shape
- **WHEN** a `signal_sensor3d` with `kind = lidar` scans a scene of primitives
- **THEN** its output is a `[N, 3]` signal whose points lie on the primitive surfaces
