## Why

mflowLink can already *compute* rich dynamics — ODE solvers, state-space, PID/LQR/MPC,
rigid-body kinematics from the robotics toolbox, quadrotor and inverted-pendulum demos —
but it can only *show* them as 2-D scope traces plus the recent `signal_scope3d`
trajectory polyline. There is no way to watch a body move, rotate, collide, or settle
under gravity in a real 3-D scene. MATLAB's **Simulink 3D Animation** (`sl3d_ug.pdf`,
R2026a) fills exactly this gap — but it does so by co-simulating with **Unreal Engine**,
a closed, multi-GB, GPU/DirectX-bound game engine that cannot be inlined into a
self-contained artifact, cannot run headless in CI, and is the opposite of this project's
"deterministic, self-contained, byte-identical golden" philosophy.

This change brings the *same capability surface* (a 3-D world; actors with geometry,
transforms, and physical attributes; lights; cameras/viewpoints; sensors; collision
events; lock-step co-simulation) to mflowLink using an **open, embeddable, WASM stack**:
**Babylon.js** for rendering and **Havok Physics** (the engine Babylon ships, distributed
as a WASM module) for in-viewer rigid-body dynamics — pluggable to **Ammo.js** via an
engine selector. The deliverable on this (compiler/runtime) side is a new
`-emit-mflowlink-babylon` lane that writes a **single self-contained `.html`** (Babylon +
Havok WASM inlined) which plays the recorded scene plus a per-step transform timeline —
matching the existing standalone-simulator and `saveas`-to-PNG precedents, testable
headlessly, and usable with no IDE.

This also composes with two already-shipped capabilities: the simulator's deterministic
timeline (so the *authoritative numbers stay in mflowLink*, as it already wraps `ode45`
rather than reimplementing it) and **N-D wire signals** (`mflow-nd-signals`) — a virtual
camera's RGB frame, a depth map, or a lidar point cloud is just a rank-3/rank-2 signal
that flows straight into the image-processing and computer-vision toolbox blocks.

## What Changes

A new `signal_*3d` block family and a Babylon emit lane, sliced into six tiers (full
detail in `docs/mflowlink_3d_animation_roadmap.md`). The headline additions:

- **3-D world + kinematic actors (primitives).** `signal_world3d` (one per model: gravity
  vector, viewpoint, pacing, output path) and `signal_actor3d` with a primitive geometry
  (`box`/`sphere`/`cylinder`/`cone`/`capsule`/`plane`) whose `translation` (3),
  `rotation` (3, roll/pitch/yaw), `scale` (3), and `color` input ports are driven by model
  signals. Simulation records a per-step transform timeline.
- **`-emit-mflowlink-babylon` HTML player.** A new emit lane lowers the recorded scene-graph
  + transform timeline to one `.html` with the scene + timeline + viewer logic embedded
  inline (the Babylon/Havok engine is CDN-referenced by default, `--babylon-cdn` overrides;
  full vendoring is a follow-on); opening it plays/scrubs the 3-D animation. Deterministic
  from the same sim output that drives the CSV.
- **Transforms, hierarchy, lights, cameras, materials.** Parent/child actors (relative
  transforms), `signal_light3d` (directional/point/spot), `signal_camera3d` (static
  viewpoint + follow-actor), per-actor material/texture color, ground plane + axis triad.
- **Mesh import — glTF/GLB + URDF.** `signal_actor3d` accepts a `mesh` path (glTF/GLB,
  Babylon's native format) or a `urdf` path; URDF reuses the shipped robotics toolbox
  (`rigidBodyTree`/`loadrobot`/`getTransform`) so a robot arm or quadrotor visualizes from
  its joint signals. (STL/FBX deferred.)
- **Viewer-side Havok physics (visual).** `signal_world3d` gains a `physics` flag; physics
  actors carry `mass`/`friction`/`restitution`/`collisionShape` and seed a Havok rigid
  body in the viewer that integrates under gravity and resolves collisions for rendering.
  `engine = "havok" | "ammo"` selects the backend.
- **Lock-step co-simulation feedback.** A deterministic C++ rigid-body/contact step inside
  `MflowLinkSim` feeds collision booleans, contact forces, and resulting poses back into
  the model as signals — `signal_actor3d` physics-state output ports + a
  `signal_collision3d` event block — so a controller can react to a collision. (The C++
  step, not the viewer WASM, is the *authoritative* number, preserving byte-identical
  goldens; Havok/Ammo remain the visualization.)
- **Sensors, synthetic data, annotations, pacing, recording.** `signal_sensor3d`: virtual
  camera RGB (rank-3 N-D signal), depth map (rank-2), semantic-segmentation labels, and
  lidar point cloud — flowing into existing image/CV blocks. Plus text annotations,
  `PacingRate`, and viewport frame capture (PNG/GIF/embedded timeline).

Each tier ships at least one `.mflow` example under `examples/mflowlink/3d/` with
`SimulateRun` checks, so every feature is exercised by a test model.

## Capabilities

### New Capabilities
- `mflow-3d-animation`: the mflowLink 3-D scene/animation model — the `signal_*3d` block
  family (world, actor, light, camera, sensor, collision), the recorded scene-graph +
  transform-timeline contract, the `-emit-mflowlink-babylon` self-contained HTML player,
  pluggable viewer physics (Havok/Ammo), and the deterministic lock-step co-simulation
  feedback path.

### Modified Capabilities
- (none expected) — the signal-flow frontend and `MflowLinkSim` describe the wire/sim
  model generically; this adds a focused 3-D capability and a new emit target. If
  conformance review finds a spec that states an explicit "2-D scope only" or "no emit
  target X" limit, a delta is added at the specs phase. `mflow-nd-signals` is reused
  as-is (camera/lidar signals are its rank-2/3 case).

## Impact

- **Model/loader** (`include/matlab/Flowchart/MflowLinkModel.h`, `flowchart_schema.md`):
  additive `signal_*3d` kinds + their `params`; no schema version bump (additive, as with
  prior mflow changes).
- **Sim** (`include/matlab/Flowchart/MflowLinkSim.h`, `lib/Flowchart/MflowLinkSim.cpp`):
  per-step actor-transform recording; the deterministic rigid-body/contact step (tier 5);
  sensor signal synthesis (tier 6, reuses N-D buffers).
- **Lowering** (`lib/Flowchart/SignalFlowLowering.cpp`): `signal_*3d` block stamping +
  shape inference for sensor signals (rank-2/3, via `mflow-nd-signals`).
- **Emit** (`tools/matlabc/`, `lib/Flowchart/`): a new `-emit-mflowlink-babylon` target
  beside `-emit-mflowlink-cpp`; the inlined Babylon/Havok viewer template.
- **Reuse, no change**: `mflow-nd-signals` (sensor signals), robotics toolbox
  (`rigidBodyTree`/FK for URDF), the existing ODE/state-space solvers (authoritative
  dynamics), the snapshot ring + DAP server (the timeline is the snapshot stream).
- **Tests**: per-tier `.mflow` fixtures under `examples/mflowlink/3d/` + `SimulateRun`
  transform/collision/sensor checks; a headless emit-and-parse check on the generated HTML
  (valid document, expected actor/keyframe counts) so CI needs no browser/GPU.
- **Docs**: `docs/mflowlink_3d_animation_roadmap.md` (the tiered plan, this change's
  companion) and a "3-D animation" section in `docs/mflowlink_blocks.md`.

## Non-Goals

- No Unreal Engine, RoadRunner import, photorealistic/PBR lighting, weather/particle
  effects, or packaged-executable scenes (the heavy Sim3D surface that depends on the game
  engine).
- No GPU/DirectX dependency and no browser in the test path — the viewer is a self-contained
  artifact; CI validates the emitted document structurally, not by rendering it.
- Havok/Ammo are the *visualization* physics; they are **not** the authoritative numbers.
  Reproducible goldens come from the deterministic C++ step, mirroring how mflowLink wraps
  its own solvers rather than depending on an external engine for the simulation result.
- STL/FBX mesh import, skeletal/bone animation, and multi-actor instancing at scale are
  deferred follow-ons.
