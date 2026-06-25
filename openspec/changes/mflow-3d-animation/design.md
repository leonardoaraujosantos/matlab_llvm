## Context

mflowLink simulates a signal-flow `.mflow` model deterministically: `MflowLinkSim` advances
time (adaptive Dormand-Prince / multirate / zero-crossing), logs per-step signal values to a
CSV-shaped buffer, and feeds a snapshot ring that powers step/step-back and the DAP server.
`-emit-mflowlink-cpp` lowers the same model to a standalone C++ simulator that reproduces the
CSV byte-for-byte. Signals are flat row-major buffers carrying a rank-1–6 shape
(`mflow-nd-signals`). The robotics toolbox already provides `rigidBodyTree`/`loadrobot`/
`getTransform` forward kinematics; `signal_scope3d` already logs an `x`/`y`/`z` column group
as a trajectory.

Simulink 3D Animation (the reference, `sl3d_ug.pdf`) provides: a **World** (`sim3d.World` —
`StopTime`/`SampleTime`/`EnablePacing`/`PacingRate`/viewpoint); **Actors** (`sim3d.Actor` —
geometry via `createShape`/`createMesh`/3-D-file import; `Translation`/`Rotation`/`Scale`;
physical attributes `Physics`/`Gravity`/`Mass`/`LinearVelocity`/`AngularVelocity`/`Force`/
`Torque`/`Friction`/`Restitution`/`Collisions`/`Mobility`; create/delete at run time;
collision callbacks/event containers); **Lights** (`sim3d.Light`); **camera viewpoints**;
**sensors** (camera/lidar/depth/semantic/point-cloud); and **lock-step co-simulation** with
Unreal's physics engine. We reproduce that surface on the open Babylon.js + Havok stack.

## Goals / Non-Goals

**Goals:**
- A mflowLink model declares a 3-D scene: one `signal_world3d`, any number of
  `signal_actor3d`/`signal_light3d`/`signal_camera3d`/`signal_sensor3d`/`signal_collision3d`.
- Actor transforms are driven by model signals (kinematic) and the sim records a per-step
  transform timeline alongside the CSV.
- `-emit-mflowlink-babylon` writes one self-contained `.html` (Babylon.js + Havok WASM
  inlined) that plays/scrubs that timeline with no external assets or network.
- Optional viewer-side Havok/Ammo physics for visual gravity/collisions; optional
  deterministic C++ lock-step physics feeding collision/contact/pose signals back into the
  model.
- Sensors emit N-D signals (RGB rank-3, depth rank-2, point cloud rank-2) reusing
  `mflow-nd-signals`, so they flow into image/CV toolbox blocks.

**Non-Goals:** Unreal/RoadRunner, PBR/weather/particles, packaged executables, GPU/browser in
CI, STL/FBX import, skeletal animation, authoritative physics from the WASM engine.

## Decisions

### D1 — A `signal_*3d` block family; one `signal_world3d` per model

3-D entities are ordinary signal-flow blocks (same `.mflow` schema, additive `kind`s — no
version bump, matching every prior mflow change). `signal_world3d` is the singleton scene
config (gravity vector, viewpoint, pacing, `engine`, `output` path); a second `world3d` is a
sourced error. Actors/lights/cameras/sensors reference the world implicitly (the lowering
binds them to the single world). **Why blocks, not a side-file:** the transform drivers are
*signals* — they must inherit sample time, propagate through Mux/Gain/etc., and step with the
solver. A side-channel scene file would fork the model and break step/step-back and emit.

### D2 — The animation is a recorded timeline, not a live engine; numbers stay in mflowLink

The sim records, per major step, each actor's `(translation, rotation, scale, color,
visible)` into a transform timeline keyed by time — the snapshot ring already holds exactly
this cadence. The viewer *plays* that timeline. So tiers 1–3 are fully deterministic from the
existing solver output, identical between `-simulate` and the emitted HTML, exactly like the
CSV. **Why:** preserves the project's byte-identical-golden contract and keeps the
authoritative dynamics in mflowLink (it already wraps `ode45` rather than delegating).

### D3 — Two physics layers, kept distinct: viewer (Havok/Ammo, visual) vs. co-sim (C++, authoritative)

- **Tier 4, viewer physics:** when `signal_world3d.physics = true`, a physics actor's
  initial pose + velocity + `mass`/`friction`/`restitution`/`collisionShape` seed a Havok
  rigid body *in the browser*; Havok integrates under gravity and resolves collisions for
  **rendering only**. `engine = "havok" | "ammo"` swaps the backend behind one viewer-side
  interface. This is visualization — its result never re-enters the model and is not part of
  any golden.
- **Tier 5, co-sim feedback:** a *deterministic C++* fixed-step rigid-body/contact solver in
  `MflowLinkSim` computes collision booleans, contact forces, and resulting poses and exposes
  them as signals (`signal_actor3d` physics-state output ports; `signal_collision3d`). These
  *are* authoritative and golden-tested.

**Why split:** a WASM engine's floating-point results vary across builds/platforms and cannot
back a byte-identical golden; but it gives far richer visuals than a minimal C++ solver. So
the engine renders, the C++ step decides. (If a user enables both, the viewer shows Havok and
the model reacts to the C++ contact signals; the roadmap notes the divergence and keeps the
co-sim path the source of truth.)

### D4 — `-emit-mflowlink-babylon` produces one self-contained `.html`

A new emit target beside `-emit-mflowlink-cpp`. It serialises (a) the static scene-graph
(actors, geometry/material/mesh refs, lights, cameras, world settings) and (b) the transform
timeline, as a JSON blob embedded in an HTML template carrying all viewer logic inline.
Opening the file renders the scene and plays the timeline with a scrub bar. **Engine
delivery:** the Babylon/Havok engine is referenced from a pinned CDN host by default
(`--babylon-cdn <url>` overrides); inlining the engine as base64 for a fully network-free
artifact is a documented packaging follow-on (a ~5 MB vendored bundle), so the headless test
path stays browser-free regardless. The embedded JSON is the documented contract (emitted as
a readable `<script type="application/json">` section, not minified), so the IDE webview can
consume the same blob later. **Why:** matches `saveas`-to-PNG and the standalone-simulator
precedents and keeps the artifact portable and the test path GPU-free.

### D5 — Sensors are N-D signals (reuse `mflow-nd-signals`)

`signal_sensor3d` synthesises its output from the recorded scene each step: a camera renders
RGB as a `[rows, cols, 3]` rank-3 signal, depth as `[rows, cols]`, semantic labels as
`[rows, cols]` of class ids, lidar as `[numPoints, 3]`. These are produced by a deterministic
C++ software rasteriser/ray-caster over the primitive/mesh scene (not the GPU), so they are
golden-stable and flow directly into the shipped image-processing / computer-vision blocks.
**Why C++ raster, not the viewer:** sensor data must be authoritative (it feeds control/AI
algorithms), so it follows the D2/D3 rule — compute in mflowLink, render in Babylon.

### D6 — Coordinate system + units fixed and documented

Right-handed, Z-up, metres; rotation as roll/pitch/yaw (X/Y/Z) in radians, matching the
robotics/sensor-fusion toolboxes and the `sl3d_ug` actor convention (roll=X, pitch=Y, yaw=Z).
The Babylon viewer (Y-up, left-handed by default) is configured `useRightHandedSystem = true`
and a fixed Z-up root transform converts at the viewer boundary, so model-side math never
sees Babylon's frame. **Why:** one documented frame avoids the classic axis-swap bugs and
keeps actor poses consistent with `getTransform`/quaternion outputs already in the project.

## Risks / Trade-offs

- **Viewer physics ≠ golden (D3).** Mitigation: tier-4 viewer physics is explicitly
  visualization-only and excluded from goldens; the doc and the `physics` param docs say so;
  tier-5 co-sim uses the C++ step for all tested numbers.
- **Self-contained HTML size (inlined Babylon + Havok WASM ~ several MB).** Mitigation: one
  shared template; the per-model payload is just the scene JSON + timeline; document the size
  and offer a `--babylon-cdn` opt-out that references a pinned CDN build for non-air-gapped
  use (default stays inlined/self-contained).
- **CI has no browser/GPU.** Mitigation: the emit test parses the generated HTML structurally
  (valid document; expected actor/light/keyframe counts; timeline length == sim steps) — it
  never renders. Rendering correctness is a manual/example check.
- **Timeline size for long runs × many actors.** Mitigation: keyframe only on change beyond a
  tolerance (delta-encode static actors), and reuse the snapshot-ring decimation already used
  for scopes; document the cap.
- **URDF FK divergence from robotics toolbox.** Mitigation: URDF actors *call* the shipped
  `rigidBodyTree`/`getTransform` path rather than re-deriving FK, so a URDF visual matches the
  robotics goldens by construction.

## Migration Plan

Purely additive. Existing models have no `signal_*3d` blocks and are byte-identical. The new
emit target is opt-in (`-emit-mflowlink-babylon`); `-emit-mflowlink-cpp` and `-simulate` are
unchanged. No schema version bump, no data migration.

## Open Questions

- Should the co-sim C++ contact solver (tier 5) be a minimal impulse/penalty rigid-body
  integrator written here, or should we attempt a *headless deterministic* Havok WASM build
  for the authoritative step too? (Proposed: minimal deterministic C++ solver for goldens —
  headless-Havok is a follow-on only if the minimal solver proves insufficient, since it
  reintroduces the cross-build determinism risk D3 was created to avoid.)
- Default viewer physics engine — Havok or Ammo? (Proposed: Havok default, Ammo via
  `engine = "ammo"`; Havok is Babylon's first-party engine with the cleaner WASM.)
- Inline the Babylon/Havok WASM by default vs. CDN reference? (Proposed: inline by default for
  the self-contained guarantee; `--babylon-cdn` for users who prefer a small file.)
- Camera-sensor RGB via software rasteriser vs. a simpler "billboard projection" first?
  (Proposed: start with depth + semantic + lidar in tier 6 — purely geometric, exact — and a
  flat-shaded raster RGB; defer textured/lit RGB.)
