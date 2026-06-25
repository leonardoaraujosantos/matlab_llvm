# mflowLink — 3-D Animation & Physics (Babylon.js + Havok)

Plan for a **3-D scene, animation, and physics** capability on top of the mflowLink
signal-flow backend — the open-stack analogue of MATLAB's **Simulink 3D Animation**
(`sl3d_ug.pdf`, R2026a). Where Sim3D co-simulates with **Unreal Engine** (closed,
multi-GB, GPU/DirectX-bound), mflowLink co-simulates with **Babylon.js** (rendering) and
**Havok Physics** (the WASM engine Babylon ships), pluggable to **Ammo.js**. The
deliverable on the compiler/runtime side is a `signal_*3d` block family plus a
`-emit-mflowlink-babylon` lane that writes one **self-contained `.html`** playing a
recorded scene + transform timeline.

**Status: Tiers 1–4 shipped (Tier 3 glTF done, URDF pending); 5–6 planned.** Tier 1 (scene
+ kinematic primitive actors + the `-emit-mflowlink-babylon` HTML player), Tier 2
(parent/child hierarchy, lights, cameras, materials), Tier 3a (glTF/GLB mesh import), and
Tier 4 (viewer-side Havok/Ammo physics) are implemented and gated (full flowchart ctest +
275 SimulateRun checks green). The OpenSpec change is
`openspec/changes/mflow-3d-animation/` (proposal / design / tasks / spec). This doc is the
tiered companion, in the style of `verilog_a_plan.md` and `mflow_link_roadmap.md`.

Companion to:
- [`mflow_link_roadmap.md`](mflow_link_roadmap.md) — the signal-flow backend this extends;
  `signal_scope3d` (the x/y/z trajectory scope) is the precedent this generalises.
- [`mflowlink_blocks.md`](mflowlink_blocks.md) — the per-block `params` catalogue; the
  `signal_*3d` blocks get a "3-D animation" section there.
- `mflow-nd-signals` (shipped) — sensor outputs (RGB / depth / lidar) are its rank-2/3 case.
- The robotics toolbox (`rigidBodyTree` / `loadrobot` / `getTransform`) — URDF actors reuse
  its forward kinematics so visuals match the robotics goldens.

Babylon.js: <https://github.com/BabylonJS/Babylon.js> · Havok:
<https://github.com/BabylonJS/havok>.

---

## 0. Why not Unreal — the design pivot

Simulink 3D Animation's entire architecture is *lock-step co-simulation with a game engine*:
MATLAB/Simulink computes the dynamics, Unreal renders and (optionally) runs the physics, and
the two exchange data each step. That model is sound; the *engine choice* is the problem for
this project:

| Sim3D / Unreal | mflowLink / Babylon + Havok |
|---|---|
| Closed, multi-GB, separate install | Open, MIT/Apache, inlined into the artifact |
| GPU + DirectX required | WebGL2/WebGPU; runs anywhere a browser does |
| Cannot be a self-contained file | One self-contained `.html` (WASM inlined) |
| No headless CI without a GPU farm | Emit + parse the HTML structurally — no GPU |
| External physics owns the numbers | **mflowLink owns the numbers**; engine renders |

The last row is the key project-philosophy fit: mflowLink already *wraps* `ode45` rather
than reimplementing it and guarantees byte-identical goldens between `-simulate` and
`-emit-mflowlink-cpp`. We keep that — the **authoritative dynamics stay in the deterministic
C++ sim**; Havok/Ammo are the *visualization* physics (and, in tier 5, a separate
deterministic C++ contact step provides authoritative collision feedback). See design
decision **D3**.

## 1. The capability map (Sim3D → mflowLink)

| Sim3D concept | mflowLink block | Notes |
|---|---|---|
| `sim3d.World` / Simulation 3D Scene Configuration | `signal_world3d` | one per model: gravity, viewpoint, pacing, engine, output |
| `sim3d.Actor` / Simulation 3D Actor | `signal_actor3d` | primitive / mesh / URDF; transform input ports; physics attrs |
| `createShape` primitives | `shape = box\|sphere\|cylinder\|cone\|capsule\|plane` | tier 1 |
| `createMesh` / import (glTF/STL/FBX/URDF) | `mesh = *.glb\|*.gltf`, `urdf = *.urdf` | glTF + URDF tier 3; STL/FBX deferred |
| `Translation`/`Rotation`/`Scale` | `translation`/`rotation`/`scale` input ports | roll/pitch/yaw rad, Z-up |
| `sim3d.Light` | `signal_light3d` | directional / point / spot |
| viewpoint / `createViewpoint` / Scene view | `signal_camera3d` | static or follow-actor |
| `Physics`/`Gravity`/`Mass`/`Friction`/`Restitution`/`Collisions` | actor physics params + `signal_world3d.physics` | tier 4 viewer (visual), tier 5 co-sim (authoritative) |
| `LinearVelocity`/`AngularVelocity`/`Force`/`Torque` | co-sim state ports / inputs | tier 5 |
| collision callbacks / event containers | `signal_collision3d` | tier 5 |
| camera / lidar / depth / semantic / point-cloud sensors | `signal_sensor3d` (`kind = rgb\|depth\|semantic\|lidar`) | tier 6; N-D signals |
| annotations | `signal_actor3d` `text` actor | tier 6 |
| `EnablePacing`/`PacingRate` | `signal_world3d.pacingRate` + viewer playback | tier 6 |
| Unreal / RoadRunner / PBR / weather / particles / packaged exe | — | **carved out** (non-goals) |

## 2. Tiers

Six tiers, matching the project's per-toolbox convention. Each ships ≥ 1 `.mflow` example
under `examples/mflowlink/3d/` with `SimulateRun` checks.

### Tier 1 — Scene + kinematic actors (primitives) + the emit lane *(the gate)*
`signal_world3d` + `signal_actor3d` (primitives) with signal-driven transform ports; the sim
records a per-step transform timeline; `-emit-mflowlink-babylon` writes the self-contained
HTML player. **Examples:** `orbit_cube.mflow`, `spin_top.mflow`. **Headline:** a box orbiting
a circle, driven by a sine/cosine pair, played in the browser from a single file.

### Tier 2 — Transforms, hierarchy, lights, cameras, materials
Parent/child relative transforms, `signal_light3d`, `signal_camera3d` (static + follow),
materials, ground plane, axis triad. **Examples:** `articulated_arm.mflow` (3-link chain),
`quadrotor_flythrough.mflow` (reuse the quadrotor pose signals + chase camera). **Headline:**
the quadrotor demo, now flown through a lit 3-D scene with a chase cam.

### Tier 3 — Mesh import (glTF/GLB + URDF)
`mesh` (glTF/GLB embedded) and `urdf` (links driven via the robotics-toolbox FK). **Examples:**
`gltf_drone.mflow`, `urdf_arm_trace.mflow`. **Headline:** a `loadrobot` arm tracing an IK path
in 3-D, each link pose equal to `getTransform`.

### Tier 4 — Viewer-side Havok/Ammo physics (visual)
`signal_world3d.physics` + actor `mass`/`friction`/`restitution`/`collisionShape`; the viewer
seeds Havok (or Ammo) rigid bodies that fall and collide for rendering. **Visualization-only**
(D3) — never fed back, never a golden. **Examples:** `falling_stack.mflow`, `ball_ramp.mflow`.
**Headline:** a stack of boxes toppling under gravity in the browser.

### Tier 5 — Lock-step co-simulation feedback (deterministic C++ physics → signals)
A deterministic fixed-step rigid-body/contact solver in `MflowLinkSim` exposes `pose`,
`velocity`, `contact` actor outputs + a `signal_collision3d` event block — authoritative,
golden-stable. **Examples:** `bounce_cosim.mflow` (analytic bounce golden),
`cart_wall_bump.mflow` (PID cart stops at a wall via the collision signal). **Headline:** a
controller that *reacts* to a collision, closing the loop.

### Tier 6 — Sensors, synthetic data, annotations, pacing, recording
`signal_sensor3d` (`rgb`/`depth`/`semantic`/`lidar`) as N-D signals feeding the
image-processing / computer-vision blocks; text annotations; `pacingRate` playback;
`--record` frame/GIF capture. **Examples:** `camera_depth_stream.mflow`, `lidar_scan.mflow`.
**Headline:** a moving camera's depth + semantic stream piped into a CV pipeline — synthetic
data generation, end to end, with no GPU.

## 3. Coordinate & physics conventions (D6 / D3)

- **Frame:** right-handed, **Z-up**, metres. Rotation is roll/pitch/yaw about X/Y/Z in
  radians (matches `sl3d_ug` and the robotics/sensor-fusion toolboxes). The Babylon viewer is
  set `useRightHandedSystem = true` with a Z-up root, so model math never sees Babylon's
  default Y-up/left-handed frame.
- **Two physics layers, kept distinct:**
  - *Viewer physics (tier 4, Havok/Ammo):* visualization only. Rich, but its floating-point
    result varies across WASM builds and is **excluded from every golden**.
  - *Co-sim physics (tier 5, deterministic C++):* authoritative. All tested collision /
    contact / pose numbers come from here and are byte-stable across platforms.

## 4. The emit artifact

`matlabc model.mflow -emit-mflowlink-babylon -o scene.html` → one self-contained file:
Babylon.js + Havok WASM inlined (base64), with the scene-graph and transform timeline as an
embedded, readable JSON blob (so the IDE webview can consume the same contract). `--babylon-cdn`
references a pinned CDN build instead of inlining (smaller file, needs network). `--record`
also writes a PNG/GIF of the timeline for headless artifacts. CI validates the document
structurally (valid HTML; actor/light/camera counts; timeline length == sim steps) — it never
renders, so no browser or GPU is required.

## 5. Carve-outs (non-goals)

Unreal Engine, RoadRunner scene import, photorealistic/PBR lighting, weather & particle
effects (rain/snow/smoke/fog), packaged-executable scenes, GPU/DirectX dependency, a browser in
the test path, STL/FBX import, skeletal/bone animation, and large-scale actor instancing. These
are the parts of Sim3D that are inseparable from the game engine or out of scope for a
deterministic, self-contained, headless-testable backend; revisit individually if a concrete
demand appears.
