# Tasks

Sliced into six tiers (matching the project's per-toolbox tier convention). Each tier ships
at least one `.mflow` example under `examples/mflowlink/3d/` plus `SimulateRun` checks, so
every feature is exercised by a test model. Tiers 1–2 are the gate (scene + timeline + the
emit lane) that everything else builds on.

## 1. Scene + kinematic actors (primitives) + the Babylon emit lane — the gate

- [x] 1.1 Add `signal_world3d` to the model/loader: `gravity = "0,0,-9.81"`, `viewpoint`,
  `engine = "havok"`, `output` (HTML path), `background`, `showGround`/`showAxes`/`physics`;
  one per model (a second is a sourced error). No input/output ports (config block).
- [x] 1.2 Add `signal_actor3d` (primitive): `shape = box|sphere|cylinder|cone|capsule|plane`,
  `size`/`radius`/`height` dims, `color`, `name`; input ports `translation` (3), `rotation`
  (3, roll/pitch/yaw rad), `scale` (3). Unconnected transform ports default to identity
  (static `translation`/`rotation`/`scale` params override; scale defaults to 1).
- [x] 1.3 `MflowLinkSim`: record a per-major-step transform timeline — `signal_actor3d` is an
  implicitly-logged sink emitting a width-9 `<id>[tx..sz]` column group via the existing log /
  snapshot stream (reused, no new structure); the emit lane reads it back. (Delta-encoding of
  unchanged actors deferred — a size optimization, not correctness.)
- [x] 1.4 `-emit-mflowlink-babylon` target: serialise scene-graph + timeline to embedded JSON
  in an HTML template carrying all viewer logic inline; right-handed Z-up metres root (D6).
  Engine CDN-referenced by default; `--babylon-cdn <url>` overrides the host. (Full engine
  vendoring/inlining is a documented packaging follow-on.)
- [x] 1.5 Headless emit test: generated HTML is a valid document; actor count and timeline
  length (== sim major steps) match the model. No browser/GPU in the test path
  (`test/Flowchart/SimulateRun/run_tests.sh`).
- [x] 1.6 Example `orbit_cube.mflow` (a `signal_sine`/`cos` pair drives a box's X/Y
  translation → circular orbit; scope + emit); `SimulateRun` asserts the recorded timeline
  matches the analytic trajectory (tx≈3 at t=π/2, tz held at 1). (`spin_top.mflow` deferred to
  the Tier-2 batch.)
- [x] 1.7 Full flowchart ctest + SimulateRun green (263/0); block-kind parity snapshot updated;
  existing models byte-identical (no `signal_*3d` ⇒ no scene, no behavior change).

## 2. Transforms, hierarchy, lights, cameras, materials

- [ ] 2.1 Actor parent/child: `parent = "<actorName>"` ⇒ the child's transform is relative to
  the parent (composed in the timeline). Cycle detection is a sourced error.
- [ ] 2.2 `signal_light3d`: `type = directional|point|spot`, `color`, `intensity`; input ports
  `position` (3), `direction` (3), `intensity` (1). Ambient term on the world.
- [ ] 2.3 `signal_camera3d`: `mode = static|follow`, `position`/`target` (static) or
  `follow = "<actorName>"` + offset; emitted as the viewer's initial/active camera.
- [ ] 2.4 Materials: per-actor `material` color/emissive/opacity; a ground `plane` and an
  XYZ axis triad as world conveniences (`showGround`, `showAxes`).
- [ ] 2.5 Example: `articulated_arm.mflow` — a 3-link chain via parent/child actors, each
  joint angle driven by a `signal_sine`; a `follow` camera + a directional light.
  `SimulateRun` checks the end-effector world transform equals the composed FK.
- [ ] 2.6 Example: `quadrotor_flythrough.mflow` — reuse the quadrotor demo's pose signals to
  drive a body actor + 4 rotor child actors; a chase camera. (Headline visual.)

## 3. Mesh import — glTF/GLB + URDF (reuse robotics toolbox)

- [ ] 3.1 `signal_actor3d` `mesh = "<file.glb|.gltf>"`: load/validate the mesh, embed it in
  the emitted scene (Babylon native loader); transforms drive it exactly as a primitive.
- [ ] 3.2 `signal_actor3d` `urdf = "<file.urdf>"`: parse via the shipped robotics toolbox
  (`rigidBodyTree`/`loadrobot`), and drive each link's transform from a `jointAngles` input
  port through `getTransform` FK — so the URDF visual matches the robotics goldens (D5/risk).
- [ ] 3.3 Example: `gltf_drone.mflow` — a glTF drone body driven by a trajectory; asserts the
  mesh is embedded and the timeline length matches.
- [ ] 3.4 Example: `urdf_arm_trace.mflow` — `loadrobot` arm; joint signals from an IK/`sine`
  source; `SimulateRun` asserts end-effector pose == `getTransform`, and the emitted scene has
  one node per link.

## 4. Viewer-side Havok/Ammo physics (visual gravity + collisions)

- [ ] 4.1 `signal_world3d.physics = true` + per-actor `physics = true`, `mass`, `friction`,
  `restitution`, `collisionShape = box|sphere|convexHull|mesh`. Emit these into the scene so
  the viewer seeds Havok rigid bodies (initial pose/velocity from the model) and integrates
  under the world gravity, resolving collisions for rendering.
- [ ] 4.2 `engine = "havok" | "ammo"` selects the viewer physics backend behind one viewer
  interface; both inlined builds available; default Havok.
- [ ] 4.3 Document explicitly (params + roadmap + emit-test comment) that tier-4 physics is
  **visualization-only** — its result never re-enters the model and is excluded from goldens.
- [ ] 4.4 Example: `falling_stack.mflow` — a stack of boxes + a ground plane, `physics = true`,
  no transform inputs ⇒ they fall and settle in the viewer. Emit test asserts physics bodies
  + mass/restitution are present in the scene JSON (no rendering assertion).
- [ ] 4.5 Example: `ball_ramp.mflow` — a sphere dropped onto an inclined plane (restitution
  bounce). Same structural emit assertions.

## 5. Lock-step co-simulation feedback (deterministic C++ physics → signals)

- [ ] 5.1 A deterministic fixed-step rigid-body/contact step in `MflowLinkSim` (impulse or
  penalty; documented method) for actors flagged `cosim = true`; integrates with the major
  step and is fully reproducible (golden-stable, platform-independent).
- [ ] 5.2 `signal_actor3d` physics-state **output** ports: `pose` (6: xyz+rpy), `velocity`
  (6), `contact` (1, boolean). Available only when `cosim = true`.
- [ ] 5.3 `signal_collision3d` event block: inputs reference two actors (or an actor + world);
  output is a collision boolean + contact force, accumulated once per major step (mirrors
  `signal_error_rate`'s once-per-step rule).
- [ ] 5.4 Example: `bounce_cosim.mflow` — a sphere under co-sim gravity bounces on the ground;
  `SimulateRun` asserts the pose/contact signals match the analytic bounce within tolerance
  (a true golden, unlike tier 4).
- [ ] 5.5 Example: `cart_wall_bump.mflow` — the inverted-pendulum-on-cart PID demo, where the
  cart hits a wall; `signal_collision3d` feeds the controller and `SimulateRun` asserts the
  cart stops at the wall. (Closes the loop: controller reacts to a collision.)

## 6. Sensors, synthetic data, annotations, pacing, recording

- [ ] 6.1 `signal_sensor3d`: deterministic C++ scene sampling →
  `kind = depth|semantic|lidar|rgb`. Depth ⇒ `[rows,cols]` rank-2; semantic ⇒ `[rows,cols]`
  class ids; lidar ⇒ `[numPoints,3]`; rgb ⇒ `[rows,cols,3]` rank-3 (flat-shaded raster).
  Outputs are N-D signals (`mflow-nd-signals`) that flow into image/CV blocks.
- [ ] 6.2 `signal_actor3d` `semanticLabel` (class id) so the semantic sensor has ground truth.
- [ ] 6.3 Annotations: a `signal_actor3d` `text` actor (billboarded label) updatable from a
  signal; world-space placement.
- [ ] 6.4 Pacing + recording: honour `pacingRate` in the emitted player (real-time playback);
  a `--record` emit flag also writes a frame sequence / GIF of the timeline for headless
  artifacts (PNG sequence via the existing `saveas` raster path where the scene is primitive).
- [ ] 6.5 Example: `camera_depth_stream.mflow` — a `follow` camera on a moving actor produces
  a depth + semantic stream into an image-processing block (e.g. `signal_image_filter` or a CV
  block); `SimulateRun` asserts the depth signal shape `[rows,cols]` and a known near/far value.
- [ ] 6.6 Example: `lidar_scan.mflow` — a lidar sensor over a few primitives; asserts the
  point-cloud shape `[N,3]` and that points lie on the primitive surfaces.

## 7. Docs

- [ ] 7.1 `docs/mflowlink_3d_animation_roadmap.md` — the tiered plan (this change's companion;
  drafted in this change).
- [ ] 7.2 `docs/mflowlink_blocks.md`: a "3-D animation" section documenting every `signal_*3d`
  block's `params` (camelCase) and ports, the coordinate convention (D6), and the
  visualization-vs-authoritative physics rule (D3).
- [ ] 7.3 Document `-emit-mflowlink-babylon` (and `--babylon-cdn`, `--record`) in the emit/CLI
  docs beside `-emit-mflowlink-cpp`; note the self-contained-HTML contract and the embedded
  scene-JSON shape (so the IDE webview can consume it).
