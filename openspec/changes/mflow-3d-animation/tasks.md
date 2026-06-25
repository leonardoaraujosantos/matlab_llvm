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

- [x] 2.1 Actor parent/child: `parent = "<actorName>"` ⇒ the child's recorded transform is its
  local frame, composed with the parent via the viewer scene graph. An unknown parent or a
  cycle in the chain is a sourced error (validated at lowering).
- [x] 2.2 `signal_light3d`: `type = directional|point|spot`, `color`, `intensity`, `position`,
  `direction` (static config block; the viewer adds a hemispheric fill + each light).
  Signal-driven light pose/intensity is a documented follow-on.
- [x] 2.3 `signal_camera3d`: `mode = static|follow`, `position`/`target` (static) or
  `follow = "<actorName>"`; emitted as the viewer's active camera (first camera wins).
- [x] 2.4 Materials: per-actor `color`/`emissive`/`opacity`; ground plane + XYZ axis triad as
  world conveniences (`showGround`, `showAxes` — shipped in Tier 1).
- [x] 2.5 Example: `articulated_arm.mflow` — a 3-link chain via parent/child actors, each joint
  rotation driven by a `signal_sine`; a `follow` camera + a directional light. `SimulateRun`
  asserts the emitted scene has the 3-link parent chain, the light, and the follow camera.
  (World-pose composition is the viewer's scene-graph job per design D2; the analytic FK
  golden moves to the URDF actor in Tier 3, which calls the robotics `getTransform`.)
- [~] 2.6 Example: `quadrotor_flythrough.mflow` — reuse the quadrotor demo's pose signals to
  drive a body actor + chase camera. Deferred to land with the Tier-3 mesh import so the body
  is a glTF drone rather than a primitive (avoids a throwaway primitive-only version).

## 3. Mesh import — glTF/GLB + URDF (reuse robotics toolbox)

- [x] 3.1 `signal_actor3d` `mesh = "<file.glb|.gltf>"`: read + validate (missing ⇒ sourced
  error), embed inline as a base64 `data:` URL resolved against the .mflow dir; the viewer
  loads it with `SceneLoader.ImportMesh` and transforms drive it exactly as a primitive.
- [x] 3.2 `signal_actor3d` `urdf = "<file.urdf>"`: a minimal URDF parser (links + box/cylinder/
  sphere visuals + joints w/ origin/axis/type) runs at emit time and emits the tree; the viewer
  builds one node per link and rotates each movable joint by `jointAngles[q]` about its axis,
  composing FK via the scene graph (no robotics-runtime linkage). The URDF actor logs up to 12
  joint angles (`<id>[q1..q12]`) alongside its base transform. Missing/unparseable URDF ⇒
  sourced error.
- [x] 3.3 Example: `gltf_drone.mflow` — a glTF drone body on a Lissajous trajectory with a
  chase camera (also subsumes the deferred Tier-2.6 flythrough); `SimulateRun` asserts the
  mesh is embedded inline and the timeline length matches the sim step count. Fixture
  `assets/drone.gltf` (minimal embedded-buffer glTF 2.0 box).
- [x] 3.4 Example: `urdf_arm_trace.mflow` — a 2-DOF URDF arm (`assets/arm2.urdf`); joint signals
  from two sines through the `jointAngles` port; `SimulateRun` asserts the logged joint angle
  tracks its driving signal (q1 = 1.2·sin(0.6·t)) and the emitted scene has 3 links + 2 joints.

## 4. Viewer-side Havok/Ammo physics (visual gravity + collisions)

- [x] 4.1 `signal_world3d.physics = true` + per-actor `physics = true`, `mass`, `friction`,
  `restitution`, `collisionShape = box|sphere|convexHull|mesh`. The viewer seeds rigid bodies
  (initial pose from the recorded keyframe, in Babylon world frame) via `PhysicsAggregate` and
  integrates under the mapped world gravity; physics actors are excluded from the timeline
  animation. The Havok WASM is loaded only for physics scenes.
- [x] 4.2 `engine = "havok" | "ammo"` selects the viewer physics backend (HavokPlugin /
  AmmoJSPlugin) behind one async-init path; default Havok.
- [x] 4.3 Documented (params doc + roadmap + the emit JS comment) that Tier-4 physics is
  **visualization-only** — its result never re-enters the model and is excluded from goldens.
- [x] 4.4 Example: `falling_stack.mflow` — a stack of boxes + ground, `physics = true`, no
  transform inputs. Emit test asserts the Havok engine + `PhysicsAggregate` bodies are present
  (no rendering assertion); a non-physics scene must omit the Havok WASM.
- [x] 4.5 Example: `ball_ramp.mflow` — a sphere dropped onto an inclined plane (restitution
  bounce). Emit test asserts physics is enabled.

## 5. Lock-step co-simulation feedback (deterministic C++ physics → signals)

- [x] 5.1 A `cosim = true` `signal_actor3d` owns 6 continuous states `[x,y,z,vx,vy,vz]`
  integrated under the world gravity by the existing RK4 (free-fall RK4-exact); a ground
  restitution bounce is resolved once per major step in `resolveCosimContacts()`. Fully
  reproducible / platform-independent (a true golden, unlike Tier-4 viewer physics).
- [x] 5.2 Co-sim actor exposes its physics state: the recorded transform carries the pose
  (position in `[tx,ty,tz]`, read by collision3d / controllers) and `x`/`y`/`z`/`vx`/`vy`/`vz`/
  `contact` are named scalar ports. (Full 6-vector `pose`/`velocity` ports are a follow-on —
  the single-`VecOut_` model carries the transform; named scalars cover the rest.)
- [x] 5.3 `signal_collision3d` event block: input ports `poseA`/`poseB` (xyz of two actors);
  emits a collision boolean on `out` + a penalty contact `force`. Loop-breaker, so it can feed
  a controller without an algebraic loop.
- [x] 5.4 Example: `bounce_cosim.mflow` — a sphere under co-sim gravity bounces on the ground;
  `SimulateRun` asserts free-fall is exact (z(0.5)=3.7737), the bounce peak follows e²·drop
  (2.056), and the body never sinks through the floor.
- [x] 5.5 Example: `cart_wall_bump.mflow` — a cosim cart slides toward a static wall;
  `signal_collision3d` emits the collision boolean (a controller-usable feedback signal) when
  they meet (t≈3.0s) and 0 before. (Closes the loop: a controller can react to the collision.)

## 6. Sensors, synthetic data, annotations, pacing, recording

- [x] 6.1 `signal_sensor3d`: a deterministic C++ raycaster (ray-sphere / ray-AABB-box /
  ground-plane) samples the scene each step → `kind = depth|semantic|lidar|rgb`. Depth ⇒
  `[rows,cols]`; semantic ⇒ `[rows,cols]` class ids; lidar ⇒ `[azimuth·elevation,3]`; rgb ⇒
  `[rows,cols,3]` (flat-shaded, depth-attenuated). Outputs are N-D signals (`mflow-nd-signals`),
  implicitly logged, that flow into the image/CV blocks.
- [x] 6.2 `signal_actor3d` `semanticLabel` (class id) is read by the semantic sensor as ground
  truth (and the raycaster reads actor geometry + color for rgb).
- [~] 6.3 Annotations (text actor) — deferred follow-on; not required for the sensor headline.
- [~] 6.4 Pacing + recording — the viewer already plays in real time against the recorded
  sample times; `pacingRate` honouring + a `--record` frame/GIF export are deferred follow-ons.
- [x] 6.5 Example: `camera_depth_stream.mflow` — a depth camera aimed at a unit sphere;
  `SimulateRun` asserts the `[6,6]` shape, the centre pixel ≈ 4 (5 − radius), and a corner ray
  reads the range (miss). Semantic/rgb share the same raycast (`semanticLabel` ground truth).
- [x] 6.6 Example: `lidar_scan.mflow` — a 24-ray lidar over a box post; `SimulateRun` asserts the
  `[24,3]` point-cloud shape and that the forward ray lands on the box front face (x≈3.5).

## 7. Docs

- [ ] 7.1 `docs/mflowlink_3d_animation_roadmap.md` — the tiered plan (this change's companion;
  drafted in this change).
- [ ] 7.2 `docs/mflowlink_blocks.md`: a "3-D animation" section documenting every `signal_*3d`
  block's `params` (camelCase) and ports, the coordinate convention (D6), and the
  visualization-vs-authoritative physics rule (D3).
- [ ] 7.3 Document `-emit-mflowlink-babylon` (and `--babylon-cdn`, `--record`) in the emit/CLI
  docs beside `-emit-mflowlink-cpp`; note the self-contained-HTML contract and the embedded
  scene-JSON shape (so the IDE webview can consume it).
