# mflowLink 3-D Animation Examples (Babylon.js + Havok)

Test models for the **`mflow-3d-animation`** capability — the open-stack analogue of
Simulink 3D Animation. Plan: `docs/mflowlink_3d_animation_roadmap.md`; spec:
`openspec/changes/mflow-3d-animation/`.

> **Status: spec fixtures.** These models reference the planned `signal_*3d` block family
> (`signal_world3d`, `signal_actor3d`, `signal_light3d`, `signal_camera3d`,
> `signal_sensor3d`, `signal_collision3d`). The blocks are not implemented yet; the models
> are the concrete targets each tier is built against. Run with
> `matlabc <model>.mflow -simulate` and render with
> `matlabc <model>.mflow -emit-mflowlink-babylon -o <name>.html` once the tier lands.

One example per tier feature, so every feature has a test model:

| Tier | Example | Exercises |
|---|---|---|
| 1 | `orbit_cube.mflow` ✅ | world + kinematic primitive actor; sine/cos → translation; scope3d + emit |
| 1 | `spin_top.mflow` ✅ | ramp → yaw rotation |
| 2 | `articulated_arm.mflow` ✅ | parent/child hierarchy; directional light; follow camera; materials |
| 2 | `quadrotor_flythrough.mflow` ✅ | body + 4 parented spinning rotors; follow cam |
| 3 | `stl_marker.mflow` ✅ | STL mesh import (binary `assets/marker.stl`) |
| 3 | `gltf_drone.mflow` ✅ | glTF/GLB mesh import (inline `assets/drone.gltf`) + chase camera |
| 3 | `urdf_arm_trace.mflow` ✅ | URDF (`assets/arm2.urdf`) → link tree, joints from a signal |
| 4 | `falling_stack.mflow` ✅ | viewer-side Havok physics (visual gravity + box collisions) |
| 4 | `ball_ramp.mflow` ✅ | sphere on an inclined plane (restitution bounce) |
| 5 | `bounce_cosim.mflow` ✅ | deterministic C++ co-sim; analytic bounce golden |
| 5 | `cart_wall_bump.mflow` ✅ | cosim cart + `signal_collision3d` → collision feedback signal |
| 6 | `camera_depth_stream.mflow` ✅ | depth sensor (raycast) → N-D `[6,6]` signal |
| 6 | `lidar_scan.mflow` ✅ | lidar point cloud `[24,3]` over a box post |

✅ = concrete fixture written; others are enumerated in the change's `tasks.md` and authored
as their tier is implemented.

## Conventions

- **Frame:** right-handed, Z-up, metres; rotation = roll/pitch/yaw (X/Y/Z) in radians.
- **Physics:** tier-4 viewer physics (Havok/Ammo) is **visualization-only** — never a golden;
  tier-5 co-sim physics is the deterministic, authoritative, golden-tested path.
- **`params` are camelCase** (per `docs/mflowlink_blocks.md`); node-level `data` fields
  (`log_signal`, `sample_time`) are snake_case.
