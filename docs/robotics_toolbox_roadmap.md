# Robotics System Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Robotics-System-Toolbox programs.

Source: *Robotics System Toolbox User's Guide* (R2026a, 10 chapters:
Robot Modeling · Inverse Kinematics Examples · Motion and Path Planning
Examples · Robot Simulation Examples · Collision Detection · Code
Generation Examples · Offroad Autonomy for Heavy Machinery · Robotics
System Toolbox Topics · Examples for Simulink Blocks · Unreal Engine
Topics).

This is the **manipulator-and-mobile-robot sibling of the Sensor Fusion
roadmap** — the two share the same foundation (`quaternion` + the
coordinate-transformation surface) and the same consumer (the user's
`examples/quadrotor/` flight controller; the UG even ships a "Plan Path
of Robotic Arm Mounted on Quadrotor" example). It rests on an unusually
deep shipped base: **inverse kinematics is `lsqnonlin` / `fminunc` over
forward kinematics** — both already shipped in the Optimization Toolbox;
**forward dynamics is ODE integration of the manipulator equations of
motion** — `ode45` / `ode23s` are shipped; **Jacobians, the IK
damped-pseudoinverse, and the mass matrix ride** the shipped
`pinv` / `svd` / `qr` / `mldivide`; **trajectory generation rides** the
shipped `interp1` / cubic interpolation; **PRM/RRT sampling rides** the
seeded PRNG; and **URDF link meshes reuse the STL/GLB importer shipped
with the PDE Toolbox**. **No external dependency** (no KDL, no MoveIt, no
Pinocchio) — every kinematics, dynamics, planning, and collision routine
is hand-coded over the shipped kernel.

**Shared foundation with Sensor Fusion**: Tier-1 here is the
coordinate-transformation layer (`se3` / `so3` / `se2` / `so2` +
`quaternion` + the `axang`/`eul`/`rotm`/`trvec`/`tform` conversion
matrix). The **`quaternion` value type is the same one planned in
[`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md)
Tier-1** — whichever toolbox ships first builds it, the other reuses it.
This is the same value-type pattern as the shipped `fi` / `datetime`
(operator-overloaded array type, *no* System-Object dependency), so the
foundation tier is light. The classdef-bearing tiers (rigidBodyTree,
solvers, planners, maps) are read-mostly built-once-then-queried objects,
not the stateful `step`-driven System Objects of DSP/Comm — so this
toolbox is **far less gated on the System-Object lowering fix** than
Fusion/DSP (only the few stateful objects — `jointSpaceMotionModel`,
`stateEstimatorPF`, `rateControl` — touch it).

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/robotics/ik_path_trace.m`](../examples/robotics/ik_path_trace.m):
*the canonical manipulator demo — `loadrobot` a predefined arm (or build
a `rigidBodyTree`), then for each point on a 2-D path, solve
`inverseKinematics` for the joint configuration that places the
end-effector there, and `show` the arm tracing the path*. This exercises
the transform foundation (T1) → rigidBodyTree forward kinematics (T2) →
inverse kinematics (T3) arc end-to-end; achieving it closes the
**manipulator core** (the UG "2-D Path Tracing with Inverse Kinematics").
The **mobile-robot tracer** (closing T5) is
[`examples/robotics/diffdrive_prm.m`](../examples/robotics/diffdrive_prm.m):
*a `binaryOccupancyMap` + `mobileRobotPRM` path + `controllerPurePursuit`
driving a `differentialDriveKinematics` robot from start to goal* (the UG
"Path Following for a Differential Drive Robot").

Companion docs: [`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md)
(the shared `quaternion` + transform foundation), [`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)
(IK rides `lsqnonlin`/`fminunc`; CHOMP rides gradient descent),
[`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md) (URDF link meshes
reuse the STL/GLB importer), [`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md)
(the path-following MPC + the quadrotor cascade consume the kinematics),
[`ode.md`](ode.md) (forward dynamics + motion models integrate via
`ode45`/`ode23s`), [`plotting.md`](plotting.md) (`show` / map / path
plots route through Cairo 3-D), [`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the **coordinate-transformation foundation** (`se3`/`so3`/`se2`/`so2`
  + `quaternion` + the full `axang`/`eul`/`rotm`/`trvec`/`tform`/`quat`
  conversion matrix + `homtrans` + the `cross`/`dot`/`deg2rad` core gaps)
  — shared with Sensor Fusion. **Tier-2** is rigid-body-tree modeling +
  forward kinematics (`rigidBodyTree`/`rigidBody`/`rigidBodyJoint` +
  `loadrobot`/`importrobot` URDF + `getTransform`/`geometricJacobian` +
  `homeConfiguration`/`randomConfiguration` + `show`). **Tier-3** is
  inverse kinematics (`inverseKinematics`/`generalizedInverseKinematics`
  + constraint objects + `analyticalInverseKinematics`), riding the
  shipped Optim solvers. **Tier-4** is trajectory generation + manipulator
  dynamics + motion models (`trapveltraj`/`cubicpolytraj`/`quinticpolytraj`/
  `bsplinepolytraj`/`minjerkpolytraj`/`rottraj`/`transformtraj`/`contopptraj`
  + `massMatrix`/`inverseDynamics`/`forwardDynamics`/`gravityTorque` +
  `jointSpaceMotionModel`/`taskSpaceMotionModel`). **Tier-5** is mobile
  robots + occupancy maps + path planning (`differentialDriveKinematics`/
  `unicycleKinematics`/`bicycleKinematics`/`ackermannKinematics` +
  `binaryOccupancyMap`/`occupancyMap`/`mapClutter`/`mapMaze` +
  `mobileRobotPRM` + `controllerPurePursuit`/`controllerVFH`). **Tier-6**
  is manipulator planning + collision detection + estimation + carve-down
  polish (`manipulatorRRT`/`manipulatorCHOMP` + `collision*`/`checkCollision`
  + `stateEstimatorPF`/`odometryMotionModel`/`rateControl`).
- **Effort** is in the existing Phase 5.6.x cadence (one focused session
  ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~1.5 wk (≈0.5
  if Fusion T1 ships the quaternion first) · T2 ~2.5 wk · T3 ~1 wk (rides
  Optim) · T4 ~2.5 wk · T5 ~2.5 wk · T6 ~3 wk (~13 wk full)**. Each tier
  is independently shippable and demoable; **T1 + T2 + T3 (~5 wk) close
  the manipulator forward/inverse-kinematics core** — the highest-value
  cut and the foundation every robotics demo needs.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started. **All 6 tiers
  shipped 2026-05-27 in one PR + a follow-on slice** (partial per the
  documented carve-downs):
  T1 ✅ full transform surface, T2 ✅ (DH `loadrobot('planar2'/'planar3')`
  **+ URDF `importrobot`** — fixed-transform tree from `<joint>`/`<link>`/
  `<inertial>`; full per-body `rigidBody`/`rigidBodyJoint` classes carved),
  T3 ✅ LM-damped IK + `constraintPoseTarget` **+ `generalizedInverseKinematics`**
  (multi-constraint LM with `constraintPositionTarget`/`constraintOrientationTarget`;
  `analyticalInverseKinematics` carved), T4 ✅ cubic/trap/transform trajs **+
  full CRBA/RNEA dynamics** (`inverseDynamics`/`massMatrix`/`forwardDynamics`/
  `gravityTorque`/`velocityProduct`/`centerOfMass`; bspline/minjerk/contopptraj
  + `jointSpaceMotionModel` carved), T5 ✅ (all four mobile kinematic models
  diff-drive/unicycle/bicycle/ackermann + `derivative` + occmap + PRM +
  pure-pursuit; range sensor + VFH + stateSpaceSE3 carved), T6 ✅
  **orientation-aware GJK** `checkCollision` over box/sphere/cylinder/capsule
  + simplified `manipulatorRRT` (EPA penetration depth + `collisionMesh` +
  CHOMP + particle filter carved).
  **Everything below is 🔵 not started** — but the substrate is deep:
  `ode45`, `pinv`/`svd`/`qr`/`mldivide`, `fminunc`/`lsqnonlin`, `interp1`,
  the PRNG, the STL importer, classdef, and Cairo plotting are all
  shipped. The genuinely new surface is the **transform value types**, the
  **rigidBodyTree + kinematics/dynamics algorithms**, the **planners +
  occupancy maps**, and the **collision geometry**.
- **Object families**: (a) **value types** — `se3`/`so3`/`quaternion` are
  operator-overloaded arrays (the `fi`/`datetime` precedent, no SO
  dependency); (b) **built-once-then-queried** — `rigidBodyTree`,
  `inverseKinematics`, `mobileRobotPRM`, `occupancyMap`,
  `collisionBox`… are classdefs constructed once then called/queried (a
  *much* lighter classdef use than the stateful DSP/Fusion System
  Objects); (c) **the few stateful objects** — `jointSpaceMotionModel`
  (ODE state), `stateEstimatorPF` (particle set), `rateControl` (timer) —
  touch the shared System-Object lowering fix (CST §12 / Comm §15 / DSP
  Tier-1), but they are a small minority here.
- **No external dependencies**: matching the project precedent — transform
  algebra hand-coded; FK/Jacobian/CRBA/RNEA hand-coded over `ode45` +
  dense linalg; IK over the shipped `lsqnonlin`/`fminunc`; PRM/RRT over
  the PRNG; GJK collision hand-coded; URDF parsed by a hand-rolled XML
  reader reusing the STL mesh importer.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Robotics code yet)

| Group | Surface (already shipped) | Location | How Robotics uses it |
|---|---|---|---|
| ODE solvers | `ode45` (Dormand-Prince), `ode23s` (stiff), `ode_events` | `runtime/matlab_runtime.cpp` | Forward-dynamics integration; `jointSpaceMotionModel`/`taskSpaceMotionModel` rollouts; mobile-kinematics integration (T4/T5). |
| Optim solvers | `lsqnonlin` (Levenberg-Marquardt), `fminunc` (BFGS), `fmincon`, `quadprog` | `runtime/toolbox/optim/runtime_optim.cpp` | **`inverseKinematics` IS `lsqnonlin`/`fminunc` over forward kinematics** (T3); `generalizedInverseKinematics` weighted constraints; CHOMP gradient steps (T6). |
| Dense linear algebra | `pinv`, `svd`, `qr`, `mldivide`, `inv`, `chol`, `norm`, `cross`/`dot` (T1) | `runtime/matlab_runtime.cpp` | The Jacobian damped-pseudoinverse, mass-matrix inverse (CRBA), null-space projection, transform inverses (T1–T4). |
| Function-handle ABI | `void *fn_p` → `matlab_mat*(*)(…)`, `LowerAnonCalls` retyping | `runtime/toolbox/optim/runtime_optim.cpp` | The IK cost/Jacobian handles; custom constraint functions; ODE rollout RHS (T3/T4). |
| Interpolation | `interp1` (linear/pchip/spline/cubic), `interp2`, `trapz`, `cumtrapz` | `lib/Sema/Resolver.cpp` → `matlab_interp1` | `cubicpolytraj`/`quinticpolytraj`/`bsplinepolytraj`/`trapveltraj` waypoint trajectories (T4). |
| PRNG | `rand`/`randn`/`randi`/`randperm` + `rng(seed)` (reproducible) | `runtime/matlab_runtime.cpp` | `randomConfiguration`, PRM node sampling, RRT extension, particle-filter resampling (T2/T5/T6). |
| STL / GLB importer | `pde_import_stl`, GLB surface + voxelize-AABB | `runtime/runtime_pde.cpp` (PDE Toolbox) | `importrobot` link visual/collision meshes (T2); `collisionMesh` geometry (T6). |
| Classdef + state | `classdef`, handle semantics, `properties`/`methods`, `persistent`, `matlab_obj_new`/`_set_*`/`_get_mat`, class-pinned dispatch, REPL persist, DAP render, value-class copy-on-assign | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The transform value types (T1), `rigidBodyTree` / solver / planner / map objects (T2–T6). |
| Value-type precedent | `fi`, `datetime`, `quaternion` (Fusion T1) operator-overloaded array types | `lib/MLIR/Passes/LowerFixedPoint.cpp`, Fusion runtime | `se3`/`so3`/`se2`/`so2` follow the same overload + compose pattern (T1). |
| Reductions / sorting | `min`/`max`, `sort`, `sum`, `norm`, `vecnorm` | `runtime/matlab_runtime.cpp` | Nearest-neighbour in PRM/RRT, collision distance, trajectory bounds (T5/T6). |
| Trig / elementary | `sin`/`cos`/`atan2`/`sqrt`/`hypot`/`mod`/`wrapToPi` | `runtime/matlab_runtime.cpp` | Transform conversions, mobile kinematics, angle wrapping (T1/T5). |
| Plotting | Cairo `plot3` / `surf` / `patch` / `quiver3` / `scatter3` / `line` | `runtime/plot/` | Headless `show(robot)` / `show(map)` / path + trajectory plots → PNG/SVG (T2/T5). |
| Quaternion (planned) | `quaternion` value type | [`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md) Tier-1 | Shared rotation primitive — built once across the two roadmaps (T1). |

**Net assessment**: the *algorithmic base* (ODE integration, the Optim
solvers that *are* the IK core, dense linalg for Jacobians/dynamics, the
PRNG for sampling planners, the STL importer for URDF meshes, classdef +
plotting) is **already shipped**. The genuinely new code is (a) the
**transform value types + conversions**, (b) the **rigidBodyTree data
structure + FK/Jacobian/CRBA/RNEA algorithms + URDF parser**, (c) the
**IK solver wrappers** over shipped Optim, (d) the **trajectory
generators + motion models**, (e) the **occupancy maps + PRM/pure-pursuit
+ mobile kinematics**, and (f) the **manipulator RRT/CHOMP + GJK
collision**. Each is a self-contained hand-coded routine over the shipped
base.

---

## 2. Tier-1 — Coordinate transformations (SE3/SO3) + quaternion 🔵 (FOUNDATION)

Goal: the transform value types + the full conversion surface. The
primitive every other tier rests on; shared with Sensor Fusion. **No
System-Object dependency** — ships first.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `se3` / `so3` / `se2` / `so2` | Homogeneous-transform / rotation value types (N-array); `*` (compose), `inv`/`transform`/`rotm`/`trvec`/`tform`/`dist`/`interp` methods. | value-type pattern (`fi`), matmul |
| 1.2 | `quaternion` | The shared rotation type — Hamilton product, `rotatepoint`/`rotateframe`, `slerp`, `euler`/`rotmat`. **Built once with [`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md) Tier-1.** | Fusion T1 |
| 1.3 | translation conversions | `trvec2tform`/`tform2trvec`. | matrix assembly |
| 1.4 | rotation-matrix conversions | `rotm2tform`/`tform2rotm`, `eul2rotm`/`rotm2eul`, `axang2rotm`/`rotm2axang`, `quat2rotm`/`rotm2quat`. | trig |
| 1.5 | Euler / axis-angle / quat ↔ tform | `eul2tform`/`tform2eul`, `axang2tform`/`tform2axang`, `quat2tform`/`tform2quat`, `eul2quat`/`quat2eul`, `axang2quat`/`quat2axang`. | 1.3/1.4 |
| 1.6 | `homtrans` + utilities | `homtrans(T, pts)` (apply transform to points), `tform2adjoint`, `wrapToPi`/`wrapTo2Pi`. | matmul |
| 1.7 | core gaps | `cross`, `dot`, `deg2rad`/`rad2deg`, `vecnorm` — small core builtins this toolbox needs (shared with Fusion). | core |
| 1.8 | display + DAP | `disp(se3)` formats the 4×4 matrix; transform/quaternion arrays render in the REPL + DAP inspector. | `disp(obj)`, DAP |

**Headline-within-tier**: a transform chain —
`T = trvec2tform([1 0 0]) * eul2tform([pi/2 0 0]); p2 = homtrans(T, p);`
+ `se3` compose/inverse round-trip. The UG "Coordinate Transformations in
Robotics" reference.

**Compile/Execute wiring**: new `runtime/toolbox/robotics/runtime_robotics.cpp`
+ `robotics_classdefs.m` (`se3`/`so3`/`se2`/`so2`); value types store the
matrix plane; operator overloads via the `fi`/CST operator-method route;
conversions are matrix-in/matrix-out builtins (`Resolver.cpp` +
`LowerTensorOps.cpp`); ship `cross`/`dot`/`deg2rad`/`vecnorm` alongside.
If Fusion's `quaternion` is already in, reuse it.

---

## 3. Tier-2 — Rigid body tree + forward kinematics + Jacobian 🔵

Goal: the manipulator model + forward kinematics — the structure the IK,
dynamics, and planning tiers all query.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `rigidBodyTree` | The kinematic tree: `addBody`/`replaceBody`/`removeBody`, `Bodies`/`BaseName`/`Gravity`/`DataFormat` (`'struct'`/`'row'`/`'column'`). | classdef |
| 2.2 | `rigidBody` / `rigidBodyJoint` | Body + joint (`'revolute'`/`'prismatic'`/`'fixed'`); `setFixedTransform` (homogeneous **or DH parameters** `[a alpha d theta]`); joint limits. | T1 transforms |
| 2.3 | `loadrobot` | Predefined models (UR10/KINOVA Gen3/Atlas/…) from baked-in kinematic tables (the Comm-5G/Image-`fspecial` lookup-table precedent). | tables |
| 2.4 | `importrobot` | URDF (XML) parser → `rigidBodyTree`; link visual/collision meshes via the shipped STL/GLB importer. | STL importer (PDE) |
| 2.5 | `getTransform` | FK: body-to-base (or body-to-body) homogeneous transform at a configuration. | T1, tree walk |
| 2.6 | `geometricJacobian` | 6×N spatial Jacobian (linear + angular) at a body for a configuration. | T1, cross products |
| 2.7 | configs | `homeConfiguration`, `randomConfiguration` (within joint limits), `constraintJointBounds`. | PRNG |
| 2.8 | `show` (headless) | Render the tree at a configuration → Cairo 3-D patch/line artifact (PNG/SVG); frame triads. | `runtime/plot/` |

**Headline-within-tier**: the FK demo —
`gen3 = loadrobot("kinovaGen3"); T = getTransform(gen3, config, "EndEffector_Link"); show(gen3, config)`.
The UG "Load Predefined Robot Models" / "Build Basic Rigid Body Tree".

**Compile/Execute wiring**: `rigidBodyTree` is a built-once classdef
holding a body/joint array + the per-joint fixed transforms; `getTransform`
/ `geometricJacobian` are class-pinned methods (CST `pole(sys)` dispatch);
the URDF parser is a hand-rolled XML reader in `runtime_robotics.cpp`;
`show` writes headless artifacts.

---

## 4. Tier-3 — Inverse kinematics 🔵 (rides shipped Optim)

Goal: solve for the joint configuration that achieves a desired
end-effector pose — almost entirely a wrapper over the shipped Optim
solvers.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `inverseKinematics` | BFGS / Levenberg-Marquardt over the pose error `dist(getTransform(q), Tgoal)`; weighted 6-DOF (orientation + position) cost; `[config, info] = ik(...)`. | `lsqnonlin`/`fminunc`, T2 |
| 3.2 | `generalizedInverseKinematics` | Multi-constraint solver (weighted sum of constraint costs). | `fmincon`, T2 |
| 3.3 | constraint objects | `constraintPoseTarget`, `constraintPositionTarget`, `constraintOrientationTarget`, `constraintAiming`, `constraintCartesianBounds`, `constraintJointBounds`, `constraintRevoluteJoint`. | classdef |
| 3.4 | `analyticalInverseKinematics` | Closed-form IK for common 6-DOF wrist-partitioned arms (auto-derive from the tree); fall back to numeric. | T1/T2 |
| 3.5 | solver parameters + info | `SolverParameters` (max iters, tol, damping), `ExitFlag`/`Iterations`/`PoseErrorNorm` info struct. | Optim info |

**Headline-within-tier**: **the roadmap headline** — `ik_path_trace.m`:
`inverseKinematics` solving each waypoint of a 2-D path so the
end-effector traces it; the UG "2-D Path Tracing with Inverse Kinematics".

**Compile/Execute wiring**: `inverseKinematics` binds the FK (T2) into a
cost handle (function-handle ABI) and hands it to the shipped
`lsqnonlin`/`fminunc`; constraints are classdef carriers summed into the
generalized cost; multi-return `[config, info]` via the splitter.

---

## 5. Tier-4 — Trajectory generation + manipulator dynamics + motion models 🔵

Goal: the trajectory + dynamics layer — generate smooth joint/task paths
and the manipulator equations of motion.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | polynomial trajectories | `cubicpolytraj`/`quinticpolytraj`/`bsplinepolytraj` — waypoint → position/velocity/accel profiles. | `interp1`, polyfit |
| 4.2 | velocity-profile trajectories | `trapveltraj` (trapezoidal), `minjerkpolytraj`/`minsnappolytraj` (minimum jerk/snap). | QP / linalg |
| 4.3 | rotation/transform trajectories | `rottraj` (slerp over time), `transformtraj` (interpolate `se3`). | T1, slerp |
| 4.4 | time-optimal | `contopptraj` (TOPP-RA time-optimal path parameterisation under velocity/accel bounds). | linalg |
| 4.5 | mass matrix + RNEA | `massMatrix` (composite-rigid-body algorithm), `inverseDynamics` (recursive Newton-Euler), `gravityTorque`, `velocityProduct`, `centerOfMass`. | T2, dense linalg |
| 4.6 | forward dynamics | `forwardDynamics` (articulated-body / `M⁻¹(τ−C−G)`) integrated by `ode45`. | `ode45`, 4.5 |
| 4.7 | motion models | `jointSpaceMotionModel` (computed-torque / PD joint control), `taskSpaceMotionModel` (Cartesian impedance) — `derivative` + `ode45` rollout. | `ode45`, 4.5 |

**Headline-within-tier**: the UG "Simulate Joint-Space Trajectory
Tracking in MATLAB" — `trapveltraj` reference + `jointSpaceMotionModel`
tracking it with computed-torque control, joint-error plot.

**Compile/Execute wiring**: trajectory generators are matrix-in/matrix-out
builtins (multi-return `[q, qd, qdd] = ...`); `massMatrix`/`inverseDynamics`
are class-pinned `rigidBodyTree` methods; `forwardDynamics`/motion models
integrate via the shipped `ode45` with the function-handle ABI.
`jointSpaceMotionModel` is one of the few stateful objects (shared SO fix).

---

## 6. Tier-5 — Mobile robots + occupancy maps + path planning 🔵

Goal: the mobile-robot half — kinematic models, maps, and the
sample-based planner + path follower.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | mobile kinematics | `differentialDriveKinematics`/`unicycleKinematics`/`bicycleKinematics`/`ackermannKinematics` — `derivative(model, state, cmd)` integrated by `ode45`. | `ode45` |
| 5.2 | occupancy maps | `binaryOccupancyMap`/`occupancyMap` (probabilistic log-odds); `setOccupancy`/`getOccupancy`/`inflate`/`raycast`/`checkOccupancy`; `mapClutter`/`mapMaze` generators. | matrix grid, PRNG |
| 5.3 | `mobileRobotPRM` | Probabilistic roadmap: sample free nodes → connect within `ConnectionDistance` → graph search (`findpath`). | PRNG, graph |
| 5.4 | `controllerPurePursuit` | Look-ahead pure-pursuit path follower → `[v, omega]` commands. | T1, geometry |
| 5.5 | `controllerVFH` | Vector field histogram obstacle avoidance from range readings → steering. | histogram |
| 5.6 | state spaces | `stateSpaceSE2`/`stateSpaceSE3` + `validatorOccupancyMap` (sampling/interpolation/validation for planners). | T1, 5.2 |
| 5.7 | range sensor | `rangeSensor` (simulated lidar/ultrasonic over an occupancy map). | 5.2, raycast |

**Headline-within-tier**: **the mobile tracer** — `diffdrive_prm.m`:
`binaryOccupancyMap` + `mobileRobotPRM` path + `controllerPurePursuit`
driving a `differentialDriveKinematics` robot to the goal; the UG "Path
Following for a Differential Drive Robot".

**Compile/Execute wiring**: kinematic models are `derivative` + `ode45`;
the occupancy map is a built-once classdef over a matrix grid;
`mobileRobotPRM` samples + graph-searches in the runtime;
`controllerPurePursuit` is a small stateful object (look-ahead waypoint
index).

---

## 7. Tier-6 — Manipulator planning + collision detection + estimation + polish 🔵

Goal: the manipulator-planning + collision + localization layer, plus the
remaining polish.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | `manipulatorRRT` | RRT over the configuration space of a `rigidBodyTree` with collision checking; `plan`/`shorten`/`interpolate`. | PRNG, 6.3, T2 |
| 6.2 | `manipulatorCHOMP` | Covariant-Hamiltonian gradient trajectory optimisation (smoothness + obstacle cost). | gradient (Optim), 6.3 |
| 6.3 | collision geometry | `collisionBox`/`collisionSphere`/`collisionCylinder`/`collisionCapsule`/`collisionMesh`; `checkCollision` (GJK + EPA distance); self-collision on `rigidBodyTree`. | GJK hand-coded, STL importer |
| 6.4 | `stateEstimatorPF` | Particle-filter localization (`predict`/`correct`/`getStateEstimate`); `odometryMotionModel` + `likelihoodFieldSensorModel`. | PRNG, 5.2 |
| 6.5 | `rateControl` | Fixed-rate execution loop (`waitfor`); overrun policy. | timer |
| 6.6 | inverse-kinematics designer outputs | `analyticalInverseKinematics` generated solver function (the app's codegen output, headless). | T3 |
| 6.7 | carve-down polish | `interactiveRigidBodyTree` (→ static config artifact), `taskSpaceMotionModel` impedance, MPPI path-following controller. | T2/T4 |

**Headline-within-tier**: the UG "Pick-and-Place Workflow Using RRT for
Manipulators" (MATLAB-only variant) — `manipulatorRRT` planning a
collision-free arm path between two configurations with `collisionBox`
obstacles.

**Compile/Execute wiring**: `manipulatorRRT`/`manipulatorCHOMP` orchestrate
the T2 FK + the T6.3 collision checker over the PRNG/gradient; collision
primitives are classdef geometry + a hand-coded GJK distance;
`stateEstimatorPF`/`rateControl` are the few stateful objects (shared SO
fix).

---

## 8. Compile/Execute · Debug/REPL · Examples · Tests (cross-cutting)

The four delivery surfaces the project always closes per toolbox — each
tier ships across **all four** before it counts as done.

### 8.1 Compile / Execute

- **Backends**: LLVM JIT + native + `-emit-c` / `-emit-cpp` are the
  primary lanes (the UG dedicates Chapter 6 to code generation — FK/IK/
  motion-planning C-codegen, which maps onto the project's `-emit-c`
  lane). `-emit-python` / `-emit-typescript` parity is a per-tier stretch
  (the transform value types + trajectory functions port cleanly; the
  planners/maps are rougher). `-emit-systemverilog` is **not** a target
  (host-side kinematics/planning) — emit a clear diagnostic.
- **Runtime**: `runtime/toolbox/robotics/runtime_robotics.cpp` (transform
  algebra, FK/Jacobian/CRBA/RNEA, trajectory generators, occupancy maps,
  PRM/RRT/CHOMP, GJK collision, URDF parser) +
  `runtime/toolbox/robotics/robotics_classdefs.m` (`se3`/`so3`,
  `rigidBodyTree`, solvers, planners, maps, collision geometry). Add to
  the strict no-C-cast list (`static_cast`), mirroring `runtime_images.cpp`.
- **Wiring**: the transform value types (T1) use the `fi`/CST operator-
  overload route; `rigidBodyTree`/solvers/planners/maps are built-once
  classdefs with class-pinned methods (`getTransform`/`plan`/`show` via
  `Lowering.cpp::CallOrIndex`); trajectory generators + conversions +
  collision are plain builtins (`Resolver.cpp` + `LowerTensorOps.cpp`,
  string-option args → `matlab_string*`); multi-return splitters for
  `[config, info]` / `[q, qd, qdd]`; the **few stateful objects**
  (`jointSpaceMotionModel`/`stateEstimatorPF`/`rateControl`) use the
  shared System-Object lowering fix (CST §12 / Comm §15 / DSP Tier-1);
  prelude-trigger the `robotics` classdefs.

### 8.2 Debug / REPL

- The transform value types (`se3`/`so3`/`quaternion`) and built-once
  objects (`rigidBodyTree`, `inverseKinematics`, `occupancyMap`) persist
  across REPL inputs and render in the DAP variable inspector — for a
  `rigidBodyTree`, the body/joint names + `DataFormat`; for a transform,
  the 4×4 matrix.
- `disp(robot)` formats the MATLAB-faithful property block
  (`rigidBodyTree with properties: NumBodies: …`); `show(robot, config)`
  writes a headless PNG so a JIT-REPL FK check produces an inspectable
  artifact.
- The stateful motion models / particle filter render their evolving state
  (joint state / particle set) in the DAP inspector under a paused loop.

### 8.3 Examples (`examples/robotics/`)

| Example | Closes | Exercises |
|---|---|---|
| `transforms_chain.m` | T1 | `se3`/`eul2tform`/`trvec2tform` compose + `homtrans` apply |
| `load_robot_fk.m` | T2 | `loadrobot` → `getTransform` → `geometricJacobian` → `show` |
| `ik_path_trace.m` | **headline (T1+T2+T3)** | `inverseKinematics` tracing a 2-D path; end-effector error |
| `gik_constraints.m` | T3 | `generalizedInverseKinematics` with pose + aiming + joint-bound constraints |
| `joint_traj_track.m` | T4 | `trapveltraj` + `jointSpaceMotionModel` computed-torque tracking |
| `arm_dynamics.m` | T4 | `massMatrix`/`inverseDynamics`/`gravityTorque` on a loaded arm |
| `diffdrive_prm.m` | **tracer (T5)** | `binaryOccupancyMap` + `mobileRobotPRM` + `controllerPurePursuit` |
| `manipulator_rrt.m` | T6 | `manipulatorRRT` collision-free plan with `collisionBox` obstacles |

### 8.4 Tests (`test/Run/`)

Gating tests follow the `robotics_*.m` convention with a `.stdout` golden
+ per-backend `.skip-emit-*` files where a lane is out of scope (SV always
skipped; Python/TS skipped where the classdef/planner path is rough,
matching the Image `image_png_roundtrip` precedent).

| Test | Tier | Asserts |
|---|---|---|
| `robotics_transforms.m` | T1 | `eul2tform`/`tform2eul` round-trip; `se3` compose/inverse; `homtrans` vs known points |
| `robotics_quat_tform.m` | T1 | `quat2tform`/`axang2tform`/`rotm2tform` against known rotations |
| `robotics_rigidbodytree.m` | T2 | build a 3-DOF tree; `getTransform` FK vs hand-computed pose |
| `robotics_jacobian.m` | T2 | `geometricJacobian` vs finite-difference of FK |
| `robotics_ik.m` | **T3** | `inverseKinematics` recovers a known config; pose error < tol (headline) |
| `robotics_gik.m` | T3 | `generalizedInverseKinematics` satisfies multiple constraints |
| `robotics_trajectory.m` | T4 | `trapveltraj`/`cubicpolytraj` hit waypoints with continuous velocity |
| `robotics_dynamics.m` | T4 | `massMatrix` symmetric PD; `inverseDynamics` vs `forwardDynamics` round-trip |
| `robotics_occupancy_prm.m` | T5 | `binaryOccupancyMap` + `mobileRobotPRM` finds a valid collision-free path |
| `robotics_purepursuit.m` | T5 | `controllerPurePursuit` drives `differentialDriveKinematics` to the goal |
| `robotics_collision.m` | T6 | `checkCollision` true/false + distance on known box/sphere pairs |

Target: **~11 gating tests** (one+ per major surface), in line with Image
(10) and Stats (12). Full regression must stay green (currently 465
run-tests) — the badge bumps to **17 toolboxes** (or higher if the other
queued toolboxes land first). **Note**: only a few stateful objects share
the System-Object lowering fix, so this toolbox is largely shippable
*without* waiting on it (unlike Fusion/DSP).

---

## 9. Carve-outs (explicitly out of scope)

Matching the per-toolbox precedent — the Simulink / 3-D-engine / ROS /
support-package / Deep-Learning surfaces are deferred. **This toolbox's
example chapters lean heavily on external simulators**, so the carve-out
list is large:

- **All Simulink examples + blocks** (Chapter 9 — Robotics Manipulator
  blocks, mobile-robot path blocks, trajectory blocks) — the MATLAB
  object/function API is the whole target; the mflowLink lane is the
  project's block-diagram answer.
- **Unreal Engine simulation** (Chapter 10 — 3-D scenes, Simulation 3D
  blocks, photoreal sensors) — external engine.
- **Gazebo co-simulation** (Chapter 4 + the Topics Gazebo sections — `gzlink`,
  co-sim blocks, the Gazebo plugin) and **ROS / ROS 2 integration** — gated
  on a future ROS Toolbox; the pure-MATLAB pick-and-place variants are in
  scope.
- **Offroad Autonomy for Heavy Machinery** (Chapter 7 — the support-package
  library, MPPI offroad controller, terrain/DEM route planners, lidar
  scene extraction) — gated on the support package + Lidar/terrain toolboxes;
  the generic `controllerMPPI`/path-following primitives could ship later.
- **Apps**: **Inverse Kinematics Designer** — interactive GUI; the
  programmatic `inverseKinematics`/`analyticalInverseKinematics` (Tier-3)
  are in scope, and the app's *generated solver* is the codegen output.
- **Deep-Learning examples** — DLCHOMP (deep-learning CHOMP), reinforcement-
  learning obstacle avoidance — gated on a future Deep Learning toolbox;
  the classical `manipulatorCHOMP` (Tier-6) is in scope.
- **Lidar / point-cloud planning** (point-cloud RRT, perceived-environment
  planning) — gated on a future Lidar / Computer Vision toolbox.
- **Navigation Toolbox planners** (`plannerRRT`/`plannerRRTStar`/
  `plannerHybridAStar`/SLAM) — a separate product; **`manipulatorRRT` /
  `mobileRobotPRM` / `controllerPurePursuit` (which live in *this*
  toolbox) are in scope**.
- **Hardware deploy / Speedgoat real-time** and **Simscape Multibody**
  dynamics co-models — beyond the `-emit-c` lane.
- **Interactive 3-D visualization** (`interactiveRigidBodyTree` live
  dragging) — ships as a **static config artifact** (PNG), not interactive.

These are documented follow-ons, not blockers: every numeric + object-API
surface a *script* uses (transforms / rigidBodyTree FK / IK / trajectories
/ dynamics / mobile kinematics / occupancy-map PRM / collision) is in
Tiers 1–6.

---

## 10. Effort summary

| Tier | Scope | Effort | Net-new code | Status |
|---|---|---|---|---|
| T1 | coordinate transforms (SE3/SO3) + quaternion | ~1.5 wk (≈0.5 if Fusion T1 ships first) | `se3`/`so3` value types + conversion matrix + core gaps | 🔵 |
| T2 | rigid body tree + FK + Jacobian | ~2.5 wk | `rigidBodyTree` + URDF parser + `getTransform`/`geometricJacobian` + `show` | 🔵 |
| T3 | inverse kinematics | ~1 wk | `inverseKinematics`/`generalizedInverseKinematics` wrappers over Optim + constraints | 🔵 |
| T4 | trajectories + dynamics + motion models | ~2.5 wk | `*poly traj`/`trapveltraj`/`contopptraj` + CRBA/RNEA + motion models | 🔵 |
| T5 | mobile robots + occupancy maps + planning | ~2.5 wk | mobile kinematics + occupancy maps + `mobileRobotPRM` + pure pursuit | 🔵 |
| T6 | manipulator planning + collision + estimation + polish | ~3 wk | `manipulatorRRT`/`CHOMP` + GJK collision + `stateEstimatorPF` | 🔵 |
| **Total** | | **~13 wk** | | |

**Recommended slice order**: **T1 → T2 → T3 first** — this closes the
**manipulator forward/inverse-kinematics core** (~5 wk), the foundation
every robotics demo needs, and it rides the *most* shipped infrastructure
(IK is nearly free over the shipped `lsqnonlin`/`fminunc`; FK/Jacobian are
linalg). **Sequencing synergy**: do this *after or alongside Sensor
Fusion Tier-1* so the `quaternion` + transform foundation is built once
and shared (cutting T1 here to ~0.5 wk) — together they form a coherent
"spatial math + estimation + kinematics" stack that directly serves the
user's quadrotor/flight-control work. T4 (trajectories + dynamics) is the
manipulator-control payoff; T5 (mobile + maps + PRM) is the
self-contained mobile-robot half; T6 (planning + collision) is the
heaviest new code and the gateway to the pick-and-place demos. Unlike
Fusion/DSP, this toolbox is **largely shippable without the System-Object
lowering fix** (only 3 stateful objects need it), so it carries less
cross-toolbox sequencing risk.
