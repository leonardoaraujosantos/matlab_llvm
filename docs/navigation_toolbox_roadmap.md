# Navigation Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Navigation-Toolbox programs.

Source: *Navigation Toolbox™ User's Guide* (R2026a, 5 chapters:
Navigation Featured Examples [~750 pp] · Navigation Topics · Coordinate-
transform & orientation utilities · nmeaParser Examples · Spatial
Representation).

This is the **capstone of the autonomy stack** the project has been
building — it sits directly on top of the just-shipped **Robotics System**
([`robotics_toolbox_roadmap.md`](robotics_toolbox_roadmap.md)) and
**Sensor Fusion and Tracking**
([`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md))
toolboxes. An unusually large fraction of the Navigation surface **is
already in the runtime**: the `quaternion` value type + coordinate
transforms, `binaryOccupancyMap` + `mobileRobotPRM` + `manipulatorRRT` +
`controllerPurePursuit`, the four mobile kinematic models +
`differentialDriveKinematics`/`derivative`, `imuSensor`/`gpsSensor` +
`ahrsfilter`/`imufilter`/`complementaryFilter`/`insfilterMARG` + `allanvar`
+ `ecompass`, and the EKF/UKF cores (System Identification Tier-5). On top
of that the project ships **dense linear algebra** (`chol`/`qr`/`svd`/
`eig`/`mldivide`/`pinv`), **Optim** (`fminunc`/`lsqnonlin`/`fmincon` for
graph optimisation), **`ode45`**, the **seeded PRNG**, the **function-handle
ABI**, the **classdef + persistent-state** machinery, and **Cairo
plotting**. **No external dependency** (no g2o/GTSAM/Ceres, no PCL, no
OMPL) — every planner, scan-matcher, graph optimiser, and GNSS routine is
hand-coded over the shipped kernel.

**The net-new surface** (everything Navigation adds *beyond* the shipped
Robotics + Fusion base) is:
1. the **sampling- and search-based planner framework** — `plannerRRT` /
   `plannerRRTStar` / `plannerBiRRT` / `plannerHybridAStar` /
   `plannerAStarGrid` over pluggable **state spaces** (`stateSpaceSE2` /
   `stateSpaceSE3` / `stateSpaceDubins` / `stateSpaceReedsShepp`) with an
   occupancy-map **state validator**;
2. **probabilistic occupancy maps** (`occupancyMap` log-odds, `occupancyMap3D`,
   map layers) — a generalisation of the shipped binary grid;
3. **lidar scan matching + SLAM** (`lidarScan`, `matchScans`,
   `matchScansGrid`, `lidarSLAM`);
4. **graph optimisation** — `poseGraph`/`poseGraph3D` + `optimizePoseGraph`
   and the general `factorGraph` + factor library;
5. **localisation + reactive control** — `monteCarloLocalization` (+
   `odometryMotionModel` / `likelihoodFieldSensorModel`), the general
   `stateEstimatorPF`, and `controllerVFH`;
6. **GNSS simulation + positioning** (`gnssSensor`, `gnssconstellation`,
   `pseudoranges`, `receiverposition`, `nmeaParser`, `rinexread`) and the
   **Frenet trajectory** layer (`referencePathFrenet` /
   `trajectoryGeneratorFrenet`).

**One shared architectural note**: unlike DSP/Comm/Fusion, Navigation is
**largely built-once-then-queried**, not stateful System Objects — the
planners, validators, occupancy maps, pose/factor graphs, and lidar SLAM
objects are constructed, populated, then queried (`plan`/`optimize`/
`addScan`/`checkOccupancy`). Only a few objects (`monteCarloLocalization`,
`stateEstimatorPF`, `controllerVFH`, `rateControl`) are stateful step-driven,
and the shipped `extendedKalmanFilter` / `trackerGNN` precedent already
proves that pattern. So Navigation, like Robotics, is **largely shippable
without the System-Object lowering fix**.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/navigation/hybrid_astar_plan.m`](../examples/navigation/hybrid_astar_plan.m):
*the canonical autonomous-navigation demo — load an `occupancyMap`, build a
`stateSpaceSE2` + `validatorOccupancyMap`, plan a collision-free
kinematically-feasible path between two poses with `plannerHybridAStar` (or
`plannerRRTStar`), and report path length + clearance*. This exercises the
state-space (T1) → occupancy-map (T1) → planner (T2) arc end-to-end;
achieving it closes the **path-planning half** of the toolbox (the UG's
flagship "Plan Path in Warehouse Scenario" workflow). The **SLAM tracer**
(closing T3 + T4) is
[`examples/navigation/lidar_slam_map.m`](../examples/navigation/lidar_slam_map.m):
*a `lidarSLAM` over a sequence of `lidarScan`s — `matchScans` incremental
odometry + loop-closure detection + `optimizePoseGraph` — producing a
drift-corrected occupancy map (the UG "Build a Map from Lidar Data Using
SLAM").* 

Companion docs:
[`robotics_toolbox_roadmap.md`](robotics_toolbox_roadmap.md) (occupancy
maps / PRM / RRT / pure-pursuit / kinematics reused wholesale),
[`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md)
(quaternion / IMU-GPS sensors / fusion filters / EKF reused wholesale),
[`ident_toolbox_roadmap.md`](ident_toolbox_roadmap.md) (the EKF/UKF cores),
[`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md) (`fminunc`/`lsqnonlin`
for pose-graph / factor-graph optimisation), [`plotting.md`](plotting.md)
(map / path / scan plots route through Cairo),
[`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1** is
  the **state-space + map foundation** (`stateSpaceSE2`/`SE3`/`Dubins`/
  `ReedsShepp` + `validatorOccupancyMap` + probabilistic `occupancyMap` +
  `navPath`) — the primitive every planner queries. **Tier-2** is the
  **planner framework** (`plannerRRT`/`plannerRRTStar`/`plannerBiRRT`/
  `plannerHybridAStar`/`plannerAStarGrid` + `pathmetrics` + path smoothing)
  — closes the planning headline. **Tier-3** is **lidar scan matching +
  SLAM** (`lidarScan`/`matchScans`/`matchScansGrid`/`lidarSLAM`/`buildMap`).
  **Tier-4** is **graph optimisation** (`poseGraph`/`poseGraph3D` +
  `optimizePoseGraph`, `factorGraph` + the factor library) — the heaviest
  new math (manifold Gauss-Newton on SE2/SE3). **Tier-5** is
  **localisation + reactive control** (`monteCarloLocalization` +
  `odometryMotionModel`/`likelihoodFieldSensorModel`, `stateEstimatorPF`,
  `controllerVFH`). **Tier-6** is **GNSS + Frenet + polish** (`gnssSensor`/
  `gnssconstellation`/`pseudoranges`/`receiverposition`/`nmeaParser`/
  `rinexread`, `referencePathFrenet`/`trajectoryGeneratorFrenet`, `insEKF`
  flexible framework).
- **Effort** is in the existing Phase-5.6.x cadence (one focused session ≈
  a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~1.5 wk · T2
  ~2.5 wk (RRT*/HybridA* are the meatiest planning code) · T3 ~2 wk · T4
  ~2.5 wk (SE2/SE3 manifold Gauss-Newton is the heaviest new math) · T5
  ~2 wk · T6 ~2 wk (~12.5 wk full)**. Each tier is independently shippable
  and demoable; **T1 + T2 (~4 wk) close the path-planning half** — the
  highest-value cut given the just-shipped Robotics mobile-robot stack.
  **T1 + T2 + T3 + T4 (~8.5 wk) close the SLAM half.**
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Everything below is 🔵 not started** — but the substrate is unusually
  deep (most of the Robotics + Sensor Fusion surface this toolbox composes
  on is already in the runtime). The genuinely new code is the **planner
  framework + state spaces**, the **scan matcher + lidar SLAM**, the **pose-
  graph / factor-graph optimiser**, the **MCL + VFH** reactive layer, and
  the **GNSS positioning + Frenet** trajectory layer.
- **No external dependencies**: matching project precedent — planners hand-
  coded over the PRNG + the shipped occupancy grid; scan matching via a
  hand-coded ICP/NDT over `svd`/`mldivide`; pose-graph / factor-graph
  optimisation via manifold Gauss-Newton over the shipped dense linalg
  (the `optimizePoseGraph` analogue of the shipped `rtsSmoother` / EKF
  Jacobian machinery); MCL/PF resampling via the seeded PRNG; GNSS least-
  squares positioning over `mldivide`.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Navigation code yet)

| Group | Surface (already shipped) | Location | How Navigation uses it |
|---|---|---|---|
| Occupancy maps + PRM | `binaryOccupancyMap` (+ `setOccupancy`/`getOccupancy`/`checkOccupancy`), `mobileRobotPRM` (+ Dijkstra `findpath`), `manipulatorRRT` | `runtime/toolbox/robotics/runtime_robotics.cpp` | The grid substrate `occupancyMap` (T1) generalises; the PRM/RRT sampling + graph search seed the planner framework (T2). |
| Mobile kinematics + control | `differentialDriveKinematics`/`unicycle`/`bicycle`/`ackermann` + `derivative`, `controllerPurePursuit` | `runtime/toolbox/robotics/runtime_robotics.cpp` | Kinematic state propagation for `plannerControlRRt` (T2) + the path-following layer; `controllerVFH` (T5) sits alongside pure pursuit. |
| Spatial math | `quaternion` + `eul`/`rotm`/`axang`/`tform` conversions, `se3` transforms, `wrapToPi` | `runtime/toolbox/fusion/runtime_fusion.cpp` + `runtime/toolbox/robotics/runtime_robotics.cpp` | SE2/SE3 state-space interpolation + distance (T1), pose-graph manifold ops (T4), coordinate-frame conversions (Ch. 5). |
| Inertial + GPS sensors + fusion | `imuSensor`/`gpsSensor`/`insSensor`, `ahrsfilter`/`imufilter`/`complementaryFilter`/`insfilterMARG`, `allanvar`, `ecompass` | `runtime/toolbox/fusion/runtime_fusion.cpp` | The inertial-navigation Ch. 1 examples reuse these directly; `insEKF` (T6) is the flexible-framework generalisation. |
| EKF / UKF cores | `matlab_ident_ekf_*`/`_ukf_*`, `extendedKalmanFilter` classdef | `runtime/toolbox/ident/runtime_ident.cpp` | EKF-based landmark SLAM + the fusion filters; the recursive-estimation loop. |
| Dense linear algebra | `chol`, `qr`, `svd`, `eig`, `mldivide`, `inv`, `pinv`, `norm` | `runtime/matlab_runtime.cpp` | Scan-match SVD (T3), pose-graph / factor-graph Gauss-Newton normal equations (T4), GNSS least-squares positioning (T6). |
| Optim | `fminunc` (BFGS), `lsqnonlin` (LM), `fmincon` | `runtime/toolbox/optim/runtime_optim.cpp` | Graph-optimisation backend (T4); optimisation-based path smoothing (T2). |
| ODE solvers | `ode45`, `ode23s` | `runtime/matlab_runtime.cpp` | Kinematic trajectory rollout for `plannerControlRRt` + the Frenet integrators (T6). |
| PRNG | `rand`/`randn`/`randi` + `rng(seed)` | `runtime/matlab_runtime.cpp` | RRT/RRT* sampling (T2), MCL/PF particle init + resampling (T5), GNSS noise (T6) — reproducible. |
| GJK collision + RRT | `collisionBox`/`Sphere`/`Cylinder`/`Capsule` + GJK `checkCollision`, `manipulatorRRT` | `runtime/toolbox/robotics/runtime_robotics.cpp` | State-validator collision checks (T1/T2); the RRT skeleton. |
| Classdef + state | `classdef`, handle semantics, `persistent`, `matlab_obj_new`/`_set_*`/`_get_mat`, class-pinned dispatch, REPL persist, DAP render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | Every planner / map / graph / filter classdef carrier. |
| Plotting | Cairo `plot`/`plot3`/`scatter`/`line`/`imagesc` | `runtime/plot/` | Headless map / path / scan / pose-graph plots → PNG/SVG. |
| Strict-typing emit lane | `-emit-c` / `-emit-cpp` strict no-C-cast | `lib/Emit/`, `runtime/*` strict-cast lists | Maps onto the UG **"Code Generation from MATLAB Code"** chapter — the planners/validators/SLAM are designed for static-memory codegen. |

**Net assessment**: the *autonomy base* (occupancy grids, PRM/RRT, mobile
kinematics, pure pursuit, quaternion/SE3, IMU/GPS sensors + fusion filters,
EKF/UKF, dense linalg, Optim, ODE, PRNG, classdef + state) is **already
shipped** across Robotics + Sensor Fusion — this toolbox reuses more
existing infrastructure than any unstarted candidate. The genuinely new
code is (a) the **planner framework + state spaces**, (b) the **probabilistic
occupancy map**, (c) the **lidar scan matcher + SLAM**, (d) the **pose-graph /
factor-graph optimiser**, (e) the **MCL + PF + VFH** reactive layer, and (f)
the **GNSS positioning + Frenet** trajectory layer. Each is a self-contained
hand-coded routine over the shipped base.

---

## 2. Tier-1 — State spaces + occupancy maps 🔵 (FOUNDATION)

Goal: the planning primitives every Tier-2 planner queries — pluggable
state spaces + the occupancy-map state validator + a probabilistic
occupancy map. **No System-Object dependency** — ships first.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `stateSpaceSE2` | [x y θ] space: `sampleUniform`, `enforceStateBounds`, `distance` (weighted), `interpolate` (linear + angular `slerp`). | `wrapToPi`, PRNG |
| 1.2 | `stateSpaceSE3` | [x y z qw qx qy qz] space; quaternion `slerp` interpolation + geodesic distance. | `quaternion`, slerp |
| 1.3 | `stateSpaceDubins` / `stateSpaceReedsShepp` | Curvature-constrained connection (Dubins forward-only / Reeds-Shepp reversible) — `MinTurningRadius`; `distance`/`interpolate` along the optimal primitive. | T1.1, geometry |
| 1.4 | `validatorOccupancyMap` | State + motion validator over an occupancy map: `isStateValid` (cell free), `isMotionValid` (interpolated sweep) with `ValidationDistance`. | occupancy map, 1.1 |
| 1.5 | `occupancyMap` | Probabilistic log-odds grid: `setOccupancy`/`getOccupancy`/`updateOccupancy`/`checkOccupancy`, `inflate`, `raycast`, `insertRay`, `move`/`syncWith`, world/grid/local coordinate conversions. | `binaryOccupancyMap` grid |
| 1.6 | `occupancyMap3D` | Octree-free dense/voxel 3-D probabilistic map; `insertPointCloud`/`setOccupancy`/`checkOccupancy`. *(carve to a flat voxel grid; octree deferred.)* | 1.5 |
| 1.7 | `navPath` | The planner output container: states (N×D) over a state space; `interpolate`, `pathLength`. | 1.1 |
| 1.8 | display + DAP | `show(map)` → Cairo `imagesc` artifact; `show(stateSpace)` / path overlay; map + path render in the REPL + DAP inspector. | `runtime/plot/`, DAP |

**Headline-within-tier**: the UG "Occupancy Grids" reference —
`map = occupancyMap(W,H,res); setOccupancy(map, xy, p); inflate(map, r);
checkOccupancy(map, pose)` + `ss = stateSpaceSE2; sv =
validatorOccupancyMap(ss, Map=map); isStateValid(sv, [x y θ])`.

**Compile/Execute wiring**: new
`runtime/toolbox/navigation/runtime_navigation.cpp` +
`navigation_classdefs.m`; state spaces store their bounds + weights as
matrix properties; the occupancy map carries the log-odds grid + resolution
+ world-limit metadata (the shipped `binaryOccupancyMap` pattern);
`isStateValid`/`isMotionValid` are class-pinned methods; conversions are
matrix-in/matrix-out builtins (`Resolver.cpp` + `LowerTensorOps.cpp`).

---

## 3. Tier-2 — Sampling- and search-based planners 🔵 (HEADLINE)

Goal: the planner framework — the path-planning headline. Sampling-based
RRT family + grid/kinematic search, all over the Tier-1 state spaces +
validator.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `plannerRRT` | Rapidly-exploring random tree: sample → nearest → steer (`MaxConnectionDistance`) → validate → extend; goal bias; `plan(planner, start, goal)` → `navPath`. | `manipulatorRRT` skeleton, T1 |
| 2.2 | `plannerRRTStar` | RRT* with near-neighbour rewiring; `BallRadiusConstant` shrinking-ball radius; asymptotically optimal. | 2.1 |
| 2.3 | `plannerBiRRT` | Bidirectional RRT (two trees, connect heuristic). | 2.1 |
| 2.4 | `plannerHybridAStar` | Kinematic A* over a discretised SE2 lattice with Reeds-Shepp expansions + analytic-expansion shortcut; `MotionPrimitiveLength`, `MinTurningRadius`. | T1.3, A* |
| 2.5 | `plannerAStarGrid` | Grid A* / Dijkstra / GBFS over a `binaryOccupancyMap`; admissible heuristics (`Euclidean`/`Manhattan`/`Chebyshev`). | grid, priority queue |
| 2.6 | `pathmetrics` | Path clearance / smoothness / `isPathValid` metrics over a validator. | T1.4 |
| 2.7 | path smoothing | `optimizePath` (optimisation-based smoothing under clearance + curvature) / `bsplinepolytraj` shortcutting. | `fmincon`, T1 |
| 2.8 | `plannerControlRRT` | Kinematic-model RRT: propagate a `differentialDriveKinematics`/`bicycle` model under sampled controls (reverse-capable). | mobile kinematics + `ode45` |

**Headline-within-tier**: **the roadmap headline** — `hybrid_astar_plan.m`:
`occupancyMap` + `stateSpaceSE2` + `validatorOccupancyMap` +
`plannerHybridAStar` planning a collision-free kinematically-feasible path
between two poses (the UG "Plan Path in Warehouse Scenario" / "Generate Code
for Path Planning Using Hybrid A Star").

**Compile/Execute wiring**: planners are built-once classdefs holding the
state space + validator references (cloned in, per the shipped IK/RRT
pattern); `plan(...)` runs the sampling/search loop in the runtime returning
a `navPath`; multi-return `[path, solnInfo]` via the splitter. Needs no SO
fix.

---

## 4. Tier-3 — Lidar scan matching + SLAM 🔵

Goal: the lidar half — register range scans + build a map incrementally.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `lidarScan` | Range-scan container (`Ranges`/`Angles` ↔ `Cartesian`); `transformScan`, `removeInvalidData`. | matrix ops |
| 3.2 | `matchScans` | Scan-to-scan registration → relative pose (point-to-point ICP via SVD; trimmed/outlier-robust). | `svd`, `mldivide` |
| 3.3 | `matchScansGrid` | Correlative grid scan matcher (branch-and-bound over a rasterised scan) — robust initial guess. | grid |
| 3.4 | `scanContextDistance` | Scan-context loop-closure descriptor + distance. | FFT / reductions |
| 3.5 | `lidarSLAM` | Incremental front-end: `addScan` → `matchScans` odometry edge + loop-closure search → underlying `poseGraph`; `scansAndPoses`, `removeLoopClosures`. | 3.2, 3.3, T4 |
| 3.6 | `buildMap` | Fuse scans-at-poses into an `occupancyMap` (`insertRay` per beam). | T1.5 |
| 3.7 | egocentric maps | `insertRay`/`raycast` egocentric occupancy update from range readings; `rangeSensor` simulated lidar over a map. | T1.5 |

**Headline-within-tier**: **the SLAM tracer** — `lidar_slam_map.m`:
`lidarSLAM` over a sequence of `lidarScan`s → incremental `matchScans`
odometry + loop closure + `optimizePoseGraph` → `buildMap` drift-corrected
occupancy map (the UG "Build a Map from Lidar Data Using SLAM" / "Implement
SLAM with Lidar Scans").

**Compile/Execute wiring**: `lidarScan` is a value container; `matchScans`
is a matrix-in/pose-out builtin (ICP loop over `svd`); `lidarSLAM` is a
stateful-ish builder owning a `poseGraph` (T4) — `addScan` mutates it
in place (the shipped `trackerGNN` accumulate pattern).

---

## 5. Tier-4 — Pose-graph + factor-graph optimisation 🔵

Goal: the back-end optimiser — the heaviest new math (manifold Gauss-Newton
on SE2/SE3). The same engine serves lidar SLAM, visual-odometry drift
correction, and multi-sensor factor-graph fusion.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | `poseGraph` / `poseGraph3D` | SE2 / SE3 pose-graph container: `addRelativePose` (edge + information matrix), `nodeEstimates`, `edges`, `findEdgeID`. | classdef, T1 |
| 4.2 | `optimizePoseGraph` | Sparse manifold Gauss-Newton / Levenberg-Marquardt over SE2/SE3 (relative-pose residuals, right-perturbation Jacobians, robust kernel option). | dense linalg, GN |
| 4.3 | `trimLoopClosures` | Reject inconsistent loop-closure edges (χ²-gated). | 4.2 |
| 4.4 | `factorGraph` | General factor graph: typed nodes (`POSE_SE2`/`POSE_SE3`/`VEL3`/`IMU_BIAS`/`POINT_XYZ`) + factor library `factorTwoPoseSE2`/`SE3`, `factorGPS`, `factorIMU`, `factorPoseSE3AndPointXYZ`, `factorIMUBiasPrior`. | 4.2 |
| 4.5 | `optimize(factorGraph)` | `factorGraphSolverOptions` (max iters, tol, trust region) → batch nonlinear least-squares over the factor residuals. | 4.2 |
| 4.6 | EKF-based landmark SLAM | The classical alternative back-end (EKF over a [pose; landmarks] state) — reuses the shipped EKF core. | `extendedKalmanFilter` |

**Headline-within-tier**: the UG "EKF-Based Landmark SLAM" + "Reduce Drift
in 3-D Visual Odometry Trajectory Using Pose Graphs" — build a `poseGraph`
with odometry + loop-closure edges, `optimizePoseGraph`, show the
before/after trajectory RMSE drop.

**Compile/Execute wiring**: `poseGraph`/`factorGraph` are built-once
classdefs over packed edge tables; `optimizePoseGraph`/`optimize` run the
manifold GN in the runtime (right-perturbation on SE2 [Δx Δy Δθ] / SE3
[Δρ Δφ]); residuals + Jacobians hand-coded over the shipped dense linalg
(the same machinery as the shipped RTS smoother + EKF). The sparse normal
equations are solved dense for the modest graph sizes the examples use
(sparse `chol` is the shipped follow-on).

---

## 6. Tier-5 — Localisation + reactive control 🔵

Goal: the on-line localisation + obstacle-avoidance layer.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | `monteCarloLocalization` | Adaptive MCL particle filter on a known `occupancyMap`: `predict`/`correct`/`getStateEstimate`; KLD-adaptive resampling. | PRNG, T1 |
| 5.2 | `odometryMotionModel` | Probabilistic odometry motion model (rotation/translation noise) for MCL prediction. | PRNG |
| 5.3 | `likelihoodFieldSensorModel` | Beam likelihood-field measurement model over an occupancy map for MCL correction. | T1.5, raycast |
| 5.4 | `stateEstimatorPF` | General particle filter (`predict`/`correct`/`getStateEstimate`, multiple resampling policies) — the framework MCL specialises. | PRNG |
| 5.5 | `controllerVFH` | Vector-field-histogram obstacle avoidance from range readings → steering direction; `DistanceLimits`/`RobotRadius`/`SafetyDistance`. | histogram |
| 5.6 | `rateControl` | Fixed-rate execution loop (`waitfor`) for the simulation loop. | timer |
| 5.7 | `binaryOccupancyMap` ↔ `occupancyMap` interop | `mapClutter`/`mapMaze` random-map generators; map-coordinate utilities. | PRNG, T1 |

**Headline-within-tier**: the UG "Localize TurtleBot Using Monte Carlo
Localization Algorithm" — `monteCarloLocalization` with `odometryMotionModel`
+ `likelihoodFieldSensorModel` converging a particle cloud to the true pose
on a known map.

**Compile/Execute wiring**: MCL / `stateEstimatorPF` are stateful step-
driven classdefs (particle matrix mutated in place — the shipped EKF/tracker
pattern); the motion/sensor models are class-pinned helpers; `controllerVFH`
is a small stateful object. The few stateful objects here share the
documented SO note but, like the shipped trackers, the
alloc-then-populate + in-place-update pattern already works.

---

## 7. Tier-6 — GNSS positioning + Frenet trajectories + polish 🔵

Goal: the satellite-navigation + structured-road-trajectory layer, plus the
flexible inertial-fusion framework and the remaining polish.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | `gnssSensor` / `gnssMeasurementGenerator` | Simulated GNSS pseudorange / Doppler measurements with clock + ionospheric + receiver noise over a constellation. | PRNG, T1 spatial |
| 6.2 | `gnssconstellation` / `lookangles` / `skyplot` | Satellite positions from a baked almanac; azimuth/elevation + visibility (DOP). | geometry |
| 6.3 | `pseudoranges` / `receiverposition` | Least-squares GNSS receiver positioning from pseudoranges (iterated LS over the geometry matrix) + `hdop`/`pdop`. | `mldivide` |
| 6.4 | `nmeaParser` / `rinexread` | Parse live/log NMEA sentences (GGA/RMC/GSV) + RINEX observation/navigation files (hand-rolled text readers). | string ops |
| 6.5 | `referencePathFrenet` | Smooth reference path (clothoid/spline) with Frenet ↔ Cartesian conversion (`global2frenet`/`frenet2global`). | spline/interp |
| 6.6 | `trajectoryGeneratorFrenet` | Lateral/longitudinal quintic-polynomial candidate trajectories along a Frenet reference (highway lane-change / merge). | `quinticpolytraj` |
| 6.7 | `insEKF` framework | The flexible `insEKF` + `insAccelerometer`/`insGyroscope`/`insMagnetometer`/`insGPS` + `insMotionPose`/`insMotionOrientation` building blocks (pluggable state + measurement models via handles). | function-handle ABI, EKF core |
| 6.8 | carve-down polish | `pathmetrics` extras, `controllerTEB` placeholder diagnostic, `globePlot`/`geoplot` → static map artifact. | T1/T2 |

**Headline-within-tier**: the UG "GPS Legacy Navigation Receiver Positioning
Using C/A-Code" (LS positioning from pseudoranges) + "Highway Trajectory
Planning Using Frenet Reference Path" (`referencePathFrenet` +
`trajectoryGeneratorFrenet` lane change).

**Compile/Execute wiring**: GNSS sims are stateful sensor classdefs (PRNG-
seeded, reproducible); `receiverposition` is a matrix-in/position-out
iterated-LS builtin; the NMEA/RINEX readers are hand-rolled text parsers
(the URDF-importrobot precedent); the Frenet layer is matrix-in/matrix-out
over the shipped spline/interp; `insEKF` reuses the function-handle ABI +
the shipped EKF core.

---

## 8. Compile/Execute · Debug/REPL · Examples · Tests (cross-cutting)

The four delivery surfaces the project always closes per toolbox — each
tier ships across **all four** before it counts as done.

### 8.1 Compile / Execute

- **Backends**: LLVM JIT + native + `-emit-c` / `-emit-cpp` are the
  primary lanes — and this toolbox maps **directly onto the UG "Code
  Generation from MATLAB Code" chapter**: the planners / validators / SLAM
  / GNSS are explicitly designed for static-memory codegen, which is the
  project's strict-cast `-emit-c` lane. `-emit-python` / `-emit-typescript`
  parity is a per-tier stretch (the maps + planners port cleanly; the
  graph-optimiser path is rougher). `-emit-systemverilog` is **not** a
  target (host-side autonomy) — emit a clear diagnostic.
- **Runtime**: `runtime/toolbox/navigation/runtime_navigation.cpp` (state
  spaces, occupancy maps, planners, scan matching, graph optimisers, MCL/PF,
  VFH, GNSS, Frenet) + `runtime/toolbox/navigation/navigation_classdefs.m`
  (the planner / map / graph / filter / sensor classdefs). Add to the strict
  no-C-cast list (`static_cast`), mirroring `runtime_robotics.cpp`.
- **Wiring**: the maps / state spaces / planners / graphs are built-once
  classdefs — constructor-intercept in `Lowering.cpp` (the
  `rigidBodyTree`/`mobileRobotPRM` precedent) + class-pinned method dispatch
  (`plan`/`optimize`/`addScan`/`checkOccupancy`/`isStateValid`); the few
  stateful objects (`monteCarloLocalization`/`stateEstimatorPF`/`gnssSensor`)
  use the alloc-then-populate + in-place-update pattern; conversions +
  `matchScans` + `receiverposition` are plain builtins (`Resolver.cpp` +
  `LowerTensorOps.cpp` pde_table, string-option args → `matlab_string*`);
  multi-return splitters for `[path, info]` / `[isValid, lastValid]` /
  `[pose, cov] = getStateEstimate(...)`; prelude-trigger the `navigation`
  classdefs (the six-place wiring map in
  [`robotics_toolbox_roadmap.md`](robotics_toolbox_roadmap.md) §8.1
  applies verbatim — `kToolboxDirs` ×2, prelude `Cls[]` + `Names[]`
  scanner + `extClassLeaf`, Resolver, Lowering, pde_table, `run_tests.sh`
  + `run_sweep.sh`).

### 8.2 Debug / REPL

- An `occupancyMap` / `navPath` / `poseGraph` persists across REPL inputs
  and renders in the DAP variable inspector (the grid, the state list, the
  node/edge counts) — the value-type + classdef render path used by the
  shipped maps/trees.
- The stateful localisers (`monteCarloLocalization`/`stateEstimatorPF`)
  persist across REPL inputs and render their particle-set summary; a paused
  `predict`/`correct` loop shows the converging estimate.
- `disp(obj)` formats the MATLAB-faithful property block; the
  `plan`/`optimize`/`addScan`/`predict`/`correct` lifecycle works under the
  JIT REPL.

### 8.3 Examples (`examples/navigation/`)

| Example | Closes | Exercises |
|---|---|---|
| `occupancy_map_basics.m` | T1 | `occupancyMap` set/inflate/checkOccupancy + `stateSpaceSE2` + `validatorOccupancyMap` |
| `hybrid_astar_plan.m` | **headline (T1+T2)** | `occupancyMap` → `stateSpaceSE2` → `plannerHybridAStar` → `navPath` + clearance |
| `rrtstar_plan.m` | T2 | `plannerRRTStar` optimal path on a cluttered grid; `pathmetrics` |
| `lidar_slam_map.m` | **tracer (T3+T4)** | `lidarSLAM` (`matchScans` + loop closure) → `optimizePoseGraph` → `buildMap` |
| `pose_graph_opt.m` | T4 | `poseGraph` with odometry + loop closures → `optimizePoseGraph` RMSE drop |
| `mcl_localize.m` | T5 | `monteCarloLocalization` + `odometryMotionModel` + `likelihoodFieldSensorModel` convergence |
| `gnss_positioning.m` | T6 | `gnssconstellation` → `pseudoranges` → `receiverposition` LS fix + DOP |
| `frenet_lane_change.m` | T6 | `referencePathFrenet` + `trajectoryGeneratorFrenet` highway lane change |

### 8.4 Tests (`test/Run/`)

Gating tests follow the `navigation_*.m` convention with a `.stdout` golden
+ per-backend `.skip-emit-*` files where a lane is out of scope (SV always
skipped; Python/TS skipped where the classdef path is rough, matching the
Robotics / Image precedent).

| Test | Tier | Asserts |
|---|---|---|
| `navigation_occmap.m` | T1 | `occupancyMap` log-odds set/get/inflate; `validatorOccupancyMap` `isStateValid` on a known grid |
| `navigation_statespace.m` | T1 | `stateSpaceSE2`/`Dubins` `distance`/`interpolate` against known values |
| `navigation_planner.m` | **T2** | `plannerRRTStar`/`plannerHybridAStar` returns a collision-free path between two poses (headline) |
| `navigation_astar.m` | T2 | `plannerAStarGrid` finds the optimal grid path around a wall |
| `navigation_scanmatch.m` | T3 | `matchScans` recovers a known relative pose between two synthetic scans |
| `navigation_posegraph.m` | **T4** | `optimizePoseGraph` reduces the trajectory error on a graph with a loop closure |
| `navigation_mcl.m` | T5 | `monteCarloLocalization` particle estimate converges < tol on a known map |
| `navigation_gnss.m` | T6 | `receiverposition` LS fix < tol vs the true receiver position |

Target: **~8 gating tests** (one+ per major surface), in line with Robotics
(7) and Sensor Fusion (5). Full regression must stay green — the badge
bumps to **24 toolboxes**.

---

## 9. Carve-outs (explicitly out of scope)

Matching the per-toolbox precedent — the Simulink / ROS / Gazebo / Deep-
Learning / other-toolbox-dependency surfaces are deferred. **This toolbox's
~750-page example chapter leans heavily on companion products**, so the
carve-out list is large:

- **All Simulink examples + blocks** (TEB local planner block, Pure Pursuit
  block, INS block, the path-following Simulink models) — the MATLAB
  object/function API is the whole target; the mflowLink lane is the
  project's block-diagram answer.
- **ROS / ROS 2 integration** ("Build and Deploy Visual SLAM with ROS",
  the TurtleBot ROS examples) and **Gazebo cosimulation** ("Simulate RGB-D
  Visual SLAM with Cosimulation in Gazebo") — gated on a future ROS Toolbox;
  the pure-MATLAB algorithm variants are in scope.
- **Deep-Learning examples** — **MPNet** (motion-planning networks: "Path
  Planning Using MPNet", "Train Deep Learning-Based Sampler", `mpnetSE2`) —
  gated on a future Deep Learning toolbox; the classical RRT/RRT*/Hybrid-A*
  planners (Tiers 2) are in scope.
- **3-D Visual SLAM / Visual-Inertial Odometry** ("Monocular VIO Using
  Factor Graph", "Performant and Deployable Monocular Visual SLAM",
  `monovslam`/`stereovslam`, feature tracking) — gated on a future Computer
  Vision / Lidar toolbox; the **lidar-scan SLAM + pose-graph / factor-graph
  back-end (Tiers 3/4)** are fully in scope (factor-graph fusion of
  IMU + GPS is in scope; the *visual* front-ends are not).
- **3-D lidar point-cloud SLAM** ("Perform SLAM Using 3-D Lidar Point
  Clouds", NDT over point clouds) — gated on a future Lidar Toolbox; **2-D
  lidar-scan SLAM is in scope**.
- **Speedgoat / real-time hardware** ("Simulate Path Following on Speedgoat
  Real-Time Target Machine") and **hardware sensor streaming** (BNO055 /
  ADIS16505 / MPU-9250 board-in-the-loop) — beyond the `-emit-c` lane.
- **Automated-driving scenario integration** (`drivingScenario`, Driving
  Scenario Designer occupancy export, "Object Tracking and Motion Planning
  Using Frenet") — gated on a future Automated Driving toolbox; the bare
  Frenet trajectory layer (Tier-6) is in scope.
- **Apps** — the SLAM Map Builder / interactive viewers — interactive GUIs;
  the programmatic `lidarSLAM` / `occupancyMap` APIs are in scope.
- **Dynamic occupancy grid RFS tracker** (`trackerGridRFS`, "Motion Planning
  in Urban Environments Using Dynamic Occupancy Grid Map") — shares the
  Sensor Fusion `trackerPHD`/RFS carve-out.
- **AUV / Doppler-velocity-log fusion** and **`factorGraph` GNSS-tightly-
  coupled** advanced variants beyond the basic factor library.

These are documented follow-ons, not blockers: every numeric + object-API
surface a *script* uses (state spaces / occupancy maps / RRT-family + Hybrid
A* planners / lidar scan matching + 2-D SLAM / pose-graph + factor-graph
optimisation / MCL + VFH / GNSS positioning + Frenet) is in Tiers 1–6.

---

## 10. Effort summary

| Tier | Scope | Effort | Net-new code | Status |
|---|---|---|---|---|
| T1 | state spaces + occupancy maps | ~1.5 wk | `stateSpaceSE2`/`SE3`/`Dubins`/`ReedsShepp` + `validatorOccupancyMap` + probabilistic `occupancyMap` + `navPath` | 🔵 |
| T2 | sampling/search planners | ~2.5 wk | `plannerRRT`/`RRTStar`/`BiRRT`/`HybridAStar`/`AStarGrid` + `pathmetrics` + smoothing | 🔵 |
| T3 | lidar scan matching + SLAM | ~2 wk | `lidarScan`/`matchScans`/`matchScansGrid`/`lidarSLAM`/`buildMap` | 🔵 |
| T4 | pose-graph + factor-graph optimisation | ~2.5 wk | `poseGraph`/`poseGraph3D` + `optimizePoseGraph` + `factorGraph` + factor library (SE2/SE3 manifold GN) | 🔵 |
| T5 | localisation + reactive control | ~2 wk | `monteCarloLocalization` + motion/sensor models + `stateEstimatorPF` + `controllerVFH` | 🔵 |
| T6 | GNSS + Frenet + polish | ~2 wk | `gnssSensor`/`gnssconstellation`/`pseudoranges`/`receiverposition` + `nmeaParser`/`rinexread` + Frenet + `insEKF` | 🔵 |
| **Total** | | **~12.5 wk** | | |

**Recommended slice order**: **T1 → T2 first** — this closes the
**path-planning half** (~4 wk), the highest-value cut given the just-shipped
Robotics mobile-robot stack (occupancy maps + PRM + kinematics + pure
pursuit are already in the runtime, so T1/T2 ride the most shipped
infrastructure). **T3 + T4 (~4.5 wk more) close the SLAM half** — the lidar
front-end + the pose-graph / factor-graph back-end, where the **SE2/SE3
manifold Gauss-Newton (T4) is the single heaviest new algorithm** and the
gateway to the mapping demos. **T5** (MCL + VFH) is the on-line localisation
layer; **T6** (GNSS + Frenet) is the satellite-navigation + structured-road
layer. **Sequencing note**: Navigation is **largely shippable without the
System-Object lowering fix** (its objects are built-once-then-queried, like
Robotics) — only the few stateful localisers touch it, and the shipped
EKF / `trackerGNN` precedent already proves that pattern. Navigation
**directly reuses the Robotics + Sensor Fusion surface shipped in the last
two PRs**, so it carries the least net-new-substrate risk of any remaining
candidate.
