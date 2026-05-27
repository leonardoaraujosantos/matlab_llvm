# Sensor Fusion and Tracking Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Sensor-Fusion-and-Tracking-Toolbox programs.

Source: *Sensor Fusion and Tracking Toolbox User's Guide* (R2026a, 6
chapters: Tracking Scenarios · Radar Detections · Inertial Sensor and
Sensor Fusion · Multi-Object Tracking · Code Generation · Featured
Examples — the last chapter alone is ~1,540 pages of worked examples).

This is a **high-leverage extension of the shipped numeric + estimation
base** and the toolbox most aligned with the user's demonstrated work
(the `examples/quadrotor/` symbolic-EOM cascade flight controller, the
MPC/PID attitude loops). The single biggest enabler is already in the
runtime: the **EKF/UKF cores shipped with System Identification Tier-5**
(`matlab_ident_ekf_init`/`_predict`/`_correct`, `matlab_ident_ukf_*`,
the `extendedKalmanFilter` / `unscentedKalmanFilter` classdefs — *the
project's first dynamic Kalman filtering loop*). The tracking filters
(`trackingEKF` / `trackingUKF`) are re-skins of those cores with
tracking-specific motion models; the inertial-fusion filters
(`ahrsfilter` / `imufilter` / `insfilter*`) are EKFs over an orientation
state. On top of that the project already ships **ODE solvers**
(`ode45`/`ode23s`), **dense linear algebra** (`chol`/`qr`/`svd`/`eig`/
`mldivide`), the **seeded PRNG** (`randn`/`rng`), the **function-handle
ABI** (`LowerAnonCalls`), and the **classdef + persistent-state**
machinery. **No external dependency** (no Eigen, no GTSAM, no Ceres) —
every filter, tracker, and quaternion routine is hand-coded over the
shipped kernel.

**One shared architectural prerequisite**: like the DSP / Comm toolboxes,
this surface is **classdef / System-Object-centric** — every tracker
(`trackerGNN` / `trackerJPDA`), every filter (`trackingEKF` /
`ahrsfilter`), every trajectory and sensor model is a stateful classdef
with `predict`/`correct`/`step` semantics. The same **System-Object
lowering fix** that gates DSP Tier-1 / Comm Tier-3+ (the documented
blocker: a tensor-typed RHS routed through `_set_f64` after
monomorphization fails the verifier — CST roadmap §12, Comm roadmap §15,
[`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md) Tier-1) is a
prerequisite for Tiers 2/3/5/6 here. If DSP Tier-1 lands first, this
toolbox inherits the fix for free; otherwise this roadmap carries it.
The shipped `extendedKalmanFilter` classdef proves the pattern already
works for a single stateful filter — the new work is the tracking/fusion
*specialisations* + the multi-object book-keeping.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/fusion/imu_gps_fusion.m`](../examples/fusion/imu_gps_fusion.m):
*the canonical inertial-navigation demo — generate a `waypointTrajectory`
ground truth, simulate noisy `imuSensor` (accel + gyro + mag) and
`gpsSensor` measurements along it, fuse them with `insfilterMARG` (an EKF
over a quaternion-orientation + position + velocity + bias state), and
report the position/orientation RMSE vs ground truth*. This exercises the
quaternion math (T1) → sensor models + fusion filter (T3) arc
end-to-end; achieving it closes the **inertial-navigation half** of the
toolbox (the user's sweet spot). The **multi-object-tracking tracer**
(closing T2 + T5) is
[`examples/fusion/gnn_air_traffic.m`](../examples/fusion/gnn_air_traffic.m):
*a `trackerGNN` over `trackingEKF` constant-velocity filters tracking
several aircraft from noisy detections, scored with `trackGOSPAMetric`*.

Companion docs: [`ident_toolbox_roadmap.md`](ident_toolbox_roadmap.md)
(the EKF/UKF cores + recursive-estimation loop are reused wholesale),
[`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md) (the shared
System-Object lowering fix), [`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md)
+ [`control_toolbox_roadmap.md`](control_toolbox_roadmap.md) (the
quadrotor/flight-control consumers of the fused state),
[`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md) (the
function-handle ABI + `fminunc` for filter `tune`),
[`plotting.md`](plotting.md) (`theaterPlot` / trajectory plots route
through Cairo), [`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the **quaternion + orientation/rotation math** (`quaternion` class +
  all the `quat*`/`eul*`/`rotm*` conversions + `ecompass` + the
  `cross`/`dot`/`deg2rad` helpers) — the foundation primitive both halves
  rest on. **Tier-2** is the estimation-filter foundation
  (`trackingKF`/`trackingEKF`/`trackingUKF`/`trackingCKF`/`trackingIMM` +
  the motion models `constvel`/`constacc`/`constturn`/`singer` + filter
  initialisers + `objectDetection`), almost all re-skinning the shipped
  EKF/UKF cores. **Tier-3** is the inertial half — IMU/GPS **sensor
  models** (`imuSensor`/`gpsSensor`/`insSensor` + `accelparams`/…) +
  **fusion filters** (`ahrsfilter`/`imufilter`/`complementaryFilter`/
  `ecompass`/`insfilterMARG`/`insfilterAsync` + `allanvar` + `tune`).
  **Tier-4** is trajectory + scenario generation (`waypointTrajectory`/
  `kinematicTrajectory`/`geoTrajectory` + `trackingScenario`/`platform`/
  `pose` + headless `theaterPlot`). **Tier-5** is multi-object tracking
  (assignment `assignmunkres`/`assignauction`/`assignjv`/`assignsd` +
  trackers `trackerGNN`/`trackerJPDA`/`trackerTOMHT` + track logic +
  `fusionRadarSensor`). **Tier-6** is track-to-track fusion + metrics +
  RFS trackers + carve-down polish (`trackFuser`/`staticDetectionFuser`,
  `trackErrorMetrics`/`trackGOSPAMetric`, `trackerPHD`, OOSM
  retrodiction).
- **Effort** is in the existing Phase 5.6.x cadence (one focused session
  ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~1.5 wk · T2
  ~1.5 wk (EKF/UKF cores shipped) · T3 ~2.5 wk · T4 ~1.5 wk · T5 ~3 wk
  (Munkres + GNN/JPDA are the meatiest new code) · T6 ~2.5 wk (~12.5 wk
  full)**. Each tier is independently shippable and demoable; **T1 + T2 +
  T3 (~5.5 wk) close the inertial-navigation half** — the highest-value
  cut given the flight-control alignment.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Tiers 1–3 ✅ shipped** 2026-05-27 (inertial-navigation half); **Tier-4
  partial 🟡** (`waypointTrajectory` + `lla2ned`/`ned2lla` shipped;
  `kinematicTrajectory` / `geoTrajectory` / `trackingScenario` / `theaterPlot`
  carved down); **Tier-5 partial 🟡** (`assignmunkres` + `objectTrack` +
  `trackerGNN` shipped — closes the `gnn_air_traffic.m` tracer; `assignjv` /
  `assignauction` / `assignsd` / `trackerJPDA` / `trackerTOMHT` /
  `fusionRadarSensor` carved down); **Tier-6 partial 🟡** (`trackFuser`
  covariance intersection + `trackGOSPAMetric` / `trackOSPAMetric` /
  `trackErrorMetrics` + `rtsSmoother` shipped — closes `track_fusion_metrics.m`;
  `trackerPHD` / OOSM retrodiction / `trackerGridRFS` carved down).
  The estimation substrate is
  unusually deep: the EKF/UKF cores, ODE solvers, dense linalg, PRNG, and
  the single-filter classdef pattern are all shipped. The genuinely new
  surface is the **quaternion type**, the **sensor noise models**, the
  **fusion-filter specialisations**, the **assignment algorithms**, and
  the **multi-object book-keeping**.
- **Two stateful-object families**: (a) the **filters/trackers/sensors**
  are stateful classdefs (`predict`/`correct`/`step` mutating internal
  state) — the System-Object pattern shared with DSP/Comm (see the
  prerequisite note above); (b) the **`quaternion`** is a *value* type
  (an N×1 array of unit quaternions with overloaded `*`/`conj`/`norm`/…)
  — closer to the shipped `fi` / `datetime` value-type precedent. Tier-1
  ships the value type first (no SO dependency); Tiers 2/3/5/6 layer the
  stateful objects on top.
- **No external dependencies**: matching the project precedent —
  quaternion algebra hand-coded; EKF/UKF reuse `matlab_ident_*`; the
  inertial filters are hand-coded EKFs over the shipped `chol`/`mldivide`;
  Munkres/auction/JV assignment hand-coded; sensor noise via the shipped
  PRNG; `tune` over the shipped `fminunc`.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Fusion code yet)

| Group | Surface (already shipped) | Location | How Fusion uses it |
|---|---|---|---|
| EKF / UKF cores | `matlab_ident_ekf_init`/`_predict`/`_correct`, `matlab_ident_ukf_predict`/`_correct`, `extendedKalmanFilter` / `unscentedKalmanFilter` classdefs | `runtime/toolbox/ident/runtime_ident.cpp` (System ID T5) | The compute core of `trackingEKF` / `trackingUKF` (T2) and every inertial-fusion EKF (`ahrsfilter` / `insfilter*`, T3). |
| ODE solvers | `ode45` (Dormand-Prince), `ode23s` (stiff), event detection | `runtime/matlab_runtime.cpp` | Continuous motion-model propagation; trajectory integration (`kinematicTrajectory`, T4). |
| Dense linear algebra | `chol`, `qr`, `svd`, `eig`, `mldivide`, `inv`, `pinv`, `norm`, `trace`, `kron` | `runtime/matlab_runtime.cpp` | Covariance factorisation (Joseph/`chol` updates), UKF sigma points, gain solves, GOSPA distance — everywhere. |
| PRNG | `rand`/`randn`/`randi`/`randperm` + `rng(seed)` (reproducible) | `runtime/matlab_runtime.cpp` | `imuSensor`/`gpsSensor` noise (T3), particle-filter resampling (`trackingPF`), clutter generation (T5). |
| Function-handle ABI | `void *fn_p` → `matlab_mat*(*)(…)`, `LowerAnonCalls` retyping | `runtime/toolbox/optim/runtime_optim.cpp` | Custom `StateTransitionFcn` / `MeasurementFcn` for `trackingEKF`; the `insEKF` flexible-model handles (T2/T3). |
| Optim | `fminunc` (BFGS), `lsqnonlin` (LM) | `runtime/toolbox/optim/runtime_optim.cpp` | Filter auto-tuning (`tune` / `tunerconfig`, T3); `allanvar` curve fit. |
| Classdef + state | `classdef`, handle semantics, `properties`/`methods`, `persistent`, `matlab_obj_new`/`_set_*`/`_get_mat`, class-pinned dispatch, REPL persist, DAP render, **value-class copy-on-assign** | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The `quaternion` value type (T1) + every stateful filter/tracker/sensor classdef (T2–T6). |
| Value-type precedent | `fi`, `datetime`, `duration`, `categorical` value types with operator overloads | `lib/MLIR/Passes/LowerFixedPoint.cpp`, `runtime/matlab_runtime.cpp` | The `quaternion` array type follows the same overload + display pattern (T1). |
| Reductions / stats | `mean`, `std`, `var`, `median`, `cov`, `sort`, `min`/`max` | `runtime/matlab_runtime.cpp` (core + Stats) | Covariance init, GOSPA/OSPA reductions, `meanrot`, metric aggregation. |
| Trig / elementary | `sin`/`cos`/`atan2`/`sqrt`/`hypot`/`mod`/`unwrap` | `runtime/matlab_runtime.cpp` | Quaternion ↔ Euler conversions, range/azimuth measurement models, angle wrapping (T1/T2). |
| Plotting | Cairo `plot` / `plot3` / `scatter` / `quiver` / `animatedline` | `runtime/plot/` | Headless `theaterPlot` / trajectory / track-vs-truth plots → PNG/SVG (T4). |
| Strict-typing emit lane | `-emit-c` / `-emit-cpp` strict no-C-cast, single-precision tracking | `lib/Emit/`, `runtime/*` strict-cast lists | Maps directly to the UG **"Code Generation with Strict Single-Precision and Non-Dynamic Memory"** chapter (the trackers/filters are designed for it). |

**Net assessment**: the *estimation base* (EKF/UKF, ODE, linalg, PRNG,
classdef + state, the single-filter classdef pattern) is **already
shipped** — this toolbox reuses more existing infrastructure than any of
the other unstarted candidates. The genuinely new code is (a) the
**`quaternion` value type** + rotation conversions, (b) the **tracking
motion models + filter re-skins** over the shipped EKF/UKF, (c) the **IMU/
GPS sensor noise models + inertial-fusion EKFs**, (d) the **trajectory/
scenario generators**, (e) the **assignment algorithms + multi-object
trackers**, and (f) the **track fusion + metrics**. Each is a
self-contained hand-coded routine over the shipped base.

---

## 2. Tier-1 — Quaternion + orientation/rotation math 🔵 (FOUNDATION)

Goal: the `quaternion` value type + the full rotation-conversion surface.
The primitive both halves of the toolbox (orientation fusion *and* 3-D
tracking) depend on. **No System-Object dependency** — ships first.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `quaternion` construction | `quaternion(w,x,y,z)`, `quaternion(M)` (N×4), from `'euler'`/`'eulerd'` (ZYX/…), `'rotmat'`, `'rotvec'`/`'rotvecd'`; N×1 array of unit quaternions. | value-type pattern (`fi`/`datetime`) |
| 1.2 | algebra overloads | `*` (Hamilton product), `.*`, `conj`, `norm`, `normalize`, `inverse`/`./`, `prod`, `exp`/`log`, `==`, indexing/`compact`/`parts`. | element-wise + custom |
| 1.3 | rotations | `rotatepoint(q,v)` / `rotateframe(q,v)` (point vs frame convention), `rotmat(q,'point'/'frame')`. | matmul |
| 1.4 | interpolation / distance | `slerp(q0,q1,t)` (spherical-linear), `dist(q1,q2)` (geodesic angle), `meanrot` (chordal mean), `angvel`. | `acos`/`sin` |
| 1.5 | Euler / matrix conversions | `euler`/`eulerd`/`rotvec`/`rotvecd` (methods); free functions `quat2eul`/`eul2quat`/`quat2rotm`/`rotm2quat`/`angle2quat`/`quat2angle`/`rotvec2quat`/`eul2rotm`/`rotm2eul`. | trig |
| 1.6 | `ecompass` | Accelerometer + magnetometer → orientation quaternion (TRIAD/Davenport). | 1.1, `cross`/`dot` |
| 1.7 | missing helpers | `cross`, `dot`, `deg2rad`/`rad2deg`, `normalize`, `mvnrnd` (multivariate-normal sampler) — small core gaps this toolbox needs. | `chol`, PRNG |
| 1.8 | display + DAP | `disp(q)` formats `w + xi + yj + zk` rows; quaternion arrays render in the REPL + DAP variable inspector. | `disp(obj)`, DAP |

**Headline-within-tier**: the rotations/quaternion round-trip —
`q = quaternion([30 45 60],'eulerd','ZYX','frame'); v2 = rotatepoint(q,v);`
+ `slerp` interpolation + `eulerd(q)` recovers the angles. The UG
"Rotations, Orientation, and Quaternions" example.

**Compile/Execute wiring**: new `runtime/toolbox/fusion/runtime_fusion.cpp`
+ `fusion_classdefs.m` (`quaternion`); the value type stores an N×4 plane
matrix; operator overloads via the CST/`fi` operator-method route;
conversions are matrix-in/matrix-out builtins (`Resolver.cpp` +
`LowerTensorOps.cpp`); ship the `cross`/`dot`/`deg2rad`/`mvnrnd` core
gaps alongside.

---

## 3. Tier-2 — Estimation filters + motion/measurement models 🔵

Goal: the tracking-filter foundation — almost entirely re-skinning the
**shipped** EKF/UKF cores with tracking-specific motion models.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `trackingKF` | Linear Kalman filter; built-in motion models (`'constvel'`/`'constacc'`/`'constturn'`); `predict`/`correct`/`distance`/`likelihood`. | shipped KF |
| 2.2 | `trackingEKF` | Extended KF over user/predefined `StateTransitionFcn` + `MeasurementFcn` (+ Jacobians); analytic or numeric Jacobian. | `matlab_ident_ekf_*` |
| 2.3 | `trackingUKF` / `trackingCKF` | Unscented + cubature KF (sigma-point / spherical-radial). | `matlab_ident_ukf_*`, `chol` |
| 2.4 | motion models | `constvel`/`constacc`/`constturn`/`singer` + Jacobians (`constveljac`/…) + process-noise builders. | — |
| 2.5 | measurement models | `cvmeas`/`cameas`/`ctmeas` (+ Jacobians); position/velocity, range/azimuth/elevation, range-rate. | trig |
| 2.6 | filter initialisers | `initcvekf`/`initcaekf`/`initctekf`/`initcvkf`/`initekfimm`/… (detection → seeded filter). | 2.1–2.4 |
| 2.7 | `objectDetection` | The measurement container: `Time`/`Measurement`/`MeasurementNoise`/`SensorIndex`/`ObjectClassID`. | classdef |
| 2.8 | `trackingIMM` / `trackingGSF` / `trackingPF` | Interacting-multiple-model (mode switching), Gaussian-sum, particle filter (resampling via PRNG). | 2.1–2.3, PRNG |

**Headline-within-tier**: the UG "Estimate 2-D Target States with Angle
and Range Measurements Using trackingEKF" — `constturn` motion +
range/azimuth `MeasurementFcn`, `predict`/`correct` loop, NEES check.

**Compile/Execute wiring**: `trackingEKF`/`UKF` are thin classdef wrappers
binding a motion-model + measurement-model handle to the shipped EKF/UKF
cores (the function-handle ABI carries the models); motion/measurement
models are plain builtins. Needs the shared SO lowering fix for the
stateful filter classdefs (see the prerequisite note).

---

## 4. Tier-3 — Inertial sensors + orientation/pose fusion 🔵

Goal: the inertial-navigation half — IMU/GPS sensor models + the fusion
filters. **The user's flight-control sweet spot**; closes the headline.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | IMU sensor model | `imuSensor` (accel + gyro + optional mag) with `accelparams`/`gyroparams`/`magparams` (bias, noise density, random walk, scale, misalignment). | quaternion, PRNG |
| 3.2 | other sensors | `gpsSensor` (lla + velocity + noise), `altimeterSensor`, `insSensor` (ground-truth pose + noise). | PRNG |
| 3.3 | `ecompass` / `imufilter` | Accel+mag tilt-compensated heading; `imufilter` (accel+gyro complementary/EKF orientation). | T1, EKF core |
| 3.4 | `ahrsfilter` / `complementaryFilter` | Accel+gyro+mag orientation EKF (the workhorse AHRS); first-order complementary filter. | EKF core, quaternion |
| 3.5 | `insfilter*` family | `insfilterMARG` / `insfilterAsync` / `insfilterErrorState` / `insfilterNonholonomic` — EKFs over quaternion-orientation + position + velocity + bias state fusing IMU + GPS. | EKF core, `chol` |
| 3.6 | `insEKF` framework | The flexible `insEKF` + `insMotion*` / `insSensor*` building blocks (pluggable state + measurement models via handles). | function-handle ABI |
| 3.7 | noise analysis + tuning | `allanvar` (Allan variance → bias-instability / random-walk), `tune`/`tunerconfig` (filter-noise auto-tuning). | `fminunc` |

**Headline-within-tier**: **the roadmap headline** —
`imu_gps_fusion.m`: `waypointTrajectory` → `imuSensor`+`gpsSensor` →
`insfilterMARG` fusion → position/orientation RMSE vs truth. Plus the UG
"Estimate Orientation Through Inertial Sensor Fusion" (`ahrsfilter`).

**Compile/Execute wiring**: sensor models are stateful classdefs emitting
noisy measurements per `step` (PRNG-seeded, reproducible); fusion filters
are EKFs over the shipped core with a quaternion-bearing state vector;
`tune` drives the shipped `fminunc`. Reuses the Tier-1 quaternion type
throughout.

---

## 5. Tier-4 — Trajectory + scenario generation 🔵

Goal: the ground-truth generators — trajectories and tracking scenarios
that feed the sensors and trackers.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | `waypointTrajectory` | Waypoint + time-of-arrival → smooth position/velocity/accel/orientation (piecewise-clothoid / minimum-jerk interpolation). | spline/interp, quaternion |
| 4.2 | `kinematicTrajectory` | Integrate body accel + angular velocity → pose (the strapdown mechanisation). | `ode45`, quaternion |
| 4.3 | `geoTrajectory` | Geodetic (LLA) waypoint trajectory + ENU/NED frame conversion. | 4.1, geodesy helpers |
| 4.4 | `trackingScenario` + `platform` | Scenario container; `platform(sc)` with `Trajectory`/`Sensors`/`Signatures`; `advance(sc)` / `record(sc)` / `pose` / `platformPoses` / `targetPoses`. | classdef + state |
| 4.5 | `theaterPlot` (headless) | Plotters (`trackPlotter`/`detectionPlotter`/`trajectoryPlotter`/`orientationPlotter`/`platformPlotter`) → Cairo PNG/SVG (no live GUI). | `runtime/plot/` |
| 4.6 | coordinate frames | NED ↔ ENU ↔ body, `lla2ned`/`ned2lla`-style helpers, `enu2lla`. | T1, geodesy |

**Headline-within-tier**: the UG "Create Tracking Scenario with Two
Platforms" — `trackingScenario` + two `waypointTrajectory` platforms +
`advance` loop plotting both positions (the example shown in Chapter 1).

**Compile/Execute wiring**: `trackingScenario`/`platform` are stateful
classdefs (the `advance` loop mutates platform poses); trajectories are
classdefs whose call/`lookupPose` returns pose-at-time; `theaterPlot`
writes headless artifacts.

---

## 6. Tier-5 — Multi-object trackers + assignment 🔵

Goal: the multi-target tracking core — assignment algorithms + the
data-association trackers built on the Tier-2 filters.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | assignment | `assignmunkres` (Munkres/Hungarian), `assignauction` (auction), `assignjv` (Jonker-Volgenant), `assignsd` (S-D Lagrangian-relaxation), `assignkbest`. | `mldivide`, custom |
| 5.2 | gating + utilities | `clusterDetectionsByGate`, gating via Mahalanobis distance, `trackHistoryLogic` / `trackScoreLogic` (confirmation/deletion). | T2 `distance` |
| 5.3 | `trackerGNN` | Global-nearest-neighbour tracker: gate → `assignmunkres` → per-track `predict`/`correct` → confirm/delete. | 5.1, 5.2, T2 |
| 5.4 | `trackerJPDA` | Joint-probabilistic data-association (weighted measurement updates over feasible joint events). | 5.1, T2 |
| 5.5 | `trackerTOMHT` | Track-oriented multi-hypothesis tracking (hypothesis tree + `assignTOMHT` + pruning). | 5.1, T2 |
| 5.6 | `objectTrack` | The track-state container (`TrackID`/`State`/`StateCovariance`/`Age`/`IsConfirmed`/…). | classdef |
| 5.7 | `fusionRadarSensor` | Statistical radar detection generation (range/azimuth/elevation/range-rate + detectability) feeding the trackers. | quaternion, PRNG |

**Headline-within-tier**: **the tracking tracer** — `gnn_air_traffic.m`:
`trackerGNN` over `trackingEKF` constant-velocity filters tracking
several aircraft from noisy `objectDetection`s. The UG "Air Traffic
Control" / "Using the Global Nearest Neighbor Tracker" examples.

**Compile/Execute wiring**: assignment algorithms are matrix-in/index-out
builtins; the trackers are stateful classdefs orchestrating a vector of
Tier-2 filters + the assignment + the track-logic — the heaviest new
book-keeping in the roadmap. Needs the shared SO lowering fix.

---

## 7. Tier-6 — Track fusion + metrics + RFS trackers + carve-down polish 🔵

Goal: the central-vs-distributed fusion layer, the evaluation metrics, and
the random-finite-set trackers.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | `trackFuser` | Track-to-track fusion (covariance-intersection / cross-covariance) of tracks from multiple sources. | T2, `inv` |
| 6.2 | `staticDetectionFuser` | Static fusion of detections from multiple geometry-diverse sensors (triangulation). | `mldivide` |
| 6.3 | tracking metrics | `trackErrorMetrics` (RMSE/NEES), `trackAssignmentMetrics`, `trackGOSPAMetric` / `trackOSPAMetric` (optimal sub-pattern assignment distance), `trackCLEARMetric`. | `assignmunkres`, reductions |
| 6.4 | `trackerPHD` | Probability-hypothesis-density filter (GM-PHD) for dense-clutter / extended-object tracking. | T2, mixtures |
| 6.5 | OOSM handling | Out-of-sequence-measurement `retrodict`/`retroCorrect` (filter rewind + re-update). | T2 |
| 6.6 | `trackerGridRFS` | Grid-based random-finite-set tracker (dynamic occupancy grid). *(stretch)* | grids |
| 6.7 | carve-down polish | `trackingGlobeViewer` (→ static map artifact), `objectDetectionDelay`, class-fusion helpers, `smoother*` (RTS smoothing). | T2/T4 |

**Headline-within-tier**: the UG "Track-Level Fusion of Radar and Lidar
Data"-style fusion + `trackGOSPAMetric` evaluation (radar-only variant,
lidar carved out — see §9).

**Compile/Execute wiring**: `trackFuser` is a stateful classdef over the
Tier-2 filters; the metrics are matrix-in/scalar-out builtins reusing
`assignmunkres` (GOSPA is an assignment problem); `trackerPHD` is the
heaviest Tier-6 item (Gaussian-mixture book-keeping).

---

## 8. Compile/Execute · Debug/REPL · Examples · Tests (cross-cutting)

The four delivery surfaces the project always closes per toolbox — each
tier ships across **all four** before it counts as done.

### 8.1 Compile / Execute

- **Backends**: LLVM JIT + native + `-emit-c` / `-emit-cpp` are the
  primary lanes — and this toolbox maps **directly onto the UG "Code
  Generation with Strict Single-Precision and Non-Dynamic Memory"
  chapter**: the trackers/filters are explicitly designed for static
  memory + single precision, which is exactly the project's strict-cast
  `-emit-c` lane. `-emit-python` / `-emit-typescript` parity is a
  per-tier stretch (the quaternion value type + matrix filters port
  cleanly; the multi-object trackers are rougher). `-emit-systemverilog`
  is **not** a target (host-side estimation) — emit a clear diagnostic.
- **Runtime**: `runtime/toolbox/fusion/runtime_fusion.cpp` (quaternion
  algebra, motion/measurement models, sensor noise models, assignment,
  metrics) + `runtime/toolbox/fusion/fusion_classdefs.m` (`quaternion`,
  the filters, trackers, sensors, trajectories). Add to the strict
  no-C-cast list (`static_cast`), mirroring `runtime_images.cpp`.
- **Wiring**: the `quaternion` value type (T1) uses the `fi`/CST operator-
  overload route; the stateful filters/trackers/sensors (T2/3/5/6) use the
  **shared System-Object lowering fix** (CST §12 / Comm §15 / DSP Tier-1)
  — coordinate so the fix lands once; motion/measurement models +
  assignment + metrics are plain builtins (`Resolver.cpp` +
  `LowerTensorOps.cpp`, string-option args → `matlab_string*`);
  multi-return splitters for `[track, age, ...]` / `[x, P] = predict(...)`;
  prelude-trigger the `fusion` classdefs.

### 8.2 Debug / REPL

- A `quaternion` array persists across REPL inputs and renders in the DAP
  variable inspector (the `w + xi + yj + zk` rows) — the value-type render
  path used by `fi`/`datetime`.
- The stateful filters/trackers/sensors persist across REPL inputs and
  render their state in the DAP inspector — for a `trackingEKF`, the
  `State` + `StateCovariance`; for a `trackerGNN`, the confirmed-track
  list. So a paused `predict`/`correct` loop shows the evolving estimate.
- `disp(obj)` formats the MATLAB-faithful property block; the
  `predict`/`correct`/`step`/`advance` lifecycle works under the JIT REPL.

### 8.3 Examples (`examples/fusion/`)

| Example | Closes | Exercises |
|---|---|---|
| `quaternion_rotations.m` | T1 | `quaternion` from Euler, `rotatepoint`, `slerp`, `eulerd` round-trip |
| `tracking_ekf_2d.m` | T2 | `trackingEKF` constturn + range/azimuth `predict`/`correct` loop |
| `imu_gps_fusion.m` | **headline (T1+T3)** | `waypointTrajectory` → `imuSensor`+`gpsSensor` → `insfilterMARG` → RMSE |
| `ahrs_orientation.m` | T3 | `imuSensor` → `ahrsfilter` orientation estimate; `allanvar` noise plot |
| `two_platform_scenario.m` | T4 | `trackingScenario` + two `waypointTrajectory` + `advance` + `theaterPlot` |
| `gnn_air_traffic.m` | **tracer (T2+T5)** | `trackerGNN` over `trackingEKF`; `trackGOSPAMetric` score |
| `jpda_clutter.m` | T5 | `trackerJPDA` tracking closely-spaced targets in clutter |
| `track_fusion_metrics.m` | T6 | `trackFuser` two-sensor fusion + `trackErrorMetrics`/`trackGOSPAMetric` |

### 8.4 Tests (`test/Run/`)

Gating tests follow the `fusion_*.m` convention with a `.stdout` golden +
per-backend `.skip-emit-*` files where a lane is out of scope (SV always
skipped; Python/TS skipped where the classdef/tracker path is rough,
matching the Image `image_png_roundtrip` precedent).

| Test | Tier | Asserts |
|---|---|---|
| `fusion_quaternion.m` | T1 | Hamilton product, `rotatepoint` vs `rotmat`, `slerp` endpoints, `eul2quat`↔`quat2eul` round-trip |
| `fusion_quat_convert.m` | T1 | `quat2rotm`/`rotm2quat`/`angle2quat` against known rotations; `ecompass` |
| `fusion_trackingkf.m` | T2 | `trackingKF` constvel `predict`/`correct` recovers a known track |
| `fusion_trackingekf.m` | T2 | `trackingEKF` range/azimuth; NEES within bounds; reuses shipped EKF core |
| `fusion_imusensor.m` | T3 | `imuSensor` noise statistics (seeded PRNG, reproducible); `accelparams` |
| `fusion_ahrsfilter.m` | T3 | `ahrsfilter` orientation error < tol on a synthetic IMU stream |
| `fusion_insfilter.m` | **T3** | `insfilterMARG` IMU+GPS fusion position/orientation RMSE < tol (headline) |
| `fusion_waypoint_traj.m` | T4 | `waypointTrajectory` hits the waypoints at the specified times |
| `fusion_assignment.m` | T5 | `assignmunkres`/`assignjv` optimal assignment on a known cost matrix |
| `fusion_trackergnn.m` | T5 | `trackerGNN` confirms N tracks on a clean multi-target scenario |
| `fusion_gospa.m` | T6 | `trackGOSPAMetric` on known track-vs-truth = expected distance |

Target: **~11 gating tests** (one+ per major surface), in line with
Image (10) and Stats (12). Full regression must stay green (currently 465
run-tests) — the badge bumps to **17 toolboxes** (or higher if Curve
Fitting / Wavelet / DSP land first). **Note**: the stateful-tracker tiers
(2/3/5/6) share the System-Object lowering fix with DSP/Comm, so
sequencing this after (or alongside) DSP Tier-1 amortises that cost.

---

## 9. Carve-outs (explicitly out of scope)

Matching the per-toolbox precedent — the Simulink / app / Deep-Learning /
other-toolbox-dependency surfaces are deferred. **This toolbox's
1,540-page example chapter leans heavily on companion products**, so the
carve-out list is large:

- **All Simulink examples + blocks** (the many "… in Simulink" featured
  examples, the INS/IMU/tracker Simulink blocks) — the MATLAB
  object/function API is the whole target; the mflowLink lane is the
  project's block-diagram answer, separately roadmapped.
- **Apps**: **Tracking Scenario Designer**, **Tracking Data Importer**,
  **Ground Truth Labeler** — interactive GUIs; the programmatic
  `trackingScenario` / `objectDetection` APIs (Tier-4) are in scope.
- **Deep-Learning examples** — DeepSORT, ReID networks, multi-object
  tracking + human-pose estimation, autoencoder anomaly detection — gated
  on a future Deep Learning toolbox.
- **Lidar / point-cloud tracking** (`pointCloud`, voxel/grid lidar
  trackers, "Track Vehicles Using Lidar") — gated on a future Lidar /
  Computer Vision toolbox; the **radar-detection** tracking path is fully
  in scope.
- **Radar-signal-level + Phased-Array dependence** (beamforming, CFAR on
  FPGA, `phased.*` waveforms, bistatic/multistatic radar examples) —
  gated on a future Radar / Phased Array toolbox. `fusionRadarSensor`
  (statistical detection generation, Tier-5) is in scope; physical
  radar-signal modelling is not.
- **Automated Driving Toolbox dependencies** (`drivingScenario`,
  lane-boundary / camera-dataset examples) and **camera/visual tracking**.
- **Hardware streaming** (BNO055/MPU-9250 wireless IMU, board-in-the-loop)
  and **deploy** beyond the `-emit-c` strict-typing lane.
- **`trackingGlobeViewer` / live 3-D globe** — ships as a **static map
  artifact** (PNG), not an interactive globe.

These are documented follow-ons, not blockers: every numeric +
object-API surface a *script* uses (quaternions / filters / IMU-GPS fusion
/ trajectories / GNN-JPDA tracking / fusion / metrics) is in Tiers 1–6.

---

## 10. Effort summary

| Tier | Scope | Effort | Net-new code | Status |
|---|---|---|---|---|
| T1 | quaternion + orientation/rotation math | ~1.5 wk | `quaternion` value type + conversions + `ecompass` + core gaps | ✅ |
| T2 | estimation filters + motion/measurement models | ~1.5 wk | `trackingKF`/`EKF`/`UKF` re-skins over shipped EKF/UKF + `constvel`/`constacc`/`constturn` + `objectDetection` + `initcvekf`/`initctekf` (`IMM`/`GSF`/`PF` deferred) | ✅ |
| T3 | inertial sensors + orientation/pose fusion | ~2.5 wk | `imuSensor`/`gpsSensor` + Mahony-style `ahrsfilter`/`imufilter`/`complementaryFilter` + simplified `insfilterMARG` headline + `allanvar` (`tune` carved as follow-on) | ✅ |
| T4 | trajectory + scenario generation | ~1.5 wk | `waypointTrajectory` (position-only) + `lla2ned`/`ned2lla` WGS-84 (full kinematicTrajectory/geoTrajectory/trackingScenario/theaterPlot are documented follow-ons) | 🟡 |
| T5 | multi-object trackers + assignment | ~3 wk | `assignmunkres` (O(n³) Hungarian) + `objectTrack` + `trackerGNN` over constvel trackingEKFs with Mahalanobis gating + age-based confirmation (assignjv/auction/sd + trackerJPDA/TOMHT + fusionRadarSensor carved as follow-ons) | 🟡 |
| T6 | track fusion + metrics + RFS + polish | ~2.5 wk | `trackFuser` covariance-intersection (trace-min ω line search) + `trackGOSPAMetric`/`trackOSPAMetric` (Munkres-based) + `trackErrorMetrics` RMSE + `rtsSmoother` (free-function RTS backward pass); `trackerPHD`/`staticDetectionFuser`/OOSM/`trackerGridRFS` carved down | 🟡 |
| **Total** | | **~12.5 wk** | | |

**Recommended slice order**: **T1 → T2 → T3 first** — this closes the
**inertial-navigation half** (~5.5 wk), which is the highest-value cut
given the user's flight-control / quadrotor work and rides the *most*
shipped infrastructure (the EKF/UKF cores make T2 nearly free; T1
quaternion is a value type with no SO dependency). T1 also seeds a future
Aerospace/Robotics toolbox (quaternions + frames are shared). T4
(scenarios) feeds T5; **T5 (multi-object trackers) is the meatiest new
code** (the assignment algorithms + tracker book-keeping) and the gateway
to the surveillance/ATC demos; T6 (fusion + metrics) is the evaluation
layer. **Sequencing note**: Tiers 2/3/5/6 depend on the shared
System-Object lowering fix (CST §12 / Comm §15 / DSP Tier-1) — landing
this toolbox after or alongside DSP Tier-1 amortises that infrastructure
cost across DSP, Comm, RF, and Fusion at once.
