# Navigation Toolbox — Tutorial

The Navigation lane compiles MATLAB mobile-robot path planning, localisation, reactive control, and GNSS positioning to native code through the MATLAB → MLIR → LLVM pipeline. All six tiers ship: probabilistic `occupancyMap`s and SE2 state spaces, the RRT planner family, Monte-Carlo localisation, the VFH reactive controller, GNSS pseudorange trilateration, and Frenet-frame trajectory generation. This tutorial walks the worked examples in `examples/navigation/`.

## Supported features

- **State spaces + maps (T1)**: `occupancyMap` (probabilistic) with `setOccupancy` / `inflate`, `stateSpaceSE2`, `validatorOccupancyMap`, `navPath`.
- **Sampling planners (T2)**: `plannerRRT` (and the `plannerRRTStar` / `plannerAStarGrid` family) with `plan` and greedy `shortenpath` smoothing.
- **Localisation + reactive control (T5)**: `monteCarloLocalization` particle filter, `controllerVFH` vector-field-histogram steering.
- **GNSS positioning (T6)**: `gnssconstellation`, `pseudoranges`, `receiverposition` (least-squares trilateration), `gnssSensor`.
- **Frenet planning (T6)**: `referencePathFrenet`, `global2frenet`, `trajectoryGeneratorFrenet` with `connect`.
- **Shared inertial/orientation surface**: reuses the Sensor Fusion `imuSensor`, `gpsSensor`, `insfilterMARG`, and `quaternion` types.

## Build & run

```bash
build/matlabc -emit-llvm examples/navigation/nav_rrt_plan.m > /tmp/nav_rrt_plan.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/nav_rrt_plan.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/nav_rrt_plan
/tmp/nav_rrt_plan
```

Swap the filename for any other example below.

### RRT path planning — headline  (`examples/navigation/nav_rrt_plan.m`)

The full Tier-1 → Tier-2 arc: build an obstacle map, wrap it in an SE2 state space + validator, plan with `plannerRRT`, then smooth with `shortenpath`. Mirrors the MathWorks "Plan Mobile Robot Paths Using RRT" page.

```matlab
map = occupancyMap(25, 25, 1);
for y = 5:18,  setOccupancy(map, [10 y], 1.0);  end   % vertical wall
for x = 12:22, setOccupancy(map, [x 14], 1.0);  end   % horizontal wall
inflate(map, 0.5);                                    % robot-radius clearance

ss = stateSpaceSE2([0 25; 0 25; -pi pi]);
sv = validatorOccupancyMap(ss, map);
sv.ValidationDistance = 0.1;

planner = plannerRRT(ss, sv);
planner.MaxConnectionDistance = 3.0;
planner.MaxIterations = 20000;
planner.GoalBias = 0.1;

result = plan(planner, [2 2 0], [23 23 0]);
% row 1 = [numStates exitflag numIters]; rows 2..N+1 = path states
states = result(2:end, :);
np    = navPath(states);
short = shortenpath(np, sv);
```

`plan` returns a packed matrix whose first row carries `[numStates exitflag numIters]`, with the path states following. The script sums Euclidean segment lengths for the raw RRT path, then runs `shortenpath` to greedily shortcut it toward the straight-line lower bound `sqrt((23-2)^2 + (23-2)^2) ≈ 29.7 m`.

### Monte-Carlo localisation  (`examples/navigation/nav_mcl_localize.m`)

A `monteCarloLocalization` particle filter seeded around the start pose tracks a robot driving across an `occupancyMap`.

```matlab
map = occupancyMap(12, 12, 1);
% ... walls give the map structure ...
mcl = monteCarloLocalization(map);
mcl.NumParticles = 1000;

empty = zeros(0, 1);
pose  = step(mcl, [5 5 0], empty, empty);   % seed the cloud at the start
for k = 1:4
    pose = step(mcl, [5+k, 5, 0], empty, empty);
end
```

Each `step` propagates particles by the odometry motion model and (when a scan is supplied) reweights by a likelihood field over occupied cells. Driving `+x` from `(5,5)` to `(9,5)`, the estimate tracks the truth and the final error `abs(pose(1)-9)` stays small.

### VFH reactive obstacle avoidance  (`examples/navigation/nav_vfh_avoid.m`)

A `controllerVFH` picks a collision-free steering direction from a lidar scan plus a goal direction.

```matlab
vfh = controllerVFH();
vfh.NumAngularSectors = 180;
vfh.DistanceLimits    = [0.05, 4.0];
vfh.RobotRadius       = 0.25;
vfh.SafetyDistance    = 0.2;

ang = (-pi/2 : 0.05 : pi/2)';
r   = 8 * ones(size(ang,1), 1);
s1  = step(vfh, r, ang, 0.0);                % clear field -> steer ~0
% put a wall dead ahead (|ang| < 0.4 -> r = 0.8), then:
s2  = step(vfh, r, ang, 0.0);                % must divert around it
s3  = step(vfh, r, ang, 0.7);               % goal-right biases the opening
```

With a clear field VFH steers straight at the goal; with a wall dead ahead it diverts to an open sector, and a right-biased goal direction shifts which opening it selects.

### GNSS receiver positioning — headline  (`examples/navigation/nav_gnss_position.m`)

Recover a receiver position from satellite geometry and pseudoranges by iterative least-squares trilateration (3-D position + clock bias).

```matlab
sats = gnssconstellation(0);
truth = [37.4275, -122.1697, 30.0];        % Stanford, CA
pr  = pseudoranges(truth, sats);            % noiseless -> inverts exactly
pos = receiverposition(pr, sats);
gps = gnssSensor();
fix = step(gps, truth, [0 0 0]);            % noisy single-epoch fix
```

Noiseless pseudoranges invert to the true lat/lon/alt at metre-level horizontal error; the `gnssSensor` adds realistic single-epoch noise on top.

### Frenet trajectory generation  (`examples/navigation/nav_frenet_planner.m`)

Build a reference path, locate a vehicle in Frenet `(s, d)` coordinates, then generate a lane-change trajectory and convert it back to global `(x, y)`.

```matlab
rp = referencePathFrenet([0 0; 20 0; 40 8; 60 8]);
fr = global2frenet(rp, [22 3]);             % -> [s d]
tg = trajectoryGeneratorFrenet(rp);
traj = connect(tg, [fr(1) 0], [fr(1)+30, 3.5], 4.0);   % +3.5 m lateral over 30 m
% traj columns: [t ... d x y]; check traj(end,3) reaches 3.5 m
```

`connect` produces a smooth lateral shift to the target lane over the requested travel; the terminal sample's lateral offset `traj(end,3)` lands on the requested 3.5 m and `traj(end,4:5)` give the global endpoint.

The remaining examples reuse the Sensor Fusion inertial stack: **`nav_ground_vehicle.m`** runs an IMU(100 Hz)/GPS(10 Hz) nested predict / `fuseaccel` / `fusegps` loop on an `insfilterMARG` to dead-reckon a 4 m/s ground vehicle; **`nav_imu_intro.m`** drives an `imuSensor` with a known yaw-rate profile and averages back the noisy measurements; **`nav_rotations.m`** covers `eul2quat` / `quat2rotm` / `slerp` / `cross` orientation maths for a navigation frame.

## Limitations & carve-outs

From [`../navigation_toolbox_roadmap.md`](../navigation_toolbox_roadmap.md) §9:

- **All Simulink examples + blocks** (TEB local planner, Pure Pursuit, INS blocks) — go through the mflowLink lane.
- **ROS / ROS 2 integration** and **Gazebo cosimulation** — gated on a future ROS toolbox; the pure-MATLAB algorithm variants are in scope.
- **Deep-Learning planners** (MPNet, `mpnetSE2`, learned samplers) — gated on the Deep Learning toolbox; classical RRT/RRT*/Hybrid-A* are in scope.
- **3-D Visual SLAM / VIO** (`monovslam`/`stereovslam`) and **3-D lidar point-cloud SLAM** (NDT over point clouds) — gated on a future CV/Lidar toolbox; **2-D lidar-scan SLAM + the SE2 pose-graph / factor-graph back-end are in scope**.
- **Speedgoat / real-time hardware** and **hardware sensor streaming** (BNO055/ADIS16505/MPU-9250) — beyond the `-emit-c` lane.
- **Automated-driving scenario integration** (`drivingScenario`) — the bare Frenet trajectory layer is in scope.
- **Apps** (SLAM Map Builder) — interactive GUIs.
- `plannerHybridAStar`/`plannerBiRRT`, `stateSpaceSE3`/Reeds-Shepp, `poseGraph3D`/full `factorGraph`, `nmeaParser`/`rinexread`/`insEKF`, and real almanac/ephemeris constellations are documented follow-ons.

## See also

- Roadmap: [`../navigation_toolbox_roadmap.md`](../navigation_toolbox_roadmap.md)
- Examples: `examples/navigation/`
- Related: the inertial/orientation surface comes from the Sensor Fusion lane — see [`sensor_fusion_tutorial.md`](sensor_fusion_tutorial.md).
