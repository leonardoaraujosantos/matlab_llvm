# Robotics System Toolbox — Tutorial

The Robotics System lane compiles MATLAB manipulator kinematics, dynamics, mobile-robot models, and motion control to native code through the MATLAB → MLIR → LLVM pipeline. All six tiers ship: `rigidBodyTree` forward kinematics, damped-least-squares `inverseKinematics`, full rigid-body dynamics (RNEA / CRBA / forward dynamics), the four mobile kinematic models, `mobileRobotPRM` planning, and `controllerPurePursuit` path following. This tutorial walks the worked examples in `examples/robotics/`.

## Supported features

- **Rigid-body kinematics (T1–T2)**: `rigidBodyTree`, `loadrobot` (`'planar2'`), packed-DH `addBody`, `getTransform` (forward kinematics), `trvec2tform`.
- **Inverse kinematics (T3)**: `inverseKinematics` (Levenberg-Marquardt damped least squares) via the `matlab_robotics_ik_solve` entry.
- **Dynamics (T4)**: `inverseDynamics` (recursive Newton-Euler), `massMatrix` (composite-rigid-body), `forwardDynamics`, `velocityProduct`, `centerOfMass`.
- **Mobile kinematic models (T5)**: `unicycleKinematics`, `differentialDriveKinematics`, `bicycleKinematics`, `ackermannKinematics`, each with `derivative`.
- **Mobile planning + control (T5)**: `binaryOccupancyMap`, `mobileRobotPRM` with `findpath`, `controllerPurePursuit`.

## Build & run

```bash
build/matlabc -emit-llvm examples/robotics/ik_path_trace.m > /tmp/ik_path_trace.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/ik_path_trace.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/ik_path_trace
/tmp/ik_path_trace
```

Swap the filename for any other example below.

## Worked examples

### Inverse-kinematics path trace — headline  (`examples/robotics/ik_path_trace.m`)

Solve `inverseKinematics` at each waypoint of a 2-D path so a planar 2-link arm's end-effector traces it; verify each pose with forward kinematics.

```matlab
arm = rigidBodyTree();
loadrobot(arm, 'planar2');
ik  = inverseKinematics(arm);

xs = [1.4, 1.5, 1.6, 1.7];
ys = [0.4, 0.5, 0.6, 0.5];

q_prev = [0.0; 0.0];
for k = 1:4
    Tgt = trvec2tform([xs(k), ys(k), 0.0]);
    res = matlab_robotics_ik_solve(ik, Tgt, q_prev, 1.0, 0.0);
    qsol = [res(1); res(2)];
    Tf = getTransform(arm, qsol);   % Tf(4),Tf(8) = EE x,y
    q_prev = qsol;                  % warm-start the next waypoint
end
```

The solver runs damped least squares from the previous solution as a warm start, and `res(5)` carries the residual. With each waypoint inside the arm's 2 m reach, residuals drive to ~1e-6 and the forward-kinematics check `Tf(4),Tf(8)` lands on each target.

### Differential-drive PRM + pure pursuit — headline  (`examples/robotics/diffdrive_prm.m`)

Plan a path on a `binaryOccupancyMap` with a central wall, then drive a differential-drive robot toward it with `controllerPurePursuit`.

```matlab
map = binaryOccupancyMap(20, 20, 1.0);
for k = 1:11, setOccupancy(map, [10.0, 4+(k-1)], 1.0); end   % wall x=10

prm  = mobileRobotPRM(map, 200, 4.0);
path = findpath(prm, [2.0, 2.0], [18.0, 18.0]);

pp = controllerPurePursuit(path, 1.5, 0.6);
dd = differentialDriveKinematics(0.1, 0.5);

px = 2.0; py = 2.0; pth = 0.0;
for k = 1:5
    pose = [px; py; pth];
    cmd  = step(pp, pose);          % [v omega]
    dxy  = matlab_robotics_diffdrive_derivative(dd, pose, cmd);
    px = px + 0.5*dxy(1);  py = py + 0.5*dxy(2);  pth = pth + 0.5*dxy(3);
end
```

`mobileRobotPRM` samples 200 nodes (4 m connection radius) and Dijkstra-finds a path around the wall; `controllerPurePursuit` emits `[v omega]` commands toward the lookahead point, integrated through the differential-drive derivative each tick.

### Rigid-body dynamics round-trip  (`examples/robotics/arm_dynamics.m`)

Full RNEA / CRBA / forward-dynamics on the planar arm, verifying the inverse/forward-dynamics round-trip at machine precision.

```matlab
arm = rigidBodyTree();  loadrobot(arm, 'planar2');
q   = [0.4; -0.6];  qd = [0.3; 0.1];  qdd = [0.8; -0.2];

tau     = inverseDynamics(arm, q, qd, qdd);   % recursive Newton-Euler
M       = massMatrix(arm, q);                 % composite-rigid-body, symmetric
qdd_rec = forwardDynamics(arm, q, qd, tau);   % must recover qdd
vp      = velocityProduct(arm, q, qd);        % Coriolis/centrifugal
com     = centerOfMass(arm, q);
```

`inverseDynamics` computes the torque needed for the commanded `qdd`; feeding that torque back through `forwardDynamics` recovers `qdd` to ~1e-15 (`abs(qdd_rec-qdd)` round-trip error), confirming the RNEA and CRBA kernels are mutually consistent.

### 2-D inverse-kinematics circle trace  (`examples/robotics/ik_2d_circle.m`)

Build a 3-link planar arm with the packed-DH `addBody` form, then solve `inverseKinematics` for each point on a circular path.

```matlab
robot = rigidBodyTree();
addBody(robot, [0.5, 0, 0, 0], 1, -pi, pi);   % dh=[a alpha d theta], joint=1 (revolute)
addBody(robot, [0.5, 0, 0, 0], 1, -pi, pi);
addBody(robot, [0.5, 0, 0, 0], 1, -pi, pi);
ik = inverseKinematics(robot);

q = [0.1; 0.1; 0.1];
for k = 1:8
    ang = 2*pi*(k-1)/8;
    T = trvec2tform([0.9 + 0.3*cos(ang), 0.3*sin(ang), 0.0]);
    res = matlab_robotics_ik_solve(ik, T, q, 1.0, 0.0);
    q  = [res(1); res(2); res(3)];
    Tf = getTransform(robot, q);
end
```

The redundant 3-link arm traces a radius-0.3 circle; the max position error over the eight waypoints stays at solver tolerance.

### Mobile-robot kinematic models  (`examples/robotics/mobile_kinematics.m`)

Construct each kinematic model and integrate a short trajectory with `derivative` + forward Euler.

```matlab
uni = unicycleKinematics();
bic = bicycleKinematics(2.0);
ack = ackermannKinematics(2.0);

% unicycle: state [x y theta], command [v omega]
d = derivative(uni, [ux; uy; uth], [1.0; 0.3]);
% bicycle:  command [v psi] (steering angle)
d = derivative(bic, [bx; by; bth], [1.0; 0.2]);
% ackermann: state [x y theta psi], command [v psidot]
d = derivative(ack, [ax; ay; ath; aps], [1.0; 0.05]);
```

Each model's `derivative` returns the state rate for a given command; a gentle left turn integrated over 2 s shows the differing turn geometries (the unicycle/differential-drive turn in place via `omega`, the bicycle/Ackermann turn via steering angle).

## Limitations & carve-outs

From [`../robotics_toolbox_roadmap.md`](../robotics_toolbox_roadmap.md) §9:

- **All Simulink examples + blocks** (manipulator/mobile/trajectory blocks) — go through the mflowLink lane.
- **Unreal Engine simulation** and **Gazebo co-simulation / ROS integration** — external engines / a future ROS toolbox; the pure-MATLAB variants are in scope.
- **Offroad Autonomy support package** (MPPI offroad controller, terrain/DEM planners, lidar scene extraction) — gated on the support package + Lidar/terrain toolboxes.
- **Apps** (Inverse Kinematics Designer) — interactive GUI; the programmatic `inverseKinematics` is in scope.
- **Deep-Learning examples** (DLCHOMP, RL obstacle avoidance) — gated on the Deep Learning toolbox; classical `manipulatorCHOMP` is in scope.
- **Lidar / point-cloud planning** — gated on a future Lidar/CV toolbox.
- **Navigation Toolbox planners** (`plannerRRT`/`plannerHybridAStar`/SLAM) live in the separate Navigation lane; `manipulatorRRT` / `mobileRobotPRM` / `controllerPurePursuit` are in this toolbox.
- **Hardware deploy / Speedgoat / Simscape Multibody** — beyond the `-emit-c` lane.
- Full per-body `rigidBody`/`rigidBodyJoint` classes (packed `addBody` ships), `analyticalInverseKinematics`, `jointSpaceMotionModel`, full GJK collision, and `stateEstimatorPF` are documented follow-ons.

## See also

- Roadmap: [`../robotics_toolbox_roadmap.md`](../robotics_toolbox_roadmap.md)
- Examples: `examples/robotics/`
- Related: mobile-robot path planning on occupancy maps overlaps the Navigation lane — see [`navigation_tutorial.md`](navigation_tutorial.md).
