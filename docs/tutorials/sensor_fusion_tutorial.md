# Sensor Fusion & Tracking Toolbox — Tutorial

The Sensor Fusion & Tracking lane compiles MATLAB orientation maths, inertial-navigation filters, and multi-object trackers straight to native code through the MATLAB → MLIR → LLVM pipeline. All six tiers ship: quaternion algebra, `imuSensor`/`gpsSensor` models, an INS filter, the Munkres-gated `trackerGNN`, covariance-intersection fusion, and the GOSPA/OSPA tracking-quality metrics. This tutorial walks the worked examples in `examples/fusion/`.

## Supported features

- **Orientation maths (T1)**: `quaternion` value type, `eul2quat` / `quat2eul` / `quat2rotm`, `slerp`, plus everyday vector helpers `cross` / `dot` / `deg2rad`.
- **Inertial + GNSS sensor models (T3)**: `imuSensor` (accelerometer + gyroscope specific-force/angular-rate simulation), `gpsSensor`.
- **Inertial-navigation filter (T3)**: `insfilterMARG` with `predict` / `fuseaccel` / `fusegps` and a readable `.State` vector (quaternion, position, velocity).
- **Multi-object tracking (T5)**: `trackerGNN` over constant-velocity tracking-EKF filters with Mahalanobis gating + Munkres assignment, `step`, `numConfirmed`, `.States`.
- **Track fusion + metrics (T6)**: `trackFuser` (per-target covariance intersection), `trackGOSPAMetric` and `trackOSPAMetric` scoring against ground truth.

## Build & run

```bash
build/matlabc -emit-llvm examples/fusion/gnn_air_traffic.m > /tmp/gnn_air_traffic.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/gnn_air_traffic.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/gnn_air_traffic
/tmp/gnn_air_traffic
```

Swap the filename for any other example below.

## Worked examples

### GNN air-traffic tracking — headline  (`examples/fusion/gnn_air_traffic.m`)

Three aircraft on parallel northbound headings emit noisy 2-D detections over 40 timesteps. A `trackerGNN` confirms all three tracks using Mahalanobis gating and Munkres (global-nearest-neighbour) assignment.

```matlab
dt   = 1.0;
trk  = trackerGNN(16);
% ... three targets at y = 0 / 30 / 60, marching in +x ...
for k = 1:nsteps
    % advance truth, add pseudo-random noise, build a 3x2 detection matrix
    det = [p1x + nz1, p1y + nz1*0.5;
           p2x + nz2, p2y + nz2*0.5;
           p3x + nz3, p3y + nz3*0.5];
    step(trk, det, dt);
end
nc = numConfirmed(trk);
S  = trk.States;
```

Each `step` runs predict → gate → assign → update. After 40 steps `numConfirmed` reports `3 / 3`, and `trk.States` packs the constant-velocity state (`[x vx y vy]` per track) so positions read out as `S(1),S(3)`, `S(5),S(7)`, `S(9),S(11)` — recovering the three lanes near `x≈40` with `y≈0/30/60`.

### IMU + GPS inertial navigation — headline  (`examples/fusion/imu_gps_fusion.m`)

End-to-end INS over a straight-line trajectory: an `imuSensor` and `gpsSensor` drive an `insfilterMARG` (complementary-filter quaternion + gravity-compensated double integrator + linear GPS correction).

```matlab
fs = 100;  dt = 1.0 / fs;  N = 400;
imu = imuSensor(fs);   gps = gpsSensor(1);   ins = insfilterMARG(fs);
for k = 1:N
    z_imu = step(imu, [0,0,9.81], [0,0,0]);   % flat & level
    acc_meas  = [z_imu(1), z_imu(2), z_imu(3)];
    gyro_meas = [z_imu(4), z_imu(5), z_imu(6)];
    predict(ins, acc_meas, gyro_meas, dt);
    fuseaccel(ins, acc_meas);
    if mod(k, 10) == 0                        % GPS correction at 10 Hz
        z_gps = step(gps, [k*dt*2,0,0], true_v);
        fusegps(ins, [z_gps(1),z_gps(2),z_gps(3)], [z_gps(4),z_gps(5),z_gps(6)]);
    end
end
S = ins.State;   % S(1:4)=quat, S(5:7)=pos, S(8:10)=vel
```

The inner loop dead-reckons from the IMU at 100 Hz; the outer cadence fuses GPS every 10th step. After 4 s the estimated position `S(5:7)` tracks the true `x = N*dt*2 = 8 m` and velocity converges to `[2,0,0] m/s`. The same nested predict/fuse structure appears in the navigation tutorial's ground-vehicle example.

### Quaternion & rotation maths  (`examples/fusion/quaternion_rotations.m`)

Tier-1 orientation algebra: build quaternions from Euler angles, rotate a body vector, interpolate with `slerp`, and round-trip through the matrix/Euler conversions.

```matlab
q1 = quaternion(1, 0, 0, 0);              % identity
q2_data = eul2quat([30*pi/180, 0, 0]);    % 30 deg yaw
R2 = quat2rotm(q2_data);
vp = R2 * [1.0; 0.0; 0.0];                % rotate body x-axis into nav frame
qhalf = slerp(q1.Data, q2.Data, 0.5);     % midpoint orientation
Eh = quat2eul(qhalf);                     % -> yaw = 0.5 * 30 deg
```

`slerp` returns the half-way orientation, so `quat2eul` recovers a 15-degree yaw. The script also exercises `cross`, `dot`, and `deg2rad`.

### Track fusion + quality metrics  (`examples/fusion/track_fusion_metrics.m`)

Two virtual sensors observe three targets; per-target estimates are fused with `trackFuser` (covariance intersection) and scored against ground truth with GOSPA/OSPA.

```matlab
g_a = trackGOSPAMetric(sensorA, truth, cutoff, p);
o_a = trackOSPAMetric(sensorA, truth, cutoff, p);
for i = 1:3
    PA = [0.4 0; 0 0.4];   PB = [0.8 0; 0 0.8];   % sensor A tighter
    F  = trackFuser([sensorA(i,1);sensorA(i,2)], PA, ...
                    [sensorB(i,1);sensorB(i,2)], PB);
    fused(i,:) = [F(1) F(2)];
end
g_f = trackGOSPAMetric(fused, truth, cutoff, p);
```

`trackFuser` weights the tighter sensor more heavily via the trace-minimising covariance-intersection blend, so the fused GOSPA `g_f` beats either single-sensor score — the printed improvement `g_a - g_f` is positive.

## Limitations & carve-outs

From [`../sensor_fusion_toolbox_roadmap.md`](../sensor_fusion_toolbox_roadmap.md) §9:

- **All Simulink examples + blocks** — the MATLAB object/function API is the target; block diagrams go through the separate mflowLink lane.
- **Apps** (Tracking Scenario Designer, Tracking Data Importer, Ground Truth Labeler) — interactive GUIs; the programmatic `trackingScenario`/`objectDetection` APIs are in scope.
- **Deep-Learning examples** (DeepSORT, ReID, pose estimation, autoencoder anomaly detection) — gated on the Deep Learning toolbox.
- **Lidar / point-cloud tracking** (`pointCloud`, voxel/grid lidar trackers) — gated on a future Lidar/CV toolbox; the radar-detection path is in scope.
- **Radar-signal-level + Phased-Array** (`phased.*`, beamforming, CFAR-on-FPGA) — gated on a future Radar toolbox.
- **Automated Driving / camera-visual tracking** and **hardware streaming** (wireless IMU, board-in-the-loop).
- The full vector-source `trackFuser`/`staticDetectionFuser` track-to-track API, `trackerJPDA`/`TOMHT`/`trackerPHD`, OOSM retrodiction, and the full 16-state EKF with Joseph-form updates are documented follow-ons; the demos exercise the per-target/single-filter primitives directly.

## See also

- Roadmap: [`../sensor_fusion_toolbox_roadmap.md`](../sensor_fusion_toolbox_roadmap.md)
- Examples: `examples/fusion/`
- Related: the navigation lane reuses `imuSensor` / `gpsSensor` / `quaternion` — see [`navigation_tutorial.md`](navigation_tutorial.md).
