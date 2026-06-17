# Sensor Fusion and Tracking Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Sensor Fusion and Tracking Toolbox in `matlab_llvm`: the `quaternion` value type, inertial-fusion filters, tracking Kalman filters with motion/measurement models, multi-object tracking, inertial/GNSS sensor models, and tracking metrics. Layered on the shipped EKF/UKF cores.

## Requirements

### Requirement: Quaternion type and orientation
The system SHALL provide a `quaternion` value type with rotation operations.

#### Scenario: Operate on quaternions
- **WHEN** a program constructs a `quaternion`, multiplies/normalizes/inverts, rotates points/frames, slerps, or converts to/from rotation matrices and Euler angles
- **THEN** the system SHALL return the resulting quaternion or rotation (matlab_fusion_quat_init_wxyz, matlab_fusion_quat_mul_data, matlab_fusion_quat_rotatepoint, matlab_fusion_quat_slerp, matlab_fusion_quat2rotm, matlab_fusion_quat_to_eul) (doc: docs/sensor_fusion_toolbox_roadmap.md) (src: runtime/toolbox/fusion/runtime_fusion.cpp, runtime/toolbox/fusion/fusion_classdefs.m)

### Requirement: Inertial fusion filters
The system SHALL fuse IMU/GPS data into orientation/pose estimates.

#### Scenario: Step an inertial fusion filter
- **WHEN** a program steps an `ahrsfilter`, `imufilter`, `complementaryFilter`, or `insfilterMARG` (predict / fuse-accel / fuse-gps), or runs `ecompass`
- **THEN** the system SHALL return the updated orientation/pose estimate (matlab_fusion_ahrs_step, matlab_fusion_imufilter_step, matlab_fusion_compfilter_step, matlab_fusion_insmarg_predict, matlab_fusion_insmarg_fuse_gps, matlab_fusion_ecompass) (doc: docs/sensor_fusion_toolbox_roadmap.md) (src: runtime/toolbox/fusion/runtime_fusion.cpp, runtime/toolbox/fusion/fusion_classdefs.m)

### Requirement: Tracking filters and motion models
The system SHALL provide trackingKF/EKF/UKF filters with standard motion/measurement models.

#### Scenario: Predict and correct a tracking filter
- **WHEN** a program initializes a `trackingKF`/`trackingEKF`/`trackingUKF`, predicts, and corrects, using `constvel`/`constacc`/`constturn` motion models and their measurement functions
- **THEN** the system SHALL return the updated state and covariance (matlab_fusion_trackingkf_predict, matlab_fusion_trackingekf_correct, matlab_fusion_trackingukf_predict, matlab_fusion_constvel, matlab_fusion_cvmeas, matlab_fusion_initcvekf) (doc: docs/sensor_fusion_toolbox_roadmap.md) (src: runtime/toolbox/fusion/runtime_fusion.cpp, runtime/toolbox/fusion/fusion_classdefs.m)

### Requirement: Multi-object tracking
The system SHALL associate detections to tracks with a GNN tracker.

#### Scenario: Step a GNN tracker
- **WHEN** a program feeds `objectDetection`s to a `trackerGNN`, which runs Munkres assignment and returns confirmed `objectTrack`s
- **THEN** the system SHALL return the updated track list and confirmed count (matlab_fusion_gnn_init, matlab_fusion_gnn_step, matlab_fusion_gnn_numconfirmed, matlab_fusion_assignmunkres, matlab_fusion_objdet_init) (doc: docs/sensor_fusion_toolbox_roadmap.md) (src: runtime/toolbox/fusion/runtime_fusion.cpp, runtime/toolbox/fusion/fusion_classdefs.m)

### Requirement: Sensor models and geodesy
The system SHALL model inertial/GNSS sensors and convert geodetic coordinates.

#### Scenario: Sample a sensor model
- **WHEN** a program steps an `imuSensor`/`gpsSensor`, computes Allan variance, follows a `waypointTrajectory`, or converts `lla2ned`/`ned2lla`
- **THEN** the system SHALL return the simulated measurements or converted coordinates (matlab_fusion_imu_step, matlab_fusion_gps_step, matlab_fusion_allanvar, matlab_fusion_waypoint_lookup, matlab_fusion_lla2ned, matlab_fusion_ned2lla) (doc: docs/sensor_fusion_toolbox_roadmap.md) (src: runtime/toolbox/fusion/runtime_fusion.cpp, runtime/toolbox/fusion/fusion_classdefs.m)

### Requirement: Tracking metrics and smoothing
The system SHALL evaluate tracking quality and smooth estimates.

#### Scenario: Compute tracking metrics
- **WHEN** a program computes GOSPA/OSPA/track-error metrics or runs an RTS smoother
- **THEN** the system SHALL return the metric value or smoothed state sequence (matlab_fusion_gospa, matlab_fusion_ospa, matlab_fusion_trackerror, matlab_fusion_rts_smoother) (doc: docs/sensor_fusion_toolbox_roadmap.md) (src: runtime/toolbox/fusion/runtime_fusion.cpp, runtime/toolbox/fusion/fusion_classdefs.m)
