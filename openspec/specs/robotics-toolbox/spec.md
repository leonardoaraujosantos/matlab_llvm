# Robotics System Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Robotics System Toolbox in `matlab_llvm`: rigid-body-tree kinematics/dynamics, inverse kinematics, mobile-robot kinematic models, path planning (PRM/RRT), collision detection, trajectory generation, and the coordinate-transformation surface.

## Requirements

### Requirement: Coordinate transformations
The system SHALL convert between rotation/transform representations.

#### Scenario: Convert between transform forms
- **WHEN** a program calls `eul2rotm`/`rotm2eul`, `axang2rotm`, `quat2tform`/`tform2quat`, `trvec2tform`, `eul2tform`, or angle wrapping
- **THEN** the system SHALL return the converted rotation matrix, transform, or wrapped angle (matlab_robotics_eul2rotm, matlab_robotics_axang2rotm, matlab_robotics_quat2tform, matlab_robotics_trvec2tform, matlab_robotics_wrapToPi) (doc: docs/robotics_toolbox_roadmap.md) (src: runtime/toolbox/robotics/runtime_robotics.cpp, runtime/toolbox/robotics/robotics_classdefs.m)

### Requirement: Rigid body tree kinematics and dynamics
The system SHALL model a `rigidBodyTree` and compute its kinematics and dynamics.

#### Scenario: Compute forward kinematics and dynamics
- **WHEN** a program builds a `rigidBodyTree`, queries `getTransform`/`geometricJacobian`, or computes mass matrix / inverse-forward dynamics / gravity torque
- **THEN** the system SHALL return the requested transform, Jacobian, or dynamics quantity (matlab_robotics_tree_init, matlab_robotics_getTransform, matlab_robotics_geometricJacobian, matlab_robotics_massMatrix, matlab_robotics_inverseDynamics, matlab_robotics_forwardDynamics, matlab_robotics_gravityTorque) (doc: docs/robotics_toolbox_roadmap.md) (src: runtime/toolbox/robotics/runtime_robotics.cpp, runtime/toolbox/robotics/robotics_classdefs.m)

### Requirement: Inverse kinematics
The system SHALL solve inverse and generalized inverse kinematics.

#### Scenario: Solve IK for a target pose
- **WHEN** a program uses `inverseKinematics` or `generalizedInverseKinematics` with pose/position/orientation constraints
- **THEN** the system SHALL return the joint configuration that satisfies the target (matlab_robotics_ik_init, matlab_robotics_ik_solve, matlab_robotics_gik_init, matlab_robotics_gik_solve, matlab_robotics_constraint_pose_init) (doc: docs/robotics_toolbox_roadmap.md) (src: runtime/toolbox/robotics/runtime_robotics.cpp, runtime/toolbox/robotics/robotics_classdefs.m)

### Requirement: Mobile robot kinematics
The system SHALL model mobile-robot kinematic models and pure-pursuit control.

#### Scenario: Integrate a kinematic model
- **WHEN** a program builds a `differentialDriveKinematics`/`bicycleKinematics`/`ackermannKinematics`/`unicycleKinematics` and computes its derivative, or steers with `controllerPurePursuit`
- **THEN** the system SHALL return the state derivative or steering command (matlab_robotics_diffdrive_derivative, matlab_robotics_bicycle_derivative, matlab_robotics_ackermann_derivative, matlab_robotics_unicycle_derivative, matlab_robotics_pursuit_step) (doc: docs/robotics_toolbox_roadmap.md) (src: runtime/toolbox/robotics/runtime_robotics.cpp, runtime/toolbox/robotics/robotics_classdefs.m)

### Requirement: Path planning and collision detection
The system SHALL plan paths and detect collisions.

#### Scenario: Plan and check for collisions
- **WHEN** a program plans with `mobileRobotPRM` or `manipulatorRRT`, or checks collisions between collision primitives (box/sphere/cylinder/capsule)
- **THEN** the system SHALL return a planned path or a collision result (matlab_robotics_prm_findpath, matlab_robotics_rrt_plan, matlab_robotics_checkCollision, matlab_robotics_gjk_collision, matlab_robotics_collbox_init) (doc: docs/robotics_toolbox_roadmap.md) (src: runtime/toolbox/robotics/runtime_robotics.cpp, runtime/toolbox/robotics/robotics_classdefs.m)

### Requirement: Trajectory generation
The system SHALL generate joint and task-space trajectories.

#### Scenario: Generate a trajectory
- **WHEN** a program calls `cubicpolytraj`, `trapveltraj`, `transformtraj`
- **THEN** the system SHALL return the sampled trajectory (positions/velocities/accelerations) (matlab_robotics_cubicpolytraj, matlab_robotics_trapveltraj, matlab_robotics_transformtraj) (doc: docs/robotics_toolbox_roadmap.md) (src: runtime/toolbox/robotics/runtime_robotics.cpp, runtime/toolbox/robotics/robotics_classdefs.m)
