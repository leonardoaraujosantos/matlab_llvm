# Navigation Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Navigation Toolbox in `matlab_llvm`: occupancy mapping, sampling-based and grid path planning, pose-graph SLAM, particle-filter localization, GNSS positioning, and trajectory/state-space utilities. Layered on the shipped Robotics and Sensor Fusion toolboxes.

## Requirements

### Requirement: Occupancy mapping
The system SHALL build and query occupancy maps.

#### Scenario: Create and query an occupancy map
- **WHEN** a program constructs an `occupancyMap`, sets/gets cells, inflates obstacles, or checks occupancy
- **THEN** the system SHALL return the updated map or queried occupancy values (matlab_nav_occmap_init, matlab_nav_occmap_set, matlab_nav_occmap_get, matlab_nav_occmap_inflate, matlab_nav_occmap_check) (doc: docs/navigation_toolbox_roadmap.md) (src: runtime/toolbox/navigation/runtime_navigation.cpp, runtime/toolbox/navigation/navigation_classdefs.m)

### Requirement: Path planning
The system SHALL plan collision-free paths with grid and sampling-based planners.

#### Scenario: Plan a path
- **WHEN** a program calls `plannerAStarGrid`, `plannerRRT`/`plannerRRTStar`, or a generic planner with a state validator
- **THEN** the system SHALL return a planned `navPath` with length and shortening utilities (matlab_nav_astar_plan, matlab_nav_planner_plan, matlab_nav_path_init, matlab_nav_path_length, matlab_nav_shortenpath, matlab_nav_validator_isstate) (doc: docs/navigation_toolbox_roadmap.md) (src: runtime/toolbox/navigation/runtime_navigation.cpp, runtime/toolbox/navigation/navigation_classdefs.m)

### Requirement: SLAM and pose-graph optimization
The system SHALL match scans and optimize pose graphs.

#### Scenario: Build and optimize a SLAM graph
- **WHEN** a program adds scans to a `lidarSLAM`, matches scans, adds pose-graph relations, and optimizes
- **THEN** the system SHALL return the optimized pose graph (matlab_nav_slam_init, matlab_nav_slam_addscan, matlab_nav_matchscans, matlab_nav_posegraph_addrel, matlab_nav_posegraph_optimize) (doc: docs/navigation_toolbox_roadmap.md) (src: runtime/toolbox/navigation/runtime_navigation.cpp, runtime/toolbox/navigation/navigation_classdefs.m)

### Requirement: Probabilistic localization
The system SHALL localize with particle/Monte-Carlo filters and VFH steering.

#### Scenario: Run a particle-filter localization step
- **WHEN** a program initializes a `stateEstimatorPF` or `monteCarloLocalization`, predicts, corrects, and estimates, or steers with `controllerVFH`
- **THEN** the system SHALL return the updated state estimate or steering direction (matlab_nav_pf_predict, matlab_nav_pf_correct, matlab_nav_pf_estimate, matlab_nav_mcl_step, matlab_nav_vfh_step) (doc: docs/navigation_toolbox_roadmap.md) (src: runtime/toolbox/navigation/runtime_navigation.cpp, runtime/toolbox/navigation/navigation_classdefs.m)

### Requirement: GNSS positioning
The system SHALL model GNSS constellations and compute receiver positions.

#### Scenario: Compute a GNSS fix
- **WHEN** a program calls `gnssconstellation`, computes pseudoranges, or solves receiver position
- **THEN** the system SHALL return satellite states or the estimated receiver position (matlab_nav_gnssconstellation, matlab_nav_pseudoranges, matlab_nav_receiverposition) (doc: docs/navigation_toolbox_roadmap.md) (src: runtime/toolbox/navigation/runtime_navigation.cpp, runtime/toolbox/navigation/navigation_classdefs.m)

### Requirement: Trajectory and state-space utilities
The system SHALL generate trajectories and sample state spaces.

#### Scenario: Generate a Frenet/Dubins trajectory
- **WHEN** a program uses `referencePathFrenet`/`trajectoryGeneratorFrenet` or a `stateSpaceDubins`/`stateSpaceSE2` to sample, interpolate, and measure distance
- **THEN** the system SHALL return the generated trajectory or sampled states (matlab_nav_frenet_init, matlab_nav_trajgen_connect, matlab_nav_ss_dubins_init, matlab_nav_ss_sample, matlab_nav_ss_interpolate, matlab_nav_ss_distance) (doc: docs/navigation_toolbox_roadmap.md) (src: runtime/toolbox/navigation/runtime_navigation.cpp, runtime/toolbox/navigation/navigation_classdefs.m)
