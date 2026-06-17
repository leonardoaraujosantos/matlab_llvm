# RF Propagation Toolbox Spec

## Purpose
Documents the shipped subset of the RF propagation surface in the matlab_llvm compiler: closed-form ITU-R/NIST and cellular path-loss models, an engineering Longley-Rice (ITM) port, Fresnel-zone and diffraction math, geographic helpers, terrain profile / line-of-sight / link-budget / coverage-grid computation, directional antenna patterns with mount orientation, multi-site coverage aggregation, and `propagationModel`/`txsite`/`rxsite` objects. Tiers 1a/1b/2a/2b/3 are shipped (2026-05-17). (doc: docs/propagation_toolbox_roadmap.md) (src: runtime/toolbox/prop)

## Requirements

### Requirement: Path-loss models
The system SHALL provide closed-form ITU-R/NIST and cellular empirical path-loss models. (doc: docs/propagation_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Compute path loss
- **WHEN** a program calls free-space, rain, gas, fog, close-in, Hata, COST231, Egli, ECC33, SUI, or Ericsson9999 path loss
- **THEN** the system SHALL return path loss in dB via the matching runtime entry (e.g. `matlab_prop_fspl`, `matlab_prop_pathloss_rain`, `matlab_prop_pathloss_gas`, `matlab_prop_pathloss_hata`, `matlab_prop_pathloss_cost231`, `matlab_prop_pathloss_sui`)

### Requirement: Longley-Rice (ITM) terrain model
The system SHALL provide an engineering Longley-Rice (ITM) path-loss model with reliability-quantile correction. (doc: docs/propagation_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Compute ITM path loss
- **WHEN** a program calls the ITM model with a terrain profile, frequency, antenna heights, polarization, climate, and reliability triple (time/location/situation)
- **THEN** the system SHALL return median path loss with Gaussian quantile correction across LOS/diffraction/troposcatter regimes via `matlab_prop_itm_pathloss`

### Requirement: Fresnel zone, diffraction, and geographic helpers
The system SHALL provide Fresnel-zone math, knife-edge/multi-edge diffraction, and geodesic helpers. (doc: docs/propagation_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Compute clearance, diffraction, or distance
- **WHEN** a program calls Fresnel-zone radius/clearance, knife-edge/Bullington/Deygout diffraction, or geodesics (haversine, Vincenty, bearing, destination point)
- **THEN** the system SHALL return the radius/clearance/loss/distance via the matching runtime entry (e.g. `matlab_prop_fresnel_zone_radius`, `matlab_prop_diff_knife_edge`, `matlab_prop_diff_bullington`, `matlab_prop_haversine`, `matlab_prop_vincenty`)

### Requirement: Terrain profile, line-of-sight, link budget, and coverage
The system SHALL provide terrain-profile sampling, line-of-sight checks, link-budget computation, and coverage-grid generation. (doc: docs/propagation_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Compute a link budget or coverage grid
- **WHEN** a program calls terrain profile, LOS obstruction/clearance, link budget, or single-TX coverage grid
- **THEN** the system SHALL return the elevation samples, LOS result, link-budget struct, or received-power matrix via `matlab_prop_terrain_profile`, `matlab_prop_los_obstruction`/`matlab_prop_los_clear`, `matlab_prop_link_budget`, or `matlab_prop_coverage_grid`

### Requirement: Directional patterns, mount orientation, and multi-site coverage
The system SHALL provide directional antenna patterns, mount-orientation transforms, and multi-site coverage aggregation. (doc: docs/propagation_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Apply patterns and aggregate multi-site coverage
- **WHEN** a program calls sector/cosine/Gaussian/isotropic patterns, mount-orientation conversion, or multi-site coverage with best-server/sum-power/SINR aggregation
- **THEN** the system SHALL return the pattern gain, local angles, or aggregated coverage matrix via `matlab_prop_pat_sector`/`matlab_prop_pat_cosine`/`matlab_prop_pat_gaussian`/`matlab_prop_pat_isotropic`, `matlab_prop_mount_to_local`, or `matlab_prop_coverage_grid_multi`

### Requirement: Site and model objects
The system SHALL provide `propagationModel`, `txsite`, and `rxsite` objects with link/coverage/signal-strength methods. (doc: docs/propagation_toolbox_roadmap.md) (src: runtime/toolbox/prop/rf_class_propagationmodel.m) (src: runtime/toolbox/prop/rf_class_txsite.m) (src: runtime/toolbox/prop/rf_class_rxsite.m)

#### Scenario: Build sites and evaluate a link
- **WHEN** a program constructs a `propagationModel` (Kind selects the model variant), `txsite`, and `rxsite`, and calls `pathloss`, `link`, `los`, `coverage`, or `sigstrength`
- **THEN** the system SHALL dispatch to the runtime model via `matlab_prop_dispatch_pathloss` and return path loss, distance, LOS flag, coverage grid, or received signal strength (`matlab_prop_sigstrength`)
