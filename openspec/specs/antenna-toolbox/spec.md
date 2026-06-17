# Antenna Toolbox Spec

## Purpose
Documents the shipped MVP subset of the Antenna Toolbox in the matlab_llvm compiler: thin-wire dipole/monopole geometry objects with a resonance-sizing `design` method, and closed-form (induced-EMF / sinusoidal-current) solvers that compute input impedance, S-parameters, and radiation patterns. The ANT-Tier-2 MVP is shipped; multi-wire and surface Method-of-Moments solvers and arrays remain not started. (doc: docs/antenna_toolbox_roadmap.md) (src: runtime/toolbox/antenna) (src: runtime/toolbox/prop/runtime_prop.cpp)

## Requirements

### Requirement: Antenna geometry objects with resonance sizing
The system SHALL provide thin-wire dipole and monopole geometry objects with a `design` method that auto-sizes to a target frequency. (src: runtime/toolbox/antenna/ant_class_dipole.m) (src: runtime/toolbox/antenna/ant_class_monopole.m)

#### Scenario: Construct and size an antenna
- **WHEN** a program constructs `AntDipole(length_m, width_m, feed_offset)` or `AntMonopole(height_m, width_m, gp_size_m)` and calls `design(ant, freq_hz)`
- **THEN** the system SHALL store the geometry properties and resize to half-wave (dipole) or quarter-wave (monopole) at the target frequency

### Requirement: Closed-form thin-wire solver
The system SHALL provide a closed-form thin-wire dipole solver returning input impedance, S11, VSWR, and return loss. (doc: docs/antenna_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Solve a thin-wire dipole
- **WHEN** a program calls `antennaWireSolve(length_m, radius_m, n_segments, freq_Hz)`
- **THEN** the system SHALL return a struct with `Zin_re`/`Zin_im`/`S11_re`/`S11_im`/`VSWR`/`ReturnLoss_dB` computed by the induced-EMF method via `matlab_ant_wire_solve`

### Requirement: Radiation pattern and gain
The system SHALL provide a closed-form radiation-pattern function and per-object peak gain. (doc: docs/antenna_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Compute a pattern or peak gain
- **WHEN** a program calls `antennaWirePattern(length_m, radius_m, n_segments, freq_Hz, n_theta)` or `antennaGain(antenna, freq)`
- **THEN** the system SHALL return the sinusoidal-current pattern struct (`Theta`/`ETheta`/`Gain_dBi`/`Directivity_dBi`) via `matlab_ant_wire_pattern`, or the peak broadside gain in dBi via `matlab_prop_antenna_gain` (AntDipole 2.15 dBi, AntMonopole 5.15 dBi)

### Requirement: S-parameter bridge to RF Toolbox
The system SHALL provide an antenna S-parameter sweep that produces an RF-Toolbox-shaped result. (doc: docs/antenna_toolbox_roadmap.md) (src: runtime/toolbox/prop/runtime_prop.cpp)

#### Scenario: Sweep antenna S-parameters
- **WHEN** a program calls `antennaWireSparameters(length_m, radius_m, n_segments, freqs_col)`
- **THEN** the system SHALL return a 1-port `RFSparameters`-shaped struct (complex `S11`, `Frequencies`, `Z0 = 50`, `NumPorts = 1`) via `matlab_ant_wire_sparameters`, ready to drop into the RF Toolbox cascade
