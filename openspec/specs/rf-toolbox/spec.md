# RF Toolbox Spec

## Purpose
Documents the shipped subset of the RF Toolbox in the matlab_llvm compiler: network-parameter objects (S/Y/Z/H/G/ABCD/T) with conversions, Touchstone I/O, 2-port and N-port analyses, cascade (T-chain and Redheffer star product), vector fitting with passivity enforcement, time-domain (ZOH/TDR/TDT) response, transmission-line geometries, matching networks and LC filters, an RF circuit-element hierarchy, and a Verilog-A export arc. The roadmap reports the toolbox 100% complete (function-form plus classdefs). (doc: docs/rf_toolbox_plan.md) (src: runtime/toolbox/rf)

## Requirements

### Requirement: Network-parameter objects and conversions
The system SHALL provide S/Y/Z/H/G/ABCD/T network-parameter objects and conversions between representations. (src: runtime/toolbox/rf/rf_class_sparameters.m) (src: runtime/toolbox/rf/runtime_rf.cpp)

#### Scenario: Build network parameters and convert representation
- **WHEN** a program constructs `RFSparameters`/`RFYparameters`/`RFZparameters`/`RFHparameters`/`RFGparameters`/`RFAbcdparameters`/`RFTparameters` or calls conversions `s2y`/`s2z`/`s2h`/`s2g`/`s2abcd`/`s2t` (and inverses `h2s`/`g2s`/`abcd2s`/`t2s`)
- **THEN** the system SHALL return the converted parameters via the matching runtime entry (e.g. `matlab_rf_s2y`, `matlab_rf_s2abcd`, `matlab_rf_abcd2s`), including N-port variants (`matlab_rf_s2y_n`, `matlab_rf_s2z_n`)

### Requirement: Touchstone I/O
The system SHALL read and write Touchstone files and expose typed parameter getters. (doc: docs/rf_toolbox_plan.md) (src: runtime/toolbox/rf/runtime_rf.cpp)

#### Scenario: Read and query a Touchstone file
- **WHEN** a program reads a `.sNp` file or writes a 2-port `.s2p` and queries port-pair parameters
- **THEN** the system SHALL parse/emit MA/DB/RI formats (Touchstone v1 and v2) via `matlab_rf_touchstone_read`/`matlab_rf_touchstone_write_s2p` and return entries via typed getters such as `matlab_rf_ts_sij`, `matlab_rf_ts_freqs`, `matlab_rf_ts_z0`

### Requirement: RF analyses and cascade
The system SHALL provide 2-port and N-port RF analyses and network cascading. (doc: docs/rf_toolbox_plan.md) (src: runtime/toolbox/rf/runtime_rf.cpp)

#### Scenario: Analyze and cascade networks
- **WHEN** a program computes input/output reflection, VSWR, power gain, stability (K/mu and stability circles), group delay, S-to-TF, or cascades networks (2-port T-chain or N-port Redheffer)
- **THEN** the system SHALL return the result via the matching runtime entry (e.g. `matlab_rf_gamma_in`/`matlab_rf_gamma_out`, `matlab_rf_vswr_from_gamma`, `matlab_rf_power_gain`, `matlab_rf_stability_k`/`matlab_rf_stability_mu`, `matlab_rf_groupdelay`, `matlab_rf_cascade2`/`matlab_rf_cascade_n_full`)

### Requirement: Vector fitting, passivity, and time-domain response
The system SHALL provide rational (vector) fitting with passivity enforcement and time-domain/TDR/TDT response. (doc: docs/rf_toolbox_plan.md) (src: runtime/toolbox/rf/rf_class_rfrational.m) (src: runtime/toolbox/rf/runtime_rf.cpp)

#### Scenario: Fit a rational model and compute its response
- **WHEN** a program calls `rationalfit` (optionally weighted), checks/enforces passivity, evaluates the fit, or computes time/TDR/TDT response
- **THEN** the system SHALL return an `RFRational` model and responses via `matlab_rf_rationalfit`/`matlab_rf_rationalfit_w`, `matlab_rf_passivity`/`matlab_rf_enforce_passivity`, `matlab_rf_freqresp`, and `matlab_rf_timeresp`/`matlab_rf_s2tdr`/`matlab_rf_s2tdt`

### Requirement: Transmission lines, matching networks, and LC filters
The system SHALL provide transmission-line geometries, matching-network synthesis, and LC filter blocks. (doc: docs/rf_toolbox_plan.md) (src: runtime/toolbox/rf/runtime_rf.cpp)

#### Scenario: Build a line, matching network, or filter
- **WHEN** a program calls a transmission-line geometry (`txline`/`coaxial`/`microstrip`/`cpw`/`parallelplate`/`twowire`), an L/T/Pi matching network, or an LC filter block
- **THEN** the system SHALL return the S-parameters or element values via the matching runtime entry (e.g. `matlab_rf_microstrip`, `matlab_rf_matchingnetwork`/`matlab_rf_matchingnetwork_t`/`matlab_rf_matchingnetwork_pi`, `matlab_rf_lc_filter`/`matlab_rf_lc_filter4`)

### Requirement: RF circuit-element hierarchy, budget, and Verilog-A export
The system SHALL provide an RF circuit-element classdef hierarchy with `analyze`, RF budget analysis, and Verilog-A export. (doc: docs/rf_toolbox_plan.md) (src: runtime/toolbox/rf/rf_class_amplifier.m) (src: runtime/toolbox/rf/runtime_rf.cpp)

#### Scenario: Analyze a circuit element, run a budget, or export Verilog-A
- **WHEN** a program constructs `RFCktAmplifier`/`RFCktMixer`/`RFCktPassive`/`RFCktSeries`/`RFCktShunt`/`RFCktCascade` and calls `analyze`, runs an RF budget, or exports Verilog-A
- **THEN** the system SHALL return synthesized S-parameters, cascaded gain/NF/IP3/SNR, or a `.va` file via the matching runtime entry (e.g. `matlab_rf_analyze_amplifier`/`matlab_rf_analyze_passive`/`matlab_rf_analyze_series`/`matlab_rf_analyze_shunt`, `matlab_rf_budget_friis`/`matlab_rf_budget_table`, `matlab_rf_write_verilog_a`)
