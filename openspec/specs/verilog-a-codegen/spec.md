# Verilog-A Code Generation Spec

## Purpose
Document the observed behavior of the Verilog-A emission path, which writes
synthesizable analog behavioral models (`.va`) that load directly into standard
SPICE-class analog simulators (Cadence Spectre, ngspice + OpenVAF, Xyce, Mentor
Eldo, Synopsys CustomSim, Keysight ADS). Unlike the digital SystemVerilog
backend, Verilog-A emission is implemented as MATLAB runtime functions
(`writeVerilogA*`) that write the `.va` as a side effect, so the same `.m` source
stays executable through every other backend.
(doc: docs/emit_verilog_a.md, doc: docs/verilog_a_plan.md, test: test/EmitVA)

## Requirements

### Requirement: Rational / TF / ZPK / state-space model export
The system SHALL emit Verilog-A behavioral modules from rational fits, transfer
functions, zero-pole-gain forms, and SISO state-space descriptions, folding
real poles to `laplace_nd` sections and complex-conjugate pole pairs to
real-coefficient biquad sections. (doc: docs/emit_verilog_a.md §Tier-1–3)

#### Scenario: rationalfit export
- **WHEN** `writeVerilogA(mdl, 'm.va')` is called on a rationalfit/RFRational instance
- **THEN** the system SHALL write a sum-of-poles `.va` module with real-pole `laplace_nd` and complex-pair biquad sections, wrapping bulk delay via `absdelay`

#### Scenario: transfer-function export
- **WHEN** `writeVerilogATF(num, den, 'm.va')` is called with descending-power-of-s columns
- **THEN** the system SHALL emit a single `laplace_nd` contribution module

### Requirement: Analog primitive and component model emission
The system SHALL emit Verilog-A modules for analog sources, comparators, Schmitt
triggers, VCOs, behavioral DACs, compact device models (diode, op-amp, RTD,
thermistor), noise sources, and lookup tables. (doc: docs/emit_verilog_a.md §Tier-4–9)

#### Scenario: temperature-dependent RTD
- **WHEN** `writeVerilogARTD(R0, alpha, T0, 'm.va')` is called
- **THEN** the system SHALL emit a module using first-class `$temperature` in the resistance expression

#### Scenario: lookup-table model with sidecar
- **WHEN** `writeVerilogATable(x_col, y_col, 'm.va')` is called
- **THEN** the system SHALL write a `.tbl` sidecar plus a `.va` referencing it via `$table_model` for 1-D linear interpolation

### Requirement: Source remains executable through other backends
The system SHALL treat Verilog-A emission as one additional output lane, leaving
the MATLAB source runnable via `-emit-llvm`, the REPL, and the DAP debugger.
(doc: docs/emit_verilog_a.md §Design philosophy)

#### Scenario: numeric sanity check on the LLVM lane
- **WHEN** a `writeVerilogA*` example is compiled with `matlabc -emit-llvm`
- **THEN** the system SHALL execute the numeric body and write its `.va` (and any `.tbl`) into the working directory

### Requirement: Lint and cosim workflows
The system SHALL provide opt-in lint and co-simulation lanes for emitted `.va`
files that prefer OpenVAF (lint) / ngspice+OpenVAF (cosim) with ADMS/Xyce
fallbacks, and SHALL skip cleanly with exit 0 when no linter or simulator is
installed. (doc: docs/emit_verilog_a.md §Tier-10; test: test/EmitVA/run_lint.sh, test/EmitVA/run_cosim.sh)

#### Scenario: lint lane without OpenVAF
- **WHEN** the `run-emit-va-admslint` lane runs on a machine with neither OpenVAF nor ADMS
- **THEN** the system SHALL skip cleanly with exit 0

#### Scenario: cosim AC sweep on a 1-in/1-out module
- **WHEN** the cosim lane runs against a 2-port `.va` with ngspice+OpenVAF available
- **THEN** the system SHALL compile the `.va` and run an AC sweep on the canonical netlist template
