# Stateflow (mStateflow) Spec

## Purpose
Document the shipped subset of mStateflow in `matlab_llvm`: a hierarchical, event-driven state-chart dialect of the `.mflow` JSON container (`settings.kind = "state_chart"`) that compiles to software (MATLAB/MIR/LLVM/C/C++) and to synthesizable FSM SystemVerilog. Status is PARTIAL: the compiler/debugger/DAP/REPL side and integer-typed Moore/Mealy/AND-parallel examples are shipped end-to-end, while broader Stateflow semantic parity remains a work in progress.

## Requirements

### Requirement: State-chart schema and loading
The system SHALL load `.mflow` files declared as the `state_chart` dialect.

#### Scenario: Load a chart file
- **WHEN** a program loads a `.mflow` file whose `settings.kind == "state_chart"` with hierarchical states, transitions, and symbols
- **THEN** the system SHALL parse it into a chart IR and validate parent resolution, AND-execution order, and default-transition multiplicity (doc: docs/mStateflow_roadmap.md) (src: lib/StateChart/StateChartIR.cpp, include/matlab/StateChart/StateChartIR.h)

### Requirement: Chart interpretation with deterministic traces
The system SHALL simulate a chart with deterministic event-driven semantics.

#### Scenario: Step a chart on an event
- **WHEN** a program drives a chart with an input event and reads active states
- **THEN** the system SHALL evaluate guards/actions, update active states, and emit a deterministic event trace (doc: docs/mStateflow_roadmap.md) (src: lib/StateChart/Interpreter.cpp, runtime/toolbox/stateflow/mstateflow_helpers.m)

### Requirement: Software lowering across emit lanes
The system SHALL lower a chart to a software `tick` function across the matlabc emit lanes.

#### Scenario: Emit software for a chart
- **WHEN** a program compiles a chart with `-emit-matlab`, `-emit-mir`, `-emit-llvm`, `-emit-c`, or `-emit-cpp`
- **THEN** the system SHALL emit a single `<chart>_tick(...)` function using flat persistent-scalar state (no struct/string literals) (doc: docs/mStateflow_roadmap.md) (src: lib/StateChart/Lowering.cpp, include/matlab/StateChart/Lowering.h)

### Requirement: Synthesizable SystemVerilog FSM generation
The system SHALL lower integer-typed charts to synthesizable FSM SystemVerilog.

#### Scenario: Emit RTL for a Moore/Mealy chart
- **WHEN** a program compiles an integer-typed Moore/Mealy/AND-parallel chart with `-emit-systemverilog` (and runs `-check-synthesizable`)
- **THEN** the system SHALL emit a verilator-lint-clean SystemVerilog module, with optional hardware report and cocotb testbench (`-emit-hardware-report`, `-emit-cocotb`) (doc: docs/mStateflow_roadmap.md) (src: lib/StateChart/Lowering.cpp, examples/stateflow)

### Requirement: stateChart classdef and history
The system SHALL expose a `stateChart` classdef wrapper and chart history operations.

#### Scenario: Use the stateChart wrapper
- **WHEN** a program constructs a `stateChart` and uses history push/pop/auto-snapshot and reset helpers
- **THEN** the system SHALL invoke the chart tick and manage the chart's live-state and history ring (doc: docs/mStateflow_roadmap.md) (src: runtime/toolbox/stateflow/stateflow_classdefs.m, runtime/toolbox/stateflow/mstateflow_helpers.m)

### Requirement: Partial parity status
The system SHALL document that mStateflow ships a partial (🟡) subset of Stateflow semantics.

#### Scenario: Consult shipped scope
- **WHEN** a developer needs to know which Stateflow features are guaranteed
- **THEN** the system SHALL provide the shipped-tier list (Tiers 0–10 UI; integer-typed Moore/Mealy/AND-parallel RTL examples) and treat features outside it as not-yet-shipped (doc: docs/mStateflow_roadmap.md) (src: lib/StateChart/Lowering.cpp, examples/stateflow/README.md)
