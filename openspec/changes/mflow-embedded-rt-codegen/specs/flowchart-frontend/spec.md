## MODIFIED Requirements

### Requirement: Subsystem-to-MATLAB lowering
The system SHALL lower signal-flow subsystems and whole diagrams into MATLAB function ASTs that feed the existing `-emit-*` backends, producing flat, dependency-free code. Beyond the per-subsystem kernel and the whole-diagram `simulate()` driver, the system SHALL additionally emit, under a real-time mode, an Embedded-Coder-style entry-point contract (`model_initialize` / `model_step` / `model_terminate` over a static state struct) and a whole-diagram SystemVerilog top module. (src: lib/Flowchart/SubsystemToMatlab.cpp, src: include/matlab/Flowchart/SubsystemToMatlab.h)

#### Scenario: Subsystem export
- **WHEN** a signal-flow subsystem is converted
- **THEN** the system SHALL emit a MATLAB function AST representing the subsystem's stateless block computation (src: lib/Flowchart/SubsystemToMatlab.cpp, test/Flowchart/EmitSubsystem)

#### Scenario: Real-time entry-point emission
- **WHEN** a whole diagram is emitted under the real-time mode (`--rt` / `--ert`)
- **THEN** the system SHALL emit `model_initialize`/`model_step`/`model_terminate` over a static caller-owned `RT_MODEL` state struct, with no runtime model interpreter linked, and the step code SHALL reproduce `matlabc -simulate` of the same model byte-for-byte (src: lib/Flowchart/SubsystemToMatlab.cpp, capability mflow-embedded-rt-codegen, test/Flowchart/EmitRt)

#### Scenario: Whole-diagram SystemVerilog
- **WHEN** a discrete whole diagram is emitted as SystemVerilog
- **THEN** the system SHALL emit one synthesizable top module composing the per-subsystem modules, rejecting continuous-time blocks with a sourced error (src: lib/Flowchart/SubsystemToMatlab.cpp, capability mflow-embedded-rt-codegen)
