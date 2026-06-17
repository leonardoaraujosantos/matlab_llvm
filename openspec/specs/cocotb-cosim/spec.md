# CocoTB Co-Simulation Spec

## Purpose
Document the observed behavior of `matlabc -emit-cocotb`, an open-source
alternative to MathWorks HDL Verifier that closes the verification loop for the
SystemVerilog backend. From a single `.m` source, it generates a self-contained
CocoTB testbench directory that drives the emitted SystemVerilog DUT and the
emitted Python reference model in lockstep against random vectors, asserting
cycle-by-cycle bit-exact equality. (doc: docs/emit_cocotb.md, test: test/EmitCocoTB)

## Requirements

### Requirement: Self-contained harness bundle
The system SHALL emit a self-contained harness directory containing the DUT
(`<stem>.sv`), the Python reference (`<stem>_ref.py`), the CocoTB testbench
(`test_<stem>.py`), the `fi` pack/unpack helpers (`cocotb_fi.py`), the Python
runtime (`matlab_runtime.py`), and a `Makefile` wiring Verilator + CocoTB.
(doc: docs/emit_cocotb.md §Output directory layout)

#### Scenario: harness directory emitted
- **WHEN** the user runs `matlabc -emit-cocotb examples/hdl/alu_16bit.m`
- **THEN** the system SHALL write all harness files into `<stem>_cocotb` and report the input/output counts and vector count

#### Scenario: bundle is portable and buildable
- **WHEN** the emitted harness directory is moved elsewhere and `make` is run
- **THEN** the system SHALL build and run the testbench without external path setup

### Requirement: Python reference versus SV DUT lockstep
The system SHALL drive identical packed-bit stimulus into the DUT and identical
real-valued stimulus (after `pack_fi`/`unpack_fi` round-trip) into the Python
reference, and SHALL compare the DUT output decoded via `unpack_fi` against the
reference return value each cycle. (doc: docs/emit_cocotb.md §Design)

#### Scenario: bit-exact comparison per cycle
- **WHEN** the testbench runs on a synthesizable HDL example
- **THEN** the system SHALL assert that the DUT output equals the Python reference value cycle-by-cycle and fail loudly on any mismatch (including X/Z bits)

#### Scenario: combinational versus sequential mode
- **WHEN** the parsed SV port list contains synthetic `clk` / `rst_n`
- **THEN** the system SHALL switch the harness to sequential mode (clock, reset sequencing, post-edge settle) rather than combinational sampling

### Requirement: Port specs derived from the emitted SV
The system SHALL derive harness port types (direction, signedness, width, name,
unpacked-array length) by parsing the port list of the SV file just produced by
`-emit-systemverilog`, rather than re-walking the MLIR module. (doc: docs/emit_cocotb.md §Source of truth for port specs)

#### Scenario: vector port driven element-by-element
- **WHEN** a port is an unpacked array `logic ... [N]`
- **THEN** the system SHALL capture the array length and drive/read it element-by-element via `dut.<name>[k].value`

### Requirement: Stimulus, latency, and seed controls
The system SHALL accept CLI controls (`-cocotb-out`, `-cocotb-vectors`,
`-cocotb-latency`, `-cocotb-seed`) and source pragmas (`% cocotb: stimulus`,
`hold`, `latency`, `cover*`), with the CLI flag winning over a source pragma for
latency. (doc: docs/emit_cocotb.md §Quick Start, §Pipeline latency)

#### Scenario: pipeline latency alignment
- **WHEN** a pipelined design is emitted with `-cocotb-latency=2`
- **THEN** the system SHALL compare the DUT's response at cycle `k+2` against the reference's response to cycle `k`'s inputs, with no tail-flush phase

#### Scenario: deterministic seed
- **WHEN** `-cocotb-seed=N` is passed
- **THEN** the system SHALL seed the harness's randomization with N (default 42)

### Requirement: CI sweep over synthesizable examples
The system SHALL provide a `cocotb-tests` CI lane that sweeps every synthesizable
HDL example, asserts each reports `TESTS=1 PASS=1 FAIL=0`, and skips (ctest code
77) when Verilator or CocoTB is missing. (doc: docs/emit_cocotb.md §Status; test: test/EmitCocoTB/run_tests.sh)

#### Scenario: full sweep passes
- **WHEN** the `cocotb-tests` lane runs with Verilator and CocoTB on PATH
- **THEN** the system SHALL verify every example bit-exact (39/39 reported) between the SV DUT and the Python reference

#### Scenario: skip when tooling absent
- **WHEN** the lane runs without Verilator or CocoTB on PATH
- **THEN** the system SHALL report ctest's Skipped status rather than failing
