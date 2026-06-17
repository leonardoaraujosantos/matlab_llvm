# SystemVerilog Code Generation Spec

## Purpose
Document the observed behavior of `matlabc -emit-systemverilog`, which lowers a
constrained MATLAB subset into vendor-neutral, synthesizable SystemVerilog RTL
targeting standard-cell ASIC synthesis (Synopsys Design Compiler, Cadence Genus,
Yosys). The backend is legality-first: it decides whether the source is
hardware-inferable, emits combinational or sequential RTL accordingly, and
rejects non-synthesizable code with source-level diagnostics rather than
silently producing broken RTL.
(doc: docs/emit_systemverilog.md, doc: docs/sv_supported_subset.md)

## Requirements

### Requirement: Synthesizable subset emission
The system SHALL accept the documented MATLAB subset — scalar/fixed-vector
combinational arithmetic, `if`/`elseif`/`else`, `switch`/`case`, `persistent`
registers and counters, FSM cascades, static and persistent `fi`-arrays, and
constant-bound `for` loops — and emit synthesizable SystemVerilog.

#### Scenario: Combinational scalar function
- **WHEN** a function of pure scalar arithmetic with `if`/`else` is compiled with `-emit-systemverilog`
- **THEN** the system SHALL emit one `module` with `input`/`output` ports and a single `always_comb` block

#### Scenario: Persistent register with reset
- **WHEN** a `persistent` variable guarded by `if isempty(reg) ... end` is updated conditionally
- **THEN** the system SHALL emit an `always_ff @(posedge clk ...)` block with a reset branch and auto-added `clk` / `rst_n` ports

#### Scenario: FSM cascade
- **WHEN** a `persistent` integer state variable is driven by a `switch`/`case` (or `if` cascade) on the state
- **THEN** the system SHALL emit a `typedef enum` state type and render the transitions as a `unique case` inside `always_comb`/`always_ff`

### Requirement: Lint cleanliness
The system SHALL emit RTL that passes Verilator `--lint-only -Wall`, and the
reference designs in `examples/hdl/` SHALL additionally pass Yosys generic
synthesis for non-FSM designs. (doc: docs/sv_supported_subset.md; test: test/EmitSV)

#### Scenario: Golden fixture lint lane
- **WHEN** the `emit-sv` test lane runs over its golden fixtures
- **THEN** each emitted `.sv` SHALL lint clean under Verilator and match its golden diff

### Requirement: Port mapping and pragmas
The system SHALL map each function argument to an `input` port and each return
value to an `output` port, and SHALL honor `% hdl: port(<name>, <kind>, <sign>, <W>, <F>)`
pragmas on both inputs and outputs to drive port type, width, and signedness.
(doc: docs/sv_supported_subset.md; test: test/EmitSVPorts)

#### Scenario: Pragma-driven output port
- **WHEN** a function carries `% hdl: port(crc, fi, unsigned, 32, 0)` for an output named `crc`
- **THEN** the system SHALL render the port as `output logic [31:0] crc` instead of the default-signed form

#### Scenario: Supported integer widths only
- **WHEN** a pragma declares a width outside {1, 8, 16, 32, 64} (e.g. a 3-bit signal)
- **THEN** the system SHALL reject it and require rounding up to the next supported native width

### Requirement: Boolean-port lint hint
The system SHALL emit an actionable warning when a multi-bit `fi` input port is
used only in boolean predicates, suggesting it be declared `% hdl: port(name, bool)`.
(test: test/EmitSVHint)

#### Scenario: 8-bit port used as boolean
- **WHEN** an 8-bit `fi` input is consumed only by boolean comparisons
- **THEN** the system SHALL warn that the port is 8 bits wide but used as a boolean

### Requirement: Rejection of non-synthesizable constructs
The system SHALL run a synthesizability gate (`HWLegalize`) up front, emit no
`.sv` output on any rejection, and report a source-level diagnostic for
unsupported constructs — `while` loops, recursion, runtime-bounded `for` loops,
floating-point datapaths without `fi`, surviving runtime calls (e.g. `disp`),
`persistent` without an `isempty` initializer, and FSM ambiguities (duplicate or
empty case arms). (doc: docs/emit_systemverilog.md; test: test/EmitSVFail)

#### Scenario: while-loop rejected
- **WHEN** a function containing a `while` loop is compiled with `-emit-systemverilog`
- **THEN** the system SHALL abort with a source-level diagnostic and write no RTL

#### Scenario: check-synthesizable mode
- **WHEN** the user runs `-check-synthesizable` on a function
- **THEN** the system SHALL run the full legality gate, produce no `.sv`, and exit non-zero on any rejection
