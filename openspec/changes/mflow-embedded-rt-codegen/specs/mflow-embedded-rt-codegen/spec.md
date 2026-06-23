## ADDED Requirements

### Requirement: ERT-style entry-point contract

The system SHALL emit, from a whole `.mflow` diagram under an `--rt` / `--ert` mode, a flat
real-time C (and C++) implementation exposing `model_initialize`, `model_step`, and
`model_terminate` over a single caller-owned state struct (`RT_MODEL` containing the discrete
work/state `*_DW`, root inputs `*_U`, and root outputs `*_Y`), declared in a generated header.
All sizes SHALL be fixed at emit time and no runtime model interpreter SHALL be linked.

#### Scenario: Reusable step entry point
- **WHEN** a stateful diagram is emitted with `--rt`
- **THEN** the output declares `model_initialize`/`model_step`/`model_terminate` over an
  `RT_MODEL` struct, where `model_step` reads `m->u`, advances the persisted state in `m->dwork`,
  and writes `m->y`, with no dynamic allocation and no `MflowLinkSim` dependency

#### Scenario: Entry-point code reproduces the interpreter
- **WHEN** the emitted `model_step` is driven N times with the same inputs the model's sources
  produce, and its captured outputs are compared against `matlabc -simulate` of the same model
- **THEN** the two traces are byte-identical

### Requirement: Fixed-step real-time scheduling

For a multirate model the system SHALL generate base-rate plus sub-rate task entry points
(`model_step(m, tid)` with rate identifiers) and an `rt_OneStep()` template carrying rate
counters that invoke each rate at its period. A single-rate model SHALL collapse to a single
`model_step(m)`.

#### Scenario: Sub-rate fires at its period
- **WHEN** a multirate diagram with a base rate and a slower sub-rate is emitted with `--rt`
- **THEN** the generated scheduler invokes the sub-rate task once per its period (correct rate
  counter sequence) while the base-rate task runs every base step

### Requirement: Static / MISRA-leaning C profile

Under `--rt-profile=misra` the system SHALL emit step-path C that performs no heap allocation,
holds all state in the caller-owned struct, uses no variable-length arrays or recursion, and
emits explicit narrowing casts for fixed-point. Constructs that cannot satisfy the subset
SHALL carry a `MISRA deviation` marker naming the rule and reason.

#### Scenario: No allocation in the step path
- **WHEN** a model is emitted with `--rt --rt-profile=misra`
- **THEN** the generated `model_step` and its callees contain no `malloc`/`calloc`/`realloc`
  and keep all persistent state in `RT_MODEL`

### Requirement: Whole-diagram SystemVerilog

The system SHALL emit a whole discrete `.mflow` diagram as a single synthesizable SystemVerilog
top module that instantiates the per-subsystem modules, wires the inter-block signals, and
exposes one `clk`/`rst_n` plus the root IO as ports. A continuous-time block SHALL be rejected
with a sourced error directing the user to discretise first.

#### Scenario: Composed top module
- **WHEN** a discrete multi-subsystem diagram is emitted as whole-diagram SystemVerilog
- **THEN** the output is one synthesizable, Verilator-lint-clean top module wiring the subsystem
  instances, and it cosimulates with `matlabc -simulate` of the same model

#### Scenario: Continuous block rejected
- **WHEN** a diagram containing a continuous-time block is emitted as SystemVerilog
- **THEN** the system reports a sourced error rather than emitting non-synthesizable RTL

### Requirement: Packaged build bundle

Under `--emit-package <dir>` the system SHALL write the generated sources, a public header
declaring the entry points and structs, and a build manifest with a compile + link recipe, such
that the bundle cross-compiles standalone without hand-editing.

#### Scenario: Self-contained package
- **WHEN** a model is emitted with `--rt --emit-package <dir>`
- **THEN** `<dir>` contains the sources, `model.h`, and a build manifest that compiles the
  package with no external runtime library on the flat C path
