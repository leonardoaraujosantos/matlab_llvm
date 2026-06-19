# matlabc Command-Line Driver Spec

## Purpose
Document the observed behavior of `matlabc`, the command-line driver for the MATLAB compiler stack. A single driver accepts `.m` and `.mflow` inputs, selects a frontend by file extension, runs the shared Lex / Parse / Sema / MLIR pipeline, and dispatches to one of many output modes selected by a mode flag. This spec records input formats, frontend selection, the set of `-emit-*` flags, and output routing as they exist today (src: tools/matlabc/main.cpp; doc: README.md, docs/build_and_run.md).

## Requirements

### Requirement: Accepted input formats
The system SHALL accept MATLAB `.m` source files and `.mflow` flowchart-graph files as program inputs to the same driver.

#### Scenario: MATLAB source compiled
- **WHEN** the user runs `matlabc -emit-llvm foo.m`
- **THEN** the system SHALL lex/parse the `.m` source and run it through the shared pipeline (doc: docs/build_and_run.md)

#### Scenario: Flowchart graph compiled
- **WHEN** the user runs `matlabc -emit-matlab foo.mflow`
- **THEN** the system SHALL accept the `.mflow` input and produce output from the same pipeline (doc: README.md "Flowchart frontend")

### Requirement: Frontend selection by extension
The system SHALL select the flowchart frontend for inputs whose path ends in `.mflow` and the MATLAB lexer/parser frontend otherwise, with both frontends producing the same `TranslationUnit` for the shared Sema + MLIR pipeline.

#### Scenario: .mflow routed to the flowchart frontend
- **WHEN** the program path ends in `.mflow`
- **THEN** the system SHALL set `IsFlow=true` and route through the flowchart frontend instead of the MATLAB lexer/parser, feeding the resulting TU into the same downstream pipeline (src: tools/matlabc/main.cpp `endsWith(G.ProgramPath, ".mflow")`)

### Requirement: Multi-file sibling resolution
For a `.m` entry point, the system SHALL merge referenced sibling `.m` function/classdef files from the entry program's directory into the entry translation unit before Sema, so a script that calls helpers defined in separate sibling files resolves them. The merge is reference-gated (only siblings the program references, transitively) and deduped by symbol name (the entry's own definitions win). This SHALL hold uniformly for the `-dap` Debug launch and the code-generating `-emit-{llvm,c,cpp,python,typescript}` modes, so Compile/Run resolves multi-file programs the same way Debug does.

#### Scenario: AOT emit resolves a sibling helper
- **WHEN** the user runs `matlabc -emit-llvm prog.m` where `prog.m` calls `helper(...)` defined in a sibling `helper.m`
- **THEN** the system SHALL merge `helper.m` into the TU and resolve the call rather than failing with `undefined name 'helper'` (src: tools/matlabc/main.cpp `mergeReferencedSiblingFiles`; test: test/Run/sib332_multifile.m)

### Requirement: Emit / mode flags
The system SHALL select exactly one output mode from a mode flag, supporting at least the following: `-dump-tokens`, `-dump-ast`, `-emit-sema`, `-dump-call-sites`, `-emit-mir`, `-emit-mlir`, `-emit-llvm`, `-emit-c`, `-emit-cpp`, `-emit-python`, `-emit-typescript` (alias `-emit-ts`), `-emit-cuda`, `-emit-metal`, `-emit-opencl`, `-emit-systemverilog` (alias `-emit-sv`), `-check-synthesizable`, `-emit-hardware-report` (alias `-emit-hw-report`), `-emit-fixed-point-report` (alias `-emit-fi-report`), `-emit-cocotb`, `-emit-trace`, `-emit-matlab` (alias `-emit-m`), `-emit-mflow` (alias `-emit-flow`), `-emit-mflowlink-cpp` (alias `-emit-signal-flow-cpp`), `-dump-flow`, `-dump-chart`, `-format`, `-repl`, `-dap`, and `-simulate`.

#### Scenario: LLVM IR emitted
- **WHEN** the user passes `-emit-llvm`
- **THEN** the system SHALL emit final LLVM IR text to stdout (src: tools/matlabc/main.cpp `Mode::EmitLLVM`; doc: docs/debug.md MLIR/LLVM IR dumps)

#### Scenario: Alias accepted
- **WHEN** the user passes `-emit-sv`
- **THEN** the system SHALL select the same mode as `-emit-systemverilog` (src: tools/matlabc/main.cpp `A == "-emit-systemverilog" || A == "-emit-sv"`)

#### Scenario: Interactive and debug modes
- **WHEN** the user passes `-repl` or `-dap`
- **THEN** the system SHALL enter the JIT-backed REPL or the DAP debug server respectively rather than emitting a file (src: tools/matlabc/main.cpp `Mode::Repl` / `Mode::Dap`)

#### Scenario: Simulation mode and modifiers
- **WHEN** the user passes `-simulate` on a `.mflow` program, optionally with `--sim-dap` (live DAP server) or `--dry-run` (lower-only)
- **THEN** the system SHALL run the signal-flow / state-chart simulator — emitting CSV by default, booting the live DAP transport under `--sim-dap`, or only lowering under `--dry-run` (src: tools/matlabc/main.cpp `Mode::Simulate`, `Opts.SimulateDap`)

#### Scenario: List supported signal-flow kinds
- **WHEN** the user passes `-simulate --list-supported-kinds` (no model file required)
- **THEN** the system SHALL print a JSON array of `{kind, supported}` for every recognised `signal_*` block kind and exit 0, so tooling can distinguish shipped evaluators from reserved kinds (src: tools/matlabc/main.cpp `Opts.ListKinds`, lib/Flowchart/SignalFlowLowering.cpp `listSignalKinds`)

#### Scenario: Usage banner advertises the mode set
- **WHEN** the user invokes `matlabc` with no input file
- **THEN** the system SHALL print a usage banner that lists the simulation lane (`-simulate [--sim-dap | --dry-run]`) alongside the other modes and exit non-zero (src: tools/matlabc/main.cpp `usage`, test: test/Flowchart/SimulateDap/run_usage.py)

### Requirement: Output destination
The system SHALL write single-stream emit output to stdout and accept `-o <dir>` (alias `--output`) to name the output directory for the multi-file GPU bundle emitters (`-emit-{cuda,metal,opencl}`).

#### Scenario: GPU bundle directory
- **WHEN** the user passes `-emit-cuda -o /tmp/out foo.m`
- **THEN** the system SHALL write the standalone CUDA bundle (kernel + host driver + Makefile) into the given directory (doc: README.md GPU bundle row; src: tools/matlabc/main.cpp `-o` / `--output`)

### Requirement: Debug and line-table flags
The system SHALL accept `-g` (alias `--debug-hooks`) to inject `matlab_dbg_hook` calls and DWARF metadata, and `-line` / `-no-line` to control `#line` directive emission in `-emit-c` / `-emit-cpp` output (off by default).

#### Scenario: DWARF emitted with -g
- **WHEN** the user runs `matlabc -emit-llvm -g foo.m`
- **THEN** the system SHALL attach DWARF line-table metadata (`!DICompileUnit` / `!DISubprogram` / `!DILocation`) to the IR (doc: docs/debug.md "Native debugging via lldb / gdb")

#### Scenario: C output opts into #line
- **WHEN** the user runs `matlabc -emit-c -line foo.m`
- **THEN** the system SHALL annotate each emitted statement with a `#line "src.m"` directive (doc: docs/debug.md "#line directives in emitted C / C++")

### Requirement: Usage and help
The system SHALL print a usage summary listing the available modes when invoked with `-h` / `--help` or with no usable arguments.

#### Scenario: Help requested
- **WHEN** the user runs `matlabc -h`
- **THEN** the system SHALL print the usage string enumerating the mode flags and exit (src: tools/matlabc/main.cpp usage text / `-h` / `--help`)
