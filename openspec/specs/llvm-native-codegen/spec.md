# LLVM Native Codegen and JIT Spec

## Purpose
Documents the observed behavior of `matlabc`'s LLVM-based code generation and in-process JIT. The frontend lowers `.m` / `.mflow` input to MLIR, then `-emit-llvm` translates the LLVM dialect to textual LLVM IR for ahead-of-time native compilation, while the REPL and DAP debugger JIT the same MLIR module in-process via `mlir::ExecutionEngine`. This is the most mature execution path; the runtime is bundled into the `matlabc` binary so JIT runs need no external shared library.

## Requirements

### Requirement: Emit textual LLVM IR
The system SHALL, in `-emit-llvm` mode, lower the program through the full MLIR pipeline to the LLVM dialect, translate it to textual LLVM IR, and write the IR to stdout.

#### Scenario: Emitting IR for a script
- **WHEN** a user runs `matlabc -emit-llvm foo.m`
- **THEN** the system SHALL print textual LLVM IR to stdout and exit 0 on success (src: tools/matlabc/main.cpp; doc: docs/build_and_run.md)

#### Scenario: Debug metadata under -g
- **WHEN** `-emit-llvm -g` is passed
- **THEN** the system SHALL attach DWARF debug metadata (DICompileUnit / DIFile / DISubprogram, `!dbg` on instructions) so a debugger can step through the original `.m` source after native compilation (src: lib/MLIR/Passes/LowerToLLVMIR.cpp)

### Requirement: Runtime symbols are external references
The system SHALL emit references to `matlab_*` runtime entry points as external symbols in the LLVM IR rather than inlining the runtime, leaving them to be resolved at link time.

#### Scenario: Linking against the runtime
- **WHEN** a user compiles the emitted IR with `clang++` and links `libMatlabRuntime.a`
- **THEN** the external `matlab_*` symbols SHALL resolve to the bundled runtime and produce a native executable (doc: docs/build_and_run.md; src: CMakeLists.txt)

### Requirement: In-process JIT execution
The system SHALL JIT-compile the lowered MLIR module in-process using `mlir::ExecutionEngine` and invoke the entry function directly, without writing any intermediate textual LLVM IR.

#### Scenario: Running a REPL turn
- **WHEN** the REPL (`matlabc -repl`) accepts a statement
- **THEN** the system SHALL lower it to MLIR with REPL mode enabled, convert the LLVM dialect, create an `ExecutionEngine`, look up the entry function, and call it (src: tools/matlabc/main.cpp; doc: docs/repl.md)

#### Scenario: Resolving runtime symbols in the JIT
- **WHEN** the JITed code references `matlab_*` runtime functions
- **THEN** the system SHALL resolve them against the running `matlabc` process image, into which `libMatlabRuntime.a` is statically linked, requiring no external `.so` (src: CMakeLists.txt; doc: docs/build_and_run.md)

### Requirement: REPL workspace and function persistence
The system SHALL persist top-level workspace variables and user-defined functions across REPL turns so later turns observe earlier state.

#### Scenario: Variable survives across turns
- **WHEN** a top-level variable is assigned in one REPL turn and referenced in a later turn
- **THEN** the system SHALL route the binding through `matlab_ws_get_*` / `matlab_ws_set_*` runtime calls backed by process-global workspace state so the value is observed (doc: docs/repl.md; src: tools/matlabc/main.cpp)

#### Scenario: User function reused in a later turn
- **WHEN** a previously defined top-level function is called in a later turn
- **THEN** the system SHALL prepend the stashed function source so the function and call site compile in the same translation unit (src: tools/matlabc/main.cpp)

### Requirement: DAP-backed JIT debugging
The system SHALL, in DAP mode, JIT the program using the same pipeline as the REPL while injecting per-statement debug hooks so a Debug Adapter Protocol client can set breakpoints and step.

#### Scenario: Breakpoint hook injection
- **WHEN** a program is run under `matlabc -dap`
- **THEN** the system SHALL compile with REPL-style workspace persistence and a debug hook before statements so the DAP client can pause execution (src: tools/matlabc/main.cpp; doc: docs/debug.md)

### Requirement: Golden test coverage for execution
The system SHALL verify native/JIT execution by running programs and comparing their stdout against stored golden files.

#### Scenario: Run-lane regression
- **WHEN** the `run-tests` lane executes the programs under `test/Run`
- **THEN** the system SHALL compare each program's stdout against its golden output (numeric-tolerance aware) and fail on divergence (src: test/Run; src: CMakeLists.txt)
