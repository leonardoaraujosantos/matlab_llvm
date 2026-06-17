# C and C++ Codegen Spec

## Purpose
Documents the observed behavior of `matlabc`'s `-emit-c` and `-emit-cpp` backends, which translate the lowered MLIR module to self-contained C or C++ source. The emitted source links against the in-tree C++20 runtime to form a standalone executable with no LLVM toolchain at the compile step. The backends share the lowering pipeline with `-emit-llvm` up through I/O lowering, keeping `scf` control flow structural so it prints as readable C.

## Requirements

### Requirement: Emit self-contained C/C++ source
The system SHALL, in `-emit-c` or `-emit-cpp` mode, translate the lowered MLIR module to C or C++ source and write it to stdout.

#### Scenario: Emitting C for a script
- **WHEN** a user runs `matlabc -emit-c foo.m > foo.c`
- **THEN** the system SHALL print self-contained C source to stdout that, when compiled with the runtime, reproduces the program's behavior (doc: docs/emit_c_cpp.md; src: lib/MLIR/Passes/EmitC.cpp)

#### Scenario: Emitting C++ for the same script
- **WHEN** a user runs `matlabc -emit-cpp foo.m`
- **THEN** the system SHALL print C++ source that differs from C output mainly in including C++ headers, wrapping runtime prototypes in `extern "C"`, and emitting `std::cout`-based `disp` instead of `printf`/`puts` (doc: docs/emit_c_cpp.md; src: test/EmitC)

### Requirement: Runtime linkage via void-pointer prototypes
The system SHALL emit its own runtime prototypes with `void*` pointer parameters so the same source compiles as C or C++ and links against the typed in-tree runtime.

#### Scenario: Compiling and linking against the runtime
- **WHEN** the emitted source is compiled with a plain C/C++ compiler and linked against `runtime/matlab_runtime.cpp` and sibling translation units
- **THEN** the `void*`-typed declarations SHALL resolve to the runtime's typed definitions and produce a standalone executable, requiring no LLVM toolchain (doc: docs/emit_c_cpp.md; doc: docs/runtime.md)

### Requirement: #line directives off by default
The system SHALL omit `#line` directives in C/C++ output by default and emit them only when explicitly requested.

#### Scenario: Default emission
- **WHEN** `-emit-c` / `-emit-cpp` runs without `-line`
- **THEN** the system SHALL produce output without `#line` directives for readability (src: tools/matlabc/main.cpp)

#### Scenario: Opt-in source mapping
- **WHEN** `-line` is passed
- **THEN** the system SHALL emit `#line` directives mapping back to the original `.m` source for debugger stepping (src: tools/matlabc/main.cpp)

### Requirement: classdef mapped to C++
The system SHALL translate MATLAB `classdef` definitions into C++ wrapper classes over the runtime object model when emitting C++.

#### Scenario: Emitting a class
- **WHEN** `-emit-cpp` processes a program containing a `classdef`
- **THEN** the system SHALL emit a C++ class with methods and property accessors bridging to the runtime (doc: docs/emit_cpp_classdef.md; src: lib/MLIR/Passes/EmitC.cpp)

### Requirement: Fail fast on unsupported ops
The system SHALL exit non-zero with a diagnostic when the module contains an MLIR op the C/C++ emitter cannot lower, rather than producing broken output silently.

#### Scenario: Unsupported op
- **WHEN** `-emit-c` encounters an op with no emitter mapping
- **THEN** the system SHALL exit non-zero and print an `emit-c: unsupported op` diagnostic to stderr (doc: docs/emit_c_cpp.md; src: test/EmitCFail)

### Requirement: Compile-and-run golden coverage
The system SHALL verify C/C++ output by compiling it with the runtime, running it, and comparing stdout against the reference, including a strict warning lane.

#### Scenario: Run-lane parity
- **WHEN** the `run-tests-emit-c` / `run-tests-emit-cpp` lanes compile and execute each program
- **THEN** the system SHALL compare stdout (tolerance-aware) and, in strict lanes, fail on `-Wall -Wextra -Werror` warnings (src: CMakeLists.txt; src: test/Run/run_tests_emitc.sh)

#### Scenario: Shape goldens
- **WHEN** the `test/EmitC` lane runs
- **THEN** the system SHALL diff generated C/C++ source against `.c.expected` / `.cpp.expected` goldens to catch cosmetic regressions (src: test/EmitC)
