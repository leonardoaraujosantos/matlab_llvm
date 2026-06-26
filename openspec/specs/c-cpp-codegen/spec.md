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
The system SHALL translate MATLAB `classdef` definitions into C++ classes over a
single, consistent object-backing model when emitting C++. The emitted class, its
property storage, and its method/constructor bodies SHALL agree on how an object
maps to runtime state, so that a program using a classdef whose properties are not
all scalar doubles (strings, vectors, or mixed) both compiles and runs, producing
output identical to the interpreter. Property reads/writes inside method bodies
SHALL NOT pass a stack C++ object where the runtime expects a heap object handle.

#### Scenario: Emitting a class
- **WHEN** `-emit-cpp` processes a program containing a `classdef`
- **THEN** the system SHALL emit a C++ class with methods and property accessors
  bridging to the runtime under a consistent object-backing model (doc:
  docs/emit_cpp_classdef.md; src: lib/MLIR/Passes/EmitC.cpp)

#### Scenario: Handle classdef with non-scalar properties compiles and runs
- **WHEN** `-emit-cpp` processes a program that constructs a handle classdef with
  string and/or vector properties (e.g. `sim3d.Actor` with `Name`/`Shape`/`Color`/
  `Size`), sets those properties, and calls its methods
- **THEN** the emitted C++ SHALL compile and run without crashing, and its stdout
  SHALL be byte-identical to `matlabc -repl` on the same program

#### Scenario: Property access does not dereference a stack object as a struct
- **WHEN** a classdef method body assigns or reads a property (`obj.Name = ...`)
- **THEN** the emitted code SHALL operate on a valid runtime object handle (not a
  stack C++ object reinterpreted as a `matlab_struct`/`matlab_obj`), so no
  out-of-bounds dereference occurs at runtime

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

### Requirement: Handle-class object identity and aliasing
Emitted handle-class (`< handle`) objects SHALL share a single runtime backing so
that passing an object to a method, or aliasing it through another variable,
observes mutations on the same underlying state — matching MATLAB handle
semantics and the interpreter.

#### Scenario: Method mutates the shared object
- **WHEN** a handle object is passed to a method that sets one of its properties,
  then a property is read back through the original variable
- **THEN** the read SHALL reflect the mutation (compiled output equals interpreter)

#### Scenario: Object stored by another object is reachable later
- **WHEN** `world.add(actor)` stores a handle and a later `world` method uses the
  stored actor
- **THEN** the stored handle SHALL still be valid (no dangling temporary)

### Requirement: Handle-class objects are stable named locals
The emitter SHALL NOT inline a handle-class constructor result into a larger
expression as a temporary; a handle-class value SHALL be emitted as a named local
whose lifetime spans its uses, because the runtime keys state on the object's
handle.

#### Scenario: Constructor result used as a method argument
- **WHEN** a program constructs an object and immediately passes it to a method
  (`world.add(sim3d.Actor(...))`)
- **THEN** the emitted C++ SHALL bind the constructed object to a named local and
  pass that local, not a temporary materialized inside the call

### Requirement: No-argument constructor emission is well-formed
For a classdef with a zero-argument constructor, the emitter SHALL NOT also emit a
defaulted constructor of the same signature, and a no-argument construction SHALL
be emitted in a form that is not parsed as a function declaration.

#### Scenario: No-arg constructor does not collide with a defaulted one (#412)
- **WHEN** `-emit-cpp` emits a classdef that defines `function obj = World()`
- **THEN** the class SHALL declare exactly one no-argument constructor (no
  `Class() = default;` alongside the user `Class() { ... }`)

#### Scenario: No-arg construction avoids the most-vexing-parse (#412)
- **WHEN** the emitter declares a local initialized by a no-argument constructor
  call
- **THEN** it SHALL emit brace-initialization (`Class name{};`), not `Class name();`

### Requirement: Handle-object operands materialize to their runtime handle
The emitter SHALL materialize a handle-class operand to its runtime handle
wherever a runtime call or a sibling classdef method expects a bare handle, such
as a binary operator reading the other operand's handle or a method parameter
that is itself a class object. The emitter SHALL NOT pass the C++ object by value
or reference into a handle-typed slot.

#### Scenario: Binary operator on model objects compiles (#411)
- **WHEN** `-emit-cpp` emits an operator method that consumes the other operand's
  runtime handle (e.g. `ss::operator+`)
- **THEN** the emitted C++ SHALL compile (no "cannot convert Class to void*") and
  run with interpreter-identical output

### Requirement: classdef compile-and-run differential coverage
The system SHALL cover the 3-D control and sim3d example programs that exercise
non-scalar-property handle classdefs with differential fixtures that run each
program through both the interpreter and the compiled emit-cpp lane and require
identical stdout. These fixtures SHALL be part of the CI gate.

#### Scenario: sim3d / control-3d examples pass differentially
- **WHEN** the differential test lane runs the added sim3d/control-3d fixtures
- **THEN** the interpreter and compiled outputs SHALL match, and the lane SHALL be
  part of the CI gate

