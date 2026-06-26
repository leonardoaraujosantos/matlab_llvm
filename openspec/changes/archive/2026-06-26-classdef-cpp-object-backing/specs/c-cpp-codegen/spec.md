## MODIFIED Requirements

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

## ADDED Requirements

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
