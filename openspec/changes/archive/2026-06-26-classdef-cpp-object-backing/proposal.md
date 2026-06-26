## Why

The `-emit-cpp` classdef emitter only works for classes whose properties are all
scalar doubles. For any stateful/handle classdef with string or vector
properties — `sim3d.World`/`Actor`, Control System Toolbox `ss`/`tf`/`zpk` model
objects — it emits C++ that either fails to compile or **compiles and then
segfaults** (issues #411, #412). This blocks the `examples/control/3d/` and
`examples/sim3d/` 3-D examples from running in compiled mode even though they run
correctly interpreted.

## What Changes

- **Reconcile the two conflicting classdef-emission models** in
  `lib/MLIR/Passes/EmitC.cpp`. Today `CppClassDef` materializes each class as a
  C++ object with native `double` fields, but constructor/method bodies store and
  read properties through the generic heap-object runtime (`matlab_obj_set_mat`
  /`matlab_obj_get_mat`), which require `this` to be a runtime `matlab_obj*`
  handle — not a stack C++ object. The emitter SHALL adopt a single consistent
  object-backing model (see design for options A/B/C and the chosen approach).
- **Correct handle identity.** Each emitted classdef object SHALL map to a stable
  runtime handle so the runtime's property/state operations dereference a real
  object, and handle-class aliasing semantics (passing an object to a method
  mutates the original; copies share state) match the interpreter.
- **Fix the surface emission bugs as part of the model** (necessary but not
  sufficient on their own — alone they turn the compile error into a crash):
  - **#412** — suppress the duplicate `Class() = default;` when the classdef has a
    zero-argument constructor; emit brace-init `Class name{};` for a no-arg ctor
    call to avoid the most-vexing-parse `Class name();`.
  - **#411** — materialize a handle-class object to its runtime handle where the
    runtime/sibling method expects one (e.g. `operator+`, `world.add(actor)`),
    and keep handle objects as stable named locals (do not inline a ctor result
    as a temporary that dangles).
- **Lifetime/inlining rule.** A handle-class value SHALL NOT be inlined into a
  larger expression as a temporary; it SHALL be a named local with a lifetime
  spanning its uses, because the runtime keys state on the object's handle.
- **Preserve interpreter parity.** Compiled output SHALL remain byte-identical to
  `matlabc -repl` for the covered programs; no regression to the existing
  classdef emit-c/cpp golden and compile-and-run tests.
- **Validate end-to-end.** Add `examples/control/3d/*_3d.m` and
  `examples/sim3d/*.m`-style programs as `test/Differential/` fixtures so the
  sim3d/ss classes are exercised in both lanes with identical output.

This is a compiler/runtime design change to an existing capability, not new
behavior visible to MATLAB authors — programs that already run interpreted should
simply also compile and run.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `c-cpp-codegen`: the **classdef mapped to C++** requirement changes from "emit a
  C++ class with property accessors bridging to the runtime" (which silently
  produces miscompiled/crashing code for non-scalar-property handle classes) to a
  consistent object-backed model that compiles and runs correctly with interpreter
  parity, plus handle-identity, lifetime, and handle-materialization guarantees.

## Impact

- **Source**: `lib/MLIR/Passes/EmitC.cpp` (`CppClassDef`, `emitCppClass`,
  `emitCppMethod`, `canInline`, ctor-call and method-call emission), and the
  shared C-mode `emitCStructTypedef` path if affected. Possibly the emitted
  runtime header (`runtime/matlab_runtime.hpp`) for the C++ wrapper/handle glue.
- **Runtime**: reuses the existing object-handle API (`matlab_obj_new`,
  `matlab_obj_get_mat`, `matlab_obj_set_mat`); new helpers only if the chosen
  design needs them.
- **Docs**: `docs/emit_cpp_classdef.md`.
- **Tests**: new `test/Differential/` fixtures (sim3d + ss); existing
  `test/EmitC` / `test/Run` classdef goldens must still pass.
- **Resolves**: #411 (`ss::operator+`), #412 (`sim3d.World` ctor); unblocks
  compiled mode for the `control-3d-examples` and sim3d example suites.
- **No impact** on the interpreter, `-emit-llvm`/python/typescript/SV lanes, or
  MATLAB-facing semantics.
