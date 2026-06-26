## 1. Audit & decide the model

- [x] 1.1 Inventory existing classdef emit-cpp coverage: which `test/EmitC` + `test/Run` fixtures emit a classdef, and which rely on native `double` fields vs the generic-object runtime path
- [x] 1.2 Map `CppClassDef` / `emitCppClass` / `emitCppMethod` / `canInline` and every property store/read lowering (`matlab_obj_set_mat`/`get_mat`, `obj_get_f64`, native field) — document the two conflicting models
- [x] 1.3 Confirm the runtime object-handle ownership model (`matlab_obj_new`, `matlab_*_new`) — who owns/frees the handle, lifetime guarantees
- [x] 1.4 Resolve design open question D1: pick model A (handle-over-runtime-object for `< handle`) vs C (hybrid); record the decision and the native-field-path retention boundary

## 2. Object-backing implementation

- [x] 2.1 Emit handle-class objects over a consistent runtime handle (per D1): construction via `matlab_obj_new`/`matlab_*_new`, a handle accessor on the wrapper, dtor policy per 1.3
- [x] 2.2 Route property store/read in ctor/method bodies to operate on the object's handle (no stack object passed to `matlab_obj_set_mat`/`get_mat`)
- [x] 2.3 Keep the existing native-field path for the value/scalar-double classes identified in 1.1 (no regression)

## 3. Surface fixes (folded into the model)

- [x] 3.1 (#412) Suppress `Class() = default;` when the classdef has a zero-argument constructor method
- [x] 3.2 (#412) Emit brace-init `Class name{};` for a no-argument constructor call (all ctor-call emission sites), avoiding the most-vexing-parse
- [x] 3.3 (Lifetime, D2) Not needed: `__h` is a heap handle from `matlab_obj_new`, so it survives even when the C++ wrapper is a temporary — inlined handle objects don't dangle
- [x] 3.4 (#411, D3) Materialize a handle-class operand to its runtime handle at method/runtime-call boundaries (handle accessor, not a blanket `operator void*`)

## 4. Validation

- [x] 4.1 `sim3d.World`/`Actor` minimal program: `-emit-cpp` compiles, runs, stdout == `-repl`
- [x] 4.2 `ss(A,B,C,D)` program (#411): compiles and runs, stdout == `-repl`
- [x] 4.3 Add `test/Differential/` fixtures for `examples/sim3d/*.m` and representative `examples/control/3d/*_3d.m` (numeric/diagnostic output; deterministic)
- [x] 4.4 Run the full `test/EmitC` (golden) + `test/Run` (emit-c/emit-cpp) + `differential-tests` + `emit-sv` lanes — no regressions
- [x] 4.5 Verify interpreter, `-emit-llvm`, python/typescript lanes unchanged for a classdef program

## 5. Docs & close-out

- [x] 5.1 Update `docs/emit_cpp_classdef.md` with the object-backing model and the handle/lifetime/materialization rules
- [x] 5.2 `openspec validate` (valid); #411 + #412 closed by PR #416
- [x] 5.3 Note in `examples/control/3d/README.md` + `examples/sim3d/README.md` that the suites now run compiled as well as interpreted
