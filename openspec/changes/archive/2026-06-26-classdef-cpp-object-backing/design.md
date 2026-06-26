## Context

`EmitC.cpp` carries two classdef-emission models that contradict each other:

- **Structural**: `CppClassDef` collects a class's *scalar-double* properties and
  `emitCppClass` emits a C++ class with native `double` fields + a defaulted ctor.
- **Runtime-object**: constructor/method bodies set and read properties through the
  generic heap-object runtime — `matlab_obj_set_mat(this, "Name", ...)` /
  `matlab_obj_get_mat(this, ...)` — which require `this` to be a `matlab_obj*`.

For a class whose properties are all scalar doubles and never touch the generic
path, the structural model happens to work. For `sim3d.Actor` (`Name`/`Shape`
strings, `Color`/`Size` vectors) and CST `ss`/`tf`/`zpk`, the bodies hit the
runtime-object path with `this` pointing at a stack C++ object → `struct_find_field`
walks a stack address as a struct and segfaults (confirmed backtrace, 2026-06-25).
The interpreter and `-emit-llvm` lanes are unaffected; this is C/C++-emit only.

Runtime already exposes the object-handle API: `matlab_obj_new(class_id)`,
`matlab_obj_get_mat`, `matlab_obj_set_mat`, plus the toolbox `matlab_*_new(this)`
registrations (sim3d keys state on the pointer passed to `world_new`/`actor_new`).

## Goals / Non-Goals

**Goals**
- One coherent object-backing model: object ↔ runtime handle is consistent across
  field storage, ctor, methods, operators, get/set, and call sites.
- `examples/control/3d/*_3d.m` and `examples/sim3d/*.m` compile and run with
  stdout byte-identical to `-repl`; covered in `test/Differential/`.
- Correct handle-class identity/aliasing and object lifetime (no dangling temporaries).
- Resolve #411 and #412 as part of the model.

**Non-Goals**
- No interpreter, `-emit-llvm`, `-emit-python/typescript`, or SV changes.
- No new MATLAB-facing classdef semantics.
- Not redesigning value-class (non-handle) numeric wrappers (e.g. `Matrix`) that
  already work; only the object-backed classdef path.
- Swing-up / behavioral changes to the examples themselves — they already run.

## Decisions

**D1 — Object-backing model (the core choice).** Evaluate three:

- **(A) Handle-over-runtime-object.** The C++ class holds a single opaque
  `matlab_obj*` (or `void*`) handle created in the ctor via `matlab_obj_new` /
  `matlab_*_new`; **all** property access goes through `matlab_obj_get/set_mat`;
  native fields are dropped. `this`→handle is trivially consistent; handle
  identity/aliasing fall out for free; supports any property type.
  Cost: every scalar-double property access becomes a runtime call (slower, less
  readable C++), and the wrapper needs copy/lifetime glue so the handle isn't
  double-freed or dangled.
- **(B) Typed native fields + native access.** Emit properly typed C++ fields
  (`double` / `Matrix` / `std::string`) and rewrite property access to member
  access (`this->Name = ...`), reserving the generic-object runtime only for
  genuinely dynamic/struct-like classes. Fast, readable. Cost: must infer each
  property's concrete type, handle handle-aliasing explicitly (a value object
  copies fields — wrong for `< handle`), and re-implement whatever runtime
  behavior the generic path provided (validation, dynamic fields).
- **(C) Hybrid.** Per class: if every property is a statically-typed scalar/Matrix
  and the class is not `< handle`, use (B); otherwise use (A). Best output for the
  common case, correct for the hard case. Cost: two code paths to maintain.

Lean: **(A) for `< handle` classes** (sim3d, and any toolbox class backed by a C
runtime — these are exactly the failing cases and need shared identity anyway),
keeping the existing fast native-field path only for non-handle scalar-double
value classes. This is effectively (C) but with the split drawn on
`< handle` + "uses the generic-object runtime", which is what the failure
correlates with. Final pick to be confirmed in task 1 after auditing how many
existing passing classdef goldens rely on native fields.

**D2 — Lifetime / no-inline.** Handle-class values become named locals; the
emitter's `canInline` returns false for any value whose `classTypeOf` is a handle
class. Construction binds to a local that lives to end of scope; the runtime keeps
the handle valid for the program duration (handles are arena/refcount-managed by
the runtime, not stack-bound).

**D3 — Handle materialization at boundaries.** Where a method parameter or runtime
call expects the bare handle, pass the object's handle accessor (a member like
`.h()` or an explicit handle field) rather than a blanket implicit `operator
void*()` — an implicit conversion is too broad (overload-resolution surprises) and
was observed to enable miscompiles. Method params that are class objects are typed
as the class and converted to the handle inside the body.

**D4 — Surface fixes folded in.** #412: suppress `Class() = default;` when a
0-arg ctor method exists; emit `Class name{};` for no-arg construction. #411:
covered by D3 (operands materialize to handles).

**D5 — Parity gate.** A `test/Differential/` fixture per target program; the lane
already diffs `-repl` vs compiled stdout and is in CI. Treat any diff as a failure.

## Risks / Trade-offs

- **Regressing currently-passing classdef goldens** → audit first (task 1); run the
  full `test/EmitC` + `test/Run` classdef set before/after; keep the native-field
  path for the classes that already work.
- **Handle lifetime / double-free** → rely on the runtime's existing object
  ownership; the wrapper stores a non-owning handle and does not free in its dtor
  unless the runtime model requires it. Decide ownership explicitly in task 2.
- **Performance of all-runtime property access (option A)** → acceptable for the
  toolbox/handle classes (not hot loops); keep native fields for value classes.
- **Implicit-conversion footguns** → avoided by D3 (explicit handle accessor, no
  blanket `operator void*`).
- **Scope creep into CST value classes (`ss` arithmetic)** → `ss::operator+`
  only needs handle materialization (D3) to compile; full CST model-object
  arithmetic correctness is validated by its own examples, not expanded here.

## Migration Plan

Additive within the C++ emitter; no on-disk format or API change. Rollout: land
behind the existing emit-cpp path (no flag needed — it replaces broken behavior).
Rollback = revert the EmitC.cpp commit; interpreter and other lanes untouched
throughout.

## Open Questions

- Final D1 pick (A vs C) — resolve after the task-1 audit of how many passing
  goldens depend on native double fields.
- Does any emitted handle wrapper need a destructor (runtime-owned vs
  wrapper-owned handle)? Confirm against `matlab_obj_new` ownership.
- Do CST value classes (`ss`/`tf`/`zpk`) also need the handle model, or only
  handle materialization at call boundaries? (#411 needs only the latter.)
