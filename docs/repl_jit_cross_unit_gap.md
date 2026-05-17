# matlabc REPL cross-unit JIT gap — investigation (2026-05-17)

## Symptom

A function defined in REPL turn *N* and called in turn *N+1* fails to
JIT with:

```
loc("<repl:N+1>":...): error: cannot be converted to LLVM IR:
  missing `LLVMTranslationDialectInterface` registration for dialect
  for op: matlab.alloc | matlab.subscript | func.func
error: ExecutionEngine::create failed: could not convert to LLVM IR
```

Affects any user-defined function. The chart-tick functions emitted by
`loadStateChart('foo.mflow')` are a concrete bite from this gap —
their demo driver runs (same TU as the definition), but a follow-on
direct call fails.

## Root cause (multi-layer)

### Layer 1 — Resolver routes the name through the workspace

In REPL mode (`Resolver::ReplMode = true`, set in
`tools/matlabc/main.cpp:787`), the resolver assumes any name not in
the current TU's symbol table is a workspace variable
(`lib/Sema/Resolver.cpp:1511–1527`). A function defined in a previous
REPL turn is *not* in the current TU; the resolver auto-declares the
name as a `Var` and routes its read through
`matlab_ws_get_mat(name, len)`.

Confirmed via `MATLABC_REPL_DUMP=1` on the IR emitted for
`chart_tick(false)` in turn *N+1*: the call is lowered as a workspace
load + `matlab_subscript1_s(ptr, 0.0)` (array index, not call).

### Layer 2 — `runLowerUserCalls` can't see the call as a call

`lib/MLIR/Passes/LowerUserCalls.cpp:467–472` walks `matlab.call` ops
and groups them by callee. Once the resolver has rewritten the call
site as a workspace load, there *is* no `matlab.call` op — the linkage
to the function definition is gone at the MLIR level. Type
monomorphisation, signature refinement, and call lowering all skip
this call site.

### Layer 3 — Workspace hook has no "Function" kind

The workspace introspection in
`runtime/runtime_io.cpp::matlab_dbg_ws_kind` enumerates kinds 0–5
(scalar / mat / obj / string / u8-mat / i32-mat). There is no
`Function` kind. The resolver therefore can't distinguish a workspace
*variable* called `foo` from a previously-defined *function* called
`foo`. `replWorkspaceKindHook` returns the same `-1` (not in
workspace) for any function defined in an earlier turn.

### Layer 4 — Same-TU forward case is also broken

A degenerate case appears even in a single REPL input:

```matlab
function y = double_it(x)
  y = 2 * x;
end
z = double_it(5);
disp(z)
```

The function is *defined* with `none`-typed args (because no call site
forces refinement at Sema time) *and* the call site is still routed
through the workspace. Static `-emit-c` on the function-after-script
form of the same code refines `x` to `f64` via
`runMonomorphiseUserCalls`. The REPL pipeline runs the same
monomorphisation pass but it has nothing to bite on, since the call
was already turned into a workspace load.

## Why this is bigger than one session

Closing the gap end-to-end requires coordinated changes in three
subsystems:

1. **Runtime / workspace** — add a `Function` kind (e.g. `6`) +
   storage for JIT-compiled function symbols + an introspection API
   so the next REPL turn can re-bind them. Each REPL turn currently
   builds + tears down a fresh `mlir::ExecutionEngine`; persisting
   user-defined function symbols across engines means either keeping
   the engine alive across turns (and re-using its module) or
   serialising the LLVM IR + reloading it.

2. **Resolver** — special-case "name resolves to a `Function` binding
   from the workspace" so the call site stays a `matlab.call`
   targeting the (now-symbol-registered) function. Also fix the
   same-TU forward case (function definition + immediate use) so
   monomorphisation can refine arg types.

3. **MLIR pipeline** — when reusing a long-lived execution engine,
   either re-add already-JIT'd functions to subsequent modules as
   `func.func declare` symbols + link against the prior engine, or
   compile each REPL turn into a separate JIT-dylib and chain them
   so cross-symbol lookups work.

Effort estimate stays at the previously-flagged **~1 wk**. The
pieces interact tightly enough that a half-implementation either
breaks existing REPL workflows or regresses static-mode codegen.

## Workarounds today

- **Within a single REPL input**, define + call in the same turn. The
  demo driver embedded by `matlabc -emit-matlab` does this — that's
  why `loadStateChart` actually exercises a chart on first load.
- **Programmatic chart drives** → `-emit-c chart.mflow` + a sibling
  `driver.m` compiled together via the file lane. Static-mode
  pipelines have none of the REPL workspace indirection.
- **Live state introspection** → `-simulate --sim-dap` is the
  supported live-interactive surface, not the REPL.

These match the wording added to `loadStateChart`'s confirmation
message and the `runtime/stateflow_classdefs.m` deprecation header.

## Recommendation

Pull this into its own dedicated workstream when the REPL becomes a
priority for cross-turn programmatic flows. Until then, the docs
correctly point users at the static / DAP paths and the gap is honest.
