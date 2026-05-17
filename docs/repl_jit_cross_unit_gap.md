# matlabc REPL cross-unit JIT gap — investigation + fix (2026-05-17)

## Symptom (historical)

A function defined in REPL turn *N* and called in turn *N+1* used to
fail to JIT with:

```
loc("<repl:N+1>":...): error: cannot be converted to LLVM IR:
  missing `LLVMTranslationDialectInterface` registration for dialect
  for op: matlab.alloc | matlab.subscript | func.func
error: ExecutionEngine::create failed: could not convert to LLVM IR
```

The chart-tick functions emitted by `loadStateChart('foo.mflow')` were a
concrete bite from this gap — their demo driver ran (same TU as the
definition), but a follow-on direct call from a later REPL prompt
failed.

## Root cause (multi-layer)

### Layer 1 — REPL accumulator splits at block boundaries

`tools/matlabc/main.cpp:1838–1853`: the REPL line-by-line
accumulator submits the input as soon as `blockDepth(Toks) == 0`.
A `function ... end` block flips depth back to 0 right after `end`,
so the function definition is always submitted as its own TU. Any
follow-on statement (call site, `disp`, anything) lands in a separate
subsequent TU.

This means there is **no "same-TU function-def + call"** case in
practice — the REPL splits them by design. (Earlier wording in this
doc treated the same-TU case as a possible fix; that turned out to
be moot.)

### Layer 2 — Resolver routes the cross-turn name through the workspace

In REPL mode (`Resolver::ReplMode = true`, set in
`tools/matlabc/main.cpp:809`), the resolver assumes any name not in
the current TU's symbol table is a workspace variable
(`lib/Sema/Resolver.cpp:1511–1527`). A function defined in a previous
REPL turn is *not* in the current TU; the resolver auto-declares the
name as a `Var` and routes its read through
`matlab_ws_get_mat(name, len)`.

Confirmed via `MATLABC_REPL_DUMP=1` on the IR emitted for
`chart_tick(false)` in turn *N+1*: the call was lowered as a
workspace load + `matlab_subscript1_s(ptr, 0.0)` (array index, not
call).

### Layer 3 — `runLowerUserCalls` can't see the call as a call

`lib/MLIR/Passes/LowerUserCalls.cpp:467–472` walks `matlab.call` ops
and groups them by callee. Once the resolver has rewritten the call
site as a workspace load, there *is* no `matlab.call` op — the
linkage to the function definition is gone at the MLIR level. Type
monomorphisation, signature refinement, and call lowering all skip
this call site.

### Layer 4 — Workspace hook has no "Function" kind

`runtime/runtime_io.cpp::matlab_dbg_ws_kind` enumerates kinds 0–5
(scalar / mat / obj / string / u8-mat / i32-mat). There is no
`Function` kind. The resolver therefore can't distinguish a workspace
*variable* called `foo` from a previously-defined *function* called
`foo`. `replWorkspaceKindHook` returns `-1` (not in workspace) for
any function defined in an earlier turn — even though the function
WAS defined, it's invisible to the resolver.

## Fix shipped (2026-05-17)

Implemented as **user-defined function persistence through the
prelude system** in `tools/matlabc/main.cpp`. Touches only the REPL
codepath; static `-emit-*` lanes are unaffected.

### Pieces

1. **`g_ReplUserFunctions: std::map<std::string, std::string>`** —
   file-scope map keyed by function name, value is the function's
   source text exactly as the user typed it.

2. **Capture in `runReplInput`** — after `parseFile` returns a
   well-formed `TU`, walk `TU->Functions` and extract each top-level
   function's source via its `Range` offsets. Skip functions whose
   offsets fall past `Src.size()` (those came from a prelude
   prepend, not from the user's input).

3. **`buildReplPrelude` extension** —
   - Detect names redefined in the current input (`function ... = NAME(`)
     and exclude them from the prelude scan so the verifier doesn't
     see two `func.func @NAME`.
   - Scan the input for mentions of stashed function names. Add each
     match to a `Wanted` set.
   - Iterate to closure: for each function already in `Wanted`, scan
     its body for mentions of other stashed names and pull them in.
     Closes transitive cases like `quad → add`.
   - Append the source of every wanted function under a
     `% --- repl-user-fn NAME ---` banner.

4. **Skip JIT when no script** — if the parsed TU has only function
   definitions and no `Script`, return early before the
   `ExecutionEngine::create` step. The function-def-only turn has no
   work to do at JIT time (the source is already stashed for future
   turns), and trying to translate uncalled `none`-typed funcs would
   error noisily.

### What works now

```
>> function y = double_it(x)
     y = 2 * x;
   end
>> disp(double_it(5))
10
```

```
>> loadStateChart('traffic_light.mflow')
loadStateChart: emitted ... — demo driver ran above; the chart's
`<name>_tick` is now stashed in the REPL's user-function table and
can be called directly on subsequent turns.
>> [r, y, g] = traffic_light_tick(false);
>> disp(r); disp(y); disp(g);
1
0
0
```

Multi-function transitive references work (`quad` → `add` →
sub-calls), redefinitions replace cleanly, and function definitions
typed but never called are silent.

### Test coverage

`test/Repl/run_tests.sh` (registered as ctest `repl-tests`) — 4
cases:
- `cross_turn_user_fn` — define + call across turns.
- `transitive_user_fns` — function calling another function (both
  stashed in earlier turns).
- `redef_user_fn` — redefinition in a turn replaces the stash
  without colliding with the prelude.
- `uncalled_user_fn` — function defined but never called: no error.

The chart-side `flowchart-simulate-dap-chart-tests` and the
`loadStateChart` path itself implicitly exercise the same machinery.

## Out of scope / deferred

The fix is sufficient for **interactive function-then-call
workflows** (chart drives, user helper functions, prelude-style
script libraries the user builds up turn-by-turn). It does **not**:

- **Persist anything other than top-level Function defs** —
  workspace variables already persist via the existing
  `matlab_dbg_ws_*` API; class instances persist via the kind=2 obj
  hook in `buildReplPrelude`. Nested functions inside a
  user-defined function aren't separately stashed (they're part of
  the parent's source body and pulled in transitively when the
  parent is).
- **Compress / GC the stash** — `g_ReplUserFunctions` grows
  unboundedly for the lifetime of the REPL session. Re-defining a
  function overwrites its entry; there is no automatic eviction of
  truly unused entries. Acceptable for typical sessions; if a user
  pastes thousands of function definitions, peak memory grows with
  the total source size.
- **Cross-process persistence** — a fresh `matlabc -repl` invocation
  starts with an empty stash. Not a real REPL workflow concern; if
  needed, the user can `source('library.m')` (which re-defines each
  function and so re-populates the stash).
- **Recover from same-name collisions with classdef-prelude
  classes** — if a user defines `function y = tf(x)` then calls
  `tf(5)`, both the user-fn prelude *and* the CST classdef prelude
  (which provides `tf` as the transfer-function class) get appended,
  producing duplicate symbols. The collision will surface as an
  MLIR verifier error. Workaround: use a different function name.

## File touchpoints

- `tools/matlabc/main.cpp` — `g_ReplUserFunctions`, the capture pass
  in `runReplInput`, the prelude scan in `buildReplPrelude`, the
  no-script-skip-JIT branch, and the updated `loadStateChart`
  confirmation message.
- `test/Repl/run_tests.sh` — new smoke fixture (4 cases).
- `CMakeLists.txt` — `repl-tests` ctest target.
