# Debug Improvements Plan

A focused, actionable plan for the remaining gaps in the DAP server
called out in [`docs/debug.md`](debug.md). The companion file is
already accurate about what works today; this doc tracks what's *not*
shipped, in priority order, with the runtime / MLIR / DAP-server
changes each item needs and the validation criteria a follow-up PR
must meet.

The DAP server lives in `tools/matlabc/main.cpp`, the runtime debug
state lives in `runtime/matlab_runtime.c`, and the per-statement hook
injection lives in `lib/MLIR/Lowering.cpp`.

## Status legend

- **shipped** — already in tree and exercised by `debug-hook-tests` /
  `debug-dap-tests`
- **partial** — wired far enough to be useful but with known gaps
- **deferred** — design clear, just not yet implemented
- **gated** — blocked on infrastructure outside the DAP server itself

---

## (1) Stack frames for `stepIn` / `error()` backtrace — **shipped**

`matlabc -dap` injects `matlab_dbg_enter_frame(name, len)` at every
user-function entry and `matlab_dbg_leave_frame()` before each
`func.return`. `stackTrace` returns the live frame list with correct
file paths and call-site lines. `error()` snapshots the frame stack
inside `matlab_set_error_msg` (before the unwind pops anything) and
prints `error: <msg>\n  at <fn> (<file>:<line>)` lines to stderr via
`write(2)` when DebugMode is on. Frame names are heap-copied on
`enter_frame` so the runtime owns null-terminated copies — closes a
latent bug where `stackTrace` was reading past unterminated JIT
globals into adjacent constants.

Validated by `scn_error_backtrace` in `test/Debug/dap_scenarios.py`
and the multi-frame `stackTrace` covered indirectly through
`scn_stack_scope_variables`.

---

## (2) Multi-file breakpoints — **gated** on Sema cross-file work

The DAP server's path → file_id table (`G.PathToFileId`) and the
`setBreakpoints` resolver are fully wired (`tools/matlabc/main.cpp`
around `:1660`). What's missing is upstream: today only the
entry-point `.m` is loaded by `SourceManager::loadFile`. Cross-file
calls — `myscript.m` calling a function defined in `helper.m` — would
need Sema to discover sibling files at name-resolution time, parse
them, and register them with the runtime via
`matlab_dbg_register_file` so breakpoint paths resolve.

This is a non-trivial frontend change (file discovery policy,
duplicate-symbol diagnostics, transitive imports for classdef
hierarchies) and lives outside the DAP layer. Once it lands the DAP
side will Just Work; no further DAP changes needed.

Tracking note for the eventual implementer: when sibling files are
parsed, call `SM.loadFile(siblingPath)` and pass the resulting FileID
+ canonical path to `matlab_dbg_register_file` from
`tools/matlabc/main.cpp::compileProgram` (already pre-walks the
SourceManager entries — extending the loop is a one-line change).

---

## (3) `setVariable` — **shipped**, all kinds

`setVariable` no longer rejects matrices / strings / structs / cells.
The handler routes through the same REPL JIT pipeline that
conditional breakpoints use: it wraps the user's text as
`<name> = (<value>);` and invokes `runReplInput`. Anything the
parser + Sema accepts is fair game — scalar literals, matrix
literals (`[1 2; 3 4]`), strings, struct accessors, function calls.
The response renders the new value via `formatVar` so the IDE's
watch box shows what actually got stored (e.g. "2x2 double" for a
matrix).

Defense-in-depth: the variable name is validated as a plain
identifier before the wrap so a malformed `name = ");; system(...)"`
can't smuggle extra statements past the literal-text concatenation.
Compile errors (parse / sema) come back as `success=false` with a
clear message; the DAP connection stays open for the user to retry.

Validated by `scn_set_variable` (covers scalar, matrix-literal,
fresh-name, malformed-RHS, and non-identifier-name paths).

---

## (4) Conditional breakpoints / log points — **shipped**

`setBreakpoints` accepts `condition` and `logMessage` per the DAP
spec; capabilities are advertised as
`supportsConditionalBreakpoints=true` and `supportsLogPoints=true`.
Conditional bps run the expression through the REPL JIT against the
script-level workspace (`evalConditionInWorkspace`); zero result
silently resumes; non-zero fires a `stopped` event with
`reason="breakpoint"`; eval failure marks the condition disabled so
subsequent hits don't keep paying the JIT cost. Log points
interpolate `{name}` placeholders from the workspace and emit a
DAP `output` event in the `console` category, never pausing.

Validated by `scn_conditional_breakpoint` and `scn_log_point`.

**Known limit** — both modes only see the script-level workspace;
locals inside a user function and for-loop induction variables
aren't visible. See item (5) below.

---

## (5) Function-frame Locals + DAP `evaluate` — **shipped**

The runtime maintains a per-frame mini-workspace alongside
`matlab_dbg.frames[]`. The lowering injects a generic
`matlab_dbg_frame_set` builtin after every `matlab.store` to a named
slot when `DebugMode` is on; `LowerTensorOps` dispatches by the
operand's lowered type to either `matlab_dbg_frame_set_f64(name,
len, val)` or `matlab_dbg_frame_set_mat(name, len, ptr)` —
late-binding the variant lets the call carry its name + value
through scalar promotion without us having to pre-commit to a
type at lowering time. `matlab_dbg_enter_frame` was hoisted to fire
*before* the parameter spill loop so the spill-store mirrors land in
the new frame, not the caller's.

The DAP server's `scopes` returns one Locals scope per requested
frame — variablesReference encoded as `1000 + DAP_frame_id`. The
legacy ref `1` is preserved as an alias for the script-level
workspace so any IDE / test that hardcodes it keeps working. The
`variables` handler decodes the reference, maps DAP frame ids back to
the runtime's outermost-first `frames[]` index, and dispatches: the
script frame gets `matlab_ws + frame_locals[0]` merged (covers
loop-induction variables that go through slot stores rather than
ws_set); function frames return their per-frame mini-ws only.

`evaluate` is wired through the same REPL JIT pipeline conditional
breakpoints already use: the user expression is wrapped as
`__matlab_dbg_eval = (<expr>);`, run through `runReplInput`, then
re-read by name and formatted with `formatVar`. Capability advertised
as `supportsEvaluateForHovers=true`. v1 evaluates against the
script-level workspace plus the script frame's mini-ws — function
frame locals aren't yet bridged into `runReplInput`, so a watch
expression referencing `n` inside `fact(5)` won't resolve. That
bridge is the natural follow-up to this commit (see `(6)` below).

Validated by the new `scn_function_locals` and `scn_evaluate`
scenarios in `test/Debug/dap_scenarios.py`. The full debug suite is
9/9 green; the existing 7 scenarios continue to pass unchanged
because legacy ref `1` still maps to the script-level workspace view.

---

## (6) Frame-scoped `evaluate` — **deferred**

Now that `variables` can render any frame's locals, the obvious next
step is letting `evaluate` operate against the same per-frame slice.
Today the evaluator runs through `runReplInput`, which compiles
against the global `matlab_ws` — so a watch expression typed while
paused inside `compute(a, b)` can reference `seed` (script ws) but
NOT `a` / `b` / `sum` (function-frame mini-ws).

### Recommended approach

The cleanest path is to bridge the requested frame's mini-ws into
`matlab_ws` for the duration of the eval:

1. **Snapshot the script ws** so the bridge is reversible.
2. **Stamp the frame's mini-ws entries** into `matlab_ws` under their
   bare names (overwriting any same-named script var temporarily).
3. **Run `runReplInput`** for `__matlab_dbg_eval = (<expr>);`.
4. **Read the result**, then **restore the script ws** from the
   snapshot.

The snapshot/restore avoids leaking function-frame state into the
script's persistent workspace once the user resumes. Implementation
lives entirely in `tools/matlabc/main.cpp`'s `evaluate` handler;
neither the runtime nor the lowering needs to change.

Validation: extend `scn_evaluate` so that, while paused inside
`compute()`, `evaluate("a + b")` returns the expected `sum` value;
afterwards script-scope `seed` is still its pre-eval value.

---

## Out of scope (separately tracked, not on the roadmap)

- **`keyboard` as a nested REPL.** Needs the scoped-eval path from
  item (5) plus a bidirectional REPL pump from the paused worker.
  No design started.
- **DWARF line tables in `-emit-llvm`.** Useful when piping
  `.m → LLVM IR → native` via clang and stepping in lldb. We emit
  `FileLineColLoc` on every op but the `-emit-llvm` text output
  doesn't carry a `!DISubprogram` / `!DILocation` graph.
  Orthogonal to DAP.
- **Function breakpoints** (`setFunctionBreakpoints`). Capability
  advertised as `false`. No design.
- **Hit-count breakpoints / data breakpoints / instruction
  breakpoints.** None advertised, none wired.
- **Reverse debugging / step-back.** `supportsStepBack=false`.
  Explicitly out of scope.
- **Always-on frame instrumentation in non-debug builds.** Current
  `enter_frame` / `leave_frame` only fire when `-g` is on (or
  implicitly under `-dap`). A production crash from `error()` in a
  non-debug build still has no backtrace. Adding always-on frames
  is a steady-state cost decision separate from this roadmap.

## Ordering

| # | Status | Depends on |
|---|---|---|
| 1 | shipped | — |
| 2 | gated   | Sema cross-file work |
| 3 | shipped | — |
| 4 | shipped | — |
| 5 | shipped | — |
| 6 | deferred | (5) shipped — has the per-frame mini-ws to bridge |

The next actionable pickup is item (6) — small focused change
(snapshot/restore around `runReplInput` inside the `evaluate`
handler) that closes the last DAP-side gap on this list. Item (2)
waits on Sema regardless of DAP work.
