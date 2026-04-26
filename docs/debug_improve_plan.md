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

## (2) Multi-file breakpoints — **shipped** (function-only siblings)

`compileProgram` in `tools/matlabc/main.cpp` walks the entry-point's
directory for sibling `.m` files. Each sibling is lexed + parsed
into its own `TranslationUnit` and, **only if** its `ScriptNode` is
empty (function-only or classdef-only files), its `Functions` and
`Classes` are merged into the main TU. The "skip script-bodied
siblings" rule prevents stitching unrelated executable code into the
launch — the DAP-test corpus has many `*.m` fixtures that are
themselves entry points and would conflict otherwise.

Each loaded file gets a distinct `SourceManager` FileID, gets
registered with the runtime via the existing
`matlab_dbg_register_file` walk, and lands in `G.PathToFileId`. An
IDE-supplied breakpoint path on `helper.m:5` now resolves through the
table; the JIT'd helper's hooks carry the right file_id (the
parser stamps each token with its source FileID, which the lowering
hook uses verbatim), so the breakpoint fires.

Sibling load order is **deterministic** (alphabetical sort of the
directory listing) so file_id assignment is reproducible across runs
— useful when comparing DAP traces.

Validated by `scn_multifile_breakpoint` in
`test/Debug/dap_scenarios.py`, which uses the
`test/Debug/multifile/` fixture (a `dap_main.m` that calls a
`helper_fn` defined in `dap_helper.m`) and verifies:
- the helper-file breakpoint comes back `verified=true`
- the stop event reports the helper file's path and line
- `stackTrace` shows `helper_fn` over `<script>` with each frame's
  source path correctly attributed
- function-frame Locals work for the helper (parameter spill +
  intermediate compute)

### Out of scope for this commit

- **Script-bodied helpers**: a sibling that mixes top-level
  statements with local-function definitions still won't have its
  helpers visible to the entry point. Real MATLAB's path-based
  resolution is more nuanced; this commit covers the common
  function-file pattern.
- **Cross-directory imports**: only the entry point's own directory
  is walked. Adding subdirectories or a `path` config flag is the
  next obvious step.
- **Duplicate-symbol diagnostics across siblings**: if two sibling
  files define a function with the same name, Sema flags the second
  one. That's the right behavior but the diagnostic could be
  friendlier.

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

## (6) Frame-scoped `evaluate` — **shipped**

The DAP `evaluate` handler in `tools/matlabc/main.cpp` now accepts an
optional `frameId`. When it points at a non-script frame, the handler
bridges that frame's mini-workspace into `matlab_ws` for the
duration of the eval and reverses the bridge on the way out:

1. **Snapshot** the pre-existing `matlab_ws` entries whose names
   collide with the frame's mini-ws.
2. **Stamp** the frame's mini-ws entries into `matlab_ws`. Tracks
   stamped names + which were pre-existing.
3. **Run** `runReplInput` for `__matlab_dbg_eval = (<expr>);`.
4. **Read** the result + format via `formatVar`.
5. **Restore**: clear `__matlab_dbg_eval` (so it doesn't pile up
   across calls), clear stamped names that didn't pre-exist (via
   `matlab_ws_clear_one`), and re-set names that were pre-existing
   to their snapshot values.

Implementation is entirely in the DAP handler — no runtime or
lowering changes. The runtime already exposed
`matlab_ws_clear_one(name, len)` (used by MATLAB's `clear name` form)
which made the restore trivial.

### Known shadowing limitation

The REPL JIT resolves bare identifiers as builtin function references
when a name matches a MATLAB builtin (`sum`, `prod`, `max`, ...) —
even after stamping. So a function-frame local named `sum` won't
resolve through `evaluate`; the user gets a compile failure. The
fixture's helper variable was renamed from `sum` to `total` to avoid
the collision. A proper fix would teach the resolver to prefer
matlab_ws entries over builtins under ReplMode, which is a
self-contained Resolver change but bigger than this commit.

Validated by `scn_evaluate_in_frame` in
`test/Debug/dap_scenarios.py`, which paused inside `compute(a, b)`
verifies:
- `evaluate("a")` without frameId silently resolves to the runtime's
  default-empty value (NOT `3`)
- `evaluate("a", frameId=inner)` returns `"3"`; same for `b`,
  `total`, and arithmetic on them
- After the bridge fires, the script frame's Locals (which surface
  `matlab_ws + frame_locals[0]`) still don't show `a`, `b`, `total`
  — the restore took them back out
- Pre-existing script-scope vars (`seed`) survive untouched

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
| 2 | shipped (function-only siblings) | — |
| 3 | shipped | — |
| 4 | shipped | — |
| 5 | shipped | — |
| 6 | shipped | — |

All originally-tracked items have landed. The remaining DAP-side
follow-ups are smaller polish:

- **Resolver-prefers-ws-over-builtin** (gated by item 6's shadowing
  limitation): teach the resolver under ReplMode to look up
  `matlab_ws` before falling back to builtin function references, so
  a workspace variable named `sum` shadows the builtin in
  `evaluate`-context expressions. Self-contained Resolver change.
- **Cross-directory multi-file**: extend item 2's directory walk to
  also cover sub-directories, or accept a `path` configuration entry.
- **Script-bodied helpers**: real MATLAB resolves local helper
  functions inside script files when set as the entry point. Not
  pursued today.

These three would close the remaining DAP-relevant gaps; nothing on
the original plan is still open.
