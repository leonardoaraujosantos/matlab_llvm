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

## (7) DWARF in `-emit-llvm` — **shipped**

`matlabc -emit-llvm -g` now attaches a DWARF line-table graph to the
emitted LLVM IR so users compiling `.m → LLVM IR → native` via clang
get source-level stepping in lldb / gdb. Implementation in
`lib/MLIR/Passes/LowerToLLVMIR.cpp::attachDebugInfo`:

- After the conversion-to-LLVM-dialect pipeline (so we're operating
  on `llvm.func` ops, not the original `func.func`), walk every
  function and stamp it with a `LLVM::DISubprogramAttr` attached via
  `mlir::FusedLoc`. The MLIR-to-LLVM-IR translator
  (`mlir::translateModuleToLLVMIR`) reads that fused location and
  emits `!DISubprogram` metadata in the resulting IR, then threads
  `!DILocation` through every instruction whose location is a
  `FileLineColLoc` parented by that fused scope.
- One `LLVM::DICompileUnitAttr` per source file (with
  `LineTablesOnly` emission kind so we skip the heavier full DWARF
  type-graph emission), one `LLVM::DIFileAttr` per file path.
- The CU map is shared, so multi-file emit (item 2's sibling pre-load)
  produces one CU per source file.
- Emission is strictly opt-in: without `-g`, the output IR has no
  DWARF metadata at all (verified by the new `debug-dwarf-tests`
  ctest).

End-to-end smoke validation:

```bash
matlabc -emit-llvm -g foo.m > foo.ll
clang -g -c -x ir foo.ll -o foo.o
clang -g foo.o runtime/matlab_runtime.c -o foo
lldb foo
(lldb) breakpoint set --file foo.m --line 7
Breakpoint 1: where = foo`main + 88 at foo.m:7:1, address = 0x...
```

Validated by `debug-dwarf-tests` (asserts metadata presence with
`-g` and absence without it). The lldb-attach step itself isn't a
CTest — runtime-attach permissions vary by host (macOS in particular
needs codesign entitlements for non-self attach).

What's NOT in the DWARF graph: variable inspection
(`DW_TAG_variable`), full type info, inlined-function info. Variable
inspection is better served by `-dap`'s per-frame Locals; types and
inlining haven't been pursued because line-tables-only is what
enables source-level stepping for the typical user. Both are
extensible from here without re-architecting.

## (8) Class instances in Locals + Watch — **shipped**

The `kind=2` path in the runtime, lowering, and DAP server makes
user-classdef instances first-class citizens in the LOCALS panel
and the watch box. Before this commit a class instance read like
`<huge>x<huge> double` because the matrix formatter was reading
`matlab_obj` internal fields as rows / cols; now it reads
`1x1 ClassName` and the row expands into one child per stored
property.

What ships:

- **Runtime.** `matlab_dbg_local` now carries `kind=2` for object
  pointers; `matlab_dbg_frame_set_obj` is the new mirror entry.
  `matlab_struct` / `matlab_ws` gained `kind=2` storage via
  `matlab_ws_set_obj` so script-level class assignments stamp the
  workspace with the right kind. `matlab_dbg_register_class` /
  `_class_name` map class IDs back to printable names; the
  matlab_obj layout already carries `class_id` at the tail of the
  struct prefix, so the registry plus the obj pointer is all the
  formatter needs. Property introspection lands as
  `matlab_dbg_obj_field_count` / `_name` / `_kind` / `_f64` / `_ptr`.
- **Lowering.** Class-bound slots get a `matlab.class_id` integer
  attribute on their `matlab.alloc` op (set in `getOrCreateSlot`,
  the IsCtorObj output, and the IsSelfParam / IsClassParam input
  sites in `lowerFunction`). `emitStore` forwards it onto the
  `matlab_dbg_frame_set` builtin so `LowerTensorOps` can pick the
  obj variant. The script-level `matlab_ws_set_*` site routes
  class-bound assignments through `matlab_ws_set_obj`. `lowerScript`
  emits one `matlab_dbg_register_class` call per classdef in the
  TU at the top of the script body when DebugMode is on.
- **DAP.** `formatVar` handles `kind=2`. `variables` requests hand
  out a `variablesReference >= 100000` for each class-instance row
  and an `ObjRefs` registry maps the handle back to the
  `matlab_obj *` for child expansion. `evaluate` promotes
  `kind=1` results back to `kind=2` when the result pointer
  matches a currently-tracked obj pointer (works around the REPL
  JIT's fresh-Sema not knowing workspace bindings are
  class-typed).

Validated by `scn_class_instance_locals` against
`dap_class_program.m` (two distinct classes plus a subclass; a
mutator runs before the breakpoint to verify property mutation
shows up in the expanded view; both `variables` and `evaluate`
exercise the obj path).

### Known limit

`acc.Balance` typed into the watch box still goes through the
REPL's struct path because the fresh Sema for the eval snippet
doesn't pin `acc` to a class. For stored-f64 properties the
struct-prefix layout means the read happens to find the field
anyway; for Dependent properties (which need `get.<Name>`
dispatch) it returns 0. Workaround: expand the row in the LOCALS
panel — that path goes through obj introspection and shows every
materialised property correctly. Dependent properties would
require a separate pass to query them via the lowered
`get.<Name>__<Class>` function; not started.

## (9) Matrix expansion in Locals + Watch — **shipped**

Numeric matrices used to render as `RxC double` with no
expansion path: the LOCALS panel had no disclosure arrow and the
watch box gave the same shape summary for any matrix expression.
Now every kind=1 row carries a `variablesReference` so the IDE
can drill into the cells.

What ships:

- **Runtime.** `matlab_dbg_mat_get(m, i, j)` exposes 1-based
  element access without leaking the `matlab_mat` layout to the
  DAP server. Out-of-range indices and complex matrices return
  `0.0` so a malformed children request can't read past the
  buffer.
- **DAP server.** `MatRefBase = 200000` + a `MatRefs` vector mirror
  the obj-ref registry. Kind=1 rows in LOCALS / per-frame Locals /
  obj-property children / watch eval results all register the
  matrix and emit a mat-ref. `variables(ref >= MatRefBase)` walks
  the buffer row-major and emits children with `(i,j)` /
  `(i)` labels (the latter for row / column vectors). 1x1 matrices
  stay leaves and unbox to the scalar in the parent value.
  `MatExpandCap = 256` caps the children list with a `…` /
  `(truncated)` trailer so a 1000x1000 matrix can't blow up the
  wire payload.

Validated by `scn_matrix_expansion` against
`dap_matrix_program.m` (covers the 2x3 / 3x1 / 1x1 formatting
paths and the watch-result variant).

### Out of scope (for now)

- **Custom matrix-viewer request.** Editor panels that want a 2D
  grid in one response (instead of 256 child rows) need a
  dedicated handler — easy to layer over the same `MatRefs`
  registry, but no IDE in the tree consumes it yet.
- **Real complex matrices** (the `MATLAB_MAT_C_MAGIC`-tagged
  descriptor). `mat_get` short-circuits to 0 there; a parallel
  `_get_real` / `_get_imag` pair would feed a richer formatter.
- **3-D matrices** (`matlab_mat3`). Pattern is identical but
  needs an `(i,j,k)` labeller and a stricter cap.

---

## Out of scope (separately tracked, not on the roadmap)

The items below are tracked here so future contributors don't
re-discover them from scratch — each has a brief sketch of what
it'd take to implement and why it's deferred. Status: **none are
started**; the shipped items above (1–7) form the supported surface.

### `keyboard` as a nested REPL

MATLAB's `keyboard` statement pauses execution at the line where
it appears and drops into an interactive prompt with full access to
the surrounding scope. The scoped-eval bridge from item (6) is
already in place; the remaining pieces are:

- **A bidirectional REPL pump** driven from the paused worker
  thread: read user input over a channel, route through item (6)'s
  snapshot/stamp/restore bridge, print result, loop until `dbcont`.
  In `-dap` mode the input channel is the IDE's debug-console pane
  via repeated `evaluate(context="repl")` requests; in standalone
  mode a tty pump on stdin/stdout works.
- **A lowering recogniser** for the `keyboard` builtin — emit a
  call to a new `matlab_dbg_keyboard()` runtime entry that flips the
  pause state with a "nested REPL" flag the DAP server / standalone
  pump knows how to respond to.
- **`dbcont` / `dbquit` / `dbstack`** as REPL commands handled
  inside the pump rather than passed through to the JIT.

Effort estimate: half-day or so once item (6) is understood. The
dependency graph is fully unblocked.

### Function breakpoints (`setFunctionBreakpoints`)

DAP lets the IDE set breakpoints by *function name* rather than by
file:line. The runtime-side change is small:

- **Runtime**: extend `matlab_dbg_state` with a parallel
  `fn_bp_names[]` table keyed by interned name. The frame-tracking
  hook (`matlab_dbg_enter_frame`) already has the function name in
  hand — compare against the table and pause if matched.
- **DAP server**: add a `setFunctionBreakpoints` handler, populate
  the runtime table, flip `supportsFunctionBreakpoints` to `true`
  in `initialize`.

Why deferred: file:line breakpoints already cover the common case
(set bp on the function's first line). The IDE-facing benefit is
"breakpoint follows the function across renames" which is uncommon
in the matlab_llvm corpus. Easy to add later without disturbing
anything else.

### Hit-count breakpoints

DAP `setBreakpoints` accepts a `hitCondition` string (e.g. `"5"`,
`">10"`, `"%2 == 0"`) so the breakpoint only fires after N hits or
on every Nth hit. Implementation:

- **Runtime**: per-bp counter alongside the existing condition /
  log fields; increment on every hit; compare against the
  hit-condition spec before deciding whether to pause.
- **DAP server**: parse `hitCondition` (a small grammar:
  literal-int → `==`, `>N`/`>=N`/`<N`/`<=N` → comparison,
  `%N == K` → modulo). Pass through to the runtime via a new
  `matlab_dbg_add_breakpoint_ex2` API or extend the existing one.
- **Capability**: advertise `supportsHitConditionalBreakpoints=true`.

Useful for "stop on the 5th iteration of this loop" without
hand-rolling a conditional that references the loop counter
(especially since the conditional evaluator can't see function-frame
loop indices today). Small enough to land in an afternoon.

### Data breakpoints

DAP `setDataBreakpoints` fires when a *value* changes (e.g. "stop
when `x` is written to"). For a JIT'd MATLAB this is the most
expensive option to implement well:

- **Approach A (per-store check)**: at every `matlab_ws_set_*` and
  `matlab_dbg_frame_set_*` call, look up the name in a watched-data
  table; pause if matched. Adds cost to every store in DebugMode but
  is mechanically simple.
- **Approach B (page-protect)**: mprotect the workspace page and
  catch SIGSEGV; only feasible if the matrix descriptor's storage
  is page-aligned, which it isn't today.

Why deferred: Approach A's per-store cost compounds with the
existing DebugMode hook overhead; Approach B is a substantial
runtime overhaul. The user-facing value (stop when a variable
changes) is mostly covered by conditional breakpoints with the
right condition expression.

### Instruction breakpoints (`setInstructionBreakpoints`)

DAP can set breakpoints on raw machine-code addresses. This is the
debugger-disassembly view's "click to break here" affordance.
Useful for native binaries; nearly meaningless for the JIT path
where instruction addresses are ephemeral and re-emitted each
launch.

Could be implemented for the `-emit-llvm -g`-clang-native flow by
deferring entirely to lldb / gdb's own instruction-bp machinery
(they already support it via DWARF). For DAP under JIT: not
pursued.

### Reverse debugging / step-back

DAP advertises `supportsStepBack` for time-travel debuggers
(`rr`, `gdb` reverse-execution, etc.). Implementing this for a JIT
needs deterministic re-execution from a checkpoint — every external
side effect (`disp`, file I/O, `randn`, threading) has to be
journaled and replayable. That's a substantial runtime project of
its own.

Explicitly out of scope. `supportsStepBack=false` and there's no
plan to flip it.

### Always-on frame instrumentation

Today `matlab_dbg_enter_frame` / `_leave_frame` only fire when
DebugMode is on (i.e. when `-g` was passed at compile time). A
production crash from `error()` in a non-debug build therefore has
no backtrace — the runtime sees an empty frame stack and the
emitted message has no "at fn (file.m:line)" lines.

To get always-on backtraces:

- **Lowering**: emit `enter_frame` / `leave_frame` unconditionally
  (drop the `if (!DebugMode) return;` early-out).
- **Cost**: two function calls per user-function invocation, plus
  one heap-copy of the function name on entry. Measurable but
  small for typical workloads; uglier for hot inner-loop functions
  called millions of times.
- **Symbol export**: the runtime already exports these symbols
  (`-rdynamic` / `--export-dynamic`); no link-time change needed.

Why deferred: it's a steady-state cost decision rather than a
correctness fix, and the typical user only needs the backtrace in
debug builds. If we want it, the change is one-line in lowering
plus a small benchmark to pin the per-call overhead.

## Ordering

| # | Status | Depends on |
|---|---|---|
| 1 | shipped | — |
| 2 | shipped (function-only siblings) | — |
| 3 | shipped | — |
| 4 | shipped | — |
| 5 | shipped | — |
| 6 | shipped | — |
| 7 | shipped | — |

All originally-tracked items plus DWARF-in-`-emit-llvm` have landed.
The remaining DAP / native-debug follow-ups are smaller polish:

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
- **DWARF variables / type info** (item 7 extension): line tables
  cover stepping; adding `DW_TAG_variable` + a partial type graph
  would let lldb's `frame variable` show locals on the native side.
  Lower priority since `-dap` already covers variable inspection
  thoroughly.

These three would close the remaining DAP-relevant gaps; nothing on
the original plan is still open.
