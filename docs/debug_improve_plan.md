# Debug Improvements Plan

Historical record of the DAP / debug build-out, plus the small
remaining follow-ups list at the end. The original plan was
items (1–9) below; everything in that range is shipped. Several
items the plan flagged as "out of scope" — `keyboard`, function
breakpoints, hit-count breakpoints, data breakpoints, reverse
stepping, disassembly, per-thread frames, memory inspection —
shipped in subsequent rounds and are documented in the
"Beyond the original plan" section further down.

The companion file [`docs/debug.md`](debug.md) is the
user-facing reference for what works today. This doc focuses on
*how* each item was built (so future contributors can extend or
debug them) and what's still open.

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

## Beyond the original plan — shipped follow-ups

The original roadmap (items 1–9 above) is fully landed. Subsequent
DAP work added the items below, all shipped in the current tree.
This section is preserved as historical record so future readers
can see what each was, what shipped, and where to find the
implementation.

### `keyboard` builtin — **shipped**

MATLAB's `keyboard;` pauses execution at the call site and drops
into the IDE's REPL panel. The lowering recognises `keyboard` in
the `ExprStmt` dispatch (`lib/MLIR/Lowering.cpp`, alongside the
`who`/`whos`/`clear` arm) and emits
`matlab.call_builtin {callee="matlab_dbg_keyboard_hook"}`;
`LowerTensorOps` lowers that to a direct `llvm.call`. The runtime's
`matlab_dbg_keyboard_hook` snapshots the calling thread's frame
chain and blocks on the same condvar a real bp uses; the DAP
monitor surfaces stop reason `"entry"` so the IDE switches to the
REPL view. Capability: stop reason routes correctly via
`matlab_dbg_was_paused_from_keyboard`.

The standalone-REPL pump (non-DAP) is still missing — see the open
follow-ups list at the bottom of this file.

### Function breakpoints (`setFunctionBreakpoints`) — **shipped**

`setFunctionBreakpoints` resolves a name against `G.FunctionTable`
(populated at `compileProgram` time by an AST walk over the TU's
`Functions` + `ClassDef::Methods`) and pins a line breakpoint at
the body's first statement. Class methods are registered under
three keys (`MethodName`, `ClassName.MethodName`,
`ClassName/MethodName`) so any form the user types resolves.
`supportsFunctionBreakpoints` is now `true`.

### Hit-count breakpoints — **shipped**

`setBreakpoints.hitCondition` accepts `N`, `==N`, `>=N`, `>N`, and
`%N`. The runtime's bp table grew `hit_count[]`, `hit_target[]`,
and `hit_op[]` fields; the hook checks the gate before declaring a
pause so a `hitCondition: ">= 100"` skips the JIT cost of cond
eval for the first 99 hits. New runtime API
`matlab_dbg_add_breakpoint_ex2` carries the gate; the v1 `_ex`
form is a back-compat wrapper.
`supportsHitConditionalBreakpoints` is now `true`.

### Data breakpoints — **shipped** (read / write / readWrite)

Approach A from the original sketch: per-store check. The runtime
maintains a watch table (`wp_name[]`, `wp_scope[]`, `wp_id[]`,
`wp_access[]`); every `matlab_ws_set_*` and
`matlab_dbg_frame_set_*` call hits `matlab_dbg_watch_check`
followed by `matlab_dbg_watch_trip` if the name matches with a
write-compatible access kind. Read watchpoints work too:
`matlab_ws_get_f64` / `_get_mat` call `matlab_ws_check_read_watch`
with a lock-free `n_wp == 0` fast path so the JIT pays no
measurable cost when no watches are armed.

Cost in practice is fine — the no-watch fast path is a single
relaxed load; the with-watches path adds a small linear scan that
only runs while the IDE has watches set. The page-protect approach
B was never pursued and isn't needed.
`supportsDataBreakpoints` is now `true`. Limitation: function-frame
reads bypass the runtime API entirely (the JIT loads from stack
slots), so a read-watch on a function local is silently invisible.

### Instruction breakpoints (`setInstructionBreakpoints`) — **deferred**

Still refused (`supportsInstructionBreakpoints=false`) — the JIT
exposes no public mapping from line to native PC, and the
disassemble path now ships uses the host triple's MCDisassembler
without needing one. Users who want byte-level breakpoints can
take the `-emit-llvm -g | clang | lldb` path which already
supports them through DWARF.

### Reverse debugging / step-back — **shipped** (per-statement undo log)

The runtime maintains a 4096-entry ring-buffer undo log. Every
`matlab_ws_set_*` and `matlab_dbg_frame_set_*` pushes a prev-value
record before the write; the hook stamps a statement boundary on
each fire. `matlab_dbg_step_back` walks the log backward from
`undo_head`, applies each non-boundary record in reverse until the
previous boundary, and refreshes the shared `frames[]` snapshot
from the paused thread's chain. The DAP `stepBack` and
`reverseContinue` handlers drive this.
`supportsStepBack` is now `true`.

v1 caveats (documented in `docs/debug.md`): per-statement
granularity (not per-instruction); rewinding a write where the
variable didn't pre-exist sets it to 0 (no remove-from-struct
API); irreversible-op markers are wired into the runtime but
`disp`/`fprintf` don't yet stamp them.

### Disassembly (`disassemble`) — **shipped**

Walks JIT-emitted machine code instruction-by-instruction using
the host triple's `MCDisassembler`. The DAP server caches the
`(target, MCAsmInfo, MCRegisterInfo, MCInstrInfo, MCSubtargetInfo,
MCContext, MCDisassembler, MCInstPrinter)` stack on first use —
`InitializeNativeTargetDisassembler` is deferred to first-request
to sidestep a static-init clash with MLIR's target registration on
some LLVM builds. Default base address is `Engine->lookup("main")`
cached as `G.MainAddr`. `supportsDisassembleRequest` is now
`true`.

### Per-thread frame chains for parfor — **shipped**

Each pthread that calls into the debug runtime gets its own
`thread_frames[i][]`, `thread_n_frames[i]`,
`thread_frame_locals[i][]`, and `thread_step_target_depth[i]`.
Concurrent parfor bodies enter/leave their own stacks instead of
trampling a shared `n_frames`. The legacy single-threaded shared
`frames[]` is now a snapshot of whichever thread last paused —
the hook refreshes it on pause so DAP inspectors that read those
arrays directly continue to work without per-thread refactoring.

### Memory inspection on matrices — **shipped**

Matrix variable rows carry a `memoryReference` (hex-formatted
data-buffer pointer); the DAP server's `MemRegions` registry
records `(ptr, byte_count)` for every buffer it hands out so
`readMemory` / `writeMemory` can validate against a known
buffer. Reads past the buffer end report `unreadableBytes`
instead of erroring. 1MB read cap per request.
`supportsReadMemoryRequest` and `supportsWriteMemoryRequest` are
now `true`.

### Always-on frame instrumentation — **still deferred**

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

## Open follow-ups

Items still on the backlog that don't fit a single feature heading:

- **Standalone-REPL `keyboard` pump** — the DAP path drops into
  the IDE's REPL panel, but `matlabc -repl` outside DAP doesn't
  have a bidirectional pump for nested REPL sessions. Would need
  the line-editor loop to interact with the worker's pause state.
- **Irreversible-op markers in `disp` / `fprintf`** — the
  reverse-stepping infrastructure has a kind=4 marker for ops
  that can't be undone (e.g. printed output). The runtime API
  `matlab_dbg_undo_record_irreversible` exists; `disp`/`fprintf`
  don't yet call it, so stepBack will silently rewind past
  printed lines. Wiring the call sites is mechanical.
- **Read watchpoints on function locals** — function-frame reads
  bypass the runtime API entirely (the JIT loads from stack
  slots), so a `read` watch on a function local is silently
  invisible. Matching the write-side coverage would need the
  lowering to emit a `matlab_dbg_frame_local_get` mirror call
  alongside every read.
- **`locations` / `setInstructionBreakpoints`** — would need a
  PC -> .m line table the JIT path doesn't maintain. Refused
  cleanly; the `-emit-llvm -g | clang | lldb` path covers users
  who need this.
- **`restartFrame` / `goto` / `gotoTargets`** — the runtime
  doesn't snapshot per-frame workspace at function entry, and
  the JIT exposes no in-frame PC manipulation primitive. Refused
  cleanly.
- **Parfor: per-thread `stackTrace` content** — per-thread frame
  chains are now correct internally, but the DAP `stackTrace`
  inspector still reads the snapshot of the *paused* thread. A
  `stackTrace(threadId)` query for a different (running) thread
  won't return that thread's stack. Useful only when the user has
  multiple parfor workers that pause concurrently.

## Status table

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Stack frames for `stepIn` / `error()` backtrace | shipped | per-thread chains added later |
| 2 | Multi-file breakpoints | shipped | function-only / classdef-only siblings |
| 3 | `setVariable` | shipped | all kinds via REPL-JIT path |
| 4 | Conditional / log breakpoints | shipped | frame-bridged in later round |
| 5 | Function-frame Locals + DAP `evaluate` | shipped | |
| 6 | Frame-scoped `evaluate` | shipped | shared `FrameBridge` helper |
| 7 | DWARF in `-emit-llvm` | shipped | line-tables-only |
| 8 | Class instances in Locals + Watch | shipped | methods exposed too |
| 9 | Matrix expansion in Locals + Watch | shipped | complex / 3-D added later |

Beyond the original plan (separate sections above):

| Item | Status |
|---|---|
| `keyboard` builtin | shipped (DAP path); standalone REPL pump still open |
| Function breakpoints | shipped; class-method names round-trip |
| Hit-count breakpoints | shipped |
| Data breakpoints (read / write / readWrite) | shipped |
| Disassembly | shipped (host triple via `MCDisassembler`) |
| Per-thread frame chains for parfor | shipped |
| Reverse stepping (`stepBack` / `reverseContinue`) | shipped (per-statement undo log) |
| Memory inspection on matrix buffers | shipped |
| Always-on frame instrumentation | deferred |
| `locations` / `setInstructionBreakpoints` / `restartFrame` / `goto` | refused cleanly |

## Smaller polish still on the table

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
