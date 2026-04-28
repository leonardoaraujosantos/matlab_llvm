# Debugging matlab_llvm programs

A tour of the debugging aids shipped today. The baseline (`dbg()`,
REPL workspace commands, opt-in `#line`-annotated C output via
`-emit-c -line`) composes with the full Debug Adapter Protocol server
(`matlabc -dap`) so you can stay in an editor when a print doesn't
cut it.

## Quick tools for "what's going on at this line?"

### `dbg(x)` / `dbg(x, 'label')`

Source-located debug print to stderr. Works anywhere in compiled or
REPL code:

```matlab
>> A = [1 2; 3 4]
>> dbg(A)
<repl:1>:1: A = [2x2]
            1          2
            3          4
>> dbg(A * 3, "scaled")
<repl:2>:1: scaled = [2x2]
            3          6
            9         12
```

`dbg` prints the source file and line the call came from, the label
(the variable's name when the argument is a bare NameExpr, or an
explicit second-argument string, or `<expr>`), the shape, and for
matrices up to 8×8 of content. Scalars print as `file:line: name =
value`.

Writes go to stderr so `matlab.disp` output on stdout stays uncluttered
when you're debugging a script's computation.

### Workspace introspection in the REPL

`who` lists the names of every variable currently in the REPL
workspace. `whos` adds size and class columns:

```
>> x = 42
>> A = [1 2; 3 4]
>> whos
  Name             Size             Class
  x                1x1              double
  A                2x2              double
```

`clear x` removes a single variable; `clear` with no arguments wipes
the whole workspace. Both work in command syntax (`clear x`) and
function syntax (`clear('x')`).

### `#line` directives in emitted C / C++

The `-emit-c` and `-emit-cpp` backends can annotate each emitted
statement with a `#line "src.m"` directive. gdb and lldb pick these
up automatically when stepping through the compiled C / C++ code, so
the debugger shows your `.m` source rather than the generated C.
`#line` markers are off by default — pass `-line` to opt in:

```
$ matlabc -emit-c -line examples/factorial.m > /tmp/fact.c
$ cc -g /tmp/fact.c runtime/matlab_runtime.c -o /tmp/fact
$ lldb /tmp/fact
(lldb) breakpoint set -f factorial.m -l 9
```

### MLIR / LLVM IR dumps

Progressively lower-level introspection, in increasing "how did this
compile" order:

```
matlabc -dump-tokens file.m      # lexer output
matlabc -dump-ast    file.m      # parsed syntax tree
matlabc -emit-sema   file.m      # resolver + type inference
matlabc -emit-mir    file.m      # reference IR (in-house)
matlabc -emit-mlir   file.m      # mlir dialect (pre-passes)
matlabc -emit-mlir -opt file.m   # after slot-promotion + scalar-arith
matlabc -emit-mlir -g  file.m    # with matlab_dbg_hook(file_id, line)
                                 # injected at every statement (the same
                                 # IR shape -dap runs against, minus the
                                 # ReplMode workspace plumbing)
matlabc -emit-llvm   file.m      # final LLVM IR text
matlabc -emit-llvm -g file.m     # ... plus DWARF (!DICompileUnit /
                                 # !DISubprogram / !DILocation) so
                                 # clang+lldb step into the .m
matlabc -emit-c      file.m      # portable C (no #line by default;
                                 # add -line for lldb/gdb stepping)
matlabc -emit-cpp    file.m      # portable C++ (classes preserved)
```

`-g` (alias `--debug-hooks`) is the same flag the test suite uses to
verify that every emitted hook lands on a real, executable source line.
Combine it with `-emit-mlir` to see exactly which statements get a
hook and which lines they report — handy when an editor stops on a
seemingly random row.

In the REPL, set `MATLABC_REPL_DUMP=1` to print the final MLIR of each
input before it's handed to the JIT — useful when a compile-time error
is surprising.

## DAP server (`matlabc -dap`)

Breakpoints, stepping, and variable inspection — speaking the Debug
Adapter Protocol over stdio. Drops into any editor that talks DAP
(VS Code's `debugpy`-style generic-debug extension, `nvim-dap`,
IntelliJ's "DAP runner", Emacs `dap-mode`, etc.).

### How the stack is wired

`matlabc -dap` compiles the target `.m` file with two special flags:

- **`ReplMode=true`** — top-level script variables route through
  `matlab_ws_get_*` / `matlab_ws_set_*` so they live in the same
  persistent workspace struct the REPL uses. The `Locals` scope
  in the debugger is a snapshot of that struct.
- **`DebugMode=true`** — every statement is prefixed in MLIR with
  a call to `matlab_dbg_hook(file_id, line)`. The runtime hook
  consults a breakpoint table and the current step mode; if it
  decides to pause, it blocks on a pthread condition variable
  waiting for a resume command from the DAP server.

The DAP server (main thread) reads JSON-RPC frames from stdin, and a
detached worker thread JIT-executes the target. Pauses surface as DAP
`stopped` events; `continue` / `next` / `stepIn` / `stepOut` write
a new action into the runtime and wake the worker.

### Launching from an editor

Any DAP client works. Minimal nvim-dap configuration:

```lua
local dap = require('dap')
dap.adapters['matlab'] = {
  type = 'executable',
  command = '/path/to/matlab_llvm/build/matlabc',
  args = { '-dap' },
}
dap.configurations.matlab = {
  {
    type = 'matlab',
    request = 'launch',
    name = 'Run current .m',
    program = '${file}',
    stopOnEntry = false,
  },
}
```

VS Code via a generic-DAP extension:

```json
{
  "type": "matlab",
  "request": "launch",
  "name": "matlabc -dap",
  "program": "${file}",
  "stopOnEntry": false,
  "adapter": {
    "command": "/path/to/matlab_llvm/build/matlabc",
    "args": ["-dap"]
  }
}
```

### What works

| Feature                         | DAP method                          |
| ------------------------------- | ----------------------------------- |
| Initialize handshake            | `initialize` + `initialized` event  |
| Launch a `.m` file              | `launch` (with `program`, `stopOnEntry`) |
| Attach to a running session     | `attach` (alias for `launch` — same single-process model) |
| Stop on entry                   | `stopOnEntry` option on `launch`    |
| Line breakpoints                | `setBreakpoints` — invalid lines (blank, comment-only) snap forward to the nearest executable row; each verified bp gets a stable `id` for `hitBreakpointIds` correlation, and a `message` field surfaces the snap or "no executable line at or after this row" for unresolvable picks |
| Conditional breakpoints         | `setBreakpoints` `condition` — evaluates against the innermost paused frame, so a bp inside `compute(a, b)` can use `a > 5` |
| Log points (no pause)           | `setBreakpoints` `logMessage` — `{name}` placeholders resolve against the innermost paused frame's mini-ws (same bridge as conditional bps) |
| Hit-count breakpoints           | `setBreakpoints` `hitCondition` — accepts `N`, `==N`, `>=N`, `>N`, `%N`. Skip-counter lives in the runtime's bp table so the JIT cost of cond eval is paid only once the gate passes |
| Data breakpoints (read / write / readWrite) | `dataBreakpointInfo` advertises all three access types and resolves a name to a stable dataId; `setDataBreakpoints` installs the watch. Write side: every `matlab_ws_set_*` and `matlab_dbg_frame_set_*` checks the watch list. Read side: `matlab_ws_get_f64` / `matlab_ws_get_mat` check too, with a lock-free n_wp==0 fast path so the JIT pays no measurable cost when no watches are armed. `stopped` events carry `reason: "data breakpoint"` and the watch's id in `hitBreakpointIds`. Limitation: function-frame reads bypass the runtime API (the JIT loads from stack slots directly), so a read-watch on a function local is silently invisible — script-scope reads only |
| Memory inspection on matrices   | Matrix variable rows carry a `memoryReference` (hex-formatted data-buffer pointer); `readMemory` and `writeMemory` decode it back, validate against a server-side region table (only matrix data buffers are exposed — refuses arbitrary addresses), and stream cell bytes as base64. Reads past the buffer end report `unreadableBytes` instead of erroring. 1MB read cap per request |
| Disassembly                     | `disassemble` walks JIT-emitted machine code instruction-by-instruction using the host triple's `MCDisassembler`. With no `memoryReference`, defaults to the JIT's `main` entry point. Each result row carries an address, raw bytes (hex), and printed asm. Decode failures emit a `.byte` recovery row and step forward 1 byte so a single bad byte doesn't collapse the response. Negative `instructionOffset` is refused (would need a backward-decoder for variable-length archs) |
| Reverse stepping                | `stepBack` and `reverseContinue` rewind through a per-statement undo log. Every `matlab_ws_set_*` and `matlab_dbg_frame_set_*` push prev-value records before the write; the hook stamps a statement boundary (with the calling thread's frame depth) on each fire. stepBack pops one statement's worth of records and reapplies them in reverse, **staying within the current function frame** — boundaries from deeper (callee) frames are skipped past, and walking off the front of the current function returns `reason: "entry"` rather than teleporting into the caller. The innermost frame's line is updated so DAP `stackTrace` reflects the rewound caret. Ring buffer holds 4096 records — enough for hundreds of statements at typical mutation rates |
| Function breakpoints            | `setFunctionBreakpoints` resolves a name against the compiled function table and pins a line bp at the body's first statement. Class methods are registered under `MethodName`, `ClassName.MethodName`, and `ClassName/MethodName` so any form the user types resolves |
| Breakpoint candidate lines      | `breakpointLocations` returns every bp-eligible line in a range (computed by an AST walker at compileProgram time) |
| Exception breakpoints           | `setExceptionBreakpoints` with the `error` filter — runtime pauses on the first hook fired after `matlab_set_error` |
| Resume                          | `continue` (also fires `continued` event) |
| Step over / in / out            | `next` / `stepIn` / `stepOut` (stops report `reason="step"`, also fire `continued` events) |
| Step-into-targets               | `stepInTargets` returns one target (MATLAB call sites have at most one user-defined call per statement) |
| Pause a running program         | `pause` (breaks at the next stmt)   |
| Stack trace across user fns     | `stackTrace`                        |
| `stopped` correlation           | `stopped` event includes `hitBreakpointIds` when the pause came from a matched bp — same id space as `setBreakpoints` / `setFunctionBreakpoints` responses |
| Matrix grid hint                | `variables` / `evaluate` rows backed by a multi-cell matrix carry `indexedVariables` (cell count) so matrix-viewer panels can render an MxN grid widget without paging children |
| Property table hint             | Class-instance rows carry `namedVariables` (property count) for the same reason |
| Stderr forwarded                | `output` event (category `stderr`) — Diag prints from REPL eval, error()-tracebacks, and any other write to fd 2 surface in the IDE's debug console with stderr styling. Tee'd to the original stderr fd so subprocess capture / CI logs still see them |
| Watch error inline              | When `evaluate` fails, the response `message` carries the captured `<file>:<line>:<col>: error: <msg>` diagnostic instead of a generic placeholder — same bytes also forwarded to the debug console for full multi-line context |
| `breakpoint` event on resolve   | `setBreakpoints` against a path the runtime hasn't loaded yet (request arrived before launch / compileProgram) returns `verified=false` and queues the bp; once the path registers, the queued bp is replayed and a `breakpoint` event with `reason: "changed"` carries the now-verified state |
| List threads                    | `threads` enumerates the runtime's lazy-registered thread table — main script worker as `id=1` ("main") plus one entry per spawned parfor worker (`id=2..N`, "parfor-1" / "parfor-2" / ...). Stop events carry the originating thread's id. `thread` events fire on the main worker's `started` / `exited` |
| Per-frame Locals                | `scopes(frameId)` returns one Locals scope per frame; `variables(ref)` reads either the script ws or that frame's mini-ws |
| Workspace variables snapshot    | `variables`                         |
| Class instances in Locals/Watch | `1x1 ClassName` rows expand into properties **and methods**. Methods render with `presentationHint.kind="method"` (function-icon glyph) and a `@name(args)` value-column signature; methods inherited from a superclass carry an `(inherited from X)` suffix. Override (same-name method on derived class) suppresses the parent entry |
| Matrix expansion in Locals/Watch | `RxC double` rows expand into one child per cell (`(i,j)` row-major; linear `(i)` for vectors); `1x1` matrices unbox to the scalar. Complex matrices show as `RxC complex` and expand to one row per cell with value rendered as `re+im*i`; `1x1` complex unboxes to `re+im*i`. 3-D arrays show as `MxNxP double` and expand with `(i,j,k)` labels in slice-major order |
| `keyboard` builtin              | A `keyboard;` call in user code pauses the worker — same machinery as a breakpoint, but triggered from the program. The `stopped` event carries `reason: "entry"` so the IDE switches to the REPL view; resumption proceeds normally with the workspace intact |
| Watch / hover / debug-console eval | `evaluate` against any frame's locals — pass `frameId` for function-frame scope, omit for script scope; `context: "watch"` (default) wraps as an assignment, `context: "repl"` runs verbatim (see below) |
| REPL prompt against the live session | `evaluate` with `context: "repl"` runs `disp(T)` / `clear x` / `T(2,2) = 99` etc. against the paused session; output flows out as `stdout` `output` events. See "REPL with debug session" below |
| Multi-file breakpoints          | Sibling `.m` files (function-only or classdef-only) in the entry-point's directory get auto-loaded; bps on their lines fire correctly |
| Loaded source list              | `loadedSources` returns every registered file; one `loadedSource` event per file is also emitted at `configurationDone` |
| Source content fetch            | `source` returns the file's bytes (used by IDEs without local fs access) |
| Completions for REPL/watch      | `completions` returns the union of workspace names + frame locals (when `frameId` is set) + a curated set of MATLAB builtins, filtered by the prefix |
| Mutate any workspace variable   | `setVariable` (any MATLAB expr on RHS — scalar, matrix, string, struct) |
| Mutate by lvalue expression     | `setExpression` — same REPL-JIT path as setVariable, but the LHS can be `s.field`, `A(i,j)`, etc. The response renders the *computed* value via a readback (so `x = 2 * 21` returns `value="42"`, not `"2 * 21"`), with `indexedVariables` / `namedVariables` hints for matrix and class-instance results |
| Exception inspection            | `exceptionInfo` returns the message + frame snapshot captured by `matlab_set_error_msg` before the unwind; survives across the failing call's return |
| Process / lifecycle events      | `process` (on launch), `thread:started` / `thread:exited`, `loadedSource:new`, `continued`, `stopped`, `output`, `exited`, `terminated` |
| Restart                         | `restart` resumes with `STOP`, sends a `terminated` event with `restart: true`, expects the client to re-launch |
| Modules pane                    | `modules` returns an empty list (single-engine JIT host has no shared-library concept) |
| Graceful terminate              | `terminate` / `terminateThreads` stops the worker and emits `terminated`; the DAP loop stays alive so the IDE may follow up with `restart` or `disconnect` |
| Forceful detach                 | `disconnect` stops the worker and exits the DAP server loop |
| Program stdout forwarded        | `output` event (category `stdout`)  |

#### Explicit refusals

These requests respond `success=false` with a precise message
explaining the missing infrastructure, instead of falling through to
the unknown-handler silent-success. The IDE knows up front what
isn't available, and `initialize` advertises `supportsX=false` for
each so clients suppress the corresponding UI affordances.

| Request                          | Reason                                                              |
| -------------------------------- | ------------------------------------------------------------------- |
| `locations`                      | No PC -> .m source mapping is maintained for JIT'd code; the `-emit-llvm -g \| clang \| lldb` path covers native-level debugging where source lines round-trip through DWARF |
| `setInstructionBreakpoints`      | The JIT exposes no public mapping from line to native PC            |
| `restartFrame`                   | The runtime does not snapshot per-frame workspace at function entry |
| `goto`, `gotoTargets`            | The JIT exposes no in-frame PC manipulation primitive               |

### Architecture notes

- **Output routing.** The JIT'd program's stdout is piped to a
  reader thread that forwards each chunk as a DAP `output` event;
  we hang on to the original `STDOUT_FILENO` for DAP frames. Without
  this, `disp()` from the program would splice into the JSON-RPC
  stream and corrupt the channel. Stderr gets the same treatment
  with two differences: (a) the reader emits `category: "stderr"`
  events (drives the IDE's error styling), and (b) the bytes are
  *also* tee'd to the original stderr fd so `subprocess.stderr`
  capture and CI logs keep working — stdout is JIT-owned, stderr
  is shared.
- **Pause signalling.** The runtime broadcasts a condvar when it
  transitions to "paused"; the server's monitor thread wakes, emits
  a `stopped` event, and blocks until the client resumes. A small
  watcher thread also polls `matlab_dbg_is_paused()` every 20 ms as
  a belt-and-braces wake-up — the runtime only broadcasts on state
  transitions, so polling catches any edge where the monitor was
  between waits. Well below human perception for stepping latency.
- **Variable formatting.** Scalars print as their `%g` form. Matrices
  print as `RxC double` — we don't dump full contents for large
  matrices in the `variables` response since VS Code's watch UI
  doesn't page cleanly. Use `dbg(M)` in the source if you need
  content.
- **Hook line normalization.** The `matlab_dbg_hook(file_id, line)`
  call injected at the start of each statement is anchored to the
  first non-blank, non-comment-only line within that statement's
  source range. The walk is bounded by the statement's end line so it
  can never overshoot into the next statement — but it does mean
  stepping reliably lands on a row that contains code, never on a
  blank separator or a `% ...` comment line. Implementation:
  `lib/MLIR/Lowering.cpp::lowerStmt`.
- **Stop reasons.** When the runtime pauses, the DAP server inspects
  `matlab_dbg_get_pause_bp()` to distinguish a real breakpoint match
  (returns `>= 0`) from a step / `pause` (returns `-1`). The `stopped`
  event's `reason` field is `"breakpoint"` or `"step"` accordingly,
  matching the DAP spec so the IDE renders the correct icon.
- **Lifecycle events.** `configurationDone` is the trigger for the
  full lifecycle handshake: the worker thread is spawned and we
  emit a `process` event (debugger identity), a `thread:started`
  event, and one `loadedSource:new` event per file the
  SourceManager registered. The monitor thread fires
  `thread:exited` + `exited` + `terminated` when the worker
  returns. Stepping requests (`continue` / `next` / `stepIn` /
  `stepOut`) all send a synchronous `continued` event after
  responding, so adapters that track stopped/continued symmetry
  stay in sync.
- **Worker-state safety gate on `evaluate`.** `runReplInput` shares
  `matlab_ws` with the JIT'd program, so evaluating while the
  worker is running races on the workspace and the JIT engine.
  The evaluate handler's first action is to refuse with
  `success=false` unless the worker is paused, pre-launch, or
  exited. This is a blanket safety check — applies to watch / hover
  / repl alike.
- **Server-side bp/function tables.** During `compileProgram`,
  before Sema runs, the server walks the parsed TU to populate
  `G.BpLocations` (per-file set of statement start lines) and
  `G.FunctionTable` (function name → (file_id, first body line)).
  These back the `breakpointLocations` and `setFunctionBreakpoints`
  responses without re-walking the AST per request.

### Conditional breakpoints and log points

`setBreakpoints` honours the optional `condition` and `logMessage`
fields; both are advertised as supported in the `initialize`
response (`supportsConditionalBreakpoints`, `supportsLogPoints`).

- **Conditional**: when the breakpoint matches, the DAP server
  evaluates the condition by piggybacking on the REPL JIT — the
  expression is wrapped as `__matlab_dbg_cond = (<expr>);` and run
  through the full Lex → Parse → Sema → MLIR → JIT pipeline against
  the persistent `matlab_ws` workspace. Non-zero result → pause;
  zero → silently resume; eval failure → log once and treat the
  condition as broken (subsequent hits skip the bp without
  re-running the JIT).

- **Log points**: emit a DAP `output` event without ever pausing.
  The template supports `{name}` placeholders that resolve to the
  matching workspace variable's printed form (1×1 matrices unbox
  to scalar). Bare identifiers only — anything more complex is
  passed through as the literal substring.

Both modes also bridge function-frame locals into the REPL JIT
when the bp fires inside a user function — a condition like
`a > 5` on a bp in `compute(a, b)` resolves `a` to the parameter,
not to a missing script-scope name. The bridge uses the same
`FrameBridge` helper the `evaluate` handler uses (stamp the
innermost frame's mini-ws into matlab_ws, run the eval, reverse).
Plan item (6) of `debug_improve_plan.md` is closed by this.

#### Exception breakpoints (`error` filter)

`initialize` advertises a single exception-breakpoint filter named
`error`. When the IDE enables it via `setExceptionBreakpoints`, the
DAP server flips `matlab_dbg.pause_on_error = 1`; the runtime hook
then pauses the worker on the first statement after `matlab_set_error`
fires. The user can inspect the failing frame's locals before
resuming.

This is not the same path as the printed `error()` traceback (that
prints to stderr and then unwinds) — the filter forces a *pause* so
the call site is still on the live frame stack. After resuming, the
flag goes back to its idle state; matlab_set_error keeps recording
the snapshot for `exceptionInfo` regardless of whether the filter
caused a pause.

### Per-frame Locals + `evaluate`

`Locals` for any frame in the call stack is rendered from a per-frame
mini-workspace the runtime maintains alongside `matlab_dbg.frames[]`.
The lowering's `emitStore` injects a `matlab_dbg_frame_set` builtin
after every store to a named slot when DebugMode is on; LowerTensorOps
dispatches by the operand's lowered type plus an optional
`matlab.class_id` attribute to one of three runtime entries:

- `matlab_dbg_frame_set_f64(name, len, val)` — scalar f64.
- `matlab_dbg_frame_set_mat(name, len, ptr)` — `matlab_mat *`.
- `matlab_dbg_frame_set_obj(name, len, ptr)` — `matlab_obj *` (user
  classdef instance). The slot's `matlab.alloc` op carries
  `matlab.class_id` whenever the binding is pinned to a classdef
  (set in `getOrCreateSlot` and the explicit class-slot sites in
  `lowerFunction`); `emitStore` forwards the attribute onto the
  `matlab_dbg_frame_set` call so `LowerTensorOps` can pick the obj
  variant.

The frame push (`matlab_dbg_enter_frame`) was hoisted to fire *before*
the parameter spill loop in `lowerFunction` so the spill-store mirrors
land in the new frame, not the caller's.

DAP-side: `scopes(frameId)` returns one Locals scope whose
`variablesReference` is `1000 + DAP_frame_id`. `variables(ref)`
decodes the reference, maps the DAP frame id back to the runtime's
outermost-first `frames[]` index, and dispatches:

- The **script frame** merges `matlab_ws` (REPL-mode top-level
  assignments) with `frame_locals[0]` (loop-induction variables and
  other slot-stored values that bypass `ws_set`).
- A **function frame** returns its own per-frame mini-ws. Names
  duplicated across `matlab_ws` and a frame's mini-ws de-dup with
  `matlab_ws` winning.

The legacy `variablesReference == 1` is preserved as an alias for
the script-level workspace view so older scenarios / IDEs that
hardcode it keep working.

`evaluate` runs the user expression through the same REPL JIT
pipeline conditional breakpoints use. The behaviour now branches on
the DAP `context` field:

- **`context: "watch"`** (default — also covers `"hover"` and the
  Variables panel's expanded watches): wrap as
  `__matlab_dbg_eval = (<expr>);`, run through `runReplInput`, then
  re-read by name and format with `formatVar`. The response carries
  the rendered `result` and (when applicable) a `variablesReference`
  for class-instance or matrix expansion. This path is value-shaped
  — statements like `disp(T)` that don't return a value can't be
  bound to the holder slot.

- **`context: "repl"`**: run the user's input verbatim (no wrap, no
  result readback) and return an empty `result`. Output flows out
  through the existing stdout-redirect pipe and surfaces as DAP
  `output` events with `category: "stdout"` — the IDE renders
  those in its REPL panel alongside the typed prompt. This path
  supports statements that don't have a value: `disp(T)`, `clear x`,
  `T(2,2) = 99`, `[u, s, v] = svd(A)`. Trailing `;` is preserved
  (in MATLAB it suppresses the implicit echo of an assignment's
  result).

Frame-scoped eval still applies on both paths: pass `frameId` and
the handler bridges that frame's mini-workspace into `matlab_ws`
for the duration of the eval, then reverses the bridge.

A worker-state gate fast-fails any evaluate that arrives while the
worker is mid-execution (`success=false`, message "evaluate is only
valid while the program is paused or has exited"). Pre-launch and
post-exit eval are both allowed — the workspace is empty pre-launch
and frozen post-exit, so concurrent access is impossible.

Capability advertised as `supportsEvaluateForHovers=true`. v1
evaluates against the script-level workspace plus the script
frame's mini-ws — function frame locals aren't yet visible to the
evaluator (see plan item (6) for the bridge). Malformed expressions
come back as `success=false`; the connection stays open.

### REPL with a debug session

The `context: "repl"` branch above is what makes the IDE's REPL
panel work *while attached to a debug session*: type code at the
prompt and have it evaluate against the paused worker's workspace,
not against an isolated `matlabc -repl` subprocess.

- **Statement execution.** `disp(A)`, `clear x`, `who`, `whos`,
  `tmp = 99`, multi-output destructures, function calls — anything
  the standalone REPL accepts. The input is appended with a `\n`
  (matching what the standalone REPL does when reading lines from
  stdin) so parser recovery on malformed input lands on a clean
  diagnostic instead of running off the end of the buffer.

- **Output routing.** Bytes written to `stdout` by the JIT'd code
  hit the stdout-redirect pipe set up at `runDap` startup
  (`dup2(pipe[1], STDOUT_FILENO)`); the reader thread forwards them
  as DAP `output` events. The REPL panel renders those events
  inline; the synchronous `evaluate` response carries an empty
  `result`. (We do *not* try to capture the bytes synchronously
  inside the eval handler — the reader thread already owns the
  pipe.)

- **Diagnostics.** `runReplInput` writes Diag messages to stderr,
  which is *not* piped through the DAP server. Parse / type / lower
  errors land on the matlabc process's stderr (visible in the IDE's
  adapter log, not in the REPL panel). Future work: redirect
  stderr too and forward as `stderr` `output` events.

- **Frame scope.** Pass `frameId` to evaluate against a function
  frame's locals; omit it to use the script workspace. Same bridge
  logic as the watch path.

- **Lifecycle.** REPL eval is allowed pre-launch (empty workspace),
  while paused (current frame), and post-exit (frozen final state).
  The "running, not paused" case is rejected because `runReplInput`
  shares `matlab_ws` with the running JIT.

### Class instances in Locals + Watch

`acc = BankAccount(...)` used to surface in the LOCALS panel as
`<huge>x<huge> double` because the runtime tracked only two kinds
(`f64` scalar and `matlab_mat *` matrix) and the matrix formatter
dereferenced the `matlab_obj *` as if it were a `matlab_mat *` —
reading internal pointer fields as rows / cols. Class instances now
flow through a dedicated `kind=2` path end-to-end:

1. **Lowering.** Slots whose binding is pinned to a user classdef
   carry a `matlab.class_id` integer attribute on their
   `matlab.alloc` op. `emitStore` forwards the attribute onto the
   `matlab_dbg_frame_set` builtin; `LowerTensorOps` reads it back
   and lowers to `matlab_dbg_frame_set_obj` (instead of `_set_mat`)
   so the runtime's per-frame Locals table records `kind=2` with
   the obj pointer borrowed from the slot. The script-level
   `matlab_ws_set_*` write site in the assignment lowerer routes
   class-bound assignments through the new `matlab_ws_set_obj`,
   stamping `kind=2` directly on the `matlab_struct` workspace
   entry.
2. **Class-name registry.** `lowerScript` emits one
   `matlab_dbg_register_class(class_id, "ClassName")` call per
   classdef in the translation unit (DebugMode only) at the very
   top of the script body, populating a small linear table inside
   `matlab_dbg`. The DAP server reads it via `matlab_dbg_class_name`
   to format `1x1 ClassName`. `matlab_obj` already carries `class_id`
   at the tail of its struct prefix, so the runtime resolves the
   name from the obj pointer alone.
3. **Property introspection.** New `matlab_dbg_obj_field_*`
   accessors expose the `matlab_obj`'s embedded `matlab_struct`
   (`names[]` / `kinds[]` / `f64_vals[]` / `ptr_vals[]`) so the DAP
   server can produce one child row per property when the IDE
   asks to expand a class-instance row.
4. **DAP plumbing.** `formatVar` handles `kind=2` directly. The
   `variables` request hands out a `variablesReference` ≥ 100000
   for each class-instance row, backed by a server-side registry
   (`ObjRefs`) that maps the handle back to the underlying
   `matlab_obj *`. Expansions read children via
   `matlab_dbg_obj_field_*`; properties that themselves hold a
   class instance recurse via the same registry.
5. **Watch promotion.** The REPL JIT compiling
   `__matlab_dbg_eval = (<expr>);` doesn't carry workspace class
   info into its fresh Sema, so an expression that yields a class
   instance lands in `matlab_ws` with `kind=1` (matlab_mat) — the
   pointer is correct but the kind tag is wrong. The `evaluate`
   handler compensates by sweeping every currently tracked
   `kind=2` pointer (across `matlab_ws` and every frame's mini-ws)
   and promoting the result to `kind=2` on a hit. Without the
   promotion the watch box would show `<huge>x<huge> double`
   again on a watched class instance.

A test scenario (`scn_class_instance_locals`) and a fixture
(`dap_class_program.m`) cover the full surface: two distinct
classes, an inherited subclass, a mutator (`acc.deposit`),
property expansion in both `variables` and `evaluate`, and the
`kind=1`→`kind=2` promotion via the watch box.

#### Known limit: dot-access on workspace class instances

Inside the watch box, `acc.Balance` evaluates against a
freshly-Sema'd REPL session that has no record of `acc`'s class.
The dot lookup falls back to `matlab_struct_get_f64`, which
correctly walks the obj's struct-compatible prefix — so for
properties that exist on the matlab_obj and were stored as f64,
the answer comes back. Properties that need `get.<Name>`
dispatch (Dependent properties) or class-method calls are *not*
resolved by the workspace evaluator yet — the user-facing watch
will return `0` rather than the computed value. Workaround:
expand the row in the LOCALS panel instead, which goes through
the obj-introspection path and shows every stored property
correctly (Dependent properties are still elided because they
aren't materialised in the obj's field table).

### Matrix expansion in Locals + Watch

`A 3x3 double` used to be the end of the line in the LOCALS panel:
clicking the row didn't reveal the cell values, the watch box gave
the same one-line summary for any matrix expression, and editor-side
"matrix viewer" panels had no DAP path to read element data. The
expansion path mirrors the class-instance path:

1. **Runtime.** `matlab_dbg_mat_get(matlab_mat *m, i, j)` returns
   the `(i, j)` cell using 1-based indexing (matches the labels the
   DAP server hands the IDE) so the server doesn't need access to
   the `matlab_mat` layout. Out-of-range indices and complex
   matrices return `0.0` defensively.
2. **DAP plumbing.** `MatRefBase = 200000` plus a `MatRefs` vector
   maps DAP variablesReferences to live `matlab_mat *` pointers. A
   kind=1 row (LOCALS, watch eval result, or an obj property
   holding a matrix) registers the pointer and ships its handle as
   the row's `variablesReference`; 1x1 matrices stay leaves and
   unbox to the scalar in the parent's `value` field. When
   `variables(ref)` arrives with `ref >= MatRefBase`,
   `appendMatChildren` walks the buffer in row-major order and
   emits one cell per child:
   - 1xN row vector → `(j)` linear labels
   - Mx1 col vector → `(i)` linear labels
   - MxN matrix → `(i,j)` two-dim labels
3. **Truncation.** `MatExpandCap = 256` keeps the response payload
   sane on large matrices; once the cap is hit a single `…` row
   with value `(truncated)` flags the elision so users know the
   IDE didn't quietly drop cells. A future Matrix Viewer protocol
   can request the full grid via a custom request without going
   through the truncated children path.

The watch path uses the same registry: an `evaluate` of `A * x`
lands as a kind=1 result in `matlab_ws`, and the response carries a
mat-ref so the IDE can drill into the product without re-typing it
into LOCALS.

Validated by `scn_matrix_expansion` against `dap_matrix_program.m`
(2x3 matrix, 3x1 column vector, 1x1 scalar — covers the three
formatting paths plus the watch-result variant).

#### Out of scope (for now)

- **Custom matrix-viewer request.** Editor panels that want a 2D
  grid in one shot (rather than 256 child rows) need a dedicated
  request like `matlab/matrix(ref)` returning `{rows, cols, data}`.
  Easy to layer on top of the existing registry — same handle
  works, different response shape — but no IDE in the tree
  consumes it yet so it isn't shipped.
- **Real complex matrices.** *Done.* The runtime exposes a kind
  discriminator (`matlab_dbg_mat_kind`) plus per-kind accessors —
  `matlab_dbg_mat_c_re/_im` for complex, `matlab_dbg_mat3_get` for
  3-D. The DAP server's `formatMatShape` and `appendMatChildren`
  dispatch on the kind: complex cells render as `re+im*i`, 3-D
  cells get `(i,j,k)` labels in slice-major order. 1×1 complex
  unboxes to `re+im*i` in the parent value column.

### `error()` backtrace

When DebugMode is on (`-dap` or `-g`-built binaries that call
`matlab_dbg_enable`), `error()` snapshots the runtime frame stack
inside `matlab_set_error_msg` *before* the unwind pops it, then emits
the diagnostic to stderr with one `at <fn> (<file>:<line>)` line per
frame:

```
error: boom
  at deeper (/path/to/script.m:17)
  at fail   (/path/to/script.m:13)
  at <script> (/path/to/script.m:9)
```

The print uses `write(2)` rather than `fprintf` so libc's stdio file
lock can't deadlock against MLIR's ExecutionEngine on shutdown. Frame
names are heap-copied on `matlab_dbg_enter_frame` so the runtime owns
null-terminated copies — fixes a latent bug where the JIT's read-only
name globals (sized exactly to the string, no trailing 0) would cause
`%s`-style readers to walk into adjacent constants.

In production (non-debug) builds `matlab_dbg.enabled` is false and the
print is suppressed — `error()` keeps its existing semantics (sets the
flag for try/catch, no stderr noise).

### `setVariable` for any RHS expression

The watch-box mutation path runs through the same REPL JIT that
conditional breakpoints use. The DAP server wraps the user's text as
`<name> = (<value>);` and runs it through Lex → Parse → Sema → MLIR →
JIT against the persistent workspace. Anything the parser accepts on
the RHS works: scalar literals, matrix literals (`[1 2; 3 4]`),
strings, struct accessors, function calls. The response renders the
new value via the same `formatVar` that the `variables` request uses,
so the IDE watch box shows `2x2 double` after a matrix set instead of
a stale scalar.

Compile errors come back as `success=false` with a clear message; the
DAP connection stays open. The variable name is validated as a plain
identifier before the wrap, so a malformed `name` like
`"x); system(...)"` can't smuggle extra statements past the literal
concatenation.

### Multi-file breakpoints

`compileProgram` walks the entry-point's directory for sibling `.m`
files, parses each independently, and merges any **function-only or
classdef-only** sibling's `Functions` / `Classes` into the main
`TranslationUnit`. Siblings with a script body are skipped (they're
treated as their own entry-point candidates). Each loaded file lands
in `SourceManager` with a fresh FileID and gets registered with the
runtime via `matlab_dbg_register_file`, so an IDE-supplied path on
`helper.m:5` resolves through `G.PathToFileId` and the breakpoint
fires when the JIT'd helper executes that line.

Sibling load order is alphabetically deterministic so file_id
assignment is reproducible across runs.

Phantom paths (a path the IDE knows about but no file is loaded for)
still come back with `verified=false` instead of crashing.

Out of scope today: cross-directory walks, script-bodied helpers,
and friendlier duplicate-symbol diagnostics. See
[`docs/debug_improve_plan.md`](debug_improve_plan.md) item 2 for the
follow-up policy options.

### Frame-scoped `evaluate`

`evaluate` accepts an optional `frameId`. When it points at a
non-script frame, the handler bridges that frame's mini-workspace
into `matlab_ws` for the duration of the eval and reverses the
bridge afterward — snapshot pre-existing entries, stamp the frame
locals on top, run `runReplInput`, restore. Stamped names that
didn't pre-exist get cleared via `matlab_ws_clear_one` so eval
doesn't leak function locals into the persistent script workspace.

The bridge logic is factored into a `FrameBridge` helper shared
across three sites: the `evaluate` handler (parameterised by
`frameId`), the conditional-bp evaluator, and the log-point
interpolator. The latter two always bridge the *innermost*
function frame so a bp inside `compute(a, b)` can use `a > 5` as
its condition. The script frame (rt index 0) needs no bridging —
its locals are already in matlab_ws / frame_locals[0].

Known shadowing limitation: the REPL JIT resolves bare identifiers
as builtin function references when the name matches a MATLAB
builtin (`sum`, `prod`, ...) — so a function-frame local named
after a builtin won't resolve through `evaluate`. The fixture's
helper variable is named `total` rather than `sum` for this reason.

### Other known limits (deferred, not blocked)

- **Reverse stepping / time-travel debugging.** *Done.* The
  runtime maintains a 4096-entry ring-buffer undo log: every
  `matlab_ws_set_*` and `matlab_dbg_frame_set_*` pushes a
  prev-value record before the write, and the hook stamps a
  statement boundary (carrying the thread's frame depth at
  stamp time) on each fire. `matlab_dbg_step_back` drops the
  head boundary (the current paused-statement marker), walks
  back applying each non-boundary record in reverse, and stops
  at the previous boundary *with the same frame depth* — so
  rewinding inside a function call stays within that function
  and skips past nested-callee boundary records that were
  stamped during the descent. Walking off the front of the
  current function returns `reason: "entry"` rather than
  silently teleporting up into the caller; the user can
  `continue` from there or use forward step-out semantics.
  The innermost frame's line is updated as part of the rewind
  so DAP `stackTrace` (which the IDE renders the caret from)
  reflects the new position. Variables that didn't pre-exist
  are removed via `matlab_struct_rmfield` so the rewound state
  matches the pre-write workspace exactly — no stale `x = 0`
  shadow. The DAP `stepBack` and `reverseContinue` handlers
  drive this. Limitations: per-statement granularity (not
  per-instruction); rewinding past the first statement returns
  `reason=entry` with `description: "stepBack: undo log
  exhausted"`. Irreversible ops can stamp a kind=4 marker that
  stops the rewind cleanly — the runtime API
  (`matlab_dbg_undo_record_irreversible`) exists, but `disp` /
  `fprintf` don't yet stamp markers, so stepBack will currently
  rewind past printed output silently. Wiring those call sites
  is follow-up.
- **Memory inspection on matrices.** *Done.* Matrix variable rows
  carry a `memoryReference` pointing at the data buffer; the DAP
  server keeps a `MemRegions` registry of (ptr, byte_count) pairs
  for every buffer it hands out. `readMemory` and `writeMemory`
  decode the hex pointer, validate against the registry to bound
  the I/O, and stream cell bytes as base64. Buffers we don't
  expose (the LLVM JIT image, complex matrices' parallel re/im
  buffers) stay opaque.
- **Disassembly.** *Done.* `disassemble` uses the host triple's
  `MCDisassembler` to walk JIT-emitted machine code instruction-
  by-instruction. The disassembler holder (target, MCInfo,
  MCRegisterInfo, MCInstrInfo, MCSubtargetInfo, MCContext,
  MCDisassembler, MCInstPrinter) is built lazily on first
  `disassemble` request — `InitializeNativeTargetDisassembler`
  is deferred to first-use so it doesn't clash with MLIR's
  startup target init. The default base address is
  `Engine->lookup("main")` cached as `G.MainAddr` in the worker.
  `locations` (PC -> source line) stays refused — we don't
  maintain a JIT line table; the `-emit-llvm -g | clang | lldb`
  path covers that need via DWARF.
- **Data breakpoints (read / write / readWrite).** *Done.* The
  runtime carries a per-name watch table with an `wp_access` byte
  per entry. Write side: every `matlab_ws_set_*` and
  `matlab_dbg_frame_set_*` calls `matlab_dbg_watch_check`/`_trip`
  after the write lands. Read side: `matlab_ws_get_f64` /
  `matlab_ws_get_mat` call `matlab_ws_check_read_watch` after the
  load, with a lock-free `n_wp == 0` fast path so the no-watch
  case has no mutex cost. Watch ids are djb2 hashes of the name
  (31-bit-truncated to stay clear of the line-bp id space) so
  they round-trip cleanly. Limitation: function-frame reads
  bypass the runtime API entirely (the JIT emits direct loads
  from stack slots), so a read-watch on a function local is
  silently invisible — read watchpoints work for script-scope
  variables only.
- **Parfor / multi-thread debugging.** *Done.* The runtime
  lazy-registers each pthread that calls into the debug API
  (`matlab_dbg_thread_slot_locked` runs on every
  `matlab_dbg_hook` entry) and assigns sequential ids: 1 = main
  worker, 2..N = parfor workers in spawn order. Each thread now
  owns its own frame chain (`thread_frames[i][]`,
  `thread_n_frames[i]`, `thread_frame_locals[i][]`,
  `thread_step_target_depth[i]`); concurrent parfor bodies
  enter/leave their own stacks without corrupting each other.
  When a thread pauses (line bp / data bp / keyboard / error),
  the hook snapshots that thread's chain into the legacy shared
  `frames[]` / `frame_locals[]` arrays so DAP inspectors that
  read those directly see the *paused* thread's stack — no
  inspector refactor needed. Step-target depth is also
  per-thread, so a step in worker A doesn't fire when worker B
  reaches its target depth. Capacity is 32 threads (table-full
  reuses slot 0 for overflow rather than refusing to track).
- **Instruction breakpoints.** Same root cause as memory — no
  byte-level addressing of the JIT image.
- **Restart-frame / goto.** Need per-frame workspace snapshots and
  in-frame PC manipulation respectively. Both refuse.
- **`disp(T)` in a watch box** *(fixed)*. The watch handler used to
  SIGSEGV when wrapping a void call (`__matlab_dbg_eval = (disp(T));`
  binds nothing). The handler now detects statement-shaped void
  calls (`disp`, `fprintf`, `error`, `warning`, `assert`, `clear`,
  `who`, `whos`, plotting calls, etc.) up front and routes them
  through the REPL branch — the watch row shows `<void>` and the
  side-effect output flows out as `output` events. False-positive
  cost is bounded: a watch on a void call shows `<void>` instead of
  the empty cell it used to show; false negatives would crash, so
  the detection list errs on the inclusive side.
- **Stderr → DAP `output`.** Compile / lower diagnostics from
  REPL-mode evaluate go to the matlabc process's stderr, not into
  the IDE's REPL panel. A second pipe redirect for stderr (mirror
  of the existing stdout one) would close this gap.

## Native debugging via `lldb` / `gdb`: DWARF in `-emit-llvm`

The DAP path is the right choice for IDE-driven debugging during
JIT execution. For users who compile `.m` → LLVM IR → native via
`clang` and want to step in `lldb` / `gdb` against the resulting
binary, `-emit-llvm -g` attaches a DWARF line-table graph to the IR:

- One `!DICompileUnit` per source file (`!DIFile` references the
  original `.m` filename + directory; emission kind is
  `LineTablesOnly` so we skip the heavier full DWARF type graph).
- One `!DISubprogram` per `llvm.func` (name, linkage name, file,
  line, scope-line — sufficient for `breakpoint set --file foo.m
  --line 5` to resolve to a binary address).
- One `!DILocation` per IR instruction whose MLIR location was a
  `FileLineColLoc`. The translator threads these automatically once
  the parent function carries a fused-location DISubprogram, which
  is the trick: we walk every `llvm.func` after the conversion-to-
  LLVM-dialect pipeline and stamp each one with a `DISubprogramAttr`
  attached via `FusedLoc`.

End-to-end:

```bash
matlabc -emit-llvm -g foo.m > foo.ll
clang -g -c -x ir foo.ll -o foo.o
clang -g foo.o runtime/matlab_runtime.c -o foo -lm -lpthread
lldb foo
(lldb) breakpoint set --file foo.m --line 7
Breakpoint 1: where = foo`main + 88 at foo.m:7:1, address = 0x...
```

Without `-g`, the `-emit-llvm` output has none of this metadata —
DWARF is strictly opt-in. The `-g` flag also enables the runtime
hook injection (same as for `-dap`); the hooks are dead calls in
this path (the runtime sees `matlab_dbg.enabled == 0` and returns
immediately) but cost a function call per statement. If you don't
need them, the `-emit-c -line` path is cheaper — `cc -g` reads
`#line` directives and produces equivalent line-table DWARF without
any runtime instrumentation.

What's NOT in the DWARF graph today: variable inspection
(`DW_TAG_variable`), full type info (struct / array shapes),
inlined-function info. Variable inspection is better served by
`-dap`'s per-frame Locals; types and inlining haven't been pursued
because the line-tables-only build is what enables source-level
stepping for the typical user. Both are extensible from here without
re-architecting.

The shape of the emitted metadata is verified by the
`debug-dwarf-tests` ctest (asserts `!DICompileUnit` /
`!DISubprogram` / `!DILocation` are present with `-g` and absent
without it). The lldb-attach path itself isn't a CTest because
runtime-attach permissions vary by host (macOS in particular
requires codesign entitlements for non-self attach).

### Tracing the wire

Every DAP client has a "trace the protocol to a file" toggle; that's
the fastest way to debug an editor integration. A minimal manual
exchange looks like:

```
-> {"seq":1,"type":"request","command":"initialize", ...}
<- {"seq":1,"type":"response","success":true,"body":{ ... caps ... }}
<- {"seq":2,"type":"event","event":"initialized"}
-> {"seq":3,"type":"request","command":"launch",
    "arguments":{"program":"foo.m","stopOnEntry":true}}
<- {"seq":4,"type":"response","success":true}
-> {"seq":5,"type":"request","command":"setBreakpoints",
    "arguments":{"source":{"path":"foo.m"},
                 "breakpoints":[{"line":10}]}}
<- {"seq":6,"type":"response","success":true,
    "body":{"breakpoints":[{"verified":true,"line":10}]}}
-> {"seq":7,"type":"request","command":"configurationDone"}
<- {"seq":8,"type":"response","success":true}
<- {"seq":9,"type":"event","event":"stopped",
    "body":{"reason":"breakpoint","line":10,"threadId":1,
            "allThreadsStopped":true}}
```

Compare to the protocol cheat sheet at the end of
[`docs/lsp.md`](lsp.md) for the equivalent LSP framing.

### Test coverage

Three ctest suites guard the debugging surface (all gated on
`MATLAB_LLVM_WITH_MLIR=ON`):

- **`debug-hook-tests`** — drives `matlabc -emit-mlir -g` over a small
  set of fixtures in `test/Debug/*.m` (blank lines, comment lines,
  `if`/`for`/`while` blocks, helper-function bodies). Extracts the
  line constant baked into every emitted `matlab_dbg_hook` call and
  asserts both the exact list per fixture *and* the property that
  every hook line points at a non-blank, non-comment-only source row.
  This is what guards "stepping never lands on a blank line", whether
  or not the lowering's normalization pass had to fire.

- **`debug-dap-tests`** — spawns `matlabc -dap` as a subprocess and
  drives the protocol with a small Python client
  (`test/Debug/dap_client.py`). Fifty-one scenarios cover the
  end-to-end surface:

  *Stepping & basic flow:*
  - plain breakpoint (`reason="breakpoint"`, expected line)
  - step-vs-breakpoint reasons (the regression where every pause
    was hardcoded as `"breakpoint"` even after `next`)
  - `threads` request + `continued` event symmetry on resume
  - `stackTrace` / `scopes` / `variables` introspection

  *Variables & evaluation:*
  - **function-frame Locals** — paused inside `compute(a, b)`,
    `variables` for the function frame shows `a` / `b` / `total` and
    NOT script-scope `seed`; the script frame's view shows `seed`
    and not the function's locals
  - **`evaluate`** (watch / script scope) — pure arithmetic
    (`1 + 1`), workspace references (`x`, `x + y`), matrix literals
    (`[1 2; 3 4]`), trailing-semicolon tolerance, malformed
    rejection
  - **`evaluate` with `context: "repl"`** — runs `disp(A)` against
    a paused matrix, captures the row output via the stdout pipe
    as `output` events; verifies REPL-scope assignment
    (`tmp_repl = 99;`) is visible to a follow-up watch read;
    trailing `;` preserved; malformed input fails cleanly
  - **`evaluate` in a function frame** — `evaluate("a", frameId=…)`
    resolves to the function's parameter; without `frameId` the same
    expression silently defaults; after the bridge reverses, the
    script ws is unchanged

  *Mutation:*
  - `setVariable` round-trip — scalar, matrix literal `[1 2; 3 4]`,
    fresh-name assignment, malformed-RHS rejection,
    non-identifier-name rejection
  - `setExpression` lvalue mutation — assigns through arbitrary
    lvalue expressions; verified by a follow-up `evaluate` read

  *Breakpoint variants:*
  - conditional breakpoint (false condition silently resumes, true
    one stops)
  - log point (emits an `output` event, never `stopped`)
  - `setFunctionBreakpoints` — resolves `compute` from
    `dap_locals_program.m`'s function table to `compute.m:10`;
    unknown name comes back with `verified=false`
  - `breakpointLocations` — returns only executable lines
    (assignment / disp lines for `dap_program.m`); blank and
    comment-only lines are excluded
  - **multi-file breakpoint** — `dap_main.m` calls `helper_fn` from
    a sibling `dap_helper.m`; bp on the helper file is `verified`,
    fires, and `stackTrace` reports the helper's source path

  *Errors:*
  - `error()` backtrace — nested user-function calls raise via
    `error('boom')`; stderr must contain the message header plus one
    frame line per call site (innermost first)
  - `setExceptionBreakpoints` + `exceptionInfo` — toggles the
    `error` filter; once the runtime hook pauses on the failing
    statement (or the program runs to completion), `exceptionInfo`
    returns the captured message and frame snapshot

  *Source / sources / completions:*
  - `loadedSources` + `source` — the entry point appears in the
    loaded list; fetching its content matches the file on disk
  - `completions` — workspace name `x` and builtin `disp` both
    surface for prefixes `x` / `dis`

  *Lifecycle / capabilities:*
  - `modules` — returns an empty list cleanly
  - **`unsupported_refusals`** — the still-refused requests
    (`locations`, `setInstructionBreakpoints`, `restartFrame`,
    `goto`, `gotoTargets`) respond with `success=false` and a
    precise reason; the connection stays open. The list shrank
    over successive rounds — `stepBack`, `reverseContinue`,
    `readMemory`, `writeMemory`, `disassemble`, and
    `setDataBreakpoints` were all moved out as the underlying
    features shipped

  *Frame-scoped evaluation:*
  - **`frame_scoped_conditional_breakpoint`** — a bp inside
    `compute(a, b)` with condition `a > 2` fires when called as
    `compute(3, 4)`; the cond evaluator stamps the function frame's
    mini-ws into matlab_ws, runs the cond, and restores. With the
    same fixture, condition `a > 99` silently resumes (no stop)
  - **`frame_scoped_log_point`** — `{a}` / `{b}` placeholders in a
    function-body logMessage interpolate against the function
    frame's mini-ws, not the script ws

  *Hit counts + class methods:*
  - **`hit_count_breakpoint`** — `hitCondition: ">= 3"` on a bp
    inside `for i = 1:3` pauses on the third iteration only;
    `i == 3` is verified through `variables` of the script frame
  - **`class_method_function_breakpoints`** — `setFunctionBreakpoints`
    resolves `Account.deposit` under all three name forms
    (`deposit`, `Account.deposit`, `Account/deposit`)

  *Lifecycle:*
  - **`pending_breakpoint_event`** — sends `setBreakpoints` BEFORE
    `launch` (DAP-permitted ordering); the response carries
    `verified=false` and the bp is queued. After `configurationDone`,
    a `breakpoint` event with `reason="changed"` arrives carrying
    `verified=true`, and the bp fires normally

  *UX hardening:*
  - `breakpoint_ids` — each bp from `setBreakpoints` carries a
    stable `id`; the `stopped` event surfaces it as
    `hitBreakpointIds` when that specific bp triggered the pause
  - `stderr_forwarded` — the error()-traceback bytes that hit
    stderr are forwarded to the IDE as `output` events with
    `category: "stderr"`, while still reaching the parent
    process's stderr (verified by the existing error-traceback
    scenario's stderr_buf assertion continuing to pass)
  - **`watch_void_promotion`** — watch-mode `disp(A)` used to
    SIGSEGV the matlabc process (the `__matlab_dbg_eval = (...);`
    wrap can't bind a void RHS). The handler now detects
    statement-shaped void calls (`disp`, `clear`, `who`, `whos`,
    plotting calls, etc.) up front and routes them through the
    REPL branch, returning `result="<void>"`; side-effect output
    flows through the existing stdout pipe

  *Composite / domain:*
  - **matrix expansion** — `dap_matrix_program.m` constructs a
    2x3, a 3x1, and a 1x1 matrix; the scenario asserts the
    `RxC double` shape labels, the `(i,j)` / `(i)` cell layout,
    the 1x1 unbox, and that an `evaluate("A")` watch result
    carries the same mat-ref so the IDE can drill into a watched
    expression
  - **class-instance Locals + Watch** — `dap_class_program.m`
    constructs `Account` and `Savings` (subclass) instances at
    script scope; the LOCALS panel reports `1x1 Account` /
    `1x1 Savings` with expandable property children, the inherited
    `Id` / `Balance` come through alongside `Savings.Rate`, and a
    watch-box `evaluate("acc")` exercises the kind=1 → kind=2
    promotion for class instances that the REPL JIT mistakenly
    stamps as matrices
  - **`class_instance_methods`** — same fixture; verifies that
    expanding `acc` shows `deposit` + `Account` constructor as
    method rows alongside the property rows, with
    `type: "method"`, `presentationHint.kind: "method"`, and a
    `@deposit(obj, amt)` signature in the value column.
    Expanding `sav` (Savings < Account) shows its own constructor
    plus an inherited `Account` ctor and `deposit` — both flagged
    `(inherited from Account)`
  - **`complex_and_3d_matrix_expansion`** — `dap_complex_program.m`
    constructs `c = 3 + 4i` (1×1 complex) and `A = ones(2,2,2)`
    (2×2×2 real). The complex 1×1 unboxes to `"3+4i"` in the
    value column with no expansion; `A` reports
    `value="2x2x2 double"` + `indexedVariables=8` + a mat-ref;
    drilling emits all eight `(i,j,k)` cells in slice-major order
    with the mutated `(1,2,1)=42` and the rest `=1`
  - **`keyboard_builtin`** — `dap_keyboard_program.m` calls
    `keyboard;` after assigning `x = 41`. The DAP `stopped` event
    arrives with `reason="entry"` at the keyboard line; the
    Locals panel still shows `x=41`; resumption produces the
    expected `disp(x)` output and exits cleanly
  - **`data_breakpoint_write`** — `dap_watchpoint_program.m`
    writes `target = 1; target = 2;`. With a write-watch on
    `target`, the runtime trips on both writes; each `stopped`
    event reports `reason="data breakpoint"` and surfaces the
    watch's id in `hitBreakpointIds`. Inspecting the workspace
    at each trip shows the freshly-written value
  - **`data_breakpoint_clear`** — verifies that an empty
    `setDataBreakpoints` list wipes prior watches; the program
    runs to termination without any `stopped` events
  - **`data_breakpoint_accesstype_advertised`** —
    `dataBreakpointInfo` returns all three access types
    (`read` / `write` / `readWrite`) so the IDE renders an
    accessType chooser
  - **`data_breakpoint_read`** — read-only watch on `target`;
    the two writes don't trip but `disp(target)` on line 8
    does. Single trip with `reason="data breakpoint"`
  - **`data_breakpoint_readwrite`** — readWrite watch trips
    on every write (lines 6, 7) and the read (line 8) — three
    trips total in the fixture
  - **`parfor_thread_enumeration`** — `dap_parfor_program.m`
    runs `parfor i = 1:3`. After the body executes, the
    `threads` request reports the main worker (id=1, "main")
    plus one entry per parfor pthread ("parfor-1" / etc.). The
    runtime's `matlab_dbg_thread_slot_locked` lazy-registers
    each pthread on its first hook fire
  - **`parfor_per_thread_frames`** — concurrent parfor body
    executes a function call (enter_frame / leave_frame) on
    three pthreads simultaneously. Without per-thread chains
    the global `n_frames` would race; with per-thread chains
    each worker mutates its own slot and the program runs
    cleanly. Confirms thread enumeration survives
  - **`disassemble`** — verifies `supportsDisassembleRequest`
    is advertised, walks four instructions starting from the JIT
    main entry, and confirms each row has an address (`0x...`),
    a hex-bytes string, and printed asm. Negative
    `instructionOffset` comes back with a clear refusal
  - **`step_back`** — `dap_revstep_program.m` runs `a=100;
    b=200; c=300; disp(c);` on lines 5-8. Pause at line 8 with
    `{a=100, b=200, c=300}`. Each stepBack walks back one
    statement, asserting both the resume line *and* the exact
    workspace state: stepBack #1 → line 7, `{a=100, b=200}`
    (c removed via `rmfield`); stepBack #2 → line 6, `{a=100}`;
    stepBack #3 → line 5, `{}`; stepBack #4 → `reason=entry`
    with `description: "stepBack: undo log exhausted"`.
    Variables that didn't pre-exist are *removed*, not zeroed
  - **`step_back_overwrites`** — `dap_revstep_overwrite_program
    .m` runs `x=1; x=2; x=3; disp(x);`. Pause at the disp;
    stepBack walks `x` through `3 → 2 → 1 → removed`,
    confirming that `prev_existed=1` records restore the prior
    value (instead of removing the binding)
  - **`step_back_inside_function`** — `examples/factorial.m`
    has `disp(fact(1))` calling a recursive `fact(n)`. Pause at
    line 14 inside `fact`; assert (a) the *innermost frame's*
    line in `stackTrace` updates to 13 after stepBack (the IDE
    renders the caret from stackTrace, not the stopped event's
    `line` field), and (b) the next stepBack refuses to cross
    out of `fact` into the script frame, returning
    `reason=entry`. Locks in the depth-aware boundary matching
  - **`reverse_continue_to_breakpoint`** — sets two bps (lines
    6 and 8) in `dap_revstep_program.m`. Hits the line-8 bp
    after continuing past line 6's first hit; `reverseContinue`
    must walk back to the line-6 bp and stop with
    `reason="breakpoint"` plus the matching `hitBreakpointIds`.
    Catches the regression where `reverseContinue` had a
    `break`-on-first-iteration in the bp scan loop and was
    really just a single-step
  - **`reverse_continue_to_entry`** — same fixture, only one bp
    (line 8). `reverseContinue` walks the entire undo log back
    and stops with `reason="entry"` plus the
    `"reverseContinue: undo log exhausted"` description
  - **`caret_consistency`** — drives `next` / `continue` /
    `stepBack` against `examples/factorial.m` and asserts that
    on every pause `stackTrace[0].line` agrees with the most
    recent `stopped` event's `line`. The IDE renders its caret
    from `stackTrace` (not the event line), so any path that
    desyncs the two leaves the user looking at a stale row.
    The helper `_assert_caret_consistent` is also called from
    `scn_basic_breakpoint`, `scn_step_reason`, `scn_step_back`,
    `scn_hit_count_breakpoint`, and
    `scn_frame_scoped_conditional_breakpoint` so the invariant
    has cross-feature coverage
  - **`write_memory_visible_in_variables`** — after
    `writeMemory` mutates the (1,1) cell of `A` to 7777.0 in
    `dap_matrix_program.m`, expanding the `A` row through
    `variables(matrixRef)` must read `(1,1) = "7777"`. Locks in
    that the byte-level mutation isn't ghost-state in a shadow
    buffer — the IDE's matrix view sees what `writeMemory`
    wrote
  - **`read_watch_on_frame_local_is_invisible`** — negative
    test for a documented limitation: a read-only data
    breakpoint on a function-frame-local name (`total` in
    `compute(a, b)`) does NOT trip when the function reads
    that local. Frame reads bypass the runtime API, so the
    watch table never sees them. If a future lowering wires
    frame reads into the watch path, this test fails and the
    docs need updating
  - **`read_write_memory`** — using `dap_matrix_program.m`'s
    `A = [1 2 3; 4 5 6]`, exercises the matrix `memoryReference`
    field: reads the first 3 doubles via `readMemory` and
    confirms they match {1, 2, 3}; reads past the buffer end
    and verifies `unreadableBytes` reports the truncated tail;
    writes a new pattern via `writeMemory` and reads it back
    to confirm the round-trip; sends a bogus memoryReference
    and verifies the handler refuses with a registration-check
    message

- **`debug-dwarf-tests`** — runs `matlabc -emit-llvm -g` and
  `-emit-llvm` (no -g) over a fixture, asserts the DWARF metadata
  graph (`!DICompileUnit`, `!DIFile`, `!DISubprogram`, `!DILocation`,
  `!llvm.dbg.cu` registration, function-level `!dbg` attachment) is
  present with `-g` and absent without it.

Run the lot via:

```bash
ctest --test-dir build -R "debug-" --output-on-failure
```

Both suites finish in well under two seconds combined; no hangs even
when a scenario fails (the Python harness uses bounded timeouts).

### `keyboard` builtin

MATLAB's `keyboard` pauses execution at the call site with access
to the surrounding scope. Under DAP it routes through the same
pause machinery as a breakpoint:

- The lowerer recognises `keyboard;` (bare-name builtin call) in
  the `ExprStmt` dispatch — same place that catches `who`/`whos`/
  `clear` — and emits `matlab.call_builtin {callee =
  "matlab_dbg_keyboard_hook"}`.
- `LowerTensorOps` maps that to a direct `llvm.call` on the
  runtime symbol.
- `matlab_dbg_keyboard_hook()` snapshots the innermost frame's
  `(file_id, line)` into the cur_* fields, sets `paused=1` and a
  `paused_from_keyboard` flag, then blocks on the same condvar a
  real breakpoint uses.
- The DAP monitor reads the flag via
  `matlab_dbg_was_paused_from_keyboard()` and surfaces stop
  reason="entry" — distinguishing keyboard-initiated pauses from
  breakpoints (`"breakpoint"`) and steps (`"step"`).
- The IDE's REPL panel (already wired via
  `evaluate context="repl"`) takes over from there; resumption
  via `continue` proceeds normally with the workspace intact.

In release-mode (non-`-g`) builds `matlab_dbg.enabled` is 0 and
`matlab_dbg_keyboard_hook` returns immediately, so a `keyboard`
call in a compiled binary is a no-op rather than an error.

#### Standalone REPL pump

Outside DAP, `matlabc -repl` would still benefit from a bidirectional
pump that read input from stdin and routed it through the same
frame-bridge for `runReplInput`. Not built; the DAP path is what most
users hit through their IDE.

## Deliberately out of scope

## See also

- [`docs/repl.md`](repl.md) — the JIT REPL that hosts `dbg()` /
  `who` / `whos` / `clear`.
- [`docs/lsp.md`](lsp.md) — the Language Server surfaces our
  `DiagnosticEngine` output as editor squiggles, and uses the
  same JSON-RPC framing as DAP.
