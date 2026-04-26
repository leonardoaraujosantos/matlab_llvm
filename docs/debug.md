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
| Stop on entry                   | `stopOnEntry` option on `launch`    |
| Line breakpoints                | `setBreakpoints`                    |
| Conditional breakpoints         | `setBreakpoints` `condition`        |
| Log points (no pause)           | `setBreakpoints` `logMessage`       |
| Resume                          | `continue`                          |
| Step over / in / out            | `next` / `stepIn` / `stepOut` (stops report `reason="step"`) |
| Pause a running program         | `pause` (breaks at the next stmt)   |
| Stack trace across user fns     | `stackTrace`                        |
| Per-frame Locals                | `scopes(frameId)` returns one Locals scope per frame; `variables(ref)` reads either the script ws or that frame's mini-ws |
| Workspace variables snapshot    | `variables`                         |
| Watch / hover / debug-console eval | `evaluate` (against script-level workspace; function-frame scoping is the planned follow-up) |
| Mutate any workspace variable   | `setVariable` (any MATLAB expr on RHS — scalar, matrix, string, struct) |
| Clean shutdown / terminate      | `disconnect` / `terminate`          |
| Program stdout forwarded        | `output` event (category `stdout`)  |

### Architecture notes

- **Output routing.** The JIT'd program's stdout is piped to a
  reader thread that forwards each chunk as a DAP `output` event;
  we hang on to the original `STDOUT_FILENO` for DAP frames. Without
  this, `disp()` from the program would splice into the JSON-RPC
  stream and corrupt the channel.
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

Both modes evaluate against the script-level workspace. The
`variables` panel exposes function-frame locals through the per-frame
mini-ws machinery described below, but the conditional / log
evaluators don't yet bridge those into the REPL JIT — so a
breakpoint condition referencing a function-local won't see it. The
follow-up that closes that gap is item (6) in
[`docs/debug_improve_plan.md`](debug_improve_plan.md).

### Per-frame Locals + `evaluate`

`Locals` for any frame in the call stack is rendered from a per-frame
mini-workspace the runtime maintains alongside `matlab_dbg.frames[]`.
The lowering's `emitStore` injects a `matlab_dbg_frame_set` builtin
after every store to a named slot when DebugMode is on; LowerTensorOps
dispatches by the operand's lowered type to either
`matlab_dbg_frame_set_f64(name, len, val)` or
`matlab_dbg_frame_set_mat(name, len, ptr)`. The frame push
(`matlab_dbg_enter_frame`) was hoisted to fire *before* the parameter
spill loop in `lowerFunction` so the spill-store mirrors land in the
new frame, not the caller's.

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
pipeline conditional breakpoints use: wrap as
`__matlab_dbg_eval = (<expr>);`, run through `runReplInput`, then
re-read by name and format with `formatVar`. Capability advertised
as `supportsEvaluateForHovers=true`. v1 evaluates against the
script-level workspace plus the script frame's mini-ws — function
frame locals aren't yet visible to the evaluator (see plan item
(6) for the bridge). Malformed expressions come back as
`success=false`; the connection stays open.

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

### Other known limits (deferred, not blocked)

- **Function breakpoints.** Not advertised as supported.
- **Multiple source files.** The DAP server keeps a path → file_id
  table seeded from every file the SourceManager loaded, and the
  `setBreakpoints` handler resolves the IDE-supplied path against
  it (canonicalized via `realpath`). Today only the entry-point
  `.m` is loaded — once Sema starts pulling in sibling `.m` files
  for cross-file calls they'll appear here automatically. Phantom
  paths still respond with `verified=false` instead of crashing.
  This is the only "gated on infrastructure outside DAP" item.

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

Two ctest suites guard the debugging surface (both gated on
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
  (`test/Debug/dap_client.py`). Nine scenarios cover the end-to-end
  surface:
  - plain breakpoint (`reason="breakpoint"`, expected line)
  - step-vs-breakpoint reasons (the regression where every pause
    was hardcoded as `"breakpoint"` even after `next`)
  - `stackTrace` / `scopes` / `variables` introspection
  - **function-frame Locals** — paused inside `compute(a, b)`,
    `variables` for the function frame shows `a` / `b` / `sum` and
    NOT script-scope `seed`; the script frame's view shows `seed`
    and not the function's locals
  - **`evaluate`** — pure arithmetic (`1 + 1`), workspace references
    (`x`, `x + y`), matrix literals (`[1 2; 3 4]`), trailing-semicolon
    tolerance, malformed-expression rejection
  - `setVariable` round-trip — scalar, matrix literal `[1 2; 3 4]`,
    fresh-name assignment, malformed-RHS rejection,
    non-identifier-name rejection
  - conditional breakpoint (false condition silently resumes, true
    one stops)
  - log point (emits an `output` event, never `stopped`)
  - `error()` backtrace — nested user-function calls raise via
    `error('boom')`; stderr must contain the message header plus one
    frame line per call site (innermost first)

Run the lot via:

```bash
ctest --test-dir build -R "debug-" --output-on-failure
```

Both suites finish in well under two seconds combined; no hangs even
when a scenario fails (the Python harness uses bounded timeouts).

## Deliberately out of scope

### `keyboard` as a nested REPL

MATLAB's `keyboard` pauses execution and opens an interactive prompt
at the paused location with access to the surrounding scope. We have
the pause machinery (`matlab_dbg_hook`), the REPL, and per-frame
Locals — but wiring an interactive evaluator that operates on a
non-script frame's mini-ws on demand still needs the
snapshot/restore bridge tracked as plan item (6). Once that lands,
`keyboard` becomes a small REPL-pump-around-pause integration
on top.

### DWARF line tables in `-emit-llvm`

Useful when compiling `.m` → LLVM IR → native with clang and then
stepping in lldb. We do emit file/line locations on every op (via
`FileLineColLoc`), but the `-emit-llvm` text output doesn't yet carry
a full `!DISubprogram` / `!DILocation` metadata graph. Separate work
from DAP; both are tractable.

## See also

- [`docs/repl.md`](repl.md) — the JIT REPL that hosts `dbg()` /
  `who` / `whos` / `clear`.
- [`docs/lsp.md`](lsp.md) — the Language Server surfaces our
  `DiagnosticEngine` output as editor squiggles, and uses the
  same JSON-RPC framing as DAP.
