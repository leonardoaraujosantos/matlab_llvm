# Debugging matlab_llvm programs

A tour of the debugging aids shipped today. The baseline (`dbg()`,
REPL workspace commands, `#line`-annotated C output) composes with the
full Debug Adapter Protocol server (`matlabc -dap`) so you can stay in
an editor when a print doesn't cut it.

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

The `-emit-c` and `-emit-cpp` backends annotate each emitted statement
with a `#line "src.m"` directive. gdb and lldb pick these up
automatically when stepping through the compiled C / C++ code, so the
debugger shows your `.m` source rather than the generated C:

```
$ matlabc -emit-c examples/factorial.m > /tmp/fact.c
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
matlabc -emit-c      file.m      # portable C (includes #line)
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
| Scopes (just `Locals` for now)  | `scopes`                            |
| Workspace variables snapshot    | `variables`                         |
| Mutate scalar variables         | `setVariable`                       |
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

Both modes only see the script-level workspace: locals inside a
user function, for-loop induction variables, and SSA scratch values
aren't visible. Per-function slot tables (Option B in
`docs/debug_improve_plan.md`) are the planned follow-up.

### Other known limits (deferred, not blocked)

- **Function breakpoints.** Not advertised as supported.
- **`setVariable`.** Scalars only — typing `x = 99` in the watch box
  while paused calls `matlab_ws_set_f64` and the new value flows
  through to subsequent `disp(x)` calls. Matrix / string / struct /
  cell targets return a clear "only scalar set is supported" error
  without dropping the DAP connection. Capability advertised as
  `supportsSetVariable=true`.
- **Multiple source files.** The DAP server keeps a path → file_id
  table seeded from every file the SourceManager loaded, and the
  `setBreakpoints` handler resolves the IDE-supplied path against
  it (canonicalized via `realpath`). Today only the entry-point
  `.m` is loaded — once Sema starts pulling in sibling `.m` files
  for cross-file calls they'll appear here automatically. Phantom
  paths still respond with `verified=false` instead of crashing.

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
  (`test/Debug/dap_client.py`). Six scenarios cover the end-to-end
  surface:
  - plain breakpoint (`reason="breakpoint"`, expected line)
  - step-vs-breakpoint reasons (the regression where every pause
    was hardcoded as `"breakpoint"` even after `next`)
  - `stackTrace` / `scopes` / `variables` introspection
  - `setVariable` round-trip (write a scalar, read it back)
  - conditional breakpoint (false condition silently resumes, true
    one stops)
  - log point (emits an `output` event, never `stopped`)

Run the lot via:

```bash
ctest --test-dir build -R "debug-" --output-on-failure
```

Both suites finish in well under two seconds combined; no hangs even
when a scenario fails (the Python harness uses bounded timeouts).

## Deliberately out of scope

### Call-stack traces in `error()`

`error()` currently prints just the message text. The frame-stack
plumbing is already wired (each user function pushes via
`matlab_dbg_enter_frame` and pops via `_leave_frame` when -g is
on), but the runtime doesn't yet snapshot the frames into the
diagnostic before unwinding. Tracked as a follow-up.

### `keyboard` as a nested REPL

MATLAB's `keyboard` pauses execution and opens an interactive prompt
at the paused location with access to the surrounding scope. We have
the pause machinery (`matlab_dbg_hook`) and the REPL, but wiring the
locals of a non-script frame through to an interactive evaluator
requires the scoped-eval path DAP's `evaluate` request would also
need. Neither is started.

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
