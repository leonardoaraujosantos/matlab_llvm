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
matlabc -emit-llvm   file.m      # final LLVM IR text
matlabc -emit-c      file.m      # portable C (includes #line)
matlabc -emit-cpp    file.m      # portable C++ (classes preserved)
```

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
| Resume                          | `continue`                          |
| Step over / in / out            | `next` / `stepIn` / `stepOut`       |
| Pause a running program         | `pause` (breaks at the next stmt)   |
| Stack trace                     | `stackTrace`                        |
| Scopes (just `Locals` for now)  | `scopes`                            |
| Workspace variables snapshot    | `variables`                         |
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

### Known limits (deferred, not blocked)

- **Step into user functions.** The hook fires inside user-function
  bodies, but we don't yet push a stack frame at function entry /
  pop at return. Effect: `stepIn` into a user call steps to the
  first statement inside it, but the `stackTrace` response still
  shows a single `<script>` frame. The runtime API for frames
  (`matlab_dbg_enter_frame` / `_leave_frame`) is in place; only the
  MLIR lowering injections are missing.
- **Conditional breakpoints / function breakpoints / log points.**
  Advertised as unsupported in the `initialize` capabilities.
- **`setVariable`.** You can inspect but not mutate from the
  debugger; advertised as unsupported.
- **Multiple source files.** `matlab_dbg_register_file` supports up
  to 256 files in the runtime table, but the DAP server currently
  only registers the entry-point `.m`. Cross-file breakpoints land
  once multi-TU compilation does.

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

## Deliberately out of scope

### Call-stack traces in `error()`

`error()` currently prints just the message text. A backtrace would
require `matlab_dbg_enter_frame` to be invoked at every user-function
entry and `_leave_frame` before each return — same plumbing the DAP
`stepIn` improvement needs. Lands together.

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
