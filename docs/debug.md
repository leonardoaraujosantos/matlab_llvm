# Debugging matlab_llvm programs

A short tour of the debugging aids shipped today, plus what's
deliberately missing and why a full Debug Adapter Protocol (DAP)
server remains deferred.

## What's available

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

## What's missing, and why

### Full DAP (breakpoints, stepping, variable inspection at stop)

The Debug Adapter Protocol layers a richer debugger model over source
code: an editor can set a breakpoint, ask the backend to run until
that line, then query local variables, step in / over / out, resume,
etc.

Our compile path is AOT. Once an `.m` file has been lowered through
the MLIR passes into LLVM IR and handed to the JIT or clang, there is
no surviving mapping from "the program's current execution point" back
to a specific source line, and no mechanism to *pause* execution at a
given location — each `func.func` is fused and inlined past a naive
source-line boundary very quickly.

Delivering DAP cleanly needs one of two investments:

1. **Breakpoint-aware JIT** — keep the MLIR-level source locations
   alive through code generation, emit proper DWARF line tables, plus
   an inserted `matlab_dbg_hook()` call at each statement that checks
   a breakpoint table and yields control to the debugger thread when
   hit. ~3–4 weeks of focused work; the debugger UI would be on top
   of LLDB / DAP via a thin adapter.

2. **Tree-walking interpreter** — a separate execution engine that
   walks the AST directly instead of compiling. Easier to instrument
   for breakpoints and variable inspection, but means carrying two
   full implementations (the compiler for production, the interpreter
   for debug). ~2 weeks to reach parity with the current feature set.

Neither is started. For now, `dbg()` + the REPL cover most of the
"what's going on at this line?" questions without that machinery.

### Call-stack traces

Errors fired via `error()` currently print just the message text. A
full trace would require maintaining a runtime stack of `(filename,
line, function_name)` entries pushed on each user-function entry and
popped on exit — plumbing work that mostly pays off once DAP lands.

### `keyboard` drops into a nested REPL

MATLAB's `keyboard` command pauses execution and opens an interactive
prompt at the paused location, with access to the surrounding scope.
Our `keyboard` is registered as a builtin but has no behavior yet.
Implementing it cleanly needs at minimum the local-variable
accessibility that a tree-walking interpreter would provide; in
compiled code the locals are register-allocated and invisible by the
time execution reaches the keyboard point.

## Roadmap

Rough priorities if debug tooling gets its own focus block:

1. Call-stack recording so `error()` includes a backtrace.
2. DWARF line tables in `-emit-llvm` output (unlocks LLDB stepping
   through compiled `.m` → native).
3. `matlab_dbg_hook()` injection at statement boundaries, with a
   simple breakpoint table consulted at runtime.
4. DAP adapter binary (`matlab-dbg`) speaking the protocol to an
   editor, forwarding break/step/continue commands to the hook.
5. `keyboard` as a nested REPL that can see the hooked stack frame.

Items 1 and 2 are independently useful; items 3–5 are the minimum for
a "set a breakpoint, inspect a variable" loop in an editor.
