# REPL / Interactive Interpreter

A JIT-backed REPL is now available as `matlabc -repl` (or `just repl`).
Each input line is lex + parse + Sema + lowered through the same
pipeline as `-emit-llvm`, then handed directly to MLIR's
`ExecutionEngine` for in-process JIT compilation and execution —
no intermediate text LLVM IR, no clang invocation, no temp files.

## State persistence

Variables assigned in one input are visible in later inputs. The
compiler reroutes script-level `Var` reads/writes through a runtime
workspace (`matlab_ws_get_*` / `matlab_ws_set_*` backed by a single
`matlab_struct` living in the matlabc process):

```
>> x = 42
x =
42
>> y = x * 2
y =
84
>> disp(x + y)
126
```

For-loop induction variables and any other binding that already owns
a function-local slot keep their slot-local semantics within their
scope; only bindings that are pure "script workspace" variables route
through the runtime table.

## Multi-line blocks

`if` / `for` / `while` / `switch` / `try` / `function` / `classdef`
auto-continue: the prompt switches to `   ` while block depth is
positive, and the whole block is compiled as a single unit once the
matching `end` is seen.

## Symbol resolution

`matlabc` links `runtime/matlab_runtime.c` directly into its own
binary (controlled by `CMakeLists.txt`). MLIR's `ExecutionEngine`
resolves `matlab_*` and `matlab_ws_*` symbols against the running
process via LLJIT's default dynamic-library search generator, so no
`.so` / `.dylib` is required at runtime.

## See also

- [`docs/debug.md`](debug.md) — `dbg(x)` and the workspace
  commands (`who` / `whos` / `clear`) that are especially useful
  inside the REPL.
- [`docs/lsp.md`](lsp.md) — non-interactive editor integration
  (diagnostics, goto-def, outline) on the same front-end stack.

## Line editing and history

The prompt runs a small termios-raw-mode line editor when stdin is a
TTY — no external dependency (no `readline`, no `libedit`). Supported
keys:

| Key | Action |
|---|---|
| `↑` / `↓` | Previous / next line from history |
| `←` / `→` | Cursor left / right |
| `Home` / `End` (or `Ctrl-A` / `Ctrl-E`) | Jump to line start / end |
| `Backspace` / `Delete` | Delete char before / at cursor |
| `Ctrl-U` / `Ctrl-K` | Kill to start / to end of line |
| `Ctrl-L` | Clear the screen |
| `Ctrl-C` | Discard the current line (doesn't exit) |
| `Ctrl-D` | Exit on an empty line; delete-char-forward otherwise |
| `Enter` | Submit |

History is bounded at 500 entries, deduplicates consecutive duplicates,
and is not persisted across sessions (in-memory only). Wrap with
`rlwrap` if you want a persistent `~/.matlab_history`.

When stdin is piped (scripted input, CI, heredocs) the editor falls
back transparently to `std::getline` — no raw-mode side effects, no
escape-sequence handling, same behavior as before.

## `help` — built-in topic browser

```
>> help
  matlab_llvm REPL help
  =====================

  Usage:
    help               — this overview
    help <topic>       — detailed help on a topic

  Topics (grouped):
    FFT                fft  ifft  fft2  ifft2
    Complex            conj  real  imag  angle  abs
    Linear algebra     inv  det  svd  eig  lu  qr  chol  pinv  norm  trace  kron
    ...

>> help fft
  fft
  ===

  GROUP:     FFT
  SYNOPSIS   Y = fft(X)
  DESCRIPTION
    DFT of a real or complex vector / matrix column. Pure-C Cooley-Tukey.
  EXAMPLES
    fft([1 2 3 4])
       10+0i  -2+2i  -2+0i  -2-2i
    ...
```

`help` is intercepted at the REPL loop level, before the compile
pipeline — it isn't a real Sema builtin, it's a REPL UX affordance
(like MATLAB's own `help`). Accepts both command syntax (`help fft`)
and function syntax (`help('fft')` / `help(fft)`). Table is defined
inline in `tools/matlabc/main.cpp` — straightforward to extend.

## Known limitations

- No JIT caching across inputs — each line rebuilds its own MLIR
  module and ExecutionEngine. Fast enough for human-paced use,
  noticeable on tight benchmark loops.
- History is in-memory only; no persistence across sessions.
- User-defined functions declared inside the REPL are compiled into
  the current module and disappear after the line runs; a follow-up
  call in the next line won't find them.
- `classdef` inside the REPL: same limitation as user functions.

---

## Historical design notes (pre-implementation)

## Scope

A command-line REPL that behaves like MATLAB's command window:

```
>> x = [1 2 3]
x =

     1     2     3

>> y = x * 2
y =

     2     4     6

>> size(y)
ans =

     1     3

>> function r = sq(n); r = n*n; end
>> sq(7)
ans =

    49
```

Requirements:
- Persistent workspace across statements (`x` survives to next prompt).
- Echo-unless-semicolon with MATLAB's display rules. Unassigned
  expressions bind to `ans`.
- Multi-line blocks: `if` / `for` / `while` / `switch` / `function`
  prompt for continuation (`...` or auto-detected).
- Error recovery: bad input or runtime errors return to the prompt,
  never exit.
- Line editing + history + tab completion over workspace and builtins.

## The core mismatch with the existing pipeline

The current AOT pipeline is whole-program and type-specializing. A REPL
breaks every assumption:

| Pipeline assumption | REPL reality |
|---|---|
| `Monomorphise` sees all call sites | New call sites appear every prompt |
| `LowerTensorOps` uses propagated shape info | Shapes change across statements; user rebinds names |
| Sema errors on unresolved identifiers | Unknown names are workspace lookups, not errors |
| Single-pass compilation | Incremental, per-statement |
| Static types per name | `x = 5` then `x = "hi"` is legal MATLAB |

This rules out simply "reusing the existing pipeline in a loop." Two
viable architectures are discussed below; the recommended path is the
tree-walking interpreter first, JIT second.

## Architecture options

### Option A — Tree-walking interpreter over AST/MIR (recommended for v1)

Skip MLIR and LLVM entirely for the REPL path. Each AST node evaluates
directly against a `Workspace` (a `std::unordered_map<std::string,
MatlabValue>`) and calls into `runtime/matlab_runtime.c` for every
primitive operation.

```
Input ──► Lexer ──► Parser (incremental) ──► Sema (REPL mode)
                                                   │
                                                   ▼
                                           AST Interpreter
                                                   │
                    ┌──────────────────────────────┼──────────────────────────┐
                    ▼                              ▼                          ▼
              Workspace                     matlab_runtime.c            Display
          (name → value map)              (add/mul/matmul/...)       (echo unless ;)
```

**Pros**
- Bypasses `Monomorphise` and static-shape constraints entirely — the
  runtime is already dynamic, the interpreter is trivially dynamic.
- No incremental compilation machinery needed.
- Runtime/language dynamism (type rebinding, shape changes, polymorphic
  calls) comes for free.
- Small: ~1500–2000 LoC for the interpreter itself, plus REPL shell.

**Cons**
- Execution is 10–100× slower than compiled code. Each `+` is a function
  call through an opaque pointer.
- Duplicates dispatch logic that already exists in the lowering passes
  (the per-op dispatch in the interpreter mirrors `LowerTensorOps`'s
  per-op lowering).

### Option B — ORC JIT, compile each statement to a function

Each statement becomes a `void exec_N(Workspace*)` function that loads
operands from the workspace via runtime calls, does the work, stores
results back. Use LLVM's ORC JIT to compile and link against the
already-built runtime.

**Pros**
- Execution speed approaches `-emit-llvm`.
- Reuses the full lowering pipeline.

**Cons**
- Every pipeline pass that assumes whole-program has to be disabled or
  made incremental for the REPL path. `Monomorphise` in particular
  needs rework — a function defined at line 5 and called at line 12
  needs its clones materialized on demand.
- Shape-specialized lowering has to be turned off: everything stays
  `matlab_mat*` and all dispatch goes through the runtime, which
  eliminates most of the reason to compile in the first place.
- JIT integration: manage an `LLJIT` instance, add module per
  statement, resolve symbols against the loaded runtime, handle
  symbol conflicts across statements.
- Debugging a miscompile in a single REPL line is painful.

### Recommendation: A, then B

Ship the interpreter first. It's the shortest path to a working REPL
that handles the full language. Add a JIT tier later (Phase 4) that
detects hot functions and compiles them in the background — this is
how CPython+Numba and Julia's tiered compilation work. Don't try to
make the AOT pipeline incremental from day one; that's a trap.

## Step-by-step plan

### Phase 1 — Minimum viable REPL (~2 weeks)

Goal: single-line statements, assignment, expression evaluation,
workspace persistence, display, the common builtins.

**Step 1.1 — Incremental parser.**
The existing `lib/Parse/` wants whole files. Add an entry point:

```cpp
enum class ParseOutcome { Complete, Incomplete, Error };
ParseOutcome parseStatement(StringRef Src, AST::Stmt *&Out,
                            Diagnostic &Diag);
```

- `Complete` — one top-level statement was consumed.
- `Incomplete` — we hit EOF inside an open block (`if` without `end`,
  `function` body without `end`, unbalanced paren). REPL prompts for
  continuation with `...>` and re-tries with accumulated buffer.
- `Error` — syntactic error; REPL prints diagnostic, discards buffer,
  back to main prompt.

Implementation: buffer input, attempt parse; if the parser's
recovery state is "expecting more tokens and saw EOF," return
`Incomplete`. ~2–3 days.

**Step 1.2 — Workspace.**
`lib/Runtime/Workspace.cpp` + `include/matlab/Runtime/Workspace.h`:

```cpp
struct MatlabValue {
  enum Kind { Mat, Struct, Cell, Scalar, String, FnHandle } K;
  void *Payload;   // matlab_mat*, matlab_struct*, etc.
};

class Workspace {
  std::unordered_map<std::string, MatlabValue> Vars;
public:
  std::optional<MatlabValue> lookup(StringRef Name) const;
  void bind(StringRef Name, MatlabValue V);
  void clear();
  void clearOne(StringRef Name);
  std::vector<StringRef> names() const;  // for `who` / `whos` / tab completion
};
```

Runtime entry points in `runtime/matlab_runtime.c`:
- `matlab_workspace_create / destroy`
- `matlab_workspace_load(ws, name) -> void*`
- `matlab_workspace_store(ws, name, value)`
- `matlab_workspace_clear / clear_one`
- `matlab_whos(ws)` — prints `Name / Size / Bytes / Class` table

~2 days.

**Step 1.3 — Sema REPL mode.**
`lib/Sema/Resolver.cpp` today errors on unresolved identifiers. Add a
`ResolveMode` enum:

- `Strict` (current AOT behavior): unresolved ident → error.
- `REPL`: unresolved ident → mark as `WorkspaceRef`, defer to runtime.

At runtime, `WorkspaceRef{name}` lowers to `matlab_workspace_load(ws,
"name")`; unbound-at-eval-time raises a MATLAB error caught by the
REPL loop. Function lookup falls back identically: workspace
function-handle → builtin → user-defined file on path → error. ~2 days.

**Step 1.4 — AST interpreter.**
`lib/Interp/` (new directory). One file, one class:

```cpp
class Interpreter {
  Workspace &WS;
public:
  MatlabValue eval(const AST::Expr &E);
  void exec(const AST::Stmt &S);
  // recursive descent, one case per AST node kind
};
```

Coverage for Phase 1:
- Literals (numeric, string), identifier references (→ workspace lookup)
- Binary / unary operators → runtime calls (`matlab_add_mm`, etc.)
- Indexing (`A(i)`, `A(i,j)`, `A(end)`, colon) → runtime `matlab_index_*`
- Assignment (simple `x = expr` and indexed `A(i) = expr`)
- Calls: builtin → runtime; user function from path → recursive `exec`
  on its AST with a child workspace
- `if` / `for` / `while` / `switch` — evaluated structurally
- Expression statement without `=` → bind to `ans`

~1 week. This is the largest single chunk and the heart of the REPL.

**Step 1.5 — Display.**
`lib/Interp/Display.cpp`. MATLAB's format rules are fiddly but
deterministic:

- Scalar double: `\n<name> =\n\n    <value>\n\n`, 5 significant digits
  default, `format long` → 15.
- Vector/matrix: column-aligned, width derived from widest element, with
  `Columns 1 through N` headers when wider than 80 cols.
- Strings: `<name> =\n\n    '<contents>'\n\n` for char arrays,
  `<name> =\n\n    "<contents>"\n\n` for string scalars.
- Structs: `<name> = \n\n  struct with fields:\n\n    f1: <val>\n    f2: <val>\n`.
- Cells: `<name> =\n\n  N×M cell array\n\n    {<repr>} {<repr>}\n`.

Match `disp`'s existing C runtime implementation where possible — it
already handles formatting; expose it as a library function the
interpreter calls after each non-suppressed statement. ~2 days for the
common cases. The tail (exact column widths, `format` modes, complex
numbers) is endless — ship v1 with scalars + real matrices + strings
well-covered and accept divergence on the rest.

**Step 1.6 — Error recovery.**
Three error sources, all must unwind to the prompt:

1. **Parse errors**: already produce diagnostics via `Diag`; REPL loop
   prints them and continues.
2. **Sema errors** (in REPL mode, much reduced): same.
3. **Runtime errors** via the existing error-flag mechanism the C
   runtime already uses. Wire the flag check to throw a C++
   `MatlabRuntimeError` exception; the REPL loop's `try { exec(...) }
   catch (MatlabRuntimeError &E) { print; }` unwinds cleanly.

Also handle SIGINT (Ctrl-C): install a handler that sets a cancellation
flag checked by the interpreter between AST nodes. Can't interrupt
mid-matmul without runtime cooperation, but per-statement interrupt is
enough for v1. ~2 days.

**Step 1.7 — REPL shell.**
`tools/matlabi/main.cpp` (new tool), or add `-i` flag to existing
`matlabc`.

Embed `replxx` (header-only BSD-licensed line editor) for:
- Line editing + history (`~/.matlab_history`)
- Multi-line input buffering
- Tab completion: candidates = `Workspace::names()` ∪ builtin list ∪
  files on the MATLAB path matching `<prefix>*.m`

Prompt: `>> ` for fresh input, `...> ` for continuation of an open
block. On `Ctrl-D` / `exit` / `quit`, teardown and exit 0.

`CMakeLists.txt`: new target `matlabi` linking `MatlabInterp` +
`matlab_runtime`. ~2 days including dependency plumbing.

**Phase 1 deliverables:**
```
lib/Interp/Interpreter.cpp
lib/Interp/Display.cpp
lib/Interp/Builtins.cpp           # dispatch table: name → fn pointer
lib/Runtime/Workspace.cpp          # or extend runtime/matlab_runtime.c
include/matlab/Interp/...
include/matlab/Runtime/Workspace.h
runtime/matlab_runtime.c           # + matlab_workspace_* entries
lib/Parse/                         # + parseStatement()
lib/Sema/Resolver.cpp              # + REPL mode
tools/matlabi/main.cpp             # or tools/matlabc/main.cpp -i flag
```

### Phase 2 — Multi-line blocks, scripts, path (~1 week)

**Step 2.1** — Function definitions at the REPL.
`function r = sq(n); r = n*n; end` typed interactively becomes a
workspace-bound function handle (or a named entry in a per-session
function table). Reuses the incremental parser's `Incomplete` handling
from Phase 1; just needs the interpreter to register the resulting
`FunctionDecl` in the workspace.

**Step 2.2** — Scripts as REPL input.
A `.m` file without a leading `function` is a script: it executes in
the caller's workspace. `matlabi foo.m` replays the file through the
same interpreter that drives the prompt, with no workspace separation.

**Step 2.3** — Path management.
`addpath` / `rmpath` / `path` builtins. Workspace-adjacent state
(separate from variables). When resolving a call to `foo`, the
interpreter checks: builtins → workspace function handles → `foo.m` on
path (parse + cache AST on first use, reparse if file mtime changed).

**Step 2.4** — Session commands.
`clear`, `clc`, `who`, `whos`, `help <topic>`, `exit`, `quit`, `ans`,
`format short` / `format long`. Implement as interpreter intrinsics
(recognized at AST level, not runtime calls).

### Phase 3 — Polish (~1 week)

- **`keyboard` / `dbstop`**: breakpoints. Enter a nested REPL with the
  current local workspace exposed. `return` or `dbcont` resumes. Needs
  the interpreter to support suspending mid-stack — which is free with
  a tree-walker since the host stack is the MATLAB stack.
- **Errors with source locations**: every AST node carries a
  `SourceLoc`; on runtime error, print a MATLAB-style stack trace
  pointing back to the offending line.
- **`input`, `pause`, `menu`**: interactive builtins that read from
  stdin outside the normal line-editor loop.
- **Figure / plot stubs**: either reject with a clear error ("plotting
  not supported in this build") or integrate a minimal backend. Out of
  scope for v1; just don't crash.

### Phase 4 — JIT tier (optional, ~3–4 weeks)

Once the interpreter is solid, hot functions can be JIT'd in the
background:

- Profile counter on every user-function entry.
- Over threshold → queue for compilation. A worker thread runs the
  existing AOT pipeline with `Monomorphise` configured for a **single**
  concrete type tuple (derived from the hot call site), targeting an
  ORC-JIT'd shared module.
- Dispatcher per user function: first N calls → interpreter; after
  compile completes, swap the function-handle payload to point at the
  JIT'd code.
- On type mismatch (caller provides args of a type the compiled
  version wasn't specialized for), fall back to the interpreter for
  that call and possibly queue a new specialization.

This is how PyPy / LuaJIT / Julia's tiered JIT work. Do not attempt
before Phase 1–3 are stable; it's an optimization, not a feature.

## Op coverage checklist

The interpreter needs to handle every AST node kind the frontend
produces. Grep `include/matlab/AST/` for the authoritative list; at
time of writing the set is approximately:

| AST kind | Interpreter action |
|---|---|
| `NumberLiteral` / `StringLiteral` | → `MatlabValue` via runtime constructor |
| `Identifier` | → `Workspace::lookup` |
| `BinaryOp (+ - * / .* ./ .^ ^ < <= > >= == ~= & \| && \|\|)` | → `matlab_<op>_mm` or scalar fast path |
| `UnaryOp (- ~ ' .')` | → `matlab_<op>` |
| `Call (f(args))` | resolve callee → dispatch (builtin / handle / path) |
| `Index (A(i) / A(i,j) / A(:) / A(end))` | → `matlab_index_*` |
| `SliceAssign (A(i) = v)` | → `matlab_assign_*` |
| `ColonExpr (a:b / a:s:b)` | → `matlab_colon` |
| `MatrixLiteral ([a b; c d])` | → `matlab_mat_from_rows` |
| `CellLiteral ({a, b})` | → `matlab_cell_from_values` |
| `StructAccess (s.f)` | → `matlab_struct_get` |
| `StructAssign (s.f = v)` | → `matlab_struct_set` |
| `If / ElseIf / Else` | recursive `exec` on branch |
| `For` | materialize range, loop, bind induction var |
| `While` | loop with condition re-eval |
| `Switch / Case / Otherwise` | linear match against case values |
| `Break / Continue / Return` | C++ exception unwinding through recursive `exec` |
| `Try / Catch` | C++ try/catch around body; bind caught error to catch-var |
| `FunctionDecl` | register in workspace or session function table |
| `AnonymousFn (@(x) x+1)` | allocate `matlab_fn_handle` with captured environment |

## Testing strategy

Three layers, CI-friendly:

1. **Golden transcript tests**: `test/REPL/*.mrepl` — each file is an
   input-output pair. Lines prefixed `>> ` are input; everything else
   is expected output. A test runner pipes inputs through `matlabi`
   and diffs stdout against the golden. Catches both computation and
   display regressions. ~50 transcripts covering the Phase 1 surface.

2. **Script equivalence tests**: every `test/Run/*.m` program already
   has a `.stdout` golden used by `-emit-llvm` / `-emit-c`. Add a
   `run-tests-repl` lane that runs each program through `matlabi
   script.m` and diffs the same golden. Reuses existing test corpus,
   catches divergence between compiled and interpreted paths.

3. **Interactive smoke**: a pexpect-based test that drives the REPL
   like a user — sends Ctrl-C, `exit`, checks that incomplete input
   triggers the `...>` prompt, confirms tab completion returns
   expected candidates. A handful of these, not a large suite.

## Effort estimate

| Phase | Scope | Effort |
|---|---|--:|
| 1 | Single-line REPL, workspace, display, error recovery, shell | ~2 weeks |
| 2 | Multi-line blocks, scripts, path, session commands | ~1 week |
| 3 | `keyboard`, stack traces, interactive builtins, polish | ~1 week |
| **Total useful v1** | Phase 1+2+3 | **~4 weeks** |
| 4 | JIT tier (optional) | ~3–4 weeks |

"Matches MATLAB behavior on a representative test corpus" is another
2–3 months of tail chasing: `ans` chaining rules, `end`-in-index
semantics, command-vs-function syntax, broadcasting edge cases,
`format` modes, complex-number display, `inputname`, `evalin`, `base`
workspace vs. function workspace distinctions.

## Non-goals

- **Matching MATLAB byte-for-byte on display formatting.** Aim for
  "looks right for common cases." Exact column widths and trailing-
  whitespace reproduction are not worth the engineering.
- **Plotting / figures / GUI.** Reject cleanly with a diagnostic.
  Integrating a graphics backend is an entirely separate project.
- **MEX / external binary interface.** No loading of compiled `.mex`
  files. If the test corpus needs C extensions, add them to the
  runtime directly.
- **Live MLIR / LLVM IR inspection commands** (`matlabi> :mlir x+y`).
  Nice to have for debugging the implementation; not a user feature.
- **Persistent workspace across sessions.** MATLAB has `save` / `load`
  for this. Implement `save workspace.mat` / `load workspace.mat` as
  builtins only if needed; skip for v1.

## Open questions

1. **Should `matlabi` be a new binary or a `-i` flag on `matlabc`?**
   Pragmatically, a `-i` flag is cheaper — one build target, one
   install. A separate `matlabi` binary is more discoverable for
   users. Lean toward `-i` for v1, split later if `matlabc` grows too
   many responsibilities.

2. **Does the interpreter share code with any emitter?**
   No direct reuse is possible — the emitters print C/Python/SystemC,
   the interpreter calls runtime functions. But **the runtime itself
   is shared**. Every operation the interpreter needs is already
   callable in `runtime/matlab_runtime.c`; we're adding very few new
   entries (workspace, possibly display helpers). This is the main
   reason the interpreter is cheap: the heavy lifting (matmul,
   broadcasting, indexing) exists and is battle-tested.

3. **What about whole-file scripts that were previously compiled?**
   The compiler keeps working unchanged. The REPL is additive. A user
   choosing `matlabi foo.m` gets interpreted execution; `matlabc
   -emit-llvm foo.m` still produces a compiled binary. The test
   corpus runs through both paths — divergence is a bug in whichever
   is wrong.

4. **Debugger integration.**
   MATLAB's debugger (`dbstop`, `dbstep`, `dbcont`, `dbquit`,
   `dbclear`) is a substantial feature. For v1, `keyboard` alone
   (nested REPL) covers the 80% case. Full breakpoint management is
   Phase 3+ or later.

## Files

| File | Purpose |
|---|---|
| `lib/Interp/Interpreter.cpp` | AST walker — recursive `eval`/`exec` over every node kind |
| `lib/Interp/Display.cpp` | MATLAB-style formatting for scalars, matrices, structs, cells, strings |
| `lib/Interp/Builtins.cpp` | Name → function-pointer dispatch table for built-in functions |
| `lib/Interp/Session.cpp` | REPL loop, prompt, multi-line buffering, tab completion, history |
| `lib/Runtime/Workspace.cpp` | `Workspace` class; C ABI wrappers in `runtime/matlab_runtime.c` |
| `include/matlab/Interp/...` | Public headers |
| `lib/Parse/Parser.cpp` | + `parseStatement(StringRef, AST::Stmt*&) -> ParseOutcome` |
| `lib/Sema/Resolver.cpp` | + `ResolveMode::REPL` (unresolved idents → `WorkspaceRef`) |
| `runtime/matlab_runtime.c` | + `matlab_workspace_*`, + error-flag → C++-exception bridge |
| `tools/matlabc/main.cpp` (or `tools/matlabi/main.cpp`) | `-i` flag or new binary entry point |
| `test/REPL/*.mrepl` | Transcript golden tests |
| `test/REPL/run_tests_repl.sh` | Runner: feed input, diff stdout |
| `CMakeLists.txt` | New `MatlabInterp` library, new `matlabi` / extended `matlabc` target, CTest lane |
| `justfile` | `repl` / `test-repl` recipes |

## Risks

1. **Display formatting tail.** Every `.mrepl` golden is a hostage to
   MATLAB's exact formatting. Pin the scope early: document which
   format cases we match and which we explicitly don't.

2. **Sema REPL mode leaking into AOT path.** The `ResolveMode` enum is
   the fence. Must guarantee that `ResolveMode::Strict` behavior is
   unchanged bit-for-bit — add a CI test that runs the existing
   frontend-tests suite through the modified Sema and confirms zero
   regression.

3. **Interpreter/compiler divergence.** Two implementations of the same
   language semantics → they will disagree. The "script equivalence"
   test lane (Phase 1 Step 1.5 + test layer 2 above) is the main
   defense. Budget time for chasing divergences, not just the first
   implementation.

4. **Runtime error unwinding.** The C runtime uses an error flag, not
   exceptions. Throwing C++ exceptions through C call frames is
   technically fine (well-defined on every platform we target) but
   needs `-fexceptions` everywhere and a discipline check that no
   runtime function leaves state half-mutated on error.
