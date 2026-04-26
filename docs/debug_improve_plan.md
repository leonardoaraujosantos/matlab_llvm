# Debug Improvements Plan

A focused, actionable plan for the four limitations called out in
[`docs/debug.md`](debug.md). Items are ordered by user-visible impact;
each lists the runtime / MLIR / DAP-server changes required and the
test scenarios that must pass before the item is considered done.

The DAP server lives in `tools/matlabc/main.cpp`, the runtime debug
state lives in `runtime/matlab_runtime.c`, and the per-statement hook
injection lives in `lib/MLIR/Lowering.cpp`.

---

## (1) Stack frames for `stepIn` / `error()` backtrace

**Today:** `matlabc -dap` exposes `stepIn` and the runtime hook fires
inside user-function bodies, but `stackTrace` always shows a single
`<script>` frame. `error()` prints the message text with no backtrace.

**Why:** the runtime side is already wired —
`matlab_dbg_enter_frame(name, len)` /
`matlab_dbg_leave_frame()` push/pop a frame stack, and
`matlab_dbg_frame_count` / `_at` walk it. The MLIR lowering just
doesn't *call* them on user-function entry/exit yet.

### Changes

- **`lib/MLIR/Lowering.cpp`** — `lowerFunction` (or wherever a
  user-function body gets emitted as `func.func`):
  - At function entry, before the first user statement, emit
    `llvm.call @matlab_dbg_enter_frame(name_ptr, name_len)` when
    `DebugMode` is on. Reuse the existing string-global lowering for
    the function name literal. Skip for the top-level script entry
    (it's already the root frame; double-pushing would shift the
    user view by one).
  - Before *every* `func.return` op, emit
    `llvm.call @matlab_dbg_leave_frame()`. For multi-return functions
    walk the body once and inject ahead of each return.
  - For an early-exit `error()` path: leave_frame must NOT be hit
    (the runtime keeps the frame around so the backtrace can read
    it). Either inject *after* the error call sites (so they
    short-circuit before reaching the leave), or have `error` itself
    walk-and-print without popping.

- **`tools/matlabc/main.cpp`** — `stackTrace` handler:
  - Already calls `matlab_dbg_frame_count()` / `_at`. Should "just
    work" once entry/leave are injected, but verify the script
    pseudo-frame (frame 0 = `<script>`) is preserved when N>0.
  - DAP frame `name` field: use the string passed to enter_frame
    (the runtime stores it). Source path lookup: the frame's
    `file_id` resolves through `matlab_dbg_file_name(id)`.

- **`runtime/matlab_runtime.c`** — minimal:
  - `matlab_dbg_frame_at` already returns name + file/line. Confirm
    it formats line as the *call site*'s line (the `dbg_hook` line
    when this frame paused), not the entry line — that's what DAP
    expects.
  - `error()` (or the equivalent) should snapshot the frame stack
    into the diagnostic before unwinding, so the printed message can
    include `at fn:line` markers.

### Validation

- Step into `fact(5)` from `examples/factorial.m`; `stackTrace`
  response shows `[fact, <script>]`.
- Throw via `error('boom')` inside a user function; stderr shows
  `error: boom\n  at userFn (file.m:N)\n  at <script> (file.m:M)`.
- `examples/fibonacci.m` recursive case: `stepIn fib`, then `stepIn
  fib` again — depth N visible in stackTrace.

---

## (2) Multi-file breakpoints

**Today:** `matlab_dbg_register_file` supports up to 256 files, but
the DAP server only registers the entry-point `.m`. A
`setBreakpoints` request for a helper function in another file
silently no-ops.

### Changes

- **`tools/matlabc/main.cpp`** — `setBreakpoints` handler:
  - When `source.path` doesn't match the entry-point, look up the
    file in a path → file_id table maintained inside the DAP server.
    Allocate a new file_id (monotonic, starting after the entry-point's)
    and call `matlab_dbg_register_file(new_id, path, len)` once. Cache
    so subsequent `setBreakpoints` reuses it.
  - The `breakpoints[].verified` flag in the response should still
    flip `true` only if the path corresponds to a file the JIT
    actually loaded (otherwise we're advertising a phantom).

- **`lib/MLIR/Lowering.cpp`** — `loc(...)`:
  - SourceManager already tracks per-file `Buffer.File` IDs across
    all parsed files. The hook injection's `St.Range.Begin.File`
    already carries the right id. Verify it matches the DAP server's
    table (i.e. both sides use the same ID for the same path), or
    pass an explicit registration call so they stay in sync.

- **JIT bring-up of helper files** — when the entry-point uses
  `function` declarations imported from sibling `.m` files, the JIT
  compiles them into the same module. The Lowering already knows
  about all parsed files via `SourceManager`. Surface that file list
  to the DAP server during `launch` so it can register *every* file
  proactively (saves a roundtrip on the first `setBreakpoints` per
  file).

### Validation

- `examples/factorial.m` calling `fact(...)` defined in a hypothetical
  `fact.m` sibling: `setBreakpoints` on `fact.m:14` is `verified=true`
  and pauses execution at line 14.
- `setBreakpoints` on a non-existent path returns
  `verified=false` and does not crash the runtime table.

---

## (3) `setVariable`

**Today:** the `Locals` scope in the debugger UI is a *snapshot* of
the workspace struct. The `variables` request returns a read-only
view; advertising `supportsSetVariable=false` in the `initialize`
response.

### Changes

- **`tools/matlabc/main.cpp`** — `setVariable` handler:
  - New request branch alongside the existing `variables` /
    `scopes`. Parse `arguments.name`, `arguments.value` (a string),
    and dispatch on the variable's runtime kind:
    - **f64** → `strtod` the value, call
      `matlab_ws_set_f64(name, len, v)`.
    - **matrix** → reject for now (no clean parse for `[1 2; 3 4]`
      from a watch box); return an error like
      `"only scalar set is supported"`.
    - **string / struct / cell** → reject similarly.
  - Update `initialize` capabilities: flip
    `supportsSetVariable` to `true`. Note the partial coverage in
    the docs.

- **`runtime/matlab_runtime.c`** — already exposes
  `matlab_ws_set_f64` and `_set_mat`; nothing to add for the scalar
  case. For matrix-set later, build a `matlab_dbg_parse_matrix(text)`
  helper that reuses the existing array-literal lexer.

### Validation

- Set a breakpoint, hit it, type `x = 99` in the watch box. Resume.
  Subsequent `disp(x)` prints `99`.
- Try to set a matrix-typed variable: returns a clear error in the
  client without dropping the DAP connection.

---

## (4) Conditional breakpoints / log points

**Today:** `setBreakpoints` ignores `condition` and `logMessage`
fields. Advertised as unsupported in `initialize` capabilities.

### Changes

- **DAP wire-up.** Extend the breakpoint table in the runtime so each
  entry stores an optional condition string and an optional log
  string. The DAP server passes them through to a new runtime API:

  ```c
  int matlab_dbg_add_breakpoint_ex(int32_t file_id, int32_t line,
                                    const char *cond, int64_t cond_len,
                                    const char *log,  int64_t log_len);
  ```

- **Expression evaluator.** Conditional and log points both need to
  evaluate a MATLAB expression in the paused location's scope. The
  REPL JIT already does this end-to-end (Lex → Parse → Sema → MLIR
  → JIT) for one-line inputs against a workspace. The plan is to:
  - Refactor the REPL's "compile and run a single statement" into a
    library entry point (`evalExprInWorkspace(text) → MatValue`).
  - In `matlab_dbg_hook`, when the breakpoint at this line carries
    a condition string, invoke that entry to evaluate against the
    *current* workspace; pause only if the result is logical-true.
  - For log points, evaluate and `printf` the expression interpolated
    into the message; never pause.

- **Scoped eval — the hard part.** The REPL workspace today is the
  top-level script's ws struct. When a paused frame is inside a
  user function, its local slots aren't in the ws struct — they're
  alloca'd / SSA values local to the JIT's function. Two options:
  - **(A) Mirror locals into a frame-scoped ws struct on entry.**
    Each `matlab_dbg_enter_frame` (item 1) allocates a small ws
    struct; each store to a tracked local mirrors into it. Cheap
    when no breakpoint is set, but always-on cost. Reject for
    perf-sensitive code.
  - **(B) Lazy snapshot on hook fire.** When the hook decides the
    breakpoint *might* fire (condition present, frame matches),
    walk the JIT's frame's named slots into a workspace, then run
    the eval. Requires a slot-name table baked into the function's
    metadata at lowering time. Heavier infra but pay-as-you-go.

  Recommended: **(B)** — emit a per-function `slots[]` table at
  lowering time keyed by `matlab.name` attrs we already carry, with
  a runtime helper `matlab_dbg_locals_for_frame(frame_idx, &count)`
  that reconstructs a ws struct from those slot pointers on demand.

- **Update `initialize` capabilities** to advertise
  `supportsConditionalBreakpoints=true` and `supportsLogPoints=true`
  once the runtime + eval path lands.

### Validation

- `setBreakpoints` with `condition: "i > 5"` on a line in a
  `for i = 1:10` loop pauses only at iterations 6-10.
- Log point with `logMessage: "i = {i}"` emits one DAP `output`
  event per iteration, no pause.
- Condition with a syntax error: surfaces a single diagnostic the
  first time it's evaluated; subsequent hits don't repeat. Doesn't
  hang the worker thread.

---

## Ordering and dependencies

| # | Depends on | Notes |
|---|---|---|
| 1 | none | unblocks `error()` backtrace too |
| 2 | none | independent of 1 |
| 3 | none | smallest patch |
| 4 | 1 (frame stack), eval-in-workspace | biggest lift |

Recommended ship order: **1 → 2 → 3 → 4**. Each lands as its own
commit; `ctest` should stay 9/9 across all of them.

## Out of scope (separately tracked)

- `keyboard` as a nested REPL — needs scoped eval (item 4) plus a
  bidirectional REPL pump from the paused worker.
- DWARF line tables in `-emit-llvm` — orthogonal; useful for clang
  pipeline users.
- Reverse debugging / time travel — not on the roadmap.
