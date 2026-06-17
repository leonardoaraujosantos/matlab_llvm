# JIT-backed Interactive REPL Spec

## Purpose
Document the observed behavior of `matlabc -repl`, a JIT-backed interactive command window. Each input line is lexed, parsed, Sema-resolved, and lowered through the same pipeline as `-emit-llvm`, then handed to MLIR's `ExecutionEngine` for in-process JIT compilation and execution. This spec records the workspace persistence, JIT execution, multi-line, and editor-affordance behavior that ships today (doc: docs/repl.md; src: tools/matlabc/main.cpp; tests: test/Repl).

## Requirements

### Requirement: JIT execution of each input
The system SHALL compile and execute every REPL input through the in-process MLIR ExecutionEngine without emitting text LLVM IR, invoking clang, or writing temp files.

#### Scenario: Single expression evaluated
- **WHEN** the user enters `disp(x + y)` at the `>>` prompt
- **THEN** the system SHALL lower the line through the `-emit-llvm` pipeline, JIT-compile it, and execute it in the running `matlabc` process, resolving `matlab_*` / `matlab_ws_*` symbols against the process via LLJIT's default dynamic-library search generator (doc: docs/repl.md "Symbol resolution")

### Requirement: Workspace persistence across lines
The system SHALL keep script-level variables assigned on one input visible to later inputs by routing script-workspace `Var` reads/writes through a runtime workspace (`matlab_ws_get_*` / `matlab_ws_set_*`) backed by a single `matlab_struct` in the process.

#### Scenario: Variable survives to the next prompt
- **WHEN** the user enters `x = 42` and then on the next line `y = x * 2`
- **THEN** the system SHALL report `y = 84`, reusing the persisted `x` (doc: docs/repl.md "State persistence"; test: test/Repl/run_tests.sh)

#### Scenario: Loop induction variables keep slot-local semantics
- **WHEN** an input contains a `for` loop whose induction variable already owns a function-local slot
- **THEN** the system SHALL keep that binding slot-local within its scope rather than routing it through the runtime workspace table

### Requirement: Multi-line block continuation
The system SHALL auto-continue input while block depth is positive for `if` / `for` / `while` / `switch` / `try` / `function` / `classdef`, switching the prompt to a continuation indentation and compiling the whole block as a single unit once the matching `end` is seen.

#### Scenario: Open block prompts for continuation
- **WHEN** the user enters `for i = 1:3` without a matching `end`
- **THEN** the system SHALL not execute yet but accept further lines until the closing `end`, then compile and run the block as one unit (doc: docs/repl.md "Multi-line blocks"; test: test/Repl/run_tests.sh `multiline_for_block`)

### Requirement: Cross-turn user-function persistence
The system SHALL persist user-defined functions declared at the REPL so a later input that names the function can call it, by stashing the source in a per-session function table (`g_ReplUserFunctions`) and replaying it via `buildReplPrelude` on subsequent turns.

#### Scenario: Function defined on one turn called on a later turn
- **WHEN** the user declares `function r = sq(n); r = n*n; end` and on a later line enters `sq(7)`
- **THEN** the system SHALL return `49` (src: tools/matlabc/main.cpp `g_ReplUserFunctions` / `buildReplPrelude`; test: test/Repl/run_tests.sh `cross_turn_user_fn`, `transitive_user_fns`, `redef_user_fn`)

### Requirement: REPL inside a DAP debug session
The system SHALL accept REPL-style input through the DAP `evaluate` request with `context: "repl"`, running it through the same `runReplInput` pipeline against the paused frame's workspace, with output returned as DAP `output` events of `category: "stdout"`.

#### Scenario: Prompt evaluated against the paused worker
- **WHEN** a `matlabc -dap` session is paused and the IDE sends `evaluate` with `context: "repl"` and input `disp(x)`
- **THEN** the system SHALL run the statement verbatim against the paused session's workspace and emit the output as a stdout `output` event (doc: docs/repl.md "REPL inside a DAP debug session"; doc: docs/debug.md)

### Requirement: Interactive line editing and history
The system SHALL run a termios raw-mode line editor when stdin is a TTY, supporting cursor movement, history navigation, and kill/clear keys, and SHALL fall back transparently to line-based reads when stdin is piped.

#### Scenario: Piped input falls back to getline
- **WHEN** REPL input is piped (scripted, CI, or heredoc) rather than a TTY
- **THEN** the system SHALL read via `std::getline` with no raw-mode side effects or escape-sequence handling (doc: docs/repl.md "Line editing and history")

#### Scenario: History bounded and in-memory
- **WHEN** the user navigates history with the up/down arrows in a TTY session
- **THEN** the system SHALL serve from an in-memory history bounded at 500 entries that deduplicates consecutive duplicates and is not persisted across sessions

### Requirement: Built-in help topic browser
The system SHALL intercept `help` at the REPL loop level (before the compile pipeline) and print an overview or per-topic help, accepting both command syntax (`help fft`) and function syntax (`help('fft')`).

#### Scenario: Topic help requested
- **WHEN** the user enters `help fft`
- **THEN** the system SHALL print the synopsis, description, and examples for `fft` from the inline help table rather than compiling the input (doc: docs/repl.md "help — built-in topic browser")
