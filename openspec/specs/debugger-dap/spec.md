# Debug Adapter Protocol Debugger Spec

## Purpose
Document the observed behavior of `matlabc -dap`, a debugger that speaks the Debug Adapter Protocol (DAP) as JSON-RPC over stdio. It compiles the target `.m` (or `.mflow`) program with `ReplMode=true` and `DebugMode=true`, runs it on a detached JIT worker thread, and exposes breakpoints, stepping, and variable inspection to any DAP-capable editor (doc: docs/debug.md; tests: test/Debug, test/Debug/dap_client.py).

## Requirements

### Requirement: Initialize handshake and lifecycle
The system SHALL implement the DAP `initialize` handshake (advertising its supported capabilities) and the launch/lifecycle handshake driven by `configurationDone`, emitting `initialized`, `process`, `thread`, `loadedSource`, `continued`, `stopped`, `output`, `exited`, and `terminated` events.

#### Scenario: Editor initializes the adapter
- **WHEN** a DAP client sends `initialize` followed by `launch` (with `program`) and `configurationDone`
- **THEN** the system SHALL respond with capabilities, emit the `initialized` event, spawn the JIT worker, and emit `process` / `thread:started` / one `loadedSource:new` per registered file (doc: docs/debug.md "Lifecycle events")

### Requirement: Breakpoints
The system SHALL support line breakpoints via `setBreakpoints` (snapping invalid lines forward to the nearest executable row and assigning each verified breakpoint a stable id), plus conditional, log-point, hit-count, function, data, and exception breakpoints.

#### Scenario: Breakpoint on a non-executable line snaps forward
- **WHEN** the client sets a breakpoint on a blank or comment-only line
- **THEN** the system SHALL snap it forward to the nearest executable row, return `verified=true` with a stable id, and surface the snap in the breakpoint `message` (doc: docs/debug.md "What works")

#### Scenario: Conditional breakpoint evaluated against the paused frame
- **WHEN** a breakpoint inside `compute(a, b)` carries the condition `a > 5`
- **THEN** the system SHALL evaluate the condition against the innermost paused frame via the REPL JIT and pause only when the result is non-zero (doc: docs/debug.md "Conditional breakpoints and log points")

#### Scenario: Exception breakpoint on the error filter
- **WHEN** the client enables the `error` exception filter via `setExceptionBreakpoints`
- **THEN** the system SHALL pause the worker on the first statement after `matlab_set_error` fires so the failing frame's locals can be inspected (doc: docs/debug.md "Exception breakpoints")

### Requirement: Stepping and execution control
The system SHALL support `continue`, `next`, `stepIn`, `stepOut`, and `pause`, reporting `reason="step"` versus `reason="breakpoint"` correctly, and SHALL support reverse execution via `stepBack` and `reverseContinue` over a per-statement undo log.

#### Scenario: Step reports the correct reason
- **WHEN** the worker pauses after a `next` request rather than at a breakpoint
- **THEN** the `stopped` event's `reason` field SHALL be `"step"` (doc: docs/debug.md "Stop reasons")

#### Scenario: Reverse step stays within the current frame
- **WHEN** the client sends `stepBack` inside a function frame
- **THEN** the system SHALL rewind one statement's worth of recorded writes, stay within the current function frame, and return `reason: "entry"` rather than teleporting into the caller when walking off the front (doc: docs/debug.md "Reverse stepping")

### Requirement: Variable inspection
The system SHALL serve per-frame `scopes` / `variables` views, expanding matrices into per-cell children, class instances into properties and methods, and exposing matrix grid / property-table hints (`indexedVariables` / `namedVariables`).

#### Scenario: Function-frame locals isolated from script scope
- **WHEN** the worker is paused inside `compute(a, b)` and the client requests `variables` for the function frame
- **THEN** the system SHALL return the function's locals (`a`, `b`, `total`) and not script-scope variables, while the script frame's view shows the script variables (doc: docs/debug.md "Per-frame Locals"; test: test/Debug `debug-dap-tests`)

#### Scenario: Matrix row expands to cells
- **WHEN** the client expands an `RxC double` variable row
- **THEN** the system SHALL emit one child per cell with `(i,j)` labels (linear `(i)` for vectors), unboxing a 1x1 matrix to its scalar (doc: docs/debug.md "Matrix expansion in Locals + Watch")

### Requirement: Expression evaluation and mutation
The system SHALL implement `evaluate` (watch/hover/repl contexts), `setVariable`, and `setExpression` through the REPL JIT against the paused workspace, gated to refuse evaluation while the worker is mid-execution.

#### Scenario: Evaluate refused while running
- **WHEN** an `evaluate` request arrives while the JIT worker is running and not paused
- **THEN** the system SHALL respond `success=false` with a message that evaluate is only valid while paused, pre-launch, or exited (doc: docs/debug.md "Worker-state safety gate on evaluate")

#### Scenario: setVariable validates the name
- **WHEN** `setVariable` is called with a non-identifier name such as `x); system(...)`
- **THEN** the system SHALL reject it before the RHS wrap so no extra statements are smuggled past the literal concatenation (doc: docs/debug.md "setVariable for any RHS expression")

### Requirement: Explicit capability refusals
The system SHALL respond `success=false` with a precise message for requests whose infrastructure is absent (`locations`, `setInstructionBreakpoints`, `restartFrame`, `goto` / `gotoTargets`) and advertise the corresponding `supportsX=false` in `initialize`.

#### Scenario: Instruction breakpoint refused
- **WHEN** the client sends `setInstructionBreakpoints`
- **THEN** the system SHALL respond `success=false` explaining the JIT exposes no public line-to-native-PC mapping (doc: docs/debug.md "Explicit refusals")

### Requirement: Output routing to DAP events
The system SHALL forward the JIT'd program's stdout and stderr as DAP `output` events (`category` `stdout` / `stderr`), preserving the original stdout fd for DAP JSON-RPC frames and tee'ing stderr back to the original fd.

#### Scenario: Program disp output does not corrupt the channel
- **WHEN** the debugged program calls `disp()`
- **THEN** the system SHALL forward the bytes as a `stdout` `output` event rather than splicing them into the JSON-RPC stream (doc: docs/debug.md "Output routing")
