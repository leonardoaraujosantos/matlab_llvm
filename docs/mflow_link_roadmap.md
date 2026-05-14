# mflowLink — Signal-Flow Simulation Backend (matlab_llvm side)

Plan for the **compiler + runtime** support behind *mflowLink*: a
Simulink-like, time-domain block-diagram simulation layer built on
top of the existing `.mflow` graphical frontend. Where today's
`.mflow` pipeline lowers a *control-flow* diagram to a structured
AST (`docs/flowchart_frontend.md`), mflowLink adds a second
**signal-flow** dialect that lowers to a simulation IR and runs
through a new ODE-driven runtime.

**Status: not started — entirely greenfield.** `matlab_llvm` has
zero signal-flow awareness today (survey 2026-05-14): the `.mflow`
loader doesn't know `settings.kind`, there's no
`runtime_mflowlink.cpp`, no `-simulate` flag, and no signal-flow
DAP verbs. This doc is the concrete compiler-side plan; the
**IDE-side** authoring surface is already largely built.

Companion to:
- `Matlab_llvm_ide/docs/mflowLink_roadmap.md` — the IDE-side
  roadmap. Its §6/§7 sketch this contract; this doc is the full
  matlab_llvm version. The IDE half (palette, canvas, inspector,
  standalone window, edit-time diagnostics) is **shipped**; this
  doc is the half that isn't.
- [`flowchart_frontend.md`](flowchart_frontend.md) — the existing
  control-flow `.mflow` frontend this builds beside.
- [`flowchart_schema.md`](flowchart_schema.md) — the `.mflow` JSON
  contract; mflowLink extends it additively (no version bump).
- [`ode.md`](ode.md) — the shipped `ode45` / `ode23` solvers the
  simulation runtime **wraps** rather than reimplements.
- [`verilog_a_plan.md`](verilog_a_plan.md) — the tiered-plan style
  this doc follows.

---

## 0. Context — the two `.mflow` dialects

`.mflow` is one file format with two dialects, picked by a new
`settings.kind` field:

    .mflow
    ├── control_flow  (default, shipped) → Loader → GraphToAST
    │                                    → structured AST → every
    │                                      existing backend
    └── signal_flow   (mflowLink, NEW)   → Loader → SignalFlowLowering
                                         → MflowLinkModel IR
                                         → runtime_mflowlink.cpp

A control-flow `.mflow` is a *program* — `if` / `for` / `while`
nodes that reduce to MATLAB statements. A signal-flow `.mflow` is a
*block diagram* — `signal_*` blocks wired by data edges that the
simulation engine integrates over time. They share the loader, the
JSON shape, the IDE editor, and the DAP transport; they diverge at
the lowering step.

The mental rule:

> If the `.mflow` describes statements that *execute in sequence*,
> it's **control_flow**. If it describes blocks whose outputs are a
> *continuous/discrete function of time and feedback*, it's
> **signal_flow**.

## 1. Goals

- Lower a signal-flow `.mflow` to a flat **`MflowLinkModel` IR**:
  block list, edge list, sorted execution order, state counts,
  sample-time partitions, zero-crossing table.
- Ship **`runtime_mflowlink.cpp`** — a simulation loop that
  *wraps* the existing ODE solvers, schedules multi-rate discrete
  blocks, locates zero-crossings, resolves algebraic loops, and
  maintains a snapshot ring for step-back.
- Add **`matlabc -simulate model.mflow`** — boots the runtime in
  DAP-server mode so the IDE can run / pause / step / step-back a
  diagram as a paused program (the "Simulink-like REPL" the IDE
  roadmap headlines).
- Add **`-emit-mflowlink-cpp`** — a standalone-executable lane for
  baking a model into a deployable simulator.
- Keep it **additive**: schema stays v0.1.0, control-flow `.mflow`
  is byte-for-byte unaffected, an old matlabc still loads a new
  IDE's control-flow files.

## 2. Non-goals

- Full Simscape physical-network simulation (across/through
  variables, modified nodal analysis). Pure signal-flow only.
- Stateflow (hierarchical FSM charts) as a third dialect — deferred.
- Production embedded code-gen (AUTOSAR, fixed-step real-time
  hooks). `-emit-mflowlink-cpp` is a plain host simulator.
- Best-effort simulation of arbitrary `signal_*` graphs — a kind
  whose evaluator hasn't shipped is rejected with a sourced
  diagnostic, exactly like the SV gate.
- Re-implementing ODE solvers. `ode45` / `ode23` already ship with
  adaptive step + event root-finding (`docs/ode.md`); the runtime
  wraps them.

## 3. What already exists — reuse, don't rebuild

A survey of the repo (2026-05-14) — the foundation mflowLink builds
on:

| Component | Location | Reuse |
|---|---|---|
| `.mflow` JSON loader + structural validation | `lib/Flowchart/Loader.cpp`, `include/matlab/Flowchart/Loader.h` (`FlowDoc` struct) | Extend `Settings` + node-kind acceptance (§5) |
| Control-flow lowering (CFG → structured AST) | `lib/Flowchart/GraphToAST.cpp` | Signal-flow gets a **sibling** pass, not a change to this one |
| ODE solvers — `ode45`, `ode23` (adaptive FSAL, dense output, full `odeset`, event root-finding) | runtime builtins; `docs/ode.md` | The scheduler hands continuous partitions to these |
| ODE-wrapped-in-a-loop precedent | `runtime/runtime_pde.cpp` (method-of-lines around `ode23s`) | The structural model for `runtime_mflowlink.cpp` |
| DAP server + CLI `Mode` enum + request dispatch | `tools/matlabc/main.cpp` | Add a `Simulate` mode + the new request handlers here |
| Runtime auto-detect / link wiring | `runtime/build_and_run.sh` (sym / plot blocks) | Add `runtime_mflowlink.cpp` to the link list (§11) |
| Backends (C/C++/Python/TS/LLVM/MLIR), `-emit-matlab`, `-emit-mflow` | existing | Unchanged — signal-flow has its own lane |

## 4. The IDE ↔ compiler contract

The IDE side is shipped; matlabc must match what it already emits.

- **Schema version stays `0.1.0`.** Every signal-flow field is
  optional/additive. The IDE codec treats `0.1.x` ↔ `0.2.x` as
  *incompatible* (pre-1.0, minor = breaking boundary), so a bump
  would gratuitously break loading. matlabc must accept
  `"version": "0.1.0"` documents carrying the new fields. Reserve
  the bump for a genuinely breaking change.
- **The IDE emits a *subset* of the planned node kinds** (see §5.2)
  plus three composite kinds the original IDE roadmap §4.2 didn't
  list: `signal_subsystem` / `signal_inport` / `signal_outport`.
- **Per-block `params` keys** are pinned by the IDE's
  `SignalFlowParamSpec` catalogue. matlabc's evaluators must read
  the *same* keys — see §13.
- **Edge `signal` metadata is not in the schema yet.** The IDE's
  `FlowEdge` carries only `id`/`kind`/`from`/`to`/`label`/
  `waypoints`. Until the IDE adds per-edge `signal` blocks, the IR
  derives signal type/units from the **source block's** `data`
  fields, defaulting to scalar `double`.

## 5. Schema acceptance — `lib/Flowchart/Loader.cpp`

### 5.1 `Settings` struct

`include/matlab/Flowchart/Loader.h`'s `Settings` today holds
`ColumnMajor` / `DefaultNumericType` / `SourceLanguage`. Add:

```cpp
struct Settings {
  // ... existing ...
  std::string Kind = "control_flow";   // "control_flow" | "signal_flow"
  std::optional<SolverConfig>   Solver;
  std::optional<SnapshotConfig> Snapshot;
};

struct SolverConfig {            // settings.solver
  std::string Type      = "variable_step";  // "fixed_step" | "variable_step"
  std::string Algorithm = "ode45";          // ode45 | ode23 | ode23s | euler | heun
  double StartTime = 0.0, StopTime = 10.0;
  std::string MaxStep = "auto", MinStep = "auto";  // "auto" | <seconds>
  double RelTol = 1e-3, AbsTol = 1e-6;
  bool   ZeroCrossing = true;
  std::string AlgebraicLoopMethod = "trust_region"; // trust_region | newton | off
};

struct SnapshotConfig {          // settings.snapshot
  bool Enabled = true;
  int  Depth   = 256;
  std::string Fields = "states"; // "states" | "states+inputs" | "all"
};
```

`Loader.cpp` parses them; an absent `kind` ⇒ `"control_flow"` (the
historical default — control-flow files are untouched).

### 5.2 Signal-flow node kinds

Recognise the `signal_*` kinds. **Accept and round-trip the full
reserved set** — a newer IDE must never trip an older loader — but
a kind whose evaluator hasn't shipped (§7) may be rejected at
*lower* time with a sourced diagnostic.

| Group | Shipped by the IDE | Reserved (accept + round-trip; lower later) |
|---|---|---|
| Sources | `signal_constant`, `signal_step`, `signal_sine`, `signal_pulse`, `signal_ramp` | `signal_chirp`, `signal_noise`, `signal_from_workspace`, `signal_clock` |
| Sinks | `signal_scope`, `signal_display`, `signal_to_workspace`, `signal_terminator` | — |
| Continuous | `signal_integrator`, `signal_derivative`, `signal_transfer_fcn`, `signal_state_space` | `signal_zero_pole`, `signal_transport_delay` |
| Discrete | `signal_unit_delay`, `signal_zoh` | `signal_discrete_integrator`, `signal_discrete_filter`, `signal_rate_transition` |
| Math | `signal_gain`, `signal_sum`, `signal_product`, `signal_abs`, `signal_saturation` | `signal_math_fcn`, `signal_trig_fcn`, `signal_dead_zone`, `signal_relop`, `signal_logical` |
| Signal routing | `signal_mux`, `signal_demux`, `signal_switch` | `signal_bus_creator`, `signal_bus_selector`, `signal_multiport_switch`, `signal_goto`, `signal_from`, `signal_merge` |
| Composite | `signal_subsystem`, `signal_inport`, `signal_outport` | — |
| Conditional | — | `signal_enabled_subsystem`, `signal_triggered_subsystem`, `signal_function_call_generator`, `signal_if_action`, `signal_switch_case_action` |
| Lookup / Logic | — | `signal_lookup_1d/2d/nd`, `signal_relay`, `signal_compare_to_zero`, `signal_compare_to_constant` |
| MATLAB / User | — | `signal_matlab_fcn`, `signal_custom` |

### 5.3 Node `data` fields

Parse the optional signal-attribute fields (snake_case on disk):

- `sample_time` — `"continuous"` | `"inherited"` | `<seconds-as-string>`
- `units` — engineering-unit string
- `data_type` — `"double"` (default) | `"single"` | `"int8"`…
- `log_signal` — bool; true ⇒ stream this block's output (§7)
- `params` — `{ key: scalar }`; scalars are bare JSON `double` /
  `bool` / `string` (not wrapper objects). Keys per §13.

### 5.4 Subsystems

`signal_subsystem` carries `data.flow_id` referencing a sub-`Flow`
in the same multi-flow document; `signal_inport` / `signal_outport`
live *inside* that sub-`Flow` and tag the boundary. The loader
already handles multi-flow documents (for `subflow_call`) — reuse
that machinery; the flattening happens in lowering (§6.2).

## 6. Signal-flow lowering — `lib/Flowchart/SignalFlowLowering.cpp`

A **new sibling** to `GraphToAST.cpp`. Control-flow goes Loader →
GraphToAST → structured AST → backends. Signal-flow takes a
different path: Loader → `SignalFlowLowering` → `MflowLinkModel`
IR. There is no statement AST — a block diagram has no control flow
to structure.

### 6.1 The `MflowLinkModel` IR

The single source of truth handed to the runtime, the codegen
lane, and the model advisor:

- **Block list** — `{ id, kind, params, sample_time_class }`.
- **Edge list** — `{ from_node, from_port, to_node, to_port }`.
- **Sorted execution order** — topological over data edges, with
  feedback resolved through the loop-breaker kinds.
- **State counts** — continuous-state count, discrete-state count.
- **Sample-time partitions** — blocks grouped by sample-time class
  (continuous / discrete-period / fixed-in-minor / constant).
- **Zero-crossing table** — one entry per block that registers a
  crossing predicate (Switch / Saturation / Relay / …).

### 6.2 Subsystem flattening

Before anything else, flatten composites. Each `signal_subsystem`
references a sub-`Flow`; recursively inline that flow's nodes/edges
into the parent, splicing through the boundary tags:

- A `signal_inport` inside the sub-`Flow` is replaced by a direct
  wire from the subsystem's external source to every internal
  target the inport fed.
- A `signal_outport` is replaced by a wire from its internal
  source to every external target the subsystem's output drove.

The result is one flat block graph — the runtime never sees a
subsystem. Recurse for nested subsystems; detect and reject
subsystem cycles (a subsystem referencing an ancestor flow).

### 6.3 Execution-order sort + loop-breakers

Topological sort over data edges. A **loop-breaker** block —
`signal_integrator`, `signal_unit_delay`, `signal_zoh` — carries
state: its output in the current step does not depend on its input
in the same step. Drop a loop-breaker's *outgoing* edges from the
sort graph; what remains acyclic is the direct-feedthrough order.

### 6.4 Algebraic-loop validation

Any cycle left after §6.3 is an **algebraic loop**. The IDE already
runs edit-time detection (`algebraicLoopNodeIDs`) — but that is
*advisory*. matlabc is authoritative: re-detect, and emit a
sourced diagnostic naming the blocks on the cycle. The runtime's
algebraic-loop *solver* (§7) handles loops the user chose not to
break; whether to solve or hard-error is `settings.solver.
algebraic_loop_method` (`trust_region` | `newton` | `off`).

## 7. Runtime — `runtime/runtime_mflowlink.cpp`

A new companion to `runtime_rf.cpp` / `runtime_comm.cpp` / etc.
**Wraps** the existing ODE solvers — the `runtime_pde.cpp`
method-of-lines wrapper is the structural precedent.

### 7.1 Sample-time scheduler

A priority queue keyed on next-hit time. Continuous partitions are
always due; discrete partitions fire on `period + offset`;
fixed-in-minor blocks ride the continuous step. The loop: pop the
next hit, evaluate the matching block partition in execution
order, advance time, repeat — Simulink's "Simulation Loop Phase".

### 7.2 ODE integration — wrap the existing solvers

A continuous partition's job each major step is to integrate its
state derivatives. Build the derivative closure from the
partition's `ddt(x) <+ …` blocks (Integrator / Transfer Fcn /
State-Space) and hand it to the `ode45` / `ode23` / `ode23s`
builtins already in the runtime (`docs/ode.md`). Tolerances and
max-step come from `settings.solver`.

### 7.3 Zero-crossing locator

When a registered crossing predicate flips sign between two
integrator steps, bracket the root by bisection / Illinois and
re-evaluate at the crossing. The ODE builtins already do event
root-finding — reuse that bracketing. Required for Switch /
Saturation / Relay correctness.

### 7.4 Algebraic-loop solver

For a direct-feedthrough cycle the user didn't break: a Newton /
trust-region inner iteration each step, converged to
`settings.solver.rel_tol`. Non-convergence surfaces as a paused
state with a clear diagnostic (see §15).

### 7.5 Snapshot ring buffer — step-back

At the end of every major step, copy
`(t, continuous_state, discrete_state, scheduler_queue, RNG seeds)`
into a fixed-depth ring (default 256, from
`settings.snapshot.depth`). `stepBackMajor` is an `O(1)` restore
from the ring. `settings.snapshot.fields` trades memory for depth:
`"states"` (recompute outputs on restore) vs `"all"`.

### 7.6 Signal log buffer

Per-`log_signal` block, a struct-of-arrays append. Flushed
periodically over the DAP `signalSample` stream (§10) and, in the
`-emit-mflowlink-cpp` lane, dumped to CSV / JSON at `stopTime`.

### 7.7 Per-block evaluator dispatch

A dispatch table keyed on node kind — one C++ function per kind
(`mfl_eval_integrator`, `mfl_eval_gain`, `mfl_eval_sine`, …). The
generated `main()` iterates the schedule and calls into the table.
**The set of kinds and their `params` keys must match §13 exactly**
— this is where IDE/runtime drift would silently corrupt a model.

Tier-3 evaluator set (enough for the `lowpass` / `pid_tracking`
demos): Constant, Step, Sine, Gain, Sum, Product, Integrator,
Transfer Fcn, Scope, To Workspace, Terminator.

## 8. The `-simulate` flag

Add a `Simulate` case to the `Mode` enum in
`tools/matlabc/main.cpp` (alongside `Dap`, `Repl`, the `Emit*`
modes).

```
matlabc -simulate model.mflow
```

Boots the simulation runtime in **DAP-server mode**, reusing the
`-dap` server scaffolding already in that file:

- **No debugger attached** — run to `stopTime`, stream logged
  signals, exit.
- **Debugger attached** — pause at `t = startTime`; stream DAP
  `stopped` events tagged `reason: "entry" | "breakpoint" |
  "step" | "pause" | "crossing"`; accept the §10 request set.

Internally: `SignalFlowLowering` → emit C++ → clang → `dlopen`
into the matlabc process — the same trick `-repl` / `-dap` use, so
there's no separate build step for the user.

## 9. The `-emit-mflowlink-cpp` codegen lane

A standalone-executable lane — bake a model into a deployable
simulator. Emits one `.cpp` plus a small `main()` linking
`runtime_mflowlink.cpp`; no DAP server, just the simulation loop
and a CSV / JSON dump of `to_workspace` / `log_signal` outputs.
Lower priority than `-simulate` — that's the interactive path the
IDE drives.

## 10. DAP protocol extensions

Handled in `tools/matlabc/main.cpp`, extending the existing `-dap`
request dispatch (which already handles `setBreakpoints`,
`continue`, `next`, `stepIn`, `evaluate`, …). New verbs, gated to
`Simulate` mode:

```jsonc
// Stepping
{ "command": "stepMajor",     "arguments": { "threadId": 1 } }
{ "command": "stepBlock",     "arguments": { "threadId": 1 } }
{ "command": "stepBackMajor", "arguments": { "threadId": 1 } }
{ "command": "stepBackBlock", "arguments": { "threadId": 1 } }

// Signal breakpoints (needs the IDE's edge `breakpoint` field —
// see §4; until then, accept but treat as no-op)
{ "command": "setSignalBreakpoints",
  "arguments": { "source": { "path": "model.mflow" },
                 "breakpoints": [ { "edgeId": "e7",
                                    "condition": "abs(value) > 1e3" } ] } }

// Time breakpoints
{ "command": "setTimeBreakpoints",
  "arguments": { "times": [ { "t": 5.0 },
                            { "t": 7.5, "condition": "x > 0" } ] } }

// Reset to t = startTime, clear snapshots
{ "command": "resetSimulation" }

// Live solver tuning while paused
{ "command": "configureSolver",
  "arguments": { "relTol": 1e-4, "maxStep": 0.01 } }
```

Events streamed up:

```jsonc
{ "event": "simulationTime",        "body": { "t": 1.234, "majorStep": 1234 } }
{ "event": "simulationActiveBlock", "body": { "nodeId": "gain_1" } }
{ "event": "signalSample",          "body": { "edgeId": "e3", "t": 1.234, "value": 0.42 } }
{ "event": "zeroCrossing",          "body": { "nodeId": "sat_1", "t": 1.234 } }
{ "event": "snapshotTaken",         "body": { "majorStep": 1234, "depth": 47 } }
```

Throttle `simulationActiveBlock` / `signalSample` to a max rate so
a fast run doesn't flood the channel. The IDE consumes these
per-window (each mflowLink window owns its own DAP session).

## 11. Build-script wiring

`runtime/build_and_run.sh` already auto-detects Symbolic Math and
plotting usage to pick extra `.cpp` files. Either add the same for
signal-flow `.mflow` inputs, **or simply add
`runtime_mflowlink.cpp` to the unconditional `RUNTIME_SRCS`** — it
has no external dependencies, so the unconditional one-liner is
cleanest (the treatment `runtime_rf.cpp` already got).

## 12. Reference examples + CTest

Land under `examples/mflowlink/` (the repo already has per-toolbox
`examples/` dirs):

| Example | Exercises |
|---|---|
| `lowpass.mflow` | Sine → Gain → Transfer Fcn → Scope — the continuous path end-to-end |
| `pid_tracking.mflow` | Step → Sum → PID (Gain+Integrator+Derivative) → Plant → Scope, closed loop — algebraic-loop avoidance |
| `multirate.mflow` | continuous → Rate Transition → 10 ms discrete controller → ZOH → continuous plant — the scheduler |
| `bouncing_ball.mflow` | free fall + a zero-crossing on position, reset via discrete event — zero-crossing + reset |

Each doubles as a CTest fixture (mirror the `scripts/va_*.sh` +
opt-in CTest lane pattern from `verilog_a_plan.md`) so regressions
show in CI. A `-simulate --dry-run` that prints the sorted
execution order makes a cheap Tier-2 smoke lane.

## 13. Per-block parameter catalogue — `docs/mflowlink_blocks.md`

**The real schema contract between the two repos.** The IDE's
`SignalFlowParamSpec` catalogue pins down which `params` keys each
`signal_*` kind expects:

- `signal_gain` → `gain`
- `signal_sine` → `amplitude`, `bias`, `frequency`, `phase`
- `signal_step` → `step_time`, `initial_value`, `final_value`
- `signal_transfer_fcn` → `num`, `den`
- `signal_state_space` → `A`, `B`, `C`, `D`, `x0`
- `signal_sum` → `signs` (e.g. `"+-"`)
- `signal_integrator` → `initial_condition`
- `signal_saturation` → `upper_limit`, `lower_limit`
- … (full list in the IDE's `SignalFlowParamSpec`)

matlabc's per-block evaluators (§7.7) must read the *same* keys.
Create `docs/mflowlink_blocks.md` as that catalogue and keep it
lockstep with `SignalFlowParamSpec` — without a single source of
truth, the IDE and the runtime silently disagree on parameter
names and a model "runs" with wrong numbers.

## 14. Tier plan

Each tier ends with a working demo + CTest lane. The IDE-side
work for Tiers 1–8 is largely shipped already — these tiers track
the **matlab_llvm** half.

### Tier A — Schema acceptance  *(this doc §5)*

`Loader.cpp` accepts `settings.kind` + `solver` + `snapshot`, the
`signal_*` kinds, and the signal-attribute `data` fields.
Round-trips the full reserved set. `flowchart_schema.md` gets a
"Signal-flow extensions" section.

**Demo:** `matlabc -dump-flow model.mflow` on a signal-flow file
prints the parsed block graph without error.

### Tier B — Lowering + static analysis  *(§6)*

`SignalFlowLowering.cpp` builds the `MflowLinkModel` IR: subsystem
flattening, execution-order sort, algebraic-loop validation, state
counts, zero-crossing table.

**Demo:** `matlabc -simulate --dry-run model.mflow` prints the
sorted execution order; a deliberate algebraic loop is rejected
with a sourced diagnostic.

### Tier C — Continuous-time simulation MVP  *(§7 + §8)*

`runtime_mflowlink.cpp` with the Tier-3 evaluator set, continuous
blocks only, wrapping `ode45`. `-simulate` runs to `stopTime` and
dumps logged signals as CSV.

**Demo:** `lowpass.mflow` — sine in, filtered sine out, CSV
matches a hand-computed reference column-by-column.

### Tier D — DAP stepping + snapshot ring  *(§8 + §10)*

DAP-server mode: entry pause, `stepMajor` / `continue` / `pause`,
the snapshot ring + `stepBackMajor`. The IDE's transport row goes
live (it flips `simulationRuntimeAvailable` true).

**Demo:** `pid_tracking.mflow` paused at entry; step to t=2 s,
step back 3 major steps, resume — the IDE shows the active-block
halo and live signal values.

### Tier E — Discrete + multirate  *(§7.1)*

The sample-time scheduler; Unit Delay / ZOH evaluators;
block-level stepping (`stepBlock` / `stepBackBlock`); zero-crossing
for Switch / Saturation / Relay.

**Demo:** `multirate.mflow` + `bouncing_ball.mflow` both run; the
IDE can step block-by-block through one major step.

### Tier F — Signal & time breakpoints  *(§10)*

`setSignalBreakpoints` / `setTimeBreakpoints` handlers (needs the
IDE's edge `breakpoint` schema field landed first). Conditional
subsystems (Enabled / Triggered / Function-Call).

### Tier G — Code-gen lane  *(§9)*  ✓ shipped

`matlabc -emit-mflowlink-cpp model.mflow` produces a self-contained
C++ source file that embeds the .mflow JSON as a raw string literal,
includes the matlab_llvm Flowchart headers, and has a `main()` which
loads the embedded model through the existing `loadMflow` /
`lowerSignalFlow` machinery, runs `MflowLinkSim::runToCompletion`,
and dumps the logged-signal CSV to stdout. The user compiles it
against the matlab_llvm Flowchart static libs via
`runtime/build_mflowlink.sh` (or any equivalent C++17 command line)
to produce a deployable simulator that does **not** need the
original .mflow at runtime.

The CTest lane `flowchart-emit-mflowlink-cpp-tests` round-trips every
shipped example .mflow through emit → compile → run and diffs the
output CSV byte-for-byte against `matlabc -simulate` on the same
input — proof that the standalone codegen lane reproduces the
in-process interpreter exactly.

**Cross-dialect composition** — the `-emit-cpp` lane learning to
embed `runtime_mflowlink` for a control-flow `.m` that calls into a
`signal_flow` sub-`Flow` — is left to a follow-up PR. It needs a
linker step that pulls `MflowLinkSim` into the regular MATLAB-program
compile path, plus a call-site convention for the host MATLAB code
to invoke a baked simulation.

### Tier H — Reserved-kind expansion

Evaluators for the §5.2 *reserved* kinds, as demand surfaces:
chirp/noise sources, discrete filters, lookup tables, bus
creator/selector, MATLAB Function block.

## 15. Risk register

| Risk | Mitigation |
|---|---|
| ODE-wrapper impedance mismatch — the solvers were built for `ode45(@f, tspan, y0)`, not a per-partition step | Expose a lower-level `step(state, t, h, deriv) -> (next, err)` entry from the existing solver core; `runtime_pde.cpp` already needed something close |
| Algebraic-loop solver non-convergence on pathological models | Default `trust_region`; surface non-convergence as a paused DAP state with the offending block list, not a hard crash |
| Zero-crossing chatter near a saturation rail | Adaptive crossing threshold + a configurable max bracketing-iteration cap |
| IDE/runtime `params`-key drift | §13 single-source catalogue (`mflowlink_blocks.md`); a CTest that diffs it against the IDE's `SignalFlowParamSpec` |
| Snapshot ring memory on big models | Configurable depth; `"states"`-only mode recomputes outputs on restore |
| Subsystem flattening blows up on deep nesting | Recursion depth cap + a subsystem-cycle check (a subsystem referencing an ancestor flow) |
| DAP event flood on a fast run | Throttle `simulationActiveBlock` / `signalSample` to a max rate per second |

## 16. Open questions

- **Implicit / stiff solver.** `ode23s` (Rosenbrock) ships;
  whether mflowLink also wants a BDF (`ode15s`-style) option is a
  Tier-E+ call once stiff demos surface.
- **Per-flow solver overrides.** Simulink allows model-reference-
  local solver settings. Punt until multi-document composition
  (Tier G) lands.
- **Edge `signal` metadata.** §10's `setSignalBreakpoints` and
  §6.1's per-edge signal typing both want the IDE's `FlowEdge` to
  carry a `signal` block (designed in the IDE roadmap §4.4, not
  yet built). Until it lands, type/units come from the source
  block. This is an **IDE-side prerequisite**, flagged here so the
  matlabc work doesn't assume it.
- **Bus signals.** First-class struct-typed signals
  (`signal_bus_creator` / `signal_bus_selector`) need a composite
  signal type in the IR — Tier-H, once the scalar path is solid.
