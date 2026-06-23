# mflowLink — Signal-Flow Simulation Backend (matlab_llvm side)

Plan for the **compiler + runtime** support behind *mflowLink*: a
Simulink-like, time-domain block-diagram simulation layer built on
top of the existing `.mflow` graphical frontend. Where today's
`.mflow` pipeline lowers a *control-flow* diagram to a structured
AST (`docs/flowchart_frontend.md`), mflowLink adds a second
**signal-flow** dialect that lowers to a simulation IR and runs
through a new ODE-driven runtime.

**Status: Tiers A–H shipped + Items 1, 2, 4, 5 of the §17.5
next-horizon priority list (2026-05-14).** The `.mflow` loader
parses `settings.kind` and signal-flow attributes, `MflowLinkSim`
runs continuous + multirate + zero-crossing simulations with
step / step-back / block-stepping via the snapshot ring,
`matlabc -simulate --sim-dap` boots a DAP server with time + signal
breakpoints, `matlabc -emit-mflowlink-cpp` produces a deployable
standalone simulator that matches `-simulate` byte-for-byte across
every shipped demo, vectors propagate through Mux / Demux /
element-wise math with sample-time inheritance, an adaptive
Dormand-Prince 5(4) integrator drives the continuous step under
PI step-size control, the algebraic-loop solver iterates
direct-feedthrough cycles to convergence, masked subsystems
clone library flows with `${name}` parameter substitution, and
the MATLAB Function block accepts a full
`function y = f(u1, u2); ... end` body parsed via the existing
matlab_llvm lexer + parser and walked by a scalar AST
interpreter. 12/12 flowchart-* CTest lanes pass, 62 SimulateRun
analytic checks green, 15 demos round-trip byte-identical through
the standalone codegen lane.

The shipped surface is summarised in §17.1, what we cut along the
way in §17.2, the still-blocked carve-outs in §17.3, the bigger
gap to "real Simulink" in §17.4, and the **updated** priority for
the next chunk of work in §17.5.

This doc is the concrete compiler-side plan; the **IDE-side**
authoring surface is documented in `Matlab_llvm_ide/docs/
mflowLink_roadmap.md`.

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

## 17. Next horizon — the gap to Simulink

This section captures the honest delta between *shipped mflowLink*
and *real Simulink* as of the close of Tier H. The roadmap above
(§1–§16) was always a **subset**; this section names the rest so
future work can be planned, sliced, and scoped without
rediscovering each gap.

### 17.1 Where we are — the scorecard

Every named tier in the roadmap landed, plus two of the six Tier-H+
carve-outs and four of the five §17.5 next-horizon items:

**Tiers A–H:**

| Tier | Item | Status | CTest lane |
|---|---|---|---|
| A | schema acceptance | ✓ | `flowchart-tests` |
| B | `SignalFlowLowering` + `MflowLinkModel` IR | ✓ | `flowchart-simulate-tests` |
| C | continuous sim MVP + `-simulate` | ✓ | `flowchart-simulate-run-tests` |
| D | DAP step / step-back / snapshot ring | ✓ | `flowchart-simulate-dap-tests` |
| E | multirate scheduler, ZC, Unit Delay / ZOH / Relay, block stepping | ✓ | `flowchart-simulate-dap-block-tests` |
| F | time + signal breakpoints, Enabled / Triggered / Function-Call subsystems | ✓ | `flowchart-simulate-dap-breakpoint-tests` |
| G | `-emit-mflowlink-cpp` standalone codegen | ✓ | `flowchart-emit-mflowlink-cpp-tests` |
| H | 21 of 23 reserved-kind evaluators + `signal_goto`/`from` + `signal_matlab_fcn` | ✓ | (folded into the lanes above) |

**§17.5 next-horizon items shipped:**

| # | Item | Status | Commit |
|---|---|---|---|
| §17.5 #1 | Vector signals + sample-time inheritance | ✓ | `8cbdf7a` |
| §17.5 #2 | Algebraic-loop solver + adaptive ODE (DOPRI5) | ✓ | `0091a2d` |
| §17.5 #4 | Block masks + library blocks | ✓ | `a2a0d49` |
| §17.5 #5 | Real MATLAB Function block (interpreter MVP) | ✓ | `7a69f96` |

**12/12 flowchart-* CTest lanes, 62/62 SimulateRun analytic checks,
15 demos round-trip byte-identical through the standalone codegen
lane.**

Demos that exercise the next-horizon work: `vector_signals.mflow`,
`sample_time_inherit.mflow`, `algebraic_loop_solved.mflow`,
`masked_library.mflow`, `matlab_function_block.mflow`.

### 17.2 Deferred deviations — places we cut a shortcut

These are decisions inside shipped tiers where the implementation
took a deliberately smaller path than the roadmap text. Each is
internally documented in the source; collected here for visibility.
Items marked ✓ closed since the original list was written.

| Gap | Status | Where I cut | What it unlocked / would unlock |
|---|---|---|---|
| Adaptive ODE wrapping | ✓ closed | §7.2 — was fixed-step RK4 | Dormand-Prince 5(4) with PI step-size control now runs when `settings.solver.type == "variable_step"` (Item 2 / `0091a2d`). Fixed-step RK4 kept as the `fixed_step` fallback |
| Algebraic-loop *solver* | ✓ closed | §7.4 — was hard-error at lowering | Fixed-point Picard iteration each step, tolerance keyed off `settings.solver.relTol`; non-convergence queued via `consumeAlgebraicLoopFailures` (Item 2 / `0091a2d`). The `"off"` method preserves the hard-error path |
| Vector signal type | ✓ closed | Tier H carve-out justification | Per-block `OutWidth` + `VecOut_`; element-wise broadcast for Gain / Sum / Product / Abs / Saturation; Mux concatenates, scope logs per-element columns; sample-time inheritance walks topo order (Item 1 / `8cbdf7a`) |
| `signal_transfer_fcn` output scaling | ✓ closed | Pre-existing bug — output divided by `Lead` (leading denominator coefficient) when it shouldn't have. Masked by every demo with `den = "1, ..."` (Lead = 1) | Surfaced by the masked-library demo (τ = 0.1 → 10× output overshoot); fixed in the Item 3 chain. Same path covers `signal_zero_pole` |
| BDF stiff solver | ◐ partial | §16 — `ode15s` ships as **BDF1** (Backward Euler) with a real Newton corrector, finite-difference Jacobian, and dense LU (`bdf1Step`); fixed-step only. The full variable-order/variable-step suite (`ode15s` BDF1–5, native `ode23`, `ode23s`/`ode23t`/`ode23tb`, mass-matrix/index-1 DAE, dense output) is scoped in OpenSpec change `mflow-variable-step-stiff-solvers` | `ode15s`-style implicit integration for stiff chemistry / electronics / thermal models |
| Per-flow solver overrides | ○ open | §16 open | Model-reference flows with different solver settings than the parent |
| Cross-dialect composition | ○ open | §9 follow-up | `-emit-cpp` linking `runtime_mflowlink` so a `.m` script can call into a baked signal-flow simulation |
| `bouncing_ball.mflow` demo | ✓ done | §12 example carve-out | `signal_integrator` external `reset`/`init` ports ship; `examples/mflowlink/bouncing_ball.mflow` is a SimulateRun fixture (zero-crossing → state-reset, energy-dissipating bounces) |
| Discrete filter — FIR path | ○ open | `signal_discrete_filter` ships the pole half of direct-form-II only | Pure-FIR designs need a `u`-history buffer (taps on the input side) |
| Backward Euler / Trapezoidal | ○ open | `signal_discrete_integrator` parses the method param but uses the Forward Euler single-sample approximation | True implicit / averaged discrete integration |
| MATLAB Function block — JIT | ✓ closed | §17.5 #8 — `tools/matlabc/MflowLinkJit.cpp` synthesises a one-level wrapper (driver + `mflowlink_jit_entry` shim + user body), runs the full lex/parse/Sema/MLIR/LLVM-ORC pipeline, casts the resolved entrypoint to a flat `(double, ...) → double` function pointer. Bodies the wrapper can't refine (e.g. triple-helper chains, `n`-as-loop-bound) fall back to the AST interpreter automatically | Demo: `examples/mflowlink/matlab_fcn_jit.mflow` |
| Vector signals — true matrix shapes | ✓ closed | §17.5 #9 — `OutRows × OutCols` carried on every `MflBlock`; lowering's shape-inference pass propagates the dominant non-scalar shape, broadcast operators stay flat-storage element-wise. New `signal_reshape` block + matrix-literal parsing on `signal_constant` (`"[1 2; 3 4]"`). Scope / to-workspace columns switch to `<id>[r,c]` for 2-D outputs | Demo: `examples/mflowlink/matrix_signals.mflow` |

### 17.3 Blocked carve-outs — five Tier-H+ kinds still reserved

Each is accepted by the loader (round-trips clean) but rejected at
lowering. They need prerequisites outside the lone "ship more
evaluators" axis:

| Kind | Blocker |
|---|---|
| `signal_bus_creator`, `signal_bus_selector` | Vector / struct signal type in the IR. Today every wire is scalar `double` — a bus needs a composite signal carrier plus per-port type checking. |
| `signal_from_workspace` | Mechanism to bind a runtime `simout`-style variable into the simulation as a time-indexed source. The matlabc workspace model and the mflowLink runtime share no such handle today. |
| `signal_custom` | Plugin registry + ABI design. Now distinct from the shipped `signal_matlab_fcn`: where `signal_matlab_fcn` covers the inline-expression case, `signal_custom` would let the IDE register evaluators implemented elsewhere (a `.cpp` the user compiles into the runtime, a remote service, …). |
| `signal_if_action`, `signal_switch_case_action` | A parent `signal_if_subsystem` / `signal_switch_case_subsystem` container that scopes which case fires. The IDE's `SignalFlowParamSpec` doesn't ship these containers yet — IDE-side prerequisite. |
| `signal_lookup_nd` | Settle the `lookup_1d` / `lookup_2d` pair (breakpoint shape + extrapolation policy + cached table format) before generalising to N dimensions. |

### 17.4 Bigger gap — what real Simulink does that mflowLink doesn't

mflowLink was always **a subset** of Simulink, not a replacement.
This sub-section is the honest census of features Simulink users
expect that aren't on the current roadmap at all.

#### 17.4.1 Core simulation semantics

| Item | What Simulink does | Today in mflowLink |
|---|---|---|
| **Vector signals** *(partial)* | First-class N-D signals on every wire | ✓ 1-D vectors with scalar-broadcast through element-wise math + Mux concat; matrix shapes still TODO |
| **Sample-time inheritance** | `-1` inherits from upstream, `auto` picks, colour-coded on the canvas | ✓ Fixpoint inheritance pass; downstream block adopts the fastest upstream period |
| **Algebraic-loop solver** | Newton / trust-region per `algebraicLoopMethod` | ✓ Fixed-point Picard iteration each step under `relTol`; non-convergence surfaces as a paused DAP state |
| **Variable-step solver** | True adaptive integration with error control | ✓ Dormand-Prince 5(4) with PI step-size control when `type == "variable_step"`; fixed-step RK4 fallback |
| **Bus signals (structs)** | Nested named-field signals; in-place type inference | ○ Tier-H+ carve-out — needs composite signal type |
| **Matrix-shaped signals** | First-class 2-D + N-D wires with per-dimension width | ○ Tier-I+ — extends the 1-D vector path |
| **Complex numbers** | Native complex-valued signals | ○ Real `double` only |
| **Fixed-point arithmetic** | Q-format with overflow handling, FxP-aware codegen | ○ Floating-point only |
| **Variable-size signals** | Array sizes change at runtime; max-size declared at design time | ○ Fixed at construction |
| **Stiff / implicit solvers** | `ode15s`, `ode23t`, `ode23tb`, fixed-step `ode4` / `ode5` / Heun | ◐ DOPRI5 (`ode45`) + RK4 + **`ode15s` BDF1** (Newton + FD Jacobian + dense LU, fixed-step); `ode23` aliases DOPRI5, `ode23s`/`ode23t`/`ode23tb` fall through to RK4. Full variable-order/variable-step suite + native `ode23` + Rosenbrock/TR + mass-matrix DAE + dense output scoped in OpenSpec `mflow-variable-step-stiff-solvers` |
| **Frame-based processing** | Process N samples per tick (DSP convention) | ○ One sample per tick |

#### 17.4.2 Authoring features Simulink users expect

| Item | Status | What Simulink does |
|---|---|---|
| **Block masks** | ✓ | A user-visible block with hidden internals plus a custom icon and parameter dialog — schema side shipped (Item 3 / `a2a0d49`); IDE-side mask UI still upstream |
| **Library blocks + linking** | ✓ | A referenced library that updates across every model — `kind: "library"` flow accepted; one library cloned into N masked instances each with their own `mask_params` |
| **Mask parameters** | ✓ | Same library block reused with different per-instance params — `${name}` placeholder substitution per Item 3 |
| **Goto tag visibility** | partial | `signal_goto` / `signal_from` shipped with global tag scope; local / scoped tag namespaces still TODO |
| **Model reference** | ○ | `simulink.ModelReference` — one model called by another, with its own solver. Closest we have is library flows; per-model solver still TODO |
| **Signal labels + propagation** | ○ | Hover, label propagation, multi-line scope plots |
| **Annotation blocks** | ○ | Free-form text / images on the canvas |
| **Signal tracing** | ○ | Click a signal → trace it through the model |
| **Model navigation** | ○ | Breadcrumbs, hyperlinks between subsystems |
| **Subsystem masking UI** | ○ | Custom icon / dialog hiding the subsystem internals (IDE-side prerequisite) |

#### 17.4.3 Block kinds Simulink has that mflowLink doesn't fully cover

| Category | Status | What Simulink does |
|---|---|---|
| **MATLAB Function block** | ✓ MVP | Item 4 / `7a69f96` — `function y = f(u1, u2); ... end` parsed by the matlab_llvm lexer + parser, walked by a scalar AST interpreter. Supports assignment, if/else, math/trig builtins, scalar locals. Loops / multi-return / vectors / user-function calls still TODO (would route through the matlab_llvm MIR/MLIR/JIT pipeline) |
| **Function-Call subsystems** | partial | Event-driven via `signal_triggered_subsystem` + `signal_function_call_generator` (Tier F). Discrete-event priority + multi-call-source orchestration still TODO |
| **Stateflow** | ○ | Hierarchical state machines (separate roadmap, called out as non-goal in §2) |
| **For-Each / Iterator subsystems** | ○ | Vector iteration with local state |
| **S-Function** | ○ | Compiled C / Fortran / MATLAB user blocks with full lifecycle hooks (closest carve-out: `signal_custom`, blocked on plugin ABI) |
| **Discrete filters (real DSP)** | ○ | Biquad, FIR Decimation / Interpolation, CIC, polyphase |
| **Continuous filters with reset / limits** | ○ | Variable transport delay, integrator with limits + external reset port (also surfaces in the `bouncing_ball.mflow` carve-out) |
| **Logic & bit operations** | ○ | Bitwise AND / OR / XOR / shift, bit-packed counters |
| **Lookup tables (advanced)** | partial | `signal_lookup_1d` / `_2d` shipped; N-D + prelookup + `Interpolation-Using-Prelookup` pattern still TODO |
| **Sources** | partial | Constant / Step / Sine / Pulse / Ramp / Chirp / Clock / Noise shipped. Repeating sequence, signal builder, from-file still TODO |
| **Sinks** | partial | Scope / Display / To-Workspace / Terminator shipped. XY graph, To-File, Floating Scope still TODO |
| **Variant subsystems / model variants** | ○ | Conditional model assembly at compile time |
| **Discrete events** | ○ | Function-call subsystems with priority, event-driven scheduling |

#### 17.4.4 Tooling Simulink has

- **Simulation Data Inspector (SDI)** — multi-run comparison,
  tolerances, baselines.
- **Linearization** — compute Jacobians around an operating
  point, generate state-space models from a Simulink model.
- **Linear Analysis** — Bode / Nyquist / step response, stability
  margins.
- **Optimization tools** — parameter tuning, Response Optimization.
- **Coverage** — decision / condition / MC/DC coverage of model
  paths.
- **Test framework** — Simulink Test, baseline tests, equivalence
  tests.
- **Verification & Validation** — requirements links,
  traceability, formal verification (Simulink Design Verifier).
- **Real-Time Workshop / External Mode** — hardware-in-the-loop,
  tuning at runtime.
- **Simulink Coder / Embedded Coder** — production C / C++
  generation with timing constraints, AUTOSAR, fixed-step real-
  time hooks. (Our Tier G is "host simulator only".)
- **PIL / SIL** — processor-in-the-loop, software-in-the-loop.
- **Simscape** — physical multi-domain modelling (called out as a
  non-goal in §2; mechanical / electrical / hydraulic / thermal
  networks with across-and-through variables and modified nodal
  analysis).
- **HDL Coder** — Simulink → Verilog / VHDL.
- **Parallel simulation** — `parsim`, Fast Restart, batch sweeps.

### 17.5 Suggested priority for the next horizon

The original list at this slot ranked 10 items; **four are
shipped** (#1, #2, #4, #5 — see §17.1 for the commits). The
remaining work, re-ranked by impact-per-effort given the new
foundation:

#### Shipped already (kept for trace)

| Original # | Item | Commit |
|---|---|---|
| 1 | Vector signals + sample-time inheritance | `8cbdf7a` |
| 2 | Algebraic-loop solver + adaptive ODE | `0091a2d` |
| 4 | Block masks + library blocks | `a2a0d49` |
| 5 | Real MATLAB Function block (interpreter MVP) | `7a69f96` |

#### Remaining work — ranked

1. **Bus signals (struct-typed wires)** *(~1–2 weeks)*. Closes the
   `signal_bus_creator` / `signal_bus_selector` Tier-H+ carve-out
   and the most-cited §17.4.1 sim-semantics gap. The per-block
   `OutWidth` + `VecOut_` machinery from Item 1 gives us a
   foundation; bus needs a *composite* signal carrier plus per-
   port type checking. Cleanest design: a new `MflBusType` IR
   record with named fields, edges carry the type reference,
   creator/selector evaluators pack / project at runtime.

2. **`bouncing_ball.mflow` + integrator-reset port** *(~50 LOC)*.
   Cleanup that closes a roadmap §12 demo carve-out and
   demonstrates the zero-crossing → state-reset pattern central
   to physical / discrete-event simulation. New `reset` input
   port on `signal_integrator`; on a rising edge of the reset
   signal, replace the integrator's continuous state with its
   `initialCondition` parameter at the start of the next major
   step.

3. **Implicit / BDF solver (ode15s-style)** *(~1 week)*. Now that
   adaptive DOPRI5 is wired, adding an implicit BDF integrator is
   self-contained. Unlocks stiff models (electronics with widely-
   separated time constants, chemistry kinetics, thermal). Same
   step-size-control infrastructure as DOPRI5; the inner Newton
   step reuses the algebraic-loop solver's iteration shape.

4. **Backward Euler / Trapezoidal for `signal_discrete_integrator`**
   *(~1 day)*. Small correctness fix: we parse the `method` param
   but always use Forward Euler. Needs a sub-sample of the input
   for Backward Euler / Trapezoidal — the scheduler already
   exposes the previous tick's value via the existing
   `Znext_` shadow buffer.

5. **Discrete filter FIR path** *(~1 day)*. `signal_discrete_filter`
   ships the pole half of direct-form-II only. Add a `u`-history
   buffer so pure-FIR designs work; the IIR path already uses a
   `Z_`-style tap shift that generalises cleanly.

6. **Cross-dialect composition** *(~3–5 days)*. The last Tier-G
   follow-up: teach `-emit-cpp` to embed `runtime_mflowlink` so a
   control-flow `.m` script can invoke a baked signal-flow
   simulation. Needs a call-site convention (a magic identifier
   like `mflowlink_run("model.mflow")`) plus the link-time
   inclusion of `MflowLinkSim`.

7. **Per-flow solver overrides** *(~3 days)*. Schema-side
   `Flow.solver` field; lowering picks the nearest enclosing
   solver per block. Mostly Loader changes plus a per-block
   `SolverIndex` in `MflBlock`.

8. **MATLAB Function block — JIT path** *(✓ shipped 2026-05-15)*.
   Two-stage delivery:
   - **Stage A (interpreter loops)** — the Item-4 scalar AST
     interpreter gained for / while / break / continue so users
     can write real MATLAB control-flow inside a
     `signal_matlab_fcn` body. `matlab_fcn_loops.mflow` shows a
     5-term harmonic sum and a Newton-iteration `√(u+1)` solver
     both producing exact analytic answers.
   - **Stage B (true MLIR JIT)** — `MflowLinkSim` now consults
     an injectable `MatlabFcnJit` factory at construction. When
     `matlabc` is the host, `installMflowLinkJit()`
     (`tools/matlabc/MflowLinkJit.cpp`) registers a factory that
     synthesises a wrapper TU, runs the full lex → parse → Sema
     → MLIR → LLVM-ORC pipeline, and resolves
     `mflowlink_jit_entry` as a flat `(double, ..., double) →
     double` function pointer. Bodies the JIT can't compile
     silently fall back to the AST interpreter, so the surface
     keeps the simpler bodies working even when the wrapper
     stalls. Demoed by `matlab_fcn_jit.mflow` (multi-return
     polar decomposition + vector-literal L2 norm, both verified
     in `test/Flowchart/SimulateRun/run_tests.sh`).
   Layering: the JIT factory lives in matlabc, not in
   `MatlabFlowchart` — the static library's dependency closure
   stays MLIR-free. Up to 8 scalar inputs per block; the
   `-emit-mflowlink-cpp` codegen lane still uses the AST
   interpreter (any host that wants JIT can install the same
   factory).

9. **Matrix-shaped signals** *(✓ MVP shipped 2026-05-15)*.
   `MflBlock` now carries `OutRows × OutCols` next to `OutWidth`;
   the lowering's shape-inference pass propagates the dominant
   non-scalar shape through every block (scalar inputs broadcast
   freely; mismatched non-scalar shapes are a sourced
   diagnostic). `signal_constant` accepts the MATLAB matrix
   literal form `"[1 2; 3 4]"`; new `signal_reshape` block
   re-stamps a flat-storage signal as `rows × cols` (element
   count must match — pure metadata pass at runtime). Scope /
   to-workspace columns switch to the `<id>[r,c]` form for 2-D
   blocks; 1-D vectors keep the legacy `<id>[k]` columns so
   existing IDE consumers stay byte-identical. Demo:
   `examples/mflowlink/matrix_signals.mflow` (2×2 constant +
   elementwise broadcast + 1×4 → 2×2 reshape, both byte-
   identical against the `-emit-mflowlink-cpp` standalone).
   Carve-outs: shape-aware Mux/Demux with an `axis` parameter
   (mux/demux still 1-D-row today); matrix multiplication
   block (`signal_matmul`) needs different broadcast semantics
   and will land as a separate slice.

10. **Stateflow** *(separate roadmap, multi-month)*. Hierarchical
    FSMs are 50% of why engineering teams pick Simulink. Out of
    scope for `mflow_link_roadmap.md`; would be a sibling
    `stateflow_roadmap.md`.

11. **Simscape-style physical networks** *(order of magnitude
    more work than the simulation engine itself)*. Non-goal per
    §2 but the single biggest reason engineering teams pick
    Simulink. Modified nodal analysis, across-and-through
    variables, multi-domain network solvers.

12. **Real-Time / PIL / Embedded Coder** *(production-grade
    codegen)*. Builds on Tier G but adds timing constraints,
    target-specific optimisations, AUTOSAR, fixed-step real-time
    hooks. Adjacent to but separate from the simulation surface.
    **Now scoped** in OpenSpec change `mflow-embedded-rt-codegen`
    (§17.7): the deployable bare-metal/RTOS slice (ERT-style
    `model_step` entry points, multirate scheduling, static/MISRA
    C, whole-diagram SV, packaging) on top of the shipped flat
    emit lanes — AUTOSAR/PIL stay out of that first slice.

### 17.6 What this section *isn't*

This isn't a commitment to ship any of §17.4 — it's the honest
horizon so future planning starts from accurate ground. The
shipped surface (§17.1) is internally consistent and tested; the
deferred deviations (§17.2) have working alternatives; the
blocked carve-outs (§17.3) wait on specific prerequisites; the
Simulink gap (§17.4) is the **product** delta, not a bug list.

### 17.7 Active OpenSpec proposals — two Simulink-parity pillars

Two of the §17.4 gaps are now formally scoped as OpenSpec changes
(under `openspec/changes/`), framed as precise **deltas** on top of
what already ships, not greenfield rewrites:

| Change | Pillar | Closes | Starting reality |
|---|---|---|---|
| `mflow-variable-step-stiff-solvers` | Variable-step / stiff solvers | §17.4.1 stiff row, §17.2 BDF row | `ode45` (DOPRI5) + `ode15s` (BDF1, Newton, fixed-step) already ship; adds native `ode23`, variable-order/variable-step BDF1–5, `ode23s`/`ode23t`/`ode23tb`, Jacobian reuse + analytic hook, mass-matrix/index-1 DAE, dense output + `Refine` |
| `mflow-embedded-rt-codegen` | Code generation (Embedded Coder) | §17.5 #12 | Flat AOT C/C++/Python/TS/SV subsystem + whole-diagram emit already ships (Tiers 1–7); adds the real-time wrapper — `model_initialize`/`model_step`/`model_terminate` over a static struct, multirate task scheduling, static/MISRA C profile, whole-diagram SystemVerilog, packaged build bundle |

Both run the same evaluator/emitter in the interpreter and the
compiled `-emit-mflowlink-cpp` binary, so additions are inherited by
compiled models with byte-parity. See each change's `proposal.md` /
`design.md` / `tasks.md` for the incremental landing plan.
