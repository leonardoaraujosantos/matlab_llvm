# Stateflow / mStateflow — Tutorial

`mStateflow` is a hierarchical, event-driven state-chart frontend that
reuses the `.mflow` JSON container as a third dialect
(`settings.kind = "state_chart"`, alongside `control_flow` and
`signal_flow`). A chart is loaded, lowered to an AST, and then runs
through **every** existing `matlabc` emit lane: software targets
(`-emit-matlab`/`-emit-c`/`-emit-cpp`/`-emit-llvm`), a C++ interpreter
(`-simulate`) with DAP live-debug, and **synthesizable
SystemVerilog** (`-emit-systemverilog`). The integer-typed
Moore/Mealy/AND-parallel examples produce verilator-lint-clean modules.

## Supported features

State-chart constructs (see [`../mStateflow_roadmap.md`](../mStateflow_roadmap.md)):

- **States** with `entryAction` / `duringAction` / `exitAction` and
  `onEventActions`.
- **Hierarchy**: OR-decomposition (`decomposition: "or"`) and
  AND-decomposition (parallel regions with execution order), nested
  substates via `parent`, default substates (`isInitial`).
- **Transitions** with a four-field label `event[guard]{condAction}/
  transAction`, priorities, inner transitions, super-transitions
  across hierarchy, and history-junction-aware entry.
- **Junctions** for flow-graph routing.
- **Data symbols** (`input` / `output` / `local`) with type and
  initial value; **events** with rising-edge triggers.
- **Moore** (outputs from state entry actions) and **Mealy** (outputs
  from a transition's condition action) styles.
- Super-step semantics: a fixed-point loop with iteration saturation;
  the `in(<state>)` predicate is available in actions.

Emit targets / lanes:

- **Software lowering** (default): one `<chart>_tick(in_X…, ev_E…) →
  out_Y…` function backed by **persistent scalars** + integer-indexed
  regions — no `struct()`, no string literals — so the MATLAB → LLVM
  lane lowers it cleanly through every emit-* mode.
- **SystemVerilog lowering** (under `-emit-systemverilog` /
  `-check-synthesizable` / `-emit-hardware-report` / `-emit-cocotb`):
  per-variable `if isempty(X), X = intW(0); end` resets, integer-typed
  locals (cast through `int16(...)`), one transition attempt per
  region per call (one clock edge = one `chart_tick`), inlined `in()`
  predicate, auto-injected `reset`/`clk`.

## Build & emit

The chart `.mflow` files live in `examples/stateflow/`. The lanes:

```sh
# Inspect the resolved chart IR (validation + structural dump).
matlabc -dump-chart   examples/stateflow/traffic_light_moore.mflow

# Lower to readable MATLAB source.
matlabc -emit-matlab  examples/stateflow/traffic_light_moore.mflow

# Lower to C / LLVM / MIR (via the emitted MATLAB).
matlabc -emit-c       examples/stateflow/traffic_light_moore.mflow > tl.c

# Run via the C++ interpreter (deterministic event trace).
matlabc -simulate     examples/stateflow/traffic_light_moore.mflow

# Live-debug + REPL via DAP.
matlabc -simulate --sim-dap examples/stateflow/traffic_light_moore.mflow

# Synthesizable SystemVerilog (verilator --lint-only clean).
matlabc -emit-systemverilog examples/stateflow/traffic_light_moore.mflow > tl.sv

# Native execution as C, end-to-end.
matlabc -emit-matlab examples/stateflow/traffic_light_moore.mflow \
  | matlabc -emit-c /dev/stdin > tl.c && cc tl.c -lm -o tl && ./tl
```

In the REPL, a one-call load emits + sources in one step:

```
matlabc -repl
>> loadStateChart('examples/stateflow/traffic_light_moore.mflow')
>> traffic_light_tick(...)   % the chart's tick fn is now callable
```

The SV lane produces the standard `(clk, rst_n, inputs..., outputs...)`
surface with an `always_comb` block for next-state logic and an
`always_ff @(posedge clk or negedge rst_n)` block for the state
registers. Integer-only data is required for synthesizable output.

## Worked examples

### Moore traffic light (`examples/stateflow/traffic_light_moore.mflow`)

A three-state OR-decomposed chart (`Red` default → `Green` → `Yellow`
→ `Red`). Each state's `entryAction` sets the three outputs, so
outputs depend **only on the active state** — classic Moore. The
`.mflow` structure:

- `symbols.data`: `red`/`yellow`/`green` (`output int32`), `timer`
  (`local int32`); `symbols.events`: `step` (rising).
- Three `state` nodes, e.g.
  `"entryAction": "red = 1; yellow = 0; green = 0; timer = 0"`, with
  `params.decomposition = "leaf"` and the initial state tagged
  `isInitial: "true"`.
- Three `transition` edges labelled `step[timer >= 30]`,
  `step[timer >= 25]`, `step[timer >= 5]` (event + guard), each with a
  `priority`.

Integer-only data means `-emit-systemverilog` yields a synthesizable
module; `-simulate` walks the event trace, advancing one transition
per `step` event.

### Mealy vending machine (`examples/stateflow/vending_machine_mealy.mflow`)

Three states (`Idle`/`Coin1`/`Coin2`) accumulate coins on the `coin`
event. The single output `dispense` is driven by a **transition's
condition action**, not a state entry — `select/dispense = 1` on the
`Coin2 → Idle` edge fires only when in `Coin2`, so the output depends
on state **and** the incoming event (Mealy). The `reset` event returns
to `Idle` via lower-priority edges (`priority: "2"`). This contrasts
directly with the Moore traffic light whose outputs are
state-resident.

### Get-started battery chart (`examples/stateflow/get_started_create_chart.mflow`)

Mirror of the MathWorks "Create a Stateflow Chart" tutorial: a
two-state model `Charge` (default) ↔ `Discharge`, guarded on the
`isCharging` boolean input. `Charge` entry sets `sentPower = 0` and
its during action grows `charge` by 4 per super-step; `Discharge`
entry sets `sentPower = 3.5` and drops `charge` by 3. This one uses
`double` data, so it targets the software lanes.

### Hierarchy chart (`examples/stateflow/get_started_hierarchy_chart.mflow`)

Extends the battery chart with substates via `parent`. `Charge` is OR-
decomposed into `FastCharge` (default, +4/tick) → `SlowCharge` (+1)
when `charge > 80` → `Full` when `charge == 100`. `Discharge`
decomposes into `Powered` (default, −3/tick) → `Empty` (sets
`sentPower = 0`) when `charge <= 3`. Top-level `Charge`↔`Discharge`
transitions remain. Demonstrates nested-state default entry and
substate-to-substate transitions.

### AND-parallel air-temperature controller (`examples/stateflow/model_air_temperature_controller.mflow`)

OR-decomposed root `PowerOff` (default) / `PowerOn`. `PowerOn`
AND-decomposes into three **parallel** regions with execution order:
`FAN1` (Off↔On at 120 °F), `FAN2` (Off↔On at 150 °F), and
`SpeedValue` — a leaf whose during action sums the `in(FAN1_On) +
in(FAN2_On)` predicates into the `airflow` output. This exercises the
`in()` predicate and parallel-region execution order. Drive it under
DAP:

```
> stateChart/setLocal { name: "power", value: 1 }
> stateChart/setLocal { name: "temp",  value: 130 }
> stateChart/emit     { name: "tick" }
> stateChart/stepSuperStep
< airflow == 1            (FAN1 on)
> stateChart/setLocal { name: "temp", value: 160 }
> stateChart/emit { name: "tick" }
> stateChart/stepSuperStep
< airflow == 2            (FAN1 + FAN2 on)
```

The DAP surface (`stateChart/setLocal`, `stateChart/emit`,
`stateChart/stepSuperStep`, breakpoints, snapshot ring) is wired
end-to-end for live introspection.

## Limitations & carve-outs

- **Integer-only data is required for the SV lane.** Charts using
  `double` data (`get_started_create_chart`,
  `get_started_hierarchy_chart`) target the software lanes only;
  Moore/Mealy/AND-parallel integer charts are the synthesizable set.
- The SV lowering performs **one transition attempt per region per
  call** — one `chart_tick` models one clock edge, not a full
  super-step. (The software lowering does run the super-step
  fixed-point loop.)
- The `in()` predicate is **inlined** in SV mode (no helper function)
  to bypass call-site type inference.
- The schema (`settings.kind`, `FlowNode.parent`, `data.params.*`,
  transition labels) is documented in
  [`../flowchart_schema.md`](../flowchart_schema.md); parent
  resolution, parent cycles, AND-execution-order, and
  default-transition multiplicity are validated at load.
- The chart dialect rides on the same window/DAP/snapshot
  infrastructure as the signal-flow sibling; the UI/UX surface lives
  in the IDE, not in `matlabc`.

## See also

- mStateflow roadmap (semantics, lowering targets, shipped tiers): [`../mStateflow_roadmap.md`](../mStateflow_roadmap.md)
- Flowchart frontend architecture: [`../flowchart_frontend.md`](../flowchart_frontend.md)
- `.mflow` schema reference: [`../flowchart_schema.md`](../flowchart_schema.md)
- Examples + per-chart notes: `examples/stateflow/` (and its `README.md`)
