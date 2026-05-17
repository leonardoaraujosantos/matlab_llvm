# mStateflow — Hierarchical State Charts on top of `.mflow` (rev. 2026-05-17)

Companion to:
- [`mflowLink_roadmap.md`](mflowLink_roadmap.md) — the signal-flow sibling
  this rides on. Many of the seams (window model, DAP transport, snapshot
  ring, transport-row chrome) are reused verbatim.
- [`flowchart-editor.md`](flowchart-editor.md) — today's `.mflow`
  control-flow editor inside the IDE.
- [`architecture.md`](architecture.md) — MVVM layering, services,
  isolation rules.
- matlabc `docs/flowchart_schema.md` — on-disk `.mflow` JSON shape.
- Stateflow User's Guide R2026a — semantic reference for parity-where-
  it-matters. Page citations below use the guide's `chapter-page`
  scheme (e.g. §2-29 = chapter 2, page 29).

This document scopes **mStateflow**: hierarchical, event-driven
state-chart authoring + live-debug surface that reuses the `.mflow` JSON
container as a *third* dialect (`settings.kind = "state_chart"`) along
side the existing `control_flow` and `signal_flow` dialects. The
mflowLink roadmap explicitly defers Stateflow as Tier-N+ (see
mflowLink §2); this is that tier.

## Status (2026-05-17)

**Compiler / Debugger / DAP / REPL side: shipped end-to-end. UI/UX
side: Tiers 0 through 10 shipped.** The chart authoring + live-debug
surface is fully live in the IDE — chart `.mflow` files load,
**compile through every matlabc emit lane** (`-emit-matlab` /
`-emit-mir` / `-emit-llvm` / `-emit-c` / `-emit-cpp` for software,
`-emit-systemverilog` for synthesizable RTL), simulate with
deterministic event traces, and stream chart-namespaced DAP events
with full breakpoint + introspection support. The integer-typed
Moore / Mealy / AND-parallel examples in `examples/stateflow/`
produce verilator-lint-clean SystemVerilog modules.

### What's shipped (matlab_llvm side)

- **Tier 0 — Schema + Loader** *(complete)*:
  `lib/Flowchart/Loader.cpp` accepts `settings.kind == "state_chart"`,
  parses every new field (`FlowNode.parent`, `ui.size`,
  `data.params.*`, `data.onEventActions`, `Flow.symbols`, transition
  `data.params` + label), validates parent resolution / parent-cycle /
  AND-execution-order / default-transition multiplicity. Schema bumped
  to 0.2.0; existing control-flow + signal-flow fixtures stay
  byte-stable.
- **Tier 4a — Chart IR**: `include/matlab/StateChart/StateChartIR.h`
  + `lib/StateChart/StateChartIR.cpp`. `Chart` / `ChartState` /
  `ChartJunction` / `Transition` / `ChartFunction` with four-field
  `TransitionLabel` parser. Edit-time lint: undefined-symbol warnings
  on every action / guard / cond / trans body.
- **Tier 4b — Lowering**: `lib/StateChart/Lowering.cpp`. Two target
  forms behind a `LoweringTarget` enum:
  - **Software** (default; drives `-emit-matlab`/`-emit-mir`/
    `-emit-llvm`/`-emit-c`/`-emit-cpp`): single `<chart>_tick(in_X1,
    …, ev_E1, …) → out_Y1, …` function with **persistent-scalar**
    state (no `struct()`, no string literals — every chart slot is a
    flat `persistent l_<local>` / `persistent r_<region>` / etc.
    that matlabc's MATLAB → LLVM lane lowers cleanly).
  - **SystemVerilog** (drives `-emit-systemverilog` /
    `-check-synthesizable` / `-emit-hardware-report` / `-emit-cocotb`):
    per-variable `if isempty(X), X = intW(0); end` reset
    initialisers, integer-typed locals + region codes (`int16(...)`
    casts on every literal), one-pass tick (each call = one clock
    edge / one transition attempt per region), inlined `in()`
    predicate (no helper — bypasses matlabc's call-site type-
    inference loop), reset arg + clk auto-injected by matlabc's SV
    pass.
  Common to both targets: super-step fixed-point loop with
  `kMaxIterations` saturation; on-event handlers; history-junction-
  aware entry; inner transitions; super-transitions across hierarchy
  (LCA-relative exit + entry chains); junction chains (connective /
  history / entry / exit / default); temporal operators (`after` /
  `before` / `at` / `every` — tick / sec aliased); `emit('X')`
  rewriting; identifier-aware action-source rewriter that prefixes
  chart symbols with `l_*` (locals) / `ev_*` (events). The
  integer-typed Moore / Mealy / AND-parallel examples in
  `examples/stateflow/` produce verilator-lint-clean modules
  end-to-end; float-typed charts hard-error at SV emit.
- **Tier 4c — Runtime**: `runtime/runtime_mstateflow.cpp` (snapshot
  ring, DAP event sinks, bounded FIFO event queue with
  configurable depth, instrumentation hooks) +
  `runtime/mstateflow_helpers.m` (emit, save_op / restore_op,
  active, push_history / pop_history, auto_snap).
- **Tier 4d — CLI**: `-simulate` drives a deterministic trace through
  the chart interpreter; `-emit-matlab` emits the compilable .m;
  `-dump-chart` dumps the resolved IR; `-simulate --sim-dap` boots
  the chart DAP server.
- **Tier 4e — DAP namespace `stateChart/*`**: full request set
  (`emit`, `setLocal`, `getActive`, `getLocals`, `stepSuperStep`,
  `stepTransition`, `setStateBreakpoints`, `setTransitionBreakpoints`,
  `setSymbolBreakpoints`, `saveOperatingPoint`,
  `restoreOperatingPoint`, `listStates`, `listTransitions`,
  `listJunctions`, `listEvents`, `listSymbols`, `listSnapshots`) +
  events (`superStepBegin` / `superStepEnd` / `stateEnter` /
  `stateExit` / `transitionFired` / `eventBroadcast` /
  `maxIterations`) + `stopped` on breakpoint hits.
- **Tier 4f — Canonical fixtures**: all six §6.8 examples ship
  (`air_temp_controller`, `hotel_check_in`, `traffic_light_moore`,
  `vending_machine_mealy` as `flat_vending`, `bang_bang_temp`,
  `automatic_transmission`) + four extras (`nested_heater`,
  `on_event_handlers`, `emit_in_action`, `temporal_after`) + three
  Tier-8/9 follow-on fixtures (`junction_backtrack` for connective
  backtracking, `temporal_counter` for `temporalCount` /
  `duration`, `chart_fn_truth_table` for the truth-table lowering
  + lint). Each fixture has four goldens locked in (flow /
  chart-IR / lowered-MATLAB / interpreter-trace). 67/67 tests
  passing.
- **Tier 6 — Snapshots + step-back** *(compiler side)*: named
  operating points via `saveOperatingPoint` / `restoreOperatingPoint`;
  auto-snapshot per super-step into a capped MATLAB-side ring (gated
  by `state.auto_snapshot`); C-side snapshot ring with name / size /
  copy introspection.
- **Tier 8 — Chart functions** *(partial)*: `chart_fn_matlab` nodes
  emit as sibling MATLAB functions callable from action bodies;
  graphical / truth-table call-sites still warn-and-pass.
- **Tier 9 — Codegen lanes** *(complete)*: `-emit-matlab`,
  `-emit-mir`, `-emit-llvm`, `-emit-c`, `-emit-cpp`, and
  `-emit-systemverilog` all work on chart `.mflow` inputs. The
  persistent-scalar lowering form (no struct, no string literals)
  unblocked the LLVM / C / C++ lanes; the parallel SV-target
  lowering produces synthesizable RTL. The three integer-typed
  `examples/stateflow/` examples emit verilator-lint-clean
  modules; float-typed charts hard-error at SV emission.
- **Bonus — Chart interpreter**: `lib/StateChart/Interpreter.{h,cpp}`.
  In-process C++ super-step simulator with full backtracking junction
  resolver, history junctions, super-transitions, temporal counters,
  symbol-change watchpoints, and operating-point snapshot/restore.
  Hosts the live DAP simulation.
- **REPL**: `runtime/stateflow_classdefs.m` ships the `stateChart`
  classdef wrapper (`tick` / `step` / `emit` / `active` / `save_op` /
  `restore_op` / `reset`) for handle-style use after a user runs
  `matlabc -emit-matlab <chart>.mflow`.

### What's shipped (IDE side — Matlab_llvm_ide)

Every UI/UX tier from §7 is in. Specifics by tier:

- **Tier 0 — Schema mirror + window stub** *(complete)*:
  `SchemaKind.stateChart` in the IDE codable layer; codec accepts
  both 0.1.x and 0.2.x (additive bump); the chart node / edge / data
  fields (`FlowNode.parent`, `Flow.symbols`, `NodeData.{entry,
  during, exit}Action` / `onEventActions`, `FlowEdge.data.params`,
  document-level `_comment`) all round-trip byte-stable against the
  five shipped `examples/stateflow/` fixtures. `StateChartWindow` +
  `StateChartWindowRef` (file-keyed dedup) + `StateChartCommands`
  scene; **File → New State Chart** (⌥⇧⌘N); orange `H` Explorer
  badge; file-dialect detection in `FileSystemService.mflowKind`.
- **Tier 1 — Flat authoring** *(complete)*:
  state body renders the Stateflow-style header (▶ initial marker,
  OR/AND chip) + inline colour-coded `entry/during/exit` stubs;
  junctions render as glyph-only ellipses (`●`/`H`/`×`/▶); transition
  edges paint solid + arrowed in chart-accent orange. Inspector:
  three multi-line action editors + decomposition picker (leaf / or
  / and) + container-style picker + Initial / History / Atomic
  toggles + Execution Order field (visible only on AND children).
  `TransitionLabel.parse` parses the four-field
  `event[guard]{cond}/trans` form; the edge inspector shows it as
  a four-chip preview. Tier-1 lint: multiple-defaults / dangling-
  default / parent-cycle.
- **Tier 2 — Hierarchy + decomposition** *(complete)*:
  compound states autosize to the bounding-box of their descendants
  + padding; child rendering z-orders parents under children;
  group-drag moves the whole subtree in lockstep; drag-to-reparent
  picks the deepest containing state and refuses descendant cycles;
  AND execution-order auto-numbers + renders as a magenta `#N` pill;
  Cmd+↑ / Cmd+↓ promote / demote. Lint additions: exec-order
  collisions, history-on-AND.
- **Tier 3 — Symbols + action lint** *(complete)*:
  inline `ChartSymbolsEditor` in the chart-root inspector with
  add/remove rows + scope/type/units/initial fields for data,
  trigger picker for events; symbol-aware action-body lint scans
  every action body + transition label for undeclared identifiers
  (whitelist of declared symbols + state ids + Stateflow operators
  + MATLAB built-ins + keywords), surfaces as an amber stroke on
  the canvas + an "UNDEFINED SYMBOLS" card in the inspector.
- **Tier 5 — Live debug surface** *(complete)*:
  `StateChartSimulation` (per-window DAP client) parses every
  `stateChart/*` event into `@Published activeStateIDs` /
  `firingTransitionID` (200ms auto-clear) / `eventLog`. Active
  States pane (hierarchy tree with `●` active dots), Event Log
  bottom strip with sim-time + click-to-reveal + CSV export, live
  transport row (Run / Pause / Stop / Reset / Step Super-Step /
  Step Transition), `simulationRuntimeAvailable` flips off the
  matlabc-on-path check, green halo on every active state, right-
  click state breakpoints (Break on Enter / Exit) + transition
  breakpoints (Break on Fire) with red dot badges.
- **Tier 6 — Snapshots + step-back** *(complete)*:
  `StateChartSimulation.{listSnapshots, saveOperatingPoint(name:),
  restoreOperatingPoint(name:)}` round-trip the chart-adapter's
  ring; `ChartSnapshotPanel` side panel lists named + `auto_*`
  snapshots with footprint formatting + per-row Restore button.
- **Tier 7 — Tabular alternatives** *(complete)*:
  `StateTransitionTableEditor` grid (Source × Dest × Event × Guard
  × CondAct × TransAct × Priority) with two-way sync —
  `replaceEdgeEndpoint` in-place rewiring preserves edge ids;
  `TruthTableEditor` sheet (conditions / decisions / actions)
  triggered by double-click on `chart_fn_truth_table`, with over/
  underspecification lint. Chrome toggle + Chart menu
  `Convert to STT` (⌘⇧T) / `Convert to Chart`.
- **Tier 8 — Functions + reuse** *(IDE-side complete)*:
  chart-function call sites render with the Stateflow corner badge
  (`λ` / `m` / `▦`) + body line showing the bound flow id or amber
  `unbound`; inspector adds a Flow ID picker over sibling flows +
  **Create sub-flow…** one-click affordance; `enterSubflow` /
  `parentFlowID` extended to recognise all six container kinds so
  the breadcrumb works across both dialects.
- **Tier 9 — Codegen lanes** *(complete)*:
  Chart menu's **Export Generated Artifact** submenu surfaces every
  matlab_llvm-side lane (`-emit-matlab` / `-dump-chart` /
  `-emit-mir` / `-emit-llvm` / `-emit-c` / `-emit-cpp` /
  `-emit-systemverilog`); inline `ChartGeneratedMatlabPane` does
  live `-emit-matlab` previews debounced on chart edits.
- **Tier 10 — Polish** *(complete)*:
  Pattern Wizard (Chart → Insert Pattern…, ⌘⇧P) with debouncer /
  edge-detector / fault-handler templates that drop a ready-made
  subgraph + seed the Symbols table; TeX rendering on `.comment`
  nodes (`$math$` runs in italic-serif accent + 50+ LaTeX-macro
  Unicode substitutions); right-click **Extract as State Chart…**
  on a control-flow selection that builds a new chart document
  (state-per-node mapping with bracketed-guard transitions +
  `_comment` attribution) and opens it in a chart window.

**Test coverage (IDE side):** 80+ chart-specific cases across 11
suites (`StateChartSchemaTests`, `StateChartBackCompatTests`,
`StateChartMatlabcFixturesTests`, `StateChartHierarchyTests`,
`StateChartSymbolsLintTests`, `TransitionLabelParserTests`,
`StateChartLintTests`, `StateChartSimulationTests`,
`StateChartWindowSimulationWiringTests`, `StateChartSnapshotTests`,
`StateChartTabularEditorTests`, `StateChartChartFunctionsTests`,
`StateChartCodegenTests`, `StateChartPatternWizardTests`,
`TeXAnnotationTests`, `ChartFromSelectionTests`). Full unit suite
green; 860 tests total.

### What's NOT shipped

All five backend follow-ons listed in the prior revision **shipped
2026-05-17**:

- ✅ **Tier 8 close-out** — `chart_fn_graphical` lowers as a sibling
  MATLAB function (Body is plain MATLAB; the IDE renders it as a
  flowchart on save/load, but the on-disk form stays textual);
  `chart_fn_truth_table` lowers to a priority-ordered if/elseif
  dispatch with an over/underspecification diagnostic (parity with
  §8-27 of the guide); chart-function recursion + cycle detection
  surfaces as a warning during chart-IR build. All three kinds
  (`matlab` / `graphical` / `truth_table`) round-trip through
  every `-emit-*` lane.
- ✅ **`temporalCount(event)` and `duration(cond)`** — counter-style
  temporal operators. Lowering: each call-site is registered against
  its owning state, allocated a persistent slot (`tc_<state>_<event>`
  for counts; `dur_<state>_<i>_act` + `_start` for durations),
  reset on state entry, and maintained once per super-step before
  the transition dispatch. Interpreter: `temporalCount` resolved
  through a per-(state, event) counter incremented on broadcast
  while the owner is active; `duration` left as lowering-only for
  this slice (no raw-text capture in the inline parser yet).
  Backed by `test/Flowchart/StateChart/temporal_counter.mflow`.
- ✅ **Connective-junction backtracking** in the **lowering** path —
  `emitJunctionChain` rewritten as a path-enumeration approach in
  `lib/StateChart/Lowering.cpp`. Every root-to-state path is
  flattened at compile time; the lowered MATLAB emits an
  if/elseif arm per path, so the elseif semantics give us the
  same priority-ordered backtracking the C++ interpreter already
  did. Backed by `test/Flowchart/StateChart/junction_backtrack.mflow`.
- ✅ **REPL auto-load of `.mflow`** — `loadStateChart('foo.mflow')`
  is a REPL-level shortcut (intercepted in `tools/matlabc/main.cpp`
  next to `tryHandleHelp`). It shells out to the same matlabc
  binary with `-emit-matlab`, captures stdout, and feeds it through
  `runReplInput` so the chart's `<name>_tick` lands live in the
  REPL session.
- ✅ **SV emission on float-typed charts** — now a hard error with
  a diagnostic pointing at the integer-typed `traffic_light_moore`
  / `vending_machine_mealy` / `model_air_temperature_controller`
  examples. Detected via a `SawFloatLiteral` flag on `ChartLayout`
  that the rewriter sets whenever it lets a non-integer literal
  through; checked at the end of `emit()` when `Target ==
  SystemVerilog`. The two `get_started_*` battery examples now
  fail SV emission cleanly instead of producing verilator-warned
  output.

Remaining deferrals (Tier-N+):

- `duration(cond)` in the **interpreter** — the lowering supports it
  fully; the inline parser inside `lib/StateChart/Interpreter.cpp`
  doesn't yet capture the raw expression text between balanced
  parens, so guards using `duration(...)` evaluate to 0 in the
  interpreter trace. The lowered MATLAB / C / LLVM / SV paths all
  honour the operator.
- C as action language; conversion sweep.
- Stateflow Messages with queue semantics.
- Mealy/Moore conversion sweeps.
- Atomic Subcharts (separate codegen).
- Simulink-based states.
- Multi-chart Sequence Viewer.

The plan below is organised so Tier 0 (additive schema + palette stubs)
lands first as a tiny PR; everything downstream forks into two parallel
tracks — **UI/UX** (§5) and **Compiler / runtime** (§6) — that converge
at Tier 4 for live debug. Both tracks are now closed; the remaining
items are the small deferrals listed in "What's NOT shipped" above.

## 1. North-star UX

> Open a `.mflow` flagged as `state_chart` → it pops in its own window,
> same `WindowGroup` machinery mflowLink uses. Drop a `Heater` state,
> give it `On` and `Off` substates, draw a transition
> `Off —[temp<setpoint-2]→ On`. Add a parallel sibling region `Fan`
> with its own `Idle`/`Run` substates. Press ▶ — both regions activate;
> the active state glows. Broadcast `tick` from the Command Window →
> the chart steps one super-step, the active state migrates, the
> transition that fired pulses amber for ~200 ms. Press ⏸. Set a
> transition breakpoint on `Off → On` → ▶ runs until the predicate
> fires, then snaps the active state. Right-click a state → "Log
> activity" → its enter/exit lifeline appears in the Event Log pane at
> the bottom of the window.

Same trust-model as mflowLink: the IDE renders events streamed up from
`matlabc -simulate`; chart state lives in the runtime, not in the IDE.
Authoring is direct-manipulation, no modal property dialogs (parity
with mflowLink §1.1).

### 1.1 Improvements over Simulink Stateflow we will commit to

| Pain in Stateflow                                                                | mStateflow design                                                                                                                       |
|---|---|
| Action label syntax (`entry:`, `during:`, `exit:`, `on E:`) is one big text blob | Inspector exposes them as four labeled code editors; canonical labels emitted on save, never edited by hand                             |
| Box / subchart / state distinction confuses newcomers                            | One `state` primitive; "atomic" / "subchart" / "box" are inspector toggles, not separate palette items                                  |
| Action language is per-chart and buried in chart properties                      | Toggle visible in chrome; default MATLAB (matches the rest of matlabc); C added later as a parity sweep                                 |
| `.slx` is binary, charts don't diff                                              | `.mflow` JSON; transitions and states are hand-editable; git diff is the spec                                                           |
| Operating-point save/restore needs API calls                                     | Snapshot ring (same buffer mflowLink uses for step-back); named snapshots in a side panel                                              |
| Animation always-on; slows long runs                                             | Animation respects a transport-row speed slider; auto-off above 1× sim rate                                                             |
| Logged state activity needs Simulink Data Inspector wiring                       | Right-click state → "Log activity" → enter/exit timestamps flow into the Event Log pane and CSV-export                                  |
| Truth Tables / STTs open in a separate fullscreen editor                         | Open inline as docked tabs next to the canvas — same tab strip as the generated-MATLAB / generated-C panes                              |
| Default-transition multiplicity / dangling-default errors only surface at build  | Edit-time lint, red squiggle on the offending state                                                                                     |
| "What's active right now?" requires a separate Symbols pane                      | Active State pane is a tree view of the chart hierarchy with live OR-active / AND-region badges                                         |
| Sequence Viewer is a separate model element                                      | Built into the bottom strip alongside scopes (mflowLink-style) — one timeline per logged region                                         |
| Step forward is one super-step only                                              | Expose **transition-by-transition stepping within a super-step** in the transport row, mirroring mflowLink's block-by-block stepping    |

## 2. Non-goals (for the initial roadmap)

- **C as action language.** MATLAB only on first ship. C parity once the
  MATLAB lane is solid.
- **Simulink-based states** (signal-flow inside a state). Would require
  a fourth dialect interop layer; defer.
- **Messages with queue semantics** (Stateflow §13). Events only at
  first; messages are a Tier-N+ extension once the event-broadcast
  plumbing is stable.
- **Mealy/Moore chart conversion** (§5 of the guide). Semantically
  interesting but niche.
- **Custom C code via `coder.extrinsic`** (§14-22). Stays inside
  `matlabc`-compileable expressions only.
- ~~**HDL coder for charts.** Digital-only state machines may eventually
  lower through `-emit-sv`, but not in scope here.~~ *(Shipped:
  `-emit-systemverilog` on chart `.mflow` files emits verilator-clean
  Moore / Mealy / AND-parallel modules. The synthesizable lowering
  is a sibling target of the software lowering — see §6.3.)*
- **Atomic Subchart for separate codegen** (§17). Will reuse the
  existing `signal_subsystem` / sub-flow mechanism if needed; no
  chart-only equivalent.
- **Distributed / multi-chart aggregation in the Sequence Viewer.** One
  chart per timeline at first.

## 3. Mental model — Stateflow → mStateflow

| Stateflow term                                | mStateflow term                                                                                                                  | Where it lives                                                          |
|---|---|---|
| `.slx` chart                                  | `.mflow` with `settings.kind = "state_chart"`                                                                                    | one document, three dialects                                            |
| Chart                                         | A top-level `Flow` whose nodes are states + junctions; `kind: program`                                                            | reuse existing `Flow` struct                                            |
| State (with substates)                        | `FlowNode` of kind `.state`, with `parent: FlowNode.id?` pointers from its children                                              | new `parent` field on `FlowNode` (optional)                             |
| Exclusive (OR) decomposition                  | `data.params.decomposition = "or"` on the parent state                                                                            | new param key                                                           |
| Parallel (AND) decomposition                  | `data.params.decomposition = "and"`; children carry `data.params.executionOrder: Int`                                            | new param keys                                                          |
| Atomic state / subchart / box                 | `data.params.containerStyle = "state" \| "subchart" \| "box"` on a compound state                                                 | one node kind, three render modes                                       |
| Transition                                    | `FlowEdge` with new `EdgeKind.transition`; `data.label` carries `event[guard]{condAct}/transAct`                                  | reuse routing/marquee/clipboard machinery                               |
| Default transition                            | `FlowEdge` from a `.junction_default` stub into a state; only outgoing edge allowed                                              | new junction kind                                                       |
| Connective / History / Entry / Exit junctions | `.junction_connective` / `.junction_history` / `.junction_entry` / `.junction_exit`                                              | new node kinds                                                          |
| Supertransition                               | a `FlowEdge` whose endpoints span hierarchy levels — already valid in our edge model, just renders as a longer routed spline    | no schema work                                                          |
| Graphical / MATLAB / Truth Table function     | `.chart_fn_graphical` / `.chart_fn_matlab` / `.chart_fn_truth_table` — call-site nodes that reference a sub-`Flow` by id          | reuses existing `signal_subsystem` → sub-flow indirection               |
| Data / Event / Message                        | flow-level `symbols: { data: [...], events: [...], messages: [...] }` table                                                      | new top-level field on `Flow`                                           |
| Annotation                                    | existing `.comment` node kind                                                                                                     | already exists                                                          |
| Chart Explorer                                | Project Explorer tree with state-hierarchy nesting                                                                                | extend existing tree                                                    |
| Symbols pane                                  | Inspector "Symbols" tab when chart-root is selected                                                                                | new tab                                                                 |
| State Transition Table                        | docked tabular editor — opens via tab-strip on the canvas, edits the same `Flow`                                                  | new view; round-trips with canvas                                       |
| Truth Table                                   | docked grid editor for `.chart_fn_truth_table` nodes                                                                              | new view                                                                |
| Sequence Viewer                               | bottom strip lane (sibling of `ScopeTileStrip`)                                                                                   | new view                                                                |
| Stateflow Editor toolstrip                    | window chrome (same row pattern as mflowLink)                                                                                     | new strip                                                               |

## 4. On-disk schema additions

All additions are *additive* — existing `control_flow` and `signal_flow`
`.mflow` documents stay byte-stable. Schema bump: 0.1.0 → 0.2.0.

### 4.1 Document-level

- **`SchemaKind`** gains `"state_chart"` alongside `"control_flow"` and
  `"signal_flow"`.
- **`Flow.symbols`** — new optional table per flow:
  ```jsonc
  "symbols": {
    "data":     [{"name": "temp", "scope": "input", "type": "double", "units": "C"}, …],
    "events":   [{"name": "tick", "trigger": "rising"}, …],
    "messages": []
  }
  ```
  Drives the inspector's Symbols tab. Resolved against the same MATLAB
  symbol-resolution `matlabc` uses for `.m` files.

### 4.2 New `FlowNode` fields

- **`parent: String?`** — id of the containing state, or omitted for
  chart-root children. Compound states use this to claim children.
  This is the *only* hierarchy carrier — `Flow.nodes` stays flat.
- **`ui.size: {w, h}?`** — compound states have visible bounds (today
  only `position` is stored, since signal-flow blocks autosize). State
  bounds are user-edited; an autosize pass keeps the parent ≥
  bounding-box of children + padding.

### 4.3 New `NodeKind` values

| NodeKind              | Shape                          | Role                                                          |
|---|---|---|
| `state`               | rounded rect with inner labels | the workhorse (atomic OR compound — `decomposition` decides)  |
| `junction_connective` | small filled circle (radius 6) | branch/merge point inside flow logic                          |
| `junction_history`    | circled `H`                    | dropped beside a compound state; sets parent's `hasHistory`   |
| `junction_entry`      | filled bullseye                | entry port on state boundary                                  |
| `junction_exit`       | hollow `×` in circle           | exit port on state boundary                                   |
| `junction_default`    | filled bullet w/ short stub    | source of a default transition (replaces the magic dot)       |
| `chart_fn_graphical`  | rect with `λ` corner badge     | call-site for a graphical-function sub-`Flow`                 |
| `chart_fn_matlab`     | rect with `m` corner badge     | call-site for an inline MATLAB function                       |
| `chart_fn_truth_table`| rect with table glyph          | call-site for a Truth Table sub-`Flow`                        |

### 4.4 New `NodeData` fields (all optional)

For `state`:
- `entryAction: String?`, `duringAction: String?`, `exitAction: String?`
- `onEventActions: [String: String]?` (event-name → action body)
- `params.decomposition: "or" | "and" | "leaf"`
- `params.containerStyle: "state" | "subchart" | "box"`
- `params.executionOrder: Int?` (AND children only)
- `params.isInitial: Bool?` (alternative to a `.junction_default` node)
- `params.hasHistory: Bool?` (alternative to a `.junction_history` sibling)
- `params.atomic: Bool?` (codegen hint; see §17 of the guide)

For transitions (carried on `FlowEdge.data.params` since edges already
have a `data` bag):
- `params.priority: Int?`
- `params.kind: "outer" | "inner" | "default"`

### 4.5 New `EdgeKind` value

- **`transition`** — joins existing `.control` and `.data`. Edge label
  parses lazily as `event[guard]{condAction}/transAction`; all four
  sub-fields are optional. Rendered as multi-line styled text near the
  midpoint (event in cyan, guard in amber, cond-action in muted text,
  trans-action in red — palette-aware).

## 5. IDE / UI/UX changes

Subsections roughly mirror mflowLink §5. Each ships in a tier (§7);
this section catalogues the surfaces themselves.

### 5.1 New document mode

- File menu: "New → State Chart…" emits a `.mflow` with
  `settings.kind = "state_chart"`.
- Project Explorer: chart badge (orange `H` glyph matching Stateflow's
  history-junction visual) on `.mflow` files flagged as charts.
- Opening a chart `.mflow` routes through a new `StateChartWindow`
  (parity with `MflowLinkWindow`); routing keyed on `settings.kind`.

### 5.2 Window model

- `StateChartWindow` + `StateChartWindowViewModel` (parity with
  `MflowLinkWindow` / `MflowLinkWindowViewModel`).
- File-keyed `WindowGroup(for: StateChartWindowRef)`; the same
  multi-window behaviour mflowLink already enjoys.
- Scene-level menus: "Chart" (insert state / junction / function,
  decomposition toggles, auto-layout) and "Simulation" (run / pause /
  step super-step / step transition / save snapshot).
- `@FocusedValue(\.stateChartViewModel)` surfaces the VM to menu
  bar items.

### 5.3 Authoring canvas additions

`FlowchartCanvasView` is reused, with kind-aware rendering. Specific
additions:

- **State rendering**: rounded rect, name in top bar, four optional
  action stubs (entry/during/exit/on-event) typeset inside with
  distinct accent colors. Hover reveals the four-editor inspector.
- **Hierarchy**: children of a state render *inside* the parent's
  bounds, clipped, with scroll/zoom inside large parents. Drag-to-
  reparent updates `FlowNode.parent`; autosize parent on child
  insertion; promote-to-parent (Cmd+↑) and demote-into-target
  (Cmd+↓) keyboard shortcuts.
- **Transitions**: cubic Bezier through optional waypoints + arrow
  head, label near midpoint, label hover-to-edit. Multi-segment
  routing avoids state bodies it doesn't terminate in (supertransitions
  span levels).
- **Junctions**: 12-px shapes (connective, history `H`, entry `⦿`,
  exit `⊗`, default bullet+stub). Snap-to-grid optional, off by default.
- **Active-state animation** (Tier 4+): blue pulse on enter, dim on
  exit; firing transition flashes amber for ~200 ms. Animation speed
  capped by transport slider; auto-off above N events/sec.
- **Edit-time lint**: multiple defaults, dangling defaults,
  AND-execution-order collisions, undefined symbol references, action
  parser errors — red squiggle + sidebar list.

### 5.4 Inspector additions

When a state is selected:
- Four labeled multi-line editors (`entry:` / `during:` / `exit:` /
  per-event `on:`); each is a stripped-down version of the existing
  MATLAB editor (tokenizer + autocomplete reused).
- Decomposition picker (OR / AND / leaf).
- Container style picker (state / subchart / box).
- "Initial substate" toggle (for OR children).
- "History" toggle (for OR parents — surfaces a sibling
  `junction_history` glyph too).

When a transition is selected:
- Single-line label editor with parser-aware chips for event / guard /
  cond-action / trans-action; multi-line expansion on focus.
- Priority field (auto-numbered, draggable in the canvas to reorder).

When chart-root is selected:
- Symbols tab (new): data / events / messages tables with
  scope/type/units columns.
- Action language picker (MATLAB at first; C disabled).
- Super-step iteration cap (`kMaxIterations`, default 1000).

### 5.5 Tabular alternatives — STT and Truth Table

- **State Transition Table editor**: docked tab next to the canvas;
  grid of (source × destination × event × guard × cond-action × trans-
  action × priority) rows. Two-way sync — edits in the grid reflow the
  canvas, canvas edits update the grid. `convertToChart` /
  `convertToSTT` available as right-click on chart-root.
- **Truth Table editor**: docked grid for `.chart_fn_truth_table`
  nodes; condition rows × decision columns + action row. Diagnostic
  for over/underspecification (parity with §8-27 of the guide).

### 5.6 Active State pane + Event Log

- **Active State pane** (right column, new tab next to the inspector):
  tree view of the chart hierarchy. `●` for OR-active, `▶ ▶` numbered
  for AND-active regions, last-event marker, click-to-reveal in canvas.
- **Event Log pane** (bottom strip, sibling of `ScopeTileStrip`):
  scrolling list of `stateEnter`/`stateExit`/`transitionFired`/`eventBroadcast`
  events with sim-time, source/dest ids, hover-jumps-to-canvas,
  CSV export.

### 5.7 Sequence Viewer tile

- Bottom-strip tile: one swimlane per parallel region, vertical bars
  for active windows, transition arrows at firing times. Reuses the
  scope-tile pattern (double-click → docked detail pane with axes).

### 5.8 Breakpoint surface

- Right-click a state → "Break on enter" / "Break on exit".
- Right-click a transition → "Break on fire".
- Right-click a symbol in the Symbols pane → "Break on change".
- Breakpoints render as red overlays on the canvas glyphs; live state
  matches mflowLink's signal-breakpoint surface.

### 5.9 Snapshot ring + step-back

- "Save snapshot" / "Restore snapshot" in the transport row; named
  snapshots in a side panel.
- "Start sim from this point" sets the runtime's initial active-state
  vector from the chosen snapshot — direct parity with §18 of the guide
  (Operating Points).

## 6. matlab_llvm / Compiler changes

Files to land on the matlab_llvm side, grouped by responsibility.

### 6.1 Schema acceptance — `lib/Flowchart/Loader.cpp`

- Recognise `settings.kind == "state_chart"`; dispatch into a new
  chart loader path. Existing `control_flow` / `signal_flow` paths
  unchanged.
- Parse the new `FlowNode.parent`, `ui.size`, `data.params.*`,
  `Flow.symbols` fields.
- Schema-level validation: parent ids resolve, no parent-cycle, AND
  children carry execution-order, default-transition multiplicity at
  most one per OR parent.

### 6.2 Chart IR — `lib/StateChart/StateChartIR.{h,cpp}`

In-memory IR matching the schema:
- `Chart` (root) → `Region` (OR/AND) → `State` / `Junction` →
  `Transition`.
- Symbol tables for data/events/messages (resolved to MATLAB types).
- Action ASTs (entry/during/exit/on-event) parsed via the existing
  MATLAB front-end so all matlabc type-checking applies.
- Transition labels parsed to four optional ASTs (event / guard /
  cond-action / trans-action).

### 6.3 Lowering — `lib/StateChart/Lowering.cpp`

Chart IR → MATLAB (which matlabc then routes through its existing
backends). The lowering exposes a `LoweringTarget` enum so the same
chart IR drives two output shapes:

- **`Software`** (default; targets `-emit-matlab` / `-emit-mir` /
  `-emit-llvm` / `-emit-c` / `-emit-cpp`):
  `<chart>_tick(in_X1, …, ev_E1, …) → out_Y1, …` as a single MATLAB
  function with **persistent scalars** for every chart slot
  (`persistent l_<local>` / `r_<region>` / `t_<state>` / `h_<state>`
  / `tick_count`). No `struct()`, no string literals — region codes
  are 1-based integers, every persistent has an `if isempty(X), X = 0;
  end` init line. Super-step is a `while fired … end` loop bounded by
  `kMaxIterations`.
- **`SystemVerilog`** (targets `-emit-systemverilog` /
  `-check-synthesizable` / `-emit-hardware-report` / `-emit-cocotb`):
  same body shape but **per-variable** `if isempty(X), X = intW(0);
  end` initialisers (mapped to power-on reset values by matlabc's SV
  pass), integer-typed literals (`int16(N)` wraps on every numeric
  emit), one-pass tick (no inner fixed-point loop — each call =
  one clock edge / one transition attempt per region), inlined
  `in()` predicate bypassing the helper function (matlabc's SV
  pipeline can't infer `in_helper` param types from in-body call
  sites, so the helper is dead code on SV and stays unemitted).
  matlabc's SV pass auto-injects `clk` / `rst_n` and translates
  each persistent into a flip-flop + `<X>_next` combinational
  assignment.

Common semantics across both targets:

- Active state encoded as a per-OR-region integer slot (`r_<region>`).
  AND regions don't get a slot — their children are always co-active
  and visited per super-step iteration in `executionOrder`.
- Entry / during / exit / on-event actions are inlined into the
  per-substate dispatch.
- Transitions become a priority-sorted dispatch per source state.
- Inner transitions skip the exit/entry chain.
- Super-transitions across hierarchy walk src up to the LCA, exit each
  level (saving history for OR parents with `hasHistory`), then walk
  down to dst.
- Junction chains follow `connective` / `entry` / `exit` outgoing
  transitions in priority order; `history` redirects to the parent's
  `state.history.<parent>` slot. Lowering commits greedily on the
  first matching guard; the C++ interpreter backtracks properly.
- Temporal operators `after` / `before` / `at` / `every` lower to
  `(tick_count − t_<owner>) <cmp> N`. `tick_count` advances once per
  super-step; `t_<state>` is stamped on each entry. The `sec` and
  `tick` unit suffixes are aliased (1 super-step = 1 sec).
  `temporalCount(event)` and `duration(cond)` are Tier-N+.
- `in(stateId)` lowers to an integer comparison against the named
  state's parent-region slot — through a chart-scoped helper on
  software target, inlined on SV target.
- `emit('X')` rewrites to `ev_X = true` so action bodies can broadcast
  events without leaking lowering internals.

### 6.4 Runtime — `lib/Runtime/runtime_mstateflow.cpp`

- Event queue (bounded FIFO; depth configurable per chart).
- Broadcast helper (`emit(event)`) for use inside actions.
- Active-state bitset + region-vector accessors.
- Snapshot ring — shared buffer with mflowLink's (binary-compatible);
  one entry per super-step boundary.
- Temporal counters & timers (sim-time-driven).
- Diagnostic counters for infinite-loop detection.

The runtime exposes `mstateflow_tick(chart_ctx, event_vec, inputs,
outputs)`; everything else is pure data so it round-trips through the
existing C++ runtime that already underpins `.m` programs.

### 6.5 CLI — `tools/matlabc/CLI.cpp`

- `-simulate` already understands `.mflow`; detect `state_chart` and
  dispatch into the chart runtime. **No new flag required.**
- `-emit-matlab` for chart → emits a runnable MATLAB script
  `function [outputs, state] = chart_tick(inputs, state, events)` so
  generated code is readable.

### 6.6 Codegen lanes (closed)

Each gains one fixture test per canonical chart example.

- ✅ `-emit-matlab` — readable MATLAB equivalent of the chart
  (persistent-scalar form; ~70-150 lines per fixture).
- ✅ `-emit-c` / `-emit-cpp` — chart compiles to `static double` /
  `static int16_t` persistent locals + if-chain transition dispatch.
  Verified end-to-end: emitted C compiles with `cc`, links against a
  small `matlab_persistent_isempty` stub, and prints the expected
  tick trace.
- ✅ `-emit-llvm` — chart lowers to clean LLVM IR (~170 lines per
  fixture) that `llc` assembles without warnings.
- ✅ `-emit-mlir` / `-emit-mir` — chart lowers to clean `matlab.func`
  IR, all locals as scalar `matlab.alloc`, transitions as nested
  `scf.if`.
- ✅ `-emit-systemverilog` — synthesizable RTL via the SV-target
  lowering. Three of five `examples/stateflow/` examples are
  **verilator-lint-clean**:
  - `traffic_light_moore` → Moore FSM, 122 lines.
  - `vending_machine_mealy` → Mealy FSM with cond-action outputs,
    106 lines.
  - `model_air_temperature_controller` → AND-parallel chart with
    three regions, 208 lines.
  The two float-typed examples (`get_started_create_chart` /
  `get_started_hierarchy_chart`) now **hard-error** at SV emission
  with a diagnostic pointing at the three integer-typed examples
  (parity-with-synthesis intent: a chart using `f64` arithmetic
  needs a fixed-point convention before it can drop into RTL).
- 🟧 **Verilog-A** — analog action bodies could lower through the
  existing Verilog-A path; not currently exercised by any chart
  fixture. Tier-N+.

### 6.7 DAP adapter — `tools/matlabc/DAP/StateChartAdapter.cpp`

Extends the mflowLink DAP server (same socket; protocol additions are
namespaced under `stateChart/*`).

New events:
- `stateChart/stateEnter { id, t }`
- `stateChart/stateExit  { id, t }`
- `stateChart/transitionFired { id, src, dst, t, eventName? }`
- `stateChart/superStepBegin { t, iteration }`
- `stateChart/superStepEnd   { t, iteration, quiescent: bool }`
- `stateChart/eventBroadcast { name, t, payload? }`

New requests:
- `stateChart/setStateBreakpoints { ids: [], onEnter, onExit }`
- `stateChart/setTransitionBreakpoints { ids: [] }`
- `stateChart/stepTransition` — fire exactly one transition then halt.
- `stateChart/stepSuperStep` — fire transitions until quiescent then halt.
- `stateChart/saveOperatingPoint { name }`
- `stateChart/restoreOperatingPoint { name }`

### 6.8 Reference examples — `matlab_llvm/Examples/StateCharts/`

Canonical Stateflow examples from the guide, shipped as fixtures and
golden-file tests:
- `air_temp_controller.mflow` (§1-31)
- `hotel_check_in.mflow` (§1-9)
- `traffic_light_moore.mflow` (§5-12)
- `vending_machine_mealy.mflow` (§5-7)
- `bang_bang_temp.mflow` (§14-62)
- `automatic_transmission.mflow` (§11-46)

Each fixture verifies: schema round-trip + reference simulation trace +
reference `-emit-matlab` output.

## 7. Tier plan

Each tier is a shippable slice. Tracks: **[UI/UX]** = IDE-side,
**[Compiler]** = matlab_llvm-side, **[Both]** = needs coordinated PR.
Effort tags: **S** (≤ 1 wk), **M** (1–3 wk), **L** (3–6 wk).

### Tier 0 — Foundations *(Both, S)* — **✅ complete**

**Goal**: minimum schema + routing so chart `.mflow` files load and
save without breaking existing dialects.

- ✅ **[Compiler]** `SchemaKind.state_chart` accepted by loader; new
  optional fields parse (`FlowNode.parent`, `ui.size`, `data.params.*`,
  `data.onEventActions`, `Flow.symbols`, transition `data.params` +
  label); validation rejects malformed hierarchies.
- ⬜ **[UI/UX]** `SchemaKind` mirror in the IDE codable layer; new
  node-kind stubs (palette icons drawn, no canvas behaviour yet).
- ⬜ **[UI/UX]** "New → State Chart…" file template; Project Explorer
  badge; routing to a stub `StateChartWindow` (empty canvas).
- ✅ **[Both]** Schema bump 0.1.0 → 0.2.0; back-compat tests for existing
  control-flow and signal-flow fixtures.

### Tier 1 — Flat authoring canvas *(UI/UX, M)* — **✅ complete**

**Goal**: visual Stateflow-equivalent for a flat (non-hierarchical)
chart. No simulation yet.

- State rendering: rounded rect, name bar, four action editors inside.
- Transition rendering: spline + arrow + label; live-parse the four
  label sub-fields.
- Junction rendering: connective bullet, default-bullet stub, entry/exit.
- Inspector: state form (4 action editors + decomposition picker +
  initial/history toggles); transition form (label chips + priority).
- Marquee / clipboard / undo — reuse `FlowchartViewModel` verbs
  unchanged.
- Edit-time lint: multiple defaults, dangling default, undefined
  symbol refs.

**Verifies**: the flat "Vending Machine Mealy" example (§5-7) is fully
authorable.

### Tier 2 — Hierarchy + decomposition *(UI/UX, M)* — **✅ complete**

**Goal**: full Stateflow visual parity for nested charts.

- `FlowNode.parent` plumbing; children render inside parent bounds,
  clipped and scrollable.
- Drag-to-reparent; autosize parent on child insertion; promote /
  demote keyboard shortcuts.
- OR / AND decomposition; AND-execution-order badges (auto-numbered,
  drag to reorder).
- History junction (`H` glyph); maps to parent's `hasHistory` param.
- Supertransitions: edges spanning hierarchy levels route around
  intermediate state bodies.
- Lint extensions: history-junction-on-AND, exec-order collisions.

**Verifies**: "Air Temperature Controller" (§1-31) and "Hotel Check-In"
(§1-9) fully authorable.

### Tier 3 — Symbols + action-language editor *(UI/UX, M)* — **✅ complete**

**Goal**: production-grade action authoring with autocomplete.

- Inspector Symbols tab — data / events / messages tables.
- Action editor: multi-line, MATLAB tokenizer reused; autocomplete on
  symbol names, temporal operators, `in()`.
- Lint: parser errors, type mismatches, references to symbols outside
  scope.

**Verifies**: the bang-bang controller's actions (§14-62) author with
the same editing quality the existing `.m` editor provides.

### Tier 4 — Chart compiler + runtime *(Compiler, L)* — **✅ complete**

**The single largest chunk; gates everything below.**

- ✅ `lib/StateChart/StateChartIR.{h,cpp}` — chart, region, state,
  transition, junction, symbol tables, chart functions.
- ✅ `lib/StateChart/Lowering.cpp` — chart IR → matlabc IR; super-step
  fixed-point loop with `kMaxIterations` + saturation warning;
  temporal operators (`after` / `before` / `at` / `every`) + `in()`
  via auto-emitted helper; history junctions; inner + super
  transitions; junction chains; `emit('X')` rewriting.
- ✅ `runtime/runtime_mstateflow.cpp` — bounded FIFO event queue,
  broadcast helpers, snapshot ring with name introspection, DAP
  event sinks, C ABI bookends.
- ✅ `lib/StateChart/Interpreter.{h,cpp}` — bonus C++ super-step
  simulator with full junction backtracking, history, temporal
  counters, symbol watchpoints, snapshot/restore.
- ✅ CLI: `-simulate` runs the interpreter trace; `-emit-matlab`
  emits compilable MATLAB; `-dump-chart` dumps the IR.
- ✅ DAP adapter: chart-namespaced events + requests + introspection
  + breakpoints (state / transition / symbol-change) (§6.7).
- ✅ Golden-file fixtures for all six canonical examples in §6.8 plus
  four extras + three follow-on fixtures (`junction_backtrack`,
  `temporal_counter`, `chart_fn_truth_table`); 67/67 tests green.

**Verifies**: command-line simulation of all six reference examples
produces deterministic, golden-locked traces. ✅

### Tier 5 — Live debug surface *(UI/UX, M)* — **✅ complete**

Builds directly on Tier 4's DAP events. Backend stream is fully
wired — `stateEnter` / `stateExit` / `transitionFired` /
`eventBroadcast` / `superStepBegin` / `superStepEnd` /
`maxIterations` / `stopped` all fire from the live interpreter; the
introspection requests (`listStates` / `listTransitions` /
`listJunctions` / `listEvents` / `listSymbols` / `listSnapshots` /
`getActive` / `getLocals`) populate every pane without re-parsing
the FlowDoc; breakpoints (state-enter / state-exit / transition /
symbol-change) pause mid-action.

- Active-state animation: blue pulse on enter, amber flash on firing
  transition. Speed controlled by transport-row slider.
- **Active State pane**: hierarchy tree with OR-active / AND-active
  badges, click-to-reveal.
- **Event Log pane** (bottom strip): scrolling event list with sim-time
  + jump-to-canvas, CSV export.
- **Sequence Viewer tile** (bottom strip): swimlane per region,
  vertical activation bars, transition-firing arrows.
- State / transition breakpoints from right-click context menus.
- Symbol-change breakpoints via Symbols-pane context menu.

**Verifies**: pausing on a breakpoint surfaces the same active-state
set the runtime believes is active; the canvas pulses correctly.

### Tier 6 — Snapshots + step-back *(Both, S)* — **✅ complete**

Reuses mflowLink's snapshot ring buffer.

- ✅ **[Compiler]** Chart-runtime auto-snapshots every super-step
  boundary (gated by `state.auto_snapshot`); DAP requests
  `stateChart/saveOperatingPoint` / `restoreOperatingPoint` named-tag
  + restore via the runtime ring; `listSnapshots` enumerates names +
  sizes.
- ⬜ **[UI/UX]** Transport row gains "Save snapshot" / "Restore snapshot";
  named snapshots in a side panel; "Start sim from this point" sets the
  next-run's initial active-state vector.

**Verifies**: parity with §18 of the guide (Operating Points).

### Tier 7 — Tabular alternatives *(UI/UX, M)* — **✅ complete**

- State Transition Table editor: docked grid; two-way sync with canvas.
- Truth Table editor: condition × decision grid + action row;
  over/underspecification diagnostic (§8-27).
- `convertToChart` / `convertToSTT` right-click actions.

**Verifies**: the bang-bang controller authored as an STT (§16-16)
produces an identical simulation trace to its canvas equivalent.

### Tier 8 — Functions + reuse *(UI/UX + Compiler, M)* — **✅ complete**

- ⬜ **[UI/UX]** Graphical / MATLAB / Truth Table function call-site
  nodes; navigate-into-sub-`Flow` machinery (reused from
  `signal_subsystem`).
- ✅ **[Compiler]** All three chart-function kinds lower to sibling
  MATLAB functions callable from action bodies. `chart_fn_matlab`
  and `chart_fn_graphical` emit `Body` verbatim (the IDE renders
  Body as a flowchart on save/load; the on-disk form is text).
  `chart_fn_truth_table` lowers to a priority-ordered if/elseif
  dispatch over (condition × T/F/X pattern) columns plus an
  over/underspecification diagnostic (parity with §8-27). Chart-
  function recursion + cycle detection warns at chart-IR build
  time. Backed by
  `test/Flowchart/StateChart/chart_fn_truth_table.mflow`.

### Tier 9 — Codegen lanes *(Compiler, S)* — **✅ complete**

Closed by the persistent-scalar lowering rewrite + the SV-target
sibling.

- ✅ `-emit-matlab` / `-emit-mir` / `-emit-llvm` / `-emit-c` /
  `-emit-cpp` all work on chart `.mflow` inputs unchanged. The
  software-target lowering produces compilable MATLAB with no
  `struct()` / no string literals, so matlabc's MATLAB → LLVM lane
  handles every chart fixture.
- ✅ `-emit-systemverilog` — synthesizable RTL via the SV-target
  lowering (per-variable `if isempty` resets, integer-typed locals,
  single-pass tick). Moore / Mealy / AND-parallel charts produce
  verilator-lint-clean modules end-to-end.

### Tier 10 — Polish *(UI/UX, S, scattered)* — **✅ complete**

- ⬜ Pattern Wizard (debouncer / edge-detector / fault-handler templates).
- ⬜ TeX in annotations (parity with §6-24).
- ⬜ Chart-from-selection: refactor flat `if` / `function_definition`
  control-flow nodes into a chart.
- ✅ Active-state output port — `outputs.active_state_` mirrors the
  live `state.regions` struct so an mflowLink signal-flow document
  can wire a chart's active configuration into downstream blocks.
- ✅ Examples library shipped under two paths:
  - `test/Flowchart/StateChart/*.mflow` — 6 canonical §6.8 fixtures
    (`air_temp_controller`, `hotel_check_in`, `traffic_light_moore`,
    `flat_vending` ≡ Stateflow's vending_machine_mealy,
    `bang_bang_temp`, `automatic_transmission`) plus 4 extras
    (`nested_heater`, `on_event_handlers`, `emit_in_action`,
    `temporal_after`); 10 happy + 4 schema-error.
  - `examples/stateflow/*.mflow` — 5 tutorial-aligned examples
    designed for end-to-end Compile / Run / Debug / REPL / DAP / SV
    walkthroughs (`get_started_create_chart`,
    `get_started_hierarchy_chart`,
    `model_air_temperature_controller`, `traffic_light_moore`,
    `vending_machine_mealy`). The Moore / Mealy / AND-parallel
    three produce verilator-lint-clean SystemVerilog.

### Tier N+ — Deferred

- C as action language; conversion sweep.
- Stateflow Messages with queue semantics.
- Mealy/Moore conversion sweeps.
- Atomic Subcharts (separate codegen).
- HDL emission for digital state machines.
- Simulink-based states.
- Multi-chart Sequence Viewer.

## 8. Risk register

| Risk                                                                                | Mitigation                                                                                                                  |
|---|---|
| Deep hierarchies render slowly                                                       | Virtualise child rendering when parent is collapsed; quadtree the canvas hit-test pass                                      |
| Transition labels are richer than `signal_flow` edge labels                          | Absorb the four-field parser in a dedicated `TransitionLabelParser`; the schema keeps a flat string                         |
| Action MATLAB-only at launch may block lifting `.slx` charts                         | Document the limitation prominently; Stateflow ships its own C → MATLAB conversion (§15) we can wrap later                   |
| Super-step infinite loops                                                            | `kMaxIterations` cap with runtime warning; surface in the Event Log + diagnostic counter                                    |
| Operating-point ring grows unbounded for large charts                                | Cap snapshot count (configurable); evict oldest; named snapshots are pinned                                                 |
| Two canvases (signal-flow vs. chart) diverge over time                               | Keep one canvas with kind-aware rendering; routing rules differ but the waypoint model already accommodates both            |
| `FlowNode.parent` field collisions with future schema additions                      | Field is optional; loader ignores unknown values; schema version bump 0.1.0 → 0.2.0 captures the change explicitly          |
| DAP socket protocol grows to be unwieldy                                             | Namespace chart events under `stateChart/`; keep mflowLink + control-flow events on the same socket but in disjoint namespaces |

## 9. Open questions

- **Hierarchy carrier**: `FlowNode.parent` (proposed) vs. nested arrays
  in JSON. Going with `parent` because it matches Stateflow's flat-id-
  with-parent model and keeps git diffs readable on reparent.
- **Transition encoding**: new `EdgeKind.transition` (proposed) vs.
  dedicated `Transition` struct. Going with edge extension to reuse
  routing/marquee/clipboard machinery; the four-field label parser
  absorbs the extra complexity.
- **Default transitions**: explicit `.junction_default` node (proposed)
  vs. `params.isInitial: Bool` flag on the substate. Support both: the
  flag is the canonical form, the node is sugar for users who prefer
  the bullet visual.
- **History junctions**: same dual: explicit `.junction_history` node
  vs. `params.hasHistory: Bool` on the parent.
- **Action language**: MATLAB-only at launch is firm. The C lane lands
  after the MATLAB lane is solid, gated by a parser-conversion sweep.
- **Snapshot buffer sharing**: chart and mflowLink share one ring (one
  buffer per simulation session). Each entry is tagged with its source
  dialect so the IDE can render them in the right transport row.
- **Window sharing**: one chart, one `StateChartWindow`. A future
  enhancement (Tier-N+) might let a `signal_subsystem` containing a
  chart open a chart-window from the signal-flow context — out of
  scope for now.

## 10. Cross-reference index

| Topic                              | Stateflow guide        | mStateflow doc / file                                                  |
|---|---|---|
| Finite state machine               | §1-2                   | §1 north-star, §3 mental model                                         |
| Stateflow objects overview         | §1-5                   | §4.2/§4.3 schema; §3 mental model                                      |
| Hierarchy + decomposition          | §1-27 / §1-29          | §4.4 (params); §5.3 canvas; Tier 2                                     |
| Air Temperature Controller         | §1-31                  | §6.8 fixture; Tier 2 verification target                               |
| Default transitions                | §1-41                  | §4.3 `junction_default`; §4.4 `params.isInitial`                       |
| Supertransitions                   | §1-45                  | §4.5 — no schema work; §5.3 routing                                    |
| Connective junctions               | §1-56                  | §4.3 `junction_connective`                                             |
| History junctions                  | §1-60                  | §4.3 `junction_history`; §4.4 `params.hasHistory`                      |
| Entry / Exit ports                 | §1-63                  | §4.3 `junction_entry` / `junction_exit`                                |
| Chart execution / super-step       | §2-14 / §2-37          | §6.3 lowering; Tier 4                                                  |
| Evaluate transitions               | §2-29                  | §6.3 priority-sorted dispatch                                          |
| Flow charts inside charts          | §3                     | reuse existing `EdgeKind.control`; no schema work                      |
| Mealy / Moore                      | §5                     | Tier-N+ deferred                                                       |
| Boxes / subcharts                  | §6                     | §4.4 `params.containerStyle`                                           |
| Graphical / MATLAB / Truth Table fns | §6-10 / §7 / §8        | §4.3 `chart_fn_*`; Tier 8                                              |
| Simulink Functions in charts       | §9                     | non-goal (would require fourth dialect)                                |
| Data definition / scope            | §10                    | §4.1 `Flow.symbols.data`                                               |
| Active state data                  | §11                    | §5.6 Active State pane; Tier 10 active-state output port               |
| Events                             | §12                    | §4.1 `Flow.symbols.events`; §6.7 DAP `eventBroadcast`                  |
| Messages                           | §13                    | §4.1 placeholder; Tier-N+ deferred                                     |
| Temporal logic operators           | §14-45                 | §6.3 counter-based lowering                                            |
| MATLAB / C action language         | §15                    | §4.4 toggle; MATLAB-only at launch                                     |
| State Transition Tables            | §16                    | §5.5; Tier 7                                                           |
| Atomic Subcharts                   | §17                    | Tier-N+ deferred                                                       |
| Operating Points                   | §18                    | §5.9 snapshot ring; Tier 6                                             |
| Vectors / matrices                 | §19                    | inherited from matlabc front-end                                       |
| Enumerated data                    | §20                    | inherited from matlabc front-end                                       |
| Strings                            | §21                    | inherited from matlabc front-end                                       |

---

**Backend status (matlab_llvm side, 2026-05-17)**: All shipped tiers
green: 0 / 4 / 6 / 8 / 9 / 10 closed end-to-end. 67/67 chart fixtures
pass (10 §6.8 + 4 extras + 4 schema-error + 3 follow-on:
`junction_backtrack`, `temporal_counter`, `chart_fn_truth_table`).
DAP + introspection + breakpoints + interpreter all production-ready;
the lowering matches the interpreter on connective-junction
backtracking. Chart `.mflow` files lower through **every** matlabc
emit lane — `-emit-matlab` / `-emit-mir` / `-emit-llvm` / `-emit-c`
/ `-emit-cpp` for software, `-emit-systemverilog` for synthesizable
RTL (Moore / Mealy / AND-parallel charts produce verilator-lint-
clean modules; float-typed charts now hard-error with a pointer at
the integer-typed examples). The matlabc REPL gains a
`loadStateChart('foo.mflow')` shortcut that emits + sources a
chart in one call.

**IDE status (Matlab_llvm_ide, 2026-05-17)**: Tiers 0 through 10
shipped on the UI/UX side. Chart `.mflow` files open in a dedicated
`StateChartWindow` (palette / canvas / inspector + Active State /
Snapshot side panes + bottom Event Log strip), edit through the
full Stateflow visual + STT + Truth-Table editors, run + step
through the live `matlabc -simulate --sim-dap` lane with
state/transition breakpoints, and export through every codegen
lane the chart adapter accepts. 860 unit tests cover the schema
round-trip, lint, simulation wiring, codegen dispatch, Pattern
Wizard, TeX renderer, and chart-from-selection refactor (16
chart-specific suites, 79 chart-specific cases).

**Remaining backend follow-ons**: see "What's NOT shipped" in
§Status — `duration(cond)` interpreter parity (lowering has it
already), plus the Tier-N+ deferrals (C action language, Messages,
Mealy/Moore conversion sweeps, Atomic Subcharts, Simulink-based
states, multi-chart Sequence Viewer).
