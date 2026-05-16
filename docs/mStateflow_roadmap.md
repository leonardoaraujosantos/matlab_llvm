# mStateflow — Hierarchical State Charts on top of `.mflow` (rev. 2026-05-16)

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

## Status (2026-05-16)

**Nothing is shipped.** This document is a green-field plan. The
existing matlab_llvm has *zero* state-chart awareness:

- the `.mflow` loader (`lib/Flowchart/Loader.cpp`) only switches between
  `control_flow` and `signal_flow` kinds,
- there is no chart IR, no chart runtime, no chart DAP adapter,
- the IDE's flowchart canvas has no hierarchy primitives (every
  `FlowNode` sits at chart root today),
- `FlowNode` has no `parent` field,
- `EdgeKind` only knows `.control` and `.data`.

The plan below is organised so Tier 0 (additive schema + palette stubs)
lands first as a tiny PR; everything downstream forks into two parallel
tracks — **UI/UX** (§5) and **Compiler / runtime** (§6) — that converge
at Tier 4 for live debug.

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
- **HDL coder for charts.** Digital-only state machines may eventually
  lower through `-emit-sv`, but not in scope here.
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

Chart IR → existing matlabc IR. Strategy:

- Active state encoded as a packed `uint8`/`uint16` vector indexed by
  region id (one slot per OR region; AND regions encode their children
  as a bitset).
- Each chart compiles to a single tick function
  `chart_tick(state, events, inputs) → (state, outputs)` that mutates
  the active-state vector + chart-local data block.
- Transitions become a priority-sorted dispatch table per region.
- Entry / during / exit / on-event actions are inlined MATLAB ASTs.
- Super-step runs as a fixed-point loop with `kMaxIterations` (default
  1000) and a runtime warning on saturation (parity with §2-41).
- Temporal operators (`after`, `before`, `every`, `at`, `duration`,
  `temporalCount`) lower to counter increments on event broadcasts
  (parity with §14-45).
- The `in(state)` operator lowers to a bitmask check against the
  active-state vector.

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

### 6.6 Codegen lanes (free, mostly)

Each gains one fixture test per canonical chart example.

- `-emit-matlab` — readable MATLAB equivalent of the chart.
- `-emit-c` / `-emit-cpp` — chart compiles to nested `switch`-cases +
  state vector + event dispatch.
- `-emit-llvm` / `-emit-mlir` — flow through existing pipelines once
  the IR is hooked up.
- **HDL / Verilog-A** — explicit non-goal.

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

### Tier 0 — Foundations *(Both, S)*

**Goal**: minimum schema + routing so chart `.mflow` files load and
save without breaking existing dialects.

- **[Compiler]** `SchemaKind.state_chart` accepted by loader; new
  optional fields parse (`FlowNode.parent`, `ui.size`, `data.params.*`,
  `Flow.symbols`); validation rejects malformed hierarchies.
- **[UI/UX]** `SchemaKind` mirror in the IDE codable layer; new node-
  kind stubs (palette icons drawn, no canvas behaviour yet).
- **[UI/UX]** "New → State Chart…" file template; Project Explorer
  badge; routing to a stub `StateChartWindow` (empty canvas).
- **[Both]** Schema bump 0.1.0 → 0.2.0; back-compat tests for existing
  control-flow and signal-flow fixtures.

### Tier 1 — Flat authoring canvas *(UI/UX, M)*

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

### Tier 2 — Hierarchy + decomposition *(UI/UX, M)*

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

### Tier 3 — Symbols + action-language editor *(UI/UX, M)*

**Goal**: production-grade action authoring with autocomplete.

- Inspector Symbols tab — data / events / messages tables.
- Action editor: multi-line, MATLAB tokenizer reused; autocomplete on
  symbol names, temporal operators, `in()`.
- Lint: parser errors, type mismatches, references to symbols outside
  scope.

**Verifies**: the bang-bang controller's actions (§14-62) author with
the same editing quality the existing `.m` editor provides.

### Tier 4 — Chart compiler + runtime *(Compiler, L)*

**The single largest chunk; gates everything below.**

- `lib/StateChart/StateChartIR.{h,cpp}` — chart, region, state,
  transition, junction, symbol tables.
- `lib/StateChart/Lowering.cpp` — chart IR → matlabc IR; super-step
  fixed-point loop with `kMaxIterations`; temporal/`in()` operators
  implemented.
- `lib/Runtime/runtime_mstateflow.cpp` — event queue, broadcast
  helper, active-state bitset, snapshot ring shared with mflowLink.
- CLI: `-simulate` detects `state_chart`; `-emit-matlab` emits a
  readable equivalent.
- DAP adapter: chart-namespaced events and requests (§6.7).
- Golden-file fixtures for the canonical examples in §6.8.

**Verifies**: command-line simulation of all six reference examples
produces deterministic, reference-matching traces.

### Tier 5 — Live debug surface *(UI/UX, M)*

Builds directly on Tier 4's DAP events.

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

### Tier 6 — Snapshots + step-back *(Both, S)*

Reuses mflowLink's snapshot ring buffer.

- **[Compiler]** Chart-runtime emits a snapshot every super-step
  boundary; DAP request `stateChart/saveOperatingPoint` named-tags one.
- **[UI/UX]** Transport row gains "Save snapshot" / "Restore snapshot";
  named snapshots in a side panel; "Start sim from this point" sets the
  next-run's initial active-state vector.

**Verifies**: parity with §18 of the guide (Operating Points).

### Tier 7 — Tabular alternatives *(UI/UX, M)*

- State Transition Table editor: docked grid; two-way sync with canvas.
- Truth Table editor: condition × decision grid + action row;
  over/underspecification diagnostic (§8-27).
- `convertToChart` / `convertToSTT` right-click actions.

**Verifies**: the bang-bang controller authored as an STT (§16-16)
produces an identical simulation trace to its canvas equivalent.

### Tier 8 — Functions + reuse *(UI/UX + Compiler, M)*

- **[UI/UX]** Graphical / MATLAB / Truth Table function call-site
  nodes; navigate-into-sub-`Flow` machinery (reused from
  `signal_subsystem`).
- **[Compiler]** Function inlining at chart-lowering time; recursion
  detection.

### Tier 9 — Codegen lanes *(Compiler, S)*

Mostly free once the chart IR lowers to the existing matlabc IR. One
fixture test per lane per canonical example.

- `-emit-c`, `-emit-cpp`, `-emit-llvm`, `-emit-mlir`.

### Tier 10 — Polish *(UI/UX, S, scattered)*

- Pattern Wizard (debouncer / edge-detector / fault-handler templates).
- TeX in annotations (parity with §6-24).
- Chart-from-selection: refactor flat `if` / `function_definition`
  control-flow nodes into a chart.
- Active-state output port — the chart publishes its current active
  state as an enum output, mirroring §11-38. Useful for cross-chart
  wiring inside an mflowLink signal-flow document.
- Examples library shipped under `Examples/StateCharts/*.mflow`.

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

**Suggested first PR**: Tier 0 alone — schema additions, file template,
empty `StateChartWindow`. It unlocks parallel work on Tier 1 (canvas)
IDE-side and on the matlab_llvm chart loader, both of which can
progress independently until Tier 4 brings them together.
