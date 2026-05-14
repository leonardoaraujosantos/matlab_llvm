# mflowLink — Per-Block Parameter Catalogue

The single source of truth for signal-flow block `params` keys. The
IDE's `SignalFlowParamSpec` catalogue (`Matlab_llvm_ide/matlab_llvm/
matlab_llvm/Models/FlowchartModels.swift:1131–1267`) defines these on
the authoring side; `lib/Flowchart/SignalFlowLowering.cpp` and
`lib/Flowchart/MflowLinkSim.cpp` read the same keys on the runtime
side. **Update this file in lockstep with the IDE** — without one
source of truth, the IDE and the runtime silently disagree on
parameter names and the model "runs" with wrong numbers.

> **Important — naming convention.** The IDE's `SignalFlowParamSpec`
> serialises param keys **camelCase** (`stepTime`, `initialCondition`,
> `upperLimit`, …), not the snake_case sketch in `mflow_link_roadmap.md`
> §13. This file is authoritative; the roadmap §13 will be aligned in
> a follow-up.

The companion *node-level* `data` fields are **snake_case** —
`sample_time`, `data_type`, `log_signal`. Only `params` is camelCase.

Tier-C support column: `✓` = evaluator shipped in
`lib/Flowchart/MflowLinkSim.cpp`. Anything else is reserved
(round-trips through the loader, rejected at lowering with a sourced
diagnostic) until its evaluator lands.

## Sources

| Kind | Tier-C | Params (camelCase key → default) | Notes |
|---|---|---|---|
| `signal_constant`     | ✓ | `value: 1.0`                                                              | One-time evaluation; `SampleClass = Constant` |
| `signal_step`         | ✓ | `stepTime: 1.0`, `initialValue: 0.0`, `finalValue: 1.0`                   | `y = t ≥ stepTime ? finalValue : initialValue` |
| `signal_sine`         | ✓ | `amplitude: 1.0`, `bias: 0.0`, `frequency: 1.0`, `phase: 0.0`             | `y = amplitude · sin(frequency · t + phase) + bias` (radians) |
| `signal_pulse`        | ✓ | `amplitude: 1.0`, `period: 1.0`, `pulseWidth: 50.0`, `phaseDelay: 0.0`    | `pulseWidth` is *percent* of period |
| `signal_ramp`         | ✓ | `slope: 1.0`, `startTime: 0.0`, `initialOutput: 0.0`                      | |
| `signal_chirp`        |   | reserved                                                                  | |
| `signal_noise`        |   | reserved                                                                  | |
| `signal_from_workspace` | | reserved                                                                | |
| `signal_clock`        |   | reserved                                                                  | |

## Sinks

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_scope`         | ✓ | `yMin: -1.0`, `yMax: 1.0`, `title: ""`, `decimation: 1` | Implicitly logged; CSV column = block id |
| `signal_display`       | ✓ | *(none)*                                                | Implicitly logged |
| `signal_to_workspace`  | ✓ | `variableName: "simout"`                                | CSV column = `variableName` |
| `signal_terminator`    | ✓ | *(none)*                                                | Drops the signal |

## Continuous-time

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_integrator`   | ✓ | `initialCondition: 0.0`                                  | One continuous state; loop-breaker |
| `signal_derivative`   |   | *(none)*                                                 | Reserved — needs filtered derivative |
| `signal_transfer_fcn` | ✓ | `num: "1"`, `den: "1, 1"`                                | Comma-separated coefficients, highest order first. Tier-C: strictly proper only (`degNum < degDen` ⇒ loop-breaker) |
| `signal_state_space`  | ✓ | `A: "0"`, `B: "1"`, `C: "1"`, `D: 0.0`, `x0: "0"`        | MATLAB-style matrix literals (`"[0 1; -1 0]"`). Tier-C: SISO with `D = 0` |
| `signal_zero_pole`    |   | reserved                                                 | |
| `signal_transport_delay` | | reserved                                              | |

## Discrete

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_unit_delay`   | ✓ | `initialValue: 0.0`, `sampleTime: 1.0`                   | Multi-rate scheduler latches `u[n-1]` at each tick (Tier E) |
| `signal_zoh`          | ✓ | `sampleTime: 1.0`                                        | Sample-and-hold; output latched at each tick boundary |
| `signal_discrete_integrator` | | reserved                                          | |
| `signal_discrete_filter`     | | reserved                                          | |
| `signal_rate_transition`     | | reserved                                          | |

## Math

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_gain`       | ✓ | `gain: 1.0`                                              | |
| `signal_sum`        | ✓ | `signs: "++"`                                            | One character per input port (`+` / `-`); port-name order is `in1, in2, …` |
| `signal_product`    | ✓ | `numInputs: 2.0`                                         | Multiplies `in1 · in2 · … · inN` |
| `signal_abs`        | ✓ | *(none)*                                                 | |
| `signal_saturation` | ✓ | `upperLimit: 1.0`, `lowerLimit: -1.0`                    | Registers a zero-crossing predicate (Tier-E root-finding) |
| `signal_math_fcn`   |   | reserved                                                 | |
| `signal_trig_fcn`   |   | reserved                                                 | |
| `signal_dead_zone`  |   | reserved                                                 | |
| `signal_relop`      |   | reserved                                                 | |
| `signal_logical`    |   | reserved                                                 | |

## Signal routing

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_mux`    | stub | `numInputs: 2.0`             | Tier-H: vector signal type; current evaluator passes through `in1` |
| `signal_demux`  | stub | `numOutputs: 2.0`            | Same caveat |
| `signal_switch` | stub | `threshold: 0.0`             | Tier-E: zero-crossing on `in2 − threshold`; current evaluator passes through `in1` |

## Composite

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_subsystem` | ✓ | `flow_id` (in `data`, *not* `params`) — id of the sub-flow | Flattened during lowering (§6.2); runtime never sees a subsystem |
| `signal_inport`    | ✓ | `port` (in `data`) — optional external-port binding         | Contracted into the parent during flattening |
| `signal_outport`   | ✓ | `port` (in `data`) — optional external-port binding         | Contracted into the parent during flattening |
| `signal_enabled_subsystem`   | ✓ | `flow_id` + `enable_block` (in `data`) — id of a sibling block whose output drives the gate | Tier-F: flattens like `signal_subsystem`; every inlined leaf inherits the gate. Runtime holds outputs / zeros derivatives while gate ≤ 0 |
| `signal_triggered_subsystem` | ✓ | `flow_id` + `enable_block` (in `data`)                                                       | Tier-F: same level-gated semantics as `signal_enabled_subsystem` today. Proper edge-triggered semantics (fire on `0 → 1` transition only) is Tier-H |

## Node-level data fields (not `params`)

These live directly under `data` (not nested in `params`) and use
**snake_case** keys, per the IDE codec's `CodingKeys` rename:

| Key | Type | Default | Notes |
|---|---|---|---|
| `sample_time`  | string  | `"inherited"` | `"continuous"` \| `"inherited"` \| `"<seconds>"` |
| `units`        | string  | `""`          | Engineering-unit string, advisory |
| `data_type`    | string  | `"double"`    | `"double"` \| `"single"` \| `"int8"` … |
| `log_signal`   | bool    | `false`       | Stream this block's output (§7.6) |
| `enable_block` | string  | `""`          | Tier-F gate. On a `signal_enabled_subsystem` / `signal_triggered_subsystem`, names a sibling block whose output drives the gate (every inlined sub-block picks it up). On any other block, gates that block individually. Empty ⇒ always enabled. |
| `params`       | object  | `{}`          | Block parameters per the table above |

## Cross-repo invariants

These are the contracts a `flowchart-simulate-tests` regression would
diff against the IDE side:

1. Every `signal_*` kind in `SignalFlowParamSpec` appears in this
   file. A kind here without a matching IDE spec round-trips through
   the loader but can never originate from a user-edited diagram.
2. Every key listed for a kind matches the IDE's `SignalFlowParamSpec`
   exactly (camelCase, no typos). The IDE's
   `roundTripSignalFlowDocument` test (`FlowchartTests.swift:648–690`)
   pins the casing on the authoring side.
3. Defaults match the IDE's `SignalFlowParamSpec`. If the runtime
   needs a different default for some reason (e.g. a numerical
   stability guard), the IDE must change too.
