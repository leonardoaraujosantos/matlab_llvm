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
| `signal_clock`        | ✓ | *(none)*                                                                  | Tier-H — outputs the current simulation time `t` |
| `signal_chirp`        | ✓ | `amplitude: 1.0`, `f0: 0.1`, `f1: 1.0`, `t1: 10.0`                        | Tier-H — linear frequency sweep `f0 → f1` over `[0, t1]` |
| `signal_noise`        | ✓ | `amplitude: 1.0`, `seed: 1.0`, `kind: "uniform"`                          | Tier-H — uniform `[-amp, +amp]` or `kind: "gaussian"` (σ = amp) via xorshift64 + Box-Muller. Seed is per-block for reproducibility |
| `signal_from_workspace` |  | reserved                                                                | Needs workspace var binding (no equivalent in our runtime today) |
| `signal_function_call_generator` | ✓ | `period: 1.0`, `phaseDelay: 0.0`                               | Tier-F carve-out — emits `1` over a 1.5×step window at every `period` boundary, `0` otherwise. Designed to drive `signal_triggered_subsystem` via a rising edge. |

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
| `signal_zero_pole`    | ✓ | `zeros: ""`, `poles: "-1"`, `gain: 1.0`                  | Tier-H — real zeros/poles in comma-separated form. Constructor expands ZPK → num/den then routes through the transfer-fcn evaluator. Complex pole pairs are a follow-up |
| `signal_transport_delay` | ✓ | `delay: 0.0`, `initialOutput: 0.0`                    | Tier-H — pure time delay via a circular history buffer; linear-interpolates the input at `t − delay`. Loop-breaker because the delayed value never depends on this-step input |

## Discrete

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_unit_delay`   | ✓ | `initialValue: 0.0`, `sampleTime: 1.0`                   | Multi-rate scheduler latches `u[n-1]` at each tick (Tier E) |
| `signal_zoh`          | ✓ | `sampleTime: 1.0`                                        | Sample-and-hold; output latched at each tick boundary |
| `signal_discrete_integrator` | ✓ | `method: "ForwardEuler"`, `initialCondition: 0.0`, `sampleTime: 1.0` | Tier-H — discrete-time accumulator. ForwardEuler ships; BackwardEuler / Trapezoidal accept the param but use the same single-sample approximation pending sub-sample sampling |
| `signal_discrete_filter` | ✓ | `num: "1"`, `den: "1, -0.9"`, `sampleTime: 1.0`       | Tier-H — z-domain IIR. Implements the pole half (feedback taps) of direct-form-II; pure-FIR / mixed designs need the `u`-history buffer (follow-up) |
| `signal_rate_transition` | ✓ | `sampleTime: 1.0`                                    | Tier-H — bridges between different sample rates; behaviourally a ZOH at the requested rate |

## Math

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_gain`       | ✓ | `gain: 1.0`                                              | |
| `signal_sum`        | ✓ | `signs: "++"`                                            | One character per input port (`+` / `-`); port-name order is `in1, in2, …` |
| `signal_product`    | ✓ | `numInputs: 2.0`                                         | Multiplies `in1 · in2 · … · inN` |
| `signal_abs`        | ✓ | *(none)*                                                 | |
| `signal_saturation` | ✓ | `upperLimit: 1.0`, `lowerLimit: -1.0`                    | Registers a zero-crossing predicate (Tier-E root-finding) |
| `signal_math_fcn`   | ✓ | `function: "sqrt"`                                       | Tier-H — `sqrt`, `exp`, `log`, `log10`, `abs`, `sign`, `square`, `reciprocal`, `pow` (two inputs), `hypot` (two), `mod` (two), `rem` (two) |
| `signal_trig_fcn`   | ✓ | `function: "sin"`                                        | Tier-H — `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` (two inputs), `sinh`, `cosh`, `tanh` |
| `signal_dead_zone`  | ✓ | `lowerLimit: -0.5`, `upperLimit: 0.5`                    | Tier-H — output is 0 inside `[lo, hi]`, `(u − hi)` above, `(u − lo)` below |
| `signal_relop`      | ✓ | `op: "<"`                                                | Tier-H — `==`/`!=`/`<`/`<=`/`>`/`>=` on `in1` vs `in2`; output is 0 or 1 |
| `signal_logical`    | ✓ | `op: "AND"`                                              | Tier-H — `AND`/`OR`/`NAND`/`NOR`/`XOR` over every connected input, or `NOT` on `in1`. Truthiness keyed off `value ≠ 0` |
| `signal_compare_to_zero`     | ✓ | `op: ">"`                                       | Tier-H — predicate on input vs. zero |
| `signal_compare_to_constant` | ✓ | `op: ">"`, `constant: 0.0`                      | Tier-H — predicate on input vs. constant |
| `signal_matlab_fcn`          | ✓ | `expression` (string, required)                 | Tier-H carve-out — inline MATLAB expression evaluator. Variables `u1`..`uN` are the input ports (also `u` ≡ `u1`), `t` is simulation time, plus `pi` and `e`. Operators `+ - * / ^ .* ./ .^` and unary `-`. Functions: sin/cos/tan/asin/acos/atan/atan2, sinh/cosh/tanh, exp/log/log10/log2, sqrt/abs/sign/floor/ceil/round, min/max/mod/rem/pow/hypot/square. Expressions are parsed once at construction and rejected at lowering with a sourced diagnostic on syntax failure |
| `signal_relay`      | ✓ | `onPoint: 0.5`, `offPoint: -0.5`, `onValue: 1.0`, `offValue: 0.0`, `initialState: 0.0` | Tier-E carve-out — hysteretic on/off switch. State flips at major-step boundaries only; the dead-band between `offPoint` and `onPoint` gives the latched output its persistence. Registers two zero-crossing predicates so the bisector lands transitions sub-step accurately. |

## Signal routing

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_mux`    | stub | `numInputs: 2.0`             | Vector signal type pending (Tier H+); current evaluator passes through `in1` |
| `signal_demux`  | stub | `numOutputs: 2.0`            | Same caveat |
| `signal_switch` | stub | `threshold: 0.0`             | Tier-E: zero-crossing on `in2 − threshold`; current evaluator passes through `in1` |
| `signal_multiport_switch` | ✓ | `defaultOutput: 0.0`     | Tier-H — `in1` is the 1-based selector; `in2`, `in3`, … are the data lines. Out-of-range selectors fall through to `defaultOutput` |
| `signal_merge`            | ✓ | `initialOutput: 0.0`     | Tier-H — first non-zero input in port order wins. Falls through to `initialOutput` when every input is zero |
| `signal_goto`             | ✓ | `tag` (in `data`, not `params`) — broadcast channel name      | Tier-H carve-out — virtual wire. Sink with one incoming edge; the Flattener's `contractGotoFrom` pass rewires every matching `signal_from`'s outgoing edges to the goto's source and drops both kinds from the IR |
| `signal_from`             | ✓ | `tag` (in `data`, not `params`)                               | Tier-H carve-out — paired with `signal_goto`. Reading from an unknown tag is a sourced lowering error |

## Lookup tables

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_lookup_1d` | ✓ | `breakpointsX: ""`, `tableData: ""`                                  | Tier-H — comma-separated breakpoint vector + same-length output vector. Linear interp; out-of-range inputs clamp to the endpoints |
| `signal_lookup_2d` | ✓ | `breakpointsX: ""`, `breakpointsY: ""`, `tableData: ""`              | Tier-H — bilinear interp. `tableData` is row-major `len(X) × len(Y)` |
| `signal_lookup_nd` |   | reserved                                                              | Generalisation pending the 1d/2d pair settling |

## Composite

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_subsystem` | ✓ | `flow_id` (in `data`, *not* `params`) — id of the sub-flow | Flattened during lowering (§6.2); runtime never sees a subsystem |
| `signal_inport`    | ✓ | `port` (in `data`) — optional external-port binding         | Contracted into the parent during flattening |
| `signal_outport`   | ✓ | `port` (in `data`) — optional external-port binding         | Contracted into the parent during flattening |
| `signal_enabled_subsystem`   | ✓ | `flow_id` + `enable_block` (in `data`) — id of a sibling block whose output drives the gate | Tier-F: flattens like `signal_subsystem`; every inlined leaf inherits the gate. Runtime holds outputs / zeros derivatives while gate ≤ 0 |
| `signal_triggered_subsystem` | ✓ | `flow_id` + `enable_block` (in `data`)                                                       | Tier-F carve-out — proper rising-edge semantics: the gated subtree fires for exactly one major step on each `0 → 1` transition of the `enable_block`'s output, then resets its outputs to zero (Simulink's "Output when disabled: reset"). Typically paired with `signal_function_call_generator`. |

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

## Tier H+ carve-outs (still reserved, with prerequisites)

These kinds are accepted by the loader (round-trip clean) but rejected
at lowering. Each needs a prerequisite that lives outside the lone
Tier-H "ship more evaluators" axis:

| Kind | Prerequisite |
|---|---|
| `signal_bus_creator`, `signal_bus_selector` | First-class vector / struct signal type in the IR. Today every wire is scalar `double` — a bus needs a composite signal carrier plus per-port type checking. Roadmap §16 punts this until the scalar path is solid. |
| `signal_from_workspace` | Mechanism to bind a runtime `simout`-style variable into the simulation as a time-indexed source. The matlabc workspace model and the mflowLink runtime currently share no such handle. |
| `signal_custom` | Plugin layer for user-defined evaluators. The shipped `signal_matlab_fcn` covers the inline-expression case; `signal_custom` would let the IDE register evaluators implemented elsewhere (a `.cpp` the user compiles into the runtime, a remote service, …). Needs a registry + ABI design. |
| `signal_if_action`, `signal_switch_case_action` | A parent `signal_if_subsystem` / `signal_switch_case_subsystem` container that scopes which case fires. The IDE's `SignalFlowParamSpec` doesn't ship these containers yet — IDE-side prerequisite. |
| `signal_lookup_nd` | Settle the `lookup_1d` / `lookup_2d` pair (breakpoint shape + extrapolation policy + cached table format) before generalising to N dimensions. |

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
