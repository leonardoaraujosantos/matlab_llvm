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
| `signal_awgn`         | ✓ | `snr: 10.0`, `signalPower: 1.0`, `seed: 1.0`                              | Communications (#343) — AWGN channel `y = x + N(0, σ²)`, `σ² = signalPower / 10^(snr/10)` (Simulink "SNR + input signal power" mode). **Vector-aware**: a width-W input (e.g. a complex `[I, Q]` symbol) gets an independent N(0, σ²) draw per component, so it models a noisy PSK/QAM link end-to-end. xorshift64 + Box-Muller generator |
| `signal_error_rate`   | ✓ | `tolerance: 0.5`                                                          | Communications (#343) — error-rate (BER) sink. Ports `tx`/`rx` (or `in1`/`in2`); output is the running mismatch ratio (a symbol counts as different when `\|tx − rx\| > tolerance`). Accumulated once per major step (not per RK4 substep), output bounded in `[0, 1]` |
| `signal_psk_mod`      | ✓ | `M: 4`, `phaseOffset: 0.0`                                               | Communications (#343) — M-PSK modulator. Symbol index → constellation point `exp(j(2πk/M + φ))`, output as a width-2 `[I, Q]` vector |
| `signal_psk_demod`    | ✓ | `M: 4`, `phaseOffset: 0.0`                                               | Communications (#343) — M-PSK demodulator. Nearest-angle hard decision on a `[I, Q]` input → scalar symbol index. Exact inverse of `signal_psk_mod` on clean points |
| `signal_qam_mod`      | ✓ | `M: 16`, `normalize: false`                                              | Communications (#343) — square M-QAM modulator (M = 4/16/64…). Symbol → `[I, Q]` on the L×L odd-integer grid (L=√M); `normalize` scales to unit average power |
| `signal_qam_demod`    | ✓ | `M: 16`, `normalize: false`                                              | Communications (#343) — square M-QAM demodulator. Nearest-grid-point hard decision on `[I, Q]` → scalar symbol. Exact inverse of `signal_qam_mod` |
| `signal_running_stats`| ✓ | `stat: "mean"`                                                           | Statistics (#343) — streaming `mean` / `var` / `std` over the input via an online Welford accumulator (numerically stable, single pass). Updated once per major step. Beats a MATLAB Function block, which can't hold persistent state in the flow today |
| `signal_kalman`       | ✓ | `A`, `C`, `Q`, `R`, opt `B`, `x0`, `P0` (matrix literals)                | Sensor Fusion (#343) — discrete Kalman filter. Ports `z` (measurement, `Mz`-vector), opt `u` (control, `P`-vector); output is the `N`-vector state estimate (`N` = rows of `A`). Standard predict (`x⁻=A·x+B·u`, `P⁻=A·P·Aᵀ+Q`) / update (`K=P⁻·Cᵀ·(C·P⁻·Cᵀ+R)⁻¹`) recursion runs once per major step. Matrices are MATLAB literals (`"1 0.05; 0 1"`). See the [worked examples](#kalman-filter-worked-examples-343) |
| `signal_fft`          | ✓ | `n` (frame size, required)                                               | DSP (#343) — frame DFT of a real `n`-point vector input. Output is the complex spectrum packed as `[Re₀…Re₍ₙ₋₁₎, Im₀…Im₍ₙ₋₁₎]`, width `2n`. O(n²) direct DFT (no FFTW); pairs with `signal_ifft`. See [DSP frame examples](#dsp-frame-transforms-343) |
| `signal_ifft`         | ✓ | `n` (frame size, required)                                               | DSP (#343) — inverse DFT. Input is a complex `[Re;Im]` width-`2n` frame; output is the real `n`-point reconstruction. `fft → ifft` is the identity |
| `signal_window`       | ✓ | `n` (frame size), `type: "hann"`                                         | DSP (#343) — windowing taper over an `n`-point frame. `type`: `hann` (default), `hamming`, `blackman`, `rect`. Output `w[k]·x[k]` (width `n`) |
| `signal_biquad`       | ✓ | `b`, `a` (or `b0..b2`/`a0..a2`), `sampleTime: 1.0`                       | DSP (#343) — streaming 2nd-order section (SOS): `H(z)=(b0+b1z⁻¹+b2z⁻²)/(a0+a1z⁻¹+a2z⁻²)`. Coefficients as space/comma vectors (`b: "0.07 0.13 0.07"`) or named scalars. Runs the discrete-filter difference engine on its `sampleTime` grid; 2 state slots. Cascade instances for higher-order IIR |
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
| `signal_integrator`   | ✓ | `initialCondition: 0.0`                                  | One continuous state; loop-breaker. Optional `reset` input port: on a rising edge (`prev ≤ 0 && now > 0`) the continuous state is reloaded at the next major step from the `init` input port if connected, else from `initialCondition` — the zero-crossing → state-reset pattern (see `examples/mflowlink/bouncing_ball.mflow`) |
| `signal_pid`          | ✓ | `Kp: 1.0`, `Ki: 0.0`, `Kd: 0.0`, `N: 100.0`, `initialIntegral: 0.0`, `upperLimit: +inf`, `lowerLimit: -inf` | Parallel form `C(s) = Kp + Ki/s + Kd·N/(s+N)`. Two continuous states (integral + derivative-filter). Direct-feedthrough ⇒ **not** a loop-breaker. Optional output saturation with clamping anti-windup (integrator frozen while pinned to a limit) |
| `signal_derivative`   |   | *(none)*                                                 | Reserved — needs filtered derivative |
| `signal_transfer_fcn` | ✓ | `num: "1"`, `den: "1, 1"`                                | Comma-separated coefficients, highest order first. Tier-C: strictly proper only (`degNum < degDen` ⇒ loop-breaker) |
| `signal_state_space`  | ✓ | `A: "0"`, `B: "1"`, `C: "1"`, `D: 0.0`, `x0: "0"`        | MATLAB-style matrix literals (`"[0 1; -1 0]"`). SISO input, `D = 0`. `x0` may be a per-state vector (`"2; 0"`); a multi-row `C` drives distinct output ports `out1..outP = (C·x)_k` (#345) |
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

## HDL / digital sequential (#343)

Clocked registers driven by an external `clk` **rising edge** (posedge), not a
fixed sample rate. They update once per major step (a single clock edge → a
single update); the held value is the output, and they are loop-breakers like
`unit_delay`. An optional active-high `reset`/`rst` input asynchronously
reloads `initialValue`. Mux/Demux and logic gates already ship as
`signal_mux`/`signal_demux`/`signal_logical`/`signal_multiport_switch`.

**The clocked registers synthesise to RTL.** `matlabc -emit-sv <model.mflow>
--subsystem <name>` lowers each to a posedge register —
`always_ff @(posedge clk or negedge rst_n) s_ff <= s_ff_next` with the output
`= s_ff` and a per-block next-state:

| Block | `s_ff_next` (combinational) |
|---|---|
| `signal_dff`     | `D` |
| `signal_tff`     | `Q + T*(1 - 2*Q)` (= `1-Q` when `T` is high or unwired; arithmetic toggle, no branch) |
| `signal_counter` | `Q + step`, wrapped via `inc - mod*(inc >= mod)` when `modulus > 0` |

All three pass `-check-synthesizable`. The block's `clk` input maps to the
module's implicit clock (single-clock design), so for synthesis leave `clk`
unwired (the module `clk` is the clock); `reset` maps to the module reset. See
`examples/mflowlink/coder/{dff,tff,counter}_register.mflow`.

| Kind | Tier-C | Params | Ports | Notes |
|---|---|---|---|---|
| `signal_dff`     | ✓ | `initialValue: 0.0`            | `d`/`in`, `clk`, opt `reset` | D flip-flop — on `clk` posedge `Q ← D`; holds otherwise. `always @(posedge clk) Q <= D` |
| `signal_tff`     | ✓ | `initialValue: 0.0`            | opt `t`/`in`, `clk`, opt `reset` | T flip-flop — toggles `Q` on `clk` posedge when `t` is high (free-toggles when `t` unconnected) |
| `signal_counter` | ✓ | `step: 1.0`, `modulus: 0.0`   | `clk`, opt `reset` | Up counter — `+step` per `clk` posedge; wraps at `modulus` (> 0) |
| `signal_jkff`    | ✓ | `initialValue: 0.0`           | `j`/`in1`, `k`/`in2`, `clk`, opt `reset` | JK flip-flop — on `clk` posedge: `00` hold, `01` reset, `10` set, `11` toggle |
| `signal_srff`    | ✓ | `initialValue: 0.0`           | `s`/`in1`, `r`/`in2`, `clk`, opt `reset` | SR flip-flop — on `clk` posedge: `10` set, `01` reset, `00`/`11` hold (`11` is undefined in HW) |

**Example circuits** (`examples/mflowlink/`, regression-checked in
`test/Flowchart/SimulateRun`):

| Model | Kind | Demonstrates |
|---|---|---|
| `hdl_half_adder.mflow`     | combinational | `SUM = A⊕B`, `CARRY = A·B` from XOR/AND gates |
| `hdl_full_adder.mflow`     | combinational | `SUM = A⊕B⊕Cin`, `COUT = AB + Cin(A⊕B)` (all 8 inputs swept) |
| `hdl_shift_register.mflow` | sequential    | 3× `signal_dff` — a serial bit marches one stage per clock |
| `hdl_freq_divider.mflow`   | sequential    | synchronous 3-bit counter — `signal_tff` + AND enable → clk/2, /4, /8 |

> **Cascade note:** chain registers *synchronously* (all flip-flops on the same
> `clk`, with combinational toggle-enable logic — as in `hdl_freq_divider`), not
> as a ripple (one flip-flop's output clocking the next). The register state is
> committed once per major step, so a downstream flip-flop clocked by an
> upstream flip-flop's output won't see that output's edge within the same step.

### Kalman filter worked examples (#343)

`signal_kalman` is a discrete linear Kalman filter. Its `A`/`C`/`Q`/`R` (and
optional `B`, `x0`, `P0`) are MATLAB matrix literals; the state estimate is an
`N`-vector (`N` = rows of `A`) published on the output — a scope on that wire
expands into one column per state (`se[1]`, `se[2]`, …). The predict/update
recursion runs once per major step, so size `A` for that step (e.g. `dt` in a
constant-velocity model = the solver `maxStep`).

| Model | Demonstrates |
|---|---|
| `kalman_constant.mflow` | 1-state filter (`A=C=1`) estimating a constant `5.0` from an AWGN-corrupted measurement — converges to 5 with ~100× lower variance than the raw measurement |
| `kalman_tracker.mflow`  | 2-state constant-velocity tracker (`A=[1 dt;0 1]`, `C=[1 0]`) — tracks a ramp's position to ~10 **and infers velocity ≈ 0.5 from noisy position alone** (the state coupling that exercises the 2×2 recursion); RMS error ~5× below the measurement |

A singular innovation covariance `S = C·P⁻·Cᵀ + R` is handled by skipping the
update and keeping the prediction for that step; non-conforming matrix
dimensions degrade the block to a pass-through of its (zero-initialised)
estimate rather than crashing.

### DSP frame transforms (#343)

`signal_fft` / `signal_ifft` / `signal_window` operate on a **frame** — a
vector signal of width `n`. Build a frame with a vector `signal_constant`
(`value: "[1 2 3 4]"`), a `signal_mux`, or any width-`n` source; a scope on a
frame expands into one column per element (`sf[1]`, `sf[2]`, …).

Complex spectra are carried as a real vector `[Re₀…Re₍ₙ₋₁₎, Im₀…Im₍ₙ₋₁₎]` of
width `2n`, so `fft → ifft` is a closed loop without a separate complex type.

| Model | Demonstrates |
|---|---|
| `fft_roundtrip.mflow` | `[1 2 3 4] → fft → ifft`. The spectrum is `Re=[10,-2,-2,-2]`, `Im=[0,2,0,-2]`; the inverse recovers `[1 2 3 4]` exactly (round-trip identity) |
| `window_taper.mflow`  | A flat frame `[1 1 1 1]` through a 4-point **Hann** window → `[0, 0.75, 0.75, 0]` |

The DFT is a direct O(n²) evaluation — frames in a control/DSP model are small,
so there's no FFTW dependency. `n` is required on `fft`/`ifft` so the output
width is stamped at lowering time.

### Communications link chain (#343)

`signal_psk_mod`/`demod` and `signal_qam_mod`/`demod` carry the complex symbol
as the same width-2 `[I, Q]` vector convention as the FFT spectra. A modulator
maps a scalar symbol index to its constellation point; a demodulator makes a
hard decision back to the nearest symbol. With `signal_awgn` on the channel and
`signal_error_rate` on the output, you can model a full `source → modulate →
channel → demodulate → BER` link on the canvas.

| Model | Demonstrates |
|---|---|
| `psk_qpsk.mflow` | a counter cycles symbols 0–3 → QPSK mod → demod; the round-trip recovers every symbol (BER 0). Symbol 1 maps to `exp(jπ/2) = (0, 1)` |
| `qam16.mflow`    | symbols 0–15 → 16-QAM mod → demod, BER 0 over all 16; symbol 0 sits at the grid corner `(I,Q) = (-3, -3)` |
| `qpsk_awgn_link.mflow` | full noisy link `counter → QPSK mod → vector AWGN → demod → error_rate`. At 20 dB SNR the link recovers every symbol (BER 0); at 0 dB the channel degrades it (BER ≈ 0.45). The noisy symbol-0 tap `(1,0)` shows **both** I and Q carry noise |

`signal_awgn` noises every component of a vector signal, so the complex `[I, Q]`
symbol picks up noise on both axes — the `mod → awgn → demod → error_rate` chain
measures real BER-vs-SNR on the canvas.

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
| `signal_matlab_fcn`          | ✓ | `expression` (string, required)                 | Tier-H carve-out — inline MATLAB expression evaluator. Variables `u1`..`uN` are the input ports (also `u` ≡ `u1`), `t` is simulation time, plus `pi` and `e`. Operators `+ - * / ^ .* ./ .^` and unary `-`. Functions: sin/cos/tan/asin/acos/atan/atan2, sinh/cosh/tanh, exp/log/log10/log2, sqrt/abs/sign/floor/ceil/round, min/max/mod/rem/pow/hypot/square. Expressions are parsed once at construction and rejected at lowering with a sourced diagnostic on syntax failure. The `function_body` form (`function y = f(u1,…)`) also supports several outputs (`function [a,b] = f(…)`) bound positionally to ports `out1..outM` (#344) |
| `signal_relay`      | ✓ | `onPoint: 0.5`, `offPoint: -0.5`, `onValue: 1.0`, `offValue: 0.0`, `initialState: 0.0` | Tier-E carve-out — hysteretic on/off switch. State flips at major-step boundaries only; the dead-band between `offPoint` and `onPoint` gives the latched output its persistence. Registers two zero-crossing predicates so the bisector lands transitions sub-step accurately. |

## Signal routing

| Kind | Tier-C | Params | Notes |
|---|---|---|---|
| `signal_mux`    | stub | `numInputs: 2.0`             | Vector signal type pending (Tier H+); current evaluator passes through `in1` |
| `signal_demux`  | stub | `numOutputs: 2.0`            | Same caveat |
| `signal_reshape` | ✓ | `rows: "<N>"`, `cols: "<M>"` *(or `shape: "rows,cols"`)* | §17.5 #9 — re-stamps a flat-storage signal as an `N × M` matrix. Total element count must match the upstream port (sourced lowering error otherwise). Pure metadata pass at runtime — flat buffer is copied verbatim |
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
Tier-H "ship more evaluators" axis.

The authoritative supported/reserved split is the machine-readable
catalogue printed by `matlabc -simulate --list-supported-kinds` (a JSON
array of `{kind, supported}`, no model file required); the IDE consumes
it to gray out unsupported palette blocks at edit time instead of
failing at simulate (#323). The five rows below are the current
`supported: false` set:

| Kind | Prerequisite |
|---|---|
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

## Adding a toolbox library block (#343)

The mflowLink library is mostly Simulink-core blocks plus a few toolbox
blocks (`signal_mpc_move`, `signal_pid`, `signal_state_space`,
`signal_transfer_fcn`). To expose more toolbox capability as drag-and-drop
blocks, follow this recipe — each block is a small, mechanical change. The
catalog of which blocks to add per domain lives in the OpenSpec change
`openspec/changes/mflow-toolbox-library-blocks/tasks.md`.

**Function-first rule:** add a `signal_*` block only where drag-and-drop
time-domain modeling beats wiring a generic MATLAB Function block
(`signal_matlab_fcn`). The toolbox *math* already exists at the function
level; a block is a thin adapter, not a re-implementation.

Per-block checklist:

1. **Register the kind + classification** in `lib/Flowchart/SignalFlowLowering.cpp`
   (`add("signal_xyz", {directFeedthrough, …})`): sample-time class (continuous /
   discrete-with-period / constant) and whether it's a loop-breaker.
2. **Simulator evaluator** in `lib/Flowchart/MflowLinkSim.cpp`: read params +
   input ports, compute the output(s), and **delegate to the existing toolbox
   runtime** (`runtime/toolbox/<domain>/runtime_*.cpp`, e.g. `matlab_fft_c`)
   rather than re-coding the algorithm. Scalar output → `Out_[I]`; a
   vector/frame output → the `VecOut_[I]` width path; several output ports →
   `PortOut_[I]["outK"]` (the per-output-port routing from #344/#345).
3. **Params row** in this file (camelCase keys, defaults) — in lockstep with
   the IDE's `SignalFlowParamSpec`.
4. **`SimulateRun` regression** — an `examples/mflowlink/*.mflow` fixture and
   `check`s in `test/Flowchart/SimulateRun/run_tests.sh` asserting an
   analytically-known value.
5. **Optional `-emit-c`/`-emit-cpp` lowering** when codegen of the block is
   wanted.
6. **Editor parity** — add the matching `NodeKind` in the IDE repo, and update
   the snapshot in `test/Flowchart/BlockKindParity/registered_block_kinds.txt`.
   The `flowchart-block-kind-parity` ctest fails until the snapshot matches the
   registered kinds, so a new block can't silently skip the editor.
