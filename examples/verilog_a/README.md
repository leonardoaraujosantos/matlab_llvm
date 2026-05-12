# `examples/verilog_a/`

Verilog-A behavioral models emitted from MATLAB source via the
`writeVerilogA*` runtime entries (Tiers 1, 2, 3 of
[`docs/verilog_a_plan.md`](../../docs/verilog_a_plan.md)).

Each `.m` here is **fully executable** through the existing matlabc
pipeline (`-emit-llvm`, REPL, `-dap` debugger, plotting if built with
`-DMATLAB_LLVM_WITH_PLOT=ON`) and also emits a `.va` Verilog-A module
consumable by ngspice (≥42 + OpenVAF), Xyce (≥7.5), Cadence Spectre,
Mentor Eldo, Synopsys CustomSim, or Keysight ADS.

## Tier-1 — `rfmodel.rational/writeVerilogA`

| File                          | What it shows                                            |
|---|---|
| `rf_rational_writeva.m`       | Vector-fit a 2-pole resonant target then export the model as a parameterized Verilog-A module (sum-of-real-poles + complex-pair biquads + absdelay wrap). |

## Tier-2 — `tf` / `zpk` filters

| File                       | What it shows                                                |
|---|---|
| `rc_lowpass_tf.m`          | 1st-order RC LP, `writeVerilogATF(num, den, ...)`.           |
| `biquad_butterworth.m`     | 2nd-order Butterworth LP via TF coefficients.                |
| `resonant_bpf_zpk.m`       | Complex-pair resonant filter via `writeVerilogAZPK` with poles extracted from a `rationalfit`. |

## Tier-3 — state-space

| File                       | What it shows                                                |
|---|---|
| `rc_lowpass_ss.m`          | 1st-order RC LP in state-space form.                         |
| `biquad_ss_controllable.m` | Controllable canonical 2nd-order biquad with `ddt(x[i])`.    |
| `butter3_observable.m`     | Observable-canonical 3rd-order Butterworth.                  |

## How to use

```bash
# Sanity-check numerical behavior on the LLVM lane:
matlabc -emit-llvm rc_lowpass_tf.m | clang -x ir - matlab_runtime.cpp ... -o /tmp/sanity
/tmp/sanity

# Or run the same source under the REPL / debugger:
matlabc -repl rc_lowpass_tf.m
matlabc -dap  rc_lowpass_tf.m

# The .va emits during the run as a side effect of writeVerilogA*:
ls /tmp/*.va
```

The generated `.va` files reference only standard Verilog-A primitives
(`laplace_nd`, `absdelay`, `ddt`) and `disciplines.vams`, so any
compliant simulator should accept them without modification.

The full plan, including the future Tiers 4–10 (sources, comparators,
PLL, DAC, sensors, noise, lookup tables) and Tier-11
(Verilog-AMS extensions), lives at
[`docs/verilog_a_plan.md`](../../docs/verilog_a_plan.md).
