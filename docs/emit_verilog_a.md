# Verilog-A Emission — User Reference

Companion to [`verilog_a_plan.md`](verilog_a_plan.md) (forward plan +
roadmap) and [`emit_systemverilog.md`](emit_systemverilog.md) (digital
backend).  This page is the **user-facing reference** for the Verilog-A
runtime entries shipped in matlab_llvm.

**Status (2026-05-12): Tiers 1 – 9 shipped, Tier-10 user doc + lint
wrapper landed alongside.**  See `docs/verilog_a_plan.md` §12 for the
forward plan and what remains.

## Quick start

```matlab
% Vector-fit a 2-pole rational, then emit a Verilog-A behavioral
% model that any standard analog simulator (Cadence Spectre, ngspice
% + OpenVAF, Xyce, Mentor Eldo, Synopsys CustomSim, Keysight ADS) can
% load directly.
mdl = rationalfit(freqs, h_re, h_im, 2, 25);
writeVerilogA(mdl, 'my_model.va');
```

The `.va` written is a parameterized module — drop it into a SPICE
netlist as a behavioral subcircuit and simulate.

## Design philosophy

Each MATLAB function below writes a `.va` file as a side effect.  Your
MATLAB source remains **fully executable through every other backend**:

- Run via `matlabc -emit-llvm foo.m` (sanity-check numerics).
- Step under `matlabc -dap foo.m` (set breakpoints, inspect state).
- Tune in the REPL via `matlabc -repl`.
- Plot intermediate signals if built with `-DMATLAB_LLVM_WITH_PLOT=ON`.

Verilog-A emission is **one more output lane**, not a separate sub-
language.  The same `foo.m` produces a numerical sanity check on the
LLVM lane *and* a `.va` for analog co-simulation.

This mirrors the SystemVerilog backend's two-backend split documented in
[`docs/emit_systemverilog.md`](emit_systemverilog.md):

    MATLAB subset
    ├── Digital subset  →  SystemVerilog     (-emit-systemverilog)
    └── Analog subset   →  Verilog-A         (writeVerilogA*)

The Verilog-A path is implemented as **runtime functions** rather than
a separate emit-time CLI walker.  This matches MathWorks'
`rfmodel.rational/writeVerilogA` API and the existing
`touchstoneWrite` pattern in `runtime/runtime_rf.cpp`.

## Runtime entries

### Tier-1 — `rfmodel.rational` export

```matlab
writeVerilogA(mdl, filename)
```

Emits a sum-of-poles module from a rationalfit / RFRational instance.
Real-pole sections fold to `laplace_nd(V(in), {r}, {-p, 1.0})`;
complex-conjugate pole pairs `(a ± jb)` with residues `(c ± jd)` fold
to a real-coefficient biquad section.  The bulk delay (if set) wraps
the contribution sum via `absdelay(sum, Delay)`.

Accepts either:
- The rationalfit return struct (field names `Poles`, `Residues`, `D`, `Delay`).
- An `RFRational` classdef instance (field names `A`, `C`, `D`, `Delay`).
  Type annotations `A complex; C complex;` on the classdef route the
  property storage through `matlab_obj_set_mat` correctly.

### Tier-2 — Continuous tf / zpk filters

```matlab
writeVerilogATF(num, den, filename)
writeVerilogAZPK(zeros, poles, k, filename)
```

`writeVerilogATF`: `num` / `den` are real columns in **descending**
power of s (MATLAB `tf` convention).  Scalar-fold dispatch shims auto-
promote `[1]` → 1×1 matrix.  Emits a single `laplace_nd` contribution.

`writeVerilogAZPK`: `zeros` / `poles` may be real or complex columns;
complex-conjugate pairs auto-fold into real-coefficient quadratic
factors.  `k` is a scalar gain.

For complex pole columns, build via:

```matlab
poles = complex(p_re, p_im);                 % preferred
% — or —
poles = rfPoles(rationalfit(...));           % from a fit
```

The `1i * real_col` arithmetic path is **not** supported today (an
upstream `matlab_emul_sm` lowering drops the imag part).  Use
`complex(re, im)` instead.

### Tier-3 — Continuous SISO state-space

```matlab
writeVerilogASS(A, B, C, D, filename)
```

`A`: N×N, `B`: N×1, `C`: 1×N, `D`: scalar.  Emits one `ddt(x[i]) <+
...` contribution per state variable + the `V(out) <+ Σ Cⱼ x[j] +
D V(in)` output equation.  Zero-coefficient terms are elided.  Scalar-
fold shim handles N=1 (`A = [-1e6]`).

### Tier-4 — Analog sources, comparators, Schmitt triggers

```matlab
writeVerilogASource(kind, amp, freq_or_tau, filename)
writeVerilogAComparator(vth, vh, vl, td, tr, filename)
writeVerilogASchmitt(vhigh, vlow, vh, vl, filename)
```

`writeVerilogASource` — `kind ∈ {0=sin, 1=cos, 2=square, 3=exp-decay}`.
Drives `V(out)` from `$abstime`.  For exp-decay, the second scalar is
the time constant τ (`V(out) = amp * exp(-$abstime/τ)`).

`writeVerilogAComparator` — single-threshold comparator with
`@(cross(V(in)-vth, ±1))` event blocks toggling an `integer state`;
`transition(state ? vh : vl, td, tr)` for output settling.

`writeVerilogASchmitt` — dual-threshold hysteresis via two
`@(cross())` events at `vhigh` (rising edge) and `vlow` (falling edge).

### Tier-5 — VCO

```matlab
writeVerilogAVCO(freq_center, gain, amp, filename)
```

Phase-accumulator VCO via `idtmod`:

```verilog-a
phase    = idtmod(2.0 * `M_PI * (freq_center + gain * V(in)), 0.0, 2.0 * `M_PI);
V(out)   <+ amp * sin(phase);
```

`idtmod` is the canonical Verilog-A primitive for clean 2π phase
wrap.

### Tier-6 — Behavioral DAC

```matlab
writeVerilogADAC(N, vref, td, tr, filename)
```

Pure-Verilog-A DAC.  `V(code)` is read as an analog-coded voltage
interpreted as a digital code in `[0, 2^N - 1]`; output is
`vref * V(code) / (2^N - 1)`, wrapped in `transition(..., td, tr)`.

For a true digital bit-bus input (Verilog-AMS `connectmodule` +
`reg [N-1:0]`), see Tier-11 in `verilog_a_plan.md` — deferred.

### Tier-7 — Compact components + sensor models

```matlab
writeVerilogADiode(Is, Vt, filename)
writeVerilogAOpAmp(gain, vsat, filename)
writeVerilogARTD(R0, alpha, T0, filename)
writeVerilogAThermistor(R0, B, T0, filename)
```

`writeVerilogADiode` — Shockley equation
`I(p,n) = Is * (exp(V(p,n)/Vt) - 1)`.

`writeVerilogAOpAmp` — `tanh`-saturated op-amp
`V(out) = vsat * tanh(gain * V(vp,vn) / vsat)`.  Smooth saturation has
no slope discontinuity at the rail, which helps SPICE convergence.

`writeVerilogARTD` — Pt-style RTD using first-class `$temperature`:

```verilog-a
I(p, n) <+ V(p, n) / (R0 * (1.0 + alpha * ($temperature - T0)));
```

`writeVerilogAThermistor` — NTC thermistor with β-equation
`R(T) = R0 * exp(B * (1/$temperature - 1/T0))`.

### Tier-8 — White + flicker noise

```matlab
writeVerilogANoise(kind, pwr, exponent, filename)
```

`kind = 0`: `V(out) <+ white_noise(pwr, "thermal");`
`kind = 1`: `V(out) <+ flicker_noise(pwr, exponent, "1_over_f");`

These are **PSD-style** noise primitives consumed by the simulator's
`.noise` analysis, **not** per-step `randn()` samples.  See the
Verilog-A LRM §4.5 for the noise model semantics.

### Tier-9 — Lookup tables

```matlab
writeVerilogATable(x_col, y_col, va_filename)
```

Writes a sidecar `.tbl` file alongside the `.va` (one row per data
point: two whitespace-separated columns) and a `.va` module that
references it via `$table_model(V(in), "name.tbl", "1L,L")` — 1-D
linear interpolation with linear extrapolation outside the data
range.

`$table_model` is part of the Verilog-A 2.4 LRM and is supported by
Cadence Spectre, Synopsys CustomSim, Mentor Eldo, and ngspice (v42+)
via OpenVAF.

## Lint workflow (Tier-10)

```bash
scripts/va_lint.sh examples/verilog_a/*.va
```

The script wraps OpenVAF (preferred) or ADMS (fallback).  Neither
installed?  The script prints a hint and exits 0 (skip), so opt-in
CI lanes can run it unconditionally.

Install OpenVAF: <https://openvaf.semimod.de/>.

## Cosim workflow (manual)

For round-trip validation against the in-tree `freqresp` / `timeresp`
references, simulate the emitted `.va` in ngspice or Xyce and compare
the frequency / step response:

```bash
# Spectre
spectre netlist.scs -raw out.raw +log spectre.log

# ngspice + OpenVAF (requires ngspice >= 42 built with OpenVAF support)
ngspice -b netlist.cir -o ngspice.log

# Xyce
Xyce netlist.cir
```

A CTest integration for cosim is on the Tier-10 plan but not shipped
yet.

## Examples

See `examples/verilog_a/` for one runnable example per Tier:

| Example                           | Tier | Demonstrates                                            |
|---|:-:|---|
| `rf_rational_writeva.m`           | 1    | rationalfit → biquad sum-of-poles export                |
| `rc_lowpass_tf.m`                 | 2    | 1st-order RC LP via `writeVerilogATF`                   |
| `biquad_butterworth.m`            | 2    | Butterworth-2 via TF                                    |
| `resonant_bpf_zpk.m`              | 2    | Complex-pair zpk via `rfPoles`                          |
| `rc_lowpass_ss.m`                 | 3    | 1st-order RC LP in state-space                          |
| `biquad_ss_controllable.m`        | 3    | Controllable canonical biquad                           |
| `butter3_observable.m`            | 3    | 3rd-order Butterworth, observable canonical             |
| `sine_source.m`                   | 4    | Sinusoidal stimulus driven by `$abstime`                |
| `comparator.m`                    | 4    | `cross()`-event comparator                              |
| `schmitt.m`                       | 4    | Schmitt trigger with hysteresis                         |
| `vco.m`                           | 5    | VCO via `idtmod` phase accumulation                     |
| `dac_8bit.m`                      | 6    | Behavioral 8-bit DAC                                    |
| `diode.m`                         | 7    | Shockley-equation diode                                 |
| `opamp_saturated.m`               | 7    | `tanh`-saturated op-amp                                 |
| `rtd_pt100.m`                     | 7    | Pt-100 RTD against `$temperature`                       |
| `thermistor_ntc.m`                | 7    | NTC thermistor (β-equation)                             |
| `noise_thermal.m`                 | 8    | Thermal-noise source (`white_noise`)                    |
| `noise_flicker.m`                 | 8    | 1/f flicker noise                                       |
| `iv_curve_table.m`                | 9    | `$table_model` 1-D lookup + `.tbl` sidecar              |

Each example is fully executable through matlabc -emit-llvm and
writes its `.va` (and a `.tbl` for Tier-9) into the working directory.

## Forward plan

See [`verilog_a_plan.md`](verilog_a_plan.md) for:
- Tier-10 polish remaining: OpenVAF CTest lane, optional ngspice/Xyce
  cosim integration.
- Tier-11 (Verilog-AMS): ADC with digital bit-bus output,
  `connectmodule` / `connectrules` discipline resolution.  Deferred
  until demand surfaces.
