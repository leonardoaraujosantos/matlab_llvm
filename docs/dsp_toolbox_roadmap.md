# DSP System Toolbox + DSP HDL Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot + Fixed-Point + emit-SV/cocotb) needs to ship in order to
faithfully **compile and execute**, **debug/REPL**, and **demo**
DSP-System-Toolbox **and DSP-HDL-Toolbox** programs. The two are covered
in one roadmap because they are the same algorithms at two abstraction
levels — `dsp.*` is the frame-based streaming surface (Tiers 1–6),
`dsphdl.*` is its cycle-accurate, valid/ready-handshaked, HDL-generating
hardware counterpart (Tiers 7–8) — and the project's emit-SV + cocotb
SIL lane is *exactly* what the `dsphdl.*` surface needs.

Sources: *DSP System Toolbox User's Guide* (R2026a, 16 chapters: DSP
Tutorials · Input/Output/Display · Data and Signal Management · Featured
Examples · Filter Analysis, Design, and Implementation · Adaptive
Filters · Multirate and Multistage Filters · Filter Analyzer and Filter
Designer · Dataflow · Simulink Block Examples (Multirate/Multistage ·
Scopes/Logging · Signal I/O · Signal Generation/Operations · DSP
System · Deep Learning) · Synthesize/Channelize Audio); and *DSP HDL
Toolbox User's Guide* (R2026a, 5 chapters: Featured Examples · HDL
Optimized System Design (FIR architectures · high-throughput frame/vector
input · **hardware control signals: valid / backpressure-ready / reset**
· IP Designer app) · Block Reference Page Examples (systolic FIR · CIC ·
NCO · CORDIC `atan2` · up/downsamplers · automatic delay matching) · HDL
Code Generation and Deployment · Radar Application Examples).

This is the **natural streaming-and-fixed-point extension of the shipped
Signal Processing Toolbox**, and the toolbox where the project's three
strongest differentiators converge: the shipped **filter/FFT/multirate
substrate** (`filter` / `sosfilt` / `butter` / `cheby1` / `fir1` /
`freqz` / `upfirdn` / `resample` / `decimate` / `fft`), the shipped
**Fixed-Point Designer** (`fi` / `numerictype` / `fimath`), and the
shipped **HDL lane** (`-emit-systemverilog` + the mflowLink Embedded
Coder + cocotb SIL). A fixed-point streaming `dsp.FIRFilter` that lowers
to *synthesizable SystemVerilog and runs a cocotb SIL test* is a payoff
no other shipped toolbox can claim — and the **DSP HDL Toolbox
(`dsphdl.*`, Tiers 7–8) is where that payoff becomes the entire point**:
every `dsphdl.*` block is fixed-point-native, exposes a cycle-accurate
valid/ready streaming interface, and exists *to generate HDL* — which is
this project's home turf, not a stretch. Where MATLAB needs HDL Coder to
turn a `dsphdl.*` object into RTL, **we are the HDL generator**.

**The defining architectural fact**: unlike Curve Fitting / Wavelet
(function-form toolboxes), the DSP System Toolbox's *primary* surface is
**System Objects** — `dsp.FIRFilter`, `dsp.LMSFilter`,
`dsp.SpectrumEstimator`, … — handle-shaped classdefs with the
`setup`/`step`/`reset`/`release` lifecycle, `Nontunable` properties
locked after the first call, and `DiscreteState` that persists internal
state across frame-based calls (`y = obj(frame)`). This is **the same
System-Object lowering work that today gates Comm Tier-3+ / RF-Tier-1+ /
Antenna classdef tiers** (the documented blocker: a tensor-typed RHS
routed through `_set_f64` after monomorphization fails the verifier — CST
roadmap §12, Comm roadmap §15). **Tier-1 of this roadmap is that fix** —
so shipping DSP's filter objects simultaneously *unblocks every SO-gated
tier across Comm / RF / Antenna*. That cross-toolbox leverage is the
strategic case for doing DSP next.

**No external dependency** (no Intel IPP, no CMSIS-DSP) — every filter
runs over the shipped `filter`/`sosfilt`, every transform over the
shipped `fft`, the adaptive cores reuse the Comm-shipped
`matlab_comm_lms`/`_rls`/`_cma`/`_dfe`, and fixed-point filters reuse the
shipped `fi` lane.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/dsp/streaming_fir.m`](../examples/dsp/streaming_fir.m):
*the canonical frame-based streaming demo — design a lowpass FIR, build a
`dsp.FIRFilter` System Object, run a frame loop (`for k=1:N; y =
firFilt(noisySineFrame); end`) where the object persists its tapped-delay
state across calls, and measure the noise reduction*. This exercises the
System-Object `setup`→`step`(×N)→`reset` lifecycle end-to-end; achieving
it closes **DSP-Tier-1** (and proves the SO model). The differentiated
**DSP-Tier-6** tracer-bullet is
[`examples/dsp/fixedpoint_fir_hdl.m`](../examples/dsp/fixedpoint_fir_hdl.m):
*the same FIR with `fi`-typed coefficients and state, lowered to
synthesizable SystemVerilog and validated by a cocotb SIL test* — the
Signal × Fixed-Point × HDL convergence. The **DSP HDL headline**
(closing **DSP-Tier-7**) is
[`examples/dsp/dsphdl_fir_stream.m`](../examples/dsp/dsphdl_fir_stream.m):
*a `dsphdl.FIRFilter` (fully-parallel systolic, fixed-point) driven by a
`[dataOut, validOut] = obj(dataIn, validIn)` valid-handshaked stream,
lowered to synthesizable SystemVerilog with the valid/ready control
interface and validated cycle-by-cycle by a cocotb SIL test against the
MATLAB streaming reference* — the cycle-accurate hardware payoff.

Companion docs: [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md)
(the filter / FFT / multirate substrate this rides), [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md)
(the System-Object lowering fix is shared — §15 there documents the
blocker; the adaptive `lms`/`rls`/`cma`/`dfe` cores are reused),
[`fixed_point_toolbox_roadmap.md`](fixed_point_toolbox_roadmap.md)
(fixed-point filter coefficients + state), [`emit_fixed_point.md`](emit_fixed_point.md)
+ [`embedded_coder_roadmap.md`](embedded_coder_roadmap.md)
(the fixed-point filter → synthesizable SV + cocotb SIL bridge),
[`wavelet_toolbox_roadmap.md`](wavelet_toolbox_roadmap.md) (the dyadic
analysis/synthesis filter banks overlap), [`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the **System-Object runtime model + core filter objects** — the
  `setup`/`step`/`reset`/`release` lifecycle over the shipped
  classdef + persistent-state infra, plus `dsp.FIRFilter` /
  `dsp.IIRFilter` / `dsp.BiquadFilter` / `dsp.SOSFilter` /
  `dsp.FilterCascade` / `dsp.Delay`. This is the keystone that unblocks
  the SO-gated Comm/RF tiers too. **Tier-2** is filter design
  (function-form, no SO needed): `designfilt` / `fdesign.*` / `firpm` /
  `firls` / `firhalfband` / `kaiserord` / `iirnotch` / `iirpeak` /
  digital frequency transformations. **Tier-3** is adaptive filters
  (`dsp.LMSFilter` / `dsp.RLSFilter` / `dsp.NLMS` /
  `dsp.AffineProjectionFilter` / `dsp.FrequencyDomainAdaptiveFilter`,
  reusing the Comm cores). **Tier-4** is multirate + multistage + filter
  banks (`dsp.FIRDecimator` / `dsp.FIRInterpolator` /
  `dsp.SampleRateConverter` / `dsp.CICDecimator` / `dsp.Channelizer` /
  `dsp.DyadicAnalysisFilterBank` / `dsp.DigitalDownConverter`).
  **Tier-5** is transforms + sources + streaming statistics +
  measurement (`dsp.FFT` / `dsp.STFT` / `dsp.ZoomFFT`; `dsp.SineWave` /
  `dsp.NCO` / `dsp.ColoredNoise`; `dsp.AsyncBuffer` / `buffer`;
  `dsp.MovingAverage` / `dsp.Mean` / `dsp.Histogram` / `dsp.PeakFinder`;
  `dsp.SpectrumEstimator` / `dsp.SpectrumAnalyzer` /
  `dsp.TimeScope`). **Tier-6** is fixed-point filters + the HDL bridge +
  carve-down polish (`fi`-typed filters, fixed-point FIR →
  synthesizable SV + cocotb SIL, `dsp.LUFactor` / `dsp.LevinsonSolver`).
  **Tier-7 (DSP HDL Toolbox)** is the cycle-accurate streaming-hardware
  foundation — the **valid / backpressure-ready / reset control-signal
  interface** + the `dsphdl.*` System-Object model + the FIR hardware
  architectures (fully-parallel systolic / partly-serial / fully-serial
  systolic / transposed / programmable / frame-vector high-throughput),
  all flowing to synthesizable SV + cocotb SIL. **Tier-8 (DSP HDL
  Toolbox)** is the rest of the `dsphdl.*` block family — multirate
  (`dsphdl.FIRDecimator` / `Interpolator` / `CICDecimator` /
  `CICInterpolator` / `Downsample` / `Upsample` / DDC/DUC), transforms
  (`dsphdl.FFT` / `IFFT` / `Channelizer` / `ChannelSynthesizer` /
  `NCO`), CORDIC math (`dsphdl.Complex2MagnitudeAngle` / `atan2` /
  `Sqrt` / `SineCosine`), and data management (`dsphdl.SampleAligner`
  FIFO + `dsphdl.DelayMatcher` automatic latency matching).
- **Effort** is in the existing Phase 5.6.x cadence (one focused session
  ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~2.5 wk
  (the SO model is the big cost) · T2 ~1.5 wk · T3 ~1 wk · T4 ~2 wk · T5
  ~2.5 wk · T6 ~2.5 wk · T7 ~3 wk (the valid/ready streaming-hardware
  model) · T8 ~3 wk (~18 wk full)**. This is by far the largest
  single-toolbox roadmap — but **T1 alone is a force multiplier** (it
  unblocks Comm/RF/Antenna SO tiers), **T1 + T2 (~4 wk) close the
  streaming-filter + filter-design 70% workflow**, and **Tiers 6–8 are
  the project's strongest differentiator** (no other shipped toolbox
  generates synthesizable, cocotb-verified DSP hardware). The DSP HDL
  tiers (7–8) depend on Tier-1 (the SO model) and Tier-6 (the
  fixed-point → SV bridge) but are otherwise independent of Tiers 2–5.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Everything below is 🔵 not started** — but the substrate is deep
  (filters, FFT, multirate, `fi`, emit-SV, classdef+persistent all
  shipped). The single genuinely new infrastructure piece is the
  **System-Object semantics**; everything after rides shipped numerics.
- **System Objects are stateful classdefs with a locked lifecycle**:
  `obj = dsp.FIRFilter('Numerator', b)` (constructor sets `Nontunable`
  props) → first `y = obj(x)` triggers implicit `setup` (allocate
  `DiscreteState`) and locks Nontunable props → subsequent `obj(x)` calls
  `step` (process a frame, mutate internal state) → `reset(obj)` zeroes
  state → `release(obj)` unlocks. The project already ships **classdef +
  handle semantics + `persistent` state + the mflow stateful-block
  model**; the new work is the lifecycle dispatch + the `obj(x)`
  call-syntax-as-`step` + the `DiscreteState` storage on the object. This
  is the **exact** blocker recorded in the Comm/CST roadmaps; resolving it
  here is the cross-toolbox payoff.
- **Function-form first where it exists**: filter *design* (Tier-2) is
  pure functions (`firpm`/`designfilt`) — no SO needed, ships
  independently. The adaptive *cores* already ship as Comm functions
  (`matlab_comm_lms`/…); Tier-3 only wraps them in the SO lifecycle. This
  keeps Tier-2 shippable even before Tier-1 lands.
- **No external dependencies**: matching the project precedent — filters
  over the shipped `filter`/`sosfilt`; CIC / polyphase / Farrow
  hand-coded; transforms over `fft`; spectral estimators over the shipped
  `pwelch`/`periodogram`/`cpsd`; fixed-point over `fi`.

---

## 1. Reusable infrastructure (Tier-0 baseline — no DSP code yet)

| Group | Surface (already shipped) | Location | How DSP uses it |
|---|---|---|---|
| Filter implementation | `filter`, `filtfilt`, `sosfilt`, `conv`, `conv2` | `lib/Sema/Resolver.cpp` → `matlab_filter` (`runtime/matlab_runtime.cpp`) | The compute kernel of every `dsp.FIRFilter` / `IIRFilter` / `BiquadFilter` / `SOSFilter` (Tier-1). |
| Filter design (Signal) | `butter`, `cheby1`, `cheby2`, `besself`, `fir1`, `buttord`/`cheb1ord`, `bilinear`, `tf2sos`/`zp2tf`/`sos2tf`, `freqz`/`impz`/`grpdelay` | `lib/Sema/Resolver.cpp` (Signal Tier-1) | The classic-design baseline; Tier-2 adds the equiripple/least-squares/specialised designers on top. |
| Multirate | `upfirdn`, `decimate`, `interp`, `resample`, `upsample`, `downsample` | `lib/Sema/Resolver.cpp` (Signal Tier-3) | `dsp.FIRDecimator` / `Interpolator` / `SampleRateConverter` (Tier-4); polyphase decomposition. |
| FFT / transforms | `fft`, `ifft`, `fft2`, `dct`/`idct`, `fwht`, `hilbert`, `goertzel`, `spectrogram` | `runtime/matlab_runtime.cpp` (Signal Tier-2) | `dsp.FFT` / `IFFT` / `STFT` / `ISTFT` / `DCT` / `ZoomFFT` (Tier-5). |
| Spectral / parametric | `pwelch`, `periodogram`, `cpsd`, `tfestimate`, `mscohere`, `levinson`, `lpc`, `aryule`, `arburg` | `runtime/matlab_runtime.cpp` (Signal Tier-2) | `dsp.SpectrumEstimator` / `CrossSpectrumEstimator` / `TransferFunctionEstimator` (Tier-5); LPC demo. |
| Adaptive cores | `matlab_comm_lms`, `matlab_comm_rls`, `matlab_comm_cma`, `matlab_comm_dfe` | `runtime/toolbox/comm/runtime_comm.cpp` (Comm Tier-4) | The compute cores of `dsp.LMSFilter` / `RLSFilter` / friends — Tier-3 wraps them in the SO lifecycle. |
| Measurements / stats | `mean`, `std`, `var`, `median`, `max`, `min`, `rms`, `findpeaks`, `xcorr`, `sort`, `cumsum` | `runtime/matlab_runtime.cpp` (core + Signal) | Streaming statistics objects `dsp.Mean` / `MovingAverage` / `MovingRMS` / `PeakFinder` / `Autocorrelator` (Tier-5). |
| Fixed-Point Designer | `fi`, `numerictype`, `fimath`, fi-array indexing + `sum`/`mean`, `persistent` fi storage, all 5 rounding modes | `lib/MLIR/Passes/LowerFixedPoint.cpp` | Fixed-point filter coefficients + `DiscreteState` (Tier-6) — the differentiated track. |
| HDL lane | `-emit-systemverilog`, mflowLink Embedded Coder (per-subsystem + whole-diagram), cocotb SIL (combinational + **sequential DUTs with pre-edge sampling**), persistent-fi → SV shift register / runtime-indexed regfile, hierarchical multi-module emit | `lib/Emit/`, `docs/embedded_coder_roadmap.md` | Fixed-point `dsp.FIRFilter` → synthesizable SV tapped-delay line + cocotb SIL (Tier-6); **the entire `dsphdl.*` surface (Tiers 7–8) — every block emits a clocked SV module driven by a valid/ready handshake and is verified cycle-by-cycle by a cocotb SIL test.** This is the single biggest reason DSP HDL is in-scope rather than carved out. |
| Classdef + state | `classdef`, handle semantics, `properties`/`methods`, `persistent`, mflow stateful blocks, `matlab_obj_new`/`_set_*`/`_get_mat`, class-pinned dispatch, REPL persist, DAP render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The substrate for the System-Object lifecycle (Tier-1) — the new work is the lifecycle dispatch on top. |
| Function-handle ABI | `void *fn_p`, `LowerAnonCalls` retyping | `runtime/toolbox/optim/runtime_optim.cpp` | Custom filter / source function handles; `dsp.SignalSource` callbacks. |
| Plotting | Cairo `plot` / `stem` / `imagesc` / `surf` | `runtime/plot/` | Headless `dsp.TimeScope` / `dsp.ArrayPlot` / `dsp.SpectrumAnalyzer` → PNG/SVG artifacts (Tier-5). |

**Net assessment**: the *numeric base* (filters, FFT, multirate, spectral
estimators, adaptive cores, fixed-point, the HDL lane, classdef +
persistent state) is **already shipped** — more than any other unstarted
toolbox. The genuinely new code is (a) the **System-Object lifecycle**
(`setup`/`step`/`reset`/`release` + `Nontunable` lock + `DiscreteState` +
`obj(x)`-as-`step` — the keystone that also unblocks Comm/RF SO tiers),
(b) the **`dsp.*` filter-object wrappers** over the shipped kernels, (c)
the **advanced filter designers** (`firpm`/`firls`/`designfilt`), (d) the
**polyphase/CIC/Farrow multirate + filter banks**, (e) the **streaming
statistics + spectral-estimator objects**, and (f) the **fixed-point
filter → SV bridge**. Each rides the shipped base — the heavy numerics
are done; the new surface is mostly *wiring numerics into the
System-Object lifecycle*.

---

## 2. Tier-1 — System-Object runtime model + core filter objects 🔵 (KEYSTONE)

Goal: the `dsp.*` System-Object lifecycle + the filter objects every DSP
program starts with. **Resolving the SO lowering blocker here unblocks
the SO-gated Comm/RF/Antenna tiers** — the strategic core of the whole
roadmap.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | **SO lifecycle** | `setup`(implicit on first call) / `step` (= `obj(x)` call-syntax) / `reset` / `release` / `isLocked`; `Nontunable` props locked after setup; `DiscreteState` allocated at setup, mutated per `step`, zeroed on `reset`. Resolves the tensor-typed-RHS-through-`_set_f64`-after-monomorphization verifier blocker (CST §12 / Comm §15). | classdef + persistent state |
| 1.2 | `dsp.FIRFilter` | Direct-form FIR over a persisted tapped-delay line; `Numerator` Nontunable; frame-based `y = firFilt(x)` carries state across calls. | `filter`/`conv` |
| 1.3 | `dsp.IIRFilter` | Direct-form-II transposed IIR; `Numerator`/`Denominator`; persisted state vector. | `filter` |
| 1.4 | `dsp.BiquadFilter` / `dsp.SOSFilter` | Cascaded second-order sections; `SOSMatrix` + `ScaleValues`; per-section state. | `sosfilt` |
| 1.5 | `dsp.FilterCascade` / `dsp.AllpassFilter` | Compose System Objects in series; allpass (lattice/wave-digital) forms. | 1.2–1.4 |
| 1.6 | `dsp.Delay` / `dsp.VariableFractionalDelay` | Integer delay line + Farrow-structure fractional delay; persisted buffer. | persistent state |
| 1.7 | `clone` / `info` / `cost` / `getDiscreteState` | SO introspection: deep-copy an object, report state/latency, read internal state. | classdef |
| 1.8 | display + DAP | `disp(obj)` formats the property block; the SO + its `DiscreteState` render in the DAP variable inspector + persist across REPL inputs. | `runtime_debug.cpp` |

**Headline-within-tier**: the streaming FIR —
`firFilt = dsp.FIRFilter('Numerator', b); for k=…; y = firFilt(frame); end`
noise reduction across a frame loop with persisted state.

**Compile/Execute wiring**: the SO lifecycle is the new front-end/lowering
work — `obj(x)` on a `dsp.*` instance lowers to a `step` method call
(class-pinned dispatch in `Lowering.cpp::CallOrIndex`); `DiscreteState`
storage is a per-object matrix property; `setup` lock is a runtime flag.
New `runtime/toolbox/dsp/runtime_dsp.cpp` + `runtime/toolbox/dsp/dsp_classdefs.m`;
prelude-trigger the `dsp.*` classdefs. **This tier's verifier fix is
shared with Comm/RF** — coordinate with `comm_toolbox_roadmap.md` §15.

---

## 3. Tier-2 — Filter design (function-form) 🔵

Goal: the modern designer surface — equiripple, least-squares,
specialised, and the unified `designfilt` API. **No System Object
needed** — ships independently of Tier-1.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `firpm` / `firpmord` | Parks-McClellan equiripple FIR (Remez exchange); order estimate. | — |
| 2.2 | `firls` / `fircls` | Least-squares + constrained-least-squares FIR. | `mldivide`/`qr` |
| 2.3 | `firhalfband` / `firnyquist` / `firgr` / `firceqrip` | Halfband / L-th-band Nyquist / generalised Remez / constrained-equiripple. | 2.1 |
| 2.4 | `kaiserord` / `kaiserwin` / `firgauss` | Kaiser-window order + design; Gaussian FIR. | windows (Signal) |
| 2.5 | `iirnotch` / `iirpeak` / `iircomb` | Second-order notch/peak/comb IIR designers. | `tf2sos` |
| 2.6 | `designfilt` | Unified spec-driven designer: `designfilt('lowpassfir','FilterOrder',…,'CutoffFrequency',…)` → a digital-filter object usable by `filter`/`freqz`. | 2.1–2.5, Signal designers |
| 2.7 | `fdesign.*` + `design` | `fdesign.lowpass`/`highpass`/`bandpass`/`bandstop`/`decimator`/`interpolator` spec objects + `design(spec, method)`. | 2.6 |
| 2.8 | frequency transformations | `firlp2lp`/`firlp2hp`/`iirlp2bp`/… digital frequency transformations of a prototype. | `freqz` |
| 2.9 | `cl` / minimax / arbitrary-magnitude | Arbitrary magnitude+phase FIR; least-Pth-norm optimal FIR/IIR. | `lsqnonlin` (Optim) |

**Headline-within-tier**: narrow-transition equiripple —
`b = firpm(order, f, a)` then `freqz(b)` showing the equiripple passband.

**Compile/Execute wiring**: all matrix-in/matrix-out builtins
(`Resolver.cpp` + `LowerTensorOps.cpp` `pde_table`, string spec/method
arg → `matlab_string*`); `designfilt`/`fdesign` return a lightweight
filter-spec descriptor (classdef carrier, like `fitoptions`).

---

## 4. Tier-3 — Adaptive filters 🔵

Goal: the adaptive-filter System Objects — wrapping the **Comm-shipped**
adaptive cores in the Tier-1 SO lifecycle.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `dsp.LMSFilter` | LMS / Normalized-LMS / Sign-data / Sign-error / Sign-sign; per-step weight update, persisted weights. | `matlab_comm_lms` |
| 3.2 | `dsp.RLSFilter` | Recursive least squares (forgetting factor); persisted inverse-correlation matrix. | `matlab_comm_rls` |
| 3.3 | `dsp.AffineProjectionFilter` | Affine-projection adaptive filter. | LMS core + projection |
| 3.4 | `dsp.BlockLMSFilter` / `dsp.FrequencyDomainAdaptiveFilter` | Block-LMS + FFT-domain (overlap-save) adaptive filter. | `fft`, LMS core |
| 3.5 | `maxstep` / weight readout | Step-size bound; `obj.Weights` / `getWeights` introspection. | classdef |

**Headline-within-tier**: acoustic noise cancellation —
`dsp.LMSFilter` adapts to cancel correlated noise from a signal+noise
mixture; the UG "Acoustic Noise Cancellation (LMS)" demo.

**Compile/Execute wiring**: each is a thin SO wrapper (Tier-1 lifecycle)
around the existing `matlab_comm_*` cores; the adaptive weight vector is
the `DiscreteState`.

---

## 5. Tier-4 — Multirate + multistage + filter banks 🔵

Goal: the sample-rate-conversion + channelizer surface — over the shipped
`upfirdn`/`resample`.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | `dsp.FIRDecimator` / `dsp.FIRInterpolator` | Polyphase decimation / interpolation by integer factor; persisted polyphase state. | `upfirdn` |
| 4.2 | `dsp.FIRRateConverter` / `dsp.SampleRateConverter` | Rational L/M rate conversion; multistage auto-decomposition for large ratios. | `resample`, 4.1 |
| 4.3 | `dsp.CICDecimator` / `dsp.CICInterpolator` / `dsp.CICCompensationDecimator` | Cascaded-integrator-comb (multiplier-free) + droop-compensation FIR. | hand-coded |
| 4.4 | `dsp.FarrowRateConverter` / `dsp.VariableFractionalDelay` | Farrow-structure arbitrary-factor / fractional resampling. | polynomial interp |
| 4.5 | `dsp.Channelizer` / `dsp.ChannelSynthesizer` | Polyphase-FFT analysis/synthesis filter bank (M-channel). | `fft`, polyphase |
| 4.6 | `dsp.DyadicAnalysisFilterBank` / `dsp.DyadicSynthesisFilterBank` | Tree-structured two-channel halfband bank — **overlaps the Wavelet roadmap** (`swt`/`modwt` filter banks). | `firhalfband` (2.3) |
| 4.7 | `dsp.DigitalDownConverter` / `dsp.DigitalUpConverter` | NCO mix + multistage decimation/interpolation (the GSM/FRS DDC/DUC demos). | 4.2, `dsp.NCO` (5.x) |

**Headline-within-tier**: multistage sample-rate conversion —
`dsp.SampleRateConverter` 192 kHz → 44.1 kHz with the auto-chosen
multistage filter, spectrum before/after.

**Compile/Execute wiring**: polyphase/CIC/Farrow cores in
`runtime_dsp.cpp`; the converter objects use the Tier-1 lifecycle with the
polyphase-commutator state as `DiscreteState`.

---

## 6. Tier-5 — Transforms + sources + streaming statistics + measurement 🔵

Goal: the rest of the streaming surface — transform objects, signal
sources, buffering, online statistics, spectral estimators, and headless
scopes.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | transform objects | `dsp.FFT` / `dsp.IFFT` / `dsp.STFT` / `dsp.ISTFT` / `dsp.DCT` / `dsp.IDCT` / `dsp.ZoomFFT`. | `fft`/`dct` |
| 5.2 | sources | `dsp.SineWave` (persisted phase) / `dsp.NCO` / `dsp.Chirp` / `dsp.ColoredNoise` / `dsp.SignalSource` / `dsp.SignalSink`. | trig + PRNG |
| 5.3 | buffering / management | `buffer` (function) / `dsp.AsyncBuffer` / `dsp.DelayLine` / `dsp.Queue` / `dsp.Counter`; overlap buffering. | persistent state |
| 5.4 | streaming statistics | `dsp.Mean` / `Variance` / `RMS` / `StandardDeviation` / `Maximum` / `Minimum` / `Median`; running (`RunningMean`-style) + moving (`dsp.MovingAverage` / `MovingRMS` / `MovingStandardDeviation` / `MovingMaximum`). | reductions, `cumsum` |
| 5.5 | detectors / correlators | `dsp.PeakFinder` / `dsp.ZeroCrossingDetector` / `dsp.Autocorrelator` / `dsp.Crosscorrelator` / `dsp.Histogram` / `dsp.DCBlocker`. | `findpeaks`, `xcorr` |
| 5.6 | spectral estimators | `dsp.SpectrumEstimator` / `dsp.CrossSpectrumEstimator` / `dsp.TransferFunctionEstimator` (Welch over frames). | `pwelch`/`cpsd`/`tfestimate` |
| 5.7 | scopes (headless) | `dsp.TimeScope` / `dsp.ArrayPlot` / `dsp.SpectrumAnalyzer` → Cairo PNG/SVG artifacts (no live GUI); `getMeasurementsData` programmatic readout. | `runtime/plot/` |

**Headline-within-tier**: streaming statistics —
a frame loop feeding `dsp.MovingAverage` + `dsp.MovingRMS` +
`dsp.PeakFinder`, plotting the running envelope.

**Compile/Execute wiring**: all Tier-1-lifecycle objects (state = running
accumulators / circular buffers); `buffer` is a plain function;
`dsp.SpectrumEstimator` reuses the shipped `pwelch`; scopes write headless
plot artifacts (the existing Cairo path).

---

## 7. Tier-6 — Fixed-point filters + HDL bridge + carve-down polish 🔵

Goal: the differentiated track — fixed-point streaming filters that lower
to **synthesizable SystemVerilog with a cocotb SIL test**. The Signal ×
Fixed-Point × HDL convergence no other toolbox can claim.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | fixed-point filter coefficients | `fi`-typed `Numerator`/`SOSMatrix`; `numerictype`/`fimath` on the filter object; coefficient quantization + quantization-error report. | `fi` lane |
| 6.2 | fixed-point `DiscreteState` | The tapped-delay line / accumulator stored as a `fi` array (the shipped persistent-fi-array path). | persistent fi |
| 6.3 | **fixed-point FIR → SV** | A fixed-point `dsp.FIRFilter` lowers to a synthesizable SV tapped-delay line + multiply-accumulate (persistent-fi → SV shift register, the shipped emit-SV path). | `-emit-systemverilog` |
| 6.4 | **cocotb SIL** | The emitted SV FIR validated cycle-by-cycle against the MATLAB reference via a cocotb SIL test (the shipped sequential-DUT SIL lane). | embedded-coder cocotb lane |
| 6.5 | scaling / overflow analysis | `Saturate`/`Wrap` per stage; word-length growth through the MAC; `freqz` of the quantized vs ideal filter. | `fi` overflow modes |
| 6.6 | linear-algebra SOs | `dsp.LUFactor` / `dsp.LDLFactor` / `dsp.LevinsonSolver` / `dsp.LowerTriangularSolver` over the shipped dense kernel. | `mldivide`/`levinson` |
| 6.7 | carve-down polish | `dsp.VariableBandwidthFIRFilter` / `dsp.NotchPeakFilter` / `dsp.CoupledAllpassFilter`; `dsp.PhaseExtractor`; codec demos (ADPCM/G.729 VAD). | Tier-1/2 |

**Headline-within-tier**: the fixed-point FIR HDL bridge —
`fi`-typed `dsp.FIRFilter` → `-emit-systemverilog` → cocotb SIL passes
bit-exact vs the MATLAB reference. The roadmap's differentiated payoff.

**Compile/Execute wiring**: reuses the shipped fixed-point lowering +
persistent-fi-array → SV shift register + the cocotb sequential-DUT SIL
harness; the only new piece is recognising a fixed-point `dsp.FIRFilter`
as an emittable streaming block.

---

## 8. Tier-7 — DSP HDL Toolbox: streaming-hardware foundation + FIR architectures 🔵

Goal: the `dsphdl.*` cycle-accurate hardware surface — the **valid /
backpressure-ready / reset control-signal streaming interface**, the
`dsphdl.*` System-Object model, and the FIR hardware-architecture family,
all flowing to synthesizable SV + cocotb SIL. This is where the project's
emit-SV lane stops being a *bridge* for `dsp.*` and becomes the *native
target* of `dsphdl.*`. **Depends on Tier-1 (SO model) + Tier-6
(fixed-point → SV).**

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 7.1 | **valid/ready/reset control signals** | The cycle-accurate streaming protocol: `[dataOut, validOut] = obj(dataIn, validIn)`, optional backpressure `ready`, synchronous `reset`. Sim model gates state updates on `validIn`; emitted SV carries `valid_in`/`valid_out`/`ready`/`rst_n` ports. The cocotb harness drives the handshake. | cocotb sequential-DUT SIL |
| 7.2 | `dsphdl.*` System-Object model | Hardware variant of the Tier-1 SO lifecycle: scalar-or-vector-per-cycle I/O, fixed-point-native (`fi` in/out), exposed pipeline `latency`, `getLatency`/`info`. | Tier-1 SO model, `fi` |
| 7.3 | `dsphdl.FIRFilter` — fully-parallel systolic | Fully-parallel systolic FIR (one multiplier per tap, pipelined adder tree) → clocked SV; the canonical DSP-HDL block. | Tier-6 SV emit |
| 7.4 | partly-/fully-serial systolic | `FilterStructure`/`SerialPartition` resource-vs-throughput trade: partly-serial (1<N<L multipliers) + fully-serial (1 multiplier, L-cycle) → distinct SV micro-architectures. | resource-share SV |
| 7.5 | transposed + programmable FIR | Fully-parallel transposed architecture; `dsphdl.ProgrammableFIRFilter` (runtime-loadable coefficients via a `coeff`/`writeAddr` port → SV regfile). | runtime-indexed regfile (shipped) |
| 7.6 | high-throughput frame/vector input | Frame-based (vector-per-cycle) FIR for gigasample throughput — N parallel data lanes → N-wide SV datapath. | vector SV datapath |
| 7.7 | `dsphdl.BiquadFilter` / `IIRFilter` | Cascaded biquad with fixed-point per-section scaling → clocked SV SOS chain. | Tier-1 1.4, Tier-6 |
| 7.8 | `dsphdl.LMSFilter` | Hardware LMS (pipelined weight-update datapath) → SV; the UG "HDL Implementation of LMS Filter" + MSE-performance demos. | `matlab_comm_lms`, SV emit |

**Headline-within-tier**: the streaming systolic FIR —
`dsphdl.FIRFilter` valid-handshaked stream → `-emit-systemverilog`
(clocked module with valid/ready ports) → cocotb SIL passes bit-exact and
cycle-aligned vs the MATLAB streaming reference.

**Compile/Execute wiring**: the new piece is the **valid/ready streaming
ABI** — a `dsphdl.*` `obj(data, valid)` call lowers to a `step` that
gates state on `valid` (sim) and to a clocked SV module with handshake
ports (emit). Reuses the shipped sequential-DUT cocotb harness
(pre-edge sampling) + persistent-fi → SV shift register + the
runtime-indexed regfile (for programmable coefficients). New
`runtime/toolbox/dsphdl/` + `dsphdl_classdefs.m`; **`-emit-systemverilog`
is a first-class lane here, not a stretch.**

---

## 9. Tier-8 — DSP HDL Toolbox: multirate + transforms + CORDIC + data management 🔵

Goal: the rest of the `dsphdl.*` block family — the multirate, transform,
CORDIC-math, and FIFO/latency-management blocks. Each is a clocked SV
module with the Tier-7 valid/ready interface + a cocotb SIL test.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 8.1 | `dsphdl.FIRDecimator` / `FIRInterpolator` / `FIRRateConverter` | Polyphase multirate with valid-rate change (output valid asserted every M / L cycles) → SV polyphase commutator. | Tier-7 7.1, `upfirdn` |
| 8.2 | `dsphdl.CICDecimator` / `CICInterpolator` / `CICCompensationDecimator` | Multiplier-free integrator-comb cascade (the FPGA-favourite) + droop-compensation FIR → SV. | hand-coded CIC + 7.1 |
| 8.3 | `dsphdl.Downsample` / `dsphdl.Upsample` | Integer rate change with phase/valid control → SV. | 7.1 |
| 8.4 | `dsphdl.DigitalDownConverter` / `DigitalUpConverter` | NCO mix + multistage CIC/FIR (the NFC/GSM DDC/DUC FPGA demos) → SV chain. | 8.1, 8.2, 8.6 |
| 8.5 | `dsphdl.FFT` / `dsphdl.IFFT` / `dsphdl.ChannelizerFFT` | Streaming pipelined / Radix-2² / burst FFT with bit-reversed or natural order, automatic latency → SV. | `fft` (ref model), 7.1 |
| 8.6 | `dsphdl.NCO` / `dsphdl.SineWave` | Phase-accumulator + LUT (quarter-wave) numerically-controlled oscillator → SV; the "Generate Sine Wave" reference block. | LUT + accumulator |
| 8.7 | `dsphdl.Channelizer` / `dsphdl.ChannelSynthesizer` | Polyphase-FFT analysis/synthesis filter bank → SV (the four-channel synthesizer/channelizer demo). | 8.1, 8.5 |
| 8.8 | CORDIC math | `dsphdl.Complex2MagnitudeAngle` / `dsphdl.SineCosine` / `dsphdl.atan2` / `dsphdl.Sqrt` / `dsphdl.Reciprocal` — iterative CORDIC (rotation/vectoring) → SV; the "Implement atan2 for HDL" reference. | hand-coded CORDIC |
| 8.9 | data management | `dsphdl.SampleAligner` (dual-RAM FIFO aligning two valid-streams, `2^nextpow2(bufDepth+5)`) + `dsphdl.StreamAnalyzer` (sim-only buffer-depth/duty-cycle estimator) + `dsphdl.DelayMatcher` (automatic latency matching across parallel paths). | RAM/FIFO SV + 7.1 |

**Headline-within-tier**: the CIC decimator chain —
`dsphdl.CICDecimator` (multiplier-free) → SV + cocotb SIL bit-exact, the
classic FPGA digital-downconverter front end.

**Compile/Execute wiring**: every block reuses the Tier-7 valid/ready ABI
+ SV emit + cocotb SIL; the new code is the per-block hardware model
(CIC integrator-comb, CORDIC iteration, polyphase commutator, NCO LUT,
SampleAligner dual-RAM controller) — each a self-contained clocked
datapath. `dsphdl.StreamAnalyzer` is **sim-only** (no SV — matches
MathWorks, which excludes it from HDL generation).

---

## 10. Compile/Execute · Debug/REPL · Examples · Tests (cross-cutting)

The four delivery surfaces the project always closes per toolbox — each
tier ships across **all four** before it counts as done.

### 10.1 Compile / Execute

- **Backends**: LLVM JIT + native + `-emit-c` / `-emit-cpp` are the
  primary lanes for the `dsp.*` tiers (the System-Object + persistent-state
  model is C/C++-shaped — a struct with state + a `step` function).
  `-emit-python` / `-emit-typescript` parity is a per-tier stretch.
  **`-emit-systemverilog` IS a first-class target for the fixed-point
  filter objects (Tier-6) and for the entire DSP-HDL `dsphdl.*` surface
  (Tiers 7–8)** — unique among the recent toolboxes — via the shipped
  persistent-fi → SV path + the valid/ready handshake + cocotb SIL. For
  `dsphdl.*`, SV+cocotb is the *primary* lane, not a stretch.
- **Runtime**: `runtime/toolbox/dsp/runtime_dsp.cpp` (filter kernels,
  designers, multirate/CIC/polyphase, streaming stats, spectral
  estimators) + `runtime/toolbox/dsp/dsp_classdefs.m` (the `dsp.*`
  System-Object hierarchy). Add to the strict no-C-cast list
  (`static_cast`), mirroring `runtime_images.cpp`.
- **Wiring**: the **System-Object lifecycle** (Tier-1) is the central
  enablement — `obj(x)` → `step` class-pinned dispatch in
  `Lowering.cpp::CallOrIndex`, `DiscreteState` as object matrix
  properties, `Nontunable`-lock + `setup`/`reset`/`release` runtime
  flags. **This resolves the shared SO verifier blocker (CST §12 / Comm
  §15)** — coordinate the fix so Comm/RF SO tiers flip green too. Builtin
  designers in `Resolver.cpp`; string spec/method args → `matlab_string*`
  in `LowerTensorOps.cpp`; prelude-trigger the `dsp.*` classdefs.

### 10.2 Debug / REPL

- A `dsp.*` / `dsphdl.*` System Object persists across REPL inputs
  (class-tagged slot) and renders in the **DAP variable inspector** —
  including its `DiscreteState` (the tapped-delay line / adaptive weights
  / pipeline registers), so a paused frame loop shows the evolving filter
  state. Reuses the shipped `runtime_debug.cpp` classdef-render path.
- `disp(obj)` formats the MATLAB-faithful property block
  (`dsp.FIRFilter with properties: Numerator: […]`).
- `isLocked(obj)` / `getDiscreteState(obj)` are REPL-inspectable; the
  `step` lifecycle works under the JIT REPL. For `dsphdl.*` objects,
  `getLatency(obj)` reports the pipeline depth.

### 10.3 Examples (`examples/dsp/`)

| Example | Closes | Exercises |
|---|---|---|
| `streaming_fir.m` | **T1 headline** | `dsp.FIRFilter` frame loop with persisted state; noise reduction |
| `biquad_eq.m` | T1 | `dsp.SOSFilter` / `dsp.BiquadFilter` cascade EQ |
| `firpm_design.m` | T2 | `firpm` equiripple + `designfilt`; `freqz` verification |
| `lms_anc.m` | **T3 tracer** | `dsp.LMSFilter` acoustic noise cancellation |
| `rate_convert.m` | T4 | `dsp.SampleRateConverter` multistage 192→44.1 kHz |
| `channelizer.m` | T4 | `dsp.Channelizer`/`ChannelSynthesizer` round-trip |
| `streaming_stats.m` | T5 | `dsp.MovingAverage`/`MovingRMS`/`PeakFinder` envelope |
| `spectrum_estimate.m` | T5 | `dsp.SpectrumEstimator` Welch PSD over frames → PNG |
| `fixedpoint_fir_hdl.m` | **T6 headline** | `fi`-typed `dsp.FIRFilter` → emit-SV + cocotb SIL bit-exact |
| `dsphdl_fir_stream.m` | **T7 headline** | `dsphdl.FIRFilter` valid-handshaked systolic stream → emit-SV (valid/ready ports) + cocotb SIL cycle-aligned |
| `dsphdl_cic_ddc.m` | T8 | `dsphdl.CICDecimator` + `dsphdl.NCO` digital-downconverter front end → SV + cocotb |
| `dsphdl_cordic_atan2.m` | T8 | `dsphdl.atan2` CORDIC vectoring → SV + cocotb vs MATLAB `atan2` |

### 10.4 Tests (`test/Run/`)

Gating tests follow the `dsp_*.m` / `dsphdl_*.m` convention with a
`.stdout` golden + per-backend `.skip-emit-*` files where a lane is out
of scope (Python/TS skipped where the SO path is rough; **SV is a
first-class lane for the fixed-point and all DSP-HDL tests**, each with a
companion `test/EmitSV/` golden + `test/Runtime/` cocotb smoke).

| Test | Tier | Asserts |
|---|---|---|
| `dsp_firfilter.m` | T1 | `dsp.FIRFilter` frame loop = `filter(b,1,x)`; state persists across `step` |
| `dsp_so_lifecycle.m` | T1 | `setup`/`reset`/`release`/`isLocked`; Nontunable lock; `clone` deep-copy |
| `dsp_sosfilter.m` | T1 | `dsp.SOSFilter` = `sosfilt`; per-section state |
| `dsp_firpm.m` | T2 | `firpm` equiripple ripple bound; `designfilt` lowpass spec met |
| `dsp_lms.m` | T3 | `dsp.LMSFilter` converges; weights match `matlab_comm_lms` |
| `dsp_decimator.m` | T4 | `dsp.FIRDecimator` = `upfirdn` decimate; rate correct |
| `dsp_channelizer.m` | T4 | `dsp.Channelizer`/synthesizer reconstruction |
| `dsp_moving_stats.m` | T5 | `dsp.MovingAverage`/`MovingRMS` vs windowed reference |
| `dsp_spectrum.m` | T5 | `dsp.SpectrumEstimator` PSD peak at known tone |
| `dsp_fixedpoint_fir.m` | **T6** | `fi`-typed `dsp.FIRFilter` numeric + **emit-SV golden + cocotb SIL** |
| `dsphdl_firfilter.m` | **T7** | `dsphdl.FIRFilter` valid/ready stream numeric + **SV golden + cocotb SIL cycle-aligned** |
| `dsphdl_systolic_arch.m` | T7 | partly-/fully-serial systolic emit distinct SV; same numeric result |
| `dsphdl_cic.m` | **T8** | `dsphdl.CICDecimator`/`Interpolator` numeric + **SV golden + cocotb** |
| `dsphdl_cordic.m` | T8 | `dsphdl.atan2`/`Sqrt` CORDIC vs MATLAB to fixed-point tol + **SV golden** |
| `dsphdl_nco.m` | T8 | `dsphdl.NCO` LUT oscillator frequency/phase + **SV golden** |

Target: **~15 gating tests** (~10 `dsp.*` + ~5 `dsphdl.*`) plus the
Tier-6/7/8 SV goldens + cocotb smokes — the DSP-HDL tiers materially grow
the `test/EmitSV/` corpus (currently 77 goldens). Full regression must
stay green (currently 465 run-tests, 77 SV goldens) — the badge bumps to
**17 toolboxes** (or higher if Curve Fitting / Wavelet land first), and
**the SO lifecycle fix is expected to flip several currently-skipped
Comm/RF SO tests green** (a regression-count bonus).

---

## 11. Carve-outs (explicitly out of scope)

Matching the per-toolbox precedent — the Simulink / app / Deep-Learning /
deploy surfaces are deferred. **This combined roadmap is unusually
Simulink- and app-heavy** (7 of 16 DSP-System chapters + the radar
chapter of DSP-HDL are Simulink/app), so the carve-out list is large:

- **All Simulink block examples** (Chapters 9–16: Dataflow, Multirate/
  Multistage blocks, Scopes/Logging, Signal I/O, Signal
  Generation/Operations, DSP System blocks, Deep Learning domain, Audio
  channelization) — the MATLAB System-Object API is the whole target; the
  Simulink block library is N/A (the mflowLink lane is the project's
  block-diagram answer, separately roadmapped).
- **Filter Analyzer** + **Filter Designer** apps (Chapter 8 GUIs) and
  `filterDesigner`/`filterAnalyzer` — interactive; the programmatic
  `designfilt`/`fdesign` API (Tier-2) is in scope, the GUI is not.
- **Dataflow domain** (Chapter 9 — multicore simulation / multicore
  codegen) — Simulink-only.
- **The Deep-Learning domain** (Chapter 15 — wavelet-scattering + LSTM/
  autoencoder anomaly detectors, DOA estimation via deep nets, TensorFlow
  Lite / Jetson / Raspberry-Pi deploy) — gated on a future Deep Learning
  toolbox.
- **`dspunfold` / multithreaded MEX generation** (Chapter 1) and **MATLAB
  Compiler standalone/UDP deploy** — host-deploy tooling, not the
  numeric surface.
- **Audio file I/O** (`dsp.AudioFileReader`/`Writer`, real-time audio) —
  Audio Toolbox dependency.
- **`scope`/live-GUI rendering** — scopes ship as **headless PNG/SVG
  artifacts + `getMeasurementsData`** programmatic readout (Tier-5), not
  interactive windows.

DSP-HDL-specific carve-outs (the `dsphdl.*` *objects* are in scope,
Tiers 7–8 — these are the surrounding GUI/deploy/app surfaces):

- **DSP HDL IP Designer app** (`dsphdlIPDesigner`, the "Generate HDL for
  Preconfigured Algorithm" app) and Filter Designer "Export to DSP HDL IP
  Designer" — interactive; the programmatic `dsphdl.*` objects +
  `-emit-systemverilog` are the whole target.
- **Radar Application Examples** (DSP-HDL Chapter 5 — FPGA beamscan DOA /
  MVDR beamformer / monopulse / CA-CFAR / ULA beamformer) — gated on the
  unshipped **Phased Array System Toolbox** (`phased.*`), not on the HDL
  lane itself. The CFAR/beamform *math* could ship later once Phased
  Array exists; the HDL streaming wrappers are ready.
- **Hardware prototyping / deploy** (DSP-HDL Chapter 4 — FPGA support
  packages, board-in-the-loop, `hdlcoder` IP-core/AXI integration) —
  the project emits SV + runs cocotb SIL; physical-board deploy and
  vendor IP-core packaging are out of scope.
- **HDL Coder licensing path** — MathWorks requires HDL Coder to turn a
  `dsphdl.*` object into RTL; **the project's `-emit-systemverilog` lane
  replaces that step**, so no HDL Coder dependency.

These are documented follow-ons, not blockers: every numeric +
System-Object surface a *script* uses (filters / design / adaptive /
multirate / transforms / streaming stats / fixed-point) is in Tiers 1–6,
and the cycle-accurate streaming-hardware `dsphdl.*` blocks (with their
synthesizable-SV + cocotb-SIL deliverables) are in Tiers 7–8.

---

## 12. Effort summary

| Tier | Toolbox | Scope | Effort | Net-new code | Status |
|---|---|---|---|---|---|
| T1 | DSP System | **SO runtime model** + core filter objects | ~2.5 wk | SO lifecycle (shared verifier fix) + `dsp.FIRFilter`/`IIR`/`Biquad`/`SOS` | 🔵 |
| T2 | DSP System | filter design (function-form) | ~1.5 wk | `firpm`/`firls`/`designfilt`/`fdesign`/`iirnotch` | 🔵 |
| T3 | DSP System | adaptive filters | ~1 wk | SO wrappers over Comm `lms`/`rls`/`cma` cores | 🔵 |
| T4 | DSP System | multirate + multistage + filter banks | ~2 wk | polyphase/CIC/Farrow + channelizer + DDC/DUC | 🔵 |
| T5 | DSP System | transforms + sources + streaming stats + measurement | ~2.5 wk | transform/source/buffer/stats/spectral SOs + headless scopes | 🔵 |
| T6 | DSP System | fixed-point filters + HDL bridge + polish | ~2.5 wk | fi-typed filters + fixed-point FIR → SV + cocotb SIL | 🔵 |
| T7 | **DSP HDL** | valid/ready streaming foundation + FIR architectures | ~3 wk | `dsphdl.*` model + valid/ready ABI + systolic/serial/transposed/programmable FIR → SV+cocotb | 🔵 |
| T8 | **DSP HDL** | multirate + transforms + CORDIC + data management | ~3 wk | CIC/decimator/channelizer/FFT/NCO/CORDIC/SampleAligner → SV+cocotb | 🔵 |
| **Total** | | | **~18 wk** | | |

**Recommended slice order**: **T1 first and foremost** — it is both the
keystone for this toolbox *and* a force multiplier that unblocks the
SO-gated Comm / RF / Antenna tiers (do it once, three toolboxes benefit).
T2 (filter design) is independent and shippable in parallel (no SO
dependency). T1 + T2 (~4 wk) close the streaming-filter + design 70%
workflow. T3 (adaptive) is cheap on top of T1 (the cores already ship in
Comm). T4 (multirate) and T5 (streaming stats / spectral) are the breadth
tiers. **T6 → T7 → T8 are the differentiated payoff** — the fixed-point
FIR → SV + cocotb SIL bridge (T6) generalises into the full
cycle-accurate, valid/ready-handshaked `dsphdl.*` hardware family
(T7–T8), the Signal × Fixed-Point × HDL convergence **no other shipped
toolbox can claim**. A high-impact minimal cut is **T1 + T6 + T7**
(~8 wk): the SO model, the fixed-point bridge, and a streaming systolic
`dsphdl.FIRFilter` that emits synthesizable, cocotb-verified SV — the
single most compelling demo this project can produce. The DSP-HDL tiers
depend only on T1 + T6, not on T2–T5, so they can be pulled forward.
