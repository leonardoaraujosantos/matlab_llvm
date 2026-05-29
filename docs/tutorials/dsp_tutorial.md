# DSP System Toolbox — Tutorial

`matlab_llvm` implements the System-Object surface of the DSP System Toolbox: `dsp.*` objects that you construct once and call frame-by-frame (`y = obj(frame)`), with the object's internal state (tapped-delay lines, adaptive weights, polyphase commutator) persisting across calls. The compiler lowers the call-syntax `obj(frame)` to the object's `step` method, and the handle semantics carry state forward so a signal processed in frames is bit-identical to filtering it whole. A `dsphdl.*` cycle-accurate hardware-counterpart surface and a fixed-point (`fi`) FIR that lowers to synthesizable SystemVerilog round out the toolbox.

## Supported features

- **Tier-1 streaming filters** — `dsp.FIRFilter`, `dsp.IIRFilter`, `dsp.BiquadFilter`, `dsp.SOSFilter`, `dsp.Delay`; System-Object lifecycle (`setup` → `step` ×N → `reset`), `getDiscreteState`, `getWeights`.
- **Tier-2 function-form designers** — `firpm` (Parks-McClellan equiripple), `firls` (least-squares), `iirnotch`, `iirpeak`.
- **Tier-3 adaptive filters** — `dsp.LMSFilter` (LMS + normalized-LMS via `Method`), `dsp.RLSFilter`; `getWeights` reads the converged taps.
- **Tier-4 multirate** — `dsp.FIRDecimator`, `dsp.FIRInterpolator`, `dsp.SampleRateConverter` (polyphase rational L/M), `dsp.CICDecimator`, `dsp.CICInterpolator`.
- **Tier-5 sources / stats** — `dsp.SineWave`, `dsp.SpectrumEstimator`, `dsp.MovingAverage`, `dsp.MovingRMS`, `dsp.PeakFinder`, `dsp.LevinsonSolver`.
- **Tier-6 convenience + fixed-point** — `dsp.LowpassFilter`, `dsp.HighpassFilter`, `dsp.NotchPeakFilter`; persistent-`fi` FIR (`fi` / `numerictype`) that lowers to synthesizable SystemVerilog via `-emit-systemverilog`.
- **Tier-7/8 HDL counterparts** — `dsphdl.FIRFilter`, `dsphdl.NCO`, `dsphdl.CICDecimator` with `getLatency()`; `cordic_atan2` CORDIC vector op. (MATLAB-side simulation today; clocked valid/ready SV emit is a documented follow-on.)

## Build & run

Compile any example end-to-end:

```bash
build/matlabc -emit-llvm examples/dsp/streaming_fir.m > /tmp/streaming_fir.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/streaming_fir.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/streaming_fir
/tmp/streaming_fir
```

The fixed-point FIR can additionally drive the hardware lanes:
`build/matlabc -emit-systemverilog examples/dsp/fixedpoint_fir_hdl.m` and
`build/matlabc -check-synthesizable examples/dsp/fixedpoint_fir_hdl.m`.

## Worked examples

### Frame-based streaming FIR  (`examples/dsp/streaming_fir.m`)

The canonical demo: build a `dsp.FIRFilter` once, stream a signal through it in frames, and prove the delay-line state carries across frame boundaries.

```matlab
b = fir1(15, 0.25);
firFilt = dsp.FIRFilter('Numerator', b);

% Stream the signal through in 5 frames of 10 samples each.
y = zeros(1, 50);
for k = 1:5
    idx = (k - 1) * 10 + (1:10);
    y(idx) = firFilt(x(idx));
end

% Confirm the streamed result equals the monolithic filter.
yref = filter(b, 1, x);
fprintf('frame-vs-whole maxdiff = %.6f\n', max(abs(y - yref)));
```

Each `firFilt(x(idx))` call is a `step` that filters the frame and carries the tapped-delay state into the next. The `maxdiff` against a whole-signal `filter` is 0 — proof that the object state model is bit-exact.

### Equiripple FIR design feeding a System Object  (`examples/dsp/firpm_design.m`)

Designs a narrow-transition lowpass with `firpm` (Parks-McClellan), compares it to `firls`, then streams a two-tone signal through a `dsp.FIRFilter` built from the equiripple taps.

```matlab
N = 30;
edges = [0 0.2 0.3 1];
amp   = [1 1 0 0];
b_pm = firpm(N, edges, amp);
b_ls = firls(N, edges, amp);

firFilt = dsp.FIRFilter('Numerator', b_pm);
y = zeros(1, 256);
for f = 1:8
    idx = (f - 1) * 32 + (1:32);
    y(idx) = firFilt(x(idx));
end
fprintf('input power  = %.4f\n', sum(x .^ 2) / 256);
fprintf('output power = %.4f\n', sum(y .^ 2) / 256);

% Notch out a 0.4-Nyquist interferer.
[bn, an] = iirnotch(0.4, 0.08);
yn = filter(bn, an, interferer);
```

`firpm`/`firls` take `(order, band-edges, amplitudes)`. The stopband tone is heavily attenuated so output power drops sharply; `iirnotch(w0, bw)` returns `(b, a)` for a second-order notch that suppresses a single interferer.

### Adaptive noise cancellation with LMS  (`examples/dsp/lms_anc.m`)

The classic adaptive-filter demo: a clean tone buried under noise that is a filtered copy of a measurable reference. `dsp.LMSFilter` models the acoustic path and subtracts it.

```matlab
mic = clean + corrupting;             % what the primary microphone hears

anc = dsp.LMSFilter('Length', 12, 'StepSize', 0.05);
anc.Method = 1;                       % normalized LMS
recovered = anc(ref, mic);

resid_rms = sqrt(mean((recovered(tail) - clean(tail)) .^ 2));
w = anc.getWeights();
fprintf('estimated path: %.2f %.2f %.2f %.2f\n', w(1), w(2), w(3), w(4));
```

`anc(ref, mic)` adapts on the streaming call; `Method = 1` selects normalized-LMS for power-independent convergence. After settling, the error output is the recovered signal and `getWeights` returns the converged echo-path estimate. `adaptive_eq.m` composes the same object as an equalizer that inverts a `dsp.FIRFilter` channel.

### Rational sample-rate conversion  (`examples/dsp/rate_convert.m`)

A 3/2 sample-rate converter (interpolate-by-3 then decimate-by-2) whose polyphase state must persist across frames.

```matlab
L = 3; M = 2;
b  = fir1(60, 0.31);
src = dsp.SampleRateConverter(L, M, b);

ys = zeros(1, 256 * L / M);
for k = 1:4
    idx_in  = (k - 1) * 64 + (1:64);
    idx_out = (k - 1) * (64 * L / M) + (1:(64 * L / M));
    ys(idx_out) = src(x(idx_in));
end

% Framed-streaming must equal a fresh whole-signal SO exactly.
src_ref = dsp.SampleRateConverter(L, M, b);
yref = src_ref(x);
fprintf('frame-vs-whole maxdiff = %.6f\n', max(abs(ys - yref)));
```

`dsp.SampleRateConverter(L, M, b)` takes the rate factors and the anti-aliasing FIR. Output length is `N·L/M`; the `maxdiff` of 0 proves the polyphase commutator + FIR state is consistent across the streaming boundary. A `dsp.CICDecimator(4)` with `NumSections = 2` demonstrates multiplier-free decimation.

### Sliding statistics and spectrum estimation  (`examples/dsp/streaming_stats.m`, `spectrum_estimate.m`)

Three stateful estimators run side-by-side over a `dsp.SineWave` source.

```matlab
src = dsp.SineWave('Frequency', 5);
src.SampleRate = 1000;
src.SamplesPerFrame = 100;

ma   = dsp.MovingAverage('WindowLength', 16);
mrms = dsp.MovingRMS('WindowLength', 32);
pf   = dsp.PeakFinder();
for k = 1:10
    idx = (k - 1) * 100 + (1:100);
    ys(idx) = ma(x(idx));
    yr(idx) = mrms(x(idx));
    yp(idx) = pf(x(idx));
end
fprintf('RMS settled   = %.3f   (sine RMS = 1/sqrt(2))\n', yr(800));
```

Each object persists its window across the 10 frames. `spectrum_estimate.m` drives a two-tone signal through `dsp.SpectrumEstimator('FFTLength', N)` with exponential averaging, then reads the PSD peaks at the expected one-sided bins `round(f/fs·N)`.

### Fixed-point FIR for SystemVerilog emit  (`examples/dsp/fixedpoint_fir_hdl.m`)

The form you reach for when targeting silicon: a `fi`-typed FIR with a persistent tapped-delay line and a constant-coefficient table, which lowers to a clocked SV module.

```matlab
function r = fir_filter_fi(x)
    %#codegen
    % hdl: port(x, fi, signed, 16, 12)
    % cocotb: latency(1)
    h = fi([1, 2, 3, 4, 3, 2, 1], 1, 16, 0);
    persistent delay_line;
    if isempty(delay_line)
        delay_line = fi(zeros(1, 7), 1, 16, 12);
    end
    delay_line = [fi(x, 1, 16, 12), delay_line(1:6)];
    r = delay_line(1)*h(1) + delay_line(2)*h(2) + ... + delay_line(7)*h(7);
end
```

The `hdl: port` directive pins the I/O `fi` types; `-emit-systemverilog` produces a clocked module with the persistent delay line as N parallel registers and the coefficient table as a static SV lookup. `dsphdl_fir_stream.m` and `fpga_ddc.m` show the `dsphdl.*` cycle-accurate counterparts (`dsphdl.FIRFilter`, `dsphdl.NCO`, `dsphdl.CICDecimator` with `getLatency()`) building a digital down-converter front-end.

## Limitations & carve-outs

- **`dsphdl.*` SV emit is a follow-on** — the objects simulate in MATLAB and report latency, but the clocked valid/ready/reset SystemVerilog datapath emit is not yet wired (only the function-form persistent-`fi` FIR lowers to SV today).
- **All Simulink block examples are out of scope** — the MATLAB System-Object API is the target; the Simulink block library is N/A (the mflowLink lane is the separate block-diagram answer).
- **No interactive apps** — Filter Designer / Filter Analyzer GUIs are out; the programmatic `designfilt` / `fdesign` API is the in-scope counterpart.
- **No Deep-Learning domain** (wavelet-scattering + LSTM/autoencoder anomaly detectors, deep-net DOA), gated on a future Deep Learning toolbox.
- **No audio file I/O** (`dsp.AudioFileReader`/`Writer`, real-time audio) — Audio Toolbox dependency.
- **Scopes are headless** — they ship as PNG/SVG artifacts + `getMeasurementsData`, not interactive windows; `dspunfold` / multithreaded MEX and host-deploy tooling are out.
- **HDL Coder licensing path replaced** — `-emit-systemverilog` + cocotb SIL replace the HDL Coder RTL step; physical-board deploy, FPGA support packages, and vendor IP-core packaging are out. FPGA radar/beamform examples are gated on the unshipped Phased Array System Toolbox.

## See also

- Roadmap / design: [`../dsp_toolbox_roadmap.md`](../dsp_toolbox_roadmap.md)
- Examples directory: `examples/dsp/`
