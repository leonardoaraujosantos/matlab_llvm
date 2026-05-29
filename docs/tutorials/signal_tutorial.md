# Signal Processing Toolbox — Tutorial

`matlab_llvm` ships a self-contained subset of the Signal Processing Toolbox that compiles MATLAB scripts straight to native code: IIR/FIR filter design and application, spectral estimation, time-frequency analysis, sample-rate conversion, pulse/peak measurements, and the polynomial primitives that underpin filter design. Every program synthesises its input inline (chirps, tones, pulse trains) and prints diagnostics, so the examples double as runnable smoke tests.

## Supported features

- **IIR design + apply** — `butter` (lowpass / highpass / bandpass via the literal `[W1 W2]` and trailing-`'high'` dispatch), `cheby1`, `cheby2`; apply with `filter`, zero-phase `filtfilt`.
- **Cascade-of-biquads** — `tf2sos` to convert `(b, a)` into an `L×6` SOS matrix, `sosfilt` for numerically robust high-order filtering.
- **FIR design** — `fir1` (windowed-sinc lowpass), plus the DSP-toolbox-shared `firpm` / `firls`.
- **Spectral estimation** — `pwelch` (Welch averaged periodogram), `periodogram`, `spectrogram` (`|STFT|²` per frame), `cpsd`, `mscohere`, `tfestimate`.
- **Parametric / linear prediction** — `levinson`, `lpc`, `aryule`, `arburg`, `pyulear`, `pburg`.
- **Transforms** — `dct` / `idct`, `fwht`, `hilbert`, `goertzel`.
- **Multirate** — `resample(x, p, q)` (polyphase), `decimate`, `interp`, `upfirdn` (anti-aliased).
- **Waveform generators** — `chirp` (linear), `sawtooth`, `square`, `gauspuls`, `rectpuls`, `tripuls`, `sinc`.
- **Pulse / peak measurements** — `findpeaks`, `rms`, `statelevels`, `pulseperiod`, `pulsewidth`, `risetime`, `falltime`, `dutycycle`, `slewrate`, `overshoot`, `undershoot`, `settlingtime`, `envelope`, `hampel`, `medfilt1`.
- **Alignment** — `finddelay` (signed cross-correlation argmax), `xcov`, `dtw` (dynamic time warping).
- **Polynomial primitives** — `roots` (Durand-Kerner), `poly`, `polyder`, `polyint`, `residue` (partial-fraction expansion).

## Build & run

Compile any example end-to-end:

```bash
build/matlabc -emit-llvm examples/signal/lowpass_design.m > /tmp/lowpass_design.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/lowpass_design.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/lowpass_design
/tmp/lowpass_design
```

## Worked examples

### Butterworth lowpass, zero-phase apply  (`examples/signal/lowpass_design.m`)

Designs a 6th-order Butterworth lowpass and applies it forward-and-backward with `filtfilt` (zero phase, steady-state initial conditions).

```matlab
fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 10, 1, 200);

% Wn is normalised to Nyquist (fs/2). 100 / (fs/2) = 0.2.
[b, a] = butter(6, 100/(fs/2));
y      = filtfilt(b, a, x);

fprintf('input  stopband rms: %.4f\n',   rms(xb));
fprintf('output stopband rms: %.4f\n',   rms(yb));
fprintf('stopband attenuation: %.1f dB\n', 20 * log10(rms(xb) / rms(yb)));
```

A 10→200 Hz chirp loses everything above 100 Hz: comparing RMS of the first 200 samples (in-band) to the last 200 (out-of-band) before/after filtering shows the stopband attenuation in dB.

### Bandpass via 2-element `Wn`  (`examples/signal/bandpass_design.m`)

The band-variant dispatch recognises a literal 2-element `[W1 W2]` vector as a bandpass spec automatically.

```matlab
fs = 1000;
x  = chirp(t, 0, 1, 200);

% Bandpass: edges at 30 Hz and 70 Hz, normalised to Nyquist.
[b, a] = butter(4, [0.06 0.14]);
y      = filtfilt(b, a, x);

fprintf('passband centre rms: %.4f\n', rms(y(225:275)));
fprintf('stopband centre rms: %.4f\n', rms(y(725:775)));
fprintf('attenuation: %.1f dB\n', ...
    20 * log10(rms(y(225:275)) / rms(y(725:775))));
```

The chirp crosses 50 Hz (mid-passband) around sample 250 and 150 Hz (deep stopband) around sample 750; the RMS ratio quantifies the band selectivity. An order-`n` bandpass yields `2n+1` coefficients in each of `b` and `a`. `highpass_design.m` is the companion: `cheby1(5, 0.5, 50/(fs/2), 'high')` drops a low-frequency drift via the trailing `'high'` string.

### Cascade-of-biquads (SOS)  (`examples/signal/sosfilt_demo.m`)

Converts a 6th-order Butterworth `(b, a)` to a 3-section SOS matrix and filters with `sosfilt`, the numerically robust alternative to direct `filter` on the transfer-function form.

```matlab
[b, a] = butter(6, 0.2);

% sos is a 3 × 6 matrix: each row is [b0 b1 b2 a0 a1 a2].
sos = tf2sos(b, a);
fprintf('SOS sections: %g\n', size(sos, 1));

y_tf  = filter(b, a, x);
y_sos = sosfilt(sos, x);
fprintf('output rms (filter):  %.4f\n', rms(y_tf));
fprintf('output rms (sosfilt): %.4f\n', rms(y_sos));
```

`tf2sos` returns an `L×6` matrix (`[b0 b1 b2 a0 a1 a2]` per row). The `filter` and `sosfilt` outputs agree to roundoff but the cascade form is far more robust to coefficient quantization at high orders.

### Welch PSD and the STFT spectrogram  (`examples/signal/pwelch_demo.m`, `spectrogram_chirp.m`)

Power spectral density via Welch's averaged periodogram with a Hamming window and 50% overlap.

```matlab
nwin = 256;
win  = hamming(nwin);
P    = pwelch(x, win, 128);

fprintf('PSD bin 25  (mid-band): %.4e\n', P(25));
fprintf('PSD bin 100 (above):    %.4e\n', P(100));
```

`pwelch(x, win, noverlap)` returns a single-sided PSD column; for a 50→200 Hz chirp the mid-band bin sits 1–2 orders of magnitude above the out-of-band bin. `spectrogram_chirp.m` is the time-frequency companion: `S = spectrogram(x, hamming(128), 64)` returns `|STFT|²` as an `(nfreq × nframe)` matrix, and the peak bin marches upward frame-by-frame as the chirp ramps.

### Pulse and peak measurements  (`examples/signal/findpeaks_demo.m`)

Exercises the full §4.3 measurement surface on a tone and a rectangular pulse train.

```matlab
[pks, locs] = findpeaks(s);
fprintf('chirp peaks found:  %g\n', length(pks));

sl = statelevels(x);
fprintf('low  state level:   %.3f\n', sl(1));
fprintf('high state level:   %.3f\n', sl(2));
fprintf('pulse period:       %.3f\n', pulseperiod(x));
fprintf('pulse width:        %.3f\n', pulsewidth(x));
fprintf('rise time:          %.3f\n', risetime(x));
fprintf('duty cycle:         %.3f\n', dutycycle(x));
```

`findpeaks` returns local maxima with their indices; `statelevels` estimates low/high states from a 100-bin histogram and feeds the 10/50/90% pulse statistics (`pulseperiod`, `pulsewidth`, `risetime`, `falltime`, `dutycycle`).

### Cross-correlation alignment  (`examples/signal/xcorr_align.m`)

Recovers a known delay between two signals via the cross-correlation argmax.

```matlab
delay = 25;
y = zeros(1, length(x));
for k = 1:length(x) - delay
  y(k + delay) = x(k);
end

d = finddelay(x, y);          % signed: -25 (y lags x)
c = xcov(x, y);               % mean-removed cross-correlation
fprintf('dtw(x, y):  %.3f\n', dtw(x, y));
```

`finddelay` returns the signed lag (negative when the second signal lags the first), `xcov` the mean-removed cross-correlation, and `dtw` a scalar dynamic-time-warping distance (small for similar signals).

Other examples: `resample_demo.m` (`resample` / `decimate` / `interp` length and energy checks) and `poly_helpers.m` (`roots` / `poly` / `polyder` / `polyint` / `residue`, the bedrock of every IIR design path).

## Limitations & carve-outs

- **No elliptic / analog-prototype designers** — `ellip`, `ellipord`, `besself`, `freqs`, standalone `bilinear`, `cheb2ord` are open.
- **No subspace / advanced spectral** — `dpss` + `pmtm` (multitaper), `pmusic` / `peig` / `rootmusic`, `pcov` / `pmcov`, `prony` / `stmcb` are open.
- **`spectrogram` / `pwelch` are single-output** — the multi-return `[P, f, t, …]` forms and `stft` / `istft` / `pspectrum` / `instfreq` are open.
- **`findpeaks` name-value options** (`MinPeakHeight`, `MinPeakDistance`, `MinPeakProminence`, `Threshold`, `SortStr`) are gated on Sema name-value parsing.
- **`chirp` is linear-method only** (no quadratic / logarithmic / hyperbolic); `pulstran`, `diric`, `gmonopuls`, `vco` are open.
- **Out of scope entirely** — interactive apps (Filter Builder / Analyzer / Designer, Signal Analyzer, Signal Labeler), Simulink data inspector, Deep-Learning entries (LSTM/GAN/anomaly detectors), MATLAB Coder codegen, Python coexecution.

## See also

- Roadmap / design: [`../signal_toolbox_roadmap.md`](../signal_toolbox_roadmap.md)
- Examples directory: `examples/signal/`
