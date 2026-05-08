# Signal Processing Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Runtime + Debug + REPL) needs to
ship in order to faithfully execute MATLAB Signal Processing Toolbox
programs. Source: *Signal Processing Toolbox User's Guide* (R2026a,
2048 pages, 27 chapters).

The repo's overall compatibility target is a **practical numeric
subset** (see `feature_status.md`), so this doc inherits the same
posture: focus on the *programmable* surface (functions returning
arrays / structs), explicitly defer the GUI surface (Filter Designer,
Signal Analyzer, Signal Labeler, Signal Feature Extractor apps) and
the deep-learning / Simulink interop chapters.

For shipped work, see [`feature_status.md`](feature_status.md). For
the cross-toolbox roadmap entries, see [`roadmap.md`](roadmap.md) —
this doc is the per-toolbox companion.

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order.
  Tier-1 items are mostly closures over already-shipped infrastructure;
  Tier-3+ items need new runtime/dialect work first.
- **Effort** is in the existing Phase 5.6.x cadence
  (one focused session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
- **REPL / Debug** rows note display + DAP variable-inspector
  expectations. Most signal functions return `matlab_mat *` and inherit
  the existing matrix display path with no extra work; only
  *new descriptor types* (e.g. `digitalFilter` system object,
  `dsp.SOSFilter`) need REPL/DAP rendering and are flagged.

---

## 1. Already shipped (Tier-0 baseline)

These are wired through Sema → MLIR → LLVM/C/C++/Python/TS lanes
today. Locations are in `runtime/matlab_runtime.cpp`.

| Group | Functions | Notes |
|---|---|---|
| Convolution | `conv`, `conv2`, `xcorr` | Polynomial product + 2-D outer-product + linear cross-correlation. Tests in `test/Runtime/test_signal.c`. |
| Filter | `filter(b, a, x)` | Direct-form II transposed; scalar IIR / FIR. |
| FFT / shift | `fft`, `ifft`, `fft2`, `ifft2`, `fftshift`, `ifftshift` | Pure-C Cooley-Tukey radix-2 + Bluestein for general N. See `complex.md`. |
| Windows tail (§2.3) | `hamming`, `hann`, `blackman`, `rectwin`, `triang`, `bartlett`, `barthannwin`, `bohmanwin`, `parzenwin`, `nuttallwin`, `blackmanharris`, `flattopwin`, `kaiser`, `tukeywin`, `gausswin`, `chebwin`, `taylorwin` | Symmetric (non-periodic) form. Two-arg parametric windows take their shape parameter as the second double; `taylorwin` takes `(n, nbar, sll)`. |
| Polynomial helpers (§2.4) | `roots`, `poly`, `polyder`, `polyint`, `polyint(p, k)`, `[r, p, k] = residue(b, a)` | Distinct-pole `residue`; multi-return shape mirrors `[V, D] = eig`. |
| IIR design + frequency response (§2.1, lowpass scope) | `[b, a] = butter(n, Wn)`, `[b, a] = cheby1(n, Rp, Wn)`, `[b, a] = cheby2(n, Rs, Wn)`, `H = freqz(b, a, N)`, `[H, w] = freqz(b, a, N)`, `[n, Wn] = buttord(Wp, Ws, Rp, Rs)`, `[n, Wn] = cheb1ord(Wp, Ws, Rp, Rs)` | Bilinear-transform design; unit DC gain. cheby2 has finite j-axis zeros via the generalized `lowpass_from_analog_pz_` helper. Order helpers split via paired `_n` / `_Wn` runtime entries. TS lane gates `freqz` tests with `.skip-emit-typescript` because `NDArray` has no native complex (same as `roots`/`fft_c`). |
| FIR design (§2.2, lowpass scope) | `b = fir1(n, Wn)`, `B = sgolay(k, f)`, `y = sgolayfilt(x, k, f)` | `fir1` is windowed-sinc with default Hamming, normalized to unit DC gain. `sgolay` returns the standard `B = V (V'V)^-1 V'` projection matrix; `sgolayfilt` applies B's middle row in steady state and the matching boundary rows at the first/last `(f-1)/2` samples. |
| Filter implementation (§2.5) | `y = filtfilt(b, a, x)`, `y = sosfilt(sos, x)`, `h = impz(b, a, N)`, `s = stepz(b, a, N)`, `gd = grpdelay(b, a, N)` | Internal direct-form-II-transposed `filter_flat_` helper. `filtfilt` uses reflection padding + zero ICs (Gustafsson initial-condition trick is a follow-on). `grpdelay` is finite-difference on `arg(H(e^{jω}))`. |
| Tier-2 §3.4 transforms — `dct`, `idct`, `fwht`, `hilbert`, `goertzel` | ✅ shipped | DCT-II/III direct O(N²); fwht in-place butterfly (Hadamard ordering, /N); hilbert via FFT zero-negative-half; goertzel single-bin (1×1 complex). **Open**: czt, dst/idst, cceps/rceps. |
| Tier-2 §3.1 nonparametric spectral — `periodogram`, `pwelch`, `cpsd`, `mscohere`, `tfestimate` | ✅ shipped | Single-output, fs = 1. `cpsd`/`tfestimate` return complex (matlab_mat_c). **Open**: dpss + pmtm (multitaper); 2-/3-return `[P, f, …]` forms. |
| Tier-2 §3.2 linear prediction + parametric PSD — `levinson`, `lpc`, `aryule`, `arburg`, `pyulear`, `pburg` | ✅ shipped | Levinson-Durbin + Burg recursion; AR PSD via σ²·\|1/A(e^{jω})\|² evaluation. **Open**: pcov/pmcov, subspace methods (pmusic/peig/rootmusic/rooteig), prony/stmcb. |
| Tier-2 §3.3 time-frequency — `spectrogram(x, win, noverlap)` | ✅ shipped (single-output) | \|STFT\|² per (freq, frame). **Open**: stft/istft, pspectrum, instfreq, instbw, cwt (Tier-4 wavelets), wvd/fsst (Tier-4). |
| Tier-3 §4.3 pulse measurements — `findpeaks`, `rms`, `peak2peak`, `peak2rms`, `rssq`, `medfilt1`, `hampel`, `envelope`, `midcross`, `risetime`, `falltime`, `dutycycle` | ✅ shipped (core surface) | Strict-monotonic findpeaks; MAD-based hampel; peak-interpolation envelope; 10/50/90% pulse statistics. **Open**: MinPeak* options for findpeaks; slewrate, pulseperiod, pulsewidth, overshoot, undershoot, settlingtime, statelevels. |
| Tier-3 §4.1 real multirate — `upfirdn`, `decimate`, `interp`, `resample` | ✅ shipped | Anti-aliased; replaces the toy upsample/downsample stubs. **Open**: polyphase, group-delay correction. |
| Tier-3 §4.2 waveform generators — `chirp`, `sawtooth`, `square`, `gauspuls`, `rectpuls`, `tripuls`, `sinc` | ✅ shipped | chirp linear method only. **Open**: chirp quadratic/log/hyperbolic, `pulstran`, `diric`, `gmonopuls`, `vco`. |
| Tier-3 §4.4 alignment helpers — `xcov`, `finddelay`, `dtw` | ✅ shipped | **Open**: `alignsignals` (multi-return), `gccphat`, xcorr scaling-options. |
| Multirate stubs | `upsample(x, n)`, `downsample(x, n)` | Zero-stuff / decimate; **no** anti-aliasing filter (raw `decimate`/`resample` still TODO). |
| Numeric utilities used by SPT | `diff`, `polyfit`, `polyval`, `interp1`, `interp2`, `trapz`, `gradient` | |
| Complex scalar / matrix arithmetic | `conj`, `real`, `imag`, `angle`, complex `+ - .* ./ * /` | Required for any spectrum / transfer-function math. |

**Coverage today** closes the full Tier-1 design-and-apply loop for
*lowpass* filters:

- **Design**: `butter` / `cheby1` / `cheby2` (IIR), `fir1` (FIR),
  `sgolay`/`sgolayfilt` (Savitzky-Golay), order helpers `buttord` /
  `cheb1ord` for picking n.
- **Apply**: `filter` (causal IIR), `filtfilt` (zero-phase), `sosfilt`
  (cascade biquad).
- **Inspect**: `freqz` (frequency response), `impz` (impulse), `stepz`
  (step), `grpdelay` (group delay).

Polynomial helpers `roots` / `poly` / `polyder` / `polyint` /
`residue` plus the 17-entry windows tail round out the §2.3 / §2.4
mathematical infrastructure.

What's still **open**:

- Highpass / bandpass / bandstop variants of `butter` / `cheby1` / `cheby2`.
- `ellip`, `besself`, analog prototypes (`buttap` etc.) as
  standalone builtins, standalone `bilinear`, analog `freqs`,
  `cheb2ord` / `ellipord`, form conversions (`tf2zp` / `tf2sos` /
  etc.).
- `fir2`, `firls`, `firpm`, `firrcos`, `kaiserord` (richer FIR design).
- `phasez`, `zerophase`, true zero-edge-transient `filtfilt`
  (Gustafsson 1996 IC trick).

Beyond Tier-1: spectral / time-frequency / multirate / measurement /
linear-prediction tiers — see §3 / §4 below.

---

## 2. Tier 1 — close the FIR/IIR design loop (lowpass core ✅, band variants 🔵)

This tier closes the smallest end-to-end loop a signal-processing user
needs: *design a filter, apply it, look at its response*. All Tier-1
items are pure-numeric and slot into the existing `matlab_mat *`
runtime; no new dialect or descriptor work is needed.

**Status snapshot**: lowpass design + apply + inspect surface is
shipped (see §1 above). The remaining 🔵 items in §2.1–§2.5 are the
band variants, the analog-prototype builtins / form conversions, the
richer FIR designers, and the Gustafsson zero-edge `filtfilt` IC trick.

### 2.1 Filter design — IIR

| Function | Form | Status |
|---|---|:-:|
| `butter(n, Wn)` lowpass + `[b, a]` multi-return | `[b,a] = butter(...)` | ✅ shipped |
| `cheby1(n, Rp, Wn)` lowpass + `[b, a]` multi-return | `[b,a] = cheby1(...)` | ✅ shipped |
| `freqz(b, a, N)` + 2-return `[H, w] = freqz(...)` | scalar N | ✅ shipped |
| `butter(n, Wn, 'high'/'bandpass'/'stop')` band variants | requires multi-band Wn parsing | 🔵 follow-on |
| `cheby2(n, Rs, Wn, ...)` | needs j-axis zero handling | 🔵 follow-on |
| `ellip(n, Rp, Rs, Wn, ...)` | needs Jacobi elliptic functions | 🔵 follow-on |
| `besself(n, Wo)` | continuous-time only | 🔵 follow-on |
| `buttap` / `cheb1ap` / `cheb2ap` / `ellipap` / `besselap` | analog prototypes — building blocks | 🔵 follow-on |
| `bilinear(b, a, fs)` | analog→digital, exposed as a builtin | 🔵 follow-on |
| `freqs(b, a, w)` | analog frequency response | 🔵 follow-on |
| `tf2zp` / `zp2tf`, `tf2sos` / `sos2tf`, `tf2ss` / `ss2tf`, `zp2sos` | form conversions | 🔵 follow-on |
| `buttord` / `cheb1ord` / `cheb2ord` / `ellipord` | order-selection helpers | 🔵 follow-on |

**What shipped (lowpass core)**:
- Bilinear-transform design from analog Butterworth / Chebyshev I
  prototypes; n complex poles (conjugate-symmetric in s-plane) → n
  complex z-plane poles (conjugate-symmetric in unit disk) → real
  `(b, a)` of length n+1 each, normalized to unit DC gain.
- Bilinear is **inlined** in `compute_butter_` / `compute_cheby1_`;
  exposing it as a separate builtin is a follow-on.
- `freqz` evaluates `H(e^{jw})` at N equally spaced points on
  `[0, π)` via direct loop (O(N·M)). Reusing the FFT runtime for
  power-of-two N is a possible optimization.

**TS-lane caveat**: `freqz` returns a complex column on the C / C++ /
LLVM / Python lanes (via `matlab_mat_c`) so polymorphic `abs(H)` /
`real(H)` / `imag(H)` work. The TS `NDArray` has no native complex
shape — same friction as the existing `roots` and `fft_c` TS
behaviour — so `sig_iir.m` carries `.skip-emit-typescript`. Gating:
`test/Run/sig_iir.m` (3-lane: C/C++/LLVM + Python with bracket-repr
`.stdout-python` override) + 5 direct C unit tests.

**Follow-on slices** (in expected order):
1. `cheby2` + `ellip` (separate j-axis-zero handling, Jacobi elliptic
   functions for `ellip`).
2. High / band / stop variants of `butter` / `cheby1` (frequency
   transformations on the analog prototype before bilinear).
3. `besself` continuous-time prototype.
4. Analog prototypes (`buttap` / `cheb1ap` / …) as standalone
   builtins — useful for users wanting custom designs.
5. `bilinear` and `freqs` as standalone builtins.
6. Order helpers (`buttord` and friends).
7. Form conversions (`tf2zp` / `tf2sos` / etc.).

### 2.2 Filter design — FIR (lowpass core ✅; richer designers 🔵)

| Function | Notes |
|---|---|
| `fir1(n, Wn[, ftype][, win])` | Windowed-sinc FIR design. Default window = `hamming`. |
| `fir2(n, f, m[, win])` | Frequency-sampling FIR design. |
| `firls(n, f, a[, w])` | Least-squares FIR. |
| `firpm(n, f, a[, w])` (Parks-McClellan) | Remez exchange — non-trivial; defer to Tier-2 if needed. |
| `firrcos(n, fc, df, fs)` | Raised-cosine FIR. |
| `kaiserord(fcuts, mags, devs[, fs])` | Kaiser-window order estimator. |
| `sgolay(k, f)`, `sgolayfilt(x, k, f)` | Savitzky-Golay design + filter. |

**Gating tests**: `sig_fir1_lp.m`, `sig_fir2_arbitrary.m`,
`sig_sgolay_smooth.m`.

### 2.3 More windows (1 session) ✅

Used by every spectral / FIR design entry above.

```
kaiser(N, beta)         barthann(N)
bartlett(N)             tukeywin(N, r)
gausswin(N, alpha)      chebwin(N, r)
taylorwin(N, nbar, sll) nuttallwin(N)
flattopwin(N)           parzenwin(N)
rectwin(N)              triang(N)
blackmanharris(N)       bohmanwin(N)
```

All are length-N column vectors with closed-form coefficients —
straight loops in the runtime, mirrored across Python / TS.

**REPL / Debug**: column vectors; nothing new.

### 2.4 Polynomial / rational helpers ✅ (distinct-pole `residue`; repeated-pole grouping 🔵)

Tier-1 filter conversions all flow through `roots` / `poly`. These
also unblock arbitrary `polyval`-style work:

| Function | Notes |
|---|---|
| `roots(p)` | Companion-matrix eigenvalues. Reuse the shipped `eig` (Phase 5 Jacobi); for non-symmetric companion matrices we need a **correct** non-symmetric eigensolver (already in roadmap.md). Until then, fall back to QR-iteration on the companion matrix. |
| `poly(r)` | Coefficients from roots — straight conv chain. |
| `residue(b, a)` / `residuez(b, a)` | Partial-fraction expansion. |
| `polyder` / `polyint` | Pure scalar arithmetic. |
| `roots`-aware `polyfit` | Already shipped; extend to accept `[p, S, mu]` 3-return for centering / scaling. |

**Risk**: `roots` correctness on complex companion matrices is the
critical blocker. If general non-symmetric eig isn't ready, ship a
*Schur-form*-based root finder narrowly scoped to the companion
matrix shape (Hessenberg by construction, so QR with implicit
shifts converges robustly).

### 2.5 Tier-1 closure: zero-phase + filtfilt ✅ (Gustafsson IC + `phasez`/`zerophase` 🔵)

| Function | Notes |
|---|---|
| `filtfilt(b, a, x)` | Forward-backward filtering. Endpoint-reflection padding + initial-condition matching (the Gustafsson 1996 trick is what MATLAB ships). |
| `sosfilt(sos, x)` | Cascade of biquads — runtime kernel; needed for numeric stability of high-order IIR. |
| `impz(b, a, n)` / `stepz(b, a, n)` | Synthesize impulse / step response — drives `filter` with a unit pulse / step. |
| `freqz(b, a, n)` (covered in 2.1) | |
| `grpdelay(b, a, n)` | Group delay — `-d arg(H) / d w`; reuse `freqz` + finite-difference. |
| `phasez(b, a, n)`, `zerophase(b, a, n)` | Phase / zero-phase response. |

After 2.1 + 2.2 + 2.3 + 2.4 + 2.5 ship, a user can write end-to-end:

```matlab
[b, a]  = butter(6, 0.2);          % design
y       = filtfilt(b, a, x);       % apply with zero phase
[h, w]  = freqz(b, a, 1024);       % inspect response
```

…and have it work in the C / C++ / Python / TypeScript lanes plus
the REPL with cross-input persistence.

---

## 3. Tier 2 — spectral analysis & time-frequency (~4 weeks)

### 3.1 Nonparametric spectral analysis ✅ (`periodogram`/`pwelch`/`cpsd`/`mscohere`/`tfestimate`); multitaper 🔵

| Function | Notes |
|---|---|
| `periodogram(x[, win, nfft, fs])` | `\|FFT\|^2 / (fs * sum(win.^2))`. 4-return form `[pxx, f, pxxc, freqlims]` — start with 2-return `[pxx, f]`. |
| `pwelch(x, win, noverlap, nfft, fs)` | Segment-and-average periodogram. Built on top of `periodogram`. |
| `pmtm(x, nw, nfft, fs)` | Multitaper (Slepian / DPSS); needs `dpss(N, NW, K)` window generator. |
| `cpsd(x, y, ...)` | Cross-spectral density. |
| `mscohere(x, y, ...)` | Magnitude-squared coherence. `cpsd` / `pwelch` ratio. |
| `tfestimate(x, y, ...)` | Transfer-function estimate. |

`dpss(N, NW, K)` requires solving a tridiagonal symmetric eigenvalue
problem — the shipped Jacobi eig handles it, but a banded solver
would be faster. Acceptable to start with the dense Jacobi.

**REPL / Debug**: `[pxx, f]` are plain matrices; reuse existing
display.

**Gating tests**: `sig_periodogram_basic.m`,
`sig_pwelch_white_noise.m`, `sig_mscohere_two_sin.m`.

### 3.2 Parametric / model-based PSD ✅ (`lpc`/`levinson`/`aryule`/`arburg`/`pyulear`/`pburg`); subspace + `prony`/`stmcb` 🔵

Builds on Tier 1.4 polynomial helpers + a small set of LP routines.

| Function | Notes |
|---|---|
| `lpc(x, p)` | Linear prediction coefficients. Levinson-Durbin recursion. |
| `levinson(r, p)` | Levinson-Durbin from autocorrelation. |
| `aryule(x, p)` | Yule-Walker AR estimator. Wraps `xcorr` + `levinson`. |
| `arburg(x, p)` | Burg AR estimator. |
| `pyulear(x, p, ...)` | AR PSD via Yule-Walker. |
| `pburg(x, p, ...)` | AR PSD via Burg. |
| `pcov` / `pmcov` | Covariance / modified covariance AR PSDs. |
| `pmusic` / `peig` / `rooteig` / `rootmusic` | Subspace methods. Need eigendecomposition of the autocorrelation matrix — leverages shipped `eig`. |
| `prony(h, nb, na)` / `stmcb(x, nb, na)` | IIR design from impulse response / time-domain. |

### 3.3 Time-frequency analysis ✅ (`spectrogram` single-output); STFT/`pspectrum`/`instfreq`/`instbw` 🔵

| Function | Notes |
|---|---|
| `spectrogram(x, win, noverlap, nfft, fs)` | STFT magnitude squared. 4-return `[s, f, t, ps]`. |
| `stft(x, fs, ...)` / `istft(s, fs, ...)` | Forward / inverse STFT with COLA window normalization. |
| `pspectrum(x, fs, type)` | Newer wrapper around `pwelch` / `spectrogram` / persistence. |
| `hilbert(x)` | Analytic signal — already ships in transform tail (3.4). Listed here because it underlies envelope / instantaneous-frequency. |
| `instfreq(x, fs)`, `instbw(x, fs)` | Instantaneous frequency / bandwidth. |
| `cwt(x, ...)` | Continuous wavelet transform — Morse / Morlet / bump wavelets. **Defer to Tier 3** (heavy lift; needs scalogram type). |
| `wvd(x)` (Wigner-Ville) | Bilinear time-frequency distribution. **Defer to Tier 3**. |
| `fsst(x, fs)` / `ifsst(s)` | Fourier synchrosqueezed transform. **Defer to Tier 3**. |

**REPL / Debug**: `spectrogram` returns a matrix `s` plus frequency
and time vectors. Display works as-is. The user-facing
*plot* form (no LHS) is a no-op in our headless runtime —
emit a message + return invisibly, matching how other
plot-only entries are handled.

### 3.4 Other transforms — close the chapter-17 surface ✅ (`dct`/`idct`/`fwht`/`hilbert`/`goertzel`); `czt`/`dst`/cepstrum 🔵

| Function | Notes |
|---|---|
| `dct(x[, n])` / `idct(y[, n])` | Type-II DCT via N-point FFT trick. |
| `dst` / `idst` | Same trick. |
| `hilbert(x)` | Analytic signal — already half-done by the FFT runtime; just zeros negative frequencies. |
| `cceps(x)` / `rceps(x)` / `icceps(x)` | Complex / real cepstrum. |
| `czt(x, m, w, a)` | Chirp Z-transform — Bluestein on a chirped grid; reuses Bluestein. |
| `goertzel(x, k)` | Single-bin DFT via Goertzel recurrence. |
| `fwht(x[, n, ord])` / `ifwht(...)` | Walsh-Hadamard transform (sequency / Hadamard / dyadic order). |

---

## 4. Tier 3 — measurements, multirate, vibration (~4 weeks)

### 4.1 Multirate ✅ (`upfirdn`/`decimate`/`interp`/`resample`); polyphase + group-delay correction 🔵

| Function | Notes |
|---|---|
| `resample(x, p, q)` / `resample(x, t, fs_new)` | Polyphase FIR upsample-by-p, downsample-by-q. Needs polyphase decomposition of the design FIR. |
| `decimate(x, r)` | Lowpass + downsample. Default Chebyshev IIR order-8. Wraps `cheby1` + `filtfilt`. |
| `interp(x, r)` | Lowpass + upsample-by-r. Hamming-windowed FIR. |
| `upfirdn(x, h, p, q)` | Polyphase resampling kernel — `resample` is built on this. |
| `polyphase(b, m)` | Polyphase decomposition. |

`resample`/`decimate`/`interp` together replace the toy
`upsample`/`downsample` we ship today.

### 4.2 Pulse / waveform generators ✅ (`chirp` linear, `sawtooth`/`square`/`gauspuls`/`rectpuls`/`tripuls`/`sinc`); `pulstran`/`diric`/`gmonopuls`/`vco` + non-linear chirp 🔵

| Function | Notes |
|---|---|
| `chirp(t, f0, t1, f1[, method])` | Linear / quadratic / logarithmic / hyperbolic. |
| `sawtooth(t)`, `square(t[, duty])` | |
| `gauspuls(t, fc, bw)` | Gaussian-modulated sinusoidal pulse. |
| `rectpuls(t[, w])`, `tripuls(t[, w[, skew]])` | |
| `pulstran(t, d, ...)` | Train of arbitrary base pulses. |
| `sinc(x)`, `diric(x, n)` | (sinc already ships as a math identity — verify normalized vs unnormalized convention.) |
| `vco(x, fc, fs)` | Voltage-controlled oscillator. Built on `cumsum`. |

### 4.3 Pulse / waveform measurements ✅ (`findpeaks`, `rms`/`peak2peak`/`peak2rms`/`rssq`, `medfilt1`/`hampel`/`envelope`, `midcross`/`risetime`/`falltime`/`dutycycle`); MinPeak* options + `slewrate`/`pulseperiod`/`pulsewidth`/`overshoot`/`undershoot`/`settlingtime`/`statelevels` 🔵

These power chapters 18 and 24 (Signal Measurement, Common
Applications) — they also feed `findpeaks` workflows.

| Function | Notes |
|---|---|
| `findpeaks(x[, ...])` | Local maxima with `MinPeakHeight`, `MinPeakDistance`, `MinPeakProminence`, `Threshold`, `SortStr`. |
| `peak2peak`, `peak2rms`, `rms`, `rssq` | Pure scalar reductions. |
| `risetime`, `falltime`, `slewrate`, `overshoot`, `undershoot`, `settlingtime`, `dutycycle`, `pulseperiod`, `pulsewidth`, `midcross`, `statelevels` | Standard pulse-characterization statistics. All operate on a single time vector. |
| `envelope(x[, np[, method]])` | Hilbert-based or peak-based envelope. |
| `hampel(x, k)` | Hampel outlier identifier. |
| `medfilt1(x, n)` | 1-D median filter (already partly in mind under image; signal-toolbox version is 1-D). |

### 4.4 Cross-correlation / alignment ✅ (`xcov`/`finddelay`/`dtw`); `alignsignals`/`gccphat` + `xcorr` scaling-options 🔵

| Function | Notes |
|---|---|
| `xcorr(x, y[, maxlag][, scaleopt])` | Existing `xcorr` is 2-arg unscaled; extend to `'biased'` / `'unbiased'` / `'normalized'` / `'coeff'` / `'none'`. |
| `xcov` | Mean-removed cross-correlation. |
| `finddelay(x, y)` | Argmax of `xcorr`. |
| `alignsignals(x, y)` | Pad-or-shift to align peaks. |
| `dtw(x, y)` | Dynamic time warping. |
| `gccphat(x, y, fs)` | Generalized cross-correlation phase transform. |

### 4.5 Linear prediction tail (covered in 3.2) — listed here for completeness.

### 4.6 Vibration analysis (heavy — defer) 🔵

Chapter 19 (`rpmtrack`, `rpmfreqmap`, `rpmordermap`, `modalfit`,
`modalsd`, `tfestimate`-MIMO) is multi-week work tied to the
**Order Tracking** + **Modal Analysis** subsystems. These need:
- a `timetable` runtime (Phase 5.4 in the cross-toolbox roadmap),
- structured outputs (`oti`, `ord`, `magnitude` arrays per channel),
- and the FFT runtime in batch / streaming form.

Carrying as **Tier 4** below; tracked as a follow-on after
Tier-1/2/3 close.

---

## 5. Tier 4 — heavy lifts and system objects

### 5.1 `digitalFilter` system object (`designfilt`) 🔵

`designfilt('lowpassiir', 'PassbandFrequency', ...)` returns a
`digitalFilter` handle that can be passed to `filter`,
`filtfilt`, `freqz`, etc. This is **a `classdef` with operator
overloading**, so it slots into the existing OOP runtime — but
we need:

- A new descriptor type (say `matlab_dfilt` with kind tag) —
  parallel to `matlab_dict`, `matlab_symmat`.
- `filter` / `filtfilt` / `freqz` / `impz` / `stepz` polymorphic
  on argument 1 (matrix-of-coefficients vs `digitalFilter`).
- REPL / DAP rendering: show
  `digitalFilter (lowpass, IIR, n=8, Fc=0.2)` summary +
  expand to `Coefficients`, `SosMatrix`, `ScaleValues` fields.

Effort: ~1 week once Tier-1 IIR/FIR design entries are in.

### 5.2 DSP HDL filter system objects 🔵

Chapters 5 + 26 cover `dsp.SOSFilter`, `dsp.FIRFilter`,
`dsp.LowpassFilter` and the DSP HDL IP Designer flow. These
overlap with the **SystemVerilog backend** already shipped — the
natural integration point is `-emit-systemverilog` consuming a
`digitalFilter` directly and lowering it into the existing
`fir_asic_pipelined` / `cic_decimator` patterns. Track as a
*backend* feature once 5.1 lands.

### 5.3 Apps, GUIs, deep-learning, Python coexecution 🔴

Chapters 4, 5 (Filter Designer GUI), 20 (Signal Analyzer App), 21
(Simulation Data Inspector), 22 (Signal Labeler), 23 (Signal
Feature Extractor), 25 (most of "Featured Examples", esp. all
deep-learning entries), 26 (code generation from MATLAB), and 27
(Python coexecution) are **out of scope**:

- The apps are interactive GUIs (Qt / Web) that aren't part of
  the language runtime.
- Code generation in MATLAB's sense uses MATLAB Coder, not our
  pipeline; our `-emit-c`/`-emit-cpp` already covers the user
  intent for our subset.
- Deep-learning entries call into Deep Learning Toolbox;
  out-of-scope for the same reason as the toolbox carve-out in
  `feature_status.md` §456.

### 5.4 Wavelets / time-frequency tail 🔵

`cwt`, `dwt`, `idwt`, `wavedec`, `waverec`, `wvd`, `fsst`,
`ifsst`, constant-Q Gabor (`cqt` / `icqt`), reassignment /
synchrosqueezing variants. These are 2–3 weeks each — stand-alone
algorithmic work — and should land *after* the Tier-1/2 surface
unblocks day-to-day signal processing.

---

## 6. REPL / Debug-side work (cross-cutting)

The bulk of Signal Processing Toolbox functions return **plain
matrices** (filter coefficients, impulse responses, spectra, time
series), so the existing REPL display + DAP variable inspector
cover them with no per-function work.

The few cases that need explicit REPL/DAP wiring:

| Item | Tier | Action |
|---|---|---|
| `digitalFilter` summary in REPL + DAP variable inspector | 5.1 | Extend `disp` dispatch + DAP `Variables` request to render the filter object; mirror across Python / TS runtimes. |
| `dsp.SOSFilter` / `dsp.FIRFilter` system objects | 5.2 | Same as above — once the descriptor type ships. |
| `spectrogram(x, ...)` with **no LHS** (plot-only call) | 3.3 | Already handled by the no-op-plot convention; verify the no-LHS path doesn't error. |
| `fvtool(b, a)`, `freqz(...)`-no-LHS, `impz(...)`-no-LHS | 2.1, 2.5 | Same no-op-plot path; ensure each new entry routes through it. |
| Function-handle-in-struct ABI for `OutputFcn`-style callbacks (`pwelch(..., 'OutputFcn', @cb)`) | Tier 2 | Already on the cross-toolbox roadmap (`ode.md`); same gate unblocks both. |

`dbg(x)` works for any `matlab_mat *` — no extra work for signal
results. `who` / `whos` already reports size+class for new bindings.
DAP-from-REPL `evaluate` already routes statement-level signal
expressions through `runReplInput`, so an interactive
`>> [b, a] = butter(6, 0.2); [h, w] = freqz(b, a)` session inside
a paused debug frame works as soon as the new builtins ship.

---

## 7. Suggested execution order

A pragmatic order that keeps each landing self-contained and
gates the next on user-visible output:

1. ~~**2.3 Windows tail**~~ ✅ shipped — unblocks every FIR design.
2. ~~**2.4 `roots` / `poly` / `polyder` / `polyint` / `residue`**~~ ✅ shipped — unblocks tf↔zp↔sos. (residue distinct-pole only; repeated-pole grouping is a follow-on.)
3. ~~**2.1 IIR design — lowpass core**~~ ✅ shipped (`butter`, `cheby1`, `freqz`).
4. ~~**2.1 IIR follow-on — cheby2 + buttord + cheb1ord**~~ ✅ shipped. **Still open**: `ellip`, `besself`, analog prototypes, standalone `bilinear`, `freqs`, high/band/stop variants, `cheb2ord`/`ellipord`, form conversions.
5. ~~**2.2 FIR design (lowpass core: `fir1`, `sgolay`, `sgolayfilt`)**~~ ✅ shipped. **Still open**: `fir2`, `firls`, `firpm`, `firrcos`, `kaiserord`.
6. ~~**2.5 `filtfilt`, `sosfilt`, `impz`, `stepz`, `grpdelay`**~~ ✅ shipped — design loop closed. **Still open**: `phasez`, `zerophase`, Gustafsson initial conditions for `filtfilt`.
7. ~~**Tier 2 §3.4 — transforms tail**~~ ✅ shipped (`dct`, `idct`, `fwht`, `hilbert`, `goertzel`). **Open**: czt, dst/idst, cceps/rceps.
8. ~~**Tier 2 §3.1 — nonparametric spectral**~~ ✅ shipped (`periodogram`, `pwelch`, `cpsd`, `mscohere`, `tfestimate`). **Open**: dpss + pmtm.
9. ~~**Tier 2 §3.2 — linear prediction + parametric PSD**~~ ✅ shipped (`levinson`, `lpc`, `aryule`, `arburg`, `pyulear`, `pburg`). **Open**: pcov/pmcov, subspace methods, prony/stmcb.
10. ~~**Tier 2 §3.3 — time-frequency**~~ ✅ shipped (`spectrogram` single-output). **Open**: stft/istft, pspectrum, instfreq/instbw.
11. ~~**Tier 3 §4.3 — pulse measurements core**~~ ✅ shipped (`findpeaks`, `rms`/`peak2peak`/`peak2rms`/`rssq`, `medfilt1`/`hampel`/`envelope`, `midcross`/`risetime`/`falltime`/`dutycycle`). **Open**: MinPeak* options, slewrate/pulseperiod/pulsewidth/overshoot/undershoot/settlingtime/statelevels.
12. ~~**Tier 3 §4.1 — real multirate**~~ ✅ shipped (`upfirdn`, `decimate`, `interp`, `resample`). **Open**: polyphase, group-delay correction.
13. ~~**Tier 3 §4.2 — waveform generators**~~ ✅ shipped (`chirp`, `sawtooth`, `square`, `gauspuls`, `rectpuls`, `tripuls`, `sinc`). **Open**: chirp non-linear methods, pulstran, diric, gmonopuls, vco.
14. ~~**Tier 3 §4.4 — alignment helpers**~~ ✅ shipped (`xcov`, `finddelay`, `dtw`). **Open**: alignsignals, gccphat, xcorr scaling-option strings.

**Remaining open work** (queued in priority order):

15. **§2.1 IIR family completion** — band variants (high/bandpass/stop)
    of `butter`/`cheby1`/`cheby2`, `ellip`, `besself`, analog
    prototypes (`buttap`/`cheb1ap`/`cheb2ap`/`ellipap`/`besselap`),
    standalone `bilinear`/`freqs`, `cheb2ord`/`ellipord`, form
    conversions (`tf2zp`/`zp2tf`/`tf2sos`/`sos2tf`/`tf2ss`/`ss2tf`/`zp2sos`).
    Bandpass peak-normalization is the known-bug entry that needs
    an analytical re-derivation pass before relanding.
16. **§2.2 richer FIR design** — `fir2`, `firls`, `firpm` (Parks-McClellan / Remez), `firrcos`, `kaiserord`.
17. **§2.5 zero-phase tail** — Gustafsson 1996 IC trick for `filtfilt`, `phasez`, `zerophase`.
18. **§3.1 multitaper** — `dpss` (Slepian sequences via tridiagonal eig), `pmtm`.
19. **§3.3 STFT family** — `stft`/`istft` (with COLA inversion), `pspectrum`, `instfreq`, `instbw`.
20. **§3.4 transforms tail** — `czt` (Bluestein on chirped grid), `dst`/`idst`, `cceps`/`rceps`/`icceps`.
21. **§3.2 covariance + subspace** — `pcov`/`pmcov`, `pmusic`/`peig`/`rootmusic`/`rooteig`, `prony`/`stmcb`.
22. **§4.3 pulse measurements tail** — `slewrate`, `pulseperiod`, `pulsewidth`, `overshoot`, `undershoot`, `settlingtime`, `statelevels`; `MinPeakHeight`/`MinPeakDistance`/`MinPeakProminence`/`Threshold`/`SortStr` options for `findpeaks`.
23. **§4.2 waveform tail** — `chirp` quadratic/log/hyperbolic, `pulstran`, `diric`, `gmonopuls`, `vco`.
24. **§4.4 alignment tail** — `alignsignals` (multi-return), `gccphat`, `xcorr` scaling-option strings.
25. **5.1 `designfilt` / `digitalFilter` system object** (~1 week; needs new descriptor type).
26. **5.4 + 4.6** wavelets and vibration tail — open-ended; schedule after 15–25.

Items 1–14 closed the practical day-to-day Signal Processing surface
(filter design, apply, inspect; spectral; LP/parametric PSD; multirate;
generators; pulse measurements; alignment). Items 15–24 round out the
toolbox to per-function MATLAB parity. Items 25–26 are descriptor /
algorithmic heavy lifts.

---

## 8. Test corpus deltas

Each tier adds a uniform set of fixtures under `test/Run/` named
`sig_<area>_<case>.m`, plus a direct unit test in
`test/Runtime/test_signal.c` for any new runtime kernel.

| Tier | New `test/Run/` fixtures (target count) | New `test/Runtime/test_*.c` |
|---|---:|---|
| Tier 1 (filter design + close-the-loop) | ~25 | extend `test/Runtime/test_signal.c` |
| Tier 2 (spectral + transforms + TF) | ~20 | new `test/Runtime/test_spectral.c` |
| Tier 3 (multirate + measurements + alignment) | ~15 | new `test/Runtime/test_pulse.c` |
| Tier 4 (system objects, vibration, wavelets) | ~15 | new `test/Runtime/test_dfilt.c` |

C / C++ / Python / TypeScript lanes must remain byte-identical (the
Python lane may carry `.stdout-python` overrides for numpy repr, as
elsewhere).

---

## 9. Out of scope (carved out, by chapter)

| Chapter | What | Why out of scope |
|---|---|---|
| 4 — Filter Builder GUI | Interactive Qt app | Not a language feature. |
| 5 (App parts) — Filter Analyzer / Designer | Interactive GUI | Same. |
| 14 — Signal Data Set Management | MATLAB-specific data-set abstractions | Tied to `audioDatastore`, `signalDatastore`; not in our runtime. |
| 15 — Experiment Manager Templates | Hyperparameter sweep app | Out of scope. |
| 20 — Signal Analyzer App | Interactive GUI | Out of scope. |
| 21 — Simulation Data Inspector | Simulink GUI | Simulink not in scope. |
| 22 — Signal Labeler | Interactive GUI | Out of scope. |
| 23 — Signal Feature Extractor | Interactive GUI; programmatic `signalFeatureExtractor` is **partially** in scope and would slot into Tier 3 if user demand surfaces. | Defer until requested. |
| 25 (deep-learning entries) | LSTM, GAN, deepSignalAnomalyDetector, … | Deep Learning Toolbox not in scope. |
| 26 — Code Generation from MATLAB | Targets MATLAB Coder | We have our own `-emit-c`/`-emit-cpp` path. |
| 27 — Python Coexecution | MATLAB↔Python bridge | Out of scope; our Python emission is the inverse direction. |
