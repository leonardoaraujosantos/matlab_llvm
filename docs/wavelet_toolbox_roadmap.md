# Wavelet Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Wavelet-Toolbox programs.

Source: *Wavelet Toolbox User's Guide* (R2026a, 17 chapters + appendix:
Wavelets/Scaling Functions/Conjugate Quadrature Mirror Filters ·
Continuous Wavelet Analysis · Discrete Wavelet Analysis ·
Time-Frequency Gallery · Wavelet Packets · Denoising, Nonparametric
Function Estimation, and Compression · Matching Pursuit · Code
Generation · Special Topics · Featured Examples (Time-Frequency ·
Discrete MRA · Denoising/Compression · ML and Deep Learning) · Wavelet
Signal Analyzer · Wavelet Image Analyzer · Wavelet Analyzer Topics ·
Generating MATLAB Code from the App · App Features Summary).

This is a **high-leverage extension of the shipped Signal Processing
Toolbox** — the discrete wavelet transform is, at its core, a
two-channel filter bank (convolve with a quadrature-mirror filter pair,
then downsample by 2), and the project already ships every primitive
that needs: `conv` / `conv2` / `filter` / `upfirdn` / `downsample` /
`fft` / `ifft` / `fft2` / `dct` / `fwht` / `hilbert` / `spectrogram`.
The continuous wavelet transform is an FFT-domain convolution of the
signal with scaled-and-shifted analysing wavelets — again, the shipped
`fft`/`ifft` carry it. The 2-D image tiers reuse the shipped Image
Processing substrate (`im2double` / `mat2gray` / 3-D indexing); the
machine-learning headline reuses the shipped Stats `fitcsvm` /
`pca` / `kmeans`. **No external dependency** (no PyWavelets, no
WaveLab) — every transform is a hand-coded filter-bank / FFT routine
over the shipped kernel, and the wavelet-family coefficients are
hard-coded lookup tables (the same precedent as the 5G-NR base matrices
in Comm and the `fspecial` kernels in Image).

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/wavelet/denoise_signal.m`](../examples/wavelet/denoise_signal.m):
*the canonical Donoho-Johnstone denoising demo — synthesise a noisy
"heavy sine"/Doppler test signal with `wnoise`, decompose it with
`wavedec(x, 5, 'sym4')`, soft-threshold the detail coefficients at the
universal threshold (`wthresh` + `thselect('sqtwolog')` + `wnoisest`),
reconstruct with `waverec`, and report the SNR improvement vs the noisy
input*. This exercises the `wavedec` → coefficient access → threshold →
`waverec` arc end-to-end; achieving it closes **Wave-Tier-1/2** (it is
both the perfect-reconstruction proof and the denoising payoff). The
companion `examples/wavelet/ecg_rwave_modwt.m` (MODWT-based R-wave
detection) is the **Wave-Tier-4** tracer-bullet, and
`examples/wavelet/scalogram_chirp.m` (CWT scalogram of a chirp) is the
**Wave-Tier-3** one.

Companion docs: [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md)
(the DWT/CWT ride the shipped filter / FFT / multirate surface — this
roadmap is its natural extension), [`image_toolbox_roadmap.md`](image_toolbox_roadmap.md)
(the 2-D wavelet + image-denoising tiers reuse `im2double` / `mat2gray`
/ 3-D indexing / `psnr`/`ssim`), [`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md)
(the wavelet-scattering ML headline reuses `fitcsvm` / `pca`),
[`plotting.md`](plotting.md) (scalogram / coefficient plots route
through the Cairo backend), [`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the discrete-wavelet core + the family filter catalogue
  (`wfilters` / `dwt` / `idwt` / `wavedec` / `waverec` / `appcoef` /
  `detcoef` / `wrcoef` / `wmaxlev` / `wextend` — the Mallat fast wavelet
  transform). **Tier-2** is denoising + compression (`wthresh` /
  `thselect` / `wnoisest` / `ddencmp` / `wden` / `wdenoise` / `wcompress`
  / `measerr`) — the killer app. **Tier-3** is the continuous wavelet
  transform + time-frequency (`cwt` / `icwt` / `cwtfilterbank` /
  `scal2frq` / `centfrq` / `wcoherence` + the scalogram). **Tier-4** is
  the undecimated transforms + 2-D (`swt` / `iswt` / `modwt` / `imodwt`
  / `modwtmra` / `modwtvar` / `modwtcorr`; `dwt2` / `idwt2` / `wavedec2`
  / `waverec2` / `wcodemat` + image denoising/compression). **Tier-5**
  is wavelet packets (`wpdec` / `wprec` / `wpcoef` / `besttree` /
  `bestlevt` / `wenergy` + the WPTREE object). **Tier-6** is the
  Special-Topics + ML surface (`ewt` / `vmd` / `emd` / `tqwt`,
  `waveletScattering`, `matchingPursuit` / `sensingDictionary`,
  `wmspca`) plus carve-down polish.
- **Effort** is in the existing Phase 5.6.x cadence (one focused session
  ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~1.5 wk · T2
  ~1 wk · T3 ~2 wk · T4 ~1.5 wk · T5 ~1.5 wk · T6 ~3 wk (~10.5 wk
  full)**. Each tier is independently shippable and demoable; **T1 + T2
  alone (~2.5 wk) close the 80% denoising/MRA workflow** — the single
  most common reason anyone reaches for this toolbox.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Everything below is 🔵 not started** — clean slate. There is no
  `dwt` / `cwt` / `wavedec` / `wthresh` / `modwt` / `wpdec` in the
  runtime today; the deep shipped Signal base (`conv`, `filter`,
  `upfirdn`, `downsample`, `fft`) is what makes the DWT/CWT cheap.
- **Wavelet families are hard-coded filter tables**: `wfilters('db4')`
  returns the four decomposition/reconstruction filters from a baked-in
  coefficient catalogue (Haar / `db1`–`db10` / `sym2`–`sym8` /
  `coif1`–`coif5` / `bior` pairs / `dmey`). This is the **exact**
  lookup-table precedent of the Comm 5G-NR base matrices and the Image
  `fspecial` kernels — caller-supplied family string → table fetch in
  `runtime_wavelet.cpp`. Continuous families (`morse` / `amor` Morlet /
  `bump`) are generated analytically in the frequency domain.
- **Decomposition structures are `[C, L]` matrices, not opaque objects**
  (for the 1-D/2-D decimated transforms): `wavedec` returns the
  concatenated coefficient vector `C` + the bookkeeping length vector
  `L`, exactly as MATLAB does — so `appcoef`/`detcoef`/`waverec` are
  plain matrix-in/matrix-out builtins with no classdef needed. Only the
  **wavelet-packet tree** (Tier-5, `WPTREE`) and `cwtfilterbank`
  (Tier-3) need the classdef descriptor (the alloc-then-populate +
  class-pinned-dispatch pattern proven by `tf`/`ss`/`LinearModel`),
  auto-prepended via `wavelet_classdefs.m`. This keeps Tiers 1–2 (the
  headline) entirely in the matrix lane.
- **No external dependencies**: matching the project precedent — DWT via
  the shipped `conv` + a hand-coded dyadic `downsample`/`upsample`; CWT
  via the shipped `fft`/`ifft`; thresholding + threshold-selection
  hand-coded; the family-filter generators (cascade algorithm for
  `wavefun`) hand-coded over `conv`.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Wavelet code yet)

| Group | Surface (already shipped) | Location | How Wavelet uses it |
|---|---|---|---|
| Convolution / filtering | `conv`, `conv2`, `filter`, `filtfilt` | `lib/Sema/Resolver.cpp` → `matlab_conv` / `matlab_filter` (`runtime/matlab_runtime.cpp`) | The two-channel filter bank — every `dwt`/`idwt`/`wavedec` level is a `conv` + dyadic decimate (Tier-1); 2-D via `conv2` (Tier-4). |
| Multirate | `upfirdn`, `decimate`, `interp`, `resample`, `upsample`, `downsample` | `lib/Sema/Resolver.cpp` (Signal Tier-3) | Dyadic `dyaddown`/`dyadup` (decimate/interpolate by 2) inside the FWT; à-trous (no-decimate) SWT/MODWT (Tier-1/4). |
| FFT family | `fft`, `ifft`, `fft2`, `ifft2` | `runtime/matlab_runtime.cpp` | The FFT-domain CWT (signal × scaled analysing wavelet) (Tier-3); fast convolution for long filters; `modwt` frequency-domain path (Tier-4). |
| Transforms | `dct`/`idct`, `fwht`, `hilbert`, `goertzel`, `spectrogram` | `runtime/matlab_runtime.cpp` (Signal Tier-2) | Sibling transforms for the Time-Frequency Gallery comparison (Tier-3); `hilbert` for analytic-signal CWT validation; `spectrogram` as the STFT baseline. |
| Spectral / parametric | `pwelch`, `periodogram`, `cpsd`, `tfestimate`, `mscohere` | `runtime/matlab_runtime.cpp` (Signal Tier-2) | `wcoherence` baseline + cross-spectral validation (Tier-3); `modwtcorr`/`modwtxcorr` lean on the same covariance machinery (Tier-4). |
| Smoothing | `sgolayfilt`, `medfilt1`, `findpeaks` | `runtime/matlab_runtime.cpp` (Signal) | Comparison baselines for denoising (Tier-2); `findpeaks` for MODWT R-wave detection (Tier-4 ECG headline). |
| Dense linear algebra | `mldivide`, `qr`, `svd`, `pinv`, `eig` | `runtime/matlab_runtime.cpp` | Lifting-scheme polyphase solves (Tier-1 stretch); `vmd` ADMM updates + multiscale-PCA (`pca` via `svd`/`eig`) (Tier-6). |
| Stats / ML | `pca`, `kmeans`, `fitcsvm`, `predict`, `std`/`var`/`median` | `runtime/toolbox/stats/runtime_stats.cpp` | `wnoisest` (robust σ = MAD/0.6745 via `median`); multiscale PCA `wmspca` (Tier-6); the wavelet-scattering → `fitcsvm` ML headline (Tier-6). |
| Image substrate | `im2double`, `mat2gray`, `imread`, `psnr`/`ssim`, 3-D indexing `A(:,:,k)`, `wcodemat`-style scaling | `runtime/toolbox/images/runtime_images.cpp` | 2-D wavelet decomposition display + image denoising/compression quality metrics (Tier-4). |
| Function-handle ABI | `void *fn_p`, `LowerAnonCalls` retyping | `runtime/toolbox/optim/runtime_optim.cpp` | Custom analysing-function CWT + `matchingPursuit` dictionary handles (Tier-3/6). |
| Classdef plumbing | `matlab_obj_new`/`_set_*`/`_get_mat`, kwarg-ctor sugar, class-pinned dispatch, REPL persist, DAP render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The `WPTREE` packet-tree + `cwtfilterbank` + `waveletScattering` descriptors (Tier-3/5/6). |
| Plotting | Cairo `plot` / `imagesc` / `surf` / `contourf` / `stem` | `runtime/plot/` | Scalogram (Tier-3), coefficient/detail plots (Tier-1), MODWTMRA panels (Tier-4), packet-tree coefficient maps (Tier-5). |

**Net assessment**: the *transform substrate* (convolution, dyadic
multirate, FFT, plotting, the Stats/Image companions) is **already
shipped**. The genuinely new code is (a) the **family-filter catalogue +
`wfilters`/`wavefun`** (lookup tables + the cascade generator), (b) the
**FWT engine** (`dwt`/`wavedec`/`waverec` + `[C,L]` bookkeeping +
`wextend` border handling), (c) the **denoising layer** (`wthresh` +
`thselect` + `wnoisest` + `wdenoise`), (d) the **CWT engine** (FFT-domain
filter bank + scalogram + `scal2frq`), (e) the **undecimated transforms**
(`swt`/`modwt`/`modwtmra`), and (f) the **packet tree** + special-topics
(`ewt`/`vmd`/scattering). Each is a self-contained hand-coded routine
over the shipped base — the heavy numeric lifting (convolution, FFT) is
done.

---

## 2. Tier-1 — Discrete wavelet core + family filters (the FWT) 🔵

Goal: the Mallat fast wavelet transform — decompose, access coefficients,
reconstruct exactly. The perfect-reconstruction backbone everything else
builds on.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `wfilters(wname)` | Family filter catalogue → `[Lo_D, Hi_D, Lo_R, Hi_R]`. Tables for `'haar'`/`'db1'`–`'db10'`, `'sym2'`–`'sym8'`, `'coif1'`–`'coif5'`, `'bior*'`/`'rbio*'`, `'dmey'`. QMF relation `Hi = qmf(Lo)`. | lookup table, `qmf` |
| 1.2 | `dwt` / `idwt` | Single-level: `cA = downsample(conv(x, Lo_D), 2)`, `cD = downsample(conv(x, Hi_D), 2)`; inverse upsamples + convolves with `Lo_R`/`Hi_R`. Border modes via `wextend`. | `conv`, `downsample`, `upsample` |
| 1.3 | `wavedec` / `waverec` | Multi-level 1-D decomposition → `[C, L]` (concatenated coeffs + bookkeeping lengths); cascade `dwt` on the approximation. `waverec` is exact (perfect reconstruction). | 1.2 |
| 1.4 | `appcoef` / `detcoef` | Extract approximation / level-`k` detail coefficients from `[C, L]`. `detcoef(C, L, 'cells')` multi-level. | `[C,L]` slicing |
| 1.5 | `wrcoef` / `upcoef` / `upwlev` | Reconstruct a single branch (approx or detail at one level) back to signal length; level merge. | 1.3 |
| 1.6 | `wextend` / `wkeep` | Border extension (`'sym'` symmetric / `'zpd'` zero-pad / `'per'` periodic / `'sp0'`/`'spd'`) + center-crop. The conditioning piece for finite-length signals. | matrix pad/slice |
| 1.7 | `wmaxlev` / `dwtmode` | Max useful decomposition level for a signal length + filter; global border-mode setting. | — |
| 1.8 | `wavefun` / `waveinfo` / `centfrq` | Cascade-algorithm reconstruction of the scaling/wavelet function pair (for plotting); family info string; center frequency. | `conv` |
| 1.9 | `wentropy` / `wenergy` | Shannon / log-energy / threshold / norm entropy of coefficients; per-level energy distribution. | reductions |
| 1.10 | display | `plot` the approximation + per-level details (the canonical MRA stack). | `runtime/plot/` |

**Headline-within-tier**: the perfect-reconstruction proof —
`[C,L]=wavedec(x,5,'db4'); xr=waverec(C,L,'db4'); max(abs(x-xr))` ≈ 0.

**Compile/Execute wiring**: new `runtime/toolbox/wavelet/runtime_wavelet.cpp`;
register `dwt`/`idwt`/`wavedec`/`waverec`/`appcoef`/`detcoef`/`wrcoef`/
`wfilters`/`wmaxlev`/`wextend`/`wentropy` in `Resolver.cpp`; `pde_table`
loose-match in `LowerTensorOps.cpp` with the string family-name arg →
`matlab_string*` (the Image `imread('f.png')` path); `[C,L]` and
`[cA,cD]` are 2-result builtins via the existing multi-output splitter.

---

## 3. Tier-2 — Denoising + nonparametric estimation + compression 🔵

Goal: the killer app — wavelet shrinkage denoising. Closes the headline.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `wthresh` | Soft (`sign(x)·max(|x|−T,0)`) and hard (`x·(|x|>T)`) thresholding. | element-wise |
| 2.2 | `thselect` | Threshold selection rules: `'sqtwolog'` (universal `σ√(2log n)`), `'rigrsure'` (SURE), `'heursure'`, `'minimaxi'`. | reductions |
| 2.3 | `wnoisest` / `wnoise` | Robust noise σ from finest-level details (MAD `median(|d|)/0.6745`); `wnoise` test-signal generator (heavy sine / Doppler / blocks / bumps). | Stats `median` |
| 2.4 | `ddencmp` | Default denoising/compression parameters (threshold + rule + scaling). | 2.2/2.3 |
| 2.5 | `wden` / `wdencmp` | Legacy automatic 1-D denoising over `[C,L]`; level-dependent + global thresholding (`'sln'`/`'mln'` noise scaling). | 2.1–2.4, Tier-1 |
| 2.6 | `wdenoise` | Modern high-level API: `wdenoise(x, level, 'Wavelet','sym4','DenoisingMethod','UniversalThreshold','ThresholdRule','Soft')`. | `wavedec`/`waverec` |
| 2.7 | `cmddenoise` / interval-dependent | Piecewise (interval-dependent) thresholds. | 2.5 |
| 2.8 | `wcompress` / `wpdencmp` | Coefficient-thresholding compression with a retained-energy / N-coeff target; compression ratio + quality report. | Tier-1 |
| 2.9 | `measerr` | Quality metrics — MSE / `psnr` / `maxerr` / L2-recovery ratio. | Image `psnr` |

**Headline-within-tier**: the denoising demo —
`wdenoise` a noisy heavy-sine, SNR improvement reported, overlay clean vs
noisy vs denoised.

**Compile/Execute wiring**: all matrix-in/matrix-out builtins layered on
the Tier-1 `[C,L]` engine; `wdenoise` name/value options read in the
runtime (the Image `fspecial`-option-string path); `measerr` reuses the
shipped Image `psnr`/`immse`.

---

## 4. Tier-3 — Continuous wavelet transform + time-frequency 🔵

Goal: the CWT + scalogram — the second pillar (analysis, not denoising).

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `cwt` / `icwt` | FFT-domain CWT: `W(a,b) = ifft( fft(x) · conj(Ψ̂(aω)) )` over a log-spaced scale/frequency set; analytic Morse (`'amor'`/`'morse'`/`'bump'`) wavelets generated in frequency. `icwt` inverse via the synthesis sum. | `fft`/`ifft` |
| 3.2 | `cwtfilterbank` | Classdef carrying the precomputed analysis filter bank (wavelet, signal length, `VoicesPerOctave`, frequency limits); `wt`/`scaleSpectrum`/`centerFrequencies` methods. | classdef |
| 3.3 | scalogram | `cwt(...)` no-output → Cairo `imagesc`/`contourf` of `|W|` over time × frequency; cone of influence overlay. | `runtime/plot/` |
| 3.4 | `scal2frq` / `centfrq` / `freq2scal` | Scale ↔ pseudo-frequency conversion via the wavelet center frequency. | — |
| 3.5 | `wcoherence` | Wavelet coherence between two signals (smoothed cross-scalogram); magnitude + phase arrows. | 3.1, smoothing |
| 3.6 | `wsst` / `iwsst` | Wavelet synchrosqueezed transform (reassign CWT energy by instantaneous frequency) + inverse; `wsstridge` ridge extraction. *(stretch within tier)* | 3.1 |
| 3.7 | Time-Frequency Gallery | Side-by-side `spectrogram` (STFT) vs `cwt` (scalogram) vs `wvd` Wigner-Ville comparison helper. | Signal `spectrogram` |

**Headline-within-tier**: the scalogram of a quadratic chirp —
`cwt(chirp, fs)` time-frequency map showing the swept ridge.

**Compile/Execute wiring**: `cwt` is a multi-return builtin
(`[wt, f, coi] = cwt(...)`); the analysing-wavelet generators live in
`runtime_wavelet.cpp`; `cwtfilterbank` is a classdef (prelude-triggered);
scalogram plotting reuses the Image-era `imagesc`/3-D display path.

---

## 5. Tier-4 — Undecimated transforms (SWT / MODWT) + 2-D 🔵

Goal: the shift-invariant transforms (better for denoising/detection) +
the image half.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | `swt` / `iswt` | Stationary (undecimated) wavelet transform — à-trous algorithm: no downsampling, upsample the filters by 2 each level. | `conv`, dyadic `dyadup` |
| 4.2 | `modwt` / `imodwt` | Maximal-overlap DWT (energy-preserving, shift-invariant); rescaled filters `/√2` per level; FFT-circular-convolution path. | `fft`/`ifft` |
| 4.3 | `modwtmra` | Multiresolution analysis — additive zero-phase detail components that sum to the signal. | 4.2 |
| 4.4 | `modwtvar` / `modwtcorr` / `modwtxcorr` | Scale-localized wavelet variance / correlation / cross-correlation with CIs. | 4.2, covariance |
| 4.5 | `dwt2` / `idwt2` | 2-D single-level — separable row-then-column filter bank → `[cA, cH, cV, cD]`. | `conv2` |
| 4.6 | `wavedec2` / `waverec2` | Multi-level 2-D decomposition → `[C, S]` (coeffs + size bookkeeping). | 4.5 |
| 4.7 | `appcoef2` / `detcoef2` / `wrcoef2` | 2-D coefficient extraction (H/V/D detail planes) + branch reconstruction. | `[C,S]` slicing |
| 4.8 | `wcodemat` / image denoise/compress | Coefficient-matrix scaling for display; 2-D `wdenoise2` / `wcompress` on images. | Image substrate |

**Headline-within-tier**: the ECG R-wave demo —
`modwtmra(ecg, 'sym4')`, isolate the QRS-energy scale, `findpeaks` the
R-waves; the UG §3 "R Wave Detection in the ECG" pipeline.

**Compile/Execute wiring**: SWT/MODWT are matrix builtins (multi-level
output as a level×N matrix); 2-D reuses the shipped `conv2` + the
Image-era 3-D coefficient stacking; `findpeaks` (Signal) closes the ECG
headline.

---

## 6. Tier-5 — Wavelet packets 🔵

Goal: the full binary-tree decomposition (both approximation *and* detail
branches split) + best-basis selection.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | `wpdec` / `wprec` | 1-D wavelet packet decomposition/reconstruction to a full tree of depth `n`. | Tier-1 `dwt` |
| 5.2 | `WPTREE` object | Classdef descriptor carrying the tree (node coeffs + structure); `read`/`write`/`get` node access. The alloc-then-populate + class-pinned-dispatch pattern. | classdef |
| 5.3 | `wpcoef` / `wprcoef` | Read packet coefficients at a node; reconstruct a single node back to signal length. | 5.1/5.2 |
| 5.4 | `besttree` / `bestlevt` | Best-basis / best-level selection by an entropy criterion (`wentropy`); prune the tree. | 5.2, `wentropy` |
| 5.5 | `wpsplt` / `wpjoin` | Split / merge tree nodes (interactive-equivalent editing). | 5.2 |
| 5.6 | `wpdec2` / `wprec2` | 2-D wavelet packets (image). | `dwt2` |
| 5.7 | `wpviewcf` / `wenergy(wpt)` | Packet-coefficient colour map; per-node energy. | `runtime/plot/` |
| 5.8 | `wpdencmp` | Packet-based denoising/compression (best-tree + threshold). | 5.4, Tier-2 |

**Headline-within-tier**: harmonic-interference removal —
`wpdec` an audio signal, `besttree`, threshold the interference node,
`wprec`; the UG "Wavelet Packet Harmonic Interference Removal" demo.

**Compile/Execute wiring**: `WPTREE` is the one classdef this tier needs
(prelude-triggered, REPL/DAP-rendered); the tree itself is stored as a
coefficient matrix + a structure vector inside the object;
`besttree` runs the entropy prune in the runtime.

---

## 7. Tier-6 — Special topics + ML + carve-down polish 🔵

Goal: the modern data-adaptive transforms + the machine-learning bridge
+ the remaining polish.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | `ewt` | Empirical Wavelet Transform — detect spectral maxima, build adaptive band-pass filters, decompose. | `fft`, `findpeaks` |
| 6.2 | `vmd` | Variational Mode Decomposition — ADMM iteration recovering narrow-band modes; `[imf, residual] = vmd(x)`. | `fft`, `mldivide` |
| 6.3 | `emd` / `hht` | Empirical Mode Decomposition (sifting) + Hilbert-Huang spectrum. | `hilbert`, spline |
| 6.4 | `tqwt` / `itqwt` | Tunable Q-factor wavelet transform (frequency-domain over/undersampled bank). | `fft`/`ifft` |
| 6.5 | `waveletScattering` | Scattering transform classdef → invariant feature matrix (cascade of `|CWT|` + lowpass); `featureMatrix`/`scatteringTransform`. | Tier-3 CWT |
| 6.6 | scattering → ML headline | `waveletScattering` features → `fitcsvm` classification (the UG "Signal Classification Using Wavelet-Based Features and SVM"). | Stats `fitcsvm` |
| 6.7 | `matchingPursuit` / `sensingDictionary` | (Orthogonal) matching pursuit over a redundant dictionary; greedy atom selection. | `mldivide`, dictionaries |
| 6.8 | `wmspca` / multivariate denoise | Multiscale PCA on the wavelet coefficients; multivariate wavelet denoising. | Stats `pca` |
| 6.9 | lifting (carve-down) | `lwt`/`ilwt` lifting-scheme transform, `liftwave`/`liftingScheme` (custom-wavelet authoring). *(stretch)* | Tier-1 |

**Headline-within-tier**: the scattering-SVM classifier — extract
`waveletScattering` features from labelled signals, train `fitcsvm`,
report accuracy (the cross-toolbox tracer-bullet: Wavelet → Stats).

**Carve-down polish (cross-tier follow-ons)**: `dualtree`/`dualtree2`
(dual-tree complex wavelet), `shearletSystem`, 3-D DWT (`wavedec3`),
`modwtcorr` table output, `cwtfreqbounds`, multifractal `dwtleader`.

---

## 8. Compile/Execute · Debug/REPL · Examples · Tests (cross-cutting)

The four delivery surfaces the project always closes per toolbox — each
tier ships across **all four** before it counts as done.

### 8.1 Compile / Execute

- **Backends**: LLVM JIT + native + `-emit-c` / `-emit-cpp` are the
  primary lanes (matching Signal/Stats — the transforms are pure numeric
  matrix code, the friendliest possible shape for C/C++).
  `-emit-python` / `-emit-typescript` parity is a per-tier stretch (the
  `[C,L]` matrix transforms port cleanly; the classdef tiers are
  rougher). `-emit-systemverilog` is **not** a target for the analysis
  surface (host-side), though a fixed-point streaming DWT filter bank is
  a natural future bridge to the shipped HDL lane — emit a clear
  diagnostic for now.
- **Runtime**: `runtime/toolbox/wavelet/runtime_wavelet.cpp` (filter
  catalogue, FWT engine, denoising, CWT, SWT/MODWT, packets, special
  topics) + `runtime/toolbox/wavelet/wavelet_classdefs.m`
  (`cwtfilterbank` / `WPTREE` / `waveletScattering`). Add to the strict
  no-C-cast list (`static_cast` throughout), mirroring
  `runtime_images.cpp`.
- **Wiring**: builtin names in `Resolver.cpp`; `pde_table` loose-match +
  string family/option arg → `matlab_string*` in `LowerTensorOps.cpp`;
  the `[C,L]` / `[cA,cD]` / `[wt,f,coi]` multi-returns via the existing
  splitter; `cwtfilterbank`/`WPTREE`/`waveletScattering` registered as
  class-returning so `pinnedOfRhs` propagates the pin, with their methods
  (`wt`, `besttree`, `featureMatrix`) as class-pinned-first-arg dispatch
  in `Lowering.cpp::CallOrIndex` (the CST `pole(sys)` route); prelude
  trigger set so a `dwt`-only program pays no classdef cost.

### 8.2 Debug / REPL

- The `[C,L]` decomposition lives in plain matrix workspace slots — fully
  visible in the REPL + DAP variable inspector with no extra work.
- `cwtfilterbank` / `WPTREE` / `waveletScattering` persist across REPL
  inputs (class-tagged slot) and render in the DAP variable view (filter
  count / tree depth / scattering paths), via the shipped
  `runtime_debug.cpp` classdef-render path.
- `disp(wpt)` formats the tree structure; scalograms/MRA stacks are
  written to PNG/SVG by the Cairo backend so a headless `cwt(x)` produces
  an inspectable artifact.

### 8.3 Examples (`examples/wavelet/`)

| Example | Closes | Exercises |
|---|---|---|
| `denoise_signal.m` | **T1/2 headline** | `wnoise` → `wavedec` → `wthresh`+`thselect`+`wnoisest` → `waverec` → SNR gain |
| `mra_stack.m` | T1 | `wavedec(x,5,'db4')` → `appcoef`/`detcoef` → plot the MRA stack; perfect-reconstruction check |
| `wdenoise_compare.m` | T2 | `wdenoise` rules (universal/SURE/minimax × soft/hard); `measerr` |
| `scalogram_chirp.m` | **T3 tracer** | `cwt` scalogram of a chirp + `scal2frq` + cone of influence |
| `wcoherence_pair.m` | T3 | `wcoherence` of two coupled oscillations |
| `ecg_rwave_modwt.m` | **T4 tracer** | `modwtmra` → QRS scale → `findpeaks` R-waves |
| `image_denoise2.m` | T4 | `wavedec2` → 2-D threshold → `waverec2`; `psnr` on a noisy image |
| `packet_bestbasis.m` | T5 | `wpdec` → `besttree` → `wpcoef`/`wprec`; node-energy map |
| `scattering_svm.m` | T6 | `waveletScattering` features → `fitcsvm` → accuracy |

### 8.4 Tests (`test/Run/`)

Gating tests follow the `wavelet_*.m` convention with a `.stdout` golden
+ per-backend `.skip-emit-*` files where a lane is out of scope (SV
always skipped; Python/TS skipped where the classdef path is rough,
matching the Image `image_png_roundtrip` precedent).

| Test | Tier | Asserts |
|---|---|---|
| `wavelet_dwt.m` | T1 | `dwt`/`idwt` single-level round-trip; `wfilters('db4')` coeff sums |
| `wavelet_wavedec.m` | T1 | `wavedec`/`waverec` perfect reconstruction (`max|x−xr|<1e-10`) |
| `wavelet_thresh.m` | T2 | `wthresh` soft/hard; `thselect('sqtwolog')`; `wnoisest` MAD |
| `wavelet_denoise.m` | T2 | `wdenoise` recovers a known signal; SNR improves |
| `wavelet_cwt.m` | T3 | `cwt` scale set + `scal2frq`; energy at a known frequency |
| `wavelet_modwt.m` | T4 | `modwt`/`imodwt` round-trip; `modwtmra` sums to signal |
| `wavelet_dwt2.m` | T4 | `wavedec2`/`waverec2` image round-trip |
| `wavelet_packet.m` | T5 | `wpdec`/`wprec` round-trip; `besttree` entropy prune |
| `wavelet_entropy.m` | T1 | `wentropy`/`wenergy` on a known coefficient vector |
| `wavelet_scatter.m` | T6 | `waveletScattering` feature-matrix shape + `fitcsvm` accuracy |

Target: **~10 gating tests** (one per major surface), in line with
Image (10) and Stats (12). Full regression must stay green
(currently 465 run-tests) — the badge bumps to **17 toolboxes** (or
**18** if Curve Fitting lands first) and the run-tests count grows by
the new gating set.

---

## 9. Carve-outs (explicitly out of scope)

Matching the per-toolbox precedent — the GUI / app / codegen-UI / deep
learning surfaces are deferred:

- **Wavelet Signal Analyzer** / **Wavelet Image Analyzer** / **Wavelet
  Analyzer** apps (Chapters 14–16) and the **Generate-MATLAB-Code-from-app**
  path (Chapter 17) — interactive GUIs; the project is headless and the
  programmatic API is the whole target. The legacy `wavemenu`/`splinetool`-style
  tools are N/A.
- **Code Generation** (Chapter 8 — MATLAB Coder C / **CUDA** / GPU
  scalograms) beyond the existing `-emit-*` lanes.
- **The Machine-Learning *and Deep Learning* featured examples** (Chapter
  13) that require the Deep Learning Toolbox (CNN/LSTM/autoencoder/DAG
  classifiers, TensorFlow-Lite/Jetson/FPGA deploy) — the **shallow-ML**
  scattering→`fitcsvm` path (6.6) *is* in scope; the deep-net variants
  are not (gated on a future Deep Learning toolbox).
- **Simulink** wavelet blocks and the Simulink code-gen demos.
- **Dual-tree complex wavelets** / **shearlet systems** / **3-D DWT** /
  **joint time-frequency scattering** — Tier-6 stretch follow-ons, not
  the core surface.
- **Lifting-scheme custom-wavelet authoring** (`liftingScheme`/`lwt`)
  beyond the built-in families — a documented Tier-6 stretch (6.9).

These are documented follow-ons, not blockers: every numeric transform a
*script* uses (DWT/CWT/SWT/MODWT/packets/denoising) is in Tiers 1–6.

---

## 10. Effort summary

| Tier | Scope | Effort | Net-new code | Status |
|---|---|---|---|---|
| T1 | DWT core + family filters (FWT) | ~1.5 wk | filter catalogue + `wavedec`/`waverec` + `[C,L]` + `wextend` | 🔵 |
| T2 | denoising + compression | ~1 wk | `wthresh`/`thselect`/`wnoisest`/`wdenoise`/`measerr` | 🔵 |
| T3 | CWT + scalogram + time-frequency | ~2 wk | FFT-domain CWT + analysing wavelets + `cwtfilterbank` + `wcoherence` | 🔵 |
| T4 | SWT/MODWT + 2-D | ~1.5 wk | à-trous + MODWT + `dwt2`/`wavedec2` | 🔵 |
| T5 | wavelet packets | ~1.5 wk | full-tree decomp + `WPTREE` + `besttree` | 🔵 |
| T6 | special topics + ML + polish | ~3 wk | `ewt`/`vmd`/`tqwt` + scattering + matching pursuit | 🔵 |
| **Total** | | **~10.5 wk** | | |

**Recommended slice order**: T1 → T2 closes the everyday 80% workflow
(~2.5 wk) and is the highest-ROI cut — at that point `wavedec` /
`waverec` / `wthresh` / `wdenoise` cover what most users mean by
"wavelets" (multiresolution analysis + shrinkage denoising). T3 (CWT
scalogram) is the next most-requested (time-frequency analysis); T4
(MODWT) unlocks the shift-invariant detection demos (ECG); T5 (packets)
and T6 (scattering/EWT/VMD) are independent advanced add-ons. Strong
synergy with the shipped Signal Processing Toolbox — this is its natural
extension, not a greenfield toolbox.
