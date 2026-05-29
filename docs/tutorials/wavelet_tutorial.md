# Wavelet Toolbox — Tutorial

A hand-coded Wavelet-Toolbox subset over the project's `conv` and `fft` — no external dependency. Orthonormal circular DWT gives exact perfect reconstruction for the `haar`/`dbN`/`symN`/`coifN` families. It covers the DWT/multiresolution, denoising, CWT scalograms, MODWT, wavelet packets, and a shallow-ML scattering→SVM path.

## Supported features

- **DWT & multiresolution:** `wavedec` / `waverec`, `appcoef` / `detcoef`, `wenergy`, `dwt` / `idwt`, `wavedec2` / `waverec2` (2-D).
- **Denoising:** `wnoise` (test signals), `wthresh`, `thselect`, `wnoisest`, `wden`, `wdenoise`, `measerr` (PSNR).
- **CWT & time-frequency:** `cwt` / `icwt` (FFT-domain Morlet), `wcoherence`, `modwt` / `modwtmra` (shift-invariant).
- **Wavelet packets:** `wpdec` / `wprec`, node-energy maps via `wenergy`, best-basis selection.
- **Shallow ML:** `waveletScattering` features feeding `fitcsvm` (Statistics Toolbox).

## Build & run

```bash
build/matlabc -emit-llvm examples/wavelet/denoise_signal.m > /tmp/denoise_signal.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/denoise_signal.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/denoise_signal
/tmp/denoise_signal
```

Swap `denoise_signal` for any other file under `examples/wavelet/`.

## Worked examples

### Donoho-Johnstone denoising — HEADLINE  (`examples/wavelet/denoise_signal.m`)

Synthesise a noisy "heavy sine" with `wnoise`, decompose with `wavedec`, estimate the noise level, and reconstruct after wavelet shrinkage with `wdenoise`. The SNR improvement is the payoff.

```matlab
xclean = wnoise(3, 11);              % heavy sine, length 2048
xn     = xclean + noise;

% 5-level sym4 decomposition, then automatic shrinkage denoising
[C, L] = wavedec(xn, 5, 'sym4');
sigma  = wnoisest(C, L, 1);
thr    = sigma * thselect(detcoef(C, L, 1), 'sqtwolog');
fprintf('estimated noise sigma = %.4f\n', sigma);
fprintf('universal threshold   = %.4f\n', thr);

xd = wdenoise(xn, 6, 'sym4');
snr_out = 20*log10(norm(xclean)/norm(xd - xclean));
fprintf('SNR improvement = %.2f dB\n', snr_out - snr_in);
```

`wnoisest` gives the MAD-estimated noise level from the finest detail band, `thselect('sqtwolog')` is the universal soft threshold, and `wdenoise` performs the full shrinkage-and-reconstruct in one call.

### Multiresolution analysis + perfect reconstruction  (`examples/wavelet/mra_stack.m`)

Decompose a two-tone signal to 5 levels with `db4`, pull out approximation and detail bands, and prove exact reconstruction.

```matlab
[C, L] = wavedec(x, 5, 'db4');
a5 = appcoef(C, L, 'db4', 5);
d1 = detcoef(C, L, 1);
e  = wenergy(C, L);
fprintf('approx energy %% = %.1f\n', e(1));

xr = waverec(C, L, 'db4');
fprintf('perfect reconstruction error = %.2e\n', max(abs(x - xr)));
```

The orthonormal circular DWT reconstructs to machine precision (error ≈ 1e-15).

### CWT scalogram of a chirp  (`examples/wavelet/scalogram_chirp.m`)

The continuous wavelet transform as an FFT-domain convolution with scaled Morlet wavelets; the magnitude shows the swept-frequency ridge.

```matlab
x = sin(2*pi*(20*t + 90*t.^2));      % chirp sweeping 20 -> 200 Hz
[wt, f] = cwt(x, fs);
mag = abs(wt);
[~, is] = max(mag(:, 64));
fprintf('ridge freq at t=0.06s : %.0f Hz\n', round(f(is)));
xr = icwt(wt);                       % inverse CWT
```

`cwt` returns both the coefficient matrix and the scale-to-frequency vector `f`; `icwt` recovers the signal shape.

### MODWT R-wave detection  (`examples/wavelet/ecg_rwave_modwt.m`)

The shift-invariant MODWT isolates the QRS-energy scale, and the R-waves are the peaks of the reconstructed detail band.

```matlab
w   = modwt(ecg, 'sym4', 5);
mra = modwtmra(w, 'sym4');
qrs = mra(3, :) + mra(4, :);         % QRS energy in detail levels 3-4
fprintf('mra reconstructs signal: %.2e\n', max(abs(ecg - sum(mra,1))));
pk = findpeaks(qrs);
fprintf('R-wave candidates found = %.0f\n', length(pk));
```

`modwtmra` produces an additive multiresolution analysis whose rows sum back to the original signal.

### Other examples (briefly)

- `wdenoise_compare.m` — `wden` with `'sqtwolog'` vs `'rigrsure'` threshold rules, scored with `measerr` (PSNR).
- `image_denoise2.m` — 2-D `wavedec2` → `wthresh` detail thresholding → `waverec2` image denoising.
- `packet_bestbasis.m` — `wpdec` full packet tree, locate the interference node from the energy map, zero it, and `wprec` to remove narrow-band interference.
- `wcoherence_pair.m` — `wcoherence` between two signals sharing a common oscillation.
- `scattering_svm.m` — `waveletScattering` features + `fitcsvm` for translation-invariant signal classification.

## Limitations & carve-outs

- **Wavelet Analyzer apps** (Signal/Image Analyzer) and generate-code-from-app — the programmatic API is the whole target; the project is headless.
- **MATLAB-Coder / CUDA / GPU codegen** beyond the existing `-emit-*` lanes.
- **Deep-learning featured examples** (CNN/LSTM/autoencoder classifiers) — the shallow `waveletScattering`→`fitcsvm` path is in scope; deep-net variants are not.
- **Simulink** wavelet blocks.
- **Dual-tree complex wavelets / shearlets / 3-D DWT / joint time-frequency scattering** and **lifting-scheme custom-wavelet authoring** (`liftingScheme`/`lwt`) — Tier-6 stretch follow-ons.

## See also

- Roadmap: [`wavelet_toolbox_roadmap.md`](../wavelet_toolbox_roadmap.md)
- Examples: `examples/wavelet/`
