# examples/signal/

Self-contained programs that exercise the most common Signal Processing
Toolbox functions shipped by `matlab_llvm`. Each example synthesises its
input signal inline (no fixture data) and prints diagnostic output, so
they're useful both as runnable smoke tests and as reading-order tours
of the SPT surface.

Run any one example with:

```sh
runtime/build_and_run.sh examples/signal/<name>.m /tmp/<name>
/tmp/<name>
```

| File | Demonstrates |
|---|---|
| `lowpass_design.m` | `butter` lowpass design + `filtfilt` zero-phase apply (with steady-state ICs that preserve constant signals exactly). Synthesises a 0..200 Hz chirp, drops everything above 100 Hz, prints passband / stopband RMS + attenuation in dB. |
| `bandpass_design.m` | `butter` bandpass — the band-variant dispatch picks up a literal 2-element `[W1 W2]` argument as bandpass. Verifies the chirp's mid-band energy survives while the high-band is killed. |
| `highpass_design.m` | `cheby1(...,'high')` — the Sema dispatcher picks up the trailing `'high'` string literal. Removes a low-frequency drift component and keeps the high-frequency tone. |
| `pwelch_demo.m` | Power spectral density via Welch's averaged-modified-periodogram with a Hamming window and 50% overlap. |
| `spectrogram_chirp.m` | Time-frequency analysis — STFT magnitude squared of a 0..250 Hz chirp. Prints peak energy per frame; the per-frame *bin* of the peak shifts upward as the chirp ramps up. |
| `resample_demo.m` | Sample-rate conversion via `resample(x, p, q)` (polyphase upsample-by-p then downsample-by-q), `decimate`, and `interp`. Verifies output lengths and energy preservation. |
| `findpeaks_demo.m` | Pulse measurements — `findpeaks` for local maxima of a tone, `statelevels` (histogram-based low/high estimation), `pulseperiod`, `pulsewidth`, `risetime`, `falltime`, `dutycycle` on a rectangular pulse train. |
| `xcorr_align.m` | Signal alignment via `finddelay` (signed argmax of cross-correlation), `xcov` (mean-removed cross-correlation), and `dtw` (dynamic time warping). |
| `sosfilt_demo.m` | High-order IIR via cascade-of-biquads — `tf2sos` to convert (b, a) into an L×6 SOS matrix, then `sosfilt` for numerically robust filtering of a chirp. |
| `poly_helpers.m` | Polynomial helpers — `roots` (Durand-Kerner), `poly`, `polyder`, `polyint`, `residue` (partial-fraction expansion). The bedrock primitives behind every IIR design path. |

For the full SPT surface and what's still open (e.g. `ellip` /
`ellipord`, `stft` / `istft`, multitaper, the strict 1996 Gustafsson
`filtfilt`, the `digitalFilter` system object), see
[`../../docs/signal_toolbox_roadmap.md`](../../docs/signal_toolbox_roadmap.md).

## Notes

- These are demonstration programs, not regression tests. The
  authoritative SPT regression corpus is in
  [`../../test/Run/sig_*.m`](../../test/Run/), which exercises 4-to-5-
  lane parity (LLVM / C / C++ / Python / TypeScript).
- A few examples use `disp(scalar)` instead of `fprintf('%.3f', scalar)`
  where the scalar comes from a stored intermediate (e.g.
  `m = max(col); fprintf(..., m)`). That's a workaround for a Sema
  scalar-typing limitation — not a recommendation against `fprintf`
  in user code.
