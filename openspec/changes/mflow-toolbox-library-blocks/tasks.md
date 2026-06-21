# Tasks

Groups 1–2 are the first slice (this change's PR). Groups 3+ are the per-domain catalog —
each block (or small same-domain set) is its own follow-on PR. Each block PR: register kind
+ classification, simulator evaluator (delegating to the toolbox runtime), `SimulateRun`
fixture + checks, `docs/mflowlink_blocks.md` row, update the parity snapshot, and file the
editor `NodeKind` (IDE repo).

## 1. First slice — recipe + parity guard

- [ ] 1.1 Write the block-authoring recipe into `docs/mflowlink_blocks.md` (the §"Adding a toolbox library block" checklist: kind registration, classification, evaluator, runtime delegation, docs, test, editor parity)
- [ ] 1.2 Add the editor↔simulator parity guard: a `test/Flowchart/` test that enumerates the registered `signal_*` kinds and diffs against a committed `registered_block_kinds.txt` snapshot (fails on drift)
- [ ] 1.3 Wire the guard into ctest; seed the snapshot with today's ~63 kinds

## 2. First slice — first DSP block (worked example)

- [ ] 2.1 Implement `signal_fft` (or `signal_fir`): register kind + sample-time/loop-breaker class in `SignalFlowLowering.cpp`
- [ ] 2.2 Simulator evaluator in `MflowLinkSim.cpp` delegating to the DSP runtime (`matlab_fft_c` / FIR), using the `VecOut_` frame path for vector output
- [ ] 2.3 `examples/mflowlink/` fixture + `SimulateRun` checks asserting a known transform/filter result; `docs/mflowlink_blocks.md` row; update parity snapshot

## 3. DSP / Signal Processing catalog (follow-on PRs)

- [ ] 3.1 `signal_fft` / `signal_ifft` — frame DFT/IDFT (`matlab_fft_c`/`matlab_ifft_c`)
- [ ] 3.2 `signal_fir` — FIR filter (taps param; frame or sample)
- [ ] 3.3 `signal_biquad` / `signal_iir` — IIR/Biquad sections
- [ ] 3.4 `signal_window` — windowing (Hann/Hamming/Blackman)
- [ ] 3.5 `signal_spectrum` — power-spectrum / spectrum-analyzer sink

## 4. Communications catalog (follow-on PRs)

- [ ] 4.1 `signal_psk_mod` / `signal_psk_demod` — PSK modulator/demodulator (comm runtime)
- [ ] 4.2 `signal_qam_mod` / `signal_qam_demod` — QAM modulator/demodulator
- [ ] 4.3 `signal_awgn` — AWGN channel (SNR/EbNo param)
- [ ] 4.4 `signal_error_rate` — error-rate calculation sink

## 5. Computer Vision / Image Processing catalog (follow-on PRs)

- [ ] 5.1 Resolve the 2-D-signal-on-a-wire question (design Open Question) — bus/vector vs a small 2-D extension
- [ ] 5.2 `signal_image_source` — image/From-File source
- [ ] 5.3 `signal_image_filter` — 2-D convolution / filter (`runtime/toolbox/images`/`vision`)
- [ ] 5.4 `signal_color_space` — color-space conversion
- [ ] 5.5 `signal_threshold` — image threshold / binarize

## 6. RF catalog (follow-on PRs)

- [ ] 6.1 Triage which RF capabilities are time-domain-meaningful as blocks vs frequency-domain functions (design Open Question)
- [ ] 6.2 Implement the agreed RF blocks (e.g. an S-parameter-driven 2-port in the time domain) delegating to `runtime/toolbox/rf`

## 7. Control / Stats round-outs (follow-on PRs)

- [ ] 7.1 Control: a few more dedicated blocks beyond `signal_pid` / `signal_state_space` / `signal_transfer_fcn` where useful (e.g. discrete LQR/observer gain block)
- [ ] 7.2 Stats: a block where streaming/time-domain makes sense (e.g. running mean/variance) — only if it beats a MATLAB Function block
