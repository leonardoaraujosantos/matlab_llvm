# Tasks

Groups 1–2 are the first slice (this change's PR). Groups 3+ are the per-domain catalog —
each block (or small same-domain set) is its own follow-on PR. Each block PR: register kind
+ classification, simulator evaluator (delegating to the toolbox runtime), `SimulateRun`
fixture + checks, `docs/mflowlink_blocks.md` row, update the parity snapshot, and file the
editor `NodeKind` (IDE repo).

## 0. Prioritization (gap analysis: 66 blocks vs 29 function-level toolboxes)

Blocks earn a dedicated kind only where time-domain drag-and-drop beats the generic
`signal_matlab_fcn` (function-first philosophy). Effort: S = evaluator delegates to existing
runtime; M = new state/port shape or codegen; L = needs a design decision or new plumbing.
Value: H = streaming/stateful/synthesizable, no ergonomic function equivalent.

**Tier 1 (high value, low effort — do next):**
- 4b.5b `signal_tff`/`signal_counter` → SystemVerilog (S/H) — finishes the HDL synthesize path
- 3.1 `signal_fft`/`signal_ifft` (S–M/H) + 3.2 `signal_window` (S/H) — DSP frame trio
- 3.4 `signal_biquad` SOS streaming IIR (S/H)
- 8.1 `signal_kalman` (M/H) — highest-value *new* block; no good function equivalent
- 4.1 PSK + 4.2 QAM mod/demod (M/H) + 4.4 `signal_error_rate` (S/H) — completes the Comms chain `awgn` started

**Tier 2 (high value, medium effort):**
- 8.2 `signal_dnn_predict` (M–L/H) — NN inference in a loop
- 4b.2/4b.3/4b.4 `signal_jkff`/`srff`/`shift_register`/`ram`/`rom` (S–M/H) — round out synthesizable HDL
- 3.3 `signal_spectrum` (M/M–H), 3.5 streaming `dcblock`/`lowpass`/`highpass` (M/M)
- 7.1 `signal_lqr`/`signal_observer` (S/M), 8.3 `signal_rl_agent` (M–L/M)

**Tier 3 (blocked on a design decision):** §5 Vision/Image (2-D-signal-on-a-wire), §6 RF (time- vs freq-domain triage), wavelet DWT, nav/robotics pose blocks.

**Tier 4 (leave as functions — low block value):** antenna, propagation, bluetooth, bioinfo,
econ, finance, curvefit, optim, gads, PDE, symbolic, GPU, ident (batch/frequency/symbolic).

## 1. First slice — recipe + parity guard (DONE, PR #350)

- [x] 1.1 Block-authoring recipe in `docs/mflowlink_blocks.md` (§"Adding a toolbox library block")
- [x] 1.2 Editor↔simulator parity guard: `test/Flowchart/BlockKindParity/run_tests.sh` diffs the registered kinds vs a committed `registered_block_kinds.txt` snapshot (fails on drift)
- [x] 1.3 Wired into ctest (`flowchart-block-kind-parity`); snapshot seeded (62 kinds → grows as blocks land)

## 2. First worked-example block (DONE, PR #351)

- [x] 2.1–2.3 `signal_awgn` (Comms AWGN channel) shipped end-to-end through the recipe — the worked example follow-on blocks copy. (A DSP transform block, §3, is still open; the linear-filter space is already covered by discrete_filter/transfer_fcn/state_space so the first block was a Comms channel instead.)

## 3. DSP / Signal Processing catalog (follow-on PRs)

- [ ] 3.1 `signal_fft` / `signal_ifft` — frame DFT/IDFT (`matlab_fft_c`/`matlab_ifft_c`, `VecOut_` frame path)
- [ ] 3.2 `signal_window` — windowing (Hann/Hamming/Blackman)
- [ ] 3.3 `signal_spectrum` — power-spectrum / spectrum-analyzer sink
- [ ] 3.4 `signal_biquad` — Biquad/SOS section (delegates to the DSP runtime; complements the existing `signal_discrete_filter` IIR)
- [ ] 3.5 `signal_dcblock` / `signal_lowpass` / `signal_highpass` — streaming filters (DSP `*_step` runtimes; need the obj-step delegation pattern)

## 4. Communications catalog (follow-on PRs)

- [x] 4.3 `signal_awgn` — AWGN channel (SNR + signal power) — DONE (PR #351)
- [ ] 4.1 `signal_psk_mod` / `signal_psk_demod` — PSK modulator/demodulator (comm runtime)
- [ ] 4.2 `signal_qam_mod` / `signal_qam_demod` — QAM modulator/demodulator
- [x] 4.4 `signal_error_rate` — error-rate (BER) calculation sink — DONE. Running mismatch ratio over `tx`/`rx`, accumulated once per major step; `error_rate.mflow` + SimulateRun checks (converges to 0.5 on a 50%-duty mismatch, bounded [0,1])

## 4b. HDL / digital catalog (follow-on PRs)

Sequential elements for synchronous digital modeling → the `-emit-{systemverilog,
verilog,cocotb}` lane. Mux/Demux + logic gates already ship (`signal_mux`/`demux`/
`logical`/`multiport_switch`).

- [x] 4b.1 `signal_dff` (D flip-flop), `signal_tff` (T flip-flop), `signal_counter` (up counter) — clocked posedge registers (simulator) — DONE
- [x] 4b.5a `signal_dff` → SystemVerilog: lowers to `always_ff @(posedge clk) s_ff <= D` via the SubsystemToMatlab persistent-register path; synthesizable (`-check-synthesizable` clean). DONE
- [x] 4b.x example circuits: `hdl_half_adder`, `hdl_full_adder` (combinational), `hdl_shift_register`, `hdl_freq_divider` (sequential) — DONE
- [x] 4b.2 `signal_jkff` / `signal_srff` — JK / SR flip-flops — DONE. Same clocked single-latch family as D/T (DigitalLatch_, once-per-major-step edge update). JK: 00 hold / 01 reset / 10 set / 11 toggle; SR: 10 set / 01 reset / 00,11 hold. `hdl_jk_sr.mflow` + SimulateRun checks (JK toggles as /2 divider, SR latches set/reset). SV emit is a follow-up like tff/counter were.
- [ ] 4b.3 `signal_shift_register` — N-bit serial/parallel shift register block (the example wires DFFs by hand today)
- [ ] 4b.4 `signal_ram` / `signal_rom` — addressable memory (addr/data/we ports; vector state)
- [x] 4b.5b emit-SystemVerilog for `signal_tff` (toggle) and `signal_counter` (increment+wrap) — DONE. Arithmetic next-state (`Q + T*(1-2Q)`, `inc - mod*(inc>=mod)`) keeps both branch-free and synthesizable; `sv_tff_smoke`/`sv_counter_smoke` in EmitSubsystem assert the always_ff register + `-check-synthesizable` clean

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
- [x] 7.2 Stats: `signal_running_stats` — streaming mean/var/std via an online Welford accumulator — DONE. Beats a MATLAB Function block (which can't hold persistent state in the flow). `running_stats.mflow` + SimulateRun checks (mean→bias, var→A²/2 on a sine over whole periods).

## 8. Estimation / ML in-the-loop catalog (follow-on PRs)

- [x] 8.1 `signal_kalman` — discrete Kalman filter — DONE. A/C/Q/R (+ optional B/x0/P0) matrix-literal params; N-vector state estimate output; standard predict/update recursion once per major step with a small in-file dense linalg kernel (matMul/matT/matInv via Gauss-Jordan). `kalman_constant.mflow` (1-state, ~100× variance reduction) + `kalman_tracker.mflow` (2-state CV tracker, infers velocity from noisy position) + SimulateRun checks. Highest-value new block delivered.
- [ ] 8.2 `signal_dnn_predict` — neural-net inference block in a control/sim loop (`runtime/toolbox/dlnet`); serialized network param
- [ ] 8.3 `signal_rl_agent` — trained policy-in-the-loop block (`runtime/toolbox/rl`)
