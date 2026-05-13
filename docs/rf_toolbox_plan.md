# RF Toolbox — closure plan (2026-05-12)

This doc captures the state of the RF Toolbox port after the
multi-session arc that landed RF-Tier-1 through RF-Tier-4 (function-
form) + the network-parameter classdef hierarchy + the complex-pair
Vector Fitting upgrade.

The companion doc `docs/comm_toolbox_roadmap.md` §9 holds the full
per-row status table; this file is the **forward** view — what's
already shipped, what's open, and which lanes are next.

## Shipped (RF-Tier-1 / 2 / 3 / 4 function-form)

### Touchstone I/O
- `touchstoneRead(filename)` — reads s1p / s2p / s3p / s4p / sNp with
  auto-detection from the `.sNp` extension.  Tolerates multi-line
  per-frequency rows, MA / DB / RI formats, Hz / kHz / MHz / GHz units.
- `touchstoneWrite(filename, data)` — writes any N-port in MA format,
  picking the historical `[s11 s21 s12 s22]` order for s2p and row-
  major `[s11 s12 … s1N; s21 …]` for sNp (N > 2).
- `touchstoneWriteS2p(...)` — 2-port-explicit alias for back-compat.
- `tsSij(data, i, j)` generic typed-getter for any port pair.

### Network parameter conversions
- 2-port closed-form: `sparamS2y`, `sparamS2z`, `sparamS2h`,
  `sparamS2abcd`, `sparamS2g`, `sparamS2t` + inverses `sparamH2s`,
  `sparamAbcd2s`, `sparamG2s` (via H-inverse route),
  `sparamT2s` (T-parameter chain).
- N-port via 2N × 2N real-equivalent complex matrix algebra:
  `sparamS2yN`, `sparamS2zN`.
- Mixed-mode 4-port: `sparamS2smm(s11..s44, block_code)` for
  block_code 0=dd / 1=dc / 2=cd / 3=cc.
- Port extraction: `snp2smp(data, port_list, m)` — matched-termination
  sub-block; `snp2smpZ(data, port_list, z_term, m)` — Schur complement
  for arbitrary non-z0 terminations at dropped ports.
- Γ ↔ Z helpers: `gamma2z(gamma, z0)` and `z2gamma(z, z0)`.

### Closed-form analyses
- `gammaIn`, `gammaOut` reflection coefficients.
- `vswr` from gamma.
- `powerGain` (transducer / available / operating via type code).
- `stabilityK` (Rollett), `stabilityMu` (Edwards-Sinsky mu1 / mu2).
- `s2tf` voltage transfer function.
- `cascadeSparams2` 2-port T-parameter chain.
- `cascadeSparamsN` N-port cascade (diagonal approximation, fast).
- `cascadeSparamsNFull` full Redheffer star product (k = N/2 case,
  exact for arbitrarily-coupled networks of even port count).

### RF system analysis
- `rfbudgetFriis(gains_dB, nfs_dB, ip3s_dBm, p_in_dBm, bw_Hz)` —
  cascaded gain, NF (Friis), IP3 (input-referred), output power,
  thermal noise, SNR.

### Vector Fitting (RF-Tier-3.1)
- `rationalfit(freq, h_re, h_im, nPoles, nIter)` — Gustavsen-Semlyen
  with both real poles AND complex-conjugate pole pairs.  Uses the
  `(α, β)` pole representation with real-arithmetic `[α β; -β α]`
  block form in the relocation matrix M.  After eig, eigenvalues
  auto-classify into real or complex pairs.  Output stores complex
  Poles + complex Residues columns.
- `freqresp(mdl, freqs)` — complex-pole-aware frequency-response
  evaluation.
- `passivity(mdl, f_lo, f_hi)` — max |H(jω)| over a dense log-spaced
  sweep.

### Time-domain (RF-Tier-3.2)
- `timeresp(mdl, u, ts)` — per-pole ZOH discretization with complex
  state for complex poles; reduces to real arithmetic on real poles.
- `s2tdr(S11, freqs, nPoles, ts, nSamples)` — TDR step response.
- `s2tdt(S21, freqs, nPoles, ts, nSamples)` — TDT step response.

### Transmission line geometries (RF-Tier-3.3)
- `rfckt_txline(Z0, εr, length, freqs, z0)` — generic line.
- `rfckt_coaxial(a, b, εr, length, freqs, z0)` — coaxial cable.
- `rfckt_microstrip(w, h, εr, length, freqs, z0)` — Hammerstad-Jensen.
- `rfckt_cpw(w, s, εr, length, freqs, z0)` — coplanar waveguide
  (Hilberg approximation).
- `rfckt_parallelplate`, `rfckt_twowire`.

### Matching networks (RF-Tier-4.1)
- `matchingnetwork(zs_re, zs_im, zl_re, zl_im, freq)` — L-section
  auto-synthesis.
- `matchingnetworkT(..., q_target)` — T-section with virtual high-
  impedance node.
- `matchingnetworkPi(..., q_target)` — Pi-section with virtual low-
  impedance node.

### LC filter blocks (RF-Tier-4.2)
- `rfckt_lcfilter(topology, comp1, comp2, freqs, z0)` — 3-element
  Lowpass-Tee (0), Lowpass-Pi (1), Highpass-Tee (2), Highpass-Pi
  (3) topologies via T-parameter chain composition of series-Z and
  shunt-Y elements.  Returns 2-port S-parameter struct.

### Smith chart numerics (RF-Tier-4.3)
- `smithGrid(r_norm, n_pts)` — constant-r and unit-circle complex
  column overlays.

### Classdef hierarchy
- `RFSparameters` (S11/S12/S21/S22) — already shipped pre-arc.
- Sibling network parameter classdefs: `RFYparameters`,
  `RFZparameters`, `RFHparameters`, `RFGparameters`,
  `RFAbcdparameters`, `RFTparameters`.
- RF circuit hierarchy: `RFCktAmplifier` (NF / Gain / IP3),
  `RFCktMixer` (NF / ConversionGain / IP3 / LO_Frequency),
  `RFCktPassive` (Loss / Label), `RFCktCascade` (Gains_dB / NFs_dB /
  IP3s_dBm columns), `RFCktParallel` / `RFCktSeries` / `RFCktShunt`
  combinator skeletons.
- `RFRational` (rfmodel.rational equivalent): A / C / D / Delay /
  Order / Error properties; populated from rationalfit struct via
  the typed getters.

### Typed-getter family
- `tsSij(data, i, j)` — generic S-param column at port pair (i, j).
- `tsYij`, `tsZij`, `tsHij`, `tsGij`, `tsTij` — Y/Z/H/G/T matrices.
- `tsAbcdA` / `tsAbcdB` / `tsAbcdC` / `tsAbcdD` — ABCD individual
  components.
- `rfPoles`, `rfResidues`, `rfD`, `rfOrder`, `rfFitError` — rationalfit
  struct fields.
- `smithRCircle`, `smithUnitCircle` — Smith-grid column extractors.
- `tsFreqs`, `tsZ0`, `tsNumPorts` — Touchstone struct accessors.

## Tests on the LLVM lane

The Run/ suite now has **67 RF-specific tests** (rf_*) under
`test/Run/`, of which 27 are `rf_writeva_*` (the Verilog-A export
surface from `verilog_a_plan.md` Tiers 1–10).  All pass at the time
of writing (322 / 322 total Run/ tests).  Fixtures:
`test/Run/fixtures/rf/test_amp.s2p`,
`test/Run/fixtures/rf/diff_pair.s4p`.

## Open — function-form
**All shipped.**  Touchstone v2 reader, LC bandpass/bandstop 4-element
topologies, and analyze-block helpers all landed.

## Open — classdef-bearing
The classdef-method bodies `analyze(block, freqs)` are syntactic
sugar over the function-form `rfAnalyze*` helpers — users can already
call those directly.  Adding the methods enables MathWorks-faithful
`analyze(amp, freqs)` method-dispatch.  ~3 sess if a contributor wants
that specific polish.

## Carved out
- ~~`writeVerilogA` / `rfmodel.rational/writeVA` Verilog-A export.~~
  **Shipped 2026-05-12 across the full Tier-1 → Tier-10 arc** of
  [`verilog_a_plan.md`](verilog_a_plan.md):
  - Tier-1 `writeVerilogA(mdl, filename)` — rationalfit / RFRational
    → parameterized .va with real-pole 1st-order sections + complex-
    conjugate-pair biquads + `absdelay` wrap.
  - Tier-2 `writeVerilogATF` / `writeVerilogAZPK` — continuous tf /
    zpk filters; closes the Verilog-A export of Control System
    Toolbox `tf`/`zpk` and SPT `butter('s')`/`cheby1('s')` returns.
  - Tier-3 `writeVerilogASS` — continuous SISO state-space.
  - Tier-4 `writeVerilogASource` / `Comparator` / `Schmitt`.
  - Tier-5 `writeVerilogAVCO` (idtmod phase accumulator).
  - Tier-6 `writeVerilogADAC`.
  - Tier-7 `writeVerilogADiode` / `OpAmp` / `RTD` / `Thermistor`.
  - Tier-7 follow-on: `writeVerilogAAmplifier` (gain × LPF × tanh),
    `writeVerilogAAM` (amplitude modulator), `writeVerilogAIQMod`
    (generic I/Q modulator — covers QAM-16 / QPSK / 8-PSK).
  - Tier-8 `writeVerilogANoise` (white + flicker, PSD).
  - Tier-9 `writeVerilogATable` (`$table_model` + .tbl sidecar).
  - Tier-10 polish: `scripts/va_lint.sh` (OpenVAF/ADMS wrapper),
    `scripts/va_cosim.sh` (ngspice+OpenVAF / Xyce+ADMS), opt-in
    CTest lanes `run-emit-va-admslint` + `run-emit-va-cosim` gated
    on `MATLAB_LLVM_WITH_VA_LINT` / `MATLAB_LLVM_WITH_VA_COSIM`,
    and user-facing reference [`emit_verilog_a.md`](emit_verilog_a.md).
  - One critical follow-on bug fix (commit `7b1f727`): the
    `rf_va_field_or` Poles→A / Residues→C fallback now reads
    `rows`/`cols` at the correct offsets for `matlab_mat_c`
    descriptors (was UB through a `matlab_mat *` alias).
- Circuit envelope simulation (multi-tone time-stepping nonlinear
  circuit solver).
- Harmonic Balance solver (Newton-Krylov on multi-tone steady-state
  nonlinear circuit residual).
- RF Budget Analyzer app, Smith Chart Tool app (Qt apps).
- Modelithics commercial component library (vendor-licensed data).
- IEEE P370 fixture characterization (niche).
- AMP file format reader (Touchstone covers nearly all use cases).
- Simulink RF Blockset (Simulink not in scope).

## Status

**RF Toolbox: 100% complete** (function-form + classdef).  Tier 1
polish all shipped:

- `gammams` / `gammaml` simultaneous-match Γ values.
- `groupdelay(S, freqs)` — d(phase)/dω via centered FDs.
- `s2tfPort(...)` with arbitrary input/output port designation.
- `rfbudgetTable(...)` returning per-stage cumulative columns.
- `stabCircleLoad(spar)` / `stabCircleSource(spar)` — Smith-chart
  stability circles (center + radius + denominator-sign per
  frequency).
- `analyze(block, freqs)` method dispatch on all rfckt classdefs
  (Amplifier / Mixer / Passive / Series / Shunt).
- MathWorks-faithful lowercase aliases:
  `s2y` / `s2z` / `s2h` / `s2g` / `s2abcd` / `s2t` +
  `h2s` / `g2s` / `abcd2s` / `t2s` + `rfbudget` / `rfwrite` /
  `sparameters`.
- `rfDelayEstimate(freqs, h_re, h_im)` — bulk delay τ from top-
  decade phase slope.
- `rfApplyDelay(freqs, h_re, h_im, τ)` — pre-fit de-delay step.
- `rfPassivityEnforce(mdl, f_lo, f_hi)` — iterative residue scaling
  to drive max|H(jω)| ≤ 1.
- `rationalfitWeighted(freqs, h_re, h_im, weight, nPoles, nIter)` —
  per-frequency weighted Vector Fitting.

Tier 2 polish (generalizations of shipped code, all shipped):
- `newref(spar, z0_new)` — renormalize an N-port S-parameter struct to
  a new reference impedance via the Γ_a-renormalization formula.
- `cascadeSparamsNFullK(A, B, k)` — Redheffer star product with
  arbitrary inner-connection port count k (generalizes
  `cascadeSparamsNFull`'s symmetric k = N/2 case to asymmetric outer
  port counts N_A ≠ N_B).
- `sparamS2abcdN(spar)` — N-port (even-N) ABCD via the Y-partition
  formula.  Returns A_ij / B_ij / C_ij / D_ij blocks of size (N/2)².
- `sparamS2hN(spar)` — N-port (even-N) H-parameters via the Y-
  partition formula.  Stored as full N×N H_ij block-stitched matrix.
- Native complex N×N LU decomposition with partial pivoting,
  transparently replacing the 2N×2N real-equivalent path inside
  `complex_mat_inv_2neq` (~4× speedup on every matrix inverse the RF
  runtime performs).  The real-equivalent path stays as a fallback
  when LU encounters a singular pivot.

Algorithm coverage: **complete** — Vector Fitting with complex
pairs, matrix algebra over arbitrary N (S↔Y/Z + Schur cascade +
Schur port termination), all closed-form analyses, all cross-
conversions (S↔Y/Z/H/G/ABCD/T and inverses), Friis cascade,
Hammerstad-Jensen, ZOH time-domain, Touchstone v1 + v2, LC filter
topologies (lowpass/highpass/bandpass/bandstop, Tee + Pi each),
stability + matching + Smith overlays + group delay.

## Forward plan

**Nothing in scope inside the RF Toolbox itself.**  Adjacent
work shipped or in flight:

- ✅ Verilog-A export — full Tier-1 → Tier-10 arc shipped via
  [`verilog_a_plan.md`](verilog_a_plan.md).  13 runtime entries
  (`writeVerilogA`, `writeVerilogATF` / `AZPK` / `ASS`,
  `writeVerilogASource` / `AComparator` / `ASchmitt`,
  `writeVerilogAVCO`, `writeVerilogADAC`,
  `writeVerilogADiode` / `AOpAmp` / `ARTD` / `AThermistor`,
  `writeVerilogANoise`, `writeVerilogATable`, plus the Tier-7
  follow-on composites `writeVerilogAAmplifier` / `AAM` / `AIQMod`),
  24 examples in `examples/verilog_a/`, 27 `rf_writeva_*` Run/ tests,
  two opt-in CTest lanes (lint + cosim).  See user reference
  [`emit_verilog_a.md`](emit_verilog_a.md).
- Circuit envelope simulation — multi-tone time-stepping nonlinear
  solver.
- Harmonic Balance — Newton-Krylov on multi-tone steady-state.
- RF Budget Analyzer / Smith Chart Tool apps (Qt).
- Modelithics commercial component library.
- IEEE P370 fixture characterization.
- AMP file format.
- Simulink RF Blockset.

## Carved out (final)

Original list minus Verilog-A export (which shipped 2026-05-12
across the full Tier-1 → Tier-10 arc of
[`verilog_a_plan.md`](verilog_a_plan.md)):

- Circuit envelope simulation
- Harmonic balance solver
- RF apps (Budget Analyzer / Smith Chart Tool)
- Modelithics commercial component library
- IEEE P370 fixture characterization
- AMP file format reader
- Simulink RF Blockset

These all need infrastructure outside the language layer (multi-tone
time-stepping solvers, Qt apps, commercial licensing).
