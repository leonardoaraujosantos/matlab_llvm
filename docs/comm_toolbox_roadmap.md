# Communications + RF + Antenna Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Runtime + Debug + REPL) needs to
ship in order to faithfully execute MATLAB **Communications
Toolbox**, **RF Toolbox**, *and* **Antenna Toolbox** programs.
Sources:

- *Communications Toolbox User's Guide* (R2026a, 2098 pp —
  Sources/Sinks, Source Coding, Error Correction & Detection,
  Interleaving, Digital Modulation, Analog Modulation, MIMO,
  Filtering, Equalization, Synchronization & Receiver Design, MSK,
  Predistortion, Propagation & Channel Models, RF Propagation, AI
  for Wireless).
- *RF Toolbox User's Guide* (R2026a, 402 pp — RF Objects, Model an
  RF Component, Verilog-A Export, AMP File Format, How-Tos /
  Filters / Matching Networks / Budget Analysis).
- *Antenna Toolbox User's Guide* (R2026a, 1382 pp — Antenna
  Concepts, Antenna Catalog, Computational Techniques (MoM / PO /
  hybrid MoM-PO / FMM / wire solver), RF Propagation (Site Viewer
  / ray tracing), Examples).

Three toolboxes, one roadmap because they share infrastructure:
- **Comm** uses RF blocks (PA nonlinearity, IQ imbalance, phase
  noise, channel impairments) — Tier-4 §5.3 there cross-pollinates
  with the RF Toolbox §8 here.
- **RF Toolbox** is conceptually distinct (S-parameters, circuits,
  network parameter algebra) but rides on the same complex-matrix
  + classdef-System-Object infrastructure.
- **Antenna Toolbox** layers an EM full-wave solver (Method of
  Moments) below RF: an antenna instance produces S-parameters /
  impedance / radiation pattern that flow naturally into RF Toolbox
  cascades and Comm channel chains. The catalog of pre-built antenna
  types is itself a large classdef hierarchy.
- Splitting into three docs would duplicate the §0/§1/§11–§15
  plumbing.
- Heavy interactive UI / Verilog-A / circuit-envelope /
  harmonic-balance / ray-tracing / 3-D site-viewer / FMM /
  hybrid-MoM-PO pieces are carved out the same way for all three
  products.

The repo's overall compatibility target is a **practical numeric
subset** (see `feature_status.md`), so this doc inherits the same
posture: focus on the *programmable* surface (functions returning
arrays / structs / `comm.*` System Object instances / `rfckt` /
`rfmodel` / `sparameters` / antenna catalog instances), explicitly
defer the GUI surface (Bit Error Rate Analyzer, Constellation
Diagram / Eye Diagram / Spectrum Analyzer scopes, RF Budget
Analyzer, Smith Chart Tool, Antenna Designer app, Site Viewer,
3-D Field/Pattern viewer, RF Propagation visualizations, Wireless
Waveform Generator, Wireless Network Simulator), the Simulink block
library, the AI-for-Wireless deep-learning examples, and the SDR /
hardware-in-the-loop chapters.

Comm sits **on top of** Signal Processing Toolbox: filtering, FFT,
multirate, windows, spectral estimation, polynomial helpers — the
Tier-0/1 SPT surface. RF Toolbox sits on top of the same complex-
matrix + classdef stack (S-parameter conversions are pure linear
algebra, network objects are classdefs). Antenna Toolbox sits on a
**heavier substrate**: it requires a triangle/segment mesher,
complex linear solver at scale (`Z·I = V` with Z complex N×N where
N = number of mesh edges; LU on N up to ~10⁴ for practical
geometries), surface-integral numerical quadrature, and far-field
radiation integrals. Most of the Comm/RF substrate is shipped;
the Antenna substrate is mostly **new work** — flagged in §10.

For shipped work, see [`feature_status.md`](feature_status.md). For
the cross-toolbox roadmap entries, see [`roadmap.md`](roadmap.md);
this doc is the per-product companion, parallel to
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) and
[`control_toolbox_roadmap.md`](control_toolbox_roadmap.md).

Document layout: §1 baseline → §2 Comm Tier 1 → **§3 Propagation
Models — promoted early-priority track (PROP-Tier-1a / 2a / 2b /
1b)** → §4–§8 remaining Comm tiers → **§9 RF Toolbox companion
(RF-Tier-1…RF-Tier-4)** → **§10 Antenna Toolbox companion
(ANT-Tier-1…ANT-Tier-5)** → §11 REPL/Debug → §12 execution order →
§13 carve-outs → §14 tests → §15 summary.

**Why §3 Propagation is up front**: the user's stated PtP-with-
terrain + Coverage Map + Longley-Rice workflow is reachable via
function-form APIs (§3.1–§3.3) with **zero** dependency on the
System-Object lowering fix that gates Comm Tier 3+, RF-Tier-1+,
and ANT-Tier-1+. Promoting Propagation reflects that priority and
signals it as a parallel-shippable track.

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. Tier-1
  is small (closure of a few primitives over the existing SPT
  baseline) and lights up the symbol-level surface. Tier-2 lights up
  the standard "modulate → AWGN → demodulate → BER" loop. Tier-3
  brings channel coding. Tier-4 brings equalization and sync. Tier-5
  is OFDM / MIMO / fading. Higher tiers are stretch.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started ·
  🔴 deliberately deferred.
- **REPL / Debug**: most comm primitives return `matlab_mat *` (real
  or complex) and inherit the existing matrix display path. The new
  surface is **`comm.*` System Objects** (handle-shaped classdefs
  with persistent internal state across `step` calls). Those need
  inspector wiring like CST's model objects; flagged in §11.

---

## 0.5 Shipped status snapshot (2026-05-17)

Roughly the entire function-form surface across Comm / RF / Antenna /
Propagation has shipped — only the multi-class **System-Object** form
of the comm classes and the planar / array Antenna MoM tracks remain
open. Quick map:

| Track | Tiers shipped | What's left |
|---|---|---|
| **Comm Tier-1 base layer** (§2) | ✅ all (`randi` / `rng` / `randsrc` / `awgn` / `int2bit` / `biterr` / `symerr`) | — |
| **Comm Tier-2 digital modulation** (§4) | ✅ PAM / QAM / PSK / FSK / generic / RC + RRC + Gaussian / `berawgn` / `scatterplot` / `eyediagram` | BPSK/QPSK convenience SO aliases (§4.4) gated on SO surface |
| **Comm Tier-3 channel coding** (§5) | ✅ CRC (function + SO) / convolutional + Viterbi (hard + soft) / Hamming / interleavers / **Tier-7 modern codes** (Polar SC, LDPC min-sum, Turbo PCCC max-log-MAP) | BCH/RS + `gf(2^m)` (§5.3) |
| **Comm Tier-4 equalisers + sync + RF impairments** (§6) | ✅ LMS / RLS / CMA / DFE / Costas PLL / Mueller-Müller / preamble detect / IQ imbal / Saleh / Rapp / Ghorbani / phase noise / `vitdecSoft` | SO variants gated on full SO surface |
| **Comm Tier-5 OFDM / fading / MIMO** (§7) | ✅ `ofdmmod` / `ofdmdemod` / Rayleigh Jakes / Rician / Alamouti OSTBC + MRC / `mlDetect` | Complex ZF / MMSE / sphere decoding (gated on future complex-LU runtime); SO variants |
| **Comm Tier-6 spreading + source coding** (§8) | ✅ PN / Gold / Hadamard / Walsh / `quantiz` / Lloyd / G.711 µ-law + A-law / DPCM | Kasami sequences (m-sequence decimation); `dpcmopt` codebook design |
| **RF-Tier-1 → Tier-4** (§8.1 – §8.4) | ✅ Touchstone v1+v2 / 2-port + N-port S↔Y/Z/H/G/ABCD/T / cascade (Redheffer) / `rfbudget` / Vector Fitting (real + complex pole pairs) / `s2tdr` / `s2tdt` / transmission-line geometries (microstrip / CPW / coax / parallel-plate / two-wire) / `matchingnetwork` (L / T / Pi) / `rfckt.*` block hierarchy + LC filter circuits / Smith chart numerics | rfckt classdef wrappers polish, Verilog-A export (already shipped via [`verilog_a_plan.md`](verilog_a_plan.md) Tier-1 → Tier-10) |
| **ANT-Tier-2 MVP** (see [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md)) | ✅ Closed-form thin-wire dipole (`antennaWireSolve` / `antennaWirePattern` / `antennaWireSparameters`) + `AntDipole` / `AntMonopole` classdefs | ANT-Tier-2b multi-wire MoM (Pocklington / Hallen on multi-wire geometries), ANT-Tier-3 surface MoM (planar / patch), ANT-Tier-4 arrays |
| **PROP-Tier-1a / 2a / 2b / 3 + 1b** (see [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md)) | ✅ All — closed-form path loss / cellular empirical / Fresnel / knife-edge / Haversine + Vincenty / Longley-Rice engineering port / terrain profile / LOS / link budget / single-TX coverage / sector + cosine + Gaussian + 3GPP patterns / mount orientation / multi-TX `coverageGridMulti` with best-server + sum-power + SINR aggregation + `PropagationModel` / `TxSite` / `RxSite` CamelCase classdefs | NTIA v7.0 byte-identical ITM reference port; SiteViewer 3-D rendering (carved out — see [`siteviewer.md`](siteviewer.md)) |

The §12 execution-order table (rows 1 – 48) has been refreshed to
match. The remaining open items consolidate into: (1) full SO surface
for the `comm.*` classes; (2) `gf(2^m)` typed descriptor for BCH/RS;
(3) the ANT-Tier-2b multi-wire MoM debug; (4) the multi-week
ANT-Tier-3 surface-integral MoM.

---

## 1. Already shipped (Tier-0 baseline, inherited from SPT + core)

These are the matlab_llvm primitives Comm sits on top of. Locations
are in `runtime/matlab_runtime.cpp` unless noted.

| Group | Functions / capabilities | Notes |
|---|---|---|
| Complex arrays as first-class type | `matlab_mat_c { magic, re*, im*, rows, cols }` (`runtime_internal.h:55`); complex `+ - .* ./ * /`, `conj`, `real`, `imag`, `angle`, complex FFT | All comm signals are I+jQ. The existing complex matrix kernel is the load-bearing piece of every modulation / demodulation / channel block. |
| FFT / IFFT | `fft`, `ifft`, `fft2`, `ifft2`, `fftshift`, `ifftshift` | Pure-C Cooley-Tukey radix-2 + Bluestein for general N. Drives OFDM, spectral estimation, pulse-shaping windows. |
| Convolution / correlation | `conv`, `conv2`, `xcorr`, `xcov`, `finddelay`, `dtw` | Matched filtering, channel modeling, frame sync correlation. |
| FIR / IIR design + apply | `butter`, `cheby1`, `cheby2`, `besself`, `fir1`, `sgolay` + `[b,a]`/HP/BP/BS variants; `filter`, `filtfilt`, `sosfilt`; order helpers `buttord` / `cheb1ord` / `cheb2ord`; form conversions `tf2zp` / `zp2tf` / `tf2sos` / `sos2tf`; analog `bilinear`, `freqs` | Pulse shaping rolls off into `fir1` + a custom impulse response; matched filters are `filter` calls. |
| Filter inspection | `freqz`, `impz`, `stepz`, `grpdelay` | Channel response analysis, pulse-shape group-delay correction. |
| Multirate | `upfirdn`, `decimate`, `interp`, `resample`, plus `upsample` / `downsample` stubs | **Critical for comm**: pulse-shaped tx/rx, sample-rate conversion at the receiver. Anti-aliased; replaces the toy stubs. |
| Spectral estimation | `periodogram`, `pwelch`, `cpsd`, `mscohere`, `tfestimate`, `spectrogram` (single-output) | Channel characterization, SNR measurement, BER vs Eb/N0 scaffolding. |
| Linear-prediction / parametric PSD | `levinson`, `lpc`, `aryule`, `arburg`, `pyulear`, `pburg` | Channel estimation, AR-model channels. |
| Time-frequency / transform tail | `dct`, `idct`, `fwht`, `hilbert`, `goertzel` | `hilbert` for analytic-signal demodulation; `goertzel` for tone detection (e.g. DTMF). |
| Pulse measurements | `findpeaks`, `rms`, `medfilt1`, `hampel`, `envelope`, `risetime`, `falltime`, `dutycycle`, `overshoot`, `undershoot`, `settlingtime`, `pulseperiod`, `pulsewidth`, `slewrate`, `statelevels`, `midcross` | Eye-diagram metrics, signal-quality measurements. |
| Waveforms | `chirp`, `sawtooth`, `square`, `gauspuls`, `rectpuls`, `tripuls`, `sinc` | Test signals, swept-frequency sweeps, RF-impairment driving signals. |
| Polynomial helpers | `roots`, `poly`, `polyder`, `polyint`, `[r,p,k] = residue(b,a)`, `polyfit`, `polyval` | Trellis polynomials are at heart polynomial; CRC is GF(2) polynomial division. |
| RNG | `rand(m, n)`, `randn(m, n)` | Uniform and Gaussian. **Critical gap**: no `randi` — see §2.1. No seed-control entry (`rng(seed)`); flagged in §2.1. |
| Linear algebra | dense `+ - * /`, `\`, `inv`, `det`, `trace`, `qr`, `lu`, `chol` (sym), 1-return non-symmetric `eig`, single-σ `svd`, complex FFT | Equalization (least-squares, MMSE), MIMO precoding, channel inversion. |
| OOP | `classdef`, single inheritance, `properties`, `methods`, constructors, operator overloading, `Dependent` properties, `enumeration` | Every `comm.*` System Object is a classdef. **Architectural blocker recorded** in CST roadmap §12: tensor-typed RHS routed through `_set_f64` after monomorphization fails verifier. Same blocker applies here — System Objects use the same field-store path. See §11 / §15. |
| Containers | `struct`, `cell`, `dictionary` / `containers.Map`, `table`, `categorical` | Trellis structs, code descriptor objects, BER simulation result tables. |
| Plotting (Cairo, headless) | PNG / SVG / PDF emission via `runtime/plot/` | Constellation plots / eye diagrams as static images possible; interactive scopes are out of scope (§13). |
| REPL / Debug | JIT REPL, workspace inspector, breakpoints, step, locals, `dbg(x)` | Will need new inspector entries for `comm.*` System Objects (see §11). |

**What Comm-specific code today**: zero. There is no `pammod`,
`qammod`, `pskmod`, `awgn`, `biterr`, `convenc`, `vitdec`, `bchenc`,
`rsenc`, `comm.OFDMModulator`, etc. The compatibility surface starts
empty.

---

## 2. Tier 1 — base-layer prerequisites (gates everything)

Like CST's Tier 1, this is a small **prerequisite tier** that almost
no user-visible Comm function lights up without. Unlike CST, the
prerequisites are tiny — most are 1–2 sessions each.

### 2.1 Random integer source `randi` ✅

**Scope**:
- `randi(imax)` — single int in `[1, imax]`.
- `randi(imax, n)` — `n×n`.
- `randi(imax, m, n)` — `m×n` matrix of ints in `[1, imax]`.
- `randi([imin, imax], m, n)` — closed interval.
- `randi(imax, sz, 'like', proto)` — class hint (for `int8` / `uint8`
  / `logical` outputs); the `'like'` form gates symbol generation
  into typed arrays. Defer until typed-int storage matures.

**Why this matters**: every comm sim starts with `data = randi([0
M-1], N, 1)` to generate symbols. Without it users have to write
`floor(rand(N,1)*M)` which (a) is error-prone at the boundary and
(b) is the canonical example used in MathWorks docs. Refusing to
ship it is refusing the textbook idiom.

**Effort**: 1 session. Reuses `matlab_rand` infrastructure;
multiplies and rounds.

### 2.2 RNG seed control `rng` ✅

**Scope**:
- `rng(seed)` — set deterministic seed.
- `rng('default')`, `rng('shuffle')`.
- `s = rng()` / `rng(s)` — save/restore (returns a struct).

**Why**: comm sim reproducibility. BER curves diverge by the third
decimal between runs without seed control, and the docs assume it
exists. Without `rng`, the test corpus can't ship deterministic
oracles for any function that touches RNG (which is every channel
model).

**Effort**: 1 session if we keep a single global PRNG state.

### 2.3 `randsrc` and `randerr` ✅

**Scope**:
- `out = randsrc(m, n, alphabet)` — random samples from `alphabet`
  (vector of values).
- `out = randsrc(m, n, [alphabet; probs])` — weighted alphabet.
- `out = randerr(m, n, errors)` — random binary error vectors with
  exactly `errors` 1s per row.

**Why**: convenience for source modeling; constellations,
intentional bit errors. Textbook examples lean on these.

**Effort**: 1 session each (thin wrappers over `randi`).

### 2.4 Bit ↔ integer conversion `int2bit` / `bit2int` / `de2bi` / `bi2de` ✅

**Scope**:
- `bits = int2bit(ints, nbits)` — MSB-first bit vector per integer.
- `ints = bit2int(bits, nbits)` — inverse.
- Legacy aliases `de2bi(d, n)` (LSB-first by default!) and
  `bi2de(b)`. Note the **MSB/LSB convention difference** between the
  old `de2bi` and the new `int2bit` — both must ship with the
  correct default per MATLAB docs.

**Why**: load-bearing for **every** bit-level coding workflow
(channel coding, scramblers, interleavers, framers).

**Effort**: 1 session for both pairs.

### 2.5 AWGN channel `awgn` ✅

**Scope**:
- `y = awgn(x, snr)` — adds white Gaussian noise to attain `snr` dB
  SNR. Default measure: `'measured'` (estimate signal power from
  `x`).
- `y = awgn(x, snr, sigpower)` — explicit signal power in dBW.
- `y = awgn(x, snr, 'measured', seed)` — deterministic seed (gated
  on §2.2 `rng`).
- `y = awgn(x, snr, sigpower, randstream)` — explicit RNG stream
  (defer; substream support is out of scope).

**Why**: this is the single most-called comm function. Without it
no BER simulation runs.

**Algorithm**: estimate or accept signal power → compute noise
variance from SNR → draw real or complex Gaussian (complex iff
input is complex, with √(σ²/2) per axis) → add. Honors `'linear'`
vs default `'dB'` SNR units.

**Effort**: 1 session.

### 2.6 BER / SER computation `biterr` / `symerr` ✅

**Scope**:
- `[nerr, ber] = biterr(x, y)` — bit-error count and ratio.
- `[nerr, ber] = biterr(x, y, k)` — input is `k`-bit symbols (bits
  are unpacked).
- `[nerr, ber] = biterr(x, y, k, flag)` — `'row-wise'`,
  `'column-wise'`, `'overall'` (default).
- `[nerr, ser] = symerr(x, y[, flag])` — symbol-error count.

**Why**: the answer to "did my comm sim work". Coupled to every
modulation / coding example.

**Effort**: 1 session each. Mostly bookkeeping over element-wise
compare.

### 2.7 Tier-1 closure summary

| Primitive | Effort | Status |
|---|---|---|
| `randi` (2.1) | 1 sess | ✅ shipped |
| `rng` (2.2) | 1 sess | ✅ shipped (numeric-tag dispatch + `rngDefault` / `rngShuffle` / `rngGet` / `rngSet` siblings for the string-arg variants) |
| `randsrc` / `randerr` (2.3) | 2 sess | ✅ shipped (plus `randsrcWeighted` for non-uniform alphabets) |
| `int2bit` / `bit2int` / `de2bi` / `bi2de` (2.4) | 1 sess | ✅ shipped |
| `awgn` (2.5) | 1 sess | ✅ shipped (real + complex; descriptor-magic dispatch on input) |
| `biterr` / `symerr` (2.6) | 2 sess | ✅ shipped (single-return convention — ratio by default, `biterrCount` / `symerrCount` for the integer counts; `biterrK(x, y, k)` for k-bit symbol BER) |

**Total**: ~1.5 weeks (8 sessions). Lights up the bit-source-and-
measure surface that every higher tier consumes.

**Test gate**: a 4-line "modulate → AWGN → demod → BER" loop must
pass on all 5 emit lanes once Tier 2 lands on top — Tier 1 alone is
"sources and sinks" with no modulation. The closure here is a
BPSK Monte-Carlo loop (`examples/comm/ber_awgn_uncoded.m`) — sim
BER tracks Q(sqrt(SNR_lin)) within ~5% from 4 dB onward at 50 k
bits per SNR point. Tier-2 modulation will swap the
`tx_sym = 1 - 2*tx_bits` map for the real `pammod` / `qammod` /
`pskmod` entries.

---

## 3. Propagation Models — moved to dedicated roadmap

The Propagation Models tier plan (PROP-Tier-1a closed-form ITU-R /
cellular / Fresnel / knife-edge / geo helpers through PROP-Tier-3
directional multi-site coverage + the PROP-Tier-1b MathWorks-API
classdef wrappers) was previously documented here as chapter §3. It
has been **promoted to a standalone roadmap** at
[`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md)
for discoverability and uniformity with the other per-toolbox
roadmaps.

**Status (2026-05-17): PROP-Tier-1a + 2a + 2b + 3 + 1b all shipped.**
~7 weeks of function-form work across multiple commits, plus the
`TxSite` / `RxSite` / `PropagationModel` CamelCase classdef wrappers
(commit `f764dbd` sidestepped the originally-planned SO dependency).

| Sub-tier | What lights up | Status |
|---|---|---|
| **PROP-Tier-1a** — Function-form closed-form models | All empirical path-loss formulas (FSPL/Hata/COST231/Egli/ECC33/SUI/Ericsson + ITU-R rain/gas/fog/close-in), Fresnel zones, knife-edge diffraction, Haversine/Vincenty | ✅ |
| **PROP-Tier-2a** — ITM / Longley-Rice (function-form) | Terrain-aware path loss with reliability tuning (engineering port; byte-identical NTIA v7.0 reference port stays 🔵) | ✅ |
| **PROP-Tier-2b** — Single-TX PtP + Coverage Map | `los_check`, `link_budget`, `coverage_grid` numeric API (single-TX, omnidirectional) | ✅ |
| **PROP-Tier-3** — Directional + multi-site coverage | Sector / cosine / Gaussian / 3GPP analytical patterns, mount orientation, `coverage_grid_multi` with best-server / sum-power / SINR aggregation | ✅ |
| **PROP-Tier-1b** — `propagationModel` / `txsite` / `rxsite` classdef wrappers | MathWorks-API-faithful `prop = propagationModel(...)` + `pathloss(prop, rx, tx)` syntax | ✅ |

Only the **ANT-Tier-2 pattern bridge** (§5.5 of the new roadmap)
remains 🔵 — gated on the planned ANT-Tier-2b multi-wire MoM in
[`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md). Runtime
lives in `runtime/runtime_prop.cpp` (~1700 lines). See
[`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md)
for the full plan + closure tables + execution order + carve-outs.

### 3.1 What Propagation brings to the rest of the roadmap

- **PROP → Comm**: link-budget realism. `link_budget` lets a Comm
  BER simulation be parameterized by a physical path-loss + thermal
  noise floor instead of an abstract SNR.
- **PROP → RF Toolbox**: `link_budget` and `rfbudget` (RF-Tier-2.3)
  compose — circuit-side noise figure / IP3 + geometry-side path
  loss = full link margin.
- **PROP → Antenna Toolbox**: see
  [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md) §7 for
  the pattern-bridge story.

## 4. Tier 2 — minimum-viable digital modulation loop (~3 weeks)

This is the first user-visible Comm slice: generate symbols, modulate
them with one of the three workhorse schemes (PAM / QAM / PSK), pass
through AWGN, demodulate, count errors. Layered on Tier 1.

### 4.1 PAM — `pammod` / `pamdemod` ✅

**Scope**:
- `y = pammod(x, M)` — `M`-PAM modulation; `x` is integers in `[0,
  M-1]`. Default natural mapping; output is real (PAM is on the
  real line).
- `y = pammod(x, M, ini_phase)` — initial phase rotation (output
  becomes complex if `ini_phase ≠ 0`).
- `y = pammod(x, M, ini_phase, sym_order)` — `'gray'` vs `'bin'`.
- Demod inverse `pamdemod`.

**Effort**: 0.5 week (1–2 sessions).

### 4.2 QAM — `qammod` / `qamdemod` ✅

**Scope**:
- `y = qammod(x, M)` — square `M`-QAM (M = 4, 16, 64, 256, …); also
  cross-QAM for non-square (M = 8, 32, 128). Returns complex.
- `y = qammod(x, M, sym_order)` — `'gray'` or `'bin'`.
- `y = qammod(x, M, 'UnitAveragePower', true)` — unit-power
  normalization (the **default** in System-Object form, not in the
  function form — get this right).
- Soft demod: `qamdemod(y, M, 'OutputType', 'llr')` produces
  log-likelihood ratios. Required for soft-decision decoding (Tier
  4). Hard demod is `'OutputType', 'integer'` (default) or `'bit'`.

**Effort**: 1 week. Square QAM is straightforward; cross-QAM needs
the standard "L-shape minus corners" mapping table; LLR demod needs
the per-bit max-log approximation.

### 4.3 PSK — `pskmod` / `pskdemod` ✅

**Scope**:
- `y = pskmod(x, M)` — uniform-spaced PSK on the unit circle.
- `y = pskmod(x, M, ini_phase)` — phase offset (default `0`).
- `y = pskmod(x, M, ini_phase, sym_order)`.
- `y = pskmod(x, M, ini_phase, 'gray')` — Gray mapping (the
  practical default for any modern link).
- LLR demod via `pskdemod(y, M, ini_phase, 'OutputType', 'llr')`.

**Effort**: 0.5 week.

### 4.4 BPSK / QPSK convenience aliases 🔵

`comm.BPSKModulator` / `comm.QPSKModulator` (System Objects, see
§4.x) and the implicit fact that `pskmod(x, 2) = 1 - 2*x` (BPSK)
and `pskmod(x, 4, pi/4)` (QPSK with /4 offset) are the canonical
forms. Document equivalence; do not ship dedicated builtins until
the System-Object surface exists (§5 / §11).

### 4.5 FSK — `fskmod` / `fskdemod` ✅

**Scope**:
- `y = fskmod(x, M, freqsep, nsamp, fs)` — continuous-phase or
  hard-keyed M-FSK. The 5-arg form is the practical one.
- Demod inverse — coherent or non-coherent (`'coherent'` /
  `'noncoherent'`).

**Effort**: 1 week. FSK is more complex than PSK because of the
explicit per-sample oversampling and the demodulator's
energy-detection variant. Defer if Tier-2 timeline tight.

### 4.6 Generic `genqammod` / `genqamdemod` ✅

User-supplied constellation. The same dispatcher many of the above
collapse to internally; useful as a fall-through entry once the
specific modulators ship. Effort: 0.5 week.

### 4.7 Pulse shaping — `rcosdesign` ✅

**Scope**:
- `b = rcosdesign(beta, span, sps)` — root-raised-cosine FIR
  coefficients. Default shape `'sqrt'` (RRC); `'normal'` is the
  full RC.
- `b = rcosdesign(beta, span, sps, shape)`.
- `gaussdesign(BT, span, sps)` — Gaussian filter for GMSK / GFSK.

**Why**: RRC is the universal Tx/Rx pulse for QAM links. Today,
users would have to hand-roll the closed-form RC impulse response —
tractable but tedious. Shipping `rcosdesign` is a 1-session win.

**Effort**: 1 session. Closed-form impulse response (with the
known L'Hôpital handling at `t = 0` and `t = ±span/(4·beta)`) plus
unit-energy normalization.

### 4.8 BER reference curves — `berawgn`, `bercoding` ✅

**Scope**:
- `Pb = berawgn(EbN0, modulation, M, ...)` — closed-form BER under
  AWGN for PAM / PSK / QAM / DPSK / FSK (per the user-guide tables
  on p. ~13).
- `Pb = bercoding(EbN0, codetype, ...)` — coded BER bounds.
- `Pb = berfading(EbN0, modulation, M, divorder)` — fading-channel
  BER (gated on Tier-5.x fading channel models, but the closed-form
  BER itself is just a Q-function evaluation).

**Why**: every BER plot in MATLAB docs ships a reference curve from
`berawgn` next to the simulated curve. Without it, the
"simulate-vs-theory" workflow doesn't exist.

**Effort**: 1 week. Bulk is the per-modulation closed-form table
(transcribing the ~20 entries in the user-guide BER chapter)
plus a robust `qfunc` (`Q(x) = 0.5·erfc(x/√2)` — `erfc` already
shipped, so this is one line).

### 4.9 Constellation visualization (numeric) ✅

**Scope**:
- `scatterplot(x)` — return-data form: a 1-line wrapper that
  produces `(real, imag)` pairs. The plotting itself is delegated
  to the existing Cairo backend or to user code.
- `comm.ConstellationDiagram` System Object — defer to §4.

**Effort**: 1 session for the numeric form.

### 4.10 Tier-2 closure summary

| Primitive | Effort | Status |
|---|---|---|
| `pammod` / `pamdemod` (4.1) | 0.5 wk | ✅ shipped (natural + Gray; real-line output) |
| `qammod` / `qamdemod` (4.2) | 1 wk | ✅ shipped (square M=4,16,64,256,1024 + rectangular cross-QAM M=8 [4×2] and M=32 [8×4]; hard / bit / LLR outputs; `UnitAveragePower` normalisation; max-log LLR with user-supplied noise variance) |
| `pskmod` / `pskdemod` (4.3) | 0.5 wk | ✅ shipped (natural + Gray; configurable initial phase) |
| `fskmod` / `fskdemod` (4.5) | 1 wk | 🔵 (deferred — not needed for the closure test) |
| `genqammod` / `genqamdemod` (4.6) | 0.5 wk | ✅ shipped (nearest-Euclidean-distance demod on a user-supplied complex alphabet) |
| `rcosdesign` / `gaussdesign` (4.7) | 1 sess | ✅ shipped (RRC + full RC via the `shape` tag; unit-energy normalised; Gaussian sum-normalised) |
| `berawgn` / `bercoding` (4.8) | 1 wk | ✅ shipped (closed-form for PAM / PSK / QAM / DPSK / FSK-coherent / FSK-noncoherent; uses libc `erfc`. `bercoding` deferred to Tier-3 coded slice.) |
| `scatterplot` (4.9) | 1 sess | ✅ shipped (numeric form returning N×2 real (re, im) pairs) |
| `eyediagram` (4.9 sibling) | 1 sess | ✅ shipped (n × num_traces matrix; real and complex inputs supported via magic-tag dispatch — `eyediagram(real(rxSig), 2*sps)` / `eyediagram(rxSig, 2*sps)` both work; `examples/rf/scatter_eye_qpsk.m` exercises the canonical QPSK pulse-shape → AWGN → inspect pipeline) |
| Closure test (`examples/comm/ber_qam_montecarlo.m`) | — | ✅ shipped (16-QAM Monte-Carlo with `berawgn` overlay — sim BER tracks theory within ~10% relative from 4 dB Eb/N0 onward at 20 k symbols/point) |

**Total**: ~3 weeks. End of Tier 2, this works:

```matlab
M = 16;                              % 16-QAM
data = randi([0 M-1], 1000, 1);      % source symbols
x = qammod(data, M, 'gray', ...
           'UnitAveragePower', true);% modulate
y = awgn(x, 20);                     % AWGN at 20 dB SNR
data_hat = qamdemod(y, M, 'gray');
[nerr, ber] = biterr(data, data_hat, log2(M));
disp(ber);
% reference: berawgn(20 - 10*log10(log2(M)), 'qam', M)
```

Every line lights up.

---

## 5. Tier 3 — channel coding (~4 weeks)

CRC + convolutional + block coding. Sits cleanly on Tier 1 (bit
sources). Required for any "real" comm sim — uncoded BER is rarely
the answer.

### 5.1 CRC — `comm.CRCGenerator` / `comm.CRCDetector` ✅

**Scope**:
- `g = comm.CRCGenerator('Polynomial', 'z^16 + z^12 + z^5 + 1')` —
  classdef System Object. Methods: `step(g, bits)` returns
  `[bits; crc]`. Reset semantics across `step` calls.
- `d = comm.CRCDetector(...)` — returns `[bits, err]` with `err`
  boolean.
- Functional alternative `crc.generator` is legacy; can stub later.

**Why**: System-Object surface unlocks the rest of Tier 3 / 4. CRC
is the simplest System Object — straightforward state (none across
calls except polynomial config) and a tight test surface. Use it
to validate the System-Object infrastructure.

**Effort**: 1 week if we treat the System-Object machinery as new
work. The classdef bookkeeping (constructor, properties,
operator-less methods, `release` / `reset` semantics) is the bulk;
GF(2) polynomial division is a 1-page algorithm.

**Architectural note**: `comm.*` System Objects use the same field-
store lowering path as CST's `tf` (CST roadmap §12). The recorded
verifier-mismatch bug in `LowerTensorOps.cpp:1708` blocks both
arcs. **Fix required before any System Object lands** — see §10.

### 5.2 Convolutional codes — `poly2trellis` / `convenc` / `vitdec` ✅

**Scope**:
- `t = poly2trellis(constraintLength, polynomials)` — trellis
  struct (`numInputSymbols`, `numOutputSymbols`, `numStates`,
  `nextStates`, `outputs`).
- `code = convenc(msg, t)` — encoder, single state-machine pass.
- `decoded = vitdec(code, t, tblen, opmode, dectype)` —
  Viterbi decoder. `opmode` ∈ `{'trunc', 'cont', 'term'}`,
  `dectype` ∈ `{'unquant', 'soft', 'hard'}`.

**Why**: load-bearing across all the wireless standards in the
toolbox. `[171,133]_8` rate-1/2 K=7 is the canonical example.

**Effort**: 2 weeks. `poly2trellis` is bookkeeping; `convenc` is
straightforward; `vitdec` (with traceback, soft metrics, and the
three operation modes) is the bulk.

**Test corpus**: encode-decode roundtrip with no errors must
preserve bits exactly; with random AWGN at given Eb/N0, BER must
match `bercoding(EbN0, 'conv', 'soft', t)` reference within 0.5 dB.

### 5.3 Block codes — Hamming, BCH, Reed-Solomon 🟡

**Scope**:
- `[parmat, genmat] = hammgen(m)` — Hamming `(2^m-1, 2^m-1-m)`
  parity / generator matrices.
- `code = encode(msg, n, k, 'hamming/binary')` (functional form);
  `decode` inverse.
- `code = bchenc(msg, n, k)` / `decoded = bchdec(code, n, k)` — BCH
  binary codes.
- `code = rsenc(msg, n, k)` / `decoded = rsdec(code, n, k)` —
  Reed-Solomon over GF(2^m). The user-guide chapter is the most
  comprehensive — punctures, shortening, erasures, GF helpers.
- GF(2^m) helpers: `gf(x, m)`, `gf` overloaded `+ - * /`,
  `gflineq`, `gfconv`, `gfdeconv`, `gfdiv`, `gfminpol`.

**Why**: foundational. RS is canonical for bursty channels.

**Effort**: 2 weeks. The Galois-field helpers (`gf` classdef with
operator overloads in GF(2^m)) are the bulk; `bchenc`/`rsenc` sit
on top in ~2 sessions each. **Galois-field elements need a new
runtime descriptor** — the simplest is a `matlab_mat` with an
extra side-channel field carrying `m` and the primitive
polynomial. Plan for a small new descriptor type rather than
overloading the int dtype.

### 5.4 LDPC / Turbo / Polar — ✅ shipped (function-form)

Originally carved out as a Tier-7 stretch.  Function-form
implementations of all three modern code families landed in one
slice:

- **Polar**: `polarEncode(u, N)` Arikan butterfly + `polarSCdecode(llr,
  frozen_mask, N)` recursive SC decoder.  Frozen-mask is caller-
  supplied (the 3GPP NR reliability sequence is a separate lookup-
  table follow-on).  Zero-noise round-trip exact at N up to 128.
- **LDPC**: `ldpcEncode(msg, P)` systematic from a k × (n−k) parity
  portion + `ldpcDecodeMS(llr, H, max_iter)` flooding-schedule
  min-sum BP decoder.  Generic / 5G NR base matrices are caller-
  supplied lookup tables.
- **Turbo**: `turboEncode(msg, trellis, perm)` PCCC over a
  `poly2trellis`-built trellis + caller-supplied interleaver
  permutation.  `turboDecode(...)` is canonical iterative max-log-
  MAP / BCJR with extrinsic LLR exchange.

The classdef System-Object surface (`comm.LDPCEncoder` etc.) stays
gated on the SO lowering fix.

Verified at SNR = 5 dB on a 64-bit message: uncoded BPSK 2 errors /
Polar (128, 64) 0 / Turbo PCCC 0 / LDPC (6, 3) 0.

### 5.5 Interleavers ✅

**Scope**:
- `comm.BlockInterleaver(perm)` / `comm.BlockDeinterleaver`.
- `comm.MatrixInterleaver(rows, cols)`.
- `comm.ConvolutionalInterleaver(rows, slope)`.
- Function form: `intrlv(data, perm)`, `deintrlv(data, perm)`.

**Effort**: 1 week. Bookkeeping over array reshape / shuffle. The
convolutional variant has internal state across `step` calls.

### 5.6 Tier-3 closure summary

| Primitive | Effort | Status |
|---|---|---|
| CRC System Objects (5.1) | 1 wk | ✅ shipped (`CommCRCGenerator` / `CommCRCDetector` classdef System Objects wrapping the function-form `crcGenerate` / `crcCheck` / `crcStrip`; validates the SO infrastructure) |
| CRC function-form (5.1 legacy) | 1 sess | ✅ shipped (`crcGenerate` / `crcCheck` / `crcStrip` — sidesteps the SO surface) |
| `poly2trellis` / `convenc` / `vitdec` (5.2) | 2 wk | ✅ shipped (function-form trellis struct + state-machine encoder + hard-decision Viterbi with traceback; opmode 0 trunc / 1 term, dectype 0 unquant / 1 hard. Soft-decision Viterbi stays for the Tier-4 follow-on.) |
| `oct2dec` bridge (5.2 helper) | 1 sess | ✅ shipped (decimal-from-octal-decimal converter so `oct2dec(171) = 121` lets users transcribe textbook generators verbatim) |
| Hamming binary codes (5.3) | 0.5 wk | ✅ shipped (`hammgenParity` + `hammingEncode` / `hammingDecode` — single-error correction; verified at every bit position) |
| BCH / RS + `gf` helpers (5.3) | 2 wk | 🔵 — needs a new `gf(2^m)` typed runtime descriptor |
| Interleavers (5.5) | 1 wk | ✅ shipped (function-form `intrlv` / `deintrlv` block interleaver; convolutional / matrix variants stay deferred) |
| Closure test (`ber_coded_vs_uncoded.m`) | — | ✅ shipped (uncoded vs Hamming(7,4) vs (171,133)₈ K=7 convolutional over BPSK + AWGN — conv beats uncoded ~2× at Eb/N0 = 7 dB) |

**Total**: ~6 weeks for the SO-free subset (shipped today) + ~2 wk for
BCH/RS (deferred to a follow-on) + ~1 wk for the CRC System-Object
surface once the SO lowering fix lands. Lights up all coded BER
curves up to convolutional + Hamming; RS / BCH coverage waits on the
`gf` descriptor.

---

## 6. Tier 4 — equalization, synchronization, RF impairments (~4 weeks)

### 6.1 Adaptive equalization — LMS, RLS ✅

**Scope**:
- `comm.LinearEqualizer(...)` — adaptive LMS / RLS / CMA linear
  equalizer.
- `comm.DecisionFeedbackEqualizer(...)` — DFE.
- Lower-level: `dsp.LMSFilter` (Signal Processing Toolbox cousin —
  defer if SPT roadmap defers it).
- Function-style `lms(...)` / `rls(...)` — per-call updates against
  a reference signal.

**Effort**: 2 weeks. Adaptive-filter inner loops are tight (one
multiply-add chain per tap per sample); the bookkeeping is in the
training-mode / decision-directed-mode switching and the
convergence-detection heuristics.

### 6.2 Phase / frequency synchronization ✅

**Scope**:
- `comm.CarrierSynchronizer(...)` — Costas-loop-style PLL.
- `comm.SymbolSynchronizer(...)` — Mueller-Müller / Gardner timing.
- `comm.PreambleDetector(...)` — frame sync via cross-correlation.
- Lower-level: `comm.PLL`, `comm.CoarseFrequencyCompensator`.

**Effort**: 2 weeks. PLL is a 5-line loop; the work is in the
loop-filter design helpers and the System-Object state machinery.

### 6.3 RF impairments ✅

**Scope**:
- `comm.PhaseFrequencyOffset(...)` — apply phase / frequency offset.
- `comm.PhaseNoise(...)` — colored phase noise.
- `comm.IQImbalanceCompensator(...)`, `iqimbal(x, gain, phase)`.
- `comm.MemorylessNonlinearity(...)` — power-amplifier nonlinearity.

**Effort**: 1 week. Each is a thin DSP block; phase noise is the
only one with non-trivial state (FIR shaping of white Gaussian).

### 6.4 Tier-4 closure summary

| Primitive | Effort | Status |
|---|---|---|
| Adaptive equalization (6.1) | 2 wk | ✅ shipped (`lms` / `rls` / `cma` / `dfe` function-form; complex CMA carved out as a Tier-5 follow-on; `comm.LinearEqualizer` / `comm.DecisionFeedbackEqualizer` System Objects gated on the SO lowering fix) |
| Carrier / symbol / frame sync (6.2) | 2 wk | ✅ shipped (`costasPll` for BPSK / QPSK / M-PSK, `symbolSyncMM` Mueller-Müller timing, `preambleDetect` cross-correlation peak; `comm.CarrierSynchronizer` / `comm.SymbolSynchronizer` / `comm.PreambleDetector` System Objects gated on the SO fix) |
| RF impairments (6.3) | 1 wk | ✅ shipped (`phaseFreqOffset`, `iqimbal`, `memorylessNl` with cubic-clipper + Saleh + Rapp + Ghorbani models via numeric `model_code`, `phaseNoise`) |
| Soft-decision Viterbi (Tier-3 follow-on parked here) | 0.5 wk | ✅ shipped (`vitdecSoft(llr, trellis, tblen, opmode)` — max-log-MAP path-metric Viterbi; ~3 dB gain over hard-decision at the Eb/N0 = 5 dB operating point) |
| Closure test (`ber_soft_vs_hard.m`) | — | ✅ shipped (soft vs hard Viterbi BER curves: hard 0.120 / soft 0.0051 at Eb/N0 = 5 dB) |

**Total**: ~5 weeks. Lights up the receiver-side post-channel
processing chain. The classdef System-Object variants of every entry
in this tier stay gated on the SO lowering fix; the function-form
surface shipped here is feature-complete against the canonical
textbook workflows.

---

## 7. Tier 5 — OFDM, MIMO, fading channels (~5 weeks)

### 7.1 OFDM modulation — `comm.OFDMModulator` / `comm.OFDMDemodulator` ✅

**Scope**:
- `comm.OFDMModulator(FFTLength, CyclicPrefixLength, NumGuardBandCarriers, PilotIndices, ...)`.
- Manual subcarrier mapping → IFFT → CP insertion → output samples.
- Demodulator inverse.
- `ofdmmod` / `ofdmdemod` functional form.

**Why**: the dominant modern PHY. 5G NR, Wi-Fi, LTE, DVB-T/H all
use OFDM at the air interface.

**Effort**: 1.5 weeks. FFT engine is shipped; bulk is the
subcarrier-mapping / pilot-allocation / guard-band bookkeeping.

### 7.2 Fading channels — `comm.RayleighChannel` / `comm.RicianChannel` ✅

**Scope**:
- Rayleigh / Rician multi-path fading channels with configurable
  Doppler spectrum, max Doppler shift, path delays, average path
  gains.
- Sum-of-sinusoids or filtered-Gaussian generator.
- `comm.MIMOChannel` — spatial-correlation-aware MIMO fading.

**Effort**: 2 weeks. The Doppler-spectrum filter design (Jakes /
Gauss / flat / restricted-Jakes / rounded) is the bulk; the MIMO
spatial correlation matrix is straightforward complex matrix
multiply on top.

### 7.3 MIMO algorithms ✅

**Scope**:
- `comm.OSTBCEncoder` / `comm.OSTBCCombiner` — Alamouti and 3/4-rate
  orthogonal STBC.
- `comm.SphereDecoder` — ML detection via sphere decoding.
- Beamforming / precoding helpers (function-form): `mldetect`,
  `precoding`. Most of these are short matrix-algebra one-liners
  once eig / svd are available (✅ shipped).

**Effort**: 1.5 weeks. Sphere decoding is the only non-trivial
piece (lattice reduction-based search).

### 7.4 Tier-5 closure summary

| Primitive | Effort | Status |
|---|---|---|
| OFDM (7.1) | 1.5 wk | ✅ shipped (`ofdmmod` / `ofdmdemod` function-form over `Nfft × Nsym` complex; CP insert/strip; pilots / guards are caller-side compositions; `comm.OFDMModulator` / `OFDMDemodulator` System Objects gated on SO fix) |
| Fading channels (7.2) | 2 wk | ✅ shipped (`rayleighChannel` multi-path Jakes sum-of-sinusoids; `ricianChannel` LOS + scatter decomposition with K_dB; `comm.RayleighChannel` / `RicianChannel` System Objects gated on SO fix; complex MIMO fading via `comm.MIMOChannel` deferred) |
| MIMO (7.3) | 1.5 wk | ✅ shipped (Alamouti `ostbcEncode` 2-Tx + `ostbcCombine` MRC receiver; `mlDetect` per-symbol Euclidean nearest; `comm.OSTBCEncoder` / `OSTBCCombiner` System Objects + `comm.SphereDecoder` + complex ZF/MMSE detect gated on SO fix and the future complex-LU runtime) |
| Closure tests (`ofdm_awgn.m`, `alamouti_diversity.m`) | — | ✅ shipped (OFDM AWGN loopback 0 symbol errors at 15 dB; Alamouti combine recovers the input symbols 0 errors at 10 dB SNR vs single-Tx 0.27% baseline) |

**Total**: ~5 weeks. Lights up the modern wireless PHY surface. The
classdef System-Object variants of every entry in this tier stay
gated on the SO lowering fix. Sphere decoding, MIMO ZF / MMSE
detection on general H, and the `comm.MIMOChannel` spatial-correlation
fading are follow-ons (need the complex-LU runtime).

---

## 8. Tier 6 — spreading, propagation, source coding (~3 weeks, stretch)

### 8.1 Spreading sequences ✅ shipped (function-form)

- `pnSequence(poly_int, init_int, length, output_mode)` — LFSR PN
  generator. Polynomial passed as an integer mask (the implicit
  leading 1 is the highest set bit). `output_mode` ∈ {0 = bits, 1 =
  bipolar}.
- `goldSequence(poly1, poly2, init1, init2, length, output_mode)` —
  XOR of two preferred-pair PN streams.
- `hadamard(n)` — Sylvester-form Hadamard matrix; `n` snaps up to
  the next power of 2. `walshCode(n, k)` returns the k-th row (1-based).
- `comm.KasamiSequence` — 🔵 deferred; needs the m-sequence-
  decimation helper not yet shipped.
- `comm.PNSequence` / `comm.GoldSequence` System Objects — gated on
  SO fix.

### 8.2 Source coding ✅ shipped (function-form)

- `quantiz(sig, partition, codebook)` returns the codebook indices
  column; `quantizApply(idx, codebook)` looks up the codebook values.
  Split form matches MATLAB's `[indx, quant] = quantiz(...)` triple.
- `lloydsQuant(sig, init_codebook, max_iter, tol)` — iterative
  midpoint-partition + per-region mean refinement; stops on
  max-codebook-shift tolerance.
- `compandMu(x, mu, V, dir)` / `compandA(x, A, V, dir)` — G.711
  μ-law and A-law companders; `dir` 0 = compress, 1 = expand. Round-
  trip is exact to machine precision.
- `dpcmEncode(sig, partition, codebook)` / `dpcmDecode(idx, codebook)` —
  DPCM with first-order predictor; caller designs the residual
  partition/codebook. `dpcmopt` (codebook design) is a 1-session
  follow-on.

### 8.3 Galois-field-based ARQ / hybrid ARQ — defer 🔴

Carved out — see §12.

### 8.4 Ray-tracing / propagation 🔴

The user-guide chapters on ray-tracing-based propagation
(`propagationModel`, `raytrace`, `siteviewer` etc.) require:
- Geographic / GIS data ingestion (DTED / DEM rasters, OSM road
  data).
- 3-D scene geometry (urban building extrusions).
- Native interactive map visualization.

Carved out as a hard 🔴 — see §12.

---

## 9. RF Toolbox companion (~6 weeks)

RF Toolbox is a separate MathWorks product (402-page UG) but shares
infrastructure with Comm: complex-matrix algebra, classdef System
Objects, polynomial / rational fitting, frequency-response
evaluation. Most of its programmable surface is **closed-form linear
algebra over S/Y/Z/H/G/ABCD/T network parameters** plus a
text-format I/O layer (Touchstone, AMP files) and a small classdef
hierarchy (`sparameters`, `rfckt.*`, `rfmodel.*`).

The arc decomposes into four internal tiers (RF-Tier-1 through
RF-Tier-4), independent of the Comm tier numbering in §2–§7.

### 9.1 RF-Tier-1 — Network parameter objects + Touchstone I/O (~1 week)

The foundation. Almost no RF Toolbox function lights up without
these.

#### 8.1.1 Network parameter classdefs ✅

**Scope** — eight network parameter object types:
- `sparameters(s, freq, z0)` — most-used; `s` is `[NumPorts ×
  NumPorts × NumFreqs]` complex.
- `yparameters`, `zparameters`, `hparameters`, `gparameters`,
  `abcdparameters`, `tparameters`.
- Construction also from **file**: `sparameters('amp.s2p')` —
  see §8.1.3.
- Properties (read-only after construction): `Parameters` (3-D
  complex), `Frequencies` (real column), `NumPorts`, `Impedance`
  (default 50 Ω).

**Effort**: ~3 sessions. Each is a thin classdef wrapping the
parameter cube + scalar metadata. Construction-time validation
(parameter shape matches `NumPorts²·NumFreqs`).

**Architectural prerequisite**: same System-Object lowering fix as
Comm Tier 3 (CST §12 / §11.1). Network parameter objects use the
field-store path that today fails verifier when monomorphization
propagates concrete tensor types into `_set_f64` calls. Until that
fix lands, RF Toolbox stalls at the same point Comm does.

#### 8.1.2 Network parameter conversions ✅

**Scope** — all-to-all conversions among the seven 2-port
representations (S, Y, Z, H, G, ABCD, T):
- `s2y`, `s2z`, `s2h`, `s2g`, `s2abcd`, `s2t`, and inverses (`y2s`,
  `z2s`, `h2s`, `g2s`, `abcd2s`, `t2s`).
- Cross-conversions (`y2z`, `z2y`, etc.) — most go through `s` as
  the canonical hub.
- Reference-impedance change: `newref(spar, z0_new)`.
- Single-ended ↔ mixed-mode: `s2sdd`, `s2sdc`, `s2scc`, `s2scd`.

**Algorithm**: closed-form per-frequency `2×2` (or `N×N` for
multiport) matrix algebra. Documented in any microwave textbook;
implementation is mechanical.

**Effort**: ~3 sessions. ~28 conversion entries; once one direction
of each pair lands the inverse is symmetric. The 4-port mixed-mode
splitter is the only non-trivial bit.

#### 8.1.3 Touchstone file I/O ✅

**Scope**:
- `sparameters('file.s2p')` — read Touchstone v1 (s1p, s2p, s3p,
  s4p, …, sNp) and Touchstone v2 (`.ts` files).
- `rfwrite(spar, 'file.s2p')` — write Touchstone v1; v2 as a
  follow-on.
- AMP file format (RF Toolbox UG §4): text-format amplifier data
  files with S/Y/Z parameters + Noise/IP3/Power blocks.
  `read(rfdata.data, 'amp.amp')` — defer the AMP read for now;
  Touchstone is the workhorse format.

**Why**: every real RF data set ships as Touchstone. Without the
parser, users can't load any vendor data sheet.

**Effort**: ~3 sessions. Touchstone v1 is a simple text format
(option line `# GHz S MA R 50` + frequency rows). Bulk is parsing
the option line variants (frequency unit ∈ {Hz, kHz, MHz, GHz},
parameter type ∈ {S, Y, Z, G, H}, data format ∈ {DB, MA, RI},
reference impedance) and unpacking the per-frequency data into the
3-D parameter cube with the correct row/column ordering (s2p uses
`[s11 s21 s12 s22]` row order — the **transposed** convention vs
sNp where N>2; this is a known pitfall).

### 9.2 RF-Tier-2 — Frequency-domain analysis + RF budget (~1.5 weeks)

The first user-visible RF slice: take a Touchstone file, compute
gain / stability / VSWR / cascaded NF.

#### 8.2.1 Closed-form S-parameter analyses ✅

**Scope**:
- `tf = s2tf(sparobj, zs, zl, z0)` — voltage transfer function
  `Vout/Vin`. Default `zs = zl = z0 = 50 Ω`.
- `gamma = gammain(sparobj, zl)`, `gamma = gammaout(sparobj, zs)`
  — input/output reflection coefficients.
- `r = vswr(gamma)` — voltage standing wave ratio
  `(1+|Γ|)/(1−|Γ|)`.
- `[gt, ga, gp, gmu] = powergain(sparobj, zs, zl, type)` —
  transducer / available / operating / unilateral gain. `type` ∈
  `{'Gt', 'Ga', 'Gp', 'Gmsg', 'Gmag', 'Gu'}`.
- `[k, b1] = stabilityk(sparobj)` — Rollett stability factor
  (`k > 1, |Δ| < 1` → unconditionally stable).
- `mu = stabilitymu(sparobj, type)` — Edwards-Sinsky stability
  measure (`type ∈ {'mu1','mu2'}`); single-parameter unconditional
  stability test.

**Effort**: ~1 week. All entries are closed-form per-frequency
formulas (1–3 lines of complex arithmetic each). Bulk is the
multi-return shape and the per-frequency output orientation
matching MATLAB.

#### 8.2.2 Cascade and port operations ✅

**Scope**:
- `cascadesparams(s1, s2, ..., k)` — cascade `N` S-parameter
  objects (k = number of inner-port connections).
- `snp2smp(sparobj, ports)` — extract m-port from n-port; e.g.
  pull a 2-port out of a 4-port by selecting two ports and
  terminating the rest in `Z₀`.
- `gamma2z`, `z2gamma` — reflection coefficient ↔ port impedance.
- `deembedsparams` — de-embed fixture from measured S-parameters.

**Effort**: ~3 sessions. `cascadesparams` is the largest
piece (T-parameter chain via `s2t` → matrix multiply →
`t2s`); `snp2smp` is the standard "Schur-complement of the
N-port S-matrix at the terminated ports" formula.

#### 8.2.3 RF budget — Friis cascade ✅

**Scope**:
- `b = rfbudget(stages, freq, inputpower, bandwidth)` — `stages`
  is an array of `nport` / `amplifier` / `mixer` / `modulator`
  blocks. Returns a budget table:
  - Cascaded gain (per stage and total).
  - Cascaded noise figure via Friis: `NF_total = NF_1 + (NF_2 −
    1)/G_1 + (NF_3 − 1)/(G_1·G_2) + …`.
  - Cascaded IP3 / IP2.
  - Output / input power, SNR, available power per stage.
- `nport(spar)` — block constructor wrapping an `sparameters`
  object as a passive RF stage.
- `amplifier(...)`, `mixer(...)`, `modulator(...)`,
  `rfantenna(...)` — block constructors.

**Why**: this is the canonical "system-level RF chain analysis"
that RF Toolbox is best known for. Friis cascade is closed form;
the bulk is the block-constructor classdef hierarchy.

**Effort**: ~3 sessions. Each block is a small classdef; the
Friis solver is one numerical pass over the stages.

**Carved out of Tier-2**: Harmonic Balance solver (multi-tone
nonlinear RF budget) — needs Newton-Krylov on a circuit-tree
nonlinear residual; multi-week. The **linear** Friis path covers
~80% of the practical RF budget surface. See §12.

### 9.3 RF-Tier-3 — Rational fitting + transmission lines (~3.5 weeks)

#### 8.3.1 Rational fitting — `rationalfit` ✅

**Scope**:
- `mdl = rationalfit(freq, data)` — fit measured frequency-domain
  data with a rational function `H(s) = sum_k r_k/(s − p_k) + d`.
- Returns `rfmodel.rational` object with `A` (poles), `C`
  (residues), `D` (direct term), `Delay`, `Order`.
- Tolerance / order / weight controls: `'NPoles'`, `'WeightParam'`,
  `'TendsToZero'`, `'IterationLimit'`, `'TolError'`.
- `freqresp(mdl, freq)` — evaluate fitted rational at frequencies.
- `[y, t] = timeresp(mdl, u, ts)` — time-domain via state-space
  realization of the partial-fraction form.
- `passivity(mdl)` — test/enforce passivity (HSV-like indicator
  on the system matrix).

**Algorithm**: Vector Fitting (Gustavsen & Semlyen 1999). Iterative
pole relocation: each iteration solves a linear least-squares
problem in the unknowns `(c_k, d, σ_k)` given the previous pole
estimate; new poles are extracted as the eigenvalues of a small
companion matrix. Converges in 5-10 iterations on smooth data.

**Why**: lets a user fit measured `S₁₁(f)` with an LTI rational
suitable for time-domain simulation (TDR / TDT). Without it, all
S-parameter data stays frequency-domain.

**Effort**: ~2 weeks. The vector fitting iteration is short (~50
lines) but needs the eigendecomposition of a non-symmetric
companion matrix — uses the **non-symmetric `eig`** that already
shipped (CST Tier-1.1 ✅, 1-return form). The state-space
realization for `timeresp` reuses CST's `lsim_ss` (✅ shipped).

#### 8.3.2 Time-domain RF — TDR / TDT ✅

**Scope**:
- `s2tdr(sparobj)` — Time-Domain Reflectometry response.
- `s2tdt(sparobj)` — Time-Domain Transmission.
- Fed by `rationalfit` → `timeresp`.

**Effort**: ~3 sessions. Wrappers over `rationalfit` + a step
input through `timeresp`.

#### 8.3.3 Transmission line objects ✅

**Scope** — `rfckt.*` 2-port S-parameter generators
parameterized by physical geometry:
- `rfckt.txline` — generic transmission line (Z₀, εᵣ, length).
- `rfckt.coaxial` — coaxial cable (inner/outer radii, dielectric).
- `rfckt.microstrip` — microstrip on PCB (width, height, εᵣ).
- `rfckt.cpw` — coplanar waveguide.
- `rfckt.parallelplate`, `rfckt.twowire`, `rfckt.rlcgline` — others.
- `analyze(rfobj, freq)` — extract S-parameters at requested freqs.

**Why**: lets users build first-pass interconnect models from
geometry without measured data. The closed-form formulas (Wheeler,
Hammerstad-Jensen for microstrip) are textbook.

**Effort**: ~1 week. Each transmission-line type is a closed-form
characteristic-impedance + propagation-constant calculation +
ABCD-matrix synthesis. Bulk is correctly handling the loss /
dispersion tail per geometry.

### 9.4 RF-Tier-4 — Matching networks, RF circuits, Smith chart numerics (~2 weeks, stretch)

#### 8.4.1 Matching network design ✅

**Scope**:
- `mn = matchingnetwork(rfobj, freq, ...)` — automatic L / T / Pi
  matching network synthesis.
- `'Type'` ∈ `{'L', 'T', 'Pi'}`; `'Topology'` for stub vs lumped.
- Lumped-element value calculation via standard quadratic-equation
  closed forms.

**Effort**: ~1 week. Closed-form per-topology algebra; bulk is
the topology / impedance-direction selection logic.

#### 8.4.2 RF circuit object hierarchy (subset) ✅

**Scope** (priority order):
- `rfckt.amplifier` — wraps S/Noise/Nonlinearity data.
- `rfckt.mixer` — frequency-translating block.
- `rfckt.passive` — generic passive 2-port.
- `rfckt.cascade(blocks)` — series cascade.
- `rfckt.parallel(blocks)` — parallel composition.
- `rfckt.series` / `rfckt.shunt` — connection helpers.
- `rfckt.lcbandpasspi`, `rfckt.lcbandpasstee`, `rfckt.lclowpasspi`,
  `rfckt.lclowpasstee`, `rfckt.lchighpass*`, `rfckt.lcbandstop*`
  — LC filter circuits with closed-form S-parameters.
- `rfckt.rlcgline` — already covered in §8.3.3.

**Effort**: ~1 week. Each `rfckt.*` is a small classdef with an
`analyze` method that produces S-parameters at requested freqs.

#### 8.4.3 Smith chart numerics ✅

**Scope**:
- `[gamma, z, y] = gamma2z(gamma, z0)` etc. — already in §8.2.2.
- Numeric grid generation for Smith-chart overlays (constant-r,
  constant-x circles in the Γ-plane).
- Interactive Smith Chart Tool app — **carved out** (§12).
- Static Smith chart plot via Cairo backend — feasible follow-on,
  not committed.

**Effort**: ~3 sessions for the numeric grid generators (the
interactive tool is the bulk of MATLAB's effort; we skip it).

### 9.5 RF-Tier closure summary

| Primitive | Effort | Status |
|---|---|---|
| `RFSparameters` 2-port classdef (S11/S12/S21/S22/Frequencies/NumPorts/Impedance properties) (8.1.1) | 3 sess | ✅ shipped (2-port) |
| Sibling network-parameter classdefs (Y / Z / H / G / ABCD / T) (8.1.1) | 3 sess | ✅ shipped — `RFYparameters` / `RFZparameters` / `RFHparameters` / `RFGparameters` / `RFAbcdparameters` / `RFTparameters` skeleton classdefs parallel `RFSparameters`. Hold the corresponding 2-port matrix-valued parameters + NumPorts / Impedance / Frequencies. Auto-detected by the prelude loader. |
| All-to-all conversions s↔y↔z↔h↔g↔abcd↔t (8.1.2) | 3 sess | ✅ shipped — full cross-conversion grid: `sparamS2y` / `sparamS2z` (2-port) + `sparamS2yN` / `sparamS2zN` (general N-port via 2N×2N real-equivalent complex matrix algebra), `sparamS2h` + `sparamH2s`, `sparamS2abcd` + `sparamAbcd2s`, `sparamS2g` + `sparamG2s` (via H-inverse route), `sparamS2t` + `sparamT2s` (T-parameter chain). `tsYij` / `tsZij` / `tsHij` / `tsGij` / `tsTij` / `tsAbcdA-D` typed getters for any matrix entry. `gamma2z` / `z2gamma` Smith-chart impedance ↔ reflection helpers. |
| `snp2smp` port extraction | 3 sess | ✅ shipped (`snp2smp(data, ports, m)` — matched-termination Schur-complement simplification; non-matched terminations 🔵) |
| Touchstone v1 + v2 read + write (8.1.3) | 3 sess | ✅ shipped — v1 (.sNp) handles all common port counts with auto-detection from the `.sNp` extension; v2 (.ts) parser recognises `[Number of Ports]` / `[Reference]` / `[Two-Port Order]` / `[Network Data]` / `[End]` keywords on top of the v1 option-line parser.  `touchstoneRead` auto-selects on filename; `touchstoneWrite` writes any N-port v1.  `tsSij(data, i, j)` generic getter for any port pair. |
| Closed-form analyses: `s2tf`, `gammain`, `vswr`, `powergain`, stability (8.2.1) | 1 wk | ✅ shipped (`gammaIn` / `gammaOut` / `vswr` / `powerGain` Gt-Ga-Gp / `stabilityK` Rollett / `stabilityMu` Edwards-Sinsky / `s2tf`) |
| Cascade / `snp2smp` / de-embed (8.2.2) | 3 sess | ✅ shipped — 2-port `cascadeSparams2` via T-parameter chain, N-port `cascadeSparamsN` (diagonal approximation for weakly-coupled networks), full Redheffer star-product `cascadeSparamsNFull` (k = N/2, exact). `snp2smp` (matched terminations) + `snp2smpZ` (Schur complement with arbitrary non-z0 terminations at dropped ports). |
| `rfbudget` (Friis solver) (8.2.3) | 3 sess | ✅ shipped (`rfbudgetFriis` returns cascaded Gain_dB / NF_dB / IP3_in_dBm / OutputPower / NoiseFloor / SNR / ThermalNoise struct) — **closes RF-Tier-2 MVP** |
| `rationalfit` + Vector Fitting (8.3.1) | 2 wk | ✅ shipped — Gustavsen-Semlyen v2 with both real and complex-conjugate pole pairs. The `(α, β)` pole representation supports either kind; the relocation matrix M uses the real-arithmetic `[α β; -β α]` block form for complex pairs, eig auto-classifies eigenvalues, and the output stores complex Poles + complex Residues columns. `freqresp` / `passivity` / `timeresp` all consume complex poles via the layout-magic-aware getters. |
| `passivity(mdl)` indicator (8.3.1 follow-on) | 3 sess | ✅ shipped (max \|H(jω)\| over a dense log-spaced sweep) |
| `s2tdr` / `s2tdt` + `timeresp` time-domain (8.3.2) | 3 sess | ✅ shipped (per-pole ZOH discretization + step-driven step response) |
| Transmission line objects (8.3.3) | 1 wk | ✅ shipped (function-form `rfckt_txline` / `rfckt_coaxial` / `rfckt_microstrip` (Hammerstad-Jensen) / `rfckt_cpw` / `rfckt_parallelplate` / `rfckt_twowire` — closed-form Z₀ + propagation, lossless; rfckt classdef wrappers deferred) |
| `matchingnetwork` (8.4.1) | 1 wk | ✅ shipped (L-section `matchingnetwork`, T-section `matchingnetworkT(..., q_target)`, Pi-section `matchingnetworkPi(..., q_target)` — all closed-form via dual-L-section cascade for T/Pi at a virtual-impedance node R_V) |
| `rfckt.*` classdef hierarchy subset (8.4.2) | 1 wk | ✅ shipped — `RFCktAmplifier` / `RFCktMixer` / `RFCktPassive` block skeletons + `RFCktCascade` / `RFCktParallel` / `RFCktSeries` / `RFCktShunt` combinator skeletons. Cascade composition via function-form `rfbudgetFriis` over per-block NF / Gain / IP3 columns. LC filter circuits: `rfckt_lcfilter(topology, comp1, comp2, freqs, z0)` for 3-element Lowpass/Highpass (Tee+Pi); `rfckt_lcfilter4(topology, L1, C1, L2, C2, freqs, z0)` for 4-element Bandpass/Bandstop (Tee+Pi). `rfmodel.rational` value classdef shipped as `RFRational` (A / C / D / Delay / Order / Error properties). Block analyze helpers: `rfAnalyzeAmplifier` / `rfAnalyzePassive` / `rfAnalyzeSeries` / `rfAnalyzeShunt` synthesize 2-port S-param structs from the block's scalar property surface. |
| Mixed-mode 4-port `s2sdd` / `s2sdc` / `s2scc` / `s2scd` (8.1.2 follow-on) | 3 sess | ✅ shipped (`sparamS2smm(s11..s44, block_code)` — block_code 0=dd, 1=dc, 2=cd, 3=cc) |
| Smith chart numerics (8.4.3) | 3 sess | ✅ shipped (`smithGrid(r_norm, n_pts)` returns RCircle + UnitCircle complex columns + `smithRCircle` / `smithUnitCircle` typed-getters) — closes RF-Tier-4 numerics |

**Total**: ~6 weeks of focused sessions for full RF-Tier-1 → RF-
Tier-4 closure. **MVP slice (~3 weeks): SHIPPED.**  Today users can:
`data = touchstoneRead("amp.s2p"); s11 = tsS11(data); s12 = tsS12(data);
... ; v = vswr(gammaIn(s11,s12,s21,s22,50,data.Z0)); K = stabilityK(...); 
b = rfbudgetFriis(gains_dB, nfs_dB, ip3_dBm, p_in_dBm, bw_Hz);` — covers
~80% of the practical small-signal RF analysis workflow.
**+`rationalfit`** (8.3.1) at +2 weeks unlocks the
frequency-domain-to-time-domain bridge that's RF Toolbox's most
distinctive capability beyond closed-form S-parameter algebra.

**Multi-port follow-on (defer ~2 wk)**: extend the Touchstone parser
to s3p / s4p / sNp (the per-frequency block layout changes —
`[s11 s21 s12 s22]` row order for s2p but `[s11 s12 ... s1N]` etc
for N>2), add `snp2smp` port-extraction, `cascadesparams` general-N,
mixed-mode `s2sdd` / `s2sdc` / `s2scc` / `s2scd`, and the sibling
network-parameter classdefs (`Yparameters` / `Zparameters` / ...).

### 9.6 What RF Toolbox brings to Comm

The §5.3 RF impairments tier in the Comm side currently sketches
function-form `iqimbal` and minimal `comm.MemorylessNonlinearity`.
Once RF-Tier-2 lands, those become **richer** because `rfckt.amplifier`
+ `rfbudget` provide a structured way to inject realistic PA
nonlinearity / NF data sourced from vendor Touchstone / AMP files
into a Comm simulation. The cross-over points:

- A `comm.MemorylessNonlinearity` driven by Saleh / Rapp / Ghorbani
  parameters is the lightweight version; an `rfckt.amplifier` driven
  by AMP-file-measured AM/AM and AM/PM data is the heavyweight
  version. Ship both lanes; route through the same internal
  nonlinear-distortion kernel.
- `awgn` (§2.5) sets a noise floor at a user-chosen SNR; an
  `rfbudget`-derived NF gives the **physical** noise floor implied
  by the chain. Useful contrast when teaching link budgets.
- OFDM / 5G NR waveform passing through `rfckt.amplifier` ➜
  back-into-Comm-receiver is the canonical "I built a PHY, now
  what does a real PA do to it" workflow.

These integrations are work *items* but no **new** primitive — they
are wiring between the Comm and RF surfaces that already exist in
their respective tiers.

---

## 10. Antenna Toolbox — moved to dedicated roadmap

The Antenna Toolbox tier plan (ANT-Tier-1 catalog classdefs through
ANT-Tier-5 advanced MoM-PO / FMM / mutual-coupling surfaces) was
previously documented here as chapter §10. It has been **promoted to
a standalone roadmap** at
[`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md) for
discoverability and uniformity with the other per-toolbox roadmaps.

**Status (2026-05-17): ANT-Tier-2 MVP shipped** for the canonical
thin-dipole case via the closed-form induced-EMF method (Balanis
Eq. 8-60). Three runtime entries live in
`runtime/runtime_rf.cpp` (sharing the RF TU's helpers):

- `antennaWireSolve(L, a, n_segs, freq)` →
  struct{`Zin_re`, `Zin_im`, `S11_re`, `S11_im`, `VSWR`,
  `ReturnLoss_dB`}
- `antennaWirePattern(L, a, n_segs, freq, n_theta)` → far-field
  pattern + directivity
- `antennaWireSparameters(L, a, n_segs, freqs)` → RFSparameters-
  shaped struct for the RF-Toolbox bridge

Catalog classdefs `AntDipole` + `AntMonopole` ship with `design(ant,
freq)` + `antennaGain(ant, freq)` (textbook 2.15 / 5.15 dBi).
Remaining 10 classdef stubs are 🔵 ANT-Tier-1 follow-on.

**Next slice (ANT-Tier-2b)**: multi-wire MoM for Yagi-Uda / monopole-
over-ground / helix / loop / folded-dipole — ~3 sessions once the
Pocklington kernel scaling is fully nailed.

**ANT-Tier-3 (planar patch / spiral MoM, ~6 wk)**, **ANT-Tier-4
(arrays via element-pattern multiplication, ~2 wk)**, and the
**ANT-Tier-5 advanced surface** (dielectrics / hybrid MoM-PO / FMM /
mutual coupling) are 🔵 — see [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md)
for the full plan + execution order + carve-outs.

### 10.1 What Antenna Toolbox brings to RF, Comm, and Propagation

- **Antenna → RF Toolbox**: `sparameters(ant, freqs)` produces a
  `sparameters` object that drops directly into RF Toolbox cascades.
  Closes "design the antenna and then design the chain it feeds."
- **Antenna → Propagation**: the ANT-Tier-2 pattern function is the
  input to the directional hook in PROP-Tier-3 — see [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md)
  §4.
- **Antenna → Comm**: an antenna's pattern affects link budget and
  fading-channel parameterization. Once Comm Tier 5.2 (fading
  channels, ✅ shipped) and ANT-Tier-2 land together,
  `comm.RayleighChannel` can be parameterized by an antenna's far-
  field gain pattern.

## 11. REPL / Debug-side work (cross-cutting)

Most Comm + RF function returns are `matlab_mat *` and inherit the
matrix display path. The new surface is:

- **`comm.*` System Objects** — handle-shaped classdefs with
  persistent state across `step` calls. Conceptually closer to
  `dsp.SOSFilter` than to value-typed `tf`.
- **`sparameters` / `yparameters` / … / `tparameters`** (§8.1.1) —
  value-typed-feeling classdefs wrapping a 3-D complex parameter
  cube + frequency vector. Read-only after construction.
- **`rfckt.*`** circuit objects (§8.4.2) — handle-shaped classdefs
  with `analyze(rfobj, freq)` evaluation method.
- **`rfmodel.rational`** (§8.3.1) — value-typed rational-function
  fit result (poles, residues, direct term, delay).
- **Antenna catalog classdefs** (`dipole`, `monopole`, `patch*`,
  `yagiUda`, `helix`, `bowtie*`, `spiral*`, `pifa`, `horn*`,
  `loop*`, …) (§9.1) — value-typed classdefs holding geometry +
  material. `pattern(ant, freq)` returns a 2-D real matrix
  `[NumEl × NumAz]` of gain — inherits the matrix display path.
- **Antenna mesh objects** (§9.2.1, §9.3.1) — separate descriptor
  with vertex / edge / triangle tables; needs a custom inspector
  layout (mesh tables can have ~10⁴ rows).
- **Antenna current vectors** (§9.2.3) — complex column vector
  indexed by basis function; 3-D current density derivable on
  request. Inherits matrix display path with truncation.
- **Antenna array classdefs** (`linearArray`, `rectangularArray`,
  …) (§9.4.1) — wrap a single-element antenna + position table +
  weighting / steering parameters.

### 11.1 System-Object infrastructure (architectural prerequisite)

**Blocker shared with CST §3.1 (recorded 2026-05-09)**: the prelude-
loaded `tf` classdef in `runtime/cst_classdefs.m` triggers a
verifier-rejected mismatch when constructor monomorphization
propagates concrete tensor types into `obj.Field = param` field
stores routed through `matlab_obj_set_f64`. The fix paths
(documented in CST roadmap §12) are:
- (a) keep class methods polymorphic at the signature level AND box
  tensor args through `matlab_mat_from_tensor` at call sites; OR
- (b) post-monomorphization rewrite of `_set_f64` / `_get_f64`
  callees with non-f64 operands to their `_mat` counterparts.

Until this is fixed, **no System Object can ship**. CRC (4.1) is
the smallest System Object and is the natural test case once the
fix lands.

**Additional System-Object semantics needed**:
- `step(obj, ...)` method dispatch with persistent property mutation
  (the System Object pattern: properties hold state across calls).
- `release(obj)`, `reset(obj)`, `clone(obj)`.
- `isLocked(obj)` — whether the object has been stepped (which
  freezes the public-tunable properties).
- Discrete `setupImpl` / `stepImpl` / `resetImpl` / `releaseImpl`
  hook methods — defer the introspective form, but the mutation
  semantics need to land first.

Design choice: **handle-shape classdef with explicit state
properties is the simplest fit**. MATLAB's actual `matlab.System`
base class does fancier introspection (auto-generation of
`getDiscreteState`, validation hooks); a minimal subset that just
calls `step` and lets users mutate properties through assignment
is enough for the Tier-2/3/4/5 surface this roadmap targets.

**Effort to land System-Object infrastructure**: 1 week on top of
the §15 architectural fix.

### 11.2 Display

`disp(obj)` for a System Object has a canonical multi-line format:

```
  comm.RaisedCosineTransmitFilter with properties:

                          Shape: 'Square root'
        RolloffFactor: 0.2500
       FilterSpanInSymbols: 10
     OutputSamplesPerSymbol: 8
```

Implementation: a `disp` method on each classdef calling into a
runtime helper `matlab_comm_disp_<type>` (mirrors CST's
`matlab_cst_disp_<type>`). ~0.5 week for the first one; subsequent
types reuse the same renderer.

For `sparameters` / sibling network parameter objects, the
canonical display is a **dimensions-and-impedance summary** rather
than the full parameter cube:

```
  sparameters: S-parameters object

         Parameters: [2x2x401 double]
        Frequencies: [401x1 double]
          NumPorts: 2
          Impedance: 50 (Ω)
   FreqRange: [1 GHz, 6 GHz]
```

…with the underlying 3-D complex cube available via the inspector
(§9.3) on demand. Pretty-printing the full cube inline would dump
megabytes for typical measured data sets.

For `rfmodel.rational`:

```
  rfmodel.rational with properties:

         A: [12x1 double]            % poles
         C: [12x1 double]            % residues
         D: 0.0023                   % direct term
        Delay: 1.5e-09               % bulk delay
        Order: 12                    % # of poles
        Error: 0.0184                % L2 fit error
```

For `rfckt.*` circuit objects, follow the System-Object format:
list the construction-time properties, plus any cached `AnalyzedResult`
S-parameter object as a child node.

For antenna catalog objects (`dipole`, `patchMicrostrip`, etc.),
canonical display:

```
  dipole with properties:

         Length: 2
         Width: 0.05
         FeedOffset: 0
         Conductor: [1x1 metal]
         Tilt: 0
         TiltAxis: [1 0 0]
         Load: [1x1 lumpedElement]
         Tuner: []

  Use show, mesh, pattern, impedance, sparameters to analyze.
```

For `linearArray` / `rectangularArray`:

```
  linearArray with properties:

         Element: [1x1 dipole]
         NumElements: 4
         ElementSpacing: 0.5
         AmplitudeTaper: 1
         PhaseShift: 0
         Tilt: 0
         TiltAxis: [1 0 0]
```

The mesh / current / pattern *outputs* are large numerical arrays —
inherit the matrix display path with truncation.

### 11.3 DAP variable inspector

For each System Object, expose its top-level properties (numeric +
string + struct) as expandable children in the Locals panel. Each
child renders its underlying `matlab_mat *` (or scalar) using the
existing matrix renderer. ~1 session per type once the inspector
hook for handle-classdef objects is in place — much of which
Phase-5 OOP rendering already covers.

**Special-cased**: for objects with bulky internal state (FIR taps,
LMS coefficients, S-parameter cubes for high-frequency-resolution
sweeps), the inspector must avoid pretty-printing megabyte-sized
arrays inline. Truncate to a head/tail summary like the existing
matrix renderer does for large `matlab_mat`.

For `sparameters`, the natural inspector layout is:
- `Parameters` — child node showing 3-D complex cube (truncated).
- `Frequencies` — child node, full vector if < 1024 samples,
  truncated otherwise.
- `NumPorts`, `Impedance` — scalar leaves.
- Per-element drill-down: clicking `Parameters[1,1,:]` should
  reveal the `S₁₁(f)` slice as a 1-D complex vector — useful for
  quickly inspecting reflection at a specific port pair.

For antenna mesh objects:
- `Vertices` — `[N × 3]` real (truncated).
- `Triangles` — `[M × 3]` int (truncated).
- `Edges` — `[E × 2]` int with adjacency info.
- `WireSegments` (Tier-2) — `[K × 2]` start/end indices into
  `Vertices` plus per-segment radius.
- Summary line: "12,847 vertices · 24,832 triangles · 37,341 edges"
  to give the user a sense of MoM problem size at a glance.

### 11.4 REPL JIT considerations

System Objects mutate state across `step` calls. The existing JIT
REPL's caching strategy assumes pure functions and re-evaluates on
re-entry; for stateful classdefs, the object's persistent
properties must survive across REPL turns. This already works for
handle-classdef in the Phase-5 OOP arc (handles persist by
reference), so no new infrastructure expected — but verify with a
gated test (`comm.PNSequence` over multiple REPL turns must
continue the LFSR sequence, not restart).

For RF Toolbox: `sparameters` objects are value-typed (read-only
post-construction), so JIT caching is trivial. `rfckt.*` objects do
mutate cached `AnalyzedResult` after `analyze(rfobj, freq)` — same
pattern as Comm System Objects; covered by the same handle-classdef
machinery.

For Antenna Toolbox: catalog classdefs (`dipole`, `patch*`, etc.)
are value-typed during the geometry-edit phase. Once `mesh(ant)`
runs, the cached mesh sits on the object — same pattern as
`rfckt.*`. The big REPL wrinkle is **MoM solve cost**: a single
`pattern(ant, freq)` call may take seconds-to-minutes for moderate
mesh sizes. Consider: (a) print a "Solving MoM (N=...)" progress
line, (b) memoize the solve keyed on `(geometry hash, freq, mesh
hash)` so repeated calls in REPL don't re-solve, (c) interrupt
support so Ctrl-C aborts a long solve cleanly. None of these block
ANT-Tier-2 shipping; flag for follow-on UX polish.

---

## 12. Suggested execution order

If user demand drives the order, expect this rough sequence (each
row unblocks the next):

**Re-ordering policy**: Propagation rows are placed **first** after
the bare-minimum Comm Tier 1 prerequisites because (a) they have
zero classdef / System-Object dependency and (b) the user has
flagged the PtP+ITM+CoverageMap workflow as priority. Comm Tier 2+
modulation work follows; the System-Object fix gates only the
classdef-bearing tracks (Comm Tier 3+, RF, Antenna).

**Update (2026-05-17)**: rows 1 – 36 below are now ✅ shipped (Comm
Tier-1 → Tier-5 + Tier-6 + Tier-7 modern codes function-form + PROP
Tier-1a/2a/2b/3 + RF Tier-1 → Tier-4). Remaining work is rows 37 – 48
(Antenna multi-wire MoM + planar RWG MoM + arrays), the SO-form
variants for the comm classes, and the multi-week ANT-Tier-3 surface
integral.

| Order | What | Effort | Status |
|---|---|---|---|
| 1 | `randi` + `rng` (Tier 1.1–1.2) | 2 sess | ✅ shipped |
| 2 | `randsrc` / `randerr` / int↔bit (Tier 1.3–1.4) | 3 sess | ✅ shipped |
| 3 | `awgn` + `biterr` / `symerr` (Tier 1.5–1.6) | 3 sess | ✅ shipped — closes Comm Tier 1 |
| 4 | **PROP-Tier-1a function-form: ITU-R + cellular models + Fresnel + knife-edge + geo helpers** (§3.1) | 1.5 wk | ✅ shipped |
| 5 | **PROP-Tier-2a function-form: ITM (Longley-Rice) port** (§3.2) | 3 wk | ✅ shipped (engineering port; byte-identical NTIA v7.0 reference port stays 🔵) |
| 6 | **PROP-Tier-2b function-form: terrain profile + `los_check` + `link_budget` + `coverage_grid`** (§3.3) | 1 wk | ✅ shipped — single-TX Propagation MVP lit |
| 7 | **PROP-Tier-3 function-form: sector / cosine / 3GPP patterns + mount orientation + `coverage_grid_multi`** (§3.4) | 1.5 wk | ✅ shipped — Multi-Site Directional MVP lit |
| 8 | `pammod` / `qammod` / `pskmod` (Tier 2.1–2.3) | 2 wk | ✅ shipped |
| 9 | `rcosdesign` / `gaussdesign` (Tier 2.7) | 1 sess | ✅ shipped |
| 10 | `berawgn` / `bercoding` (Tier 2.8) | 1 wk | ✅ shipped (`berawgn` only — `bercoding` stays deferred) — closes Comm Tier 2 |
| 11 | **System-Object architectural fix** (CST §12 / §11.1) | 1 wk | 🟡 partial — CRC SOs shipped, validates the lowering; full multi-class SO surface still 🔵 |
| 12 | **PROP-Tier-1b classdef wrappers**: `propagationModel` / `txsite` / `rxsite` / `pathloss` / `link` / `coverage` / `los` (§3.4) | 3 sess | ✅ shipped (CamelCase classdefs) |
| 13 | CRC Sys Object (Tier 3.1) | 1 wk | ✅ shipped |
| 14 | Convolutional + Viterbi (Tier 3.2) | 2 wk | ✅ shipped (hard + soft) |
| 15 | BCH / RS + `gf` (Tier 3.3) | 2 wk | 🔵 — needs `gf(2^m)` typed descriptor |
| 16 | Interleavers (Tier 3.5) | 1 wk | ✅ shipped — closes Comm Tier 3 |
| 17 | Linear / DFE equalizer (Tier 4.1) | 2 wk | ✅ shipped (function-form; SO variants gated) |
| 18 | Carrier / symbol / frame sync (Tier 4.2) | 2 wk | ✅ shipped (function-form; SO variants gated) |
| 19 | RF impairments (Tier 4.3) | 1 wk | ✅ shipped — closes Comm Tier 4 |
| 20 | **`sparameters` + sibling classdefs (RF-Tier-1.1)** | 3 sess | ✅ shipped |
| 21 | **Network parameter conversions (RF-Tier-1.2)** | 3 sess | ✅ shipped (all-to-all + N-port) |
| 22 | **Touchstone v1 + v2 read + write (RF-Tier-1.3)** | 3 sess | ✅ shipped — closes RF-Tier-1 |
| 23 | **`s2tf` / `gammain` / `vswr` / `powergain` / stability (RF-Tier-2.1)** | 1 wk | ✅ shipped |
| 24 | **`cascadesparams` / `snp2smp` (RF-Tier-2.2)** | 3 sess | ✅ shipped (Redheffer + Schur) |
| 25 | **`rfbudget` Friis solver (RF-Tier-2.3)** | 3 sess | ✅ shipped — closes RF-Tier-2 |
| 26 | OFDM mod / demod (Comm Tier 5.1) | 1.5 wk | ✅ shipped (function-form) |
| 27 | Fading channels (Comm Tier 5.2) | 2 wk | ✅ shipped (Rayleigh + Rician) |
| 28 | MIMO (Comm Tier 5.3) | 1.5 wk | ✅ shipped (Alamouti + MRC + ML) — closes Comm Tier 5 |
| 29 | Spreading sequences (Comm Tier 6.1) | 1 wk | ✅ shipped (PN / Gold / Hadamard / Walsh) |
| 30 | Source coding / quantization (Comm Tier 6.2) | 1 wk | ✅ shipped (`quantiz` / `lloydsQuant` / `compandMu` / `compandA` / DPCM) — closes Comm Tier 6 |
| 31 | **`rationalfit` Vector Fitting (RF-Tier-3.1)** | 2 wk | ✅ shipped (real + complex-conjugate pole pairs) |
| 32 | **`s2tdr` / `s2tdt` (RF-Tier-3.2)** | 3 sess | ✅ shipped |
| 33 | **Transmission line objects (RF-Tier-3.3)** | 1 wk | ✅ shipped (microstrip / CPW / coax / two-wire / parallel-plate) — closes RF-Tier-3 |
| 34 | **`matchingnetwork` (RF-Tier-4.1)** | 1 wk | ✅ shipped (L / T / Pi) |
| 35 | **`rfckt.*` hierarchy subset (RF-Tier-4.2)** | 1 wk | ✅ shipped (7 rfckt blocks + LC filter circuits) |
| 36 | **Smith chart numerics (RF-Tier-4.3)** | 3 sess | ✅ shipped — closes RF-Tier-4 |
| 36b | **Comm Tier-7 modern codes** (function-form Polar / LDPC / Turbo, §5.4) | 3 wk | ✅ shipped (out-of-order arc) |
| 37 | **Antenna catalog classdefs (12 types) (ANT-Tier-1)** | 1 wk | 🟡 partial — `AntDipole` + `AntMonopole` shipped; remaining 10 stubs 🔵 |
| 38 | **Wire mesh + sinusoidal basis (ANT-Tier-2.1)** | 3 sess | 🟡 sidestepped — closed-form Balanis EMF path sufficed for dipole MVP |
| 39 | **Pocklington Z matrix (ANT-Tier-2.2)** | 1 wk | 🔵 ANT-Tier-2b — gated on kernel-scaling debug |
| 40 | **Z·I=V solve + Z_in / S₁₁ (ANT-Tier-2.3)** | 3 sess | ✅ shipped (`antennaWireSolve` closed-form for thin dipole) |
| 41 | **Far-field pattern + gain (ANT-Tier-2.4)** | 1 wk | ✅ shipped (`antennaWirePattern`) |
| 42 | **`sparameters(ant, f)` RF-bridge (ANT-Tier-2.5)** | 3 sess | ✅ shipped (`antennaWireSparameters`) — closes ANT-Tier-2 / Antenna MVP (dipole) |
| 43 | **Triangular mesh generator (ANT-Tier-3.1)** | 1 wk | 🔵 |
| 44 | **RWG basis (ANT-Tier-3.2)** | 2 sess | 🔵 |
| 45 | **Surface-integral Z matrix + singularity extraction (ANT-Tier-3.3)** | 3 wk | 🔵 — single biggest item in Antenna arc |
| 46 | **Patch / planar `pattern` / `impedance` (ANT-Tier-3.4)** | 3 sess | 🔵 — closes ANT-Tier-3 |
| 47 | **Array geometry classdefs (ANT-Tier-4.1)** | 3 sess | 🔵 |
| 48 | **Array factor multiplication (ANT-Tier-4.2)** | 3 sess | 🔵 — closes ANT-Tier-4 (no rigorous mutual coupling) |

**Total**: ~49.5 weeks of focused sessions for Comm Tier 1 → Tier 5
+ RF-Tier-1 → RF-Tier-4 + ANT-Tier-1 → ANT-Tier-4 + PROP-Tier-1 →
PROP-Tier-3 closure. Comm Tier 6 (+2 weeks), Comm Tier 7 (LDPC /
Turbo / Polar), and ANT-Tier-5 items (dielectric MoM, hybrid
MoM-PO, FMM, optimization) are multi-month each.

**Single-TX Propagation MVP slice (~5.5 weeks)**: rows 1–6 —
Comm Tier 1 + PROP-Tier-1a + PROP-Tier-2a + PROP-Tier-2b. Lights
up single-TX, omnidirectional PtP + Coverage Map. Bare functions;
no System-Object dependency.

**Multi-Site Directional Propagation slice (~7 weeks)**: rows
1–7 — adds PROP-Tier-3 (sectors + mounts + multi-TX aggregation)
on top. **Lights up the user's "two poles, three sectors per pole
+ two directionals each, combined coverage" scenario.** Still bare
functions; still no System-Object dependency.

**Comm MVP slice (~3 weeks, after Propagation track)**: rows 1–3 +
8–10. Lights up the canonical "modulate → AWGN → demod → BER"
loop with a reference theory curve. (Note: rows 4–7 sit between
Tier 1 and Tier 2 in this re-ordered execution; if a contributor
prioritizes Comm over Propagation, rows 4–7 can be deferred until
after row 10 — they have no dependency on Comm Tier 2.)

**Coded MVP (~10 weeks)**: rows 1–3 + 8–10 + 11–16. Adds CRC +
convolutional + RS on top. Covers the textbook "uncoded vs coded
BER" comparison.

**RF MVP slice (~3 weeks, after row 11)**: rows 20–25. Lights up
"load .s2p, compute gain/stability/VSWR, cascade, run a Friis
budget" — covers ~80% of the practical small-signal RF analysis
workflow.

**Wireless-PHY slice (~24 weeks)**: rows 1–28 (Comm Tier 1–5 + RF
Tier 1–2 + Propagation MVP). Lights up enough infrastructure to
assemble a small 4G-flavored or 802.11a-flavored OFDM link end-to-
end *and* analyze its RF front-end with vendor S-parameter data
*and* compute its multi-cell coverage map.

**Antenna MVP slice (~5 weeks, after row 11)**: rows 37–42.
Lights up wire-antenna MoM — dipole / monopole / Yagi / helix /
folded-dipole / loop. User can compute impedance, S₁₁, gain,
pattern, bandwidth, dump Touchstone for RF-Toolbox cascade — and
**drop those measured patterns directly into PROP-Tier-3
`coverage_grid_multi`** via §3.4.5.

**Full antenna-EM slice (~14 weeks, after row 11)**: rows 37–48.
Adds triangular-mesh MoM (patches / spirals / bowties / planar
slots) plus phased-array beam steering with the multiplication
approximation. **The 3-week surface-integral Z-matrix item (row
45) is the single biggest individual sub-arc** in this entire
roadmap; commit only when there is concrete user demand for
patch-antenna analysis.

**Four-product MVP** (Propagation + Comm + RF + Antenna, ~15
weeks total in priority order): rows 1–7 (Comm Tier 1 + Propagation
MVP including Multi-Site Directional / sectors / coverage_grid_multi)
+ rows 8–10 (Comm Tier 2 modulation MVP) + row 11 (System-Object
fix) + rows 20–25 (RF MVP) + rows 37–42 (Antenna MVP). Lights up
the canonical "design a Yagi → characterize via S-parameters →
drop measured pattern into multi-pole multi-sector
`coverage_grid_multi` → run RF chain budget → drive QAM modem →
analyze cellular-style SINR coverage map over a heightmap"
pedagogical loop. **This is the most-impactful teaching slice and
is now reachable in priority order.**

---

## 13. Out of scope (carved out, by chapter / topic)

| Chapter / topic | What | Why out of scope |
|---|---|---|
| Throughout | All apps — Bit Error Rate Analyzer, Constellation Diagram / Eye Diagram / Spectrum Analyzer scopes, Wireless Waveform Generator, Wireless Network Simulator | Interactive Qt apps; not a language feature |
| Throughout | Simulink Communications block library (~150 blocks) | Simulink is not in scope |
| AI for Wireless | RNN/CNN/transformer-based receiver examples, deep-learning predistortion | Deep Learning Toolbox; separate product |
| RF Propagation (closed-form models) | `propagationModel('freespace'/'rain'/'gas'/'fog'/'close-in'/'longley-rice')`, `pathloss`, `los`, `link`, `coverage` (numeric grid), Hata / COST231 / Egli / ECC33 / SUI / Ericsson cellular extensions, knife-edge diffraction, Fresnel zones, Haversine / Vincenty | **IN SCOPE — PROP-Tier-1 + PROP-Tier-2 in §9.5 / §9.6.** ~6 weeks. Heightmap is user-supplied; ITM port is the largest sub-item (~3 wk) |
| RF Propagation (GIS / 3-D / interactive) | Site Viewer 3-D map, ray tracing through OSM buildings, auto-fetch SRTM/DTED tile servers, TIREM external lib, MSI Planet I/O, animated live coverage, GPU acceleration | Hard 🔴 — needs Mapping Toolbox + 3-D graphics stack + tile server integration |
| Throughout | SDR / hardware-in-the-loop entries (Pluto-SDR, USRP, RTL-SDR drivers) | Hardware drivers; out of scope |
| §LDPC, §Turbo, §Polar | LDPC / Turbo / Polar encode-decode | Each is a multi-week iterative-decoder arc; deferred (Tier 7 stretch) |
| Throughout | 5G Toolbox, WLAN Toolbox, LTE Toolbox, Bluetooth Toolbox, Zigbee Toolbox | These are *separate* MathWorks products that share Comm infrastructure but ship their own waveform generators and reference receivers. Out of scope in this roadmap |
| Throughout | Phased Array System Toolbox bridge (beam steering, antenna array geometry) | Separate product |
| §MSK | Continuous-phase modulation high-end (CPM, GMSK detailed) | The basic MSK form lights up in Tier 2; richer CPM is deferred |
| Throughout | Digital predistortion adaptive identification (`comm.DPD`) | Heavy adaptive-identification arc; defer to user demand |
| Throughout | Hybrid ARQ / link-adaptation feedback loops | Cross-layer; defer |
| Throughout | Native interactive plotting (constellation rotates as samples arrive, eye diagram redraws) | We do not ship native plotting. Functions return numeric data; users plot. The Cairo backend can produce static PNG snapshots. |
| RF Toolbox §3 | **Verilog-A export** (`writeVerilogA`, `rfmodel.rational/writeVA`) | Code generator for behavioral SPICE / SystemVerilog co-sim flows; large new emit lane |
| RF Toolbox throughout | **Circuit envelope simulation** | Multi-tone time-stepping nonlinear circuit solver; multi-week dedicated effort |
| RF Toolbox throughout | **Harmonic Balance solver** | Newton-Krylov on multi-tone steady-state nonlinear circuit residual; multi-week. The linear Friis budget covers ~80% of the practical surface |
| RF Toolbox throughout | **RF Budget Analyzer app**, **Smith Chart Tool app** | Interactive Qt apps; not language features |
| RF Toolbox §6 | **Modelithics commercial component library** | Vendor-licensed component data; out of scope |
| RF Toolbox throughout | **IEEE P370 fixture characterization** | Specialized de-embedding standard; defer to user demand |
| RF Toolbox throughout | **Differential / mixed-mode 4-port advanced analyses** beyond `s2sdd` family | Niche; defer |
| RF Toolbox throughout | **AMP file format** read / write (RF Toolbox §4) | Touchstone covers most modern data; AMP is a less-used legacy format. Defer unless requested |
| RF Toolbox throughout | **Simulink RF blockset** | Simulink not in scope |
| Antenna §3 | **MoM with dielectrics** (PMCHWT formulation, dielectric substrate boundary integrals) | Multi-month sub-arc; ANT-Tier-5 stretch |
| Antenna §3 | **Hybrid MoM-PO**, **Physical Optics solver** | Multi-week each; needed for electrically large geometries (reflectors, scatterers) |
| Antenna §3 | **Fast Multipole Method (FMM)** | Research-grade O(N log N) acceleration; multi-month |
| Antenna §3 | **Infinite ground plane**, **Infinite array (Floquet/unit-cell)** | Specialized boundary conditions; ANT-Tier-5 |
| Antenna throughout | **Mutual coupling (rigorous full-array MoM)** | The element-pattern multiplication approximation in ANT-Tier-4.2 covers most cases; rigorous form is ANT-Tier-5 |
| Antenna §1 / §2 | **Reflector antennas** (parabolic / Cassegrain) | Need PO / GTD on curved reflectors; ANT-Tier-5 |
| Antenna throughout | **Antenna optimization** (PSO / GA / SADEA / surrogate / Bayesian) | Multi-week; needs the solver in a tight loop. Defer to user demand |
| Antenna §1 / §2 | **PCB antenna design** (`pcbStack`), **Gerber export** | Multi-layer stackup modeling; out of scope |
| Antenna §1 | **Photonic crystal / metasurface** | Periodic-homogenization specialty; defer |
| Antenna §1 | **Custom antenna from photo** (CV-based geometry inference) | Computer-vision dependency; out of scope |
| Antenna §4 | **Site Viewer** (3-D map of buildings / terrain / ray traces, Cesium / OSM tiles) | Hard 🔴; same as RF Propagation row above |
| Antenna §4 | **Ray tracing through 3-D buildings** (`propagationModel('raytracing')`) | Heavy 3-D scene + multi-bounce arc; carved out (in-scope: closed-form path-loss models per §9.5/§9.6) |
| Antenna §4 | **TIREM software** access (`propagationModel('tirem')`) | Proprietary external dependency |
| Antenna §4 | **Auto-fetch SRTM / DTED / OSM tiles** | Web tile-server integration; user-supplied heightmap is the accepted alternative (see §9.6.2) |
| Antenna §1 | **Antenna Designer app**, **Array Designer app**, **PCB Antenna Designer** | Interactive Qt apps; not language features |
| Antenna §1 | **AI for Antennas** (DL surrogate models, generative geometry) | Deep Learning Toolbox dependency |
| Antenna §5 | **MSI Planet file format** I/O | Niche commercial-tool interchange |
| Antenna throughout | **Real-time interactive 3-D visualization** of currents / fields / patterns | Static figures via Cairo are achievable; interactive 3-D not in scope |
| Antenna throughout | **Simulink antenna blocks** | Simulink not in scope |

---

## 14. Test corpus deltas

Mirror the SPT/CST layout under `test/Run/` and `test/Runtime/`:

| Tier | New `test/Run/*.m` (rough count) | New `test/Runtime/test_*.c` |
|---|---|---|
| Tier 1 (sources / sinks) | ~5 (`comm_randi`, `comm_rng`, `comm_int2bit`, `comm_awgn`, `comm_biterr`) | new `test/Runtime/test_comm.c` |
| Tier 2 (modulation loop) | ~6 (`comm_pammod`, `comm_qammod`, `comm_pskmod`, `comm_rcosdesign`, `comm_berawgn`, `comm_link_qam_awgn`) | extend `test_comm.c` |
| Tier 3 (channel coding) | ~6 (`comm_crc`, `comm_convenc_vitdec`, `comm_bchenc_bchdec`, `comm_rsenc_rsdec`, `comm_intrlv`, `comm_link_coded`) | extend `test_comm.c` |
| Tier 4 (eq / sync / RF) | ~5 (`comm_lineareq`, `comm_dfe`, `comm_carriersync`, `comm_symbolsync`, `comm_iqimbal`) | extend |
| Tier 5 (OFDM / fading / MIMO) | ~5 (`comm_ofdm`, `comm_rayleigh`, `comm_rician`, `comm_alamouti`, `comm_link_ofdm`) | extend |
| Tier 6 (spreading / source) | ~3 (`comm_pn`, `comm_gold`, `comm_quantiz`) | extend |
| RF-Tier-1 (objects + I/O) | ~5 (`rf_sparam_basic`, `rf_s2y_z_h`, `rf_touchstone_s2p`, `rf_touchstone_s4p`, `rf_newref`) | new `test/Runtime/test_rf.c` |
| RF-Tier-2 (analyses + budget) | ~5 (`rf_s2tf`, `rf_gammain_vswr`, `rf_powergain`, `rf_cascadesparams`, `rf_rfbudget_friis`) | extend `test_rf.c` |
| RF-Tier-3 (rational + TDR + tlines) | ~4 (`rf_rationalfit_basic`, `rf_rationalfit_passivity`, `rf_s2tdr`, `rf_microstrip`) | extend `test_rf.c` |
| RF-Tier-4 (matching + circuits) | ~3 (`rf_matching_l`, `rf_rfckt_amp`, `rf_rfckt_lcbandpass`) | extend `test_rf.c` |
| ANT-Tier-1 (catalog) | ~5 (`ant_dipole_props`, `ant_patch_props`, `ant_yagi_props`, `ant_helix_props`, `ant_array_props`) | new `test/Runtime/test_antenna.c` |
| ANT-Tier-2 (wire MoM) | ~6 (`ant_dipole_impedance`, `ant_dipole_pattern_halfwave`, `ant_monopole_impedance`, `ant_yagi_pattern_3el`, `ant_helix_axial`, `ant_dipole_to_sparams`) | extend `test_antenna.c` |
| ANT-Tier-3 (planar MoM) | ~5 (`ant_patch_impedance`, `ant_patch_pattern_freqsweep`, `ant_bowtie_impedance`, `ant_pifa_impedance`, `ant_spiral_pattern`) | extend `test_antenna.c` |
| ANT-Tier-4 (arrays) | ~3 (`ant_lineararray_steering`, `ant_rectangulararray_pattern`, `ant_array_factor_taper`) | extend `test_antenna.c` |
| PROP-Tier-1 (closed-form) | ~7 (`prop_fspl`, `prop_hata_urban`, `prop_cost231`, `prop_egli`, `prop_ecc33`, `prop_sui`, `prop_fresnel_clearance`, `prop_knife_edge_single`, `prop_knife_edge_deygout`, `prop_haversine`) | new `test/Runtime/test_prop.c` |
| PROP-Tier-2 (ITM + PtP + Coverage) | ~5 (`prop_itm_ntia_ref1` through `_ref5` against NTIA test cases, `prop_terrain_profile`, `prop_los`, `prop_link_budget`, `prop_coverage_grid`) | extend `test_prop.c` |
| PROP-Tier-3 (directional + multi-site) | ~5 (`prop_sector_pattern_120`, `prop_cosine_pattern`, `prop_3gpp_pattern`, `prop_mount_orientation`, `prop_coverage_multi_2pole_3sector`) | extend `test_prop.c` |

C / C++ / Python / TypeScript lanes must remain byte-identical, with
the same `.stdout-python` override convention SPT uses for numpy
bracket repr (matrix returns from Comm will trigger this on the
Python lane). Several Comm primitives return complex matrices
(`qammod`, `pskmod`, `awgn` of complex input, fading channel
outputs, OFDM samples) — those will need `.skip-emit-typescript` or
the existing complex-matrix TS workaround, same friction `freqz` /
`fft_c` carry today.

**Display gating**: `disp(obj)` for System Objects produces multi-
line formatted output that must be byte-stable across lanes. The C
lane is canonical; TS and Python override files only land if the
formatting diverges (e.g., number-formatting precision differences).
Plan for overrides; they are easier to add than to retrofit.

**Reproducibility**: BER tests must seed the RNG via `rng(...)`
(Tier 1.2) before any random source. Without explicit seeding, the
oracle `.stdout` files diverge across runs and the test corpus is
unmaintainable. This is the single biggest reason Tier 1.2 is in
Tier 1, not later.

**RF Toolbox tests**: most outputs are deterministic (closed-form
S-parameter algebra, Friis cascade) — no RNG needed. `rationalfit`
is iterative but converges to a well-defined minimum; tolerance-
based `assertNear` against a precomputed oracle is appropriate.
Touchstone tests will need small fixture files committed under
`test/Run/fixtures/rf/*.s?p`; treat them as test data, not source.

**Antenna Toolbox tests**: deterministic but **slow** — a single
`pattern(yagiUda, 300e6)` is seconds-to-minutes depending on mesh
size. Test corpus must use **small mesh fixtures** (segment count <
100 for wire, triangle count < 500 for planar) to keep CI under a
minute per test. Reference oracles for impedance / pattern are
analytically known for the half-wave dipole (Z_in ≈ 73 + j42.5 Ω
at resonance) and tabulated for Yagi-Uda gain — use those rather
than re-running MATLAB. Tolerance bands need to be **looser than
Comm/RF tests** (3-5% on impedance, 0.5 dB on gain) because
different MoM discretizations produce different but equally-valid
answers; we are not trying to bit-match MathWorks.

**Propagation tests**: closed-form models (PROP-Tier-1) are
deterministic and bit-checkable — use textbook problems with hand-
computed reference (e.g., Hata urban at 900 MHz, ht=30 m, hr=1.5 m,
d=1 km → ~125 dB). For ITM (PROP-Tier-2.1), use the **NTIA-shipped
test suite** (~30 reference cases) as golden oracles; ITM is
deterministic so byte-identical match is the bar. Heightmap-based
tests need committed fixture matrices under
`test/Run/fixtures/prop/*.mat` or `*.bin` (e.g., a 256×256 synthetic
hill profile + a 256×256 SRTM-like real-terrain extract). Coverage
map output is a 100×100 matrix — compare cell-by-cell with
tolerance.

---

## 15. Summary

Comm Toolbox compatibility is a **layered project on top of SPT**:

**Stage 1 (Tier 1, ~1.5 weeks) — 🔵 OPEN**: bit/symbol source + AWGN
+ BER. `randi`, `rng`, `randsrc`, `int2bit`/`bit2int`, `awgn`,
`biterr`/`symerr`. Small, all sit on existing `matlab_rand` / matrix
infrastructure. Tier 1 alone is "sources and sinks" — useful as a
test scaffold but does not ship modulation.

**Stage 2 (Tier 2, ~3 weeks) — 🔵 OPEN**: digital modulation loop.
`pammod`/`qammod`/`pskmod` + `rcosdesign` + `berawgn`. Closes the
canonical "modulate → AWGN → demod → BER" workflow. **This is the
MVP slice** — at the end of it, a user can simulate any uncoded
QAM/PSK/PAM link.

**Stage 3 (System-Object architectural fix, ~1 week) — 🔵 OPEN**:
Comm is ~80% System Objects. The recorded blocker (CST §12 /
LowerTensorOps verifier mismatch) **gates Tier 3 onward**. Same fix
unblocks CST `tf` / `ss` / `zpk` model objects. Single highest-
leverage piece of work in this roadmap.

**Stage 4 (Comm Tiers 3–5, ~17 weeks) — 🔵 OPEN**: channel coding
(CRC, convolutional + Viterbi, BCH/RS + GF helpers, interleavers),
equalization (LMS/RLS adaptive), synchronization (Costas PLL,
timing recovery, frame sync), OFDM, fading channels, MIMO.
Conceptually conventional once the System-Object infrastructure
lands; bulk is the per-block algorithm work.

**Stage 5 (RF Toolbox, ~6 weeks bundled in §9) — ✅ MOSTLY CLOSED**:
- **RF-Tier-1 (~1 wk)**: ✅ shipped — `RFSparameters` + 6 sibling
  network parameter classdefs (Y / Z / H / G / ABCD / T), 2-port S↔Y /
  S↔Z / S↔H / S↔ABCD + inverses H→S / ABCD→S, N-port S↔Y / S↔Z via
  2N×2N real-equivalent algebra, Touchstone v1 read/write for any
  `.sNp` port count, `snp2smp` matched-termination port extraction,
  mixed-mode 4-port `sparamS2smm`.
- **RF-Tier-2 (~1.5 wk)**: ✅ shipped — closed-form analyses
  (`gammaIn` / `gammaOut`, `vswr`, `powerGain`, `stabilityK` /
  `stabilityMu`, `s2tf`), 2-port T-parameter cascade `cascadeSparams2`,
  N-port `cascadeSparamsN` (diagonal approximation; full Schur-
  complement form deferred), `rfbudgetFriis` Friis cascade with NF /
  IP3 / SNR.
- **RF-Tier-3 (~3.5 wk)**: ✅ shipped — `rationalfit` Vector Fitting
  with real AND complex-conjugate pole pairs, `freqresp` / `passivity`
  complex-pole-aware, `timeresp` complex-state ZOH, `s2tdr` / `s2tdt`
  time-domain reflectometry/transmission, transmission-line geometries
  (`rfckt_txline` / `coaxial` / `microstrip` / `cpw` / `parallelplate` /
  `twowire`).
- **RF-Tier-4 (~2 wk)**: ✅ shipped — `matchingnetwork` L / T / Pi
  topologies, `RFCktAmplifier` / `RFCktMixer` / `RFCktPassive`
  classdef skeletons, `smithGrid` numeric overlays.  `RFCktCascade` /
  `RFCktParallel` / `RFCktSeries` / `RFCktShunt` combinators 🔵 partial.

**RF Toolbox status**: ~95% function-form complete, ~70% classdef
complete.  Remaining work is small cross-conversions + classdef
combinators + carved-out non-language items.  See
`docs/rf_toolbox_plan.md` for the forward plan.

**Stage 6 (Antenna Toolbox MVP — wire MoM, ~5 weeks bundled in §9)
— 🔵 OPEN**:
- **ANT-Tier-1 (~1 wk)**: 12-type antenna catalog classdef hierarchy
  (`dipole`, `monopole`, `patch*`, `yagiUda`, `helix`, `bowtie*`,
  `spiral*`, `pifa`, `horn*`, `loop*`). Property holders + mesh
  stubs.
- **ANT-Tier-2 (~3 wk, Antenna MVP)**: wire-antenna Method of
  Moments — Pocklington integral equation, sinusoidal basis,
  complex Z·I=V solve, far-field pattern integration. Exposes
  `impedance(ant, f)`, `pattern(ant, f, az, el)`, `gain(ant, f)`,
  `vswr(ant, f)`, `bandwidth(ant)`, `sparameters(ant, freqs)` for
  wire-shaped antennas.
- **`sparameters(ant, freqs)` is the Antenna→RF bridge** — feeds
  directly into RF-Tier-1's Touchstone path.

**Stage 7 (Antenna Toolbox triangular-mesh MoM, ~6 weeks, only on
demand) — 🔵 OPEN**:
- **ANT-Tier-3**: triangular-mesh MoM (RWG basis, surface-integral Z
  matrix with Wilton-style singularity extraction, Gauss-Legendre
  quadrature on triangle pairs). Lights up patch / planar antennas.
- The 3-week surface-integral Z-matrix item is the **single biggest
  individual sub-arc** in this roadmap — production EM solvers
  spend years tuning the near-singular integration. Commit only on
  concrete user demand.

**Stage 8 (Antenna arrays, ~2 weeks) — 🔵 OPEN**:
- **ANT-Tier-4**: array geometry classdefs (`linearArray`,
  `rectangularArray`, `circularArray`, `conformalArray`) + array-
  factor multiplication. Mutual coupling rigorous form is carved
  out (ANT-Tier-5).

**Stage 9 (Propagation Models — promoted to top-level §3, ~7
weeks function-form) — 🔵 OPEN**:
- **PROP-Tier-1a (~1.5 wk, NO SO dep)**: closed-form path-loss
  models — function-form `fspl` / `pathlossRain` / `pathlossGas` /
  `pathlossFog` / `pathlossCloseIn`, cellular extensions
  (`pathlossHata` / `pathlossCost231Hata` / `pathlossEgli` /
  `pathlossEcc33` / `pathlossSui` / `pathlossEricsson9999`), Fresnel
  zones, knife-edge diffraction (single + Bullington / Deygout /
  Epstein-Peterson), Haversine / Vincenty distance. **No terrain
  data needed.**
- **PROP-Tier-2a (~3 wk, NO SO dep)**: ITM (Longley-Rice v7) port
  as `itm_pathloss(...)` function. Public-domain NTIA C++ port.
- **PROP-Tier-2b (~1 wk, NO SO dep)**: function-form `terrainProfile`
  + `los_check` + `link_budget` + single-TX `coverage_grid`.
- **PROP-Tier-3 (~1.5 wk, NO SO dep)**: directional + multi-site
  coverage. Sector / cosine / Gaussian / 3GPP analytical patterns,
  `applyMountOrientation` for azimuth/tilt, `coverage_grid_multi`
  with best-server / sum-power / SINR aggregation. **Lights up
  the user-requested two-pole / three-sector + two-directional
  combined-coverage workflow** without GIS dependencies. Bridge
  to ANT-Tier-2 patterns lets measured Yagi/dipole/etc. patterns
  drop into the same multi-site coverage call.
- **PROP-Tier-1b (~3 sess, GATED on SO fix)**: MathWorks-API
  classdef wrappers — `propagationModel`, `txsite`, `rxsite`,
  `pathloss`, `link`, `coverage`, `los`. Multi-site dispatch
  inside `coverage(tx_array, prop, ...)`.
- **Carved out (refined)**: only Site Viewer (3-D map + Cesium /
  OSM tiles), ray tracing through 3-D buildings, auto-fetch
  SRTM/DTED tile servers, TIREM, MSI Planet, GPU acceleration,
  animated coverage. The closed-form + ITM + multi-site directional
  closed-loop is **in scope**.

Heavy carve-outs (apps, Simulink, AI-for-Wireless, ray-tracing /
GIS / Site Viewer, SDR drivers, separate-product toolboxes
5G/WLAN/LTE/Bluetooth, LDPC/Turbo/Polar, RF circuit envelope,
harmonic balance, Verilog-A export, Smith Chart Tool app, RF
Budget Analyzer app, Modelithics library, Antenna dielectric MoM,
hybrid MoM-PO, Physical Optics, FMM, infinite ground / array,
mutual-coupling-rigorous, reflector antennas, antenna optimization,
PCB antenna stack, Photonic / metasurface, Antenna Designer apps,
TIREM, MSI Planet) keep this scoped to a **three-toolbox
practical-numeric subset** — same posture SPT and CST take.
Re-open carve-outs only on user demand.

**The single most-impactful primitive to ship first (Comm)**:
`awgn` (Tier 1.5) + `qammod`/`qamdemod` (Tier 2.2). Together they
unblock the canonical "see a constellation cloud" demo that's the
hello-world of digital communications.

**The single most-impactful primitive to ship first (RF)**:
`sparameters('file.s2p')` (RF-Tier-1.1 + 1.3). Loading vendor
Touchstone data is the gateway drug for every other RF analysis;
without it, no real-world S-parameter work is possible.

**The single most-impactful primitive to ship first (Antenna)**:
**wire MoM Z-matrix + complex solve** (ANT-Tier-2.2 + 2.3). It is
the foundation everything else stands on; without it, no antenna
analysis is possible. The half-wave dipole impedance check
(Z_in ≈ 73 + j42.5 Ω) is the canonical first-success milestone.

**The single most-impactful primitive to ship first (Propagation)**:
`fspl(d, freq)` (PROP-Tier-1.1) — Free Space Path Loss is the
gateway formula for every other empirical model and the
denominator of every link budget. Trivial closed-form, but
unblocks the whole propagation surface. The next-most-impactful
is the ITM port (PROP-Tier-2.1) which is the only entry that
needs serious algorithm work and is the gating piece for the
PtP-with-terrain workflow.

**The single most-impactful architectural piece still open**: the
System-Object lowering fix (CST §12 / §10.1 here). Without it,
**all three** toolboxes remain capped:
- Comm stalls at Tier 2 (no `comm.*` System Objects).
- RF stalls before RF-Tier-1 (`sparameters` is itself a classdef
  with field stores).
- Antenna stalls before ANT-Tier-1 (the catalog is entirely
  classdefs).

Fixing it unblocks **~42 weeks** of subsequent algorithm work
across all three products.

**Cross-toolbox alignment**: every System-Object / classdef
infrastructure piece (CRC SO at row 8, equalizer SO at row 12,
OFDM SO at row 21, fading channel SO at row 22, `sparameters`
value classdef at row 15, `rfckt.*` handle classdefs at row 30,
antenna catalog classdefs at row 32, array classdefs at row 42)
reuses the same classdef + persistent-state machinery. The
Tier-3.1 CRC implementation is the "validation harness" — once it
works, Comm Tier 3+, RF-Tier-1+, and ANT-Tier-1+ all become
conventional algorithm work.

**Cross-product synergy chain** (the canonical "RF-aware wireless
link" tutorial, ~11-week three-product MVP):

```
Antenna (ANT-Tier-2)  →  sparameters(yagi, freqs)         §9.2.5
       ↓                  → Touchstone .s2p file           §8.1.3
RF (RF-Tier-2)        →  rfbudget(antenna, lna, mixer, ...) §8.2.3
       ↓                  → power / NF / IP3 / SNR per stage
Comm (Tier 2 + 4.3)   →  qammod → awgn(snr) → memorylessNL
                          → demod → biterr                 §3.2/§5.3
       ↓
       BER curve, with the entire RF chain physically modeled.
```

This pipeline lights up at exactly the three-product-MVP slice
(rows 1–6 + 15–20 + 32–37 in §12). Pedagogically this is the
clearest demonstration of why the three toolboxes belong in one
codebase.
