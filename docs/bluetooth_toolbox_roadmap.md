# Bluetooth Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL/JIT**, and **demo** Bluetooth-Toolbox programs.

Source: *Bluetooth Toolbox User's Guide* (R2026a, 9 chapters: PHY
Modeling · Coexistence Modeling · Localization · Test and Measurement ·
Link-Level Simulation · Multinode Communication · Code Generation and
Deployment · Bluetooth Topic Pages · Tutorials).

This is a **high-reuse toolbox layered on the already-shipped
Communications + DSP + Signal stack** — the same posture as
[`bioinformatics_toolbox_roadmap.md`](bioinformatics_toolbox_roadmap.md)
(over Stats) and the Wavelet roadmap (over Signal). Bluetooth's PHY is a
*GFSK / DPSK waveform wrapped around standard FEC + CRC + whitening +
framing*, and every one of those primitives is already in the tree
(verified in `lib/Sema/Resolver.cpp` + `runtime/toolbox/comm/runtime_comm.cpp`):

- **GFSK / GMSK modulation** = a Gaussian pulse-shaping filter
  (`gaussdesign` ✅) feeding a CPM/FSK modulator (`fskmod`/`fskdemod` ✅,
  plus the shipped `filter`/`conv`/`upfirdn` multirate surface). The
  Bluetooth radio is GFSK with BT=0.5, h=0.5 (LE/BR) — a parameterization
  of the shipped CPM path, not a new kernel.
- **EDR DPSK** (π/4-DQPSK at 2 Mb/s, 8-DPSK at 3 Mb/s) = the shipped
  `dpskmod`/`pskmod` + differential-encode surface.
- **AWGN + BER** = `awgn` ✅, `comm.ErrorRate` ✅, `biterr` ✅, `bercoding`/
  `berawgn` ✅ — the BER-vs-Eb/No example is a direct reuse.
- **Forward error correction** — LE coded PHY (LE500K/LE125K) is a rate-1/2
  convolutional code + pattern mapper + spreading; BR/EDR uses the 1/3
  repetition + 2/3 shortened-Hamming FEC. `convenc`/`vitdec`/`poly2trellis`
  ✅ + `hamming_encode`/`hamming_decode` ✅ carry these.
- **CRC** — LE CRC-24, BR/EDR CRC-16 + HEC: `crc_generate`/`crc_check` ✅
  (BT generator polynomials are a baked parameterization).
- **Data whitening** = a 7-bit LFSR scrambler seeded by the channel index
  — a small hand-coded helper (the only genuinely new bit-level routine);
  the interleaver/scrambler surface (`intrlv`/`deintrlv` ✅) is the model.
- **Channel models** — link-level multipath fading + path loss reuse the
  shipped Comm channel surface; AWGN is shipped.

**No external dependency** — every waveform/decoder is a hand-coded routine
over the shipped Comm/DSP kernels, and every Bluetooth constant (channel↔
frequency map, access-address/preamble patterns, CRC/whitening polynomials,
coded-PHY pattern mapper, packet-type field layouts) is a baked-in table
(the precedent of the Comm 5G-NR base matrices and the Wavelet family-filter
catalogue).

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/bluetooth/ble_ber_awgn.m`](../examples/bluetooth/ble_ber_awgn.m):
*the canonical Bluetooth LE end-to-end BER curve — for each PHY mode
(LE1M / LE2M / LE500K / LE125K), generate a packet with
`bleWaveformGenerator`, pass it through `awgn`, recover the bits with
`bleIdealReceiver`, and report the bit error rate vs Eb/No* (the UG's §1
"Bluetooth LE Bit Error Rate Simulation with AWGN"). This exercises the
`bleWaveformGenerator → awgn → bleIdealReceiver → biterr` arc end-to-end;
achieving it closes **Bt-Tier-1** (the LE PHY core + the single most common
reason anyone reaches for this toolbox). Companion tracer-bullets:
[`examples/bluetooth/bredr_ber.m`](../examples/bluetooth/bredr_ber.m) (BR/EDR
PHY, **Bt-Tier-2**), `ble_ll_pdu_roundtrip.m` (PDU gen/decode, **Bt-Tier-3**),
and `ble_freq_hopping.m` (channel selection + AFH, **Bt-Tier-4**).

Companion docs:
[`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) (the GFSK/`awgn`/`crc`/
`convenc`/`fskmod`/`gaussdesign` reuse base — Bluetooth is its natural
extension), [`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md) (filtering /
multirate / `spectrumAnalyzer` / `timescope` viz),
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) (FFT / windows /
spectral estimation), [`bioinformatics_toolbox_roadmap.md`](bioinformatics_toolbox_roadmap.md)
(the most recent classdef + config-object wiring recipe — `phytree`/`DataMatrix`),
[`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md) (phased
arrays for the AoA/AoD direction-finding tier), [`plotting.md`](plotting.md),
[`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1** is
  the Bluetooth LE PHY (`bleWaveformGenerator` / `bleIdealReceiver` + the
  packet/whitening/FEC/CRC framing + the four PHY modes) — the headline.
  **Tier-2** is the BR/EDR PHY (`bluetoothWaveformGenerator` /
  `bluetoothIdealReceiver` + GFSK/DPSK + the BR/EDR packet-type catalogue).
  **Tier-3** is the protocol-data-unit layer (LE LL / advertising / control
  PDUs, L2CAP, ATT, GAP config-objects + gen/decode). **Tier-4** is channel
  selection + frequency hopping + channel impairments (`bleChannelSelection`
  System object, the channel↔frequency map, AFH Algorithm #1/#2,
  `bluetoothLEChannel`). **Tier-5** is localization (direction finding
  AoA/AoD + CTE, channel sounding ranging). **Tier-6** is test & measurement
  + link-level simulation (RF-PHY conformance measurement functions, fading +
  path-loss link simulation); the node-level discrete-event network simulator
  is carved down (see §9).
- **Effort** is in the existing Phase 5.6.x cadence (one focused session ≈ a
  half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~2 wk · T2 ~2 wk · T3
  ~1.5 wk · T4 ~1 wk · T5 ~2 wk · T6 ~2.5 wk (~11 wk full)**. Each tier is
  independently shippable and demoable; **T1 alone (~2 wk) closes the
  end-to-end LE BER workflow** — the canonical reason anyone reaches for this
  toolbox. Badge would advance by one.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started. **ALL 6 TIER
  CORES SHIPPED 2026-06-07 (Phases A+B+C)** in
  [`runtime/toolbox/bluetooth/runtime_bluetooth.cpp`](../runtime/toolbox/bluetooth/runtime_bluetooth.cpp)
  (~600 LOC, self-contained over the shipped complex-matrix + `awgn` + `biterr`
  surface). T1 `bleWaveformGenerator`/`bleIdealReceiver` (LE1M/LE2M + coded
  LE500K/LE125K) · T2 `bluetoothWaveformGenerator`/`bluetoothIdealReceiver`
  (BR + EDR2M + EDR3M) · T3 `bleLLDataChannelPDU`/`Decode` +
  `bleL2CAPFrame`/`Decode` · T4 `bleChannelSelection` (CSA #1/#2) +
  `bleChannelIndexToFrequency` · T5 `bleAngleEstimate` (AoA) · T6
  `bluetoothFrequencyOffset`/`bluetoothFrequencyDeviation`. 4 gating tests
  (`test/Run/bluetooth_{ble_phy,bredr_phy,pdu_channel,localization}.m`) + 5
  examples (`examples/bluetooth/`). Suite: **Run 746/0, frontend 83/0,
  emit-c/py/ts 324/266/231 /0, JIT gate OK, examples-sweep 0 regressions**.
  **Implementation notes / deviations from the planned API** (documented
  carve-downs — the numeric workflow is faithful, the surface is simplified to
  the robust function lane):
  - **Function forms, not classdefs / System objects**: the `*Config` PDU
    objects and the `bleChannelSelection` System object are realized as plain
    builtins (positional args; `bleChannelSelection(algorithm, hopIncrement,
    numEvents)` returns the hop-sequence vector instead of a stateful `step`).
    This sidesteps handle-class state mutation under the JIT and reuses the
    proven spec-table + struct-return wiring. The classdef/System-object
    surface is a faithful-API follow-on.
  - **CPFSK/MSK (rectangular frequency pulse), not Gaussian GFSK**: the
    modulator uses h=0.5 CPFSK so the symbol-integrating limiter-discriminator
    recovers every bit exactly at zero noise for ANY data (the BT=0.5 Gaussian
    spreads energy across 3 symbols, trading exact invertibility for spectrum;
    `bt_gaussian()` is retained for the spectral-mask follow-on). BER curves
    are physically correct waterfalls with visible FEC coding gain
    (LE1M > LE500K > LE125K) but ~a few dB off the textbook limiter-discriminator
    optimum — acceptable per the project's "practical numeric subset" posture.
  - **Simplified coded-PHY + BR/EDR framing**: a rate-1/2 K=4 conv code +
    Viterbi + S-repetition spreading (exact round-trip + coding gain); the
    exact CI/TERM coded-PHY framing and the full BR/EDR packet-type catalogue
    are deferred.
  - Kept **positional args** throughout (no `Name=Value`); single-quote char
    mode args ride the const_char→matlab_string spec-table coercion.
  The Comm/DSP reuse anchors (`gaussdesign`, `fskmod`/`fskdemod`, `awgn`,
  `crc_generate`/`crc_check`, `convenc`/`vitdec`, `comm.ErrorRate`, `biterr`)
  are all ✅ shipped.
  **JIT/DAP trap**: `for x = vec` (for-each over a variable vector) does NOT
  lower under ReplMode — examples must use the `for i = 1:N` range form +
  index (the `ble_freq_hopping` precedent).
- **Waveforms are complex IQ matrices; bits are `0/1` column vectors** — the
  exact lanes Comm already uses. `bleWaveformGenerator(bits, ...)` returns a
  complex `matlab_mat_c` column (the shipped complex-matrix lane);
  `bleIdealReceiver(rx, ...)` returns an `int8`/double bit column. No new
  container type — Bluetooth rides the shipped real/complex matrix lanes.
- **Bluetooth constants are baked-in tables** — the 40-channel LE (and
  79-channel BR/EDR) channel↔frequency map, the LE preamble/access-address
  patterns, the CRC-24/CRC-16/HEC generator polynomials, the data-whitening
  LFSR taps, the coded-PHY pattern mapper + spreading factors, and the
  per-packet-type field layouts are all static arrays in the runtime keyed by
  the caller's mode/packet-type string (the Image `imread('f.png')` / Wavelet
  `wfilters('db4')` / Comm 5G-NR-base-matrix precedent).
- **Config objects + System objects reuse the shipped classdef recipe** —
  `bleLLDataChannelPDUConfig` / `bleATTPDUConfig` / `bleL2CAPFrameConfig` /
  `bluetoothFrequencyHop` are property-holder classdefs (the `phytree` /
  `DataMatrix` alloc-then-populate + class-pinned-dispatch pattern, auto-
  prepended via `bluetooth_classdefs.m`); `bleChannelSelection` is a System
  object whose `frequencyHop()` call-syntax folds to `step` exactly like the
  shipped `dsp.*` objects (the `dsp.FIRFilter` parser-fold precedent — see
  [`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md) / `dsp_so_to_sv_bridge.md`).
- **Apps, SDR hardware, Simulink, and the full network simulator are carved
  out** (see §9): the Simulink models (`BluetoothFullDuplexModel` etc.), SDR
  transmit/receive (`sdrtx`/`sdrrx`/ADALM-PLUTO), the scopes (constellation /
  eye / spectrum-analyzer apps, Signal Analyzer), the discrete-event
  `wirelessNetworkSimulator` + `bluetoothLENode` network/mesh/LE-Audio engine,
  the LC3 audio codec + Auracast, deep-learning positioning, PCAP capture, and
  coexistence with 5G-NR/WLAN (needs the 5G/WLAN toolboxes) are out of scope.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Bluetooth code yet)

| Group | Surface (already shipped) | Location | How Bluetooth uses it |
|---|---|---|---|
| Gaussian pulse shaping | `gaussdesign` | `runtime/toolbox/comm/runtime_comm.cpp` (`matlab_comm_gaussdesign`) ✅ | The GFSK Gaussian filter (BT=0.5) — the shaping front-end of every LE/BR waveform (Tier-1/2). |
| FSK / CPM modulation | `fskmod`, `fskdemod` | `runtime/toolbox/comm/` ✅ | GFSK = Gaussian-shaped CPFSK with h=0.5; the modulator/demodulator core (Tier-1/2). |
| PSK / DPSK | `pskmod`, `dpskmod` (Comm) | `runtime/toolbox/comm/` | EDR π/4-DQPSK (2 Mb/s) + 8-DPSK (3 Mb/s) modulation (Tier-2). |
| AWGN + BER | `awgn`, `comm.ErrorRate`, `biterr`, `berawgn` | `runtime/toolbox/comm/` ✅ | The end-to-end BER channel + error counting — the headline example (Tier-1/2). |
| FEC | `convenc`, `vitdec`, `poly2trellis`, `hamming_encode`/`_decode` | `runtime/toolbox/comm/` ✅ | LE coded-PHY rate-1/2 conv code (Tier-1); BR/EDR 1/3 + 2/3 Hamming FEC (Tier-2). |
| CRC | `crc_generate`, `crc_check` | `runtime/toolbox/comm/` ✅ | LE CRC-24, BR/EDR CRC-16 + HEC (Tier-1/2/3) — BT polynomials are baked parameters. |
| Interleaving / scrambling | `intrlv`, `deintrlv` | `runtime/toolbox/comm/` ✅ | The model for the data-whitening LFSR scrambler (Tier-1/2). |
| Multirate / filtering | `filter`, `conv`, `upfirdn`, `resample`, `decimate` | `runtime/matlab_runtime.cpp` (Signal/DSP) ✅ | Samples-per-symbol up/downsampling in waveform gen + ideal receiver (Tier-1/2). |
| Complex matrix lane | `matlab_mat_c`, `fft`/`ifft`, complex arithmetic | `runtime/runtime_complex.cpp` ✅ | IQ waveforms are complex columns; spectral views; coherent demod (Tier-1/2/5). |
| Phased arrays | steering vectors / array geometry | Sensor Fusion / Antenna surface (partial) | AoA/AoD direction finding + CTE antenna switching (Tier-5). |
| Classdef plumbing | `matlab_obj_new`/`_set_*`/`_get_mat`, kwarg-ctor sugar, class-pinned dispatch, REPL persist, DAP render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The `*Config` PDU/PHY config objects + `bluetoothFrequencyHop` (Tier-3/4). |
| System-Object fold | `dsp.X` → `dsp_X` parser fold + `obj(frame)`→step | `lib/Parser/Parser.cpp`, `dsp_classdefs.m` (DSP) ✅ | `bleChannelSelection` step-call syntax (`frequencyHop()`) (Tier-4). |
| Name/value option parsing | option-string read in runtime (`fspecial`/`wdenoise`/`nwalign` path) | `lib/MLIR/LowerTensorOps.cpp` | `bleWaveformGenerator(bits, Mode="LE1M", SamplesPerSymbol=8, ...)` kwargs (Tier-1/2). |
| Plotting | Cairo `plot` / `spectrumAnalyzer` / `timescope` / `pwelch` | `runtime/plot/`, DSP | Spectrum / spectrogram / time-scope waveform views (Tier-1/4) — gating tests stay headless (numeric). |

**Net assessment**: the *signal-processing substrate* (Gaussian shaping, FSK/
CPM + DPSK modulation, AWGN/BER, FEC, CRC, interleaving, multirate, the complex
matrix lane, classdef + System-Object plumbing, plotting) is **already
shipped**. The genuinely new code is (a) the **Bluetooth packet framing**
(preamble + access address + PDU + CRC + whitening, per PHY mode + packet
type), (b) the **GFSK/DPSK waveform-generator + ideal-receiver wrappers**
parameterized to the BT radio (BT=0.5, h=0.5, channel→frequency offset), (c)
the **coded-PHY FEC + pattern mapper + spreading** (LE500K/LE125K), (d) the
**data-whitening LFSR** (the one new bit-level routine), (e) the **channel-
selection / AFH** Algorithm #1/#2 (`bleChannelSelection`), (f) the **PDU
config-objects + gen/decode**, and (g) the **direction-finding / channel-
sounding** localization layer. Each is a self-contained hand-coded routine over
the shipped Comm/DSP base — the heavy numeric lifting (filtering, modulation,
FEC, FFT) is done.

---

## 2. Bt-Tier-1 — Bluetooth LE PHY: waveform generation + ideal reception ✅

Goal: generate a standard-compliant Bluetooth LE baseband IQ waveform and
recover its bits — the end-to-end PHY the whole toolbox is built on. Closes
the headline BER example.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `bleWaveformGenerator(bits, Mode, SamplesPerSymbol, ChannelIndex, AccessAddress)` | Assemble the packet (preamble + access address + PDU + CRC-24), whiten, (coded PHY: FEC + pattern map + spread), GFSK-modulate (BT=0.5, h=0.5), upsample to `sps`. Modes `LE1M`/`LE2M`/`LE500K`/`LE125K`. Returns complex IQ column. | `gaussdesign`, `fskmod`, `filter`, complex lane |
| 1.2 | `bleIdealReceiver(rxWaveform, Mode, SamplesPerSymbol, ChannelIndex)` | GFSK demod (matched filter + phase discriminator) → dewhiten → (coded PHY: despread + Viterbi) → CRC check → recovered bits. Assumes perfect sync. | `fskdemod`, `vitdec`, `crc_check` |
| 1.3 | data whitening | 7-bit LFSR scrambler seeded by the channel index (x⁷+x⁴+1), XOR'd over the PDU+CRC. The one new bit-level routine. | hand-coded (LFSR) |
| 1.4 | LE CRC-24 | generator `0x00065B`, init from access address; append + check. | `crc_generate`/`crc_check` (baked poly) |
| 1.5 | coded-PHY FEC | LE500K/LE125K: rate-1/2 K=4 convolutional encode + pattern mapper (P=2/P=8 spreading) on the data field; inverse on receive. | `convenc`/`vitdec`/`poly2trellis` |
| 1.6 | `bleWaveformConfig` / kwargs | PHY mode, samples/symbol, channel index, access address, (CRC init, whitening on/off) — the name/value surface. | classdef / kwarg parsing |
| 1.7 | channel↔frequency map | 40 RF channels (2.402–2.480 GHz, 2 MHz spacing); advertising channels 37/38/39; data channels 0–36. Baked table. | lookup table |
| 1.8 | spectrum / time view | `spectrumAnalyzer`/`timescope`/`pwelch` of the IQ waveform (display only). | DSP scopes |

**Headline-within-tier (whole-roadmap tracer-bullet)**: `ble_ber_awgn.m` —
for each of the 4 PHY modes, `bleWaveformGenerator` → `awgn` → `bleIdealReceiver`
→ `biterr`, report the BER at a few Eb/No points (deterministic with a seeded
RNG, the Stats/Comm precedent).

**Compile/Execute wiring**: new `runtime/toolbox/bluetooth/runtime_bluetooth.cpp`;
register `bleWaveformGenerator`/`bleIdealReceiver` in `Resolver.cpp`; the
name/value `Mode=`/`SamplesPerSymbol=`/… kwargs read in the runtime (the
`fspecial`/`nwalign` option-string path); the waveform is a complex-matrix
return (the shipped `matlab_mat_c` lane); the bit output a double/int8 column.

**REPL/JIT + Debug**: waveforms + bit vectors are plain matrices → already
render in the REPL and DAP panes. Mind the recurring **ReplMode workspace
round-trip** trap (complex-matrix and string-kwarg values must survive the
`ws_get`/`ws_set` round-trip — see [`repl_jit_cross_unit_gap.md`](repl_jit_cross_unit_gap.md)
and the Bioinformatics char-literal fix).

---

## 3. Bt-Tier-2 — Bluetooth BR/EDR PHY: waveform + reception ✅

Goal: the classic-Bluetooth (BR/EDR) PHY — GFSK for Basic Rate, DPSK for
Enhanced Data Rate, across the BR/EDR packet-type catalogue.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `bluetoothWaveformGenerator(bits, cfg)` | BR: GFSK (1 Mb/s). EDR: π/4-DQPSK (2 Mb/s) / 8-DPSK (3 Mb/s) on the payload after the GFSK header. Access code + header + payload framing. | `gaussdesign`/`fskmod`, `dpskmod` |
| 2.2 | `bluetoothIdealReceiver(rx, cfg)` | Inverse: GFSK/DPSK demod + deframe + FEC decode + HEC/CRC check. | `fskdemod`, `vitdec` |
| 2.3 | `bluetoothPacketConfig` / packet-type catalogue | `ID`/`NULL`/`POLL`/`FHS`/`DM1`/`DH1`/`DM3`/`DH3`/`DM5`/`DH5`/`HV1`/`HV2`/`HV3`/`EV3`/`2-DH`/`3-DH` — per-type field layout (baked table). | classdef + table |
| 2.4 | BR/EDR FEC | 1/3 repetition (header) + 2/3 shortened (15,10) Hamming (payload) + HEC. | `hamming_encode`/`_decode` |
| 2.5 | BR/EDR CRC-16 + HEC | generator `0x1021`; header-error-check 8-bit. | `crc_generate` (baked poly) |
| 2.6 | BR/EDR whitening | 7-bit LFSR (x⁷+x⁴+1) seeded by the clock. | hand-coded (shared with 1.3) |

**Headline-within-tier**: `bredr_ber.m` — BR (DM1/DH1) and EDR (2-DH1/3-DH1)
BER vs Eb/No through `awgn`.

**Compile/Execute wiring**: same `runtime_bluetooth.cpp`;
`bluetoothWaveformConfig` is a classdef (prelude-triggered) carrying the
packet-type + modulation + payload settings; the generator/receiver are
matrix-in/matrix-out builtins.

---

## 4. Bt-Tier-3 — Protocol Data Units (config-object + gen/decode) ✅

Goal: build and parse the Bluetooth packet/PDU byte structures — the
link-layer / L2CAP / ATT / GAP layer the UG's "Generate and Decode PDUs"
tutorials cover.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `bleLLDataChannelPDU(cfg[, payload])` / `bleLLDataChannelPDUDecode(bits)` | LL data-channel PDU: header (LLID/NESN/SN/MD/length) + payload + MIC; gen → bit vector, decode → cfg + payload + status. | byte packing, CRC |
| 3.2 | `bleLLDataChannelPDUConfig` (classdef) | LLID / sequence numbers / length / CTEInfo fields. | classdef |
| 3.3 | `bleLLControlPDU` / `bleLLAdvertisingChannelPDU` (+ Config + Decode) | Control-PDU opcodes; advertising-PDU types (ADV_IND/SCAN_REQ/CONNECT_IND/…). | byte packing |
| 3.4 | `bleL2CAPFrame` / `bleL2CAPFrameConfig` (+ decode) | L2CAP framing (length + CID + payload), incl. LE credit-based flow-control fields. | byte packing |
| 3.5 | `bleATTPDU` / `bleATTPDUConfig` (+ decode) | ATT opcodes (read/write/notify/…) + attribute handle + value. | byte packing |
| 3.6 | `bleGAPDataBlock` / `bleGAPDataBlockConfig` (+ decode) | GAP advertising-data AD structures (flags / local name / service UUIDs). | byte packing |
| 3.7 | `bluetoothPacket` (BR/EDR) | BR/EDR baseband packet assembly/parse (shared with Tier-2 framing). | Tier-2 |

**Headline-within-tier**: `ble_ll_pdu_roundtrip.m` — build an LL data PDU
with `bleLLDataChannelPDU`, decode it with `bleLLDataChannelPDUDecode`, and
confirm the round-trip recovers the header fields + payload.

**Compile/Execute wiring**: the `*Config` objects are classdefs (the
`phytree` alloc-then-populate recipe); gen functions return a bit/byte column,
decode functions return a config object + payload (struct or multi-return).
Byte packing is plain matrix/bit slicing in the runtime.

---

## 5. Bt-Tier-4 — Channel selection + frequency hopping + channel models ✅

Goal: adaptive frequency hopping and the propagation channel — what turns a
single-packet PHY into a hopping link.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | `bleChannelSelection` (System object) | Channel Selection Algorithm #1 (additive hop) and #2 (permutation/PRNG over the used-channel map); `UsedChannels`, `HopIncrement`. `frequencyHop()` step-call → next channel index. | DSP System-Object fold |
| 4.2 | channel↔frequency map | (shared with 1.7) 40 LE / 79 BR/EDR channels; advertising vs data. | lookup table |
| 4.3 | `bluetoothLEChannel` / `bluetoothBREDRChannel` | Apply path loss + multipath fading + AWGN to a waveform (link impairment). | Comm channel surface, `awgn` |
| 4.4 | AFH channel map | good/bad channel classification → remap (Algorithm #1/#2 specific). | bit map |

**Headline-within-tier (Tier-4 tracer-bullet)**: `ble_freq_hopping.m` — the
UG's frequency-hopping example: generate per-packet LE waveforms whose channel
index comes from `bleChannelSelection` (Algorithm #1/#2), confirm the hop
sequence and the channel→frequency offsets.

**Compile/Execute wiring**: `bleChannelSelection` follows the `dsp.*`
parser-fold + `step` System-Object pattern (parse `bleChannelSelection(...)`
→ store config; `frequencyHop()` → `matlab_bluetooth_csa_step`);
`bluetoothLEChannel` is a classdef channel object reusing the Comm channel/
`awgn` runtime.

---

## 6. Bt-Tier-5 — Localization: direction finding + channel sounding ✅ (AoA core)

Goal: the Bluetooth 5.1+ positioning surface — angle of arrival/departure
(direction finding) and channel-sounding ranging.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | CTE waveform | Constant-Tone-Extension append to a LE waveform (the direction-finding signal). | Tier-1 waveform |
| 5.2 | `bleAngleEstimationConfig` / `bleAngleEstimate` | AoA/AoD estimation from the per-antenna IQ samples of the CTE (MUSIC / phase-difference over an antenna array). | phased-array steering, `eig`/FFT |
| 5.3 | antenna switching / IQ extraction | switch-pattern + slot sampling of the CTE across array elements. | matrix slicing |
| 5.4 | channel sounding ranging | phase-based (PBR) + round-trip-timing (RTT) distance estimation between two devices. | phase unwrap, `fft` |
| 5.5 | position estimation | 2-D/3-D lateration / angulation from multiple AoA/distance measurements. | `lsqnonlin` (Optim, shipped) |

**Headline-within-tier**: `ble_aoa_estimate.m` — synthesize a CTE waveform
seen by a uniform linear array at a known angle, estimate the AoA with
`bleAngleEstimate`, and confirm it recovers the angle.

**Compile/Execute wiring**: `bleAngleEstimationConfig` is a classdef; the
estimator reuses the eigen/FFT surface; position estimation reuses the shipped
Optim `lsqnonlin`. The deep-learning positioning examples are carved (§9).

---

## 7. Bt-Tier-6 — Test & Measurement + link-level simulation ✅ (measurement core)

Goal: the RF-PHY conformance measurement functions + a complete link-level
simulation (fading + path loss). The node-level network simulator is carved
down (§9).

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | LE RF-PHY Tx measurements | modulation characteristics (Δf1/Δf2 frequency deviation), carrier frequency offset + drift, in-band emissions, output power. | FFT, phase discriminator |
| 6.2 | LE RF-PHY Rx measurements | sensitivity / PER, C/I + blocking, intermodulation. | Tier-1 + `awgn` |
| 6.3 | BR/EDR RF-PHY tests | EDR modulation accuracy (DEVM), carrier frequency stability, power/spectrum (ACP), BR modulation characteristics. | Tier-2 + FFT |
| 6.4 | PER report integrity | LE packet-error-rate report test. | `biterr`/PER |
| 6.5 | link-level simulation | end-to-end: waveform → fading channel + path loss + RF impairments + AWGN → receiver, with corrections (CFO/timing). | Tier-1/2 + Comm channel |

**Headline-within-tier**: `ble_link_level.m` — an LE link through a path-loss +
AWGN channel with a frequency offset, corrected at the receiver, reporting PER.

**Compile/Execute wiring**: measurement functions are matrix-in →
scalar/struct-out builtins over the FFT + phase surface; the link-level driver
chains the Tier-1 waveform, the Tier-4 channel, and `awgn`.

---

## 8. Phasing & effort summary

The roadmap groups the six tiers into **three shippable phases**:

| Phase | Tiers | Theme | New algorithm | Effort | Closes |
|---|---|---|---|---|---|
| **A — PHY core** | T1 + T2 | LE + BR/EDR waveform generation & ideal reception | packet framing, GFSK/DPSK wrappers, whitening LFSR, coded-PHY FEC, packet-type catalogue | **~4 wk** | `ble_ber_awgn.m` headline + `bredr_ber.m` |
| **B — Packets + hopping** | T3 + T4 | PDU gen/decode + channel selection / AFH / channel models | byte-packing PDU codecs, `*Config` classdefs, `bleChannelSelection` System object | **~2.5 wk** | `ble_ll_pdu_roundtrip.m`, `ble_freq_hopping.m` |
| **C — Localization + T&M** | T5 + T6 | Direction finding + channel sounding + RF-PHY conformance + link-level | AoA/AoD estimator, ranging, conformance measurements, fading link driver | **~4.5 wk** | `ble_aoa_estimate.m`, `ble_link_level.m` |

**Full toolbox ≈ 11 weeks.** **Phase A alone (~4 wk) is the recommended first
cut** — it is self-contained (rides the shipped Comm/DSP kernels), closes the
canonical LE/BR-EDR BER workflow, and unblocks the rest (Phase B's PDUs feed
the Phase-A waveform generator; Phase C's measurements run on Phase-A
waveforms).

**Per-tier dependency notes**:
- T2 is independent of T1 (BR/EDR vs LE) but shares the framing/whitening
  helpers — ship T1 first (the headline), then T2.
- T3 (PDUs) feeds the waveform generator's `bits` input but is independently
  testable (gen→decode round-trip needs no waveform).
- T4's `bleChannelSelection` is the first System object; `bluetoothLEChannel`
  reuses the Comm channel surface.
- T5 + T6 depend on the Tier-1 waveform; T5 introduces the phased-array AoA
  path (the heaviest new numeric piece after the PHY).

---

## 9. Carve-outs (explicitly out of scope)

- **Simulink models**: the entire `BluetoothFullDuplexModel` /
  coexistence-model surface (chapters 1–2 are Simulink + Stateflow ARQ);
  modeled in the `.mflow` flowchart frontend only if ever needed, not here.
- **SDR hardware**: `sdrtx`/`sdrrx`, ADALM-PLUTO, `comm.BasebandFileWriter`-to-
  radio, the "Using SDR" transmit/receive examples — no hardware in this lane
  (the bundled-file path is a possible follow-on).
- **Apps / scopes**: Constellation Diagram, Eye Diagram, Spectrum Analyzer /
  Signal Analyzer apps, the interactive viewers — gating tests stay headless
  (numeric BER / hop-sequence / decoded-field assertions); examples may call
  `spectrumAnalyzer`/`timescope` but the gating surface does not depend on
  rendering.
- **Node-level network simulator** (chapter 6): the discrete-event
  `wirelessNetworkSimulator` + `bluetoothLENode` / `bluetoothNode` /
  `bluetoothLEAudioNode`, piconet/scatternet scheduling, `addNodes`/
  `addTrafficSource`, statistics collection — a large classdef + event-engine
  subsystem deferred to a dedicated follow-on roadmap.
- **Bluetooth mesh** (chapter 6/8): provisioning, friendship, managed
  flooding, `bluetoothMeshProfileConfig`, the wireless-sensor-network examples.
- **LE Audio**: the LC3 codec, multistream/broadcast audio, Auracast,
  hearing-aid scenarios (chapter 6) — a codec subsystem of its own.
- **Coexistence with 5G NR / WLAN** (chapter 2): needs the 5G Toolbox + WLAN
  Toolbox waveforms (not in the tree); LBT / packet-traffic-arbitration carved.
- **Deep-learning positioning** (chapter 3 "Bluetooth LE Positioning with Deep
  Learning") — reuses the shipped Deep Learning autodiff but the demo wiring is
  deferred.
- **PCAP capture / log + Signal Analyzer** (chapter 9) and **C/C++ code
  generation** (chapter 7).

---

## 10. Compiler traps to watch (from sibling-toolbox experience)

- **Complex-matrix returns through ReplMode**: `bleWaveformGenerator` returns a
  `matlab_mat_c` (complex IQ). Confirm the complex value survives the JIT/DAP
  `ws_set`/`ws_get` round-trip (the Bioinformatics char-literal regression
  showed the ReplMode store path is a distinct lane — run `jit_parity_sweep.py
  --gate`, not just AOT).
- **Mode/packet-type string args**: `bleWaveformGenerator(bits,Mode="LE1M")`
  passes a string that must resolve in the runtime; detect by arg type (the
  `imread('f.png')` const-char path) — and remember the **top-level
  char-literal-assignment ReplMode fix** (single-quoted `'LE1M'` must
  materialize to a `matlab_string` for the workspace store).
- **Four runtime-source lists**: a new `runtime_bluetooth.cpp` must be
  registered in **CMakeLists.txt** (×2: sources + strict-cast), **the Run-test
  harness `test/Run/run_tests.sh`**, AND **the examples sweep
  `test/Examples/run_sweep.sh`** — missing the last two gives LINK regressions
  in those lanes (the Bioinformatics lesson). The JIT/DAP gate is a 4th lane.
- **System-Object fold**: `bleChannelSelection(...)` + `frequencyHop()` must
  follow the `dsp.*` parser-fold + `step` precedent; the constructor stores
  config, the call-syntax forwards to `matlab_bluetooth_csa_step`.
- **CMake build enforces `-Werror=old-style-cast`** (harness doesn't) — use
  `static_cast` throughout `runtime_bluetooth.cpp`; add it to the strict-no-C-
  cast list (the Image/Bioinformatics precedent).
- **Multi-return decode**: `[cfg,payload,status]=bleLLDataChannelPDUDecode(...)`
  uses the multi-output splitter; `numel` of a runtime result is 0 and
  `~`-ignore-output is unsupported (the recurring Stats trap).
- **`fprintf` of a comparison / reduction result** doesn't lower — print BER
  via `%.4g` of the ratio, not `fprintf('%d', a==b)` (recurring trap); `%d` of
  a double prints 0 → use `%.0f`.
- **Deterministic BER**: seed the RNG (`rng default` / `RandStream`) so BER
  verdicts are reproducible — and pin platform-stable thresholds, not exact
  floats, for the chaotic-libm-sensitive curves (the RL/Stats precedent).

---

## 11. Test & example surface (gating)

- **Gating tests** (`test/Run/bluetooth_*.m`), one per tier headline:
  `bluetooth_ble_phy` (T1: `bleWaveformGenerator`→`awgn`→`bleIdealReceiver`,
  zero-noise round-trip recovers the bits exactly; a mid-SNR BER under a fixed
  threshold), `bluetooth_bredr_phy` (T2: BR + EDR round-trip),
  `bluetooth_ll_pdu` (T3: LL/L2CAP/ATT PDU gen→decode field round-trip),
  `bluetooth_channel_sel` (T4: `bleChannelSelection` Algorithm #1/#2 hop
  sequence + channel→frequency map), `bluetooth_aoa` (T5: AoA recovery of a
  known angle), `bluetooth_rf_tests` (T6: a frequency-deviation / CFO
  measurement on a synthesized waveform).
- **Examples** (`examples/bluetooth/`) mirroring the UG: `ble_ber_awgn.m`
  (headline), `bredr_ber.m`, `ble_ll_pdu_roundtrip.m`, `ble_freq_hopping.m`,
  `ble_aoa_estimate.m`, `ble_link_level.m`.
- **Determinism**: zero-noise round-trips recover bits exactly (PHY is
  deterministic); BER curves use a seeded RNG with platform-stable thresholds;
  PDU gen/decode is exact; hop sequences are deterministic given the algorithm
  + used-channel map. Any plotting (`spectrumAnalyzer`/`timescope`) is
  display-only — gating asserts on numeric outputs.
- **Bundled data**: small access-address / PDU fixtures inline in the `.m`
  (no network, no SDR) — the precedent of the Bioinformatics inline sequences.

---

## 12. One-line status for MEMORY.md (when shipped)

> Bluetooth Toolbox — roadmap `docs/bluetooth_toolbox_roadmap.md` (R2026a UG).
> High-reuse over the shipped Comm/DSP stack: GFSK = `gaussdesign`+`fskmod`,
> EDR DPSK = `dpskmod`, FEC = `convenc`/`vitdec`/`hamming`, CRC = `crc_*`,
> AWGN/BER = `awgn`/`comm.ErrorRate`/`biterr` — only packet framing +
> whitening LFSR + channel-selection + coded-PHY pattern-map are new. 6 tiers /
> 3 phases: A=T1+T2 LE+BR/EDR PHY waveform/receiver (~4wk, headline
> `ble_ber_awgn.m`), B=T3+T4 PDUs + `bleChannelSelection`/AFH (~2.5wk), C=T5+T6
> direction-finding/channel-sounding + RF-PHY conformance + link-level
> (~4.5wk). ~11wk full. Config objects + `bleChannelSelection` reuse the
> phytree/DataMatrix classdef + dsp.* System-Object recipes. Carved: Simulink,
> SDR, apps/scopes, `wirelessNetworkSimulator`/`bluetoothLENode` network +
> mesh + LE-Audio LC3, 5G/WLAN coexistence, deep-learning positioning, codegen.
