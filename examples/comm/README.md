# examples/comm — Communications Toolbox Tier-1 demos

Function-form base layer per `docs/comm_toolbox_roadmap.md §2`. Runs through the standard compile-and-execute path:

```bash
bash runtime/build_and_run.sh examples/comm/<name>.m /tmp/<name>
/tmp/<name>
```

| Example | Tier | What it demonstrates |
|---|---|---|
| `comm_tier1_smoke.m` | 1 | One canonical call per Tier-1 entry: `rng` + `randi` + `randsrc` + `randerr` + `int2bit` + `bit2int` + `de2bi` + `bi2de` + `awgn` + `biterr` + `symerr`. |
| `source_bits.m` | 1 | Bit / symbol source primitives: deterministic seeding, uniform integer source, custom-alphabet sampling, weighted alphabet, crafted error vectors, MSB-first vs LSB-first round-trips. |
| `ber_awgn_uncoded.m` | 1 | Sample-and-count BER curve for uncoded BPSK over AWGN — canonical Tier-1 Monte-Carlo loop (`source → map → AWGN → threshold → biterr`) at SNR ∈ {0, 2, 4, 6, 8, 10} dB. |
| `tier2_smoke.m` | 2 | One canonical call per Tier-2 entry: `pammod` + `pamdemod` + `pskmod` + `pskdemod` + `qammod` + `qamdemod` + `qamdemodBit` + `genqammod` + `rcosdesign` + `gaussdesign` + `berawgn` + `qfunc` + `scatterplot`. |
| `pulse_shape_demo.m` | 2 | RRC + full RC + Gaussian (GSM/Bluetooth BT) pulse-shaping FIRs, plus the canonical "RRC ⊗ RRC ≈ raised-cosine" matched-filter cascade with the Nyquist-zero check at integer symbol multiples. |
| `ber_qam_montecarlo.m` | 2 | **Tier-2 closure** — 16-QAM Monte-Carlo `source → qammod → awgn → qamdemod → biterrK` against the `berawgn` closed-form curve at Eb/N0 ∈ {4, 6, 8, 10, 12, 14} dB. Tracks theory within ~10% relative from 4 dB onward at 20 k symbols/point. |
| `tier3_smoke.m` | 3 | One canonical call per Tier-3 entry: `crcGenerate` / `crcCheck` / `crcStrip`, `poly2trellis` / `convenc` / `vitdec` (with `oct2dec` bridge for textbook octal generators), `hammgenParity` / `hammingEncode` / `hammingDecode` (every-position 1-bit error correction), `intrlv` / `deintrlv`. |
| `ber_coded_vs_uncoded.m` | 3 | **Tier-3 closure** — coded-vs-uncoded BER curve over BPSK + AWGN at Eb/N0 ∈ {2, 3, 4, 5, 6, 7} dB. Compares uncoded BPSK against Hamming(7, 4) and (171, 133)₈ K=7 hard-decision Viterbi. Conv crosses over and beats uncoded by ~2× at 7 dB Eb/N0. |
| `tier4_smoke.m` | 4 | One canonical call per Tier-4 entry: `lms` / `rls` / `cma` / `dfe` adaptive equalisers; `costasPll`, `symbolSyncMM`, `preambleDetect` sync; `phaseFreqOffset`, `iqimbal`, `memorylessNl`, `phaseNoise` impairments; `vitdecSoft` soft-decision Viterbi. |
| `ber_soft_vs_hard.m` | 4 | **Tier-4 closure** — hard vs soft Viterbi BER curves on (171, 133)₈ K=7 convolutional + BPSK + AWGN at Eb/N0 ∈ {1, 2, 3, 4, 5} dB. Soft sits ~3 dB to the left of hard (at 50 k bits/point: hard 0.120 / soft 0.0051 at Eb/N0 = 5 dB). |
| `impairment_demo.m` | 4 | Applies each of the four canonical RF impairments to a clean QPSK constellation in isolation (`phaseFreqOffset`, `iqimbal`, `memorylessNl` Rapp, `phaseNoise`) plus a combined chain, reporting the per-step distortion via `norm(abs(y) - abs(clean))`. |
| `tier5_smoke.m` | 5 | One canonical call per Tier-5 entry: `ofdmmod` + `ofdmdemod` round-trip, `rayleighChannel` + `ricianChannel` multi-path channels (with two-tap delay / gain vectors), `ostbcEncode` Alamouti 2-Tx encoder, `mlDetect` per-symbol Euclidean ML decision against a 4-PSK alphabet. |
| `ofdm_awgn.m` | 5 | Single-symbol OFDM loopback over AWGN at SNR = 15 dB: 64 QPSK subcarriers + CP = 16 → 0 errors after `ofdmdemod` + `mlDetect`. |
| `alamouti_diversity.m` | 5 | **Tier-5 closure** — Alamouti 2-Tx encode → known scalar channel `(h1, h2)` + AWGN → maximum-ratio combine → `mlDetect`. At 10 dB SNR Alamouti reaches 0 errors vs the single-Tx baseline 0.0027 symerr (the combiner's coherent gain). |

## API conventions

To keep the runtime dispatch reachable without strings or function handles, the API uses numeric values everywhere and exposes the named-string variants of `rng` as separate functions:

- `rng(seed)` — set deterministic seed (integer, cast to f64).
- `rngDefault()` — replaces `rng('default')`.
- `rngShuffle()` — replaces `rng('shuffle')` (wall-clock derived).
- `s = rngGet()` — save the current state as a scalar (the 64-bit xorshift state cast to f64; round-trip works within the session).
- `rngSet(s)` — restore from a saved scalar.

The PRNG state is shared with the existing `rand` / `randn` primitives, so seeding via `rng(...)` is deterministic end-to-end across the rand/randn path AND the comm Tier-1 primitives.

### `biterr` / `symerr` return value

To stay inside the single-return dispatch convention, `biterr(x, y)` returns just the ratio (the second of MATLAB's `[nerr, ratio]` pair, since the ratio is what almost every script consumes). The matching count-only variants are `biterrCount(x, y)` and `symerrCount(x, y)`.

### k-bit symbol BER

`biterrK(x, y, k)` treats each entry of `x` / `y` as a k-bit symbol and counts the bit-mismatch ratio across the unpacked bits.

### AWGN polymorphism

`awgn(x, snr_dB)` dispatches on the descriptor magic — real-typed `x` produces real noise; complex `x` produces complex noise with `σ²/2` per axis so the total variance matches `signal_power / snr_lin`. Pass a third argument to `awgn(x, snr_dB, sigpower_dBW)` for an explicit signal-power baseline instead of the measured one.

### Tier-2 numeric tags

| Tag | Values |
|---|---|
| `order` (every modulator) | 0 = natural binary, 1 = Gray |
| `qammod` / `qamdemod` `unit_avg` | 0 = natural-power constellation, 1 = unit-mean-power (scale by `1/√(2(M−1)/3)`) |
| `rcosdesign` `shape` | 0 = root-raised-cosine ('sqrt'), 1 = full RC ('normal') |
| `berawgn` `mod_code` | 0 PAM / 1 PSK / 2 QAM / 3 DPSK / 4 FSK orthogonal coherent / 5 FSK orthogonal non-coherent |
| `qamdemod` flavours | `qamdemod(y, M, order, unit_avg)` → integer labels; `qamdemodBit(y, M, order, unit_avg)` → `N·log2(M)` MSB-first bits; `qamdemodLlr(y, M, order, unit_avg, noise_var)` → max-log LLR |

### Tier-3 numeric tags

| Tag | Values |
|---|---|
| `crcGenerate` / `crcCheck` / `crcStrip` `poly_int` | The lower `nbits` bits of the polynomial; the leading 1 is implicit. E.g. CRC-16-CCITT poly `0x11021` → `crcGenerate(bits, 4129, 16)`. |
| `poly2trellis` generators | Decimal integers. Use `oct2dec(171)` to convert from textbook octal (gives 121). |
| `vitdec` `opmode` | 0 = truncated, 1 = terminated (assume end-state 0) |
| `vitdec` `dectype` | 0 = unquantised, 1 = hard-decision (soft is a Tier-4 follow-on) |
| Hamming `m` | The parity check length; gives `n = 2^m - 1`, `k = n - m`. m=3 → Hamming(7, 4); m=4 → (15, 11). |
| Interleaver `perm` | 1-based permutation vector. `intrlv(data, perm)` writes `out(i) = data(perm(i))`; `deintrlv` is the exact inverse. |

### Tier-4 numeric tags

| Tag | Values |
|---|---|
| `costasPll` `M_psk` | 2 (BPSK squarer), 4 (QPSK 4-PSK error), other (atan2 generic) |
| `memorylessNl` `model_code` | 0 = cubic clipper (`p1` = saturation amplitude); 1 = Saleh (`p1` = α_a, `p2` = β_a, `p3` = α_p, `p4` = β_p); 2 = Rapp (`p1` = smoothness `p`, `p2` = `Asat`); 3 = Ghorbani-style 4-parameter form |
| `vitdecSoft` `opmode` | 0 = truncated, 1 = terminated (assume end-state 0) |
| `vitdecSoft` input | Real values where positive ⇒ favours bit = 0 (matches the `qamdemodLlr` convention) |

### Tier-5 conventions

| Entry | Convention |
|---|---|
| `ofdmmod(data, Nfft, cp_len)` | `data` is an `Nfft × Nsym` complex matrix (rows = subcarriers, columns = OFDM symbols). Pilots / nulls / guards are caller-side compositions: zero out the relevant subcarrier rows of `data` before calling. Output is `(Nfft + cp_len) · Nsym × 1` complex. |
| `ofdmdemod(samples, Nfft, cp_len)` | Inverse: strips the per-symbol cyclic prefix then FFTs each block. Returns `Nfft × Nsym` complex. |
| `rayleighChannel` / `ricianChannel` `delays_samples`, `gains_dB` | Both must have ≥ 2 elements (single-element `[0]` literal gets typed as scalar f64 and fails the ptr-arg dispatch). For a degenerate single-tap channel use `[0; 0]` paired with `[0; -200]` (the second path is then ~80 dB below the first). |
| `rayleighChannel` length convention | Output length = `length(x) + max(delays_samples)` (per-path delays extend the output beyond the input). |
| `ostbcCombine` channel gains | Pass real / imag components as four separate scalar args (no complex-scalar dispatch yet). Channel is assumed flat across the burst; for time-varying channels split the burst into coherence-time chunks. |
| Complex outputs `size` / indexing | The `size`/`length` runtime entries read the real-matrix layout; on a `matlab_mat_c` result they return garbage. Take `abs(...)` first when you need shape introspection or scalar indexing on the magnitude. |
