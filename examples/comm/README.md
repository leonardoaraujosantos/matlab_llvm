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
