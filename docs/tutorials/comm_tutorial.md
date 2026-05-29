# Communications Toolbox — Tutorial

The Communications Toolbox surface in matlab_llvm is a **function-form** base layer: bit/symbol sources, modulators, channels, channel codes, equalisers, and modern iterative decoders, all reachable from the MLIR → LLVM compile path with no System-Object or string-arg dispatch required. To keep runtime dispatch numeric, mode selectors (Gray vs binary, RRC vs RC, climate codes, etc.) are passed as small integer tags rather than option strings, and multi-return MATLAB calls like `biterr` are split into single-return variants (`biterr`/`biterrCount`/`biterrK`). This tutorial is grounded in the shipped examples under `examples/comm/`.

## Supported features

- **Tier 1 — sources & metrics:** `rng`/`rngDefault`/`rngShuffle`/`rngGet`/`rngSet`, `randi`, `randsrc`, `randerr`, `int2bit`/`bit2int`, `de2bi`/`bi2de`, `awgn` (real/complex polymorphic), `biterr`/`biterrCount`/`biterrK`, `symerr`/`symerrCount`.
- **Tier 2 — modulation & pulse shaping:** `pammod`/`pamdemod`, `pskmod`/`pskdemod`, `qammod`/`qamdemod`/`qamdemodBit`/`qamdemodLlr`, `genqammod`, `rcosdesign`, `gaussdesign`, `berawgn`, `qfunc`, `scatterplot`.
- **Tier 3 — block & convolutional codes:** `crcGenerate`/`crcCheck`/`crcStrip`, `poly2trellis`/`convenc`/`vitdec` (with `oct2dec` octal bridge), `hammingEncode`/`hammingDecode`/`hammgenParity`, `intrlv`/`deintrlv`.
- **Tier 4 — equalisation, sync, impairments:** `lms`/`rls`/`cma`/`dfe`; `costasPll`, `symbolSyncMM`, `preambleDetect`; `phaseFreqOffset`, `iqimbal`, `memorylessNl` (cubic/Saleh/Rapp/Ghorbani), `phaseNoise`; `vitdecSoft` soft-decision Viterbi.
- **Tier 5 — OFDM, fading, MIMO:** `ofdmmod`/`ofdmdemod`, `rayleighChannel`/`ricianChannel`, `ostbcEncode`/`ostbcCombine` (Alamouti), `mlDetect`.
- **Tier 6 — spreading & quantisation:** `pnSequence`, `goldSequence`, `hadamard`, `walshCode`, `quantiz`/`quantizApply`, `lloydsQuant`, `compandMu`/`compandA` (μ-law / A-law), `dpcmEncode`/`dpcmDecode`.
- **Tier 7 — modern iterative codes:** `polarEncode`/`polarSCdecode`, `ldpcEncode`/`ldpcDecodeMS` (min-sum), `turboEncode`/`turboDecode` (PCCC, max-log-MAP/BCJR).

## Build & run

```bash
build/matlabc -emit-llvm examples/comm/ber_awgn_uncoded.m > /tmp/ber_awgn_uncoded.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/ber_awgn_uncoded.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/ber_awgn_uncoded
/tmp/ber_awgn_uncoded
```

The examples directory also ships `runtime/build_and_run.sh examples/comm/<name>.m /tmp/<name>` as a convenience wrapper for the same compile-and-execute path.

## Worked examples

### Uncoded BPSK BER over AWGN  (`examples/comm/ber_awgn_uncoded.m`)

The canonical Tier-1 Monte-Carlo loop: source → map → AWGN → threshold detect → `biterr`, overlaid against the closed-form Q-function curve.

```matlab
rng(2026);
N   = 50000;             % bits per SNR point
snr = [0 2 4 6 8 10];    % dB (matlab awgn convention)
for k = 1:6
    s = snr(k);
    tx_bits = randi(2, N, 1) - 1;        % {0, 1}
    tx_sym  = 1 - 2 * tx_bits;            % {+1, -1} BPSK
    rx = awgn(tx_sym, s);
    rx_bits = (rx < 0);                   % hard-decision threshold at 0
    sim_ber = biterr(tx_bits, rx_bits);
    snr_lin = pow_10(s / 10.0);
    theo_ber = q_func(sqrt(snr_lin));
    fprintf('  %5.1f   | %.6f    | %.6f\n', s, sim_ber, theo_ber);
end
```

Note the AWGN convention: `awgn(x, snr_dB)` treats `snr_dB = 10·log10(signal/noise)`, so for unit-energy BPSK the hard-decision BER is `Q(sqrt(SNR_lin))` — 3 dB offset from the textbook `Q(sqrt(2·Eb/N0))` because `Eb/N0 = SNR/2` for one-axis real BPSK. `biterr` returns the ratio (the count-only form is `biterrCount`).

### 16-QAM Monte-Carlo vs theory  (`examples/comm/ber_qam_montecarlo.m`)

Tier-2 closure: `qammod` → `awgn` → `qamdemod`, with the symbol-energy BER measured by `biterrK` (treats each label as a `k`-bit symbol) against `berawgn`.

```matlab
M = 16; k = 4; N = 20000;
ebn0 = [4 6 8 10 12 14];
k_dB = 10 * log(k) / log(10);
for i = 1:6
    eb = ebn0(i);
    data    = randi(M, N, 1) - 1;
    tx      = qammod(data, M, 1, 1);     % Gray (tag 1), unit avg power (tag 1)
    rx      = awgn(tx, eb + k_dB);        % shift Eb/N0 to symbol SNR
    rx_data = qamdemod(rx, M, 1, 1);
    sim_ber = biterrK(data, rx_data, k);
    theo_ber = berawgn(eb, M, 2);         % mod_code 2 = QAM
end
```

The fourth argument to `qammod`/`qamdemod` is the `unit_avg` tag (1 = unit-mean-power constellation); the third is the mapping order (1 = Gray). `berawgn`'s `mod_code` is `0 PAM / 1 PSK / 2 QAM / 3 DPSK / 4-5 FSK`.

### Coded vs uncoded BER  (`examples/comm/ber_coded_vs_uncoded.m`)

Tier-3 closure comparing uncoded BPSK against Hamming(7,4) and a (171,133)₈ K=7 convolutional code with hard-decision Viterbi. `oct2dec` bridges the textbook octal generators into the decimal form `poly2trellis` expects.

```matlab
gens = [oct2dec(171), oct2dec(133)];
t    = poly2trellis(7, gens);
% ... per Eb/N0 point:
snr_c    = eb + 10 * log(0.5) / log(10);   % rate-1/2 -> -3 dB channel SNR
msg      = randi(2, N, 1) - 1;
code     = convenc(msg, t);
tx       = 1 - 2 * code;
rx       = awgn(tx, snr_c);
decoded  = vitdec(rx < 0, t, 35, 0, 1);    % tblen 35, opmode 0 (trunc), hard (1)
ber_c    = biterr(msg, decoded);
```

The convolutional code crosses over and beats uncoded BPSK by roughly 2× at 7 dB Eb/N0. `vitdec`'s last two args are `opmode` (0 truncated / 1 terminated) and `dectype` (0 unquantised / 1 hard); the soft-decision path is `vitdecSoft` in Tier 4.

### OFDM loopback over AWGN  (`examples/comm/ofdm_awgn.m`)

A single OFDM symbol of 64 QPSK subcarriers with a 16-sample cyclic prefix, recovered with `ofdmdemod` + `mlDetect`.

```matlab
Nfft = 64; Lcp = 16; M = 4;
data = randi(M, Nfft, 1) - 1;
sym  = pskmod(data, M, pi/4, 1);     % unit-power QPSK
tx   = ofdmmod(sym, Nfft, Lcp);
rx   = awgn(tx, 15);
rx_data  = ofdmdemod(rx, Nfft, Lcp);
alpha    = pskmod((0:M-1)', M, pi/4, 1);
data_hat = mlDetect(rx_data, alpha);
fprintf('symerr: %.4f\n', symerr(data, data_hat));
```

`ofdmmod` takes an `Nfft × Nsym` complex matrix (rows = subcarriers); pilots/guards are caller-side compositions (zero the relevant rows). Note that `size`/indexing on a complex result reads the real-matrix layout, so the example calls `size(abs(tx), 1)` when it needs shape introspection on a complex array.

### Alamouti diversity  (`examples/comm/alamouti_diversity.m`)

Alamouti 2-Tx encode → flat channel `(h1,h2)` + AWGN → maximum-ratio combine → `mlDetect`, beating the single-Tx baseline thanks to the combiner's coherent gain. `ostbcCombine` takes the channel as four separate real scalars (no complex-scalar dispatch yet).

```matlab
encoded = ostbcEncode(tx);            % N x 2
h_eff_re = (h1_re + h2_re) / sqrt(2);
h_eff_im = (h1_im + h2_im) / sqrt(2);
y_n = awgn(tx * complex(h_eff_re, h_eff_im), snr_dB);
rx_alam = ostbcCombine(y_n, h1_re, h1_im, h2_re, h2_im);
data_hat_alam = mlDetect(rx_alam, alpha);
```

### Modern codes (Polar / LDPC / Turbo)  (`examples/comm/modern_codes_ber.m`)

Tier-7 closure: at SNR = 5 dB over a 64-bit message, uncoded BPSK shows errors while Polar(128,64) SC, Turbo PCCC, and LDPC(6,3) min-sum all decode cleanly. The example builds the polar frozen mask by hand, derives LLRs as `2*rx`, and runs `turboDecode` with a shift-by-11 interleaver and 6 BCJR iterations:

```matlab
cw_pol  = polarEncode(u_pol, N_pol);            % frozen mask is caller-designed
u_hat   = polarSCdecode(2 * rx_pol, frozen_pol, N_pol);
code_tur = turboEncode(msg, t, perm);            % [sys; parity1; parity2]
dec_tur  = turboDecode(llr_sys, llr_p1, llr_p2, t, perm, 6);
cw       = ldpcEncode(chunk, P_ldpc);            % G = [I | P]
dec_cw   = ldpcDecodeMS(2 * rx, H_ldpc, 20);     % H = [P^T | I], 20 iters
```

### Other examples (briefly)

- **`cdma_walsh_demo.m`** — two-user Walsh-Hadamard CDMA round-trip; verifies orthogonality via the `‖A+B‖² − ‖A−B‖²` identity, then spread/sum/despread both users to 0 symbol errors.
- **`ber_soft_vs_hard.m`** — hard vs soft Viterbi on the (171,133)₈ code; soft (`vitdecSoft`) sits ~3 dB to the left of hard.
- **`impairment_demo.m`** — the four RF impairments (`phaseFreqOffset`, `iqimbal`, `memorylessNl` Rapp, `phaseNoise`) applied to a clean QPSK constellation in isolation and as a chain.
- **`pulse_shape_demo.m`** — RRC / full-RC / Gaussian FIRs via `rcosdesign`/`gaussdesign`, plus the matched-filter `RRC ⊗ RRC ≈ RC` Nyquist-zero check.
- **`source_bits.m`** — bit/symbol source primitives (deterministic seeding, custom/weighted alphabets, MSB-first vs LSB-first round-trips).
- **`tier1_smoke.m` … `tier7_smoke.m`** — one canonical call per entry in each tier; the fastest reference for exact call signatures.

## Limitations & carve-outs

From `docs/comm_toolbox_roadmap.md §13`:

- **System Objects** (`comm.LinearEqualizer`, `comm.RayleighChannel`, `comm.MIMOChannel`, etc.) are gated on the System-Object lowering fix; the function-form equivalents ship today.
- **Simulink Communications block library** (~150 blocks) — Simulink is not in scope.
- **SDR / hardware-in-the-loop** (Pluto-SDR, USRP, RTL-SDR drivers) — hardware drivers out of scope.
- **Separate MathWorks products** — 5G / WLAN / LTE / Bluetooth / Zigbee Toolboxes are out of scope.
- **BCH / RS codes** deferred; **complex MIMO fading** (`comm.MIMOChannel`) deferred; **`bercoding`** deferred (only `berawgn` shipped).
- **Polar frozen-set design** (3GPP reliability sequence) is not shipped — the caller supplies the frozen mask.
- **Interactive constellation/eye scopes** are out of scope; static PNG/SVG via `runtime/plot/` is possible.

## See also

- Roadmap / design: [`../comm_toolbox_roadmap.md`](../comm_toolbox_roadmap.md) — the bundled Comm + RF + Antenna + Propagation plan; Comm tiers + numeric-tag tables + closure tests.
- Examples directory: `examples/comm/` (see its `README.md` for the full numeric-tag reference per tier).
