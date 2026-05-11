# examples/comm — Communications Toolbox Tier-1 demos

Function-form base layer per `docs/comm_toolbox_roadmap.md §2`. Runs through the standard compile-and-execute path:

```bash
bash runtime/build_and_run.sh examples/comm/<name>.m /tmp/<name>
/tmp/<name>
```

| Example | What it demonstrates |
|---|---|
| `comm_tier1_smoke.m` | One canonical call per Tier-1 entry: `rng` + `randi` + `randsrc` + `randerr` + `int2bit` + `bit2int` + `de2bi` + `bi2de` + `awgn` + `biterr` + `symerr`. |
| `source_bits.m` | Bit / symbol source primitives: deterministic seeding, uniform integer source, custom-alphabet sampling, weighted alphabet, crafted error vectors, MSB-first vs LSB-first round-trips. |
| `ber_awgn_uncoded.m` | Sample-and-count BER curve for uncoded BPSK over AWGN — canonical Tier-1 Monte-Carlo loop (`source -> map -> AWGN -> threshold -> biterr`) at Eb/N0 in {0, 2, 4, 6, 8, 10} dB. |

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
