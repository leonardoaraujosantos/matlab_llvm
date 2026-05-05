# CocoTB Verification Harness Emission (`-emit-cocotb`)

This document scopes the `-emit-cocotb` mode of `matlabc`: an
**open-source alternative to MathWorks's HDL Verifier** that closes
the verification loop for the SystemVerilog backend. From a single
`.m` source file, `matlabc` generates a self-contained
[CocoTB](https://www.cocotb.org/) testbench directory that drives
the emitted SystemVerilog DUT and the emitted Python reference
model in lockstep against random vectors, asserting cycle-by-cycle
equality.

**Status: shipped — CI lane is 36/36.** The lane sweeps 36 of the
39 synthesizable HDL examples and asserts each verifies bit-exact
between the SV DUT and the matlab-emitted Python reference. The
remaining 3 fixtures (`cordic_pipe`, `cordic_step`, `cic_decimator`)
need per-op wrap insertion in the Python emitter — every binary
arith op result wraps to the result type's width, mirroring SV's
mid-computation truncation. Quality-of-life additions in this
ship: `% cocotb: latency(N)` source pragma, parallel CI runner
(~6.5× speedup), enriched mismatch diagnostics with fi-decoded
values + canonical fault hints + VCD pointer on first failure,
and `% cocotb: cover(<port>, min_bins=N)` coverage gates.

---

## Why It Matters

Until v1 of this feature shipped, the SV backend was verified only
by Verilator's `--lint-only -Wall` lane. Lint catches syntax,
signedness, width, and unused-signal issues, but it does **not**
prove the generated RTL *behaves* the same as the MATLAB source.

CocoTB-based co-simulation closes that gap. The flow:

1. The MATLAB function is the **golden reference**.
2. `matlabc` emits both the SystemVerilog DUT (via
   `-emit-systemverilog`) and a Python reference model (via
   `-emit-python`) — both honor the same `% hdl: port(...)` pragma
   surface and fi (Fixed-Point Designer) semantics.
3. The generated testbench drives **identical packed-bit stimulus**
   into the DUT and identical real-valued stimulus (after
   `pack_fi`/`unpack_fi` round-trip) into the Python reference.
4. Per cycle, the DUT output is sampled, decoded back to a real
   value via `unpack_fi`, and compared against the reference's
   return value.

The result is the same workflow MATLAB users get from HDL Verifier,
without the licensing or proprietary tooling: one source of truth,
two emissions, one harness.

---

## Quick Start

### Command-line surface

```sh
matlabc -emit-cocotb FILE.m \
    [-cocotb-out=DIR] \
    [-cocotb-vectors=N] \
    [-cocotb-latency=L] \
    [-cocotb-seed=N]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-emit-cocotb` | — | Selects the harness-emit mode. |
| `-cocotb-out=DIR` | `<input-dir>/<stem>_cocotb` | Output directory. Created if missing. |
| `-cocotb-vectors=N` | 100 | Number of random stimulus vectors driven. |
| `-cocotb-latency=L` | 0 | Pipeline latency. DUT output at cycle `k+L` is compared against the reference's response to cycle `k`'s inputs. See [Pipeline latency](#pipeline-latency-cocotb-latencyl). The same value can live in the source as `% cocotb: latency(L)`; the CLI flag wins when both are present. |
| `-cocotb-seed=N` | 42 | Seed for the harness's `random` calls. Override to explore different randomization schedules without editing the generated file. |

### Output directory layout

```
<stem>_cocotb/
    <stem>.sv             # DUT (matlabc -emit-systemverilog)
    <stem>_ref.py         # Reference model (matlabc -emit-python)
    test_<stem>.py        # CocoTB testbench (random-vector lockstep)
    cocotb_fi.py          # fi pack / unpack helpers (mirror of
                          # runtime/cocotb_fi.py, embedded so the
                          # harness directory is self-contained)
    matlab_runtime.py     # Python fi runtime, copied from the
                          # source tree so the reference imports
                          # work without PYTHONPATH gymnastics
    Makefile              # Verilator + CocoTB invocation
```

The harness directory is **self-contained**. Move it anywhere, drop
into it, and `make`:

```sh
$ matlabc -emit-cocotb examples/hdl/alu_16bit.m
matlabc: wrote CocoTB harness to examples/hdl/alu_16bit_cocotb \
    (3 inputs, 2 outputs, combinational, 100 vectors)

$ cd examples/hdl/alu_16bit_cocotb
$ make
...
** TESTS=1 PASS=1 FAIL=0 SKIP=0 **
```

### Dependencies

- [Verilator](https://www.veripool.org/wiki/verilator) (default `SIM`
  in the generated Makefile) — install with `brew install verilator`
  on macOS, your distro's package manager elsewhere.
- [CocoTB 2.x](https://www.cocotb.org/) — `pip install cocotb`.

The Makefile sets `SIM ?= verilator` so users can override with
`SIM=icarus make` (Icarus Verilog) or `SIM=questa make` etc. without
touching the harness.

---

## Design

### Source of truth for port specs

The harness emitter does not re-walk the lowered MLIR module to
derive port types. Instead it **parses the SV port list** from the
file `matlabc -emit-systemverilog` just produced. The SV emitter has
already type-refined every signal (direction, signedness, width,
name); rebuilding the same info from MLIR would mean replaying most
of the SV pipeline.

Recognized port shapes (one per line, possibly with trailing comma):

```
input  logic signed [W-1:0] name,
input  logic [W-1:0] name,
input  logic name,             // 1-bit (clk, rst_n, bool)
output ... (same forms)
```

Synthetic `clk` / `rst_n` are filtered into a `Sequential` flag
rather than appearing as harness-driven INPUTS. Vector / unpacked-
array ports (`logic ... [N]`) capture the array length and are
driven element-by-element via `dut.<name>[k].value` (v3.3).

### `cocotb_fi.py` — fi pack / unpack

A small fixed helper module translates between the Python
reference's real-valued representation and the SV DUT's packed bit
vectors:

```python
pack_fi(value, signed: bool, wl: int, fl: int) -> int
unpack_fi(bits, signed: bool, wl: int, fl: int) -> float
fi_range(signed: bool, wl: int, fl: int) -> (lo, hi)
```

- **Saturating, not wrapping.** `pack_fi(300, signed=False, wl=8,
  fl=0)` returns 255; `pack_fi(-5, signed=False, wl=8, fl=0)`
  returns 0. Matches `matlab_runtime`'s saturating int casts on
  every other backend.
- **Two's-complement on signed pack.** `pack_fi(-1, signed=True,
  wl=8, fl=0)` returns `0xFF` (`255`). The DUT decodes via
  `unpack_fi(0xFF, signed=True, wl=8, fl=0) → -1.0`.
- **Resolvability check on unpack.** A signal carrying X / Z bits
  raises `ValueError` so the testbench fails loudly instead of
  silently masking a bad sample.

`fi_range` returns the legal real-value range a port can carry, so
the random-vector generator stays inside the legal grid (otherwise
`pack_fi` would saturate and we'd lose coverage on the upper end).

### Combinational vs sequential

A port-list `clk` / `rst_n` flips the harness into sequential mode:

| Mode | Clock | Reset | Sample timing |
|------|-------|-------|---------------|
| Combinational | none | none | drive inputs, settle 1 ns, sample |
| Sequential | 10 ns period (`Clock(dut.clk, 10, "ns")`) | hold `rst_n=0` for 2 cycles, then deassert | drive inputs, `await RisingEdge(clk)`, settle 1 ns, sample |

The post-edge 1 ns settle is intentional: without it,
`getattr(dut, name).value` reads the **pre-edge** register state
because cocotb's `await RisingEdge` returns at the moment of the
edge, before the simulator has propagated the non-blocking
assignments. Without the settle, every cycle is off by one.

### Pipeline latency (`-cocotb-latency=L`)

Mirrors MathWorks HDL Verifier's `Latency` parameter. Contract:

- Drive `N` stimulus vectors.
- The reference is called at cycle `k` against `input[k]`.
- The DUT's response to that input surfaces at cycle `k+L`.
- Recorded refs sit in a FIFO until their corresponding DUT sample
  comes due.
- The first `L` cycles are pipeline warm-up (no comparison); cycles
  `L..N-1` each yield one comparison, for `N-L` valid checks total.

There is **no tail-flush phase**: driving zeros after the last user
input would corrupt late-cycle DUT samples on stateful pipelines
(FIR with feedback through `delay_line`, etc.) and produce false
mismatches.

#### When to use `L > 0`

Set `L` to the pipeline depth between the inputs and the registered
output you want to verify. For
[`fir_asic_pipelined.m`](../examples/hdl/fir_asic_pipelined.m):

```matlab
% delay_line ─► reg_products ─► reg_acc ─► reg_output
%   (1 stage)    (1 stage)        (combo)    (output)
```

`ovfl` reads `full_res = reg_acc * gain` combinationally, so the
effective pipeline depth from `x` to `ovfl` is `2` (one cycle for
`reg_products`, one for `reg_acc`). Running with `-cocotb-latency=2`
takes that fixture from `2/200` mismatches in v1 to a clean pass:

```sh
$ matlabc -emit-cocotb -cocotb-latency=2 examples/hdl/fir_asic_pipelined.m
$ cd examples/hdl/fir_asic_pipelined_cocotb && make
** TESTS=1 PASS=1 FAIL=0 **
```

For the simpler examples (`alu_16bit`, FSMs, counter, mux), `L=0`
is correct because their outputs are combinational of the
post-edge state, so the post-edge settle in the harness already
gives the right alignment.

---

## Status

CI sweep across the 39 synthesizable HDL examples (cocotb 2.0.1,
Verilator 5.x). 28 modules pass cleanly at `L=0` (or the noted
pipeline depth) against random vectors and are exercised by the
`cocotb-tests` lane. Per-fixture latency lives in source as
`% cocotb: latency(N)` — the runner just walks fixture names.

| Tier-1 (8) | Tier-2 (14) | Tier-3 (6) |
|---|---|---|
| `alu_16bit` (L=0)   | `computed_state_fsm` (L=0) | `axi_handshake` (L=0) |
| `counter_0_to_10` (L=0) | `hamming74` (L=0) | `booth_mul` (L=0) |
| `fir_asic_pipelined` (L=4) | `i2c_bit_bang` (L=0) | `edge_detector` (L=0) |
| `mealy_fsm` (L=0)   | `leading_zero_detector` (L=0) | `fifo` (L=0) |
| `moore_fsm` (L=0)   | `median3` (L=0) | `manchester_enc` (L=0) |
| `mux_4to_1_16bit` (L=0) | `mmap_periph` (L=0) | `sync_2ff` (L=1) |
| `vector_processor` (L=0) | `popcount` (L=0) | |
| `sequential_processor` (L=4) | `priority_encoder` (L=0) | |
|                     | `pwm` (L=0) | |
|                     | `regfile` (L=0) | |
|                     | `rr_arbiter` (L=0) | |
|                     | `spi_master` (L=0) | |
|                     | `uart_rx` (L=0) | |
|                     | `up_down_counter` (L=0) | |

Tier-3 modules cleared after the Python emitter learned to handle
`matlab.not` (the bool-NOT op the frontend emits for `~rst`-style
expressions). The handler is the same shape as the existing
`matlab.bxor` / `matlab.band` lowerings — `(not <operand>)` for
i1 results, with the operand's truthiness coercion handled by
Python's standard semantics.

Mode selection is automatic per-input from the source pragmas:

- No `% cocotb:` stimulus pragma → random (with `% cocotb: hold`
  honored) or replay (when a sibling `test_*.m` exists).
- `% cocotb: stimulus(<name>, ...)` for any input → that input
  uses the deterministic shape; other inputs keep their default
  mode.

The hardest case (`sequential_processor`, a 4-stage persistent
FIR cascade) required v3.2.x's deterministic stimulus pragmas to
align the DUT's per-cycle pipeline propagation with the per-call
Python reference: an impulse on `x` plus a constant `gain` makes
the DUT and the reference walk through the impulse response in
lockstep — `ref(impulse).y` at call k matches `DUT.y` at cycle
`k+L` for L = pipeline depth.

### Python-emit gaps blocking cocotb expansion

The remaining 12 modules in `examples/hdl/` aren't in the lane
because the **Python emitter** (the source of truth for the
reference model) has gaps that surface only on HDL-style code.
These aren't cocotb-harness defects — `-emit-systemverilog`
ships clean RTL for every one — but the harness compares against
a Python reference, so a broken reference blocks verification.

| Class | Modules | Root cause |
|---|---|---|
| ~~`matlab.not` not handled~~ | ~~6~~ → 0 | ✅ Fixed; see Tier-3 above. |
| ~~Harness-side CDC timing~~ | ~~1~~ → 0 | ✅ False alarm — `sync_2ff` just needed `% cocotb: latency(1)`. The MATLAB-source's blocking-assignment semantics (`stage1=async_in; stage2=stage1` reads stage1 post-write) collapses the visible delay from 2 cycles to 1 against an SV DUT that uses non-blocking assignment. The Persist-count hint already pointed at L=1 (`PersistCount-1`); the fix was wiring it via the new `% cocotb: latency(N)` source pragma. |
| `matlab.alloc` not handled | `crc8`, `crc32`, `cordic_step` | Specific slot patterns produced by Stage F / RefineSlotTypes survive into the Python lowering with `matlab.alloc` ops the emitter doesn't recognise. |
| `matlab.call_builtin` unhandled | `cic_decimator` | A builtin (likely `bitshift` in a specific operand shape) isn't in the Python emitter's dispatch table. |
| Float-vs-int bitwise ops | `async_fifo`, `cordic_pipe`, `fnv1a`, `galois_lfsr` | The Python ref emits `wp ^ rp` / `s2y >> 1` / `h ^ b32` / `state & 1` where the LHS is a Python `float` (the `+ 0` snapshot of an f64 ABI load). Python (unlike MATLAB / C / SV) doesn't auto-coerce — it raises `TypeError: unsupported operand type(s) for >>: 'float' and 'int'`. Fix: wrap fi-typed values in `int(...)` before bitwise ops. |
| SV vs Python fi-saturation divergence | `aes_round`, `barrel_shifter`, `multi_cycle_mul` | The SV DUT correctly truncates / saturates `uint{8,16,32}` results on overflow. The Python ref computes the unbounded mathematical result. E.g. `bitshift(uint16(41905), 6)` → SV: 60480 (saturated / truncated), Python: 2681920 (unbounded). |

The float-vs-int cluster (4 modules) is the next-cheapest unblock
— wrap operands of `arith.shrui` / `arith.shli` / `arith.andi` /
`arith.ori` / `arith.xori` in `int(...)` whenever the operand
type traces back to an f64 ABI load. The saturation cluster (3
modules) needs a per-fi-spec wrap-and-clamp helper inserted at
each backend-narrow op (trunci / shrui / shli that crosses a
declared width).

Mode selection is automatic per-input from the source pragmas:

- No `% cocotb:` stimulus pragma → random (with `% cocotb: hold`
  honored) or replay (when a sibling `test_*.m` exists).
- `% cocotb: stimulus(<name>, ...)` for any input → that input
  uses the deterministic shape; other inputs keep their default
  mode.

---

## Roadmap

Each item below is **concrete, sized, and not yet attempted**.
Effort is calendar time at one focused implementation session per
stage.

### v3.1 — Input-stability semantics ✅ shipped

**Status.** Implemented. New `% cocotb: hold(<input>, <cycles>)`
pragma — drop it inside the function body next to the `% hdl:
port(...)` lines and the harness will pin the named input to the
same random value for `<cycles>` consecutive iterations before
drawing a fresh sample. Mismatched names emit a clear warning and
are ignored (stale pragma after a rename).

```matlab
function y = filter(x, gain)
    %#codegen
    % hdl: port(x, fi, signed, 16, 14)
    % hdl: port(gain, fi, signed, 16, 12)
    % cocotb: hold(gain, 4)        # gain stays stable for 4 cycles
    ...
```

The pragma covers the simple "input X must be stable across L
cycles for the SV pipeline to converge" case. It does **not** by
itself solve the multi-stage cascade reference divergence (that
needs impulse-style stimulus shapes — v3.2.x). For that reason
`sequential_processor` is still deferred; the pragma is shipped
mechanically and ready for the cases where input stability alone
is enough.

### v3.2 — `test_<stem>.m`-derived stimulus ✅ shipped

**Status.** Implemented. When `-emit-cocotb FILE.m` finds a sibling
`test_*.m` whose script body calls the DUT function, the harness
extracts the stimulus sequence via AST walk and embeds it as a
deterministic vector list. Recognised tester shapes:

1. **Single device call, no loop** — `result = device(a, b, ...)`.
2. **Vector-driven loop** — `vec = [...]; for i = 1:length(vec):
   device(vec(i), other)`.
3. **Fixed-count loop with conditionals** — `for i = 1:N: if i ==
   K, x = a; else x = b; end; device(x)`.

Tester discovery is name-flexible: tries the strict
`test_<stem>.m` first, then any `test_*.m` in the same directory
whose body calls the DUT function. The existing `examples/hdl/`
naming (`test_mealy.m`, `test_fsm_moore.m`, `test_mux.m`,
`test_counter.m`) all match cleanly. Falls back to random vectors
with a diagnostic when the tester uses a shape outside the
recognised set.

**Side effect.** v3.2 also exposed and fixed a bug in
`-emit-python` where `rt.persistent_isempty(id)` returned True
forever for scalar persistents (function-attribute storage was
invisible to the runtime's `_persistent_ptr` table). The fix
emits a `rt.persistent_set_ptr(<id>, True)` marker alongside
each scalar's module-level init, so the runtime sees the slot
as set after import. Caught by the FSM testers under v3.2 —
random vectors had been masking it because most uint8 inputs
aren't `==1` so the FSM stayed in the reset arm regardless.

### v3.x — Mealy combinational output sampling ✅ shipped

**Status.** Implemented via dual-edge sampling. For sequential
DUTs, the harness now samples every output *both* before the
rising edge (combinational propagation of new inputs through the
still-valid old state — Mealy-style alignment) and after (the
just-latched register value — Moore / counter / FIR-style
alignment). For each output the comparison accepts whichever
sample matches the reference. No per-port metadata needed; the
harness picks the correct alignment automatically.

The pre-edge sample is captured after a 1 ns Timer settle (lets
combinational signals propagate the new inputs against the still-
valid old state); then `await RisingEdge` advances the state;
then a second 1 ns settle yields the post-edge sample.

### v3.3 — Vector / unpacked-array port driving ✅ shipped

**Status.** Implemented. The SV port-list parser now captures the
`[N]` array suffix (was previously a hard short-circuit). The
generated harness drives unpacked-array inputs element-by-element
(`dut.<name>[k].value = pack_fi(...)`) and reads outputs the same
way. Helper functions `_drive` / `_read` / `_eq` / `_gen_random`
emitted at the top of `test_<stem>.py` keep the test body uniform
across scalar and vector ports — most users will never need to
touch them.

### v3.2.x — Stimulus-shape extensions ✅ shipped

**Status.** Implemented. New pragmas:

```matlab
% cocotb: stimulus(<input>, impulse, <value>)        # value@0, zeros after
% cocotb: stimulus(<input>, constant, <value>)       # same value every cycle
% cocotb: stimulus(<input>, ramp, <start>, <stride>) # start, start+stride, ...
```

Each input is independently controlled — combine impulse on the
data input with constant on a coefficient port to match
`ref(impulse)` evaluated by the per-call reference against
`DUT.y[k+L]` once the pipeline has settled.

`sequential_processor.m` now ships with:

```matlab
% cocotb: stimulus(x, impulse, 1.0)
% cocotb: stimulus(gain, constant, 0.25)
% cocotb: stimulus(reset, constant, 0)
```

…driving its 4-stage persistent pipeline through a clean
impulse response with `-cocotb-latency=4`. The `just
verify-cocotb examples/hdl/sequential_processor.m 4` flow exercises
exactly that.

The pragma also covers two adjacent use cases that fall out of
the same machinery:
- **Constant** lets the user pin a coefficient / mode-select port
  while exercising the rest of the design. Useful for verifying
  one operating mode of a multi-mode block at a time.
- **Ramp** (`start, stride`) walks an input linearly through its
  range. For ALU-style blocks this exercises every operand value
  the port can hold, complementing random testing.

### v3.4 / B4 — Precise pipeline-latency hint ✅ shipped

**Status.** Implemented. When neither the CLI flag nor a
`% cocotb: latency(N)` pragma supplied a value, the matlabc emit
message prints a per-fixture suggestion based on two complementary
signals:

1. **Scalar-persistent chain.** N independent `persistent <var>;`
   declarations that feed each other in source order produce a
   visible delay of N-1 cycles (MATLAB blocking semantics: the
   body's `stage1 = in; stage2 = stage1` reads stage1's same-cycle
   written value, so the SV's two-flop chain shows up as a
   one-cycle delay against the Python ref). Source: `sync_2ff` →
   hint L=1.
2. **fi-array shift register.** `fi(zeros(1, N), ...)` declares an
   N-element shift register; Stage F splits it into N parallel
   scalar persistents and the natural pipeline depth from input to
   the last tap is N. Source: `sequential_processor` (`zeros(1,
   4)` → hint L=4) and `fir_asic_pipelined` (multi-stage with N=4
   → hint L=4).

The hint reports the larger of the two estimates, so designs that
mix shapes (FIR-style pipelined fixtures with a few extra scalar
state registers alongside the fi-array shift register) get the
right value too. Counter / FSM modules with a single self-feeding
persistent get no hint — L=0 is correct despite having state.

### v3.4.x — `% cocotb: latency(N)` source pragma ✅ shipped

**Status.** Implemented. The same value the `-cocotb-latency` CLI
flag carries can now live in the source next to the `% hdl:
port(...)` lines:

```matlab
function [y, ovfl] = fir_asic_pipelined(x, gain, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 14)
    % hdl: port(gain, fi, signed, 16, 12)
    % hdl: port(reset, bool)
    % cocotb: stimulus(gain, constant, 0.25)
    % cocotb: stimulus(reset, constant, 0)
    % cocotb: latency(4)
    ...
```

Resolution rules:
- If the user passes `-cocotb-latency=N` (any N, including 0), the
  CLI flag wins — explicit override semantics.
- Else, if the source has `% cocotb: latency(N)`, that's used.
- Else, default 0.

This lets the CI lane drop hardcoded `(module, latency)` tuples and
just sweep every fixture by name. The matching fixtures
(`fir_asic_pipelined`, `sequential_processor`) carry the pragma
in-source.

### v3.5 — CI lane (`cocotb-tests`) ✅ shipped

**Status.** Implemented. `ctest --test-dir build -R cocotb-tests`
runs the sweep across every supported `examples/hdl/*.m` module
and asserts each reports `TESTS=1 PASS=1 FAIL=0`. The lane is
gated on `verilator` + `cocotb` being on `PATH`; missing either
yields ctest's `Skipped` status (return code 77), matching how
the emit-typescript lane handles a missing Node. Driver script:
`test/EmitCocoTB/run_tests.sh`.

### v3.6 — `-cocotb-seed=N` knob ✅ shipped

**Status.** Implemented. `-cocotb-seed=N` plumbs through to
`random.seed(N)` in the harness; default 42 keeps existing
goldens byte-stable.

### v3.7 — Coverage report ✅ shipped

**Status.** Implemented. The harness now writes `coverage.txt`
to its own directory at the end of every run — best-effort, a
write failure (read-only fs / permissions) doesn't fail the
test. Per-port stats: count, min, max, mean. Narrow ports
(WL ≤ 8 bits, e.g. mux selectors and FSM inputs) also include
a value histogram so the user can see which input states the
random / tester vectors actually exercised. Sample output:

```
## Inputs
  a : signed 16 bits, FL=0
    samples=100  min=-32134.0  max=32720.0  mean=2036.92
  sel : unsigned 8 bits, FL=0
    samples=100  min=1.0  max=255.0  mean=127.02
    histogram:
        1   1  #
        2   1  #
       12   2  ##
       ...
```

### v3.8 — Multi-clock testbenches 🔵 (deferred — see scope below)

**Status.** Deferred. The cocotb harness today assumes a single
`clk` / `rst_n` pair. Multi-clock support is meaningful only once
the upstream `-emit-systemverilog` backend produces multi-clock
designs and the MATLAB source language has a way to express them.
Both are missing today; this section scopes the cross-cutting
work needed end-to-end so the right pieces ship together when a
real user need surfaces.

#### What "multi-clock" actually means

A multi-clock DUT has more than one independent clock signal,
each driving its own register set (a "clock domain"). Typical
shapes:

- **CDC across distinct clocks.** Two functional units running on
  unrelated clocks (a slow sensor capture domain at 100 MHz, a
  fast DSP pipeline at 500 MHz). Cross-domain signals need
  synchronizers.
- **Multi-rate processing.** Decimation / interpolation filters
  where input rate != output rate, expressed as different clock
  periods on the same DUT.
- **Async FIFO bridges.** Producer and consumer on different
  clocks, FIFO with read / write pointers in their own domains
  and gray-coded across.

Each scenario has different verification needs. The cocotb
harness side is the smallest piece; the real lift is in the
language and SV backend.

#### Layer 1 — MATLAB source language

**Currently missing.** No way to express clock domains in MATLAB
source today.

**What's needed.**

1. **Per-function clock declaration.** A `% hdl: clock(<name>,
   <period_ns>)` pragma at the function level, repeatable. First
   declaration is the primary clock; subsequent ones are
   additional domains.

   ```matlab
   function y = foo(x_a, x_b)
       %#codegen
       % hdl: clock(clk_fast, 2)        # 500 MHz
       % hdl: clock(clk_slow, 10)       # 100 MHz
       % hdl: port(x_a, fi, signed, 16, 14)  % implicitly on clk_fast (default = primary)
       % hdl: port(x_b, fi, signed, 16, 14, clock=clk_slow)
       ...
   ```

2. **Per-persistent clock binding.** A `% hdl: clock_domain(<name>)`
   on each `persistent` so the SV emit knows which `always_ff`
   block latches it.

   ```matlab
   persistent fast_acc;
   % hdl: clock_domain(clk_fast)
   persistent slow_state;
   % hdl: clock_domain(clk_slow)
   ```

3. **CDC marking.** A `% hdl: cdc(<src_clk>, <dst_clk>)` annotation
   on cross-domain reads, driving the SV emit to insert a
   double-flop synchronizer (or similar) at that crossing.

**Effort.** ~2 sessions for the pragma surface + AST plumbing.
Bigger lift is the type-system implication: every value carries
an implicit clock-domain tag, and the inference rules need to
catch unmarked CDC crossings as a synthesizability error.

#### Layer 2 — SV backend (`-emit-systemverilog`)

**Currently missing.** Single hardcoded `clk` / `rst_n`. Stage F
(`LowerPersistentFiArrays`) and `HWStateInfer` produce one
`always_ff @(posedge clk or negedge rst_n)` block.

**What's needed.**

1. **Per-domain register grouping.** Stage F + HWStateInfer pass
   the clock-domain tag through and emit one `always_ff` per
   distinct clock. Module port list gains one input per declared
   clock (and optionally per-clock reset signals — async-low
   default per the SV style guide).

2. **Synchronizer cells.** For each `% hdl: cdc(...)` crossing,
   emit a 2-flop synchronizer in the destination domain:

   ```sv
   logic sync_meta, sync_q;
   always_ff @(posedge clk_dst or negedge rst_n_dst) begin
     if (!rst_n_dst) begin sync_meta <= 0; sync_q <= 0; end
     else            begin sync_meta <= signal_src; sync_q <= sync_meta; end
   end
   ```

3. **Lint cleanliness.** `verilator --lint-only` rules tighten in
   multi-clock designs (CLKDATA, MULTIDRIVEN, CMPCONST). The
   golden-diff lane needs to keep passing once these emit.

**Effort.** ~5 sessions. Touches every SV-emit pass that knows
about clocks. The current single-clock assumption is baked into
the reset arm logic, the FSM-encoding emit, and the
`HWStateInfer` matcher.

#### Layer 3 — Python reference (`-emit-python`)

**Mostly OK as-is.** The Python reference already runs as a
synchronous function call — cycle accuracy isn't a concept there.
Each call advances all persistents, regardless of which "clock"
they conceptually belong to.

**What's needed for tight verification.**

1. **Synchronizer modeling.** The reference needs to emit at
   least a 2-cycle delay on signals tagged as CDC crossings, so
   the harness comparison (DUT[k+L_dst] vs ref[k]) matches the
   SV's synchronizer latency. Implementable as a per-CDC FIFO in
   the emitted Python.

2. **Optional: per-clock-domain stepping.** Instead of one ref
   call per "compound cycle", the reference exposes
   `step_clk_fast()` / `step_clk_slow()` so the harness can
   simulate the actual interleaving. Heavier, only needed if
   single-stepping isn't bit-exact (it isn't, in general).

**Effort.** ~3 sessions for the synchronizer modeling. The
per-clock stepping option doubles that and is probably not worth
it before a real user case demands it.

#### Layer 4 — Cocotb harness (`-emit-cocotb`)

**Most of the work happens here, and most of it is already
sketched.**

1. **Clock detection in the SV port-list parser.** Today only
   `clk` / `rst_n` are recognised. Extend to any input named
   `clk*` (and `rst_n*` / `reset_n*`). Build a per-clock spec:
   `(name, period_ns, reset_name)`.

2. **Per-clock `Clock` instances.** One `cocotb.start_soon(
   Clock(dut.<name>, <period>, "ns").start())` per detected
   clock. Default period 10 ns; per-clock override via
   `% cocotb: clock(<name>, <period_ns>)`.

3. **Reset sequencing.** Hold each clock's reset low for 2 of its
   own clock cycles, then deassert. Multi-clock systems may need
   to coordinate reset deassertion across domains — typically
   the slowest clock's deassertion is the synchronization
   point.

4. **Per-output clock binding.** Each output port carries its
   "this signal is valid on posedge clk_<X>" tag. The harness
   uses `await RisingEdge(dut.clk_<X>)` for that output's sample
   window. Today the harness samples on the single `clk`'s
   posedge for everything.

5. **Per-clock latency.** The HDL Verifier-equivalent `Latency`
   parameter becomes a map: `{clk_a: L_a, clk_b: L_b}`. CLI
   form: `-cocotb-latency=clk_a:2,clk_b:1` (or repeated
   `-cocotb-latency=clk_a:2 -cocotb-latency=clk_b:1`).

6. **Frequency-ratio bookkeeping.** Cycles aren't 1:1 across
   clocks. The harness's outer loop drives stimulus at the
   rate of the slowest clock; faster clocks tick in between.
   Sampling and reference advancement happen at slow-clock
   posedges by default (override via pragma).

**Effort.** ~3 sessions. Mostly mechanical once Layer 2
provides multi-clock SV to parse against.

#### Layer 5 — Verification methodology

**The hardest piece, for the user.** Multi-clock equivalence
between an SV DUT and a single-call Python reference isn't
mechanical the way single-clock is. Even with synchronizer
modeling, true async clocks have **timing nondeterminism** —
the relative phase of the two clocks at simulation start
affects which cycle a CDC sample lands. Verification flow
options:

1. **Constrained equivalence.** Drive both clocks at known
   integer ratios (e.g., 2:1) with deterministic phase. The
   reference walks at the slow rate; the harness samples
   outputs at the slow-clock posedge. Easy to verify; doesn't
   exercise the actual async behavior.

2. **Bounded-skew sweep.** Run multiple phases of the fast
   clock relative to the slow clock; assert all produce
   semantically-equivalent output (allowing for +/- 1 cycle
   slip on CDC paths). Closer to a real CDC test; expensive
   and not bit-exact.

3. **Formal CDC checking.** Out of scope for this harness —
   would need a formal tool (JasperGold's CDC App, etc.).

For the v3.8 "shipped" definition, **option 1 is the target.**
Options 2 and 3 are explicitly "use a real verification
toolchain".

#### Trigger to ship

v3.8 is reasonable to schedule when **any one** of the following
becomes a real user need:

- A user files a request with a multi-clock MATLAB program they
  want to verify.
- The roadmap picks up CDC-aware DSP examples (multi-rate FIR,
  decimation, etc.).
- The SV backend lands its own multi-clock support for an
  unrelated reason (asic-flow improvements, formal lint
  expansion).

Until then, all the pieces above stay shipped-but-unwritten in
their respective backends. The cost of building multi-clock
without a real consumer is high — every layer's design choices
get locked in by whatever first synthetic example we pick, and
real users will want different shapes.

**Total effort if all five layers ship together.** ~13 focused
sessions (~3 weeks). The cocotb-harness slice (Layer 4) is a
~3-session subset and could ship first as a forward-looking
stub, but without Layers 1–2 it has nothing to test against.

### Out of scope (intentionally)

- **UVM-style structured verification.** v3 is still
  random-vector + golden-reference, not constrained-random with
  scoreboards / coverage groups. Power users can build that on top
  by importing the generated testbench as a starting point.
- **Formal proof.** Bit-exact equality across N random vectors is
  not a proof of equivalence; it's a strong empirical signal. For
  formal proof, users should bring their own toolchain (SymbiYosys
  + Verilator-front-end, JasperGold, etc.).
- **Timing closure.** The harness verifies functional equivalence,
  not synthesis QoR. Users still need their synthesis flow for
  area / timing / power.

---

## Implementation pointers

If you want to read the code:

- **CLI plumbing** — `tools/matlabc/main.cpp`, `Options::Mode::EmitCocotb`
  branch + the `-cocotb-*` flag parsers near the top of `parseArgs`.
- **Pipeline driver** — `emitCocotbHarness()` in the same file. Drives
  the SV + Python sub-emits via self-spawn (`std::system`), parses
  the SV port list, walks the module for the function-name match,
  renders the harness, and stages the output directory.
- **Port-list parser** — `parseCocotbSpecFromSv()`. Pure text parser
  on the SV file just emitted; ~100 lines including v3.3's
  unpacked-array suffix capture.
- **Pragma scanner** — `scanCocotbPragmas()`. Walks the source for
  `% cocotb:` lines and returns the (name → `hold` cycles, name →
  `stimulus` spec) maps. Extending the pragma surface (multi-clock
  v3.8, future v3.9 items) starts here.
- **Harness template** — `renderCocotbHarness()`. Plain
  `std::string` concatenation; no template engine. Emits per-port
  helper functions (`_drive` / `_read` / `_eq` / `_gen_random` /
  `_stim_value`) plus the `_Coverage` class at the top so the
  test body stays compact regardless of port shape.
- **Tester-stimulus extractor** — `TesterStimulus::extract()`. AST
  walks a sibling `test_<stem>.m`'s script body, evaluates the
  recognised loop / single-call shapes, returns a flat list of
  per-cycle input tuples. Returns `nullopt` when the tester uses
  a shape outside the recognised set; caller falls back to random.
- **fi helpers** — `runtime/cocotb_fi.py`, mirrored as the embedded
  string `kCocotbFiHelperPy` in `main.cpp`. The mirror is regenerated
  by hand when the canonical file changes; the embed exists so the
  harness directory is portable across machines.
- **CI lane** — `test/EmitCocoTB/run_tests.sh` + `add_test(NAME
  cocotb-tests ...)` in `CMakeLists.txt`. Skip-if-missing on
  verilator / cocotb (returns 77, ctest's `Skipped` code).

---

## See Also

- [`emit_systemverilog.md`](emit_systemverilog.md) — the SV backend
  that produces the DUT.
- [`emit_python.md`](emit_python.md) — the Python emitter that
  produces the reference model.
- [`emit_fixed_point.md`](emit_fixed_point.md) — fi semantics shared
  by both emit paths and the harness's `pack_fi` / `unpack_fi`.
- [`feature_status.md`](feature_status.md) — authoritative status
  matrix; this doc is the design-side companion.
