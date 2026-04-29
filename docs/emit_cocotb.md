# CocoTB Verification Harness Emission (`-emit-cocotb`)

This document scopes the `-emit-cocotb` mode of `matlabc`: an
**open-source alternative to MathWorks's HDL Verifier** that closes
the verification loop for the SystemVerilog backend. From a single
`.m` source file, `matlabc` generates a self-contained
[CocoTB](https://www.cocotb.org/) testbench directory that drives
the emitted SystemVerilog DUT and the emitted Python reference
model in lockstep against random vectors, asserting cycle-by-cycle
equality.

**Status: v2 shipped.** 6 of 7 supported `examples/hdl/` modules
pass bit-exact under Verilator + CocoTB; the 7th (`vector_processor`)
is intentionally deferred pending vector-port driving (v3). See the
[Status](#status) and [Roadmap](#roadmap) sections for the running
list of remaining work.

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
    [-cocotb-latency=L]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-emit-cocotb` | — | Selects the harness-emit mode. |
| `-cocotb-out=DIR` | `<input-dir>/<stem>_cocotb` | Output directory. Created if missing. |
| `-cocotb-vectors=N` | 100 | Number of random stimulus vectors driven. |
| `-cocotb-latency=L` | 0 | Pipeline latency. DUT output at cycle `k+L` is compared against the reference's response to cycle `k`'s inputs. See [Pipeline latency](#pipeline-latency-cocotb-latencyl). |

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
array ports (`logic ... [N]`) currently short-circuit harness
generation with a clear diagnostic.

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

Sweep across `examples/hdl/*.m` (cocotb 2.0.1, Verilator 5.x):

| Module                | L | Result | Notes                        |
| --------------------- | - | ------ | ---------------------------- |
| `alu_16bit`           | 0 | PASS 100/100 | combinational                |
| `counter_0_to_10`     | 0 | PASS 100/100 | sequential, persistent       |
| `mealy_fsm`           | 0 | PASS 100/100 | sequential FSM               |
| `moore_fsm`           | 0 | PASS 100/100 | sequential FSM               |
| `mux_4to_1_16bit`     | 0 | PASS 100/100 | combinational                |
| `fir_asic_pipelined`  | 2 | PASS    | requires `-cocotb-latency=2` |
| `sequential_processor`| any | mismatch | multi-input pipeline (see roadmap) |
| `vector_processor`    | — | SKIP w/ warning | unpacked-array ports (see roadmap) |

`sequential_processor` is the one supported-shape module where v2's
random-vector strategy doesn't suffice. Its pipeline samples `gain`
at cycle `k+L` while the Python reference at cycle `k` consumes
`gain[k]`; with random per-cycle gain values the two paths
legitimately diverge. The fix is in v3 (input-stability — see
below), not in the harness internals.

---

## Roadmap

Each item below is **concrete, sized, and not yet attempted**.
Effort is calendar time at one focused implementation session per
stage.

### v3.1 — Input-stability semantics 🔵

**Problem.** Multi-stage pipelined DUTs sample different inputs at
different cycles. For `sequential_processor`, the SV samples `gain`
at cycle `k+L` (after the pipeline propagates) while the Python
reference at cycle `k` uses `gain[k]`. With random per-cycle gain,
they don't match — even at the correct `L`.

**Plan.** A new `% cocotb: hold(<input>, <cycles>)` pragma (or a
top-level `-cocotb-hold-inputs=N` flag) tells the harness to hold a
named input stable for `<cycles>` consecutive cycles before
advancing. The reference-vs-DUT comparison still aligns at `k+L`,
but every sample within the hold window agrees.

**Effort.** ~1 session. The SV port-list parser already knows port
names; the harness emitter just needs to track per-input hold
counters and skip the random-uniform call until the hold expires.

### v3.2 — `test_<stem>.m`-derived stimulus 🔵

**Problem.** Random vectors don't exercise corner cases (state
transitions, saturation boundaries, reset sequences) reliably. The
existing `examples/hdl/test_*.m` driver scripts already encode
hand-picked stimulus sequences (e.g.,
`bits = [1 0 1 1 0 0 1]; for i = 1:length(bits): mealy_fsm(bits(i))`).

**Plan.** When `-emit-cocotb` finds a sibling `test_<stem>.m`
alongside `<stem>.m`, replace the random-vector loop with the
sequence the test driver feeds the function. The reference Python
model already runs that sequence (it's just the
`-emit-python` of `test_<stem>.m`); the harness drives the same
sequence into the DUT.

**Effort.** ~2 sessions. Needs a small AST / IR walk to extract the
stimulus loop's iteration count and per-cycle input expressions
from the driver script.

### v3.3 — Vector / unpacked-array port driving 🔵

**Problem.** `vector_processor` declares unpacked-array ports
(`input logic signed [15:0] vec_a [3]`). The current harness bails
with a warning because cocotb's `dut.vec_a.value = single_int`
assignment doesn't work for arrays.

**Plan.** Detect the `[N]` array suffix in the SV port-list parser
(already done — used to bail). Generate per-element drive (`for k
in range(N): dut.vec_a[k].value = pack_fi(...)`) and per-element
sample on the read side. The reference Python model already takes
list-shaped arguments for vector inputs.

**Effort.** ~1 session. Mostly a code-generation change in
`renderCocotbHarness`.

### v3.4 — Auto-detect pipeline latency 🔵

**Problem.** Today the user has to know `L` per fixture (`L=2` for
`fir_asic_pipelined`, `L=0` for everything else that passes).
Mistakes silently produce mismatches.

**Plan.** During the in-process module walk used to render the
harness, count the longest persistent-update chain reaching each
output — that's a tight upper bound on pipeline latency. Set the
default `L` from that count when `-cocotb-latency` isn't passed
explicitly. A diagnostic line in the matlabc emit message reports
the inferred value so the user can override if it's wrong.

**Effort.** ~1 session. Reuses the same pass that decides whether
the DUT is sequential.

### v3.5 — CI lane (`cocotb-tests`) 🔵

**Problem.** No automatic regression check exists today. The user
runs the sweep manually after changes.

**Plan.** New ctest entry `cocotb-tests` that:

1. Probes for `cocotb-config` and `verilator` on `PATH`.
2. If either is missing, **skips** rather than fails (matches the
   policy used for `emit-typescript` skips on platforms missing
   Node).
3. Otherwise runs `-emit-cocotb` + `make` for every supported
   `examples/hdl/*.m` and asserts each prints `PASS=1 FAIL=0`.

**Effort.** ~1 session. Driver script in `test/EmitCocoTB/` mirroring
the existing `test/EmitSV/` lane.

### v3.6 — `-cocotb-seed=N` knob 🔵

**Problem.** The harness hard-codes `random.seed(42)`. Useful for
reproducibility, but doesn't let the user explore different
randomization schedules without editing the generated file.

**Plan.** `-cocotb-seed=N` flag plumbed through to `random.seed(N)`
in the harness. Default still 42 for byte-stable test output.

**Effort.** ~15 minutes. Trivial.

### v3.7 — Coverage report 🔵

**Problem.** Pass / fail tells the user equality holds for the
random vectors driven; it doesn't tell them which input ranges,
state transitions, or output values were actually exercised.

**Plan.** A second testbench helper that records per-cycle
(input, output) tuples and prints a histogram-style summary at the
end: input-range coverage per port, output-range coverage per port,
and (for FSMs) state-transition coverage. Saves to a `coverage.txt`
alongside the test results.

**Effort.** ~2 sessions. Pure Python; no C++ changes.

### v3.8 — Multi-clock testbenches 🔵

**Problem.** Today the harness assumes a single `clk` input. The SV
backend doesn't currently emit multi-clock designs, but if it ever
does (CDC support, async-FIFO style modules), the harness needs to
match.

**Plan.** Detect multiple `input logic clk_*` ports in the parsed
SV port list and emit one `cocotb.start_soon(Clock(...))` per
clock with configurable per-clock periods.

**Effort.** ~3 sessions. Needs SV emitter coordination on naming
conventions.

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
  on the SV file just emitted; ~80 lines.
- **Harness template** — `renderCocotbHarness()`. Plain
  `std::string` concatenation; no template engine. Combinational vs
  sequential branching is inline.
- **fi helpers** — `runtime/cocotb_fi.py`, mirrored as the embedded
  string `kCocotbFiHelperPy` in `main.cpp`. The mirror is regenerated
  by hand when the canonical file changes; the embed exists so the
  harness directory is portable across machines.

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
