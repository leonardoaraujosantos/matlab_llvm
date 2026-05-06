# Tutorial: MATLAB → SystemVerilog → cocotb verification

This walks through the full HDL flow end-to-end with one running
example. By the end you'll have written a small DUT in MATLAB,
emitted synthesizable SystemVerilog, and verified it bit-exact
against a Python reference under cocotb — without touching SV by
hand.

If you just want to run the working examples, jump to
[Just-run-everything](#just-run-everything) at the bottom.

---

## What you need

- A built `matlabc` (one of: `cmake --build build` from a clone, or
  the binary from a release).
- `verilator` ≥ 5.0 — `brew install verilator` on macOS, your
  distro's package manager elsewhere.
- `cocotb` ≥ 2.0 — `pip install cocotb`.

The cocotb lane gates on both being installed; without them, the
SV emit still works but you can't run lockstep verification.

---

## Step 1 — Write a tiny DUT

Save this as `mux2.m`:

```matlab
function y = mux2(a, b, sel)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(sel, bool)
    if sel
        y = a;
    else
        y = b;
    end
end
```

Two things to notice:

- The `%#codegen` directive marks this as a code-gen target.
- The `% hdl: port(...)` comments declare the port shapes — `fi,
  signed, 16, 0` is a 16-bit signed integer (`fi` = MATLAB's
  Fixed-Point Designer numeric type), `bool` is a 1-bit signal.
  Without these pragmas, the SV emitter has no way to know the
  function's argument types.

The output `y`'s type is **inferred** from the assignment. For
clean control over the output port's signedness, add an output
pragma too — see [Output port pragmas](#output-port-pragmas)
below.

---

## Step 2 — Emit SystemVerilog

```sh
matlabc -emit-systemverilog mux2.m > mux2.sv
```

The result:

```sv
module mux2 (
    input  logic signed [15:0] a,
    input  logic signed [15:0] b,
    input  logic sel,
    output logic signed [15:0] y
);
    always_comb begin
        y = sel ? a : b;
    end
endmodule
```

(The simple if/else collapses to a ternary at emit time — the
`IfStoreToSelect` pass folds single-store if/else into
`arith.select` before SV emission.)

Two checks before going further:

```sh
# Lint with Verilator. -Wall + the suppress flags catch all the
# real bugs while ignoring style/cosmetic noise.
verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL \
    --top-module mux2 mux2.sv

# (Optional) sanity-check with Yosys generic synth.
yosys -p "read_verilog -sv mux2.sv; synth -top mux2"
```

Both should pass clean for any synthesizable design `matlabc`
emits — the SV golden lane runs both on every fixture in
`test/EmitSV/`.

---

## Step 3 — Verify against the Python reference

This is where cocotb takes over. One command:

```sh
matlabc -emit-cocotb mux2.m
```

Output:

```
matlabc: wrote CocoTB harness to mux2_cocotb (3 inputs, 1 outputs, combinational, 100 random vectors)
```

The harness directory is **self-contained**. Drop in and `make`:

```sh
cd mux2_cocotb && make
```

After ~5 seconds:

```
** TESTS=1 PASS=1 FAIL=0 SKIP=0 **
```

What just happened:

1. `matlabc -emit-cocotb` ran `-emit-systemverilog` to produce
   `mux2.sv` and `-emit-python` to produce `mux2_ref.py` (a
   reference model in Python with the same fi semantics as the
   MATLAB source).
2. It generated `test_mux2.py` — a cocotb testbench that drives
   100 random `(a, b, sel)` triples into both the SV simulation
   (via Verilator) and the Python reference, comparing outputs
   cycle-by-cycle.
3. `make` ran Verilator + cocotb + the harness; both DUT and
   reference returned identical values for every vector.

The output directory layout:

```
mux2_cocotb/
    mux2.sv             # the SV DUT
    mux2_ref.py         # the Python reference
    test_mux2.py        # cocotb testbench (drives + compares)
    cocotb_fi.py        # fi pack / unpack helpers
    matlab_runtime.py   # Python fi runtime (copied from source tree)
    Makefile            # Verilator + cocotb invocation
    coverage.txt        # per-port stats (after running `make`)
    dump.vcd            # waveform (after running `make`)
```

`coverage.txt` after the run summarises per-port min/max/mean +
histograms for narrow ports. `dump.vcd` is the Verilator trace —
open in GTKWave or Surfer to inspect cycle-by-cycle behaviour.

---

## Adding state (sequential designs)

Save this as `counter.m`:

```matlab
function n = counter(en, reset)
    %#codegen
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 8-bit counter that increments while `en` is high.
    persistent count;
    if isempty(count) || reset
        count = uint8(0);
    end
    if en
        count = count + uint8(1);
    end
    n = count;
end
```

Three new ingredients:

1. **`persistent` declaration** — backs a `count_reg` flop in SV.
   Every persistent variable becomes a register.
2. **`if isempty(count) || reset`** — the canonical reset-init
   pattern. The init value (`uint8(0)`) becomes the SV register's
   reset value.
3. **`% cocotb: stimulus(reset, constant, 0)`** — pin `reset` low
   throughout the random-vector run. Without this, random `reset`
   values would fire often enough that the test sees more reset
   cycles than counts.

Emitting:

```sh
matlabc -emit-cocotb counter.m
cd counter_cocotb && make
```

The SV gains a `clk`/`rst_n` port and an `always_ff` block:

```sv
module counter (
    input  logic clk,
    input  logic rst_n,
    input  logic en,
    input  logic reset,
    output logic [7:0] n
);
    logic [7:0] count;
    logic [7:0] count_next;

    always_comb begin
        count_next = count;
        if (reset) count_next = 8'sd0;
        if (en) count_next = count + 8'sd1;
        n = count;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) count <= 8'sd0;
        else        count <= count_next;
    end
endmodule
```

The cocotb harness automatically detects the sequential shape and
adds clock generation + reset sequencing. No knobs needed for the
common case.

---

## Pipelined designs (`% cocotb: latency(N)`)

When the SV has a register chain between an input and an output,
the cocotb harness needs to know the depth so it can compare
`DUT.y[k+L]` against `ref(input[k])`. Declare it in source:

```matlab
function y = pipe2(x, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    % cocotb: latency(2)              <-- two flops between x and y

    persistent stage1;
    persistent stage2;
    if isempty(stage1) || reset; stage1 = int16(0); end
    if isempty(stage2) || reset; stage2 = int16(0); end

    s1 = stage1 + int16(0);   % snapshot pattern; see Common gotchas
    s2 = stage2 + int16(0);

    stage1 = x;
    stage2 = s1;
    y = s2;
end
```

`matlabc -emit-cocotb pipe2.m` will print a hint if no latency is
declared:

```
hint: 2 `persistent` decls — pipelined; if outputs are registered, add `% cocotb: latency(1)` near the `% hdl: port(...)` lines, or pass `-cocotb-latency=1` on the CLI.
```

(For shift-register-style fi-arrays, the hint reads the array
length: `hint: 4-tap fi-array shift register — pipelined; ...`.)

Pass the value via either the source pragma (preferred) or the
CLI flag. The CLI wins when both are present.

---

## Output port pragmas

Source pragmas can also declare output port types, not just
inputs:

```matlab
function crc = crc32(data_in, en, reset)
    %#codegen
    % hdl: port(data_in, bool)
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % hdl: port(crc, fi, unsigned, 32, 0)   <-- output pragma
    ...
```

Without the output pragma, the SV emitter renders multi-bit
outputs as `signed` by default. With it, `crc` renders as
`output logic [31:0] crc` (unsigned). Important when the source
intent is clearly unsigned (CRC accumulators, hashes, byte
streams) — the cocotb harness's DUT-side decode aligns with the
Python reference's value rather than relying on the modulo-2^WL
fallback in `_eq`.

---

## Common gotchas

### Persistent reads need to be "anchored"

This snippet looks fine but breaks SV emission:

```matlab
persistent crc_reg;
if isempty(crc_reg); crc_reg = uint8(0); end
msb = bitand(bitshift(crc_reg, -7), uint8(1));   % FAILS
```

Inside the body, `crc_reg`'s read goes through a runtime
`matlab_global_get_f64` call (returns f64 from the runtime ABI).
The bitwise lowering needs both operands to be matching integer
types, which fails when one side is f64. Fix: snapshot the
register into a local with a `+ uint8(0)` (or equivalent typed-
zero) anchor:

```matlab
cur = crc_reg + uint8(0);   % anchored as i8
msb = bitand(bitshift(cur, -7), uint8(1));   % works
```

`crc_reg + uint8(0)` is free at synthesis (the synth tool folds
the `+0`), but in MLIR-land it forces the result through a typed
arithmetic op which anchors the value at i8. The `examples/hdl/`
fixtures all use this pattern.

### `if isempty(p) || reset` is the only supported init shape

`isempty(p)` alone works too. But anything more complex
(`isempty(p) && some_cond`, etc.) is rejected — the SV emitter
expects a single guarded init that becomes the register's reset
value.

### `bool` vs `fi(_, _, 1, 0)` for 1-bit ports

Use `% hdl: port(name, bool)` for any 1-bit signal (clock-enable,
ready, valid, etc.). It renders as `logic name` (single-bit,
unsigned). Declaring 1-bit ports as `fi(_, _, 1, 0)` works but
the compiler emits a hint suggesting `bool` instead.

### MATLAB blocking semantics on chained writes

This MATLAB code is technically correct but produces different
behaviour in SV:

```matlab
int1 = int1 + x;
int2 = int2 + int1;   % reads int1's just-written value (MATLAB)
int3 = int3 + int2;   %                                  (MATLAB)
```

In SV, the equivalent `int1_next = int1 + x; int2_next = int2 +
int1` reads `int1`'s pre-edge value — each integrator stage
adds a one-cycle pipeline delay that's invisible to the Python
reference. The cocotb lockstep comparison will mismatch.

**Workaround**: snapshot every persistent at the top of the body
and use the snapshots in expressions:

```matlab
int1_s = int1 + fi(0, 1, 22, 0);   % pre-cycle snapshots
int2_s = int2 + fi(0, 1, 22, 0);
int3_s = int3 + fi(0, 1, 22, 0);
int1 = int1_s + x;
int2 = int2_s + int1_s;
int3 = int3_s + int2_s;
```

This makes Python and SV agree. The pattern is documented in the
[supported subset reference](sv_supported_subset.md) under
"Source-side patterns that need restructuring."

---

## Just-run-everything

The repository ships 39 reference HDL designs under
`examples/hdl/`. Run the cocotb sweep over the 38 verified
modules in parallel:

```sh
ctest --test-dir build -R cocotb-tests        # CI form
# or directly:
bash test/EmitCocoTB/run_tests.sh build/matlabc
```

The runner walks every fixture in
[`test/EmitCocoTB/run_tests.sh`](../test/EmitCocoTB/run_tests.sh)
in parallel (defaults to 8 workers; override with
`COCOTB_PARALLEL=N`). Per-fixture latency lives in source as
`% cocotb: latency(N)`.

Or pick one to inspect by hand:

```sh
just emit-sv examples/hdl/fir_asic_pipelined.m         # write SV to a temp file
just verify-cocotb examples/hdl/fir_asic_pipelined.m   # full lockstep run
```

`just verify-cocotb` writes the harness, runs `make`, and streams
the cocotb output. Failures include the failing cycle, fi-decoded
values, canonical fault hints (saturation suspected, latency
suspected, sign-interpretation), and a path to `dump.vcd` for
waveform inspection.

---

## Where to go from here

| Topic | Doc |
|---|---|
| Full SystemVerilog supported subset (every pragma + every limitation) | [`sv_supported_subset.md`](sv_supported_subset.md) |
| SV backend architecture — passes, lowering pipeline, op handling | [`emit_systemverilog.md`](emit_systemverilog.md) |
| Cocotb design — port-list parser, stimulus pragmas, harness rendering | [`emit_cocotb.md`](emit_cocotb.md) |
| Fixed-point semantics — fi(...), saturation, quantization | [`emit_fixed_point.md`](emit_fixed_point.md) |
| Python emitter — fi runtime, persistent storage | [`emit_python.md`](emit_python.md) |

**Reference designs by category** (all in `examples/hdl/`):

- Combinational: `alu_16bit`, `mux_4to_1_16bit`, `popcount`,
  `priority_encoder`, `leading_zero_detector`, `barrel_shifter`,
  `median3`, `hamming74`, `aes_round`, `cordic_step`
- FSMs: `mealy_fsm`, `moore_fsm`, `computed_state_fsm`, `uart_rx`,
  `spi_master`, `i2c_bit_bang`
- Pipelined DSP: `fir_asic_pipelined`, `cic_decimator`,
  `cordic_pipe`, `sequential_processor`
- Arithmetic engines: `multi_cycle_mul`, `booth_mul`, `crc8`,
  `crc32`, `fnv1a`, `galois_lfsr`
- Memory / dataflow: `regfile`, `regfile_dyn` (runtime indexing),
  `fifo`, `async_fifo`, `axi_handshake`, `mmap_periph`,
  `sync_2ff`, `manchester_enc`
- Hierarchy: `hier_combinational`, `hier_sequential`

Each one compiles, lints clean, and (with one documented
exception — `cic_decimator` — see
[`emit_cocotb.md`](emit_cocotb.md) for the integrator-chain
caveat) verifies cycle-exact under cocotb.
