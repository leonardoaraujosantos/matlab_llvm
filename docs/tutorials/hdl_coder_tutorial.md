# HDL Coder (MATLAB → SystemVerilog + cocotb) — Tutorial

`matlabc` compiles a constrained MATLAB subset into **synthesizable,
vendor-neutral SystemVerilog** and, from the same source, a
**cocotb** testbench that proves the emitted RTL is bit-exact against
a Python reference model. You write the DUT once in MATLAB; the
compiler emits the SV DUT, a Python golden reference, and a lockstep
testbench that drives random vectors into both and compares them
cycle-by-cycle. No hand-written HDL, no hand-written verification.

## Supported features

Language subset (full table in [`../sv_supported_subset.md`](../sv_supported_subset.md)):

- Combinational arithmetic (`+ - * / %`), bitwise ops, all six
  comparisons (signed + unsigned).
- Width-typed `fi(...)` — signed/unsigned at widths {1, 8, 16, 32, 64}.
  Saturation via `'OverflowAction', 'Saturate'` auto-hoists a
  per-width helper.
- `if`/`elseif`/`else` (single-store if/else folds to a ternary via
  the `IfStoreToSelect` pass); `switch`/`case` renders as `unique case`.
- `persistent` registers → flops; each needs an
  `if isempty(reg) || reset` initializer that supplies the reset value.
- FSM cascades (≤ ~16 states) auto-typedef an `enum` and split into
  next-state `always_comb` + state-register `always_ff`.
- Static and persistent `fi`-arrays (1-D and 2-D with constant
  indices); runtime-indexed persistent arrays auto-decode to a
  mux + decoded write enables.
- `% hdl: port(...)` pragmas drive port type/width/signedness on
  **inputs and outputs**.
- Hierarchical multi-module: a `func.call` to a user function becomes
  a SV module instantiation, with `clk`/`rst_n` auto-wired when the
  callee is sequential.

Emit targets: `-emit-systemverilog`, `-emit-python` (reference),
`-emit-cocotb` (the full verification harness), plus
`-check-synthesizable` and `-emit-hardware-report`.

Verification flow: `-emit-cocotb` produces a self-contained directory
that, on `make`, runs Verilator + cocotb driving 100 random vectors
into the SV DUT and the Python reference simultaneously and asserts
cycle-exact equality. The repo's CI lane verifies **39/39** reference
designs bit-exact.

## Build & emit

You need a built `matlabc`, `verilator` ≥ 5.0, and `cocotb` ≥ 2.0
(`pip install cocotb`). The cocotb lane gates on both tools; without
them the SV emit still works but lockstep verification can't run.

```sh
# 1. Emit synthesizable SystemVerilog.
matlabc -emit-systemverilog mux2.m > mux2.sv

# 2. Lint with Verilator (-Wall plus the cosmetic suppressions).
verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL \
    --top-module mux2 mux2.sv

# 3. Emit the cocotb verification harness, then verify.
matlabc -emit-cocotb mux2.m
cd mux2_cocotb && make
#   ** TESTS=1 PASS=1 FAIL=0 SKIP=0 **
```

`-emit-cocotb` internally runs `-emit-systemverilog` (the DUT) and
`-emit-python` (the reference), then writes `test_<name>.py`, the fi
pack/unpack helpers, the Python fi runtime, and a `Makefile` into
`<name>_cocotb/`. After `make`, the directory also holds
`coverage.txt` (per-port stats), `dump.vcd` (waveform), and
`args_trail.jsonl` (per-cycle inputs for replay).

Seed/vector overrides without re-emitting:

```sh
COCOTB_SEED=7 COCOTB_VECTORS=20 make   # alternate seed, shorter run
make sweep N=50                        # sweep 50 seeds, report any FAIL
make replay TRAIL=saved/repro.jsonl    # deterministic re-drive
```

Run the whole reference suite (39 designs, parallel):

```sh
ctest --test-dir build -R cocotb-tests
# or one design by hand:
just emit-sv     examples/hdl/fir_asic_pipelined.m
just verify-cocotb examples/hdl/fir_asic_pipelined.m
```

## Worked examples

### Combinational mux — port pragmas + ternary fold

The minimal DUT: declare port shapes with `% hdl: port(...)`, let the
output type be inferred from the assignment.

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

`%#codegen` marks the function as a code-gen target; the `fi, signed,
16, 0` pragma is a 16-bit signed integer; `bool` is a 1-bit signal.
The single-store if/else collapses to `y = sel ? a : b;` in an
`always_comb` block. The cocotb harness recognises this as
combinational (no clock) and drives 100 random `(a, b, sel)` triples.

### Counter — `persistent` registers + reset (`examples/hdl/counter_0_to_10.m`)

```matlab
function count = counter_0_to_10(reset)
    %#codegen
    % hdl: port(reset, bool)
    persistent count_reg;
    if isempty(count_reg)
        count_reg = fi(0, 0, 4, 0); % 4 bits, counts to 10
    end
    if reset
        count_reg = fi(0, 0, 4, 0);
    else
        if count_reg >= 10
            count_reg = fi(0, 0, 4, 0);
        else
            count_reg = count_reg + 1;
        end
    end
    count = count_reg;
end
```

Each `persistent` becomes a register; the init store inside the
`if isempty(...)` guard becomes its reset value. The emitter adds
`clk`/`rst_n` ports, an `always_comb` computing `count_reg_next`, and
an `always_ff @(posedge clk or negedge rst_n)` that registers it. The
cocotb harness auto-detects the sequential shape and adds clock
generation + reset sequencing.

### Moore FSM — auto enum + `unique case` (`examples/hdl/mealy_fsm.m`)

A `switch` on a persistent state variable with a small contiguous
constant set (`S0=0, S1=1, S2=2`) lowers to a `typedef enum logic
[1:0]` plus a `unique case`, with the `otherwise` arm becoming
`default`. Moore outputs are pure combinational functions of the
state; the Mealy variant (`examples/hdl/mealy_fsm.m`) uses the same
source shape but the output expression also reads inputs. Two
coverage pragmas are worth adding for FSMs:

```matlab
% cocotb: cover(state_display, min_bins=3)        % ≥3 distinct states seen
% cocotb: cover_pairs(state_display, min_pairs=4) % ≥4 distinct transitions
```

`cover` catches the case where 100 random inputs never reach a state;
`cover_pairs` catches an untraversed transition edge even when every
state was visited.

### Pipelined FIR — chained persistents + latency (`examples/hdl/fir_asic_pipelined.m`)

```matlab
function [y, ovfl] = fir_asic_pipelined(x, gain, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 14)
    % hdl: port(gain, fi, signed, 16, 12)
    % hdl: port(reset, bool)
    % cocotb: stimulus(gain, constant, 0.25)
    % cocotb: stimulus(reset, constant, 0)
    % cocotb: latency(4)

    h = fi([0.1, 0.2, 0.3, 0.4], 1, 16, 15);
    persistent delay_line; persistent reg_products;
    persistent reg_acc;    persistent reg_output;
    if isempty(delay_line) || reset
        delay_line = fi(zeros(1, 4), 1, 16, 14);
        ...
    end
    delay_line = [fi(x, 1, 16, 14), delay_line(1:3)]; % shift register
    for i = 1:4
        reg_products(i) = delay_line(i) * h(i);
    end
    ...
    reg_output = fi(full_res, 1, 16, 12, 'OverflowAction', 'Saturate');
    y = reg_output;
end
```

Four pipeline stages = four `persistent` writes. `% cocotb:
stimulus(gain, constant, 0.25)` pins `gain` so random values don't
swamp the filter; `% cocotb: stimulus(reset, constant, 0)` keeps
reset deasserted after init. The constant-bound `for i = 1:4` loops
fully unroll. The Python emitter inserts pre-edge snapshots for
persistent reads that feed other persistents (matching SV's
non-blocking `always_comb` semantics), so chained stages compare
correctly. `% cocotb: latency(4)` skips the first 4 warm-up cycles.

### CRC-8 LFSR — bitwise feedback + register anchoring (`examples/hdl/crc8.m`)

```matlab
cur = crc_reg + uint8(0);                 % anchor the f64 register read as i8
msb = bitand(bitshift(cur, -7), uint8(1));
feedback = bitxor(msb, data_in_u8);
shifted = bitand(bitshift(cur, 1), uint8(254));
crc_reg = bitxor(shifted, feedback * uint8(7));   % polynomial 0x07 taps
```

A persistent register read returns f64 from the runtime ABI; the
bitwise lowering needs both operands to be matching integer types, so
the canonical fix is to snapshot into a typed local with `+ uint8(0)`
(free at synthesis). This pattern recurs across the bitwise fixtures.

### Other reference designs

The CORDIC pipeline (`cordic_pipe.m`, 12 chained registers across 4
stages), the synchronous FIFO (`fifo.m`, case-decoded write port +
multi-source read mux + pointer counters), and `popcount.m`
(constant-position bit extraction into a balanced adder tree) round
out the common shapes. The repo ships 39 designs grouped as:
combinational (`alu_16bit`, `barrel_shifter`, `aes_round`,
`hamming74`, …), FSMs (`uart_rx`, `spi_master`, `i2c_bit_bang`, …),
pipelined DSP (`cic_decimator`, `cordic_pipe`, …), arithmetic engines
(`booth_mul`, `crc32`, `galois_lfsr`, …), memory/dataflow (`regfile`,
`async_fifo`, `axi_handshake`, `mmap_periph`, …), and hierarchy
(`hier_combinational`, `hier_sequential`). Each pairs with a
`*_cocotb/` directory and verifies bit-exact.

## Limitations & carve-outs

From [`../sv_supported_subset.md`](../sv_supported_subset.md):

- `for` loops must have constant bounds (they fully unroll); `while`
  and recursion are rejected at HW legalize.
- Floating-point datapaths are rejected unless wrapped in `fi(...)`.
- Cell arrays / structs in the datapath, and dynamic allocation
  (`zeros(N)` with runtime `N`), are software-only and rejected.
- Integer widths are restricted to i1/i8/i16/i32/i64 — round narrower
  signals up to the next supported width.
- The fixed-point fractional position `F` in `fi(_, S, W, F)` is
  **not** preserved in SV output (storage is integer at width `W`);
  the `F` still drives compile-time saturation bounds.
- `if isempty(p) || reset` (or `isempty(p)` alone) is the only
  supported persistent-init shape.
- Per the cocotb harness: don't drive `reset` high mid-run — the
  fixtures pin `reset=0` after init.
- FPGA primitives, soft-IP wrappers, and dynamic constructs are out
  of scope; the backend targets synthesizable, vendor-neutral SV.

## See also

- Tutorial (end-to-end walkthrough): [`../tutorial_hdl.md`](../tutorial_hdl.md)
- SV supported subset (every pragma + limitation): [`../sv_supported_subset.md`](../sv_supported_subset.md)
- SV backend architecture: [`../emit_systemverilog.md`](../emit_systemverilog.md)
- Cocotb harness design + stimulus/coverage pragmas: [`../emit_cocotb.md`](../emit_cocotb.md)
- Fixed-point semantics: [`../emit_fixed_point.md`](../emit_fixed_point.md)
- Examples: `examples/hdl/`
