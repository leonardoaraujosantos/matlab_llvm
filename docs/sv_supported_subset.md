# SystemVerilog supported subset

Reference for what `matlabc -emit-systemverilog` accepts and rejects.
The backend targets **synthesizable, vendor-neutral SV** for ASIC
flows; FPGA primitives, soft-IP wrappers, and dynamic constructs are
out of scope.

This is a curated, mostly-narrow subset. The compiler diagnoses the
common rejections at source level, but not every gap has a tailored
diagnostic — some surface as MLIR-verifier errors. When in doubt,
follow the patterns in `examples/hdl/`; every module there compiles,
lints clean under Verilator `-Wall`, and (for non-FSM designs) passes
Yosys synthesis.

## Quick reference

| Pattern | Status | Notes |
|---|---|---|
| Combinational arithmetic | ✅ | `+`, `-`, `*`, `/`, `%`, bitwise ops |
| Comparisons | ✅ | All six predicates, signed and unsigned |
| Width-typed `fi(...)` | ✅ | Signed/unsigned, widths in {1, 8, 16, 32, 64} |
| Saturation (`'OverflowAction', 'Saturate'`) | ✅ | Auto-hoists per-width helper functions |
| `switch`/`case` | ✅ | Renders as `unique case`; integer-equality only |
| `if`/`elseif`/`else` | ✅ | Lowers to nested `if`/`else` blocks |
| `persistent` registers | ✅ | Each needs an `if isempty(reg) ... end` initializer |
| FSM cascades | ✅ | Up to ~16 states; auto-typedef enum, encoding-aware |
| Static fi-arrays | ✅ | `fi(zeros(1, N), ...)` with constant indices |
| Persistent fi-arrays | ✅ | Lowers to N parallel scalar persistents (Stage F) |
| Boolean ports declared `bool` | ✅ | Renders as `logic` |
| Bit-slicing `x(hi:lo)` | ✅ | Constant range on scalar int; result widens to next native size |
| Runtime-indexed persistent arrays | ✅ | `arr(addr+1) = v` / `y = arr(addr+1)` auto-decode to mux + decoded enables |
| `% hdl: port(...)` pragmas | ✅ | Drives port type/width/signedness on **inputs and outputs** |
| Multi-source persistent reads (regfile) | ✅ | New: HW-aware slot type-unification (B-workstream) |
| Verbatim source comments preserved | ✅ | Forwarded as `// ...` SV comments |
| Hierarchical multi-module | ✅ | `func.call` to user fn becomes a SV module instantiation; clk/rst_n auto-wire when callee is sequential |
| `for` loop with runtime trip count | ❌ | Only constant-bound loops, get fully unrolled |
| `while` | ❌ | Rejected at HW legalize |
| Recursion | ❌ | Rejected at HW legalize |
| Floating-point datapath | ❌ | Rejected unless wrapped in `fi(...)` |
| Cell arrays / structs in datapath | ❌ | Software-only constructs |
| Dynamic allocation (`zeros(N)` with runtime N) | ❌ | Static shapes only |

## Required idioms

### Persistent registers — initialization

Every `persistent` variable needs its own `if isempty(...) || reset`
guard. The init store inside the guard becomes the register's reset
value.

```matlab
persistent counter;
if isempty(counter) || reset
    counter = uint8(0);   % reset value
end
counter = counter + uint8(1);
```

Multiple persistents share an `if isempty(first_var)` guard ONLY when
they're persistent fi-arrays (Stage F replicates the guard
structurally). For independent scalar persistents, write one guard
per register.

### Type widths

The HW backend accepts only **i1, i8, i16, i32, i64** as integer
widths. Pragma-declared widths outside this set (e.g.
`fi, unsigned, 3, 0` for a 3-bit signal) are rejected. Round up
to the next supported width — the synthesis tool will optimize the
unused bits.

Fixed-point fractional positions (`F` in `fi(_, S, W, F)`) are NOT
preserved in the SV output. The backend renders integer storage at
width `W` only; the user manages scaling externally. The fi spec
remains useful for compile-time saturation bounds.

### Output port pragmas

`% hdl: port(<output_name>, fi, signed/unsigned, W, F)` works for
output port names too — the pragma name resolves against the
function's result attribute (`matlab.name`). The pragma controls
the SV port declaration's signedness and width, overriding the
default-signed-multi-bit rule for output ports whose source
expression doesn't clearly trace back to a typed input or
persistent.

```matlab
function crc = crc32(data_in, en, reset)
    %#codegen
    % hdl: port(data_in, bool)
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % hdl: port(crc, fi, unsigned, 32, 0)   <-- output pragma
    ...
```

Without the output pragma, `crc` would render as
`output logic signed [31:0] crc` (the SV emitter's default for
multi-bit values whose chain doesn't reach a fi-tagged source).
With the pragma, it renders as `output logic [31:0] crc`,
matching the user's intent and aligning the cocotb harness's
DUT-side decode with the Python ref's value.

### Boolean ports

Declare 1-bit signals as `bool` rather than `fi(_, _, 1, 0)` to get
proper `logic` (single-bit, unsigned) port rendering. The compiler
emits an actionable hint when a multi-bit fi port is only used in
boolean predicates:

```
warning: input port 'reset' is 8 bits wide but only used as a boolean
— consider declaring it as `% hdl: port(reset, bool)` ...
```

### CocoTB stimulus pragmas

Functions with explicit `% hdl: port(...)` declarations can also
carry `% cocotb:` pragmas to drive the lockstep test harness. For
pipelined designs, hold streaming inputs constant when their values
shouldn't pipeline-skew (e.g. `fir_asic_pipelined.m` holds `gain`
constant so the SV's per-cycle `gain × reg_acc` aligns with the
Python ref's same-call computation).

```matlab
% hdl: port(gain, fi, signed, 16, 12)
% cocotb: stimulus(gain, constant, 0.25)
```

## Common pitfalls

### "function 'X' result 0 has unsynthesizable type"

The function's return type is still `none` or `f64` after lowering.
Causes:

1. **Multi-source slot picked f64 ABI**: every store into the return
   slot is a `matlab_global_get_f64` ABI call. Fix shipped: HW-aware
   slot retyping in `RefineSlotTypes` recognizes this pattern and
   narrows to the persistent register's underlying integer width.
2. **Hierarchical call with unrefined return**: `y = sub_fn(...)`
   where `sub_fn`'s return type didn't get refined. Inline the
   call body into the parent function as a workaround.
3. **Genuinely unsupported value type** (e.g. f64 in datapath, cell
   array, struct).

### Slot-drop bug (output silently driven by reset)

If your output port reads as `'0` in the SV body with no other
assignment, the slot-drop bug from B1/B2 is a candidate. The fix
ships in `LowerScalarSlots`; see `test/EmitSV/persist_to_output.m`
for the pattern that exercises it.

### Saturation chains render as ternary instead of helper calls

The B1 saturation rendering emits one `function automatic
sat_<sign><IW>_b<W>(arg)` per unique tuple per module. If your
chain renders as inline ternaries, the LowerFiSaturate pass didn't
tag the SelectOps — usually because the saturation goes through
the runtime call (`matlab_fi_quantize_s`) instead of
`matlab_fi_sat_s64`. This happens when the input isn't typed at
the right width before the cast. Try:
- explicit `fi(value, S, IW, F)` recast on the input first
- check that the input is integer-typed (not `none`)

### Boolean / multi-bit port confusion

If a 1-bit signal renders as `logic [7:0]`, the source has
`% hdl: port(name, fi, unsigned, 8, 0)` instead of
`% hdl: port(name, bool)`. The lint hint flags this.

## Designs in `examples/hdl/`

Each compiles, lints clean, and demonstrates a category:

| Module | Demonstrates |
|---|---|
| `aes_round` | AES MixColumns + AddRoundKey on a 32-bit word (xtime XOR network) |
| `alu_16bit` | Combinational arithmetic + bitwise ops + signed overflow |
| `async_fifo` | Single-clock approximation of an async FIFO with gray pointers |
| `axi_handshake` | AXI-Stream-style register slice (valid/ready handshake) |
| `barrel_shifter` | `bitshift(x, K)` with constant K → `arith.shli` |
| `booth_mul` | Signed 8x8 Booth multiplier (signed counterpart to `multi_cycle_mul`) |
| `cic_decimator` | Multi-stage CIC filter, downsample counter, sat at output |
| `computed_state_fsm` | FSM/counter hybrid (post computed-vs-constant gatherFSMs fix) |
| `cordic_pipe` | 4-stage pipelined CORDIC (12 chained persistent registers) |
| `cordic_step` | Single CORDIC iteration in rotation mode (signed shift + cond add/sub) |
| `counter_0_to_10` | Persistent register with reset and modulo wraparound |
| `crc8` | LFSR with XOR feedback into persistent state |
| `crc32` | CRC-32 LFSR (IEEE 802.3 polynomial) — 32-bit XOR feedback |
| `edge_detector` | Single-FF + bool NOT (`matlab.not` rendering) |
| `fifo` | 4-deep synchronous FIFO with full/empty + counter pointers |
| `fir_asic_pipelined` | 3-stage pipelined FIR with persistent fi-arrays |
| `fnv1a` | FNV-1a 32-bit streaming hash (XOR + multiply on persistent i32) |
| `galois_lfsr` | 16-bit Galois LFSR with polynomial XOR feedback |
| `hamming74` | Hamming(7,4) parity-XOR network + bit-pack output |
| `i2c_bit_bang` | I2C master bit-banger — 6-state FSM with phase sub-counter |
| `leading_zero_detector` | 16-input LZD via reverse-priority chain |
| `manchester_enc` | Manchester encoder — 2-state phase FSM + bool XOR/NOT |
| `mealy_fsm` | 2-state Mealy with output dependence on input |
| `median3` | 3-input median filter via min/max compare-select network |
| `mmap_periph` | 4-register memory-mapped peripheral with read/write decode |
| `moore_fsm` | 3-state Moore, output decode from state register |
| `multi_cycle_mul` | 16x8 shift-add multiplier with 3-state FSM |
| `mux_4to_1_16bit` | Combinational case mux |
| `popcount` | Bit-extraction via shift+mask, conditional accumulator |
| `priority_encoder` | 8-input priority encoder via long if/elseif chain |
| `pwm` | Counter + comparator output gating |
| `regfile` | Multi-source persistent read (post type-unification fix) |
| `rr_arbiter` | 4-input round-robin arbiter with rotating priority pointer |
| `sequential_processor` | MAC pipeline with explicit `acc(:)` pattern |
| `spi_master` | SPI master mode 0 — 4-state FSM driving MOSI/SCLK/CS#/done |
| `sync_2ff` | Classic 2-FF clock domain crossing synchronizer |
| `uart_rx` | 11-state FSM (post FSM-cascade aggregation fix) |
| `up_down_counter` | Conditional persistent set with direction control |
| `vector_processor` | Vector arg ports + dot-product + magnitude squared |

## Source-side patterns that need restructuring

These work in MATLAB but currently need to be rewritten before
the SV backend accepts them:

### `uint8(bool_value)` cast — runtime call, not synthesizable

The frontend lowers `uint8(some_bool)` to a runtime call
(`matlab_uint8_s`) which the SV backend can't unwind for runtime
operands (only literal-constant casts get folded in
`IntCastConstantFold`). Workaround: branch.

```matlab
% NOT supported:
%   x_u8 = uint8(rx);   % rx is bool

% Use a branch:
x_u8 = uint8(0);
if rx
    x_u8 = uint8(1);
end
```

The same applies to `int8(bool)`, `uint16(bool)`, etc.

### Persistent-get + bitwise op chain — needs typed snapshot

The runtime ABI's `matlab_global_get_f64` returns f64 regardless
of the register's actual width. Bitwise lowering (`bitand`,
`bitor`, `bitxor`, `bitshift`) requires both operands to share
the same scalar integer type. Without a snapshot, the f64 from
the get-call defeats the lowering.

```matlab
% NOT supported:
%   msb = bitand(bitshift(crc_reg, -7), uint8(1));

% Snapshot first to coerce to the register's i8 width:
cur = crc_reg + uint8(0);
msb = bitand(bitshift(cur, -7), uint8(1));
```

The `+ uint8(0)` is free at synthesis but anchors the SSA value
at i8 (matlab.add propagates through the lowering).

### Multi-source slot writes with mixed types

When a slot is written from multiple branches with different IR
result types (e.g. one branch produces an `arith` result, another
a `matlab.call_builtin` result), the slot's type stays `none` and
HW legalize rejects it. Workaround: pre-seed the slot with a
single-source typed expression and use mutually-exclusive
conditional adds.

```matlab
% Often fails:
%   if cond
%       y = bitxor(a, uint8(1));
%   else
%       y = a;
%   end

% Works:
y_default = a;
y_xor1 = bitxor(a, uint8(1));
if cond
    y = y_xor1;
else
    y = y_default;
end
```

(In practice, hoisting BOTH branches to compute concrete-typed
values at top-level avoids the conditional store mixing types.
See `examples/hdl/crc8.m` for the additive-zero pattern that
sidesteps this.)

## Workarounds for unsupported patterns

### Need a register file with N entries?

Two equivalent forms work — pick whichever reads cleaner:

**Persistent fi-array with runtime indexing** (preferred for >4
entries):

```matlab
persistent regs;
if isempty(regs) || reset
    regs = fi(zeros(1, 8), 1, 16, 0);
end
if we
    regs(waddr + 1) = wdata;     % runtime addr
end
rdata = regs(raddr + 1);
```

Stage F expands each runtime-indexed read into an N-input mux and
each runtime-indexed write into N decoded write enables (one per
register, gated on `addr_int == k`). The synth tool sees the
canonical regfile pattern. See `test/EmitSV/regfile_dyn.m` for a
worked 4-entry example. Scales cleanly to 8–16 entries; larger
files (32+) may want a memory macro instead.

**Manual scalar persistents + switch decode** (the older idiom — see
`examples/hdl/regfile.m`). Compiles to identical RTL but the source
gets verbose past 4 entries.

### Hierarchical sub-modules

A user-defined function called from another function becomes a SV
module instantiation in the caller. Each call site gets a unique
instance name (`u_<callee>_<idx>`) and a wire per result that flows
back into the caller's `always_comb`.

```matlab
function y = top(a, b, c, d)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(c, fi, signed, 16, 0)
    % hdl: port(d, fi, signed, 16, 0)
    s1 = add2(a, b);
    s2 = add2(c, d);
    y = s1 + s2;
end

function s = add2(x, y)
    %#codegen
    s = x + y;
end
```

Sequential helpers (`persistent` registers, FSMs, port pipelining)
get `clk` + `rst_n` ports auto-added. The instantiation site wires
both through, and the caller is itself promoted to a sequential
module if any of its callees needs a clock — the SV emitter computes
the transitive closure across the module's call graph before
emission.

Constraints:
  - Each callee must be present in the same file (or visible in
    the same MLIR module). Cross-file inclusion isn't supported.
  - Recursion is rejected at HW legalize (`hasCycleFrom`).
  - All port types are inferred from the callee's pragmas / function
    signature. Hierarchical type propagation runs as part of the
    standard `LowerScalarsToArith` + `LowerUserCalls` fixpoint.

See `test/EmitSV/hier_combinational.m` and `hier_sequential.m` for
worked examples.

### Bit-slicing — `x(hi:lo)` syntax

Constant-range indexing on a scalar integer extracts a bit-slice. The
result is an unsigned scalar of the rounded-up next-native width
(1, 8, 16, 32, or 64). MATLAB itself treats `x(7:0)` on a scalar as
an empty array, so this overlay doesn't shadow valid MATLAB code —
it's strictly an HDL extension.

```matlab
% hdl: port(x, fi, unsigned, 32, 0)
low_byte   = x(7:0);     % uint8: bits 7..0    → SV `x[7:0]`
high_byte  = x(31:24);   % uint8: high byte    → SV `x[31:24]`
top_bit    = x(31:31);   % logical: MSB        → SV `x[31:31]`
low_word   = x(15:0);    % uint16: low half    → SV `x[15:0]`
three_bit  = x(6:4);     % 3-bit slice → uint8 → SV `8'(x[6:4])`
twelve_bit = x(23:12);   % 12-bit  → uint16    → SV `16'(x[23:12])`
```

Constraints:
  - The range must be a literal `hi:lo` (no explicit step) with
    `hi >= lo >= 0` and `hi < bitwidth(x)`.
  - The source must be a typed scalar integer (port pragma, function
    parameter, or anchored via the `+ uint8(0)` snapshot pattern for
    persistent reads).
  - Slice width must be 1..64.
  - Bit-select rendering as a clean `x[hi:lo]` only fires when the
    source is a function port (its SV-level width matches its MLIR
    width). Slices on intermediate values lower to `arith.shrui` /
    `arith.trunci` / `arith.andi` and render via the generic
    `<W>'(...)` size cast — same gates, slightly noisier syntax.

For non-constant or LHS-side slicing (`y(7:0) = v`), use the
bitand+bitshift workaround — full bit-vector concatenation and
write-side slicing aren't yet supported.

## See also

- `docs/emit_systemverilog.md` — backend architecture and pass pipeline
- `docs/emit_fixed_point.md` — fi op lowering details
- `examples/hdl/` — reference designs
- `test/EmitSV/` — golden fixtures (also reference designs, but
  bench-style)
- `test/EmitSVHint/` — lint-hint test cases
- `test/EmitSVFail/` — patterns the backend explicitly rejects with
  diagnostics
