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
| `% hdl: port(...)` pragmas | ✅ | Drives port type/width on function-only files |
| Multi-source persistent reads (regfile) | ✅ | New: HW-aware slot type-unification (B-workstream) |
| Verbatim source comments preserved | ✅ | Forwarded as `// ...` SV comments |
| Hierarchical multi-module | ⚠️ | `matlab.call` to user fn; return-type refinement gap |
| Dynamic indexing on persistent arrays | ❌ | `mem(i) = v` with runtime `i` not lowered |
| Bit slicing / range select | ❌ | No `x(7:0)` syntax support |
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
| `cordic_step` | Single CORDIC iteration in rotation mode (signed shift + cond add/sub) |
| `counter_0_to_10` | Persistent register with reset and modulo wraparound |
| `crc8` | LFSR with XOR feedback into persistent state |
| `crc32` | CRC-32 LFSR (IEEE 802.3 polynomial) — 32-bit XOR feedback |
| `edge_detector` | Single-FF + bool NOT (`matlab.not` rendering) |
| `fifo` | 4-deep synchronous FIFO with full/empty + counter pointers |
| `fir_asic_pipelined` | 3-stage pipelined FIR with persistent fi-arrays |
| `galois_lfsr` | 16-bit Galois LFSR with polynomial XOR feedback |
| `hamming74` | Hamming(7,4) parity-XOR network + bit-pack output |
| `leading_zero_detector` | 16-input LZD via reverse-priority chain |
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

Use the case-decoded write port + multi-source mux read pattern from
`examples/hdl/regfile.m`. Scales to 8-16 entries cleanly; for larger
files (32+), the per-cell flop count is excessive and a memory
macro is the right answer (out of this backend's scope today).

### Need dynamic-index reads on a small array?

Generate a switch/case manually:

```matlab
% NOT this (gap):
%   y = arr(idx);
% Instead:
switch idx
    case 0; y = arr(1);
    case 1; y = arr(2);
    ...
end
```

### Need hierarchical sub-modules?

Inline the sub-function body into the parent. The optimizer will
deduplicate logic at synthesis. Until the multi-module emission
gap is closed, this is the recommended pattern.

### Need a bit slice (`x(7:0)`)?

Use bitwise mask + shift:

```matlab
low_byte = bitand(x, uint16(255));     % x[7:0]
high_byte = bitshift(bitand(x, uint16(65280)), -8); % x[15:8]
```

## See also

- `docs/emit_systemverilog.md` — backend architecture and pass pipeline
- `docs/emit_fixed_point.md` — fi op lowering details
- `examples/hdl/` — reference designs
- `test/EmitSV/` — golden fixtures (also reference designs, but
  bench-style)
- `test/EmitSVHint/` — lint-hint test cases
- `test/EmitSVFail/` — patterns the backend explicitly rejects with
  diagnostics
