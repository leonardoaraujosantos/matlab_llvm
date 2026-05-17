# Fixed-Point Designer — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ HDL emit) needs to ship in order to faithfully **compile and
execute**, **debug/REPL**, **emit HDL from**, and **demo** Fixed-Point
Designer programs.

This roadmap is the tier-organised **compatibility** view. For the
**implementation-detail** view (type-system extension, MIR / MLIR
representation, per-backend code-generation rules, runtime storage
layout), see the companion [`emit_fixed_point.md`](emit_fixed_point.md)
— the implementation plan that drove the shipped phases.

**Runtime location**: scalar `fi` arithmetic + array constructors +
fi-array runtime helpers live in `runtime/matlab_runtime.cpp` (the
core TU) under the `matlab_fi_*` and `matlab_mat_i64_*` /
`matlab_mat_u64_*` prefixes. The lowering passes that emit idiomatic
shift-and-saturate code live in `lib/MLIR/Passes/LowerFixedPoint.cpp`,
`lib/MLIR/Passes/LowerFiSaturate.cpp`, and
`lib/MLIR/Passes/LowerPersistentFiArrays.cpp`.

Source: *Fixed-Point Designer User's Guide* (R2026a). Companion docs:
[`emit_fixed_point.md`](emit_fixed_point.md) (full implementation
plan + lowering rules + per-backend code-gen), [`emit_systemverilog.md`](emit_systemverilog.md)
(SV emission consumes the same fi descriptors for synthesis),
[`sv_supported_subset.md`](sv_supported_subset.md) (fi-array port
declarations + bit-slicing), [`feature_status.md`](feature_status.md),
[`roadmap.md`](roadmap.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. The
  Fixed-Point Designer roadmap was originally framed as **Phases**
  in `emit_fixed_point.md` (Phase 1 scalar → Phase 5 persistent
  fi-arrays → Phase 6 SV synthesis polish). This roadmap re-frames
  the shipped Phases as **Tiers 1–5** to uniformly match the other
  per-toolbox roadmaps.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started ·
  🔴 deliberately deferred.
- **Tiers 1 → 5 are all ✅ shipped.** Headline outcome: a MATLAB
  `fi` program lowers to **idiomatic int + shift code** in
  C / C++ / Python / TypeScript / LLVM, and to **synthesizable
  fixed-point SystemVerilog** with bit-exact cycle-by-cycle cocotb
  parity. Open Tier-6 follow-ons: function-internal fi typing
  across user calls (the biggest UX gap), 2-D fi matrices,
  reductions tail (`prod`/`min`/`max`/`cumsum`/`dot`), fi parfor
  reductions, slope/bias scaling, complex fi, 3-D fi arrays.
- **REPL / Debug**: fi values render with their full numerictype
  banner in `disp`, the DAP variable inspector renders `Q<int>.<frac>`
  + integer + real-world value, `whos` shows the integer storage
  class. Persistent fi arrays survive REPL boundaries.

---

## 1. Tier 0 — baseline (inherited from core)

These primitives sit underneath the fi surface:

| Group | Functions / capabilities | Notes |
|---|---|---|
| Native int storage | `int8` / `int16` / `int32` / `int64` / `uint8` / `uint16` / `uint32` / `uint64` matrix descriptors | Saturating arithmetic, comparisons, casts, REPL+DAP display |
| Scalar arith dispatch | `+ - * / \ ^` with auto-promotion | LowerScalarsToArith path; mixed-mode fi op double goes through the implicit-cast rule |
| Workspace renderer | `disp` / `whos` for typed-int + fi values | Includes the `numerictype` / `fimath` banner |

---

## 2. Tier 1 — scalar `fi` MVP ✅ **shipped**

The foundational slice: scalar Q-format arithmetic with explicit
`Saturate` / `Wrap` overflow and the canonical `Floor` / `Nearest`
rounding modes.

### 2.1 Constructors ✅

| Form | Meaning |
|---|---|
| `fi(value)` | Default `numerictype(1, 16, 15)` — Q1.15 signed |
| `fi(value, signed, WL, FL)` | Explicit signed / word length / fraction length |
| `fi(value, T)` | `T` is a `numerictype` object |
| `fi(value, T, F)` | `T` + `F` (`fimath` object) |

### 2.2 Operators ✅

`+ - *` shipped with auto-FL-tracking: `Q(a).f + Q(b).f → max(f_a,f_b)`
on the FL, `Q(a).f * Q(b).f → f_a + f_b`. Quantize-and-saturate is
applied on `lhs(:) = rhs` assignment per the user's stored numerictype.

### 2.3 Methods / property access ✅

- `int(n)` / `storedInteger(n)` — raw integer storage
- `double(n)` — real-world value
- `bin(n)` / `hex(n)` / `dec(n)` — base-formatted display
- `numerictype(n)` — extract the type tag
- `fimath(n)` — extract the math attributes (`OverflowAction`,
  `RoundingMethod`, `ProductWordLength`, `SumWordLength`, etc.)

### 2.4 Numeric configuration ✅

- **Overflow modes**: `Saturate` (default), `Wrap`. Both shipped
  in the runtime + LowerFixedPoint.
- **Rounding modes**: `Floor`, `Nearest`, `Zero`, `Ceiling`,
  `Convergent` — all five shipped (the original Phase 1 plan
  promised Floor + Nearest only; the rest landed in the same arc).
- **Word lengths**: native lanes `WL ∈ {8, 16, 32, 64}` plus
  sub-native (e.g. WL=12 in the i16 lane) via masking.
- **Fraction length**: `FL ≤ WL`.

### 2.5 Tier-1 closure summary

| Primitive | Status |
|---|---|
| Scalar `fi` constants, `+ - *`, signed + unsigned (§2.1, §2.2) | ✅ shipped |
| `Saturate` / `Wrap` overflow (§2.4) | ✅ shipped |
| All 5 rounding modes (§2.4) | ✅ shipped |
| WL ∈ {8, 16, 32, 64} + sub-native (§2.4) | ✅ shipped |
| FL ≤ WL (§2.4) | ✅ shipped |
| Scalar `lhs(:) = rhs` (§2.2) | ✅ shipped |
| Gating test: `examples/fi_basic.m` (constructor + arithmetic + disp) | ✅ shipped |

**Status**: ✅ shipped. Covers `apply_gain` and the scalar MAC.

---

## 3. Tier 2 — fimath / numerictype object surface ✅ **shipped**

Promotes `numerictype` and `fimath` from string tags to first-class
classdef objects with cross-input persistence.

### 3.1 `numerictype` ✅

```matlab
T = numerictype(1, 16, 15);     % signed Q1.15
T = numerictype('Signed', true, 'WordLength', 16, 'FractionLength', 15);
disp(T.WordLength);             % 16
disp(T.FractionLength);         % 15
disp(T.Signed);                 % 1
```

### 3.2 `fimath` ✅

```matlab
F = fimath('OverflowAction', 'Saturate', ...
           'RoundingMethod', 'Nearest', ...
           'ProductWordLength', 32, ...
           'ProductFractionLength', 30);
setfimath(F);
% ... fi arithmetic uses F's settings
removefimath();
```

### 3.3 `reinterpretcast` ✅

`reinterpretcast(n, T)` re-tags an existing fi value with a new
numerictype without changing the underlying bits — useful for
bit-juggling FFT scaling, CORDIC pipelines, etc.

### 3.4 Tier-2 closure

| Primitive | Status |
|---|---|
| `numerictype` first-class classdef (§3.1) | ✅ shipped |
| `fimath` first-class classdef (§3.2) | ✅ shipped |
| `setfimath` / `removefimath` (§3.2) | ✅ shipped |
| `reinterpretcast` (§3.3) | ✅ shipped |
| `-emit-fixed-point-report` driver flag (per-`fi` summary of WL/FL/saturate sites) | ✅ shipped |

---

## 4. Tier 3 — fi arrays + FIR gating workload ✅ **shipped**

The first user-visible fi slice **at array scale**: scalar Q-format
plus vector concat + slice + reductions, enough to write an FIR
filter that lowers to idiomatic shift-and-add.

### 4.1 fi-array constructors ✅

```matlab
delay_line = fi(zeros(1, N), 1, 16, 12);     % zero-init 16-bit Q12 vector
h = fi(taps, 1, 16, 14);                     % filter taps
```

### 4.2 fi-array operations ✅

- **Indexing**: `delay_line(i)`, `delay_line(1:end-1)` (slice)
- **Concat**: `[x, delay_line(1:end-1)]`
- **Reductions**: `sum`, `mean`
- **Element-wise**: `*`, `+` element-by-element

### 4.3 Persistent fi-array storage ✅

```matlab
function y = my_filter(x)
  persistent delay_line
  if isempty(delay_line)
    delay_line = fi(zeros(1, N), 1, 16, 12);
  end
  delay_line = [x, delay_line(1:end-1)];
  y = sum(delay_line .* h);
end
```

The `persistent` keyword lowers to a static-storage fi array, with
proper init-on-first-call gating. The same pattern emits a clean
shift register in SystemVerilog (see [`emit_systemverilog.md`](emit_systemverilog.md)).

### 4.4 Implicit promotion ✅

`fi + double` auto-promotes the double to fi using the fi operand's
numerictype, so users don't have to write `fi(0.5)` everywhere.

### 4.5 Tier-3 closure

| Primitive | Status |
|---|---|
| `fi(zeros(1, N), ...)` array constructor (§4.1) | ✅ shipped |
| Element indexing + slice (§4.2) | ✅ shipped |
| Vector concat `[x, v(1:end-1)]` (§4.2) | ✅ shipped |
| `sum` / `mean` reductions on fi (§4.2) | ✅ shipped |
| `persistent` fi-array storage (§4.3) | ✅ shipped |
| Implicit `fi + double` promotion (§4.4) | ✅ shipped |
| **Gating test**: `examples/fi_filter.m` (full FIR filter) | ✅ shipped |

---

## 5. Tier 4 — emit-* lanes parity ✅ **shipped**

The same `fi` source program produces idiomatic output across every
backend. This Tier is closed in the sense that **the headline FIR
test runs bit-identically across LLVM / C / C++ / Python**; TypeScript
has one residual rough edge (see Tier-6).

| Backend | Status | Notes |
|---|---|---|
| LLVM IR / native | ✅ | Shift + saturate ops lower through `LowerFixedPoint.cpp` |
| `-emit-c` | ✅ | Idiomatic `int16_t y = (int16_t)((acc + (1 << (s-1))) >> s)` style |
| `-emit-cpp` | ✅ | Same as C; `auto` opt-in via `-cpp-auto` |
| `-emit-python` | ✅ | numpy `int16` / `int32` arithmetic |
| `-emit-typescript` | 🟡 | FIR gating test is `.skip-emit-typescript` — mixed BigInt × number arithmetic on fi-array element reads needs a coercion pass (Tier-6 §7.2) |
| `-emit-systemverilog` | ✅ | `% hdl: port(...)` pragmas + scalar fi + 1-D fi-array shift registers + bit-slicing `x(hi:lo)` (1..64 wide) all lint clean under Verilator |
| `-emit-cocotb` | ✅ | Cycle-by-cycle co-simulation of the SV DUT against the Python reference bit-identical |
| `-emit-fixed-point-report` | ✅ | Per-`fi` summary of WL / FL / saturate sites |
| `-emit-hardware-report` | ✅ | Includes fi register counts + saturation overflow points |

---

## 6. Tier 5 — synthesis polish (persistent fi-arrays as SV regfiles) ✅ **shipped**

The biggest SV-side advance: a persistent fi-array becomes a
runtime-indexed regfile, with the standard `wr_en` / `wr_data` /
`rd_data` interface that any HDL author would expect.

### 6.1 Persistent fi-arrays → SV shift registers ✅

```matlab
function y = fir_shift(x)
  persistent dly
  if isempty(dly)
    dly = fi(zeros(1, 4), 1, 16, 14);
  end
  dly = [x, dly(1:end-1)];
  y = sum(dly .* h);
end
```

Lowers to:
```systemverilog
reg signed [15:0] dly [0:3];
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin dly[0] <= 0; dly[1] <= 0; ...
  end else begin
    dly[3] <= dly[2]; dly[2] <= dly[1]; dly[1] <= dly[0]; dly[0] <= x;
  end
end
```

### 6.2 Runtime-indexed fi-arrays → SV regfile ✅

```matlab
function y = regfile_read(addr)
  persistent regs
  if isempty(regs)
    regs = fi(zeros(1, 16), 1, 16, 0);
  end
  y = regs(addr);
end
```

Lowers to an auto-decoded address-mux regfile pattern — the case
statement is emitted from the `dly(addr)` MATLAB indexing.

### 6.3 Tier-5 closure

| Primitive | Status |
|---|---|
| Persistent fi-array shift register (§6.1) | ✅ shipped |
| Runtime-indexed persistent fi-array → auto-regfile (§6.2) | ✅ shipped |
| Hierarchical multi-module emission (`func.call` → SV instance with auto-wired `clk` / `rst_n`) | ✅ shipped |
| 7 fi-spec port-declaration regression tests in `test/EmitSVPorts/` | ✅ shipped |
| 2 boolean-port lint-hint tests in `test/EmitSVHint/` | ✅ shipped |
| 10 synthesizability-gate diagnostic tests in `test/EmitSVFail/` | ✅ shipped |

---

## 7. Tier 6 — open follow-ons 🔵 **next slices**

Five items remain open. They're sized in dependency order; the
function-internal typing fix is the biggest UX unlock.

### 7.1 Function-internal fi typing across user calls 🔵 (~1 week)

```matlab
function y = my_helper(x)        % what numerictype does x have?
  y = x * 2;
end
% Caller:
v = fi(0.5, 1, 16, 14);
out = my_helper(v);              % what numerictype does out have?
```

Today Sema doesn't propagate the fi spec through user-function
boundaries — `x` inside `my_helper` is `f64`. Workaround: inline
the helper or repeat the `fi(...)` wrap inside the helper.

**Plan**: Extend Sema's user-function specialisation pass to
monomorphise on caller-side fi numerictype. Already proven for
typed-int matrices in Phase 1.1; this is the same template applied
to fi descriptors.

**Effort**: ~1 week. This is the **biggest UX gap** in the shipped
surface.

### 7.2 2-D fi matrices 🔵 (~1.5 weeks)

Tier 3 ships 1-D fi vectors only. `A(i, j)` on a 2-D fi matrix has
a runtime path through `matlab_mat_i64_subscript2_s`, but tested
only via 1-D shape today. Needs concrete 2-D indexing tests,
slice2, and matmul on fi matrices (the shift-and-accumulate
pattern, not just element-wise).

**Effort**: ~1.5 weeks. Unlocks fi matmul / image-processing-style
code.

### 7.3 fi reductions tail 🔵 (~3 days)

`sum` / `mean` shipped in Tier 3. `prod` / `min` / `max` / `cumsum`
/ `dot` return `any` from Sema today, so the `(:)` clamp on the
result fails. Each is a small Sema + lowering hookup that mirrors
the `sum` path.

**Effort**: ~3 days.

### 7.4 fi `parfor` reductions 🔵 (~1 week)

The pthread fan-out runtime needs to know the integer storage
class. Originally this needed the typed-int runtime to land first
(it has — Phase 3) so this is now actually doable. Reduction
operator (`+`, `*`, `min`, `max`, `bitand`, `bitor`) per-thread
accumulator scheduling + cross-thread merge with the right
saturation semantics.

**Effort**: ~1 week.

### 7.5 emit-typescript fi-array shifts on BigInt 🔵 (~3 days)

The FIR gating test is `.skip-emit-typescript` because mixed
BigInt × number arithmetic on fi-array element reads needs a
coercion pass in `EmitTypeScript.cpp`. Either teach the emitter
to wrap any operand of a shift in `BigInt(...)` when the producer
is a fi-array subscript, or adopt a number-only TS shim for
WL ≤ 32.

**Effort**: ~3 days.

### 7.6 Tier-6 closure summary

| Primitive | Effort | Status |
|---|---|---|
| Function-internal fi typing across user calls (§7.1) | 1 wk | 🔵 — biggest UX gap |
| 2-D fi matrices (§7.2) | 1.5 wk | 🔵 — unlocks fi matmul / image-processing code |
| fi reductions tail (`prod` / `min` / `max` / `cumsum` / `dot`) (§7.3) | 3 days | 🔵 — small hookups |
| fi `parfor` reductions (§7.4) | 1 wk | 🔵 — typed-int runtime prerequisite has shipped |
| emit-typescript fi-array shifts on BigInt (§7.5) | 3 days | 🔵 — single coercion pass in `EmitTypeScript.cpp` |

**Total Tier-6**: ~3 weeks of focused sessions.

---

## 8. Tier 7+ — long-tail features 🔵 **not started**

Less-common but documented MATLAB fi surface that hasn't been
needed yet:

| Feature | Effort | Notes |
|---|---|---|
| Slope / bias scaling (`numerictype('Scaling', 'SlopeBias')`) | ~1 week | More general than binary-point scaling; needed for sensor calibration |
| Complex `fi` (`fi(complex(re, im), ...)`) | ~1.5 wk | Two real fi arrays packaged as one complex fi — used in QAM / OFDM HDL |
| 3-D fi arrays | ~1 wk | Same template as 2-D once that lands |
| `fipref` display preferences | 3 sess | Per-session control of how fi values render |
| `quantizer` standalone object | 3 sess | A `quantizer` without storage — useful as a parametric scale-and-saturate function |
| `tic` / `toc` instrumentation of fi blocks | 1 sess | Profile saturation rates |
| `coder.config('fixed-point', ...)` MATLAB Coder compat | — | Carved out — see §9 |

---

## 9. Out of scope (Fixed-Point Designer carve-outs)

- **Fixed-Point Tool app** — interactive auto-conversion GUI.
- **`coder.config('fixed-point', ...)` MATLAB Coder integration** —
  this project *is* a code generator; MathWorks Coder compatibility
  is a different product.
- **Floating-to-fixed automatic conversion** (the `fxpopt` /
  `propose_fl` analysis pass). Could be done as a separate offline
  tool, but not for Tier-6.
- **Histogram-based scaling analysis** (the data-driven WL/FL
  search that the Coder UI runs). Same reason.
- **HDL Coder app integration**. We emit synthesizable SV directly;
  HDL Coder is its own product.
- **Embedded MATLAB Function block** (Simulink). Out of scope.

---

## 10. What Fixed-Point Designer brings to the rest of the roadmap

- **Fi → SystemVerilog (synthesis)**: scalar fi + persistent fi-arrays
  are the building blocks the SV emit lane reads. Every `% hdl:
  port(...)` pragma that names a fixed-point type uses the same
  `numerictype` descriptor. See [`emit_systemverilog.md`](emit_systemverilog.md)
  and [`sv_supported_subset.md`](sv_supported_subset.md).
- **Fi → cocotb co-simulation**: the bit-identical SV vs. Python
  reference path is what makes fi a credible HDL design tool. See
  [`emit_cocotb.md`](emit_cocotb.md).
- **Fi → Signal Processing Toolbox**: FIR / IIR / sosfilt all
  accept `fi` inputs and propagate via the §7.1 function-internal
  typing work once it lands. Today the typing fix is the gating
  item.
- **Fi → Control System Toolbox**: digital controllers expressed in
  fixed-point (the `c2d_tustin` of an analog plant + a fi-quantised
  state-space). Same gating: §7.1 function-internal typing.
- **Fi → Embedded Coder**: the mflowLink whole-diagram emit (per
  [`embedded_coder_roadmap.md`](embedded_coder_roadmap.md))
  consumes fi-typed signals end-to-end on the SV lane — the cocotb
  SIL examples (`cocotb_pid_sil.mflow`) rely on Tier-1+3+5 already
  being shipped.

---

## 11. Execution order — if user demand drives prioritization

| Order | What | Effort | Status |
|---|---|---|---|
| 1 | Tier 1 scalar `fi` MVP (constructors / `+ - *` / saturate / rounding modes / WL ∈ 8/16/32/64) | 2.5 wk | ✅ shipped (Phase 1) |
| 2 | Tier 2 `numerictype` / `fimath` first-class objects + `reinterpretcast` + `-emit-fixed-point-report` | 1.5 wk | ✅ shipped (Phase 4) |
| 3 | Tier 3 fi-array constructors + indexing + slice + concat + `sum`/`mean` + `persistent` storage + implicit promotion + FIR gating test | 2 wk | ✅ shipped (Phase 3) |
| 4 | Tier 4 emit-* parity across LLVM / C / C++ / Python / SV / cocotb (TS one rough edge) | 1 wk | ✅ shipped (Phase 2) |
| 5 | Tier 5 synthesis polish (persistent fi-arrays → SV shift registers / runtime-indexed → regfile) + hierarchical multi-module emit | 1.5 wk | ✅ shipped (Phase 5) |
| 6 | Tier-6.1 function-internal fi typing across user calls | 1 wk | 🔵 — biggest UX gap |
| 7 | Tier-6.2 2-D fi matrices | 1.5 wk | 🔵 — unlocks fi matmul / image-processing |
| 8 | Tier-6.3 fi reductions tail (`prod` / `min` / `max` / `cumsum` / `dot`) | 3 days | 🔵 |
| 9 | Tier-6.4 fi `parfor` reductions | 1 wk | 🔵 |
| 10 | Tier-6.5 emit-typescript fi-array shifts on BigInt | 3 days | 🔵 |
| 11 | Tier-7 slope/bias scaling + complex fi + 3-D fi + `fipref` + `quantizer` | ~5 wk | 🔵 — long-tail |

**Tier-6 closure**: ~3 weeks. Lights up function-call-based fi
designs and 2-D matrix arithmetic — the most-requested gaps in the
shipped surface.

---

## 12. Gating tests + internal references

- Runtime: [`runtime/matlab_runtime.cpp`](../runtime/matlab_runtime.cpp)
  (`matlab_fi_*` + `matlab_mat_i64_*` + `matlab_mat_u64_*` entries)
- Lowering passes: [`lib/MLIR/Passes/LowerFixedPoint.cpp`](../lib/MLIR/Passes/LowerFixedPoint.cpp),
  [`lib/MLIR/Passes/LowerFiSaturate.cpp`](../lib/MLIR/Passes/LowerFiSaturate.cpp),
  [`lib/MLIR/Passes/LowerPersistentFiArrays.cpp`](../lib/MLIR/Passes/LowerPersistentFiArrays.cpp)
- Frontend: builtins registered in `lib/Sema/Builtins.cpp` under
  the `fi` / `numerictype` / `fimath` / `setfimath` / `removefimath`
  / `reinterpretcast` / `int` / `storedInteger` / `double` / `bin`
  / `hex` / `dec` groups
- Implementation reference: [`emit_fixed_point.md`](emit_fixed_point.md)
  (the original phase-organised plan that drove the shipped
  surface — kept as the in-tree reference for the type-system
  extension, MIR layout, MLIR ops, per-backend code-gen, and the
  16-section implementation map)
- HDL counterpart: [`emit_systemverilog.md`](emit_systemverilog.md)
  §5 ("Advanced Arithmetic — fixed-point"),
  [`sv_supported_subset.md`](sv_supported_subset.md) (fi port
  pragmas + bit-slicing)
- Cocotb co-sim: [`emit_cocotb.md`](emit_cocotb.md)
- Project-wide roadmap: [`roadmap.md`](roadmap.md) Phase 1.x /
  Phase 4.x history
- Authoritative compat matrix: [`feature_status.md`](feature_status.md)
  §3 ("Numeric types & values") `fi` row

**Gating tests**:

| Lane | Tests | What it gates |
|---|---|---|
| `test/Run/fi_basic.m` | constructor + arithmetic + disp | Tier 1 |
| `test/Run/fi_numerictype.m` | first-class `numerictype` + `fimath` | Tier 2 |
| `test/Run/fi_filter.m` | full FIR filter end-to-end | Tier 3 (the headline gating test) |
| `test/EmitSVPorts/` (7 tests) | fi port-decl regressions | Tier 5 |
| `test/EmitSVHint/` (2 tests) | boolean-port lint hints | Tier 5 |
| `test/EmitSVFail/` (10 tests) | synthesizability gate diagnostics | Tier 5 |
| `examples/hdl/cic_decimator.m` + `fir_asic_pipelined.m` + cocotb cosim | bit-identical SV vs Python reference | Tiers 4 + 5 |
