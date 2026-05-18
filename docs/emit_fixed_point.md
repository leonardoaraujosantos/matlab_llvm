# Fixed-Point (`fi`) Support — Plan

This document scopes adding **MATLAB Fixed-Point Designer (`fi`)**
support to `matlab_llvm`. It is the planning artifact for the
`feat/fixed-point` branch / worktree at
`/Users/leonardoaraujo/work/matlab_llvm-fi`.

The intent is that user code such as

```matlab
a = fi(1.5, 1, 16, 8);   % signed Q8.8
b = fi(2.25, 1, 16, 8);
c = a + b;
disp(c);
```

is parsed, type-checked, lowered through MLIR, and emitted as portable
C/C++ that does the math with native integer types and explicit shifts
— exactly the shape MathWorks' MATLAB Coder produces, and the same
shape `docs/emit_systemverilog.md` Phase 5 ("fixed-point arithmetic
policies") will need.

This is a frontend + Sema + lowering + runtime addition, not a new
backend.

## 1. Goals

- Parse and type-check `fi`, `fimath`, `numerictype`, `fipref` and
  the property-access surface used in everyday fixed-point code.
- Track word length, fraction length, signedness, overflow mode, and
  rounding mode in the Sema type lattice end-to-end.
- Lower `fi` arithmetic to integer ops + explicit shifts (no f64 in
  the datapath of `-emit-c` / `-emit-cpp` output).
- Provide a small runtime helper set (saturation, rounding helpers)
  in `runtime/matlab_runtime.{c,h,hpp}`.
- Produce numerically faithful results for the supported subset
  (overflow + rounding behavior matches MATLAB to bit, for the modes
  we ship).
- Round-trip `fi` through `-emit-c`, `-emit-cpp`, `-emit-llvm`, and
  the REPL/`disp`.

## 2. Non-Goals

- `fi` arrays beyond what the existing `matlab_mat` 2-D dense layout
  can carry. (3-D fi arrays are explicitly out — same boundary as
  the rest of the project.)
- Slope/Bias scaling beyond binary-point-only (`Slope = 2^-FL`,
  `Bias = 0`). [Slope Bias] arithmetic — section 3.14 of the User's
  Guide — is deferred.
- Custom word lengths > 64 bits. We rely on native `int8/16/32/64`
  and `uint8/16/32/64` storage; arbitrary precision (e.g. WL = 73)
  is out.
- `fipref` GUI / display preferences beyond what `disp` already
  needs.
- `fiaccel`, the automated double→fixed-point conversion workflow
  (Chapter 7 of the User's Guide). Out of scope; users supply
  explicit `fi(...)` calls.
- Lookup-table replacement of transcendentals (`exp` → table).
- Bit-exact agreement with MathWorks for `Slope ≠ 2^-FL` paths.
- Live `fxpopt` / `fxptdlg` tooling.

## 3. MATLAB Surface To Support

Phase-1 (MVP) surface, drawn from the User's Guide chapters 1–5:

### 3.1 Constructors

```matlab
n = fi(value)                          % default 16/15 signed
n = fi(value, signed, wordLength)
n = fi(value, signed, wordLength, fractionLength)
n = fi(value, numerictypeObj)
n = fi(value, numerictypeObj, fimathObj)
```

### 3.2 Companion objects

```matlab
T = numerictype(signed, WL, FL)
F = fimath('OverflowAction','Saturate', 'RoundingMethod','Floor', ...)
n = fi(value, T, F)
```

### 3.3 Operators

`+ - * .* ./ - (unary)` between `fi` operands, and between `fi` and
`double` (with explicit cast lifting).

### 3.4 Methods / property access

- `n.Value`, `n.WordLength`, `n.FractionLength`, `n.Signed`
- `n.IntegerLength` (== WL − FL − Signed)
- `int(n)` — stored integer
- `double(n)` — real-world value
- `bin(n)`, `hex(n)`, `dec(n)` — display helpers
- `reinterpretcast(n, T)` — bit-reinterpret without changing storage
- `removefimath(n)`, `setfimath(n, F)`
- `storedInteger`, `storedIntegerToDouble` (synonyms in modern code)

### 3.5 Numeric configuration we honor

- **Signedness**: signed two's complement, unsigned.
- **Word length**: 8, 16, 32, 64 (round up to the nearest native
  storage class on emission). Sub-byte widths (e.g. 4) are *legal*
  inputs and tracked exactly in the type — but emitted code always
  stores them in the smallest containing native int and mask/saturate
  on every write.
- **Fraction length**: any integer ≥ 0 and ≤ WL. Negative FL ("scaling
  larger than 1 LSB") is deferred.
- **Overflow modes**: `Saturate`, `Wrap`. Both shipped in Phase 1.
- **Rounding modes**: `Floor` (truncate toward −∞), `Nearest`,
  `Zero` (truncate toward 0), `Convergent` (banker's), `Ceiling`.
  Phase 1 ships `Floor` + `Nearest`; rest are stubs that error
  cleanly.

### 3.6 Out of scope for MVP

- `Slope ≠ 2^-FL` (general slope-bias scaling)
- complex `fi`
- 3-D `fi` arrays
- `fi` as `classdef` property type (works mechanically since classdef
  properties are dynamic, but no special path)

## 4. Type-System Extension

Today `Dtype` is a flat `enum class` (`include/matlab/Sema/Type.h`).
Fixed-point doesn't fit because it needs WL/FL/signedness/overflow/
rounding parameters per value.

### 4.1 Add a `Dtype::Fixed` tag

```cpp
enum class Dtype : uint8_t {
  Unknown, Logical, Char, Double, Single, Complex,
  Int8,  Int16,  Int32,  Int64,
  UInt8, UInt16, UInt32, UInt64,
  Fixed,            // <-- new
};
```

### 4.2 Carry params on `ArrayType`

`ArrayType` gains an optional `FixedSpec` payload (only populated
when `Elt == Dtype::Fixed`):

```cpp
struct FixedSpec {
  bool Signed;
  uint8_t WordLength;       // 1..64
  int8_t  FractionLength;   // 0..WL  (negative deferred)
  enum class Overflow : uint8_t { Saturate, Wrap } OF;
  enum class Rounding : uint8_t { Floor, Nearest, Zero,
                                  Convergent, Ceiling } RM;
  bool operator==(const FixedSpec &) const = default;
};
```

`TypeContext::fixedScalar(spec)` and `fixedArray(spec, shape)` get
interned the same way `arrayOf` is today. `promoteDtype` extends to
return a joined `Fixed` (matched WL/FL via the `fimath`-style
arithmetic rules in §3.10–3.14 of the User's Guide).

### 4.3 Sema/Resolver

- `lib/Sema/Resolver.cpp:45` — add `"fi"`, `"numerictype"`, `"fimath"`,
  `"fipref"` to the builtin name list.
- `lib/Sema/TypeInference.cpp:684` — extend the cast-builtin block:
  - `fi(v, ...)` returns
    `ArrayType{Fixed, shape(v), FixedSpec(...)}`, folding constant
    args at compile time when present (almost always are).
  - `int(n)`, `storedInteger(n)` return the matching native integer.
  - `double(n)` returns `Double`.

### 4.4 Display / dumping

- `dtypeName(Dtype::Fixed)` returns a parameter-bearing name like
  `numerictype(1,16,8)` for sema dumps.
- `Type::toString()` includes `FixedSpec` so `-emit-sema` is
  inspectable.

## 5. AST / Parser

No new syntax. `fi(...)`, `numerictype(...)`, `fimath(...)` parse
today as ordinary `CallOrIndex`. Property access (`n.WordLength`,
`n.Value`) parses as ordinary `FieldAccess`.

The only parser-side touch point is in Sema, mapping field accesses
on a `Fixed` array to compile-time-known integer reads (no runtime
lookup needed).

## 6. Runtime Representation

The runtime stays integer-native — *no* f64 detour for fi values.

### 6.1 Storage policy

- WL ≤ 8  → `int8_t`  / `uint8_t`
- WL ≤ 16 → `int16_t` / `uint16_t`
- WL ≤ 32 → `int32_t` / `uint32_t`
- WL ≤ 64 → `int64_t` / `uint64_t`

For sub-native widths (e.g. WL=12), arithmetic is done in the
containing class and **every assignment** goes through a
mask+saturate helper that enforces the actual WL.

### 6.2 New scalar runtime helpers

```c
// Saturating cast from an arbitrary-width signed/unsigned integer
// down to a (signed?, WL)-clipped lane. Compiler emits the call only
// when the source range can exceed the destination.
int64_t  matlab_fi_sat_s64(int64_t x, uint8_t WL);   // signed
uint64_t matlab_fi_sat_u64(uint64_t x, uint8_t WL);  // unsigned

// Rounding helpers for shifting a wide product down by `shift` bits
// under the chosen rounding mode. Returns the shifted value.
int64_t  matlab_fi_round_floor_s   (int64_t x, uint8_t shift);
int64_t  matlab_fi_round_nearest_s (int64_t x, uint8_t shift);
uint64_t matlab_fi_round_floor_u   (uint64_t x, uint8_t shift);
uint64_t matlab_fi_round_nearest_u (uint64_t x, uint8_t shift);
// (zero/convergent/ceiling stubs land later, error-flag on call)

// Real-world value -> stored integer (compile-time helper, used by
// the constant folder; runtime version exists for dynamic fi(value)).
int64_t  matlab_fi_quantize_s(double v, uint8_t WL, int8_t FL,
                              uint8_t overflow, uint8_t rounding);
uint64_t matlab_fi_quantize_u(double v, uint8_t WL, int8_t FL,
                              uint8_t overflow, uint8_t rounding);

// disp / fprintf — render a fi value as its real-world double.
void matlab_fi_disp_s(int64_t  stored, uint8_t WL, int8_t FL);
void matlab_fi_disp_u(uint64_t stored, uint8_t WL, int8_t FL);
```

### 6.3 Array layout

`fi` arrays reuse the existing `matlab_mat` descriptor with an
**element-class swap**: instead of `double *data`, we add a
`matlab_mat_i` family parameterized by element kind (i8, i16, i32,
i64, u8…u64). This mirrors how the project already plans the
"typed integer runtime" item in `docs/feature_status.md` §8 — fi is
the forcing function for that work, and the two land together.

For Phase 3 we ship `matlab_mat_i64` / `matlab_mat_u64` only and
store all fi arrays in 64-bit lanes; tighter lanes follow once
codegen proves out.

## 7. Lowering

### 7.1 MIR / MLIR

- The MLIR `matlab` dialect grows two scalar ops:
  - `matlab.fi.const` — folds `fi(constant, …)` into an
    `arith.constant` of the appropriate integer type, with the
    FixedSpec on an attribute.
  - `matlab.fi.cast` — convert between two FixedSpecs (the
    `fi(x, T2)` rebind operation).
- Arithmetic stays as ordinary `arith.addi`, `arith.muli`,
  `arith.shr*i`, `arith.shli`, plus calls into the helpers above for
  saturation / rounding when the inferred FixedSpec demands it.
- A new pass `LowerFixedPoint` runs *between* `runLowerScalarsToArith`
  and `runLowerSeqLoops` (i.e. inserted at `tools/matlabc/main.cpp:215`
  and `:1444`). Responsibilities:
  - lift FixedSpec metadata off `matlab.fi.*` ops onto `arith` ops
    it rewrites
  - decide per binop whether overflow can happen given the operand
    specs (skip the saturate call when the result type already covers
    the worst case)
  - decide per binop whether rounding is needed (no rounding when
    `FL_out == FL_in_left == FL_in_right`)

### 7.2 Arithmetic rules (User's Guide §1.3, §3.10–3.14)

- **add/sub** — operands aligned to the larger FL by left-shift;
  result WL grows by 1 unless `fimath.SumMode` says otherwise.
  Default `KeepLSB` matches MATLAB.
- **mul** — stored integers multiplied in the next-wider native
  class (`int32 × int32 → int64`); FL of result is FL_a + FL_b, then
  shifted back per `fimath.ProductMode`. **Project default is
  `KeepLSB`** (matches MathWorks Coder's emission shape; MATLAB's
  literal default `FullPrecision` is exposed via Phase-4 `fimath`
  surface).
- **unary minus** — for unsigned, errors at compile time (matches
  MATLAB).
- **cast `fi → fi`** — shift+saturate+round chain emitted inline.

### 7.3 Code generation (`-emit-c` / `-emit-cpp`)

For the canonical Q8.8 add example,

```matlab
a = fi(1.5, 1, 16, 8);
b = fi(2.25, 1, 16, 8);
c = a + b;
```

emits roughly:

```c
int16_t a = (int16_t)384;            // 1.50 * 2^8
int16_t b = (int16_t)576;            // 2.25 * 2^8
int16_t c = (int16_t)((int32_t)a + (int32_t)b);   // no shift; FL aligned
```

For multiplication:

```matlab
y = a * b;     % keep 16/8
```

emits:

```c
int16_t y = (int16_t)(((int32_t)a * (int32_t)b) >> 8);
```

For saturating overflow, the cast wraps in a helper:

```c
int16_t y = (int16_t)matlab_fi_sat_s64((int64_t)((int32_t)a * (int32_t)b) >> 8, 16);
```

#### Worked example — `apply_gain` (the MATLAB Coder canonical case)

```matlab
function y = apply_gain(x)         % x: fi(_, 1, 16, 8)
    gain = fi(1.5, 1, 16, 8);      % stored int = 384
    y = x * gain;                  % result: fi(_, 1, 16, 8)
end
```

Sema folds the `fi(1.5, 1, 16, 8)` constructor at compile time into
the literal `384` with `FixedSpec{signed=1, WL=16, FL=8}`. The mul
widens, then `LowerFixedPoint` shifts the product back from FL=16
down to FL=8.

Generated C:

```c
int16_t apply_gain(int16_t x) {
    const int16_t gain_fixpt = 384;       /* 1.5 * 2^8 */
    int32_t prod = (int32_t)x * gain_fixpt;
    return (int16_t)(prod >> 8);          /* Wrap, Floor */
}
```

With `Saturate` + `Nearest` (also Phase 1):

```c
int16_t apply_gain(int16_t x) {
    const int16_t gain_fixpt = 384;
    int32_t prod = (int32_t)x * gain_fixpt;
    int32_t round = matlab_fi_round_nearest_s(prod, 8);  /* +0.5 LSB then >> 8 */
    return (int16_t)matlab_fi_sat_s64(round, 16);
}
```

This is exactly the form a Cortex-M0 / DSP toolchain expects — no
`rtwtypes.h`, only `<stdint.h>`.

#### Worked example — FIR filter (the Phase 3 gating case)

```matlab
function y = fir_filter_fixpt(x, h)        % x: fi 16/14, h: fi[1×4] 16/14
    persistent delay_line;
    if isempty(delay_line)
        delay_line = fi(zeros(1, 4), 1, 16, 14);
    end
    delay_line = [x, delay_line(1:end-1)];

    acc = fi(0, 1, 32, 28);                % wider acc to absorb bit growth
    for i = 1:length(h)
        acc(:) = acc + (delay_line(i) * h(i));
    end
    y = fi(acc, 1, 16, 14);
end
```

This is the gating example for Phase 3. It exercises:

- scalar `fi` arithmetic on `acc` — Phase 1
- `fi` *array* construction (`fi(zeros(1,4), …)`) — Phase 3
- vector concat + slice (`[x, delay_line(1:end-1)]`) on `fi` — Phase 3
- `length(h)` / indexing `h(i)` / `delay_line(i)` on `fi` — Phase 3
- `persistent` storage of an `fi` array — needs the persistent
  runtime extension in §12
- the **`acc(:) = ...` type-preserving assignment** idiom (without
  `(:)`, the bit width would grow on each iteration) — see §11

Expected C emission:

```c
static int16_t delay_line[4];
static int8_t  delay_line_init = 0;

int16_t fir_filter_fixpt(int16_t x, const int16_t h[4]) {
    if (!delay_line_init) {
        for (int32_t k = 0; k < 4; ++k) delay_line[k] = 0;
        delay_line_init = 1;
    }
    /* shift register: delay_line = [x, delay_line(1:end-1)] */
    delay_line[3] = delay_line[2];
    delay_line[2] = delay_line[1];
    delay_line[1] = delay_line[0];
    delay_line[0] = x;

    int32_t acc = 0;            /* fi(0, 1, 32, 28) — FL=14+14=28 already */
    for (int32_t i = 0; i < 4; ++i) {
        acc += (int32_t)delay_line[i] * h[i];   /* MAC, no shift */
    }
    return (int16_t)(acc >> 14);                /* narrow back to 16/14 */
}
```

Key codegen points the lowering must get right:

- `fi(zeros(1,4), 1, 16, 14)` constant-folds to a zero-init
  `int16_t[4]`, not a runtime `matlab_zeros + cast` call.
- The MAC body emits `acc += int32 * int16` with **no shift** because
  the accumulator's FL=28 exactly matches `FL_a + FL_b = 14 + 14`.
  The `LowerFixedPoint` pass must recognize this "no shift needed"
  case to avoid emitting `>> 0`.
- The final narrowing happens once, on the `y = fi(acc, 1, 16, 14)`
  cast — that's a single `>> 14` (and an optional saturate / round
  helper if the function-level `fimath` requests them).
- Whether `delay_line` becomes `static int16_t[4]` or a heap-allocated
  `matlab_mat_i16*` depends on the persistent-storage extension
  (§12). For Phase 3 we ship the static-array form when the size
  is compile-time constant — closest to the MathWorks Coder shape
  and the synthesizable shape the future SV backend will need.

Generated code is target-agnostic C99 — readable, no MathWorks
`rtwtypes.h` dependency. We use `<stdint.h>` only.

### 7.4 `-emit-llvm`

Same lowering as C-emit; falls out of MLIR's existing `arith` →
LLVM conversion (`tools/matlabc/main.cpp:248`,
`createArithToLLVMConversionPass`). No backend-specific work expected.

### 7.5 `-emit-python` / `-emit-typescript`

Both shims fall back to a pure-numeric implementation that stores
the underlying integer in a NumPy / TS `BigInt` lane and applies the
same shifts on every op. These are simulation-grade, not bit-exact
to the C path for `WL > 53`. Marked as a known limitation in the
docs for those backends.

## 8. CLI / Tooling Surface

- (Phase 4) `-emit-fixed-point-report` flag: dumps a per-variable
  summary (WL/FL/overflow-saturation count seen at compile time) —
  equivalent in spirit to the type-proposal report MATLAB Coder
  shows.
- REPL: `disp(fi(1.5,1,16,8))` shows `1.5` plus a one-line type tag
  (e.g. `numerictype(1,16,8)`), matching MATLAB's display.
- Formatter: nothing to do — `fi(...)` is already a plain call.
- LSP: hover on a fixed-point value shows its FixedSpec.
- DAP: `Locals` renders fi values as their real-world value with
  the type tag.

## 9. Test Corpus

Add a dedicated lane `fixed-point-tests` under `test/Run/` (matches
the existing convention — see `test/Run/run_tests.sh` and
`CMakeLists.txt:158`):

- `fi_basic.m` — constructor, property access, `disp`. (Phase 1)
- `fi_add_align.m` — same-FL add, mixed-FL add (forces shift). (1)
- `fi_mul_keep.m` — Q8.8 × Q8.8 → Q8.8 (default `ProductMode`). (1)
- `fi_mul_full.m` — `FullPrecision` (Q16.16 result). (4)
- `fi_overflow_wrap.m` — `Wrap` mode wraps as expected. (1)
- `fi_overflow_saturate.m` — `Saturate` clamps to range. (1)
- `fi_round_floor.m`, `fi_round_nearest.m`. (1)
- `fi_unsigned.m`. (1)
- `fi_mac_scalar.m` — scalar multiply-accumulate (the Phase-1
  acceptance test): `acc(:) = acc + a*b` × 2, then narrow. (1)
- `fi_array.m` — 1-D vector of `fi`, sum/dot product. (3)
- `fi_to_double.m` — round-trip `fi` → `double` → `fi`. (1)
- `fi_filter.m` — full FIR (the Phase 3 gating test). (3)

Plus golden-output files (`*.stdout`) for each, run on all four
emission lanes (`-emit-llvm`, `-emit-c`, `-emit-cpp`, `-emit-python`).
The Python lane gets a `*.skip-emit-python` marker for the
high-WL cases that BigInt can't faithfully reproduce.

Examples gallery: one `examples/fi_filter.m` mirroring the FIR test.

## 10. Phasing

| Phase | Surface | Notes |
|---|---|---|
| 1 | Scalar `fi` constants, `+ - *`, signed + unsigned, `Saturate` (default), `Floor` + `Nearest`, WL ∈ {8,16,32,64}, FL ≤ WL, scalar `lhs(:) = rhs` | **✅ Shipped.** Covers `apply_gain` and the scalar MAC. `Wrap` overflow is implemented in the runtime + LowerFixedPoint, but currently unreachable from MATLAB syntax (needs the `fimath` object surface — Phase 4). |
| 2 | Sub-native WL (e.g. WL=12), implicit `fi + double` promotion, `bin/hex/dec` display, `int(n)` / `storedInteger(n)` / `double(n)` | **✅ Shipped.** Tests: `fi_subnative.m`, `fi_mixed_double.m`, `fi_bin_hex.m`, `fi_int_extract.m`. |
| 3 | `fi` arrays (1-D), `length`/`size`/`numel`, indexing `A(i)` and slicing `A(1:end-1)`, vector concat `[x, A(1:end-1)]`, `persistent` storage of `fi` arrays, reductions on `fi` (`sum`, `mean`) | **✅ Shipped (script form).** Backed by `matlab_mat_i64` / `matlab_mat_u64` heap descriptors. Gating test `fi_filter.m` exercises the full FIR shape from §7.3. Function-internal fi typing across user calls is deferred — the FIR runs at script scope. emit-typescript: scalar shifts on BigInt operands need a coercion pass; FIR test marked skipped on that lane. |
| 4 | `fimath`, `numerictype` as first-class objects; `setfimath`/`removefimath`; `-emit-fixed-point-report` | **✅ Shipped.** `numerictype(s,WL,FL)` and `fimath('OverflowAction','Wrap'\|'Saturate','RoundingMethod','Floor'\|'Nearest')` are compile-time-only types; `fi(value, T)` and `fi(value, T, F)` read the spec out at type-inference time and fold the constructor away. The `Wrap` overflow mode is now reachable from MATLAB syntax. `fipref` accepted but no-op for now. Tests: `fi_numerictype.m`, `fi_fimath_wrap.m`, `fi_setfimath.m`. |
| 5 | Convergent / Zero / Ceiling rounding, `reinterpretcast` | **✅ Shipped.** All five rounding modes (Floor, Nearest, Zero, Ceiling, Convergent/banker's) flow through both the constructor quantize path and the runtime shift path used by mul/cast. `reinterpretcast(n, T)` bit-reinterprets the stored integer as a different numerictype with matching storage width. Tests: `fi_round_modes.m`, `fi_reinterpretcast.m`. |
| 6 | Slope/Bias scaling | Deferred — likely never. |

Phase 1 is the gating deliverable. Everything after it is an
incremental, independently-shippable patch.

### 10.1 What's still missing after Phase 5

The phasing table above is **closed** for Phases 1–5 and Phase 6 stays
deferred. The work below is what's *not* in the original plan but
remains visible in real fi programs — grouped by criticality.

#### High-impact gaps

| Gap | Scope | Why it matters | Reference |
|---|---|---|---|
| **Function-internal fi typing across user calls** | ~1 week | `function y = apply_gain(x)` doesn't propagate the fi spec from the call site into the body. Workaround today: keep fi arithmetic at script scope (the FIR gating test does this). The fix extends `runMonomorphiseUserCalls` to split on `FixedSpec` (not just `Dtype`), so `apply_gain(fi(_, 1, 16, 8))` clones a body that types `x` as `fi 1/16/8`. | §11 — original "function input typing" open question |
| **2-D fi arrays (matrix subscripts)** | ~1.5 weeks | Phase 3 ships 1-D fi vectors only. `A(i,j)` on a 2-D fi matrix has the path through `matlab_mat_i64_subscript2_s`, but tested only via 1-D shape today. Needs concrete 2-D indexing tests, slice2, and matmul on fi matrices (the shift-and-accumulate pattern, not just element-wise). | §6.3 |
| **emit-typescript fi-array shifts on BigInt** | ~3 days | The FIR gating test is `.skip-emit-typescript` because mixed BigInt × number arithmetic on fi-array element reads needs a coercion pass in `EmitTypeScript.cpp`. Either teach the emitter to wrap any operand of a shift in `BigInt(...)` when the producer is a fi-array subscript, or adopt a number-only TS shim for WL ≤ 32. | Phase 3 commit |
| **fi reductions tail** (`prod`, `min`, `max`, `cumsum`, `dot`) | ~3 days | `sum`/`mean` shipped in Phase 3. The other reductions return `any` from Sema today, so the (:) clamp on the result fails. Each is a small Sema + lowering hookup that mirrors the `sum` path. | §3.4 |

#### Medium-impact gaps

| Gap | Scope | Notes |
|---|---|---|
| **fi parfor reductions** | ~1 week | The pthread fan-out runtime needs to know the integer storage class. The §11 plan flagged this as needing the typed-int runtime to land first (it has — Phase 3) so this is now actually doable. |
| **`fipref` honored for display formatting** | ~2 days | Recognised as a builtin but no-op; MATLAB's `fipref` controls precision and number-of-digits output. The `disp(fi)` path always prints `%g`. |
| **LSP hover with FixedSpec** | ~1 day | `matlab-lsp` doesn't show fi metadata on hover. The Sema type is already there; just needs the LSP `hover` handler to detect `Dtype::Fixed` and format the spec. |
| **DAP `Locals` renders fi as real-world value** | ~1 day | Today fi values show as their raw stored integer in the debugger. Should call `matlab_fi_disp_*` formatting machinery for the variables panel. |
| **Diagnostic when `fi + double` literal doesn't fit** | ~half-day | §11 wanted "diagnose loudly when the double constant doesn't fit". Today the `fi + 5.0e9` case silently saturates to the WL max. |
| **`fi(x, T)` where `T` is non-literal numerictype** | ~2 days | Works when `T = numerictype(1, 16, 8)` is constant-folded at compile time. A runtime-built `T` (e.g. selected by an `if`) bails to `any`. |

#### Low-impact / deliberate non-goals

| Item | Status | Reason |
|---|---|---|
| Slope/Bias scaling (`Slope ≠ 2^-FL`) | ❌ Deferred — likely never | §2 non-goal. No real call for it in DSP/HDL workflows since the FFT/filter literature already standardised on binary-point-only. |
| Custom WL > 64 | ❌ Out of scope | §2 non-goal. Storage class would have to grow (i128 is rare in MATLAB code). |
| Complex `fi` (fi values with imaginary parts) | ❌ Out of scope | §3.6 non-goal. Real-only fi covers the gating DSP cases. |
| 3-D fi arrays | ❌ Out of scope | §3.6 non-goal — bounded by the same project-wide 2-D limit. |
| `fi` as `classdef` property type | 🟡 Mechanically works, untested | §3.6: classdef properties are dynamic so a fi value stores fine, but no special path / no test. |
| Lookup-table replacement of transcendentals (`exp` → table) | ❌ Out of scope | §2 non-goal — explicit. |
| `fxpopt` / `fxptdlg` tooling | ❌ Out of scope | §2 non-goal — explicit. |
| `fiaccel` (auto double→fi conversion) | ❌ Out of scope | §2 non-goal — users supply explicit `fi(...)` calls. |

#### Suggested next-up order

If pulling the next deliverable from this list, the order that gives the
most user-visible win per day:

1. **Function-internal fi typing** (high-impact, unlocks `function y = apply_gain(x)` form of every example).
2. **2-D fi arrays** (high-impact, unlocks fi matmul / image-processing-style code).
3. **fi reductions tail** (low-effort, fills out the §3.4 surface).
4. **emit-typescript BigInt coercion** (cleans up the one outstanding test skip).
5. **DAP / LSP polish** (small standalone wins for users actively debugging fi code).

## 11. Open Questions

- **Promotion rule for mixed `fi + double`.** MATLAB's default is to
  cast the double to the fi's numerictype (with the fi's
  rounding/overflow). We adopt that, but we should diagnose loudly
  when the double constant doesn't fit.
- **Storage of compile-time-constant `fi` literals in the AST.** We
  currently fold `IntegerLiteral`/`FPLiteral` in `lib/MIR/Lowering.cpp`.
  Two viable options: a new `FixedLiteral` AST node, or attaching a
  `FixedSpec` to `IntegerLiteral` post-Sema. Leaning toward the
  latter — fewer node kinds, no parser change.
- **Function input typing.** `apply_gain(x)` doesn't say what type
  `x` is. The project uses Sema-driven monomorphization at call
  sites today (`feature_status.md` §2). We adopt that for fi too:
  no `coder.typeof` annotation surface in MVP. Add an explicit
  annotation only if a real example forces it.
- **Interaction with `parfor`.** Reduction on `fi` requires the
  pthread runtime to know the integer storage class. Doable, but it
  forces the typed-integer runtime work in Phase 3 to land before
  parallel fi reductions are claimed.
- **Display formatting.** MATLAB shows `fi(1.5, 1, 16, 8)` as
  `1.5000` plus a "DataTypeMode: Fixed-point: binary point scaling"
  block. We will ship the value-only display in Phase 1 and a
  proper type banner in Phase 4 alongside the `numerictype` object
  surface.
- **`acc(:) = rhs` type-preserving assignment.** This idiom is
  essential to fixed-point code — without it, the natural assignment
  `acc = acc + x*y` would let `acc` re-infer its FixedSpec each
  iteration and grow bit width unboundedly. Sema needs to recognize
  `lhs(:) = rhs` for an *fi scalar* `lhs` as "cast `rhs` into `lhs`'s
  spec, do not re-infer". The parser already accepts this syntax;
  the work is purely in `TypeInference.cpp` and `LowerFixedPoint`.

## 12. `persistent` Runtime Extension

The current `persistent` runtime in `runtime/matlab_runtime.cpp`
(grep for `Global / persistent storage`, currently ~line 13095)
is a flat 128-slot **scalar f64 table**. The FIR example in §7.3
needs persistent storage of an *integer array*. Two pieces of work
land in Phase 3:

- Generalize the slot table to hold a typed pointer instead of an
  f64 — `union { double f; void *ptr; }` keyed by an enum, or a
  parallel pointer table indexed by the same ID. The compiler
  already namespaces persistent IDs per declaring function, so no
  resolver changes are needed.
- Emit the C-side declaration as `static <storage>[N] name;` plus a
  `static int8_t name_init = 0;` flag for the `isempty(...)` check —
  this matches MathWorks Coder and is the form the SV backend can
  later lift to a registered RAM/ROM. For non-constant-size persistent
  state we fall back to the heap-allocated `matlab_mat_*` path.

This extension is independent of fi (it unblocks ordinary
`persistent` arrays of any dtype), but the FIR example is what
forces it onto the critical path. Worth doing it under the fi work
since fi arrays are its first real consumer.

## 13. Relationship to `emit_systemverilog.md`

Phase 5 of `emit_systemverilog.md` ("Advanced Arithmetic — fixed-
point arithmetic policies") presupposes exactly this work. Once
`fi` is in the type lattice and the C backend emits explicit-shift
integer code, the SV backend can mirror that lowering verbatim —
`int16_t` becomes `logic signed [15:0]`, the same shift chains map
to combinational RTL.

The two backends should share a single `LowerFixedPoint` pass; SV
emission becomes a thin printer over the same lowered arith ops.

## 14. Documentation Updates

- `docs/feature_status.md` §3 (Numeric types & values): add row for
  `fi` / fixed-point.
- `docs/feature_status.md` §9 roadmap: insert "Fixed-point (`fi`)
  scalar + arithmetic" between current items 2 and 3.
- `docs/emit_c_cpp.md`: new section on the fi lowering, with the
  Q8.8 add/mul examples above.
- `docs/README.md`: add an index entry for this file.
- `README.md` "Main Features": add a fi example block.

## 15. Out-of-Tree References

- *Fixed-Point Designer User's Guide* (R2026a),
  `~/Downloads/fixedpoint.pdf` — chapters 1–5 cover the surface this
  plan supports; chapters 7–9 (`fiaccel`, single-precision
  conversion, manual workflow) are explicitly out of scope.
- HDL Coder User's Guide — no direct dependency, but the SV plan's
  Phase 5 numeric policy must align with what this document fixes.

---

## 16. Implementation Map

This is the file-by-file checklist. Each entry lists what needs to
change and which phase it lands in. Items marked **[P1]** are
required for Phase 1; **[P3]** for Phase 3; **[Px]** for later.

### 16.1 Frontend headers — `include/matlab/`

| File | Change | Phase |
|---|---|---|
| `Sema/Type.h` | Add `Dtype::Fixed` enum value. Add `FixedSpec` struct. Add `ArrayType::FxSpec` field (only valid when `Elt == Fixed`). Add `TypeContext::fixedScalar(spec)` and `fixedArray(spec, shape)`. | **P1** |
| `Sema/TypeInference.h` | Add helper signatures for fi arithmetic spec promotion (`promoteFixedAdd`, `promoteFixedMul`, `promoteFixedCast`). | **P1** |
| `MLIR/TypeMapper.h` | Add `mapFixedToInt(FixedSpec) -> mlir::IntegerType`. | **P1** |
| `MLIR/Passes/Passes.h` | Declare `bool runLowerFixedPoint(mlir::ModuleOp)`. | **P1** |

### 16.2 Frontend implementation — `lib/`

| File | Change | Phase |
|---|---|---|
| `Sema/Type.cpp` | Extend `dtypeName(Dtype::Fixed)` to render `numerictype(s,WL,FL)`. Extend `promoteDtype`. Add `isInteger(Dtype::Fixed) == true`. Implement `TypeContext::fixedScalar`/`fixedArray` interning (parallel to `arrayOf`). Extend `ArrayKey` hash/eq to include `FixedSpec`. | **P1** |
| `Sema/Resolver.cpp:45` | Add `"fi"`, `"numerictype"`, `"fimath"`, `"fipref"`, `"storedInteger"`, `"reinterpretcast"`, `"removefimath"`, `"setfimath"` to the builtin name table. | **P1** |
| `Sema/TypeInference.cpp:684` | Extend the cast-builtin block: handle `fi(value)`, `fi(value, signed, WL)`, `fi(value, signed, WL, FL)`, `fi(value, T)`, `fi(value, T, F)`. Constant-fold the constructor when args are literals. Handle `int(n)`, `storedInteger(n)`, `double(n)`. | **P1** |
| `Sema/TypeInference.cpp` (binop visitor) | Apply `promoteFixedAdd/Mul/Cast` when either operand has `Dtype::Fixed`. Implement the `lhs(:) = rhs` type-preserving assignment for fi scalars. | **P1** |
| `Sema/TypeInference.cpp` (FieldAccess visitor) | Resolve `n.WordLength` / `n.FractionLength` / `n.Signed` / `n.Value` etc. as compile-time integer reads when `n` is an `Array` of `Fixed`. | **P1** |
| `Sema/SemaDumper.cpp` | Render `FixedSpec` in `-emit-sema` output. | **P1** |
| `MIR/MIR.cpp` / `MIR/Builder.cpp` | New `MIR::FixedConst` op (or extend `Const` with FixedSpec). Optional — folding can also live in `MLIR/Lowering.cpp` directly. | **P1** |
| `MIR/Lowering.cpp` | Constant-fold `fi(literal, …)` into a stored-integer constant + FixedSpec metadata. Today the equivalent for plain numbers is the `IntegerLiteral` / `FPLiteral` fold path here. | **P1** |
| `MIR/Printer.cpp` | Pretty-print fi constants as `fi(<value>, <spec>)`. | **P1** |

### 16.3 MLIR — `lib/MLIR/`

| File | Change | Phase |
|---|---|---|
| `Dialect/MatlabDialect.cpp` | Register `matlab.fi.const` and `matlab.fi.cast` ops. The FixedSpec rides as an `mlir::Attribute` (`FixedSpecAttr`) on each op. | **P1** |
| `TypeMapper.cpp` | Map `Dtype::Fixed` to the corresponding `IntegerType` (i8/i16/i32/i64) on its native storage class. Sub-native WLs round up; the FixedSpec attribute carries the actual WL for downstream passes. | **P1** |
| `Lowering.cpp` | Emit `matlab.fi.const` for folded fi literals; emit `matlab.fi.cast` for `fi(x, T)` rebinds; emit ordinary `matlab.add` / `matlab.mul` / `matlab.minus` for fi binops with FixedSpec attributes. | **P1** |
| `Passes/LowerFixedPoint.cpp` | **NEW FILE.** Walks the module after `LowerScalarsToArith`; rewrites each `matlab.fi.*` and arith op carrying a FixedSpec attribute into the integer-shift sequence described in §7.2. Decides per binop whether overflow / rounding helpers are needed. Also handles the constant-fold path: a `matlab.fi.const` becomes an `arith.constant` of integer type. | **P1** |
| `Passes/EmitC.cpp` | Recognize the integer-shift sequences emitted by `LowerFixedPoint` and pretty-print them in idiomatic C (e.g. preserve the `(int32_t)x * y >> 8` form rather than emitting an opaque `arith.shrsi`). Wire the `matlab_fi_*` runtime helper calls. | **P1** |
| `Passes/EmitPython.cpp` | Same pretty-printing rules using NumPy `np.int*` types. Add a Python shim path that uses `BigInt`-equivalent (Python's native unbounded ints) for WL > 53. | **P1** |
| `Passes/EmitTypeScript.cpp` | Mirror EmitPython but emit TS `BigInt` for WL > 32 (TS `number` is f64 ≈ 53-bit safe-int), `number` otherwise. | **P1** |
| `Passes/LowerToLLVMIR.cpp` | No change — once `LowerFixedPoint` rewrites everything to plain `arith.*` ops, the existing `createArithToLLVMConversionPass` covers the rest. | — |

### 16.4 Pipeline registration — `tools/matlabc/main.cpp`

Insert the new pass in two places (the `-emit-c`/`-emit-cpp` path
and the `-emit-llvm` path):

| Site | Insertion |
|---|---|
| `tools/matlabc/main.cpp:215` | Add `mlirgen::runLowerFixedPoint(M);` between `runLowerScalarsToArith` (line 214) and the second `runSlotPromotion` (line 215) — so FixedPoint sees the same SSA shape as the rest of the scalar lowering. |
| `tools/matlabc/main.cpp:1444` | Same insertion in the second pipeline copy. |

(The dual sites are an existing pattern in the driver — one for the
LLVM/JIT path, one for the emit-C path — so the new pass needs to
live in both.)

### 16.5 Runtime — `runtime/`

| File | Change | Phase |
|---|---|---|
| `matlab_runtime.h` | Declare the helpers from §6.2 (`matlab_fi_sat_*`, `matlab_fi_round_*`, `matlab_fi_quantize_*`, `matlab_fi_disp_*`). | **P1** |
| `matlab_runtime.cpp` | Implement those helpers. ~80 lines. Add right after the existing `matlab_int*_s` block (currently ~line 6278; grep for `matlab_int8_s`). | **P1** |
| `matlab_runtime.cpp` (persistent storage; grep `Global / persistent storage`) | Generalize the 128-slot f64 table to hold typed pointers as well — see §12. | **P3** |
| `matlab_runtime.h` (persistent extension) | New `matlab_persistent_get_ptr(id)` / `_set_ptr(id, ptr)` declarations. | **P3** |
| `matlab_runtime.cpp` (typed integer matrix) | Add `matlab_mat_i64` / `matlab_mat_u64` descriptors with the same shape API as `matlab_mat`. Reductions and `disp` get integer-aware overloads. **Status (2026-05): shipped** — see `runtime.md` §3.1 typed-integer row. | **P3** |
| `matlab_runtime.h` (typed integer matrix) | Declarations for the above. | **P3** |
| `matlab_runtime.hpp` | Thin C++ wrappers if any (mirrors existing pattern). | **P1** |
| `matlab_runtime.py` | Python shim with the same five helper signatures, backed by Python int / NumPy. | **P1** |
| `matlab_runtime.ts` | TypeScript shim with `BigInt`-backed implementations. | **P1** |

### 16.6 Tests — `test/Run/` and `test/`

| File | Change | Phase |
|---|---|---|
| `test/Run/fi_basic.m` + `.stdout` | Constructor + property access + disp. | **P1** |
| `test/Run/fi_add_align.m`, `fi_mul_keep.m`, `fi_overflow_wrap.m`, `fi_overflow_saturate.m`, `fi_round_floor.m`, `fi_round_nearest.m`, `fi_unsigned.m`, `fi_mac_scalar.m`, `fi_to_double.m` (+ `.stdout`) | Phase-1 acceptance corpus. | **P1** |
| `test/Run/fi_array.m`, `fi_filter.m` (+ `.stdout`) | Phase-3 corpus. | **P3** |
| `test/Run/fi_*.skip-emit-python` (selected) | Mark Python lane skips for high-WL tests. | **P1** |
| `CMakeLists.txt` | No change needed — the existing `run-tests` / `run-tests-emit-c` / `run-tests-emit-cpp` lanes (lines 158–195) auto-pick up new `.m` files in `test/Run/`. | — |
| `test/Run/run_tests.sh` | No change — golden-output diff is already file-driven. | — |

### 16.7 Examples gallery — `examples/`

| File | Change | Phase |
|---|---|---|
| `examples/fi_filter.m` | The FIR filter from §7.3. | **P3** |
| `examples/fi_apply_gain.m` | The simpler gain example from §7.3. | **P1** |
| `examples/README.md` | Add entries for the new examples. | **P1** / **P3** |

### 16.8 Documentation — `docs/`

| File | Change | Phase |
|---|---|---|
| `docs/emit_fixed_point.md` | This file. Kept up to date as phases land. | **P1** |
| `docs/feature_status.md` §3 | Add a `fi` row to the numeric-types table. | **P1** |
| `docs/feature_status.md` §9 | Insert a "Fixed-point" item in the roadmap. | **P1** |
| `docs/emit_c_cpp.md` | New "Fixed-point lowering" section with the §7.3 examples. | **P1** |
| `docs/emit_python.md` / `docs/emit_systemverilog.md` | Cross-link to this doc. | **P1** |
| `docs/README.md` | Index entry. | **P1** |
| `README.md` | One example block under "Main Features". | **P1** |

### 16.9 Phase-1 ordering (suggested commit sequence)

1. Type-system extension (`Type.h`, `Type.cpp`) — pure data, no
   semantics. Lands first because everything else depends on it.
2. Sema (`Resolver.cpp`, `TypeInference.cpp`) — `fi(...)` recognized
   as a builtin, returns a `Fixed` array. `-emit-sema` shows the
   spec.
3. Runtime helpers (`matlab_runtime.{h,c}`) — saturation + rounding +
   disp, with their own focused unit test (compile-and-link, no
   matlabc involvement).
4. MLIR dialect ops (`matlab.fi.const`, `matlab.fi.cast`).
5. The `LowerFixedPoint` pass.
6. Pipeline insertion in `main.cpp`.
7. EmitC / EmitPython / EmitTypeScript pretty-printing.
8. Phase-1 tests in `test/Run/`.
9. Docs touch-up + README example.

Each step is independently mergeable to `feat/fixed-point` and can
be reviewed in isolation.

### 16.10 Total surface area (Phase 1)

- **2 headers modified** (`Type.h`, `Passes.h`)
- **6 implementation files modified** (`Type.cpp`, `Resolver.cpp`,
  `TypeInference.cpp`, `MIR/Lowering.cpp`, `MLIR/Lowering.cpp`,
  `MLIR/Dialect/MatlabDialect.cpp`)
- **1 new MLIR pass** (`LowerFixedPoint.cpp`)
- **3 emit passes touched** (`EmitC.cpp`, `EmitPython.cpp`,
  `EmitTypeScript.cpp`)
- **2 driver insertions** (`tools/matlabc/main.cpp`)
- **3 runtime files modified** (`matlab_runtime.{h,c,py,ts}`)
- **~10 new tests** + **2 examples**
- **5 docs edits**

Estimate: **~2.5 weeks of focused work** for Phase 1, dominated by
`LowerFixedPoint` and the test corpus.

Phase 3 adds:
- typed integer matrix runtime (~3 days)
- persistent runtime extension (~2 days)
- fi array indexing + slicing in lowering (~3 days)
- the FIR filter test passing end-to-end (~2 days)

Estimate: **~2 weeks** on top of Phase 1.
