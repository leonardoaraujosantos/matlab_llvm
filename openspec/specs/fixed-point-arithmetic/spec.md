# Fixed-Point Arithmetic Spec

## Purpose
Document the observed behavior of MATLAB Fixed-Point Designer (`fi`) support,
which parses, type-checks, lowers, and emits fixed-point arithmetic as portable
integer code with explicit shifts (no `double` in the datapath of `-emit-c` /
`-emit-cpp`). The `fi` type carries word length, fraction length, signedness,
overflow mode, and rounding mode through the Sema type lattice, and is the shared
numeric foundation the SystemVerilog and cocotb backends build on.
(doc: docs/emit_fixed_point.md, doc: docs/fixed_point_toolbox_roadmap.md, test: test/Run/fi_*.m)

## Requirements

### Requirement: fi type and companion objects
The system SHALL recognize `fi`, `numerictype`, `fimath`, and `fipref` as
builtins and track a per-value FixedSpec (signedness, word length 1..64, fraction
length, overflow mode, rounding mode) in the type system, folding constant
constructors at compile time. (doc: docs/emit_fixed_point.md §3, §4; test: test/Run/fi_basic.m)

#### Scenario: constructor folds to stored integer
- **WHEN** `fi(1.5, 1, 16, 8)` is evaluated with constant arguments
- **THEN** the system SHALL fold it to the stored integer 384 carrying FixedSpec{signed, WL=16, FL=8}

#### Scenario: numerictype/fimath companions
- **WHEN** `fi(value, numerictype(s,WL,FL), fimath(...))` is used
- **THEN** the system SHALL read the spec out at type-inference time and fold the constructor away

### Requirement: Integer-native lowering with explicit shifts
The system SHALL lower `fi` arithmetic to native integer ops plus explicit shifts
(no f64 detour), aligning add/sub by fraction length and shifting multiply
results back per the product mode, emitting `<stdint.h>`-only C with no
`rtwtypes.h` dependency. (doc: docs/emit_fixed_point.md §6, §7)

#### Scenario: Q8.8 multiply
- **WHEN** two Q8.8 `fi` values are multiplied
- **THEN** the system SHALL emit `(int16_t)(((int32_t)a * (int32_t)b) >> 8)`

#### Scenario: aligned add needs no shift
- **WHEN** two `fi` operands share the same fraction length
- **THEN** the system SHALL emit the addition without a shift

### Requirement: Quantization, overflow, and rounding modes
The system SHALL implement `Saturate` (default) and `Wrap` overflow, and all five
rounding modes (Floor, Nearest, Zero, Ceiling, Convergent), routing
saturation/rounding through `matlab_fi_*` runtime helpers only when the operand
specs require them. (doc: docs/emit_fixed_point.md §3.5, §6.2, §10 Phases 1/4/5)

#### Scenario: saturating cast
- **WHEN** a result can exceed the destination range under `Saturate`
- **THEN** the system SHALL wrap the cast in `matlab_fi_sat_s64` (or the unsigned/round variant) to clamp to range

#### Scenario: wrap overflow via fimath
- **WHEN** `fimath('OverflowAction','Wrap')` is selected
- **THEN** the system SHALL apply wrap semantics on the affected operations

### Requirement: fi arrays, persistent storage, and round-trip
The system SHALL support 1-D `fi` arrays with indexing, slicing, concat,
reductions (`sum`/`mean`), `persistent` storage of `fi` arrays, and round-trip
through `-emit-c`, `-emit-cpp`, `-emit-llvm`, `-emit-python`, and the REPL.
(doc: docs/emit_fixed_point.md §10 Phase 3; test: test/Run/fi_array.m, test/Run/fi_filter.m)

#### Scenario: FIR shift-register filter
- **WHEN** the FIR `fi_filter.m` gating example is compiled
- **THEN** the system SHALL emit a static integer delay-line array, MAC the taps without a shift when the accumulator FL matches, and narrow once on the final cast

#### Scenario: type-preserving (:) assignment
- **WHEN** `acc(:) = acc + a*b` is used on an `fi` scalar
- **THEN** the system SHALL cast the RHS into the LHS's spec without re-inferring (preventing unbounded bit-width growth)
