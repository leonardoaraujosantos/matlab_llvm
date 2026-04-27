# SystemVerilog Emission for Hardware Inference — Plan

This document scopes a future backend that lowers a constrained MATLAB
subset into **synthesizable SystemVerilog**.

The key requirement is not just emission. The tool must also decide
whether the MATLAB source is **hardware-inferable** at all:
- if the source is combinational, emit combinational RTL
- if the source implies registers, counters, or FSM behavior, emit
  sequential RTL
- if the source cannot be mapped to predictable hardware, reject it with
  a source-level diagnostic

This is a legality-first design. Silent fallback is not acceptable.

## Target: ASIC, Not FPGA

Output is **vendor-neutral synthesizable SystemVerilog** intended for
standard-cell synthesis (Synopsys Design Compiler, Cadence Genus,
Yosys). Specifically:

- No FPGA primitives — no `RAMB36E1`, `DSP48`, `xpm_*`, distributed-RAM
  styles, LUT-shape pragmas.
- No vendor attributes — no `(* ram_style = "block" *)`,
  `(* use_dsp = "yes" *)`, `(* keep *)` quirks tied to one toolchain.
- No FPGA platform glue — no AXI/AXI-Lite wrappers, IP-core packaging,
  block-design TCL, board files.
- Generic inferable RTL only. Memories are emitted as the canonical
  `always_ff` synchronous-read pattern; the synthesis tool maps it to
  a memory-compiler instance, register file, or flop array based on
  its own area/timing model.
- ASIC reset convention by default: **async-assert / sync-deassert**,
  active-low `rst_n` (typical Synopsys flow). A flag selects sync
  reset for teams that prefer it.
- Clock-gating left to the synth tool. We emit explicit `if (en)`
  enables inside `always_ff`; the tool inserts integrated clock-gating
  cells (ICGs) when its library has them. We do not hand-instantiate
  any standard-cell ICG.

If FPGA targeting becomes a goal later, that's a separate flag and a
separate output style — not a parameter of this backend.

## Goals

- Generate synthesizable SystemVerilog, not simulation-only Verilog.
- Support two primary hardware classes:
  - combinational datapaths
  - sequential logic: registers, counters, and FSMs
- Detect unsupported MATLAB constructs early and explain why they are
  not hardware-inferable.
- Keep the generated RTL structurally obvious enough for downstream
  synthesis, lint, and review.

## Non-Goals

- Full MATLAB compatibility
- Dynamic allocation or runtime-based execution
- Best-effort translation of arbitrary scripts
- Preserving MATLAB execution semantics when those semantics do not map
  cleanly to hardware

## Target User Model

The user writes MATLAB that behaves like an RTL algorithm:
- fixed-size inputs and outputs
- statically known types and widths
- explicit state updates where sequential behavior is intended
- bounded loops or explicit next-state logic

If the source instead looks like a software program, the tool rejects it.

## Known Limitations

These are the inherent constraints any MATLAB-to-RTL flow inherits from
the static, parallel nature of synthesizable hardware. Anything that
depends on dynamic memory, recursion, or runtime software objects
cannot be synthesized. The categories below mirror the limits
documented for MathWorks' HDL Coder and apply equally to this backend.

### Data Types And Variables

- **Strings and text.** Hardware has no notion of `char` or `string`.
  `fprintf`, `disp`, and any text manipulation inside HDL code are
  rejected.
- **Variable-size arrays.** In MATLAB an array can grow (e.g.
  `A = [A, x]`). In RTL every signal and array must have a fixed,
  compile-time-known shape.
- **`double` / `single`.** Native floating-point is not synthesized
  efficiently — translating it produces a large, inefficient float
  pipeline. The supported path is fixed-point (`fi`); native float is
  only available behind an explicit policy flag.
- **Recursion.** A function cannot call itself. Hardware needs a fully
  defined logic graph.

### Functions And Libraries

- **Visualization.** No `plot`, `imshow`, `grid`, or any GUI call.
- **High-level toolboxes.** Most Deep Learning and Image Processing
  Toolbox entry points are not supported. Only functions explicitly
  marked for "C/C++ Code Generation Support" or HDL are eligible.
- **System calls.** `input()`, `pause()`, `eval()`, `load`, and `save`
  cannot exist in silicon.

### Control Flow

- **Unbounded `while` loops.** A `while` loop is only legal when the
  compiler can prove (or the user states) a maximum trip count, so the
  logic can be unrolled or mapped to an FSM.
- **`try` / `catch`.** Software exception handling has no RTL
  equivalent.
- **Objects and classes.** OOP support is narrow. Dynamic instantiation
  and complex polymorphism are not supported.

### Memory And Pointers

- **Dynamic allocation.** No `malloc` or runtime object creation. All
  storage (registers and RAM) must be pre-allocated through
  `persistent` variables or dedicated memory blocks.
- **Pointers.** No raw memory pointers — there is no equivalent of a
  C/C++ pointer.

### Complex Arithmetic

- **Division.** Fixed-point `/` is supported but produces heavy
  hardware. Prefer powers of two (shifts) over variable divisors.
- **Non-linear math.** `sin`, `cos`, `log`, `sqrt` do not map to simple
  gates. Implementations rely on CORDIC or lookup tables, which
  consume significant chip area.

### Quick Self-Check

MathWorks' HDL Coder exposes `checkPotentialHDLCode('your_file.m')` for
exactly this purpose — it enumerates the lines that violate hardware
rules. Our `HWLegalize` pass should expose an equivalent diagnostic
mode so users can validate a function against this subset before
invoking emission.

## Hardware Optimizations and Capabilities

Beyond direct combinational/sequential mapping, the backend should
target the higher-leverage transformations that make hand-written RTL
impractical. The capabilities below mirror the headline features of
HDL Coder and define the value proposition of this path versus a thin
"MATLAB to Verilog" syntactic translator.

### 1. Loop Serialization (Area vs. Throughput)

A `for` loop over a compile-time-bounded range has two legal lowerings:
- **Unrolled** — every iteration becomes parallel hardware (one
  multiplier per iteration). High throughput, high area.
- **Serialized** — a single shared datapath runs the iterations across
  cycles. Lower area, longer latency.

#### Controlling the Lowering

Two mechanisms select between the strategies, mirroring HDL Coder's
pragma + project-option split:

- **In-source pragma.** `coder.unroll` (or our equivalent attribute)
  inside the function forces full unroll regardless of the default
  policy. A `coder.serialize` (or `streaming_factor`) attribute does
  the inverse.
- **Compiler option.** A streaming factor passed to the backend
  selects how many physical datapath copies to instantiate. A
  streaming factor of `1` over a loop of trip count `N` produces a
  single shared multiplier driven for `N` cycles; a factor of `N`
  produces full unroll. Intermediate factors produce partial unrolls.

```matlab
function y = process_vector(vec_in)
    %#codegen
    y = fi(zeros(1, 4), 1, 16, 8);

    % Force full unroll: 4 parallel multipliers, single-cycle result.
    coder.unroll;
    for i = 1:4
        y(i) = vec_in(i) * fi(1.5, 1, 16, 8);
    end
end
```

Omitting `coder.unroll` and passing a streaming factor of `1` would
instead emit one multiplier plus a small index counter that drives the
loop across four cycles, with the iteration value muxed into the right
output slot.

#### Tradeoffs

| Strategy   | Hardware                                       | Latency  | Area                |
| ---------- | ---------------------------------------------- | -------- | ------------------- |
| Unroll     | N independent datapaths                        | 1 cycle  | High (N × resource) |
| Serialize  | 1 shared datapath + index counter + output mux | N cycles | Low (1 × resource)  |

#### Default Policy

- Small loops (trip count below a configured threshold, e.g. 8)
  unroll by default — the area cost is acceptable and parallel
  evaluation is the obvious shape.
- Large loops (e.g. 1024-iteration vector ops) serialize by default,
  and their backing storage is mapped to RAM rather than a register
  bank. Forcing unroll on a 1024-wide loop produces an unrealistic
  resource footprint, so the compiler should emit a warning when an
  explicit `coder.unroll` is honored beyond the threshold.

### 2. RAM Inference

Large `persistent` arrays must not consume thousands of flip-flops.
The backend should detect single-read/single-write port patterns with
clocked indexing and map the storage to a synchronous RAM primitive
instead of a register bank.

```matlab
function data_out = ram_inference(addr, data_in, we)
    %#codegen
    persistent ram_block;
    if isempty(ram_block)
        ram_block = fi(zeros(1, 1024), 1, 16, 0);
    end

    if we
        ram_block(addr) = data_in;
    end
    data_out = ram_block(addr);
end
```

Size thresholds and supported access shapes belong in
`docs/hardware_subset.md` once that file exists.

### 3. CORDIC for Trigonometric and Hyperbolic Functions

`sin`, `cos`, `atan2`, `sqrt`, and similar non-linear primitives
should be lowered to CORDIC engines (shifts + adds) rather than huge
lookup tables or synthesized float pipelines. `HWLegalize` should
accept these calls only when the active numeric policy authorizes the
substitution.

```matlab
function [s, c] = hardware_sine_cosine(theta)
    %#codegen
    s = sin(theta);
    c = cos(theta);
end
```

### 4. Adaptive Pipelining

When a combinational expression's logic depth exceeds the target clock
period, the backend should automatically insert pipeline registers to
break the critical path while preserving functional equivalence across
cycles. The user supplies a target frequency; the compiler picks the
register boundaries.

```matlab
function result = complex_math(a, b, c, d)
    %#codegen
    val1 = a * b + c;
    val2 = d * b - a;
    result = val1 * val2;
end
```

### 5. Multi-Rate Hardware via Clock Enables

Different parts of an algorithm can run at different effective rates
under a single master clock by emitting clock-enable signals. This
keeps timing closure simpler than multi-clock designs while still
exposing rate adaptation in the generated RTL.

### Additional Capabilities

- **Float-to-fixed conversion.** Profile a `double` reference
  implementation against representative inputs and propose `fi`
  parameters with sufficient dynamic range and precision.
- **Resource report.** Pre-synthesis estimate of multipliers, RAM
  macros, and flip-flops the generated module will consume, so the
  cost is visible before invoking downstream synthesis.
- **Co-simulation testbench.** Auto-generate a SystemVerilog testbench
  that drives the emitted module with the same vectors as the MATLAB
  reference and asserts bit-exact equality.

## Documentation Set To Write

This feature needs a small doc set instead of one oversized design note.

### 1. Overview Doc

File:
- `docs/emit_systemverilog.md`

Purpose:
- explain what the backend does
- define the synthesizable MATLAB subset
- explain the split between combinational and sequential generation
- define what gets rejected

### 2. Synthesizable MATLAB Subset

File:
- `docs/hardware_subset.md`

Purpose:
- list supported data types
- list supported operators
- list supported statements
- define legal loop forms
- define legal function-call patterns
- define legal state-holding patterns

This should become the source of truth for "is this MATLAB hardware-like
enough to synthesize?"

### 3. RTL Inference Rules

File:
- `docs/rtl_inference.md`

Purpose:
- explain exactly how MATLAB maps to RTL structures
- separate combinational, register, counter, and FSM inference
- show small side-by-side examples: MATLAB, inferred hardware class, and
  emitted SystemVerilog shape

### 4. Rejection And Diagnostics

File:
- `docs/hardware_legality.md`

Purpose:
- document why code is rejected
- define each legality rule
- show example diagnostics
- explain what the user should rewrite

### 5. Pragmas / User Annotations

File:
- `docs/hardware_annotations.md`

Purpose:
- document optional MATLAB comments or attributes such as:
  - clock / reset intent
  - signedness / bit width
  - RAM / ROM hints
  - FSM encoding hints
  - unroll / pipeline hints

This should remain optional. The default path must still reject unclear
or ambiguous code rather than guessing.

## Backend Shape

The backend should be framed around three stages after the existing
frontend and lowering pipeline:

1. Hardware legality analysis
2. RTL classification and inference
3. SystemVerilog emission

High-level pipeline:

```text
MATLAB
  -> parse / sema / shape inference
  -> hardware legality pass
  -> RTL inference pass
  -> SV emission
```

Recommended new passes:
- `HWLegalize`
- `HWBitWidthInfer`
- `HWStateInfer`
- `HWFSMExtract`
- `EmitSystemVerilog`

## Hardware Classes To Document

The docs should center on four inference classes.

### 1. Combinational Logic

Definition:
- outputs depend only on current inputs
- no retained state across cycles
- no feedback through registers

Typical MATLAB shape:

```matlab
function y = addmul(a, b, c)
    y = a + b * c;
end
```

Expected RTL shape:
- `always_comb`
- continuous expressions
- no state register

Docs should define:
- legal operators
- legal temporary variables
- if/else as mux trees
- restrictions on loops for pure combinational expansion

### 2. Registers And Counters

Definition:
- value persists across cycles
- next value depends on current value and inputs

Typical MATLAB shape:

```matlab
function count = step(en, rst)
    persistent c;
    if isempty(c)
        c = uint8(0);
    end
    if rst
        c = uint8(0);
    elseif en
        c = c + uint8(1);
    end
    count = c;
end
```

Expected RTL shape:
- `always_ff @(posedge clk)`
- register declaration
- reset branch
- enable branch
- counter increment

Docs should define:
- how persistent state maps to registers
- reset semantics
- width rules and overflow semantics
- when a persistent variable becomes a simple register vs. a counter vs.
  a RAM

### 3. FSMs

Definition:
- state variable selects control behavior across cycles
- next-state and output logic are separable

Typical MATLAB shape:

```matlab
function [done, y] = controller(start, x)
    persistent state acc;
    if isempty(state)
        state = uint8(0);
        acc = uint8(0);
    end

    done = false;
    switch state
        case 0
            if start
                acc = x;
                state = uint8(1);
            end
        case 1
            acc = acc - uint8(1);
            if acc == 0
                done = true;
                state = uint8(0);
            end
    end
    y = acc;
end
```

Expected RTL shape:
- state enum or localparam encoding
- `always_ff` for state and registered data
- `always_comb` for next-state and outputs

Docs should define:
- accepted FSM coding styles
- required explicit state variables
- how `switch`/`if` map to next-state logic
- one-process vs. two-process emission policy
- how unreachable or ambiguous transitions are diagnosed

### 4. Static Datapaths With Bounded Loops

Definition:
- loops with compile-time known bounds
- no data-dependent trip counts

Typical MATLAB shape:

```matlab
function y = dot4(a, b)
    acc = int16(0);
    for i = 1:4
        acc = acc + a(i) * b(i);
    end
    y = acc;
end
```

Expected RTL shape:
- unrolled combinational datapath, or
- staged sequential datapath if explicitly annotated

Docs should define:
- default unroll behavior
- threshold for rejecting excessive expansion
- when a loop may become an FSM instead of unrolled logic

## Legality Rules

The tool needs a first-class section in the docs for rejection rules.
These rules should also directly drive diagnostics in the compiler.

### Hard Rejects

Reject these by default:
- dynamic array growth
- variable-size arrays
- recursion
- anonymous functions and function handles
- `eval`, `feval`, dynamic dispatch
- file I/O, console I/O, strings in datapath logic
- heap-like runtime constructs
- floating behavior without a supported hardware numeric policy
- data-dependent `while` loops without an explicit sequential/FSM form
- unsupported `persistent` initialization patterns
- non-constant indexing into unsupported storage shapes
- side effects across function boundaries that do not map cleanly to RTL

### Restricted Constructs

These are only legal under narrow conditions:
- `for` loops: bounds must be compile-time constants
- `while` loops: only if rewritten into an approved FSM form, or proven
  bounded and classifiable
- `switch`: case values must be compile-time constants
- matrices: fixed shape only
- persistent arrays: fixed shape only, with explicit mapping policy
- division, sqrt, trig: only if a supported hardware operator policy is
  documented

### Diagnostic Quality Requirement

Each rejection should answer:
- what construct was rejected
- why it is not hardware-inferable
- whether the issue is combinational-only, sequential-only, or fully
  unsupported
- what rewrite the user should consider

Example diagnostic style:

```text
error: non-synthesizable MATLAB while-loop
note: loop trip count depends on runtime data `x`
note: hardware generation requires either a constant bound or an explicit state-machine form
help: rewrite using an explicit persistent state variable and switch-based next-state logic
```

## Inference Rules The Docs Must Nail Down

The docs should avoid hand-wavy "the compiler figures it out" language.
They need exact inference rules.

### Combinational Inference

Infer `always_comb` only when:
- all assignments are acyclic within the evaluation step
- no persistent or retained state is read or written
- all loop expansions are static

### Register Inference

Infer registers when:
- a `persistent` variable or explicit state variable survives across calls
- next value depends on current value or inputs
- initialization and reset semantics are well-defined

### Counter Inference

Infer a counter when:
- a register updates by a constant increment or decrement
- optional enable/reset branches are structurally recognizable
- width is known

### FSM Inference

Infer an FSM when:
- there is an explicit persistent state variable
- state transitions are encoded with `switch` or a canonical `if` tree
- all next-state writes are unambiguous within a cycle

Reject when:
- state is implicit in control flow only
- transitions depend on hidden side effects
- multiple state variables behave as loosely coupled controllers without a
  documented mapping rule

## Numeric Policy

The docs must define a hardware numeric story early, otherwise legality
checks will be inconsistent.

Recommended policy:
- prefer explicit integer and fixed-point types
- allow `logical`
- allow floating-point only behind an explicit policy flag and supported
  operator list

The docs should specify:
- width inference rules
- signedness inference
- overflow behavior
- cast semantics
- fixed-point annotation format

## SystemVerilog Emission Policy

The docs should make the emitted RTL style predictable.

Recommended defaults:
- `always_comb` for combinational blocks
- `always_ff @(posedge clk)` for sequential state
- `logic` for internal nets/registers
- `typedef enum logic [...]` for FSM states
- separate next-state and state registers for FSMs
- optional `unique case` when legality checks prove exclusivity

This keeps synthesis intent obvious and lint-friendly.

## Implementation Plan

This section is the concrete plan: files to add, passes to write, tests
to land, in the order they need to land. It mirrors the structure of
[`docs/emit_systemc.md`](emit_systemc.md), which has shipped through a
similar phased build-out.

### Status snapshot — where we are vs the original plan

The original Phase 1 → 5 build-out is essentially **done**. Every
phase has shipped at least a v1; many ship more than the original
plan called for (the FSM v2 work, the comment / port-name /
inlining work in 5.6 weren't in the initial outline). What remains
is a small set of **deferred features** with explicit triggers,
plus some upstream-pipeline lifts that block three of the eight
`examples/hdl/` modules.

| Phase | Status | What ships today |
|---|---|---|
| 1 — combinational MVP, scalars only | ✅ shipped | scalar arith, if/else, ports + always_comb, HWLegalize gate, HWBitWidthInfer, Verilator lint lane |
| 2 — fixed-size vectors + bounded `for` | ✅ shipped | unrolled `for i = 1:N`, packed-array slot decls, GEP-based load/store |
| 3 — registers + `persistent` | ✅ shipped | `HWStateInfer`, two-process `always_ff` + `always_comb`, async-low / sync-high / sync-low reset (`-sv-reset=...`), counter / register classification |
| 4 v1 — FSM emission via Phase-3 cascade | ✅ shipped | persistent state + switch + case → `unique case` cascade, both Mealy and Moore styles lint clean |
| 4 v2 — FSM polish | ✅ shipped | `typedef enum` state types, FSM-aware enum literals at compares + `_next` writes, ambiguity diagnostics (duplicate / empty case), encoding flag (`-sv-fsm-encoding=binary|one_hot|gray`), `% hdl:` pragma scanner with per-FSM `fsm_encoding(...)` override |
| 4 RAM inference | 🟡 deferred | depends on persistent fi-array recognition; the SV emitter side is a small extension once the upstream pipeline lifts that |
| 4.5.1 multi-store slot retyping | ✅ shipped | `RefineSlotTypes` pass — `[data_out, overflow] = ...` style functions emit cleanly |
| 4.5.2 `if <fi>` truthy | ✅ shipped | `RefineIfConds` rewrites `unrealized_conversion_cast` + retyped slot loads to `cmpi ne, 0` |
| 4.5.3 `true` / `false` literals | ✅ shipped | Resolver special-cases them; lower to `arith.constant 0/1 : i1` |
| 4.5.4 static fi arrays | ✅ shipped | `LowerStaticFiArrays` rewrites `fi(zeros(1, N), ...)` to `llvm.alloca` + GEP + load/store |
| 4.5.5 vector function arguments | 🟡 deferred | scalar-args + local-array workarounds shipped (`vector_proc3.m`, `vector_proc_local.m`); end-to-end vector args is a multi-week pipeline lift across Sema + MIR + user-call refinement + LLVM tensor lowering |
| 5.1 fi Saturate semantics | ✅ shipped | `LowerFiSaturate` — explicit clamp circuit (cmpi + select), narrow-width peel optimization, `OverflowAction='Saturate'` is the default |
| 5.2 v1 port pipelining | ✅ shipped | `% hdl: input_pipeline(N)` + `output_pipeline(N)`, dedicated always_ff shifts the chain, increases function output latency by N |
| 5.2 v2 adaptive distributed pipelining | 🟡 deferred | trigger: a real DSP user asks for `-sv-target-freq=N`-driven retiming. v1 covers the common "stages between combinational stages" need |
| 5.3 v1 large-loop unroll warning | ✅ shipped | trip count > 64 emits a warning suggesting `% hdl: loopspec('stream')`; unrolls anyway |
| 5.3 v2 actual streaming | 🟡 deferred | trigger: a user hits the warning and needs the area win. v2 builds an iteration-counter FSM + shared body instance using the Phase-4 cascade machinery |
| 5.4 constant-multiplier optimization | ✅ shipped | `ConstMulCSD` — ×0/×1/×-1/×2^k/×(2^k±1) rewrites; full Booth/CSD recoding deferred to v2 |
| 5.5 hardware report | ✅ shipped | `-emit-hardware-report` walks the post-pipeline IR; per-func operator counts, register widths, FSM state counts, encoding |
| 5.6.1 `% hdl: port(...)` pragma | ✅ shipped | function-only `.m` files emit SV without a separate driver |
| 5.6.2a function-result name preservation | ✅ shipped | `[data_out, overflow] = f(...)` emits `output ... data_out, output ... overflow` instead of `y, y1` |
| 5.6.2b leading-comment forwarding | ✅ shipped | side-channel `SourceManager` scan in the emitter; comment lines outside the function body's range are dropped |
| 5.6.3 SSA-temp inlining + slot-output collapse | ✅ shipped | `vN_1` scratch signals gone; pure single-use ops inline at use site; same-named slot + output port share one signal |
| 5.6.4 trailing same-line comment forwarding | ✅ shipped | `case 0 % Soma` and `y = a + b; % sum` now both forward as `// ...` lines |

### Vector-DSP roadmap (closing the last 3 examples/hdl/ modules)

The three remaining unblocked modules — `vector_processor`,
`fir_asic_pipelined`, `sequential_processor` — together exercise
8 distinct pipeline gaps. Sequenced for incremental wins: each
stage either closes one example outright or delivers a reusable
piece the next stage builds on.

| Stage | Items | Effort | Closes |
|---|---|--:|---|
| A | (split — see notes below) | — | — |
| A.1 | fi-spec propagation across function-call boundaries (precondition for clean Saturate-cast on the function-arg path) | ~3 days | enables `fi(arg, ...)` casts to clamp correctly, used by fir/seq's final stage |
| A.2 | Constant-index reads on vector args | ~1 day | small extension once Stage B lands; no value standalone |
| B | Vector function arguments (Sema + MIR + user-call + LLVM tensor lowering) | ~4 days | `vector_processor` ✅ |
| C | Static array literal init (`fi([0.1, 0.2, ...], ...)` → alloca + per-element stores) | ~2 days | coefficient-table half of fir / seq |
| D | Loop-iv array indexing (`for i = 1:N; arr(i) ...; end`) | ~3 days | for-loop bodies in fir / seq |
| E | Vector concat with static shapes (`[x, delay(1:end-1)]`) | ~3 days | shift-register pattern in fir / seq |
| F | Persistent fi-arrays + whole-vector assign (`persistent` + `acc(:) = ...`) | ~6 days | `fir_asic_pipelined` ✅, `sequential_processor` ✅ |

Total: ~4–5 weeks of focused work, **one stage per
implementation session** (the existing Phase 5.6.x cadence). The
shift-register-with-persistent-fi-array idiom in fir / seq depends
on three orthogonal lifts (literal init, loop-iv indexing, vector
concat) all landing before stage F can complete.

#### Stage A re-scoped (was "verification")

Original plan called Stage A "saturate-cast verification" treating
it as a half-day check. Closer inspection shows the verification
actually fails on the path used by fir / seq:

  ```matlab
  function y = step(x)              % x is i32 fi(_, 32, 29)
      y = fi(x, 1, 16, 12, 'OverflowAction', 'Saturate');
  end
  ```

`Lowering` emits this fi-cast with `callee = matlab_fi_quantize_*`
(the constructor form) regardless of input type.
`LowerFixedPoint::rewriteFiCast` only takes the constructor path
when the input is `f64` / `f32`; integer inputs need the clamp
path with `fi_clamp` attr **and** `fi_lhs_*` attrs naming the
source spec. Neither is set today, so the cast survives to the
SV emitter as `unsupported op`.

The proper fix needs **fi-spec propagation across function calls**:
the call-site fi spec (signed/WL/FL) attaches to the function arg
as `matlab.fi_arg_*` attrs; `LowerFixedPoint` reads those to
populate the missing `fi_lhs_*` on the cast; the existing clamp
path (Phase 5.1) handles the rewrite.

Without this, `fi(<fi_value>, ...)` casts simply fail in the SV
pipeline whenever the input crosses a function boundary — which
is exactly the FIR / sequential_processor final stage. So Stage
A.1 is genuinely a multi-day effort, not a verification.

### Out-of-scope items still on the plan

These items are deferred — concrete, but not yet attempted:

- **Persistent fi-arrays** (`persistent delay_line; if isempty
  delay_line = fi(zeros(1, N), ...);`) — needs
  `matlab_persistent_set_ptr` / `_get_ptr` recognition, plus
  RAM-or-register-bank inference. Blocker for
  `fir_asic_pipelined.m`, `sequential_processor.m`.
- **Loop-iv array indexing** (`for i = 1:N; acc += arr(i); end`)
  — Phase 4.5.4 v1 only handles constant indices; loop-iv
  indices need the iv to lower to integer (currently f64,
  rejected by HWLegalize as datapath).
- **Vector concat / slice** (`[x, delay(1:end-1)]`,
  `delay(1:end-1)`) — runtime-call shapes today; static cases
  could fold to GEP-based shuffles.
- **Array literal init** (`h = [0.1, 0.2, 0.3, 0.4]`) — currently
  goes through `matlab_mat_from_buf`. Static cases could fold
  to per-element stores into an alloca, similar to 4.5.4.
- **Constant-index reads on vector function args** —
  Phase 4.5.4 only handles locals; reading `vec_a(1)` from a
  vector-typed function arg needs a small extension once vector
  args themselves land (Stage B → A).
- **Whole-vector colon-assignment** (`acc(:) = expr`) — when
  `acc` is scalar this is just a store; when it's a vector this
  is broadcast / element-wise assign. Recognized once loop-iv
  indexing lands.
- **2-D fi matrices** + persistent fi matrices.
- **N-dim arrays** beyond what 4.5.4 produces.
- **`sin` / `cos` / `sqrt`** — CORDIC lowering.
- **Float fallbacks** + policy flags.

### Closure assessment

For the SV-backend feature itself: **closed** for the scalar +
fixed-vector + state + FSM corpus, with first-class fi support,
optimization passes, and human-readable output. Five of the
eight `examples/hdl/` modules (`alu_16bit`, `mux_4to_1_16bit`,
`counter_0_to_10`, `mealy_fsm`, `moore_fsm`) emit clean
lint-passing SV with source comments + source-level identifier
names preserved.

The three remaining modules (`vector_processor`,
`fir_asic_pipelined`, `sequential_processor`) need the
vector-DSP roadmap above (Stages A–F, ~4 weeks) — each stage's
lift is concrete and sized in days; the dependency between
literal init / loop-iv indexing / vector concat is what makes
fir + seq a multi-stage closeout rather than a single-shot
feature.

### Original phase detail (kept for reference)

The full per-phase plan below is preserved as the original
target shape. Each "shipped" phase still describes the full
v1+v2 contract including the pieces that landed.

### Why this differs from `-emit-systemc`

`-emit-systemc` produces input for a downstream HLS tool, which owns
loop scheduling, FSM extraction, and pipelining. `-emit-systemverilog`
goes **direct to RTL** — there is no HLS tool in the loop, so this
backend owns those decisions itself. Concretely, this means three
extra MLIR passes (`HWStateInfer`, `HWFSMExtract`, `HWPipeline`) that
have no counterpart in the SystemC path, plus tighter rejection on
anything that would otherwise need an HLS tool to clean up.

The two backends share the legality and bit-width-inference passes
conceptually but keep separate implementations — the SV path is
stricter (e.g. it rejects `scf.while` outright unless it has an
explicit FSM annotation, where the SystemC path can punt to the HLS
tool).

### Architecture

```text
AST ──► MLIR ──► [existing pipeline ... LowerIO]
                       │
                       ├──► emitC()        ──► .c           (existing)
                       ├──► emitSystemC()  ──► .cpp + .h    (planned)
                       │
                       └──► [HWLegalize]      ───► reject or tag
                            [HWBitWidthInfer] ───► annotate values with !sv.type
                            [HWStateInfer]    ───► classify persistent vars: reg / counter / RAM
                            [HWFSMExtract]    ───► persistent state + switch → hw.fsm
                            [HWPipeline]      ───► retiming when target frequency given
                                 │
                                 └──► emitSystemVerilog() ──► .sv
```

`HWLegalize` and `HWBitWidthInfer` are conceptually similar to the
SystemC versions but separate implementations, since the rules and
the type lattice (`logic [W-1:0]` / `logic signed [W-1:0]`) differ.

### Synthesizability gate

**The emitter detects, up front, whether the MATLAB source can be
synthesized at all and rejects it with a source-level diagnostic if
it cannot.** This is a first-class deliverable, not a side effect of
emission. The "Legality Rules" section above defines the rules; this
section defines how they are surfaced and enforced.

**Driver mode `-check-synthesizable`.** Runs the full frontend +
lowering + `HWLegalize` and stops. Produces no `.sv` output — its job
is to answer "would emission succeed?" and list every offending line.
Mirrors MathWorks' `checkPotentialHDLCode('your_file.m')`. Exit code
is 0 on clean, non-zero on any rejection. CI lanes consuming this
mode can gate merges on hardware-readiness without depending on the
SV emission lane.

**`-emit-systemverilog` itself runs the same gate.** Emission never
silently produces non-synthesizable RTL — if `HWLegalize` reports any
error, the emitter aborts with the same diagnostic and writes
nothing. There is no `--force` or fallback path. Silent emission of
broken RTL is the failure mode this whole backend exists to prevent.

**Detection categories** (each maps to a checker in `HWLegalize`):

| Category | Examples |
|---|---|
| **Runtime calls** | any `matlab_*` symbol survives lowering — `disp`, `fprintf`, `parfor`, `eval`, file I/O |
| **Dynamic shape** | `llvm.alloca` of non-constant size, growth assignment `A = [A, x]`, variable-size matrices |
| **Recursion** | call-graph cycle — direct or indirect |
| **Indirect calls** | function handles, anonymous functions, `feval` |
| **Floating-point without policy** | bare `f64` / `f32` arithmetic, `sin`/`cos`/`exp`/`sqrt` without an `fi`-or-LUT lowering enabled |
| **Unbounded control flow** | `scf.while` without a constant trip-count proof or an explicit FSM annotation; `try`/`catch` |
| **Unsupported types** | `string`, `cell`, struct fields in datapath, sparse matrices, N-D (>3D) arrays |
| **Unsynthesizable persistent shapes** | `persistent` arrays accessed with non-scalar indices, missing `isempty(...)` initializer, multi-driven across function-call boundaries |
| **Inferred-latch hazards** | `always_comb` whose conditional branches do not fully assign every output on every path |
| **Cross-call side effects** | global writes, persistent var read after a call that may write to it without a mapping rule |

**Detection coverage grows by phase.** The synthesizability gate is
not "all rules at once"; it expands as features land:

| Phase | Rules added to `HWLegalize` |
|---|---|
| 1 | Runtime calls, dynamic shape, recursion, indirect calls, FP-without-policy, *all* `scf.while`, strings/cells/structs, inferred-latch hazards |
| 2 | `scf.for` with non-constant bounds (was already error-out via missing constant fold; now explicit), array shapes that don't constant-fold |
| 3 | `persistent` without `isempty` init, persistent writes outside `always_ff`-mappable patterns, multi-clock hints (out of scope for v1) |
| 4 | FSM ambiguity (multiple unconditional next-state writes in one arm, missing `otherwise`, unreachable states), RAM-pattern violations (slice access, whole-array copy, data-dependent index *shape*) |
| 4.5 | Multi-store slots that disagree on type, `if <non-i1, non-fi>`, `true`/`false` used in non-bool contexts, fi-array access patterns outside the canonical scalar read/write shape, vector args with non-constant length |
| 5 | Retiming barriers crossed (saturating `fi` ops moved across pipeline registers), unsupported `fi` overflow/rounding combos, `coder.hdl.*` annotation conflicts |

**Diagnostic format.** Every rejection answers four things, mirroring
the "Diagnostic Quality Requirement" section above:

```text
error: <one-line summary of the offending construct>
  --> path/to/file.m:LINE:COL
note: <why it cannot be synthesized>
note: <whether the issue is combinational-only / sequential-only / fully unsupported>
help: <concrete rewrite the user should consider, with a doc link>
```

Each rule lives in its own checker file under
`lib/MLIR/Passes/HWLegalize/`, named after the rule, with a
`.stderr` golden test under `test/EmitSVFail/<rule>/` so regressions
in the *diagnostic* are caught the same way as regressions in the
*emitter*.

**Coverage report.** `-check-synthesizable -hardware-report` (Phase 5)
emits a Markdown summary of which rules fired, how many times, and
on which lines — the same shape as the fixed-point report. This is
what teams hand to a reviewer along with a diff to demonstrate "this
function is hardware-ready".

### Phase 1 — Combinational MVP, scalars only (~1 week)

**Goal.** A `foo.m` of pure scalar arithmetic with `if`/`else` compiles
to one `module` with `input`/`output` ports and a single `always_comb`
block. No loops, no arrays, no state.

**Step 1.1** — CLI flag.
`tools/matlabc/main.cpp`: `Mode::EmitSystemVerilog`, `-emit-systemverilog`
parser, dispatch branch.
`include/matlab/MLIR/Passes/Passes.h`: declare
`emitSystemVerilog(ModuleOp, raw_ostream&)`.

**Step 1.2** — `HWLegalize` (Phase-1 scope).
`lib/MLIR/Passes/HWLegalize.cpp`. Walk post-`LowerIO` module and
diagnose via `mlir::emitError(loc)`:
- any `llvm.call` to a `matlab_*` runtime symbol
- recursion (call-graph cycle)
- `llvm.alloca` of non-constant size
- any `scf.while` (Phase 1 rejects all `while`; later phases relax)
- string/cell/struct globals
- `f64` values without an explicit fixed-point or integer policy

**Step 1.3** — `HWBitWidthInfer`.
`lib/MLIR/Passes/HWBitWidthInfer.cpp`. Attaches a discardable
`sv.type` attribute to every SSA value:
- `i1` → `logic`
- `i8/16/32/64` (unsigned) → `logic [W-1:0]`
- `i8/16/32/64` (signed)   → `logic signed [W-1:0]`
- fixed-point (Phase 5 surface) → `logic signed [W-1:0]` plus a
  binary-point attribute for downstream operator widening
- `f64` rejected unless the user opts into a soft-float operator
  policy (out-of-scope for v1)

**Step 1.4** — Emitter.
`lib/MLIR/Passes/EmitSystemVerilog.cpp`. Each top-level `func.func`
becomes a SystemVerilog `module`:
- one `input` port per argument, one `output` per return value
- a single `always_comb` block holds the lowered body
- `arith.constant` → SV literal with explicit width
  (`16'd5`, `16'sd-3`)
- `arith.addi/subi/muli/and/or/xor/shl/shr` → matching SV operators
- `scf.if` with results → ternary `cond ? a : b`
- multi-line `if/else` with side effects → blocking-assignment `=` in
  `always_comb` writing to a temporary `logic`
- temporaries declared at the top of the module via `logic [W-1:0]`

**Step 1.5** — Tests.
`test/EmitSV/combinational/*.m` — scalar programs. Goldens are the
emitted `.sv`. A smoke check runs **Verilator** in lint-only mode:

```bash
verilator --lint-only -Wall foo.sv
```

Make Verilator optional at configure time (`-DMATLAB_LLVM_VERILATOR=ON`)
so CI without Verilator still passes the golden-diff lane.

`test/EmitSVFail/*.m` — legality rejections with `.stderr` goldens,
mirroring the existing `test/EmitCFail/` contract.

### Phase 2 — Fixed-size vectors, statically bounded `for` (~1 week)

**Goal.** MATLAB `for i = 1:N` with constant `N` and writes into a
fixed-shape vector compiles to **unrolled** combinational logic, all
inside a single `always_comb`.

**Step 2.1** — Reuse `HWLegalize` to enforce:
- every `scf.for` has constant bounds (otherwise reject)
- every `llvm.alloca` for an array has a constant shape

**Step 2.2** — `HWUnroll` (or extend an existing pass).
For `scf.for` with constant trip count ≤ unroll threshold (default
64, configurable), fully unroll in place. The threshold is policy:
above it, the pass emits a warning and unrolls anyway in Phase 2;
serialization is Phase 4.

**Step 2.3** — Emitter updates.
- `llvm.alloca` of `!llvm.array<N x iW>` → declare a packed array:
  `logic [W-1:0] arr [N];` (or unpacked, configurable)
- `llvm.store` / `llvm.load` → `arr[i] = v;` / `v = arr[i];`
- unrolled iterations emit a sequence of statements; the SV synth
  tool collapses identical structure cheaply

**Tests.** `test/EmitSV/vectors/*.m` — dot products, fixed-tap FIR
combinational form, small constant-coefficient transforms. Goldens
plus Verilator lint.

### Phase 3 — Registers, counters, and `persistent` (~1 week)

**Goal.** `persistent` variables compile to clocked registers. The
canonical HDL Coder counter shape works:

```matlab
function count = step(en, rst)
    persistent c;
    if isempty(c)
        c = uint8(0);
    end
    if rst
        c = uint8(0);
    elseif en
        c = c + uint8(1);
    end
    count = c;
end
```

**Step 3.1** — `HWStateInfer`.
`lib/MLIR/Passes/HWStateInfer.cpp`. Recognizes the
`persistent + isempty(init) + conditional update` pattern in MLIR and
classifies each persistent variable as one of:
- **register** — arbitrary next-state expression
- **counter** — constant ±1 update guarded by enable, optional reset
- **RAM candidate** — array with single-element scalar accesses (Phase 4)

The pass attaches an `hw.role = #hw.role<reg|counter|ram>` attr and a
reset-init value attr. It also adds an implicit clock + reset port to
each `func.func` that holds at least one stateful var.

**Step 3.2** — Module-port synthesis.
The emitter adds `input logic clk` and either `input logic rst_n`
(default ASIC: async-assert, sync-deassert) or `input logic rst`
(sync-reset mode) to any module containing state. Configured via
`-sv-reset=async-low|sync-high|sync-low` (default `async-low`).

**Step 3.3** — Emitter: `always_ff`.
Emit one `always_ff @(posedge clk or negedge rst_n)` block per
state-bearing function, with the reset branch first:

```systemverilog
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)         c <= 8'd0;
    else if (rst)       c <= 8'd0;
    else if (en)        c <= c + 8'd1;
end
```

Counter recognition isn't strictly necessary for correctness — a plain
register lowering produces identical RTL — but tagging it improves
diagnostics ("inferred 8-bit up-counter") and unlocks later width
analysis.

**Tests.** `test/EmitSV/registers/*.m` — counter, accumulator,
shift register, edge-detector. Goldens plus Verilator lint plus an
optional simulation lane (`just test-sv-sim`) that drives a
testbench against the MATLAB reference.

### Phase 4 — FSMs and RAM inference (~2 weeks; v1 ships only FSMs)

**Phase 4 v1 (shipped) scope.** FSM emission is much smaller than
this section originally projected — the **Phase 3 two-process pattern
already handles FSMs end-to-end** when the user expresses state
transitions as a `switch (state)` (or nested `if (state == c)`
cascade) on a `persistent` integer state variable. Concretely, v1
adds **one missing op handler** to the SV emitter — `arith.cmpf` for
state-equality checks lowered through the persistent-get f64 ABI —
and lets Phase 3's existing always_comb / always_ff machinery do the
rest. Both Mealy and Moore styles emit clean lint-passing SV today
(`test/EmitSV/fsm_2state.m`, `fsm_moore3.m`).

**Phase 4 v2 (shipped):**
- `typedef enum logic [W-1:0] { S0, S1, ... } reg_t;` declared
  at module scope (W = ⌈log2(N)⌉ for binary encoding) and
  deduped per persistent register.
- The state register declared as `reg_t reg, reg_next;` instead
  of raw `logic [W-1:0]`.
- Cascades of `scf.if cmpf(oeq, get(reg), const)` (or the
  `matlab.eq` mixed-type equivalent) render as `unique case
  (reg) S0: ... default: ... endcase` inside `always_comb`.
  Inner cascade `scf.if`s and their cmpf/eq operands are
  suppressed; their then-regions become case arm bodies; the
  deepest else-region (or the implicit hold-by-default) becomes
  the `default:` arm.
- `state_next = <const>` writes render as `state_next = S<n>`
  enum literals; the `always_ff` reset uses the same enum
  literal for the reset state (`state <= S0` instead of `state
  <= 8'sd0`).
- Non-cascade comparisons of an FSM register against a case
  constant (e.g. the Moore-style `if state == S2` output decode
  before the second cascade collapses) render the constant as
  the matching enum literal too — Verilator otherwise warns on
  width mismatch between the small enum and the wider integer
  literal.
- Multiple cascades on the same register (state-transition
  switch + output decode) share one typedef; both render as
  `unique case`. Output ports that return the FSM register get
  an explicit width cast (`y1 = 8'(state)`) so the assignment
  to a non-enum port stays width-clean.

**Phase 4 v2.3 (shipped) — FSM ambiguity diagnostics**:
- **Duplicate case label** — two `case <c>` arms with the same
  constant in the same switch. The second arm is unreachable and
  `unique case` would also be malformed. Hard error.
- **Empty case arm** — a recognized cascade arm whose body is
  empty. Almost always an oversight (state stuck without explicit
  transitions / outputs). Hard error.
- (Skipped) **State written but never matched** — false-positive
  prone in Moore-style designs whose output-decode cascade
  intentionally covers only a subset of states and routes the
  rest through `default:`.
- (Skipped) **Multiple unconditional writes per arm** — MATLAB
  execution semantics legitimately allows this (later write
  wins); too high false-positive rate to flag.

Both diagnostics fire under `-emit-systemverilog` and
`-check-synthesizable`. The check-mode invocation runs the SV
emitter dry (output discarded) so FSM-time issues surface
alongside the HWLegalize gate.

**Phase 4 v2.5 (shipped) — state-encoding flag**:
- `-sv-fsm-encoding=binary` (default) — sequential ints, width
  ⌈log2(N)⌉. Smallest register; synth tools re-encode anyway.
- `-sv-fsm-encoding=one-hot` (alias `one_hot`) — one bit per
  state, width N. Fastest decode, largest register. Common for
  high-frequency control paths.
- `-sv-fsm-encoding=gray` — reflected-binary gray code, width
  ⌈log2(N)⌉. Single-bit transitions between adjacent states;
  useful for CDC sync FIFOs and similar metastability-sensitive
  paths.

The chosen encoding shows up only in the typedef enum's
explicit per-state values (`S0 = 3'd1, S1 = 3'd2, S2 = 3'd4` for
3-state one-hot etc.); everything else (case statement, register
declaration, reset assignment) is unchanged. The
`emit-sv-tests` lane verifies all 5 FSM fixtures lint clean
under all 3 encodings (`fsm-encoding sweep` line in the
runner's summary).

**Phase 4 v2.6 (shipped) — `% hdl:` pragma scanner**:

A small generic pass `runScanHWPragmas(M, SM)` walks every
user `func.func`, finds the source line range its body ops
cover (plus one line above to catch pragmas immediately
before the function), and scans those lines for `% hdl:
<directive>(<args>)` comments. Each recognized directive
attaches as a discardable string attribute on the
function: `hdl.<directive> = "<arg>"`.

```matlab
function out = controller(...)
    % hdl: fsm_encoding('one_hot')
    persistent state;
    ...
end
```

The first directive shipped:

- **`fsm_encoding('binary' | 'one_hot' | 'gray')`** — overrides
  the CLI-wide `-sv-fsm-encoding` flag for this function's
  FSMs only. Lets a single design mix encodings (e.g. a
  control FSM kept binary for area and a fast-path FSM
  marked one-hot for decode latency). Saved + restored
  around each function so siblings can use different
  encodings.

The pragma infrastructure is intentionally generic — Phase 5
will reuse it for `pipeline`, `loopspec`, `ram`, etc.
without re-doing the comment-scanning plumbing. Unknown
directives are silently ignored (forward-compatibility),
malformed pragmas (no `(...)`, mismatched quotes) emit a
warning.

**Phase 4 v2 deferred to a future round:**
- Hierarchical FSM extraction (an `hw.fsm` op carrying explicit
  state table + per-state IR blocks) — not needed for direct
  emission but useful for downstream passes that operate on
  FSMs (state minimization, formal extraction, alternate
  emission targets). Pure forward-investment with no current
  consumer; revisit when a concrete pass needs explicit FSM
  ops.

**RAM inference deferred to Phase 4.5.** The persistent-array storage
pattern depends on `LowerStaticFiArrays` from Phase 4.5.4, which
turns `fi(zeros(1, N), ...)` into an `llvm.alloca [N x iW]`. Once
that lands, RAM-inference is a small extension of the Phase 3
`HWStateInfer` pass (recognize `persistent + array + scalar
read/write` and tag with `hw.role<ram>`) plus the canonical
synchronous-RAM pattern in the SV emitter.

The original Phase 4 design content below is kept verbatim as the
target shape for the full subset.

**Goal.** Persistent state variable + `switch` compiles to a
two-process FSM. Persistent arrays with single-element scalar access
patterns compile to inferable synchronous RAM.

**Step 4.1** — `HWFSMExtract`.
`lib/MLIR/Passes/HWFSMExtract.cpp`. Recognizes:
- a persistent state variable initialized to a constant in the
  `isempty` branch
- a `switch` on that variable with a `case` arm per state and a
  catch-all `otherwise`
- next-state writes inside each arm

Rewrites the construct into an `hw.fsm` op carrying the state enum,
per-state next-state IR, and per-state output IR. This op is what the
emitter pattern-matches on.

Diagnostics for ambiguous transitions (multiple unconditional writes
to the same state inside one arm), missing `otherwise`, or unreachable
states.

**Step 4.2** — Emitter: two-process FSM.
```systemverilog
typedef enum logic [1:0] { S0, S1, S2, S3 } state_t;
state_t state, next_state;

always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) state <= S0;
    else        state <= next_state;

always_comb begin
    next_state = state;
    z = '0;
    unique case (state)
        S0: begin /* outputs + transitions */ end
        ...
        default: next_state = S0;
    endcase
end
```

`unique case` is emitted only when `HWFSMExtract` proves arm
exclusivity; otherwise plain `case`.

State encoding default is binary; `% hdl: fsm_encoding('one_hot' | 'gray')`
or a CLI flag overrides.

**Step 4.3** — RAM inference in `HWStateInfer` (full).
Extend the Phase-3 pass to recognize the HDL Coder RAM pattern:
- persistent array of fixed shape, initialized once via `zeros(...)`
- every read access is `arr(scalar_index)`
- every write access is `arr(scalar_index) = scalar_value`
- index expression is data-dependent in the *runtime* sense but
  structurally a single load/store — no slicing, no whole-array copy

Tag matching arrays with `hw.role = #hw.role<ram>` and an interface
descriptor (single-port read/write, 1-cycle read latency).

**Step 4.4** — Emitter: synchronous-read RAM (vendor-neutral).
Emit the canonical pattern that every standard-cell synth tool
recognizes:

```systemverilog
logic [W-1:0] mem [DEPTH];
logic [W-1:0] mem_q;

always_ff @(posedge clk) begin
    if (we) mem[addr] <= din;
    mem_q <= mem[addr];
end

assign dout = mem_q;
```

We do **not** emit any vendor `(* ram_style = ... *)` attribute. The
synthesis tool picks register-file vs. memory-compiler instance based
on its own thresholds. For ASIC flows that need a specific memory
compiler instance, the user wraps our output and replaces the array
manually.

**Tests.** `test/EmitSV/fsm/*.m`, `test/EmitSV/ram/*.m` — Mealy and
Moore controllers, single-port RAM, register file. Verilator lint
plus simulation-equivalence lane.

### Phase 4.5 — Pipeline Hardening for HDL examples (~1.5 weeks)

**Goal.** Unblock the `examples/hdl/` corpus that today's Phase 1-3
backend cannot accept due to limitations in the *existing* lowering
pipeline (not the SV emitter itself). Each item below is an
orthogonal pipeline issue surfaced by a real example program; each
also fails identically under `-emit-c` today, so fixes here help
every backend, not just SV.

These are intentionally not Phase 4: FSM extraction and RAM
inference are SV-emitter work, while the items here are
front-end / type-flow / scalar-promotion plumbing.

**Trigger examples** (drawn from `examples/hdl/`):
- `alu_16bit.m` — multi-store scalar slot retyping + `false` literal
- `counter_0_to_10.m` — `if <fi-truthy>` conversion
- `vector_processor.m` — vector function args + `fi(vec, T)` cast
- `fir_asic_pipelined.m`, `sequential_processor.m` —
  `fi(zeros(1, N), ...)` static fi arrays

#### 4.5.1 Multi-store scalar slot retyping

**Problem.** A `matlab.alloc` (slot) initialized once with a typed
store and then over-written in multiple branches keeps its
`none`-typed result, even when every store agrees on a concrete
scalar integer / fi type. Today `LowerScalarSlots` only promotes
slots whose type is already concrete, so the `data_out` slot in
`alu_16bit` stays `none` and downstream typing breaks.

**Fix.** Extend the slot-type inference inside `LowerUserCalls` (or
add a small post-pass) so a `matlab.alloc` whose every store has
the same concrete scalar type adopts that type. The slot then
participates in normal scalar arith lowering.

**Test:** `alu_16bit.m` with a typed driver should emit clean SV
with a single signed `data_out` register-style local plus the
case-tree.

#### 4.5.2 `if <fi-typed value>` truthy lowering

**Problem.** MATLAB's `if x` (where `x` is a fi-typed integer) is
the natural truthiness pattern. Today the lowering produces
`scf.if` with a non-`i1` condition operand, which fails MLIR
verification:

```
'scf.if' op operand #0 must be 1-bit signless integer, but got 'none'
```

**Fix.** When Sema sees an `IfStmt` whose condition is an integer
or fi value, emit an implicit `cond != 0` comparison so the
resulting `scf.if` consumes a clean `i1`. Alternatively, do the
fix-up in `LowerScalarsToArith` (rewrite `scf.if` whose condition
is a non-`i1` integer to `arith.cmpi ne, cond, 0`).

**Test:** `counter_0_to_10.m` (which writes `if reset` /
`if count_val >= 10`) should compile without manual
`> fi(0, ...)` rewrites.

#### 4.5.3 `false` / `true` as bool literals (not handles)

**Problem.** `overflow = false` produces a `matlab.make_handle`
with `callee = "false"` because `true` and `false` are registered
as builtin function names, not constants. The slot ends up
holding a function handle, not a bool, and the function's
`overflow` output stays `none`-typed.

**Fix.** Special-case `true` / `false` in Sema (or in MIR
lowering) to emit `arith.constant 1 : i1` / `0 : i1` instead of
`make_handle`. Update Resolver to mark them as constants rather
than builtins.

**Test:** `alu_16bit.m`'s second output `overflow` should emit as
a 1-bit SV port driven from a clean comparison (after 4.5.4 lands
the multi-store slot retyping covers `data_out`).

#### 4.5.4 Static fi-array lowering

**Problem.** `fi(zeros(1, N), S, W, F)` lowers to a runtime call
chain (`matlab_mat_i64_zeros` → `matlab_persistent_set_ptr` →
`matlab_mat_i64_subscript1_s`). Synthesis can't accept any of
those — they're heap allocations and dynamic dispatches.

**Fix.** Add a `LowerStaticFiArrays` pass that runs before
`LowerTensorOps` and recognizes the canonical pattern:

```matlab
arr = fi(zeros(1, N), S, W, F);    % constant N, constant fi spec
... arr(i) ...                      % constant or for-loop iv index
arr(i) = ...
```

For Phase 4.5, the supported access shapes mirror the HDL Coder
RAM-inference rules (see Phase 4 §4.3): single-element scalar
read or write, no slicing or whole-array copy. The pass rewrites
the storage to an `llvm.alloca [N x iW]` (with the fi `S/F` spec
threaded via a discardable `sv.fi_spec` attr for Phase 5
arithmetic) and the access calls to `llvm.getelementptr` +
`llvm.load` / `llvm.store`. The SV emitter then renders the array
as `logic [W-1:0] arr [N];` with index expressions, identical to
the Phase 2 array shape.

**Test:** `vector_processor.m` (constant-index reads of
`vec_a(1)` / `vec_a(2)` / `vec_a(3)`) — pure unrolled
combinational. `fir_asic_pipelined.m` and
`sequential_processor.m` exercise the same path inside a
`for i = 1:4` body, also unrolled by Phase 2's loop handling.

#### 4.5.5 Vector function arguments

**Problem.** `function y = f(vec_a)` with `vec_a` typed as a
3-element fi vector at the call site today refines to a
`!llvm.ptr` argument going through `matlab_fi_quantize_s` —
runtime cast, not synthesizable. Worse, the user-call refinement
*collapses* the call-site `!llvm.ptr` to the function arg's
inferred scalar element type, so the body's `vec_a(1)` becomes a
malformed `matlab.subscript(scalar_i16, 1.0)`.

**Status (v1 not shipped).** Supporting vector args end-to-end
needs Sema + MIR-to-MLIR lowering + user-call refinement + LLVM
tensor-op lowering all to track vector function-argument shapes
through the pipeline. Multi-week effort that touches multiple
non-SV-specific subsystems. Out of scope for the current Phase
4.5 round.

**Recommended workarounds today** — both lint clean and emit
correct SV; documented as fixtures in `test/EmitSV/`:

  - `vector_proc3.m` — pass elements individually:
    `function [...] = f(a1, a2, a3, b1, b2, b3)`. The script
    side "unrolls" the vector into N scalars per argument.
    Module emits with N input ports per vector.
  - `vector_proc_local.m` — same scalar-args boundary, but the
    body builds a local `fi(zeros(1, N), ...)` array (Phase
    4.5.4) from the scalars and uses array-style access
    inside. Keeps the array shape readable in the source while
    the function signature stays scalar.

**Future fix sketch.** Extend the user-call refinement so a
vector arg with constant shape (matched by `fi(vec, T)` at the
call site) lowers to a fixed-size `!llvm.array<N x iW>`
parameter passed by value, mirroring the static-array storage
in 4.5.4. The SV emitter renders these as input port arrays
(`input logic [W-1:0] vec_a [N]`). Test target:
`vector_processor(fi([1 2 3], T), fi([4 5 6], T))` emits with
two i16 input arrays of length 3 and two i32 scalar outputs.

#### 4.5.6 What stays out of Phase 4.5

- **Vector function arguments** (4.5.5 above) — multi-week
  upstream pipeline work; scalar-args + local-array workarounds
  ship as fixtures.
- **Persistent fi-arrays** (`persistent delay_line; if isempty
  delay_line = fi(zeros(1, N), ...);`) — needs
  `matlab_persistent_set_ptr` / `_get_ptr` recognition, plus
  RAM-or-register-bank inference per the Phase 4 RAM-inference
  goal that itself depends on 4.5.4.
- **Loop-iv array indexing** (`for i = 1:N; acc = acc + arr(i);
  end`) — Phase 4.5.4 v1 only handles constant indices. Adding
  loop-iv indices needs the iv to lower to an integer (today
  it's f64 and Phase 2 explicitly rejected iv-as-datapath uses).
- **Vector concat / slice** (`[x, delay(1:end-1)]`,
  `delay(1:end-1)`) — `matlab_mat_i64_concat_row` and
  `matlab_mat_i64_slice1` runtime calls. Static-shape variants
  could fold to GEP-based shuffles.
- **Array literal init** (`h = [0.1, 0.2, 0.3, 0.4]`) — currently
  goes through `matlab_mat_from_buf`. Static cases could fold to
  per-element stores into an alloca, similar to 4.5.4's zero-init.
- **2-D fi matrices** and persistent fi matrices — the existing
  feature_status doc lists this as a separate gap; the SV path
  inherits whatever the upstream pipeline supports.
- **N-dim arrays beyond what Phase 4.5.4 produces.**
- `sin` / `cos` / `sqrt` etc. (CORDIC lowering) — Phase 5.
- Float fallbacks and policy flags — Phase 5.

The three remaining `examples/hdl/` modules (`vector_processor`,
`fir_asic_pipelined`, `sequential_processor`) all hit
combinations of these. They're realistic FIR designs that
exercise persistent fi-arrays + loop-iv array indexing + vector
concat — the full set is a follow-up phase, not Phase 4.5.

#### Driving the examples/hdl/ corpus

Function-only `.m` files (the MATLAB convention — one function
per file) carry no caller, so the user-call refinement pass has
no typed call site to fix port widths. Each synthesizable
example in `examples/hdl/` therefore ships with a small
`<name>_synth.m` driver alongside it that performs a single
typed call, e.g.:

```matlab
% examples/hdl/alu_16bit_synth.m
T = numerictype(1, 16, 0);
S = numerictype(0, 8, 0);
[d, o] = alu_16bit(fi(5, T), fi(3, T), fi(2, S));
disp(d);
```

Compile via the multi-file recipe:

```sh
just emit-sv-multi examples/hdl/alu_16bit_synth.m \
                   examples/hdl/alu_16bit.m
```

Today five examples ship with drivers — `alu_16bit`,
`mux_4to_1_16bit`, `counter_0_to_10`, `mealy_fsm`, `moore_fsm`.
The remaining three (`vector_processor`, `fir_asic_pipelined`,
`sequential_processor`) need vector-port + persistent-fi-array
shift register support that's a separate follow-up phase. A
future `% hdl: port(...)` pragma (planned next phase) will let
function-only files declare their port types inline, removing
the need for a separate driver file.

#### Effort budget

| Item | Effort | Affects beyond SV |
|---|--:|---|
| 4.5.1 multi-store slot retyping | ~2 days | yes — emit-c also benefits |
| 4.5.2 if-fi truthy | ~1 day | yes |
| 4.5.3 true/false constants | ~0.5 day | yes |
| 4.5.4 static fi arrays | ~1 week | yes — biggest win |
| 4.5.5 vector args | ~3 days | yes |
| **Total** | **~1.5 weeks** | |

Phase 4.5 is intentionally orthogonal to Phase 4. The two can
ship in either order; pick by which corpus matters more — FSMs
(Phase 4) or vector / array DSP (Phase 4.5).

### Phase 5 — Fixed-point, optimizations, reports (~2 weeks)

**Goal.** First-class `fi` support in the SV path, plus the
optimization knobs that make the output competitive with hand RTL.

**Step 5.1** — Fixed-point Saturate semantics. **Shipped as
Phase 5.1 v1.**

Replaces every runtime-call `matlab_fi_sat_s64(val, W)` (and
`_u64`) in the post-pipeline IR with an explicit clamp circuit
built from `arith.cmpi` + `arith.select`:

  signed:    out = (val > MAX) ? MAX : (val < MIN ? MIN : val)
              MAX =  2^(W-1) - 1,    MIN = -2^(W-1)
  unsigned:  out = (val > MAX) ? MAX : val
              MAX =  2^W - 1

Renders in SV as the matching ternary chain (`v6_1 = v5_1 >
32'sd65535;  v8_1 = v7_1 ? -32'sd65536 : v5_1;  v9_1 = v6_1 ?
32'sd65535 : v8_1;`), which synthesizes to a comparator + 2-way
mux per bound — small, well-understood by every standard-cell
synth tool.

Replaces the earlier passthrough DCE in `LowerStaticFiArrays`
which was correct only for Wrap-mode fi (the trunci downstream
produced the same value as the saturate for non-overflowing
inputs). Saturate-mode programs now get correct semantics on
overflow.

**Width-narrowing peel.** When the saturate's input is
`arith.extsi narrow → wide` and the saturate target ≤ narrow's
width, the clamp emits at the narrow width (with a single
extsi back to wide for downstream consumers) instead of the
wide intermediate. The downstream `extsi/trunci` collapse
folds the round-trip away, so the SV ends up with no wide
unused-bit signals — Verilator stays clean.

Implementation: `lib/MLIR/Passes/LowerFiSaturate.cpp`. Runs in
the SV pipeline immediately after the user-call iteration loop
and `LowerStaticFiArrays`, before `ConstMulCSD` and the
`HWLegalize` gate.

**Reduce-mode optimization** for fi paths that only need Wrap
semantics (where the truncation downstream gives equivalent
results on overflow): not yet exposed. Today the explicit clamp
runs unconditionally; a future `% hdl: fi_overflow('wrap')`
pragma could opt back into the cheaper passthrough form.

**Step 5.2** — Pipelining. **Phase 5.2 v1 (port pipelining)
shipped.**

Two pragmas, scanned by Phase 4 v2.6's `% hdl:` infrastructure:

  - `% hdl: input_pipeline(N)` — adds N flop stages on every
    input port. The body's references go through the last
    stage (`<arg>_dN`) instead of the raw port; an always_ff
    shifts the chain on every clock.
  - `% hdl: output_pipeline(N)` — adds N flop stages between
    the body's combinational output and the actual port. The
    always_comb writes to `<port>_d0`; the always_ff shifts
    `_d0 → _d1 → … → _dN`; an `assign port = <port>_dN`
    drives the port.

Either pragma adds `clk` + `rst_n` to the module's port list
(with the chosen reset polarity / synchronicity from
`-sv-reset=...`). Function output latency increases by the
sum of input + output pipeline depths. Mirrors HDL Coder's
"Input/Output Pipelining = N" project option.

Adaptive distributed pipelining (cell-cost-driven register
insertion across combinational depth, driven by
`-sv-target-freq=N`) is a v2 follow-up. The retiming-pass
shape is sketched out in the original Phase 5.2 design but
not yet implemented — port pipelining covers the common
"add N stages between combinational stages" need that real
DSP designs hit first.

**Step 5.3** — Loop serialization. **Phase 5.3 v1 (warning
only) shipped.**

For-loops with trip count > 64 (configurable threshold) emit
an informational warning suggesting the user mark them with
`% hdl: loopspec('stream')` for serialization. The actual
streaming transformation (sequential body + iteration
counter FSM, shared datapath instance) is a v2 follow-up:

  - `coder.hdl.loopspec('stream', 1)` → one shared body
    instance, iteration counter, output mux. Lower area,
    higher latency.
  - factor `N` (= trip count) → full unroll (Phase 2's
    default).
  - intermediate factors → partial unroll.

Today the synth tool unrolls every constant-bound loop fully
(it sees the `for (int i = ...; ...; ++i) begin ... end`
inside `always_comb` and replicates the body N times). For
N=1024 that's an unrealistic resource footprint. The
warning surfaces this before the user discovers it from
their synth tool's report. Threshold knob is reserved
(`-sv-unroll-threshold=N`) but the default is hard-coded at
64 today.

Serialized bodies will become small FSMs reusing the
Phase 4 v2 cascade machinery; v2 follow-up work.

**Step 5.4** — Constant-multiplier optimization. **Shipped as
Phase 5.4 v1.**

Recognizes `arith.muli %x, %c` (or `%c, %x`) where `%c` is a
compile-time constant and rewrites to a shift-add tree using
the most-common coefficient patterns:

  ×0          → 0           (folded)
  ×1          → x           (passthrough)
  ×-1         → 0 - x
  ×2^k        → x << k
  ×-(2^k)     → 0 - (x << k)
  ×(2^k - 1)  → (x << k) - x      (×3, ×7, ×15, ×31, ...)
  ×(2^k + 1)  → (x << k) + x      (×5, ×9, ×17, ×33, ...)

Other constants stay as ordinary `muli`. Full Booth/CSD
recoding for arbitrary coefficients is a v2 follow-up; v1
captures the patterns that account for most DSP coefficients
users actually write.

CLI: `-sv-const-mul=off|auto|csd` (default `auto` = on for the
SV pipeline only; the C/Python/TS backends still emit `*`
directly to match user-side semantics there).

Implementation: `lib/MLIR/Passes/ConstMulCSD.cpp`. Runs after
the user-call iteration loop and the static-fi-array lowering,
before the verifier check. The hardware report (`-emit-
hardware-report`) reflects the rewrite — patterns that get
collapsed appear as shift / add / sub counts instead of mul.

**Step 5.5** — Reports. **Shipped as Phase 5.5 v1.**

`-emit-hardware-report` driver flag (also accepts
`-emit-hw-report`), parallel to the existing
`-emit-fixed-point-report`. Emits a Markdown summary per
user function: inferred class (combinational / clocked /
FSM-bearing), input/output port widths, operator counts
(add / sub / mul / div / cmp / shift / bitop / mux),
register count + total flip-flop bits, FSM state counts +
chosen encoding (per the active CLI flag or per-FSM
pragma).

The estimate is **pre-synthesis**; absolute gate counts
come from the user's downstream synthesis tool. The
report's purpose is visibility before synthesis: see the
cost shape of each module, diff resource changes across
PRs, and catch unintended widenings before they hit the
synth tool's timing report.

Implementation: `lib/MLIR/Passes/EmitHardwareReport.cpp`.
Runs the same SV pipeline as `-emit-systemverilog` up to
and including `HWLegalize`, then walks the post-pipeline
module collecting counts. The walker tallies arith /
matlab dialect operators by kind, uses
`gatherHWPersistentState` for register info, and matches
the FSM cascade pattern to count state-equality scf.if's
per persistent register.

RAM inference is still deferred (Phase 4 RAM-inference
follow-up depends on Phase 4.5.4 static-fi-array
infrastructure for the persistent-array case). Today the
report always shows "RAM: none".

Justfile recipe: `just report-hw FILE.m`.

**Tests.** `test/EmitSV/fi/*.m` — fixed-point FIR, IIR, CIC.
`test/EmitSV/pipelined/*.m` — long-path datapath under various
target frequencies. `test/EmitSV/reports/*.m` — golden hardware
reports.

### Phase 5.6 — Port-type pragmas + source-name preservation (~3 days). **Shipped.**

**Goal.** Let function-only `.m` files emit synthesizable SV
*without* a separate typed-driver file, and carry source-level
identifiers all the way to the generated module so the output is
human-readable for downstream review.

All three sub-items shipped in v1:
  - 5.6.1 — `% hdl: port(<name>, <kind>, ...)` pragma scanner +
    `ApplyPortTypePragmas` pass that retypes the function
    signature before the user-call refinement loop.
  - 5.6.2a — `matlab.name` result attr propagated through
    `Lowering` and read by `EmitSystemVerilog::emitPortList`.
  - 5.6.2b — source-comment forwarding implemented as a
    side-channel in the SV emitter (no lex/AST changes): the
    emitter tracks the last-emitted source line per function and
    scans `SourceManager::getLineText` for `%` comment-only lines
    between the previous op and the next, emitting them as `//`
    inside the `always_comb` body. Scoped to the function's body
    line range so script-driver and file-header prose are
    excluded. Trailing same-line comments (`x = 1; % bar`) need
    a real lexer change and are still deferred.

Phase 5.6.3 — readability follow-up (also shipped). The SV emitter
now inlines pure single-use ops at their use site instead of
materializing a `vN_1 = ...` scratch signal per SSA value. Covers
arith ops (`addi/subi/muli/and/or/xor/shl/shr*/cmpi/cmpf/select/
ext/trunc`), the matlab-dialect binops that survive lowering
(`matlab.add/sub/matmul/emul/eq/ne/lt/le/gt/ge`), and `llvm.load`
of a slot. A `stripOuterParens` helper drops redundant parens in
unambiguous RHS contexts (`if (...)`, `... = expr;`, yield assigns)
so the emitted SV reads as ordinary expressions rather than
double-wrapped calls. Slot-output collapse: when an `llvm.alloca`'s
`matlab.name` matches an output port's name AND every load of it
flows only into the func.return that drives that port, the slot
and the port share one signal — body stores write `<port>` directly,
the `<port> = <port>` self-assign at return is suppressed, no
scratch `<name>_1` declaration. Net effect on the
`examples/hdl/alu_16bit` module: ~95 lines → ~50 lines, no
auxiliary signals, all source comments + names preserved.

Two independent gaps, shipped together because they share
infrastructure (the lexer / `ScanHWPragmas` pass and the
emitter's name-resolution path).

#### 5.6.1 — `% hdl: port(...)` port-type pragma

**Problem.** Today the user-call refinement pass needs a typed
caller (e.g. `y = f(fi(0,1,16,0))`) to fix the function's port
widths. Without one, args default to `f64` and the
synthesizability gate rejects them. The `examples/hdl/*_synth.m`
driver convention works but adds a parallel file per module.

**Fix.** Extend the existing `% hdl: <directive>(<arg>)` scanner
(`lib/MLIR/Passes/ScanHWPragmas.cpp`, currently parses
`fsm_encoding`, `input_pipeline`, `output_pipeline`) to also
recognize:

```matlab
function y = mux_4to1_16bit(in0, in1, in2, in3, sel)
    %#codegen
    % hdl: port(in0, fi, signed, 16, 0)
    % hdl: port(in1, fi, signed, 16, 0)
    % hdl: port(in2, fi, signed, 16, 0)
    % hdl: port(in3, fi, signed, 16, 0)
    % hdl: port(sel, uint, 8)
    ...
```

Pragma forms:
  - `port(<arg>, fi, signed|unsigned, <W>, <F>)` — full fi spec
  - `port(<arg>, uint, <W>)` / `port(<arg>, int, <W>)` — plain
    integer
  - `port(<arg>, bool)` — i1
  - `port(<return>, ...)` — same syntax for returns by name

Lowering: `ScanHWPragmas` attaches a synthetic `matlab.porttype`
attribute on the func. A new pre-pipeline pass
`ApplyPortTypePragmas` rewrites the function signature to the
declared types, then re-runs the existing `RefineFuncSigs` to
propagate. Bare `just emit-sv examples/hdl/mux_4to_1_16bit.m`
then works standalone — no driver, no multi-file invocation.

When both a typed caller AND a `port(...)` pragma exist, they
must agree; mismatch is a hard error.

#### 5.6.2 — Source-name preservation

**Problem.** Today's SV output preserves arg names, top-level
local names, persistent register names, and FSM enum labels
(`S0`, `S1`, ...) via the `matlab.name` attr propagated through
the slot pipeline. But it drops two categories:

1. **Function return-variable names** — `[data_out, overflow] =
   alu_16bit(...)` emits ports `output logic ... y, output logic
   y1` instead of `output logic ... data_out, output logic
   overflow`. `EmitSystemVerilog.cpp:emitPortList` hardcodes `y`,
   `y1`, ... for results.
2. **Source comments** — MATLAB-source comments
   (`% Switch case to select the operation`, `% Reset and count
   logic`) never reach MLIR; the lexer drops them as trivia.

**Fix 5.6.2a — Return-variable names** (~0.5 day):
  - Resolver / lowering: when constructing the func, set
    `matlab.result_name` on each result via `setResultAttr` from
    the source-level return-variable identifier (the names in
    `[data_out, overflow] = ...`).
  - `EmitSystemVerilog.cpp:emitPortList` (line 426): read the
    attr, fall back to the existing `y`/`y1` scheme when absent
    or when the name collides with a reserved port (`clk`, `rst`,
    args). The arg-name path already does the
    `Used.insert + suffix` collision dance; reuse it.
  - Mirror in `gatherHWPersistentState` / hardware-report so
    `report-hw` shows the user names.

**Fix 5.6.2b — Source comments** (~2 days, the bulk):
  - Lexer: keep `Comment` tokens (or attach trailing-comment
    strings to the next non-trivia token) instead of dropping
    them. The existing dump-tokens mode already roundtrips comments,
    so the data is there.
  - AST → MLIR lowering: attach attached-comment text as a
    `matlab.comment` string array attr on the producing op.
    Already done implicitly via location info for diagnostics; we
    just need a stable attribute carrier separate from `Loc`.
  - Emitter: when emitting a statement-equivalent op (e.g. the
    body of an `always_comb` for a top-level statement, the
    `case` arms of a switch), prefix `// <comment>` lines.

**Out of scope.** Inline `data_out =` rename for SSA-numbered
intermediates (`v0_1, v1_1, ...`) — those are MLIR SSA values
without source names. We *could* attach `matlab.name` from the
producing ASTNode for the *first* user-visible store, but
that's a much bigger heuristic. Defer.

#### Effort budget

| Item | Effort | Affects beyond SV |
|---|--:|---|
| 5.6.1 port-type pragma scanner | ~0.5 day | no — SV-specific surface |
| 5.6.1 ApplyPortTypePragmas pass | ~1 day | yes — emit-c also benefits from explicit port types |
| 5.6.2a return-name preservation | ~0.5 day | yes — emit-c/cpp/python/ts return-variable names also improve |
| 5.6.2b comment preservation (lexer + lowering + emit) | ~2 days | yes — every emitter benefits |
| Tests | ~0.5 day | — |

Total: ~3 days, biased toward 5.6.2b.

**Tests.** Each `examples/hdl/*.m` gets a `_pragma.m` variant
that uses `port(...)` pragmas instead of the `_synth.m` driver,
verifying both produce the same SV. Comment-preservation tests
in `test/EmitSV/comments/*.m` golden-diff a function with rich
comments against the expected SV with `// <comment>` lines.

**Migration.** Once 5.6.1 ships, the `_synth.m` drivers in
`examples/hdl/` become a redundant convention. Decide then
whether to drop them or keep both as parallel demos (the typed
driver is a more familiar MATLAB-engineer pattern, the pragma is
a more familiar HDL-engineer pattern).

### Op mapping table

#### Combinational (`always_comb`)

| MLIR | SystemVerilog |
|---|---|
| `arith.constant 5 : i16` + `sv.type = logic [15:0]` | `16'd5` |
| `arith.addi / subi / muli / divi` | `+` / `-` / `*` / `/` (rejected by default; warn) |
| `arith.andi / ori / xori / shli / shrui / shrsi` | `&` / `\|` / `^` / `<<` / `>>` / `>>>` |
| `arith.cmpi` | `==`, `<`, etc., producing 1-bit `logic` |
| `arith.select` | `cond ? a : b` |
| `arith.extsi` / `extui` / `trunci` | sign/zero-extend / slice |
| `scf.if -> T` | ternary (preferred), else temp + `if/else` in `always_comb` |
| `func.func` (leaf, stateless) | `module` with `always_comb` body |
| `llvm.alloca` of static array | `logic [W-1:0] arr [N];` |
| `llvm.load / store` (constant index) | `v = arr[i];` / `arr[i] = v;` |

#### Sequential (`always_ff`)

| MLIR | SystemVerilog |
|---|---|
| `hw.role<reg>` persistent var | `logic [W-1:0] r;` + `always_ff` register |
| `hw.role<counter>` persistent var | counter register + reset/enable arms |
| `hw.fsm` op (Phase 4) | `typedef enum`, `state_reg`, two-process FSM |
| `hw.role<ram>` persistent array | inferable sync-RAM block |
| iter-arg of pipelined `scf.for` | pipeline register stage |

#### Rejected (Phase 1+)

| MLIR | Diagnostic |
|---|---|
| any `llvm.call @matlab_*` runtime call | `error: runtime call <name> has no synthesizable form` |
| `llvm.call @matlab_disp_*` / `_fprintf_*` | `error: I/O is not synthesizable` |
| `llvm.call @matlab_parfor_*` | `error: parfor has no RTL form; use static for-loop` |
| `llvm.alloca` of dynamic size | `error: array size must be compile-time constant` |
| recursion | `error: recursion not synthesizable` |
| function handle / anonymous fn | `error: function values not synthesizable` |
| string literal in datapath | `error: string types not synthesizable` |
| `scf.while` without FSM annotation (Phase 1–3) | `error: data-dependent while-loop needs explicit FSM form` |
| `f64` value without numeric policy | `error: floating-point requires explicit fi() conversion` |

### Pragma surface

MATLAB-side annotations are recognized by the frontend as comments
above the relevant line and threaded through MLIR as `hw.pragma`
attributes. Vendor-neutral — no Vivado/Quartus directives:

```matlab
% hdl: clock(name='clk')
% hdl: reset(name='rst_n', polarity='active_low', kind='async')
% hdl: pipeline(stages=2)
% hdl: fsm_encoding('one_hot')
% hdl: ram(latency=1, ports=1)
% hdl: loopspec('stream', factor=1)
% hdl: keep   % do not retime across this expression
```

These mirror HDL Coder's `coder.hdl.*` calls but stay in comment form
so the source compiles unchanged in stock MATLAB.

### Files

| File | Purpose |
|---|---|
| `lib/MLIR/Passes/HWLegalize.cpp` | Rejects non-synthesizable IR with source-line diagnostics |
| `lib/MLIR/Passes/HWBitWidthInfer.cpp` | Annotates every value with `sv.type = logic [W-1:0]` (signed/unsigned) |
| `lib/MLIR/Passes/HWStateInfer.cpp` | Classifies persistent vars: register / counter / RAM |
| `lib/MLIR/Passes/HWFSMExtract.cpp` | `persistent state + switch` → `hw.fsm` op |
| `lib/MLIR/Passes/HWUnroll.cpp` | Loop unrolling + (Phase 5) serialization |
| `lib/MLIR/Passes/HWPipeline.cpp` | Adaptive pipelining via retiming |
| `lib/MLIR/Passes/EmitSystemVerilog.cpp` | Walker that prints `.sv`; combinational, FSM, RAM paths |
| `include/matlab/MLIR/Passes/Passes.h` | Pass declarations |
| `tools/matlabc/main.cpp` | `-emit-systemverilog` flag, `-sv-reset`, `-sv-target-freq`, `-sv-streaming-factor`, `-sv-const-mul`, `-emit-hardware-report` |
| `test/EmitSV/combinational/*.m` | Phase 1–2 tests: golden `.sv` + Verilator lint |
| `test/EmitSV/registers/*.m` | Phase 3 tests |
| `test/EmitSV/fsm/*.m` | Phase 4 tests |
| `test/EmitSV/ram/*.m` | Phase 4 tests |
| `test/EmitSV/fi/*.m` | Phase 5 fixed-point tests |
| `test/EmitSV/pipelined/*.m` | Phase 5 retiming tests |
| `test/EmitSVFail/*.m` | Legality rejection tests with `.stderr` goldens |
| `justfile` | `emit-sv FILE`, `lint-sv FILE`, `test-sv`, `test-sv-sim` recipes |

### Testing strategy

Three layers, mirroring the SystemC plan but adapted for direct-to-RTL:

1. **Golden-diff** (fast, always in CI). Each `test/EmitSV/*.m` pairs
   with a `.sv` golden. Catches emitter regressions without running
   anything.
2. **Verilator lint** (medium, optional in CI behind
   `-DMATLAB_LLVM_VERILATOR=ON`). `verilator --lint-only -Wall` on
   every emitted `.sv`. Catches syntax errors, missing widths,
   inferred-latch hazards, multi-driven nets.
3. **Simulation equivalence** (slow, opt-in). For Phase 3+ tests,
   auto-generate a SystemVerilog testbench that drives the emitted
   module with the same input vectors used by the MATLAB reference
   and asserts bit-exact equality. Run under `just test-sv-sim`,
   not by default in CI. Verilator's `--binary` mode is sufficient;
   no commercial simulator required.

Synthesis itself is **not** in our test matrix. ASIC synthesis takes
hours and requires a target standard-cell library the user owns. We
optimize for "the synthesis tool will accept this without warnings"
via the lint lane, and "this matches the MATLAB reference" via the
sim lane.

### Effort estimate

| Phase | Scope | Effort |
|---|---|--:|
| 1 | Scalar combinational + legality + bit-width infer + emitter skeleton | ~1 week |
| 2 | Static arrays + fully-unrolled `for` | ~1 week |
| 3 | `persistent` → register/counter, ASIC reset, `always_ff` emission | ~1 week |
| 4 | FSM extraction + RAM inference | ~2 weeks |
| 4.5 | Pipeline hardening: multi-store slots, `if <fi>`, `true`/`false`, static fi arrays, vector args | ~1.5 weeks |
| 5 | Fixed-point + adaptive pipelining + serialization + const-mul + report | ~2 weeks |
| **Total to a useful v1** | Phase 1+2+3 | **~3 weeks** |
| **Total to full subset** | Phase 1+2+3+4+4.5+5 | **~8.5 weeks** |

Phase 1+2+3 is already useful: pure DSP datapaths, registered
accumulators, FIR filters, simple pipelines compile and lint clean.
Phase 4 unlocks controllers and memory-backed designs. Phase 4.5
unblocks the `examples/hdl/` corpus that depends on cross-pipeline
fixes (multi-store slots, `if <fi>`, static arrays). Phase 5 brings
the optimizations that make output competitive with hand-written
RTL on a real ASIC flow.

### Open questions

1. **Reset convention default.** Async-assert/sync-deassert active-low
   `rst_n` is the dominant ASIC convention but not universal — some
   teams (especially ARM/embedded) prefer sync reset. Default
   `async-low`, surface loudly in docs, configurable via `-sv-reset`.

2. **Fixed-point overflow propagation in pipelined paths.** Our
   `fi` semantics already define overflow per operator (saturate /
   wrap). Pipeline register insertion does not change semantics, but
   *retiming across* a saturating operator is unsound. `HWPipeline`
   must treat saturating ops as retiming barriers — confirm this is
   modeled correctly before shipping.

3. **Multi-clock designs.** Out of scope for v1. Single clock domain
   only; cross-clock-domain logic requires explicit user wrapping
   today. A future `% hdl: clock(name='clk2')` annotation could enable
   it without changing the core architecture.

4. **Inferred-latch detection.** Verilator catches most cases.
   `HWLegalize` should additionally reject any `always_comb` whose
   conditional outputs aren't fully assigned on every path — the
   standard "missing else" trap. Phase 1 enforces this.

5. **State encoding default.** Binary is the safest default for
   pre-RTL — synth tools re-encode aggressively anyway. One-hot is
   useful for high-frequency designs but produces wider state regs.
   Default `binary`, override per-FSM via `% hdl: fsm_encoding(...)`.

6. **`std::sort`-style algorithms.** Some MATLAB patterns
   (sorting networks, priority queues) need explicit hardware
   primitives. Out of scope for v1 — reject with a pointer to the
   hardware-subset doc.

## Design alternatives

### Alternative A — Lower through CIRCT

CIRCT (the LLVM hardware subproject) has its own MLIR dialects (`hw`,
`comb`, `seq`, `fsm`, `sv`) and goes directly to Verilog via its
`ExportVerilog` pass.

**Pros**: structurally correct hardware IR (no ad-hoc `hw.fsm` op),
mature Verilog emitter, less code to write.
**Cons**: hard build dependency on the CIRCT distribution,
interfaces still churning, and our subset of MATLAB that survives
the conversion is small enough that a direct emitter is comparable
in code size and far more controllable.

Revisit when (a) we want to share infrastructure with other CIRCT
frontends, or (b) CIRCT's `sv` dialect stabilizes enough that the
import cost is once-only.

### Alternative B — Lower through SystemC then through HLS

Use the existing `-emit-systemc` pipeline and rely on a downstream
HLS tool to produce SV.

**Pros**: zero new code in this repo.
**Cons**: shifts vendor dependency to the user (HLS tool license),
gives up control over RTL style (each HLS tool emits different
shapes), and produces output that's harder to review than direct
RTL emission. Defeats the point of an ASIC-targeted backend.

### Alternative C — Emit Verilog-2001 instead of SystemVerilog

**Pros**: works with older synthesis tools.
**Cons**: loses `always_comb`/`always_ff`/`unique case`/`logic` —
the very constructs that make synthesis intent unambiguous. The
ASIC tool ecosystem fully supports SystemVerilog-2017; there's no
practical reason to downgrade.

## Relationship To Existing SystemC Plan

The current repository already has
[`docs/emit_systemc.md`](emit_systemc.md), which targets synthesizable
SystemC/HLS. This SystemVerilog plan should not replace it.

Recommended positioning:
- `emit_systemc.md`: HLS/SystemC-oriented path
- `emit_systemverilog.md`: direct RTL-oriented path

The legality and inference docs can be shared conceptually between both
paths, but the emission rules differ enough that they should remain
separate.

## References

This plan is consistent with the general structure described in the HDL
Coder User's Guide PDF the user provided:
- synthesizable subsets need explicit support matrices for functions,
  types, operators, and control flow
- bounded control flow matters for hardware generation
- persistent/stateful data maps naturally to hardware storage
- unsupported source must be diagnosed early instead of deferred to
  downstream synthesis tools
