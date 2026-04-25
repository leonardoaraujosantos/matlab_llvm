# SystemVerilog Emission for Hardware Inference — Plan

This document scopes a future backend that lowers a constrained MATLAB
subset into **synthesizable SystemVerilog** for FPGA or ASIC flows.

The key requirement is not just emission. The tool must also decide
whether the MATLAB source is **hardware-inferable** at all:
- if the source is combinational, emit combinational RTL
- if the source implies registers, counters, or FSM behavior, emit
  sequential RTL
- if the source cannot be mapped to predictable hardware, reject it with
  a source-level diagnostic

This is a legality-first design. Silent fallback is not acceptable.

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

## Suggested Implementation Phases

### Phase 1: Combinational MVP

Document and implement:
- scalar combinational functions
- fixed-size vectors
- `if`/`else`
- statically bounded `for`
- no persistent state

Output:
- pure `always_comb`

### Phase 2: Registers And Counters

Document and implement:
- `persistent` -> register mapping
- reset/init rules
- enable patterns
- increment/decrement pattern recognition

Output:
- `always_ff`
- simple counters and registers

### Phase 3: FSMs

Document and implement:
- explicit state-variable coding style
- switch-based next-state logic
- state encoding rules
- legality checks for ambiguous transitions

Output:
- two-process FSM style in SystemVerilog

### Phase 4: Storage Inference

Document and implement:
- fixed-size persistent arrays
- register bank vs. RAM inference
- indexing restrictions

### Phase 5: Advanced Arithmetic

Document and implement:
- fixed-point arithmetic policies
- optional floating-point operator subset
- vendor/operator library hooks if needed

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
