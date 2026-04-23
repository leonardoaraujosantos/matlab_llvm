# SystemC (Behavioral) Emission — Plan

Forward-looking design doc for adding a `-emit-systemc` backend that
targets **synthesizable SystemC** — a constrained subset of MATLAB
that maps to combinational logic and explicit finite state machines
(FSMs) suitable for high-level synthesis (HLS) tools such as Vitis HLS,
Catapult, or Stratus.

Nothing here has shipped yet. This document scopes what's synthesizable,
what has to be rejected, the new passes required, and the phased
milestone plan.

## Scope

One new driver flag:

| Flag | Output |
|---|---|
| `-emit-systemc` | Self-contained `.cpp` (+ matching `.h`) SystemC module |

Generated code compiles against a SystemC distribution
(`-I$SYSTEMC_HOME/include`, `-lsystemc`) and is intended to be consumed
by an HLS tool. Simulation runs against the SystemC reference kernel;
synthesis is performed by the downstream HLS tool, not by this project.

## Why this is fundamentally harder than `-emit-c` / `-emit-python`

The C and Python backends are **retargeting exercises**: same pipeline,
new pretty-printer, new runtime shim. SystemC-for-synthesis forbids the
runtime entirely. There is no `malloc`, no `printf`, no dynamic shapes,
no recursion, no function pointers, no standard-library `std::vector`.
Every loop bound and every array dimension must be a compile-time
constant. Every integer must have a known bit width.

The bulk of the work is **not in the emitter** — it is in three new
MLIR passes that reshape the IR into something synthesizable, plus a
legality pass that produces good errors for anything outside the
subset.

## Target subset

A MATLAB function is synthesizable if, after the full lowering pipeline
plus the new passes in this document, it satisfies all of:

| Requirement | Rationale |
|---|---|
| All tensor shapes are compile-time constants | HLS requires fixed-size arrays |
| All numeric types have a known bit width (`int8`..`int64`, `fixed<W,I>`, `bool`) | `sc_int<W>` / `sc_fixed<W,I>` target types |
| No `matlab_*` runtime calls survive | No heap, no libm, no I/O |
| No recursion | Call graph must be a DAG |
| No function pointers / anonymous functions | All calls statically resolved |
| No `cell` arrays, no `struct` fields, no strings | No variant types in HLS |
| No variadic I/O (`disp`, `fprintf`) | No printf in synthesis |
| No parfor | Parallelism is expressed via pragmas, not runtime dispatch |
| Every `scf.while` has either a proven constant trip count (→ fully unrolled) or an explicit FSM encoding | HLS tools can handle `while`, but our FSM path is the predictable one |

Anything failing these gets rejected with a diagnostic pointing at the
offending MATLAB source line. That diagnostic quality **is half the
product** — otherwise users see cryptic HLS tool errors on generated
code they can't read.

## Architecture

Six new pieces. Three are MLIR passes run *before* emission; two are
supporting infrastructure; one is the emitter itself.

```
AST ──► MLIR ──► [existing pipeline ... LowerIO]
                       │
                       ├──► emitC() ──► .c / .c++                (existing)
                       │
                       └──► [HLSLegalize]  ───► reject or tag
                            [HLSInlineRuntime] ───► replace matlab_* with static loops
                            [HLSBitWidthInfer] ───► annotate every value with !sc_int<W> / !sc_fixed<W,I>
                            [HLSFSMExtract]   ───► classify each scf.while: unroll vs FSM
                                 │
                                 └──► emitSystemC() ──► .cpp + .h
```

## Step-by-step plan

### Phase 1 — MVP: pure combinational, scalars only (~1 week)

Goal: a `foo.m` consisting of scalar arithmetic with `scf.if` compiles
to a single `SC_METHOD` sensitive to all inputs. No loops, no arrays,
no FSM.

**Step 1.1** — CLI flag.
`tools/matlabc/main.cpp`: `Mode::EmitSystemC`, `-emit-systemc` parser, dispatch branch.
`include/matlab/MLIR/Passes/Passes.h`: declare `emitSystemC(ModuleOp, raw_ostream&)`.

**Step 1.2** — New pass `HLSLegalize`.
`lib/MLIR/Passes/HLSLegalize.cpp`. Walk the post-`LowerIO` module and
emit an error diagnostic (via `mlir::emitError(loc)`) for each:
- `llvm.call` to any `matlab_*` symbol that isn't in an allowlist (initially: the allowlist is empty, so any runtime call rejects)
- `func.func` with recursion (detected via call-graph cycle)
- `llvm.alloca` of a non-constant array size
- `scf.while` with an undecidable trip count *and* no FSM annotation (Phase 1 rejects all `scf.while`; Phase 3 relaxes)
- `llvm.mlir.global` of a string type (no strings in v1)

Tests: `test/EmitSystemCFail/*.m` + `.stderr` goldens, mirroring the
existing `test/EmitCFail/` contract.

**Step 1.3** — New pass `HLSBitWidthInfer`.
`lib/MLIR/Passes/HLSBitWidthInfer.cpp`. Walks the module and attaches a
`hls.type` discardable attribute to every SSA value:
- `f64` without annotation → `sc_fixed<32,16>` (default; documented).
- `f32` → `sc_fixed<16,8>` (default).
- `i64` → `sc_int<64>`; `i32` → `sc_int<32>`; `i1` → `bool`.
- MATLAB-side casts (`int8(x)`, `uint16(x)`) propagate through
  `arith.extsi` / `arith.trunci` / `arith.sitofp` and override defaults.
- User override: a comment like `% hls: fixed<24,8>` on the variable
  declaration, recognized by the frontend, threaded through
  `matlab.name` as a sibling `hls.type` attribute.

The defaults are a **policy decision** — pick conservative widths first
(32-bit fixed-point), document them, iterate on real designs.

**Step 1.4** — Emitter `emitSystemC.cpp`.
Clone `EmitC.cpp`. Differences:
- Wrap each `func.func` in an `SC_MODULE` with one `sc_in<T>` port per
  argument, one `sc_out<T>` port per return, and an `SC_METHOD`
  sensitive to all input ports. For Phase 1 this is the *only* emitted
  shape — no `SC_CTHREAD`, no FSM.
- Type-print helper: `hls.type` attr → `sc_int<W>` / `sc_fixed<W,I>` /
  `bool`. Falls back to a diagnostic error if the attribute is missing
  (Phase 1 requires `HLSBitWidthInfer` ran successfully).
- No runtime extern block — the runtime is forbidden in synthesis.
- `scf.if` with results uses the ternary form (`cond ? a : b`) rather
  than the mutable-local form `EmitC.cpp` uses, because HLS tools
  prefer dataflow shapes for combinational paths.
- Always emit both `.cpp` (implementation) and `.h` (module
  declaration). CLI flag `-emit-systemc=foo` writes `foo.cpp` and
  `foo.h`; reading from `stdout` concatenates them with a `// === foo.h ===`
  marker.

**Step 1.5** — Tests.
`test/HLS/combinational/*.m` — scalar-only MATLAB programs. Golden:
the `.cpp` output. (We do not run HLS in CI; that's the user's
downstream concern.) A smoke check compiles each output against
SystemC headers to catch syntax errors:

```bash
c++ -std=c++17 -I$SYSTEMC_HOME/include -c foo.cpp -o /dev/null
```

### Phase 2 — Fixed-size arrays, fully-unrolled loops (~1 week)

Goal: MATLAB `for i = 1:N; A(i) = ...; end` with constant `N` compiles
to an unrolled `SC_METHOD` that writes every element combinationally.

**Step 2.1** — New pass `HLSInlineRuntime`.
Replaces specific `llvm.call @matlab_*` ops with inline `scf.for`
unrolled loops over `llvm.alloca`'d arrays of static shape. Targets:
- `matlab_zeros(N, M)` → static array + nested `scf.for` writing zero
- `matlab_ones(N, M)` → same with `1`
- `matlab_transpose(A)` → nested `scf.for` with swapped indices
- `matlab_matmul_mm(A, B)` → triple-nested `scf.for` dot-product
- Element-wise `matlab_add_mm`, `matlab_mul_mm`, etc. → nested `scf.for`

Only triggers when all input shapes are compile-time constants
(verified via `LowerTensorOps` shape tracking — if shapes aren't
constant, `HLSLegalize` rejects in Phase 1 already).

**Step 2.2** — `HLSFSMExtract` (partial): constant-trip-count unroll.
For each `scf.while` / `scf.for` with a statically provable trip
count ≤ an unroll threshold (default 64, configurable), fully unroll
in place. Remaining loops defer to Phase 3.

**Step 2.3** — Emitter updates.
- `llvm.alloca` of `!llvm.array<N x f64>` → `sc_fixed<W,I> arr[N]` as a
  module-private member (not a port).
- `llvm.getelementptr` + `llvm.store` → `arr[i] = v`.
- `scf.for` with constant bounds → unrolled sequence of statements (no
  C `for` loop — HLS tools unroll fine on their own, but unrolling at
  emit time keeps output readable and the `SC_METHOD` sensitivity list
  correct).

### Phase 3 — FSM for data-dependent loops (~2 weeks)

Goal: `while (x > 0) { x = f(x); }` with data-dependent trip count
compiles to an `SC_CTHREAD` with an explicit state register and
`wait()` between states.

**Step 3.1** — `HLSFSMExtract` (full).
For each non-unrollable `scf.while`:
- Allocate a state enum: `S_ENTRY`, `S_BODY_0`, ..., `S_EXIT`.
- Each basic block becomes one state. The before-region (condition
  check) is one state; the after-region body is one or more states
  depending on whether it contains its own nested structured control.
- Iter-args become module-private registers (`sc_signal<T>` or plain
  members, depending on coding style).
- Transition logic: next-state computed combinationally from current
  state + condition value.

The pass rewrites the `scf.while` into a synthetic `hls.fsm` op
carrying the state table and the per-state IR blocks. The emitter
recognizes `hls.fsm` specifically and emits the `SC_CTHREAD` + `switch
(state)` scaffolding.

**Step 3.2** — Emitter: `SC_CTHREAD` path.
- `hls.fsm` → `void run() { while (true) { switch (state) { case S_0: ...; wait(); break; ... } } }`
- Sensitivity: clock + reset, not inputs.
- Reset behavior: all state registers initialized in the reset block.

**Step 3.3** — Function calls to non-leaf functions.
A `func.call` inside a `SC_CTHREAD` to another function that itself
has an FSM is a **hierarchical FSM call** — the callee becomes a
sub-module, the caller drives its input ports and `wait`s on a `done`
signal. This is non-trivial; defer until there's a demand.

For Phase 3 v1, restrict calls inside FSM bodies to **leaf functions**
(pure combinational, no inner FSM) and inline them at the IR level
before FSM extraction.

### Phase 4 — Pragma surface (~1 week)

HLS tools consume pragmas for pipeline depth, loop unroll factors,
array partitioning, etc. Surface these as MATLAB comments:

```matlab
% hls: pipeline II=1
for i = 1:16
  % hls: unroll
  A(i) = B(i) * C(i);
end
```

The frontend already parses leading comments; extend it to recognize
`% hls:` and attach as `hls.pragma` attributes on the corresponding
MLIR op. The emitter prints them as `#pragma HLS ...` directly above
the relevant block.

Keep pragma syntax vendor-agnostic in the MATLAB source; vendor
translation (Vitis vs. Catapult vs. Stratus) happens in the emitter
via a `-systemc-vendor=vitis|catapult|stratus` flag.

## Op mapping table

### Combinational (`SC_METHOD`)

| MLIR | SystemC |
|---|---|
| `arith.constant 1.5 : f64` + `hls.type = sc_fixed<32,16>` | `sc_fixed<32,16>(1.5)` |
| `arith.addf / subf / mulf / divf` | `+` / `-` / `*` / `/` on `sc_fixed` |
| `arith.addi / subi / muli` | `+` / `-` / `*` on `sc_int` |
| `arith.cmpf / cmpi` | `==`, `<`, etc., producing `bool` |
| `arith.select` | `cond ? a : b` |
| casts between int/fixed widths | `sc_int<W>(x)` / `sc_fixed<W,I>(x)` |
| `scf.if -> T` | `cond ? a : b` (preferred), else a temp + `if/else` |
| `func.func` (leaf) | `SC_MODULE` with `SC_METHOD` sensitive to inputs |
| `func.call` (leaf) | inlined sub-module instance + port connection, or inlined function call (policy choice) |
| `llvm.alloca` of static array | `sc_fixed<W,I> arr[N];` as module member |
| `llvm.store / load` to static array | `arr[i] = v;` / `v = arr[i];` |

### Sequential (`SC_CTHREAD` + FSM)

| MLIR | SystemC |
|---|---|
| `hls.fsm` with N states | `switch(state) { case S_i: ...; wait(); state = S_j; }` inside a clocked thread |
| iter-arg of `scf.while` | module-private register initialized in reset |
| non-unrollable `scf.for` | FSM with counter register |

### Rejected (Phase 1+)

| MLIR | Diagnostic |
|---|---|
| `llvm.call @matlab_disp_*` | `error: I/O not supported in synthesis` |
| `llvm.call @matlab_parfor_dispatch` | `error: parfor requires explicit pragma-based parallelism` |
| `llvm.alloca` of dynamic size | `error: array size must be compile-time constant` |
| recursion | `error: recursion not supported` |
| function handle / anonymous fn | `error: function values not supported` |
| string literal / `matlab_cell_*` / `matlab_struct_*` | `error: <feature> not supported in synthesis subset` |

## Files

| File | Purpose |
|---|---|
| `lib/MLIR/Passes/HLSLegalize.cpp` | Rejects non-synthesizable IR with source-line diagnostics |
| `lib/MLIR/Passes/HLSInlineRuntime.cpp` | Replaces `matlab_*` runtime calls with static loop nests |
| `lib/MLIR/Passes/HLSBitWidthInfer.cpp` | Annotates every value with `hls.type = sc_int/sc_fixed/bool` |
| `lib/MLIR/Passes/HLSFSMExtract.cpp` | Classifies `scf.while`/`scf.for` as unroll vs FSM; rewrites non-unrollable loops to `hls.fsm` |
| `lib/MLIR/Passes/EmitSystemC.cpp` | Walker that prints `.cpp` + `.h`; handles combinational and FSM paths |
| `include/matlab/MLIR/Passes/Passes.h` | Declarations for the above |
| `tools/matlabc/main.cpp` | `-emit-systemc` flag; runs the 4 new passes in order after `runLowerIO`, then calls `emitSystemC()` |
| `test/HLS/combinational/*.m` | Phase 1–2 tests: golden `.cpp` + SystemC compile smoke |
| `test/HLS/fsm/*.m` | Phase 3 tests: ditto |
| `test/HLS/fail/*.m` | Legality rejection tests with `.stderr` goldens |
| `justfile` | `emit-systemc FILE`, `compile-systemc FILE`, `test-systemc` recipes |

## Testing strategy

Unlike `-emit-c` which tests execution equivalence against the LLVM
path, SystemC cannot easily be tested end-to-end in CI:
- Running HLS synthesis takes minutes to hours and requires vendor tools.
- Simulation with the SystemC kernel is feasible but slow.

Three-layer test pyramid:

1. **Golden-diff tests** (fast, always in CI). Each `test/HLS/*.m`
   pairs with a `.cpp` golden. Catches regressions in the emitter
   without executing anything.
2. **SystemC-compile smoke** (medium, always in CI). `c++ -std=c++17
   -I$SYSTEMC_HOME/include -c` on every generated `.cpp`. Catches
   syntax errors, missing includes, type mismatches. Requires a
   SystemC install; make it optional behind `-DMATLAB_LLVM_SYSTEMC=ON`.
3. **Simulation** (slow, opt-in). For Phase-3 FSM tests, write a
   SystemC testbench that drives inputs and checks outputs against a
   MATLAB reference run. Run under a `just test-systemc-sim` target,
   not by default in CI.

HLS synthesis itself is **not** in our test matrix — users run their
own HLS tool against the output.

## Effort estimate

| Phase | Scope | Effort |
|---|---|--:|
| 1 | Scalar combinational + legality + bit-width infer + emitter skeleton | ~1 week |
| 2 | Static arrays + runtime-call inlining + unroll-only loops | ~1 week |
| 3 | FSM extraction + `SC_CTHREAD` emission + leaf-call inlining | ~2 weeks |
| 4 | Pragma surface + vendor-specific rendering | ~1 week |
| **Total to a useful v1** | Phase 1+2 | **~2 weeks** |
| **Total to full subset** | Phase 1+2+3+4 | **~5 weeks** |

Phase 1+2 is already useful: any pure-dataflow DSP kernel (FIR filter,
small FFT stage, fixed-size matrix multiply) fits. Phase 3 extends to
iterative algorithms (Newton, CORDIC, bit-serial arithmetic).

## Non-goals

- **Full MATLAB coverage.** The synthesizable subset is a small
  fraction of the language. Programs outside it get a diagnostic, not
  best-effort output. Silent fallbacks produce unsynthesizable code,
  which is strictly worse than rejection.
- **HLS tool integration.** We emit SystemC; the user runs Vitis /
  Catapult / Stratus. No vendor runtime, no Tcl scripts, no project
  files generated.
- **RTL emission.** SystemC is the target; going all the way to Verilog
  is the downstream HLS tool's job. If someone later wants direct-to-Verilog,
  the right target is the upstream CIRCT dialects, not an extension of
  this backend.
- **Floating-point.** `sc_fixed<W,I>` is the default numeric. IEEE 754
  single/double can be forced via an attribute, but most HLS flows
  punt FP to a hardened block; emitting it from our path is not a
  priority.

## Design alternatives

### Alternative A — Target upstream CIRCT instead of SystemC

CIRCT (the LLVM hardware subproject) has its own MLIR dialects for
hardware (`hw`, `comb`, `seq`, `fsm`) and goes directly to Verilog.

**Pros**: strictly more control, no SystemC dependency, direct-to-RTL.
**Cons**: far more scaffolding — we'd be writing our own HLS instead
of leaning on a commercial tool. The subset of MATLAB that would
survive is even smaller. CIRCT is under active development and its
interfaces churn. For this project, SystemC + commercial HLS is the
pragmatic target.

Revisit if we ever want to remove the HLS-tool dependency or target
non-standard hardware.

### Alternative B — Target an existing HLS-friendly C dialect

Some HLS tools accept C/C++ directly with vendor pragmas; no SystemC
needed. Easier on the emitter (could almost reuse `EmitC.cpp`), but
gives up the FSM-as-`SC_CTHREAD` abstraction — we'd encode FSMs as
manually-written `switch` + `#pragma` blocks in C, which is harder
for users to read and harder for tools to optimize.

SystemC's `SC_CTHREAD` model maps cleanly to our FSM extraction and is
the portable intermediate. Prefer it.

### Alternative C — Skip the emitter, generate vendor-specific C++ directly

Write one emitter per HLS vendor. Strictly worse: N emitters to
maintain, output is vendor-locked, no portable intermediate for
simulation.

## Open questions

1. **Default bit widths.** `sc_fixed<32,16>` is a guess. Real designs
   tune per-variable via pragmas. Ship with a conservative default and
   surface it loudly in the docs; add per-variable override via MATLAB
   comment syntax in Phase 1.

2. **Reset style.** Synchronous or asynchronous reset? Active-high or
   active-low? Add a `-systemc-reset=sync-high|sync-low|async-high|async-low`
   flag; default to sync-active-high (Xilinx convention).

3. **Clock name / handshake protocol.** For Phase 3, the FSM needs a
   clock port and optionally a `start`/`done` handshake for hierarchical
   use. Start with a hardcoded convention (`clk`, `rst`, `start`, `done`),
   revisit if a concrete design needs more.

4. **Multiple return values.** MATLAB `[a, b] = foo(x)` lowers to a
   function with multiple results or out-parameters. SystemC modules
   use output ports; multiple outputs are natural. The C emitter
   rejects multi-result functions today — we'll need to un-reject
   that specifically for the SystemC path (or accept single-output
   functions only in v1).

5. **Interaction with `Monomorphise`.** The existing pipeline runs
   `Monomorphise` before emission; if a function is called with
   multiple distinct shape/type tuples, it's already duplicated into
   per-type clones. The SystemC path inherits this for free, which is
   exactly what HLS wants — one module per concrete type instantiation.
