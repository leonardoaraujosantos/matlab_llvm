# Design — mflowLink embedded / real-time codegen

## Context

mflowLink codegen has two lanes (see proposal). The relevant one here is **Path A**
(`embedded_coder_roadmap.md` §3): a `.mflow` model → a synthesised MATLAB-function AST
(`buildSubsystemTU` / `buildDiagramTU` in `lib/Flowchart/SubsystemToMatlab.cpp`) → the existing
MLIR pipeline + `-emit-{c,cpp,python,ts,sv}` backends. Tiers 1–7 already produce flat, inlined,
dependency-free kernels with per-subsystem state, multirate scheduling, continuous→discrete
conversion, and fixed-point.

Whole-diagram emit today wraps the kernel in a `simulate()` function: an internal time loop
that drives sources → kernel → sinks and logs CSV. That is a *driver*, not a reusable
entry-point. This change adds the production real-time wrapper around the same kernel.

## Goals / Non-goals

**Goals**: ERT-style `model_initialize`/`model_step`/`model_terminate` over a static, caller-
owned state struct; fixed-step multirate task scheduling; a static/MISRA-leaning C profile;
whole-diagram SystemVerilog; a packaged build bundle; optional buffer-in-place reuse. Reuse the
existing flat kernel — do not re-implement per-block codegen.

**Non-goals**: AUTOSAR RTE / `.arxml`; PIL board bring-up and on-target timing; Simscape /
Stateflow codegen; an OS abstraction layer (we emit *templates/hooks*, not an RTOS port);
full ISO-MISRA certification (we target a defensible subset + documented deviations).

## Decisions

### 1. The RT wrapper composes the existing kernel; it does not replace it
`--rt` keeps using `buildDiagramTU` to get the flat step kernel, then emits a thin ERT shell:

```c
/* model.h (generated) */
typedef struct { /* unit-delay/integrator/filter state, fixed sizes */ } Model_DW;
typedef struct { /* root inports */ } Model_U;
typedef struct { /* root outports / logged signals */ } Model_Y;
typedef struct { Model_DW dwork; Model_U u; Model_Y y; } RT_MODEL;

void model_initialize(RT_MODEL *m);   /* zero/seed state + initial conditions */
void model_step(RT_MODEL *m);         /* one base-rate step: reads m->u, writes m->y */
void model_terminate(RT_MODEL *m);    /* no-op for static models; hook for symmetry */
```

The body of `model_step` is the existing inlined kernel with its locals; state that today
lives in struct fields persists in `Model_DW`; root `signal_inport`/`signal_outport` (or
sources/sinks) map to `Model_U`/`Model_Y`. Sinks that were CSV logs become `Model_Y` fields
(the caller decides what to do with them). The old `simulate()` driver stays available as a
non-`--rt` convenience and as the reference oracle for tests.

### 2. Real-time scheduling: base rate + sub-rate task IDs
Reuse the Tier-6c multirate sample-time inference. For a single-rate model, `model_step(m)` is
the whole step. For a multirate model, emit `model_step(m, tid)` where `tid` selects a rate,
plus a generated `rt_OneStep()` template that maintains rate counters and calls each rate at
its period — the classic rate-monotonic pattern. The template is a *hook* the user wires to a
timer ISR or an RTOS task; we do not assume an OS.

### 3. Static / MISRA-leaning profile
`--rt-profile=misra` constrains the C backend: no `malloc`/`free` in the step path, all sizes
constant at emit time, all state in `RT_MODEL`, no VLAs, no recursion, single-return helpers
where practical, explicit narrowing casts for fixed-point. Where a construct is unavoidable,
emit a `/* MISRA deviation: <rule> — <reason> */` marker. This is a profile over the existing
emitter, not a new backend. The smoke test greps the step path for `malloc(`/`calloc(` and
fails on any hit.

### 4. Whole-diagram SystemVerilog by composition
Per-subsystem SV already exists and hard-rejects continuous blocks. Whole-diagram SV emits one
top module that instantiates each subsystem module and wires the inter-block signals, with a
single `clk`/`rst_n` and the root IO as ports. Continuous-time blocks are rejected with a
sourced error (as today) — the user must discretise first. This closes the "per-subsystem only"
gap noted in `embedded_coder_roadmap.md`.

### 5. Packaging
`--emit-package <dir>` writes: the generated source(s), `model.h` (entry points + structs), and
a `build.mk` / manifest with the compile + link recipe (no runtime libs for the flat C path).
This makes the output cross-compilable without hand-editing include paths.

### 6. Buffer-in-place (best-effort, last)
A signal consumed exactly once by a downstream block can reuse its producer's temporary instead
of a fresh `Model_DW`/local slot. Implement as a simple single-use alias pass over the lowered
AST; skip anything multi-fanout or stateful. Purely a footprint optimisation — gated so it
never changes numeric output.

## The key correctness contract

The ERT entry-point code must reproduce the interpreter. The `EmitRt` test compiles the
generated `model_step` against a tiny harness that feeds the same inputs the `.mflow` sources
produce, steps `N` times, and diffs the captured `Model_Y` against `matlabc -simulate` of the
same model. This is the same "compiled == interpreted" contract the existing
`emit-mflowlink-cpp` parity lane enforces, applied to the flat RT code.

## Risks / Mitigations

- **State-struct mapping bugs** (a delay/integrator state not persisted in `Model_DW`) →
  the `EmitRt` parity diff catches any divergence from `-simulate`; start with single-rate
  stateful models, then multirate.
- **Multirate scheduler off-by-one** (a sub-rate fires on the wrong tick) → golden the rate
  counter sequence; reuse the Tier-6c inference rather than re-deriving periods.
- **MISRA profile churn** → ship the profile as additive constraints with deviation markers;
  do not block the main `--rt` path on full MISRA cleanliness.
- **Scope creep toward AUTOSAR/PIL** → those stay explicitly out and on the roadmaps; this
  change stops at deployable bare-metal/RTOS C + whole-diagram SV + packaging.

## Migration

Additive. Existing `-emit-*` (including the whole-diagram `simulate()` driver and per-subsystem
emit) are unchanged. `--rt`, `--rt-profile`, `--emit-package`, and whole-diagram SV are opt-in.
