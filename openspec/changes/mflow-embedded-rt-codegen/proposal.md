## Why

mflowLink already has two codegen lanes, and one of them is genuinely Embedded-Coder-class:

- **`-emit-mflowlink-cpp`** emits a self-contained C++ that embeds the `.mflow` JSON and
  *interprets* it at runtime via `MflowLinkSim` (links `libMatlabFlowchart.a`). Deployable,
  but it is a runtime interpreter, not flat real-time code.
- **Subsystem / whole-diagram emit** (`-emit-{c,cpp,python,ts,sv} model.mflow [--subsystem]`,
  the "Embedded Coder" Tiers 1–7) lowers the model to a MATLAB-function AST and feeds it
  through the existing MLIR pipeline + language backends. This produces **flat, inlined,
  dependency-free step code** with per-subsystem state, multirate scheduling, continuous→
  discrete conversion, and fixed-point — no runtime dispatch.

So the flat-code kernel exists. What's missing is the **production real-time wrapper** that
makes that kernel drop-in for an MCU/RTOS — the "§17.5 #12 / Tier-7 follow-up" the roadmaps
already flag:

1. **No ERT-style entry-point contract.** Whole-diagram emit produces a `simulate()` time-loop
   driver (it runs the model to completion and logs CSV). Embedded targets need the reusable
   trio `model_initialize()` / `model_step()` / `model_terminate()` over a static, externally-
   owned state struct — the caller (a timer ISR, an RTOS task, a PIL harness) drives the step.
2. **No real-time scheduling harness.** Multirate is handled *inside* the kernel, but there is
   no generated base-rate + sub-rate task scheduler (rate-monotonic counters / `model_step(tid)`)
   that an RTOS or bare-metal super-loop can call.
3. **No MISRA-leaning / static-only C mode.** Step code should do no dynamic allocation, keep
   all state in a caller-visible struct, and avoid constructs a safety linter rejects.
4. **No whole-diagram SystemVerilog.** SV emit is per-subsystem only; a whole `.mflow` cannot
   be emitted as one synthesizable top module.
5. **No target packaging.** There is no generated build manifest / entry-point header bundle
   to hand to a cross-compiler.
6. **No buffer-in-place optimisation.** Block-chain temporaries are not aliased/reused.

This change adds the real-time/ERT layer **on top of** the existing flat emit — it does not
re-implement code generation.

## What Changes

- **ERT entry-point contract for whole-diagram C/C++.** A new emit mode (e.g.
  `-emit-c model.mflow --rt` / `--ert`) generates `model_initialize(RT_MODEL*)`,
  `model_step(RT_MODEL*)`, and `model_terminate(RT_MODEL*)` over a static state/work struct
  (`*_DW`) and external I/O structs (`*_U` / `*_Y`) in a generated header — the standard
  Embedded-Coder shape — instead of (or alongside) the existing `simulate()` driver.
- **Fixed-step real-time scheduling.** For multirate models, generate a base-rate `model_step`
  plus sub-rate task IDs with rate counters, and an optional `rt_OneStep()` template that an
  ISR/RTOS task calls. Single-rate models collapse to one `model_step`.
- **Static / MISRA-leaning C profile.** A codegen profile that forbids heap allocation in the
  step path, keeps all state in the caller-owned struct, fixes all sizes at emit time, and
  avoids constructs flagged by common MISRA-C subsets; documented deviations where unavoidable.
- **Whole-diagram SystemVerilog.** Emit a whole `.mflow` as a single synthesizable top module
  (composing the per-subsystem SV already supported), with a generated clock/reset/IO contract.
- **Target packaging.** Emit a small bundle: the generated sources, a public header with the
  entry-point + struct declarations, and a build manifest (compile + link recipe) so the
  output cross-compiles without hand-editing.
- **Buffer-in-place optimisation (best-effort).** Reuse block-chain temporaries via simple
  alias analysis where a signal is consumed exactly once, reducing the work struct footprint.

Scope guard: this targets **deployable bare-metal / RTOS C** and **whole-diagram SV**. AUTOSAR
RTE generation, PIL board bring-up, and Simscape/Stateflow codegen remain out of scope and stay
on the roadmaps.

## Capabilities

### New Capabilities
- `mflow-embedded-rt-codegen`: the mflowLink production real-time codegen layer — ERT-style
  `model_initialize`/`model_step`/`model_terminate` entry points over a static state struct,
  fixed-step multirate task scheduling, a static/MISRA-leaning C profile, whole-diagram
  SystemVerilog, and a packaged build bundle.

### Modified Capabilities
- `flowchart-frontend`: the "Subsystem-to-MATLAB lowering" / codegen requirement gains the
  real-time entry-point and whole-diagram-SV scenarios (mflowLink codegen lives here).

## Impact

- **Codegen** (`lib/Flowchart/SubsystemToMatlab.cpp`, the whole-diagram TU builder, the C/C++
  and SV backends in `tools/matlabc`): a new `--rt`/`--ert` path that wraps the existing flat
  kernel in the entry-point trio + static struct; the multirate scheduler template; the SV
  whole-diagram composer; the packaging writer.
- **Reuse, no change**: the MLIR pipeline, the per-block AST lowering, fixed-point stamping,
  multirate sample-time inference, and the per-subsystem SV emitter — the RT layer composes
  these.
- **CLI / Loader**: `--rt`/`--ert`, `--rt-profile=misra`, `--emit-package` flags.
- **Tests**: an `EmitRt` lane — emit the ERT trio for a stateful multirate model, compile it
  against a tiny generated harness, run it, and diff against `-simulate` (the entry-point code
  must reproduce the interpreter); a whole-diagram SV Verilator-lint + cosim check; a MISRA
  smoke (no malloc in the step path).
- **Docs**: `docs/embedded_coder_roadmap.md` (§17.5 #12 RT tier), `docs/mflow_link_roadmap.md`
  codegen section, a new "Real-time / embedded deployment" section in `docs/mflowlink_blocks.md`.
