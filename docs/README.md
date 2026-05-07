# Documentation Guide

Use this page as the entry point to the repo docs.

## Start Here

- [`../README.md`](../README.md): project overview, build, CLI, and main features
- [`feature_status.md`](feature_status.md): authoritative compatibility matrix
- [`roadmap.md`](roadmap.md): forward-looking work tracker (CocoTB verification, SV→MATLAB lift, runtime / REPL / HDL improvements; the block language is shipped — see [`flowchart_frontend.md`](flowchart_frontend.md))
- [`../examples/README.md`](../examples/README.md): runnable sample programs

## Frontend And Sema

- [`sema.md`](sema.md): tour of `lib/Sema/` — bindings, scopes, name resolution, and type inference
- [`flowchart_frontend.md`](flowchart_frontend.md): `.mflow` graphical block-language frontend — design, architecture, and shipped phases (graph → AST → every existing backend; round-trips to MATLAB source for free)
- [`flowchart_schema.md`](flowchart_schema.md): `.mflow` JSON schema reference — field-by-field contract, every block kind's required data fields and port conventions, validation rules. **Read this when implementing the IDE save/load.**
- [`save_load_compat.md`](save_load_compat.md): plan to bring `save` / `load` to the real MATLAB API and `.mat` v5 file format

## Backends And Runtime

- [`emit_c_cpp.md`](emit_c_cpp.md): C and C++ emission design and guarantees
- [`emit_cpp_classdef.md`](emit_cpp_classdef.md): plan to translate MATLAB `classdef` to real C++ classes (replaces the runtime-hash wrapper, queued for a focused follow-up)
- [`emit_python.md`](emit_python.md): Python emission status, workflow, and limits
- [`complex.md`](complex.md): complex numbers, FFT, and DSP-oriented runtime support
- [`emit_systemverilog.md`](emit_systemverilog.md): direct RTL/SystemVerilog backend plan, including combinational, register, counter, and FSM inference
- [`emit_cocotb.md`](emit_cocotb.md): `-emit-cocotb` harness — open-source HDL Verifier alternative; cycle-by-cycle co-simulation of the SV DUT against the Python reference, with HDL-Verifier-style `Latency` parameter
- [`emit_fixed_point.md`](emit_fixed_point.md): Fixed-Point Designer (`fi`) support — Phase 1 scalar arithmetic shipped, Phase 3 arrays / FIR planned
- [`sym.md`](sym.md): Symbolic Math Toolbox (`sym` / `syms`) backed by [SymPP](https://github.com/leonardoaraujosantos/SymPP) — diff/int/simplify/solve/dsolve/pdsolve/transforms/assume/vpa/taylor/limit; opt-in via `-DMATLAB_LLVM_WITH_SYM=ON`
- [`ode.md`](ode.md): Initial-value ODE solvers — `ode45`, `ode23` (Dormand–Prince / Bogacki–Shampine, scalar and **vector `y`**) with adaptive FSAL, cubic-Hermite dense output, full odeset (`RelTol`, `AbsTol`, `MaxStep`, `InitialStep`, `Refine`, `Stats`), 2-/3-return forms, forward and backward integration, user-time-grid `tspan`
- [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md): tiered plan for Signal Processing Toolbox compatibility. **Tier-1 + Tier-2 bulk + Tier-3 §4.3 are closed**. Tier-1: windows (17), polynomial helpers, IIR + order helpers, FIR, filter response, filter application. Tier-2: transforms, spectral, linear prediction + parametric PSD, time-frequency. Tier-3 §4.3: `findpeaks`, scalar reductions, `medfilt1`/`hampel`/`envelope`, pulse statistics (`midcross`/`risetime`/`falltime`/`dutycycle`). **Open follow-ons**: `ellip`/`besself`/band variants/analog prototypes (§2.1); `fir2`/`firls`/`firpm` (§2.2); Gustafsson IC for `filtfilt` (§2.5); `dpss`/`pmtm`/`czt`/`stft`/`istft`/subspace methods (Tier-2 follow-on); MinPeak* options for findpeaks, slewrate/pulseperiod/pulsewidth (§4.3 follow-on); real multirate (§4.1), `digitalFilter` system object (§5.1). Apps, GUIs, deep-learning, MATLAB Coder, Python coexecution explicitly carved out

## Interactive Tooling

- [`repl.md`](repl.md): JIT-backed REPL and workspace behavior
- [`debug.md`](debug.md): DAP mode, `dbg`, and runtime debugging aids
- [`lsp.md`](lsp.md): `matlab-lsp` capabilities and editor setup

## How To Read The Status Docs

- Treat [`feature_status.md`](feature_status.md) as the source of truth for
  what is implemented, partial, or missing.
- Treat backend docs as design and behavior notes for specific codegen paths.
- Treat the examples and tests as the best concrete reference for supported
  source patterns.
