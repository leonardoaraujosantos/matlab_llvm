# Documentation Guide

Use this page as the entry point to the repo docs.

## Start Here

- [`../README.md`](../README.md): project overview, build, CLI, and main features
- [`feature_status.md`](feature_status.md): authoritative compatibility matrix
- [`../examples/README.md`](../examples/README.md): runnable sample programs

## Frontend And Sema

- [`sema.md`](sema.md): tour of `lib/Sema/` — bindings, scopes, name resolution, and type inference
- [`save_load_compat.md`](save_load_compat.md): plan to bring `save` / `load` to the real MATLAB API and `.mat` v5 file format

## Backends And Runtime

- [`emit_c_cpp.md`](emit_c_cpp.md): C and C++ emission design and guarantees
- [`emit_cpp_classdef.md`](emit_cpp_classdef.md): plan to translate MATLAB `classdef` to real C++ classes (replaces the runtime-hash wrapper, queued for a focused follow-up)
- [`emit_python.md`](emit_python.md): Python emission status, workflow, and limits
- [`complex.md`](complex.md): complex numbers, FFT, and DSP-oriented runtime support
- [`emit_systemverilog.md`](emit_systemverilog.md): direct RTL/SystemVerilog backend plan, including combinational, register, counter, and FSM inference
- [`emit_systemc.md`](emit_systemc.md): future SystemC backend plan
- [`emit_fixed_point.md`](emit_fixed_point.md): Fixed-Point Designer (`fi`) support — Phase 1 scalar arithmetic shipped, Phase 3 arrays / FIR planned

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
