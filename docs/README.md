# Documentation Guide

Use this page as the entry point to the repo docs.

## Start Here

- [`../README.md`](../README.md): project overview, build, CLI, and main features
- [`feature_status.md`](feature_status.md): authoritative compatibility matrix
- [`../examples/README.md`](../examples/README.md): runnable sample programs

## Backends And Runtime

- [`emit_c_cpp.md`](emit_c_cpp.md): C and C++ emission design and guarantees
- [`emit_python.md`](emit_python.md): Python emission status, workflow, and limits
- [`complex.md`](complex.md): complex numbers, FFT, and DSP-oriented runtime support
- [`emit_systemc.md`](emit_systemc.md): future SystemC backend plan

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
