# examples/

This directory contains short, runnable programs that exercise the main
language and runtime features the project supports today.

Each example is intended to compile and run end-to-end through the normal
pipeline:

`MATLAB source -> parse -> Sema -> MLIR -> backend -> executable`

Quick check for any one example:

```sh
just compile examples/<name>.m /tmp/<name> && /tmp/<name>
```

Run the whole directory with:

```sh
just examples
```

## Feature Coverage

| File | Demonstrates |
|---|---|
| `hello.m` | `disp`, `fprintf`, basic script execution |
| `matrix_mult.m` | matrix literals, `*`, `.*`, transpose |
| `solve_linear.m` | left division `A \ b` |
| `eigendecomp.m` | `eig`, `det`, `inv` |
| `logical_mask.m` | logical indexing and reductions |
| `stats.m` | `numel`, `sum`, `mean`, `min`, `max`, derived scalar math |
| `for_loop.m` | nested `for`, non-unit step, negative step |
| `even_odd.m` | simple `if`/`else` inside a `for` loop with branch-conditional accumulators |
| `while_loop.m` | `while` loops |
| `fibonacci.m` | loop-carried state and iterative control flow |
| `factorial.m` | recursion and user-defined functions |
| `persistent_counter.m` | `persistent` — function-local state that survives across calls |
| `parfor.m` | `parfor` reductions and helper calls |
| `func_handles.m` | builtin and user function handles |
| `anon_capture.m` | anonymous functions with captures |
| `bank_account.m` | `classdef`, properties, methods, `Dependent`, inheritance-style object model, operator overloading |
| `traffic_action.m` | branching and simple classification |
| `is_old.m` | boolean logic and predicate-style functions |
| `fi_apply_gain.m` | Fixed-Point Designer (`fi`) constructor + `*` + `(:)` clamp + `disp` |
| `fi_fir_filter.m` | fi arrays + vector concat + scalar MAC accumulator (Phase 3 gating shape) |
| `ode_solver.m` | `ode45` / `ode23` with the 2-element and user-grid `tspan` shapes, `odeset` (`RelTol`, `AbsTol`, `MaxStep`, `Stats`), backward-time integration |
| `symbolic_demo.m` | Symbolic Math Toolbox via SymPP — `syms` / `diff` / `int` / `simplify` / `solve` / `dsolve` / `pdsolve` / `laplace` / `fourier` / `ztrans` / `assume` / `vpa` / `taylor` / `limit` / symbolic matrices (`sym_matrix`, `sym_det`, `sym_inv`, `sym_linsolve`, `sym_dsolve_system`, `sym_solve_2x2`). Requires `-DMATLAB_LLVM_WITH_SYM=ON` at configure time |

## Flowchart (`.mflow`) examples

The [`mflow/`](mflow/README.md) subdirectory holds counterparts of
the canonical text examples expressed as MatForge IDE diagrams.
Each `.mflow` is JSON the IDE saves; `matlabc` reads them through
the same pipeline as `.m` source so every existing emit backend
works unchanged. Includes `hello`, `for_loop`, `is_old`,
`factorial`, plus two `custom`-block demos (inline `source` and
sibling-`.m` `path` provenance). See
[`mflow/README.md`](mflow/README.md) for usage.

## HDL examples

The `hdl/` subdirectory holds 8 synthesizable modules targeting the
`-emit-systemverilog` backend. These compile to vendor-neutral
SystemVerilog (Verilator lint-clean) and most also compile to C / C++ /
Python / TypeScript via the standard backends:

| Example | Shape | Notes |
|---|---|---|
| `alu_16bit.m` | combinational | switch-case ALU with overflow detection on add/sub |
| `mux_4to_1_16bit.m` | combinational | 4:1 multiplexer; renders as `unique case` in SV |
| `counter_0_to_10.m` | sequential | persistent counter with reset; canonical `if isempty(_); _ = init; end` idiom |
| `mealy_fsm.m` | sequential FSM | 2-state Mealy detecting "11"; `typedef enum` in SV, `unique case` on state |
| `moore_fsm.m` | sequential FSM | 3-state Moore; output decoded from state register only |
| `vector_processor.m` | combinational | 3-element vector dot product + magnitude squared with saturate |
| `sequential_processor.m` | sequential | 4-tap FIR with persistent shift register + accumulator |
| `fir_asic_pipelined.m` | sequential | 4-tap pipelined FIR with N parallel persistent fi-array registers |

Most have a paired `<name>_synth.m` typed driver for the C/C++/Python
/TS path; the SV path uses inline `% hdl: port(...)` pragmas instead.

See [`../docs/emit_systemverilog.md`](../docs/emit_systemverilog.md) for
the SV pipeline and `% hdl:` pragma reference.

## Notes

- These are demonstration programs, not an exhaustive compatibility
  suite. The full supported surface is broader than this directory.
- The authoritative feature inventory is
  [`../docs/feature_status.md`](../docs/feature_status.md).
- If you want broader coverage, inspect `test/Run/`, which holds the main
  execution corpus used for backend parity checking. Hardware-output
  parity is in `test/EmitSV/`, `test/EmitSVPorts/`, and
  `test/EmitSVFail/`.
