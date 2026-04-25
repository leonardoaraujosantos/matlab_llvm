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
| `while_loop.m` | `while` loops |
| `fibonacci.m` | loop-carried state and iterative control flow |
| `factorial.m` | recursion and user-defined functions |
| `parfor.m` | `parfor` reductions and helper calls |
| `func_handles.m` | builtin and user function handles |
| `anon_capture.m` | anonymous functions with captures |
| `bank_account.m` | `classdef`, properties, methods, `Dependent`, inheritance-style object model, operator overloading |
| `traffic_action.m` | branching and simple classification |
| `is_old.m` | boolean logic and predicate-style functions |

## Notes

- These are demonstration programs, not an exhaustive compatibility
  suite. The full supported surface is broader than this directory.
- The authoritative feature inventory is
  [`../docs/feature_status.md`](../docs/feature_status.md).
- If you want broader coverage, inspect `test/Run/`, which holds the main
  execution corpus used for backend parity checking.
