# examples/

Short, self-contained MATLAB programs that the `matlab_llvm` compiler can
take end-to-end: parse → Sema → MLIR → LLVM IR → native executable.

Every file here is expected to compile cleanly with:

```sh
just compile examples/<name>.m /tmp/<name> && /tmp/<name>
```

| File              | Demonstrates                                                     |
| ----------------- | ---------------------------------------------------------------- |
| `hello.m`         | `disp` of a char literal; `fprintf` with an inline format string |
| `matrix_mult.m`   | Matrix construction, `*`, `.*`, transpose                        |
| `solve_linear.m`  | Left-division `A \ b` for a small linear system                  |
| `eigendecomp.m`   | `eig`, `det`, `inv` on a symmetric tridiagonal                   |
| `logical_mask.m`  | Logical indexing `A(A > 0)`, `mean`, `sum`-of-logical            |
| `stats.m`         | `numel`, `sum`, `mean`, `min`, `max`, `sqrt` of `sum(x.*x)`      |
| `for_loop.m`      | Vectorised substitutes for loops (outer product, `sum`)          |
| `factorial.m`     | Single-recursion user function (`if/else`, `*`, `-`)             |
| `parfor.m`        | `parfor` reductions (single/multi/step) + calls to user helpers  |
| `func_handles.m`  | `@sin` / `@sqrt` / `@abs` / `@exp` handles via indirect call     |
| `anon_capture.m`  | `@(x) x + k` — by-value captures of outer scalar variables       |

## Current limitations the examples work around

- Sequential `for` / `while` lowering isn't wired to LLVM yet (only
  `parfor` is), so `for_loop.m` uses vectorised equivalents.
- `fprintf` with `%f`/`%d` against a computed scalar whose Sema type is
  `any` (e.g. `mean(A(:))`) falls off the fast-path today. Use `disp()`
  when printing aggregate results.
- Two recursive self-calls in one expression (`fib(n-1) + fib(n-2)`)
  isn't handled by `LowerUserCalls` yet, so the classic Fibonacci is
  replaced with `factorial` above.
