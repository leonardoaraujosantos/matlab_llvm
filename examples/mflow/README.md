# `.mflow` Examples

Flowchart counterparts of the canonical `.m` programs in
`examples/`, plus two custom-block demos. Each `.mflow` is the
JSON form a MatForge IDE diagram saves to disk; `matlabc` reads
them through the same pipeline as `.m` text so every existing
emit backend (`-emit-c`, `-emit-cpp`, `-emit-python`,
`-emit-typescript`, `-emit-systemverilog`, `-emit-llvm`,
`-emit-mlir`) works unchanged.

## Examples

| File | Mirrors | Demonstrates |
|---|---|---|
| [`hello.mflow`](hello.mflow) | [`../hello.m`](../hello.m) | Linear chain: `display` + `expression` blocks |
| [`for_loop.mflow`](for_loop.mflow) | [`../for_loop.m`](../for_loop.m) | `for` loop with body + back-edge → `disp(total)` |
| [`matrix_mult.mflow`](matrix_mult.mflow) | [`../matrix_mult.m`](../matrix_mult.m) | `matrix_literal` blocks + `*`, `.*`, transpose displayed inline |
| [`solve_linear.mflow`](solve_linear.mflow) | [`../solve_linear.m`](../solve_linear.m) | Solve Ax = b via the left-divide operator (`assignment` block with `rhs: "A \\ b"`) |
| [`is_old.mflow`](is_old.mflow) | [`../is_old.m`](../is_old.m) | Multi-flow: program calls a `function`-kind sub-flow |
| [`factorial.mflow`](factorial.mflow) | [`../factorial.m`](../factorial.m) | Recursive function (sub-flow body has `if/else` and a self-call) |
| [`custom_inline_gain.mflow`](custom_inline_gain.mflow) | — | `custom` block with **inline `source`** (function body lives in the JSON) |
| [`custom_clamp.mflow`](custom_clamp.mflow) | — | `custom` block with **`path`** provenance (function body in `blocks/clamp.m`); three callers share one inserted Function |

## Generating new `.mflow` examples

Any `.m` file can be auto-converted to a `.mflow` diagram via the
reverse-direction `-emit-mflow` mode:

```bash
matlabc -emit-mflow examples/factorial.m > examples/mflow/factorial.mflow
just emit-mflow examples/for_loop.m  # via the justfile recipe
```

The output is in IDE-canonical JSON (alphabetical keys, 2-space
indent, blank-line empty arrays) so it diffs cleanly against IDE
re-saves. Auto-layout assigns column-shaped positions; the IDE
re-layouts on first open. Round-trip is idempotent from the second
iteration onward (`.m → .mflow → .m → .mflow` produces
byte-identical second `.mflow`).

## Running

```bash
# Round-trip to MATLAB source (the formatter preserves the inlined
# function definitions for sub-flows and custom blocks).
matlabc -emit-matlab examples/mflow/factorial.mflow

# Compile through any existing backend.
matlabc -emit-c       examples/mflow/factorial.mflow
matlabc -emit-python  examples/mflow/factorial.mflow
matlabc -emit-cpp     examples/mflow/factorial.mflow
matlabc -emit-llvm    examples/mflow/factorial.mflow

# Build & run the C output end-to-end.
matlabc -emit-c examples/mflow/factorial.mflow > /tmp/fact.c && \
  cc /tmp/fact.c runtime/matlab_runtime.c -o /tmp/fact -lm && \
  /tmp/fact
# fact(1..6):
# 1
# 2
# 6
# 24
# 120
# 720

# Inspect the parsed FlowDoc (validation + structural dump).
matlabc -dump-flow examples/mflow/factorial.mflow
```

The `custom_clamp.mflow` example uses a sibling `.m` file. The
loader resolves `data.path` relative to the `.mflow`'s directory,
so the example works from any working directory:

```bash
matlabc -emit-c examples/mflow/custom_clamp.mflow > /tmp/cc.c
cc /tmp/cc.c runtime/matlab_runtime.c -o /tmp/cc -lm && /tmp/cc
# clamp(42, 0, 10) = 10
# clamp(-3, 0, 10) = 0
# clamp(5,  0, 10) = 5
```

For `library_id` provenance (not shown above — see
`test/Flowchart/EmitMatlab/custom_library.mflow`), pass
`--block-path DIR` or set `MATFORGE_BLOCK_PATH=DIR1:DIR2` so the
loader can resolve the named library function.

## Editor support

`matlab-lsp` accepts `.mflow` URIs and surfaces loader / builder
diagnostics inline. Open one in your editor; a missing `data.cond`
on an `if` block, an unknown `flow_id`, or a malformed schema all
land on the offending JSON byte range.

## Reference

- **Schema reference (authoritative for IDE save/load):** [`../../docs/flowchart_schema.md`](../../docs/flowchart_schema.md)
- Architecture and design rationale: [`../../docs/flowchart_frontend.md`](../../docs/flowchart_frontend.md)
- Feature status: [`../../docs/feature_status.md`](../../docs/feature_status.md)
