# `.mflow` Flowchart Frontend — Tutorial

`.mflow` is a graphical block language: a flowchart of nodes and edges,
saved as JSON by the MatForge IDE, that `matlabc` compiles through the
**same** pipeline as `.m` text. There is no separate backend — a
`.mflow` is loaded into the exact `TranslationUnit` the textual parser
produces, so Sema, MLIR lowering, and every `-emit-*` backend work on
it unchanged. You draw control flow as boxes; the compiler reduces the
graph to structured AST (`if` / `for` / `while` / functions) and emits
C, C++, Python, TypeScript, SystemVerilog, LLVM, or MLIR.

This page is the hands-on guide to the **flowchart** dialect and
doubles as the overview of the `.mflow` family.

Design reference: [`../flowchart_frontend.md`](../flowchart_frontend.md).
Field-by-field schema: [`../flowchart_schema.md`](../flowchart_schema.md).
Examples: [`../../examples/mflow/`](../../examples/mflow/).

---

## The three `.mflow` dialects

`.mflow` is one file format with three dialects. They share the JSON
envelope (`schema`, `version`, `flows`, `nodes`, `edges`) but mean
different things and compile through different lowerings:

| Dialect | What it models | Examples | Tutorial |
|---|---|---|---|
| **flowchart** (this doc) | Control-flow programs that mirror `.m` source: `start` / `if` / `for` / `while` / function sub-flows | [`examples/mflow/`](../../examples/mflow/) | (this page) |
| **signal_flow** | Simulink-like block diagrams (`signal_*` blocks wired by data edges, integrated over time) | [`examples/mflowlink/`](../../examples/mflowlink/) | [`embedded_coder_tutorial.md`](embedded_coder_tutorial.md) |
| **state_chart** | Hierarchical state charts (states, transitions, super-steps) | [`examples/stateflow/`](../../examples/stateflow/) | [`stateflow_tutorial.md`](stateflow_tutorial.md) |

The flowchart dialect is the default (`settings.kind` absent or
`"control_flow"`). The other two are additive extensions of the same
schema — see the schema doc's "Signal-flow extensions" section for how
`settings.kind` selects between them. The rest of this tutorial is
about the flowchart dialect only.

---

## Anatomy of a `.mflow` file

A document is a JSON object with a fixed envelope and one or more
*flows*. The smallest interesting program is
[`examples/mflow/hello.mflow`](../../examples/mflow/hello.mflow):

```jsonc
{
  "entry": "main",                 // name of the program-kind entry flow
  "schema": "matforge.flowchart",  // must be this exact literal
  "version": "0.1.0",
  "settings": {
    "columnMajor": true,           // MATLAB storage order
    "defaultNumericType": "double"
  },
  "flows": [
    {
      "id": "flow_main",
      "kind": "program",           // "program" | "function"
      "name": "main",
      "signature": { "inputs": [], "outputs": [] },
      "nodes": [ /* the boxes */ ],
      "edges": [ /* the arrows */ ]
    }
  ]
}
```

### Flows

A `flow` is one graph. Its `kind` is either:

- `program` — lifts to the script body (the entry flow must be
  `program`; exactly one `start`, at least one `end`).
- `function` — lifts to a top-level `Function`. Its `signature.inputs`
  / `signature.outputs` become the MATLAB parameters and returns; the
  function's name is the flow's `name` (not its `id`).

Multiple flows live in one document — that is how a program references
helper functions (see [is_old / factorial](#sub-flows--functions-is_old-factorial)).

### Nodes

Each node is a box with an `id` (unique within its flow), a `kind`, and
per-kind `data`. The `kind` maps 1:1 to an AST construction. Here is
`hello.mflow`'s `display` block:

```jsonc
{
  "id": "d1",
  "kind": "display",
  "data": { "expression": "'Hello, world!'" },
  "ports": {
    "in":  [ { "id": "in" } ],
    "out": [ { "id": "out" } ]
  }
}
```

The block kinds and the `data` fields each requires:

| Kind | Lowers to | Required `data` |
|---|---|---|
| `start` / `end` | CFG markers (no statement) | — |
| `variable` / `constant` | `name = value;` | `name`, `value` |
| `assignment` | `lhs = rhs;` | `lhs`, `rhs` |
| `expression` | a parsed top-level statement | `expression` |
| `display` | `disp(expression);` | `expression` |
| `input` | `name = input('prompt');` | `name` (`prompt` optional) |
| `matrix_literal` | `name = [rows];` | `name`, `rows` |
| `function_call` | `lhs = callee(args);` | `callee` (`args`, `lhs` optional) |
| `if` | two-way branch | `cond` |
| `for` | range loop | `var`, `iter` |
| `while` | conditional loop | `cond` |
| `break` / `continue` / `return` | loop/function exit | — |
| `switch` | multi-way branch | `discriminant`, `cases` |
| `try` | error-handling branch | (`catch_var` optional) |
| `function_definition` | visual marker for a sub-flow | `flow_id` |
| `subflow_call` | call a sub-flow by id | `flow_id` |
| `custom` | inline a user MATLAB function | one of `source` / `path` / `library_id`, plus `callee`/`name` |
| `comment` | dropped (formatter has no comment retention) | `text` |

String `data` fields (`expression`, `cond`, `iter`, `value`, `rhs`, …)
are fed to the **same** MATLAB lexer + parser the textual frontend
uses, so the expression grammar inside a block is identical to MATLAB
text — no duplicate grammar.

### Ports and edges

Nodes connect through named ports. Port id conventions per kind:

| Kind | `in` ports | `out` ports |
|---|---|---|
| `start` | — | `out` |
| `end` | `in` | — |
| linear (`variable`, `display`, `expression`, `function_call`, …) | `in` | `out` |
| `if` | `in` | `true`, `false` |
| `for`, `while` | `in` | `body`, `done` |

An edge wires one node's out-port to another's in-port:

```jsonc
{
  "id": "e_1",
  "kind": "control",                         // "control" | "data"
  "from": { "node": "s",  "port": "out" },
  "to":   { "node": "d1", "port": "in" }
}
```

Only `kind: "control"` edges drive the compiler's CFG reducer.
`kind: "data"` edges are **reserved** for the future dataflow extension
and ignored by the v1 flowchart loader.

So `hello.mflow` is the linear chain
`start → display('Hello, world!') → expression(fprintf(...)) → end`,
which reduces to:

```matlab
disp('Hello, world!');
fprintf('Greetings from matlab_llvm!\n');
```

---

## Compiling and running

`matlabc` picks the frontend by file extension — pass a `.mflow` to any
mode and it loads through the flowchart path before Sema. Everything
downstream is identical to `.m`.

Build and run the LLVM lane (same recipe as a `.m` program):

```bash
build/matlabc -emit-llvm examples/mflow/hello.mflow > /tmp/hello.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/hello.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/hello
/tmp/hello
# Hello, world!
# Greetings from matlab_llvm!
```

Every other backend accepts `.mflow` too, with no per-backend code:

```bash
build/matlabc -emit-c              examples/mflow/factorial.mflow
build/matlabc -emit-cpp            examples/mflow/factorial.mflow
build/matlabc -emit-python         examples/mflow/factorial.mflow
build/matlabc -emit-typescript     examples/mflow/factorial.mflow
build/matlabc -emit-systemverilog  examples/mflow/factorial.mflow
build/matlabc -emit-mlir           examples/mflow/factorial.mflow
```

Two flowchart-specific modes help while authoring:

```bash
# Round-trip the diagram to clean MATLAB source.
build/matlabc -emit-matlab examples/mflow/factorial.mflow

# Validate + dump the parsed FlowDoc structure (no AST build).
build/matlabc -dump-flow   examples/mflow/factorial.mflow
```

---

## Worked examples

### Linear chain — hello (`examples/mflow/hello.mflow`)

The simplest shape: `start` → linear blocks → `end`, one control edge
between each pair. The `display` block carries
`data.expression = "'Hello, world!'"` and the `expression` block carries
a full statement (`fprintf('Greetings from matlab_llvm!\n')`). No
branching, no loops — the reducer emits a flat `Block` of statements in
edge order.

### Loops — for_loop (`examples/mflow/for_loop.mflow`)

A `for` block has one `in` port and two out-ports, `body` and `done`.
The loop body chain must end with a **back-edge** to the loop head's
`in` port; the `done` port carries the continuation:

```jsonc
{
  "id": "for1", "kind": "for",
  "data": { "var": "i", "iter": "1:10" },
  "ports": {
    "in":  [ { "id": "in" } ],
    "out": [ { "id": "body" }, { "id": "done" } ]
  }
}
```

The relevant edges: `for1:body → step` (the accumulate block
`total = total + i`), then `step:out → for1:in` (the back-edge that
closes the loop), and `for1:done → lbl` (the continuation toward the
final `disp(total)`). The reducer recognises the back-edge to the head
and emits:

```matlab
total = 0;
for i = 1:10
    total = total + i;
end
disp('sum(1..10) =');
disp(total);
```

### Branching — even_odd (`examples/mflow/even_odd.mflow`)

An `if` inside a `for`, demonstrating branch reconvergence. The `if`
block's out-ports are `true` and `false`:

```jsonc
{
  "id": "n_if_1", "kind": "if",
  "data": { "cond": "mod(i, 2) == 0" },
  "ports": {
    "in":  [ { "id": "in" } ],
    "out": [ { "id": "true" }, { "id": "false" } ]
  }
}
```

The `true` port flows to the even branch (`fprintf('%g is even\n', i)`
then `even_count = even_count + 1`); the `false` port flows to the odd
branch. Both branch tails edge back to `n_for_1:in` (the shared
reconvergence — they re-join at the loop head). The reducer's `findJoin`
computes the common reconvergence point and produces a structured
`if/else` inside the `for`. This example was itself auto-generated from
`even_odd.m` via `-emit-mflow` (see [round-tripping](#round-tripping-m--mflow)).

### Sub-flows / functions — is_old, factorial

A function is just another flow with `kind: "function"`. The program
references it two ways:

**`function_definition`** is a visual marker — it emits no statement but
validates the cross-reference and tells the IDE to draw the helper.
[`is_old.mflow`](../../examples/mflow/is_old.mflow) has a program flow
with an `fd` node pointing at the helper flow:

```jsonc
{ "id": "fd", "kind": "function_definition",
  "data": { "flow_id": "flow_is_old" } }
```

The second flow defines the function body and its signature:

```jsonc
{
  "id": "flow_is_old",
  "kind": "function",
  "name": "is_old",
  "signature": { "inputs": ["age"], "outputs": ["r"] },
  "nodes": [
    { "id": "fs", "kind": "start", ... },
    { "id": "fa", "kind": "assignment",
      "data": { "lhs": "r", "rhs": "age > 18" } },
    { "id": "fe", "kind": "end", ... }
  ]
}
```

The program's `display` blocks then call it by name
(`is_old(10)`, `is_old(18)`, `is_old(25)`). The whole Phase 2/3
control-flow surface works **inside** function bodies, so a sub-flow can
itself contain `if` / `for` / `while`.

[`factorial.mflow`](../../examples/mflow/factorial.mflow) shows a
recursive helper: its `flow_fact` flow has an `if` on `n <= 1` whose
`true` branch sets `y = 1` and whose `false` branch sets
`y = n * fact(n - 1)` — a self-call inside the function body. Both
branches re-join at the function `end`. Compiling and running the C
output prints `1 2 6 24 120 720`.

### Custom blocks — custom_inline_gain & custom_clamp

A `custom` block embeds a user-written MATLAB function as its behavior —
the extensibility hook for anything the fixed palette doesn't cover. It
is equivalent to a `function_call` whose callee is defined in the same
compilation; the difference is *provenance*. Exactly one of three
fields supplies the body:

- `source` — inline MATLAB text typed into the JSON.
- `path` — a `.m` file resolved relative to the `.mflow` location.
- `library_id` — a function resolved against a block search path
  (`--block-path DIR` flags first, then `MATFORGE_BLOCK_PATH`).

**Inline `source`** —
[`custom_inline_gain.mflow`](../../examples/mflow/custom_inline_gain.mflow)
carries the whole function in `data.source`:

```jsonc
{
  "id": "cb", "kind": "custom",
  "data": {
    "name": "gain_plus_bias",
    "callee": "gain_plus_bias",
    "args": "sample, 4, 1",
    "lhs": "scaled",
    "inputs":  ["x", "k", "b"],
    "outputs": ["y"],
    "source": "function y = gain_plus_bias(x, k, b)\n    y = k * x + b;\nend\n"
  }
}
```

The loader parses the `source` through the normal lexer/parser, inserts
the resulting `Function` into the TU, and the block becomes the call
`scaled = gain_plus_bias(sample, 4, 1);`. The optional `inputs` /
`outputs` arrays let the loader validate arity against the parsed
signature (here 3 inputs, 1 output) before lowering. The program prints
`gain_plus_bias(12, 4, 1) =` then `49`.

**`path` provenance, shared body** —
[`custom_clamp.mflow`](../../examples/mflow/custom_clamp.mflow) has
*three* custom blocks all pointing at the same sibling file:

```jsonc
{ "id": "cb1", "kind": "custom",
  "data": { "name": "clamp", "callee": "clamp",
            "args": "42, 0, 10", "lhs": "high",
            "path": "blocks/clamp.m" } }
```

`blocks/clamp.m` is an ordinary MATLAB function with `if/elseif/else`:

```matlab
function y = clamp(x, lo, hi)
    if x < lo
        y = lo;
    elseif x > hi
        y = hi;
    else
        y = x;
    end
end
```

The body is inserted **once**, deduped by `callee` name; the three
blocks (`cb1`, `cb2`, `cb3`) all call the single inserted `clamp`
function. Because `path` runs the file through the SourceManager,
diagnostics and LSP "go to definition" land on the real `clamp.m`.
Build and run:

```bash
build/matlabc -emit-c examples/mflow/custom_clamp.mflow > /tmp/cc.c
cc /tmp/cc.c runtime/matlab_runtime.c -o /tmp/cc -lm && /tmp/cc
# clamp(42, 0, 10) = 10
# clamp(-3, 0, 10) = 0
# clamp(5,  0, 10) = 5
```

Because a custom block's body is regular MATLAB AST, HDL pragmas
(`% hdl: port(...)`) inside it work as-is, so a custom block can be
`-emit-systemverilog`-ready and a block library can ship synthesizable
primitives with pragmas pre-applied.

---

## Round-tripping `.m` ↔ `.mflow`

The relationship is bidirectional:

- **`.mflow` → `.m`** is the `-emit-matlab` mode (reduces the graph and
  pretty-prints the TU).
- **`.m` → `.mflow`** is the inverse `-emit-mflow` mode, which serialises
  *any* `TranslationUnit` back to a canonical flowchart document:

```bash
build/matlabc -emit-mflow examples/factorial.m > examples/mflow/factorial.mflow
build/matlabc -emit-mflow examples/even_odd.m  > examples/mflow/even_odd.mflow
```

The walker is the structural inverse of the reducer: linear statements
map to block kinds (`name = literal` → `variable`, a `MatrixLiteral` RHS
→ `matrix_literal`, `disp(...)` → `display`, other calls →
`function_call`); `IfStmt` → `if` block with `true`/`false` ports;
`ForStmt`/`WhileStmt` → loop blocks with a `body`/`done` shape and an
explicit back-edge; each `Function` becomes a `function`-kind sub-flow.
`elseif` chains re-fold into nested `if` blocks on the false branch (no
`elseif` block kind needed).

Output is IDE-canonical JSON: 2-space indent, alphabetical keys,
blank-line empty arrays — so it diffs cleanly against IDE re-saves.
Auto-layout assigns column positions (`x=200, y=index*120`); the IDE
re-layouts on open. The round-trip is **idempotent**:
`.m → .mflow → .m → .mflow` produces a byte-identical second `.mflow`
from iteration 2 onward. `--preserve-layout PATH` merges an existing
file's `ui.position` values into a fresh emission by matching node ids,
so re-emitting doesn't clobber manual placements.

---

## Debugging & tooling

The flowchart frontend is wired into the same debug and editor surfaces
as `.m`:

- **DAP debugging.** `matlabc -dap` accepts a `.mflow` entry point.
  Each synthesized statement's source range is rewritten to the
  originating block's byte offset in the JSON, so breakpoints set on a
  block's JSON line resolve and fire when execution reaches it. Stack
  frames carry a `[block:<id>]` tag so the IDE can highlight the active
  block on the canvas, and step-over collapses a multi-statement block
  into a single logical step.
- **LSP.** `matlab-lsp` accepts `.mflow` URIs and routes them through
  the flowchart loader + builder into the same Sema pipeline. Loader and
  reducer diagnostics — a missing `data.cond` on an `if`, an unresolved
  `flow_id`, a malformed schema, a custom-block arity mismatch — surface
  inline on the offending JSON byte range. `data.path` resolves relative
  to the opened file; both DAP and LSP read `initializationOptions.blockPath`
  to configure `library_id` search paths.

---

## Limitations & status

**Status: v1 shipped (Phases 1–7), with Phase 8 polish complete.** Six
ctest lanes guard the surface: loader, emit-matlab, cross-backend
equivalence, LSP, DAP, and emit-mflow round-trip. Phases cover the
loader/validation, linear and structured control-flow reduction,
sub-flows/functions, custom blocks, cross-backend round-trip, DAP, and
the `-emit-mflow` inverse. Phase 8 added block-id stack frames,
per-block step granularity, IDE-supplied block paths, `--preserve-layout`,
and `switch` / `try`-`catch` block kinds.

Carve-outs and reserved areas:

- **Data edges** (`kind: "data"`) are reserved and ignored by the v1
  flowchart loader — they belong to the signal-flow dialect / a future
  dataflow extension.
- **Irreducible (unstructured) CFGs** are refused with a diagnostic;
  there are no synthetic `goto`s. Loops must have a single back-edge to
  their head; `if` branches must reconverge at a common join (or both
  terminate at `end` / `break` / `continue` / `return`).
- **`comment` blocks are dropped** — the formatter doesn't retain
  comments, so they wouldn't survive the `.m` round-trip anyway.
- **`ui.position`** is round-trip-only metadata; the compile path
  ignores it.
- **`classdef`** has no dedicated flow kind; on `-emit-mflow` a class
  emits as fallback text inside an `expression` block.
- **Continuous-time / multi-rate simulation** is not part of the
  flowchart dialect — every flowchart program is fixed-rate discrete.

---

## See also

- Design: [`../flowchart_frontend.md`](../flowchart_frontend.md); schema: [`../flowchart_schema.md`](../flowchart_schema.md)
- Block-diagram (signal_flow) dialect: [`embedded_coder_tutorial.md`](embedded_coder_tutorial.md)
- State-chart dialect: [`stateflow_tutorial.md`](stateflow_tutorial.md)
- Examples: [`../../examples/mflow/`](../../examples/mflow/) (with its [README](../../examples/mflow/README.md))
- Compile-and-run basics: [`../build_and_run.md`](../build_and_run.md)
