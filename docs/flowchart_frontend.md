# Flowchart Frontend — `.mflow` → AST → MLIR (and back to MATLAB)

Plan for a graphical block-language frontend that consumes `.mflow`
JSON files (produced by the MatForge IDE shown in `tools/`) and
compiles them through the existing `matlab_llvm` pipeline.

This is the concrete implementation plan for roadmap item #6 (Block
language). The high-level rationale lives in
[`roadmap.md`](roadmap.md#6-block-language-visual-nodes--mlir-);
this doc nails down architecture, schema, and phases.

**Status: v1 shipped (Phases 1–5).** `matlabc` and `matlab-lsp`
both accept `.mflow` files; every existing `-emit-*` backend works
unchanged on flowcharts. Two ctest lanes
(`flowchart-cross-backend-tests`, `flowchart-lsp-tests`) plus the
two earlier lanes guard the surface; `feature_status.md` has the
shipped row.

> **Editor / IDE implementers:** the field-by-field contract for
> the `.mflow` JSON format — every block kind, required data
> fields, port conventions, validation rules — is in
> [`flowchart_schema.md`](flowchart_schema.md). Read that one
> when implementing save/load. This doc is the architecture +
> phase log.

---

## 0. Goals

1. `matlabc Sample.mflow` compiles a flowchart program through every
   existing emit path (`-emit-mlir`, `-emit-llvm`, `-emit-c`,
   `-emit-cpp`, `-emit-python`, `-emit-typescript`,
   `-emit-systemverilog`, `-emit-hardware-report`, …) with no per-
   backend code added.
2. `matlabc -emit-matlab Sample.mflow` produces clean MATLAB source
   from the diagram. This is the round-trip the IDE needs to let
   users switch from blocks to text once a diagram outgrows the
   visual surface.
3. The schema is small enough that the MatForge IDE and `matlabc`
   can converge on it without coupling either side to the other's
   data model.

---

## 1. Architecture choice

Two viable shapes were considered:

| | A. `.mflow` → MLIR direct | B. `.mflow` → **AST** → existing pipeline |
|---|---|---|
| Reuse | Backends only | Sema + MLIR lowering + every backend |
| MATLAB emission | Reimplement | **Free** via `formatAST` |
| Type inference | Reimplement | Reuses `lib/Sema` |
| New surface area | Graph→MLIR walker, types, bindings | Graph→AST walker only |

**Decision: B.** The repo already has a stable `TranslationUnit` /
`Function` / `Block` / `IfStmt` / `ForStmt` / `AssignStmt` /
`CallOrIndex` AST (`include/matlab/AST/AST.h`) and a `formatAST`
that pretty-prints any TU back to canonical MATLAB
(`include/matlab/AST/Formatter.h`). Producing a TU is strictly less
work than producing MLIR, and unlocks every existing backend plus
free MATLAB emission.

```
.mflow JSON
    │
    ▼
lib/Flowchart/Loader.cpp        — parse + validate
    │
    ▼
lib/Flowchart/GraphToAST.cpp    — CFG → structured AST
    │
    ▼
TranslationUnit  ──►  formatAST  ──►  .m source        (NEW free path)
    │
    ▼
Resolver → TypeInference → lowerToMLIR
    │
    ▼
all existing -emit-* backends                          (reused unchanged)
```

The new pieces are `lib/Flowchart/` (loader + reducer) and a single
dispatch hook in `tools/matlabc/main.cpp` that picks the frontend
by file extension. Nothing under `lib/AST`, `lib/Sema`, `lib/MLIR`,
or `lib/Parse` changes.

---

## 2. `.mflow` schema

The format used in the existing sample (`Downloads/Sample.mflow`):

```jsonc
{
  "schema": "matforge.flowchart",
  "version": "0.1.0",
  "entry": "main",
  "settings": {
    "columnMajor": true,
    "defaultNumericType": "double",
    "sourceLanguage": "matforge"
  },
  "flows": [
    {
      "id": "flow_main",
      "kind": "program",          // "program" | "function"
      "name": "main",
      "signature": { "inputs": [], "outputs": [] },
      "nodes": [ /* see block kinds below */ ],
      "edges": [ /* control + data edges */ ],
      "layout": { "direction": "TB", "zoom": 1 }
    }
  ]
}
```

### 2.1 Block kinds (from the IDE palette)

The `kind` field on each node maps 1:1 to AST construction:

| Block kind            | AST shape produced                                  | Required `data` fields            |
|-----------------------|-----------------------------------------------------|-----------------------------------|
| `start`               | (no statement; CFG entry marker)                    | —                                 |
| `end`                 | (no statement; CFG exit marker)                     | —                                 |
| `comment`             | dropped (formatter doesn't preserve comments)       | `text`                            |
| `variable`            | `AssignStmt`: `name = value`                        | `name`, `value`                   |
| `assignment`          | `AssignStmt`: `lhs = rhs`                           | `lhs`, `rhs`                      |
| `expression`          | parsed as a top-level statement via existing Parser | `expression`                      |
| `constant`            | `AssignStmt` of a literal                           | `name`, `value`                   |
| `display`             | `ExprStmt` of `disp(expression)`                    | `expression`                      |
| `input`               | `AssignStmt` of `name = input(prompt)`              | `name`, optional `prompt`         |
| `if`                  | `IfStmt` with `Cond` parsed from `data.cond`        | `cond`                            |
| `for`                 | `ForStmt` with `Var` + `Iter` parsed                | `var`, `iter`                     |
| `while`               | `WhileStmt`                                         | `cond`                            |
| `break`               | `BreakStmt`                                         | —                                 |
| `continue`            | `ContinueStmt`                                      | —                                 |
| `return`              | `ReturnStmt`                                        | —                                 |
| `function_definition` | top-level `Function` (body = referenced sub-flow)   | `flow_id`, optional `name`        |
| `function_call`       | `ExprStmt` of `CallOrIndex`                         | `callee`, `args`, optional `lhs`  |
| `subflow_call`        | call into a non-`program` flow                      | `flow_id`, `args`, optional `lhs` |
| `matrix_literal`      | `AssignStmt` with `MatrixLiteral` RHS               | `name`, `rows`                    |
| `custom`              | inlines a user-supplied MATLAB function (see §2.3)  | one of `source` / `path` / `library_id`, plus port→param mapping |

The "schema gaps" called out below (§5) are entirely about which
`data` fields each kind must carry — block kinds themselves are
fixed by the IDE palette.

### 2.2 Ports and edges

```jsonc
{
  "ports": {
    "in":  [ { "id": "in" } ],
    "out": [ { "id": "true" }, { "id": "false" } ]   // for `if`
  }
}
```

Edge shape:

```jsonc
{
  "id": "e_3",
  "kind": "control",                  // "control" | "data"
  "from": { "node": "n_if_1", "port": "true" },
  "to":   { "node": "n_display_1", "port": "in" }
}
```

Port id conventions per kind:

| Kind                  | `in` ports     | `out` ports         |
|-----------------------|----------------|---------------------|
| `start`               | —              | `out`               |
| `end`                 | `in`           | —                   |
| linear (variable, expression, display, input, function_call, …) | `in` | `out` |
| `if`                  | `in`           | `true`, `false`     |
| `for`, `while`        | `in`           | `body`, `done`      |
| `function_definition` | `in`           | `out`               |

`kind: "control"` edges drive the CFG reducer in §4. `kind: "data"`
edges (not used in the v1 sample) are reserved for a future
dataflow-style block extension and ignored by the v1 loader.

### 2.3 Custom blocks (user-supplied MATLAB functions)

A `custom` block embeds a user-written MATLAB function as its
behavior. This is the primary extensibility hook: anything the
fixed palette doesn't cover (a domain-specific filter, an FSM
helper, a numeric primitive) ships as a custom block instead of
forcing a palette change.

Conceptually, a `custom` block is equivalent to a `function_call`
whose callee happens to be defined in the same compilation. The
difference is provenance — the function body comes from one of:

```jsonc
{
  "id": "n_my_filter_1",
  "kind": "custom",
  "label": "My FIR Filter",
  "data": {
    // Exactly one of `source` / `path` / `library_id` must be set:
    "source": "function y = my_filter(x, k)\n  y = k * x + 1;\nend",
    "path":   "blocks/my_filter.m",
    "library_id": "matforge.dsp/fir_4tap",

    // Port-to-parameter mapping (required):
    "name":    "my_filter",        // function name; defaults to file basename
    "inputs":  ["x", "k"],          // in-port id → MATLAB param, in order
    "outputs": ["y"]                // out-port id → MATLAB return, in order
  },
  "ports": {
    "in":  [ { "id": "x" }, { "id": "k" } ],
    "out": [ { "id": "y" } ]
  }
}
```

Three provenance modes — listed in priority order when multiple
are set the loader picks the first and warns:

| Source         | Behavior                                                                                       |
|----------------|------------------------------------------------------------------------------------------------|
| `source`       | Inline MATLAB text. Loader parses it via the existing `Lexer` + `Parser` and inserts the resulting `Function` into the TU. Best for one-off blocks the user typed inside the IDE. |
| `path`         | Reads the `.m` file relative to the `.mflow` location, then same as `source`. Reuses the existing multi-file `ExtraInputs` mechanism (`tools/matlabc/main.cpp:131`) — the file is added to the source manager so debug locations and LSP "go to definition" land on the original `.m` file. |
| `library_id`   | Names a function from a registered block library (e.g. a vendored set of DSP / control primitives). The loader resolves `library_id` to a `.m` file via a search path, then same as `path`. |

Lowering to AST happens in two parts:

1. **Body insertion (once per unique function).** The block's
   `Function` AST node is added to `TU.Functions` at most once,
   keyed by `data.name`. Multiple `custom` blocks that point at
   the same `library_id` or `path` share a single `Function`
   definition.
2. **Call site (per block instance).** The block itself becomes an
   `AssignStmt` of `[<outputs>] = <name>(<inputs>);` where the
   inputs are the variables flowing into each `in` port and the
   outputs are fresh names assigned per the `data.outputs` list.
   The CFG reducer treats the block as a single linear node — same
   shape as `expression` or `function_call`.

### Port/parameter binding rules

- `data.inputs[i]` names the function parameter that receives the
  value flowing into `ports.in[i]`.
- `data.outputs[j]` names the function return assigned to the
  variable flowing out of `ports.out[j]`.
- The loader validates that `len(inputs) == len(ports.in)` and
  `len(outputs) == len(ports.out)`, that each name is a valid
  MATLAB identifier, and that the named function actually accepts
  / returns those arities.

### Type checking and pragmas

Because the function body is regular MATLAB AST, every existing
mechanism applies unchanged:

- The existing `Resolver` + `TypeInference` typecheck the body and
  the call site, with the same diagnostics a hand-written
  `function_call` would get.
- HDL pragmas (`% hdl: port(name, fi, signed, W, F)`) inside the
  function body work as-is, so a `custom` block can be
  `-emit-systemverilog`-ready.
- The block library can ship `.m` files with pragmas pre-applied,
  letting non-HDL users drop in synthesizable primitives without
  understanding the pragma syntax.

### Custom block vs. `function_definition` / `subflow_call`

These three look similar but solve different problems:

| Block                 | Function body comes from                     | Use when                                                    |
|-----------------------|----------------------------------------------|-------------------------------------------------------------|
| `function_definition` | another flow in the same `.mflow` (visual)   | The user wants the helper itself to be drawn graphically.   |
| `subflow_call`        | another flow, called from this one           | A graph-defined helper is invoked from multiple sites.      |
| `custom`              | inline MATLAB / `.m` file / library          | The behavior is easier (or only practical) to express in MATLAB text — DSP kernels, dense numerics, library code. |

A future extension could let the IDE "convert custom block to
sub-flow" by lifting the MATLAB body back into a graph, but that
needs the MATLAB → blocks direction (out of scope for v1; see §8).

---

## 3. Loader (`lib/Flowchart/Loader.cpp`)

Plain C++ JSON reader (the project doesn't have a JSON dep yet —
either vendor a small single-header reader like `nlohmann/json` or
write a ~300-line recursive-descent reader; the latter is consistent
with the rest of the repo's "no third-party deps" posture).

The loader produces a typed in-memory representation:

```cpp
struct FlowDoc {
  std::string Entry;
  std::vector<Flow> Flows;
  Settings Settings;
};

struct Flow {
  std::string Id, Name, Kind;
  Signature Sig;
  std::vector<Node> Nodes;
  std::vector<Edge> Edges;
};

struct Node {
  std::string Id, Kind, Label;
  std::map<std::string, std::string> Data;
  Ports In, Out;
};

struct Edge {
  std::string Id, Kind;
  Endpoint From, To;
};
```

Validation pass before any AST construction:

1. Unique node ids per flow.
2. Every edge endpoint resolves; `from.port` is in `node.out`, etc.
3. Exactly one `start` and one `end` node per `program` flow.
4. Sub-flows (`kind: "function"`) have a clean signature.
5. Reachability: nodes not reachable from `start` are flagged
   warn-and-skip (the example `Sample.mflow` has 5 disconnected
   palette nodes; ignore them — don't error).
6. Every required `data` field for the node kind is present
   (table in §2.1).

All validation errors are reported through the existing
`DiagnosticEngine` so the LSP / IDE can render them inline.

---

## 4. Graph-to-AST reducer (`lib/Flowchart/GraphToAST.cpp`)

The hard piece. Standard structured-control reduction:

1. Build a CFG from the control edges.
2. Compute dominators / post-dominators.
3. Reduce regions bottom-up using Hammock / T1-T2 reduction:
   - Linear chain of single-in / single-out nodes → `Block` of
     `Stmt`s in order.
   - `if` node with two branches that re-converge at a common
     post-dominator J → `IfStmt` whose `Then`/`Else` are the
     reduced sub-regions and whose continuation is everything
     dominated by J.
   - `for` / `while` node where `body` post-dominates the head and
     loops back → `ForStmt` / `WhileStmt`.
4. The final reduced region of a `program` flow is a single `Block`,
   wrapped in a `Script` (or a `Function` named after the flow's
   `name`).

For `Sample.mflow` (linear: `start → variable → expression →
display → end`), the reducer produces:

```matlab
v = 1;
v = v + 1;
disp(v);
```

For sub-flows (`function_definition` block referencing `flow_id`),
each non-`program` flow becomes a top-level `Function` in the same
`TranslationUnit`. `function_call` blocks lower to `CallOrIndex`.
The TU already supports multiple `Function`s, so this is purely
additive.

### 4.1 Per-block AST construction

For block kinds whose `data` carries a string (`expression`,
`cond`, `iter`, `value`, `rhs`, …), reuse the existing
`Lexer` + `Parser` to convert the string into an `Expr` or `Stmt`:

- `data.expression = "v = v + 1"` → `Parser::parseStmt()` →
  `AssignStmt`.
- `data.cond = "v > 0"` → `Parser::parseExpr()` → `BinaryOpExpr`.
- `data.iter = "1:10"` → `Parser::parseExpr()` → `RangeExpr`.

This means the textual frontend's expression grammar IS the
expression grammar inside blocks, with no duplication. Diagnostics
from the inner parse carry a synthetic `SourceRange` that points
at "inside block `n_expression_1`, field `expression`" so errors
land on the right block in the IDE.

---

## 5. Schema gaps in v0.1.0

The current `Sample.mflow` (saved by the IDE today) has empty
`data` blocks for several kinds. The IDE and compiler need to
agree on these before the loader can produce useful programs:

| Block               | Missing fields                                                   |
|---------------------|------------------------------------------------------------------|
| `display`           | `data.expression` — what to print                                |
| `if`                | `data.cond` — branch condition                                   |
| `for`               | `data.var`, `data.iter` — induction variable and range           |
| `while`             | `data.cond`                                                      |
| `input`             | `data.name` — assigned variable; optional `data.prompt`          |
| `function_definition` | `data.flow_id` (target sub-flow), `data.name`, params/returns  |
| `function_call`     | `data.callee`, `data.args` (CSV expression list), `data.lhs`     |
| `custom`            | exactly one of `source` / `path` / `library_id`, plus `name`, `inputs`, `outputs` (see §2.3) |

These are tracked in the loader's required-fields table (§2.1).
v0.1.0 of the schema is fixed once we publish this doc.

The disconnected palette nodes seen in the sample (`n_if_1`,
`n_for_1`, `n_input_1`, `n_function_definition_1`,
`n_function_call_1`) are unreachable from `start` and are ignored
with a warning, not an error.

---

## 6. CLI integration

`tools/matlabc/main.cpp` already dispatches by mode, not by input
shape. Add an extension check next to the current `loadFile` site
(`tools/matlabc/main.cpp:5207`):

```cpp
if (endsWith(Opts.InputPath, ".mflow")) {
  TU = matlab::flowchart::loadAndBuildAST(Opts.InputPath, AstCtx, Diag);
} else {
  // existing Lex → Parse path
}
```

After this point everything is identical: same Sema, same lowering,
same backends. Add one new `-emit-matlab` mode that runs `formatAST`
on the TU and writes to stdout — a 10-line addition.

CLI surface after this change:

| Command                                    | Result                  |
|--------------------------------------------|-------------------------|
| `matlabc -emit-c Sample.mflow`             | C source                |
| `matlabc -emit-systemverilog foo.mflow`    | SV (with HDL pragmas)   |
| `matlabc -emit-matlab Sample.mflow`        | `.m` source (NEW)       |
| `matlabc -emit-mlir Sample.mflow`          | MLIR module             |
| `matlabc Sample.mflow`                     | LLVM IR (default mode)  |

`-emit-matlab` also works on `.m` input — it's just `parse →
formatAST`, which already exists internally as the `-format` path
but writes to stdout instead of in-place rewriting.

---

## 7. Phases

### Phase 1 — schema + loader (shipped)

- `lib/Flowchart/Loader.cpp` + `include/matlab/Flowchart/Loader.h`:
  hand-rolled JSON reader, validation, in-memory `FlowDoc`. No
  third-party deps.
- `matlabc -dump-flow FILE.mflow` introspects loaded flows.
- Test lane `flowchart-tests` (9 fixtures: linear / disconnected
  palette nodes / 7 error cases).

### Phase 2 — linear CFG → AST (shipped)

- `lib/Flowchart/GraphToAST.cpp` walks the linear `start → ... →
  end` chain and synthesizes a `TranslationUnit`.
- Per-block translators for `variable`, `expression`, `display`,
  `input`, `assignment`, `function_call`, `constant`,
  `matrix_literal`. String-form data fields ride through the
  existing `Lexer` + `Parser` so the grammar isn't duplicated.
- `matlabc -emit-matlab` (alias `-emit-m`) prints the round-tripped
  MATLAB. Every existing `-emit-*` mode also accepts `.mflow`.
- Test lane `flowchart-emit-matlab-tests`.

### Phase 3 — structured control flow (shipped)

- Reducer in `lib/Flowchart/GraphToAST.cpp` handles `if` / `for` /
  `while` / `break` / `continue` / `return`, including arbitrary
  nesting. Each `if` is reduced via two-pointer `findJoin` (with
  memoization through `IfJoinCache`); loops use the loop head's id
  as a `Stop` boundary so the body's back-edge cleanly closes the
  walk.
- `if`-without-else is detected (false branch goes directly to the
  join) and the empty `Else` block is dropped before formatting.
- Irreducible CFGs surface as `findJoin → ""`, which fans out to a
  diagnostic at the offending `if` — no synthetic `goto`.
- Direct AST construction for `IfStmt` / `ForStmt` / `WhileStmt`
  (no source-string round-trip) keeps source ranges precise. The
  inner `data.cond` / `data.iter` are still parsed via the existing
  Pratt parser by synthesizing a small `EXPR;\n` buffer.
- Test fixtures: `if_else`, `if_no_else`, `for_loop`,
  `nested_for_if`, `while_break`. All seven happy-path Phase 2/3
  fixtures execute end-to-end through `-emit-c` and produce
  correct values (`55`, `30`, `3`, `9.28`, ...).

### Phase 4 — sub-flows / function_definition / function_call (shipped)

- `liftFunctionFlow` in `lib/Flowchart/GraphToAST.cpp` walks each
  non-`program` flow with the same `Builder` the entry flow uses
  and emits a top-level `Function` AST. The whole Phase 2/3
  control-flow surface (if / for / while / break / nested) works
  inside function bodies for free.
- `function_definition` blocks are visual markers — they emit no
  statement but validate the `data.flow_id` cross-reference at
  AST-build time, so a typo in the IDE fails fast with a precise
  diagnostic.
- `subflow_call` blocks resolve `data.flow_id` to the target
  flow's `name` and emit a regular `lhs = name(args);` call. The
  lifted `Function` carries the matching signature so Sema and
  every backend treat it identically to a hand-written user fn.
- Duplicate function names across flows are rejected at the TU
  level (mirrors the textual frontend's same-name shadowing
  constraint).
- Test fixtures: `multi_flow.mflow` (`doubled(10) = 20`),
  `subflow_call.mflow` (Pythagorean `hypot2(3,4) = 5`),
  `Errors/missing_subflow.mflow`, `Errors/unsupported_funcdef.mflow`.

### Phase 4b — custom blocks (shipped)

- All three provenance modes from §2.3 implemented in
  `lib/Flowchart/GraphToAST.cpp::handleCustom`:
  - `data.source` — inline MATLAB text added to the `SourceManager`
    as `<flow:NODEID:source>`.
  - `data.path` — relative path resolved against the `.mflow`
    file's directory, loaded via `SM.loadFile` so diagnostics and
    LSP land on the real `.m` file.
  - `data.library_id` — resolved against the block search path
    (CLI `--block-path DIR` repeatable + `MATFORGE_BLOCK_PATH`
    colon-separated env var, in that order).
- Function insertion is deduped at the `TUContext` level: many
  blocks pointing at the same library function share one
  `Function` AST node.
- Optional arity validation: when the IDE supplies `data.inputs`
  / `data.outputs` arrays, the loader stores them in the new
  `Node::DataArrays` map and `handleCustom` rejects mismatches
  before lowering, with a diagnostic naming both the declared and
  parsed arities.
- Test fixtures: `custom_inline` (`gain_plus_bias(7,3,1)=22`),
  `custom_path` (sibling `clamp.m` with if/elseif/else,
  `clamp(42,0,10)=10`), `custom_library` (two callers of
  `dsp/scale.m` deduped to one Function, `scale(scale(10,4),2)=80`),
  plus error cases for missing/conflicting provenance and arity
  mismatch.

**Note.** The originally-planned HDL-pragma SV exit criterion is
deferred to Phase 5: nothing about Phase 4b prevents `% hdl: port(...)`
custom blocks (they're just plain MATLAB), but the cross-backend
SV golden lives in the Phase 5 test corpus.

### Phase 5 — round-trip tests + LSP hooks (shipped)

- New ctest lane `flowchart-cross-backend-tests`: for every
  fixture under `test/Flowchart/EmitMatlab/`, runs `-emit-X`
  on both the `.mflow` and the round-tripped MATLAB source
  (the output of `-emit-matlab`) and asserts blank-line-
  insensitive equivalence across X ∈ {c, cpp, python,
  typescript}. 12 fixtures × 4 backends = 48 checks per CI run.
  Diff uses `diff -uB` because the C/Python/TS emitters preserve
  source-paragraph blanks via SourceManager gap inspection (only
  fires on the .m's single buffer, not on the .mflow's per-block
  synthetic buffers — same emitted *structure*, different
  whitespace).
- New ctest lane `flowchart-lsp-tests`: drives `matlab-lsp`
  via stdio with `textDocument/didOpen` events for valid and
  malformed `.mflow` payloads. Asserts the LSP publishes the
  expected diagnostic count + message substring.
- `matlab-lsp` dispatches on the `.mflow` URI extension in
  `reparse`: routes through `flowchart::loadMflow` +
  `flowchart::buildAST` and feeds the resulting TU to the same
  Sema pipeline. `BuildOptions::MflowDirectory` is derived from
  the URI so `data.path` resolves relative to the file the user
  opened. (`--block-path` resolution for `library_id` is a v2
  concern — the LSP would need a config option to thread it
  through.)
- Test corpus growth: total ctest lanes went from 18 → 20.

**Exit criterion met.** Roadmap item #6 v1 scope is shipped.

---

## 8. Out of scope (for v1)

These are deliberate deferrals, mirroring the roadmap entry:

- **Round-trip text → blocks.** v1 is one-way (blocks → MATLAB).
  Going the other direction needs a layout heuristic that decides
  where each generated block lands on the canvas.
- **Continuous-time / sample-rate-different simulation.** Every
  `.mflow` program is sample-rate-fixed discrete, same as the SV
  pipeline.
- **2-D / image-pipeline blocks.** Roadmap item #7 territory.
- **Irreducible (unstructured) CFGs.** Refuse with a diagnostic;
  no synthetic loop / multi-entry handling.
- **Data edges.** v1 ignores `kind: "data"`; the dataflow-style
  block extension is a separate phase.
- **`.mflow` as an output target.** No `-emit-mflow` — codegen
  goes one way only, blocks → MLIR / MATLAB.

---

## 9. Risks and open questions

1. **JSON dependency.** Either vendor a small header (~3k LOC) or
   hand-roll a recursive-descent reader. The repo currently has no
   third-party deps; hand-rolling matches house style. Decide at
   Phase 1 kickoff.
2. **Source locations into block fields.** Diagnostics from the
   inner `Lexer`/`Parser` need a `SourceRange` that points back
   to the offending `.mflow` byte range, not just "node id +
   field name". The `SourceManager` accepts byte offsets, so
   tracking the JSON byte position of each `data.*` value during
   load is enough — record it in the loader.
3. **Layout preservation on round-trip.** v1 is one-way, so the
   `ui.position` field is preserved through the loader but not
   used. If we ever do MATLAB → blocks, layout becomes a first-
   class problem.
4. **Comment blocks.** The `comment` block kind is dropped today
   (the formatter doesn't preserve comments either). Revisit
   when the formatter learns comment retention.
5. **Custom-block library trust boundary.** `library_id` resolves
   against a search path under user / project control —
   essentially "include any `.m` file from this directory." This
   is the same trust posture as `addpath` in MATLAB or `-I` in C,
   so it's not a regression, but it's worth documenting that
   block libraries execute as if their `.m` files were typed
   directly into the user's program.
6. **Inline `source` size limit.** A custom block's `data.source`
   field can in principle hold an entire program. The IDE should
   warn the user past some threshold (~200 lines) and suggest
   converting to a `path` block, since a giant string inside JSON
   is hard to diff and review. Compiler-side: no hard limit, but
   the loader should refuse `source` fields above 1 MB to keep
   parse-error blast radius bounded.

---

## 10. Update cadence

This doc gets demoted from "plan" to "design + behavior notes" once
Phase 5 ships, at which point the implemented bits move to
`feature_status.md` and the file becomes the canonical reference for
the `.mflow` frontend.
