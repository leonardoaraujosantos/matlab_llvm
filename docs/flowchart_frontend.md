# Flowchart Frontend — `.mflow` → AST → MLIR (and back to MATLAB)

Plan for a graphical block-language frontend that consumes `.mflow`
JSON files (produced by the MatForge IDE shown in `tools/`) and
compiles them through the existing `matlab_llvm` pipeline.

This is the concrete implementation plan for roadmap item #6 (Block
language). The high-level rationale lives in
[`roadmap.md`](roadmap.md#6-block-language-visual-nodes--mlir-);
this doc nails down architecture, schema, and phases.

**Status: v1 shipped (Phases 1–7).**

- Phases 1–5 (loader / linear / structured control flow / sub-flows
  / custom blocks / cross-backend round-trip / LSP).
- Phase 6: `matlabc -dap` accepts `.mflow` programs; breakpoints set
  on JSON lines fire correctly.
- Phase 7: `-emit-mflow` is the inverse of the loader — any
  TranslationUnit serialises back to a canonical `.mflow` document
  (idempotent on repeat emission).

Six ctest lanes guard the surface (loader / emit-matlab /
cross-backend / LSP / DAP / emit-mflow). `feature_status.md` has
the shipped row.

**Phase 8 (in progress).** 8a (block-id stack frames) and 8c
(`--block-path` via DAP / LSP `initializationOptions`) shipped.
Remaining: 8b (per-block step granularity), 8d (`-emit-mflow
--preserve-layout`), 8e (`switch` / `try`-`catch` block kinds).
See §7 Phase 8 for the breakdown.

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

### Phase 6 — DAP support (shipped)

- `matlabc -dap` accepts a `.mflow` entry point. The DAP
  `compileProgram` path dispatches on the extension (the same
  way `matlabc` and `matlab-lsp` do) and routes through the
  flowchart loader + builder before Sema.
- Source-location remap in `lib/Flowchart/GraphToAST.cpp`: every
  per-block synthesized statement gets its `Range.Begin`
  rewritten to the originating block's `.mflow` byte offset, so
  breakpoints set on a block's JSON line resolve correctly via
  the existing `G.PathToFileId` table and fire when execution
  reaches the synthesized statements. Same fix benefits the LSP.
- Synthesized per-block buffers (`<flow:NODEID>`) are filtered
  from the DAP `loadedSources` registration so the IDE doesn't
  see them as openable files.
- Sibling-`.m` autoload is skipped for `.mflow` entries —
  flowchart programs reference helpers through `function`-kind
  sub-flows or `custom` blocks, not ad-hoc sibling files.
- `MATFORGE_BLOCK_PATH` is honoured for `library_id` custom
  blocks. (CLI `--block-path` is wired for `matlabc`-direct;
  threading it through DAP launch arguments via
  `initializationOptions` is a v2 polish.)
- Test lane `flowchart-dap-tests`: drives a real DAP session
  for three `.mflow` programs (`hello`, `for_loop`,
  `nested_for_if`), verifies breakpoint registration, the
  `stopped` event with `reason="breakpoint"`, and that the top
  stack frame's `source.path` ends in `.mflow`.

What still doesn't work for `.mflow` debugging (deferred polish):
- Block-id surfacing in stack frames (today the frame shows the
  `.mflow` path + line; the IDE has to map line → block on its
  own). Block ids in the frame `name` field would let the IDE
  highlight the active block on the canvas.
- Per-block step granularity: stepping today goes per-statement,
  so a block that synthesizes multiple statements steps multiple
  times. Collapsing to per-block steps would require tagging
  hooks with the block id and skipping when the id doesn't
  change.
- `--block-path` plumbed into the DAP launch surface (currently
  only `MATFORGE_BLOCK_PATH` reaches the DAP path).

### Phase 7 — `-emit-mflow` (AST → flowchart) (shipped)

Closes the asymmetry between the two frontends. Previously every
existing `-emit-*` mode worked on `.mflow` inputs but the inverse
direction wasn't available — the IDE was the only producer of
`.mflow` files.

- New module `lib/Flowchart/ASTToGraph.cpp` walks a `TranslationUnit`
  and emits a `.mflow` JSON document. The walker is the structural
  inverse of Phase 2-3's reducer:
  - Linear `Stmt`s map 1:1 to block kinds (`AssignStmt(name = literal)`
    → `variable`, `MatrixLiteral` RHS → `matrix_literal`,
    `disp(EXPR)` → `display`, `name(args)` calls → `function_call`,
    everything else → `assignment` or `expression`).
  - `IfStmt` → `if` block with `true`/`false` ports;
    `IfStmt::Elseifs` re-fold into a chain of nested `if` blocks on
    the false branch (no `elseif` block kind needed in the schema).
  - `ForStmt` / `WhileStmt` → loop blocks with `body`/`done` ports
    and an explicit back-edge from the body's exit pad to the loop
    head's `in` port.
  - `BreakStmt` / `ContinueStmt` / `ReturnStmt` → terminator blocks
    that produce an empty exit pad (subsequent statements in the
    same block are unreachable and dropped).
  - Each `Function` in `TU.Functions` becomes a `function`-kind
    sub-flow; `Inputs` / `Outputs` populate the flow's `signature`.
- A `Pad` abstraction (a vector of `(node_id, port_id)` source
  endpoints) drives wiring. Linear stmts produce single-source pads;
  if-branches produce multi-source pads (then-tail + else-tail) so
  the next statement gets edges from BOTH branch tails. This
  cleanly handles if-no-else (false-branch pad is just the if's
  `false` port, propagated forward).
- Output is byte-identical to the MatForge IDE's pretty-printed
  format: 2-space indent, `" : "` around the colon, alphabetical
  keys, blank-line empty arrays/objects. Successive `-emit-mflow`
  runs of the same TU produce byte-identical output, suitable for
  source-control diffs.
- Auto-layout: each block gets `ui.position = (x=200, y=index*120)`.
  The IDE re-layouts on first save; in the meantime the column-
  shaped diagram is at least readable.
- Idempotency property guaranteed: `.m → .mflow → .m → .mflow`
  produces a byte-identical second `.mflow` from iteration 2 onward.
- New `formatExpr(ostream&, const Expr&)` helper added to
  `lib/AST/Formatter.cpp` so the emitter can render `data.cond` /
  `data.iter` / `data.expression` etc. via the canonical formatter
  rather than duplicating expression printing.
- Test lane `flowchart-emit-mflow-tests`: 11 fixtures (canonical
  examples + flowchart corpus). Each runs the
  `input → -emit-mflow → -emit-matlab → -emit-mflow` pipeline and
  asserts the second `.mflow` matches the first byte-for-byte.

What's deferred:
- Round-trip preservation of IDE-set `ui.position`: `-emit-mflow`
  always writes its auto-layout x/y. A future "merge" mode could
  preserve positions by reading the current `.mflow` and copying
  its `ui.position` for any node whose id matches the new emission.
- `switch` and `try`/`catch` block kinds: degraded to
  `expression` blocks carrying the formatted source text. The
  textual round-trip works, but the diagram doesn't represent the
  branching shape.

### Phase 8 — Debug UX polish + layout merging (planned)

The `.mflow` frontend is functionally complete (compile + LSP + DAP
+ both round-trip directions ship), but the developer experience
still has rough edges that this phase addresses. Each item is
independent; pick by user pressure.

**8a. Block-id stack frames + canvas highlight (shipped).**

  - New `BlockLineMap` output parameter on `flowchart::buildAST`
    populates a `(file_id, line) → block_id` table as each block
    is emitted. `GraphToAST::recordBlock` is called from every
    site that already tags `Stmt::Range.Begin = N.Loc` (linear
    walk, handleIf, handleFor, handleWhile, handleCustom).
  - DAP `compileProgram` allocates the map for `.mflow` entry
    points and stashes it on `G.BlockByLine`. The `stackTrace`
    handler appends `[block:<id>]` to each frame's `name` when
    `(file_id, line)` is found in the map. No-op for `.m`
    programs (the map stays empty).
  - The IDE parses the block id from the frame name on each
    `stopped` event and highlights the active block on the canvas
    (out of scope for `matlabc`).
  - Test: `flowchart-dap-tests` asserts every stop's top frame
    name contains `[block:` so the IDE-facing surface stays
    contractual.

**8b. Per-block step granularity (~1 week).**
Stepping today is per-statement. A `display` block lowers to one
`disp(...)` call → one step. But:

  - A `custom` block lowers to one call-site statement plus an
    inserted `Function` definition. Stepping into the call follows
    into the function — that's correct already.
  - An `expression` block whose `data.expression` happens to be
    multi-statement (`v = v + 1; w = v * 2`) lowers to two
    `Stmt`s in the AST. Step-over visits each separately, which
    breaks the "one block = one step" mental model.
  - A `for` block with an empty body steps once on the loop head
    and again on the back-edge — duplicate stops on what looks
    like a single block.

  Fix: tag each MLIR debug hook with the originating block id
  (already known from the side-table in 8a); step-over walks
  through hooks that share the same block id and only stops when
  the id changes. New runtime helper
  `matlab_dbg_step_over_block(...)` keyed off the tag.

**8c. `--block-path` via DAP / LSP `initializationOptions` (shipped).**

  - Both `matlabc -dap` and `matlab-lsp` now read
    `initializationOptions.blockPath` (a JSON string array) on the
    `initialize` request. The DAP path stores it on
    `G.BlockPathFromIDE` and prepends entries to
    `BuildOptions::BlockSearchPath` ahead of the
    `MATFORGE_BLOCK_PATH` env-var entries on every `compileProgram`.
    The LSP path holds the entries in a `ServerBlockPath` global
    and forwards them on every `.mflow` reparse.
  - Resolution order is consistent across both surfaces and the
    standalone CLI: IDE-supplied entries first, env-var entries
    second; first match wins.
  - The IDE typically populates the array from a project setting
    (e.g. `${workspaceFolder}/blocks`), so projects can configure
    block libraries through their launch / LSP config without
    setting environment variables on the spawned subprocess.

**8d. `ui.position` merge on `-emit-mflow` (~3-4 days).**
Currently `-emit-mflow` always writes column-shape auto-layout
positions. Re-emitting an existing `.mflow` blows away whatever
the user dragged around in the IDE. Need a "merge" mode.

  - New flag `--preserve-layout PATH`: read PATH (an existing
    `.mflow`), index its nodes by id, and when the new emission
    produces a node with a matching id, copy its
    `ui.position` from the old file. Unmatched nodes (newly added)
    fall back to auto-layout.
  - Stable id generation across runs is already guaranteed by
    `n_<kind>_<counter>` ordering; the merge works as long as the
    program shape is unchanged.
  - When the program shape changes (a block was added / removed),
    the merge degrades gracefully: matched nodes keep their
    positions, new nodes are auto-laid in the column. The IDE will
    fix up the layout on save.
  - Stretch: `--preserve-layout` infers the previous `.mflow` from
    the same path as the input when it's a `.mflow` file (i.e.
    re-emitting in place picks up the file's own positions).

**8e. `switch` and `try` / `catch` block kinds (~1 week).**
Both are currently degraded to `expression` blocks carrying the
formatted source text. Diagram doesn't show the branching shape;
DAP can't set breakpoints on individual cases.

  - Schema additions (also bumps to `flowchart_schema.md`):
    - `switch`: `in` port + N `case` ports (one per case label) +
      `default` port. `data.discriminant` carries the switch
      expression.
    - `try`: `in` port + `body` + `catch` ports.
  - Reducer additions in `lib/Flowchart/GraphToAST.cpp`:
    - `switch` reduces N+1 branches via the same
      reconvergence detection used for `if` (multi-branch findJoin).
    - `try` is structurally an `if-else` whose condition is "did an
      error fire" — add `TryStmt` construction.
  - Emitter additions in `lib/Flowchart/ASTToGraph.cpp`:
    - `SwitchStmt` → `switch` block, one `case` port per case.
    - `TryStmt` → `try` block.

**Effort recap.** 8a–8c are the debug-UX polish pieces (~1.5 weeks
total); 8d is the layout merging (~half a week); 8e is the
control-flow expansion (~1 week). Each ships independently — the
debug polish has the highest user impact, the layout merge keeps
saved-then-emitted diagrams stable, and `switch`/`try` close the
last semantic gaps in the schema.

---

## 8. Out of scope (carried forward)

These v1 deferrals remain explicit non-goals. Items previously in
this list that have shipped (one-way text → blocks via
`-emit-mflow`, `.mflow` as an output target) are removed.

- **Continuous-time / sample-rate-different simulation.** Every
  `.mflow` program is sample-rate-fixed discrete, same as the SV
  pipeline.
- **2-D / image-pipeline blocks.** Roadmap item #7 territory.
- **Irreducible (unstructured) CFGs.** Refuse with a diagnostic;
  no synthetic loop / multi-entry handling.
- **Data edges.** `kind: "data"` is reserved in the schema and
  ignored by the v1 loader; the dataflow-style block extension is
  a separate phase, not v1+ work.
- **MATLAB `classdef` as a flow shape.** Class definitions in the
  AST currently emit through `-emit-mflow` as a top-level fallback
  (text inside an `expression` block). A `classdef` flow kind would
  let the IDE render properties / methods graphically, but it's
  out of scope for the v1 surface.

---

## 9. Risks and open questions

Resolved during v1 (kept here as design notes):

1. **JSON dependency** — resolved by hand-rolling a recursive-descent
   reader in `lib/Flowchart/Loader.cpp`. No third-party dep added.
2. **Source locations into block fields** — resolved Phase 6. The
   loader records every `data.*` field's byte offset in
   `Node::DataLocs`, and `GraphToAST` rewrites synthesized
   statements' `Range.Begin` to the originating block's byte
   offset, so DAP / LSP diagnostics land on the offending block.

Active open questions:

3. **Layout preservation on round-trip.** Phase 7 shipped
   `-emit-mflow` with auto-layout — repeat emission overwrites the
   IDE's manual placements. Fix planned in Phase 8d
   (`--preserve-layout` flag merging old `ui.position` into new
   emission by node id).
4. **Comment blocks.** The `comment` block kind is dropped during
   compilation; the formatter doesn't preserve comments either, so
   round-trip through `.m` would lose them anyway. Revisit when
   the formatter learns comment retention.
5. **Custom-block library trust boundary.** `library_id` resolves
   against a search path under user / project control —
   essentially "include any `.m` file from this directory." Same
   trust posture as `addpath` in MATLAB or `-I` in C, so it's not
   a regression, but worth documenting that block libraries
   execute as if their `.m` files were typed directly into the
   user's program.
6. **Inline `source` size limit.** A custom block's `data.source`
   field can in principle hold an entire program. The IDE should
   warn past some threshold (~200 lines) and suggest converting
   to a `path` block, since a giant string inside JSON is hard to
   diff and review. Compiler-side: no hard limit shipped; the
   loader should refuse `source` fields above 1 MB to keep
   parse-error blast radius bounded.
7. **Multi-statement `expression` blocks vs. step granularity.**
   Phase 8b — when the user writes `v = v + 1; w = v * 2` inside
   a single `expression` block's `data.expression`, the parser
   produces two `Stmt`s. Currently DAP step-over treats each as a
   separate stop. The block-id tagging in Phase 8a/b is the fix.

---

## 10. Update cadence

Phases 1–7 are shipped; this doc serves as the canonical
design-and-behavior reference for the `.mflow` frontend. The
shipped surface is mirrored in
[`feature_status.md`](feature_status.md), and the IDE-facing
schema contract is in [`flowchart_schema.md`](flowchart_schema.md).

Remaining work is the Phase 8 polish list (§7). The phase
boundaries there are independent — pick by user pressure rather
than sequencing — and items get demoted from "planned" to
"shipped" with a brief summary as each lands, matching the format
already used for Phases 1–7.
