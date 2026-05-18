# `.mflow` Schema Reference

Authoritative reference for the JSON file format produced by the
MatForge IDE and consumed by `matlabc` / `matlab-lsp`. This is the
contract between editor and compiler — both sides must agree on
shape, field names, and required fields per block kind.

For architecture and design rationale, see
[`flowchart_frontend.md`](flowchart_frontend.md). For shipped status,
see [`feature_status.md`](feature_status.md).

- **Schema name:** `matforge.flowchart`
- **Schema version:** `0.1.0`
- **File extension:** `.mflow`
- **Encoding:** UTF-8 JSON; LF or CRLF line endings.
- **Root type:** object (see §1).

---

## Table of contents

1. [Document structure](#1-document-structure)
2. [Flow object](#2-flow-object)
3. [Node object](#3-node-object)
4. [Edge object](#4-edge-object)
5. [Port conventions](#5-port-conventions)
6. [Block reference](#6-block-reference)
7. [Validation rules (loader)](#7-validation-rules-loader)
8. [Reduction rules (compiler)](#8-reduction-rules-compiler)
9. [Diagnostics & error UX](#9-diagnostics--error-ux)
10. [Editor implementation guide](#10-editor-implementation-guide)
11. [Versioning & forward compatibility](#11-versioning--forward-compatibility)

---

## 1. Document structure

```jsonc
{
  "schema":  "matforge.flowchart",   // REQUIRED — exact match
  "version": "0.1.0",                // REQUIRED — semver string
  "entry":   "main",                 // REQUIRED — name of the entry flow
  "settings": {                      // optional
    "columnMajor":         true,     // bool — matches MATLAB's storage order
    "defaultNumericType":  "double", // string — "double" | "single" | "int*" | "fi"
    "sourceLanguage":      "matforge"
  },
  "flows":   [ /* see §2 */ ],       // REQUIRED — array, must be non-empty
  "id":      "project_xxxxxx",       // optional — opaque project id
  "name":    "untitled.mflow",       // optional — display name
  "metadata": { /* opaque */ }       // optional — IDE-private state
}
```

### Field rules

- `schema` MUST be the literal string `matforge.flowchart`. The
  loader rejects everything else.
- `entry` MUST name a flow whose `kind` is `program`. The loader
  rejects unknown entry names and non-program entries.
- `flows` MUST contain at least one flow. The entry flow is found
  by `name`, not by `id`.
- `settings`, `id`, `name`, and `metadata` may be absent — defaults
  apply (`columnMajor=true`, `defaultNumericType="double"`).
- Unknown top-level fields are ignored (forward compatibility).

---

## 2. Flow object

```jsonc
{
  "id":   "flow_main",                          // REQUIRED — unique within document
  "kind": "program",                            // "program" | "function"
  "name": "main",                               // REQUIRED — MATLAB identifier
  "signature": {                                // optional; required for "function" kind
    "inputs":  ["x", "y"],                      // ordered, same as the function's params
    "outputs": ["r"]                            // ordered, same as the function's returns
  },
  "nodes": [ /* see §3 */ ],                    // REQUIRED
  "edges": [ /* see §4 */ ],                    // REQUIRED (may be empty)
  "layout": {                                   // optional, IDE-only
    "direction": "TB",                          // "TB" | "BT" | "LR" | "RL"
    "zoom":      1
  }
}
```

### Flow kinds

| `kind`       | Lifts to               | Constraints |
|--------------|------------------------|-------------|
| `program`    | a `Script` (entry only)| Exactly one `start` node and at least one `end` node. The flow named by the document's `entry` field MUST be `program`. |
| `function`   | a top-level `Function` | `signature.inputs` / `signature.outputs` must match the body's parameter and return variable usage. The function's MATLAB name comes from the flow's `name`, not its `id`. |

Multiple `function`-kind flows with the **same `name`** are
rejected (collides with MATLAB's same-name shadowing rule).
Different `id`s with different names are fine.

---

## 3. Node object

```jsonc
{
  "id":    "n_variable_1",                      // REQUIRED — unique within flow
  "kind":  "variable",                          // REQUIRED — see §6
  "label": "Variable",                          // optional — display text only
  "data":  { /* per-kind, see §6 */ },          // optional
  "ports": {                                    // REQUIRED for non-trivial kinds
    "in":  [ { "id": "in"  } ],
    "out": [ { "id": "out" } ]
  },
  "ui": {                                       // optional — IDE-only
    "position": { "x": 219, "y": -3 }
  }
}
```

### `data` field types

`data` is an object whose values may be:

| JSON type | Stored as | Use |
|---|---|---|
| string  | `Node::Data` (string) | Most fields: `name`, `value`, `expression`, `cond`, `iter`, `lhs`, `rhs`, `args`, `callee`, `path`, `library_id`, `source`, `flow_id` |
| number  | `Node::Data` (raw text) | Numeric fields written as JSON numbers (the loader keeps the source text) |
| boolean | `Node::Data` (`"true"` / `"false"`) | Future flags |
| array of strings | `Node::DataArrays` | `inputs[]`, `outputs[]` on custom blocks |
| object  | reserved | silently ignored in v1; reserved for future |

### Port objects

Each port object must have an `id` (string). Additional fields are
ignored. Per-block port id conventions are in §5.

---

## 4. Edge object

```jsonc
{
  "id":   "e_1",                                // optional but recommended for diagnostics
  "kind": "control",                            // "control" | "data"
  "from": { "node": "main_start", "port": "out" },
  "to":   { "node": "n_variable_1", "port": "in" }
}
```

| `kind`     | Meaning                                                               |
|------------|-----------------------------------------------------------------------|
| `control`  | Sequential control transfer (the only kind that drives the reducer). |
| `data`     | **Reserved for a future dataflow extension. v1 ignores these silently.** |

### Edge invariants

- `from.node` and `to.node` MUST resolve to nodes within the same flow.
- `from.port` MUST exist in that node's `ports.out`.
- `to.port` MUST exist in that node's `ports.in`.
- Edge `id` (when provided) MUST be unique within the flow.
- Multiple edges may target the same `(node, in-port)` pair (this is
  how loop back-edges and `if`-join points are expressed).

---

## 5. Port conventions

| Block kind                                                                                                                      | `in` ports | `out` ports |
|--------------------------------------------------------------------------------------------------------------------------------|------------|-------------|
| `start`                                                                                                                        | none       | `out`       |
| `end`                                                                                                                          | `in`       | none        |
| `variable`, `constant`, `assignment`, `expression`, `display`, `input`, `function_call`, `subflow_call`, `function_definition`, `custom`, `matrix_literal`, `comment` | `in`       | `out`       |
| `if`                                                                                                                           | `in`       | `true`, `false` |
| `for`, `while`                                                                                                                 | `in`       | `body`, `done` |
| `break`, `continue`, `return`                                                                                                  | `in`       | `out` *(unused — control transfer is implicit)* |
| `switch`                                                                                                                       | `in`       | `case_0`, `case_1`, …, `case_<N-1>`, `default` |
| `try`                                                                                                                          | `in`       | `body`, `catch` |

The IDE SHOULD enforce these counts and names at edit time so the
saved file is loader-clean. The compiler also validates them at
load time.

### Loop back-edges

A `for` / `while` block has its `body` chain end with an edge that
returns to the loop head's `in` port. This is how the body
terminates:

```
predecessor → for_head:in
              for_head:body  → body_node_1 → ... → body_node_n
                                                    └→ for_head:in   (back-edge)
              for_head:done  → continuation
```

The body chain MUST loop back to the loop head, and only that loop
head. Edges from the body to an outer loop head, an `end`, or
arbitrary nodes are not supported (use `break` / `continue` / `return`).

### `if` reconvergence

Both `true` and `false` branches must eventually reach a common
node (the join), or both must reach an `end` node, or a `break`
/ `continue` / `return` block. Diverging branches with no
reconvergence are rejected by the reducer.

---

## 6. Block reference

The full set of block kinds. **Bold** entries on `data` are required;
others are optional with the noted defaults.

### 6.1 Structural

#### `start`
- **Ports:** out=`out` only.
- **Data:** none.
- **Lowers to:** nothing — control marker only.
- **Constraints:** exactly one per `program` flow.

#### `end`
- **Ports:** in=`in` only.
- **Data:** none.
- **Lowers to:** nothing — control marker only.
- **Constraints:** at least one per `program` flow.

#### `comment`
- **Ports:** in=`in`, out=`out`.
- **Data:** `text` (string, optional, free-form).
- **Lowers to:** nothing. Comments are dropped because the formatter
  doesn't preserve comments through `-emit-matlab` round-trip.

### 6.2 Linear statements

#### `variable`
Declare and initialize a variable.

- **Data:** **`name`** (identifier), **`value`** (expression).
- **Lowers to:** `name = value;`
- **Example:**
  ```jsonc
  "data": { "name": "x", "value": "1" }
  // → x = 1;
  ```

#### `constant`
Identical to `variable` semantically; conventionally used for
literals that won't be reassigned. The compiler treats them the
same.

- **Data:** **`name`**, **`value`**.

#### `assignment`
Assign an arbitrary expression to an arbitrary l-value.

- **Data:** **`lhs`** (l-value MATLAB expression — name, indexed
  name, struct field, etc.), **`rhs`** (any MATLAB expression).
- **Lowers to:** `lhs = rhs;`
- **Example:**
  ```jsonc
  "data": { "lhs": "y", "rhs": "x * 2 + 1" }
  // → y = x * 2 + 1;
  ```

#### `expression`
Free-form statement. The IDE-supplied string is fed to the existing
MATLAB parser as a top-level statement.

- **Data:** **`expression`** (any MATLAB statement; trailing `;`
  optional — the loader appends one).
- **Lowers to:** `expression;`
- **Use cases:** in-place updates (`v = v + 1`), function calls
  with side effects, persistent declarations, command-form syntax.

#### `display`
Print an expression with `disp(...)`.

- **Data:** **`expression`** (any MATLAB expression).
- **Lowers to:** `disp(expression);`
- **Example:**
  ```jsonc
  "data": { "expression": "v" }
  // → disp(v);
  ```

#### `input`
Read a value from stdin.

- **Data:** **`name`** (identifier), `prompt` (string, optional).
- **Lowers to:** `name = input('prompt');` (or `input('')` if no prompt).
- **Example:**
  ```jsonc
  "data": { "name": "age", "prompt": "Enter age: " }
  // → age = input('Enter age: ');
  ```

#### `function_call`
Call a named function (built-in or user-defined).

- **Data:** **`callee`** (function name), `args` (CSV expression
  list, default empty), `lhs` (target l-value, default empty).
- **Lowers to:** `lhs = callee(args);` or `callee(args);` if no `lhs`.
- **Example:**
  ```jsonc
  "data": { "callee": "max", "args": "M(1, :)", "lhs": "y" }
  // → y = max(M(1, :));
  ```

#### `matrix_literal`
Assign a matrix literal to a variable.

- **Data:** **`name`**, **`rows`** (the bare interior of `[ ... ]`,
  with `;` separating rows).
- **Lowers to:** `name = [rows];`
- **Example:**
  ```jsonc
  "data": { "name": "M", "rows": "1 2 3; 4 5 6" }
  // → M = [1 2 3; 4 5 6];
  ```

### 6.3 Control flow

#### `if`
Two-way branch.

- **Ports:** in=`in`, out=`true`, `false`.
- **Data:** **`cond`** (any MATLAB expression).
- **Branch contract:** both `true` and `false` ports MUST have an
  outgoing edge. If the user wants an "if without else", route the
  `false` port directly to the same node the `true` branch
  eventually reaches — the compiler detects this and emits
  `if cond ... end` (no `else` block).
- **Lowers to:**
  ```matlab
  if cond
      <true branch statements>
  else
      <false branch statements>
  end
  ```

#### `for`
Range-style loop.

- **Ports:** in=`in`, out=`body`, `done`.
- **Data:** **`var`** (loop variable identifier), **`iter`** (range
  expression, e.g. `1:10`, `1:2:N`, `[1 3 5]`).
- **Body contract:** the `body` chain MUST end with an edge back
  to this `for` block's `in` port (the back-edge).
- **Lowers to:**
  ```matlab
  for var = iter
      <body statements>
  end
  ```

#### `while`
Condition-style loop.

- **Ports:** in=`in`, out=`body`, `done`.
- **Data:** **`cond`** (boolean expression).
- **Body contract:** same as `for` — body chain ends with a
  back-edge to this block's `in`.
- **Lowers to:**
  ```matlab
  while cond
      <body statements>
  end
  ```

#### `break`, `continue`, `return`
Loop / function exit.

- **Ports:** in=`in`, out=`out` (the out-port edge is unused but
  the IDE may keep it for placement).
- **Data:** none.
- **Constraints:** must appear inside a loop body (`break`,
  `continue`) or function body (`return`). Statements after a
  `break`/`continue`/`return` in the same chain are dropped.

#### `switch`
Multi-way branch on a discriminant. Each case body is its own
sub-chain wired from a numbered case port; `otherwise` wires from
the `default` port.

- **Ports:** in=`in`; out=`case_0`, `case_1`, … `case_<N-1>`,
  `default` (one `case_<i>` for each entry in `data.cases`, plus
  always a `default`).
- **Data:** **`discriminant`** (any MATLAB expression),
  **`cases`** (string array of case-value expressions, one per
  `case_<i>` port in order).
- **Branch contract:** all branches reconverge at a single join.
  The IDE may route the `default` port directly to the join when
  the user's MATLAB has no `otherwise` clause — the emitter
  handles either shape.
- **Lowers to:**
  ```matlab
  switch discriminant
  case <cases[0]>
      <case_0 branch statements>
  case <cases[1]>
      <case_1 branch statements>
  ...
  otherwise
      <default branch statements>
  end
  ```

#### `try`
Two-branch error-handling block. The `body` chain runs first; if
any statement fires `error()`, control transfers to the `catch`
chain.

- **Ports:** in=`in`, out=`body`, `catch`.
- **Data:** `catch_var` (optional — name of the exception object
  bound in the catch body's scope; corresponds to MATLAB's
  `catch err`).
- **Branch contract:** both `body` and `catch` chains must
  reconverge at a single join (or both must reach `end` /
  `break` / `continue` / `return`).
- **Lowers to:**
  ```matlab
  try
      <body branch statements>
  catch <catch_var?>
      <catch branch statements>
  end
  ```

### 6.4 Multi-flow

#### `function_definition`
Visual marker that the program uses a function defined as another
flow in the same document. Required for the IDE to render the
function in the canvas; not needed for the compiler (every
`function`-kind flow lifts automatically).

- **Data:** **`flow_id`** (id of a `function`-kind flow in this
  document), `name` (optional override; defaults to the target
  flow's `name`).
- **Lowers to:** nothing (no statement).
- **Validation:** the loader fails if `flow_id` doesn't resolve.

#### `subflow_call`
Call a `function`-kind flow by `flow_id` rather than by name.

- **Data:** **`flow_id`**, `args` (CSV, default empty), `lhs`
  (default empty).
- **Lowers to:** `lhs = <target.name>(args);`
- **Use vs. `function_call`:** prefer `subflow_call` when
  refactoring a sub-flow's `name` should automatically update its
  callers. `function_call` is name-based and won't track
  rename-by-id.

### 6.5 Custom blocks (user-supplied MATLAB)

#### `custom`
Embed a user-written MATLAB function as a block. Three provenance
modes — exactly one MUST be set:

| Field          | Meaning                                                  |
|----------------|----------------------------------------------------------|
| `source`       | Inline MATLAB text (full `function ... end` block).     |
| `path`         | Path to a `.m` file, resolved relative to the `.mflow`. |
| `library_id`   | Library function name resolved against the block search path. |

- **Data:**
  - **One of:** `source` / `path` / `library_id` (multiple is an
    error).
  - **`callee`** OR **`name`** — name of the function to call (and
    to find inside the source for arity validation).
  - `args` (CSV, default empty), `lhs` (default empty).
  - `inputs` (string array, optional) — declared input names; the
    loader checks `len(inputs) == function.Inputs.size()`.
  - `outputs` (string array, optional) — declared output names;
    same check against `function.Outputs.size()`.
- **Lowers to:**
  - The named function's body is inserted **once** into the
    `TranslationUnit`'s `Functions` list, deduped across multiple
    callers with the same `callee` name.
  - The block emits `lhs = callee(args);` at the call site.
- **`library_id` resolution order:** `--block-path` flags first
  (in CLI order), then `MATFORGE_BLOCK_PATH` (colon-separated).
  First matching `<dir>/<lib_id>.m` wins.

##### Custom block example

```jsonc
{
  "id": "n_filter_1",
  "kind": "custom",
  "label": "FIR Tap",
  "data": {
    "name":       "fir_tap",
    "callee":     "fir_tap",
    "args":       "x, k",
    "lhs":        "y",
    "source":     "function y = fir_tap(x, k)\n    y = k * x;\nend\n",
    "inputs":     ["x", "k"],
    "outputs":    ["y"]
  },
  "ports": {
    "in":  [ { "id": "in" } ],
    "out": [ { "id": "out" } ]
  }
}
```

---

## 7. Validation rules (loader)

The loader rejects a `.mflow` file that violates any of these
invariants. Every rejection includes a byte-precise source location
in the JSON file (line + column).

### Document level

- `schema` is missing or not the literal `matforge.flowchart`.
- `flows` is missing, not an array, or empty.
- `entry` (when set) doesn't name any flow.

### Flow level

- `id`, `name`, or `kind` is missing.
- `kind` is not one of `program` / `function`.
- Two flows share the same `id`.
- For `program` flows: != 1 `start` node, or 0 `end` nodes.

### Node level

- `id` or `kind` is missing.
- Two nodes in the same flow share `id`.
- A port object is missing its `id`.

### Edge level

- `from` or `to` is missing or not an object.
- `from.node` / `to.node` is empty.
- An endpoint node id doesn't resolve to any node in the flow.
- An endpoint port id isn't declared on the referenced node.
- Two edges in the same flow share `id` (when set).

### Reachability (warning, not error)

Nodes not reachable from `start` produce a warning per node, not
an error. The IDE leaves disconnected palette nodes on the canvas
during editing; warn-and-skip lets those files load. The
exception: `end` is checked for reachability via incoming edges,
not from `start`.

---

## 8. Reduction rules (compiler)

After the loader validates structure, the AST builder runs a
recursive walker that produces a `TranslationUnit`. Rejection at
this stage means the file is structurally valid but semantically
unreducible.

### Per-block requirements

| Block kind         | Required `data` fields            |
|--------------------|-----------------------------------|
| `variable`, `constant` | `name`, `value`               |
| `assignment`       | `lhs`, `rhs`                      |
| `expression`       | `expression`                      |
| `display`          | `expression`                      |
| `input`            | `name`                            |
| `function_call`    | `callee`                          |
| `matrix_literal`   | `name`, `rows`                    |
| `if`               | `cond`                            |
| `for`              | `var`, `iter`                     |
| `while`            | `cond`                            |
| `function_definition` | `flow_id` (validated cross-ref) |
| `subflow_call`     | `flow_id`                         |
| `custom`           | exactly one of `source` / `path` / `library_id`, plus `callee` or `name` |

### Control-flow shape requirements

- **Linear node** (everything except `if` / `for` / `while` /
  `start` / `end` / `break` / `continue` / `return`): MUST have
  exactly one outgoing control edge.
- **`if` block:** MUST reach a common reconvergence node (or both
  branches must terminate at `end` / `break` / `continue` /
  `return`). Branches that diverge without reconvergence are
  rejected.
- **`for` / `while` block:** the `body` port's reachable subgraph
  MUST eventually have an edge back to the loop head's `in` port
  (the back-edge). Bodies that don't loop back are rejected.
- **Single back-edge per loop:** only one body branch may close
  the loop. If multiple body branches loop back, the file is
  rejected. (The IDE can collapse them with an explicit join node
  before the back-edge.)

### Currently rejected even when structurally valid

- Irreducible CFGs (the reducer's `findJoin` returns "no join").
- Multi-back-edge loops.
- Goto-style jumps to unrelated nodes.
- Custom blocks whose declared `inputs` / `outputs` arity disagree
  with the parsed function signature.

---

## 9. Diagnostics & error UX

Every loader and reducer error carries a `SourceLocation` whose
`File` is the `.mflow` and whose `Offset` is the byte position of
the offending JSON token. `matlab-lsp` translates these into LSP
`textDocument/publishDiagnostics` events with line / column ranges
the editor can render directly on the canvas.

### Severity mapping

| Compiler severity | LSP severity |
|---|---|
| Error             | 1 (Error)    |
| Warning           | 2 (Warning)  |
| Note              | 3 (Information) |

### Recommended editor surface

- **Diagnostics on save**: rebuild on every save; render unique
  diagnostics on the affected blocks via the source-offset → block
  mapping (the loader can emit the offending node's id alongside
  the message; the IDE matches by id).
- **Inline cond / iter / expression validation**: when the user
  types into a `data.cond` / `data.iter` / `data.expression`
  field, the IDE can pre-parse via the same compiler library
  (`matlabc -dump-flow` over a temp file) to catch syntax errors
  before save.
- **Cross-flow lints**: an unconnected `function_definition` block
  whose `flow_id` doesn't resolve, a `subflow_call` to a missing
  flow, a `custom` block whose `library_id` can't be found in the
  configured search path — all surface as compiler diagnostics
  the editor can highlight.

---

## 10. Editor implementation guide

### Save path checklist

The IDE SHOULD enforce these constraints at edit time so saved
files load cleanly:

- [ ] Document has a `schema`, `version`, and `entry` field.
- [ ] At least one flow exists, and the `entry` matches a
  `program`-kind flow's `name`.
- [ ] Every node has a unique id within its flow.
- [ ] Every edge has resolved `from.node` / `to.node`.
- [ ] Port id strings on edges match the target node's port lists.
- [ ] Every flow with `kind: "program"` has exactly one `start`
  and at least one `end`.
- [ ] Per-block-kind required `data` fields (§6, §8) are filled
  before allowing save.

### Load path checklist

When opening an existing `.mflow`:

- Treat **unknown block kinds** as opaque — preserve them on
  re-save so a newer-schema file edited in an older IDE
  round-trips cleanly. Render them as "?" placeholder blocks.
- Treat **unknown top-level fields** the same way (`metadata`,
  future top-level extensions).
- Honour `ui.position` if present; otherwise auto-layout from
  `layout.direction`.

### Suggested IDE UX

- **Validation panel:** mirror the compiler's diagnostics list,
  one row per diagnostic, click-to-select the offending block.
- **"Run via matlabc" command:** invoke
  `matlabc -emit-matlab <file>.mflow` and show the output in a
  side panel. Round-trippable by-design.
- **Block search-path config:** a project-level setting that the
  IDE can pass as `MATFORGE_BLOCK_PATH` to `matlabc` and to
  `matlab-lsp` (the latter requires LSP `initialize`
  `initializationOptions` plumbing — currently TODO on the
  matlab-lsp side).

### Block library shipped with `matlabc`

`matlabc` ships no blocks of its own. Custom-block libraries are
user-curated `.m` directories pointed to by `--block-path` /
`MATFORGE_BLOCK_PATH`. The IDE may ship its own default library
(common DSP / control kernels); this is purely an IDE convention.

---

## 11. Versioning & forward compatibility

### Semver discipline

The schema is versioned by the document's `version` field
(currently `0.1.0`).

- **Patch** (`0.1.0` → `0.1.1`): bug fixes, clarifications, no
  syntactic change.
- **Minor** (`0.1.0` → `0.2.0`): backward-compatible additions —
  new block kinds, new optional `data` fields, new top-level
  fields. Older readers ignore the additions.
- **Major** (`0.1.0` → `1.0.0`): breaking change — required field
  rename, semantic change to an existing block kind, removed
  block kind. Migration path documented per release.

### Forward-compatible additions (no major bump)

These can land in `0.1.x` / `0.2.x` without breaking existing
files:

- New block kinds (older `matlabc` will reject as "unknown kind",
  but the loader still parses the document).
- New optional `data` fields on existing blocks.
- New top-level fields (`extensions`, `documentation`, etc.).
- New port-id conventions on a new block kind (existing kinds
  keep their ports stable).

### Reserved for future use

These shapes appear in the format but are **silently ignored** by
v1, reserved for later expansion:

- `kind: "data"` edges — Simulink-style dataflow wiring; would let
  custom blocks read inputs from in-port edges instead of from a
  manually-typed `args` string.
- Nested objects inside `data` — for richer per-block config.
- `ui.position` — preserved but not used (one-way blocks → MATLAB
  in v1; matters when MATLAB → blocks lands).

### Migration policy

When a major version lands, `matlabc` will accept files at the
**previous** major version (with deprecation warnings) for at
least 6 months. The IDE SHOULD prompt the user to upgrade the
file format on save (preserving original under `.mflow.bak`).

---

## Appendix: minimal example

The smallest valid `.mflow` — one program flow with a single
`disp` call:

```json
{
  "schema":  "matforge.flowchart",
  "version": "0.1.0",
  "entry":   "main",
  "flows": [
    {
      "id":   "flow_main",
      "kind": "program",
      "name": "main",
      "nodes": [
        { "id": "s",  "kind": "start",
          "ports": { "in": [], "out": [{"id": "out"}] } },
        { "id": "d",  "kind": "display",
          "data": { "expression": "'Hello, world!'" },
          "ports": { "in": [{"id": "in"}], "out": [{"id": "out"}] } },
        { "id": "e",  "kind": "end",
          "ports": { "in": [{"id": "in"}], "out": [] } }
      ],
      "edges": [
        { "id": "e_1", "kind": "control",
          "from": {"node": "s", "port": "out"}, "to": {"node": "d", "port": "in"} },
        { "id": "e_2", "kind": "control",
          "from": {"node": "d", "port": "out"}, "to": {"node": "e", "port": "in"} }
      ]
    }
  ]
}
```

Compile with:

```bash
matlabc -emit-matlab hello.mflow
# → disp('Hello, world!');

runtime/build_and_run.sh hello.mflow   # builds + runs through the C lane
# → Hello, world!
```

For more example shapes covering every block kind, see
[`examples/mflow/`](../examples/mflow/) and the test corpus under
[`test/Flowchart/EmitMatlab/`](../test/Flowchart/EmitMatlab/).

---

## Signal-flow extensions (mflowLink)

`.mflow` is one file format with two dialects, selected by
`settings.kind`:

- `"control_flow"` (default; absent ⇒ this) — the structured
  *program* described above: `start` / `if` / `for` / … nodes that
  lower to MATLAB statements.
- `"signal_flow"` — an **mflowLink** *block diagram*: `signal_*`
  blocks wired by `"kind": "data"` edges, integrated over time by
  the simulation runtime. See [`mflow_link_roadmap.md`](mflow_link_roadmap.md).

All signal-flow fields are **additive** — the schema version stays
`0.1.0`, and a control-flow `.mflow` is byte-for-byte unaffected. An
older loader simply ignores the fields it doesn't recognise.

### `settings.kind` / `settings.solver` / `settings.snapshot`

```jsonc
"settings": {
  "kind": "signal_flow",              // "control_flow" (default) | "signal_flow"
  "solver": {                         // optional; defaults shown
    "type":                "variable_step",  // "fixed_step" | "variable_step"
    "algorithm":           "ode45",           // ode45|ode23|ode23s|ode15s|euler|heun
    "startTime":           0.0,
    "stopTime":            10.0,
    "maxStep":             "auto",            // "auto" | "<seconds>"
    "minStep":             "auto",
    "relTol":              1e-3,
    "absTol":              1e-6,
    "zeroCrossing":        true,
    "algebraicLoopMethod": "trust_region"     // trust_region | newton | off
  },
  "snapshot": {                       // optional; defaults shown
    "enabled": true,
    "depth":   256,
    "fields":  "states"               // "states" | "states+inputs" | "all"
  }
}
```

Sub-field keys are camelCase on disk, matching the IDE's `JSONEncoder`
output. An unknown `kind` is a hard load error; absent `solver` /
`snapshot` blocks fall back to the defaults above.

### Signal-flow nodes

A signal-flow flow has **no `start` / `end` nodes** — the
program-shape validation is skipped for `signal_flow` documents.
Nodes use the `signal_*` kinds (`signal_sine`, `signal_gain`,
`signal_transfer_fcn`, `signal_integrator`, `signal_scope`, …); the
full reserved set is in [`mflow_link_roadmap.md`](mflow_link_roadmap.md) §5.2.
The loader accepts and round-trips **any** `signal_*` kind — a kind
whose evaluator hasn't shipped is rejected later, at lowering time,
not at load time.

Signal-flow nodes carry extra optional `data` fields:

```jsonc
{ "id": "k", "kind": "signal_gain",
  "data": {
    "sample_time": "continuous",     // "continuous" | "inherited" | "<seconds>"
    "units":       "Nm",
    "data_type":   "double",
    "log_signal":  true,             // stream this block's output
    "params": { "gain": 2.0 }        // per-kind block parameters
  },
  "ports": { "in": [{"id": "in"}], "out": [{"id": "out"}] } }
```

`data.params` is a nested object of scalar block parameters (number /
bool / string). Its keys are pinned per block kind by the IDE's
`SignalFlowParamSpec` catalogue — mirrored in
[`mflowlink_blocks.md`](mflowlink_blocks.md). Edges between signal
blocks use `"kind": "data"`.
