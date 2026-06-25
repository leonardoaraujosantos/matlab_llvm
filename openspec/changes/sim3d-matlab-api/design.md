## Context

The `mflow-3d-animation` work added a Babylon.js 3-D animation surface reachable
only through `.mflow` block diagrams. `lib/Flowchart/EmitBabylon.cpp` already
separates three concerns: (1) a **scene JSON** array built by iterating
`signal_actor3d` blocks (geometry/material), (2) a **transform timeline** read
from `MflowLinkSim::logColumns()` / `logColumnNames()` as per-step
`<id>[tx,ty,tz,rx,ry,rz,sx,sy,sz]` width-9 groups, and (3) the **HTML + viewer
template** (`OS << "<!doctype html>…"` at ~line 505 and the `R"JS(…)"` viewer at
~line 552), which converts rpy→quaternion in a right-handed Z-up metres frame.

MATLAB's Simulink 3D Animation toolbox is fully command-driveable via the `sim3d`
object framework (`sim3d.World`, `sim3d.Actor`, `.Translation`/`.Rotation`/
`.Scale`, `add`/`open`/`run`/`close`). This codebase supports handle classes
(`classdef X < handle`, `IsHandle` in `lib/Sema/Resolver.cpp`) but has **no**
`+package` namespace support anywhere in the lexer/parser/resolver, and no
`+folder` exists in the tree.

## Goals / Non-Goals

**Goals:**
- Faithful `sim3d.World()` / `sim3d.Actor(name, shape)` API with handle semantics
  and dotted property assignment, usable from `matlabc -repl` and scripts.
- `run(world, dt)` accumulates keyframes; `close`/`sim3d.export` writes one
  self-contained Babylon HTML player — the same artifact shape as
  `-emit-mflowlink-babylon`.
- General `+package` namespace resolution in the frontend (first consumer:
  `sim3d`), reusable by other toolboxes.
- Preserve byte-identical output of the existing block-diagram emitter.

**Non-Goals:**
- Live/interactive 3-D windows (headless; the HTML player is the deliverable).
- Unreal Engine / VRML/X3D parity, photorealism, or `sim()`-on-a-model semantics.
- glTF/URDF import or viewer-side physics from the command API (block-diagram
  path keeps those; the command API ships primitives first).
- `import pkg.*` / wildcard imports (only qualified references in this change).

## Decisions

### D1: Parse-time package fold (match existing convention) — REVISED
Recon during implementation found this codebase has **no** `+folder`/namespace
facility; every package (`dsp.FIRFilter`, `coder.gpu.*`, `gpuArray.*`) is handled
by folding the dotted name to a flat name in `lib/Parse/Parser.cpp` `parsePostfix`.
Per the confirmed decision, `sim3d` follows the same convention: fold
`sim3d.World`→`sim3d_World`, `sim3d.Actor`→`sim3d_Actor`, `sim3d.export`→
`sim3d_export`. This delivers the identical faithful user-facing syntax
(`sim3d.World()`), matches convention, and avoids a large novel subsystem.
- **Why changed from the original spec**: building real `+folder` discovery +
  qualified resolution is invasive and unprecedented here; the user-facing goal
  (faithful `sim3d.World()` syntax) is fully met by the parse-fold.

### D2: World/Actor as handle classdefs backed by a C runtime (validated) — REVISED
Recon + capability probes against the built `matlabc` established what works:
scalar handle-property mutation ✓, `get.`/`set.` accessors on **`Dependent`**
properties ✓ (plain-property assignment does NOT dispatch a setter — only
Dependent does, `Lowering.cpp:5811`), methods forwarding the receiver to
`matlab_*` runtime externs ✓. These do NOT work: cell-arrays-of-objects in
properties, subscripting an Any-typed method return. Therefore:
- `sim3d_World` / `sim3d_Actor` are `classdef … < handle` prelude files (in
  `runtime/toolbox/sim3d/`), holding only an opaque `Id` scalar. **All** scene +
  timeline state lives C-side in `runtime/toolbox/sim3d/runtime_sim3d.cpp`.
- Faithful `a.Translation = [x y z]`: `Translation`/`Rotation`/`Scale`/material
  are `properties (Dependent)`; `set.Translation(obj,v)` forwards `v` to
  `matlab_sim3d_set_translation(obj, v)` and `get.Translation(obj)` reads it back.
- `add`/`open`/`run`/`close` are `sim3d_World` methods; `sim3d.export` folds to a
  packaged `sim3d_export` function. Each forwards the receiver `obj` to a
  `matlab_sim3d_*` runtime entry — the exact `matlab_dsp_*` System-Object
  convention.
- **Registration (per existing convention, 3 surfaces)**: symbol in the
  `Resolver.cpp` builtin list; LLVM signature in the `LowerTensorOps.cpp`
  object-runtime table (`{name, symbol, PtrTy, {argTys}}`, receiver/matrix →
  pointer, scalar → f64); classdef files registered for REPL (`buildReplPrelude`
  `Want[]`) and AOT (`userMentionsExtClasses` + `extClassLeaf`), with `"sim3d"`
  added to both `kToolboxDirs` arrays.
- **Alternative considered**: pure-MATLAB state in class properties — rejected;
  the unsupported cell-of-objects / Any-subscript paths make it infeasible and
  export must reuse the C++ Babylon emitter anyway.

### D3: Refactor EmitBabylon around a neutral scene/timeline source
Extract the scene-JSON builder and the HTML/viewer writer to accept a small
neutral struct — `BabylonScene { actors[], frameNames[], frameRows[] }` — instead
of `(MflowLinkModel&, MflowLinkSim&)`. Provide two adapters:
1. existing block-diagram adapter (model blocks + `Sim.logColumns()`),
2. new `sim3d` runtime adapter (recorded actors + accumulated keyframes).
The viewer JS, frame convention, and quaternion math are shared verbatim. The
block-diagram entry point keeps its signature and output (golden-stable).

### D4: Accumulate-then-export, headless by default
`open` allocates the timeline; `run(world, dt)` snapshots each actor's current
transform into a new keyframe row and appends the time column; `close` finalises;
`close`/`export` invoke the shared emitter to a file. No display is touched, so
it runs in CI. `close` may additionally no-op when nothing was recorded (with a
diagnostic), matching the spec's "run before open / empty world" rejections.

### D5: Transform group layout reused exactly
Each actor contributes a width-9 `[tx,ty,tz,rx,ry,rz,sx,sy,sz]` group keyed by
its name — identical to the `signal_actor3d` log group — so the parity scenario
holds and the viewer needs no changes.

## Risks / Trade-offs

- **[Namespace resolution is invasive in Sema]** Folding dotted chains into
  qualified names risks regressing struct/handle field access. → Gate on
  "leftmost segment is unbound AND matches a discovered `+folder`"; add explicit
  regression tests for variable-shadows-package and struct field access.
- **[EmitBabylon refactor could shift block-diagram output]** → Keep the
  block-diagram entry point's signature; assert byte-identical golden output for
  existing `examples/mflowlink/3d/*` before/after the refactor.
- **[Handle-class method dispatch in the REPL]** Cross-turn REPL state +
  classdef preludes have known fragility (see prior `#77` work). → Cover REPL
  construction/mutation/export in tests; reuse the existing classdef-prelude path.
- **[Scope creep vs the block-diagram 3D change]** → Strictly additive; do not
  edit `signal_*3d` blocks or the open `mflow-3d-animation` change.

## Migration Plan

Additive feature; no breaking changes. Land in order: (1) `+package` namespace
resolution + frontend tests, (2) EmitBabylon neutral-source refactor with
golden-parity check, (3) `runtime/sim3d` + lowering, (4) `+sim3d` classdefs +
examples + tests. Rollback = revert the change; the block-diagram path is
untouched throughout.

## Open Questions

- Should `sim3d.export` accept emit options (CDN base / inline engine) mirroring
  `BabylonEmitOptions`, or keep a minimal `(world, path)` first cut? (Lean:
  minimal first, add name-value options later.)
- Primitive set for v1 — box/sphere/cylinder/plane confirmed; add `text` labels
  now or defer? (Lean: defer to a follow-on.)
