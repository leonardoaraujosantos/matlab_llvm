## Why

The recent `mflow-3d-animation` work gave mflowLink (the block-diagram surface) a
Simulink-3D-Animation-style capability: `signal_world3d` / `signal_actor3d` blocks
whose recorded transform timeline renders in a self-contained Babylon.js player
(`matlabc -emit-mflowlink-babylon`). That 3D surface is reachable **only** by
authoring a `.mflow` model. MATLAB's real Simulink 3D Animation toolbox is also
driveable entirely from the command line via the `sim3d` object framework
(`sim3d.World`, `sim3d.Actor`, `.Translation`/`.Rotation`/`.Scale`, `add`/`open`/
`run`/`close`). This change brings that programmatic surface to the interpreted
MATLAB command line (`matlabc -repl`) and to scripts, reusing the Babylon.js
viewer we already build instead of Unreal Engine.

## What Changes

- **New `+package` namespace resolution in the MATLAB frontend.** The lexer,
  parser, and resolver gain support for `+folder` package namespaces and
  package-dotted references (`sim3d.World`, `pkg.sub.Fn`). This is a general
  language-frontend capability that other toolboxes can adopt; `sim3d` is its
  first consumer. (No namespace support exists today — verified: no `+folders`,
  no namespace handling in `Resolver`/`Parser`.)
- **New `sim3d` command-line 3D animation API**, faithful to MathWorks:
  - `sim3d.World()` — handle object owning a scene + recorded keyframe timeline.
  - `sim3d.Actor(name, shape)` — handle object with primitive geometry; settable
    `Translation` `[x y z]` (m), `Rotation` `[roll pitch yaw]` (rad), `Scale`,
    plus material props (color/emissive/opacity).
  - Methods: `add(world, actor)`, `open(world)`, `run(world, dt)`,
    `close(world)`, `delete(world)`, and `sim3d.export(world, 'scene.html')`.
  - Works from `matlabc -repl` and in compiled/JIT scripts.
- **Accumulate-then-export render model.** `run(world, dt)` records one transform
  keyframe per actor per call; `close`/`export` writes one self-contained Babylon
  HTML player — the same artifact shape as `-emit-mflowlink-babylon`. Headless-
  friendly; no live window is required.
- **Refactor `EmitBabylon`** so its scene-JSON builder and HTML/viewer template
  accept a **programmatic** scene + timeline (not only an `MflowLinkModel` +
  `MflowLinkSim`). Actors emit the same width-9 `[tx,ty,tz,rx,ry,rz,sx,sy,sz]`
  transform group and reuse the existing rpy→quaternion (yaw-pitch-roll),
  right-handed Z-up metres frame, so the viewer JS is unchanged and the block-
  diagram path keeps byte-identical output.
- **New `sim3d` runtime** alongside `matlab_plot` (`runtime/`) that stores actor
  geometry/material and appends per-frame transforms, exposed to the handle-class
  methods through dedicated lowering.
- **Examples + tests**: an orbiting-cube and a moving-vehicle script (the user's
  sedan example expressed with our primitive shapes), plus regression coverage of
  namespace resolution, property assignment, and headless HTML export.

## Capabilities

### New Capabilities
- `simulink-3d-animation-api`: the command-line `sim3d` object framework (World/
  Actor handle classes, transform properties, add/open/run/close/export) and its
  accumulate-then-Babylon-HTML render pipeline, callable from the REPL and scripts.

### Modified Capabilities
- `matlab-language-frontend`: add `+package` namespace resolution — `+folder`
  package discovery and package-dotted name/class references (`pkg.Name`,
  `pkg.sub.Fn`) in the lexer/parser/resolver.

## Impact

- **Frontend**: `lib/Parse/Parser.cpp`, `lib/Sema/Resolver.cpp`, lexer — new
  namespace lookup; existing `IsHandle` classdef support is reused as-is.
- **Babylon emitter**: `lib/Flowchart/EmitBabylon.cpp` + `include/matlab/Flowchart/
  EmitBabylon.h` refactored to a reusable scene/timeline source; block-diagram
  `-emit-mflowlink-babylon` behavior preserved (golden-stable).
- **Runtime**: new `runtime/sim3d/*` + `runtime/matlab_sim3d.h`; lowering wired
  similar to `lib/MLIR/Passes/LowerPlot.cpp`.
- **Surfaces**: `matlabc -repl` and script execution gain the `sim3d` API.
- **Docs/examples**: `examples/sim3d/*`, docs section; `docs/mflowlink_blocks.md`
  cross-reference.
- **Not touched**: the separate, still-open `mflow-3d-animation` change (block-
  diagram surface) and its `signal_*3d` blocks.
