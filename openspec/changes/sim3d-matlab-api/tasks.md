## 1. Frontend: sim3d package fold + runtime-symbol registration

- [x] 1.1 Add a `sim3d.<member>` parse fold in `lib/Parse/Parser.cpp` `parsePostfix` (mirror the `dsp.FIRFilter` pattern): `sim3d.World`→`sim3d_World`, `sim3d.Actor`→`sim3d_Actor`, `sim3d.export`→`sim3d_export`, via an `isSim3dMember` helper
- [x] 1.2 Register the `matlab_sim3d_*` runtime symbols in the Sema builtin table (`lib/Sema/Resolver.cpp`)
- [x] 1.3 Add LLVM signatures for the `matlab_sim3d_*` entries to the object-runtime lowering table (`lib/MLIR/Passes/LowerTensorOps.cpp`): receiver/matrix → PtrTy, scalar → F64
- [x] 1.4 Regression test: `sim3d.World`/`sim3d.Actor`/`sim3d.export` fold to the flat names; field access on an unrelated member is unaffected

## 2. EmitBabylon refactor to a neutral scene/timeline source

- [x] 2.1 Introduce a neutral `BabylonScene { actors[], frameNames[], frameRows[] }` struct in `include/matlab/Flowchart/EmitBabylon.h`
- [x] 2.2 Extract the scene-JSON builder and HTML/viewer writer to consume `BabylonScene` (viewer JS, Z-up metres frame, yaw-pitch-roll quaternion unchanged)
- [x] 2.3 Add the block-diagram adapter (`MflowLinkModel` + `MflowLinkSim` → `BabylonScene`) and keep `emitMflowLinkBabylon(...)` signature/output intact
- [x] 2.4 Golden-parity check: `examples/mflowlink/3d/*` Babylon HTML is byte-identical before/after the refactor

## 3. sim3d runtime

- [x] 3.1 Create `runtime/matlab_sim3d.h` + `runtime/sim3d/*.cpp` (mirrors `runtime/matlab_plot`): world/actor registry holding geometry/material and an accumulated keyframe timeline
- [x] 3.2 Implement `matlab_sim3d_*` entry points: world_new/actor_new, set_translation/rotation/scale/color/emissive/opacity (and getters), add, open, run(dt), close, export(path)
- [x] 3.3 Build the `sim3d`-runtime → `BabylonScene` adapter and wire `export`/`close` to the shared emitter (width-9 `[tx,ty,tz,rx,ry,rz,sx,sy,sz]` groups keyed by actor name)
- [x] 3.4 Diagnostics: unsupported shape, `run` before `open`, `run`/`open` on an empty world
- [x] 3.5 Register the runtime in CMake and link it into the JIT/AOT runtime surface

## 4. Lowering + MATLAB-facing classes

- [x] 4.1 Add lowering for the `sim3d` method/constructor calls (LowerPlot-style: recognise calls, rewrite to `matlab_sim3d_*` externs)
- [x] 4.2 Author `+sim3d/World.m` and `+sim3d/Actor.m` as `classdef … < handle` wrappers holding an opaque runtime id; constructors and property set/get call the runtime
- [x] 4.3 Author packaged methods `+sim3d/add.m`, `open.m`, `run.m`, `close.m`, `delete.m`, `export.m` (or expose as methods) routing to the runtime
- [x] 4.4 Place `+sim3d` on the resolution path the REPL/JIT/AOT use (prelude/search-path wiring)

## 5. Examples

- [x] 5.1 `examples/sim3d/orbit_cube.m` — orbiting box, exported to HTML
- [x] 5.2 `examples/sim3d/moving_vehicle.m` — box "vehicle" translating along X (sedan example expressed with primitives)
- [x] 5.3 `examples/sim3d/README.md` documenting the API and the headless export workflow

## 6. Tests

- [x] 6.1 `sim3d.World`/`sim3d.Actor` construction + handle semantics (REPL + script)
- [x] 6.2 Transform/material property assignment and read-back; scalar-scale broadcast
- [x] 6.3 `open`/`run`/`close` records one keyframe per `run`; per-step transforms match
- [x] 6.4 Export produces a valid Babylon player (scene graph + timeline + viewer JS) and succeeds headless
- [x] 6.5 Parity test: equivalent `sim3d` scene and `.mflow` `signal_actor3d` scene share geometry/frame/transform-group layout
- [x] 6.6 REPL-vs-script byte-identical export

## 7. Docs + validation

- [x] 7.1 Add a docs section for the command-line `sim3d` API and cross-reference `docs/mflowlink_blocks.md`
- [x] 7.2 `openspec validate sim3d-matlab-api --strict` passes; sweep/ctest green for new examples and tests
