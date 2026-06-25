# sim3d — command-line 3-D animation API

The `sim3d` package brings MathWorks' [Simulink 3D Animation](https://www.mathworks.com/products/3d-animation.html)
command-line workflow to the interpreted MATLAB surface, rendering through the
embedded Babylon.js viewer (the same player as `matlabc -emit-mflowlink-babylon`)
instead of Unreal Engine. Authored programmatically — no `.mflow` block diagram
required — it runs in `matlabc -repl` and as a compiled script with
byte-identical output.

See `examples/sim3d/` for runnable demos and the API table.

## Usage

```matlab
w = sim3d.World();
a = sim3d.Actor('cube', 'box');   % box | sphere | cylinder | plane
a.Color = [0.2 0.6 1.0];
w.add(a);

w.open();
for k = 1:60
    th = k * 0.1;
    a.Translation = [3*cos(th) 3*sin(th) 1.0];  % metres, right-handed Z-up
    a.Rotation    = [0 0 th];                   % radians, roll-pitch-yaw
    w.run(0.05);                                % record one keyframe
end
w.close();

sim3d.export(w, 'scene.html');    % self-contained Babylon.js HTML player
```

## How it works

- **Parser fold** (`lib/Parse/Parser.cpp`): `sim3d.World`/`sim3d.Actor` fold to
  the flat classdef names `sim3d_World`/`sim3d_Actor`, and `sim3d.export` folds
  to the `matlab_sim3d_export` runtime builtin — the same package-fold
  convention as `dsp.*`. The classdefs live in
  `runtime/toolbox/sim3d/sim3d_classdefs.m` and are auto-prepended (REPL and
  AOT) when the source mentions `sim3d.*`.
- **Handle classes + Dependent properties**: `World`/`Actor` are thin
  `handle` wrappers; `Translation`/`Rotation`/`Scale` are `Dependent`
  properties whose `set.`/`get.` accessors forward to the C++ runtime (plain
  property assignment does not dispatch a setter — only `Dependent` does).
- **Runtime** (`runtime/toolbox/sim3d/runtime_sim3d.cpp`): holds all scene +
  timeline state, keyed by the handle object pointer. `run(world, dt)` records
  one keyframe of each added actor's current transform; `export` builds the
  scene JSON and writes the HTML via the shared writer
  (`include/matlab/Flowchart/BabylonDocument.h`), which the block-diagram
  emitter also uses — so both paths stay byte-for-byte in sync.

## Relationship to the block-diagram surface

This is the command-line counterpart of the `signal_world3d` / `signal_actor3d`
blocks documented in [mflowlink_blocks.md](mflowlink_blocks.md). Both emit the
same Babylon.js player and share the right-handed Z-up metres frame and
yaw-pitch-roll quaternion convention. The block-diagram surface is authored in
`.mflow`; this one is authored in MATLAB code.
