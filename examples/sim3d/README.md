# sim3d — command-line 3-D animation

These examples drive the `sim3d` command-line surface, a faithful subset of
MathWorks' [Simulink 3D Animation](https://www.mathworks.com/products/3d-animation.html)
`sim3d` object framework, rendering through the embedded Babylon.js viewer
(the same player produced by `matlabc -emit-mflowlink-babylon`) instead of
Unreal Engine.

Unlike the block-diagram surface (`examples/mflowlink/3d/*.mflow`), these are
plain MATLAB programs: build a `World`, add `Actor`s, set their transforms in a
loop, and export. They work in the **interpreted REPL** and as **compiled
scripts** with byte-identical output.

## Run

```sh
# Interpreted MATLAB commands mode:
matlabc -repl < orbit_cube.m
# Then open the emitted HTML in any browser:
xdg-open orbit_cube.html
```

## API

| Call | Meaning |
|------|---------|
| `w = sim3d.World()` | Create an empty 3-D scene (a handle). |
| `a = sim3d.Actor(name, shape)` | Create an actor with primitive geometry (`box`, `sphere`, `cylinder`, `plane`). |
| `a.Translation = [x y z]` | Position in metres (right-handed, Z-up). |
| `a.Rotation = [roll pitch yaw]` | Orientation in radians. |
| `a.Scale = [sx sy sz]` | Scale (a scalar broadcasts). |
| `a.Color = [r g b]`, `a.Size = [x y z]` | Material / geometry (RGB in `[0,1]`). |
| `w.add(a)` | Register an actor into the world. |
| `w.open()` | Begin recording. |
| `w.run(dt)` | Record one keyframe of every actor's current transform; advance time by `dt`. |
| `w.close()` | Finish recording. |
| `sim3d.export(w, 'scene.html')` | Write the self-contained Babylon.js HTML player. |

## Rendering model

`run(world, dt)` accumulates a keyframe timeline; `sim3d.export` writes one
self-contained HTML document (scene graph + keyframe timeline + viewer logic
inline, Babylon engine from CDN). Headless-friendly — no display is required to
produce the file; open it in a browser to play.

## Examples

- **orbit_cube.m** — a single cube orbiting in a circle (command-line
  counterpart of `examples/mflowlink/3d/orbit_cube.mflow`).
- **moving_vehicle.m** — a box "vehicle" driving forward over a ground plane
  (the sim3d moving-vehicle demo with primitive shapes).
