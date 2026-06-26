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

They also compile and run — the `sim3d.World`/`Actor` handle classes are backed
by the runtime object model in both the C and C++ lanes:

```sh
matlabc -emit-cpp orbit_cube.m > orbit_cube.cpp
c++ -std=c++20 -I runtime orbit_cube.cpp build/libMatlabRuntime.a -lm -o orbit_cube
./orbit_cube            # writes the same orbit_cube.html
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

## Applied: 3-D control demos

[`examples/control/3d/`](../control/3d/README.md) uses this same `sim3d` surface
to animate **inverted-pendulum** controllers — cart-pole and double inverted
pendulum, each stabilized by PID, LQR, and pole placement. Those programs build
a parented kinematic chain (cart → hub → link) and drive it from a nonlinear
plant, so they double as a Control System Toolbox tour with a moving picture.
