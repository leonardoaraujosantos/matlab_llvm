## ADDED Requirements

### Requirement: sim3d.World and sim3d.Actor object construction

The system SHALL provide a `sim3d` package exposing handle classes `World` and
`Actor` constructible from interpreted MATLAB commands and scripts. `sim3d.World()`
SHALL return a handle to an empty 3-D scene. `sim3d.Actor(name, shape)` SHALL
return a handle to an actor with the given identifier and primitive geometry,
where `shape` is one of the supported primitives (`box`, `sphere`, `cylinder`,
`plane`/`ground`). Both SHALL have reference (handle) semantics so a variable
copy refers to the same underlying object.

#### Scenario: Construct a world and an actor in the REPL

- **WHEN** `matlabc -repl` evaluates `w = sim3d.World();` then `a = sim3d.Actor('cube','box');`
- **THEN** `w` is a `sim3d.World` handle and `a` is a `sim3d.Actor` handle, with no error

#### Scenario: Handle semantics

- **WHEN** `a2 = a;` and a later mutation sets `a.Translation = [1 0 0];`
- **THEN** reading `a2.Translation` reflects `[1 0 0]` (same underlying object)

#### Scenario: Unsupported shape is rejected

- **WHEN** an actor is constructed with an unsupported `shape` string
- **THEN** the system raises a diagnostic naming the offending shape and listing the supported primitives

### Requirement: Actor transform and material properties

`sim3d.Actor` SHALL expose settable properties `Translation` (`[x y z]` in
metres), `Rotation` (`[roll pitch yaw]` in radians), and `Scale` (`[sx sy sz]` or
a scalar), plus material properties `Color`, `Emissive` (RGB in `[0,1]`) and
`Opacity`. Reading a property after assignment SHALL return the assigned value.
Defaults SHALL be translation `[0 0 0]`, rotation `[0 0 0]`, scale `[1 1 1]`.

#### Scenario: Assign and read back a transform property

- **WHEN** `a.Translation = [2 0 1];` then `a.Rotation = [0 0 pi/2];`
- **THEN** `a.Translation` returns `[2 0 1]` and `a.Rotation` returns `[0 0 pi/2]`

#### Scenario: Scalar scale broadcasts

- **WHEN** `a.Scale = 2;`
- **THEN** the actor is scaled uniformly to `[2 2 2]`

### Requirement: Scene assembly and simulation loop

The system SHALL support assembling and stepping a scene: `add(world, actor)`
registers an actor into a world; `open(world)` initialises the recording (a
no-op-render under headless execution); `run(world, dt)` records one keyframe of
every registered actor's current transform and advances the timeline by `dt`
seconds; `close(world)` finalises the recording. Calling `run` before `open`, or
`run`/`open` on a world with no actors, SHALL raise a diagnostic.

#### Scenario: Stepping records one keyframe per call

- **WHEN** a world with one actor is opened and `run(world, 0.02)` is called 50 times, moving the actor each step
- **THEN** the recorded timeline contains 50 keyframes whose per-step transforms match the assigned values

#### Scenario: run before open is rejected

- **WHEN** `run(world, dt)` is called before `open(world)`
- **THEN** the system raises a diagnostic indicating the world has not been opened

### Requirement: Accumulate-then-export to a Babylon.js HTML player

`close(world)` and `sim3d.export(world, path)` SHALL write a single
self-contained HTML document that renders and plays the recorded animation in
Babylon.js. The emitted document SHALL be structurally equivalent to the
`matlabc -emit-mflowlink-babylon` artifact: an inline scene graph, an inline
keyframe timeline, and the embedded viewer logic, using the same right-handed
Z-up metres frame and yaw-pitch-roll quaternion convention as the block-diagram
path. Export SHALL succeed under headless execution without a display.

#### Scenario: Export produces a valid Babylon player

- **WHEN** a recorded world is exported via `sim3d.export(w, 'scene.html')`
- **THEN** `scene.html` is written, contains the Babylon scene-graph + keyframe timeline + viewer JS, and references the actor's transform keyframes

#### Scenario: Headless export

- **WHEN** export runs with no display available (headless CI)
- **THEN** the HTML file is still written and no rendering window is required

#### Scenario: Parity with the block-diagram emitter

- **WHEN** an equivalent scene is built both via `sim3d` commands and via a `.mflow` `signal_actor3d` model and both are exported
- **THEN** the two HTML documents carry the same actor geometry, frame convention, and transform-group layout (`[tx,ty,tz,rx,ry,rz,sx,sy,sz]`)

### Requirement: Availability in interpreted command mode and scripts

The `sim3d` API SHALL be usable from `matlabc -repl` (the interpreted MATLAB
commands mode) and SHALL produce identical recorded output when the same
sequence runs as a compiled/JIT script.

#### Scenario: REPL and script parity

- **WHEN** the same `sim3d` orbit-cube sequence runs once line-by-line in `-repl` and once as a script
- **THEN** both produce a byte-identical exported HTML player
