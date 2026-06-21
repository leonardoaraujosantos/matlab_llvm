## ADDED Requirements

### Requirement: Toolbox library-block authoring contract
A toolbox-domain mflowLink library block SHALL be a `signal_*` block kind that exposes a
toolbox capability as a drag-and-drop block, authored through a fixed contract so each
block is mechanical to add: (a) the kind is registered with its sample-time and
loop-breaker classification in the signal-flow lowering; (b) it has a simulator evaluator
that produces the block's outputs each step, normally delegating to the existing toolbox
runtime entry rather than re-implementing the math; (c) it carries a `docs/mflowlink_blocks.md`
row and a `SimulateRun` regression asserting an analytically-known value; (d) optional
`-emit-c`/`-emit-cpp` lowering when codegen of the block is wanted.

#### Scenario: A library block round-trips through the simulator
- **WHEN** a `.mflow` model places a registered toolbox library block, wires it, and is run
  with `matlabc -simulate`
- **THEN** the block SHALL be accepted at lowering (classified, not rejected) and its
  evaluator SHALL produce the correct output signal (verified against a known value in the
  `SimulateRun` lane)

#### Scenario: A block delegates to the toolbox runtime
- **WHEN** a library block wraps an existing toolbox function (e.g. a DSP/Comm/RF runtime
  entry under `runtime/toolbox/*`)
- **THEN** its simulator evaluator SHALL call that runtime entry rather than duplicating the
  algorithm, keeping block and function numerically identical

### Requirement: Editor↔simulator block-kind parity guard
The set of `signal_*` block kinds the simulator/lowering registers SHALL be covered by an
in-repo guard so that adding or removing a kind is a visible, reviewable change and a
reminder to mirror it in the IDE editor's block library. (The editor lives in a separate
repo; this guard owns the in-repo half of the #343 §1 parity action.)

#### Scenario: Adding a block kind is flagged
- **WHEN** a new `signal_*` kind is registered (or one is removed) and the parity guard runs
- **THEN** the guard SHALL fail until its committed snapshot of registered kinds is updated,
  surfacing the change for editor mirroring

### Requirement: Prioritized per-domain coverage targets
The change SHALL maintain a prioritized catalog of the concrete library blocks to add for
the developed toolboxes, so coverage is tracked and each block lands as its own PR. Blocks
SHALL be added where drag-and-drop time-domain modeling adds value over the generic MATLAB
Function block (the function-first philosophy is preserved; not every function becomes a
block).

#### Scenario: DSP gets its first dedicated transform/filter block
- **WHEN** the first DSP library block (e.g. `signal_fft` or `signal_fir`) is implemented
  through the authoring contract
- **THEN** it SHALL ship with a simulator evaluator delegating to the DSP runtime, a
  `SimulateRun` regression, a docs row, and the parity-guard snapshot updated

#### Scenario: Catalog is the source of truth for remaining blocks
- **WHEN** a contributor picks up mflow block work for a domain (DSP / Comm / RF / CV-Image
  / Control / Stats)
- **THEN** the change's `tasks.md` catalog SHALL list the targeted blocks, their toolbox
  runtime backing, and their status, so the next PR is unambiguous
