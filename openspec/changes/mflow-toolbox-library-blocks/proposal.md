## Why

Issue #343: toolbox **function**-level compatibility is largely complete (MPC, Control,
Comm, RF, Antenna, DSP, Computer Vision, Image Processing, Stats, …) and verified by the
`*_roadmap.md` / `openspec/specs/*` suites. But the mflowLink **editor/simulator** ships
~63 `signal_*` block kinds that are almost entirely Simulink-**core** (gain, sum,
integrator, transfer_fcn, state_space, lookup, mux/demux, scope) plus a handful of
toolbox blocks (`signal_mpc_move`, `signal_pid`). Every other toolbox capability is
reachable in a `.mflow` model only through the generic **MATLAB Function block**
(`signal_matlab_fcn`).

That's fine for one-off math, but it means a user can't drag a *Spectrum*, *FIR Filter*,
*QAM Modulator*, *AWGN Channel*, or *Image Filter* onto the canvas — the domains where
drag-and-drop time-domain modeling is the whole point. #343 asks whether these domains
want dedicated library blocks; this change answers "yes, where it adds real value over a
MATLAB Function block" and **scopes** the work: a repeatable recipe, an editor↔simulator
parity guard, and a prioritized per-domain block catalog delivered as follow-on PRs.

## What Changes

This is a scoping + first-slice change. The bulk is delivered incrementally as
**separate PRs** (one per block or per small domain tier).

- **Block-authoring recipe** — document the end-to-end anatomy of a toolbox library block
  so each follow-on PR is mechanical: register the kind + sample-time/loop-breaker class
  in `SignalFlowLowering.cpp`; add a simulator evaluator in `MflowLinkSim.cpp` (usually
  delegating to the existing toolbox runtime entry, e.g. `runtime/toolbox/dsp/*`); wire
  optional `-emit-c/cpp` lowering; add a `docs/mflowlink_blocks.md` row, a `SimulateRun`
  regression, and the editor `NodeKind` (separate IDE repo).
- **Editor↔simulator parity guard** *(first slice)* — a within-repo test that snapshots the
  registered `signal_*` block kinds, so adding/removing a kind is a visible, reviewable
  diff and a reminder to mirror it in the editor (the #343 §1 action item this repo can own).
- **First concrete block** *(first slice)* — one high-value DSP block (`signal_fft` or
  `signal_fir`) end-to-end through the recipe, as the worked example follow-on PRs copy.
- **Per-domain catalog** — a prioritized list (in `tasks.md` + the spec) of the concrete
  blocks to add per developed toolbox: DSP (FFT/IFFT, FIR, Biquad/IIR, window, spectrum),
  Communications (PSK/QAM mod-demod, AWGN channel, error-rate), RF (network/S-parameter
  blocks where time-domain-meaningful), Computer Vision / Image Processing (image source,
  convolution/filter, color-space, threshold), plus a few Control / Stats round-outs.
  Each entry is a future PR.

Non-goals: re-implementing toolbox math (the runtimes exist); the IDE editor `NodeKind`
additions (separate repo — this change only keeps them honest via the parity guard);
boiling the ocean across all ~38 toolboxes in one pass.

## Capabilities

### New Capabilities
- `mflow-library-blocks`: the catalog + authoring contract for toolbox-domain `signal_*`
  library blocks — the recipe each block follows (kind registration, simulator evaluator,
  optional emit lowering, docs/test), the editor↔simulator parity guarantee, and the
  prioritized per-domain coverage targets.

### Modified Capabilities
- (none) — `flowchart-frontend` already specifies the mflowlink block model + simulator
  generically; this adds a focused capability for the toolbox-block catalog rather than
  changing existing requirements.

## Impact

- **Code (per follow-on block)**: `lib/Flowchart/SignalFlowLowering.cpp` (kind + class),
  `lib/Flowchart/MflowLinkSim.cpp` (evaluator), occasionally a thin `runtime/toolbox/*`
  shim, `lib/MLIR/Passes/Emit{C,Cpp}.cpp` if codegen is wanted.
- **First slice**: a new `test/Flowchart/` parity-guard test + the first DSP block's
  evaluator, fixture, and `SimulateRun` checks.
- **Docs**: `docs/mflowlink_blocks.md` (per-block rows), `docs/mflow_link_roadmap.md`
  (domain coverage), this change's `tasks.md` (the live catalog).
- **Cross-repo**: the IDE (`matlab_llvm_ide_linux`) mirrors each new kind as a `NodeKind`;
  the parity guard flags drift.
