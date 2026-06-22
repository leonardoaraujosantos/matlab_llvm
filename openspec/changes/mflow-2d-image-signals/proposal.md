## Why

mflowLink now has dedicated blocks for every developed toolbox that benefits from
time-domain modeling **except Computer Vision / Image Processing** (89 block kinds
across Comms, HDL, DSP, Control, Sensor Fusion, Stats, Deep Learning, RL). Those two
toolboxes have no blocks for one reason: a mflowLink wire today carries a *1-D* signal
(a width-`W` vector with row/col shape metadata), and there is no agreed way to flow a
**2-D image** between blocks. This single unresolved question gates the entire
Vision/Image block tier (image source, 2-D filter, color-space, threshold — OpenSpec
`mflow-toolbox-library-blocks` §5).

This change decides the 2-D-signal representation and lands the first image blocks on
top of it. It is deliberately design-first: the representation choice is hard to reverse
once blocks depend on it, so we settle it explicitly before building.

## What Changes

- **Decide the 2-D-signal-on-a-wire representation.** Adopt **flattened row-major
  vector + explicit `(rows, cols, channels)` shape**, reusing the existing
  `OutWidth`/`OutRows`/`OutCols` machinery rather than introducing a new wire type.
  (Alternatives — a dedicated 2-D wire type, or a bus/struct — are evaluated and
  rejected in `design.md`.) A wire's element count stays `rows·cols·channels`; the shape
  travels as already-tracked metadata, so existing 1-D blocks and the width-inference
  pass are unaffected.
- **Add an image-shape contract** to the simulator: blocks may declare a 2-D output
  shape; `signal_reshape` already carries `(rows, cols)`, so a vector source + reshape is
  the canvas idiom for "make a frame an image" until a dedicated source lands.
- **Land the first image blocks** on the new representation:
  - `signal_image_source` — emit a constant/from-param image (rows×cols, grayscale).
  - `signal_image_filter` — 2-D convolution / separable kernel (box/Gaussian/Sobel),
    delegating to `runtime/toolbox/images` where available, else an in-sim 2-D conv.
  - `signal_threshold` — per-pixel binarize at a level.
  - `signal_color_space` — left as a follow-on (needs the `channels > 1` path); scoped
    here, implemented after the grayscale blocks validate the representation.
- **Author the block-recipe extension** for 2-D blocks (shape stamping, the
  `OutRows/OutCols` round-trip, a `SimulateRun` numeric check on a small image) so future
  Vision/Image blocks follow a documented pattern.

No breaking changes: 1-D signals keep width-`W`, shape `(1, W)`; the 2-D path is additive.

## Capabilities

### New Capabilities
- `mflow-image-signals`: the 2-D-image signal representation on a mflowLink wire
  (flattened row-major + `(rows, cols, channels)` shape, the inference rules, and the
  conformance contract) plus the first grayscale image blocks built on it.

### Modified Capabilities
- (none) — `flowchart-frontend` specifies the mflowlink block/signal model generically;
  this adds a focused capability for the 2-D-image extension rather than re-specifying the
  frontend. If the conformance review finds the frontend spec must state the shape
  contract explicitly, a delta will be added at the specs phase.

## Impact

- **Simulator** (`lib/Flowchart/MflowLinkSim.cpp`): 2-D-aware reads/writes over the
  existing `VecOut_` + `OutRows_`/`OutCols_`; the new image-block evaluators.
- **Lowering** (`lib/Flowchart/SignalFlowLowering.cpp`): output-shape stamping for the
  image blocks; width inference unchanged (element count is still `rows·cols·channels`).
- **Runtime** (`runtime/toolbox/images`, `runtime/toolbox/vision`): delegation targets
  for 2-D convolution / color-space where a function-level entry already exists.
- **Tests**: `test/Flowchart/SimulateRun` image fixtures + checks; the block-kind parity
  snapshot grows by the new kinds; editor `NodeKind` parity (IDE repo).
- **Docs**: `docs/mflowlink_blocks.md` — a "2-D image signals" section + the per-block
  rows; the authoring recipe gains a 2-D variant.
- **Cross-repo**: the IDE must render a 2-D signal (shape badge on the wire) and the new
  blocks' ports — noted for the editor team, not built here.
