# Tasks

Implementation is sliced so the representation (group 1–2) is validated by the first
block before the filter/threshold blocks build on it. Each block follows the 2-D
authoring recipe: register kind + shape stamping, simulator evaluator, `SimulateRun`
image fixture + numeric check, `docs/mflowlink_blocks.md` row, parity-snapshot bump, and
editor `NodeKind` (IDE repo).

## 1. Representation + conformance (the gate)

- [x] 1.1 Confirmed the `(OutRows, OutCols)` round-trip carries a 2-D signal through `.mflow`; documented the row-major flat-index contract `r·cols + c` (grayscale; `channels` is the color follow-on, group 6).
- [x] 1.2 Grayscale slice uses `OutWidth == rows·cols` over the existing shape mirroring; `channels` (default 1) deferred to group 6 so the model is validated before color multiplies the surface.
- [x] 1.3 Image-shape conformance check in `SignalFlowLowering` — `signal_image_source` errors when `data` pixel count ≠ `rows·cols`.
- [x] 1.4 All existing 1-D blocks + width inference unaffected — full flowchart ctest green (19/19), 195/195 SimulateRun.

## 2. First image block — `signal_image_source` (validates the model)

- [x] 2.1 Registered `signal_image_source`; stamps `OutWidth/OutRows/OutCols` from `rows`/`cols`.
- [x] 2.2 Evaluator emits the row-major image from `data` into `VecOut_`.
- [x] 2.3 Covered by `image_blocks.mflow` (feeds the filter/threshold paths) + the shape-naming check; parity snapshot → 92 kinds.

## 3. `signal_image_filter` — 2-D convolution

- [x] 3.1 Registered; output shape = input shape via the shape-inference fixpoint (catch-all inherit).
- [x] 3.2 Evaluator: direct 2-D correlation with a `kernel` literal or named `type` (box/gaussian3/sobelx/sobely), zero-padded borders. (Runtime-delegation path left open; in-sim conv is adequate for the small frames in scope.)
- [x] 3.3 `SimulateRun`: normalized box preserves a constant image's interior (unity DC, center=5); Sobel-x responds at a vertical edge (center=4) and is 0 in flat regions.

## 4. `signal_threshold` — per-pixel binarize

- [x] 4.1 Registered; shape preserved via inference.
- [x] 4.2 Evaluator: per-pixel `> level ? 1 : 0`.
- [x] 4.3 `SimulateRun`: a 2×2 ramp binarizes at 0.5 → `[0 1; 0 1]`, shape preserved.

## 5. Recipe + docs

- [x] 5.1 `docs/mflowlink_blocks.md` "2-D image signals" section (representation, row-major contract, `[row,col]` scope columns) + per-block rows + worked example.
- [x] 5.2 Block-authoring recipe extended with the 2-D variant (shape stamping for defining/preserving/element-count-changing blocks, the row-major flat-index formula, the small-image regression-check step) in `docs/mflowlink_blocks.md`.
- [x] 5.3 Marked `mflow-toolbox-library-blocks` §5 done; cross-referenced this capability.

## 6. Color follow-on (scoped, after grayscale proves out)

- [x] 6.1 Channel layout RESOLVED: **interleaved** RGB triples (matches typical image buffers; `(r·cols+c)·channels+ch`).
- [x] 6.2 `signal_color_space` — RGB↔grayscale conversion over interleaved triples (rgb2gray 3→1 Rec.601 luma; gray2rgb 1→3). Width handled by a rule in the width-inference fixpoint. `color_space.mflow` + SimulateRun (red→0.299, green→0.587; gray2rgb→[v v v]).
- [ ] 6.3 Color **image** integration — a `channels` param on `signal_image_source` and per-channel `signal_image_filter` so a 2-D color image carries its channel count in the shape. (Residual: `color_space` already converts; this threads channels through the 2-D image shape end-to-end.)
