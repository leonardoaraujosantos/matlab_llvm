# Tasks

Implementation is sliced so the representation (group 1–2) is validated by the first
block before the filter/threshold blocks build on it. Each block follows the 2-D
authoring recipe: register kind + shape stamping, simulator evaluator, `SimulateRun`
image fixture + numeric check, `docs/mflowlink_blocks.md` row, parity-snapshot bump, and
editor `NodeKind` (IDE repo).

## 1. Representation + conformance (the gate)

- [ ] 1.1 Confirm the `(OutRows, OutCols)` round-trip carries through `.mflow` for a 2-D signal and document the row-major flat-index contract `(r·cols + c)·channels + ch`
- [ ] 1.2 Add a `channels` shape field (default 1) and the `OutWidth == rows·cols·channels` invariant; thread it through the simulator's shape mirroring
- [ ] 1.3 Add the image-shape conformance check in `SignalFlowLowering` — sourced error when a declared image shape's element count mismatches the signal width
- [ ] 1.4 Verify all 89 existing 1-D blocks and the width-inference pass are unaffected (full flowchart ctest green)

## 2. First image block — `signal_image_source` (validates the model)

- [ ] 2.1 Register `signal_image_source`; stamp `OutWidth/OutRows/OutCols` from `rows`/`cols` params
- [ ] 2.2 Evaluator: emit the row-major image from `data` param into `VecOut_`
- [ ] 2.3 `image_source.mflow` fixture + `SimulateRun` check (width = rows·cols, shape columns, known pixel values); bump parity snapshot

## 3. `signal_image_filter` — 2-D convolution

- [ ] 3.1 Register + shape-stamp (output shape = input shape; preserved via inference)
- [ ] 3.2 Evaluator: direct 2-D correlation with a `kernel` matrix literal or named `type` (box/gaussian3/sobelx/sobely), zero-padded borders; delegate to `runtime/toolbox/images` where an entry exists
- [ ] 3.3 `SimulateRun` checks: normalized box kernel preserves a constant image (unity DC gain); a Sobel kernel responds at an edge and is ~zero in flat regions

## 4. `signal_threshold` — per-pixel binarize

- [ ] 4.1 Register + shape-stamp (shape preserved)
- [ ] 4.2 Evaluator: per-pixel `> level ? 1 : 0`
- [ ] 4.3 `SimulateRun` check: a ramp image binarizes at the level, shape preserved

## 5. Recipe + docs

- [ ] 5.1 `docs/mflowlink_blocks.md`: a "2-D image signals" section (representation, row-major contract, shape badge) + the per-block rows
- [ ] 5.2 Extend the block-authoring recipe with the 2-D variant (shape stamping, flat-index formula, image regression-check example)
- [ ] 5.3 Mark `mflow-toolbox-library-blocks` §5.1–5.3/5.5 done; cross-reference this capability

## 6. Color follow-on (scoped, after grayscale proves out)

- [ ] 6.1 Resolve the channel-layout open question (interleaved vs planar) and record it in the spec
- [ ] 6.2 Implement the `channels > 1` stride path and `signal_color_space` (e.g. RGB↔grayscale)
- [ ] 6.3 `signal_image_source` / `signal_image_filter` color coverage + checks
