# Tasks

Sliced so the `OutShape` carrier + back-compat (group 1) is proven before reshape/scope
(group 2) and the image/cross-stack work (groups 3–4) build on it.

## 1. OutShape carrier + backward compatibility (the gate)

- [x] 1.1 Add `std::vector<int> OutShape` to `MflBlock` and `std::vector<std::vector<int>> OutShape_` to `MflowLinkSim`; document the row-major flat-index contract and the rank-6 ceiling
- [x] 1.2 Populate `OutShape` for every block: `[OutWidth]` (1-D) or `[OutRows, OutCols]` (2-D); keep `OutRows`/`OutCols` as the derived 2-D projection (`OutShape[0]`, `prod(rest)`)
- [x] 1.3 Enforce `prod(OutShape) == OutWidth` and a sourced error on rank > 6
- [x] 1.4 Full flowchart ctest + SimulateRun green with 1-D/2-D CSV byte-identical (regression: no existing column name or value changes)

## 2. N-D reshape + scope rendering

- [x] 2.1 Generalize `signal_reshape` to a `shape = "d1,…,dN"` param (1–6 dims); element-count mismatch is a sourced error
- [x] 2.2 Generalize the scope column namer to `<id>[i1,…,iN]` (row-major divmod over `OutShape`); rank 1/2 stay byte-identical
- [x] 2.3 Generalize the shape-inference fixpoint to propagate the full `OutShape`
- [x] 2.4 `nd_reshape.mflow` fixture + `SimulateRun` checks (a width-24 frame → `[2,3,4]`; rank-3 scope column names; a mismatched reshape errors)

## 3. Color image as a rank-3 signal (subsumes mflow-2d-image-signals 6.3)

- [x] 3.1 `signal_image_source` gains `channels` → `[rows, cols, channels]` (grayscale = `channels` absent/1)
- [~] 3.2 `signal_color_space` carries the channel axis (rank-3 in/out works today). Per-channel `signal_image_filter` (filtering each channel of a rank-3 image independently) is the one remaining follow-up — the filter currently treats the input as a single 2-D plane; threading `OutShape[2]` into a per-channel loop is a small, isolated change for when a color-image-filter pipeline is needed.
- [x] 3.3 `nd_color_image.mflow` + checks (an RGB image source shape `[2,2,3]`; rgb2gray → `[2,2,1]`)

## 4. Cross-stack conformance (verify, don't rebuild)

- [x] 4.1 Confirm a rank-6 signal flows through the `-emit-*` lowering (shape preserved; bump any incidental sub-6 cap found)
- [x] 4.2 Confirm a `signal_matlab_fcn` over an N-D input behaves (AST-interpreter fallback for vectors, as today) and a DAP inspect renders the N-D shape
- [x] 4.3 Mark `mflow-2d-image-signals` group 6.3 done (subsumed); cross-reference this capability

## 5. Docs

- [x] 5.1 `docs/mflowlink_blocks.md`: an "N-D signals" section (rank 1–6, row-major contract, `[i1,…,iN]` scope columns); the 2-D image text becomes a special case
- [x] 5.2 Note the reused rank-N infra (runtime `matlab_matN`, MLIR `RankedTensorType`, DAP 8-D display) so contributors know the cap was mflowLink-only
