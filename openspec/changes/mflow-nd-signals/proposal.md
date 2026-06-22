## Why

A mflowLink wire today carries a flat buffer plus a **2-D** shape (`OutRows × OutCols`).
That cap blocks color images (rows × cols × channels), tensor signals (NN activations,
batched data), and any model that wants to move a rank-≥3 array between blocks. The
narrower "color-image channels" task is one instance of a general need: **let a wire
carry an N-D shape, up to 6-D.**

Crucially, the rest of the stack is *already* N-D: the runtime rank-N descriptor
(`matlab_matN`, issue #76) stores an unbounded `ndims` + dynamic `dims` (binary ops cap
at 16-D); the MLIR type system uses `RankedTensorType` with an arbitrary `Dims` vector;
the DAP debugger renders shapes up to 8-D. The **only** hard 2-D limit is the mflowLink
signal-wire model. So this change is about lifting that one cap and reusing the existing
rank-N machinery — not building new tensor infrastructure.

## What Changes

- **Generalize the wire shape to N-D (≤ 6).** Replace the `OutRows`/`OutCols` pair with
  an `OutShape` dimension list (length 1–6) on `MflBlock` and in `MflowLinkSim`. Element
  count stays `prod(OutShape)`; the flat row-major buffer (`VecOut_`) is unchanged.
  `OutRows`/`OutCols` become a 2-D projection kept for backward compatibility.
- **Backward-compatible by construction.** A 1-D signal is `OutShape = [W]` (scope columns
  stay `<id>[k]`); a 2-D signal is `[R, C]` (`<id>[r,c]`). Existing models, tests, and CSV
  output remain byte-identical.
- **N-D scope columns.** A scope on a rank-N signal renders `<id>[i1,i2,…,iN]` (row-major),
  generalizing the existing `[r,c]`.
- **N-D shape inference + reshape.** The width/shape-inference fixpoint propagates the full
  `OutShape`; `signal_reshape` accepts a `shape = "d1,d2,…"` (up to 6 dims) so a flat frame
  can be reshaped into any rank.
- **Color images as a 3-D signal.** `signal_image_source` gains a `channels` param →
  `[rows, cols, channels]`; this subsumes the color-image-channels residual
  (`mflow-2d-image-signals` 6.3) as a special case of the N-D model.
- **Verify the cross-stack path.** Confirm a 6-D mflowLink signal flows through the
  `-emit-*` lowering, the `signal_matlab_fcn` JIT/AST interpreter, and the DAP debugger
  (all already ≥6-D capable); bump any incidental display cap below 6 if found.

No breaking changes; the 2-D cap is lifted, not replaced.

## Capabilities

### New Capabilities
- `mflow-nd-signals`: the N-D (rank 1–6) mflowLink wire-signal model — shape carrier,
  element-count contract, N-D shape inference + reshape, N-D scope rendering, and the
  cross-stack (emit / JIT / debugger) conformance.

### Modified Capabilities
- (none) — `flowchart-frontend` describes the wire/signal model generically; this adds a
  focused N-D capability. If conformance review finds the frontend spec states a 2-D cap
  explicitly, a delta is added at the specs phase. The grayscale `mflow-2d-image-signals`
  capability is the rank-2 special case and stays valid.

## Impact

- **Model/sim** (`include/matlab/Flowchart/MflowLinkModel.h`, `MflowLinkSim.h`,
  `lib/Flowchart/MflowLinkSim.cpp`): `OutShape` carrier, N-D scope rendering, reads/writes
  over the existing flat `VecOut_` (~65 `OutRows_/OutCols_` sites).
- **Lowering** (`lib/Flowchart/SignalFlowLowering.cpp`): N-D shape stamping + inference;
  N-D `signal_reshape`; `channels` on `signal_image_source`.
- **Reuse, no change**: runtime `matlab_matN` (≤16-D), MLIR `RankedTensorType`, DAP shape
  display (8-D) — all already exceed 6-D.
- **Tests**: `SimulateRun` N-D reshape + color-image fixtures; parity snapshot; the
  byte-identical 1-D/2-D guarantee re-checked against the full suite.
- **Docs**: `docs/mflowlink_blocks.md` "N-D signals" section; the 2-D image text becomes a
  special case.
