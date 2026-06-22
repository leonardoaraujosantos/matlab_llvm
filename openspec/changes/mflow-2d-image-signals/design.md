## Context

A mflowLink signal is a width-`W` value on a wire. The simulator (`MflowLinkSim`)
stores it as a scalar `Out_[I]` plus, when `W > 1`, a flat `VecOut_[I]` of length `W`,
and already tracks a 2-D **shape** alongside it: `OutRows_[I]` / `OutCols_[I]` with the
invariant `OutRows·OutCols == OutWidth` (see the constructor's shape mirroring and
`signal_reshape`). The lowering (`SignalFlowLowering`) stamps `OutWidth`/`OutRows`/
`OutCols` per block and runs a fixpoint width-inference pass; vector `signal_constant`
and `signal_reshape` already round-trip a `(rows, cols)` shape through `.mflow`.

So the data plane for 2-D already exists — what is missing is a **decision** that an
image *is* a width-`rows·cols·channels` signal carrying that shape, and the block-level
conventions that follow. Vision/Image is the only developed-toolbox tier with no
mflowLink blocks purely because this convention was never settled
(`mflow-toolbox-library-blocks` §5.1 is an explicit Open Question). Picking it wrong is
expensive: every image block's port contract and the editor's wire rendering depend on
it. Hence this design-first change.

## Goals / Non-Goals

**Goals:**
- Settle the 2-D-image signal representation on a wire, with alternatives weighed.
- Keep all 89 existing 1-D blocks and the width-inference pass working unchanged.
- Reuse the existing `OutRows`/`OutCols`/`VecOut_` machinery — no new signal container.
- Land enough grayscale image blocks (source, 2-D filter, threshold) to validate the
  representation end-to-end with deterministic numeric tests.
- Document a 2-D variant of the block-authoring recipe.

**Non-Goals:**
- Multi-channel/color is scoped but not implemented here (the `channels > 1` path); a
  grayscale-first slice validates the model before color lands.
- Streaming video (a time sequence of frames) — each step already carries one frame;
  inter-frame buffering is out of scope.
- Large-image performance / FFT-based convolution — direct 2-D conv over the small
  frames typical of a control/vision model, matching the existing "small frames, no
  FFTW" stance of the DSP blocks.
- Editor rendering of 2-D wires (cross-repo; noted, not built).

## Decisions

### D1 — Represent a 2-D image as a flattened row-major vector + `(rows, cols, channels)` shape

An image signal is a width-`rows·cols·channels` vector in `VecOut_`, **row-major**
(`index = (r·cols + c)·channels + ch`), with the shape carried in the existing
`OutRows`/`OutCols` (and a new per-block `channels`, defaulting to 1). The element-count
contract (`OutWidth == rows·cols·channels`) is unchanged, so width inference, `mux`,
`reshape`, and every 1-D consumer keep working — an image is "just a vector with a
shape."

**Why over the alternatives:**

- **(chosen) Flattened vector + shape metadata.** Zero new data structures; reuses the
  shape fields and the `.mflow` round-trip that `reshape`/vector-`constant` already use.
  1-D and 2-D signals interoperate (a 1-D filter can still run per-pixel). Cost: blocks
  must read/write with stride math, and a wire alone doesn't enforce "this is 2-D"
  (mitigated by the shape metadata + a conformance check).
- **(rejected) Dedicated 2-D wire/tensor type.** A separate `MatOut_` container with its
  own type tag. Cleaner typing, but forks the signal model in two: every routing/mux/
  scope/inference path would need a 2-D branch, and the 89 existing blocks would need
  audit. High blast radius for marginal gain; the rank-N descriptor work (issue #76,
  closed) already showed flattened+shape is the project's idiom.
- **(rejected) Bus/struct of (data, rows, cols).** Reuses `signal_bus_creator`. But it
  pushes shape into named fields the simulator would have to special-case per block, and
  makes the common "image in → image out" wire clumsy (un/repack each hop). Worse
  ergonomics than metadata that rides the existing signal.

### D2 — Image blocks declare their output shape at lowering time

Like `signal_fft`/`signal_lqr`, each image block stamps `OutRows`/`OutCols`/`OutWidth`
in the per-kind dispatch (NOT via the inherit sentinel) so a downstream consumer sees the
2-D shape without running the block. Shape-preserving blocks (`filter`, `threshold`)
copy the input shape via the inference pass when their own params don't override it;
shape-defining blocks (`image_source`) set it from `rows`/`cols` params. This follows the
established rule — *any block whose output width/shape differs from its input must stamp
explicitly, else the catch-all makes it inherit* (the demod/lqr/dnn lesson).

### D3 — 2-D convolution semantics

`signal_image_filter` does direct 2-D correlation/convolution of the `rows×cols` input
with an `kh×kw` kernel (`kernel` matrix literal, or a named `type`: `box`/`gaussian3`/
`sobelx`/`sobely`), zero-padded at the borders (`padding: "zero"` default; `"replicate"`
optional). Delegates to a `runtime/toolbox/images` entry when one exists; otherwise an
in-sim O(rows·cols·kh·kw) loop — adequate for the small frames in scope and consistent
with the inline-DFT/inline-matmul stance.

### D4 — Grayscale-first, color as a typed follow-on

`channels` defaults to 1 and the first three blocks are grayscale. `signal_color_space`
(and any RGB filter) needs the `channels > 1` stride and is implemented only after the
grayscale path is proven, so the representation is validated before color multiplies the
test surface.

## Risks / Trade-offs

- **A bare wire doesn't self-identify as 2-D** → the `(rows, cols, channels)` shape
  metadata is the source of truth; a conformance check rejects a block whose declared
  shape mismatches its element count, and the parity/SimulateRun tests pin behavior.
- **Row-major vs column-major confusion** (MATLAB is column-major) → fix row-major
  explicitly in the spec and the recipe, document the index formula, and assert it in a
  test where transpose would change the result.
- **Scope creep into a full image toolbox** → the Non-Goals fence it to grayscale
  source/filter/threshold; color and video are deferred with explicit gates.
- **Editor can't render 2-D yet** → blocks are usable headless (simulate + numeric
  check) without the IDE; editor work is tracked separately and not a blocker.

## Migration Plan

Additive, no migration. 1-D signals are unchanged (`shape (1, W)`). The new capability
spec + blocks layer on top; if a future review wants the `flowchart-frontend` spec to
state the shape contract, that delta is added without touching existing behavior.

## Open Questions

- Does `flowchart-frontend` need a delta to state the shape-on-wire contract explicitly,
  or is the new `mflow-image-signals` capability sufficient? (Resolve at the specs phase.)
- Color layout when `channels > 1`: interleaved (`...ch`) vs planar (`r·cols·... ` per
  channel)? Interleaved is proposed (matches typical image buffers); confirm before the
  color follow-on.
- Should `signal_image_source` also support loading from a file path, or only
  param/constant images in this slice? (Proposed: param/constant only here; file source
  needs the same workspace-binding work `signal_from_workspace` is blocked on.)
