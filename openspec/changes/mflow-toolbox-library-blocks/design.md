## Context

The mflowLink simulator (`lib/Flowchart/MflowLinkSim.cpp`) evaluates a `.mflow` block graph
in execution order; lowering (`lib/Flowchart/SignalFlowLowering.cpp`) registers each
`signal_*` kind with a sample-time + loop-breaker classification. ~63 kinds exist today,
mostly Simulink-core. Toolbox math lives in `runtime/toolbox/<domain>/runtime_*.cpp`
(e.g. `runtime/toolbox/dsp/`, `comm/`, `rf/`, `vision/`, `images/`) and is already
function-complete and tested. The MATLAB Function block (`signal_matlab_fcn`) can call any
of it, but isn't a discoverable, parameterised, drag-and-drop library block.

## Goals / Non-Goals

**Goals**
- A repeatable, low-friction recipe so each toolbox block is a small mechanical PR.
- An in-repo parity guard so the registered-kind set can't silently drift from the editor.
- A prioritized catalog so the highest-value blocks land first.
- One worked DSP block end-to-end as the template.

**Non-Goals**
- Re-implementing toolbox algorithms (delegate to the existing runtimes).
- The IDE editor `NodeKind` additions (separate repo; the guard keeps them honest).
- Implementing the whole catalog here — only the recipe + guard + first block.

## Decisions

### Decision 1 — Blocks delegate to the existing toolbox runtime
A library block's evaluator extracts params + input signals and calls the toolbox runtime
entry (e.g. `matlab_fft_c`, a DSP FIR step). 
- **Why:** the math is already implemented and tested at the function level; a block is a
  thin adapter (param parse + signal marshalling), so block == function numerically and the
  surface added per block is tiny.
- **Alternative — reimplement in the simulator:** rejected (duplicates math, drift risk).

### Decision 2 — Sample-time + signal-width are explicit per block
Each block declares its sample time (continuous / discrete-with-period / constant) and
output width at registration, reusing the existing classification + the `VecOut_` vector
path for multi-element signals (FFT bins, image rows). Frame/vector blocks use `VecOut_`;
sample-based blocks use the scalar `Out_`.
- **Why:** the simulator's scheduler + algebraic-loop breaker already key off these; a DSP
  filter is a discrete block, an FFT emits a vector frame.

### Decision 3 — Parity guard is a committed snapshot test
A `test/Flowchart/` test enumerates the registered `signal_*` kinds (parsed from the
lowering registration or a single exported list) and diffs against a committed
`registered_block_kinds.txt`. Adding a kind fails the test until the snapshot is updated,
which is the reviewer's cue to file/track the editor `NodeKind`.
- **Why over a cross-repo check:** the editor is a separate repo; an in-repo snapshot is the
  honest, owned half. It also documents the canonical kind list in one place.
- **Alternative — derive from a shared schema:** heavier; revisit if the editor and compiler
  ever share a generated kind enum.

### Decision 4 — First block: a DSP transform/filter
Pick `signal_fft` (vector-frame output, delegates to `matlab_fft_c`) or `signal_fir` (sample
or frame FIR, delegates to the DSP runtime) as the worked example, because DSP already has
partial block coverage (so the surrounding infra — discrete sample times, the `VecOut_`
frame path — is exercised) and the value is obvious.

## Risks / Trade-offs

- **[Per-block param/port conventions drift]** → Mitigation: the recipe pins naming (`u1..uN`
  inputs, `out`/`out1..outM` outputs, param keys) and every block ships a `SimulateRun`
  regression.
- **[Vector/frame signals stress the scalar-centric simulator]** → Mitigation: reuse the
  existing `VecOut_` width path (already used by mux/demux/vector gain); the first DSP block
  validates it for a transform.
- **[Editor parity is cross-repo and can still lag]** → Mitigation: the in-repo guard makes
  the compiler-side list authoritative and the lag visible; the editor mirrors from it.
- **[Catalog scope creep]** → Mitigation: function-first rule — only add a block where
  drag-and-drop modeling beats a MATLAB Function block; the catalog is prioritized, not
  exhaustive.

## Migration Plan

1. This change: recipe (design) + parity guard + first DSP block + catalog (tasks).
2. Follow-on PRs: one block (or a small same-domain tier) each, per the catalog priority —
   DSP transforms/filters → Comm mod/demod + AWGN → CV/Image filters → RF → Control/Stats
   round-outs. Each updates the parity snapshot and `docs/mflowlink_blocks.md`.

Rollback: blocks are additive; reverting a block PR removes its kind + evaluator + snapshot
line with no effect on existing models.

## Open Questions

- Frame vs sample semantics for DSP blocks: do FFT/FIR operate on a buffered frame
  (rate-transition-style) or sample-by-sample? Settle per block against the DSP runtime's
  shape.
- Which RF capabilities are time-domain-meaningful as blocks vs inherently frequency-domain
  (better left to functions)? Decide during the RF tier.
- Image blocks imply 2-D signals on wires — confirm the bus/vector model carries an image,
  or gate image blocks behind a small 2-D signal extension.
