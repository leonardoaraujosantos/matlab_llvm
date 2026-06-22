## Context

mflowLink signals are flat `VecOut_[I]` buffers with `OutWidth_[I]` elements and a 2-D
shape `(OutRows_[I], OutCols_[I])`, invariant `OutRows·OutCols == OutWidth`. The lowering
stamps these per block and runs a width- then shape-inference fixpoint. The 2-D cap is the
sole rank limit in the whole stack — verified:

| Layer | Rank support | Source |
|---|---|---|
| Runtime `matlab_matN` descriptor | unbounded `ndims`; binary ops clip at 16-D | `runtime/runtime_internal.h:91`, `matlab_runtime.cpp:7447` |
| MLIR type system | arbitrary (`RankedTensorType`, `std::vector<int64_t> Dims`) | `include/matlab/Sema/Type.h:89`, `lib/MLIR/TypeMapper.cpp:50` |
| DAP debugger shape display | 8-D (`dims[8]`) | `tools/matlabc/main.cpp:5524` |
| DAP runtime reflection | unbounded | `runtime/runtime_debug.cpp:2972` |
| **mflowLink wire** | **2-D (`OutRows`/`OutCols`)** | `MflowLinkModel.h:76`, `MflowLinkSim.h:422` |

So 6-D mflowLink signals require lifting one cap and reusing everything else.

## Goals / Non-Goals

**Goals:**
- A mflowLink wire carries a rank 1–6 shape; element count = `prod(shape)` over the
  unchanged flat row-major buffer.
- 1-D and 2-D signals stay byte-identical (models, tests, CSV).
- N-D shape inference + reshape; N-D scope columns; color image = rank-3 special case.
- Confirm 6-D signals survive the emit lowering, the matlab_fcn JIT/AST path, and the DAP
  debugger.

**Non-Goals:**
- Rank > 6 (the chosen ceiling; runtime allows 16, but 6 covers images+batch+NN and keeps
  fixed-size shape arrays cheap).
- New tensor math kernels — reuse the runtime/MLIR rank-N ops.
- Per-axis broadcasting semantics beyond what 1-D already does (scalar broadcast stays).

## Decisions

### D1 — `OutShape` (length 1–6) is the canonical shape; `OutRows`/`OutCols` a derived 2-D projection

Add `std::vector<int> OutShape` to `MflBlock` and `std::vector<std::vector<int>> OutShape_`
to the simulator. The invariant becomes `prod(OutShape) == OutWidth`. Keep `OutRows`/
`OutCols` populated as `OutShape[0]` and `prod(OutShape[1:])` (a 2-D view) so the ~65
existing call sites compile and behave identically for 1-D/2-D.

**Why over the alternatives:**
- **(chosen) Add `OutShape`, keep R/C as a view.** Smallest blast radius — existing 2-D
  code keeps reading R/C; only shape-defining/inference/rendering paths read `OutShape`.
  Back-compat is structural, not just tested.
- **(rejected) Replace R/C with `OutShape` everywhere.** Cleaner end state but touches all
  65 sites at once, risking 2-D regressions for no functional gain over the view approach.
- **(rejected) A separate N-D side-channel only for new blocks.** Forks the model; 2-D and
  N-D paths diverge and reshape can't bridge them.

### D2 — Fixed ceiling 6, capacity-checked at lowering

`OutShape` holds ≤ 6 dims; a declared/reshaped shape with rank > 6 is a sourced error.
6 is the contract ceiling (images = 3, batched images = 4, NN activations = 4–5, leaving
headroom). Trailing singleton dims are squeezed (MATLAB convention), matching `matN_alloc`.

### D3 — N-D scope rendering generalizes `[r,c]`

The scope column namer walks the row-major index for rank N: element `e` → indices
`(i1,…,iN)` via successive divmods over `OutShape`, rendered `<id>[i1,…,iN]` (1-based).
Rank 1 stays `<id>[k]`, rank 2 stays `<id>[r,c]` — identical bytes to today.

### D4 — Color image = rank-3 signal; subsumes `mflow-2d-image-signals` 6.3

`signal_image_source` with `channels = c` emits `OutShape = [rows, cols, c]` (interleaved,
the resolved layout). The grayscale path is `c = 1` (rank-2). `signal_color_space` already
converts the flat interleaved triples; with the N-D shape it now carries the channel axis
explicitly. Per-channel `image_filter` reads `OutShape[2]` for the channel count.

### D5 — Cross-stack: verify, don't rebuild

The emit lowering (`SubsystemToMatlab` → MLIR) and the matlab_fcn JIT/AST interpreter
operate on the flat buffer; a 6-D mflow signal lowers to a rank-N MLIR tensor / a flat
vector the interpreter already handles. The DAP debugger renders ≤ 8-D. The task list
includes a conformance check (a 6-D signal through emit + a `signal_matlab_fcn` + a DAP
inspect) rather than new code; any incidental sub-6 cap found is bumped.

## Risks / Trade-offs

- **2-D regression from the R/C view** → keep `OutRows`/`OutCols` exactly `OutShape[0]` /
  `prod(rest)` and re-run the full suite; the 1-D/2-D CSV byte-identity is an explicit test.
- **Shape/width-inference divergence for N-D** → inference propagates the whole `OutShape`;
  the element-count check (`prod == OutWidth`) is the single source of truth, as today.
- **Silent rank-6 overflow** → lowering errors on rank > 6 (D2), not truncation.
- **matlab_fcn over an N-D input** → the JIT path is single-scalar/≤8-arity; an N-D signal
  into a matlab_fcn block falls back to the AST interpreter (already the rule for vectors),
  flagged in the conformance check.

## Migration Plan

Additive. `OutShape` defaults to `[OutWidth]` (1-D) or `[OutRows, OutCols]` (2-D) for every
existing block, so nothing changes without an explicit N-D declaration. No data migration.

## Open Questions

- Squeeze trailing singletons on the wire, or preserve declared rank for round-trip
  fidelity? (Proposed: squeeze on compute, preserve the declared `shape` param for display —
  matches `matN_alloc`.)
- Should `signal_reshape`'s element-count error mention the N-D shape, and should a
  `permute`/`squeeze` block follow? (Proposed: reshape only in this change; permute is a
  follow-on if needed.)
