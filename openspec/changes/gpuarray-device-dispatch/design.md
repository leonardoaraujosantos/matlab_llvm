## Context

`gpuArray`/`gather` are identity builtins (`matlab_gpuArray_ctor(X){return X;}`,
`matlab_gather(X){return X;}`, `gpuDeviceCount()→1.0`) in
`runtime/toolbox/gpu/runtime_gpu_helpers.cpp` (post #333). #333 deliberately chose
identity after the earlier **classdef carrier** failed: matlabc does not dispatch
free functions / `size` / `gather` to class methods, and the object did not
round-trip through the REPL workspace, so `Ag*Bg` ran `matlab_matmul` on the object
pointer and returned empty. Identity restored host correctness but left **no hook**
to route an op to a device.

The runtime already dispatches polymorphic ops by a **magic tag** on the descriptor:
complex matrices (`matlab_mat_c`) and sparse matrices (`0xC0FFEE05`) are recognized at
runtime by a tag in the `matlab_mat` header and routed to the right kernel. gpuArray
fits the same pattern.

## Goals / Non-Goals

**Goals (Tier A, this change):**
- A device-resident value representation that (1) the lowering/runtime recognizes
  per-op, (2) round-trips through the REPL workspace, (3) stays host-correct with no
  device backend linked.
- A single per-op dispatch point that selects device vs host (always host in Tier A).
- Identical behavior on AOT and JIT/REPL lanes.

**Non-Goals (deferred to Tiers B–E):**
- Real device kernels / `runtime/gpu/*` linkage / h2d-d2h transfer (Tier C).
- Full operator surface coverage beyond what Tier A needs to prove the path (Tier B).
- A real-GPU CI lane and example rewiring (Tiers D, E).

## Decisions

### Decision 1 — Represent gpuArray as a magic-tagged runtime descriptor, not a classdef object
Use a `matlab_mat`-compatible descriptor carrying a **device magic tag** (mirroring the
sparse `0xC0FFEE05` / complex tags) plus retained host `data` and an opaque
`void* device_ptr` (null in Tier A). 

- **Why over a classdef carrier:** matlabc doesn't route free-function/`size`/`gather`
  calls to methods, and objects don't survive the REPL workspace round-trip — the two
  failures #333 documented. A tagged descriptor is just a `ptr`, so the workspace
  stores/loads it like any matrix and the tag persists.
- **Why over an MLIR-type-only tag:** a lowering-side type attribute is lost when the
  value crosses the runtime workspace boundary (REPL turn → turn). The tag must live in
  the runtime descriptor.
- **Alternative considered — reuse `matlab_mat` unchanged + a side table:** rejected;
  a side table keyed by pointer is fragile across copies and the workspace.

### Decision 2 — `gpuArray(X)` wraps, `gather(g)` unwraps, host-correct in Tier A
`matlab_gpuArray_ctor(X)` allocates the tagged descriptor sharing/copying X's host data
(no device upload yet). `matlab_gather(g)` returns the plain host matrix. Both keep the
exact host values, so `gather(Ag*Bg) == A*B` — the correctness #333 secured.

### Decision 3 — One dispatch shim per supported op, recognized by the tag
Each routed op (`mtimes`, element-wise, reductions, `gather`) calls a dispatch entry
that checks the device tag: tag present → device backend (Tier C) **or** host fallback
(Tier A); no tag → unchanged host path. The result of a device-resident op is itself
device-resident (re-tagged), so chains stay on the device path until `gather`. The
lowering routes an op to the dispatch entry when Sema/typing marks an operand as a
gpuArray; absent that, the existing host lowering is untouched (zero blast radius on
non-gpuArray code).

### Decision 4 — Tag check in the runtime, lane-agnostic
Because the tag lives in the descriptor and the dispatch shims are plain runtime
functions, AOT and JIT/REPL lanes share the exact same code path — satisfying the
"uniform across lanes" requirement without per-lane logic.

## Risks / Trade-offs

- **[A re-tag chain leaks the gpuArray tag into host code]** → Mitigation: `gather`
  is the only un-tagging exit; every dispatch shim that produces a host-observable
  scalar (e.g. a reduction to 1×1 used in control flow) returns host-tagged data, and
  fixtures assert `class(gather(...))`-equivalent host correctness.
- **[Descriptor layout drift vs `matlab_mat`]** → Mitigation: the tag reuses the
  existing header field the complex/sparse tags already occupy; no struct-size change.
- **[Operator coverage gaps make some gpuArray ops silently host-run as untagged]** →
  Mitigation: Tier A scopes the proven surface (`mtimes` + a representative element-wise
  + reduction + `gather`) and `log()`s/documents which ops are not yet routed; full
  coverage is Tier B.
- **[Perf: Tier A adds a tag check per op]** → negligible (one header read), and the
  fallback path is the same kernel as today.

## Migration Plan

1. Tier A (this change): replace identity with the tagged descriptor + dispatch shims
   (host fallback); add host-correctness fixtures; update roadmap Tier-1.4 status.
2. Tier B: extend dispatch coverage to the full element-wise/reduction surface across
   AOT + JIT.
3. Tier C: link `runtime/gpu/*`, implement h2d/d2h + device kernels behind the tag,
   `MATLAB_GPU_TARGET=auto`.
4. Tier D: real-GPU CI lane (RTX 5060) asserting parity + speedup.
5. Tier E: rewire `examples/gpu/*`; make the benchmark meaningful.

Rollback: Tier A is behavior-preserving on the host (results identical to identity);
reverting restores the identity builtin with no data-format migration.

## Open Questions

- Exact device magic-tag constant and which `matlab_mat` header field carries it
  (reuse the complex/sparse tag slot vs a dedicated flag) — settle in Tier-A
  implementation against `runtime/runtime_internal.h`.
- Whether `single`-typed gpuArrays need a distinct tag now or can defer to Tier C
  (the CPU lane is double-only today).
