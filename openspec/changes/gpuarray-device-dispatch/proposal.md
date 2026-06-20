## Why

Issue #335: today "GPU" is a fiction on every in-process lane. `examples/gpu/*.m`
all use `Ag * Bg` (mtimes → host BLAS `matlab_matmul`); the GPU dispatcher
`matlab_gpu_gemm` is only reached via `gpucoder.gemm`, which no example uses.
Post #333, `gpuArray`/`gather` are pure identity builtins with **no hook** to
route an op to a device, and the CUDA/Metal/OpenCL backends under `runtime/gpu/`
are not linked into the REPL/examples/test build. Net: compile/run/JIT/Debug/REPL
are 100% CPU, and `benchmark_gpu_backend.m`'s "speedup" is CPU-vs-CPU (~1×).

A user should be able to write `Ag = gpuArray(A); Cg = Ag*Bg; C = gather(Cg)` and
have every op run on the device when one is present — uniformly across the
compiled (`-emit-*`) and interpreted (JIT/`-dap`/`-repl`) lanes — falling back to
the host CPU, numerically correct, when no device is available.

## What Changes

This is a multi-tier epic (roadmap Tier-1.4 carrier + Tier-5.9 operator overloads).
It is decomposed into five ordered sub-capabilities, delivered as **separate PRs**;
only Tier A is implemented in the first slice.

- **Tier A — routable `gpuArray` representation** *(first slice; this change)*. Replace
  the identity builtin with a value the lowering recognizes as device-resident and
  can route per-op, and that round-trips through the REPL workspace (the old
  classdef carrier did not). Provide the dispatch **hook** while keeping the host
  lane numerically correct — no real device backend yet (CPU fallback executes the
  op). Reconciles with #333: identity is replaced by a tagged-but-host-backed value.
- **Tier B — per-operation GPU dispatch** for the common surface (`*` mtimes,
  `+ - .* ./ .^`, relational/logical, reductions `sum`/`max`/`min`, `gather`),
  uniform across AOT and the JIT pipeline (`runJitSoftwareLowering`).
- **Tier C — real device init + host↔device transfer**: link the existing
  `runtime/gpu/{cuda,metal,opencl}` backends in-process; `gpuArray(X)` → h2d upload,
  ops stay on device, `gather` → d2h; `MATLAB_GPU_TARGET=auto` escalates to the
  present device, CPU fallback otherwise.
- **Tier D — real-GPU CI/dev lane** (the RTX 5060 box) asserting numeric parity
  vs CPU and speedup ≥ 1× at N ≥ 1024.
- **Tier E — wire `examples/gpu/*`** to the device path so the benchmark's speedup
  is meaningful.

## Capabilities

### New Capabilities
- `gpu-runtime-dispatch`: the in-process gpuArray model — device-resident value
  representation, per-operation dispatch (host fallback vs device), host↔device
  transfer, and `MATLAB_GPU_TARGET` device selection — shared by the AOT and JIT
  lanes. Tier A establishes the representation + dispatch hook + CPU-correct
  fallback; Tiers B–E extend it.

### Modified Capabilities
- (none) — the in-process `gpuArray`/`gather` surface was never spec'd as a
  requirement (the existing `gpu-codegen` capability covers only the standalone
  `-emit-{cuda,metal,opencl}` bundle emitters, which are unchanged). The #333
  identity behavior is superseded by the new `gpu-runtime-dispatch` capability
  rather than a modified requirement.

## Impact

- **Code**: `runtime/toolbox/gpu/runtime_gpu_helpers.cpp` (carrier + dispatch),
  `lib/MLIR/Lowering.cpp` / `lib/MLIR/Passes/LowerTensorOps.cpp` (recognize the
  representation, route ops), `tools/matlabc/main.cpp` (`runJitSoftwareLowering`,
  REPL workspace round-trip). Later tiers: `runtime/gpu/*`, the build/link graph,
  CI workflows, `examples/gpu/*`.
- **Tests**: new `test/Run` fixtures asserting host correctness of `gather(Ag*Bg)`
  and the element-wise/reduction surface through the gpuArray path on the CPU
  fallback (Tier A); a real-GPU parity/speedup lane lands in Tier D.
- **Docs**: `docs/gpu_coder_roadmap.md` Tier-1.4 / Tier-5.9 status; reconciles the
  #333 note.
- **Reconciliation**: #333 (gpuArray identity builtin) is revisited — host
  correctness it secured must be preserved by the new representation.
