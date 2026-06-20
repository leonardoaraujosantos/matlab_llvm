# Tasks

Tiers B–E are tracked here for the full epic but are **separate future PRs**; only
Tier A (groups 1–4) is implemented by the first slice.

## 1. Tier A — device-resident representation (runtime)

- [ ] 1.1 Pick the device magic-tag constant + the `matlab_mat` header field to carry it (reuse the complex/sparse tag slot), documented in `runtime/runtime_internal.h`
- [ ] 1.2 Add a tagged gpuArray descriptor allocator/accessors retaining host `data` + an opaque `void* device_ptr` (null in Tier A) + an `is_gpu(mat)` predicate
- [ ] 1.3 Rewrite `matlab_gpuArray_ctor(X)` to wrap X in the tagged descriptor (share/copy host data, no upload); `matlab_gather(g)` to return the plain host matrix; keep `gpuDeviceCount()` honest (0 when no backend linked)
- [ ] 1.4 Unit-level runtime check: `gather(ctor(A))` equals A; tag set on the wrapped value, cleared on the gathered value

## 2. Tier A — per-op dispatch shims (runtime)

- [ ] 2.1 Add dispatch entries for the proving surface: `mtimes`, one representative element-wise op (`.*`), one reduction (`sum`), and `gather`, each checking `is_gpu` → host fallback (Tier A) and re-tagging the result device-resident
- [ ] 2.2 Ensure a reduction that yields a host-observable scalar returns host-tagged data (no tag leak into control flow)
- [ ] 2.3 Mixed-operand rule: a binary op with one gpuArray operand promotes the host operand and produces a device-resident result

## 3. Tier A — lowering routes gpuArray ops to the dispatch shims

- [ ] 3.1 Recognize a device-resident operand in lowering (`lib/MLIR/Lowering.cpp` / `LowerTensorOps.cpp`) and emit the dispatch entry instead of the direct host runtime call, gated so non-gpuArray code is untouched (zero blast radius)
- [ ] 3.2 Verify the value round-trips through the REPL workspace store/load and stays recognized as device-resident across turns
- [ ] 3.3 Confirm identical lowering on the AOT path and the JIT pipeline (`runJitSoftwareLowering` in `tools/matlabc/main.cpp`)

## 4. Tier A — tests, docs, reconcile #333

- [ ] 4.1 `test/Run` fixture: `Ag=gpuArray(A); Cg=Ag*Bg; C=gather(Cg)` equals host `A*B`; element-wise + reduction through the gpuArray path host-correct (LLVM lane; emit-lane skips per the runtime-feature convention if needed)
- [ ] 4.2 REPL/JIT parity fixture asserting the same result interpreted as compiled
- [ ] 4.3 Update `docs/gpu_coder_roadmap.md` Tier-1.4 status (identity → tagged routable carrier) and reconcile the #333 note
- [ ] 4.4 Validate locally: `run-tests` + relevant `gpu-emit`/smoke lanes green; CI Full green before merge

## 5. Tier B — full per-op dispatch surface (future PR)

- [ ] 5.1 Extend dispatch to `+ - ./ .^`, relational/logical, `max`/`min`, and the remaining reductions, uniform across AOT + JIT
- [ ] 5.2 Coverage report: `log()`/document any gpuArray op not yet routed

## 6. Tier C — real device init + transfer (future PR)

- [ ] 6.1 Link `runtime/gpu/{cuda,metal,opencl}` into the REPL/examples/test build behind the tag
- [ ] 6.2 `gpuArray(X)` h2d upload, ops stay on device, `gather` d2h; `MATLAB_GPU_TARGET=auto` device escalation with CPU fallback

## 7. Tier D — real-GPU CI/dev lane (future PR)

- [ ] 7.1 RTX 5060 lane asserting numeric parity vs CPU and speedup ≥ 1× at N ≥ 1024

## 8. Tier E — wire the examples (future PR)

- [ ] 8.1 Route `examples/gpu/*` (`benchmark_gpu_backend`, `test_gpuarray_*`) through the device path; make the benchmark's speedup meaningful
