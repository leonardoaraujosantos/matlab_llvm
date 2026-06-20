# Tasks

Tiers B–E are tracked here for the full epic but are **separate future PRs**; only
Tier A (groups 1–4) is implemented by the first slice.

## 1. Tier A — device-resident representation (runtime)

- [x] 1.1 Pick the device magic-tag constant + the `matlab_mat` header field to carry it (reuse the complex/sparse tag slot), documented in `runtime/runtime_internal.h` — `MATLAB_GPU_MAGIC 0xC0FFEE06`, magic at offset 0 like the complex/sparse tags
- [x] 1.2 Add a tagged gpuArray descriptor allocator/accessors retaining host `data` + an opaque `void* device_ptr` (null in Tier A) + an `is_gpu(mat)` predicate — `struct matlab_gpu`, `matlab_gpu_wrap`, `mat_is_gpu`, `mat_gpu_host`
- [x] 1.3 Rewrite `matlab_gpuArray_ctor(X)` to wrap X in the tagged descriptor (share host data, no upload); `matlab_gather(g)` to return the plain host matrix. (`gpuDeviceCount()` left at 1 — honest device detection deferred to Tier C, to avoid breaking `test/Run/gpu_device.stdout`)
- [x] 1.4 Verified: `gather(ctor(A))` equals A; tag set on the wrapped value, cleared on the gathered value (fixture)

## 2. Tier A — per-op dispatch shims (runtime)

- [x] 2.1 Dispatch entries for the proving surface: `mtimes` (`matlab_matmul_mm`), the element-wise `BINARY_MM/MS/SM` family, reductions (`COLWISE_REDUCE` + `sum_all`), and `gather` — each checks `mat_is_gpu` → host fallback and re-tags the result device-resident
- [x] 2.2 Inspectors (`size`/`numel`/`length`/`disp`) and the scalar chokepoint `matlab_mat_to_scalar` unwrap (no re-tag), so a gpuArray scalar reaching control flow / print sees host data
- [x] 2.3 Mixed-operand rule: `BINARY_MS`/`BINARY_SM` route a gpuArray×scalar to a device-resident result (verified `Ag .* 3`)

## 3. Tier A — dispatch is runtime-level, lane-uniform

- [x] 3.1 Dispatch implemented at the **runtime** op entry (tag sniff), NOT in lowering — more robust and lane-uniform by construction; lowering/Sema unchanged, so non-gpuArray code has zero blast radius (all ~770 run-tests fixtures pass)
- [x] 3.2 Verified the value round-trips through the REPL workspace and stays device-resident across turns (`-repl`: `Ag=gpuArray(A)` then `gather(Ag*Ag)`)
- [x] 3.3 Identical result on the AOT (`-emit-llvm`) and JIT/REPL lanes (the dispatch is in the shared runtime)

## 4. Tier A — tests, docs, reconcile #333

- [x] 4.1 `test/Run/gpuarray_device_dispatch.m`: `gather(Ag*Bg)==A*B`, element-wise, mixed scalar, reductions, inspectors all host-correct (LLVM + emit-typescript; emit-c/cpp/python skip per the gpu-fixture convention — harnesses don't link the gpu toolbox, python matrix-disp is numpy repr)
- [x] 4.2 REPL/JIT parity verified interpreted vs compiled (same result)
- [x] 4.3 Updated `docs/gpu_coder_roadmap.md` Tier-1.4 status (identity → tagged routable carrier) and reconciled the #333 note
- [ ] 4.4 Validate locally + CI: `run-tests` + emit-c/cpp/typescript green locally (0 failures); CI Full green before merge

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
