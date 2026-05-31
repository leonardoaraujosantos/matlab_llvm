# GPU Coder Toolbox — Compatibility + Multi-Backend Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug +
emit lanes) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** GPU-Coder programs — **extended beyond the
MathWorks CUDA-only lane to three backends**: NVIDIA **CUDA** + portable
**OpenCL** on Linux, and **Apple Metal** on macOS.

> **🟢 On-hardware validation (2026-05-31, issue #25).** The CUDA and
> OpenCL backends are now validated end-to-end on real NVIDIA hardware
> (RTX 5060, sm_120) — not just emission. CUDA runs cuBLAS `Dgemm` (fp64)
> + NVRTC-JIT kernels via the driver API; OpenCL runs an fp64 GEMM kernel
> + the AXPY `-emit-opencl` bundle via the device's ICD. Both are opt-in
> CMake flags (`-DMATLAB_LLVM_GPU_CUDA=ON` / `-DMATLAB_LLVM_GPU_OPENCL=ON`,
> default OFF) with HW-gated validation lanes
> (`test/Run/run_gpu_{cuda,opencl}_validation.sh`). Metal (Tier-2) was
> already validated on Apple silicon. Details in the Tier-3 / Tier-4
> status notes below.

Source: *GPU Coder™ User's Guide* (R2026a, 7 chapters): Functions
Supported for GPU Code Generation · Kernel Creation from MATLAB Code
(element-wise loops · reductions · library calls · custom CUDA kernels ·
stencils · matrix-matrix · GPU memory manager · GPU arrays · half
precision · cuBLAS/cuSOLVER/cuFFT examples) · Kernel Creation from
Simulink Models · Deep Learning (cuDNN / TensorRT / ARM Mali / INT8) ·
Targeting Embedded GPU Devices (Jetson · DRIVE · packNGo · PIL) ·
Troubleshooting (reports · traceability · profiler · kernel analysis ·
memory bottlenecks · loop dependencies) · Troubleshooting CUDA Errors.

This is the project's **first compute-accelerator backend** — the
existing emit lanes (`-emit-c`/`-emit-cpp`/`-emit-python`/
`-emit-typescript`/`-emit-systemverilog`/`-emit-cocotb`) all target host
CPUs or RTL. GPU Coder adds the **device offload axis**: a tagged loop
body or `arrayfun`/`gpucoder.*` call must be outlined, transferred to
device memory, and dispatched as a kernel — and *that whole pipeline must
work across three different driver stacks and two operating systems*.

**The defining architectural fact**: MATLAB ships GPU Coder as
NVIDIA-only (CUDA / cuBLAS / cuSOLVER / cuFFT / cuDNN / TensorRT). This
roadmap is *deliberately* broader — we treat the **kernel IR** as the
single source of truth and ship *three independent backends* that
consume it:

- **macOS / Apple Silicon → Metal** (Metal Shading Language source +
  Metal Performance Shaders + `metal-cpp` host driver, JIT through
  `MTLLibrary newLibraryWithSource:`). Unified Memory on Apple Silicon
  collapses the host↔device copy.
- **Linux / NVIDIA → CUDA** (NVRTC-compiled `.cu` source + cuBLAS /
  cuSOLVER / cuFFT for library replacement, half via `__half`).
- **Linux / portable → OpenCL** (`clBuildProgram` of `.cl` source +
  clBlast / clFFT for library replacement, runs on NVIDIA / AMD / Intel
  / Mali). Linux fallback when CUDA isn't installed; also the path that
  matches GPU Coder's published ARM Mali story.

A `coder.gpu.kernelfun`-tagged MATLAB function compiles to *the same*
`matlab.gpu.kernel` op + outlined private `func.func`; the difference is
only which **emit-pass** lowers it to which **kernel-source dialect**
(MSL / CUDA-C / OpenCL-C) and which **runtime ABI** dispatches it
(`runtime/gpu/metal/` / `runtime/gpu/cuda/` / `runtime/gpu/opencl/`).
This mirrors the existing `-emit-c` / `-emit-python` / `-emit-ts`
split — same MLIR up front, language-specific printers at the back —
applied to a *device* code generator.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/gpu/mandelbrot_gpu.m`](../examples/gpu/mandelbrot_gpu.m): *the
canonical GPU Coder UG demo — `coder.gpu.kernelfun` on a per-pixel
Mandelbrot loop, the same source compiles and runs on Metal (macOS) /
CUDA (Linux) / OpenCL (Linux), and a PIL-style numerical-equivalence
test confirms each device result matches the CPU reference within
`1e-9` absolute*. This exercises kernel outlining + grid/block sizing +
host↔device transfer + library-free element-wise dispatch on all three
backends end-to-end; achieving it closes **GPU-Tier-1** + the first
backend of **GPU-Tier-2/3/4**. The differentiated **GPU-Tier-5** demo
is [`examples/gpu/streaming_fft.m`](../examples/gpu/streaming_fft.m):
*a windowed FFT pipeline mapping `gpucoder.reduce` + a frame loop +
device FFT to MPSGraph on Metal / cuFFT on CUDA / clFFT on OpenCL,
verified against the CPU `fft` reference*. The **multi-backend AOT
emit** demo (**GPU-Tier-6**) is
[`examples/gpu/sobel_emit.m`](../examples/gpu/sobel_emit.m): *one MATLAB
file emits a self-contained `.metal` + driver / `.cu` + driver / `.cl`
+ driver from a single source, each builds with its native toolchain and
links to the platform's BLAS/FFT lib*.

Companion docs:
[`embedded_coder_roadmap.md`](embedded_coder_roadmap.md) (the existing
per-target emit-lane convention — Python/C/C++/TS/SV; GPU follows the
same architectural template),
[`emit_cocotb.md`](emit_cocotb.md) (the existing JIT-of-emitted-source
pattern — cocotb's `Verilator` is the Linux-CUDA NVRTC analogue),
[`emit_systemverilog.md`](emit_systemverilog.md) (per-pass emit pattern
in `lib/MLIR/Passes/EmitSystemVerilog.cpp` — the GPU emit passes mirror
it), [`feature_status.md`](feature_status.md),
[`debug.md`](debug.md) + [`repl.md`](repl.md) (the DAP + REPL surface
that the `gpuArray` inspector hooks),
[`fixed_point_toolbox_roadmap.md`](fixed_point_toolbox_roadmap.md) (the
half-precision codegen story leverages the same parametric-width
storage).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order.
  **Tier-1** is the **portable GPU kernel IR + outlining + host carrier**
  — the `matlab.gpu.kernel`/`reduce`/`stencil`/`gemm` ops, kernel-body
  outlining (reusing the `LowerParfor.cpp` scaffolding), `gpuArray`
  classdef + handle ABI, runtime-agnostic launch dispatch, DAP + REPL
  inspector. This is the keystone — everything below picks a backend.
  **Tier-2** is the **Metal backend on macOS** (Metal C++ host driver +
  MSL emit + MPS GEMM/FFT, JIT compile via `MTLLibrary`, Apple Silicon
  Unified Memory fast path). **Tier-3** is the **CUDA backend on Linux**
  (NVRTC JIT of emitted `.cu`, cuBLAS / cuSOLVER / cuFFT library
  replacement, half via `__half`). **Tier-4** is the **OpenCL backend
  on Linux** (`clBuildProgram` JIT of emitted `.cl`, clBlast / clFFT,
  AMD / Intel / Mali coverage). **Tier-5** is the **GPU Coder design
  patterns** (`gpucoder.reduce` / `matrixMatrixKernel` / `stencilfun` /
  `coder.ceval` / `coder.gpu.constantMemory`) — wired across all three
  backends. **Tier-6** is the **AOT emit lanes** —
  `-emit-cuda`/`-emit-opencl`/`-emit-metal` — producing a
  self-contained host driver + one or more device-source files (the
  GPU equivalent of the shipped `-emit-cocotb`). **Tier-7** is the
  **numerical-equivalence + half-precision + memory-manager** polish:
  PIL-style harness, fp16 codegen, GPU memory manager (reusable
  device-buffer pool — the cudaMalloc-coalescing optimisation from the
  UG).
- **Effort** in the existing per-tier session cadence (one focused
  session ≈ a half-day; "week" ≈ 5 sessions). Rough totals: **T1
  ~3 wk (kernel IR + outlining + carrier + REPL is the heavy lift) ·
  T2 ~2.5 wk (Metal first because it's the dev box) · T3 ~2 wk
  (CUDA — NVRTC + lib link) · T4 ~2 wk (OpenCL) · T5 ~2 wk
  (design patterns × 3 backends) · T6 ~1.5 wk (AOT emit) · T7 ~1.5 wk
  (PIL + fp16 + mempool) — ~14 wk full**. The fastest demonstrable
  payoff is **T1 + T2 (~5.5 wk) closes the Apple Silicon Mandelbrot
  + Sobel + matrix-multiply story** — the dev box, the first three
  UG demos, single-backend. **T1 + T2 + T6 (~7 wk) closes the
  "one-source, multi-target" story** because once Metal works the
  CUDA/OpenCL emit-pipes are mostly source-printer changes.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Everything below is 🔵 not started**. The substrate that exists:
  `matlab.parfor` outlining (`lib/MLIR/Passes/LowerParfor.cpp` — the
  template for `matlab.gpu.kernel`), the per-emit-pass family
  (`lib/MLIR/Passes/EmitC.cpp` / `EmitPython.cpp` / `EmitTypeScript.cpp`
  / `EmitSystemVerilog.cpp` — the template for `EmitMetal.cpp` /
  `EmitCUDA.cpp` / `EmitOpenCL.cpp`), the classdef + handle ABI
  (`matlab_obj_new` / `_set_*` / `_get_mat` — the template for
  `gpuArray`'s thin device-buffer-handle carrier), the
  function-handle ABI (`LowerAnonCalls` retype — needed for `arrayfun`
  on the GPU), `runtime/toolbox/<name>/` layout, and the cocotb-SIL
  pattern of "AOT-emit a device source + JIT-compile + verify against
  host reference" — that's *literally* what the GPU emit-lane needs at
  runtime, just with `MTLLibrary newLibraryWithSource:` /
  `nvrtcCompileProgram` / `clBuildProgram` instead of `verilator`.
- **A `coder.gpu.kernelfun` is a `parfor` with a device target**:
  this is the load-bearing simplification. The existing
  `lib/MLIR/Passes/LowerParfor.cpp` already outlines a `matlab.parfor`
  body into a private `func.func`, collects free variables as captures,
  and replaces the op with a call to `matlab_parfor_dispatch`. The
  GPU lane forks at the dispatch call: instead of `pthread`-based CPU
  workers, dispatch goes to the active GPU runtime
  (`matlab_gpu_launch_kernel_metal` / `_cuda` / `_opencl`). Same
  outlining, same capture protocol, same induction variable. This is
  why **Tier-1 is implementable on top of shipped infrastructure
  rather than a green-field MLIR design**.
- **`gpuArray(x)` is `matlab_obj_new` with a device pointer**: the
  shipped classdef carrier (used by `dsp.FIRFilter` /
  `ClassificationModel` / `idss` / `affine2d` / hundreds of others)
  is the *exact right shape* for a GPU array handle — a host-side
  carrier with opaque-pointer + size + dtype slots that the runtime
  fills with `MTLBuffer*` / `CUdeviceptr` / `cl_mem`. Existing DAP
  variable inspector and REPL persistence pick it up for free; the
  GPU-specific work is the printer (`gpuArray (1024×1024 double, Metal)`)
  and `gather()`/`mat2gpu()` conversion runtime entry points.
- **JIT-of-emitted-source over LLVM-IR-backend**: LLVM has NVPTX
  (NVIDIA) + AMDGPU + SPIR-V backends, but no upstream Metal backend
  (Apple's compiler is downstream + closed for AIR). The *simpler and
  more portable* route — and the one Apple's own `metal-cpp` tutorials
  use — is to **emit textual MSL/CUDA-C/OpenCL-C from the outlined
  kernel function and let the platform driver compile it at first
  launch**. This is exactly the project's existing pattern for cocotb
  (emit `.sv` source + cocotb's `Verilator` JIT-compiles it). The cost
  is sub-millisecond first-launch overhead per kernel, amortised by a
  hash-keyed runtime cache. The benefit is *no LLVM-target-bringup
  work* — emit passes are textual printers that read the same MLIR
  the existing C/Python emitters read.
- **No external dependency at host runtime beyond the platform
  driver**: Metal links `Metal.framework` + `MetalPerformanceShaders.framework`
  (already on every macOS box); CUDA links `libcuda` + `libnvrtc` +
  `libcublas` + `libcufft` + `libcusolver` (CUDA Toolkit, the same
  prerequisite GPU Coder has); OpenCL links `libOpenCL` + clBlast
  + clFFT (clBlast and clFFT are small, header-only-ish vendored
  builds). MathWorks-only deps (cuDNN / TensorRT / Coder UI / packNGo
  / Jetson Support Package / Simulink GPU model / Deep Learning
  blocks) are **carved out** to a later DL roadmap.

---

## 1. Reusable infrastructure (Tier-0 baseline — no GPU code yet)

| Group | Surface (already shipped) | Location | How GPU Coder uses it |
|---|---|---|---|
| Parallel loop outlining | `matlab.parfor` op + body outlining + captures + range op | `lib/MLIR/Passes/LowerParfor.cpp` | **The skeleton for `matlab.gpu.kernel`** — same outlining, same range, same capture protocol; only the dispatch target changes from `matlab_parfor_dispatch` (pthread) to `matlab_gpu_launch_kernel_*` (device driver). Tier-1.1/1.2 reuse this verbatim. |
| Per-target emit passes | `EmitC.cpp` / `EmitCpp` / `EmitPython.cpp` / `EmitTypeScript.cpp` / `EmitSystemVerilog.cpp` | `lib/MLIR/Passes/` | **Template for `EmitMetal.cpp` / `EmitCUDA.cpp` / `EmitOpenCL.cpp`** (Tier-2/3/4) — same Op walker + printer pattern, output is MSL / CUDA-C / OpenCL-C source text instead of C / Python / TS. |
| Classdef + handle ABI | `classdef`, handle semantics, `properties`/`methods`, `matlab_obj_new`/`_set_*`/`_get_mat`, class-pinned dispatch, REPL persist, DAP variable render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The carrier for `gpuArray` (Tier-1.4) — the GPU array is a handle classdef whose property is a device pointer + size + dtype; existing DAP inspector picks it up. |
| Function-handle ABI | `void *fn_p`, `LowerAnonCalls` retyping, capture lowering | `runtime/toolbox/optim/runtime_optim.cpp`, `lib/MLIR/Passes/LowerAnonCalls.cpp` | `arrayfun(@f, A)` on the GPU (Tier-5.3) — the anon body is inlined into a kernel, captures become kernel arguments. |
| Driver mode enum + flag parser | `Mode::EmitC`/`EmitCpp`/`EmitPython`/`EmitTypeScript`/`EmitSystemVerilog`/`EmitCocotb` + `tools/matlabc/main.cpp` flag plumbing | `tools/matlabc/main.cpp` | New `Mode::EmitCUDA`/`EmitOpenCL`/`EmitMetal` slots (Tier-6) — adding three switch arms in the existing dispatch is mechanical. |
| Cocotb-SIL JIT pattern | AOT-emit `.sv` + `.py` + JIT-compile via `verilator` + verify against CPU reference within tolerance | `docs/emit_cocotb.md`, `lib/MLIR/Passes/EmitSystemVerilog.cpp` | **Direct template for the GPU runtime's per-kernel JIT** — replace `verilator` with `MTLLibrary newLibraryWithSource:` / `nvrtcCompileProgram` / `clBuildProgram`, replace SV port matching with kernel-args descriptor. Tier-1.6 + Tier-7.1 (PIL). |
| Element-wise + reductions on host | element-wise math, `sum`/`prod`/`mean`/`min`/`max`/`norm` reductions, broadcasting | `runtime/matlab_runtime.cpp` | The host-side fallback path + the **per-tile CPU reference** that the PIL harness compares against (Tier-7.1). |
| Linear algebra (host) | `*`/`mtimes` / `mldivide` / `qr` / `svd` / `chol` / `eig` / `lu` | `runtime/matlab_runtime.cpp` | The host fallback when the GPU library replacement isn't enabled or the matrix is below the size threshold (matches GPU Coder's 128-element minimum for cuSOLVER, UG p. 2-19). |
| FFT (host) | `fft` / `ifft` / `fft2` | `runtime/matlab_runtime.cpp` | Host fallback / CPU reference for `dsp.FFT`-on-GPU (Tier-5.2 + Tier-7.1). |
| Plotting | Cairo `plot` / `imagesc` / `surf` | `runtime/plot/` | Headless `imagesc(mandelbrot_image)` — the Tier-2 demo's visual artifact, no GPU code, just shared. |
| DAP + REPL persistence | classdef objects survive across REPL inputs, expand in the variable inspector, format via `disp` | `runtime/runtime_debug.cpp`, `docs/repl.md` | A `gpuArray` shows in the variable pane with size + device + dtype + sample preview (Tier-1.7). |
| `coder.*` config carriers | `coder.gpuConfig` will join the existing carrier-objects family (like `fitoptions`, `nlpcOptions`, `optimoptions`) | `lib/Sema/Resolver.cpp` (Constructor-path classdef intercept) | `cfg = coder.gpuConfig('exe')` returns a thin config record read by `-emit-cuda` / `-emit-metal` / `-emit-opencl`. |
| Pragma front-end | `%#codegen` already parsed (silent no-op); the resolver already does `dsp.X` → `dsp_X` NameExpr fold (DSP Tier-1) | `lib/Parse/Parser.cpp`, `lib/Sema/Resolver.cpp` | The same fold turns `coder.gpu.kernelfun` / `coder.gpu.kernel` / `gpucoder.reduce` / `gpucoder.matrixMatrixKernel` / `stencilfun` into recognised builtin calls. |

**Net assessment**: the *outlining substrate* (parfor body extraction +
capture lowering + per-target emit-pass pattern + classdef carrier +
cocotb-JIT-of-emitted-source) is **already shipped**, used daily by
the existing parfor / cocotb / DSP lanes. The genuinely new code is:
**(a)** the **kernel IR ops** (`matlab.gpu.kernel` / `matlab.gpu.reduce`
/ `matlab.gpu.stencil` / `matlab.gpu.gemm` — modelled on `matlab.parfor`),
**(b)** the **three textual-source emit passes** (`EmitMetal.cpp` /
`EmitCUDA.cpp` / `EmitOpenCL.cpp`), **(c)** the **three thin host-runtime
ABIs** (`runtime/gpu/metal/` / `runtime/gpu/cuda/` / `runtime/gpu/opencl/`
— each ~600-1000 LOC, mostly buffer mgmt + library wrappers + JIT-compile
+ launch), **(d)** the **`gpuArray` classdef** + `gather` /
`existsOnGPU` / `arrayfun-on-GPU`, and **(e)** the **design-pattern
helpers** (`gpucoder.reduce` / `matrixMatrixKernel` / `stencilfun`).
None of these is novel infrastructure — each plugs into a shipped slot.

---

## 2. Tier-1 — Portable GPU kernel IR + outlining + host carrier 🔵 (KEYSTONE)

Goal: the **target-agnostic** GPU lifecycle — front-end recognises the
`coder.gpu.*` pragmas and `gpucoder.*` helpers, outliner extracts the
kernel body, MLIR represents it in a `matlab.gpu.*` op family, the host
runtime carries `gpuArray` handles and dispatches launches to whichever
backend is active. **No backend yet** — everything below the `matlab.gpu.*`
op is a stub that errors with "no GPU backend loaded" until Tier-2/3/4.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | **`matlab.gpu.kernel` op + body region** | Models an outlined kernel: induction range(s), captured-by-value scalar args, captured-by-reference array buffers (`gpuArray` handles), body region with the loop computation. Verifier rules: induction-only access into output slots, no host I/O, no recursion. Mirrors `matlab.parfor` 1:1 except for the device-target attribute. | `LowerParfor.cpp` |
| 1.2 | **Kernel outlining pass** | `LowerGpuKernels.cpp` — for each `matlab.gpu.kernel` op: clone body into a fresh private `func.func` named `__gpu_kernel_<id>`, lift free variables to function args, lift `coder.gpu.constantMemory`-tagged values to a `gpu.const_mem` attribute, replace the op with a call to `matlab_gpu_launch_kernel(target, src_id, name, grid, block, args)`. Body still in MLIR (lowered separately to source by Tier-2/3/4 emit passes). | `LowerParfor.cpp` outlining |
| 1.3 | **Pragma + helper front-end fold** | `coder.gpu.kernelfun` → mark enclosing func as one big kernel candidate (whole function-body becomes a `matlab.gpu.kernel`); `coder.gpu.kernel` → marks the next `for` loop nest; `coder.gpu.constantMemory(X)` → flags `X` capture as constant memory; `gpucoder.reduce(X,@f,...)` → `matlab.gpu.reduce` op; `gpucoder.matrixMatrixKernel(...)` → `matlab.gpu.gemm` op; `stencilfun(@f, A, sz)` → `matlab.gpu.stencil` op; `coder.ceval('cudaDeviceSynchronize')` → `matlab.gpu.sync`. | Resolver fold (DSP `dsp.X` precedent) |
| 1.4 | **`gpuArray` classdef + carrier** | Handle classdef with `Underlying` (host shape/dtype), `Device` (which backend), `DevicePtr` (opaque `void*`), `Stream` (per-device queue). `gpuArray(X)` → upload + return handle; `gather(g)` → device→host copy + return matrix; `existsOnGPU(g)` / `classUnderlying(g)` / `size`/`numel`/`isa` overloads. The classdef carrier reuses `matlab_obj_new`. | `matlab_obj_new`/`_set_*`/`_get_mat` |
| 1.5 | **Runtime-agnostic launch ABI** | Single C ABI in `runtime/gpu/runtime_gpu.cpp`: `matlab_gpu_init(target)` / `matlab_gpu_malloc(bytes,dtype)` / `matlab_gpu_h2d(dst, src, bytes)` / `matlab_gpu_d2h(dst, src, bytes)` / `matlab_gpu_d2d(...)` / `matlab_gpu_launch_kernel(src_id, name, grid[3], block[3], args, nargs)` / `matlab_gpu_sync()` / `matlab_gpu_free(ptr)`. Routes by active backend (`MATLAB_GPU_TARGET=metal\|cuda\|opencl\|auto`). | Cocotb runtime-dispatch pattern |
| 1.6 | **JIT-compile cache** | Hash kernel-source text + capture-arg signature, look up in `~/.cache/matlab_llvm/gpu/<target>/<hash>.{metallib\|cubin\|cl-bin}`, if miss → compile via the active driver (Tier-2/3/4 supplies the back end), store. Survives `matlabc` restarts. | Cocotb's emit+JIT pattern |
| 1.7 | **REPL + DAP inspector** | `disp(g)` formats `gpuArray (1024×1024 double, Metal, queue 0)`; DAP variables pane expands to show `Underlying`/`Device`/`DevicePtr (hex)` + a `Preview` row that gathers the first 8 elements on-demand. `coder.gpuConfig('exe')` returns a config record; `gpuDevice()` reports `name=Apple M2 / NVIDIA A100 / Intel Iris`. | `runtime_debug.cpp` DAP renderer |
| 1.8 | **CPU fallback / debug lane** | `--gpu-debug-on-host` (and matlabc default when no backend is available) emits the kernel body inline and runs it on host with a sequential for-loop — the same body that the kernel emits is executable as CPU code, so `dbstop in mandelbrot at 7` works the way every other breakpoint works in the project. PIL harness (Tier-7.1) is the formal verifier; this is the *interactive* fallback. | Existing `LowerParfor`'s single-threaded fallback |
| 1.9 | **`coder.gpuConfig` carrier** | Classdef record with `Target` (`mex`/`lib`/`exe`/`dll`), `GpuConfig.EnableCUBLAS`/`EnableCUSOLVER`/`EnableCUFFT`/`EnableMemoryManager`/`StackLimitPerThread`/`MallocMode`/`HalfType`/`ComputeCapability`/`EnableMPS`/`OpenCLPlatform`. Read by the AOT emit lanes (Tier-6). | Existing `optimoptions`/`fitoptions` carrier pattern |

**Headline-within-tier**: a working `coder.gpu.kernelfun` at the MLIR
level — the Mandelbrot kernel function outlined into a private
`func.func`, replaced with a `matlab_gpu_launch_kernel` stub call, and
**executable on the CPU-debug lane** (Tier-1.8) producing the correct
image. No backend yet — but the IR, the carrier, the REPL inspector,
and the host-side debug experience are *done*.

**Compile/Execute wiring**: new MLIR ops in `lib/MLIR/Dialect/` (modelled
on the existing `matlab.parfor`); new `LowerGpuKernels.cpp` pass run
right after `LowerParfor.cpp`; new `runtime/gpu/runtime_gpu.cpp` +
`runtime/toolbox/gpu/gpu_classdefs.m` (`gpuArray`, `coder.gpuConfig`,
`gpuDevice`); resolver fold for `coder.gpu.*` and `gpucoder.*` (DSP-style
NameExpr fold in `Parser.cpp::parsePostfix`); DAP renderer extension
keyed by classdef name; build flag `MATLAB_LLVM_GPU=ON` gates the GPU
lane (off by default in CI to keep the macOS-only / CUDA-only test
matrices smaller).

---

## 3. Tier-2 — Metal backend (macOS) 🔵

Goal: the **first working backend** — on Apple Silicon (the dev box), a
`coder.gpu.kernelfun`-tagged MATLAB program runs end-to-end through
Metal, with Metal Performance Shaders covering GEMM/FFT and MSL
covering element-wise + reductions + stencils.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | **`runtime/gpu/metal/` host driver** | `metal-cpp` headers (Apple's official C++ bridge — no Obj-C++ in our tree). Wraps `MTLDevice` / `MTLCommandQueue` / `MTLBuffer` / `MTLLibrary` / `MTLComputePipelineState` / `MTLComputeCommandEncoder`. Singleton device + queue, per-launch encoder. | — |
| 2.2 | **`EmitMetal.cpp` source emitter** | MLIR Op walker that prints MSL: `kernel void __gpu_kernel_N(device const real_t* X [[buffer(0)]], device real_t* Y [[buffer(1)]], constant uint& N [[buffer(2)]], uint tid [[thread_position_in_grid]]) { ... }`. Mirrors `EmitC.cpp` structure — Op-by-Op printer with type mapping (`f64` → `double` on macOS 13+, fp32 fallback / fp16 via `half`). | `EmitC.cpp` |
| 2.3 | **JIT compile via `MTLLibrary newLibraryWithSource:`** | First launch: emit MSL → `MTLDevice newLibraryWithSource:` (returns a `MTLLibrary`) → `newFunctionWithName:` → `newComputePipelineStateWithFunction:`. Cache the pipeline state by source-hash. On subsequent launches, look up cached pipeline → encode → commit → `waitUntilCompleted` (synchronous semantics matching the existing CPU host lane). | Tier-1.6 cache |
| 2.4 | **Unified Memory fast path (Apple Silicon)** | Detect Apple Silicon at init; allocate `MTLBuffer` with `MTLResourceStorageModeShared` (CPU + GPU share the same physical page). `matlab_gpu_h2d` / `d2h` become pointer assignments — zero copy. On Intel Macs (discrete GPU), fall back to `MTLResourceStorageModeManaged` + `didModifyRange:` / `synchronize`. | — |
| 2.5 | **MPS — Metal Performance Shaders for BLAS** | When `cfg.GpuConfig.EnableMPS = true` (default on Metal): `mtimes(a,b)` of GPU arrays → `MPSMatrixMultiplication` (the cuBLAS-`gemm` analogue); `mldivide` → `MPSMatrixDecompositionLU` + `MPSMatrixSolveLU` (the cuSOLVER analogue); `inv` → `MPSMatrixInverse`; per-element vec ops on small sizes stay in our MSL kernels (MPS is overkill below ~128 elements, mirroring GPU Coder's documented cuSOLVER threshold). | — |
| 2.6 | **FFT — MPSGraph FFT** | `fft(gpuArray(x))` → `MPSGraph` with an `FFT` node, single-shot execution. 2-D and higher follow GPU Coder's batched-1-D-FFT pattern (UG p. 2-22). FFTW-style real-/complex-input variants. | host `fft` reference |
| 2.7 | **Half-precision (`half`)** | MATLAB `half` type → MSL `half` (IEEE 754 binary16, native on Apple GPU). Promote/demote at `gpuArray(half(X))` boundary. Required for the Tier-7.2 fp16 sweep. | `Underlying` carrier slot |
| 2.8 | **Stream + sync** | One default `MTLCommandQueue`; explicit streams via `coder.gpu.stream()` map to multiple queues. `wait(g)` / `coder.ceval('cudaDeviceSynchronize')` → `[encoder endEncoding]; [cmdbuf commit]; [cmdbuf waitUntilCompleted];` on the active queue. | Tier-1.5 ABI |
| 2.9 | **Profiling capture** | `MTLCaptureManager` shim — `gpuPerformanceAnalyzer` writes a `.gputrace` bundle openable in Xcode's Metal Frame Capture. Skipped on CI; user-only ergonomic. | — |
| 2.10 | **Error surface** | MSL compile errors → propagate `MTLLibrary newLibraryWithSource:`'s `NSError` into our `matlabc` error stream, mapped back to source line via the `#line N "user.m"` directive emitted by Tier-2.2. | `EmitC.cpp` `#line` precedent |

**Headline-within-tier**: `examples/gpu/mandelbrot_gpu.m` running on
Apple Silicon end-to-end — `coder.gpu.kernelfun` decorated, compiled by
matlabc, MSL emitted, JIT-compiled by Metal at first launch, dispatched
to the M-series GPU via Unified Memory (zero copy), result imaged via
the headless Cairo plot lane to `mandelbrot.png`. **First working GPU
demo on any backend.**

**Compile/Execute wiring**: `runtime/gpu/metal/` (~700 LOC C++ — device
init, buffer mgmt, MSL JIT, MPS GEMM/FFT, stream); `lib/MLIR/Passes/EmitMetal.cpp`
(~600 LOC printer, structured like `EmitC.cpp`); CMake gate
`MATLAB_LLVM_GPU_METAL=ON` (auto-on for Apple targets, off elsewhere);
new CTest lane `gpu-metal-gate` runs the Tier-2 demos on macOS CI only.

---

## 4. Tier-3 — CUDA backend (Linux) 🔵

Goal: feature-parity with **MathWorks GPU Coder's CUDA lane** — the
same source code that ran on Metal in Tier-2 now runs on NVIDIA Linux
through NVRTC + cuBLAS + cuSOLVER + cuFFT.

> **Status (2026-05-31, issue #25): first on-hardware validation.**
> Validated on an RTX 5060 (sm_120) via `test/Run/run_gpu_cuda_validation.sh`:
> - **3.1 host driver** (`runtime/gpu/cuda/runtime_gpu_cuda.cpp`) — driver-API
>   `CUcontext`/`CUdeviceptr` lifecycle + device probe. ✅
> - **3.3 NVRTC JIT** — `nvrtcCompileProgram` → PTX → `cuModuleLoadData` →
>   `cuLaunchKernel`, AXPY exact. ✅
> - **3.4 H2D/D2H** — `cuMemcpyHtoD`/`DtoH`. ✅
> - **3.5 cuBLAS GEMM** — `cublasDgemm` (fp64) wired into the
>   `matlab_gpu_gemm` dispatcher; matches the host lane to 0 ULP. ✅
> - **`-emit-cuda` bundle** — the emitter now translates the scalar AXPY
>   kernel body fully (was an identity FALLBACK) and emits an **nvcc-free**
>   NVRTC host driver that JIT-compiles + launches the real kernel; builds
>   and runs end-to-end. ✅
>
> Build is opt-in (`-DMATLAB_LLVM_GPU_CUDA=ON`, default OFF); CMake prefers
> a system CUDA toolkit and falls back to the pip-wheel CUDA libs. The TU
> uses only the driver API + NVRTC + hand-declared cuBLAS prototypes, so no
> full nvcc toolkit is required. Remaining items (3.2 full kernel-body
> codegen, 3.6 cuSOLVER, 3.7 cuFFT, 3.8 `__half`, 3.9 streams) are future
> work; `matlab_gpu_launch_cuda` currently uses the host-fallback loop
> (parity with the Metal backend — AOT-to-JIT kernel-source linkage is not
> wired for any backend yet).

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | **`runtime/gpu/cuda/` host driver** | Links `libcuda` (driver API) + `libnvrtc` + `libcudart`. Wraps `CUdevice` / `CUcontext` / `CUstream` / `CUdeviceptr` / `CUmodule` / `CUfunction`. Init resolves the first capable device; explicit selection via `gpuDevice(N)`. | — |
| 3.2 | **`EmitCUDA.cpp` source emitter** | MLIR Op walker that prints `.cu`: `__global__ __launch_bounds__(512, 1) void __gpu_kernel_N(const double* x, double* y, int N) { int tid = ...; if (tid < N) y[tid] = f(x[tid]); }`. Matches the UG p. 2-3 generated code exactly. Type mapping uses `double` / `float` / `__half` / `int32_t`. | `EmitMetal.cpp` (90% shared structure) |
| 3.3 | **JIT via NVRTC** | First launch: emit `.cu` → `nvrtcCreateProgram` → `nvrtcCompileProgram` → `nvrtcGetPTX` → `cuModuleLoadData` → `cuModuleGetFunction` → cached `CUfunction`. Subsequent launches: `cuLaunchKernel(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedBytes, stream, args, extra)`. Compile cache keyed by source-hash + compute-capability. | Tier-1.6 cache |
| 3.4 | **`cudaMemcpy` H2D / D2H / D2D** | `matlab_gpu_h2d` → `cuMemcpyHtoD`; `_d2h` → `cuMemcpyDtoH`; `_d2d` → `cuMemcpyDtoD`. All synchronous on the default stream by default; async variants for explicit streams. | Tier-1.5 ABI |
| 3.5 | **cuBLAS library replacement (Tier-3a)** | `cfg.GpuConfig.EnableCUBLAS = true` (default): when both operands of `*` are `gpuArray` and shape is matrix-matrix above the 128-element threshold (matches UG p. 2-12), lower to `cublasDgemm` / `cublasSgemm` / `cublasHgemm` via a `matlab.gpu.gemm` op that the CUDA backend lowers to a library call instead of an emitted kernel. | host `mtimes` |
| 3.6 | **cuSOLVER library replacement (Tier-3b)** | `cfg.GpuConfig.EnableCUSOLVER = true`: `mldivide` / `qr` / `lu` / `chol` / `svd` on `gpuArray` → `cusolverDnXgesv` / `Xgeqrf` / `Xgetrf` / `Xpotrf` / `Xgesvd`. Threshold matches MathWorks (cusolver only above 128). | host LAPACK |
| 3.7 | **cuFFT library replacement (Tier-3c)** | `cfg.GpuConfig.EnableCUFFT = true`: `fft` / `ifft` / `fft2` on `gpuArray` → `cufftPlan1d` / `cufftPlan2d` / `cufftPlanMany` + `cufftExecZ2Z` / `cufftExecD2Z`. 2-D and higher dispatched as batched 1-D (matches UG p. 2-23). | host `fft` |
| 3.8 | **`__half` half-precision** | `gpuArray(half(X))` → `__half`-typed buffer; emitted kernels use `__half` arithmetic + `__hfma`/`__hadd`/etc. cuBLAS GEMM uses `cublasHgemm`. Required by the Tier-7.2 fp16 demo and the UG's Sobel half-precision example. | half storage |
| 3.9 | **Streams + concurrency** | Multiple `CUstream` mapped to `coder.gpu.stream()` handles; `cuStreamSynchronize` for explicit sync. Default behaviour matches Tier-2 (synchronous). | — |
| 3.10 | **Build flag + CI** ✅ | CMake `MATLAB_LLVM_GPU_CUDA=ON` (default OFF); discovery prefers `find_package(CUDAToolkit)`, falls back to pip-wheel CUDA libs. Validation lane `test/Run/run_gpu_cuda_validation.sh` runs only when an NVIDIA GPU is present (HW-gated, SKIPs cleanly otherwise) — ready to wire behind a self-hosted-runner label. | — |

**Headline-within-tier**: the same `examples/gpu/mandelbrot_gpu.m`
source running on Linux + NVIDIA, plus
`examples/gpu/cublas_gemm_demo.m` (UG p. 2-17 1024×1024 `A*B` falling
into the cuBLAS lane, validated against host BLAS within 1e-10).

**Compile/Execute wiring**: `runtime/gpu/cuda/` (~900 LOC — driver init,
NVRTC JIT, cuBLAS/cuSOLVER/cuFFT shims); `lib/MLIR/Passes/EmitCUDA.cpp`
(~600 LOC printer, forked from `EmitMetal.cpp` then dialect-substituted);
`matlab.gpu.gemm` / `matlab.gpu.fft` / `matlab.gpu.solve` ops in
`lib/MLIR/Dialect/` (so the same op can be lowered to MPS on Metal or
cuBLAS on CUDA or clBlast on OpenCL — the *backend* picks the
implementation, not the *front end*).

---

## 5. Tier-4 — OpenCL backend (Linux + portable) 🔵

Goal: the **portable cross-vendor backend** — runs on NVIDIA / AMD /
Intel / ARM Mali through OpenCL, no vendor lock-in. Matches the GPU
Coder UG's "Code Generation for Deep Learning Networks Targeting ARM
Mali GPUs" lane in spirit but generalised to any OpenCL device.

> **Status (2026-05-31, issue #25): first on-hardware validation.**
> Validated on an RTX 5060 via its NVIDIA OpenCL ICD (vendor-agnostic —
> the same lane runs on AMD/Intel ICDs) using
> `test/Run/run_gpu_opencl_validation.sh`:
> - **4.1 host driver** (`runtime/gpu/opencl/runtime_gpu_opencl.cpp`) —
>   platform/device/context/queue lifecycle + device probe. ✅
> - **fp64 GEMM** — `matlab_gpu_opencl_gemm_double` JIT-builds a naive
>   fp64 kernel and is wired into the `matlab_gpu_gemm` dispatcher;
>   matches the host lane to 0 ULP. ✅ (clBLAST is a future swap-in,
>   mirroring cuBLAS on the CUDA side.)
> - **`-emit-opencl` bundle** — the emitter now translates the scalar
>   AXPY kernel body fully (was an identity FALLBACK) and emits an
>   **SDK-free** host driver (hand-declares the OpenCL API when no
>   `<CL/cl.h>` is installed) that builds + runs end-to-end. ✅
>
> Build is opt-in (`-DMATLAB_LLVM_GPU_OPENCL=ON`, default OFF); the TU
> hand-declares the OpenCL 1.2 API so it links against just the ICD
> loader (`libOpenCL`) — no SDK headers. `matlab_gpu_launch_opencl` uses
> the host-fallback loop (parity with the CUDA/Metal backends). The
> emit-pass fix is shared with CUDA via the `GpuKernelInfo` descriptor.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | **`runtime/gpu/opencl/` host driver** | Links `libOpenCL`. Wraps `cl_platform_id` / `cl_device_id` / `cl_context` / `cl_command_queue` / `cl_mem` / `cl_program` / `cl_kernel`. Platform selection via `MATLAB_GPU_OPENCL_PLATFORM` env (NVIDIA / AMD / Intel / POCL / Mali). | — |
| 4.2 | **`EmitOpenCL.cpp` source emitter** | MLIR Op walker that prints OpenCL C 1.2 / 2.0: `__kernel void __gpu_kernel_N(__global const double* x, __global double* y, int N) { int tid = get_global_id(0); if (tid < N) y[tid] = ...; }`. 90% structurally identical to `EmitCUDA.cpp`/`EmitMetal.cpp`; differs in `__kernel`/`__global` qualifiers + `get_global_id(0)` vs `threadIdx`. | `EmitCUDA.cpp` |
| 4.3 | **JIT via `clBuildProgram`** | First launch: emit `.cl` → `clCreateProgramWithSource` → `clBuildProgram` → `clCreateKernel`. Cache the `cl_program` binary via `clGetProgramInfo(... CL_PROGRAM_BINARIES ...)` keyed by source-hash + device. Subsequent launches: `clEnqueueNDRangeKernel`. | Tier-1.6 cache |
| 4.4 | **H2D / D2H** | `clEnqueueWriteBuffer` / `ReadBuffer`. SVM (shared virtual memory) on capable devices for the unified-memory fast path; pinned host memory (`CL_MEM_ALLOC_HOST_PTR`) on others. | Tier-1.5 ABI |
| 4.5 | **clBlast for BLAS** | `mtimes` on GPU arrays → `CLBlastDgemm` (the cuBLAS-`gemm` analogue, vendor-portable). clBlast is small (~10 KLOC C++), vendored into `external/clBlast/`. | Tier-3.5 op |
| 4.6 | **clFFT for FFT** | `fft` on GPU arrays → `clfftSetPlanDim` + `clfftEnqueueTransform`. Same dispatch fork as Tier-3.7 — the `matlab.gpu.fft` op dispatches by backend. | Tier-3.7 op |
| 4.7 | **fp16 via `cl_khr_fp16`** | Conditional emit — query device caps, emit `#pragma OPENCL EXTENSION cl_khr_fp16 : enable` when supported (Intel + Mali + most modern AMD). Fall back to fp32 on devices without `cl_khr_fp16` (older NVIDIA OpenCL stack). | — |
| 4.8 | **CPU-OpenCL via POCL** | Document the POCL (Portable Computing Language) installation as the CI smoke-test path — runs OpenCL on the CPU, so the `gpu-opencl-gate` CI lane passes on runners without a GPU. (Matches the spirit of `verilator`-on-Linux for cocotb gating.) | — |
| 4.9 | **Build flag + CI** | CMake `MATLAB_LLVM_GPU_OPENCL=ON`; `find_package(OpenCL REQUIRED)`. CI lane `gpu-opencl-gate` uses POCL on the CPU so it always runs; a separate hardware-gated lane runs on a real GPU runner. | — |

**Headline-within-tier**: `examples/gpu/mandelbrot_gpu.m` running
**unchanged** through `--gpu-target=opencl` on the same Linux box, with
the POCL CPU OpenCL implementation, validated bit-identical (to the
host CPU reference within 1e-10). The "*same MATLAB file → three
backends*" story now closes for at least one platform.

**Compile/Execute wiring**: `runtime/gpu/opencl/` (~800 LOC — platform
init, JIT, clBlast/clFFT shims); `lib/MLIR/Passes/EmitOpenCL.cpp`
(~550 LOC printer); `external/clBlast/` + `external/clFFT/` vendored;
POCL listed as the CI install recipe in the Dockerfile.

---

## 6. Tier-5 — GPU Coder design patterns (helpers across all 3 backends) 🔵

Goal: ship the **MathWorks-specific helper surface** — `gpucoder.reduce`,
`gpucoder.matrixMatrixKernel`, `stencilfun`, `coder.ceval`,
`coder.gpu.constantMemory` — across **all three backends**. Each helper
gets a per-backend lowering arm so a single MATLAB program written
against the GPU Coder API runs on Metal, CUDA, and OpenCL with no
source change.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | **`gpucoder.reduce(X,@f,...,'preprocess',@g,'dim',k)`** | Tree-reduction (UG p. 2-5 binary tree). For each backend: shared-memory + thread-block reduction kernel (CUDA + OpenCL workgroup-local memory + Metal `threadgroup` memory), `@g` inlined as the preprocess step, `@f` inlined as the combine step. `dim=1` reduces rows (sum-by-column), `dim=2` reduces cols, no `dim` collapses to scalar. Tested against host `sum`/`prod`/`max`/`min` reference. | function-handle ABI; existing `LowerAnonCalls` |
| 5.2 | **`gpucoder.matrixMatrixKernel(C, op, A, B)`** | Custom GEMM-shaped op with a user-provided op (`@(a,b) a*b`/`@max`/`@plus`) instead of `+`-of-products. Each backend: tiled kernel with shared-memory loads (the canonical GPU Coder matmul template, UG p. 2-39). Cross-backend dispatch via the same `matlab.gpu.gemm` op as Tier-3.5/4.5, with a `customOp` attribute that routes to the per-backend tiled kernel emitter when present (otherwise to the library call). | function-handle ABI |
| 5.3 | **`stencilfun(@f, A, [m n])`** | Kernel where each output element depends on an `m×n` window of inputs (the GPU Coder UG p. 2-37 canonical stencil — Sobel, mean, Gaussian, etc.). Per backend: emit a kernel that loads an `(blockX+m-1)×(blockY+n-1)` halo'd tile into shared/threadgroup memory and applies `@f`. Replaces deprecated `gpucoder.stencilKernel`. | `LowerAnonCalls` |
| 5.4 | **`coder.gpu.kernel`** | Loop-pragma: marks the next `for` (possibly nested) for kernelisation with explicit `[blockX,blockY,blockZ]` / `[gridX,gridY,gridZ]` sizing. Bypasses parallel-loop dependence analysis (user assertion). | Tier-1.2 outliner |
| 5.5 | **`coder.gpu.kernelfun`** | Function-level pragma: makes the whole function body one big kernel-candidate region; the outliner picks per-loop kernels and host glue. The default in 80% of GPU Coder UG examples. | Tier-1.2 outliner |
| 5.6 | **`coder.gpu.constantMemory(X)`** | Lifts capture `X` to per-kernel `__constant__` (CUDA) / `__constant` (OpenCL) / `constant device` (Metal) memory. Helps when `X` is a small read-only filter table accessed by every thread (Sobel kernels, FIR coefficients). | — |
| 5.7 | **`coder.ceval('cudaDeviceSynchronize')`** | Maps to `matlab.gpu.sync` op → `[cmdbuf waitUntilCompleted]` (Metal) / `cuStreamSynchronize` (CUDA) / `clFinish` (OpenCL). Other `coder.ceval` strings dispatched per-backend (e.g. `cudaMalloc` rewrites to the backend-native allocator). | Tier-1.5 ABI |
| 5.8 | **`gpucoder.sort` / `gpucoder.batchedMatMul`** | Library-backed: CUB `radixSort` on CUDA, Bitonic-sort kernel on Metal + OpenCL; cuBLAS `gemmBatched` / MPSMatrixMultiplicationBatch / clBlast batched GEMM. Same `matlab.gpu.*` op dispatch fork as 5.2. | — |
| 5.9 | **`gpuArray`-aware operator overloads** | `+`, `-`, `.*`, `./`, `.^`, `>`, `<`, `==`, `&`, `|`, `~`, unary `-` between two `gpuArray`s (or `gpuArray` × scalar) dispatch to a single-kernel element-wise body without the user touching any pragma. This is the *80% case* from the UG. | Tier-1.4 carrier + Tier-2/3/4 emitters |
| 5.10 | **`arrayfun(@f, gpuArray)` / `bsxfun`** | Element-wise apply where `@f` is an anon. The anon body is inlined into an emitted kernel; captures become arguments. Matches the existing `LowerAnonCalls` retype lane. | function-handle ABI |

**Headline-within-tier**: `examples/gpu/streaming_fft.m` —
`gpucoder.reduce` for the windowed mean + `gpucoder.matrixMatrixKernel`
for the per-bin amplitude × phase combine + Tier-2/3/4-backed `fft`
over a frame loop, all running on Metal AND CUDA AND OpenCL with `≤1e-9`
maxdiff vs the host reference. Closes the "design-patterns" claim of
the UG against three live backends.

**Compile/Execute wiring**: each helper is one `matlab.gpu.*` op +
per-backend emitter arm in `EmitMetal.cpp` / `EmitCUDA.cpp` /
`EmitOpenCL.cpp` (Tier-2/3/4). New shared file
`lib/MLIR/Passes/LowerGpuPatterns.cpp` does the host-side recognition
+ op construction; the lowerings are textual.

---

## 7. Tier-6 — AOT emit lanes (`-emit-cuda` / `-emit-opencl` / `-emit-metal`) 🔵

Goal: the **standalone-source emit lane** — `matlabc -emit-cuda foo.m`
writes a self-contained `foo.cu` (kernel) + `foo.cpp` (host driver) +
`Makefile` that builds without `matlabc` at runtime. Same shape for
`-emit-metal` (writes `.metal` + Obj-C++ / Swift driver) and
`-emit-opencl` (writes `.cl` + C++ driver). This is the GPU equivalent
of the shipped `-emit-cocotb` lane (which emits SV + Python and JIT-runs
through cocotb's `verilator`).

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | **`Mode::EmitCUDA` / `EmitOpenCL` / `EmitMetal`** | Three new enum slots in `Options::Mode` (`tools/matlabc/main.cpp:172`), three new `-emit-cuda` / `-emit-opencl` / `-emit-metal` flag arms. | `Mode::EmitC`/etc precedent |
| 6.2 | **Standalone `.cu` + driver** | `matlabc -emit-cuda foo.m` writes `foo_kernel.cu` (the device code emitted by `EmitCUDA.cpp` — Tier-3.2) + `foo_main.cpp` (host driver that calls `cudaMalloc` / `cudaMemcpy` / `<<<grid,block>>>` + result print) + `Makefile` that links against the local CUDA Toolkit. Matches the GPU Coder UG "Generate CUDA Executable" example (p. 5-4). | Tier-3.2 + new HostDriverPrinter |
| 6.3 | **Standalone `.metal` + driver** | `matlabc -emit-metal foo.m` writes `foo_kernel.metal` + `foo_main.mm` (Obj-C++ driver using `metal-cpp`, since pure C++ macOS apps can host Metal) + `Makefile` targeting `xcrun -sdk macosx metal` + `clang++` linking `Metal.framework` + `MetalPerformanceShaders.framework`. | Tier-2.2 + new HostDriverPrinter |
| 6.4 | **Standalone `.cl` + driver** | `matlabc -emit-opencl foo.m` writes `foo_kernel.cl` + `foo_main.cpp` (uses `clCreateProgramWithSource` / `clBuildProgram` at runtime — i.e. the standalone executable JIT-compiles its own `.cl` on first run, like a single-purpose copy of our Tier-4 runtime) + `Makefile` linking `libOpenCL`. | Tier-4.2 + new HostDriverPrinter |
| 6.5 | **`packNGo`-style bundle** | `matlabc -emit-cuda foo.m --pack` produces `foo_codegen.zip` with sources + `Makefile` + a `README.md` listing toolchain prerequisites. The MathWorks `packNGo` analogue (UG p. 5-13), per backend. | — |
| 6.6 | **Numerical reference companion** | Each `-emit-*` lane also writes `foo_ref.cpp` — the **CPU-only reference** built from the same MLIR (just lowered through `EmitC` instead of `EmitCUDA`) — and the `Makefile` builds a `run-equiv` target that runs both and asserts max-abs-diff `< 1e-9`. This is the "PIL" feature (UG p. 3-39) usable from outside `matlabc`. | Tier-7.1; existing `EmitC.cpp` |
| 6.7 | **Whole-file vs per-function** | `matlabc -emit-cuda foo.m` emits *all* `coder.gpu.kernelfun`-tagged functions in `foo.m` + one `main()` that drives the entry-point function with example inputs (sized from `coder.typeof` annotations, like the existing AOT lanes). | Existing `-emit-c` AOT |

**Headline-within-tier**: `examples/gpu/sobel_emit.m` — one source file
emits a CUDA bundle (Linux), a Metal bundle (macOS), and an OpenCL
bundle (cross-platform) via three `matlabc` invocations; each bundle
builds + runs standalone and produces a Sobel-edge-detected image
matching the host CPU reference. The **multi-backend AOT
"one-source-three-targets" claim** is now demonstrable.

**Compile/Execute wiring**: extends `tools/matlabc/main.cpp:172` with
three enum slots + flag parsing; adds `HostDriverPrinter` (one ~400 LOC
file shared across backends, parameterised on `dialect`); adds a tiny
`Makefile` template printer per backend.

---

## 8. Tier-7 — Numerical equivalence + half-precision + memory manager 🔵

Goal: the **polish & verification tier** — PIL-style host↔device
equivalence harness, fp16 codegen, the GPU memory manager (the
documented optimisation that pools `cudaMalloc` to avoid per-launch
allocator overhead — UG p. 2-42).

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 7.1 | **PIL gate (host↔device equivalence)** | New CTest lane `gpu-equiv-gate`: for each `examples/gpu/*.m`, run both the CPU-lane (`matlabc -emit-c → cc → run`) and the GPU lane (`matlabc -gpu-target=$T → run`) on the same inputs, assert `max(abs(cpu - gpu)) < tol` with `tol=1e-9` for fp64, `1e-4` for fp32, `1e-2` for fp16. Mirrors the cocotb SIL gate's *exactly*. Runs only when the active backend has a working device (`gpu-equiv-gate-metal` always on macOS; `gpu-equiv-gate-cuda` only when CUDA + GPU are present; `gpu-equiv-gate-opencl` always via POCL). | Cocotb SIL gate template |
| 7.2 | **fp16 codegen sweep** | `gpuArray(half(X))` carrier flows through MSL `half` (Tier-2.7), CUDA `__half` (Tier-3.8), and OpenCL `half` (Tier-4.7). Headline demo: the UG p. 2-121 Sobel-in-half-precision example, validated against the fp64 reference within fp16 tolerance. | Tier-2.7 + 3.8 + 4.7 |
| 7.3 | **GPU memory manager** | `cfg.GpuConfig.EnableMemoryManager = true` (default): per-thread reusable buffer pool keyed by `(size, dtype, target)`. Allocations satisfied from the pool when a matching-or-larger buffer exists; otherwise allocated and returned to the pool on free. Wins the same fog-rectification benchmark that the UG cites (p. 2-43: 39 `cudaMalloc`s → 1). | — |
| 7.4 | **Performance counters + report** | `gpuPerformanceAnalyzer(@f, args)` profiles wall-clock + per-kernel device time (`MTLCommandBuffer GPUStartTime`/`GPUEndTime` / `cudaEvent` / `clGetEventProfilingInfo`). Generates a JSON timeline + a text summary in the `matlabc -gpu-report` mode. The lighter analogue of MathWorks GPU Performance Analyzer (UG p. 6-2). | DAP profiler hook |
| 7.5 | **Kernel-analysis warnings** | Static checks during outlining (Tier-1.2): loop-carried dependency (UG p. 6-27) → warn + don't kernelise; `break` inside the loop (UG p. 6-23) → warn; unsupported function call inside kernel body → fall back to host with a diagnostic naming the call. | Tier-1.2 outliner |
| 7.6 | **Traceability tags** | Emit `// matlab_source: foo.m:42` (CUDA / OpenCL) / `// 42 "foo.m"` (Metal MSL — Metal accepts `#line`-style comments in source listings) above each kernel body chunk, so source-level navigation works in Xcode / Nsight (UG p. 6-8). | Existing `#line` directive emit in `EmitC.cpp` |

**Headline-within-tier**: the **`gpu-equiv-gate-metal` CI lane green
on macOS for all `examples/gpu/*.m`** within the documented per-dtype
tolerance, plus the Sobel-fp16 example matching the fp64 reference
within fp16 tolerance. Closes the equivalence-testing claim of UG ch. 3.

**Compile/Execute wiring**: new `test/Run/gpu_equiv/*.m` test family;
`CMakeLists.txt` adds the three `gpu-equiv-gate-*` CTest lanes;
`runtime/gpu/mempool.cpp` implements the pool used by all three runtimes.

---

## 9. Carve-outs (explicit non-goals — separate roadmap follow-ups)

The following GPU Coder UG chapters are **out of scope** for this
roadmap; each gets a dedicated follow-up doc when its dependencies land.

| Carve-out | Why deferred | Future roadmap home |
|---|---|---|
| **Deep Learning (cuDNN / TensorRT / Mali ACL) — UG ch. 4** | The project has no `dlnetwork` / CNN / LSTM substrate yet (Stats/ML stops at ensembles / trees / SVM / k-means / fitcecoc, no CNN layers). cuDNN + TensorRT replacement requires the network IR first. | Future `deep_learning_toolbox_roadmap.md` — will fork the Stats/ML lane with a CNN layer family + `dlnetwork` carrier, *then* GPU Coder Tier-8 wires cuDNN / MPSCNN / OpenVINO on top. |
| **Simulink GPU acceleration + Simulink GPU code-gen — UG ch. 3** | The project's flowchart / mflow lane uses its own block library; Simulink-block-level GPU is the mflowLink Embedded Coder × GPU lane, not GPU Coder × Simulink. | Future `mflowlink_gpu_extension.md` follow-on after Tier-2 lands — wire `mflowLink` blocks to dispatch through the Tier-1.5 ABI. |
| **Jetson / DRIVE / NVIDIA embedded boards — UG ch. 5 (5-22 onwards)** | Hardware-specific deployment + the MATLAB Coder Jetson Support Package needs board-side runtime + serial / UDP / MAVLink / MODBUS support. Out of scope for a *codegen* roadmap. | Future `embedded_gpu_deployment_roadmap.md` once Tier-3 (CUDA) lands; the Tier-6 AOT lane's standalone `.cu` + `Makefile` is *most of the way there* already. |
| **GPU Coder App (GUI) — UG ch. 2 + 5** | The project is a CLI + LSP. The GUI is a MathWorks IDE feature, not a codegen capability. | Out of scope permanently; the **`coder.gpuConfig` carrier (Tier-1.9) is the headless equivalent** and is sufficient for every CLI workflow. |
| **External Mode parameter tuning — UG ch. 3 (3-47)** | Requires a Simulink-style host↔target live link. Project doesn't have this for any backend. | Out of scope. |
| **packNGo over network — UG ch. 5 (5-20)** | The Tier-6 AOT bundle is the offline analogue; a network-relocation flow is a deployment niche. | Out of scope. |
| **`coder.gpu.constantMemory` for non-scalar non-trivial types** | First cut handles scalar arrays of constant size; struct constant memory needs the host-driven const-memory upload protocol. | Tier-5 follow-up. |
| **GPU `dlarray` / `dlnetwork` half/INT8 — UG ch. 4 (4-77, 4-84)** | Same DL gap as the cuDNN carve-out. | Future DL roadmap. |
| **`nvlink` register-count tuning — UG ch. 7** | Register-pressure tuning is a CUDA-deployment niche; first emit pass uses `__launch_bounds__` matching the UG example and trusts the compiler. | Tier-3 follow-up if it becomes a real issue. |
| **Code Generation Reports — UG ch. 6 (6-2 to 6-14)** | The matlabc trace lane already covers the source-map need; the deeper "Code Insights" report is a GUI feature. | Out of scope; basic kernel summary in `-gpu-report` (Tier-7.4) covers 80%. |
| **AMD ROCm / HIP** | OpenCL + clBlast cover AMD; HIP would be an additional Linux backend. | Future `hip_backend.md` if user demand materialises — would be a fork of the Tier-3 CUDA backend (HIP source is `cuda.h`-compatible at the API level). |
| **Vulkan compute / SPIR-V backend** | OpenCL covers the cross-vendor use case; Vulkan compute is a third orthogonal path. | Future possibility; SPIR-V LLVM target exists upstream, would be the IR-not-textual path. |

---

## 10. Effort + sequencing

```
Tier  Title                                          Weeks   Status   Gate
────  ─────────────────────────────────────────────  ─────   ──────   ────────────────────────────────
T1    GPU kernel IR + outlining + carrier            ~3      🔵       Mandelbrot CPU-debug lane runs
T2    Metal backend (macOS, dev box)                 ~2.5    🔵       Mandelbrot on Metal, MPS GEMM
T3    CUDA backend (Linux)                           ~2      🔵       Mandelbrot on CUDA, cuBLAS GEMM
T4    OpenCL backend (Linux + POCL)                  ~2      🔵       Mandelbrot on OpenCL (POCL CPU)
T5    Design patterns (reduce / matmul / stencil)    ~2      🔵       streaming_fft demo on 3 backends
T6    AOT emit (-emit-{cuda,metal,opencl})           ~1.5    🔵       sobel_emit produces 3 bundles
T7    PIL + fp16 + memory manager                    ~1.5    🔵       gpu-equiv-gate green
                                                     ─────
                                                     ~14.5
```

**Critical path**: T1 → T2 (~5.5 wk) is the fastest path to a working
demo on Apple Silicon; T1 → T3 (~5 wk) is the same on Linux+NVIDIA;
T1 → T4 (~5 wk) is the same on Linux+OpenCL/POCL. **Any of the three
backend tiers can ship independently of the other two once T1 lands.**

**Parallelisable**: T2 / T3 / T4 are largely independent (different
emit-passes, different runtimes); T5 picks them up together; T6 / T7
follow.

**Recommended order**: T1 → T2 (closes the dev-box experience first) →
T7.1 (PIL gate — gives us numerical-equivalence confidence) → T3 (CUDA)
→ T5 (cross-backend design patterns) → T4 (OpenCL) → T6 (AOT) → T7.2-5
(fp16 + mempool + reports + tracing polish).

---

## 11. Headline tracer-bullets

The headlines below are the **acceptance bar** — when the named example
runs end-to-end producing the right output, the named tier is shipped.

| Demo | Tier | What it proves |
|---|---|---|
| `examples/gpu/mandelbrot_gpu.m` (Metal) | T1 + T2 | Tier-1 kernel IR + outlining + carrier works; Metal backend launches a real device kernel and produces correct output with the Unified Memory fast path; first working GPU demo on **any** backend. |
| `examples/gpu/mandelbrot_gpu.m` (CUDA) | T1 + T3 | Same MATLAB source compiles+runs on NVIDIA via NVRTC; one-source-multi-target story has 2 of 3 platforms. |
| `examples/gpu/mandelbrot_gpu.m` (OpenCL/POCL) | T1 + T4 | Third backend; one-source-three-targets claim closes for at least the toy demo. |
| `examples/gpu/cublas_gemm_demo.m` | T2 + T3 + T4 | Per-backend library replacement (MPS / cuBLAS / clBlast) works; matches the UG p. 2-17 demo against the host BLAS reference. |
| `examples/gpu/streaming_fft.m` | T5 | `gpucoder.reduce` + `gpucoder.matrixMatrixKernel` + GPU `fft` + frame loop across all three backends, validated against host. **Closes the GPU Coder design-pattern claim.** |
| `examples/gpu/sobel_emit.m` | T6 | AOT lane writes self-contained bundles per backend; each builds and runs standalone via its native toolchain. **Closes the standalone-codegen claim.** |
| `examples/gpu/sobel_half.m` | T7.2 | fp16 codegen on all three backends within fp16 tolerance vs the fp64 reference. |
| `gpu-equiv-gate-metal` CI lane green | T7.1 | PIL gate confirms host↔device numerical equivalence for the entire `examples/gpu/*.m` family on the macOS runner. |

---

## 12. Open questions / decision points

These need to be resolved (with the user) **before** Tier-1 commits to
final shapes. Listed in priority order.

1. **`MATLAB_LLVM_GPU` build-time gate strategy**: default `ON` on macOS
   (Metal is always available), default `OFF` on Linux (CUDA / OpenCL
   require external toolchains)? Or always-`OFF` with explicit opt-in?
   → Recommendation: **macOS auto-on (Metal), Linux opt-in** — keeps CI
   matrix simple, matches current cocotb-on-CI pattern.
2. **CPU-fallback semantics in REPL**: should `gpuArray(X)` in the REPL
   *without* a GPU backend silently fall back to host-CPU storage (with
   a one-time warning), or hard-error?
   → Recommendation: **silent fallback + one-time warning** (matches
   MathWorks behaviour when Parallel Computing Toolbox is unavailable).
3. **`MPSGraph` vs hand-coded MSL kernels for `mtimes`**: MPSGraph wins
   on Apple Silicon for medium-large matrices but has measurable
   per-launch overhead. Threshold at 128 (matches UG cuSOLVER threshold)
   or benchmark-driven?
   → Recommendation: **128 as initial cutoff** (matches the UG); revisit
   with a microbenchmark in T7.4.
4. **OpenCL version target**: OpenCL 1.2 (broadest support, including
   macOS-OpenCL — though Apple deprecated OpenCL on macOS 10.14+) or
   OpenCL 2.0+ (SVM, work-group functions, shared virtual memory)?
   → Recommendation: **OpenCL 1.2 baseline + 2.0 feature-gated** (POCL
   supports 1.2 cleanly).
5. **Half-precision storage in the host carrier**: store `half` as
   `uint16_t` on the host side and convert at `gpuArray`/`gather`
   boundaries, or store as `float` and demote on upload? `uint16_t` is
   bit-exact + memory-efficient; `float` is simpler.
   → Recommendation: **`uint16_t` storage** — bit-exact matters for the
   T7.1 PIL gate.
6. **Async vs sync default kernel-launch semantics**: every other lane
   in the project (CPU host, parfor, mflowlink simulate) is
   synchronous. GPU Coder's generated code is also synchronous by
   default (each `cudaMemcpy` blocks). Stay synchronous by default,
   expose `coder.gpu.async()` opt-in?
   → Recommendation: **synchronous by default**.
7. **Tier-7.3 memory pool granularity**: per-launch reset (cheap, no
   long-lived savings) vs per-program-lifetime pool (matches UG memory
   manager, more cudaMalloc savings)? → Recommendation: **per-program
   lifetime, matching the UG**.

---

## 13. Relationship to other roadmaps

- **`embedded_coder_roadmap.md`**: the per-target emit-lane template
  (Python/C/C++/TS/SV) — Tier-6 of *this* roadmap is the GPU
  generalisation. Once shipped, the `-emit-{cuda,metal,opencl}` flags
  sit in the same flag family as `-emit-{c,cpp,python,ts,sv}` in the
  matlabc CLI.
- **`emit_cocotb.md`**: the *exact* JIT-of-emitted-source pattern that
  Tier-1.6 + Tier-2.3 + Tier-3.3 + Tier-4.3 reuse. The cocotb story is
  "emit `.sv` + Python, JIT through `verilator`, verify against
  host"; the GPU story is "emit `.cu`/`.cl`/`.metal`, JIT through
  `nvrtc`/`clBuildProgram`/`MTLLibrary`, verify against host". *Same
  shape*.
- **`fixed_point_toolbox_roadmap.md`**: fp16 is the "fi-like" precision
  type, but on the device side. The shipped `fi` lane's parametric-width
  storage informs Tier-7.2 but doesn't share code (fp16 is IEEE binary16,
  hardware-native on every modern GPU — no rounding/quantisation
  emulation needed).
- **`mflow_link_roadmap.md`**: a future mflowLink-GPU lane (carved out
  in §9) would route per-subsystem dispatch through the Tier-1.5 ABI,
  so an mflow signal graph could mark certain subsystems as
  "execute on GPU" and get device offload free.
- **`dsp_toolbox_roadmap.md`**: shipped DSP System Objects (`dsp.FIRFilter`
  / `dsp.FFT` / `dsp.LMSFilter`) become natural T5/T7 demo targets
  — a `dsp.FIRFilter` operating on a `gpuArray` frame is the
  embarrassingly-parallel happy path for `gpucoder.reduce` +
  `matrixMatrixKernel`. Cross-roadmap synergy without dependency.
- **`feature_status.md`**: add a "GPU" column once T1 lands; mark cells
  per backend (`✅M` for Metal, `✅C` for CUDA, `✅O` for OpenCL).

---

## 14. Open questions for the user (sequencing decisions)

Beyond the technical decisions in §12, the user should confirm the
following before T1 work commits:

- **Confirm Metal is the first backend** (macOS dev box, shortest path
  to a working demo)? Alternative: CUDA first if a Linux+NVIDIA box is
  available + the user wants the cross-vendor proof early.
- **Confirm Deep Learning carve-out**: cuDNN / TensorRT / MPSCNN are
  deferred to a future DL roadmap; this one is **kernel-codegen
  only**. OK?
- **Confirm Tier-6 AOT lane shape**: should `-emit-cuda` write a
  Makefile + a CMakeLists.txt, just a Makefile, or just the sources
  (let the user wire their own build)? Recommendation: Makefile +
  README; matches the UG `packNGo` shape and what cocotb does today.
