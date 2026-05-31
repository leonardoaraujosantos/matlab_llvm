/* runtime/gpu/runtime_gpu.cpp — GPU Coder runtime ABI (T1 skeleton).
 *
 * This TU exports the C ABI the MLIR LowerGpuKernels pass emits a call
 * to: `matlab_gpu_launch_kernel(start, step, end, fn_ptr, state_ptr,
 * kernel_id)`.  The dispatcher routes by the active backend selected
 * at process start by the `MATLAB_GPU_TARGET` env var:
 *
 *   "cpu" or unset       → CPU-debug fallback (sequential host loop)
 *   "metal"              → Tier-2 Metal backend (see runtime/gpu/metal/)
 *   "cuda"               → Tier-3 CUDA backend  (Linux only)
 *   "opencl"             → Tier-4 OpenCL backend
 *   "auto"               → Metal on macOS, CUDA → OpenCL → CPU on Linux
 *
 * T1 ships only the CPU-debug fallback.  The Metal/CUDA/OpenCL arms
 * are weak-linked: when their backend isn't built in, the dispatcher
 * hard-errors with a clear diagnostic naming the missing target and
 * the build flag that enables it.  This matches the locked §12
 * decision: no silent fallback when a backend is explicitly requested.
 *
 * The outlined kernel function has the signature
 *     void __gpu_kernel_<id>(double iv, void *state)
 * — identical to the `__parfor_body_<id>` signature so the CPU-debug
 * lane can call it directly without any per-target glue.
 *
 * The `state` pointer carries the reduction-pointer array (one ptr per
 * reduction var) or null if the kernel has no reductions.  Same shape
 * as LowerParfor.
 */

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "runtime_internal.h"

/* Forward declarations of matlab_obj accessors defined in matlab_runtime.cpp.
 * These aren't in runtime_internal.h yet; the dsp/toolbox lane uses the
 * same pattern (see runtime/toolbox/dsp/runtime_dsp.cpp). */
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name,
                                          int64_t len);

namespace {

enum class GpuTarget : int {
  Unset  = 0,  /* falls back to CPU-debug */
  Cpu    = 1,
  Metal  = 2,
  Cuda   = 3,
  OpenCl = 4,
};

GpuTarget activeTarget() {
  static GpuTarget T = []() {
    const char *S = std::getenv("MATLAB_GPU_TARGET");
    if (!S || !*S) return GpuTarget::Cpu;  /* default to CPU-debug */
    if (!std::strcmp(S, "cpu"))    return GpuTarget::Cpu;
    if (!std::strcmp(S, "metal"))  return GpuTarget::Metal;
    if (!std::strcmp(S, "cuda"))   return GpuTarget::Cuda;
    if (!std::strcmp(S, "opencl")) return GpuTarget::OpenCl;
    if (!std::strcmp(S, "auto")) {
#if defined(__APPLE__)
      return GpuTarget::Metal;
#else
      /* Linux: try CUDA → OpenCL → CPU at runtime; for now default
       * to CPU and let the backend init code escalate. */
      return GpuTarget::Cpu;
#endif
    }
    std::fprintf(stderr,
        "matlab_gpu: unknown MATLAB_GPU_TARGET='%s' "
        "(expected cpu|metal|cuda|opencl|auto). Falling back to cpu.\n", S);
    return GpuTarget::Cpu;
  }();
  return T;
}

const char *targetName(GpuTarget T) {
  switch (T) {
    case GpuTarget::Cpu:    return "cpu";
    case GpuTarget::Metal:  return "metal";
    case GpuTarget::Cuda:   return "cuda";
    case GpuTarget::OpenCl: return "opencl";
    default:                return "unset";
  }
}

/* Backend init / launch hooks.  T1 has only the CPU lane implemented;
 * Metal / CUDA / OpenCL backends register themselves at link time by
 * overriding the weak symbols below.  When a requested backend's
 * symbol is the default weak stub, we report "not built in" and
 * abort. */

/* Kernel-source registry — populated by the emit passes (Tier-2/3/4)
 * before launch.  Each entry holds the (target, source-text, fn-name)
 * tuple that the per-backend JIT consumes.  T1 leaves this empty; the
 * CPU lane doesn't need it. */
struct GpuKernelSource {
  const char *target;
  const char *name;
  const char *source;
};
/* Reserve a fixed slot table for now; backends grow this when they
 * land.  Linear scan is fine — kernel counts per program are O(10). */
static constexpr std::size_t kMaxKernels = 256;
static GpuKernelSource KernelSources[kMaxKernels];
static std::atomic<std::size_t> NumKernels{0};

extern "C" void matlab_gpu_register_kernel(int kernel_id, const char *target,
                                           const char *name,
                                           const char *source) {
  /* Bounded table; refuse silent overflow. */
  std::size_t idx = static_cast<std::size_t>(kernel_id);
  if (idx >= kMaxKernels) {
    std::fprintf(stderr,
        "matlab_gpu_register_kernel: kernel_id %d exceeds table size %zu\n",
        kernel_id, kMaxKernels);
    std::abort();
  }
  KernelSources[idx] = {target, name, source};
  std::size_t cur = NumKernels.load(std::memory_order_relaxed);
  while (idx + 1 > cur &&
         !NumKernels.compare_exchange_weak(cur, idx + 1,
                                           std::memory_order_relaxed))
    ;  /* retry */
}

/* Tier-2/3/4 backends define and *strong*-export these; the default
 * stubs below report the missing-backend error. */
extern "C" __attribute__((weak)) int
matlab_gpu_launch_metal(double, double, double, void *, void *, int) {
  std::fprintf(stderr,
      "matlab_gpu: MATLAB_GPU_TARGET=metal but the Metal backend is not "
      "built in.  Reconfigure with -DMATLAB_LLVM_GPU_METAL=ON.\n");
  std::abort();
}

/* Phase 4 of LAPACK roadmap (#45 §4) — Metal MPS gemm hook.  Defined
 * strongly by runtime_gpu_metal.mm when the Metal TU is in the link
 * line; the weak stub returns nullptr so the dispatcher below falls
 * back to the CPU lane on hosts without Metal linked in. */
extern "C" __attribute__((weak)) matlab_mat *
matlab_gpu_metal_gemm_double(matlab_mat *, matlab_mat *) {
  return nullptr;  /* "Metal backend not linked — fall back" */
}
extern "C" __attribute__((weak)) int
matlab_gpu_launch_cuda(double, double, double, void *, void *, int) {
  std::fprintf(stderr,
      "matlab_gpu: MATLAB_GPU_TARGET=cuda but the CUDA backend is not "
      "built in.  Reconfigure with -DMATLAB_LLVM_GPU_CUDA=ON.\n");
  std::abort();
}

/* Tier-3 CUDA cuBLAS gemm hook — defined strongly by
 * runtime/gpu/cuda/runtime_gpu_cuda.cpp when the CUDA TU is in the link
 * line; the weak stub returns nullptr so the gemm dispatcher below
 * falls back to the CPU lane on hosts without CUDA linked in. */
extern "C" __attribute__((weak)) matlab_mat *
matlab_gpu_cuda_gemm_double(matlab_mat *, matlab_mat *) {
  return nullptr;  /* "CUDA backend not linked — fall back" */
}
/* Device-name probe — strong in the CUDA TU; weak stub here. */
extern "C" __attribute__((weak)) const char *
matlab_gpu_cuda_device_name(void) {
  return nullptr;  /* "CUDA backend not linked" */
}
extern "C" __attribute__((weak)) int
matlab_gpu_launch_opencl(double, double, double, void *, void *, int) {
  std::fprintf(stderr,
      "matlab_gpu: MATLAB_GPU_TARGET=opencl but the OpenCL backend is not "
      "built in.  Reconfigure with -DMATLAB_LLVM_GPU_OPENCL=ON.\n");
  std::abort();
}

}  /* namespace */

extern "C" {

/* ====================================================================
 * matlab_gpu_launch_kernel — dispatch entry point emitted by
 * LowerGpuKernels.  Calls the per-target backend on hot paths; the
 * CPU-debug lane (T1 default) calls the outlined function pointer
 * directly in a sequential loop.
 *
 * The outlined kernel signature is `void(*)(double iv, void *state)`.
 * State carries the reduction-pointer array (or null).
 * ==================================================================== */
void matlab_gpu_launch_kernel(double start, double step, double end,
                              void *fn_ptr, void *state, int kernel_id) {
  GpuTarget T = activeTarget();
  switch (T) {
    case GpuTarget::Metal:
      matlab_gpu_launch_metal(start, step, end, fn_ptr, state, kernel_id);
      return;
    case GpuTarget::Cuda:
      matlab_gpu_launch_cuda(start, step, end, fn_ptr, state, kernel_id);
      return;
    case GpuTarget::OpenCl:
      matlab_gpu_launch_opencl(start, step, end, fn_ptr, state, kernel_id);
      return;
    case GpuTarget::Cpu:
    case GpuTarget::Unset:
    default:
      break;
  }
  /* CPU-debug fallback — invoke the outlined function once per
   * iteration sequentially on the host.  Matches the parfor CPU
   * dispatch except without the pthread fan-out, so single-step
   * debugging works frame-by-frame.
   *
   * Loop step semantics match MATLAB `start:step:end`:
   *   - positive step  → iterate while iv <= end
   *   - negative step  → iterate while iv >= end
   *   - zero step      → infinite loop in MATLAB (don't enforce; we
   *                       refuse and abort to stop pathological code)
   */
  using KernelFn = void(*)(double, void *);
  KernelFn Fn = reinterpret_cast<KernelFn>(fn_ptr);
  if (step == 0.0) {
    std::fprintf(stderr, "matlab_gpu: kernel range step is zero — aborting\n");
    std::abort();
  }
  if (step > 0.0) {
    for (double iv = start; iv <= end; iv += step) Fn(iv, state);
  } else {
    for (double iv = start; iv >= end; iv += step) Fn(iv, state);
  }
}

/* ====================================================================
 * Minimal device-buffer ABI.  T1 only needs the host-side carrier
 * (gpuArray classdef property values + gather()).  Until a real
 * backend lands, malloc / free / h2d / d2h are simple host allocs +
 * memcpy so the carrier round-trips correctly.
 * ==================================================================== */
void *matlab_gpu_malloc(std::size_t bytes) {
  void *p = std::malloc(bytes);
  if (!p && bytes) {
    std::fprintf(stderr, "matlab_gpu_malloc: out of memory (%zu bytes)\n",
                 bytes);
    std::abort();
  }
  return p;
}

void matlab_gpu_free(void *ptr) { std::free(ptr); }

void matlab_gpu_h2d(void *dst, const void *src, std::size_t bytes) {
  std::memcpy(dst, src, bytes);
}

void matlab_gpu_d2h(void *dst, const void *src, std::size_t bytes) {
  std::memcpy(dst, src, bytes);
}

void matlab_gpu_d2d(void *dst, const void *src, std::size_t bytes) {
  std::memcpy(dst, src, bytes);
}

void matlab_gpu_sync(void) { /* CPU lane is synchronous */ }

/* gpuDevice() reports a 1-line summary of the active backend.  In T1
 * this is the CPU-debug lane; later backends override `target_name`
 * via real device-info queries. */
const char *matlab_gpu_device_name(void) {
  GpuTarget T = activeTarget();
  switch (T) {
    case GpuTarget::Metal:  return "Apple Metal (Tier-2)";
    case GpuTarget::Cuda: {
      /* Real device name when the CUDA backend is linked + a device is
       * present; otherwise the generic Tier-3 label. */
      const char *N = matlab_gpu_cuda_device_name();
      return N ? N : "NVIDIA CUDA (Tier-3)";
    }
    case GpuTarget::OpenCl: return "OpenCL (Tier-4)";
    case GpuTarget::Cpu:    return "CPU debug lane (Tier-1)";
    default:                return "CPU debug lane (Tier-1)";
  }
}

int matlab_gpu_target_id(void) {
  return static_cast<int>(activeTarget());
}

const char *matlab_gpu_target_name(void) {
  return targetName(activeTarget());
}

const char *matlab_gpu_active_target_name(void) {
  return matlab_gpu_target_name();
}

/* T1 gpuArray ABI shims.  These are the runtime entries the
 * gpu_classdefs.m methods forward to.  In T1 the device buffer is
 * just the host buffer (CPU-debug lane); when a real backend lands
 * (T2 Metal / T3 CUDA / T4 OpenCL), these dispatch to the active
 * backend's allocator + h2d/d2h. */

/* matlab_gpu_upload(host_mat) returns an opaque DevicePtr that the
 * gpuArray carrier stashes in its DevicePtr property.  T1 returns
 * the host pointer unchanged — the CPU-debug lane never needs to
 * round-trip a real device copy. */
void *matlab_gpu_upload(void *host_mat) {
  if (activeTarget() == GpuTarget::Cpu || activeTarget() == GpuTarget::Unset) {
    return host_mat;  /* CPU-debug: device == host */
  }
  /* Backends will implement: alloc device buffer, h2d copy. */
  return host_mat;  /* Stub until T2 wires this. */
}

/* matlab_gpu_download(gpuArray_obj) — copy back to host.  T1 reads
 * the object's `Underlying` property and returns its value (a
 * matlab_mat*). */
void *matlab_gpu_download(void *obj) {
  return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(obj),
                            "Underlying", 10);
}

double matlab_gpu_exists_on_gpu(void *obj) {
  /* True when the carrier exists.  Refined when DevicePtr tracking
   * becomes meaningful in T2+. */
  return obj ? 1.0 : 0.0;
}

/* ====================================================================
 * matlab_gpu_gemm — Phase 4 (LAPACK roadmap §4) entry point.
 *
 * The user-facing surface lets the MATLAB level invoke the active
 * backend's GEMM library replacement (MPSMatrixMultiplication on
 * Metal; cuBLAS sgemm on CUDA — future).  Mirrors the CPU-side
 * matlab_matmul_mm signature so caller and call site shape are
 * symmetric.
 *
 * Falls back to the host CPU lane (matlab_matmul_mm) when the active
 * target is CPU, when the matrix shape is below the GPU-launch
 * threshold, or when the backend hook returns nullptr (e.g. Metal
 * device unavailable).
 *
 * Threshold: matches MathWorks' cusolver dispatch (N >= 128) — below
 * that the upload + downcast + launch overhead dwarfs the kernel
 * itself even on M-series UMA.  Tunable via MATLAB_GPU_GEMM_MIN env
 * var for benchmark sweeps. */
extern "C" matlab_mat *matlab_matmul_mm(matlab_mat *A, matlab_mat *B);

matlab_mat *matlab_gpu_gemm(matlab_mat *A, matlab_mat *B) {
  if (!A || !B) return nullptr;
  static int threshold = -1;
  if (threshold < 0) {
    const char *env = std::getenv("MATLAB_GPU_GEMM_MIN");
    threshold = (env && *env) ? std::atoi(env) : 128;
  }
  int64_t M = A->rows, K = A->cols, N = B->cols;
  bool big_enough = (M >= threshold && N >= threshold && K >= threshold);
  GpuTarget T = activeTarget();
  if (T == GpuTarget::Metal && big_enough) {
    matlab_mat *C = matlab_gpu_metal_gemm_double(A, B);
    if (C) return C;
    /* Backend hook unavailable / failed — fall through to CPU. */
  }
  if (T == GpuTarget::Cuda && big_enough) {
    matlab_mat *C = matlab_gpu_cuda_gemm_double(A, B);
    if (C) return C;
    /* Backend hook unavailable / failed — fall through to CPU. */
  }
  return matlab_matmul_mm(A, B);
}

}  /* extern "C" */
