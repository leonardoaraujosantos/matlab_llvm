// runtime/gpu/cuda/runtime_gpu_cuda.cpp — CUDA backend host driver
//
// T3 of docs/gpu_coder_roadmap.md.  Strong-overrides the weak
// matlab_gpu_launch_cuda / matlab_gpu_cuda_gemm_double stubs in
// runtime/gpu/runtime_gpu.cpp.  Built only when configured with
// -DMATLAB_LLVM_GPU_CUDA=ON (see CMakeLists.txt); on hosts without the
// flag the weak stubs stay live and report "backend not built in".
//
// Implementation notes
// --------------------
// We deliberately use only the **CUDA driver API** (cuda.h), **NVRTC**
// (nvrtc.h), and a handful of **hand-declared cuBLAS v2 prototypes**.
// We do NOT include <cuda_runtime.h> or <cublas_v2.h>: those headers
// pull in cuda_fp16.h, which on the pip-wheel CUDA stack needs the
// libcudacxx `<nv/target>` header that isn't always shipped.  The
// driver API + NVRTC is also the natural level for a JIT runtime, and
// declaring the three cuBLAS entry points we need keeps the build
// dependency-free beyond linking libcublas.
//
// Mirrors the structure of runtime/gpu/metal/runtime_gpu_metal.mm:
//   - Lazy CUcontext / CUdevice singleton (ensureCudaDevice()).
//   - matlab_gpu_cuda_alloc/free/h2d/d2h: cuMemAlloc round-trip.
//   - matlab_gpu_cuda_jit_compile: NVRTC -> PTX -> cuModuleLoadData,
//     cached by source-hash; matlab_gpu_cuda_dispatch: cuLaunchKernel.
//   - matlab_gpu_cuda_gemm_double: cuBLAS Dgemm (fp64), row-major C=A*B.
//   - matlab_gpu_launch_cuda: host-fallback sequential loop calling the
//     outlined fn_ptr — identical to what the Metal backend's
//     matlab_gpu_launch_metal does today (AOT-to-JIT kernel-source
//     linkage is not wired for any backend yet; this keeps
//     MATLAB_GPU_TARGET=cuda from aborting and runs kernelfun fixtures
//     correctly on the host).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#include "../../runtime_internal.h"

// ---- minimal cuBLAS v2 surface, declared by hand (avoid fp16 headers) ----
extern "C" {
typedef void *cublasHandle_t;
typedef int   cublasStatus_t;     // 0 == CUBLAS_STATUS_SUCCESS
typedef int   cublasOperation_t;  // 0 == CUBLAS_OP_N
cublasStatus_t cublasCreate_v2(cublasHandle_t *);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);
cublasStatus_t cublasDgemm_v2(cublasHandle_t, cublasOperation_t,
                              cublasOperation_t, int m, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta,
                              double *C, int ldc);
}
#define MATLAB_CUBLAS_OP_N 0

namespace {

CUdevice  g_Device = 0;
CUcontext g_Context = nullptr;
int g_CcMajor = 0, g_CcMinor = 0;
bool g_Ok = false;
std::atomic<bool> g_Initialized{false};
std::mutex g_InitMtx;

bool debugOn() {
  static int v = -1;
  if (v < 0) v = std::getenv("MATLAB_GPU_DEBUG") ? 1 : 0;
  return v != 0;
}

const char *cuErr(CUresult r) {
  const char *s = nullptr;
  cuGetErrorString(r, &s);
  return s ? s : "(unknown CUDA error)";
}

// Lazy driver-API context init.  Returns true once a usable device +
// primary context is available.  Thread-safe; idempotent.
bool ensureCudaDevice() {
  if (g_Initialized.load(std::memory_order_acquire)) return g_Ok;
  std::lock_guard<std::mutex> Lock(g_InitMtx);
  if (g_Initialized.load(std::memory_order_relaxed)) return g_Ok;

  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "matlab_gpu_cuda: cuInit failed: %s\n", cuErr(r));
    g_Initialized.store(true, std::memory_order_release);
    return false;
  }
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
    std::fprintf(stderr, "matlab_gpu_cuda: no CUDA devices found\n");
    g_Initialized.store(true, std::memory_order_release);
    return false;
  }
  if (cuDeviceGet(&g_Device, 0) != CUDA_SUCCESS) {
    g_Initialized.store(true, std::memory_order_release);
    return false;
  }
  cuDeviceGetAttribute(&g_CcMajor,
                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_Device);
  cuDeviceGetAttribute(&g_CcMinor,
                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_Device);
  // Retain the primary context (shared with the cudart-based libraries
  // like cuBLAS, which also bind to the primary context on the device).
  r = cuDevicePrimaryCtxRetain(&g_Context, g_Device);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "matlab_gpu_cuda: primary ctx retain failed: %s\n",
                 cuErr(r));
    g_Initialized.store(true, std::memory_order_release);
    return false;
  }
  cuCtxSetCurrent(g_Context);
  g_Ok = true;
  if (debugOn()) {
    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), g_Device);
    std::fprintf(stderr, "matlab_gpu_cuda: device=%s sm_%d%d\n", name,
                 g_CcMajor, g_CcMinor);
  }
  g_Initialized.store(true, std::memory_order_release);
  return g_Ok;
}

// Bind the retained primary context to the calling thread.  cuBLAS /
// driver-API calls need a current context; the runtime is called from
// arbitrary host threads (JIT, parfor workers), so set it on entry.
void bindContext() {
  if (g_Context) cuCtxSetCurrent(g_Context);
}

cublasHandle_t cublasHandle() {
  static cublasHandle_t h = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    bindContext();
    if (cublasCreate_v2(&h) != 0) h = nullptr;
  });
  return h;
}

struct CachedModule {
  CUmodule mod;
  CUfunction fn;
};

std::unordered_map<std::string, CachedModule> &moduleCache() {
  static std::unordered_map<std::string, CachedModule> Cache;
  return Cache;
}

}  // namespace

extern "C" {

// ====================================================================
// Device-buffer ABI.  Driver-API cuMemAlloc round-trip.  Caller treats
// the return value as an opaque void* (a CUdeviceptr widened to 64-bit).
// ====================================================================
void *matlab_gpu_cuda_alloc(std::size_t bytes) {
  if (!ensureCudaDevice()) return std::malloc(bytes);
  bindContext();
  CUdeviceptr d = 0;
  CUresult r = cuMemAlloc(&d, bytes ? bytes : 1);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "matlab_gpu_cuda_alloc: %s (%zu bytes)\n", cuErr(r),
                 bytes);
    return nullptr;
  }
  return reinterpret_cast<void *>(static_cast<std::uintptr_t>(d));
}

void matlab_gpu_cuda_free(void *ptr) {
  if (!ptr) return;
  if (!g_Ok) {
    std::free(ptr);
    return;
  }
  bindContext();
  cuMemFree(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(ptr)));
}

void matlab_gpu_cuda_h2d(void *dst, const void *src, std::size_t bytes) {
  if (!g_Ok) {
    std::memcpy(dst, src, bytes);
    return;
  }
  bindContext();
  cuMemcpyHtoD(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(dst)),
               src, bytes);
}

void matlab_gpu_cuda_d2h(void *dst, const void *src, std::size_t bytes) {
  if (!g_Ok) {
    std::memcpy(dst, src, bytes);
    return;
  }
  bindContext();
  cuMemcpyDtoH(dst,
               static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(src)),
               bytes);
}

void matlab_gpu_cuda_sync(void) {
  if (!g_Ok) return;
  bindContext();
  cuCtxSynchronize();
}

// ====================================================================
// NVRTC JIT compile + cache.  Compiles CUDA-C source via NVRTC for the
// active device's compute capability, loads the PTX, and looks up the
// named kernel.  Caches the (module, function) by source+name hash so
// repeat launches skip the compile.  Returns 0 on success and stashes
// an opaque CUfunction in *out_fn.
// ====================================================================
int matlab_gpu_cuda_jit_compile(const char *src, const char *name,
                                void **out_fn) {
  if (!ensureCudaDevice()) return -1;
  bindContext();
  std::string Key(src);
  Key += "\n@@@KNAME@@@";
  Key += name;
  auto &Cache = moduleCache();
  auto It = Cache.find(Key);
  if (It != Cache.end()) {
    if (out_fn) *out_fn = reinterpret_cast<void *>(It->second.fn);
    return 0;
  }

  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, src, name, 0, nullptr, nullptr) !=
      NVRTC_SUCCESS)
    return -2;
  char arch[40];
  std::snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", g_CcMajor,
                g_CcMinor);
  const char *opts[] = {arch};
  nvrtcResult cr = nvrtcCompileProgram(prog, 1, opts);
  if (cr != NVRTC_SUCCESS) {
    std::size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::vector<char> log(logSize ? logSize : 1);
    nvrtcGetProgramLog(prog, log.data());
    std::fprintf(stderr,
                 "matlab_gpu_cuda_jit_compile: NVRTC failed for '%s':\n%s\n",
                 name, log.data());
    nvrtcDestroyProgram(&prog);
    return -3;
  }
  std::size_t ptxSize = 0;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::vector<char> ptx(ptxSize);
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);

  CUmodule mod = nullptr;
  CUresult r = cuModuleLoadData(&mod, ptx.data());
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "matlab_gpu_cuda_jit_compile: module load: %s\n",
                 cuErr(r));
    return -4;
  }
  CUfunction fn = nullptr;
  r = cuModuleGetFunction(&fn, mod, name);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "matlab_gpu_cuda_jit_compile: get function '%s': %s\n",
                 name, cuErr(r));
    cuModuleUnload(mod);
    return -5;
  }
  Cache[Key] = {mod, fn};
  if (out_fn) *out_fn = reinterpret_cast<void *>(fn);
  return 0;
}

// Launch a precompiled kernel.  fn_handle is the opaque CUfunction from
// jit_compile; kernel_args is the cuLaunchKernel-style array of pointers
// to each argument; grid_size is the 1-D thread count (block size 256).
int matlab_gpu_cuda_dispatch(void *fn_handle, void **kernel_args,
                             int grid_size) {
  if (!g_Ok || !fn_handle) return -1;
  bindContext();
  CUfunction fn = reinterpret_cast<CUfunction>(fn_handle);
  const int block = 256;
  int gridBlocks = (grid_size + block - 1) / block;
  if (gridBlocks < 1) gridBlocks = 1;
  CUresult r = cuLaunchKernel(fn, gridBlocks, 1, 1, block, 1, 1, 0, nullptr,
                              kernel_args, nullptr);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "matlab_gpu_cuda_dispatch: launch: %s\n", cuErr(r));
    return -2;
  }
  if (cuCtxSynchronize() != CUDA_SUCCESS) return -3;
  return 0;
}

// ====================================================================
// matlab_gpu_launch_cuda — strong override of the weak stub.  Host-
// fallback sequential loop (parity with matlab_gpu_launch_metal).  This
// runs the outlined kernel function pointer once per iteration on the
// host: the AOT-to-JIT kernel-source linkage that would dispatch the
// registered NVRTC source isn't wired for any backend yet, so this keeps
// MATLAB_GPU_TARGET=cuda functional for coder.gpu.kernelfun fixtures
// instead of aborting.
// ====================================================================
int matlab_gpu_launch_cuda(double start, double step, double end, void *fn_ptr,
                           void *state, int kernel_id) {
  // Touch the device so gpuDevice()/diagnostics report a live context,
  // even though the compute below runs on the host for now.
  ensureCudaDevice();
  if (debugOn())
    std::fprintf(stderr,
                 "matlab_gpu_cuda: launch kernel_id=%d range=[%g:%g:%g] "
                 "(host fallback; AOT-to-JIT linkage pending)\n",
                 kernel_id, start, step, end);
  using KernelFn = void (*)(double, void *);
  KernelFn Fn = reinterpret_cast<KernelFn>(fn_ptr);
  if (step == 0.0) {
    std::fprintf(stderr, "matlab_gpu_cuda: kernel range step is zero\n");
    std::abort();
  }
  if (step > 0.0) {
    for (double iv = start; iv <= end; iv += step) Fn(iv, state);
  } else {
    for (double iv = start; iv >= end; iv += step) Fn(iv, state);
  }
  return 0;
}

// ====================================================================
// matlab_gpu_cuda_gemm_double — cuBLAS Dgemm (fp64).  Strong override of
// the weak stub.  Computes row-major C = A * B.  cuBLAS is column-major,
// so a row-major array X[m,n] is seen by cuBLAS as X^T[n,m]; computing
// C^T = B^T * A^T in column-major (i.e. cublasDgemm(B, A) with no
// transpose) yields exactly the row-major C we want.  RTX-class GPUs
// support fp64 natively, so the result matches the CPU lane to ~1e-12.
//
// Returns a freshly mat_alloc'd C, or nullptr on dim mismatch / failure
// (the runtime_gpu.cpp dispatcher then falls back to the CPU GEMM).
// ====================================================================
matlab_mat *matlab_gpu_cuda_gemm_double(matlab_mat *A, matlab_mat *B) {
  if (!A || !B) return nullptr;
  int64_t M = A->rows, K = A->cols, Kb = B->rows, N = B->cols;
  if (K != Kb || M <= 0 || N <= 0 || K <= 0) return nullptr;
  if (!ensureCudaDevice()) return nullptr;
  bindContext();
  cublasHandle_t h = cublasHandle();
  if (!h) return nullptr;

  std::size_t bytesA = static_cast<std::size_t>(M) * K * sizeof(double);
  std::size_t bytesB = static_cast<std::size_t>(K) * N * sizeof(double);
  std::size_t bytesC = static_cast<std::size_t>(M) * N * sizeof(double);
  CUdeviceptr dA = 0, dB = 0, dC = 0;
  if (cuMemAlloc(&dA, bytesA) != CUDA_SUCCESS) return nullptr;
  if (cuMemAlloc(&dB, bytesB) != CUDA_SUCCESS) {
    cuMemFree(dA);
    return nullptr;
  }
  if (cuMemAlloc(&dC, bytesC) != CUDA_SUCCESS) {
    cuMemFree(dA);
    cuMemFree(dB);
    return nullptr;
  }
  cuMemcpyHtoD(dA, A->data, bytesA);
  cuMemcpyHtoD(dB, B->data, bytesB);

  // Row-major C(MxN) = A(MxK) * B(KxN)  ==  col-major gemm(B, A):
  //   m=N, n=M, k=K; A_arg=dB (lda=N), B_arg=dA (ldb=K), C=dC (ldc=N).
  double one = 1.0, zero = 0.0;
  cublasStatus_t st = cublasDgemm_v2(
      h, MATLAB_CUBLAS_OP_N, MATLAB_CUBLAS_OP_N, static_cast<int>(N),
      static_cast<int>(M), static_cast<int>(K), &one,
      reinterpret_cast<const double *>(dB), static_cast<int>(N),
      reinterpret_cast<const double *>(dA), static_cast<int>(K), &zero,
      reinterpret_cast<double *>(dC), static_cast<int>(N));
  cuCtxSynchronize();
  if (st != 0) {
    std::fprintf(stderr, "matlab_gpu_cuda_gemm_double: cublasDgemm rc=%d\n",
                 st);
    cuMemFree(dA);
    cuMemFree(dB);
    cuMemFree(dC);
    return nullptr;
  }

  matlab_mat *C = mat_alloc(M, N);
  cuMemcpyDtoH(C->data, dC, bytesC);
  cuMemFree(dA);
  cuMemFree(dB);
  cuMemFree(dC);
  return C;
}

// Device-name probe for gpuDevice() / diagnostics.
const char *matlab_gpu_cuda_device_name(void) {
  if (!ensureCudaDevice()) return "CUDA (no device)";
  static char Name[300] = {0};
  if (!Name[0]) {
    char raw[256] = {0};
    cuDeviceGetName(raw, sizeof(raw), g_Device);
    std::snprintf(Name, sizeof(Name), "%s (sm_%d%d)", raw[0] ? raw : "NVIDIA GPU",
                  g_CcMajor, g_CcMinor);
  }
  return Name;
}

}  // extern "C"
