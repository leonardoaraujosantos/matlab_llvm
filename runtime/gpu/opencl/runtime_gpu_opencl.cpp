// runtime/gpu/opencl/runtime_gpu_opencl.cpp — OpenCL backend host driver
//
// T4 of docs/gpu_coder_roadmap.md, issue #25.  Strong-overrides the weak
// matlab_gpu_launch_opencl / matlab_gpu_opencl_gemm_double stubs in
// runtime/gpu/runtime_gpu.cpp.  Built only when configured with
// -DMATLAB_LLVM_GPU_OPENCL=ON (see CMakeLists.txt).
//
// Validated on an NVIDIA GPU via its OpenCL ICD (the same lane works on
// AMD / Intel ICDs).  Like the CUDA backend, this TU hand-declares the
// minimal OpenCL 1.2 API rather than including <CL/cl.h>, so it builds
// against just the ICD loader (libOpenCL) with no OpenCL SDK headers.
//
// Scope (parity with the CUDA backend's first HW-validated cut):
//   - Lazy platform/device/context/queue singleton.
//   - matlab_gpu_opencl_gemm_double: fp64 C=A*B via a JIT-built naive
//     GEMM kernel (no clBLAST dependency; clBLAST is a future swap-in,
//     mirroring cuBLAS on the CUDA side).
//   - matlab_gpu_launch_opencl: host-fallback loop (parity with the
//     CUDA/Metal backends — AOT-to-JIT kernel-source linkage is not
//     wired for any backend yet).
//   - matlab_gpu_opencl_device_name for gpuDevice() / diagnostics.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "../../runtime_internal.h"

// ---- minimal OpenCL 1.2 surface, hand-declared (no CL headers needed) ----
extern "C" {
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_program;
typedef void *cl_kernel;
typedef void *cl_mem;
typedef cl_ulong cl_mem_flags;
typedef cl_ulong cl_device_type;
typedef cl_uint cl_bool;
cl_int clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *,
                      cl_uint *);
cl_int clGetDeviceInfo(cl_device_id, cl_uint, std::size_t, void *,
                       std::size_t *);
cl_context clCreateContext(const std::intptr_t *, cl_uint, const cl_device_id *,
                           void *, void *, cl_int *);
cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_ulong,
                                      cl_int *);
cl_program clCreateProgramWithSource(cl_context, cl_uint, const char **,
                                     const std::size_t *, cl_int *);
cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *,
                      void *, void *);
cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_uint, std::size_t,
                             void *, std::size_t *);
cl_kernel clCreateKernel(cl_program, const char *, cl_int *);
cl_mem clCreateBuffer(cl_context, cl_mem_flags, std::size_t, void *, cl_int *);
cl_int clReleaseMemObject(cl_mem);
cl_int clSetKernelArg(cl_kernel, cl_uint, std::size_t, const void *);
cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint,
                              const std::size_t *, const std::size_t *,
                              const std::size_t *, cl_uint, const void *,
                              void *);
cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, std::size_t,
                            std::size_t, const void *, cl_uint, const void *,
                            void *);
cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, std::size_t,
                           std::size_t, void *, cl_uint, const void *, void *);
cl_int clFinish(cl_command_queue);
}
#define MATLAB_CL_DEVICE_TYPE_GPU 4UL
#define MATLAB_CL_DEVICE_TYPE_ALL 0xFFFFFFFFUL
#define MATLAB_CL_DEVICE_NAME 0x102B
#define MATLAB_CL_PROGRAM_BUILD_LOG 0x1183
#define MATLAB_CL_MEM_READ_WRITE 1UL
#define MATLAB_CL_TRUE 1

namespace {

cl_platform_id g_Plat = nullptr;
cl_device_id g_Dev = nullptr;
cl_context g_Ctx = nullptr;
cl_command_queue g_Queue = nullptr;
cl_kernel g_GemmKernel = nullptr;
bool g_Ok = false;
std::atomic<bool> g_Initialized{false};
std::mutex g_InitMtx;

bool debugOn() {
  static int v = -1;
  if (v < 0) v = std::getenv("MATLAB_GPU_DEBUG") ? 1 : 0;
  return v != 0;
}

bool ensureOpenCLDevice() {
  if (g_Initialized.load(std::memory_order_acquire)) return g_Ok;
  std::lock_guard<std::mutex> Lock(g_InitMtx);
  if (g_Initialized.load(std::memory_order_relaxed)) return g_Ok;

  cl_uint np = 0;
  if (clGetPlatformIDs(1, &g_Plat, &np) != 0 || np == 0) {
    std::fprintf(stderr, "matlab_gpu_opencl: no OpenCL platform\n");
    g_Initialized.store(true, std::memory_order_release);
    return false;
  }
  cl_uint nd = 0;
  if (clGetDeviceIDs(g_Plat, MATLAB_CL_DEVICE_TYPE_GPU, 1, &g_Dev, &nd) != 0 ||
      nd == 0) {
    /* Fall back to any device type (CPU ICD, etc.). */
    if (clGetDeviceIDs(g_Plat, MATLAB_CL_DEVICE_TYPE_ALL, 1, &g_Dev, &nd) != 0 ||
        nd == 0) {
      std::fprintf(stderr, "matlab_gpu_opencl: no OpenCL device\n");
      g_Initialized.store(true, std::memory_order_release);
      return false;
    }
  }
  cl_int err = 0;
  g_Ctx = clCreateContext(nullptr, 1, &g_Dev, nullptr, nullptr, &err);
  if (!g_Ctx) {
    g_Initialized.store(true, std::memory_order_release);
    return false;
  }
  g_Queue = clCreateCommandQueue(g_Ctx, g_Dev, 0, &err);
  g_Ok = (g_Queue != nullptr);
  if (g_Ok && debugOn()) {
    char name[256] = {0};
    clGetDeviceInfo(g_Dev, MATLAB_CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::fprintf(stderr, "matlab_gpu_opencl: device=%s\n", name);
  }
  g_Initialized.store(true, std::memory_order_release);
  return g_Ok;
}

// JIT-build the fp64 GEMM kernel once.  Row-major C(MxN)=A(MxK)*B(KxN).
cl_kernel gemmKernel() {
  if (g_GemmKernel) return g_GemmKernel;
  static const char *src =
      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
      "__kernel void matlab_gemm(__global const double* A,\n"
      "    __global const double* B, __global double* C, int M, int N, int K){\n"
      "  int r = get_global_id(0); int c = get_global_id(1);\n"
      "  if (r >= M || c >= N) return;\n"
      "  double s = 0.0;\n"
      "  for (int k = 0; k < K; ++k) s += A[r*K + k] * B[k*N + c];\n"
      "  C[r*N + c] = s;\n"
      "}\n";
  cl_int err = 0;
  std::size_t len = 0;
  cl_program prog = clCreateProgramWithSource(g_Ctx, 1, &src, &len, &err);
  if (clBuildProgram(prog, 1, &g_Dev, "", nullptr, nullptr) != 0) {
    char log[8192] = {0};
    clGetProgramBuildInfo(prog, g_Dev, MATLAB_CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, nullptr);
    std::fprintf(stderr, "matlab_gpu_opencl: GEMM kernel build:\n%s\n", log);
    return nullptr;
  }
  g_GemmKernel = clCreateKernel(prog, "matlab_gemm", &err);
  return g_GemmKernel;
}

}  // namespace

extern "C" {

// Strong override of the weak stub — host-fallback loop (parity with the
// CUDA / Metal backends).
int matlab_gpu_launch_opencl(double start, double step, double end,
                             void *fn_ptr, void *state, int kernel_id) {
  ensureOpenCLDevice();
  if (debugOn())
    std::fprintf(stderr,
                 "matlab_gpu_opencl: launch kernel_id=%d range=[%g:%g:%g] "
                 "(host fallback; AOT-to-JIT linkage pending)\n",
                 kernel_id, start, step, end);
  using KernelFn = void (*)(double, void *);
  KernelFn Fn = reinterpret_cast<KernelFn>(fn_ptr);
  if (step == 0.0) {
    std::fprintf(stderr, "matlab_gpu_opencl: kernel range step is zero\n");
    std::abort();
  }
  if (step > 0.0) {
    for (double iv = start; iv <= end; iv += step) Fn(iv, state);
  } else {
    for (double iv = start; iv >= end; iv += step) Fn(iv, state);
  }
  return 0;
}

// fp64 GEMM via the JIT-built naive kernel.  Row-major C = A * B.
matlab_mat *matlab_gpu_opencl_gemm_double(matlab_mat *A, matlab_mat *B) {
  if (!A || !B) return nullptr;
  int64_t M = A->rows, K = A->cols, Kb = B->rows, N = B->cols;
  if (K != Kb || M <= 0 || N <= 0 || K <= 0) return nullptr;
  if (!ensureOpenCLDevice()) return nullptr;
  cl_kernel k = gemmKernel();
  if (!k) return nullptr;

  std::size_t bytesA = static_cast<std::size_t>(M) * K * sizeof(double);
  std::size_t bytesB = static_cast<std::size_t>(K) * N * sizeof(double);
  std::size_t bytesC = static_cast<std::size_t>(M) * N * sizeof(double);
  cl_int err = 0;
  cl_mem dA = clCreateBuffer(g_Ctx, MATLAB_CL_MEM_READ_WRITE, bytesA, nullptr,
                             &err);
  cl_mem dB = clCreateBuffer(g_Ctx, MATLAB_CL_MEM_READ_WRITE, bytesB, nullptr,
                             &err);
  cl_mem dC = clCreateBuffer(g_Ctx, MATLAB_CL_MEM_READ_WRITE, bytesC, nullptr,
                             &err);
  if (!dA || !dB || !dC) {
    if (dA) clReleaseMemObject(dA);
    if (dB) clReleaseMemObject(dB);
    if (dC) clReleaseMemObject(dC);
    return nullptr;
  }
  clEnqueueWriteBuffer(g_Queue, dA, MATLAB_CL_TRUE, 0, bytesA, A->data, 0,
                       nullptr, nullptr);
  clEnqueueWriteBuffer(g_Queue, dB, MATLAB_CL_TRUE, 0, bytesB, B->data, 0,
                       nullptr, nullptr);
  int m = static_cast<int>(M), n = static_cast<int>(N), kk = static_cast<int>(K);
  clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
  clSetKernelArg(k, 1, sizeof(cl_mem), &dB);
  clSetKernelArg(k, 2, sizeof(cl_mem), &dC);
  clSetKernelArg(k, 3, sizeof(int), &m);
  clSetKernelArg(k, 4, sizeof(int), &n);
  clSetKernelArg(k, 5, sizeof(int), &kk);
  std::size_t gws[2] = {static_cast<std::size_t>(M),
                        static_cast<std::size_t>(N)};
  cl_int le = clEnqueueNDRangeKernel(g_Queue, k, 2, nullptr, gws, nullptr, 0,
                                     nullptr, nullptr);
  clFinish(g_Queue);
  if (le != 0) {
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    return nullptr;
  }
  matlab_mat *C = mat_alloc(M, N);
  clEnqueueReadBuffer(g_Queue, dC, MATLAB_CL_TRUE, 0, bytesC, C->data, 0,
                      nullptr, nullptr);
  clReleaseMemObject(dA);
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
  return C;
}

const char *matlab_gpu_opencl_device_name(void) {
  if (!ensureOpenCLDevice()) return "OpenCL (no device)";
  static char Name[300] = {0};
  if (!Name[0]) {
    char raw[256] = {0};
    clGetDeviceInfo(g_Dev, MATLAB_CL_DEVICE_NAME, sizeof(raw), raw, nullptr);
    std::snprintf(Name, sizeof(Name), "%s (OpenCL)", raw[0] ? raw : "GPU");
  }
  return Name;
}

}  // extern "C"
