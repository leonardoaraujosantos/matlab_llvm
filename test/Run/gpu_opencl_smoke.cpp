// test/Run/gpu_opencl_smoke.cpp — OpenCL backend device-validation smoke.
//
// T4 of docs/gpu_coder_roadmap.md, issue #25.  Mirrors gpu_cuda_smoke.cpp.
// Requires an OpenCL ICD + device (validated on NVIDIA via its ICD; also
// works on AMD / Intel).  Validates, fp64 ±1e-9:
//   1. matlab_gpu_opencl_gemm_double — fp64 C=A*B via the JIT-built kernel.
//   2. The matlab_gpu_gemm dispatcher under MATLAB_GPU_TARGET=opencl —
//      proves the runtime_gpu.cpp OpenCL arm routes through the backend.
// (AXPY is covered end-to-end by the -emit-opencl bundle in the runner.)

#include <cmath>
#include <cstdint>
#include <cstdio>

struct matlab_mat {
  double *data;
  int64_t rows;
  int64_t cols;
};
extern "C" matlab_mat *mat_alloc(int64_t m, int64_t n);
extern "C" {
matlab_mat *matlab_gpu_opencl_gemm_double(matlab_mat *A, matlab_mat *B);
const char *matlab_gpu_opencl_device_name(void);
matlab_mat *matlab_gpu_gemm(matlab_mat *A, matlab_mat *B);
}

static int g_fail = 0;
static void check(bool ok, const char *what, double err) {
  std::printf("  %-32s %s (err=%.3g)\n", what, ok ? "PASS" : "FAIL", err);
  if (!ok) g_fail = 1;
}
static matlab_mat *makeMat(int m, int n, const double *v) {
  matlab_mat *M = mat_alloc(m, n);
  for (int i = 0; i < m * n; ++i) M->data[i] = v[i];
  return M;
}
static double gemmErr(const matlab_mat *C, const double *ref, int m, int n) {
  double e = 0.0;
  if (!C) return 1e9;
  for (int i = 0; i < m * n; ++i) e = std::fmax(e, std::fabs(C->data[i] - ref[i]));
  return e;
}

int main() {
  std::printf("opencl smoke: device = %s\n", matlab_gpu_opencl_device_name());

  // 1. low-level GEMM: A=[[1,2],[3,4]] B=[[5,6],[7,8]] -> [[19,22],[43,50]]
  double a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8}, ref[4] = {19, 22, 43, 50};
  matlab_mat *A = makeMat(2, 2, a), *B = makeMat(2, 2, b);
  matlab_mat *C = matlab_gpu_opencl_gemm_double(A, B);
  check(gemmErr(C, ref, 2, 2) < 1e-9, "opencl fp64 GEMM", gemmErr(C, ref, 2, 2));

  // 2. high-level dispatcher (target=opencl): 4x4 identity * counter == counter
  double a4[16], b4[16];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      a4[i * 4 + j] = (i == j) ? 1.0 : 0.0;
      b4[i * 4 + j] = i * 4 + j + 1;
    }
  matlab_mat *A4 = makeMat(4, 4, a4), *B4 = makeMat(4, 4, b4);
  matlab_mat *C4 = matlab_gpu_gemm(A4, B4);
  check(gemmErr(C4, b4, 4, 4) < 1e-9, "matlab_gpu_gemm dispatch",
        gemmErr(C4, b4, 4, 4));

  std::printf("opencl smoke: %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail;
}
