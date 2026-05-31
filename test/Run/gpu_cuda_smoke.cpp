// test/Run/gpu_cuda_smoke.cpp — CUDA backend device-validation smoke.
//
// T3 of docs/gpu_coder_roadmap.md, issue #25.  Mirrors the Metal smoke
// (gpu_metal_mps_smoke.mm).  Requires an NVIDIA GPU + the CUDA backend
// linked in (built by run_gpu_cuda_validation.sh).  Validates three
// things on real hardware, ±1e-9 (fp64):
//
//   1. cuBLAS Dgemm via matlab_gpu_cuda_gemm_double — row-major C = A*B.
//   2. NVRTC compile + driver-API launch via the matlab_gpu_cuda_*
//      buffer + jit_compile + dispatch ABI (AXPY: out = a*x + y).
//   3. The high-level matlab_gpu_gemm dispatcher under
//      MATLAB_GPU_TARGET=cuda (set by the runner) — proves the
//      runtime_gpu.cpp CUDA arm routes through cuBLAS.
//
// Prints "cuda smoke: PASS" + exits 0 on success; non-zero on any
// mismatch or device failure.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

/* matlab_mat layout (runtime/runtime_internal.h) + mat_alloc from
 * MatlabRuntime, which the runner links. */
struct matlab_mat {
  double *data;
  int64_t rows;
  int64_t cols;
};
extern "C" matlab_mat *mat_alloc(int64_t m, int64_t n);

/* CUDA backend ABI (runtime/gpu/cuda/runtime_gpu_cuda.cpp). */
extern "C" {
void *matlab_gpu_cuda_alloc(std::size_t bytes);
void matlab_gpu_cuda_free(void *ptr);
void matlab_gpu_cuda_h2d(void *dst, const void *src, std::size_t bytes);
void matlab_gpu_cuda_d2h(void *dst, const void *src, std::size_t bytes);
int matlab_gpu_cuda_jit_compile(const char *src, const char *name,
                                void **out_fn);
int matlab_gpu_cuda_dispatch(void *fn, void **args, int grid_size);
matlab_mat *matlab_gpu_cuda_gemm_double(matlab_mat *A, matlab_mat *B);
const char *matlab_gpu_cuda_device_name(void);
/* High-level dispatcher (runtime/gpu/runtime_gpu.cpp). */
matlab_mat *matlab_gpu_gemm(matlab_mat *A, matlab_mat *B);
}

static int g_fail = 0;
static void check(bool ok, const char *what, double err) {
  std::printf("  %-32s %s (err=%.3g)\n", what, ok ? "PASS" : "FAIL", err);
  if (!ok) g_fail = 1;
}

/* Fill a freshly-alloc'd matrix row-major from a literal. */
static matlab_mat *makeMat(int m, int n, const double *vals) {
  matlab_mat *M = mat_alloc(m, n);
  for (int i = 0; i < m * n; ++i) M->data[i] = vals[i];
  return M;
}

static double gemmErr(const matlab_mat *C, const double *ref, int m, int n) {
  double e = 0.0;
  for (int i = 0; i < m * n; ++i)
    e = std::fmax(e, std::fabs(C->data[i] - ref[i]));
  return e;
}

int main() {
  std::printf("cuda smoke: device = %s\n", matlab_gpu_cuda_device_name());

  // ---- 1. low-level cuBLAS Dgemm: A=[[1,2],[3,4]] B=[[5,6],[7,8]] ----
  //      C = [[19,22],[43,50]]
  double a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};
  double ref[4] = {19, 22, 43, 50};
  matlab_mat *A = makeMat(2, 2, a), *B = makeMat(2, 2, b);
  matlab_mat *C = matlab_gpu_cuda_gemm_double(A, B);
  if (!C) {
    std::printf("  cublas Dgemm                     FAIL (returned null)\n");
    return 1;
  }
  check(gemmErr(C, ref, 2, 2) < 1e-9, "cublas Dgemm (fp64)",
        gemmErr(C, ref, 2, 2));

  // ---- 2. NVRTC AXPY: out = a*x + y, a=2, x=i, y=100 ----
  const char *src =
      "extern \"C\" __global__ void axpy(double a, const double* x,\n"
      "    const double* y, double* o, int n){\n"
      "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
      "  if (i < n) o[i] = a*x[i] + y[i];\n"
      "}\n";
  void *fn = nullptr;
  int rc = matlab_gpu_cuda_jit_compile(src, "axpy", &fn);
  if (rc != 0 || !fn) {
    std::printf("  nvrtc axpy compile               FAIL (rc=%d)\n", rc);
    return 1;
  }
  const int n = 8;
  double hx[n], hy[n], ho[n];
  for (int i = 0; i < n; ++i) {
    hx[i] = i;
    hy[i] = 100.0;
  }
  void *dx = matlab_gpu_cuda_alloc(n * sizeof(double));
  void *dy = matlab_gpu_cuda_alloc(n * sizeof(double));
  void *dout = matlab_gpu_cuda_alloc(n * sizeof(double));
  matlab_gpu_cuda_h2d(dx, hx, n * sizeof(double));
  matlab_gpu_cuda_h2d(dy, hy, n * sizeof(double));
  double aa = 2.0;
  int nn = n;
  void *args[] = {&aa, &dx, &dy, &dout, &nn};
  rc = matlab_gpu_cuda_dispatch(fn, args, n);
  matlab_gpu_cuda_d2h(ho, dout, n * sizeof(double));
  double axpyErr = 0.0;
  for (int i = 0; i < n; ++i)
    axpyErr = std::fmax(axpyErr, std::fabs(ho[i] - (2.0 * i + 100.0)));
  check(rc == 0 && axpyErr < 1e-9, "nvrtc axpy launch", axpyErr);
  matlab_gpu_cuda_free(dx);
  matlab_gpu_cuda_free(dy);
  matlab_gpu_cuda_free(dout);

  // ---- 3. high-level matlab_gpu_gemm dispatcher (target=cuda) ----
  //      4x4: A = identity, B = counter; expect C == B.
  double a4[16], b4[16];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      a4[i * 4 + j] = (i == j) ? 1.0 : 0.0;
      b4[i * 4 + j] = i * 4 + j + 1;
    }
  matlab_mat *A4 = makeMat(4, 4, a4), *B4 = makeMat(4, 4, b4);
  matlab_mat *C4 = matlab_gpu_gemm(A4, B4);
  double dispErr = C4 ? gemmErr(C4, b4, 4, 4) : 1e9;
  check(C4 && dispErr < 1e-9, "matlab_gpu_gemm dispatch", dispErr);

  std::printf("cuda smoke: %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail;
}
