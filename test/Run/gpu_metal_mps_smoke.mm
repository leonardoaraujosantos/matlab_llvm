// T2.D smoke test — MPS GEMM on Apple GPU.  Computes C = A * B for a
// small 4x4 matmul via MPSMatrixMultiplication and validates against
// host-side reference.  Proves the MPS GEMM dispatch path is live.

#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
void *matlab_gpu_metal_alloc(unsigned long bytes);
void  matlab_gpu_metal_free(void *ptr);
void  matlab_gpu_metal_h2d(void *dst, const void *src, unsigned long bytes);
void  matlab_gpu_metal_d2h(void *dst, const void *src, unsigned long bytes);
int   matlab_gpu_metal_gemm_f32(void *a_buf, void *b_buf, void *c_buf,
                                int M, int N, int K);
const char *matlab_gpu_metal_device_name(void);
}

int main() {
  std::printf("mps gemm smoke: device = %s\n",
              matlab_gpu_metal_device_name());

  const int M = 4, K = 4, N = 4;
  float A[M*K], B[K*N], C[M*N], Cref[M*N];

  /* Simple test matrices: A is the identity, B is a counter; expect
   * C = B exactly. */
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j)
      A[i*K + j] = (i == j) ? 1.0f : 0.0f;
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < N; ++j)
      B[i*N + j] = static_cast<float>(i * N + j + 1);

  /* Host reference: C_ref = A * B */
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[i*K + k] * B[k*N + j];
      Cref[i*N + j] = s;
    }

  void *bA = matlab_gpu_metal_alloc(M*K * sizeof(float));
  void *bB = matlab_gpu_metal_alloc(K*N * sizeof(float));
  void *bC = matlab_gpu_metal_alloc(M*N * sizeof(float));
  matlab_gpu_metal_h2d(bA, A, M*K * sizeof(float));
  matlab_gpu_metal_h2d(bB, B, K*N * sizeof(float));

  int rc = matlab_gpu_metal_gemm_f32(bA, bB, bC, M, N, K);
  if (rc != 0) {
    std::fprintf(stderr, "gemm failed: rc=%d\n", rc);
    return 1;
  }
  matlab_gpu_metal_d2h(C, bC, M*N * sizeof(float));

  float max_err = 0.0f;
  for (int i = 0; i < M*N; ++i) {
    float e = std::fabs(C[i] - Cref[i]);
    if (e > max_err) max_err = e;
  }
  std::printf("mps gemm smoke: ok M=%d N=%d K=%d max_err=%g\n",
              M, N, K, max_err);

  matlab_gpu_metal_free(bA);
  matlab_gpu_metal_free(bB);
  matlab_gpu_metal_free(bC);
  return (max_err < 1e-4f) ? 0 : 2;
}
