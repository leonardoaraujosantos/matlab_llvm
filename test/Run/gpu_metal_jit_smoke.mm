// T2.C smoke test — JIT-compile + dispatch a Metal kernel through the
// runtime ABI, validating the MTLLibrary newLibraryWithSource: path
// works end-to-end against a real Apple GPU.
//
// Build: see test/Run/gpu_metal_jit_smoke.build.sh
//
// Apple GPUs don't support `double` in MSL (fp64 isn't part of the
// language).  The MSL source below uses `float` to match — this is
// also why MATLAB's GPU lane defaults to `single()` casts.  The host
// buffers are `float` to match the kernel ABI.
//
// Validates: front-end → MSL emit → in-process Metal compiler → MTLBuffer
// dispatch → result round-trip back to host.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

extern "C" {
int matlab_gpu_metal_jit_compile(const char *src, const char *name,
                                  void **out_pso);
int matlab_gpu_metal_dispatch(void *pso_handle, void **buffer_ptrs,
                               int buffer_count, int grid_size);
void *matlab_gpu_metal_alloc(unsigned long bytes);
void  matlab_gpu_metal_free(void *ptr);
void  matlab_gpu_metal_h2d(void *dst, const void *src, unsigned long bytes);
void  matlab_gpu_metal_d2h(void *dst, const void *src, unsigned long bytes);
const char *matlab_gpu_metal_device_name(void);
}

static const char *kAxpyMSL =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void axpy_kernel(\n"
    "    device float *out [[buffer(0)]],\n"
    "    constant float &a [[buffer(1)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "  float iv = float(tid) + 1.0f;\n"
    "  out[(int)(iv) - 1] = (a * iv);\n"
    "}\n";

int main() {
  std::printf("metal jit smoke: device = %s\n",
              matlab_gpu_metal_device_name());

  void *pso = nullptr;
  int rc = matlab_gpu_metal_jit_compile(kAxpyMSL, "axpy_kernel", &pso);
  if (rc != 0) {
    std::fprintf(stderr, "jit_compile failed: rc=%d\n", rc);
    return 1;
  }
  std::printf("metal jit smoke: kernel compiled\n");

  const int n = 8;
  void *bx = matlab_gpu_metal_alloc(n * sizeof(float));
  void *bA = matlab_gpu_metal_alloc(sizeof(float));
  float a = 2.5f;
  matlab_gpu_metal_h2d(bA, &a, sizeof(float));

  void *bufs[2] = {bx, bA};
  rc = matlab_gpu_metal_dispatch(pso, bufs, 2, n);
  if (rc != 0) {
    std::fprintf(stderr, "dispatch failed: rc=%d\n", rc);
    return 1;
  }

  float host[n];
  matlab_gpu_metal_d2h(host, bx, n * sizeof(float));

  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float expected = a * static_cast<float>(i + 1);
    float e = std::fabs(host[i] - expected);
    if (e > max_err) max_err = e;
  }
  std::printf("metal jit smoke: ok n=%d max_err=%g\n", n, max_err);

  matlab_gpu_metal_free(bx);
  matlab_gpu_metal_free(bA);
  return (max_err < 1e-6f) ? 0 : 2;
}
