/* runtime/toolbox/gpu/runtime_gpu_helpers.cpp — GPU Coder design-pattern
 * runtime entries.  T5 of docs/gpu_coder_roadmap.md.
 *
 * These are the C runtime impls of the MathWorks GPU Coder helpers:
 *   gpucoder.reduce               → matlab_gpucoder_reduce
 *   gpucoder.matrixMatrixKernel   → matlab_gpucoder_matmatkernel
 *   stencilfun                    → matlab_stencilfun
 *   gpucoder.sort                 → matlab_gpucoder_sort
 *
 * v1 is the CPU-debug lane reference impl — runs on the host with the
 * user's function-handle invoked per inner cell.  Tier-2/3/4 backends
 * will pattern-match each call in their respective EmitMetal/EmitCUDA/
 * EmitOpenCL pass and swap in tiled-kernel / shared-memory tree-reduce
 * versions; the runtime function remains the host fallback when the
 * call shape doesn't match the kernel template.
 *
 * Function-handle ABI: `fn_p` is a void* to the lowered anon function.
 * LowerAnonCalls retypes the indirect call signature per call site so
 * the typed call below matches.
 */

#include "runtime_internal.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {

/* mat_alloc lives in matlab_runtime.cpp — forward-declare here. */
matlab_mat *mat_alloc(int64_t m, int64_t n);

typedef double (*binary_fn_t)(double, double);
typedef double (*stencil_fn_t)(matlab_mat *);

/* matlab_gpucoder_reduce(X, fn) — fold X to a scalar via binary fn.
 * fn is associative + commutative (the only shape the parallel
 * tree-reduce can leverage; on the CPU lane the order is left-to-
 * right but the result matches mathematically). */
double matlab_gpucoder_reduce(matlab_mat *X, void *fn_p) {
  if (!X || !fn_p) return 0.0;
  binary_fn_t fn = reinterpret_cast<binary_fn_t>(fn_p);
  std::size_t n = static_cast<std::size_t>(X->rows) *
                  static_cast<std::size_t>(X->cols);
  if (n == 0) return 0.0;
  double acc = X->data[0];
  for (std::size_t i = 1; i < n; ++i) acc = fn(acc, X->data[i]);
  return acc;
}

/* matlab_gpucoder_matmatkernel(op_p, A, B) — matrix-matrix kernel where
 * inner accumulation uses `op` instead of `+`.  C(i,j) = reduce(@op,
 * A(i,:) .* B(:,j)).  Returns a fresh ma×nb matlab_mat. */
matlab_mat *matlab_gpucoder_matmatkernel(void *op_p, matlab_mat *A,
                                         matlab_mat *B) {
  if (!A || !B || !op_p) return nullptr;
  binary_fn_t op = reinterpret_cast<binary_fn_t>(op_p);
  int64_t ma = A->rows, ka = A->cols;
  int64_t kb = B->rows, nb = B->cols;
  if (ka != kb) {
    std::fprintf(stderr,
        "matlab_gpucoder_matmatkernel: inner dimensions must match "
        "(A is %lldx%lld, B is %lldx%lld)\n",
        static_cast<long long>(ma), static_cast<long long>(ka),
        static_cast<long long>(kb), static_cast<long long>(nb));
    return nullptr;
  }
  matlab_mat *C = mat_alloc(ma, nb);
  for (int64_t i = 0; i < ma; ++i) {
    for (int64_t j = 0; j < nb; ++j) {
      double a0 = A->data[i * ka + 0];
      double b0 = B->data[0 * nb + j];
      double acc = a0 * b0;
      for (int64_t k = 1; k < ka; ++k) {
        double ax = A->data[i * ka + k];
        double bx = B->data[k * nb + j];
        acc = op(acc, ax * bx);
      }
      C->data[i * nb + j] = acc;
    }
  }
  return C;
}

/* matlab_stencilfun(f_p, A, sz_p) — apply f over each m×n window of A.
 * sz_p must be a matlab_mat holding [wm wn] (or scalar wm meaning
 * square window).  Output is "valid" boundary (no padding); size
 * (ma-wm+1) × (na-wn+1). */
matlab_mat *matlab_stencilfun(void *f_p, matlab_mat *A, matlab_mat *sz_p) {
  if (!A || !sz_p || !f_p) return nullptr;
  stencil_fn_t f = reinterpret_cast<stencil_fn_t>(f_p);
  int64_t sz_n = sz_p->rows * sz_p->cols;
  int64_t wm = (sz_n >= 1) ? static_cast<int64_t>(sz_p->data[0]) : 1;
  int64_t wn = (sz_n >= 2) ? static_cast<int64_t>(sz_p->data[1]) : wm;
  int64_t ma = A->rows, na = A->cols;
  int64_t om = ma - wm + 1;
  int64_t on = na - wn + 1;
  if (om <= 0 || on <= 0) return mat_alloc(0, 0);
  matlab_mat *Y = mat_alloc(om, on);
  for (int64_t i = 0; i < om; ++i) {
    for (int64_t j = 0; j < on; ++j) {
      matlab_mat *W = mat_alloc(wm, wn);
      for (int64_t ii = 0; ii < wm; ++ii) {
        for (int64_t jj = 0; jj < wn; ++jj) {
          W->data[ii * wn + jj] = A->data[(i + ii) * na + (j + jj)];
        }
      }
      Y->data[i * on + j] = f(W);
      /* W is registry-tracked; auto-freed on program teardown.  For a
       * tight per-cell allocation this is wasteful but simple — Tier-2+
       * emit lanes inline the window into shared/threadgroup memory
       * instead, never hitting this path. */
    }
  }
  return Y;
}

/* matlab_gpucoder_sort(X) — sort the flat view of X ascending.  v1 is
 * a copy + std::qsort; backends swap in CUB radix-sort (CUDA) /
 * bitonic-sort (Metal + OpenCL). */
static int cmp_double(const void *a, const void *b) {
  double da = *static_cast<const double *>(a);
  double db = *static_cast<const double *>(b);
  if (da < db) return -1;
  if (da > db) return 1;
  return 0;
}

matlab_mat *matlab_gpucoder_sort(matlab_mat *X) {
  if (!X) return nullptr;
  matlab_mat *Y = mat_alloc(X->rows, X->cols);
  std::size_t n = static_cast<std::size_t>(X->rows) *
                  static_cast<std::size_t>(X->cols);
  std::memcpy(Y->data, X->data, n * sizeof(double));
  if (n > 1) std::qsort(Y->data, n, sizeof(double), cmp_double);
  return Y;
}

}  /* extern "C" */
