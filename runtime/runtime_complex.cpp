/* runtime_complex.cpp — complex matrix descriptor + FFT family.
 *
 * Extracted from runtime/matlab_runtime.cpp in Phase 2.5 of the runtime
 * port (docs/port_runtime_2_cpp.md). The body is byte-identical to the
 * original "Complex numbers" block; only the surrounding wrappers are
 * new. mat_c_alloc, the matlab_mat_c layout, and the magic constants
 * are exposed via runtime_internal.h so callers in matlab_runtime.cpp
 * (matlab_roots, the polymorphic real-side dispatchers) and
 * runtime_debug.cpp (matlab_dbg_mat_c_*) can find them.
 *
 * All matlab_* and matlab_dbg_* exports are wrapped in extern "C" so
 * the JIT-emitted code resolves them by C name unchanged.
 */

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "runtime_internal.h"

#include <vector>

extern "C" {

/*===========================================================================
 * Complex numbers.
 *
 * Representation mirrors the real matrix: a heap descriptor with separate
 * real / imaginary f64 planes (row-major, rows*cols entries each). Separate
 * planes (rather than interleaved pairs) keep the existing SIMD-friendly
 * contiguous-loop shape on the real-only fast path and let us share scalar
 * math kernels between real and complex matrices.
 *
 * Scalars are 1x1 matrices — same trick the real runtime uses so the
 * compiler only has to plumb one MLIR type (`!llvm.ptr`). `matlab.const_complex`
 * from the frontend lowers to `matlab_complex_scalar(re, im)` which
 * allocates a 1x1 matlab_mat_c.
 *
 * Interop: a `matlab_mat_c` never mixes with `matlab_mat` at runtime. The
 * Lowerer decides, per binop, whether to route to the real or complex
 * variant based on Sema's propagated Dtype. When the operands disagree
 * (e.g. `real + 2i`), the Lowerer promotes the real operand via
 * matlab_mat_c_from_real before the complex op.
 *===========================================================================*/

/* matlab_mat_c layout lives in runtime_internal.h (Phase-2 split).
 * MATLAB_MAT_C_MAGIC + mat_is_complex() also live there so the
 * polymorphic real-side entries (matlab_disp_mat, etc.) can
 * discriminate the layout without pulling in the full complex runtime
 * upfront. */
matlab_mat_c *mat_c_alloc(int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    matlab_mat_c *A = (matlab_mat_c *)calloc(1, sizeof(matlab_mat_c));
    A->magic = MATLAB_MAT_C_MAGIC;
    A->rows = m; A->cols = n;
    A->re = (double *)calloc((size_t)(m * n + 1), sizeof(double));
    A->im = (double *)calloc((size_t)(m * n + 1), sizeof(double));
    return A;
}

/* DAP-side accessors for complex matrices. Forward-declared in the
 * matlab_dbg section above; defined here where the matlab_mat_c
 * layout is fully visible. Each defends against being called with
 * a non-complex descriptor by re-checking the magic byte. */
int64_t matlab_dbg_mat_c_rows(const matlab_mat_c *m) {
    if (!m || !mat_is_complex(m)) return 0;
    return m->rows;
}
int64_t matlab_dbg_mat_c_cols(const matlab_mat_c *m) {
    if (!m || !mat_is_complex(m)) return 0;
    return m->cols;
}
double matlab_dbg_mat_c_re(const matlab_mat_c *m, int64_t i, int64_t j) {
    if (!m || !mat_is_complex(m) || !m->re) return 0.0;
    if (i < 1 || j < 1) return 0.0;
    if (i > m->rows || j > m->cols) return 0.0;
    return m->re[(i - 1) * m->cols + (j - 1)];
}
double matlab_dbg_mat_c_im(const matlab_mat_c *m, int64_t i, int64_t j) {
    if (!m || !mat_is_complex(m) || !m->im) return 0.0;
    if (i < 1 || j < 1) return 0.0;
    if (i > m->rows || j > m->cols) return 0.0;
    return m->im[(i - 1) * m->cols + (j - 1)];
}

/* Constructors ----------------------------------------------------------*/

matlab_mat_c *matlab_complex_scalar(double re, double im) {
    matlab_mat_c *A = mat_c_alloc(1, 1);
    A->re[0] = re; A->im[0] = im;
    return A;
}

/* Promote a real matrix to complex (zero imag). Used at binop sites where
 * one operand is complex and the other is real. Allocates a fresh
 * descriptor; the caller's real matrix is unchanged. */
matlab_mat_c *matlab_mat_c_from_real(matlab_mat *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
    return C;
}

matlab_mat_c *matlab_mat_c_from_buf(const double *re, const double *im,
                                     double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    matlab_mat_c *A = mat_c_alloc(rm, cn);
    memcpy(A->re, re, (size_t)(rm * cn) * sizeof(double));
    if (im) memcpy(A->im, im, (size_t)(rm * cn) * sizeof(double));
    return A;
}

/* Unary ------------------------------------------------------------------*/

/* conj / real / imag / angle / abs are polymorphic: they accept either a
 * real matlab_mat* or a complex matlab_mat_c*. The mat_is_complex() check
 * distinguishes the two at the ABI boundary. On the real path, imag()
 * returns zeros, conj() is an identity copy, etc. */
matlab_mat_c *matlab_conj_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
        memcpy(C->re, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
        return C;  /* imag is already zeroed by calloc */
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) {
        C->re[k] =  A->re[k];
        C->im[k] = -A->im[k];
    }
    return C;
}

matlab_mat_c *matlab_neg_c(matlab_mat_c *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) {
        C->re[k] = -A->re[k];
        C->im[k] = -A->im[k];
    }
    return C;
}

/* fftshift / ifftshift — circular shift that moves DC to the centre
 * (fftshift) or back (ifftshift). Polymorphic on real/complex inputs;
 * always returns a complex descriptor so chained spectra survive.
 * Shift amount per axis: floor((d+1)/2) forward, floor(d/2) inverse.
 * On a vector axis the singleton dim is left alone. */
static matlab_mat_c *fftshift_impl(void *Aptr, int forward) {
    if (!Aptr) return mat_c_alloc(0, 0);
    int complex_in = mat_is_complex(Aptr);
    int64_t m, n;
    const double *re_in, *im_in;
    if (complex_in) {
        matlab_mat_c *A = (matlab_mat_c *)Aptr;
        m = A->rows; n = A->cols; re_in = A->re; im_in = A->im;
    } else {
        matlab_mat *A = (matlab_mat *)Aptr;
        m = A->rows; n = A->cols; re_in = A->data; im_in = NULL;
    }
    matlab_mat_c *C = mat_c_alloc(m, n);
    if (m == 0 || n == 0) return C;
    int64_t sr = (m == 1) ? 0 : (forward ? (m + 1) / 2 : m / 2);
    int64_t sc = (n == 1) ? 0 : (forward ? (n + 1) / 2 : n / 2);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) {
            int64_t ii = (i + sr) % m;
            int64_t jj = (j + sc) % n;
            C->re[ii * n + jj] = re_in[i * n + j];
            if (im_in) C->im[ii * n + jj] = im_in[i * n + j];
        }
    return C;
}

matlab_mat_c *matlab_fftshift_c(void *A)  { return fftshift_impl(A, 1); }
matlab_mat_c *matlab_ifftshift_c(void *A) { return fftshift_impl(A, 0); }

/* Tier-2 roots — Durand-Kerner (Weierstrass) iteration. Simultaneously
 * refines n initial complex guesses on a circle, converging to all n
 * roots of the polynomial. Returns a complex column vector.
 *
 * p is in MATLAB's highest-power-first order. Leading zeros are
 * stripped (polynomial degree drops accordingly). Trailing zeros
 * become explicit roots at the origin. */
static void cmul_(double ar, double ai, double br, double bi,
                  double *rr, double *ri) {
    *rr = ar * br - ai * bi;
    *ri = ar * bi + ai * br;
}
static void cdiv_(double ar, double ai, double br, double bi,
                  double *rr, double *ri) {
    double d = br * br + bi * bi;
    *rr = (ar * br + ai * bi) / d;
    *ri = (ai * br - ar * bi) / d;
}
matlab_mat_c *matlab_roots(matlab_mat *p) {
    if (!p) return mat_c_alloc(0, 0);
    int64_t np = p->rows * p->cols;
    int64_t lead = 0;
    while (lead < np && p->data[lead] == 0.0) lead++;
    if (lead == np) return mat_c_alloc(0, 0);
    int64_t deg = (np - 1) - lead;
    if (deg == 0) return mat_c_alloc(0, 0);
    int64_t trail = 0;
    while (trail < deg && p->data[np - 1 - trail] == 0.0) trail++;
    int64_t deg_eff = deg - trail;
    matlab_mat_c *R = mat_c_alloc(deg, 1);
    for (int64_t i = 0; i < trail; ++i) {
        R->re[deg_eff + i] = 0.0; R->im[deg_eff + i] = 0.0;
    }
    if (deg_eff == 0) return R;
    /* Phase-4 RAII: q, zr, zi go from manual malloc/free to std::vector. */
    int64_t qn = deg_eff + 1;
    std::vector<double> q(qn);
    double lead_c = p->data[lead];
    for (int64_t i = 0; i < qn; ++i) q[i] = p->data[lead + i] / lead_c;
    std::vector<double> zr(deg_eff), zi(deg_eff);
    double cur_r = 1.0, cur_i = 0.0;
    for (int64_t k = 0; k < deg_eff; ++k) {
        zr[k] = cur_r; zi[k] = cur_i;
        double nr, ni; cmul_(cur_r, cur_i, 0.4, 0.9, &nr, &ni);
        cur_r = nr; cur_i = ni;
    }
    for (int iter = 0; iter < 200; ++iter) {
        double max_delta = 0.0;
        for (int64_t k = 0; k < deg_eff; ++k) {
            double pr = q[0], pi = 0.0;
            for (int64_t j = 1; j < qn; ++j) {
                double nr, ni;
                cmul_(pr, pi, zr[k], zi[k], &nr, &ni);
                pr = nr + q[j]; pi = ni;
            }
            double dr = 1.0, di = 0.0;
            for (int64_t j = 0; j < deg_eff; ++j) {
                if (j == k) continue;
                double nr, ni;
                cmul_(dr, di, zr[k] - zr[j], zi[k] - zi[j], &nr, &ni);
                dr = nr; di = ni;
            }
            double sr, si;
            cdiv_(pr, pi, dr, di, &sr, &si);
            zr[k] -= sr; zi[k] -= si;
            double mag = sqrt(sr * sr + si * si);
            if (mag > max_delta) max_delta = mag;
        }
        if (max_delta < 1e-12) break;
    }
    for (int64_t k = 0; k < deg_eff; ++k) {
        R->re[k] = zr[k]; R->im[k] = zi[k];
    }
    return R;
}

matlab_mat *matlab_real_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        /* real(A) on a real A is just A (copy to avoid aliasing). */
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        memcpy(R->data, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
        return R;
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    memcpy(R->data, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    return R;
}

matlab_mat *matlab_imag_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        matlab_mat *A = (matlab_mat *)Aptr;
        return mat_alloc(A->rows, A->cols);  /* zeros */
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    memcpy(R->data, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    return R;
}

matlab_mat *matlab_angle_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        /* angle(a) for real a: 0 if a >= 0, pi if a < 0. */
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        int64_t n = A->rows * A->cols;
        for (int64_t k = 0; k < n; ++k)
            R->data[k] = A->data[k] < 0.0 ? M_PI : 0.0;
        return R;
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) R->data[k] = atan2(A->im[k], A->re[k]);
    return R;
}

matlab_mat *matlab_abs_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        int64_t n = A->rows * A->cols;
        for (int64_t k = 0; k < n; ++k) R->data[k] = fabs(A->data[k]);
        return R;
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k)
        R->data[k] = hypot(A->re[k], A->im[k]);
    return R;
}

/* Element-wise binary ---------------------------------------------------*/

/* Broadcast: 1x1 matches any shape. Returns the shape to use. 0 on
 * incompatible shapes. */
static int mat_c_bcast(matlab_mat_c *A, matlab_mat_c *B,
                        int64_t *m_out, int64_t *n_out) {
    if (!A || !B) return 0;
    if (A->rows == B->rows && A->cols == B->cols) {
        *m_out = A->rows; *n_out = A->cols; return 1;
    }
    if (A->rows == 1 && A->cols == 1) {
        *m_out = B->rows; *n_out = B->cols; return 1;
    }
    if (B->rows == 1 && B->cols == 1) {
        *m_out = A->rows; *n_out = A->cols; return 1;
    }
    return 0;
}

#define MAT_C_BINARY(NAME, OP_RE, OP_IM) \
    matlab_mat_c *matlab_##NAME##_cc(matlab_mat_c *A, matlab_mat_c *B) { \
        int64_t m, n; \
        if (!mat_c_bcast(A, B, &m, &n)) return mat_c_alloc(0, 0); \
        matlab_mat_c *C = mat_c_alloc(m, n); \
        int a_scalar = A->rows == 1 && A->cols == 1; \
        int b_scalar = B->rows == 1 && B->cols == 1; \
        for (int64_t k = 0; k < m * n; ++k) { \
            double ar = a_scalar ? A->re[0] : A->re[k]; \
            double ai = a_scalar ? A->im[0] : A->im[k]; \
            double br = b_scalar ? B->re[0] : B->re[k]; \
            double bi = b_scalar ? B->im[0] : B->im[k]; \
            C->re[k] = (OP_RE); \
            C->im[k] = (OP_IM); \
            (void)ar; (void)ai; (void)br; (void)bi; \
        } \
        return C; \
    }

MAT_C_BINARY(add, ar + br, ai + bi)
MAT_C_BINARY(sub, ar - br, ai - bi)
MAT_C_BINARY(emul, ar*br - ai*bi, ar*bi + ai*br)
/* Element-wise divide: (a + bi) / (c + di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2) */
matlab_mat_c *matlab_ediv_cc(matlab_mat_c *A, matlab_mat_c *B) {
    int64_t m, n;
    if (!mat_c_bcast(A, B, &m, &n)) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(m, n);
    int a_scalar = A->rows == 1 && A->cols == 1;
    int b_scalar = B->rows == 1 && B->cols == 1;
    for (int64_t k = 0; k < m * n; ++k) {
        double ar = a_scalar ? A->re[0] : A->re[k];
        double ai = a_scalar ? A->im[0] : A->im[k];
        double br = b_scalar ? B->re[0] : B->re[k];
        double bi = b_scalar ? B->im[0] : B->im[k];
        double denom = br*br + bi*bi;
        C->re[k] = (ar*br + ai*bi) / denom;
        C->im[k] = (ai*br - ar*bi) / denom;
    }
    return C;
}

#undef MAT_C_BINARY

/* Matrix multiply: C(i,j) = sum_k A(i,k) * B(k,j). Naive O(m*n*p). */
matlab_mat_c *matlab_matmul_cc(matlab_mat_c *A, matlab_mat_c *B) {
    if (!A || !B || A->cols != B->rows) return mat_c_alloc(0, 0);
    int64_t m = A->rows, p = A->cols, n = B->cols;
    matlab_mat_c *C = mat_c_alloc(m, n);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double sr = 0, si = 0;
            for (int64_t k = 0; k < p; ++k) {
                double ar = A->re[i*p + k], ai = A->im[i*p + k];
                double br = B->re[k*n + j], bi = B->im[k*n + j];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            C->re[i*n + j] = sr;
            C->im[i*n + j] = si;
        }
    }
    return C;
}

/* Transpose variants: `.'` keeps entries as-is; `'` (ctranspose) also
 * conjugates. Both swap dims. */
matlab_mat_c *matlab_transpose_c(matlab_mat_c *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i)
        for (int64_t j = 0; j < A->cols; ++j) {
            C->re[j*A->rows + i] = A->re[i*A->cols + j];
            C->im[j*A->rows + i] = A->im[i*A->cols + j];
        }
    return C;
}

matlab_mat_c *matlab_ctranspose_c(matlab_mat_c *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i)
        for (int64_t j = 0; j < A->cols; ++j) {
            C->re[j*A->rows + i] =  A->re[i*A->cols + j];
            C->im[j*A->rows + i] = -A->im[i*A->cols + j];
        }
    return C;
}

/* Display ---------------------------------------------------------------*/

static void disp_complex_scalar(double re, double im) {
    /* MATLAB-ish: "re + imi" or "re - imi"; drop parts that are exactly 0
     * except when both are zero (print "0"). */
    if (im == 0.0 && re == 0.0) { printf("0\n"); return; }
    if (im == 0.0) { printf("%g\n", re); return; }
    if (re == 0.0) {
        printf("%gi\n", im);
        return;
    }
    if (im < 0.0) printf("%g - %gi\n", re, -im);
    else          printf("%g + %gi\n", re, im);
}

void matlab_disp_mat_c(matlab_mat_c *A) {
    if (!A) return;
    if (A->rows == 0 || A->cols == 0) return;
    pthread_mutex_lock(&matlab_io_mutex);
    if (A->rows == 1 && A->cols == 1) {
        disp_complex_scalar(A->re[0], A->im[0]);
        pthread_mutex_unlock(&matlab_io_mutex);
        return;
    }
    for (int64_t i = 0; i < A->rows; ++i) {
        for (int64_t j = 0; j < A->cols; ++j) {
            double re = A->re[i*A->cols + j];
            double im = A->im[i*A->cols + j];
            if (j) printf("   ");
            if (im >= 0.0) printf("%9.4g + %.4gi", re, im);
            else           printf("%9.4g - %.4gi", re, -im);
        }
        printf("\n");
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* Shape queries / rows / cols (DAP formatter reads these via the real
 * runtime; duplicate for complex). */
int64_t matlab_mat_c_rows(matlab_mat_c *A) { return A ? A->rows : 0; }
int64_t matlab_mat_c_cols(matlab_mat_c *A) { return A ? A->cols : 0; }

/*===========================================================================
 * FFT — pure-C Cooley-Tukey.
 *
 * Two code paths:
 *   - Power-of-two N: standard iterative radix-2 DIT with bit reversal.
 *     O(N log N), exact result (modulo rounding) for all power-of-2 N.
 *   - General N: Bluestein's algorithm — expresses DFT(x) as a convolution
 *     (which is itself radix-2 FFT'd at the next power of 2 >= 2N-1). O(N
 *     log N) asymptotically; a few-× slower than direct radix-2 but no
 *     dependency on N's factorization.
 *
 * Both operate on a matlab_mat_c. Input is either (1, N) or (N, 1) for a
 * 1-D vector; matrix inputs get fft applied along the first non-singleton
 * dim (MATLAB convention — we only ship fft-along-rows-or-single-column
 * here, matching typical use).
 *===========================================================================*/

static int is_power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* In-place radix-2 DIT FFT on the size-N arrays re[], im[]. inverse=1
 * applies the conjugate twiddle (caller is responsible for the 1/N scale
 * that MATLAB's ifft applies). Pre: N is a power of 2 and N >= 1. */
static void fft_radix2_inplace(double *re, double *im, int64_t N, int inverse) {
    if (N < 2) return;
    /* Bit-reversal permutation. */
    int64_t j = 0;
    for (int64_t i = 1; i < N; ++i) {
        int64_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    /* Iterative butterflies. */
    double sign = inverse ? 1.0 : -1.0;
    for (int64_t len = 2; len <= N; len <<= 1) {
        double ang = sign * 2.0 * M_PI / (double)len;
        double wlen_r = cos(ang), wlen_i = sin(ang);
        for (int64_t i = 0; i < N; i += len) {
            double w_r = 1.0, w_i = 0.0;
            int64_t half = len >> 1;
            for (int64_t k = 0; k < half; ++k) {
                double u_r = re[i + k];
                double u_i = im[i + k];
                double v_r = re[i + k + half] * w_r - im[i + k + half] * w_i;
                double v_i = re[i + k + half] * w_i + im[i + k + half] * w_r;
                re[i + k] = u_r + v_r;
                im[i + k] = u_i + v_i;
                re[i + k + half] = u_r - v_r;
                im[i + k + half] = u_i - v_i;
                double nw_r = w_r * wlen_r - w_i * wlen_i;
                double nw_i = w_r * wlen_i + w_i * wlen_r;
                w_r = nw_r; w_i = nw_i;
            }
        }
    }
}

/* Bluestein's algorithm for arbitrary N. Builds chirp a[n] = x[n] * w^(-n^2/2)
 * and convolves with b[n] = w^(n^2/2), then multiplies by a[n]^-1 again.
 * We use a power-of-2 FFT of size M >= 2N-1 for the convolution. inverse=1
 * applies the conjugate twiddle (caller handles the 1/N normalization). */
static void fft_bluestein(double *re, double *im, int64_t N, int inverse) {
    if (N < 2) return;
    /* M = next power of 2 >= 2N - 1. */
    int64_t M = 1;
    while (M < 2 * N - 1) M <<= 1;

    double *chirp_r = (double *)calloc((size_t)N, sizeof(double));
    double *chirp_i = (double *)calloc((size_t)N, sizeof(double));
    double *a_r = (double *)calloc((size_t)M, sizeof(double));
    double *a_i = (double *)calloc((size_t)M, sizeof(double));
    double *b_r = (double *)calloc((size_t)M, sizeof(double));
    double *b_i = (double *)calloc((size_t)M, sizeof(double));

    double sign = inverse ? 1.0 : -1.0;
    /* chirp[n] = exp(sign * i * pi * n^2 / N). Precompute for n in [0, N).
     * Use (n*n) mod (2N) to keep the argument small for large N. */
    for (int64_t n = 0; n < N; ++n) {
        /* ang = sign * pi * (n^2 mod (2N)) / N */
        int64_t nn = (n * n) % (2 * N);
        double ang = sign * M_PI * (double)nn / (double)N;
        chirp_r[n] = cos(ang);
        chirp_i[n] = sin(ang);
    }
    /* a[n] = x[n] * conj(chirp[n]) */
    for (int64_t n = 0; n < N; ++n) {
        a_r[n] = re[n] * chirp_r[n] + im[n] * chirp_i[n];
        a_i[n] = im[n] * chirp_r[n] - re[n] * chirp_i[n];
    }
    /* b[n] = chirp[n] for 0 <= n < N, and b[M-n] = chirp[n] for 1 <= n < N.
     * Zero elsewhere. This is the "symmetric" Bluestein kernel shape. */
    b_r[0] = chirp_r[0]; b_i[0] = chirp_i[0];
    for (int64_t n = 1; n < N; ++n) {
        b_r[n] = chirp_r[n]; b_i[n] = chirp_i[n];
        b_r[M - n] = chirp_r[n]; b_i[M - n] = chirp_i[n];
    }
    /* Convolution via FFT(a) * FFT(b), then IFFT. */
    fft_radix2_inplace(a_r, a_i, M, 0);
    fft_radix2_inplace(b_r, b_i, M, 0);
    for (int64_t k = 0; k < M; ++k) {
        double pr = a_r[k] * b_r[k] - a_i[k] * b_i[k];
        double pi = a_r[k] * b_i[k] + a_i[k] * b_r[k];
        a_r[k] = pr; a_i[k] = pi;
    }
    fft_radix2_inplace(a_r, a_i, M, 1);
    /* Scale the inverse-FFT result by 1/M and multiply by conj(chirp). */
    for (int64_t n = 0; n < N; ++n) {
        double yr = a_r[n] / (double)M;
        double yi = a_i[n] / (double)M;
        re[n] = yr * chirp_r[n] + yi * chirp_i[n];
        im[n] = yi * chirp_r[n] - yr * chirp_i[n];
    }
    free(chirp_r); free(chirp_i);
    free(a_r); free(a_i);
    free(b_r); free(b_i);
}

/* Apply 1-D FFT to each column of the caller's matrix in place. */
static void fft_columns_inplace(double *re, double *im,
                                 int64_t rows, int64_t cols, int inverse) {
    double *col_r = (double *)malloc((size_t)rows * sizeof(double));
    double *col_i = (double *)malloc((size_t)rows * sizeof(double));
    for (int64_t c = 0; c < cols; ++c) {
        for (int64_t r = 0; r < rows; ++r) {
            col_r[r] = re[r * cols + c];
            col_i[r] = im[r * cols + c];
        }
        if (is_power_of_two(rows))
            fft_radix2_inplace(col_r, col_i, rows, inverse);
        else
            fft_bluestein(col_r, col_i, rows, inverse);
        for (int64_t r = 0; r < rows; ++r) {
            re[r * cols + c] = col_r[r];
            im[r * cols + c] = col_i[r];
        }
    }
    free(col_r); free(col_i);
}

/* Apply 1-D FFT to each row. */
static void fft_rows_inplace(double *re, double *im,
                              int64_t rows, int64_t cols, int inverse) {
    for (int64_t r = 0; r < rows; ++r) {
        if (is_power_of_two(cols))
            fft_radix2_inplace(re + r * cols, im + r * cols, cols, inverse);
        else
            fft_bluestein(re + r * cols, im + r * cols, cols, inverse);
    }
}

/* Public fft / ifft entries. Take an opaque ptr — either matlab_mat*
 * (real) or matlab_mat_c* (complex). mat_is_complex() discriminates
 * by the magic marker at byte 0 and the real path is auto-promoted
 * via matlab_mat_c_from_real. MATLAB's dim rule: vectors are FFT'd
 * along their non-singleton dim; matrices along columns (dim 1). */
matlab_mat_c *matlab_fft_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    if (A->rows == 1) {
        fft_rows_inplace(C->re, C->im, C->rows, C->cols, 0);
    } else {
        fft_columns_inplace(C->re, C->im, C->rows, C->cols, 0);
    }
    return C;
}

matlab_mat_c *matlab_ifft_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    int64_t n = (A->rows == 1) ? A->cols : A->rows;
    if (A->rows == 1) {
        fft_rows_inplace(C->re, C->im, C->rows, C->cols, 1);
    } else {
        fft_columns_inplace(C->re, C->im, C->rows, C->cols, 1);
    }
    /* MATLAB ifft applies the 1/N scale. */
    double inv = 1.0 / (double)n;
    int64_t total = C->rows * C->cols;
    for (int64_t k = 0; k < total; ++k) {
        C->re[k] *= inv;
        C->im[k] *= inv;
    }
    return C;
}

/* 2-D variants: FFT along rows then columns (or vice versa — order
 * doesn't matter for a separable transform). */
matlab_mat_c *matlab_fft2_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    fft_rows_inplace(C->re, C->im, C->rows, C->cols, 0);
    fft_columns_inplace(C->re, C->im, C->rows, C->cols, 0);
    return C;
}

matlab_mat_c *matlab_ifft2_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    fft_rows_inplace(C->re, C->im, C->rows, C->cols, 1);
    fft_columns_inplace(C->re, C->im, C->rows, C->cols, 1);
    double inv = 1.0 / (double)(C->rows * C->cols);
    int64_t total = C->rows * C->cols;
    for (int64_t k = 0; k < total; ++k) {
        C->re[k] *= inv;
        C->im[k] *= inv;
    }
    return C;
}
} /* extern "C" */
