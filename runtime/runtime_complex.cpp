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

/* Forward decls of FFT entries defined later in this file. Used by
 * hilbert / periodogram / pwelch immediately below. */
matlab_mat_c *matlab_fft_c(void *Aptr);
matlab_mat_c *matlab_ifft_c(void *Aptr);

/*===========================================================================
 * Tier-2 SPT §3.1 nonparametric spectral estimation.
 *
 *   P = periodogram(x)              |FFT(x)|² / N, single-sided
 *   P = pwelch(x, win, noverlap)    Welch's averaged-modified-periodogram
 *
 * Single-output form. The 2-return [P, f] form would also give the
 * frequency vector — deferable. Default fs = 1; we use unit fs
 * throughout this slice. (Two-arg form `pwelch(x, win, noverlap, nfft, fs)`
 * is a Tier-2 follow-on — needs the multi-arg dispatch wired.)
 */
matlab_mat *matlab_periodogram(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N == 0) return mat_alloc(0, 0);
    matlab_mat_c *X = matlab_fft_c((void *)x);
    int64_t M = N / 2 + 1;
    matlab_mat *P = mat_alloc(M, 1);
    auto sq = [&](int k) {
        return X->re[k] * X->re[k] + X->im[k] * X->im[k];
    };
    P->data[0] = sq(0) / (double)N;
    int64_t mid_end = (N % 2 == 0) ? (M - 1) : M;
    for (int64_t k = 1; k < mid_end; ++k)
        P->data[k] = 2.0 * sq((int)k) / (double)N;
    if (N % 2 == 0)
        P->data[M - 1] = sq((int)(N / 2)) / (double)N;
    free(X->re); free(X->im); free(X);
    return P;
}

matlab_mat *matlab_pwelch(matlab_mat *x, matlab_mat *win, double noverlap_d) {
    if (!x || !win) return mat_alloc(0, 0);
    int64_t N  = x->rows * x->cols;
    int64_t L  = win->rows * win->cols;
    int     no = (int)noverlap_d;
    if (no < 0) no = 0;
    if (no >= L) no = L - 1;
    int step = (int)L - no;
    if (step < 1) step = 1;
    if (N < L) {
        matlab_mat *P = mat_alloc(L / 2 + 1, 1);
        return P;          /* not enough data — return zeros. */
    }
    int K = (int)((N - L) / step) + 1;
    int64_t M = L / 2 + 1;
    /* Window energy U = sum(win^2). */
    double U = 0.0;
    for (int64_t i = 0; i < L; ++i) U += win->data[i] * win->data[i];
    matlab_mat *Pxx = mat_alloc(M, 1);
    matlab_mat seg = { /*data*/ nullptr, /*rows*/ 1, /*cols*/ L };
    std::vector<double> xseg((size_t)L);
    seg.data = xseg.data();
    for (int s = 0; s < K; ++s) {
        for (int64_t i = 0; i < L; ++i)
            xseg[(size_t)i] = x->data[s * step + i] * win->data[i];
        matlab_mat_c *X = matlab_fft_c((void *)&seg);
        auto sq = [&](int k) {
            return X->re[k] * X->re[k] + X->im[k] * X->im[k];
        };
        Pxx->data[0] += sq(0);
        int64_t mid_end = (L % 2 == 0) ? (M - 1) : M;
        for (int64_t k = 1; k < mid_end; ++k)
            Pxx->data[k] += 2.0 * sq((int)k);
        if (L % 2 == 0)
            Pxx->data[M - 1] += sq((int)(L / 2));
        free(X->re); free(X->im); free(X);
    }
    double denom = (double)K * U;
    if (denom > 0.0)
        for (int64_t k = 0; k < M; ++k) Pxx->data[k] /= denom;
    return Pxx;
}

/*===========================================================================
 * Tier-2 SPT §3.4 transforms — hilbert + goertzel.
 *
 *   y = hilbert(x)        analytic signal: complex output with the
 *                         same magnitude as x in the FFT positive
 *                         half, zero in the negative half.
 *   X = goertzel(x, k)    single-bin DFT at index k (1-based, MATLAB
 *                         convention). Returns a 1×1 complex.
 */
matlab_mat_c *matlab_hilbert(matlab_mat *x) {
    if (!x) return mat_c_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N == 0) return mat_c_alloc(x->rows, x->cols);
    /* Compute FFT of x via the existing complex-FFT runtime. */
    matlab_mat_c *X = matlab_fft_c((void *)x);
    /* Apply the hilbert mask: H[0] = 1; H[1..N/2-1] = 2; H[N/2] = 1
     * (only when N is even); H[N/2+1..N-1] = 0. */
    for (int64_t k = 0; k < N; ++k) {
        double m;
        if (k == 0) m = 1.0;
        else if (k < N / 2) m = 2.0;
        else if (k == N / 2 && (N & 1) == 0) m = 1.0;
        else m = 0.0;
        X->re[k] *= m;
        X->im[k] *= m;
    }
    /* Inverse FFT. matlab_ifft_c expects a complex matrix descriptor. */
    matlab_mat_c *Y = matlab_ifft_c((void *)X);
    return Y;
}

matlab_mat_c *matlab_goertzel(matlab_mat *x, double k_d) {
    if (!x) return mat_c_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int k = (int)k_d - 1;            /* MATLAB 1-based -> 0-based */
    matlab_mat_c *Y = mat_c_alloc(1, 1);
    if (N == 0 || k < 0) return Y;
    double w = 2.0 * M_PI * (double)k / (double)N;
    double cw = cos(w), sw = sin(w);
    double s_prev = 0.0, s_prev2 = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        double s = x->data[n] + 2.0 * cw * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev  = s;
    }
    /* Final complex output: y = s[N-1] - exp(-jw) * s[N-2]. */
    Y->re[0] = s_prev - cw * s_prev2;
    Y->im[0] = sw * s_prev2;
    return Y;
}

/* poly(r) — coefficients of the monic polynomial whose roots are r.
 * Returns a row vector c of length n+1 with c[0] = 1.
 *
 * Builds (x - r_1)(x - r_2)...(x - r_n) by repeated convolution
 * starting from the constant 1 and multiplying by [1, -r_k] each step.
 * Operates on the complex plane internally so complex-conjugate-pair
 * inputs produce a real result.
 *
 * Output shape: 1 × (n+1) real matrix when the imaginary part is
 * negligible (within ~1e-10 of zero relative to the magnitude); else
 * a 1 × (n+1) complex matrix. The real-or-complex switch is encoded
 * by returning matlab_mat * vs matlab_mat_c * via the same void *
 * convention used by other polymorphic complex helpers — but for
 * simplicity we always return matlab_mat * here, dropping any
 * residual imaginary part. Callers that need the complex form can
 * route through poly_c. */
matlab_mat *matlab_poly(void *rptr) {
    if (!rptr) return mat_alloc(0, 0);
    bool is_c = mat_is_complex(rptr);
    int64_t n;
    if (is_c) {
        matlab_mat_c *R = (matlab_mat_c *)rptr;
        n = R->rows * R->cols;
    } else {
        matlab_mat *R = (matlab_mat *)rptr;
        n = R->rows * R->cols;
    }
    if (n == 0) {
        /* poly([]) = 1 (the trivial monic polynomial of degree 0). */
        matlab_mat *C = mat_alloc(1, 1);
        C->data[0] = 1.0;
        return C;
    }
    /* Coefficient buffer in highest-power-first order, complex
     * arithmetic throughout to handle conjugate pairs cleanly. */
    std::vector<double> cr(n + 1, 0.0), ci(n + 1, 0.0);
    cr[0] = 1.0;
    int64_t cur_deg = 0;       /* current polynomial degree */
    for (int64_t k = 0; k < n; ++k) {
        double rkr, rki;
        if (is_c) {
            matlab_mat_c *R = (matlab_mat_c *)rptr;
            rkr = R->re[k]; rki = R->im[k];
        } else {
            matlab_mat *R = (matlab_mat *)rptr;
            rkr = R->data[k]; rki = 0.0;
        }
        /* Multiply current polynomial by [1, -rk]. New coefficients:
         *   new[i] = old[i] - rk * old[i-1]    (with old[-1] = 0). */
        std::vector<double> nr(cur_deg + 2, 0.0), ni(cur_deg + 2, 0.0);
        for (int64_t i = 0; i <= cur_deg; ++i) {
            nr[i] += cr[i];
            ni[i] += ci[i];
        }
        for (int64_t i = 0; i <= cur_deg; ++i) {
            /* (- rk) * old[i] is appended at position i+1. */
            double ar = cr[i], ai = ci[i];
            double pr = -rkr * ar + rki * ai;
            double pi = -rkr * ai - rki * ar;
            nr[i + 1] += pr;
            ni[i + 1] += pi;
        }
        for (int64_t i = 0; i <= cur_deg + 1; ++i) {
            cr[i] = nr[i];
            ci[i] = ni[i];
        }
        cur_deg++;
    }
    /* Drop the imaginary part — MATLAB returns a real vector when the
     * input is conjugate-symmetric, and tiny residual imaginary noise
     * would propagate to downstream consumers. We return matlab_mat *
     * unconditionally; the imaginary cleanup happens here. */
    matlab_mat *C = mat_alloc(1, n + 1);
    for (int64_t i = 0; i <= n; ++i) C->data[i] = cr[i];
    return C;
}

/* residue(b, a) — partial-fraction expansion of B(s)/A(s).
 *
 * Returns three pieces via separate runtime entries (mirrors the
 * eig_V/eig_D precedent so each MATLAB output slot maps to one
 * independent runtime call):
 *
 *   matlab_residue_r(b, a)  -> matlab_mat_c *  (residues, complex column)
 *   matlab_residue_p(b, a)  -> matlab_mat_c *  (poles,    complex column)
 *   matlab_residue_k(b, a)  -> matlab_mat *    (direct term, real row)
 *
 * Algorithm:
 *   1. If deg(b) >= deg(a), long-divide → quotient k (real row vector),
 *      remainder b' (used in the residue formula).
 *   2. Find poles p = roots(a) (complex column, length deg(a)).
 *   3. Distinct-pole cover-up rule: r_i = b'(p_i) / a'(p_i)
 *      where a'(s) is polyder(a) evaluated at p_i.
 *
 * Scope: Tier-1 ships only the distinct-pole case. Repeated poles
 * fall through to the same formula with reduced numerical accuracy
 * (a'(p_i) tends to zero for repeated p_i, so r_i blows up). The
 * multiplicity-grouping path is a follow-on slice — its FP-tolerance
 * choice is non-trivial and most DSP filter designs produce distinct
 * poles by construction.
 */

/* Evaluate a real-coefficient polynomial at a complex point via
 * Horner's method. p[0] is highest-power first. */
static void polyval_c_at_(const double *p, int64_t np,
                          double zr, double zi,
                          double *out_re, double *out_im) {
    double r = 0.0, i = 0.0;
    for (int64_t k = 0; k < np; ++k) {
        double nr, ni;
        cmul_(r, i, zr, zi, &nr, &ni);
        r = nr + p[k];
        i = ni;
    }
    *out_re = r; *out_im = i;
}

/* Long-divide b by a (both highest-power-first). Returns
 * quotient (length nb - na + 1) in *qout and remainder (length na - 1)
 * in *rout. Caller owns both buffers. */
static void poly_long_divide_(const double *b, int64_t nb,
                              const double *a, int64_t na,
                              std::vector<double> &qout,
                              std::vector<double> &rout) {
    if (nb < na) {
        qout.clear();
        rout.assign(b, b + nb);
        return;
    }
    int64_t nq = nb - na + 1;
    qout.assign(nq, 0.0);
    std::vector<double> r(b, b + nb);
    double a0 = a[0];
    for (int64_t i = 0; i < nq; ++i) {
        double c = r[i] / a0;
        qout[i] = c;
        for (int64_t j = 0; j < na; ++j) r[i + j] -= c * a[j];
    }
    /* Remainder is the last (na-1) entries. */
    rout.assign(r.begin() + nq, r.end());
}

/* Internal: compute the full decomposition. Each output buffer is
 * resized; complex outputs split into separate re/im vectors. */
static void compute_residue_(matlab_mat *b, matlab_mat *a,
                             std::vector<double> &rr,
                             std::vector<double> &ri,
                             std::vector<double> &pr,
                             std::vector<double> &pi,
                             std::vector<double> &k) {
    rr.clear(); ri.clear(); pr.clear(); pi.clear(); k.clear();
    if (!b || !a) return;
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    if (na == 0) return;
    /* Strip leading zeros from a. */
    int64_t a_lead = 0;
    while (a_lead < na && a->data[a_lead] == 0.0) a_lead++;
    if (a_lead == na) return;        /* a is all zero */
    const double *a_eff = a->data + a_lead;
    int64_t na_eff = na - a_lead;
    if (na_eff == 1) {
        /* Constant a: H(s) = b/a is itself a polynomial; no poles. */
        k.assign(nb, 0.0);
        for (int64_t i = 0; i < nb; ++i) k[i] = b->data[i] / a_eff[0];
        return;
    }
    /* Long-divide b / a. */
    std::vector<double> rem;
    poly_long_divide_(b->data, nb, a_eff, na_eff, k, rem);
    /* Find poles via roots(a_eff). Build a temporary matlab_mat * to
     * call into the existing matlab_roots — avoids duplicating the
     * Durand-Kerner core. RAII frees both the input and the complex
     * roots descriptor regardless of which path returns. */
    matlab::runtime::MatPtr atmp = matlab::runtime::make_mat(1, na_eff);
    memcpy(atmp->data, a_eff, (size_t)na_eff * sizeof(double));
    matlab::runtime::MatCPtr poles(matlab_roots(atmp.get()));
    int64_t nP = poles ? (poles->rows * poles->cols) : 0;
    /* Compute polyder(a) — coefficients of a'(s). */
    int64_t nad = na_eff - 1;
    std::vector<double> ad(nad);
    for (int64_t i = 0; i < nad; ++i) {
        double power = (double)(na_eff - 1 - i);
        ad[i] = power * a_eff[i];
    }
    /* Pad remainder to a fixed length so polyval_c_at_ sees the right
     * leading zeros. The remainder has length na_eff-1; that already
     * matches polyval's expected length when we evaluate it as-is. */
    pr.resize(nP); pi.resize(nP);
    rr.resize(nP); ri.resize(nP);
    for (int64_t j = 0; j < nP; ++j) {
        double zr = poles->re[j], zi = poles->im[j];
        pr[j] = zr; pi[j] = zi;
        double br_at = 0.0, bi_at = 0.0;
        if (!rem.empty())
            polyval_c_at_(rem.data(), (int64_t)rem.size(), zr, zi,
                          &br_at, &bi_at);
        double dr_at = 0.0, di_at = 0.0;
        if (nad > 0)
            polyval_c_at_(ad.data(), nad, zr, zi, &dr_at, &di_at);
        if (dr_at == 0.0 && di_at == 0.0) {
            /* Repeated-pole or numeric singularity: leave residue 0
             * and tag with zero. The Tier-1 distinct-pole scope
             * doesn't claim correctness here; downstream tests skip. */
            rr[j] = 0.0; ri[j] = 0.0;
            continue;
        }
        double v_re, v_im;
        cdiv_(br_at, bi_at, dr_at, di_at, &v_re, &v_im);
        rr[j] = v_re; ri[j] = v_im;
    }
    /* atmp + poles freed by RAII on scope exit. */
}

matlab_mat_c *matlab_residue_r(matlab_mat *b, matlab_mat *a) {
    std::vector<double> rr, ri, pr, pi, k;
    compute_residue_(b, a, rr, ri, pr, pi, k);
    int64_t n = (int64_t)rr.size();
    matlab_mat_c *R = mat_c_alloc(n, n > 0 ? 1 : 0);
    for (int64_t i = 0; i < n; ++i) { R->re[i] = rr[i]; R->im[i] = ri[i]; }
    return R;
}

matlab_mat_c *matlab_residue_p(matlab_mat *b, matlab_mat *a) {
    std::vector<double> rr, ri, pr, pi, k;
    compute_residue_(b, a, rr, ri, pr, pi, k);
    int64_t n = (int64_t)pr.size();
    matlab_mat_c *P = mat_c_alloc(n, n > 0 ? 1 : 0);
    for (int64_t i = 0; i < n; ++i) { P->re[i] = pr[i]; P->im[i] = pi[i]; }
    return P;
}

matlab_mat *matlab_residue_k(matlab_mat *b, matlab_mat *a) {
    std::vector<double> rr, ri, pr, pi, k;
    compute_residue_(b, a, rr, ri, pr, pi, k);
    int64_t n = (int64_t)k.size();
    matlab_mat *K = mat_alloc(n > 0 ? 1 : 0, n);
    for (int64_t i = 0; i < n; ++i) K->data[i] = k[i];
    return K;
}

/*===========================================================================
 * IIR filter design (Tier-1 SPT §2.1) — lowpass scope.
 *
 * butter(n, Wn) and cheby1(n, Rp, Wn) build a digital lowpass IIR filter
 * via the standard chain:
 *
 *   1. Analog prototype poles on the s-plane (Butterworth: unit-circle
 *      arc; Chebyshev I: ellipse via the cosh/sinh closed form).
 *   2. Frequency pre-warp: Wa = 2 * tan(pi * Wn / 2). The factor of 2
 *      matches MATLAB's T = 2 normalization for the bilinear transform.
 *   3. Scale poles to the cutoff: p_warped = p_analog * Wa.
 *   4. Bilinear transform: each pole p maps to z = (1 + p) / (1 - p);
 *      analog zeros at infinity map to z = -1 (n zeros at Nyquist for
 *      both Butterworth and Chebyshev I lowpass).
 *   5. Build (b, a) = (real(poly(zeros)), real(poly(poles))). Imaginary
 *      noise vanishes because the poles come in conjugate pairs.
 *   6. Normalize: scale b so |H(z=1)| matches MATLAB's convention
 *      (unit DC gain; for even-order Chebyshev I, MATLAB's |H(1)| =
 *      10^(-Rp/20) — this slice uses unit DC gain across the board,
 *      matching MATLAB's odd-order behavior on both filter types).
 *
 * The output is two real row vectors, b (numerator, length n+1) and a
 * (denominator, length n+1). Multi-return is split into independent
 * runtime entries matlab_<filt>_b / _a, mirroring the eig precedent.
 */

/* Multiply complex (ar, ai) and (br, bi); store result in (rr, ri). */
static inline void cmul2(double ar, double ai, double br, double bi,
                         double &rr, double &ri) {
    rr = ar * br - ai * bi;
    ri = ar * bi + ai * br;
}
static inline void cdiv2(double ar, double ai, double br, double bi,
                         double &rr, double &ri) {
    double d = br * br + bi * bi;
    rr = (ar * br + ai * bi) / d;
    ri = (ai * br - ar * bi) / d;
}

/* Build the n+1 real coefficients of poly(roots) where roots is a
 * complex set assumed conjugate-symmetric. Repeated convolution by
 * [1, -r_k] in the complex plane; imaginary part is dropped at the
 * end. Output is highest-power-first. */
static void poly_from_complex_(const std::vector<double> &rr,
                               const std::vector<double> &ri,
                               std::vector<double> &out) {
    int64_t n = (int64_t)rr.size();
    std::vector<double> cr(n + 1, 0.0), ci(n + 1, 0.0);
    cr[0] = 1.0;
    int64_t cur_deg = 0;
    for (int64_t k = 0; k < n; ++k) {
        std::vector<double> nr(cur_deg + 2, 0.0), ni(cur_deg + 2, 0.0);
        for (int64_t i = 0; i <= cur_deg; ++i) {
            nr[i] += cr[i];
            ni[i] += ci[i];
        }
        for (int64_t i = 0; i <= cur_deg; ++i) {
            double pr_ = -rr[k] * cr[i] + ri[k] * ci[i];
            double pi_ = -rr[k] * ci[i] - ri[k] * cr[i];
            nr[i + 1] += pr_;
            ni[i + 1] += pi_;
        }
        cr = nr; ci = ni;
        cur_deg++;
    }
    out.resize(n + 1);
    for (int64_t i = 0; i <= n; ++i) out[i] = cr[i];
}

/* Bilinear transform of a single complex pole p. Returns (zr, zi) =
 * (1 + p) / (1 - p). T = 2 normalization absorbed into the prewarp. */
static inline void bilinear_pole_(double pr_, double pi_,
                                  double &zr_, double &zi_) {
    double num_r = 1.0 + pr_, num_i = pi_;
    double den_r = 1.0 - pr_, den_i = -pi_;
    cdiv2(num_r, num_i, den_r, den_i, zr_, zi_);
}

/* Compute (b, a) for a digital lowpass IIR designed from a set of
 * analog poles. Caller supplies the (already scaled to Wa) pole list.
 * For Butterworth and Chebyshev I lowpass, all n zeros map to z = -1
 * after bilinear, so we hard-code that here. Output b and a have
 * length n+1 each. */
static void lowpass_from_analog_poles_(const std::vector<double> &pr_,
                                       const std::vector<double> &pi_,
                                       std::vector<double> &b,
                                       std::vector<double> &a) {
    int64_t n = (int64_t)pr_.size();
    /* Bilinear-transform each pole to the z-plane. */
    std::vector<double> zr(n), zi(n);
    for (int64_t k = 0; k < n; ++k)
        bilinear_pole_(pr_[k], pi_[k], zr[k], zi[k]);
    /* a(z) = poly(z-poles). */
    poly_from_complex_(zr, zi, a);
    /* b(z) = (1 + z^-1)^n — n zeros at z = -1. The poly() of n -1's
     * gives binomial coefficients. */
    std::vector<double> nzr(n, -1.0), nzi(n, 0.0);
    poly_from_complex_(nzr, nzi, b);
    /* Normalize for unit DC gain: H(z=1) = sum(b)/sum(a) -> 1.
     * Multiply b by sum(a)/sum(b). */
    double sumb = 0.0, suma = 0.0;
    for (int64_t i = 0; i <= n; ++i) { sumb += b[i]; suma += a[i]; }
    if (sumb != 0.0) {
        double g = suma / sumb;
        for (int64_t i = 0; i <= n; ++i) b[i] *= g;
    }
}

/* Generalized lowpass-from-analog-{poles, zeros}. Used by Chebyshev II
 * which has finite j-axis zeros. Padding: if fewer zeros than poles
 * are supplied, the remainder are treated as zeros at infinity and
 * map to z = -1 after bilinear (matches the analog lowpass response
 * tail). Output is normalized for unit DC gain. */
static void lowpass_from_analog_pz_(const std::vector<double> &ppr,
                                    const std::vector<double> &ppi,
                                    const std::vector<double> &zpr,
                                    const std::vector<double> &zpi,
                                    std::vector<double> &b,
                                    std::vector<double> &a) {
    int64_t n  = (int64_t)ppr.size();
    int64_t nz = (int64_t)zpr.size();
    /* Bilinear-transform poles. */
    std::vector<double> pdr(n), pdi(n);
    for (int64_t k = 0; k < n; ++k)
        bilinear_pole_(ppr[k], ppi[k], pdr[k], pdi[k]);
    /* Bilinear-transform finite zeros + pad infinity-zeros at z = -1. */
    std::vector<double> zdr(n), zdi(n);
    for (int64_t k = 0; k < nz; ++k)
        bilinear_pole_(zpr[k], zpi[k], zdr[k], zdi[k]);
    for (int64_t k = nz; k < n; ++k) { zdr[k] = -1.0; zdi[k] = 0.0; }
    /* Build coefficients. */
    poly_from_complex_(pdr, pdi, a);
    poly_from_complex_(zdr, zdi, b);
    /* Normalize for unit DC gain. */
    double sumb = 0.0, suma = 0.0;
    for (int64_t i = 0; i <= n; ++i) { sumb += b[i]; suma += a[i]; }
    if (sumb != 0.0) {
        double g = suma / sumb;
        for (int64_t i = 0; i <= n; ++i) b[i] *= g;
    }
}

/* Internal: run the full Butterworth design and produce (b, a). */
static void compute_butter_(int n, double Wn,
                            std::vector<double> &b,
                            std::vector<double> &a) {
    if (n < 1) n = 1;
    if (Wn <= 0.0) Wn = 1e-12;
    if (Wn >= 1.0) Wn = 1.0 - 1e-12;
    /* Pre-warp the digital cutoff to the analog frequency. */
    double Wa = 2.0 * tan(M_PI * Wn / 2.0);
    /* Analog Butterworth prototype: poles on the unit circle, evenly
     * spaced on the LHS arc. p_k = exp(j * π * (2k + n - 1) / (2n))
     * for k = 1..n. Scale by Wa. */
    std::vector<double> pr_(n), pi_(n);
    for (int k = 0; k < n; ++k) {
        double theta = M_PI * (double)(2 * (k + 1) + n - 1) / (2.0 * (double)n);
        pr_[k] = Wa * cos(theta);
        pi_[k] = Wa * sin(theta);
    }
    lowpass_from_analog_poles_(pr_, pi_, b, a);
}

/* Internal: run the full Chebyshev I design and produce (b, a).
 * Rp is the passband ripple in dB. */
static void compute_cheby1_(int n, double Rp, double Wn,
                            std::vector<double> &b,
                            std::vector<double> &a) {
    if (n < 1) n = 1;
    if (Rp <= 0.0) Rp = 1e-12;
    if (Wn <= 0.0) Wn = 1e-12;
    if (Wn >= 1.0) Wn = 1.0 - 1e-12;
    double Wa = 2.0 * tan(M_PI * Wn / 2.0);
    /* Chebyshev I closed-form analog poles. */
    double eps = sqrt(pow(10.0, Rp / 10.0) - 1.0);
    double mu  = asinh(1.0 / eps) / (double)n;
    double sh  = sinh(mu), ch = cosh(mu);
    std::vector<double> pr_(n), pi_(n);
    for (int k = 0; k < n; ++k) {
        double theta = M_PI * (double)(2 * (k + 1) - 1) / (2.0 * (double)n);
        /* The s-plane pole is on an ellipse with half-axes (sh, ch)
         * about the origin. The standard form is
         *   s_k = -sh * sin(theta) + j * ch * cos(theta)
         * which already lies in the LHS for theta ∈ (0, π). */
        pr_[k] = Wa * (-sh * sin(theta));
        pi_[k] = Wa * ( ch * cos(theta));
    }
    lowpass_from_analog_poles_(pr_, pi_, b, a);
}

/* Internal: run the full Chebyshev II design and produce (b, a).
 * Rs is the stopband attenuation in dB (positive number, e.g. 40). */
static void compute_cheby2_(int n, double Rs, double Wn,
                            std::vector<double> &b,
                            std::vector<double> &a) {
    if (n < 1) n = 1;
    if (Rs <= 0.0) Rs = 1e-12;
    if (Wn <= 0.0) Wn = 1e-12;
    if (Wn >= 1.0) Wn = 1.0 - 1e-12;
    double Wa = 2.0 * tan(M_PI * Wn / 2.0);
    /* Chebyshev II analog poles + zeros.
     *
     *   eps = 1 / sqrt(10^(Rs/10) - 1)   (inverse of cheby1's eps)
     *   mu  = (1/n) * asinh(1/eps)
     *
     * For k = 1..n at theta_k = π(2k-1)/(2n):
     *
     *   chord_pole_k = -sinh(mu)*sin(theta_k) + j*cosh(mu)*cos(theta_k)
     *   cheby2_pole_k = 1 / chord_pole_k        (reciprocal — j-axis flip)
     *   cheby2_zero_k = j / cos(theta_k)        if cos(theta_k) != 0
     *
     * For odd n, the middle theta_k = π/2 produces cos = 0 so that
     * "zero" is at infinity (handled by the lowpass_from_analog_pz_
     * padding rule). For even n, all n zeros are finite j-axis pairs.
     */
    double eps = 1.0 / sqrt(pow(10.0, Rs / 10.0) - 1.0);
    double mu  = asinh(1.0 / eps) / (double)n;
    double sh  = sinh(mu), ch = cosh(mu);
    std::vector<double> ppr(n), ppi(n);
    std::vector<double> zpr, zpi;
    for (int k = 0; k < n; ++k) {
        double theta = M_PI * (double)(2 * (k + 1) - 1) / (2.0 * (double)n);
        double cr = -sh * sin(theta);
        double ci =  ch * cos(theta);
        double m2 = cr * cr + ci * ci;
        /* Reciprocal: 1 / (cr + j*ci) = (cr - j*ci) / |c|^2. */
        ppr[k] = Wa * ( cr / m2);
        ppi[k] = Wa * (-ci / m2);
        double ct = cos(theta);
        if (fabs(ct) > 1e-12) {
            zpr.push_back(0.0);
            zpi.push_back(Wa / ct);
        }
    }
    lowpass_from_analog_pz_(ppr, ppi, zpr, zpi, b, a);
}

matlab_mat *matlab_cheby2_b(double n_d, double Rs, double Wn) {
    std::vector<double> b, a;
    compute_cheby2_((int)n_d, Rs, Wn, b, a);
    int64_t L = (int64_t)b.size();
    matlab_mat *B = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) B->data[i] = b[i];
    return B;
}
matlab_mat *matlab_cheby2_a(double n_d, double Rs, double Wn) {
    std::vector<double> b, a;
    compute_cheby2_((int)n_d, Rs, Wn, b, a);
    int64_t L = (int64_t)a.size();
    matlab_mat *A = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) A->data[i] = a[i];
    return A;
}

/* buttord(Wp, Ws, Rp, Rs) — minimum order for Butterworth lowpass to
 * meet specs. Returns (n, Wn) where n is the order and Wn the natural
 * (3 dB) cutoff in normalised digital frequency.
 *
 * Algorithm:
 *   1. Pre-warp digital specs to analog: Wpa = 2*tan(π*Wp/2), same Ws.
 *   2. n = ceil(log10((10^(Rs/10) − 1)/(10^(Rp/10) − 1))
 *               / (2 * log10(Wsa / Wpa)))
 *   3. Wn (analog) = Wpa / (10^(Rp/10) − 1)^(1/(2n))
 *   4. Convert back: Wn = (2/π) * atan(Wn_analog / 2)
 *
 * Lowpass scope only — band variants and highpass use different
 * geometric formulas (a follow-on slice). */
static void compute_buttord_(double Wp, double Ws, double Rp, double Rs,
                             double &n_out, double &Wn_out) {
    if (Wp <= 0.0) Wp = 1e-12;
    if (Ws <= 0.0) Ws = 1e-12;
    if (Wp >= 1.0) Wp = 1.0 - 1e-12;
    if (Ws >= 1.0) Ws = 1.0 - 1e-12;
    double Wpa = 2.0 * tan(M_PI * Wp / 2.0);
    double Wsa = 2.0 * tan(M_PI * Ws / 2.0);
    double num = log10((pow(10.0, Rs / 10.0) - 1.0)
                     / (pow(10.0, Rp / 10.0) - 1.0));
    double den = 2.0 * log10(Wsa / Wpa);
    int n = (int)ceil(num / den);
    if (n < 1) n = 1;
    double Wna = Wpa / pow(pow(10.0, Rp / 10.0) - 1.0, 1.0 / (2.0 * (double)n));
    n_out  = (double)n;
    Wn_out = (2.0 / M_PI) * atan(Wna / 2.0);
}
double matlab_buttord_n(double Wp, double Ws, double Rp, double Rs) {
    double n_out, Wn_out;
    compute_buttord_(Wp, Ws, Rp, Rs, n_out, Wn_out);
    return n_out;
}
double matlab_buttord_Wn(double Wp, double Ws, double Rp, double Rs) {
    double n_out, Wn_out;
    compute_buttord_(Wp, Ws, Rp, Rs, n_out, Wn_out);
    return Wn_out;
}

/* cheb1ord(Wp, Ws, Rp, Rs) — minimum order for Chebyshev I lowpass.
 * Algorithm:
 *   n = ceil(acosh(sqrt((10^(Rs/10)−1)/(10^(Rp/10)−1)))
 *           / acosh(Wsa / Wpa))
 *   Wn = Wp                (Cheby I always meets passband at Wp)
 */
static void compute_cheb1ord_(double Wp, double Ws, double Rp, double Rs,
                              double &n_out, double &Wn_out) {
    if (Wp <= 0.0) Wp = 1e-12;
    if (Ws <= 0.0) Ws = 1e-12;
    if (Wp >= 1.0) Wp = 1.0 - 1e-12;
    if (Ws >= 1.0) Ws = 1.0 - 1e-12;
    double Wpa = 2.0 * tan(M_PI * Wp / 2.0);
    double Wsa = 2.0 * tan(M_PI * Ws / 2.0);
    double num = acosh(sqrt((pow(10.0, Rs / 10.0) - 1.0)
                          / (pow(10.0, Rp / 10.0) - 1.0)));
    double den = acosh(Wsa / Wpa);
    int n = (int)ceil(num / den);
    if (n < 1) n = 1;
    n_out  = (double)n;
    Wn_out = Wp;
}
double matlab_cheb1ord_n(double Wp, double Ws, double Rp, double Rs) {
    double n_out, Wn_out;
    compute_cheb1ord_(Wp, Ws, Rp, Rs, n_out, Wn_out);
    return n_out;
}
double matlab_cheb1ord_Wn(double Wp, double Ws, double Rp, double Rs) {
    double n_out, Wn_out;
    compute_cheb1ord_(Wp, Ws, Rp, Rs, n_out, Wn_out);
    return Wn_out;
}

matlab_mat *matlab_butter_b(double n_d, double Wn) {
    std::vector<double> b, a;
    compute_butter_((int)n_d, Wn, b, a);
    int64_t L = (int64_t)b.size();
    matlab_mat *B = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) B->data[i] = b[i];
    return B;
}
matlab_mat *matlab_butter_a(double n_d, double Wn) {
    std::vector<double> b, a;
    compute_butter_((int)n_d, Wn, b, a);
    int64_t L = (int64_t)a.size();
    matlab_mat *A = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) A->data[i] = a[i];
    return A;
}
matlab_mat *matlab_cheby1_b(double n_d, double Rp, double Wn) {
    std::vector<double> b, a;
    compute_cheby1_((int)n_d, Rp, Wn, b, a);
    int64_t L = (int64_t)b.size();
    matlab_mat *B = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) B->data[i] = b[i];
    return B;
}
matlab_mat *matlab_cheby1_a(double n_d, double Rp, double Wn) {
    std::vector<double> b, a;
    compute_cheby1_((int)n_d, Rp, Wn, b, a);
    int64_t L = (int64_t)a.size();
    matlab_mat *A = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) A->data[i] = a[i];
    return A;
}

/* freqz(b, a, n) — discrete-time frequency response.
 * Evaluates H(e^{jw}) at n equally spaced points w_k on [0, π) and
 * returns a complex column of length n (matching MATLAB's default
 * 'whole' = false behavior). The 2-return form also produces the
 * frequency-axis vector w. */
static void compute_freqz_(matlab_mat *bp, matlab_mat *ap, int N,
                           std::vector<double> &h_re,
                           std::vector<double> &h_im,
                           std::vector<double> &w_out) {
    h_re.clear(); h_im.clear(); w_out.clear();
    if (!bp || !ap || N <= 0) return;
    int64_t nb = bp->rows * bp->cols;
    int64_t na = ap->rows * ap->cols;
    if (na == 0 || ap->data[0] == 0.0) return;
    /* Normalize a so a[0] = 1. */
    std::vector<double> bn(nb), an(na);
    double a0 = ap->data[0];
    for (int64_t i = 0; i < nb; ++i) bn[i] = bp->data[i] / a0;
    for (int64_t i = 0; i < na; ++i) an[i] = ap->data[i] / a0;
    h_re.resize(N); h_im.resize(N); w_out.resize(N);
    for (int k = 0; k < N; ++k) {
        double w = M_PI * (double)k / (double)N;     /* 0..π exclusive */
        w_out[k] = w;
        /* Numerator: sum b_n * e^{-jwn} for n = 0..nb-1. */
        double num_r = 0.0, num_i = 0.0;
        for (int64_t i = 0; i < nb; ++i) {
            double a_ = -w * (double)i;
            num_r += bn[i] * cos(a_);
            num_i += bn[i] * sin(a_);
        }
        /* Denominator: same shape. */
        double den_r = 0.0, den_i = 0.0;
        for (int64_t i = 0; i < na; ++i) {
            double a_ = -w * (double)i;
            den_r += an[i] * cos(a_);
            den_i += an[i] * sin(a_);
        }
        double hr, hi;
        cdiv2(num_r, num_i, den_r, den_i, hr, hi);
        h_re[k] = hr; h_im[k] = hi;
    }
}

matlab_mat_c *matlab_freqz(matlab_mat *b, matlab_mat *a, double N_d) {
    std::vector<double> hr, hi, w;
    compute_freqz_(b, a, (int)N_d, hr, hi, w);
    int64_t L = (int64_t)hr.size();
    matlab_mat_c *H = mat_c_alloc(L, L > 0 ? 1 : 0);
    for (int64_t i = 0; i < L; ++i) { H->re[i] = hr[i]; H->im[i] = hi[i]; }
    return H;
}
matlab_mat_c *matlab_freqz_h(matlab_mat *b, matlab_mat *a, double N_d) {
    return matlab_freqz(b, a, N_d);
}
matlab_mat *matlab_freqz_w(matlab_mat *b, matlab_mat *a, double N_d) {
    std::vector<double> hr, hi, w;
    compute_freqz_(b, a, (int)N_d, hr, hi, w);
    int64_t L = (int64_t)w.size();
    matlab_mat *W = mat_alloc(L, L > 0 ? 1 : 0);
    for (int64_t i = 0; i < L; ++i) W->data[i] = w[i];
    return W;
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

    /* Phase-4 RAII: six manual calloc/free pairs go to value-initialised
     * std::vectors (zero-fill is implicit). */
    std::vector<double> chirp_r(N), chirp_i(N);
    std::vector<double> a_r(M), a_i(M), b_r(M), b_i(M);

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
    fft_radix2_inplace(a_r.data(), a_i.data(), M, 0);
    fft_radix2_inplace(b_r.data(), b_i.data(), M, 0);
    for (int64_t k = 0; k < M; ++k) {
        double pr = a_r[k] * b_r[k] - a_i[k] * b_i[k];
        double pi = a_r[k] * b_i[k] + a_i[k] * b_r[k];
        a_r[k] = pr; a_i[k] = pi;
    }
    fft_radix2_inplace(a_r.data(), a_i.data(), M, 1);
    /* Scale the inverse-FFT result by 1/M and multiply by conj(chirp). */
    for (int64_t n = 0; n < N; ++n) {
        double yr = a_r[n] / (double)M;
        double yi = a_i[n] / (double)M;
        re[n] = yr * chirp_r[n] + yi * chirp_i[n];
        im[n] = yi * chirp_r[n] - yr * chirp_i[n];
    }
}

/* Apply 1-D FFT to each column of the caller's matrix in place. */
static void fft_columns_inplace(double *re, double *im,
                                 int64_t rows, int64_t cols, int inverse) {
    /* Phase-4 RAII: scratch column buffers via std::vector. */
    std::vector<double> col_r(rows), col_i(rows);
    for (int64_t c = 0; c < cols; ++c) {
        for (int64_t r = 0; r < rows; ++r) {
            col_r[r] = re[r * cols + c];
            col_i[r] = im[r * cols + c];
        }
        if (is_power_of_two(rows))
            fft_radix2_inplace(col_r.data(), col_i.data(), rows, inverse);
        else
            fft_bluestein(col_r.data(), col_i.data(), rows, inverse);
        for (int64_t r = 0; r < rows; ++r) {
            re[r * cols + c] = col_r[r];
            im[r * cols + c] = col_i[r];
        }
    }
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
