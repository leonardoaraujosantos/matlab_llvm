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
