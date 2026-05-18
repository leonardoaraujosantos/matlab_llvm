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

/* `complex(re_col, im_col)` for real-matrix args.  Combines element-
 * wise into a single complex matlab_mat_c of the same shape.  Handles
 * the three scalar/matrix broadcast variants too (scalar replicated to
 * the matrix shape). */
matlab_mat_c *matlab_complex_mm(matlab_mat *re, matlab_mat *im) {
    if (!re && !im) return mat_c_alloc(0, 0);
    int64_t r_re = re ? re->rows : 0, c_re = re ? re->cols : 0;
    int64_t r_im = im ? im->rows : 0, c_im = im ? im->cols : 0;
    int64_t R = (r_re >= r_im) ? r_re : r_im;
    int64_t C = (c_re >= c_im) ? c_re : c_im;
    matlab_mat_c *out = mat_c_alloc(R, C);
    int64_t N = R * C;
    for (int64_t k = 0; k < N; ++k) {
        out->re[k] = (re && r_re * c_re > 0) ? re->data[k % (r_re * c_re)] : 0.0;
        out->im[k] = (im && r_im * c_im > 0) ? im->data[k % (r_im * c_im)] : 0.0;
    }
    return out;
}
matlab_mat_c *matlab_complex_sm(double re_scalar, matlab_mat *im) {
    if (!im) return matlab_complex_scalar(re_scalar, 0.0);
    int64_t N = im->rows * im->cols;
    matlab_mat_c *out = mat_c_alloc(im->rows, im->cols);
    for (int64_t k = 0; k < N; ++k) {
        out->re[k] = re_scalar;
        out->im[k] = im->data[k];
    }
    return out;
}
matlab_mat_c *matlab_complex_ms(matlab_mat *re, double im_scalar) {
    if (!re) return matlab_complex_scalar(0.0, im_scalar);
    int64_t N = re->rows * re->cols;
    matlab_mat_c *out = mat_c_alloc(re->rows, re->cols);
    for (int64_t k = 0; k < N; ++k) {
        out->re[k] = re->data[k];
        out->im[k] = im_scalar;
    }
    return out;
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

/* Forward decl — pwelch is defined later in this file. */
matlab_mat *matlab_pwelch(matlab_mat *x, matlab_mat *win, double noverlap);

/* Cross-spectral density via Welch averaging. Returns a single-sided
 * complex column of length L/2+1. The averaging step over K segments
 * is identical to pwelch but multiplies X·conj(Y) instead of |X|². */
matlab_mat_c *matlab_cpsd(matlab_mat *x, matlab_mat *y, matlab_mat *win,
                          double noverlap_d) {
    if (!x || !y || !win) return mat_c_alloc(0, 0);
    int64_t Nx = x->rows * x->cols;
    int64_t Ny = y->rows * y->cols;
    int64_t L  = win->rows * win->cols;
    int64_t N  = Nx < Ny ? Nx : Ny;
    int     no = (int)noverlap_d;
    if (no < 0) no = 0;
    if (no >= L) no = L - 1;
    int step = (int)L - no;
    if (step < 1) step = 1;
    int64_t M = L / 2 + 1;
    if (N < L) return mat_c_alloc(M, 1);
    int K = (int)((N - L) / step) + 1;
    double U = 0.0;
    for (int64_t i = 0; i < L; ++i) U += win->data[i] * win->data[i];
    matlab_mat_c *Pxy = mat_c_alloc(M, 1);
    matlab_mat segx = { /*data*/ nullptr, /*rows*/ 1, /*cols*/ L };
    matlab_mat segy = { /*data*/ nullptr, /*rows*/ 1, /*cols*/ L };
    std::vector<double> xseg((size_t)L), yseg((size_t)L);
    segx.data = xseg.data();
    segy.data = yseg.data();
    for (int s = 0; s < K; ++s) {
        for (int64_t i = 0; i < L; ++i) {
            xseg[(size_t)i] = x->data[s * step + i] * win->data[i];
            yseg[(size_t)i] = y->data[s * step + i] * win->data[i];
        }
        matlab_mat_c *X = matlab_fft_c((void *)&segx);
        matlab_mat_c *Y = matlab_fft_c((void *)&segy);
        for (int64_t k = 0; k < M; ++k) {
            /* Pxy = X·conj(Y) summed; one-sided doubling on mid bins. */
            double xr = X->re[k], xi = X->im[k];
            double yr = Y->re[k], yi = Y->im[k];
            double cr = xr * yr + xi * yi;
            double ci = xi * yr - xr * yi;
            double s_ = ((k != 0 && (L % 2 == 0 ? k != L / 2 : 1))) ? 2.0 : 1.0;
            Pxy->re[k] += s_ * cr;
            Pxy->im[k] += s_ * ci;
        }
        free(X->re); free(X->im); free(X);
        free(Y->re); free(Y->im); free(Y);
    }
    double denom = (double)K * U;
    if (denom > 0.0) {
        for (int64_t k = 0; k < M; ++k) {
            Pxy->re[k] /= denom;
            Pxy->im[k] /= denom;
        }
    }
    return Pxy;
}

/* mscohere(x, y, win, noverlap) = |Pxy|² / (Pxx · Pyy). Returns a real
 * single-sided column. */
matlab_mat *matlab_mscohere(matlab_mat *x, matlab_mat *y, matlab_mat *win,
                            double noverlap_d) {
    matlab_mat   *Pxx = matlab_pwelch(x, win, noverlap_d);
    matlab_mat   *Pyy = matlab_pwelch(y, win, noverlap_d);
    matlab_mat_c *Pxy = matlab_cpsd  (x, y, win, noverlap_d);
    int64_t M = Pxx->rows * Pxx->cols;
    matlab_mat *C = mat_alloc(M, 1);
    for (int64_t k = 0; k < M; ++k) {
        double pmag2 = Pxy->re[k] * Pxy->re[k] + Pxy->im[k] * Pxy->im[k];
        double denom = Pxx->data[k] * Pyy->data[k];
        C->data[k] = denom > 0 ? pmag2 / denom : 0.0;
    }
    free(Pxx->data); free(Pxx);
    free(Pyy->data); free(Pyy);
    free(Pxy->re); free(Pxy->im); free(Pxy);
    return C;
}

/* tfestimate(x, y, win, noverlap) = Pxy / Pxx — complex transfer
 * function estimate. */
matlab_mat_c *matlab_tfestimate(matlab_mat *x, matlab_mat *y,
                                matlab_mat *win, double noverlap_d) {
    matlab_mat   *Pxx = matlab_pwelch(x, win, noverlap_d);
    matlab_mat_c *Pxy = matlab_cpsd  (x, y, win, noverlap_d);
    int64_t M = Pxx->rows * Pxx->cols;
    matlab_mat_c *T = mat_c_alloc(M, 1);
    for (int64_t k = 0; k < M; ++k) {
        double d = Pxx->data[k];
        if (d > 0) {
            T->re[k] = Pxy->re[k] / d;
            T->im[k] = Pxy->im[k] / d;
        }
    }
    free(Pxx->data); free(Pxx);
    free(Pxy->re); free(Pxy->im); free(Pxy);
    return T;
}

/* spectrogram(x, win, noverlap) — single-output |STFT|² per (freq, frame).
 *
 * Returns a (M × K) matrix where M = L/2 + 1 is the single-sided
 * frequency-bin count and K is the number of frames. Each column is
 * the magnitude-squared periodogram of one windowed segment. Matches
 * MATLAB's default 1-output `S = spectrogram(x, win, noverlap)` shape. */
matlab_mat *matlab_spectrogram(matlab_mat *x, matlab_mat *win, double noverlap_d) {
    if (!x || !win) return mat_alloc(0, 0);
    int64_t N  = x->rows * x->cols;
    int64_t L  = win->rows * win->cols;
    int     no = (int)noverlap_d;
    if (no < 0) no = 0;
    if (no >= L) no = L - 1;
    int step = (int)L - no;
    if (step < 1) step = 1;
    int64_t M = L / 2 + 1;
    if (N < L) return mat_alloc(M, 0);
    int K = (int)((N - L) / step) + 1;
    matlab_mat *S = mat_alloc(M, K);
    matlab_mat seg = { /*data*/ nullptr, /*rows*/ 1, /*cols*/ L };
    std::vector<double> xseg((size_t)L);
    seg.data = xseg.data();
    for (int s = 0; s < K; ++s) {
        for (int64_t i = 0; i < L; ++i)
            xseg[(size_t)i] = x->data[s * step + i] * win->data[i];
        matlab_mat_c *X = matlab_fft_c((void *)&seg);
        for (int64_t k = 0; k < M; ++k) {
            double re = X->re[k];
            double im = X->im[k];
            S->data[k * K + s] = re * re + im * im;
        }
        free(X->re); free(X->im); free(X);
    }
    return S;
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
 * (2 + p) / (2 - p). The T = 2 convention here pairs with the prewarp
 * Wa = 2·tan(π·Wn/2): together they make the digital cutoff land exactly
 * at the requested ω = π·Wn (this is the standard MATLAB/Octave/scipy
 * convention). */
static inline void bilinear_pole_(double pr_, double pi_,
                                  double &zr_, double &zi_) {
    double num_r = 2.0 + pr_, num_i = pi_;
    double den_r = 2.0 - pr_, den_i = -pi_;
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

/*===========================================================================
 * IIR family-completion infrastructure — band variants (high/bandpass/stop)
 * + ellip / besself / analog prototypes / form conversions for §2.1.
 *
 * The design pipeline is now factored into three stages:
 *   1.  Build the analog lowpass prototype (Wn = 1) for the chosen family
 *       (Butterworth / Chebyshev I / Chebyshev II / elliptic / Bessel).
 *       The prototype is described as (finite poles, finite zeros,
 *       n_zeros_at_infinity).
 *   2.  Apply the analog frequency transformation for the requested filter
 *       type:
 *         lp2lp(Wa)            scale poles/zeros by Wa
 *         lp2hp(Wa)            replace s with Wa/s; LP zeros at ∞ become
 *                              finite HP zeros at s=0; LP zeros at finite z
 *                              become Wa/z; same shape conversion for poles.
 *         lp2bp(Wa1, Wa2)      replace s with (s²+W0²)/(s·BW); each LP pole
 *                              produces 2 BP poles via a quadratic; LP zeros
 *                              at ∞ become n finite BP zeros at s=0 + n BP
 *                              zeros at ∞.
 *         lp2bs(Wa1, Wa2)      replace s with (s·BW)/(s²+W0²); each LP pole
 *                              produces 2 BS poles via a quadratic; LP zeros
 *                              at ∞ become 2n BS finite zeros at ±j·W0.
 *   3.  Bilinear-transform poles + finite zeros, append n_zeros_at_∞ digital
 *       zeros at z = -1, build (b, a) polynomials, and normalise gain at
 *       the appropriate digital frequency (DC for LP / BS, Nyquist for HP,
 *       2·atan(W0) for BP).
 *
 * The earlier `lowpass_from_analog_poles_` / `lowpass_from_analog_pz_`
 * helpers stay as-is so existing lowpass entries don't shift; the new code
 * lives alongside.
 */

/* Pre-warp a normalised digital cutoff to the analog frequency. Matches
 * the convention of compute_butter_ / compute_cheby1_ — `Wa = 2·tan(...)`. */
static inline double prewarp_(double Wn) {
    if (Wn <= 0.0) Wn = 1e-12;
    if (Wn >= 1.0) Wn = 1.0 - 1e-12;
    return 2.0 * tan(M_PI * Wn / 2.0);
}

/* Complex sqrt with conventional branch (Re ≥ 0). */
static inline void csqrt_(double xr, double xi, double &yr, double &yi) {
    double m = sqrt(xr * xr + xi * xi);
    yr = sqrt((m + xr) * 0.5);
    yi = (xi >= 0.0 ? 1.0 : -1.0) * sqrt((m - xr) * 0.5);
}

/* Build digital (b, a) from analog poles + finite zeros + count of
 * zeros at infinity, then normalise so |H(e^{j·omega_norm})| = 1. */
static void digitize_pz_(const std::vector<double> &apr,
                         const std::vector<double> &api,
                         const std::vector<double> &azr,
                         const std::vector<double> &azi,
                         int n_zeros_at_inf,
                         double omega_norm,
                         std::vector<double> &b,
                         std::vector<double> &a) {
    int n_poles    = (int)apr.size();
    int n_finite_z = (int)azr.size();
    /* Bilinear poles. */
    std::vector<double> dpr(n_poles), dpi(n_poles);
    for (int k = 0; k < n_poles; ++k)
        bilinear_pole_(apr[k], api[k], dpr[k], dpi[k]);
    /* Bilinear finite zeros, then append n_zeros_at_inf copies of -1. */
    std::vector<double> dzr, dzi;
    dzr.reserve(n_finite_z + n_zeros_at_inf);
    dzi.reserve(n_finite_z + n_zeros_at_inf);
    for (int k = 0; k < n_finite_z; ++k) {
        double zr_d, zi_d;
        bilinear_pole_(azr[k], azi[k], zr_d, zi_d);
        dzr.push_back(zr_d); dzi.push_back(zi_d);
    }
    for (int k = 0; k < n_zeros_at_inf; ++k) {
        dzr.push_back(-1.0); dzi.push_back(0.0);
    }
    /* Polynomials. */
    poly_from_complex_(dpr, dpi, a);
    poly_from_complex_(dzr, dzi, b);
    /* Pad b with leading zeros if total digital zeros < poles (pure-IIR). */
    while ((int)b.size() < n_poles + 1) b.insert(b.begin(), 0.0);
    /* Normalise. Horner-evaluate b(z) and a(z) at z = e^{j·omega_norm}. */
    double zr_n = cos(omega_norm), zi_n = sin(omega_norm);
    auto eval = [&](const std::vector<double> &p,
                    double &out_r, double &out_i) {
        out_r = p[0]; out_i = 0.0;
        for (size_t i = 1; i < p.size(); ++i) {
            double new_r = out_r * zr_n - out_i * zi_n + p[i];
            double new_i = out_r * zi_n + out_i * zr_n;
            out_r = new_r; out_i = new_i;
        }
    };
    double br, bi, ar, ai;
    eval(b, br, bi);
    eval(a, ar, ai);
    double mag2_b = br * br + bi * bi;
    double mag2_a = ar * ar + ai * ai;
    if (mag2_b > 0.0 && mag2_a > 0.0) {
        double g = sqrt(mag2_a / mag2_b);
        for (auto &v : b) v *= g;
    }
}

/* Build the analog Butterworth lowpass prototype (Wn = 1).
 * Output: n finite poles on the LHS unit circle, 0 finite zeros,
 * n_zeros_at_inf = n. */
static void buttap_proto_(int n,
                          std::vector<double> &pr_,
                          std::vector<double> &pi_) {
    pr_.resize(n); pi_.resize(n);
    for (int k = 0; k < n; ++k) {
        double theta = M_PI * (double)(2 * (k + 1) + n - 1) / (2.0 * (double)n);
        pr_[k] = cos(theta);
        pi_[k] = sin(theta);
    }
}

/* Chebyshev I lowpass prototype (Wn = 1). n poles on an LHS ellipse with
 * half-axes (sinh(mu), cosh(mu)); 0 finite zeros; n zeros at infinity. */
static void cheb1ap_proto_(int n, double Rp,
                           std::vector<double> &pr_,
                           std::vector<double> &pi_) {
    if (Rp <= 0.0) Rp = 1e-12;
    double eps = sqrt(pow(10.0, Rp / 10.0) - 1.0);
    double mu  = asinh(1.0 / eps) / (double)n;
    double sh  = sinh(mu), ch = cosh(mu);
    pr_.resize(n); pi_.resize(n);
    for (int k = 0; k < n; ++k) {
        double theta = M_PI * (double)(2 * (k + 1) - 1) / (2.0 * (double)n);
        pr_[k] = -sh * sin(theta);
        pi_[k] =  ch * cos(theta);
    }
}

/* Chebyshev II lowpass prototype (Ws = 1, finite j-axis zeros). n poles
 * via reciprocal of Chebyshev I; n−1 finite zeros for odd n, n for even n;
 * n_zeros_at_inf = 1 if odd, 0 if even. */
static void cheb2ap_proto_(int n, double Rs,
                           std::vector<double> &pr_,
                           std::vector<double> &pi_,
                           std::vector<double> &zr_,
                           std::vector<double> &zi_,
                           int &n_zeros_at_inf) {
    if (Rs <= 0.0) Rs = 1e-12;
    double eps = 1.0 / sqrt(pow(10.0, Rs / 10.0) - 1.0);
    double mu  = asinh(1.0 / eps) / (double)n;
    double sh  = sinh(mu), ch = cosh(mu);
    pr_.resize(n); pi_.resize(n);
    zr_.clear();   zi_.clear();
    n_zeros_at_inf = 0;
    for (int k = 0; k < n; ++k) {
        double theta = M_PI * (double)(2 * (k + 1) - 1) / (2.0 * (double)n);
        double cr = -sh * sin(theta);
        double ci =  ch * cos(theta);
        double m2 = cr * cr + ci * ci;
        pr_[k] =  cr / m2;
        pi_[k] = -ci / m2;
        double ct = cos(theta);
        if (fabs(ct) > 1e-12) {
            zr_.push_back(0.0);
            zi_.push_back(1.0 / ct);
        } else {
            n_zeros_at_inf++;
        }
    }
}

/* Apply lowpass-to-highpass to an analog prototype. New finite poles =
 * Wa / old_poles. Each LP zero at infinity becomes a finite HP zero at
 * s = 0; finite LP zeros become Wa / z. n_zeros_at_inf is set to 0 (HP
 * has no zeros at ∞). */
static void lp2hp_(double Wa,
                   const std::vector<double> &lp_pr,
                   const std::vector<double> &lp_pi,
                   const std::vector<double> &lp_zr,
                   const std::vector<double> &lp_zi,
                   int lp_n_zeros_at_inf,
                   std::vector<double> &hp_pr,
                   std::vector<double> &hp_pi,
                   std::vector<double> &hp_zr,
                   std::vector<double> &hp_zi,
                   int &hp_n_zeros_at_inf) {
    int np = (int)lp_pr.size();
    hp_pr.resize(np); hp_pi.resize(np);
    for (int k = 0; k < np; ++k) {
        double m2 = lp_pr[k] * lp_pr[k] + lp_pi[k] * lp_pi[k];
        hp_pr[k] =  Wa * lp_pr[k] / m2;
        hp_pi[k] = -Wa * lp_pi[k] / m2;
    }
    int nz_in = (int)lp_zr.size();
    hp_zr.clear(); hp_zi.clear();
    for (int k = 0; k < nz_in; ++k) {
        double m2 = lp_zr[k] * lp_zr[k] + lp_zi[k] * lp_zi[k];
        if (m2 == 0.0) continue;
        hp_zr.push_back(Wa * lp_zr[k] / m2);
        hp_zi.push_back(-Wa * lp_zi[k] / m2);
    }
    /* LP zeros at ∞ become HP zeros at 0. */
    for (int k = 0; k < lp_n_zeros_at_inf; ++k) {
        hp_zr.push_back(0.0); hp_zi.push_back(0.0);
    }
    hp_n_zeros_at_inf = 0;
    /* If HP has fewer finite zeros than poles after the conversion (e.g.
     * Cheby2 odd-n ellipordering), pad to match. */
    while ((int)hp_zr.size() < np) {
        hp_zr.push_back(0.0); hp_zi.push_back(0.0);
    }
}

/* Apply lowpass-to-bandpass. */
static void lp2bp_(double Wa1, double Wa2,
                   const std::vector<double> &lp_pr,
                   const std::vector<double> &lp_pi,
                   const std::vector<double> &lp_zr,
                   const std::vector<double> &lp_zi,
                   int lp_n_zeros_at_inf,
                   std::vector<double> &bp_pr,
                   std::vector<double> &bp_pi,
                   std::vector<double> &bp_zr,
                   std::vector<double> &bp_zi,
                   int &bp_n_zeros_at_inf) {
    double BW   = Wa2 - Wa1;
    double W0sq = Wa1 * Wa2;
    int np = (int)lp_pr.size();
    bp_pr.clear(); bp_pi.clear();
    bp_zr.clear(); bp_zi.clear();
    /* Each LP pole p produces 2 BP poles satisfying s² - p·BW·s + W0² = 0. */
    for (int k = 0; k < np; ++k) {
        double pbr = lp_pr[k] * BW;
        double pbi = lp_pi[k] * BW;
        double dr  = pbr * pbr - pbi * pbi - 4.0 * W0sq;
        double di  = 2.0 * pbr * pbi;
        double sr, si; csqrt_(dr, di, sr, si);
        bp_pr.push_back((pbr + sr) * 0.5);
        bp_pi.push_back((pbi + si) * 0.5);
        bp_pr.push_back((pbr - sr) * 0.5);
        bp_pi.push_back((pbi - si) * 0.5);
    }
    /* Each finite LP zero z → 2 BP zeros via the same quadratic. */
    int nz_in = (int)lp_zr.size();
    for (int k = 0; k < nz_in; ++k) {
        double zbr = lp_zr[k] * BW;
        double zbi = lp_zi[k] * BW;
        double dr  = zbr * zbr - zbi * zbi - 4.0 * W0sq;
        double di  = 2.0 * zbr * zbi;
        double sr, si; csqrt_(dr, di, sr, si);
        bp_zr.push_back((zbr + sr) * 0.5);
        bp_zi.push_back((zbi + si) * 0.5);
        bp_zr.push_back((zbr - sr) * 0.5);
        bp_zi.push_back((zbi - si) * 0.5);
    }
    /* Each LP zero at ∞ becomes 1 BP zero at s=0 + 1 BP zero at ∞. */
    for (int k = 0; k < lp_n_zeros_at_inf; ++k) {
        bp_zr.push_back(0.0); bp_zi.push_back(0.0);
    }
    bp_n_zeros_at_inf = lp_n_zeros_at_inf;
}

/* Apply lowpass-to-bandstop. */
static void lp2bs_(double Wa1, double Wa2,
                   const std::vector<double> &lp_pr,
                   const std::vector<double> &lp_pi,
                   const std::vector<double> &lp_zr,
                   const std::vector<double> &lp_zi,
                   int lp_n_zeros_at_inf,
                   std::vector<double> &bs_pr,
                   std::vector<double> &bs_pi,
                   std::vector<double> &bs_zr,
                   std::vector<double> &bs_zi,
                   int &bs_n_zeros_at_inf) {
    double BW   = Wa2 - Wa1;
    double W0sq = Wa1 * Wa2;
    int np = (int)lp_pr.size();
    bs_pr.clear(); bs_pi.clear();
    bs_zr.clear(); bs_zi.clear();
    /* Each LP pole p → 2 BS poles satisfying p·s² - BW·s + p·W0² = 0,
     * i.e. s = (BW ± sqrt(BW² - 4·p²·W0²)) / (2·p). */
    for (int k = 0; k < np; ++k) {
        double pr_ = lp_pr[k], pi_ = lp_pi[k];
        /* p² */
        double p2r = pr_ * pr_ - pi_ * pi_;
        double p2i = 2.0 * pr_ * pi_;
        /* BW² - 4·p²·W0² */
        double dr = BW * BW - 4.0 * W0sq * p2r;
        double di =          - 4.0 * W0sq * p2i;
        double sr, si; csqrt_(dr, di, sr, si);
        /* (BW ± (sr+j·si)) / (2·p) */
        double m2 = pr_ * pr_ + pi_ * pi_;
        if (m2 == 0.0) continue;
        for (int sign = +1; sign >= -1; sign -= 2) {
            double nr = BW + sign * sr;
            double ni =      sign * si;
            /* divide by 2·p = 2·(pr + j·pi) */
            double dnr = 2.0 * pr_, dni = 2.0 * pi_;
            double dm2 = dnr * dnr + dni * dni;
            double sx = (nr * dnr + ni * dni) / dm2;
            double sy = (ni * dnr - nr * dni) / dm2;
            bs_pr.push_back(sx);
            bs_pi.push_back(sy);
        }
    }
    /* Each LP zero at ∞ → 2 BS zeros at ±j·W0. */
    double W0 = sqrt(W0sq);
    for (int k = 0; k < lp_n_zeros_at_inf; ++k) {
        bs_zr.push_back(0.0); bs_zi.push_back( W0);
        bs_zr.push_back(0.0); bs_zi.push_back(-W0);
    }
    /* Finite LP zeros: same quadratic transform as poles. */
    int nz_in = (int)lp_zr.size();
    for (int k = 0; k < nz_in; ++k) {
        double zr_ = lp_zr[k], zi_ = lp_zi[k];
        double z2r = zr_ * zr_ - zi_ * zi_;
        double z2i = 2.0 * zr_ * zi_;
        double dr = BW * BW - 4.0 * W0sq * z2r;
        double di =          - 4.0 * W0sq * z2i;
        double sr, si; csqrt_(dr, di, sr, si);
        double m2 = zr_ * zr_ + zi_ * zi_;
        if (m2 == 0.0) continue;
        for (int sign = +1; sign >= -1; sign -= 2) {
            double nr = BW + sign * sr;
            double ni =      sign * si;
            double dnr = 2.0 * zr_, dni = 2.0 * zi_;
            double dm2 = dnr * dnr + dni * dni;
            double sx = (nr * dnr + ni * dni) / dm2;
            double sy = (ni * dnr - nr * dni) / dm2;
            bs_zr.push_back(sx);
            bs_zi.push_back(sy);
        }
    }
    bs_n_zeros_at_inf = 0;     /* BS has no zeros at ∞. */
}

/* Per filter family + type, run the full design pipeline and produce
 * digital (b, a). Wn1/Wn2 are normalised digital frequencies (0..1).
 * For LP/HP only Wn1 is used. Family-specific Rp/Rs are passed via the
 * `r1`/`r2` parameters (interpretation depends on family). */
enum FilterType { FT_LP = 0, FT_HP, FT_BP, FT_BS };
enum FilterFamily { FF_BUTTER = 0, FF_CHEBY1, FF_CHEBY2 };

static void compute_iir_(FilterFamily fam, FilterType ft,
                         int n, double r1,
                         double Wn1, double Wn2,
                         std::vector<double> &b,
                         std::vector<double> &a) {
    if (n < 1) n = 1;
    /* 1. Build the LP prototype with normalised cutoff. */
    std::vector<double> lp_pr, lp_pi, lp_zr, lp_zi;
    int lp_n_zeros_at_inf = 0;
    switch (fam) {
    case FF_BUTTER:
        buttap_proto_(n, lp_pr, lp_pi);
        lp_n_zeros_at_inf = n;
        break;
    case FF_CHEBY1:
        cheb1ap_proto_(n, r1, lp_pr, lp_pi);
        lp_n_zeros_at_inf = n;
        break;
    case FF_CHEBY2:
        /* The Cheby2 prototype is built normalised at the stopband edge
         * Wn (where the analog ripple peaks live at j·1/cos(θ_k)). For
         * Tier-1 lowpass scope we reuse `cheb2ap_proto_` directly. */
        cheb2ap_proto_(n, r1, lp_pr, lp_pi, lp_zr, lp_zi, lp_n_zeros_at_inf);
        break;
    }
    /* 2. Apply analog frequency transformation. */
    double Wa1 = prewarp_(Wn1);
    std::vector<double> ap, ai_p, az, ai_z;          /* finite analog poles + zeros */
    int n_zeros_at_inf = 0;
    double omega_norm = 0.0;                          /* digital ω where |H| = 1 */
    if (ft == FT_LP) {
        /* Scale prototype by Wa1. */
        ap.resize(lp_pr.size()); ai_p.resize(lp_pi.size());
        for (size_t k = 0; k < lp_pr.size(); ++k) {
            ap[k]   = Wa1 * lp_pr[k];
            ai_p[k] = Wa1 * lp_pi[k];
        }
        az.resize(lp_zr.size()); ai_z.resize(lp_zi.size());
        for (size_t k = 0; k < lp_zr.size(); ++k) {
            az[k]   = Wa1 * lp_zr[k];
            ai_z[k] = Wa1 * lp_zi[k];
        }
        n_zeros_at_inf = lp_n_zeros_at_inf;
        omega_norm     = 0.0;                         /* DC */
    } else if (ft == FT_HP) {
        lp2hp_(Wa1, lp_pr, lp_pi, lp_zr, lp_zi, lp_n_zeros_at_inf,
               ap, ai_p, az, ai_z, n_zeros_at_inf);
        omega_norm = M_PI;                            /* Nyquist */
    } else {
        double Wa2 = prewarp_(Wn2);
        if (Wa1 > Wa2) std::swap(Wa1, Wa2);
        if (ft == FT_BP) {
            lp2bp_(Wa1, Wa2, lp_pr, lp_pi, lp_zr, lp_zi, lp_n_zeros_at_inf,
                   ap, ai_p, az, ai_z, n_zeros_at_inf);
            double W0 = sqrt(Wa1 * Wa2);
            /* z = (2+s)/(2-s) maps s = j·W to angle 2·atan(W/2). */
            omega_norm = 2.0 * atan(W0 / 2.0);
        } else {
            lp2bs_(Wa1, Wa2, lp_pr, lp_pi, lp_zr, lp_zi, lp_n_zeros_at_inf,
                   ap, ai_p, az, ai_z, n_zeros_at_inf);
            omega_norm = 0.0;                         /* DC (BS keeps DC) */
        }
    }
    /* 3. Bilinear + gain normalise. */
    digitize_pz_(ap, ai_p, az, ai_z, n_zeros_at_inf, omega_norm, b, a);
}

/*===========================================================================
 * End of IIR family-completion infrastructure.
 */

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

/* cheb2ord(Wp, Ws, Rp, Rs) — minimum order for Chebyshev II lowpass.
 * Same Cheby formula as cheb1ord but the natural cutoff Wn is anchored
 * at the **stopband** edge Ws (Cheby II meets the stopband attenuation
 * exactly at Ws). */
static void compute_cheb2ord_(double Wp, double Ws, double Rp, double Rs,
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
    Wn_out = Ws;     /* Cheby II anchors at the stopband edge. */
}
double matlab_cheb2ord_n(double Wp, double Ws, double Rp, double Rs) {
    double n_out, Wn_out;
    compute_cheb2ord_(Wp, Ws, Rp, Rs, n_out, Wn_out);
    return n_out;
}
double matlab_cheb2ord_Wn(double Wp, double Ws, double Rp, double Rs) {
    double n_out, Wn_out;
    compute_cheb2ord_(Wp, Ws, Rp, Rs, n_out, Wn_out);
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

/*===========================================================================
 * Band-variant runtime entries (highpass / bandpass / bandstop) for
 * butter / cheby1 / cheby2. Each pair (_b, _a) returns one polynomial of
 * length n+1 (HP) or 2n+1 (BP, BS).
 *
 * Bandpass / bandstop entries take Wn1, Wn2 as separate doubles rather
 * than a 2-element vector, to keep the runtime ABI scalar-only. The
 * LowerTensorOps dispatch unpacks the matrix-shaped Wn into the two
 * element loads at the call site.
 */
static matlab_mat *iir_pack_(const std::vector<double> &v) {
    int64_t L = (int64_t)v.size();
    matlab_mat *M = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) M->data[i] = v[i];
    return M;
}

/* Butterworth band variants. */
matlab_mat *matlab_butter_hp_b(double n_d, double Wn) {
    std::vector<double> b, a;
    compute_iir_(FF_BUTTER, FT_HP, (int)n_d, 0.0, Wn, 0.0, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_butter_hp_a(double n_d, double Wn) {
    std::vector<double> b, a;
    compute_iir_(FF_BUTTER, FT_HP, (int)n_d, 0.0, Wn, 0.0, b, a);
    return iir_pack_(a);
}
matlab_mat *matlab_butter_bp_b(double n_d, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_BUTTER, FT_BP, (int)n_d, 0.0, W1, W2, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_butter_bp_a(double n_d, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_BUTTER, FT_BP, (int)n_d, 0.0, W1, W2, b, a);
    return iir_pack_(a);
}
matlab_mat *matlab_butter_bs_b(double n_d, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_BUTTER, FT_BS, (int)n_d, 0.0, W1, W2, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_butter_bs_a(double n_d, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_BUTTER, FT_BS, (int)n_d, 0.0, W1, W2, b, a);
    return iir_pack_(a);
}

/* Chebyshev I band variants. */
matlab_mat *matlab_cheby1_hp_b(double n_d, double Rp, double Wn) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY1, FT_HP, (int)n_d, Rp, Wn, 0.0, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_cheby1_hp_a(double n_d, double Rp, double Wn) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY1, FT_HP, (int)n_d, Rp, Wn, 0.0, b, a);
    return iir_pack_(a);
}
matlab_mat *matlab_cheby1_bp_b(double n_d, double Rp, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY1, FT_BP, (int)n_d, Rp, W1, W2, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_cheby1_bp_a(double n_d, double Rp, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY1, FT_BP, (int)n_d, Rp, W1, W2, b, a);
    return iir_pack_(a);
}
matlab_mat *matlab_cheby1_bs_b(double n_d, double Rp, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY1, FT_BS, (int)n_d, Rp, W1, W2, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_cheby1_bs_a(double n_d, double Rp, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY1, FT_BS, (int)n_d, Rp, W1, W2, b, a);
    return iir_pack_(a);
}

/* Chebyshev II band variants. */
matlab_mat *matlab_cheby2_hp_b(double n_d, double Rs, double Wn) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY2, FT_HP, (int)n_d, Rs, Wn, 0.0, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_cheby2_hp_a(double n_d, double Rs, double Wn) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY2, FT_HP, (int)n_d, Rs, Wn, 0.0, b, a);
    return iir_pack_(a);
}
matlab_mat *matlab_cheby2_bp_b(double n_d, double Rs, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY2, FT_BP, (int)n_d, Rs, W1, W2, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_cheby2_bp_a(double n_d, double Rs, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY2, FT_BP, (int)n_d, Rs, W1, W2, b, a);
    return iir_pack_(a);
}
matlab_mat *matlab_cheby2_bs_b(double n_d, double Rs, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY2, FT_BS, (int)n_d, Rs, W1, W2, b, a);
    return iir_pack_(b);
}
matlab_mat *matlab_cheby2_bs_a(double n_d, double Rs, double W1, double W2) {
    std::vector<double> b, a;
    compute_iir_(FF_CHEBY2, FT_BS, (int)n_d, Rs, W1, W2, b, a);
    return iir_pack_(a);
}

/*===========================================================================
 * Standalone analog→digital bilinear, analog frequency response, and
 * tf↔zp form conversions.
 *
 *   [bd, ad] = bilinear(b, a, fs)   analog (b, a) → digital via z = (2fs+s)/(2fs-s)
 *   H        = freqs(b, a, w)        analog frequency response B(jw)/A(jw)
 *   [z, p, k]= tf2zp(b, a)           polynomial form → zero/pole/gain
 *   [b, a]   = zp2tf(z, p, k)        zero/pole/gain → polynomial form
 *
 * tf2zp / zp2tf split via the eig precedent — separate `_z` / `_p` / `_k`
 * (and `_b` / `_a`) runtime entries. Multi-LHS dispatch in
 * LowerTensorOps.cpp.
 */

/* Bilinear transform with sample rate `fs` (T = 1/fs). For `fs = 1`
 * this matches the internal bilinear used by butter/cheby designs. */
static inline void bilinear_pole_fs_(double pr_, double pi_, double fs,
                                     double &zr_, double &zi_) {
    double f2 = 2.0 * fs;
    double num_r = f2 + pr_, num_i = pi_;
    double den_r = f2 - pr_, den_i = -pi_;
    cdiv2(num_r, num_i, den_r, den_i, zr_, zi_);
}

static void compute_bilinear_(matlab_mat *bp, matlab_mat *ap, double fs,
                              std::vector<double> &bd,
                              std::vector<double> &ad) {
    bd.clear(); ad.clear();
    if (!bp || !ap) return;
    int64_t nb = bp->rows * bp->cols;
    int64_t na = ap->rows * ap->cols;
    if (na == 0) return;
    /* Use the existing matlab_roots to find analog zeros + poles. */
    matlab_mat_c *bz = matlab_roots(bp);
    matlab_mat_c *ap_roots = matlab_roots(ap);
    int64_t nz = bz ? bz->rows * bz->cols : 0;
    int64_t np = ap_roots ? ap_roots->rows * ap_roots->cols : 0;
    /* Analog → digital roots. */
    std::vector<double> dpr(np), dpi(np);
    std::vector<double> dzr(nz), dzi(nz);
    for (int64_t k = 0; k < np; ++k) {
        bilinear_pole_fs_(ap_roots->re[k], ap_roots->im[k], fs,
                          dpr[k], dpi[k]);
    }
    for (int64_t k = 0; k < nz; ++k) {
        bilinear_pole_fs_(bz->re[k], bz->im[k], fs, dzr[k], dzi[k]);
    }
    /* Pad zeros at z = -1 if degree(b) < degree(a) (n_zeros_at_inf). */
    while ((int64_t)dzr.size() < np) {
        dzr.push_back(-1.0); dzi.push_back(0.0);
    }
    /* Build polynomials. */
    poly_from_complex_(dpr, dpi, ad);
    poly_from_complex_(dzr, dzi, bd);
    while ((int64_t)bd.size() < (int64_t)ad.size()) bd.insert(bd.begin(), 0.0);
    /* Scale by analog leading-coefficient ratio so that the digital
     * filter preserves the analog gain at z = 1 (DC for low/lowband
     * filters; for highpass-style analog filters the user typically
     * post-scales). The analog gain factor at s = 0 is bp[end]/ap[end]
     * (constant terms). The bilinear preserves the s = 0 ↔ z = 1 map. */
    double sb = 0.0, sa = 0.0;
    for (auto v : bd) sb += v;
    for (auto v : ad) sa += v;
    /* Match analog DC gain b(0)/a(0). For polynomials b(s) = b[0]s^n +
     * ... + b[n], evaluation at s = 0 gives b[n] (the constant term). */
    double an_dc = bp->data[nb - 1] / ap->data[na - 1];
    if (sb != 0.0 && sa != 0.0) {
        double g = an_dc * sa / sb;
        for (auto &v : bd) v *= g;
    }
    if (bz) { free(bz->re); free(bz->im); free(bz); }
    if (ap_roots) { free(ap_roots->re); free(ap_roots->im); free(ap_roots); }
}

matlab_mat *matlab_bilinear_b(matlab_mat *b, matlab_mat *a, double fs) {
    std::vector<double> bd, ad;
    compute_bilinear_(b, a, fs, bd, ad);
    int64_t L = (int64_t)bd.size();
    matlab_mat *B = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) B->data[i] = bd[i];
    return B;
}
matlab_mat *matlab_bilinear_a(matlab_mat *b, matlab_mat *a, double fs) {
    std::vector<double> bd, ad;
    compute_bilinear_(b, a, fs, bd, ad);
    int64_t L = (int64_t)ad.size();
    matlab_mat *A = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) A->data[i] = ad[i];
    return A;
}

/* Analog frequency response: H(jw) = B(jw) / A(jw). */
matlab_mat_c *matlab_freqs(matlab_mat *b, matlab_mat *a, matlab_mat *w) {
    if (!b || !a || !w) return mat_c_alloc(0, 0);
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    int64_t N  = w->rows * w->cols;
    matlab_mat_c *H = mat_c_alloc(N, 1);
    for (int64_t k = 0; k < N; ++k) {
        double wk = w->data[k];
        /* Horner-evaluate b at jw: result = b[0], then result = result*jw + b[i]. */
        double br_ = b->data[0], bi_ = 0.0;
        for (int64_t i = 1; i < nb; ++i) {
            double new_r = -bi_ * wk + b->data[i];
            double new_i =  br_ * wk;
            br_ = new_r; bi_ = new_i;
        }
        double ar_ = a->data[0], ai_ = 0.0;
        for (int64_t i = 1; i < na; ++i) {
            double new_r = -ai_ * wk + a->data[i];
            double new_i =  ar_ * wk;
            ar_ = new_r; ai_ = new_i;
        }
        double hr, hi;
        cdiv2(br_, bi_, ar_, ai_, hr, hi);
        H->re[k] = hr; H->im[k] = hi;
    }
    return H;
}

/* tf2zp(b, a) — polynomial → zero/pole/gain.
 *   z = roots(b),  p = roots(a),  k = b[0] / a[0]
 * Multi-return splits via three independent runtime entries. */
matlab_mat_c *matlab_tf2zp_z(matlab_mat *b, matlab_mat *a) {
    (void)a;
    return matlab_roots(b);
}
matlab_mat_c *matlab_tf2zp_p(matlab_mat *b, matlab_mat *a) {
    (void)b;
    return matlab_roots(a);
}
double matlab_tf2zp_k(matlab_mat *b, matlab_mat *a) {
    if (!b || !a || b->rows * b->cols == 0 || a->rows * a->cols == 0)
        return 0.0;
    if (a->data[0] == 0.0) return 0.0;
    return b->data[0] / a->data[0];
}

/* zp2tf(z, p, k) — zero/pole/gain → polynomial.
 *   b = k * poly(z),  a = poly(p)
 * z and p are complex matrices (matlab_mat_c). */
static void zp2tf_build_poly_(matlab_mat_c *roots_c,
                              std::vector<double> &out) {
    int64_t n = roots_c ? roots_c->rows * roots_c->cols : 0;
    if (n == 0) { out = {1.0}; return; }
    std::vector<double> rr(n), ri(n);
    for (int64_t i = 0; i < n; ++i) { rr[i] = roots_c->re[i]; ri[i] = roots_c->im[i]; }
    poly_from_complex_(rr, ri, out);
}

matlab_mat *matlab_zp2tf_b(matlab_mat_c *z, matlab_mat_c *p, double k) {
    (void)p;
    std::vector<double> coefs;
    zp2tf_build_poly_(z, coefs);
    int64_t L = (int64_t)coefs.size();
    matlab_mat *B = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) B->data[i] = k * coefs[i];
    return B;
}
matlab_mat *matlab_zp2tf_a(matlab_mat_c *z, matlab_mat_c *p, double k) {
    (void)z; (void)k;
    std::vector<double> coefs;
    zp2tf_build_poly_(p, coefs);
    int64_t L = (int64_t)coefs.size();
    matlab_mat *A = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) A->data[i] = coefs[i];
    return A;
}

/* besself(n, Wo) — analog Bessel-Thomson lowpass.
 *
 * MATLAB convention: poles of the unit Bessel polynomial scaled by Wo.
 * The transfer function is H(s) = B_n(0)·Wo^n / B_n(s/Wo), normalised
 * to unit DC gain. With the s → s/Wo substitution:
 *   B_n,Wo(s) = sum_i Bn[i]·s^(n-i) / Wo^(n-i)
 * Multiplying through by Wo^n to keep `a` monic gives
 *   a[i] = Bn[i] · Wo^i           (MATLAB-order: i = 0..n)
 * and b = [a(end)] so DC gain b/a(end) = 1.
 *
 * Returns (b, a) of length 1 / n+1 respectively. Multi-return splits
 * via _b / _a entries (eig precedent).
 */
static void bessel_recur_(int n, std::vector<double> &coefs) {
    /* coefs is in MATLAB order: coefs[0]·s^n + ... + coefs[n]. */
    if (n == 0) { coefs = {1.0}; return; }
    std::vector<double> Bm1 = {1.0, 1.0};       /* B_1 = s + 1 */
    std::vector<double> Bm2 = {1.0};            /* B_0 = 1 */
    if (n == 1) { coefs = Bm1; return; }
    for (int k = 2; k <= n; ++k) {
        std::vector<double> Bk((size_t)(k + 1), 0.0);
        /* (2k-1) · Bm1, padded to degree k. */
        double a = (double)(2 * k - 1);
        /* Bm1 has degree k-1, so Bm1 has k entries; align to degree k. */
        for (size_t i = 0; i < Bm1.size(); ++i)
            Bk[i + 1] += a * Bm1[i];
        /* s² · Bm2: shift Bm2 (degree k-2) up by 2 → degree k. */
        for (size_t i = 0; i < Bm2.size(); ++i)
            Bk[i] += Bm2[i];
        Bm2 = Bm1;
        Bm1 = Bk;
    }
    coefs = Bm1;
}

static void compute_besself_analog_(int n, double Wo,
                                    std::vector<double> &b,
                                    std::vector<double> &a) {
    if (n < 1) n = 1;
    if (Wo <= 0.0) Wo = 1.0;
    std::vector<double> Bn;
    bessel_recur_(n, Bn);
    a.resize(Bn.size());
    for (size_t i = 0; i < Bn.size(); ++i)
        a[i] = Bn[i] * pow(Wo, (double)i);
    b.clear();
    b.push_back(a.back());
}

matlab_mat *matlab_besself_b(double n_d, double Wo) {
    std::vector<double> b, a;
    compute_besself_analog_((int)n_d, Wo, b, a);
    int64_t L = (int64_t)b.size();
    matlab_mat *B = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) B->data[i] = b[i];
    return B;
}
matlab_mat *matlab_besself_a(double n_d, double Wo) {
    std::vector<double> b, a;
    compute_besself_analog_((int)n_d, Wo, b, a);
    int64_t L = (int64_t)a.size();
    matlab_mat *A = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) A->data[i] = a[i];
    return A;
}

/* tf2sos(b, a) — polynomial → second-order-section cascade.
 *
 * Output is an L × 6 matrix where each row is [b0 b1 b2 a0 a1 a2] for
 * one biquad section. L = ceil(N/2) where N = max(deg b, deg a).
 *
 * Pairing strategy: walk the root list, pair each complex root with
 * its conjugate, treat real roots as a quadratic (s - r)·1 → [1, -r, 0].
 * One numerator pair + one denominator pair per section. Numerator pad
 * with [0, 0] if there are fewer numerator pairs than denominator pairs
 * (i.e., for a strictly-proper transfer function that is the all-pole
 * case). The leading section absorbs the overall gain b[0]/a[0].
 *
 * Multi-return single-LHS only (no separate `g` output for now —
 * MATLAB's `[sos, g] = tf2sos(...)` is a follow-on).
 */
static void pair_conj_roots_(matlab_mat_c *roots_c,
                             std::vector<std::pair<double, double>> &out_quads) {
    /* Each output entry is (linear_coef, constant_coef) of a real
     * quadratic z^2 + linear·z + const. Real-paired conjugate (a+bi),
     * (a-bi) gives (-2a, a²+b²). Lone real root r gives (-r, 0). */
    out_quads.clear();
    if (!roots_c) return;
    int64_t n = roots_c->rows * roots_c->cols;
    std::vector<bool> used((size_t)n, false);
    for (int64_t i = 0; i < n; ++i) {
        if (used[(size_t)i]) continue;
        double rr = roots_c->re[i];
        double ri = roots_c->im[i];
        if (fabs(ri) < 1e-9) {
            out_quads.push_back({-rr, 0.0});
            used[(size_t)i] = true;
            continue;
        }
        /* Find the conjugate (rr, -ri) among remaining unused roots. */
        int64_t j = -1;
        double best = 1e30;
        for (int64_t k = i + 1; k < n; ++k) {
            if (used[(size_t)k]) continue;
            double dr = roots_c->re[k] - rr;
            double di = roots_c->im[k] + ri;       /* conjugate match */
            double d = dr * dr + di * di;
            if (d < best) { best = d; j = k; }
        }
        if (j < 0) {
            /* Unpaired complex — emit linear with the magnitude. */
            out_quads.push_back({-rr, rr * rr + ri * ri});
            used[(size_t)i] = true;
        } else {
            out_quads.push_back({-2.0 * rr, rr * rr + ri * ri});
            used[(size_t)i] = used[(size_t)j] = true;
        }
    }
}

matlab_mat *matlab_tf2sos(matlab_mat *b, matlab_mat *a) {
    if (!b || !a) return mat_alloc(0, 6);
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    if (nb == 0 || na == 0 || a->data[0] == 0.0) return mat_alloc(0, 6);
    matlab_mat_c *bz = matlab_roots(b);
    matlab_mat_c *az = matlab_roots(a);
    std::vector<std::pair<double, double>> b_qs, a_qs;
    pair_conj_roots_(bz, b_qs);
    pair_conj_roots_(az, a_qs);
    /* Pad b_qs with (0, 0) sections if fewer than a_qs (all-pole or
     * strictly-proper case). */
    while (b_qs.size() < a_qs.size()) b_qs.push_back({0.0, 0.0});
    while (a_qs.size() < b_qs.size()) a_qs.push_back({0.0, 0.0});
    int64_t L = (int64_t)a_qs.size();
    matlab_mat *S = mat_alloc(L, 6);
    /* Overall gain: b[0]/a[0] absorbed into the first section's b. */
    double g = b->data[0] / a->data[0];
    for (int64_t i = 0; i < L; ++i) {
        double *r = S->data + i * 6;
        double bg = (i == 0) ? g : 1.0;
        r[0] = bg * 1.0;
        r[1] = bg * b_qs[(size_t)i].first;
        r[2] = bg * b_qs[(size_t)i].second;
        r[3] = 1.0;
        r[4] = a_qs[(size_t)i].first;
        r[5] = a_qs[(size_t)i].second;
    }
    if (bz) { free(bz->re); free(bz->im); free(bz); }
    if (az) { free(az->re); free(az->im); free(az); }
    return S;
}

/* sos2tf(sos) — convolve all sections' numerators and denominators
 * into single (b, a) polynomials. Multi-LHS splits into _b / _a. */
static void compute_sos2tf_(matlab_mat *sos,
                            std::vector<double> &b,
                            std::vector<double> &a) {
    b = {1.0};
    a = {1.0};
    if (!sos) return;
    int64_t L = sos->rows;
    int64_t W = sos->cols;
    if (W != 6 || L == 0) return;
    auto convolve = [](const std::vector<double> &p,
                       const std::vector<double> &q) -> std::vector<double> {
        std::vector<double> r(p.size() + q.size() - 1, 0.0);
        for (size_t i = 0; i < p.size(); ++i)
            for (size_t j = 0; j < q.size(); ++j)
                r[i + j] += p[i] * q[j];
        return r;
    };
    for (int64_t s = 0; s < L; ++s) {
        const double *r = sos->data + s * 6;
        std::vector<double> bs = {r[0], r[1], r[2]};
        std::vector<double> as = {r[3], r[4], r[5]};
        /* Trim trailing zeros so 1st-order sections [1, -r, 0] don't
         * inflate the convolution length needlessly. */
        while (bs.size() > 1 && bs.back() == 0.0) bs.pop_back();
        while (as.size() > 1 && as.back() == 0.0) as.pop_back();
        b = convolve(b, bs);
        a = convolve(a, as);
    }
}

matlab_mat *matlab_sos2tf_b(matlab_mat *sos) {
    std::vector<double> b, a;
    compute_sos2tf_(sos, b, a);
    int64_t L = (int64_t)b.size();
    matlab_mat *B = mat_alloc(1, L);
    for (int64_t i = 0; i < L; ++i) B->data[i] = b[i];
    return B;
}
matlab_mat *matlab_sos2tf_a(matlab_mat *sos) {
    std::vector<double> b, a;
    compute_sos2tf_(sos, b, a);
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
