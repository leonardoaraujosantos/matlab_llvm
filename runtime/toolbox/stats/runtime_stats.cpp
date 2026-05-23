/* ============================================================================
 * runtime_stats.cpp — Statistics and Machine Learning Toolbox runtime
 * ----------------------------------------------------------------------------
 * Tier-1: descriptive statistics, covariance/correlation, probability
 * distributions (pdf/cdf/inv), and distribution random-number generators.
 *
 * Every entry is matlab_mat* in / matlab_mat* out (scalars arrive as 1x1
 * descriptors — pde_table boxes f64 literals via matlab_mat_from_scalar).
 * Reductions are column-wise for matrices and whole-vector for vectors,
 * mirroring the shipped `matlab_var`.  No external dependency: the normal
 * CDF rides libc `erf`/`erfc`, the inverse normal is Acklam's rational
 * approximation, and the RNGs reuse the shared `rng`-seeded PRNG through
 * the shipped `matlab_rand` / `matlab_randn`.
 *
 * Companion classdef (makedist / fitdist / ProbDistUnivParam) lives in
 * stats_classdefs.m; this TU holds the numeric cores those methods call.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <vector>

/* RNG reuse — both run over the shared rng-seeded xorshift state. */
extern "C" matlab_mat *matlab_rand(double m, double n);
extern "C" matlab_mat *matlab_randn(double m, double n);
extern "C" matlab_mat *matlab_mat_from_scalar(double x);
/* matlab_struct_new / matlab_struct_set_f64 come from matlab_runtime.h
 * (used to build the [.,.,.,stats] hypothesis-test output struct). */
/* Object accessors (distribution-object dispatch, makedist/fitdist). */
extern "C" double matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void   matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
extern "C" int    matlab_obj_is_known(const void *p);
/* Forward decls so the helpers can reuse the exported cores. */
extern "C" matlab_mat *matlab_stats_cov(matlab_mat *X);
extern "C" matlab_mat *matlab_stats_normpdf(matlab_mat *x, matlab_mat *mu, matlab_mat *sg);
extern "C" matlab_mat *matlab_stats_normcdf(matlab_mat *x, matlab_mat *mu, matlab_mat *sg);
extern "C" matlab_mat *matlab_stats_norminv(matlab_mat *p, matlab_mat *mu, matlab_mat *sg);
extern "C" matlab_mat *matlab_stats_exppdf(matlab_mat *x, matlab_mat *mu);
extern "C" matlab_mat *matlab_stats_expcdf(matlab_mat *x, matlab_mat *mu);
extern "C" matlab_mat *matlab_stats_expinv(matlab_mat *p, matlab_mat *mu);
extern "C" matlab_mat *matlab_stats_unifpdf(matlab_mat *x, matlab_mat *a, matlab_mat *b);
extern "C" matlab_mat *matlab_stats_unifcdf(matlab_mat *x, matlab_mat *a, matlab_mat *b);
extern "C" matlab_mat *matlab_stats_unifinv(matlab_mat *p, matlab_mat *a, matlab_mat *b);
extern "C" matlab_mat *matlab_stats_normrnd(matlab_mat *mu, matlab_mat *sg, matlab_mat *m, matlab_mat *n);
extern "C" matlab_mat *matlab_stats_exprnd(matlab_mat *mu, matlab_mat *m, matlab_mat *n);
extern "C" matlab_mat *matlab_stats_unifrnd(matlab_mat *a, matlab_mat *b, matlab_mat *m, matlab_mat *n);
extern "C" matlab_mat *matlab_stats_fit_normal(matlab_mat *x);
extern "C" matlab_mat *matlab_stats_fit_exponential(matlab_mat *x);
extern "C" void matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);

/* ===== file-scope helpers (C++ linkage — templates can't be extern "C") === */

static const double SQRT2   = 1.41421356237309504880;
static const double SQRT2PI = 2.50662827463100050242;

/* Scalar read with a default when the descriptor is empty/missing. */
static double sstat_sc(const matlab_mat *m, double dflt) {
    return (m && m->data && m->rows * m->cols > 0) ? m->data[0] : dflt;
}

static int64_t sstat_len(const matlab_mat *m) {
    return m ? m->rows * m->cols : 0;
}

/* Is this descriptor a row/column vector (or scalar)? */
static bool sstat_is_vec(const matlab_mat *m) {
    return m && (m->rows <= 1 || m->cols <= 1);
}

/* Column extractor: copies column j (matrix) or the whole thing (vector). */
static std::vector<double> sstat_col(const matlab_mat *A, int64_t j) {
    std::vector<double> v;
    if (!A || !A->data) return v;
    if (sstat_is_vec(A)) {
        int64_t n = A->rows * A->cols;
        v.assign(A->data, A->data + n);
    } else {
        v.reserve(static_cast<size_t>(A->rows));
        for (int64_t i = 0; i < A->rows; ++i)
            v.push_back(A->data[i * A->cols + j]);
    }
    return v;
}

static double sstat_mean(const std::vector<double> &c) {
    if (c.empty()) return 0.0;
    double s = 0.0;
    for (double v : c) s += v;
    return s / static_cast<double>(c.size());
}

/* central moment (1/n) sum (x-mean)^k */
static double sstat_cmoment(const std::vector<double> &c, int k) {
    double mu = sstat_mean(c), s = 0.0;
    for (double v : c) s += pow(v - mu, k);
    return c.empty() ? 0.0 : s / static_cast<double>(c.size());
}

/* MATLAB's exact percentile: sorted samples sit at plotting positions
 * 100*(i-0.5)/n; linear interpolation between, clamped to the extremes. */
static double sstat_prctile_one(std::vector<double> s, double p) {
    int64_t n = static_cast<int64_t>(s.size());
    if (n == 0) return NAN;
    if (n == 1) return s[0];
    std::sort(s.begin(), s.end());
    double pos = p / 100.0 * static_cast<double>(n) - 0.5;  /* 0-based */
    if (pos <= 0.0) return s[0];
    if (pos >= static_cast<double>(n - 1)) return s[static_cast<size_t>(n - 1)];
    int64_t lo = static_cast<int64_t>(floor(pos));
    double frac = pos - static_cast<double>(lo);
    return s[static_cast<size_t>(lo)] * (1.0 - frac) +
           s[static_cast<size_t>(lo + 1)] * frac;
}

/* Acklam's inverse standard-normal CDF (≈1.15e-9 abs error). */
static double sstat_norminv_std(double p) {
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return INFINITY;
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - 0.02425;
    double q, r;
    if (p < plow) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > phigh) {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

/* ---- special functions: regularized incomplete gamma / beta ----------- *
 * Numerical-Recipes-style series + continued-fraction.  These give the
 * t / F / chi-square CDFs the hypothesis tests (Tier-2) need, and unblock
 * the wider distribution library later. */
static double sgammp_series(double a, double x) {       /* lower P(a,x), x<a+1 */
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 200; ++n) {
        ap += 1.0; del *= x / ap; sum += del;
        if (fabs(del) < fabs(sum) * 1e-15) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}
static double sgammq_cf(double a, double x) {           /* upper Q(a,x), x>=a+1 */
    const double FPMIN = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / FPMIN, d = 1.0 / b, h = d;
    for (int i = 1; i < 200; ++i) {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b; if (fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; double del = d * c; h *= del;
        if (fabs(del - 1.0) < 1e-15) break;
    }
    return exp(-x + a * log(x) - lgamma(a)) * h;
}
/* regularized lower incomplete gamma P(a,x) */
static double sgammp(double a, double x) {
    if (x <= 0.0 || a <= 0.0) return 0.0;
    return (x < a + 1.0) ? sgammp_series(a, x) : 1.0 - sgammq_cf(a, x);
}
/* continued fraction for the incomplete beta */
static double sbetacf(double a, double b, double x) {
    const double FPMIN = 1e-300;
    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d; double h = d;
    for (int m = 1; m < 300; ++m) {
        double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; double del = d * c; h *= del;
        if (fabs(del - 1.0) < 1e-15) break;
    }
    return h;
}
/* regularized incomplete beta I_x(a,b) */
static double sbetai(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) +
                    a * log(x) + b * log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) return bt * sbetacf(a, b, x) / a;
    return 1.0 - bt * sbetacf(b, a, 1.0 - x) / b;
}
/* Student-t / F / chi-square CDFs. */
static double stcdf(double t, double nu) {
    double x = nu / (nu + t * t);
    double ib = 0.5 * sbetai(nu / 2.0, 0.5, x);
    return (t > 0.0) ? 1.0 - ib : ib;
}
static double sfcdf(double f, double d1, double d2) {
    if (f <= 0.0) return 0.0;
    return sbetai(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2));
}
static double schi2cdf(double x, double k) { return sgammp(k / 2.0, x / 2.0); }
/* inverse Student-t CDF via bisection (used for confidence intervals). */
static double stinv(double p, double nu) {
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return INFINITY;
    double lo = -1e4, hi = 1e4;
    for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        if (stcdf(mid, nu) < p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

/* ===== Tier-3 regression helpers (file scope) ============================ */

/* Dense m×m inverse via Gauss-Jordan with partial pivoting (m is small —
 * the number of regression coefficients). Returns identity on singular. */
static std::vector<double> sinv_dense(std::vector<double> A, int m) {
    std::vector<double> I(static_cast<size_t>(m * m), 0.0);
    for (int i = 0; i < m; ++i) I[static_cast<size_t>(i * m + i)] = 1.0;
    for (int col = 0; col < m; ++col) {
        int piv = col; double best = fabs(A[static_cast<size_t>(col * m + col)]);
        for (int r = col + 1; r < m; ++r) {
            double v = fabs(A[static_cast<size_t>(r * m + col)]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return I;          /* singular — bail */
        if (piv != col)
            for (int j = 0; j < m; ++j) {
                std::swap(A[static_cast<size_t>(col * m + j)], A[static_cast<size_t>(piv * m + j)]);
                std::swap(I[static_cast<size_t>(col * m + j)], I[static_cast<size_t>(piv * m + j)]);
            }
        double d = A[static_cast<size_t>(col * m + col)];
        for (int j = 0; j < m; ++j) {
            A[static_cast<size_t>(col * m + j)] /= d;
            I[static_cast<size_t>(col * m + j)] /= d;
        }
        for (int r = 0; r < m; ++r) {
            if (r == col) continue;
            double f = A[static_cast<size_t>(r * m + col)];
            for (int j = 0; j < m; ++j) {
                A[static_cast<size_t>(r * m + j)] -= f * A[static_cast<size_t>(col * m + j)];
                I[static_cast<size_t>(r * m + j)] -= f * I[static_cast<size_t>(col * m + j)];
            }
        }
    }
    return I;
}

/* Build the n×q design matrix (row-major) from predictor matrix X (n×p);
 * prepends a column of ones when `intercept`. */
static std::vector<double> sdesign(const matlab_mat *X, int64_t &n, int64_t &p,
                                   int &q, bool intercept) {
    if (X && (X->rows <= 1 || X->cols <= 1)) { n = X->rows * X->cols; p = 1; }
    else if (X) { n = X->rows; p = X->cols; }
    else { n = 0; p = 0; }
    q = static_cast<int>(p) + (intercept ? 1 : 0);
    std::vector<double> D(static_cast<size_t>(n * q), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        int c = 0;
        if (intercept) D[static_cast<size_t>(i * q + c++)] = 1.0;
        for (int64_t j = 0; j < p; ++j) {
            double v = (X->rows <= 1 || X->cols <= 1) ? X->data[i]
                                                      : X->data[i * X->cols + j];
            D[static_cast<size_t>(i * q + c++)] = v;
        }
    }
    return D;
}

/* Weighted normal-equations solve: beta = (Dᵀ W D + λI)⁻¹ Dᵀ W z, with
 * per-row weights w (nullptr = ones) and ridge λ.  Also returns (DᵀWD+λI)⁻¹
 * in `covOut` (for coefficient standard errors). */
static std::vector<double> snorm_solve(const std::vector<double> &D, int64_t n, int q,
                                       const std::vector<double> &z,
                                       const double *w, double lambda,
                                       std::vector<double> &covOut) {
    std::vector<double> XtX(static_cast<size_t>(q * q), 0.0), Xtz(static_cast<size_t>(q), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        double wi = w ? w[i] : 1.0;
        for (int a = 0; a < q; ++a) {
            double Da = D[static_cast<size_t>(i * q + a)];
            Xtz[static_cast<size_t>(a)] += Da * wi * z[static_cast<size_t>(i)];
            for (int b = 0; b < q; ++b)
                XtX[static_cast<size_t>(a * q + b)] += Da * wi * D[static_cast<size_t>(i * q + b)];
        }
    }
    for (int a = 0; a < q; ++a) XtX[static_cast<size_t>(a * q + a)] += lambda;
    covOut = sinv_dense(XtX, q);
    std::vector<double> beta(static_cast<size_t>(q), 0.0);
    for (int a = 0; a < q; ++a)
        for (int b = 0; b < q; ++b)
            beta[static_cast<size_t>(a)] += covOut[static_cast<size_t>(a * q + b)] * Xtz[static_cast<size_t>(b)];
    return beta;
}

/* Thread-local hypothesis-test result.  Each test computes everything and
 * stashes it here; the per-output reader symbols (test_o2 / test_ci /
 * test_stats) pull the secondary outputs back.  `out1` is the first
 * MATLAB output (h for ttest…, p for ranksum…), `out2` the second. */
struct STestResult {
    double out1, out2;          /* first / second MATLAB return */
    double ci_lo, ci_hi;        /* confidence interval */
    double stat, df;            /* test statistic + degrees of freedom */
};
static thread_local STestResult g_stest = {0, 0, 0, 0, 0, 0};

/* Apply a column reduction f(col)->scalar over a matrix (1xncols result)
 * or the whole vector (1x1 result), matching matlab_var's shape rules. */
template <class F>
static matlab_mat *sstat_reduce(const matlab_mat *A, F f) {
    if (!A || !A->data || A->rows * A->cols == 0) return mat_alloc(0, 0);
    if (sstat_is_vec(A)) {
        matlab_mat *R = mat_alloc(1, 1);
        std::vector<double> c = sstat_col(A, 0);
        R->data[0] = f(c);
        return R;
    }
    matlab_mat *R = mat_alloc(1, A->cols);
    for (int64_t j = 0; j < A->cols; ++j) {
        std::vector<double> c = sstat_col(A, j);
        R->data[j] = f(c);
    }
    return R;
}

/* Element-wise apply g(xi; p1, p2) over x, preserving shape. */
template <class G>
static matlab_mat *sstat_elem(const matlab_mat *x, double p1, double p2, G g) {
    if (!x || !x->data) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    matlab_mat *R = mat_alloc(x->rows, x->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = g(x->data[i], p1, p2);
    return R;
}

/* corr(X) / corrcoef(X): Pearson correlation = cov normalised by std. */
static matlab_mat *sstat_corr_impl(matlab_mat *X) {
    matlab_mat *C = matlab_stats_cov(X);
    if (!C || C->rows == 0) return C;
    int64_t p = C->rows;
    std::vector<double> sd(static_cast<size_t>(p));
    for (int64_t i = 0; i < p; ++i) sd[static_cast<size_t>(i)] = sqrt(C->data[i * p + i]);
    matlab_mat *R = mat_alloc(p, p);
    for (int64_t aa = 0; aa < p; ++aa)
        for (int64_t bb = 0; bb < p; ++bb) {
            double dd = sd[static_cast<size_t>(aa)] * sd[static_cast<size_t>(bb)];
            R->data[aa * p + bb] = (dd > 0.0) ? C->data[aa * p + bb] / dd : (aa == bb ? 1.0 : 0.0);
        }
    return R;
}

/* ===== exported runtime entries =========================================== */

extern "C" {

/* ----- §1.1 descriptive reductions --------------------------------------- */

/* prctile(x, p): p may be a scalar or a vector of percentiles.  For a data
 * vector the result has the shape of p; for a matrix (scalar p) it is a
 * column-wise row vector. */
matlab_mat *matlab_stats_prctile(matlab_mat *x, matlab_mat *p) {
    if (!x || !x->data) return mat_alloc(0, 0);
    int64_t np = sstat_len(p);
    if (sstat_is_vec(x) && np > 1) {
        std::vector<double> s = sstat_col(x, 0);
        matlab_mat *R = mat_alloc(p->rows, p->cols);
        for (int64_t k = 0; k < np; ++k) R->data[k] = sstat_prctile_one(s, p->data[k]);
        return R;
    }
    double pv = sstat_sc(p, 50.0);
    return sstat_reduce(x, [&](std::vector<double> &c) { return sstat_prctile_one(c, pv); });
}

/* quantile(x, q) == prctile(x, 100*q). */
matlab_mat *matlab_stats_quantile(matlab_mat *x, matlab_mat *q) {
    if (!x || !x->data) return mat_alloc(0, 0);
    int64_t nq = sstat_len(q);
    if (sstat_is_vec(x) && nq > 1) {
        std::vector<double> s = sstat_col(x, 0);
        matlab_mat *R = mat_alloc(q->rows, q->cols);
        for (int64_t k = 0; k < nq; ++k) R->data[k] = sstat_prctile_one(s, q->data[k] * 100.0);
        return R;
    }
    double qv = sstat_sc(q, 0.5) * 100.0;
    return sstat_reduce(x, [&](std::vector<double> &c) { return sstat_prctile_one(c, qv); });
}

matlab_mat *matlab_stats_iqr(matlab_mat *x) {
    return sstat_reduce(x, [](std::vector<double> &c) {
        return sstat_prctile_one(c, 75.0) - sstat_prctile_one(c, 25.0);
    });
}

matlab_mat *matlab_stats_range(matlab_mat *x) {
    return sstat_reduce(x, [](std::vector<double> &c) {
        if (c.empty()) return 0.0;
        double lo = c[0], hi = c[0];
        for (double v : c) { if (v < lo) lo = v; if (v > hi) hi = v; }
        return hi - lo;
    });
}

matlab_mat *matlab_stats_mode(matlab_mat *x) {
    return sstat_reduce(x, [](std::vector<double> &c) {
        if (c.empty()) return 0.0;
        std::sort(c.begin(), c.end());          /* ties -> smallest value */
        double best = c[0], cur = c[0];
        int bestCnt = 1, curCnt = 1;
        for (size_t i = 1; i < c.size(); ++i) {
            if (c[i] == cur) { curCnt++; }
            else { cur = c[i]; curCnt = 1; }
            if (curCnt > bestCnt) { bestCnt = curCnt; best = cur; }
        }
        return best;
    });
}

matlab_mat *matlab_stats_skewness(matlab_mat *x) {
    return sstat_reduce(x, [](std::vector<double> &c) {
        double m2 = sstat_cmoment(c, 2), m3 = sstat_cmoment(c, 3);
        return (m2 > 0.0) ? m3 / pow(m2, 1.5) : 0.0;   /* biased (MATLAB default) */
    });
}

matlab_mat *matlab_stats_kurtosis(matlab_mat *x) {
    return sstat_reduce(x, [](std::vector<double> &c) {
        double m2 = sstat_cmoment(c, 2), m4 = sstat_cmoment(c, 4);
        return (m2 > 0.0) ? m4 / (m2 * m2) : 0.0;       /* not excess (normal=3) */
    });
}

matlab_mat *matlab_stats_geomean(matlab_mat *x) {
    return sstat_reduce(x, [](std::vector<double> &c) {
        if (c.empty()) return 0.0;
        double s = 0.0;
        for (double v : c) s += log(v);
        return exp(s / static_cast<double>(c.size()));
    });
}

matlab_mat *matlab_stats_harmmean(matlab_mat *x) {
    return sstat_reduce(x, [](std::vector<double> &c) {
        if (c.empty()) return 0.0;
        double s = 0.0;
        for (double v : c) s += 1.0 / v;
        return static_cast<double>(c.size()) / s;
    });
}

/* ----- §1.2 covariance / correlation ------------------------------------- */

/* cov(X): n x p data matrix -> p x p covariance.  A vector is treated as a
 * single column -> 1x1 variance.  Two-arg cov(x,y) is left to a follow-on. */
matlab_mat *matlab_stats_cov(matlab_mat *X) {
    if (!X || !X->data || X->rows * X->cols == 0) return mat_alloc(0, 0);
    int64_t n, p;
    std::vector<std::vector<double>> cols;
    if (sstat_is_vec(X)) { n = X->rows * X->cols; p = 1; cols.push_back(sstat_col(X, 0)); }
    else {
        n = X->rows; p = X->cols;
        for (int64_t j = 0; j < p; ++j) cols.push_back(sstat_col(X, j));
    }
    std::vector<double> mu(static_cast<size_t>(p));
    for (int64_t j = 0; j < p; ++j) mu[static_cast<size_t>(j)] = sstat_mean(cols[static_cast<size_t>(j)]);
    matlab_mat *C = mat_alloc(p, p);
    double denom = (n > 1) ? static_cast<double>(n - 1) : 1.0;
    for (int64_t aa = 0; aa < p; ++aa)
        for (int64_t bb = 0; bb < p; ++bb) {
            double s = 0.0;
            for (int64_t i = 0; i < n; ++i)
                s += (cols[static_cast<size_t>(aa)][static_cast<size_t>(i)] - mu[static_cast<size_t>(aa)]) *
                     (cols[static_cast<size_t>(bb)][static_cast<size_t>(i)] - mu[static_cast<size_t>(bb)]);
            C->data[aa * p + bb] = s / denom;
        }
    return C;
}

matlab_mat *matlab_stats_corr(matlab_mat *X)     { return sstat_corr_impl(X); }
matlab_mat *matlab_stats_corrcoef(matlab_mat *X) { return sstat_corr_impl(X); }

/* ----- §1.4-1.5 distributions: element-wise pdf / cdf / inv --------------- */

matlab_mat *matlab_stats_normpdf(matlab_mat *x, matlab_mat *mu, matlab_mat *sg) {
    return sstat_elem(x, sstat_sc(mu, 0.0), sstat_sc(sg, 1.0), [](double v, double m, double s) {
        double z = (v - m) / s;
        return exp(-0.5 * z * z) / (s * SQRT2PI);
    });
}
matlab_mat *matlab_stats_normcdf(matlab_mat *x, matlab_mat *mu, matlab_mat *sg) {
    return sstat_elem(x, sstat_sc(mu, 0.0), sstat_sc(sg, 1.0), [](double v, double m, double s) {
        return 0.5 * erfc(-(v - m) / (s * SQRT2));
    });
}
matlab_mat *matlab_stats_norminv(matlab_mat *p, matlab_mat *mu, matlab_mat *sg) {
    return sstat_elem(p, sstat_sc(mu, 0.0), sstat_sc(sg, 1.0), [](double v, double m, double s) {
        return m + s * sstat_norminv_std(v);
    });
}
/* 1-arg standard-normal convenience forms. */
matlab_mat *matlab_stats_normpdf1(matlab_mat *x) { return matlab_stats_normpdf(x, nullptr, nullptr); }
matlab_mat *matlab_stats_normcdf1(matlab_mat *x) { return matlab_stats_normcdf(x, nullptr, nullptr); }
matlab_mat *matlab_stats_norminv1(matlab_mat *p) { return matlab_stats_norminv(p, nullptr, nullptr); }

matlab_mat *matlab_stats_exppdf(matlab_mat *x, matlab_mat *mu) {
    return sstat_elem(x, sstat_sc(mu, 1.0), 0.0, [](double v, double m, double) {
        return (v < 0.0) ? 0.0 : exp(-v / m) / m;
    });
}
matlab_mat *matlab_stats_expcdf(matlab_mat *x, matlab_mat *mu) {
    return sstat_elem(x, sstat_sc(mu, 1.0), 0.0, [](double v, double m, double) {
        return (v < 0.0) ? 0.0 : 1.0 - exp(-v / m);
    });
}
matlab_mat *matlab_stats_expinv(matlab_mat *p, matlab_mat *mu) {
    return sstat_elem(p, sstat_sc(mu, 1.0), 0.0, [](double v, double m, double) {
        return -m * log(1.0 - v);
    });
}

matlab_mat *matlab_stats_unifpdf(matlab_mat *x, matlab_mat *a, matlab_mat *b) {
    return sstat_elem(x, sstat_sc(a, 0.0), sstat_sc(b, 1.0), [](double v, double lo, double hi) {
        return (v >= lo && v <= hi) ? 1.0 / (hi - lo) : 0.0;
    });
}
matlab_mat *matlab_stats_unifcdf(matlab_mat *x, matlab_mat *a, matlab_mat *b) {
    return sstat_elem(x, sstat_sc(a, 0.0), sstat_sc(b, 1.0), [](double v, double lo, double hi) {
        if (v <= lo) return 0.0;
        if (v >= hi) return 1.0;
        return (v - lo) / (hi - lo);
    });
}
matlab_mat *matlab_stats_unifinv(matlab_mat *p, matlab_mat *a, matlab_mat *b) {
    return sstat_elem(p, sstat_sc(a, 0.0), sstat_sc(b, 1.0), [](double v, double lo, double hi) {
        return lo + v * (hi - lo);
    });
}

/* ----- §1.6 distribution RNGs (rng-reproducible) ------------------------- */

matlab_mat *matlab_stats_normrnd(matlab_mat *mu, matlab_mat *sg, matlab_mat *m, matlab_mat *n) {
    double M = sstat_sc(m, 1.0), N = sstat_sc(n, 1.0);
    double mean = sstat_sc(mu, 0.0), sd = sstat_sc(sg, 1.0);
    matlab_mat *Z = matlab_randn(M, N);
    int64_t total = Z->rows * Z->cols;
    for (int64_t i = 0; i < total; ++i) Z->data[i] = mean + sd * Z->data[i];
    return Z;
}
matlab_mat *matlab_stats_unifrnd(matlab_mat *a, matlab_mat *b, matlab_mat *m, matlab_mat *n) {
    double M = sstat_sc(m, 1.0), N = sstat_sc(n, 1.0);
    double lo = sstat_sc(a, 0.0), hi = sstat_sc(b, 1.0);
    matlab_mat *U = matlab_rand(M, N);
    int64_t total = U->rows * U->cols;
    for (int64_t i = 0; i < total; ++i) U->data[i] = lo + (hi - lo) * U->data[i];
    return U;
}
matlab_mat *matlab_stats_exprnd(matlab_mat *mu, matlab_mat *m, matlab_mat *n) {
    double M = sstat_sc(m, 1.0), N = sstat_sc(n, 1.0), mean = sstat_sc(mu, 1.0);
    matlab_mat *U = matlab_rand(M, N);
    int64_t total = U->rows * U->cols;
    for (int64_t i = 0; i < total; ++i) {
        double u = U->data[i];
        if (u <= 0.0) u = 1e-300;
        U->data[i] = -mean * log(u);
    }
    return U;
}

/* ----- §1.8 fitdist cores ------------------------------------------------ *
 * Return a parameter vector; the classdef populate step copies it into a
 * ProbDistUnivParam.  Normal: [mu sigma] with the unbiased (n-1) std to
 * match MATLAB's reported ParameterValues.  Exponential: [mu]. */
matlab_mat *matlab_stats_fit_normal(matlab_mat *x) {
    matlab_mat *R = mat_alloc(1, 2);
    if (!x || !x->data || x->rows * x->cols == 0) { R->data[0] = 0.0; R->data[1] = 1.0; return R; }
    std::vector<double> c = sstat_col(x, 0);
    double mu = sstat_mean(c), s = 0.0;
    for (double v : c) s += (v - mu) * (v - mu);
    int64_t n = static_cast<int64_t>(c.size());
    R->data[0] = mu;
    R->data[1] = (n > 1) ? sqrt(s / static_cast<double>(n - 1)) : 0.0;
    return R;
}
matlab_mat *matlab_stats_fit_exponential(matlab_mat *x) {
    matlab_mat *R = mat_alloc(1, 1);
    std::vector<double> c = sstat_col(x, 0);
    R->data[0] = sstat_mean(c);
    return R;
}

/* ----- §1.7 distribution objects (makedist / fitdist) -------------------- *
 * Dist codes: 1=Normal, 2=Exponential, 3=Uniform.  Params are carried on
 * the ProbDistUnivParam classdef as `mu`/`sigma` (Normal: mean/std; Exp:
 * mu=mean, sigma unused; Uniform: mu=lower, sigma=upper).  fitdist_init is
 * the alloc-then-populate step (the Lowering allocs the shell, the runtime
 * computes the MLE and writes the fields); pd_* read the fields back and
 * dispatch to the numeric cores above. */

/* fitdist(x, dist): MLE-populate an already-allocated ProbDistUnivParam.
 * Returns a ptr (the populated object) per the alloc-then-populate ABI;
 * the Lowering discards it and keeps the ctor result. */
matlab_mat *matlab_stats_fitdist_init(matlab_obj *obj, matlab_mat *x, double distcode) {
    if (!obj) return mat_alloc(0, 0);
    int code = static_cast<int>(distcode);
    double a = 0.0, b = 1.0;
    if (code == 2) {                    /* Exponential */
        matlab_mat *p = matlab_stats_fit_exponential(x);
        a = p->data[0]; b = 0.0;
    } else {                            /* Normal (default) */
        code = (code == 0) ? 1 : code;
        matlab_mat *p = matlab_stats_fit_normal(x);
        a = p->data[0]; b = p->data[1];
    }
    matlab_obj_set_f64(obj, "DistCode", 8, static_cast<double>(code));
    matlab_obj_set_f64(obj, "mu", 2, a);
    matlab_obj_set_f64(obj, "sigma", 5, b);
    return mat_alloc(0, 0);
}

static int  spd_code(matlab_obj *pd) { return static_cast<int>(matlab_obj_get_f64(pd, "DistCode", 8)); }
static double spd_a(matlab_obj *pd)  { return matlab_obj_get_f64(pd, "mu", 2); }
static double spd_b(matlab_obj *pd)  { return matlab_obj_get_f64(pd, "sigma", 5); }

matlab_mat *matlab_stats_pd_pdf(matlab_obj *pd, matlab_mat *x) {
    if (!pd || !matlab_obj_is_known(pd)) return mat_alloc(0, 0);
    int code = spd_code(pd);
    if (code == 2) return matlab_stats_exppdf(x, matlab_mat_from_scalar(spd_a(pd)));
    if (code == 3) return matlab_stats_unifpdf(x, matlab_mat_from_scalar(spd_a(pd)),
                                               matlab_mat_from_scalar(spd_b(pd)));
    return matlab_stats_normpdf(x, matlab_mat_from_scalar(spd_a(pd)),
                                matlab_mat_from_scalar(spd_b(pd)));
}
matlab_mat *matlab_stats_pd_cdf(matlab_obj *pd, matlab_mat *x) {
    if (!pd || !matlab_obj_is_known(pd)) return mat_alloc(0, 0);
    int code = spd_code(pd);
    if (code == 2) return matlab_stats_expcdf(x, matlab_mat_from_scalar(spd_a(pd)));
    if (code == 3) return matlab_stats_unifcdf(x, matlab_mat_from_scalar(spd_a(pd)),
                                               matlab_mat_from_scalar(spd_b(pd)));
    return matlab_stats_normcdf(x, matlab_mat_from_scalar(spd_a(pd)),
                                matlab_mat_from_scalar(spd_b(pd)));
}
matlab_mat *matlab_stats_pd_icdf(matlab_obj *pd, matlab_mat *p) {
    if (!pd || !matlab_obj_is_known(pd)) return mat_alloc(0, 0);
    int code = spd_code(pd);
    if (code == 2) return matlab_stats_expinv(p, matlab_mat_from_scalar(spd_a(pd)));
    if (code == 3) return matlab_stats_unifinv(p, matlab_mat_from_scalar(spd_a(pd)),
                                               matlab_mat_from_scalar(spd_b(pd)));
    return matlab_stats_norminv(p, matlab_mat_from_scalar(spd_a(pd)),
                                matlab_mat_from_scalar(spd_b(pd)));
}
matlab_mat *matlab_stats_pd_random(matlab_obj *pd, matlab_mat *m, matlab_mat *n) {
    if (!pd || !matlab_obj_is_known(pd)) return mat_alloc(0, 0);
    int code = spd_code(pd);
    if (code == 2) return matlab_stats_exprnd(matlab_mat_from_scalar(spd_a(pd)), m, n);
    if (code == 3) return matlab_stats_unifrnd(matlab_mat_from_scalar(spd_a(pd)),
                                               matlab_mat_from_scalar(spd_b(pd)), m, n);
    return matlab_stats_normrnd(matlab_mat_from_scalar(spd_a(pd)),
                                matlab_mat_from_scalar(spd_b(pd)), m, n);
}

/* ===== Tier-2 — hypothesis tests + ANOVA ================================= *
 * Each test stashes its full result in g_stest and returns the first
 * MATLAB output; the splitter wires the secondary outputs to the readers
 * below.  Default significance level alpha = 0.05. */

static const double STEST_ALPHA = 0.05;

/* one-sample / paired t-test.  ttest(x): mean=0; ttest(x,m) m scalar:
 * mean=m; ttest(x,y) y same-size: paired (test mean(x-y)=0). */
double matlab_stats_ttest(matlab_mat *x, matlab_mat *second) {
    std::vector<double> d = sstat_col(x, 0);
    double m0 = 0.0;
    if (second && sstat_len(second) == 1) {
        m0 = second->data[0];                       /* ttest(x, m) */
    } else if (second && sstat_len(second) == sstat_len(x)) {
        std::vector<double> y = sstat_col(second, 0);  /* paired */
        for (size_t i = 0; i < d.size(); ++i) d[i] -= y[i];
    }
    int n = static_cast<int>(d.size());
    double mu = sstat_mean(d), s = sqrt(sstat_cmoment(d, 2) * n / (n - 1.0));
    double se = s / sqrt(static_cast<double>(n)), df = n - 1.0;
    double t = (mu - m0) / se;
    double p = 2.0 * (1.0 - stcdf(fabs(t), df));
    double tc = stinv(1.0 - STEST_ALPHA / 2.0, df);
    g_stest.out1 = (p < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.out2 = p;
    g_stest.ci_lo = (mu - m0) - tc * se + m0;   /* CI on the mean */
    g_stest.ci_hi = (mu - m0) + tc * se + m0;
    g_stest.stat = t; g_stest.df = df;
    return g_stest.out1;
}
double matlab_stats_ttest1(matlab_mat *x) { return matlab_stats_ttest(x, nullptr); }

/* two-sample pooled t-test. */
double matlab_stats_ttest2(matlab_mat *x, matlab_mat *y) {
    std::vector<double> a = sstat_col(x, 0), b = sstat_col(y, 0);
    int n1 = static_cast<int>(a.size()), n2 = static_cast<int>(b.size());
    double m1 = sstat_mean(a), m2 = sstat_mean(b);
    double v1 = sstat_cmoment(a, 2) * n1 / (n1 - 1.0);
    double v2 = sstat_cmoment(b, 2) * n2 / (n2 - 1.0);
    double sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2.0);
    double se = sqrt(sp2 * (1.0 / n1 + 1.0 / n2)), df = n1 + n2 - 2.0;
    double t = (m1 - m2) / se;
    double p = 2.0 * (1.0 - stcdf(fabs(t), df));
    double tc = stinv(1.0 - STEST_ALPHA / 2.0, df);
    g_stest.out1 = (p < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.out2 = p;
    g_stest.ci_lo = (m1 - m2) - tc * se;
    g_stest.ci_hi = (m1 - m2) + tc * se;
    g_stest.stat = t; g_stest.df = df;
    return g_stest.out1;
}

/* two-sample F-test for equal variances. */
double matlab_stats_vartest2(matlab_mat *x, matlab_mat *y) {
    std::vector<double> a = sstat_col(x, 0), b = sstat_col(y, 0);
    int n1 = static_cast<int>(a.size()), n2 = static_cast<int>(b.size());
    double v1 = sstat_cmoment(a, 2) * n1 / (n1 - 1.0);
    double v2 = sstat_cmoment(b, 2) * n2 / (n2 - 1.0);
    double F = v1 / v2, d1 = n1 - 1.0, d2 = n2 - 1.0;
    double cdf = sfcdf(F, d1, d2);
    double p = 2.0 * std::min(cdf, 1.0 - cdf);
    g_stest.out1 = (p < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.out2 = p; g_stest.stat = F; g_stest.df = d1;
    g_stest.ci_lo = 0.0; g_stest.ci_hi = 0.0;
    return g_stest.out1;
}

/* one-sample z-test (known sigma): ztest(x, m, sigma). */
double matlab_stats_ztest(matlab_mat *x, matlab_mat *m, matlab_mat *sg) {
    std::vector<double> a = sstat_col(x, 0);
    int n = static_cast<int>(a.size());
    double mu = sstat_mean(a), m0 = sstat_sc(m, 0.0), sigma = sstat_sc(sg, 1.0);
    double se = sigma / sqrt(static_cast<double>(n));
    double z = (mu - m0) / se;
    double p = 2.0 * (1.0 - 0.5 * erfc(-fabs(z) / SQRT2));
    double zc = sstat_norminv_std(1.0 - STEST_ALPHA / 2.0);
    g_stest.out1 = (p < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.out2 = p;
    g_stest.ci_lo = mu - zc * se; g_stest.ci_hi = mu + zc * se;
    g_stest.stat = z; g_stest.df = 0.0;
    return g_stest.out1;
}

/* one-sample Kolmogorov-Smirnov test vs the standard normal. */
double matlab_stats_kstest(matlab_mat *x) {
    std::vector<double> a = sstat_col(x, 0);
    std::sort(a.begin(), a.end());
    int n = static_cast<int>(a.size());
    double D = 0.0;
    for (int i = 0; i < n; ++i) {
        double F = 0.5 * erfc(-a[static_cast<size_t>(i)] / SQRT2);
        double d1 = fabs((i + 1.0) / n - F);
        double d2 = fabs(F - static_cast<double>(i) / n);
        D = std::max(D, std::max(d1, d2));
    }
    /* asymptotic Kolmogorov distribution: Q(lambda). */
    double en = sqrt(static_cast<double>(n));
    double lam = (en + 0.12 + 0.11 / en) * D;
    double q = 0.0;
    for (int j = 1; j <= 100; ++j) {
        double term = 2.0 * ((j % 2) ? 1.0 : -1.0) * exp(-2.0 * j * j * lam * lam);
        q += term;
        if (fabs(term) < 1e-12) break;
    }
    if (q < 0.0) q = 0.0; if (q > 1.0) q = 1.0;
    g_stest.out1 = (q < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.out2 = q; g_stest.stat = D; g_stest.df = 0.0;
    g_stest.ci_lo = 0.0; g_stest.ci_hi = 0.0;
    return g_stest.out1;
}

/* rank helper: average ranks of `v` (ties shared). */
static std::vector<double> srank(const std::vector<double> &v) {
    int n = static_cast<int>(v.size());
    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return v[static_cast<size_t>(a)] < v[static_cast<size_t>(b)]; });
    std::vector<double> r(static_cast<size_t>(n));
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n &&
               v[static_cast<size_t>(idx[static_cast<size_t>(j + 1)])] ==
               v[static_cast<size_t>(idx[static_cast<size_t>(i)])]) j++;
        double avg = (i + j) / 2.0 + 1.0;
        for (int k = i; k <= j; ++k) r[static_cast<size_t>(idx[static_cast<size_t>(k)])] = avg;
        i = j + 1;
    }
    return r;
}

/* Wilcoxon rank-sum (Mann-Whitney), normal approximation.  Returns p
 * first (MATLAB order [p,h,stats]). */
double matlab_stats_ranksum(matlab_mat *x, matlab_mat *y) {
    std::vector<double> a = sstat_col(x, 0), b = sstat_col(y, 0);
    int n1 = static_cast<int>(a.size()), n2 = static_cast<int>(b.size());
    std::vector<double> all = a; all.insert(all.end(), b.begin(), b.end());
    std::vector<double> r = srank(all);
    double W = 0.0;
    for (int i = 0; i < n1; ++i) W += r[static_cast<size_t>(i)];
    double mu = n1 * (n1 + n2 + 1.0) / 2.0;
    double sd = sqrt(n1 * n2 * (n1 + n2 + 1.0) / 12.0);
    double z = (W - mu) / sd;
    double p = 2.0 * (1.0 - 0.5 * erfc(-fabs(z) / SQRT2));
    g_stest.out1 = p;
    g_stest.out2 = (p < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.stat = z; g_stest.df = 0.0;
    g_stest.ci_lo = 0.0; g_stest.ci_hi = 0.0;
    return g_stest.out1;
}

/* Wilcoxon signed-rank (paired), normal approximation.  Returns p first. */
double matlab_stats_signrank(matlab_mat *x, matlab_mat *y) {
    std::vector<double> a = sstat_col(x, 0);
    std::vector<double> diff;
    if (y && sstat_len(y) == sstat_len(x)) {
        std::vector<double> b = sstat_col(y, 0);
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) diff.push_back(a[i] - b[i]);
    } else {
        for (double v : a) if (v != 0.0) diff.push_back(v);
    }
    int n = static_cast<int>(diff.size());
    std::vector<double> mag(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) mag[static_cast<size_t>(i)] = fabs(diff[static_cast<size_t>(i)]);
    std::vector<double> r = srank(mag);
    double Wp = 0.0;
    for (int i = 0; i < n; ++i) if (diff[static_cast<size_t>(i)] > 0.0) Wp += r[static_cast<size_t>(i)];
    double mu = n * (n + 1.0) / 4.0;
    double sd = sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 24.0);
    double z = (Wp - mu) / sd;
    double p = 2.0 * (1.0 - 0.5 * erfc(-fabs(z) / SQRT2));
    g_stest.out1 = p;
    g_stest.out2 = (p < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.stat = z; g_stest.df = 0.0;
    g_stest.ci_lo = 0.0; g_stest.ci_hi = 0.0;
    return g_stest.out1;
}

/* sign test (paired), normal approximation.  Returns p first. */
double matlab_stats_signtest(matlab_mat *x, matlab_mat *y) {
    std::vector<double> a = sstat_col(x, 0);
    int pos = 0, tot = 0;
    if (y && sstat_len(y) == sstat_len(x)) {
        std::vector<double> b = sstat_col(y, 0);
        for (size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            if (d != 0.0) { tot++; if (d > 0.0) pos++; }
        }
    } else {
        for (double v : a) if (v != 0.0) { tot++; if (v > 0.0) pos++; }
    }
    double mu = tot / 2.0, sd = sqrt(tot / 4.0);
    double z = (sd > 0.0) ? (pos - mu) / sd : 0.0;
    double p = 2.0 * (1.0 - 0.5 * erfc(-fabs(z) / SQRT2));
    if (p > 1.0) p = 1.0;
    g_stest.out1 = p;
    g_stest.out2 = (p < STEST_ALPHA) ? 1.0 : 0.0;
    g_stest.stat = static_cast<double>(pos); g_stest.df = 0.0;
    g_stest.ci_lo = 0.0; g_stest.ci_hi = 0.0;
    return g_stest.out1;
}

/* one-way ANOVA: X columns are groups (balanced).  Returns p. */
double matlab_stats_anova1(matlab_mat *X) {
    if (!X || !X->data) return 1.0;
    int64_t k, n;            /* k groups, n per group */
    std::vector<std::vector<double>> g;
    if (sstat_is_vec(X)) { g.push_back(sstat_col(X, 0)); k = 1; n = X->rows * X->cols; }
    else { k = X->cols; n = X->rows; for (int64_t j = 0; j < k; ++j) g.push_back(sstat_col(X, j)); }
    double grand = 0.0; int64_t N = 0;
    for (auto &c : g) { for (double v : c) grand += v; N += static_cast<int64_t>(c.size()); }
    grand /= static_cast<double>(N);
    double ssb = 0.0, ssw = 0.0;
    for (auto &c : g) {
        double gm = sstat_mean(c);
        ssb += static_cast<double>(c.size()) * (gm - grand) * (gm - grand);
        for (double v : c) ssw += (v - gm) * (v - gm);
    }
    double dfb = k - 1.0, dfw = N - k;
    double F = (ssb / dfb) / (ssw / dfw);
    double p = 1.0 - sfcdf(F, dfb, dfw);
    g_stest.out1 = p; g_stest.out2 = F; g_stest.stat = F; g_stest.df = dfb;
    return p;
}

/* ===== Tier-3 — regression ============================================== *
 * LinearModel / GeneralizedLinearModel share one ProbDist-style classdef
 * carrying Beta + a Coefficients matrix (Estimate/SE/tStat/pValue) + R² /
 * RMSE.  ModelType: 1 = OLS, 2 = logistic GLM.  fit*_init is the alloc-
 * then-populate step; lm_predict reads Beta + ModelType back. */

/* Populate a LinearModel obj with an OLS fit of y on the columns of X
 * (intercept auto-added). */
matlab_mat *matlab_stats_fitlm_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (!obj) return mat_alloc(0, 0);
    int64_t n, p; int q;
    std::vector<double> D = sdesign(X, n, p, q, true);
    std::vector<double> yy = sstat_col(y, 0);
    std::vector<double> cov;
    std::vector<double> beta = snorm_solve(D, n, q, yy, nullptr, 0.0, cov);
    /* residuals, R², adjusted R² */
    double ybar = sstat_mean(yy), sse = 0.0, sst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double yhat = 0.0;
        for (int a = 0; a < q; ++a) yhat += D[static_cast<size_t>(i * q + a)] * beta[static_cast<size_t>(a)];
        double e = yy[static_cast<size_t>(i)] - yhat;
        sse += e * e; sst += (yy[static_cast<size_t>(i)] - ybar) * (yy[static_cast<size_t>(i)] - ybar);
    }
    double dfe = static_cast<double>(n - q);
    double sigma2 = (dfe > 0) ? sse / dfe : 0.0;
    double r2 = (sst > 0) ? 1.0 - sse / sst : 0.0;
    double r2adj = (sst > 0 && dfe > 0) ? 1.0 - (sse / dfe) / (sst / (n - 1.0)) : 0.0;
    /* coefficient table: Estimate | SE | tStat | pValue */
    matlab_mat *B = mat_alloc(q, 1);
    matlab_mat *Cf = mat_alloc(q, 4);
    for (int a = 0; a < q; ++a) {
        double se = sqrt(sigma2 * cov[static_cast<size_t>(a * q + a)]);
        double t = (se > 0) ? beta[static_cast<size_t>(a)] / se : 0.0;
        double pv = 2.0 * (1.0 - stcdf(fabs(t), dfe));
        B->data[a] = beta[static_cast<size_t>(a)];
        Cf->data[a * 4 + 0] = beta[static_cast<size_t>(a)];
        Cf->data[a * 4 + 1] = se;
        Cf->data[a * 4 + 2] = t;
        Cf->data[a * 4 + 3] = pv;
    }
    matlab_obj_set_mat(obj, "Beta", 4, B);
    matlab_obj_set_mat(obj, "Coefficients", 12, Cf);
    matlab_obj_set_f64(obj, "Rsquared", 8, r2);
    matlab_obj_set_f64(obj, "RsquaredAdj", 11, r2adj);
    matlab_obj_set_f64(obj, "RMSE", 4, sqrt(sigma2));
    matlab_obj_set_f64(obj, "ModelType", 9, 1.0);
    matlab_obj_set_f64(obj, "NumPred", 7, static_cast<double>(p));
    return mat_alloc(0, 0);
}

/* Populate a LinearModel obj with a logistic-GLM (IRLS) fit. */
matlab_mat *matlab_stats_fitglm_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (!obj) return mat_alloc(0, 0);
    int64_t n, p; int q;
    std::vector<double> D = sdesign(X, n, p, q, true);
    std::vector<double> yy = sstat_col(y, 0);
    std::vector<double> beta(static_cast<size_t>(q), 0.0), cov;
    for (int iter = 0; iter < 30; ++iter) {
        std::vector<double> wv(static_cast<size_t>(n)), z(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            double eta = 0.0;
            for (int a = 0; a < q; ++a) eta += D[static_cast<size_t>(i * q + a)] * beta[static_cast<size_t>(a)];
            double mu = 1.0 / (1.0 + exp(-eta));
            double w = mu * (1.0 - mu); if (w < 1e-9) w = 1e-9;
            wv[static_cast<size_t>(i)] = w;
            z[static_cast<size_t>(i)] = eta + (yy[static_cast<size_t>(i)] - mu) / w;
        }
        beta = snorm_solve(D, n, q, z, wv.data(), 0.0, cov);
    }
    matlab_mat *B = mat_alloc(q, 1);
    for (int a = 0; a < q; ++a) B->data[a] = beta[static_cast<size_t>(a)];
    matlab_obj_set_mat(obj, "Beta", 4, B);
    matlab_obj_set_f64(obj, "ModelType", 9, 2.0);
    matlab_obj_set_f64(obj, "NumPred", 7, static_cast<double>(p));
    return mat_alloc(0, 0);
}

/* predict(mdl, Xnew): [1 Xnew]·Beta, with the logit link for ModelType 2. */
matlab_mat *matlab_stats_lm_predict(matlab_obj *mdl, matlab_mat *Xnew) {
    if (!mdl || !matlab_obj_is_known(mdl) || !Xnew || !Xnew->data) return mat_alloc(0, 0);
    matlab_mat *B = matlab_obj_get_mat(mdl, "Beta", 4);
    if (!B) return mat_alloc(0, 0);
    int q = static_cast<int>(B->rows * B->cols);
    int pexp = q - 1;            /* feature count the model expects */
    int type = static_cast<int>(matlab_obj_get_f64(mdl, "ModelType", 9));
    /* Resolve Xnew shape against the known feature count: a row matching
     * pexp is a single observation; an n×pexp matrix is n observations;
     * a plain vector with pexp==1 is n single-feature observations. */
    int64_t nobs; int p;
    int64_t rows = Xnew->rows, cols = Xnew->cols, tot = rows * cols;
    const double *xd = Xnew->data;
    int64_t stride;
    if (pexp == 1) { nobs = tot; p = 1; stride = 1; }
    else if (cols == pexp) { nobs = rows; p = pexp; stride = cols; }
    else if (tot == pexp) { nobs = 1; p = pexp; stride = 0; }
    else return mat_alloc(0, 0);
    matlab_mat *R = mat_alloc(nobs, 1);
    for (int64_t i = 0; i < nobs; ++i) {
        double eta = B->data[0];                       /* intercept */
        for (int j = 0; j < p; ++j) {
            double xv = (pexp == 1) ? xd[i] : xd[i * stride + j];
            eta += xv * B->data[j + 1];
        }
        R->data[i] = (type == 2) ? 1.0 / (1.0 + exp(-eta)) : eta;
    }
    return R;
}

/* ridge(y, X, k): ridge regression on centered data (no intercept penalty).
 * Returns the p slope coefficients on the original scale. */
matlab_mat *matlab_stats_ridge(matlab_mat *y, matlab_mat *X, matlab_mat *kk) {
    int64_t n, p; int q;
    std::vector<double> D = sdesign(X, n, p, q, false);   /* no intercept col */
    std::vector<double> yy = sstat_col(y, 0);
    double k = sstat_sc(kk, 0.0);
    /* center columns + y */
    double ybar = sstat_mean(yy);
    std::vector<double> yc(yy);
    for (auto &v : yc) v -= ybar;
    std::vector<double> cmean(static_cast<size_t>(q), 0.0);
    for (int64_t i = 0; i < n; ++i) for (int a = 0; a < q; ++a) cmean[static_cast<size_t>(a)] += D[static_cast<size_t>(i * q + a)];
    for (int a = 0; a < q; ++a) cmean[static_cast<size_t>(a)] /= static_cast<double>(n);
    std::vector<double> Dc(D);
    for (int64_t i = 0; i < n; ++i) for (int a = 0; a < q; ++a) Dc[static_cast<size_t>(i * q + a)] -= cmean[static_cast<size_t>(a)];
    std::vector<double> cov;
    std::vector<double> beta = snorm_solve(Dc, n, q, yc, nullptr, k, cov);
    matlab_mat *R = mat_alloc(q, 1);
    for (int a = 0; a < q; ++a) R->data[a] = beta[static_cast<size_t>(a)];
    return R;
}

/* regress(y, X): OLS where X already carries any intercept column.
 * Returns the coefficient vector; the [b,bint,r,rint,stats] form stashes
 * the rest in g_reg for the splitter readers. */
struct SRegResult { matlab_mat *b, *r, *stats; };
static thread_local SRegResult g_reg = {nullptr, nullptr, nullptr};

matlab_mat *matlab_stats_regress(matlab_mat *y, matlab_mat *X) {
    int64_t n, p; int q;
    std::vector<double> D = sdesign(X, n, p, q, false);   /* X has its own ones col */
    std::vector<double> yy = sstat_col(y, 0), cov;
    std::vector<double> beta = snorm_solve(D, n, q, yy, nullptr, 0.0, cov);
    double ybar = sstat_mean(yy), sse = 0.0, sst = 0.0;
    matlab_mat *res = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) {
        double yhat = 0.0;
        for (int a = 0; a < q; ++a) yhat += D[static_cast<size_t>(i * q + a)] * beta[static_cast<size_t>(a)];
        double e = yy[static_cast<size_t>(i)] - yhat;
        res->data[i] = e;
        sse += e * e; sst += (yy[static_cast<size_t>(i)] - ybar) * (yy[static_cast<size_t>(i)] - ybar);
    }
    double dfe = static_cast<double>(n - q), dfr = static_cast<double>(q - 1);
    double r2 = (sst > 0) ? 1.0 - sse / sst : 0.0;
    double F = (dfr > 0 && sse > 0) ? ((sst - sse) / dfr) / (sse / dfe) : 0.0;
    double pF = 1.0 - sfcdf(F, dfr, dfe);
    matlab_mat *b = mat_alloc(q, 1);
    for (int a = 0; a < q; ++a) b->data[a] = beta[static_cast<size_t>(a)];
    matlab_mat *st = mat_alloc(1, 4);
    st->data[0] = r2; st->data[1] = F; st->data[2] = pF; st->data[3] = sse / dfe;
    g_reg.b = b; g_reg.r = res; g_reg.stats = st;
    return b;
}
matlab_mat *matlab_stats_reg_r(void)     { return g_reg.r     ? g_reg.r     : mat_alloc(0, 0); }
matlab_mat *matlab_stats_reg_stats(void) { return g_reg.stats ? g_reg.stats : mat_alloc(0, 0); }

/* ===== Tier-4 — unsupervised learning (PCA + clustering) ================ */

/* Symmetric-matrix eigensolver (cyclic Jacobi).  A is n×n row-major
 * (overwritten); fills eigenvalues `ev` and eigenvectors `V` (n×n,
 * column k = k-th eigenvector), then sorts both by descending eigenvalue.
 * n is small here (number of features), so Jacobi is more than adequate. */
static void sjacobi_eig(std::vector<double> A, int n,
                        std::vector<double> &ev, std::vector<double> &V) {
    V.assign(static_cast<size_t>(n * n), 0.0);
    for (int i = 0; i < n; ++i) V[static_cast<size_t>(i * n + i)] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) for (int qd = p + 1; qd < n; ++qd)
            off += A[static_cast<size_t>(p * n + qd)] * A[static_cast<size_t>(p * n + qd)];
        if (off < 1e-300) break;
        for (int p = 0; p < n; ++p) for (int qd = p + 1; qd < n; ++qd) {
            double apq = A[static_cast<size_t>(p * n + qd)];
            if (fabs(apq) < 1e-300) continue;
            double app = A[static_cast<size_t>(p * n + p)], aqq = A[static_cast<size_t>(qd * n + qd)];
            double phi = 0.5 * atan2(2.0 * apq, aqq - app);
            double c = cos(phi), s = sin(phi);
            for (int i = 0; i < n; ++i) {
                double aip = A[static_cast<size_t>(i * n + p)], aiq = A[static_cast<size_t>(i * n + qd)];
                A[static_cast<size_t>(i * n + p)] = c * aip - s * aiq;
                A[static_cast<size_t>(i * n + qd)] = s * aip + c * aiq;
            }
            for (int i = 0; i < n; ++i) {
                double api = A[static_cast<size_t>(p * n + i)], aqi = A[static_cast<size_t>(qd * n + i)];
                A[static_cast<size_t>(p * n + i)] = c * api - s * aqi;
                A[static_cast<size_t>(qd * n + i)] = s * api + c * aqi;
            }
            for (int i = 0; i < n; ++i) {
                double vip = V[static_cast<size_t>(i * n + p)], viq = V[static_cast<size_t>(i * n + qd)];
                V[static_cast<size_t>(i * n + p)] = c * vip - s * viq;
                V[static_cast<size_t>(i * n + qd)] = s * vip + c * viq;
            }
        }
    }
    ev.assign(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) ev[static_cast<size_t>(i)] = A[static_cast<size_t>(i * n + i)];
    /* sort eigenpairs by descending eigenvalue */
    std::vector<int> ord(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) ord[static_cast<size_t>(i)] = i;
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b){ return ev[static_cast<size_t>(a)] > ev[static_cast<size_t>(b)]; });
    std::vector<double> ev2(static_cast<size_t>(n)), V2(static_cast<size_t>(n * n));
    for (int k = 0; k < n; ++k) {
        ev2[static_cast<size_t>(k)] = ev[static_cast<size_t>(ord[static_cast<size_t>(k)])];
        for (int i = 0; i < n; ++i) V2[static_cast<size_t>(i * n + k)] = V[static_cast<size_t>(i * n + ord[static_cast<size_t>(k)])];
    }
    ev = ev2; V = V2;
}

/* Read a data matrix into n (observations) × p (features), flat row-major.
 * The ML convention is rows = observations, columns = features: a column
 * vector (k×1) is k observations of 1 feature; a row (1×k) is 1
 * observation of k features.  (Distinct from the descriptive-stats vector
 * convention in sstat_col, which treats any vector as one sample.) */
static std::vector<double> sdata(const matlab_mat *X, int64_t &n, int64_t &p) {
    if (!X || !X->data || X->rows * X->cols == 0) { n = 0; p = 0; return {}; }
    n = X->rows; p = X->cols;
    std::vector<double> D(static_cast<size_t>(n * p));
    for (int64_t i = 0; i < n * p; ++i) D[static_cast<size_t>(i)] = X->data[i];
    return D;
}

/* pca: thread-local result (coeff p×p, score n×p, latent p×1, explained p×1). */
struct SPcaResult { matlab_mat *coeff, *score, *latent, *explained; };
static thread_local SPcaResult g_pca = {nullptr, nullptr, nullptr, nullptr};

matlab_mat *matlab_stats_pca(matlab_mat *X) {
    int64_t n, p;
    std::vector<double> D = sdata(X, n, p);
    if (n == 0 || p == 0) return mat_alloc(0, 0);
    /* center columns */
    std::vector<double> mu(static_cast<size_t>(p), 0.0);
    for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j) mu[static_cast<size_t>(j)] += D[static_cast<size_t>(i * p + j)];
    for (int64_t j = 0; j < p; ++j) mu[static_cast<size_t>(j)] /= static_cast<double>(n);
    for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j) D[static_cast<size_t>(i * p + j)] -= mu[static_cast<size_t>(j)];
    /* covariance p×p */
    std::vector<double> C(static_cast<size_t>(p * p), 0.0);
    for (int64_t a = 0; a < p; ++a) for (int64_t b = 0; b < p; ++b) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += D[static_cast<size_t>(i * p + a)] * D[static_cast<size_t>(i * p + b)];
        C[static_cast<size_t>(a * p + b)] = s / (n - 1.0);
    }
    std::vector<double> ev, V;
    sjacobi_eig(C, static_cast<int>(p), ev, V);
    matlab_mat *coeff = mat_alloc(p, p);
    for (int64_t i = 0; i < p * p; ++i) coeff->data[i] = V[static_cast<size_t>(i)];
    matlab_mat *score = mat_alloc(n, p);
    for (int64_t i = 0; i < n; ++i) for (int64_t k = 0; k < p; ++k) {
        double s = 0.0;
        for (int64_t j = 0; j < p; ++j) s += D[static_cast<size_t>(i * p + j)] * V[static_cast<size_t>(j * p + k)];
        score->data[i * p + k] = s;
    }
    matlab_mat *latent = mat_alloc(p, 1);
    double tot = 0.0;
    for (int64_t k = 0; k < p; ++k) { latent->data[k] = ev[static_cast<size_t>(k)]; tot += ev[static_cast<size_t>(k)]; }
    matlab_mat *expl = mat_alloc(p, 1);
    for (int64_t k = 0; k < p; ++k) expl->data[k] = (tot > 0) ? 100.0 * ev[static_cast<size_t>(k)] / tot : 0.0;
    g_pca.coeff = coeff; g_pca.score = score; g_pca.latent = latent; g_pca.explained = expl;
    return coeff;
}
matlab_mat *matlab_stats_pca_score(void)     { return g_pca.score     ? g_pca.score     : mat_alloc(0, 0); }
matlab_mat *matlab_stats_pca_latent(void)    { return g_pca.latent    ? g_pca.latent    : mat_alloc(0, 0); }
matlab_mat *matlab_stats_pca_explained(void) { return g_pca.explained ? g_pca.explained : mat_alloc(0, 0); }
matlab_mat *matlab_stats_pca_empty(void)     { return mat_alloc(0, 0); }   /* the tsquared slot */

/* kmeans: Lloyd's algorithm + k-means++ init over the shared PRNG. */
struct SKmResult { matlab_mat *C, *sumd, *D; };
static thread_local SKmResult g_km = {nullptr, nullptr, nullptr};

matlab_mat *matlab_stats_kmeans(matlab_mat *X, matlab_mat *kk) {
    int64_t n, p;
    std::vector<double> D = sdata(X, n, p);
    int k = static_cast<int>(sstat_sc(kk, 1.0));
    if (n == 0 || k < 1) return mat_alloc(0, 0);
    if (k > static_cast<int>(n)) k = static_cast<int>(n);
    auto dist2 = [&](int64_t i, const std::vector<double> &cen, int c) {
        double s = 0.0;
        for (int64_t j = 0; j < p; ++j) {
            double d = D[static_cast<size_t>(i * p + j)] - cen[static_cast<size_t>(c * p + j)];
            s += d * d;
        }
        return s;
    };
    /* k-means++ seeding */
    std::vector<double> cen(static_cast<size_t>(k * p), 0.0);
    matlab_mat *u0 = matlab_rand(1, 1);
    int64_t first = static_cast<int64_t>(u0->data[0] * n); if (first >= n) first = n - 1;
    for (int64_t j = 0; j < p; ++j) cen[static_cast<size_t>(j)] = D[static_cast<size_t>(first * p + j)];
    std::vector<double> dmin(static_cast<size_t>(n), 1e300);
    for (int c = 1; c < k; ++c) {
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double d = dist2(i, cen, c - 1);
            if (d < dmin[static_cast<size_t>(i)]) dmin[static_cast<size_t>(i)] = d;
            sum += dmin[static_cast<size_t>(i)];
        }
        matlab_mat *u = matlab_rand(1, 1);
        double target = u->data[0] * sum, acc = 0.0; int64_t pick = n - 1;
        for (int64_t i = 0; i < n; ++i) { acc += dmin[static_cast<size_t>(i)]; if (acc >= target) { pick = i; break; } }
        for (int64_t j = 0; j < p; ++j) cen[static_cast<size_t>(c * p + j)] = D[static_cast<size_t>(pick * p + j)];
    }
    /* Lloyd iterations */
    std::vector<int> idx(static_cast<size_t>(n), 0);
    for (int iter = 0; iter < 200; ++iter) {
        bool changed = false;
        for (int64_t i = 0; i < n; ++i) {
            int best = 0; double bd = 1e300;
            for (int c = 0; c < k; ++c) { double d = dist2(i, cen, c); if (d < bd) { bd = d; best = c; } }
            if (idx[static_cast<size_t>(i)] != best) { idx[static_cast<size_t>(i)] = best; changed = true; }
        }
        std::vector<double> nc(static_cast<size_t>(k * p), 0.0);
        std::vector<int> cnt(static_cast<size_t>(k), 0);
        for (int64_t i = 0; i < n; ++i) {
            int c = idx[static_cast<size_t>(i)]; cnt[static_cast<size_t>(c)]++;
            for (int64_t j = 0; j < p; ++j) nc[static_cast<size_t>(c * p + j)] += D[static_cast<size_t>(i * p + j)];
        }
        for (int c = 0; c < k; ++c) if (cnt[static_cast<size_t>(c)] > 0)
            for (int64_t j = 0; j < p; ++j) cen[static_cast<size_t>(c * p + j)] = nc[static_cast<size_t>(c * p + j)] / cnt[static_cast<size_t>(c)];
        if (!changed && iter > 0) break;
    }
    matlab_mat *idxm = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) idxm->data[i] = idx[static_cast<size_t>(i)] + 1.0;   /* 1-based */
    matlab_mat *Cm = mat_alloc(k, p);
    for (int64_t i = 0; i < k * p; ++i) Cm->data[i] = cen[static_cast<size_t>(i)];
    matlab_mat *Dm = mat_alloc(n, k), *sd = mat_alloc(k, 1);
    for (int64_t i = 0; i < n; ++i) for (int c = 0; c < k; ++c) {
        double d = sqrt(dist2(i, cen, c));
        Dm->data[i * k + c] = d;
        if (idx[static_cast<size_t>(i)] == c) sd->data[c] += d;
    }
    g_km.C = Cm; g_km.sumd = sd; g_km.D = Dm;
    return idxm;
}
matlab_mat *matlab_stats_km_C(void)    { return g_km.C    ? g_km.C    : mat_alloc(0, 0); }
matlab_mat *matlab_stats_km_sumd(void) { return g_km.sumd ? g_km.sumd : mat_alloc(0, 0); }
matlab_mat *matlab_stats_km_D(void)    { return g_km.D    ? g_km.D    : mat_alloc(0, 0); }

/* pdist2(X, Y): nx×ny euclidean distance matrix. */
matlab_mat *matlab_stats_pdist2(matlab_mat *X, matlab_mat *Y) {
    int64_t nx, px, ny, py;
    std::vector<double> A = sdata(X, nx, px), B = sdata(Y, ny, py);
    if (px != py) return mat_alloc(0, 0);
    matlab_mat *R = mat_alloc(nx, ny);
    for (int64_t i = 0; i < nx; ++i) for (int64_t j = 0; j < ny; ++j) {
        double s = 0.0;
        for (int64_t d = 0; d < px; ++d) {
            double diff = A[static_cast<size_t>(i * px + d)] - B[static_cast<size_t>(j * py + d)];
            s += diff * diff;
        }
        R->data[i * ny + j] = sqrt(s);
    }
    return R;
}

/* pdist(X): condensed 1×(n(n-1)/2) euclidean vector (upper triangle). */
matlab_mat *matlab_stats_pdist(matlab_mat *X) {
    int64_t n, p;
    std::vector<double> A = sdata(X, n, p);
    int64_t m = n * (n - 1) / 2;
    matlab_mat *R = mat_alloc(1, m > 0 ? m : 0);
    int64_t idx = 0;
    for (int64_t i = 0; i < n; ++i) for (int64_t j = i + 1; j < n; ++j) {
        double s = 0.0;
        for (int64_t d = 0; d < p; ++d) {
            double diff = A[static_cast<size_t>(i * p + d)] - A[static_cast<size_t>(j * p + d)];
            s += diff * diff;
        }
        R->data[idx++] = sqrt(s);
    }
    return R;
}

/* squareform(v): condensed vector -> symmetric matrix (and vice versa). */
matlab_mat *matlab_stats_squareform(matlab_mat *V) {
    if (!V || !V->data) return mat_alloc(0, 0);
    int64_t len = V->rows * V->cols;
    if (V->rows > 1 && V->cols > 1) {                 /* square -> condensed */
        int64_t n = V->rows, m = n * (n - 1) / 2;
        matlab_mat *R = mat_alloc(1, m); int64_t idx = 0;
        for (int64_t i = 0; i < n; ++i) for (int64_t j = i + 1; j < n; ++j)
            R->data[idx++] = V->data[i * V->cols + j];
        return R;
    }
    /* condensed -> square: solve n(n-1)/2 = len */
    int64_t n = static_cast<int64_t>((1.0 + sqrt(1.0 + 8.0 * len)) / 2.0 + 0.5);
    matlab_mat *R = mat_alloc(n, n); int64_t idx = 0;
    for (int64_t i = 0; i < n; ++i) for (int64_t j = i + 1; j < n; ++j) {
        R->data[i * n + j] = V->data[idx];
        R->data[j * n + i] = V->data[idx];
        idx++;
    }
    return R;
}

/* silhouette(X, idx): per-point silhouette value s_i in [-1, 1]. */
matlab_mat *matlab_stats_silhouette(matlab_mat *X, matlab_mat *idxm) {
    int64_t n, p;
    std::vector<double> A = sdata(X, n, p);
    std::vector<double> id = sstat_col(idxm, 0);
    int kmax = 0;
    for (double v : id) if (static_cast<int>(v) > kmax) kmax = static_cast<int>(v);
    matlab_mat *S = mat_alloc(n, 1);
    auto edist = [&](int64_t i, int64_t j) {
        double s = 0.0;
        for (int64_t d = 0; d < p; ++d) {
            double diff = A[static_cast<size_t>(i * p + d)] - A[static_cast<size_t>(j * p + d)];
            s += diff * diff;
        }
        return sqrt(s);
    };
    for (int64_t i = 0; i < n; ++i) {
        int ci = static_cast<int>(id[static_cast<size_t>(i)]);
        std::vector<double> sumc(static_cast<size_t>(kmax + 1), 0.0);
        std::vector<int> cntc(static_cast<size_t>(kmax + 1), 0);
        for (int64_t j = 0; j < n; ++j) {
            if (j == i) continue;
            int cj = static_cast<int>(id[static_cast<size_t>(j)]);
            sumc[static_cast<size_t>(cj)] += edist(i, j); cntc[static_cast<size_t>(cj)]++;
        }
        double a = (cntc[static_cast<size_t>(ci)] > 0) ? sumc[static_cast<size_t>(ci)] / cntc[static_cast<size_t>(ci)] : 0.0;
        double b = 1e300;
        for (int c = 1; c <= kmax; ++c) {
            if (c == ci || cntc[static_cast<size_t>(c)] == 0) continue;
            double avg = sumc[static_cast<size_t>(c)] / cntc[static_cast<size_t>(c)];
            if (avg < b) b = avg;
        }
        double denom = std::max(a, b);
        S->data[i] = (denom > 0 && b < 1e299) ? (b - a) / denom : 0.0;
    }
    return S;
}

/* ===== Tier-5 — supervised classification =============================== *
 * One generic `ClassificationModel` classdef carries the fit; ModelType
 * 1=kNN, 2=Gaussian naive Bayes, 3=LDA, 4=CART tree, 5=linear SVM (binary),
 * 6=ECOC (one-vs-one linear SVM).  kNN/NB/LDA stash the training set and
 * re-derive at predict; the tree and SVM stash a compact Params matrix.
 * `predict(mdl, Xnew)` is runtime-dispatched on the model class. */

/* distinct sorted class labels of y. */
static std::vector<double> sclasses(const std::vector<double> &y) {
    std::vector<double> c = y;
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

/* store the training set + class list into a ClassificationModel obj. */
static void sclf_store_xy(matlab_obj *obj, matlab_mat *X, matlab_mat *y, int type) {
    int64_t n, p; std::vector<double> D = sdata(X, n, p);
    std::vector<double> yy = sstat_col(y, 0);
    std::vector<double> cls = sclasses(yy);
    matlab_mat *Xm = mat_alloc(n, p);
    for (int64_t i = 0; i < n * p; ++i) Xm->data[i] = D[static_cast<size_t>(i)];
    matlab_mat *Ym = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) Ym->data[i] = yy[static_cast<size_t>(i)];
    matlab_mat *Cm = mat_alloc(static_cast<int64_t>(cls.size()), 1);
    for (size_t i = 0; i < cls.size(); ++i) Cm->data[i] = cls[i];
    matlab_obj_set_mat(obj, "Xtr", 3, Xm);
    matlab_obj_set_mat(obj, "Ytr", 3, Ym);
    matlab_obj_set_mat(obj, "Classes", 7, Cm);
    matlab_obj_set_f64(obj, "ModelType", 9, static_cast<double>(type));
    matlab_obj_set_f64(obj, "NumClass", 8, static_cast<double>(cls.size()));
    matlab_obj_set_f64(obj, "NumPred", 7, static_cast<double>(p));
}

matlab_mat *matlab_stats_fitknn_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (obj) { sclf_store_xy(obj, X, y, 1); matlab_obj_set_f64(obj, "K", 1, 5.0); }
    return mat_alloc(0, 0);
}
matlab_mat *matlab_stats_fitnb_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (obj) sclf_store_xy(obj, X, y, 2);
    return mat_alloc(0, 0);
}
matlab_mat *matlab_stats_fitlda_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (obj) sclf_store_xy(obj, X, y, 3);
    return mat_alloc(0, 0);
}

/* ---- CART (fitctree) ---------------------------------------------------- *
 * Recursive Gini-split tree.  Encoded as a node matrix, one row per node:
 *   [ feature(1-based; -1 if leaf), threshold, leftRow(1-based), rightRow,
 *     leafClass ]
 * A point goes left when x(feature) < threshold. */
struct STreeBuilder {
    const std::vector<double> *D; int64_t n, p;
    const std::vector<double> *y;
    std::vector<std::array<double, 5>> nodes;
    int maxDepth = 8, minLeaf = 1;
    int featSub = 0;            /* >0 = random-forest split over a feature subset */

    double gini(const std::vector<int> &rows) {
        std::vector<double> cnt;
        std::vector<double> labs;
        for (int r : rows) {
            double l = (*y)[static_cast<size_t>(r)];
            size_t k = 0; for (; k < labs.size(); ++k) if (labs[k] == l) break;
            if (k == labs.size()) { labs.push_back(l); cnt.push_back(0); }
            cnt[k] += 1.0;
        }
        double tot = static_cast<double>(rows.size()), g = 1.0;
        for (double c : cnt) { double f = c / tot; g -= f * f; }
        return g;
    }
    double majority(const std::vector<int> &rows) {
        std::vector<double> labs, cnt;
        for (int r : rows) {
            double l = (*y)[static_cast<size_t>(r)];
            size_t k = 0; for (; k < labs.size(); ++k) if (labs[k] == l) break;
            if (k == labs.size()) { labs.push_back(l); cnt.push_back(0); }
            cnt[k] += 1.0;
        }
        double best = labs.empty() ? 0.0 : labs[0]; double bc = -1;
        for (size_t k = 0; k < labs.size(); ++k) if (cnt[k] > bc) { bc = cnt[k]; best = labs[k]; }
        return best;
    }
    int build(std::vector<int> rows, int depth) {
        int self = static_cast<int>(nodes.size());
        nodes.push_back({-1.0, 0.0, 0.0, 0.0, 0.0});
        bool pure = true;
        for (size_t i = 1; i < rows.size(); ++i)
            if ((*y)[static_cast<size_t>(rows[i])] != (*y)[static_cast<size_t>(rows[0])]) { pure = false; break; }
        if (pure || depth >= maxDepth || static_cast<int>(rows.size()) <= minLeaf) {
            nodes[static_cast<size_t>(self)][4] = majority(rows);
            return self;
        }
        double parentG = gini(rows), bestGain = 0.0; int bestF = -1; double bestT = 0.0;
        /* candidate features: all, or a random subset (random forest). */
        std::vector<int64_t> feats;
        if (featSub <= 0 || featSub >= p) { for (int64_t f = 0; f < p; ++f) feats.push_back(f); }
        else {
            std::vector<int64_t> all; for (int64_t f = 0; f < p; ++f) all.push_back(f);
            for (int t = 0; t < featSub && !all.empty(); ++t) {
                matlab_mat *u = matlab_rand(1, 1);
                size_t pick = static_cast<size_t>(u->data[0] * all.size());
                if (pick >= all.size()) pick = all.size() - 1;
                feats.push_back(all[pick]); all.erase(all.begin() + static_cast<long>(pick));
            }
        }
        for (int64_t f : feats) {
            std::vector<double> vals;
            for (int r : rows) vals.push_back((*D)[static_cast<size_t>(r * p + f)]);
            std::sort(vals.begin(), vals.end());
            for (size_t v = 1; v < vals.size(); ++v) {
                if (vals[v] == vals[v - 1]) continue;
                double thr = 0.5 * (vals[v] + vals[v - 1]);
                std::vector<int> L, R;
                for (int r : rows) (((*D)[static_cast<size_t>(r * p + f)] < thr) ? L : R).push_back(r);
                if (L.empty() || R.empty()) continue;
                double g = parentG - (L.size() * gini(L) + R.size() * gini(R)) / rows.size();
                if (g > bestGain) { bestGain = g; bestF = static_cast<int>(f); bestT = thr; }
            }
        }
        if (bestF < 0) { nodes[static_cast<size_t>(self)][4] = majority(rows); return self; }
        std::vector<int> L, R;
        for (int r : rows) (((*D)[static_cast<size_t>(r * p + bestF)] < bestT) ? L : R).push_back(r);
        int li = build(L, depth + 1), ri = build(R, depth + 1);
        nodes[static_cast<size_t>(self)] = {static_cast<double>(bestF + 1), bestT,
                                            static_cast<double>(li + 1), static_cast<double>(ri + 1), 0.0};
        return self;
    }
};
matlab_mat *matlab_stats_fittree_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (!obj) return mat_alloc(0, 0);
    int64_t n, p; std::vector<double> D = sdata(X, n, p);
    std::vector<double> yy = sstat_col(y, 0);
    STreeBuilder tb; tb.D = &D; tb.n = n; tb.p = p; tb.y = &yy;
    std::vector<int> rows(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) rows[static_cast<size_t>(i)] = static_cast<int>(i);
    tb.build(rows, 0);
    matlab_mat *T = mat_alloc(static_cast<int64_t>(tb.nodes.size()), 5);
    for (size_t r = 0; r < tb.nodes.size(); ++r)
        for (int c = 0; c < 5; ++c) T->data[r * 5 + c] = tb.nodes[r][static_cast<size_t>(c)];
    matlab_obj_set_mat(obj, "Params", 6, T);
    matlab_obj_set_f64(obj, "ModelType", 9, 4.0);
    matlab_obj_set_f64(obj, "NumPred", 7, static_cast<double>(p));
    return mat_alloc(0, 0);
}

/* ---- linear SVM (binary) + ECOC (one-vs-one) ---------------------------- *
 * Linear soft-margin SVM trained in the primal by L2-regularized squared-
 * hinge minimization (gradient descent) — robust and dependency-light; the
 * decision is w·x + b, so only (w, b) need storing.  ECOC stores one such
 * (class_a, class_b, w…, b) row per class pair and predicts by majority
 * vote. */
static void ssvm_train(const std::vector<double> &D, int64_t n, int64_t p,
                       const std::vector<int> &lab /* ±1 */,
                       std::vector<double> &w, double &b) {
    w.assign(static_cast<size_t>(p), 0.0); b = 0.0;
    double C = 1.0, lr = 0.01;
    for (int it = 0; it < 2000; ++it) {
        std::vector<double> gw(static_cast<size_t>(p), 0.0); double gb = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double f = b;
            for (int64_t j = 0; j < p; ++j) f += w[static_cast<size_t>(j)] * D[static_cast<size_t>(i * p + j)];
            double m = lab[static_cast<size_t>(i)] * f;
            if (m < 1.0) {                       /* squared-hinge gradient */
                double g = -2.0 * C * (1.0 - m) * lab[static_cast<size_t>(i)];
                for (int64_t j = 0; j < p; ++j) gw[static_cast<size_t>(j)] += g * D[static_cast<size_t>(i * p + j)];
                gb += g;
            }
        }
        for (int64_t j = 0; j < p; ++j) { gw[static_cast<size_t>(j)] += w[static_cast<size_t>(j)];
            w[static_cast<size_t>(j)] -= lr / n * gw[static_cast<size_t>(j)]; }
        b -= lr / n * gb;
    }
}
matlab_mat *matlab_stats_fitsvm_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (!obj) return mat_alloc(0, 0);
    int64_t n, p; std::vector<double> D = sdata(X, n, p);
    std::vector<double> yy = sstat_col(y, 0);
    std::vector<double> cls = sclasses(yy);
    std::vector<int> lab(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) lab[static_cast<size_t>(i)] = (yy[static_cast<size_t>(i)] == cls[0]) ? -1 : 1;
    std::vector<double> w; double b;
    ssvm_train(D, n, p, lab, w, b);
    /* Params: 1 row = [class_neg, class_pos, w(1..p), b]. */
    matlab_mat *P = mat_alloc(1, p + 3);
    P->data[0] = cls[0]; P->data[1] = cls.size() > 1 ? cls[1] : cls[0];
    for (int64_t j = 0; j < p; ++j) P->data[2 + j] = w[static_cast<size_t>(j)];
    P->data[2 + p] = b;
    matlab_obj_set_mat(obj, "Params", 6, P);
    matlab_obj_set_f64(obj, "ModelType", 9, 5.0);
    matlab_obj_set_f64(obj, "NumPred", 7, static_cast<double>(p));
    return mat_alloc(0, 0);
}
matlab_mat *matlab_stats_fitecoc_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y) {
    if (!obj) return mat_alloc(0, 0);
    int64_t n, p; std::vector<double> D = sdata(X, n, p);
    std::vector<double> yy = sstat_col(y, 0);
    std::vector<double> cls = sclasses(yy);
    int K = static_cast<int>(cls.size());
    std::vector<std::vector<double>> rows;        /* each: [ca cb w… b] */
    for (int a = 0; a < K; ++a) for (int bb = a + 1; bb < K; ++bb) {
        std::vector<double> Dab; std::vector<int> lab;
        for (int64_t i = 0; i < n; ++i) {
            if (yy[static_cast<size_t>(i)] == cls[static_cast<size_t>(a)]) lab.push_back(-1);
            else if (yy[static_cast<size_t>(i)] == cls[static_cast<size_t>(bb)]) lab.push_back(1);
            else continue;
            for (int64_t j = 0; j < p; ++j) Dab.push_back(D[static_cast<size_t>(i * p + j)]);
        }
        int64_t nab = static_cast<int64_t>(lab.size());
        std::vector<double> w; double b;
        ssvm_train(Dab, nab, p, lab, w, b);
        std::vector<double> row;
        row.push_back(cls[static_cast<size_t>(a)]); row.push_back(cls[static_cast<size_t>(bb)]);
        for (int64_t j = 0; j < p; ++j) row.push_back(w[static_cast<size_t>(j)]);
        row.push_back(b);
        rows.push_back(row);
    }
    matlab_mat *P = mat_alloc(static_cast<int64_t>(rows.size()), p + 3);
    for (size_t r = 0; r < rows.size(); ++r)
        for (int64_t c = 0; c < p + 3; ++c) P->data[static_cast<size_t>(r) * (p + 3) + c] = rows[r][static_cast<size_t>(c)];
    matlab_obj_set_mat(obj, "Params", 6, P);
    matlab_obj_set_f64(obj, "ModelType", 9, 6.0);
    matlab_obj_set_f64(obj, "NumPred", 7, static_cast<double>(p));
    matlab_obj_set_f64(obj, "NumClass", 8, static_cast<double>(K));
    return mat_alloc(0, 0);
}

/* ---- ensembles (fitcensemble bagging / TreeBagger random forest) ------- *
 * Train `T` CART trees, each on a bootstrap resample; the random-forest
 * form also restricts each split to a random feature subset (~√p).  Trees
 * are concatenated into one node matrix (child indices offset-adjusted to
 * global rows) with an Offsets vector marking each tree's root; predict
 * majority-votes the per-tree leaf classes.  ModelType 7. */
matlab_mat *matlab_stats_fitensemble_init(matlab_obj *obj, matlab_mat *X, matlab_mat *y,
                                          double ntrees_d, double featsub_d) {
    if (!obj) return mat_alloc(0, 0);
    int64_t n, p; std::vector<double> D = sdata(X, n, p);
    std::vector<double> yy = sstat_col(y, 0);
    std::vector<double> cls = sclasses(yy);
    int T = static_cast<int>(ntrees_d); if (T < 1) T = 1;
    int fs = (featsub_d < 0) ? static_cast<int>(round(sqrt(static_cast<double>(p))))
                             : static_cast<int>(featsub_d);   /* <0 = √p (random forest) */
    std::vector<std::array<double, 5>> all;
    std::vector<double> offs;
    for (int t = 0; t < T; ++t) {
        std::vector<int> rows(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            matlab_mat *u = matlab_rand(1, 1);
            int64_t r = static_cast<int64_t>(u->data[0] * n); if (r >= n) r = n - 1;
            rows[static_cast<size_t>(i)] = static_cast<int>(r);
        }
        STreeBuilder tb; tb.D = &D; tb.n = n; tb.p = p; tb.y = &yy; tb.featSub = fs;
        tb.build(rows, 0);
        int base = static_cast<int>(all.size());
        offs.push_back(static_cast<double>(base));
        for (auto nd : tb.nodes) {
            if (nd[0] >= 0) { nd[2] += base; nd[3] += base; }   /* shift children to global */
            all.push_back(nd);
        }
    }
    matlab_mat *P = mat_alloc(static_cast<int64_t>(all.size()), 5);
    for (size_t r = 0; r < all.size(); ++r) for (int c = 0; c < 5; ++c) P->data[r * 5 + c] = all[r][static_cast<size_t>(c)];
    matlab_mat *O = mat_alloc(static_cast<int64_t>(offs.size()), 1);
    for (size_t i = 0; i < offs.size(); ++i) O->data[i] = offs[i];
    matlab_mat *Cm = mat_alloc(static_cast<int64_t>(cls.size()), 1);
    for (size_t i = 0; i < cls.size(); ++i) Cm->data[i] = cls[i];
    matlab_obj_set_mat(obj, "Params", 6, P);
    matlab_obj_set_mat(obj, "Offsets", 7, O);
    matlab_obj_set_mat(obj, "Classes", 7, Cm);
    matlab_obj_set_f64(obj, "ModelType", 9, 7.0);
    matlab_obj_set_f64(obj, "NumPred", 7, static_cast<double>(p));
    matlab_obj_set_f64(obj, "NumClass", 8, static_cast<double>(cls.size()));
    return mat_alloc(0, 0);
}

/* predict(mdl, Xnew) — dispatch on ModelType, return n×1 labels. */
matlab_mat *matlab_stats_clf_predict(matlab_obj *mdl, matlab_mat *Xnew) {
    if (!mdl || !matlab_obj_is_known(mdl) || !Xnew || !Xnew->data) return mat_alloc(0, 0);
    int type = static_cast<int>(matlab_obj_get_f64(mdl, "ModelType", 9));
    int p = static_cast<int>(matlab_obj_get_f64(mdl, "NumPred", 7));
    int64_t nt = Xnew->rows, ct = Xnew->cols;
    /* a row matching p features is a single observation */
    int64_t nobs = (ct == p) ? nt : ((nt * ct == p) ? 1 : (ct == 1 ? nt : nt));
    int64_t stride = (ct == p) ? ct : ((nt * ct == p) ? 0 : ct);
    const double *xt = Xnew->data;
    auto feat = [&](int64_t i, int j) { return (stride == 0) ? xt[j] : xt[i * stride + j]; };
    matlab_mat *R = mat_alloc(nobs, 1);

    if (type == 4) {                              /* CART tree */
        matlab_mat *T = matlab_obj_get_mat(mdl, "Params", 6);
        for (int64_t i = 0; i < nobs; ++i) {
            int node = 0;
            while (true) {
                double f = T->data[node * 5 + 0];
                if (f < 0) { R->data[i] = T->data[node * 5 + 4]; break; }
                int fi = static_cast<int>(f) - 1;
                double thr = T->data[node * 5 + 1];
                node = static_cast<int>(feat(i, fi) < thr ? T->data[node * 5 + 2] : T->data[node * 5 + 3]) - 1;
            }
        }
        return R;
    }
    if (type == 7) {                              /* ensemble (bagged trees) */
        matlab_mat *P = matlab_obj_get_mat(mdl, "Params", 6);
        matlab_mat *O = matlab_obj_get_mat(mdl, "Offsets", 7);
        int64_t T = O->rows;
        for (int64_t i = 0; i < nobs; ++i) {
            std::vector<double> vc; std::vector<int> vn;
            for (int64_t t = 0; t < T; ++t) {
                int node = static_cast<int>(O->data[t]);
                while (true) {
                    double f = P->data[node * 5 + 0];
                    if (f < 0) { double lc = P->data[node * 5 + 4];
                        size_t k = 0; for (; k < vc.size(); ++k) if (vc[k] == lc) break;
                        if (k == vc.size()) { vc.push_back(lc); vn.push_back(0); }
                        vn[k]++; break; }
                    int fi = static_cast<int>(f) - 1;
                    node = static_cast<int>(feat(i, fi) < P->data[node * 5 + 1]
                               ? P->data[node * 5 + 2] : P->data[node * 5 + 3]) - 1;
                }
            }
            double best = vc.empty() ? 0.0 : vc[0]; int bc = -1;
            for (size_t k = 0; k < vc.size(); ++k) if (vn[k] > bc) { bc = vn[k]; best = vc[k]; }
            R->data[i] = best;
        }
        return R;
    }
    if (type == 5 || type == 6) {                 /* linear SVM / ECOC */
        matlab_mat *P = matlab_obj_get_mat(mdl, "Params", 6);
        int64_t nm = P->rows, w0 = 2;             /* columns: ca cb w… b */
        for (int64_t i = 0; i < nobs; ++i) {
            if (type == 5) {
                double f = P->data[w0 + p];        /* b */
                for (int j = 0; j < p; ++j) f += P->data[w0 + j] * feat(i, j);
                R->data[i] = (f < 0) ? P->data[0] : P->data[1];
            } else {                               /* ECOC majority vote */
                std::vector<double> votecls; std::vector<int> votecnt;
                for (int64_t m = 0; m < nm; ++m) {
                    double *row = &P->data[m * (p + 3)];
                    double f = row[w0 + p];
                    for (int j = 0; j < p; ++j) f += row[w0 + j] * feat(i, j);
                    double pick = (f < 0) ? row[0] : row[1];
                    size_t k = 0; for (; k < votecls.size(); ++k) if (votecls[k] == pick) break;
                    if (k == votecls.size()) { votecls.push_back(pick); votecnt.push_back(0); }
                    votecnt[k]++;
                }
                double best = votecls.empty() ? 0.0 : votecls[0]; int bc = -1;
                for (size_t k = 0; k < votecls.size(); ++k) if (votecnt[k] > bc) { bc = votecnt[k]; best = votecls[k]; }
                R->data[i] = best;
            }
        }
        return R;
    }

    /* kNN / NB / LDA — re-derive from the stored training set. */
    matlab_mat *Xm = matlab_obj_get_mat(mdl, "Xtr", 3);
    matlab_mat *Ym = matlab_obj_get_mat(mdl, "Ytr", 3);
    matlab_mat *Cm = matlab_obj_get_mat(mdl, "Classes", 7);
    int64_t n = Xm->rows; int K = static_cast<int>(Cm->rows);
    auto Xtr = [&](int64_t i, int j) { return Xm->data[i * p + j]; };
    auto cls = [&](int c) { return Cm->data[c]; };

    if (type == 1) {                              /* kNN */
        int kk = static_cast<int>(matlab_obj_get_f64(mdl, "K", 1));
        if (kk < 1) kk = 1;
        for (int64_t i = 0; i < nobs; ++i) {
            std::vector<std::pair<double, double>> dl(static_cast<size_t>(n));
            for (int64_t r = 0; r < n; ++r) {
                double s = 0.0;
                for (int j = 0; j < p; ++j) { double d = feat(i, j) - Xtr(r, j); s += d * d; }
                dl[static_cast<size_t>(r)] = {s, Ym->data[r]};
            }
            std::partial_sort(dl.begin(), dl.begin() + std::min<int64_t>(kk, n), dl.end());
            std::vector<double> vc(static_cast<size_t>(K), 0.0);
            for (int t = 0; t < kk && t < n; ++t)
                for (int c = 0; c < K; ++c) if (dl[static_cast<size_t>(t)].second == cls(c)) vc[static_cast<size_t>(c)]++;
            int best = 0; for (int c = 1; c < K; ++c) if (vc[static_cast<size_t>(c)] > vc[static_cast<size_t>(best)]) best = c;
            R->data[i] = cls(best);
        }
        return R;
    }
    /* per-class mean + variance (NB) / pooled covariance (LDA). */
    std::vector<std::vector<double>> mean(static_cast<size_t>(K), std::vector<double>(static_cast<size_t>(p), 0.0));
    std::vector<int> nc(static_cast<size_t>(K), 0);
    for (int64_t r = 0; r < n; ++r) {
        int c = 0; for (; c < K; ++c) if (Ym->data[r] == cls(c)) break;
        nc[static_cast<size_t>(c)]++;
        for (int j = 0; j < p; ++j) mean[static_cast<size_t>(c)][static_cast<size_t>(j)] += Xtr(r, j);
    }
    for (int c = 0; c < K; ++c) if (nc[static_cast<size_t>(c)] > 0)
        for (int j = 0; j < p; ++j) mean[static_cast<size_t>(c)][static_cast<size_t>(j)] /= nc[static_cast<size_t>(c)];

    if (type == 2) {                              /* Gaussian naive Bayes */
        std::vector<std::vector<double>> var(static_cast<size_t>(K), std::vector<double>(static_cast<size_t>(p), 0.0));
        for (int64_t r = 0; r < n; ++r) {
            int c = 0; for (; c < K; ++c) if (Ym->data[r] == cls(c)) break;
            for (int j = 0; j < p; ++j) { double d = Xtr(r, j) - mean[static_cast<size_t>(c)][static_cast<size_t>(j)]; var[static_cast<size_t>(c)][static_cast<size_t>(j)] += d * d; }
        }
        for (int c = 0; c < K; ++c) for (int j = 0; j < p; ++j)
            var[static_cast<size_t>(c)][static_cast<size_t>(j)] = (nc[static_cast<size_t>(c)] > 1)
                ? var[static_cast<size_t>(c)][static_cast<size_t>(j)] / (nc[static_cast<size_t>(c)] - 1) + 1e-9 : 1e-9;
        for (int64_t i = 0; i < nobs; ++i) {
            int best = 0; double bs = -1e300;
            for (int c = 0; c < K; ++c) {
                double lp = log(static_cast<double>(nc[static_cast<size_t>(c)]) / n);
                for (int j = 0; j < p; ++j) {
                    double v = var[static_cast<size_t>(c)][static_cast<size_t>(j)], d = feat(i, j) - mean[static_cast<size_t>(c)][static_cast<size_t>(j)];
                    lp += -0.5 * (log(2.0 * M_PI * v) + d * d / v);
                }
                if (lp > bs) { bs = lp; best = c; }
            }
            R->data[i] = cls(best);
        }
        return R;
    }
    /* type == 3: LDA — pooled covariance, linear discriminant. */
    std::vector<double> Sig(static_cast<size_t>(p * p), 0.0);
    for (int64_t r = 0; r < n; ++r) {
        int c = 0; for (; c < K; ++c) if (Ym->data[r] == cls(c)) break;
        for (int a = 0; a < p; ++a) for (int bcol = 0; bcol < p; ++bcol)
            Sig[static_cast<size_t>(a * p + bcol)] +=
                (Xtr(r, a) - mean[static_cast<size_t>(c)][static_cast<size_t>(a)]) *
                (Xtr(r, bcol) - mean[static_cast<size_t>(c)][static_cast<size_t>(bcol)]);
    }
    double denom = (n - K > 0) ? static_cast<double>(n - K) : 1.0;
    for (int a = 0; a < p; ++a) { for (int bcol = 0; bcol < p; ++bcol) Sig[static_cast<size_t>(a * p + bcol)] /= denom;
        Sig[static_cast<size_t>(a * p + a)] += 1e-6; }
    std::vector<double> Inv = sinv_dense(Sig, p);
    for (int64_t i = 0; i < nobs; ++i) {
        int best = 0; double bs = -1e300;
        for (int c = 0; c < K; ++c) {
            /* δ_c = xᵀΣ⁻¹μ_c − ½ μ_cᵀΣ⁻¹μ_c + log π_c */
            std::vector<double> Sm(static_cast<size_t>(p), 0.0);
            for (int a = 0; a < p; ++a) for (int bcol = 0; bcol < p; ++bcol)
                Sm[static_cast<size_t>(a)] += Inv[static_cast<size_t>(a * p + bcol)] * mean[static_cast<size_t>(c)][static_cast<size_t>(bcol)];
            double term = log(static_cast<double>(nc[static_cast<size_t>(c)]) / n);
            for (int a = 0; a < p; ++a) term += feat(i, a) * Sm[static_cast<size_t>(a)];
            for (int a = 0; a < p; ++a) term -= 0.5 * mean[static_cast<size_t>(c)][static_cast<size_t>(a)] * Sm[static_cast<size_t>(a)];
            if (term > bs) { bs = term; best = c; }
        }
        R->data[i] = cls(best);
    }
    return R;
}

/* confusionmat(ytrue, ypred): K×K counts over the union of labels. */
matlab_mat *matlab_stats_confusionmat(matlab_mat *yt, matlab_mat *yp) {
    std::vector<double> a = sstat_col(yt, 0), b = sstat_col(yp, 0);
    std::vector<double> all = a; all.insert(all.end(), b.begin(), b.end());
    std::vector<double> cls = sclasses(all);
    int K = static_cast<int>(cls.size());
    matlab_mat *M = mat_alloc(K, K);
    auto idxof = [&](double v) { for (int k = 0; k < K; ++k) if (cls[static_cast<size_t>(k)] == v) return k; return 0; };
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        M->data[idxof(a[i]) * K + idxof(b[i])] += 1.0;
    return M;
}

/* ===== Tier-6 — Hidden Markov Models ==================================== *
 * MATLAB convention: states 1..N, symbols 1..M, the model begins in
 * state 1 (the first emission follows a transition out of state 1).  TRANS
 * is N×N, EMIS is N×M, sequences are 1-based symbol-index row vectors.
 * Scaled forward-backward keeps the recursions numerically stable. */
struct SHmmResult { matlab_mat *states; double logp; matlab_mat *emis; };
static thread_local SHmmResult g_hmm = {nullptr, 0.0, nullptr};

/* hmmgenerate(L, TRANS, EMIS) -> [seq, states]. */
matlab_mat *matlab_stats_hmmgenerate(matlab_mat *Lm, matlab_mat *TR, matlab_mat *EM) {
    int L = static_cast<int>(sstat_sc(Lm, 0.0));
    int N = static_cast<int>(TR->rows), M = static_cast<int>(EM->cols);
    matlab_mat *seq = mat_alloc(1, L), *st = mat_alloc(1, L);
    int cur = 0;
    auto sample = [&](const double *row, int len) {
        matlab_mat *u = matlab_rand(1, 1); double r = u->data[0], acc = 0.0;
        for (int j = 0; j < len; ++j) { acc += row[j]; if (r <= acc) return j; }
        return len - 1;
    };
    for (int i = 0; i < L; ++i) {
        cur = sample(&TR->data[cur * N], N);
        int sym = sample(&EM->data[cur * M], M);
        st->data[i] = cur + 1; seq->data[i] = sym + 1;
    }
    g_hmm.states = st;
    return seq;
}
matlab_mat *matlab_stats_hmm_states(void) { return g_hmm.states ? g_hmm.states : mat_alloc(0, 0); }

/* hmmviterbi(seq, TRANS, EMIS) -> most-likely state path. */
matlab_mat *matlab_stats_hmmviterbi(matlab_mat *seqm, matlab_mat *TR, matlab_mat *EM) {
    std::vector<double> seq = sstat_col(seqm, 0);
    int L = static_cast<int>(seq.size());
    int N = static_cast<int>(TR->rows), M = static_cast<int>(EM->cols);
    auto lg = [](double v) { return v > 0 ? log(v) : -1e300; };
    std::vector<double> v(static_cast<size_t>(N), -1e300);
    std::vector<std::vector<int>> bp(static_cast<size_t>(L), std::vector<int>(static_cast<size_t>(N), 0));
    int s0 = static_cast<int>(seq[0]) - 1; if (s0 < 0) s0 = 0; if (s0 >= M) s0 = M - 1;
    for (int s = 0; s < N; ++s) v[static_cast<size_t>(s)] = lg(TR->data[0 * N + s]) + lg(EM->data[s * M + s0]);
    for (int t = 1; t < L; ++t) {
        int sy = static_cast<int>(seq[static_cast<size_t>(t)]) - 1; if (sy < 0) sy = 0; if (sy >= M) sy = M - 1;
        std::vector<double> nv(static_cast<size_t>(N), -1e300);
        for (int s = 0; s < N; ++s) {
            int best = 0; double bv = -1e301;
            for (int q = 0; q < N; ++q) {
                double val = v[static_cast<size_t>(q)] + lg(TR->data[q * N + s]);
                if (val > bv) { bv = val; best = q; }
            }
            nv[static_cast<size_t>(s)] = bv + lg(EM->data[s * M + sy]);
            bp[static_cast<size_t>(t)][static_cast<size_t>(s)] = best;
        }
        v = nv;
    }
    int last = 0; for (int s = 1; s < N; ++s) if (v[static_cast<size_t>(s)] > v[static_cast<size_t>(last)]) last = s;
    matlab_mat *path = mat_alloc(1, L);
    int cur = last;
    for (int t = L - 1; t >= 0; --t) { path->data[t] = cur + 1; cur = bp[static_cast<size_t>(t)][static_cast<size_t>(cur)]; }
    return path;
}

/* scaled forward-backward shared by hmmdecode / hmmtrain. */
static double shmm_fb(const std::vector<double> &seq, const matlab_mat *TR, const matlab_mat *EM,
                      int N, int M, std::vector<std::vector<double>> &gamma,
                      std::vector<std::vector<std::vector<double>>> *xi) {
    int L = static_cast<int>(seq.size());
    std::vector<std::vector<double>> a(static_cast<size_t>(L), std::vector<double>(static_cast<size_t>(N), 0.0));
    std::vector<std::vector<double>> b(static_cast<size_t>(L), std::vector<double>(static_cast<size_t>(N), 0.0));
    std::vector<double> c(static_cast<size_t>(L), 0.0);
    auto sym = [&](int t) { int s = static_cast<int>(seq[static_cast<size_t>(t)]) - 1; if (s < 0) s = 0; if (s >= M) s = M - 1; return s; };
    /* forward (state 1 start) */
    for (int s = 0; s < N; ++s) a[0][static_cast<size_t>(s)] = TR->data[0 * N + s] * EM->data[s * M + sym(0)];
    for (int s = 0; s < N; ++s) c[0] += a[0][static_cast<size_t>(s)];
    if (c[0] <= 0) c[0] = 1e-300;
    for (int s = 0; s < N; ++s) a[0][static_cast<size_t>(s)] /= c[0];
    for (int t = 1; t < L; ++t) {
        for (int s = 0; s < N; ++s) {
            double sum = 0.0;
            for (int q = 0; q < N; ++q) sum += a[static_cast<size_t>(t - 1)][static_cast<size_t>(q)] * TR->data[q * N + s];
            a[static_cast<size_t>(t)][static_cast<size_t>(s)] = sum * EM->data[s * M + sym(t)];
            c[static_cast<size_t>(t)] += a[static_cast<size_t>(t)][static_cast<size_t>(s)];
        }
        if (c[static_cast<size_t>(t)] <= 0) c[static_cast<size_t>(t)] = 1e-300;
        for (int s = 0; s < N; ++s) a[static_cast<size_t>(t)][static_cast<size_t>(s)] /= c[static_cast<size_t>(t)];
    }
    /* backward */
    for (int s = 0; s < N; ++s) b[static_cast<size_t>(L - 1)][static_cast<size_t>(s)] = 1.0;
    for (int t = L - 2; t >= 0; --t)
        for (int s = 0; s < N; ++s) {
            double sum = 0.0;
            for (int q = 0; q < N; ++q)
                sum += TR->data[s * N + q] * EM->data[q * M + sym(t + 1)] * b[static_cast<size_t>(t + 1)][static_cast<size_t>(q)];
            b[static_cast<size_t>(t)][static_cast<size_t>(s)] = sum / c[static_cast<size_t>(t + 1)];
        }
    gamma.assign(static_cast<size_t>(N), std::vector<double>(static_cast<size_t>(L), 0.0));
    for (int t = 0; t < L; ++t) for (int s = 0; s < N; ++s)
        gamma[static_cast<size_t>(s)][static_cast<size_t>(t)] = a[static_cast<size_t>(t)][static_cast<size_t>(s)] * b[static_cast<size_t>(t)][static_cast<size_t>(s)];
    if (xi) {
        xi->assign(static_cast<size_t>(L - 1),
                   std::vector<std::vector<double>>(static_cast<size_t>(N), std::vector<double>(static_cast<size_t>(N), 0.0)));
        for (int t = 0; t < L - 1; ++t)
            for (int s = 0; s < N; ++s) for (int q = 0; q < N; ++q)
                (*xi)[static_cast<size_t>(t)][static_cast<size_t>(s)][static_cast<size_t>(q)] =
                    a[static_cast<size_t>(t)][static_cast<size_t>(s)] * TR->data[s * N + q] *
                    EM->data[q * M + sym(t + 1)] * b[static_cast<size_t>(t + 1)][static_cast<size_t>(q)] / c[static_cast<size_t>(t + 1)];
    }
    double logp = 0.0; for (int t = 0; t < L; ++t) logp += log(c[static_cast<size_t>(t)]);
    return logp;
}

/* hmmdecode(seq, TRANS, EMIS) -> [pstates (N×L), logpseq]. */
matlab_mat *matlab_stats_hmmdecode(matlab_mat *seqm, matlab_mat *TR, matlab_mat *EM) {
    std::vector<double> seq = sstat_col(seqm, 0);
    int N = static_cast<int>(TR->rows), M = static_cast<int>(EM->cols), L = static_cast<int>(seq.size());
    std::vector<std::vector<double>> g;
    double logp = shmm_fb(seq, TR, EM, N, M, g, nullptr);
    matlab_mat *P = mat_alloc(N, L);
    for (int s = 0; s < N; ++s) for (int t = 0; t < L; ++t) P->data[s * L + t] = g[static_cast<size_t>(s)][static_cast<size_t>(t)];
    g_hmm.logp = logp;
    return P;
}
double matlab_stats_hmm_logp(void) { return g_hmm.logp; }

/* hmmtrain(seq, TRANS0, EMIS0) -> [TRANS, EMIS] via Baum-Welch. */
matlab_mat *matlab_stats_hmmtrain(matlab_mat *seqm, matlab_mat *TR0, matlab_mat *EM0) {
    std::vector<double> seq = sstat_col(seqm, 0);
    int N = static_cast<int>(TR0->rows), M = static_cast<int>(EM0->cols), L = static_cast<int>(seq.size());
    matlab_mat *TR = mat_alloc(N, N), *EM = mat_alloc(N, M);
    for (int i = 0; i < N * N; ++i) TR->data[i] = TR0->data[i];
    for (int i = 0; i < N * M; ++i) EM->data[i] = EM0->data[i];
    auto sym = [&](int t) { int s = static_cast<int>(seq[static_cast<size_t>(t)]) - 1; if (s < 0) s = 0; if (s >= M) s = M - 1; return s; };
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<std::vector<double>> g;
        std::vector<std::vector<std::vector<double>>> xi;
        shmm_fb(seq, TR, EM, N, M, g, &xi);
        /* re-estimate TRANS */
        for (int s = 0; s < N; ++s) {
            double den = 0.0;
            for (int t = 0; t < L - 1; ++t) den += g[static_cast<size_t>(s)][static_cast<size_t>(t)];
            for (int q = 0; q < N; ++q) {
                double num = 0.0;
                for (int t = 0; t < L - 1; ++t) num += xi[static_cast<size_t>(t)][static_cast<size_t>(s)][static_cast<size_t>(q)];
                TR->data[s * N + q] = (den > 0) ? num / den : TR->data[s * N + q];
            }
        }
        /* re-estimate EMIS */
        for (int s = 0; s < N; ++s) {
            double den = 0.0;
            for (int t = 0; t < L; ++t) den += g[static_cast<size_t>(s)][static_cast<size_t>(t)];
            for (int m = 0; m < M; ++m) {
                double num = 0.0;
                for (int t = 0; t < L; ++t) if (sym(t) == m) num += g[static_cast<size_t>(s)][static_cast<size_t>(t)];
                EM->data[s * M + m] = (den > 0) ? num / den : EM->data[s * M + m];
            }
        }
    }
    g_hmm.emis = EM;
    return TR;
}
matlab_mat *matlab_stats_hmm_emis(void) { return g_hmm.emis ? g_hmm.emis : mat_alloc(0, 0); }

/* ===== Tier-6 — Bayesian optimization =================================== *
 * bayesopt(fun, lb, ub): minimize an expensive black-box over a box with a
 * Gaussian-process surrogate (squared-exponential kernel) + expected-
 * improvement acquisition.  Functional form over the shipped 1-arg
 * objective-handle ABI (the optimizableVariable / results-object API is a
 * documented carve-down).  Returns the best point as a column vector. */
typedef double (*stats_obj_fn)(matlab_mat *);

static double sbo_eval(stats_obj_fn f, const std::vector<double> &x) {
    matlab_mat *m = mat_alloc(static_cast<int64_t>(x.size()), 1);
    for (size_t i = 0; i < x.size(); ++i) m->data[i] = x[i];
    return f(m);
}
matlab_mat *matlab_stats_bayesopt(void *fn_p, matlab_mat *lbm, matlab_mat *ubm) {
    if (!fn_p) return mat_alloc(0, 0);
    stats_obj_fn f = reinterpret_cast<stats_obj_fn>(fn_p);
    std::vector<double> lo = sstat_col(lbm, 0), hi = sstat_col(ubm, 0);
    int n = static_cast<int>(lo.size());
    if (n < 1) return mat_alloc(0, 0);
    double span = 0.0; for (int i = 0; i < n; ++i) span += (hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)]);
    double ell = 0.2 * span / n; if (ell <= 0) ell = 1.0;
    double ell2 = 2.0 * ell * ell;
    auto kern = [&](const std::vector<double> &a, const std::vector<double> &b) {
        double s = 0.0; for (int i = 0; i < n; ++i) { double d = a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]; s += d * d; }
        return exp(-s / ell2);
    };
    auto rnd_pt = [&]() {
        std::vector<double> x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) { matlab_mat *u = matlab_rand(1, 1);
            x[static_cast<size_t>(i)] = lo[static_cast<size_t>(i)] + u->data[0] * (hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)]); }
        return x;
    };
    auto NPHI = [](double z) { return 0.5 * erfc(-z / 1.41421356237309504880); };
    auto NPDF = [](double z) { return exp(-0.5 * z * z) / 2.50662827463100050242; };

    std::vector<std::vector<double>> X;
    std::vector<double> fv;
    int ninit = std::max(5, 2 * n);
    std::vector<double> xbest; double fbest = 1e300;
    for (int i = 0; i < ninit; ++i) {
        std::vector<double> x = rnd_pt(); double v = sbo_eval(f, x);
        X.push_back(x); fv.push_back(v);
        if (v < fbest) { fbest = v; xbest = x; }
    }
    int budget = 40 + 10 * n;
    for (int it = 0; it < budget; ++it) {
        int N = static_cast<int>(X.size());
        std::vector<double> K(static_cast<size_t>(N * N));
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j)
            K[static_cast<size_t>(i * N + j)] = kern(X[static_cast<size_t>(i)], X[static_cast<size_t>(j)]) + (i == j ? 1e-6 : 0.0);
        std::vector<double> Kinv = sinv_dense(K, N);
        std::vector<double> Kf(static_cast<size_t>(N), 0.0);
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) Kf[static_cast<size_t>(i)] += Kinv[static_cast<size_t>(i * N + j)] * fv[static_cast<size_t>(j)];
        std::vector<double> bestCand; double bestEI = -1e300;
        for (int c = 0; c < 200; ++c) {
            std::vector<double> cand = rnd_pt();
            std::vector<double> ks(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i) ks[static_cast<size_t>(i)] = kern(cand, X[static_cast<size_t>(i)]);
            double mu = 0.0; for (int i = 0; i < N; ++i) mu += ks[static_cast<size_t>(i)] * Kf[static_cast<size_t>(i)];
            double quad = 0.0;
            for (int i = 0; i < N; ++i) { double row = 0.0;
                for (int j = 0; j < N; ++j) row += Kinv[static_cast<size_t>(i * N + j)] * ks[static_cast<size_t>(j)];
                quad += ks[static_cast<size_t>(i)] * row; }
            double s2 = 1.0 - quad; if (s2 < 1e-9) s2 = 1e-9;
            double sd = sqrt(s2), z = (fbest - mu) / sd;
            double ei = (fbest - mu) * NPHI(z) + sd * NPDF(z);
            if (ei > bestEI) { bestEI = ei; bestCand = cand; }
        }
        double v = sbo_eval(f, bestCand);
        X.push_back(bestCand); fv.push_back(v);
        if (v < fbest) { fbest = v; xbest = bestCand; }
    }
    matlab_mat *R = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) R->data[i] = xbest[static_cast<size_t>(i)];
    return R;
}

/* ----- secondary-output readers (pull from g_stest) ---------------------- */
double      matlab_stats_test_o2(void)  { return g_stest.out2; }
matlab_mat *matlab_stats_test_ci(void)  {
    matlab_mat *R = mat_alloc(1, 2);
    R->data[0] = g_stest.ci_lo; R->data[1] = g_stest.ci_hi;
    return R;
}
matlab_mat *matlab_stats_test_stats(void) {
    matlab_struct *s = matlab_struct_new();
    matlab_struct_set_f64(s, "tstat", 5, g_stest.stat);
    matlab_struct_set_f64(s, "df", 2, g_stest.df);
    return reinterpret_cast<matlab_mat *>(s);
}

}  /* extern "C" */
