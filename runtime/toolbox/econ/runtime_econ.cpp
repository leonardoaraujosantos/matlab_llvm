/* ============================================================================
 * runtime_econ.cpp — Econometrics Toolbox runtime
 * ----------------------------------------------------------------------------
 * Per docs/econometrics_toolbox_roadmap.md.  Everything here is hand-coded
 * over the shipped numeric base (LAPACK lane, Stats CDFs, Optim, Ident PEM)
 * — no external dependency.
 *
 * Tier-1: data preprocessing (price2ret/ret2price/hpfilter), ACF/PACF
 * (autocorr/parcorr/crosscorr), diagnostic tests (lbqtest/archtest/aicbic),
 * unit-root + stationarity tests (adftest/pptest/kpsstest/lmctest/vratiotest),
 * HAC/FGLS.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
matlab_mat *mat_alloc(int64_t r, int64_t c);
}

/* ---- small local helpers ------------------------------------------------ */
namespace {

/* Flatten a matlab_mat column-or-row vector into a std::vector<double>. */
std::vector<double> vecOf(const matlab_mat *m) {
    std::vector<double> v;
    if (!m || !m->data) return v;
    int64_t n = m->rows * m->cols;
    v.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = m->data[i];
    return v;
}

matlab_mat *colVec(const std::vector<double> &v) {
    matlab_mat *out = mat_alloc(static_cast<int64_t>(v.size()), 1);
    for (size_t i = 0; i < v.size(); ++i) out->data[i] = v[i];
    return out;
}

double meanOf(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

/* Sample autocovariance at lag k (divisor N, MATLAB convention). */
double autocov(const std::vector<double> &y, double ybar, int k) {
    int64_t N = static_cast<int64_t>(y.size());
    double s = 0.0;
    for (int64_t t = k; t < N; ++t)
        s += (y[static_cast<size_t>(t)] - ybar) *
             (y[static_cast<size_t>(t - k)] - ybar);
    return s / static_cast<double>(N);
}

} // namespace

extern "C" {

/* ============================================================================
 * §T1 — Data transformations
 * ==========================================================================*/

/* price2ret(P) — continuous (log) returns: r_t = log(P_t / P_{t-1}). */
matlab_mat *matlab_econ_price2ret(matlab_mat *P) {
    std::vector<double> p = vecOf(P);
    if (p.size() < 2) return mat_alloc(0, 0);
    std::vector<double> r(p.size() - 1);
    for (size_t t = 1; t < p.size(); ++t) r[t - 1] = std::log(p[t] / p[t - 1]);
    return colVec(r);
}

/* ret2price(R) — invert log returns to a price series with P0 = 1.
 * Returns an (N+1)-vector. */
matlab_mat *matlab_econ_ret2price(matlab_mat *R) {
    std::vector<double> r = vecOf(R);
    std::vector<double> p(r.size() + 1);
    p[0] = 1.0;
    for (size_t t = 0; t < r.size(); ++t) p[t + 1] = p[t] * std::exp(r[t]);
    return colVec(p);
}

/* hpfilter(y, lambda) — Hodrick-Prescott trend.  Solves
 *   (I + lambda D'D) tau = y   where D is the 2nd-difference operator.
 * Returns the trend; the cycle is y - trend.  Banded pentadiagonal system
 * solved by a dense symmetric Gaussian elimination (N small in practice). */
matlab_mat *matlab_econ_hpfilter_l(matlab_mat *Y, double lambda) {
    std::vector<double> y = vecOf(Y);
    int64_t N = static_cast<int64_t>(y.size());
    if (N < 3) return colVec(y);
    /* Build A = I + lambda * D'D (N x N, symmetric pentadiagonal). */
    std::vector<double> A(static_cast<size_t>(N * N), 0.0);
    auto at = [&](int64_t i, int64_t j) -> double & {
        return A[static_cast<size_t>(i * N + j)];
    };
    for (int64_t i = 0; i < N; ++i) at(i, i) = 1.0;
    /* D is (N-2) x N with rows [1 -2 1].  Accumulate lambda * D'D. */
    for (int64_t r = 0; r < N - 2; ++r) {
        int64_t c0 = r, c1 = r + 1, c2 = r + 2;
        double w[3] = {1.0, -2.0, 1.0};
        int64_t cc[3] = {c0, c1, c2};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                at(cc[a], cc[b]) += lambda * w[a] * w[b];
    }
    /* Solve A tau = y by Gaussian elimination with partial pivoting. */
    std::vector<double> M = A, rhs = y;
    for (int64_t col = 0; col < N; ++col) {
        int64_t piv = col;
        double best = std::fabs(M[static_cast<size_t>(col * N + col)]);
        for (int64_t r = col + 1; r < N; ++r) {
            double v = std::fabs(M[static_cast<size_t>(r * N + col)]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) {
            for (int64_t j = 0; j < N; ++j)
                std::swap(M[static_cast<size_t>(col * N + j)],
                          M[static_cast<size_t>(piv * N + j)]);
            std::swap(rhs[static_cast<size_t>(col)],
                      rhs[static_cast<size_t>(piv)]);
        }
        double d = M[static_cast<size_t>(col * N + col)];
        if (std::fabs(d) < 1e-300) continue;
        for (int64_t r = col + 1; r < N; ++r) {
            double f = M[static_cast<size_t>(r * N + col)] / d;
            if (f == 0.0) continue;
            for (int64_t j = col; j < N; ++j)
                M[static_cast<size_t>(r * N + j)] -=
                    f * M[static_cast<size_t>(col * N + j)];
            rhs[static_cast<size_t>(r)] -= f * rhs[static_cast<size_t>(col)];
        }
    }
    std::vector<double> tau(static_cast<size_t>(N), 0.0);
    for (int64_t r = N - 1; r >= 0; --r) {
        double s = rhs[static_cast<size_t>(r)];
        for (int64_t j = r + 1; j < N; ++j)
            s -= M[static_cast<size_t>(r * N + j)] * tau[static_cast<size_t>(j)];
        double d = M[static_cast<size_t>(r * N + r)];
        tau[static_cast<size_t>(r)] = (std::fabs(d) < 1e-300) ? 0.0 : s / d;
    }
    return colVec(tau);
}

/* hpfilter(y) — default smoothing lambda = 1600 (quarterly data). */
matlab_mat *matlab_econ_hpfilter(matlab_mat *Y) {
    return matlab_econ_hpfilter_l(Y, 1600.0);
}

/* ============================================================================
 * §T1 — ACF / PACF
 * ==========================================================================*/

/* autocorr(y, numLags) — sample ACF for lags 0..numLags (length numLags+1,
 * acf[0] = 1). */
matlab_mat *matlab_econ_autocorr_n(matlab_mat *Y, double numLags) {
    std::vector<double> y = vecOf(Y);
    int64_t N = static_cast<int64_t>(y.size());
    int L = static_cast<int>(numLags);
    if (N < 2 || L < 0) return mat_alloc(0, 0);
    if (L > N - 1) L = static_cast<int>(N - 1);
    double ybar = meanOf(y);
    double c0 = autocov(y, ybar, 0);
    std::vector<double> acf(static_cast<size_t>(L + 1));
    for (int k = 0; k <= L; ++k)
        acf[static_cast<size_t>(k)] =
            (c0 == 0.0) ? (k == 0 ? 1.0 : 0.0) : autocov(y, ybar, k) / c0;
    return colVec(acf);
}

/* autocorr(y) — default numLags = min(20, N-1). */
matlab_mat *matlab_econ_autocorr(matlab_mat *Y) {
    std::vector<double> y = vecOf(Y);
    int64_t N = static_cast<int64_t>(y.size());
    double L = (N - 1 < 20) ? static_cast<double>(N - 1) : 20.0;
    return matlab_econ_autocorr_n(Y, L);
}

/* parcorr(y, numLags) — PACF via Durbin-Levinson recursion.  Returns
 * length numLags+1 with pacf[0] = 1. */
matlab_mat *matlab_econ_parcorr_n(matlab_mat *Y, double numLags) {
    std::vector<double> y = vecOf(Y);
    int64_t N = static_cast<int64_t>(y.size());
    int L = static_cast<int>(numLags);
    if (N < 2 || L < 0) return mat_alloc(0, 0);
    if (L > N - 1) L = static_cast<int>(N - 1);
    double ybar = meanOf(y);
    double c0 = autocov(y, ybar, 0);
    std::vector<double> rho(static_cast<size_t>(L + 1));
    for (int k = 0; k <= L; ++k)
        rho[static_cast<size_t>(k)] =
            (c0 == 0.0) ? (k == 0 ? 1.0 : 0.0) : autocov(y, ybar, k) / c0;
    std::vector<double> pacf(static_cast<size_t>(L + 1), 0.0);
    pacf[0] = 1.0;
    if (L >= 1) {
        std::vector<double> phi(static_cast<size_t>(L + 1), 0.0);
        std::vector<double> prev(static_cast<size_t>(L + 1), 0.0);
        phi[1] = rho[1];
        pacf[1] = rho[1];
        double v = 1.0 - rho[1] * rho[1];
        for (int k = 2; k <= L; ++k) {
            prev = phi;
            double num = rho[static_cast<size_t>(k)];
            for (int j = 1; j < k; ++j)
                num -= prev[static_cast<size_t>(j)] *
                       rho[static_cast<size_t>(k - j)];
            double phikk = (v <= 0.0) ? 0.0 : num / v;
            phi[static_cast<size_t>(k)] = phikk;
            for (int j = 1; j < k; ++j)
                phi[static_cast<size_t>(j)] =
                    prev[static_cast<size_t>(j)] -
                    phikk * prev[static_cast<size_t>(k - j)];
            pacf[static_cast<size_t>(k)] = phikk;
            v *= (1.0 - phikk * phikk);
        }
    }
    return colVec(pacf);
}

matlab_mat *matlab_econ_parcorr(matlab_mat *Y) {
    std::vector<double> y = vecOf(Y);
    int64_t N = static_cast<int64_t>(y.size());
    double L = (N - 1 < 20) ? static_cast<double>(N - 1) : 20.0;
    return matlab_econ_parcorr_n(Y, L);
}

/* crosscorr(y1, y2, numLags) — sample cross-correlation for lags
 * -numLags..numLags (length 2*numLags+1). */
matlab_mat *matlab_econ_crosscorr(matlab_mat *Y1, matlab_mat *Y2) {
    std::vector<double> a = vecOf(Y1), b = vecOf(Y2);
    int64_t N = std::min(static_cast<int64_t>(a.size()),
                         static_cast<int64_t>(b.size()));
    if (N < 2) return mat_alloc(0, 0);
    int L = (N - 1 < 20) ? static_cast<int>(N - 1) : 20;
    double ma = meanOf(a), mb = meanOf(b);
    double sa = 0.0, sb = 0.0;
    for (int64_t t = 0; t < N; ++t) {
        sa += (a[static_cast<size_t>(t)] - ma) * (a[static_cast<size_t>(t)] - ma);
        sb += (b[static_cast<size_t>(t)] - mb) * (b[static_cast<size_t>(t)] - mb);
    }
    double denom = std::sqrt(sa * sb);
    std::vector<double> xcf(static_cast<size_t>(2 * L + 1), 0.0);
    for (int k = -L; k <= L; ++k) {
        double s = 0.0;
        for (int64_t t = 0; t < N; ++t) {
            int64_t tk = t + k;
            if (tk < 0 || tk >= N) continue;
            s += (a[static_cast<size_t>(t)] - ma) *
                 (b[static_cast<size_t>(tk)] - mb);
        }
        xcf[static_cast<size_t>(k + L)] = (denom == 0.0) ? 0.0 : s / denom;
    }
    return colVec(xcf);
}

} // extern "C"

/* ============================================================================
 * §T1 — diagnostic + unit-root tests : shared statistics helpers
 * ==========================================================================*/
namespace {

/* Regularized lower incomplete gamma P(a,x) via series / continued fraction
 * (Numerical Recipes).  Used for the chi-square CDF. */
double gser(double a, double x) {
    if (x <= 0.0) return 0.0;
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 1; n <= 500; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * 1e-14) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}
double gcf(double a, double x) {
    const double FPMIN = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / FPMIN, d = 1.0 / b, h = d;
    for (int i = 1; i <= 500; ++i) {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < 1e-14) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}
double gammap(double a, double x) {
    if (x < 0.0 || a <= 0.0) return 0.0;
    return (x < a + 1.0) ? gser(a, x) : 1.0 - gcf(a, x);
}
/* chi-square upper-tail (survival) probability = P(X > x), X ~ chi2(k). */
double chi2_sf(double x, double k) {
    if (x <= 0.0) return 1.0;
    return 1.0 - gammap(k / 2.0, x / 2.0);
}
/* standard normal CDF. */
double norm_cdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

/* OLS: solve (X'X) beta = X'y for X (n x p, row-major) and y (n).  Returns
 * beta (p) and optionally the residual SSE and (X'X)^{-1}.  Small dense
 * Gaussian elimination — p is tiny in practice. */
bool ols(const std::vector<double> &X, const std::vector<double> &y,
         int64_t n, int64_t p, std::vector<double> &beta,
         std::vector<double> *XtXinv = nullptr) {
    std::vector<double> A(static_cast<size_t>(p * p), 0.0);
    std::vector<double> b(static_cast<size_t>(p), 0.0);
    for (int64_t i = 0; i < p; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (int64_t t = 0; t < n; ++t)
                s += X[static_cast<size_t>(t * p + i)] *
                     X[static_cast<size_t>(t * p + j)];
            A[static_cast<size_t>(i * p + j)] = s;
        }
        double s = 0.0;
        for (int64_t t = 0; t < n; ++t)
            s += X[static_cast<size_t>(t * p + i)] * y[static_cast<size_t>(t)];
        b[static_cast<size_t>(i)] = s;
    }
    /* Solve A beta = b by Gauss-Jordan on the augmented matrix
     * [ A | b | I ].  Columns: 0..p-1 = A, col p = rhs, cols p+1..2p = I
     * (the identity block is only laid down when the inverse is wanted). */
    int64_t m = (p + 1) + (XtXinv ? p : 0);
    std::vector<double> M(static_cast<size_t>(p * m), 0.0);
    for (int64_t i = 0; i < p; ++i) {
        for (int64_t j = 0; j < p; ++j)
            M[static_cast<size_t>(i * m + j)] = A[static_cast<size_t>(i * p + j)];
        M[static_cast<size_t>(i * m + p)] = b[static_cast<size_t>(i)];
        if (XtXinv)
            M[static_cast<size_t>(i * m + (p + 1) + i)] = 1.0;
    }
    for (int64_t col = 0; col < p; ++col) {
        int64_t piv = col;
        double best = std::fabs(M[static_cast<size_t>(col * m + col)]);
        for (int64_t r = col + 1; r < p; ++r) {
            double v = std::fabs(M[static_cast<size_t>(r * m + col)]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return false;
        if (piv != col)
            for (int64_t j = 0; j < m; ++j)
                std::swap(M[static_cast<size_t>(col * m + j)],
                          M[static_cast<size_t>(piv * m + j)]);
        double d = M[static_cast<size_t>(col * m + col)];
        for (int64_t j = 0; j < m; ++j) M[static_cast<size_t>(col * m + j)] /= d;
        for (int64_t r = 0; r < p; ++r) {
            if (r == col) continue;
            double f = M[static_cast<size_t>(r * m + col)];
            if (f == 0.0) continue;
            for (int64_t j = 0; j < m; ++j)
                M[static_cast<size_t>(r * m + j)] -=
                    f * M[static_cast<size_t>(col * m + j)];
        }
    }
    beta.assign(static_cast<size_t>(p), 0.0);
    for (int64_t i = 0; i < p; ++i)
        beta[static_cast<size_t>(i)] = M[static_cast<size_t>(i * m + p)];
    if (XtXinv) {
        XtXinv->assign(static_cast<size_t>(p * p), 0.0);
        for (int64_t i = 0; i < p; ++i)
            for (int64_t j = 0; j < p; ++j)
                (*XtXinv)[static_cast<size_t>(i * p + j)] =
                    M[static_cast<size_t>(i * m + (p + 1) + j)];
    }
    return true;
}

} // namespace

extern "C" {

/* ============================================================================
 * §T1 — diagnostic + comparison tests
 * ==========================================================================*/

/* lbqtest(res, numLags) — Ljung-Box Q test for residual autocorrelation.
 * Q = N(N+2) Σ ρ_k²/(N-k) ~ chi2(numLags) under H0 (no autocorrelation).
 * Returns the reject decision h (1 = reject no-autocorrelation @ 5%). */
double matlab_econ_lbqtest_n(matlab_mat *Res, double numLags) {
    std::vector<double> e = vecOf(Res);
    int64_t N = static_cast<int64_t>(e.size());
    int m = static_cast<int>(numLags);
    if (N < 3 || m < 1) return 0.0;
    if (m > N - 1) m = static_cast<int>(N - 1);
    double ebar = meanOf(e);
    double c0 = autocov(e, ebar, 0);
    double Q = 0.0;
    for (int k = 1; k <= m; ++k) {
        double rk = (c0 == 0.0) ? 0.0 : autocov(e, ebar, k) / c0;
        Q += rk * rk / static_cast<double>(N - k);
    }
    Q *= static_cast<double>(N) * static_cast<double>(N + 2);
    double p = chi2_sf(Q, static_cast<double>(m));
    return (p < 0.05) ? 1.0 : 0.0;
}
double matlab_econ_lbqtest(matlab_mat *Res) {
    std::vector<double> e = vecOf(Res);
    int64_t N = static_cast<int64_t>(e.size());
    double L = (N - 1 < 20) ? static_cast<double>(N - 1) : 20.0;
    return matlab_econ_lbqtest_n(Res, L);
}

/* archtest(res, numLags) — Engle's ARCH test.  Regress e_t² on a constant
 * and numLags lagged squared residuals; LM = N*R² ~ chi2(numLags).
 * Returns h (1 = reject homoscedasticity / ARCH effects present @ 5%). */
double matlab_econ_archtest_n(matlab_mat *Res, double numLags) {
    std::vector<double> e = vecOf(Res);
    int64_t N = static_cast<int64_t>(e.size());
    int m = static_cast<int>(numLags);
    if (m < 1) m = 1;
    if (N <= m + 2) return 0.0;
    std::vector<double> e2(static_cast<size_t>(N));
    for (int64_t t = 0; t < N; ++t)
        e2[static_cast<size_t>(t)] =
            e[static_cast<size_t>(t)] * e[static_cast<size_t>(t)];
    int64_t n = N - m;       /* usable rows */
    int64_t p = m + 1;       /* constant + m lags */
    std::vector<double> X(static_cast<size_t>(n * p)), yy(static_cast<size_t>(n));
    for (int64_t r = 0; r < n; ++r) {
        int64_t t = r + m;
        X[static_cast<size_t>(r * p + 0)] = 1.0;
        for (int j = 1; j <= m; ++j)
            X[static_cast<size_t>(r * p + j)] =
                e2[static_cast<size_t>(t - j)];
        yy[static_cast<size_t>(r)] = e2[static_cast<size_t>(t)];
    }
    std::vector<double> beta;
    if (!ols(X, yy, n, p, beta)) return 0.0;
    double ybar = meanOf(yy), sst = 0.0, sse = 0.0;
    for (int64_t r = 0; r < n; ++r) {
        double yhat = 0.0;
        for (int64_t j = 0; j < p; ++j)
            yhat += X[static_cast<size_t>(r * p + j)] *
                    beta[static_cast<size_t>(j)];
        double yt = yy[static_cast<size_t>(r)];
        sse += (yt - yhat) * (yt - yhat);
        sst += (yt - ybar) * (yt - ybar);
    }
    double R2 = (sst == 0.0) ? 0.0 : 1.0 - sse / sst;
    double LM = static_cast<double>(n) * R2;
    double pv = chi2_sf(LM, static_cast<double>(m));
    return (pv < 0.05) ? 1.0 : 0.0;
}
double matlab_econ_archtest(matlab_mat *Res) {
    return matlab_econ_archtest_n(Res, 1.0);
}

/* aicbic(logL, numParam[, numObs]) — Akaike information criterion
 * aic = -2 logL + 2 k.  (The companion BIC is a documented multi-output
 * follow-on; the scalar form returns AIC for model comparison.) */
double matlab_econ_aic(double logL, double numParam) {
    return -2.0 * logL + 2.0 * numParam;
}
double matlab_econ_aic_n(double logL, double numParam, double numObs) {
    (void)numObs;
    return -2.0 * logL + 2.0 * numParam;
}

/* lratiotest(logLu, logLr, dof) — likelihood-ratio test.
 * stat = 2(logLu - logLr) ~ chi2(dof) under H0 (restricted model adequate).
 * Returns h (1 = reject restricted model @ 5%). */
double matlab_econ_lratiotest(double logLu, double logLr, double dof) {
    double stat = 2.0 * (logLu - logLr);
    if (stat < 0.0) stat = 0.0;
    double p = chi2_sf(stat, dof);
    return (p < 0.05) ? 1.0 : 0.0;
}

/* Quadratic form r' V^{-1} r for r (q) and symmetric V (q x q, row-major)
 * via Gaussian elimination on [V | r]. */
static double quad_form(const std::vector<double> &V,
                        const std::vector<double> &r, int64_t q) {
    std::vector<double> M(static_cast<size_t>(q * (q + 1)), 0.0);
    int64_t w = q + 1;
    for (int64_t i = 0; i < q; ++i) {
        for (int64_t j = 0; j < q; ++j)
            M[static_cast<size_t>(i * w + j)] = V[static_cast<size_t>(i * q + j)];
        M[static_cast<size_t>(i * w + q)] = r[static_cast<size_t>(i)];
    }
    for (int64_t col = 0; col < q; ++col) {
        int64_t piv = col;
        double best = std::fabs(M[static_cast<size_t>(col * w + col)]);
        for (int64_t rr = col + 1; rr < q; ++rr) {
            double v = std::fabs(M[static_cast<size_t>(rr * w + col)]);
            if (v > best) { best = v; piv = rr; }
        }
        if (best < 1e-300) return 0.0;
        if (piv != col)
            for (int64_t j = 0; j < w; ++j)
                std::swap(M[static_cast<size_t>(col * w + j)],
                          M[static_cast<size_t>(piv * w + j)]);
        double d = M[static_cast<size_t>(col * w + col)];
        for (int64_t j = 0; j < w; ++j) M[static_cast<size_t>(col * w + j)] /= d;
        for (int64_t rr = 0; rr < q; ++rr) {
            if (rr == col) continue;
            double f = M[static_cast<size_t>(rr * w + col)];
            for (int64_t j = 0; j < w; ++j)
                M[static_cast<size_t>(rr * w + j)] -=
                    f * M[static_cast<size_t>(col * w + j)];
        }
    }
    double s = 0.0;
    for (int64_t i = 0; i < q; ++i)
        s += r[static_cast<size_t>(i)] * M[static_cast<size_t>(i * w + q)];
    return s;
}

/* waldtest(r, EstCov) — Wald test of q linear restrictions.
 * W = r' EstCov^{-1} r ~ chi2(q).  Returns h (reject @ 5%). */
double matlab_econ_waldtest(matlab_mat *R, matlab_mat *V) {
    std::vector<double> r = vecOf(R);
    int64_t q = static_cast<int64_t>(r.size());
    if (!V || V->rows != q || V->cols != q || q < 1) return 0.0;
    std::vector<double> Vv(static_cast<size_t>(q * q));
    for (int64_t i = 0; i < q * q; ++i) Vv[static_cast<size_t>(i)] = V->data[i];
    double W = quad_form(Vv, r, q);
    return (chi2_sf(W, static_cast<double>(q)) < 0.05) ? 1.0 : 0.0;
}

/* lmtest(score, V) — Lagrange-multiplier (score) test.
 * LM = score' V^{-1} score ~ chi2(q).  Returns h (reject @ 5%). */
double matlab_econ_lmtest(matlab_mat *S, matlab_mat *V) {
    std::vector<double> s = vecOf(S);
    int64_t q = static_cast<int64_t>(s.size());
    if (!V || V->rows != q || V->cols != q || q < 1) return 0.0;
    std::vector<double> Vv(static_cast<size_t>(q * q));
    for (int64_t i = 0; i < q * q; ++i) Vv[static_cast<size_t>(i)] = V->data[i];
    double LM = quad_form(Vv, s, q);
    return (chi2_sf(LM, static_cast<double>(q)) < 0.05) ? 1.0 : 0.0;
}

/* hac(X, y) — Newey-West HAC covariance of the OLS coefficients.
 * Returns the p x p coefficient covariance matrix. */
matlab_mat *matlab_econ_hac(matlab_mat *Xm, matlab_mat *Ym) {
    if (!Xm || !Ym || !Xm->data) return mat_alloc(0, 0);
    int64_t n = Xm->rows, p = Xm->cols;
    std::vector<double> X(static_cast<size_t>(n * p));
    for (int64_t i = 0; i < n * p; ++i) X[static_cast<size_t>(i)] = Xm->data[i];
    std::vector<double> y = vecOf(Ym);
    if (static_cast<int64_t>(y.size()) != n) return mat_alloc(0, 0);
    std::vector<double> beta, XtXinv;
    if (!ols(X, y, n, p, beta, &XtXinv)) return mat_alloc(0, 0);
    /* residuals */
    std::vector<double> u(static_cast<size_t>(n));
    for (int64_t t = 0; t < n; ++t) {
        double yhat = 0.0;
        for (int64_t j = 0; j < p; ++j)
            yhat += X[static_cast<size_t>(t * p + j)] *
                    beta[static_cast<size_t>(j)];
        u[static_cast<size_t>(t)] = y[static_cast<size_t>(t)] - yhat;
    }
    /* Bartlett-kernel meat: S = Σ_l w_l Σ_t x_t x_{t-l}' u_t u_{t-l}.
     * Bandwidth L = floor(4 (n/100)^{2/9}) (Newey-West rule of thumb). */
    int L = static_cast<int>(std::floor(4.0 *
                std::pow(static_cast<double>(n) / 100.0, 2.0 / 9.0)));
    if (L < 1) L = 1;
    std::vector<double> S(static_cast<size_t>(p * p), 0.0);
    auto addOuter = [&](int64_t t, int64_t s, double w) {
        double ut = u[static_cast<size_t>(t)] * u[static_cast<size_t>(s)];
        for (int64_t i = 0; i < p; ++i)
            for (int64_t j = 0; j < p; ++j)
                S[static_cast<size_t>(i * p + j)] += w *
                    X[static_cast<size_t>(t * p + i)] *
                    X[static_cast<size_t>(s * p + j)] * ut;
    };
    for (int64_t t = 0; t < n; ++t) addOuter(t, t, 1.0);
    for (int l = 1; l <= L; ++l) {
        double w = 1.0 - static_cast<double>(l) / static_cast<double>(L + 1);
        for (int64_t t = l; t < n; ++t) {
            addOuter(t, t - l, w);
            addOuter(t - l, t, w);
        }
    }
    /* Cov = (X'X)^{-1} S (X'X)^{-1}.  XtXinv is p x p. */
    std::vector<double> tmp(static_cast<size_t>(p * p), 0.0);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < p; ++k)
                s += XtXinv[static_cast<size_t>(i * p + k)] *
                     S[static_cast<size_t>(k * p + j)];
            tmp[static_cast<size_t>(i * p + j)] = s;
        }
    matlab_mat *out = mat_alloc(p, p);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < p; ++k)
                s += tmp[static_cast<size_t>(i * p + k)] *
                     XtXinv[static_cast<size_t>(k * p + j)];
            out->data[i * p + j] = s;
        }
    return out;
}

/* fgls(X, y) — feasible GLS for heteroscedastic errors.  Two-step: OLS,
 * model log(u²) on X to get variance weights, then weighted LS.  Returns
 * the estimated coefficient vector. */
matlab_mat *matlab_econ_fgls(matlab_mat *Xm, matlab_mat *Ym) {
    if (!Xm || !Ym || !Xm->data) return mat_alloc(0, 0);
    int64_t n = Xm->rows, p = Xm->cols;
    std::vector<double> X(static_cast<size_t>(n * p));
    for (int64_t i = 0; i < n * p; ++i) X[static_cast<size_t>(i)] = Xm->data[i];
    std::vector<double> y = vecOf(Ym);
    if (static_cast<int64_t>(y.size()) != n) return mat_alloc(0, 0);
    std::vector<double> beta;
    if (!ols(X, y, n, p, beta)) return mat_alloc(0, 0);
    std::vector<double> w(static_cast<size_t>(n));
    for (int64_t t = 0; t < n; ++t) {
        double yhat = 0.0;
        for (int64_t j = 0; j < p; ++j)
            yhat += X[static_cast<size_t>(t * p + j)] *
                    beta[static_cast<size_t>(j)];
        double r = y[static_cast<size_t>(t)] - yhat;
        double v = r * r;
        if (v < 1e-8) v = 1e-8;
        w[static_cast<size_t>(t)] = 1.0 / v;     /* inverse-variance weight */
    }
    /* Weighted LS: scale rows by sqrt(w). */
    std::vector<double> Xw(static_cast<size_t>(n * p)), yw(static_cast<size_t>(n));
    for (int64_t t = 0; t < n; ++t) {
        double sw = std::sqrt(w[static_cast<size_t>(t)]);
        for (int64_t j = 0; j < p; ++j)
            Xw[static_cast<size_t>(t * p + j)] =
                X[static_cast<size_t>(t * p + j)] * sw;
        yw[static_cast<size_t>(t)] = y[static_cast<size_t>(t)] * sw;
    }
    std::vector<double> bg;
    if (!ols(Xw, yw, n, p, bg)) return colVec(beta);
    return colVec(bg);
}

/* ============================================================================
 * §T1 — unit-root + stationarity tests
 * ==========================================================================*/

/* ADF regression t-statistic for the 'AR' model (no constant, no trend):
 *   Δy_t = ρ y_{t-1} + Σ_{i=1}^{lags} φ_i Δy_{t-i} + e_t
 * Returns the t-statistic on ρ (negative under stationarity). */
static double adf_tstat(const std::vector<double> &y, int lags) {
    int64_t N = static_cast<int64_t>(y.size());
    if (N < lags + 3) return 0.0;
    std::vector<double> dy(static_cast<size_t>(N - 1));
    for (int64_t t = 1; t < N; ++t)
        dy[static_cast<size_t>(t - 1)] =
            y[static_cast<size_t>(t)] - y[static_cast<size_t>(t - 1)];
    int64_t start = lags;             /* first usable index into dy */
    int64_t n = static_cast<int64_t>(dy.size()) - start;
    int64_t p = 1 + lags;             /* y_{t-1} + lagged diffs */
    std::vector<double> X(static_cast<size_t>(n * p)), rhs(static_cast<size_t>(n));
    for (int64_t r = 0; r < n; ++r) {
        int64_t i = start + r;        /* index into dy: dy[i] = y[i+1]-y[i] */
        X[static_cast<size_t>(r * p + 0)] = y[static_cast<size_t>(i)]; /* y_{t-1} */
        for (int k = 1; k <= lags; ++k)
            X[static_cast<size_t>(r * p + k)] = dy[static_cast<size_t>(i - k)];
        rhs[static_cast<size_t>(r)] = dy[static_cast<size_t>(i)];
    }
    std::vector<double> beta, XtXinv;
    if (!ols(X, rhs, n, p, beta, &XtXinv)) return 0.0;
    double sse = 0.0;
    for (int64_t r = 0; r < n; ++r) {
        double yhat = 0.0;
        for (int64_t j = 0; j < p; ++j)
            yhat += X[static_cast<size_t>(r * p + j)] *
                    beta[static_cast<size_t>(j)];
        double e = rhs[static_cast<size_t>(r)] - yhat;
        sse += e * e;
    }
    double s2 = sse / static_cast<double>(n - p);
    double se_rho = std::sqrt(s2 * XtXinv[0]);
    if (se_rho < 1e-300) return 0.0;
    return beta[0] / se_rho;
}

/* adftest(y, lags) — ADF unit-root test, 'AR' model.  Returns h (1 = reject
 * unit root, i.e. series is stationary @ 5%).  5% DF critical value (no
 * constant) ≈ -1.95. */
double matlab_econ_adftest_n(matlab_mat *Y, double lags) {
    std::vector<double> y = vecOf(Y);
    int l = static_cast<int>(lags);
    if (l < 0) l = 0;
    double t = adf_tstat(y, l);
    return (t < -1.95) ? 1.0 : 0.0;
}
double matlab_econ_adftest(matlab_mat *Y) {
    return matlab_econ_adftest_n(Y, 0.0);
}

/* pptest(y) — Phillips-Perron.  Uses the DF t-statistic (lags=0); the
 * nonparametric long-run-variance correction is a documented refinement.
 * Same 5% critical value as ADF 'AR'. */
double matlab_econ_pptest(matlab_mat *Y) {
    std::vector<double> y = vecOf(Y);
    double t = adf_tstat(y, 0);
    return (t < -1.95) ? 1.0 : 0.0;
}

/* KPSS-type stationarity statistic around a level (constant) regression:
 *   eta = (1/N²) Σ_t S_t² / s²,  S_t = Σ_{i≤t} (y_i - ybar). */
static double kpss_stat(const std::vector<double> &y) {
    int64_t N = static_cast<int64_t>(y.size());
    if (N < 3) return 0.0;
    double ybar = meanOf(y);
    std::vector<double> e(static_cast<size_t>(N));
    for (int64_t t = 0; t < N; ++t)
        e[static_cast<size_t>(t)] = y[static_cast<size_t>(t)] - ybar;
    /* long-run variance via Bartlett kernel */
    int L = static_cast<int>(std::floor(4.0 *
                std::pow(static_cast<double>(N) / 100.0, 0.25)));
    if (L < 1) L = 1;
    double g0 = 0.0;
    for (int64_t t = 0; t < N; ++t)
        g0 += e[static_cast<size_t>(t)] * e[static_cast<size_t>(t)];
    g0 /= static_cast<double>(N);
    double s2 = g0;
    for (int l = 1; l <= L; ++l) {
        double gl = 0.0;
        for (int64_t t = l; t < N; ++t)
            gl += e[static_cast<size_t>(t)] * e[static_cast<size_t>(t - l)];
        gl /= static_cast<double>(N);
        s2 += 2.0 * (1.0 - static_cast<double>(l) / static_cast<double>(L + 1)) * gl;
    }
    double St = 0.0, sumsq = 0.0;
    for (int64_t t = 0; t < N; ++t) {
        St += e[static_cast<size_t>(t)];
        sumsq += St * St;
    }
    if (s2 < 1e-300) return 0.0;
    return sumsq / (static_cast<double>(N) * static_cast<double>(N) * s2);
}

/* kpsstest(y) — KPSS stationarity test (null = stationary).  Returns h
 * (1 = reject stationarity / evidence of unit root @ 5%).  5% level-
 * stationary critical value = 0.463. */
double matlab_econ_kpsstest(matlab_mat *Y) {
    std::vector<double> y = vecOf(Y);
    double eta = kpss_stat(y);
    return (eta > 0.463) ? 1.0 : 0.0;
}

/* lmctest(y) — Leybourne-McCabe stationarity test (null = stationary AR).
 * Approximated by the KPSS statistic (documented refinement: the LM
 * variant filters an AR(1) first).  Same critical value. */
double matlab_econ_lmctest(matlab_mat *Y) {
    std::vector<double> y = vecOf(Y);
    double eta = kpss_stat(y);
    return (eta > 0.463) ? 1.0 : 0.0;
}

/* vratiotest(y) — Lo-MacKinlay variance-ratio test (null = random walk).
 * VR(q) = Var(q-diff) / (q Var(1-diff)); z ~ N(0,1) under H0 (homoscedastic).
 * Returns h (1 = reject random walk @ 5%, two-sided).  q = 2. */
double matlab_econ_vratiotest(matlab_mat *Y) {
    std::vector<double> y = vecOf(Y);
    int64_t N = static_cast<int64_t>(y.size());
    if (N < 5) return 0.0;
    int q = 2;
    /* 1-period diffs */
    std::vector<double> d(static_cast<size_t>(N - 1));
    for (int64_t t = 1; t < N; ++t)
        d[static_cast<size_t>(t - 1)] =
            y[static_cast<size_t>(t)] - y[static_cast<size_t>(t - 1)];
    double mu = meanOf(d);
    double var1 = 0.0;
    for (double x : d) var1 += (x - mu) * (x - mu);
    var1 /= static_cast<double>(d.size());
    /* q-period diffs (overlapping) */
    double varq = 0.0; int64_t cnt = 0;
    for (int64_t t = q; t < N; ++t) {
        double dq = y[static_cast<size_t>(t)] - y[static_cast<size_t>(t - q)];
        double m = static_cast<double>(q) * mu;
        varq += (dq - m) * (dq - m);
        ++cnt;
    }
    varq /= static_cast<double>(cnt) * static_cast<double>(q);
    if (var1 < 1e-300) return 0.0;
    double VR = varq / var1;
    double n = static_cast<double>(N - 1);
    /* asymptotic variance of VR(q) under homoscedastic RW */
    double phi = 2.0 * (2.0 * q - 1.0) * (q - 1.0) /
                 (3.0 * static_cast<double>(q) * n);
    if (phi < 1e-300) return 0.0;
    double z = (VR - 1.0) / std::sqrt(phi);
    double pv = 2.0 * (1.0 - norm_cdf(std::fabs(z)));
    return (pv < 0.05) ? 1.0 : 0.0;
}

} // extern "C"

/* ============================================================================
 * §T2 — Conditional Mean Models: arima(p,D,q)
 * ----------------------------------------------------------------------------
 * Estimation: Hannan-Rissanen (fit a long AR by OLS to recover innovation
 * proxies, then OLS-regress the differenced series on its own lags + the
 * innovation lags).  Forecasting: recursive MMSE on the differenced series,
 * then integrated back D times.  All over the shipped `ols` helper above.
 * ==========================================================================*/

extern "C" {
double matlab_obj_get_f64(struct matlab_obj_s *o, const char *name, int64_t len);
matlab_mat *matlab_obj_get_mat(struct matlab_obj_s *o, const char *name, int64_t len);
void matlab_obj_set_f64(struct matlab_obj_s *o, const char *name, int64_t len, double v);
void matlab_obj_set_mat(struct matlab_obj_s *o, const char *name, int64_t len, matlab_mat *m);
}

namespace {

/* Difference a series D times: w = (1-L)^D y. */
std::vector<double> differenceD(const std::vector<double> &y, int D) {
    std::vector<double> w = y;
    for (int d = 0; d < D; ++d) {
        std::vector<double> nw(w.size() > 0 ? w.size() - 1 : 0);
        for (size_t t = 1; t < w.size(); ++t) nw[t - 1] = w[t] - w[t - 1];
        w = nw;
    }
    return w;
}

/* Hannan-Rissanen estimation of ARMA(p,q) (with intercept) on series w.
 * Fills phi (p), theta (q), and returns the intercept c; resid variance in
 * sigma2; in-sample residuals in eHat (length = w.size()). */
double arma_hr(const std::vector<double> &w, int p, int q,
               std::vector<double> &phi, std::vector<double> &theta,
               double &sigma2, std::vector<double> &eHat) {
    int64_t M = static_cast<int64_t>(w.size());
    phi.assign(static_cast<size_t>(p), 0.0);
    theta.assign(static_cast<size_t>(q), 0.0);
    eHat.assign(static_cast<size_t>(M), 0.0);
    sigma2 = 0.0;
    if (M < p + q + 4) return 0.0;
    /* Stage 1: long AR(m) by OLS to get innovation proxies. */
    int m = p + q + 8;
    if (m > static_cast<int>(M / 3)) m = static_cast<int>(M / 3);
    if (m < 1) m = 1;
    int64_t n1 = M - m, pp1 = m + 1;
    std::vector<double> X1(static_cast<size_t>(n1 * pp1)),
        y1(static_cast<size_t>(n1));
    for (int64_t r = 0; r < n1; ++r) {
        int64_t t = r + m;
        X1[static_cast<size_t>(r * pp1 + 0)] = 1.0;
        for (int i = 1; i <= m; ++i)
            X1[static_cast<size_t>(r * pp1 + i)] = w[static_cast<size_t>(t - i)];
        y1[static_cast<size_t>(r)] = w[static_cast<size_t>(t)];
    }
    std::vector<double> b1;
    if (!ols(X1, y1, n1, pp1, b1)) return 0.0;
    for (int64_t t = m; t < M; ++t) {
        double yhat = b1[0];
        for (int i = 1; i <= m; ++i)
            yhat += b1[static_cast<size_t>(i)] * w[static_cast<size_t>(t - i)];
        eHat[static_cast<size_t>(t)] = w[static_cast<size_t>(t)] - yhat;
    }
    /* Stage 2: regress w_t on [1, w_{t-1..p}, eHat_{t-1..q}]. */
    int64_t start = m;            /* eHat valid from index m */
    if (start < p) start = p;
    if (start < q) start = q;
    int64_t n2 = M - start, pp2 = 1 + p + q;
    if (n2 < pp2 + 1) { sigma2 = 0.0; return b1[0]; }
    std::vector<double> X2(static_cast<size_t>(n2 * pp2)),
        y2(static_cast<size_t>(n2));
    for (int64_t r = 0; r < n2; ++r) {
        int64_t t = start + r;
        int64_t col = 0;
        X2[static_cast<size_t>(r * pp2 + col++)] = 1.0;
        for (int i = 1; i <= p; ++i)
            X2[static_cast<size_t>(r * pp2 + col++)] =
                w[static_cast<size_t>(t - i)];
        for (int j = 1; j <= q; ++j)
            X2[static_cast<size_t>(r * pp2 + col++)] =
                eHat[static_cast<size_t>(t - j)];
        y2[static_cast<size_t>(r)] = w[static_cast<size_t>(t)];
    }
    std::vector<double> b2;
    if (!ols(X2, y2, n2, pp2, b2)) return b1[0];
    double c = b2[0];
    for (int i = 0; i < p; ++i) phi[static_cast<size_t>(i)] = b2[static_cast<size_t>(1 + i)];
    for (int j = 0; j < q; ++j) theta[static_cast<size_t>(j)] = b2[static_cast<size_t>(1 + p + j)];
    /* Recompute residuals with the ARMA recursion and the variance. */
    std::vector<double> e(static_cast<size_t>(M), 0.0);
    double sse = 0.0; int64_t cnt = 0;
    for (int64_t t = 0; t < M; ++t) {
        if (t < start) { e[static_cast<size_t>(t)] = 0.0; continue; }
        double pred = c;
        for (int i = 0; i < p; ++i)
            pred += phi[static_cast<size_t>(i)] * w[static_cast<size_t>(t - 1 - i)];
        for (int j = 0; j < q; ++j)
            pred += theta[static_cast<size_t>(j)] * e[static_cast<size_t>(t - 1 - j)];
        e[static_cast<size_t>(t)] = w[static_cast<size_t>(t)] - pred;
        sse += e[static_cast<size_t>(t)] * e[static_cast<size_t>(t)];
        ++cnt;
    }
    eHat = e;
    sigma2 = (cnt > 0) ? sse / static_cast<double>(cnt) : 0.0;
    return c;
}

/* Box-Muller normal from a fixed LCG (deterministic simulate). */
struct Lcg { uint64_t s; double next() {
    s = (6364136223846793005ULL * s + 1442695040888963407ULL);
    return static_cast<double>(s >> 11) / 9007199254740992.0; } };
double normRand(Lcg &g) {
    double u1 = g.next(), u2 = g.next();
    if (u1 < 1e-12) u1 = 1e-12;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

} // namespace

extern "C" {

/* estimate(fresh, template, y) — copy the (p,D,q) orders from the template
 * onto the freshly-constructed object, fit by Hannan-Rissanen, and populate
 * the fresh object in place. */
struct matlab_obj_s *matlab_econ_arima_estimate(struct matlab_obj_s *mdl,
                                                 struct matlab_obj_s *tmpl,
                                                 matlab_mat *Y) {
    if (!mdl) return mdl;
    int p = static_cast<int>(matlab_obj_get_f64(tmpl, "P", 1));
    int D = static_cast<int>(matlab_obj_get_f64(tmpl, "D", 1));
    int q = static_cast<int>(matlab_obj_get_f64(tmpl, "Q", 1));
    matlab_obj_set_f64(mdl, "P", 1, static_cast<double>(p));
    matlab_obj_set_f64(mdl, "D", 1, static_cast<double>(D));
    matlab_obj_set_f64(mdl, "Q", 1, static_cast<double>(q));
    std::vector<double> y = vecOf(Y);
    std::vector<double> w = differenceD(y, D);
    std::vector<double> phi, theta, eHat;
    double sigma2 = 0.0;
    double c = arma_hr(w, p, q, phi, theta, sigma2, eHat);
    matlab_obj_set_f64(mdl, "Constant", 8, c);
    matlab_obj_set_f64(mdl, "Variance", 8, sigma2);
    matlab_mat *arm = mat_alloc(1, p > 0 ? p : 1);
    for (int i = 0; i < p; ++i) arm->data[i] = phi[static_cast<size_t>(i)];
    matlab_mat *mam = mat_alloc(1, q > 0 ? q : 1);
    for (int j = 0; j < q; ++j) mam->data[j] = theta[static_cast<size_t>(j)];
    matlab_obj_set_mat(mdl, "AR", 2, arm);
    matlab_obj_set_mat(mdl, "MA", 2, mam);
    return mdl;
}

/* infer(Mdl, y) — in-sample innovations (residuals) on the differenced
 * series, length = numel(diff^D y). */
matlab_mat *matlab_econ_arima_infer(struct matlab_obj_s *mdl, matlab_mat *Y) {
    if (!mdl) return mat_alloc(0, 0);
    int p = static_cast<int>(matlab_obj_get_f64(mdl, "P", 1));
    int D = static_cast<int>(matlab_obj_get_f64(mdl, "D", 1));
    int q = static_cast<int>(matlab_obj_get_f64(mdl, "Q", 1));
    double c = matlab_obj_get_f64(mdl, "Constant", 8);
    matlab_mat *arm = matlab_obj_get_mat(mdl, "AR", 2);
    matlab_mat *mam = matlab_obj_get_mat(mdl, "MA", 2);
    std::vector<double> y = vecOf(Y);
    std::vector<double> w = differenceD(y, D);
    int64_t M = static_cast<int64_t>(w.size());
    std::vector<double> e(static_cast<size_t>(M), 0.0);
    int start = (p > q ? p : q);
    for (int64_t t = start; t < M; ++t) {
        double pred = c;
        for (int i = 0; i < p && arm; ++i)
            pred += arm->data[i] * w[static_cast<size_t>(t - 1 - i)];
        for (int j = 0; j < q && mam; ++j)
            pred += mam->data[j] * e[static_cast<size_t>(t - 1 - j)];
        e[static_cast<size_t>(t)] = w[static_cast<size_t>(t)] - pred;
    }
    return colVec(e);
}

/* forecast(Mdl, h, y) — h-step MMSE forecast (column vector). */
matlab_mat *matlab_econ_arima_forecast(struct matlab_obj_s *mdl, double hh,
                                       matlab_mat *Y) {
    if (!mdl) return mat_alloc(0, 0);
    int h = static_cast<int>(hh);
    if (h < 1) return mat_alloc(0, 0);
    int p = static_cast<int>(matlab_obj_get_f64(mdl, "P", 1));
    int D = static_cast<int>(matlab_obj_get_f64(mdl, "D", 1));
    int q = static_cast<int>(matlab_obj_get_f64(mdl, "Q", 1));
    double c = matlab_obj_get_f64(mdl, "Constant", 8);
    matlab_mat *arm = matlab_obj_get_mat(mdl, "AR", 2);
    matlab_mat *mam = matlab_obj_get_mat(mdl, "MA", 2);
    std::vector<double> y = vecOf(Y);
    std::vector<double> w = differenceD(y, D);
    int64_t M = static_cast<int64_t>(w.size());
    /* in-sample residuals */
    std::vector<double> e(static_cast<size_t>(M), 0.0);
    int start = (p > q ? p : q);
    for (int64_t t = start; t < M; ++t) {
        double pred = c;
        for (int i = 0; i < p && arm; ++i)
            pred += arm->data[i] * w[static_cast<size_t>(t - 1 - i)];
        for (int j = 0; j < q && mam; ++j)
            pred += mam->data[j] * e[static_cast<size_t>(t - 1 - j)];
        e[static_cast<size_t>(t)] = w[static_cast<size_t>(t)] - pred;
    }
    /* extend w (and e=0 for future) recursively */
    std::vector<double> we = w;
    std::vector<double> ee = e;
    for (int k = 0; k < h; ++k) {
        double pred = c;
        int64_t t = static_cast<int64_t>(we.size());
        for (int i = 0; i < p && arm; ++i)
            if (t - 1 - i >= 0)
                pred += arm->data[i] * we[static_cast<size_t>(t - 1 - i)];
        for (int j = 0; j < q && mam; ++j)
            if (t - 1 - j >= 0)
                pred += mam->data[j] * ee[static_cast<size_t>(t - 1 - j)];
        we.push_back(pred);
        ee.push_back(0.0);     /* future innovation expectation = 0 */
    }
    /* integrate the forecast differences back D times.  We need the last D
     * "levels" at each integration stage; rebuild from y. */
    std::vector<double> fdiff(we.begin() + M, we.end());  /* h forecast diffs */
    /* For D integrations, carry the tail values of each differenced order. */
    if (D == 0) return colVec(fdiff);
    /* Build the D successive difference series of y and take their tails. */
    std::vector<std::vector<double>> levels;       /* levels[0] = y, [1]=diff y, ... */
    levels.push_back(y);
    for (int d = 1; d <= D; ++d) levels.push_back(differenceD(y, d));
    /* Integrate: start from order D forecast (fdiff) up to order 0. */
    std::vector<double> cur = fdiff;               /* order-D forecasts */
    for (int d = D - 1; d >= 0; --d) {
        std::vector<double> &lev = levels[static_cast<size_t>(d)];
        double last = lev.empty() ? 0.0 : lev.back();
        std::vector<double> integ(cur.size());
        double acc = last;
        for (size_t k = 0; k < cur.size(); ++k) { acc += cur[k]; integ[k] = acc; }
        cur = integ;
    }
    return colVec(cur);
}

/* simulate(Mdl, n) — one simulated path of length n (deterministic LCG). */
matlab_mat *matlab_econ_arima_simulate(struct matlab_obj_s *mdl, double nn) {
    if (!mdl) return mat_alloc(0, 0);
    int n = static_cast<int>(nn);
    if (n < 1) return mat_alloc(0, 0);
    int p = static_cast<int>(matlab_obj_get_f64(mdl, "P", 1));
    int D = static_cast<int>(matlab_obj_get_f64(mdl, "D", 1));
    int q = static_cast<int>(matlab_obj_get_f64(mdl, "Q", 1));
    double c = matlab_obj_get_f64(mdl, "Constant", 8);
    double v = matlab_obj_get_f64(mdl, "Variance", 8);
    if (v <= 0.0) v = 1.0;
    matlab_mat *arm = matlab_obj_get_mat(mdl, "AR", 2);
    matlab_mat *mam = matlab_obj_get_mat(mdl, "MA", 2);
    double sd = std::sqrt(v);
    Lcg g{88172645463325252ULL};
    int burn = 50;
    int M = n + D + burn;
    std::vector<double> w(static_cast<size_t>(M), 0.0),
        e(static_cast<size_t>(M), 0.0);
    for (int t = 0; t < M; ++t) {
        double et = sd * normRand(g);
        e[static_cast<size_t>(t)] = et;
        double val = c + et;
        for (int i = 0; i < p && arm; ++i)
            if (t - 1 - i >= 0)
                val += arm->data[i] * w[static_cast<size_t>(t - 1 - i)];
        for (int j = 0; j < q && mam; ++j)
            if (t - 1 - j >= 0)
                val += mam->data[j] * e[static_cast<size_t>(t - 1 - j)];
        w[static_cast<size_t>(t)] = val;
    }
    /* drop burn-in, then integrate D times */
    std::vector<double> ws(w.begin() + burn, w.end());   /* length n + D */
    std::vector<double> cur = ws;
    for (int d = 0; d < D; ++d) {
        std::vector<double> integ(cur.size());
        double acc = 0.0;
        for (size_t k = 0; k < cur.size(); ++k) { acc += cur[k]; integ[k] = acc; }
        cur = integ;
    }
    /* take the last n points */
    std::vector<double> out(cur.end() - n, cur.end());
    return colVec(out);
}

} // extern "C"
