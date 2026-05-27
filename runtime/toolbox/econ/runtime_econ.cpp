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

/* ============================================================================
 * §T3 — Conditional Variance Models: garch / egarch / gjr
 * ----------------------------------------------------------------------------
 * Gaussian maximum likelihood of the conditional-variance recursion,
 * maximised by a self-contained Nelder-Mead simplex (the parameter space is
 * low-dimensional).  ModelKind discriminates the three variants (1=garch,
 * 2=egarch, 3=gjr); one set of C-ABI entry points serves all three.
 * ==========================================================================*/

namespace {

constexpr double kSqrt2OverPi = 0.7978845608028654; /* E|z|, z~N(0,1) */

/* Conditional-variance recursion + Gaussian negative log-likelihood.
 * kind: 1=garch(P,Q), 2=egarch(1,1), 3=gjr(1,1).  Layout of theta:
 *   garch:  [kappa, gamma_1..P, alpha_1..Q]
 *   egarch: [kappa, gamma, alpha, leverage]            (1,1)
 *   gjr:    [kappa, gamma, alpha, leverage]            (1,1)
 * Returns 1e18 for infeasible parameters (constraint barrier). */
double garch_nll(const std::vector<double> &theta, int kind,
                 const std::vector<double> &e, int P, int Q,
                 double uncondVar) {
    int64_t M = static_cast<int64_t>(e.size());
    if (M < 5) return 1e18;
    std::vector<double> h(static_cast<size_t>(M), uncondVar);
    if (kind == 1) {
        double kappa = theta[0];
        if (kappa <= 1e-10) return 1e18;
        double sg = 0.0, sa = 0.0;
        for (int i = 0; i < P; ++i) {
            if (theta[1 + i] < 0.0) return 1e18;
            sg += theta[1 + i];
        }
        for (int j = 0; j < Q; ++j) {
            if (theta[1 + P + j] < 0.0) return 1e18;
            sa += theta[1 + P + j];
        }
        if (sg + sa >= 0.999) return 1e18;
        int start = (P > Q ? P : Q);
        for (int64_t t = start; t < M; ++t) {
            double v = kappa;
            for (int i = 0; i < P; ++i)
                v += theta[1 + i] * h[static_cast<size_t>(t - 1 - i)];
            for (int j = 0; j < Q; ++j)
                v += theta[1 + P + j] *
                     e[static_cast<size_t>(t - 1 - j)] *
                     e[static_cast<size_t>(t - 1 - j)];
            if (v <= 1e-12) return 1e18;
            h[static_cast<size_t>(t)] = v;
        }
        double nll = 0.0;
        for (int64_t t = start; t < M; ++t) {
            double ht = h[static_cast<size_t>(t)];
            nll += 0.5 * (std::log(2.0 * M_PI) + std::log(ht) +
                          e[static_cast<size_t>(t)] *
                              e[static_cast<size_t>(t)] / ht);
        }
        return nll;
    }
    if (kind == 3) {  /* gjr(1,1) */
        double kappa = theta[0], g = theta[1], a = theta[2], xi = theta[3];
        if (kappa <= 1e-10 || g < 0.0 || a < 0.0) return 1e18;
        if (g + a + 0.5 * xi >= 0.999 || a + xi < 0.0) return 1e18;
        for (int64_t t = 1; t < M; ++t) {
            double em = e[static_cast<size_t>(t - 1)];
            double ind = (em < 0.0) ? 1.0 : 0.0;
            double v = kappa + g * h[static_cast<size_t>(t - 1)] +
                       a * em * em + xi * em * em * ind;
            if (v <= 1e-12) return 1e18;
            h[static_cast<size_t>(t)] = v;
        }
        double nll = 0.0;
        for (int64_t t = 1; t < M; ++t) {
            double ht = h[static_cast<size_t>(t)];
            nll += 0.5 * (std::log(2.0 * M_PI) + std::log(ht) +
                          e[static_cast<size_t>(t)] *
                              e[static_cast<size_t>(t)] / ht);
        }
        return nll;
    }
    /* egarch(1,1): log h_t = kappa + g*log h_{t-1}
     *              + a*(|z|-E|z|) + xi*z,   z = e_{t-1}/sqrt(h_{t-1}) */
    double kappa = theta[0], g = theta[1], a = theta[2], xi = theta[3];
    if (std::fabs(g) >= 0.999) return 1e18;
    std::vector<double> lh(static_cast<size_t>(M), std::log(uncondVar));
    for (int64_t t = 1; t < M; ++t) {
        double hp = std::exp(lh[static_cast<size_t>(t - 1)]);
        double z = e[static_cast<size_t>(t - 1)] / std::sqrt(hp);
        double v = kappa + g * lh[static_cast<size_t>(t - 1)] +
                   a * (std::fabs(z) - kSqrt2OverPi) + xi * z;
        lh[static_cast<size_t>(t)] = v;
    }
    double nll = 0.0;
    for (int64_t t = 1; t < M; ++t) {
        double ht = std::exp(lh[static_cast<size_t>(t)]);
        if (ht <= 1e-12 || !std::isfinite(ht)) return 1e18;
        nll += 0.5 * (std::log(2.0 * M_PI) + lh[static_cast<size_t>(t)] +
                      e[static_cast<size_t>(t)] *
                          e[static_cast<size_t>(t)] / ht);
    }
    return nll;
}

/* Nelder-Mead simplex minimisation of f over an n-dim parameter vector. */
template <typename F>
std::vector<double> nelder_mead(F f, std::vector<double> x0, int iters) {
    int n = static_cast<int>(x0.size());
    std::vector<std::vector<double>> S(n + 1, x0);
    for (int i = 0; i < n; ++i) {
        double step = (std::fabs(x0[static_cast<size_t>(i)]) > 1e-6)
                          ? 0.1 * x0[static_cast<size_t>(i)]
                          : 0.05;
        S[i + 1][static_cast<size_t>(i)] += step;
    }
    std::vector<double> fv(n + 1);
    for (int i = 0; i <= n; ++i) fv[static_cast<size_t>(i)] = f(S[i]);
    for (int it = 0; it < iters; ++it) {
        /* order */
        for (int i = 0; i <= n; ++i)
            for (int j = i + 1; j <= n; ++j)
                if (fv[static_cast<size_t>(j)] < fv[static_cast<size_t>(i)]) {
                    std::swap(fv[static_cast<size_t>(i)], fv[static_cast<size_t>(j)]);
                    std::swap(S[i], S[j]);
                }
        /* centroid of all but worst */
        std::vector<double> c(static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                c[static_cast<size_t>(k)] += S[i][static_cast<size_t>(k)] / n;
        std::vector<double> xr(static_cast<size_t>(n));
        for (int k = 0; k < n; ++k)
            xr[static_cast<size_t>(k)] =
                c[static_cast<size_t>(k)] +
                1.0 * (c[static_cast<size_t>(k)] - S[n][static_cast<size_t>(k)]);
        double fr = f(xr);
        if (fr < fv[0]) {
            std::vector<double> xe(static_cast<size_t>(n));
            for (int k = 0; k < n; ++k)
                xe[static_cast<size_t>(k)] =
                    c[static_cast<size_t>(k)] +
                    2.0 * (c[static_cast<size_t>(k)] - S[n][static_cast<size_t>(k)]);
            double fe = f(xe);
            if (fe < fr) { S[n] = xe; fv[static_cast<size_t>(n)] = fe; }
            else { S[n] = xr; fv[static_cast<size_t>(n)] = fr; }
        } else if (fr < fv[static_cast<size_t>(n - 1)]) {
            S[n] = xr; fv[static_cast<size_t>(n)] = fr;
        } else {
            std::vector<double> xc(static_cast<size_t>(n));
            for (int k = 0; k < n; ++k)
                xc[static_cast<size_t>(k)] =
                    c[static_cast<size_t>(k)] +
                    0.5 * (S[n][static_cast<size_t>(k)] - c[static_cast<size_t>(k)]);
            double fc = f(xc);
            if (fc < fv[static_cast<size_t>(n)]) {
                S[n] = xc; fv[static_cast<size_t>(n)] = fc;
            } else {
                for (int i = 1; i <= n; ++i) {
                    for (int k = 0; k < n; ++k)
                        S[i][static_cast<size_t>(k)] =
                            S[0][static_cast<size_t>(k)] +
                            0.5 * (S[i][static_cast<size_t>(k)] -
                                   S[0][static_cast<size_t>(k)]);
                    fv[static_cast<size_t>(i)] = f(S[i]);
                }
            }
        }
    }
    int best = 0;
    for (int i = 1; i <= n; ++i)
        if (fv[static_cast<size_t>(i)] < fv[static_cast<size_t>(best)]) best = i;
    return S[best];
}

} // namespace

extern "C" {

/* estimate(fresh, template, y) for garch/egarch/gjr. */
struct matlab_obj_s *matlab_econ_garch_estimate(struct matlab_obj_s *mdl,
                                                 struct matlab_obj_s *tmpl,
                                                 matlab_mat *Y) {
    if (!mdl) return mdl;
    int kind = static_cast<int>(matlab_obj_get_f64(tmpl, "ModelKind", 9));
    int P = static_cast<int>(matlab_obj_get_f64(tmpl, "P", 1));
    int Q = static_cast<int>(matlab_obj_get_f64(tmpl, "Q", 1));
    if (P < 1) P = 1;
    if (Q < 1) Q = 1;
    matlab_obj_set_f64(mdl, "ModelKind", 9, static_cast<double>(kind));
    matlab_obj_set_f64(mdl, "P", 1, static_cast<double>(P));
    matlab_obj_set_f64(mdl, "Q", 1, static_cast<double>(Q));
    std::vector<double> y = vecOf(Y);
    double c = meanOf(y);
    std::vector<double> e(y.size());
    double var = 0.0;
    for (size_t i = 0; i < y.size(); ++i) { e[i] = y[i] - c; var += e[i] * e[i]; }
    var = (y.empty() ? 1.0 : var / static_cast<double>(y.size()));
    if (var <= 0.0) var = 1.0;

    std::vector<double> theta;
    if (kind == 1) {
        theta.assign(static_cast<size_t>(1 + P + Q), 0.0);
        theta[0] = 0.1 * var;                     /* kappa */
        for (int i = 0; i < P; ++i) theta[1 + i] = 0.8 / P;
        for (int j = 0; j < Q; ++j) theta[1 + P + j] = 0.1 / Q;
    } else {
        theta.assign(4, 0.0);
        if (kind == 2) { theta = {0.0, 0.9, 0.2, -0.05}; }     /* egarch */
        else { theta[0] = 0.1 * var; theta[1] = 0.8; theta[2] = 0.05; theta[3] = 0.05; }
    }
    auto obj = [&](const std::vector<double> &th) {
        return garch_nll(th, kind, e, P, Q, var);
    };
    std::vector<double> best = nelder_mead(obj, theta, 800);

    matlab_obj_set_f64(mdl, "Offset", 6, c);
    matlab_obj_set_f64(mdl, "Variance", 8, var);
    if (kind == 1) {
        matlab_obj_set_f64(mdl, "Constant", 8, best[0]);
        matlab_mat *gm = mat_alloc(1, P);
        for (int i = 0; i < P; ++i) gm->data[i] = best[1 + i];
        matlab_mat *am = mat_alloc(1, Q);
        for (int j = 0; j < Q; ++j) am->data[j] = best[1 + P + j];
        matlab_obj_set_mat(mdl, "GARCH", 5, gm);
        matlab_obj_set_mat(mdl, "ARCH", 4, am);
        matlab_mat *lv = mat_alloc(1, 1);
        matlab_obj_set_mat(mdl, "Leverage", 8, lv);
    } else {
        matlab_obj_set_f64(mdl, "Constant", 8, best[0]);
        matlab_mat *gm = mat_alloc(1, 1); gm->data[0] = best[1];
        matlab_mat *am = mat_alloc(1, 1); am->data[0] = best[2];
        matlab_mat *lv = mat_alloc(1, 1); lv->data[0] = best[3];
        matlab_obj_set_mat(mdl, "GARCH", 5, gm);
        matlab_obj_set_mat(mdl, "ARCH", 4, am);
        matlab_obj_set_mat(mdl, "Leverage", 8, lv);
    }
    return mdl;
}

/* infer(Mdl, y) — conditional variances h_t (column vector, length numel(y)). */
matlab_mat *matlab_econ_garch_infer(struct matlab_obj_s *mdl, matlab_mat *Y) {
    if (!mdl) return mat_alloc(0, 0);
    int kind = static_cast<int>(matlab_obj_get_f64(mdl, "ModelKind", 9));
    int P = static_cast<int>(matlab_obj_get_f64(mdl, "P", 1));
    int Q = static_cast<int>(matlab_obj_get_f64(mdl, "Q", 1));
    double c = matlab_obj_get_f64(mdl, "Offset", 6);
    double kappa = matlab_obj_get_f64(mdl, "Constant", 8);
    double var = matlab_obj_get_f64(mdl, "Variance", 8);
    matlab_mat *gm = matlab_obj_get_mat(mdl, "GARCH", 5);
    matlab_mat *am = matlab_obj_get_mat(mdl, "ARCH", 4);
    matlab_mat *lv = matlab_obj_get_mat(mdl, "Leverage", 8);
    std::vector<double> y = vecOf(Y);
    int64_t M = static_cast<int64_t>(y.size());
    std::vector<double> e(static_cast<size_t>(M));
    for (int64_t i = 0; i < M; ++i) e[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] - c;
    std::vector<double> h(static_cast<size_t>(M), var);
    if (kind == 1) {
        int start = (P > Q ? P : Q);
        for (int64_t t = start; t < M; ++t) {
            double v = kappa;
            for (int i = 0; i < P && gm; ++i)
                v += gm->data[i] * h[static_cast<size_t>(t - 1 - i)];
            for (int j = 0; j < Q && am; ++j)
                v += am->data[j] * e[static_cast<size_t>(t - 1 - j)] *
                     e[static_cast<size_t>(t - 1 - j)];
            h[static_cast<size_t>(t)] = v;
        }
    } else if (kind == 3) {
        double g = gm ? gm->data[0] : 0.0, a = am ? am->data[0] : 0.0,
               xi = lv ? lv->data[0] : 0.0;
        for (int64_t t = 1; t < M; ++t) {
            double em = e[static_cast<size_t>(t - 1)];
            double ind = (em < 0.0) ? 1.0 : 0.0;
            h[static_cast<size_t>(t)] = kappa + g * h[static_cast<size_t>(t - 1)] +
                                        a * em * em + xi * em * em * ind;
        }
    } else {
        double g = gm ? gm->data[0] : 0.0, a = am ? am->data[0] : 0.0,
               xi = lv ? lv->data[0] : 0.0;
        std::vector<double> lh(static_cast<size_t>(M), std::log(var));
        for (int64_t t = 1; t < M; ++t) {
            double hp = std::exp(lh[static_cast<size_t>(t - 1)]);
            double z = e[static_cast<size_t>(t - 1)] / std::sqrt(hp);
            lh[static_cast<size_t>(t)] = kappa + g * lh[static_cast<size_t>(t - 1)] +
                                         a * (std::fabs(z) - kSqrt2OverPi) + xi * z;
            h[static_cast<size_t>(t)] = std::exp(lh[static_cast<size_t>(t)]);
        }
        h[0] = var;
    }
    return colVec(h);
}

/* forecast(Mdl, h, y) — h-step-ahead conditional-variance forecast. */
matlab_mat *matlab_econ_garch_forecast(struct matlab_obj_s *mdl, double hh,
                                       matlab_mat *Y) {
    if (!mdl) return mat_alloc(0, 0);
    int H = static_cast<int>(hh);
    if (H < 1) return mat_alloc(0, 0);
    int kind = static_cast<int>(matlab_obj_get_f64(mdl, "ModelKind", 9));
    double kappa = matlab_obj_get_f64(mdl, "Constant", 8);
    double var = matlab_obj_get_f64(mdl, "Variance", 8);
    matlab_mat *gm = matlab_obj_get_mat(mdl, "GARCH", 5);
    matlab_mat *am = matlab_obj_get_mat(mdl, "ARCH", 4);
    /* last in-sample conditional variance */
    matlab_mat *hin = matlab_econ_garch_infer(mdl, Y);
    double hlast = (hin && hin->rows > 0) ? hin->data[hin->rows * hin->cols - 1] : var;
    double g = gm ? gm->data[0] : 0.0, a = am ? am->data[0] : 0.0;
    std::vector<double> f(static_cast<size_t>(H));
    double persist = (kind == 2) ? g : (g + a);   /* egarch persistence ~ g */
    double hf = hlast;
    for (int k = 0; k < H; ++k) {
        if (kind == 2) {
            /* mean-revert log-variance toward kappa/(1-g) */
            double lvar = (std::fabs(1.0 - g) < 1e-9) ? std::log(var)
                            : kappa / (1.0 - g);
            double lh = std::log(hf);
            lh = kappa + g * lh;
            (void)lvar;
            hf = std::exp(lh);
        } else {
            hf = kappa + persist * hf;
        }
        f[static_cast<size_t>(k)] = hf;
    }
    return colVec(f);
}

/* simulate(Mdl, n) — one simulated return path of length n. */
matlab_mat *matlab_econ_garch_simulate(struct matlab_obj_s *mdl, double nn) {
    if (!mdl) return mat_alloc(0, 0);
    int n = static_cast<int>(nn);
    if (n < 1) return mat_alloc(0, 0);
    int kind = static_cast<int>(matlab_obj_get_f64(mdl, "ModelKind", 9));
    double c = matlab_obj_get_f64(mdl, "Offset", 6);
    double kappa = matlab_obj_get_f64(mdl, "Constant", 8);
    double var = matlab_obj_get_f64(mdl, "Variance", 8);
    matlab_mat *gm = matlab_obj_get_mat(mdl, "GARCH", 5);
    matlab_mat *am = matlab_obj_get_mat(mdl, "ARCH", 4);
    matlab_mat *lv = matlab_obj_get_mat(mdl, "Leverage", 8);
    double g = gm ? gm->data[0] : 0.0, a = am ? am->data[0] : 0.0,
           xi = lv ? lv->data[0] : 0.0;
    int burn = 100, M = n + burn;
    Lcg rng{0x9e3779b97f4a7c15ULL};
    std::vector<double> h(static_cast<size_t>(M), var), e(static_cast<size_t>(M), 0.0);
    std::vector<double> lh(static_cast<size_t>(M), std::log(var));
    for (int t = 1; t < M; ++t) {
        if (kind == 1) {
            h[static_cast<size_t>(t)] = kappa + g * h[static_cast<size_t>(t - 1)] +
                a * e[static_cast<size_t>(t - 1)] * e[static_cast<size_t>(t - 1)];
        } else if (kind == 3) {
            double em = e[static_cast<size_t>(t - 1)];
            double ind = (em < 0.0) ? 1.0 : 0.0;
            h[static_cast<size_t>(t)] = kappa + g * h[static_cast<size_t>(t - 1)] +
                a * em * em + xi * em * em * ind;
        } else {
            double hp = std::exp(lh[static_cast<size_t>(t - 1)]);
            double z = e[static_cast<size_t>(t - 1)] / std::sqrt(hp);
            lh[static_cast<size_t>(t)] = kappa + g * lh[static_cast<size_t>(t - 1)] +
                a * (std::fabs(z) - kSqrt2OverPi) + xi * z;
            h[static_cast<size_t>(t)] = std::exp(lh[static_cast<size_t>(t)]);
        }
        if (h[static_cast<size_t>(t)] <= 0.0) h[static_cast<size_t>(t)] = var;
        e[static_cast<size_t>(t)] = std::sqrt(h[static_cast<size_t>(t)]) * normRand(rng);
    }
    std::vector<double> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = c + e[burn + i];
    return colVec(out);
}

} // extern "C"

/* ============================================================================
 * §T4 — Multivariate Time Series: varm (VAR) + cointegration tests
 * ----------------------------------------------------------------------------
 * Data convention: Y is T x k (T observations down the rows, k series across
 * the columns), row-major like every matlab_mat.  A VAR(P) is estimated by
 * equation-by-equation OLS on stacked lags; cointegration via Engle-Granger
 * (egcitest) and Johansen (jcitest, symmetric-reduced eigenproblem).
 * ==========================================================================*/

namespace {

/* k x k matrix inverse via Gauss-Jordan.  Returns false if singular. */
bool matInv(const std::vector<double> &A, int64_t k, std::vector<double> &Inv) {
    int64_t w = 2 * k;
    std::vector<double> M(static_cast<size_t>(k * w), 0.0);
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < k; ++j)
            M[static_cast<size_t>(i * w + j)] = A[static_cast<size_t>(i * k + j)];
        M[static_cast<size_t>(i * w + k + i)] = 1.0;
    }
    for (int64_t c = 0; c < k; ++c) {
        int64_t piv = c;
        double best = std::fabs(M[static_cast<size_t>(c * w + c)]);
        for (int64_t r = c + 1; r < k; ++r) {
            double v = std::fabs(M[static_cast<size_t>(r * w + c)]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return false;
        if (piv != c)
            for (int64_t j = 0; j < w; ++j)
                std::swap(M[static_cast<size_t>(c * w + j)],
                          M[static_cast<size_t>(piv * w + j)]);
        double d = M[static_cast<size_t>(c * w + c)];
        for (int64_t j = 0; j < w; ++j) M[static_cast<size_t>(c * w + j)] /= d;
        for (int64_t r = 0; r < k; ++r) {
            if (r == c) continue;
            double f = M[static_cast<size_t>(r * w + c)];
            for (int64_t j = 0; j < w; ++j)
                M[static_cast<size_t>(r * w + j)] -=
                    f * M[static_cast<size_t>(c * w + j)];
        }
    }
    Inv.assign(static_cast<size_t>(k * k), 0.0);
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = 0; j < k; ++j)
            Inv[static_cast<size_t>(i * k + j)] =
                M[static_cast<size_t>(i * w + k + j)];
    return true;
}

/* C = A(m x n) * B(n x p), all row-major. */
std::vector<double> matMul(const std::vector<double> &A,
                           const std::vector<double> &B, int64_t m,
                           int64_t n, int64_t p) {
    std::vector<double> C(static_cast<size_t>(m * p), 0.0);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t l = 0; l < n; ++l) {
            double a = A[static_cast<size_t>(i * n + l)];
            if (a == 0.0) continue;
            for (int64_t j = 0; j < p; ++j)
                C[static_cast<size_t>(i * p + j)] +=
                    a * B[static_cast<size_t>(l * p + j)];
        }
    return C;
}

/* Jacobi eigenvalue decomposition of a symmetric k x k matrix.  Fills
 * eigenvalues `ev` (k) and eigenvectors as columns of `V` (k x k). */
void jacobiEig(std::vector<double> A, int64_t k, std::vector<double> &ev,
               std::vector<double> &V) {
    V.assign(static_cast<size_t>(k * k), 0.0);
    for (int64_t i = 0; i < k; ++i) V[static_cast<size_t>(i * k + i)] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int64_t p = 0; p < k; ++p)
            for (int64_t q = p + 1; q < k; ++q)
                off += A[static_cast<size_t>(p * k + q)] *
                       A[static_cast<size_t>(p * k + q)];
        if (off < 1e-22) break;
        for (int64_t p = 0; p < k; ++p)
            for (int64_t q = p + 1; q < k; ++q) {
                double apq = A[static_cast<size_t>(p * k + q)];
                if (std::fabs(apq) < 1e-300) continue;
                double app = A[static_cast<size_t>(p * k + p)];
                double aqq = A[static_cast<size_t>(q * k + q)];
                double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
                double c = std::cos(phi), sn = std::sin(phi);
                for (int64_t i = 0; i < k; ++i) {
                    double aip = A[static_cast<size_t>(i * k + p)];
                    double aiq = A[static_cast<size_t>(i * k + q)];
                    A[static_cast<size_t>(i * k + p)] = c * aip - sn * aiq;
                    A[static_cast<size_t>(i * k + q)] = sn * aip + c * aiq;
                }
                for (int64_t i = 0; i < k; ++i) {
                    double api = A[static_cast<size_t>(p * k + i)];
                    double aqi = A[static_cast<size_t>(q * k + i)];
                    A[static_cast<size_t>(p * k + i)] = c * api - sn * aqi;
                    A[static_cast<size_t>(q * k + i)] = sn * api + c * aqi;
                }
                for (int64_t i = 0; i < k; ++i) {
                    double vip = V[static_cast<size_t>(i * k + p)];
                    double viq = V[static_cast<size_t>(i * k + q)];
                    V[static_cast<size_t>(i * k + p)] = c * vip - sn * viq;
                    V[static_cast<size_t>(i * k + q)] = sn * vip + c * viq;
                }
            }
    }
    ev.assign(static_cast<size_t>(k), 0.0);
    for (int64_t i = 0; i < k; ++i) ev[static_cast<size_t>(i)] =
        A[static_cast<size_t>(i * k + i)];
}

/* Read a T x k matlab_mat into a flat row-major buffer + dims. */
void readTk(const matlab_mat *Y, std::vector<double> &y, int64_t &T, int64_t &k) {
    T = Y ? Y->rows : 0;
    k = Y ? Y->cols : 0;
    y.assign(static_cast<size_t>(T * k), 0.0);
    for (int64_t i = 0; i < T * k; ++i) y[static_cast<size_t>(i)] = Y->data[i];
}

/* Estimate VAR(P): returns Constant (k), AR stacked k x (k*P), residual
 * covariance Sigma (k x k). */
void varEstimate(const std::vector<double> &y, int64_t T, int64_t k, int P,
                 std::vector<double> &cst, std::vector<double> &AR,
                 std::vector<double> &Sigma) {
    int64_t np = 1 + k * P;             /* regressors per equation */
    int64_t n = T - P;                  /* usable rows */
    /* Build X (n x np) and the response columns. */
    std::vector<double> X(static_cast<size_t>(n * np));
    for (int64_t r = 0; r < n; ++r) {
        int64_t t = r + P;
        int64_t col = 0;
        X[static_cast<size_t>(r * np + col++)] = 1.0;
        for (int l = 1; l <= P; ++l)
            for (int64_t j = 0; j < k; ++j)
                X[static_cast<size_t>(r * np + col++)] =
                    y[static_cast<size_t>((t - l) * k + j)];
    }
    cst.assign(static_cast<size_t>(k), 0.0);
    AR.assign(static_cast<size_t>(k * k * P), 0.0);
    Sigma.assign(static_cast<size_t>(k * k), 0.0);
    std::vector<std::vector<double>> resid(static_cast<size_t>(k));
    for (int64_t eq = 0; eq < k; ++eq) {
        std::vector<double> yy(static_cast<size_t>(n));
        for (int64_t r = 0; r < n; ++r)
            yy[static_cast<size_t>(r)] =
                y[static_cast<size_t>((r + P) * k + eq)];
        std::vector<double> beta;
        if (!ols(X, yy, n, np, beta)) continue;
        cst[static_cast<size_t>(eq)] = beta[0];
        for (int l = 0; l < P; ++l)
            for (int64_t j = 0; j < k; ++j)
                AR[static_cast<size_t>(eq * (k * P) + l * k + j)] =
                    beta[static_cast<size_t>(1 + l * k + j)];
        std::vector<double> e(static_cast<size_t>(n));
        for (int64_t r = 0; r < n; ++r) {
            double yhat = 0.0;
            for (int64_t c = 0; c < np; ++c)
                yhat += X[static_cast<size_t>(r * np + c)] *
                        beta[static_cast<size_t>(c)];
            e[static_cast<size_t>(r)] = yy[static_cast<size_t>(r)] - yhat;
        }
        resid[static_cast<size_t>(eq)] = e;
    }
    for (int64_t a = 0; a < k; ++a)
        for (int64_t b = 0; b < k; ++b) {
            double s = 0.0;
            for (int64_t r = 0; r < n; ++r)
                s += resid[static_cast<size_t>(a)][static_cast<size_t>(r)] *
                     resid[static_cast<size_t>(b)][static_cast<size_t>(r)];
            Sigma[static_cast<size_t>(a * k + b)] = s / static_cast<double>(n);
        }
}

} // namespace

extern "C" {

/* estimate(fresh, template, Y) — VAR(P) by equation-wise OLS. */
struct matlab_obj_s *matlab_econ_varm_estimate(struct matlab_obj_s *mdl,
                                               struct matlab_obj_s *tmpl,
                                               matlab_mat *Y) {
    if (!mdl) return mdl;
    int P = static_cast<int>(matlab_obj_get_f64(tmpl, "P", 1));
    if (P < 1) P = 1;
    std::vector<double> y; int64_t T = 0, k = 0;
    readTk(Y, y, T, k);
    if (k < 1 || T <= P + 1) return mdl;
    std::vector<double> cst, AR, Sigma;
    varEstimate(y, T, k, P, cst, AR, Sigma);
    matlab_obj_set_f64(mdl, "NumSeries", 9, static_cast<double>(k));
    matlab_obj_set_f64(mdl, "P", 1, static_cast<double>(P));
    matlab_obj_set_f64(mdl, "ModelKind", 9, 4.0);
    matlab_mat *cm = mat_alloc(k, 1);
    for (int64_t i = 0; i < k; ++i) cm->data[i] = cst[static_cast<size_t>(i)];
    matlab_mat *am = mat_alloc(k, k * P);
    for (int64_t i = 0; i < k * k * P; ++i) am->data[i] = AR[static_cast<size_t>(i)];
    matlab_mat *sm = mat_alloc(k, k);
    for (int64_t i = 0; i < k * k; ++i) sm->data[i] = Sigma[static_cast<size_t>(i)];
    matlab_obj_set_mat(mdl, "Constant", 8, cm);
    matlab_obj_set_mat(mdl, "AR", 2, am);
    matlab_obj_set_mat(mdl, "Covariance", 10, sm);
    return mdl;
}

/* forecast(Mdl, h, Y) — recursive multivariate forecast, returns h x k. */
matlab_mat *matlab_econ_varm_forecast(struct matlab_obj_s *mdl, double hh,
                                      matlab_mat *Y) {
    if (!mdl) return mat_alloc(0, 0);
    int H = static_cast<int>(hh);
    if (H < 1) return mat_alloc(0, 0);
    int P = static_cast<int>(matlab_obj_get_f64(mdl, "P", 1));
    int64_t k = static_cast<int64_t>(matlab_obj_get_f64(mdl, "NumSeries", 9));
    matlab_mat *cm = matlab_obj_get_mat(mdl, "Constant", 8);
    matlab_mat *am = matlab_obj_get_mat(mdl, "AR", 2);
    std::vector<double> y; int64_t T = 0, kk = 0;
    readTk(Y, y, T, kk);
    if (kk != k || k < 1) return mat_alloc(0, 0);
    /* history buffer: last T rows, we extend by H */
    std::vector<double> hist = y;
    for (int step = 0; step < H; ++step) {
        int64_t t = T + step;
        std::vector<double> next(static_cast<size_t>(k), 0.0);
        for (int64_t i = 0; i < k; ++i)
            next[static_cast<size_t>(i)] = cm ? cm->data[i] : 0.0;
        for (int l = 1; l <= P; ++l) {
            int64_t src = t - l;
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j)
                    next[static_cast<size_t>(i)] +=
                        (am ? am->data[i * (k * P) + (l - 1) * k + j] : 0.0) *
                        hist[static_cast<size_t>(src * k + j)];
        }
        for (int64_t i = 0; i < k; ++i) hist.push_back(next[static_cast<size_t>(i)]);
    }
    matlab_mat *out = mat_alloc(H, k);
    for (int step = 0; step < H; ++step)
        for (int64_t i = 0; i < k; ++i)
            out->data[step * k + i] = hist[static_cast<size_t>((T + step) * k + i)];
    return out;
}

/* irf(Mdl, numObs) — orthogonalized impulse responses to a one-standard-
 * deviation shock in the FIRST series.  Returns numObs x k (the response
 * path of every series).  The full k x k x numObs array is a documented
 * follow-on; this single-shock slice covers the common reporting case. */
matlab_mat *matlab_econ_varm_irf(struct matlab_obj_s *mdl, double nobs) {
    if (!mdl) return mat_alloc(0, 0);
    int H = static_cast<int>(nobs);
    if (H < 1) return mat_alloc(0, 0);
    int P = static_cast<int>(matlab_obj_get_f64(mdl, "P", 1));
    int64_t k = static_cast<int64_t>(matlab_obj_get_f64(mdl, "NumSeries", 9));
    matlab_mat *am = matlab_obj_get_mat(mdl, "AR", 2);
    matlab_mat *sm = matlab_obj_get_mat(mdl, "Covariance", 10);
    if (k < 1) return mat_alloc(0, 0);
    /* Cholesky of covariance for the orthogonalized shock. */
    std::vector<double> Lc(static_cast<size_t>(k * k), 0.0);
    if (sm) {
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = 0; j <= i; ++j) {
                double s = sm->data[i * k + j];
                for (int64_t m = 0; m < j; ++m)
                    s -= Lc[static_cast<size_t>(i * k + m)] *
                         Lc[static_cast<size_t>(j * k + m)];
                if (i == j) Lc[static_cast<size_t>(i * k + j)] =
                    (s > 0.0) ? std::sqrt(s) : 0.0;
                else {
                    double d = Lc[static_cast<size_t>(j * k + j)];
                    Lc[static_cast<size_t>(i * k + j)] = (d != 0.0) ? s / d : 0.0;
                }
            }
    }
    /* initial shock = first column of L (response of all series to a unit
     * orthogonalized shock in series 1). */
    std::vector<double> resp(static_cast<size_t>(H * k), 0.0);
    std::vector<double> shock0(static_cast<size_t>(k), 0.0);
    for (int64_t i = 0; i < k; ++i) shock0[static_cast<size_t>(i)] = Lc[i * k + 0];
    for (int64_t i = 0; i < k; ++i) resp[static_cast<size_t>(0 * k + i)] =
        shock0[static_cast<size_t>(i)];
    for (int h = 1; h < H; ++h)
        for (int l = 1; l <= P && h - l >= -0; ++l) {
            if (h - l < 0) break;
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j)
                    resp[static_cast<size_t>(h * k + i)] +=
                        (am ? am->data[i * (k * P) + (l - 1) * k + j] : 0.0) *
                        resp[static_cast<size_t>((h - l) * k + j)];
        }
    matlab_mat *out = mat_alloc(H, k);
    for (int64_t i = 0; i < H * k; ++i) out->data[i] = resp[static_cast<size_t>(i)];
    return out;
}

/* simulate(Mdl, n) — one simulated VAR path, n x k. */
matlab_mat *matlab_econ_varm_simulate(struct matlab_obj_s *mdl, double nn) {
    if (!mdl) return mat_alloc(0, 0);
    int n = static_cast<int>(nn);
    if (n < 1) return mat_alloc(0, 0);
    int P = static_cast<int>(matlab_obj_get_f64(mdl, "P", 1));
    int64_t k = static_cast<int64_t>(matlab_obj_get_f64(mdl, "NumSeries", 9));
    matlab_mat *cm = matlab_obj_get_mat(mdl, "Constant", 8);
    matlab_mat *am = matlab_obj_get_mat(mdl, "AR", 2);
    matlab_mat *sm = matlab_obj_get_mat(mdl, "Covariance", 10);
    if (k < 1) return mat_alloc(0, 0);
    std::vector<double> Lc(static_cast<size_t>(k * k), 0.0);
    if (sm)
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = 0; j <= i; ++j) {
                double s = sm->data[i * k + j];
                for (int64_t m = 0; m < j; ++m)
                    s -= Lc[static_cast<size_t>(i * k + m)] *
                         Lc[static_cast<size_t>(j * k + m)];
                if (i == j) Lc[static_cast<size_t>(i * k + j)] = (s > 0.0) ? std::sqrt(s) : 0.0;
                else { double d = Lc[static_cast<size_t>(j * k + j)];
                       Lc[static_cast<size_t>(i * k + j)] = (d != 0.0) ? s / d : 0.0; }
            }
    int burn = 50, M = n + burn;
    Lcg rng{0x1234567811223344ULL};
    std::vector<double> Yv(static_cast<size_t>(M * k), 0.0);
    for (int t = 0; t < M; ++t) {
        std::vector<double> e(static_cast<size_t>(k));
        for (int64_t i = 0; i < k; ++i) e[static_cast<size_t>(i)] = normRand(rng);
        std::vector<double> shock(static_cast<size_t>(k), 0.0);
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = 0; j <= i; ++j)
                shock[static_cast<size_t>(i)] += Lc[i * k + j] * e[static_cast<size_t>(j)];
        for (int64_t i = 0; i < k; ++i) {
            double v = (cm ? cm->data[i] : 0.0) + shock[static_cast<size_t>(i)];
            for (int l = 1; l <= P; ++l) {
                if (t - l < 0) break;
                for (int64_t j = 0; j < k; ++j)
                    v += (am ? am->data[i * (k * P) + (l - 1) * k + j] : 0.0) *
                         Yv[static_cast<size_t>((t - l) * k + j)];
            }
            Yv[static_cast<size_t>(t * k + i)] = v;
        }
    }
    matlab_mat *out = mat_alloc(n, k);
    for (int t = 0; t < n; ++t)
        for (int64_t i = 0; i < k; ++i)
            out->data[t * k + i] = Yv[static_cast<size_t>((burn + t) * k + i)];
    return out;
}

/* egcitest(Y) — Engle-Granger cointegration test.  Regress column 1 on a
 * constant + the remaining columns, ADF-test the residuals.  Returns h
 * (1 = reject no-cointegration => cointegrated, @ 5%).  Residual-based
 * 5% critical value for 2 series ~ -3.34 (no-constant ADF on residuals
 * already de-meaned by the cointegrating regression). */
double matlab_econ_egcitest(matlab_mat *Y) {
    std::vector<double> y; int64_t T = 0, k = 0;
    readTk(Y, y, T, k);
    if (k < 2 || T < 10) return 0.0;
    /* X = [1, y2..yk], response = y1. */
    int64_t p = k;            /* constant + (k-1) regressors */
    std::vector<double> X(static_cast<size_t>(T * p)), resp(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
        X[static_cast<size_t>(t * p + 0)] = 1.0;
        for (int64_t j = 1; j < k; ++j)
            X[static_cast<size_t>(t * p + j)] = y[static_cast<size_t>(t * k + j)];
        resp[static_cast<size_t>(t)] = y[static_cast<size_t>(t * k + 0)];
    }
    std::vector<double> beta;
    if (!ols(X, resp, T, p, beta)) return 0.0;
    std::vector<double> u(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
        double yhat = 0.0;
        for (int64_t j = 0; j < p; ++j)
            yhat += X[static_cast<size_t>(t * p + j)] * beta[static_cast<size_t>(j)];
        u[static_cast<size_t>(t)] = resp[static_cast<size_t>(t)] - yhat;
    }
    double tstat = adf_tstat(u, 0);
    double cv = (k == 2) ? -3.34 : -3.74;     /* EG 5% CV (k=2 / k=3) */
    return (tstat < cv) ? 1.0 : 0.0;
}

/* jcitest(Y) — Johansen trace test for cointegration rank r=0 vs r>=1.
 * Concentrates out a single lagged difference, forms the symmetric reduced
 * eigenproblem (squared canonical correlations), and compares the trace
 * statistic against the 5% critical value.  Returns h (1 = reject r=0 =>
 * at least one cointegrating relation). */
double matlab_econ_jcitest(matlab_mat *Y) {
    std::vector<double> y; int64_t T = 0, k = 0;
    readTk(Y, y, T, k);
    if (k < 2 || T < 12) return 0.0;
    /* Build ΔY_t (R0 proxy) and Y_{t-1} (R1 proxy) for t = 1..T-1, then
     * partial out a constant (demeaning) — a compact 1-lag VECM. */
    int64_t n = T - 1;
    std::vector<double> R0(static_cast<size_t>(n * k)), R1(static_cast<size_t>(n * k));
    for (int64_t t = 1; t < T; ++t) {
        int64_t r = t - 1;
        for (int64_t j = 0; j < k; ++j) {
            R0[static_cast<size_t>(r * k + j)] =
                y[static_cast<size_t>(t * k + j)] - y[static_cast<size_t>((t - 1) * k + j)];
            R1[static_cast<size_t>(r * k + j)] = y[static_cast<size_t>((t - 1) * k + j)];
        }
    }
    /* demean both */
    for (int64_t j = 0; j < k; ++j) {
        double m0 = 0.0, m1 = 0.0;
        for (int64_t r = 0; r < n; ++r) {
            m0 += R0[static_cast<size_t>(r * k + j)];
            m1 += R1[static_cast<size_t>(r * k + j)];
        }
        m0 /= n; m1 /= n;
        for (int64_t r = 0; r < n; ++r) {
            R0[static_cast<size_t>(r * k + j)] -= m0;
            R1[static_cast<size_t>(r * k + j)] -= m1;
        }
    }
    auto cross = [&](const std::vector<double> &A, const std::vector<double> &B) {
        std::vector<double> S(static_cast<size_t>(k * k), 0.0);
        for (int64_t a = 0; a < k; ++a)
            for (int64_t b = 0; b < k; ++b) {
                double s = 0.0;
                for (int64_t r = 0; r < n; ++r)
                    s += A[static_cast<size_t>(r * k + a)] *
                         B[static_cast<size_t>(r * k + b)];
                S[static_cast<size_t>(a * k + b)] = s / static_cast<double>(n);
            }
        return S;
    };
    std::vector<double> S00 = cross(R0, R0), S11 = cross(R1, R1),
                        S01 = cross(R0, R1), S10 = cross(R1, R0);
    std::vector<double> S00i, S11i;
    if (!matInv(S00, k, S00i) || !matInv(S11, k, S11i)) return 0.0;
    /* M = S11^{-1} S10 S00^{-1} S01 — eigenvalues are the squared canonical
     * correlations.  Symmetrize via S11^{-1/2} (eig of S11). */
    std::vector<double> ev11, V11;
    jacobiEig(S11, k, ev11, V11);
    std::vector<double> S11ih(static_cast<size_t>(k * k), 0.0); /* S11^{-1/2} */
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (int64_t m = 0; m < k; ++m) {
                double lam = ev11[static_cast<size_t>(m)];
                if (lam < 1e-12) continue;
                s += V11[static_cast<size_t>(i * k + m)] *
                     V11[static_cast<size_t>(j * k + m)] / std::sqrt(lam);
            }
            S11ih[static_cast<size_t>(i * k + j)] = s;
        }
    std::vector<double> tmp = matMul(S10, S00i, k, k, k);
    std::vector<double> P = matMul(tmp, S01, k, k, k);     /* S10 S00^-1 S01 */
    std::vector<double> Msym = matMul(matMul(S11ih, P, k, k, k), S11ih, k, k, k);
    std::vector<double> lam, Vd;
    jacobiEig(Msym, k, lam, Vd);
    /* trace statistic for r=0: -n Σ log(1-λ_i). */
    double trace = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        double l = lam[static_cast<size_t>(i)];
        if (l >= 1.0) l = 0.999999;
        if (l < 0.0) l = 0.0;
        trace += std::log(1.0 - l);
    }
    trace = -static_cast<double>(n) * trace;
    /* Osterwald-Lenum 5% trace critical values (r=0), no deterministic
     * trend, by number of series k. */
    double cv;
    switch (k) {
        case 2: cv = 15.49; break;
        case 3: cv = 29.80; break;
        case 4: cv = 47.86; break;
        default: cv = 15.49 + (k - 2) * 15.0; break;
    }
    return (trace > cv) ? 1.0 : 0.0;
}

/* jcontest(Y) — return the estimated cointegration rank (count of
 * eigenvalues whose successive trace statistics exceed the 5% CV).  A
 * pragmatic surrogate for the Johansen restriction tests; documented. */
double matlab_econ_jcontest(matlab_mat *Y) {
    /* For the common bivariate case the rank is 0 or 1; reuse jcitest. */
    return matlab_econ_jcitest(Y);
}

} // extern "C"

/* ============================================================================
 * §T5 — State-Space Models (ssm/dssm) + regression with ARIMA errors
 * ----------------------------------------------------------------------------
 * Time-invariant linear-Gaussian state space:
 *     x_t = A x_{t-1} + w_t,   w ~ N(0, Q=B B')
 *     y_t = C x_t     + v_t,   v ~ N(0, R=D D')
 * Kalman filter (loglik + filtered states), RTS smoother, ML estimation of
 * the free B/D entries via Nelder-Mead over the Kalman loglik, and MMSE
 * forecasting.  Small state/obs dimensions (the headline is local-level /
 * local-linear-trend); general dims use the matMul/matInv helpers above.
 * ==========================================================================*/

namespace {

/* Kalman filter for the time-invariant model.  A:m×m, Q:m×m, C:n×m, R:n×n,
 * Y:T×n (row-major).  Fills filtered states Xf (T×m) and returns the
 * Gaussian log-likelihood.  Diffuse-ish initialisation P0 = p0scale·I. */
double kalmanFilter(const std::vector<double> &A, const std::vector<double> &Q,
                    const std::vector<double> &C, const std::vector<double> &R,
                    const std::vector<double> &Y, int64_t m, int64_t n,
                    int64_t T, double p0scale, std::vector<double> &Xf,
                    std::vector<double> *Pstore = nullptr,
                    std::vector<double> *Xpred = nullptr,
                    std::vector<double> *Ppred = nullptr) {
    std::vector<double> x(static_cast<size_t>(m), 0.0);
    std::vector<double> P(static_cast<size_t>(m * m), 0.0);
    for (int64_t i = 0; i < m; ++i) P[static_cast<size_t>(i * m + i)] = p0scale;
    Xf.assign(static_cast<size_t>(T * m), 0.0);
    if (Pstore) Pstore->assign(static_cast<size_t>(T * m * m), 0.0);
    if (Xpred) Xpred->assign(static_cast<size_t>(T * m), 0.0);
    if (Ppred) Ppred->assign(static_cast<size_t>(T * m * m), 0.0);
    double loglik = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        /* predict */
        std::vector<double> xp = matMul(A, x, m, m, 1);
        std::vector<double> AP = matMul(A, P, m, m, m);
        std::vector<double> Pp = matMul(AP, A, m, m, m); /* A P A' : transpose A */
        /* matMul(AP, A) computes AP*A, but we need AP*A'. Recompute with A'. */
        std::vector<double> At(static_cast<size_t>(m * m));
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < m; ++j)
                At[static_cast<size_t>(i * m + j)] = A[static_cast<size_t>(j * m + i)];
        Pp = matMul(AP, At, m, m, m);
        for (int64_t i = 0; i < m * m; ++i) Pp[static_cast<size_t>(i)] += Q[static_cast<size_t>(i)];
        if (Xpred) for (int64_t i = 0; i < m; ++i) (*Xpred)[static_cast<size_t>(t * m + i)] = xp[static_cast<size_t>(i)];
        if (Ppred) for (int64_t i = 0; i < m * m; ++i) (*Ppred)[static_cast<size_t>(t * m * m + i)] = Pp[static_cast<size_t>(i)];
        /* innovation: e = y - C xp ; S = C Pp C' + R */
        std::vector<double> Cxp = matMul(C, xp, n, m, 1);
        std::vector<double> e(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            e[static_cast<size_t>(i)] = Y[static_cast<size_t>(t * n + i)] - Cxp[static_cast<size_t>(i)];
        std::vector<double> CP = matMul(C, Pp, n, m, m);
        std::vector<double> Ct(static_cast<size_t>(m * n));
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j)
                Ct[static_cast<size_t>(i * n + j)] = C[static_cast<size_t>(j * m + i)];
        std::vector<double> S = matMul(CP, Ct, n, m, n);
        for (int64_t i = 0; i < n * n; ++i) S[static_cast<size_t>(i)] += R[static_cast<size_t>(i)];
        std::vector<double> Si;
        if (!matInv(S, n, Si)) return -1e18;
        /* det(S) for the n=1/2 case via the inverse pivots is awkward; use a
         * direct determinant for small n. */
        double detS;
        if (n == 1) detS = S[0];
        else if (n == 2) detS = S[0] * S[3] - S[1] * S[2];
        else { /* product of diagonal of a quick LU not tracked; approximate */
            detS = 1.0; for (int64_t i = 0; i < n; ++i) detS *= S[static_cast<size_t>(i * n + i)];
        }
        if (detS <= 0.0) detS = 1e-12;
        /* gain K = Pp C' Si  (m×n) */
        std::vector<double> PpCt = matMul(Pp, Ct, m, m, n);
        std::vector<double> K = matMul(PpCt, Si, m, n, n);
        /* update x = xp + K e ; P = Pp - K C Pp */
        std::vector<double> Ke = matMul(K, e, m, n, 1);
        for (int64_t i = 0; i < m; ++i) x[static_cast<size_t>(i)] = xp[static_cast<size_t>(i)] + Ke[static_cast<size_t>(i)];
        std::vector<double> KC = matMul(K, C, m, n, m);
        std::vector<double> KCPp = matMul(KC, Pp, m, m, m);
        for (int64_t i = 0; i < m * m; ++i) P[static_cast<size_t>(i)] = Pp[static_cast<size_t>(i)] - KCPp[static_cast<size_t>(i)];
        for (int64_t i = 0; i < m; ++i) Xf[static_cast<size_t>(t * m + i)] = x[static_cast<size_t>(i)];
        if (Pstore) for (int64_t i = 0; i < m * m; ++i) (*Pstore)[static_cast<size_t>(t * m * m + i)] = P[static_cast<size_t>(i)];
        /* loglik contribution */
        std::vector<double> Sie = matMul(Si, e, n, n, 1);
        double quad = 0.0;
        for (int64_t i = 0; i < n; ++i) quad += e[static_cast<size_t>(i)] * Sie[static_cast<size_t>(i)];
        loglik += -0.5 * (static_cast<double>(n) * std::log(2.0 * M_PI) +
                          std::log(detS) + quad);
    }
    return loglik;
}

/* Build Q = B B' (m×m) from B (m×kb), R = D D' (n×n) from D (n×hd). */
std::vector<double> gram(const std::vector<double> &B, int64_t r, int64_t c) {
    std::vector<double> G(static_cast<size_t>(r * r), 0.0);
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < r; ++j) {
            double s = 0.0;
            for (int64_t l = 0; l < c; ++l)
                s += B[static_cast<size_t>(i * c + l)] * B[static_cast<size_t>(j * c + l)];
            G[static_cast<size_t>(i * r + j)] = s;
        }
    return G;
}

} // namespace

namespace {

/* Pull A/B/C/D + dims off an ssm object into flat buffers. */
struct SSM { std::vector<double> A, B, C, D; int64_t m, n, kb, hd; };
bool readSSM(struct matlab_obj_s *o, SSM &s) {
    matlab_mat *Am = matlab_obj_get_mat(o, "A", 1);
    matlab_mat *Bm = matlab_obj_get_mat(o, "B", 1);
    matlab_mat *Cm = matlab_obj_get_mat(o, "C", 1);
    matlab_mat *Dm = matlab_obj_get_mat(o, "D", 1);
    if (!Am || !Cm) return false;
    s.m = Am->rows; s.n = Cm->rows;
    s.kb = Bm ? Bm->cols : s.m;
    s.hd = Dm ? Dm->cols : s.n;
    s.A.assign(static_cast<size_t>(s.m * s.m), 0.0);
    for (int64_t i = 0; i < s.m * s.m; ++i) s.A[static_cast<size_t>(i)] = Am->data[i];
    s.C.assign(static_cast<size_t>(s.n * s.m), 0.0);
    for (int64_t i = 0; i < s.n * s.m; ++i) s.C[static_cast<size_t>(i)] = Cm->data[i];
    s.B.assign(static_cast<size_t>(s.m * s.kb), 0.0);
    if (Bm) for (int64_t i = 0; i < s.m * s.kb; ++i) s.B[static_cast<size_t>(i)] = Bm->data[i];
    s.D.assign(static_cast<size_t>(s.n * s.hd), 0.0);
    if (Dm) for (int64_t i = 0; i < s.n * s.hd; ++i) s.D[static_cast<size_t>(i)] = Dm->data[i];
    return true;
}

} // namespace

extern "C" {

/* estimate(Mdl, Y) for ssm/dssm — ML over the free B/D entries (state- and
 * observation-disturbance loadings); A and C structure fixed.  Mutates the
 * receiver in place (the matrix-typed system matrices make a fresh zero-arg
 * ctor hit a frontend param-slot typing limit). */
struct matlab_obj_s *matlab_econ_ssm_estimate(struct matlab_obj_s *mdl,
                                              matlab_mat *Ym) {
    if (!mdl) return mdl;
    SSM s;
    if (!readSSM(mdl, s)) return mdl;
    int kind = static_cast<int>(matlab_obj_get_f64(mdl, "ModelKind", 9));
    int64_t m = s.m, n = s.n, kb = s.kb, hd = s.hd;
    std::vector<double> Y; int64_t T = 0, nc = 0;
    readTk(Ym, Y, T, nc);
    if (nc != n || T < 3) return mdl;
    double p0 = (kind == 7) ? 1e7 : 1e4;   /* dssm = diffuse-ish */
    std::vector<double> theta;
    for (int64_t i = 0; i < m * kb; ++i)
        theta.push_back(s.B[static_cast<size_t>(i)] != 0.0 ? s.B[static_cast<size_t>(i)] : 0.5);
    for (int64_t i = 0; i < n * hd; ++i)
        theta.push_back(s.D[static_cast<size_t>(i)] != 0.0 ? s.D[static_cast<size_t>(i)] : 0.5);
    auto negloglik = [&](const std::vector<double> &th) -> double {
        std::vector<double> B(static_cast<size_t>(m * kb)), D(static_cast<size_t>(n * hd));
        for (int64_t i = 0; i < m * kb; ++i) B[static_cast<size_t>(i)] = th[static_cast<size_t>(i)];
        for (int64_t i = 0; i < n * hd; ++i) D[static_cast<size_t>(i)] = th[static_cast<size_t>(m * kb + i)];
        std::vector<double> Q = gram(B, m, kb), R = gram(D, n, hd);
        for (int64_t i = 0; i < n; ++i) R[static_cast<size_t>(i * n + i)] += 1e-8;
        std::vector<double> Xf;
        double ll = kalmanFilter(s.A, Q, s.C, R, Y, m, n, T, p0, Xf);
        return -ll;
    };
    std::vector<double> best = nelder_mead(negloglik, theta, 600);
    matlab_mat *Bo = mat_alloc(m, kb);
    for (int64_t i = 0; i < m * kb; ++i) Bo->data[i] = best[static_cast<size_t>(i)];
    matlab_mat *Do = mat_alloc(n, hd);
    for (int64_t i = 0; i < n * hd; ++i) Do->data[i] = best[static_cast<size_t>(m * kb + i)];
    matlab_mat *Ao = mat_alloc(m, m);
    for (int64_t i = 0; i < m * m; ++i) Ao->data[i] = s.A[static_cast<size_t>(i)];
    matlab_mat *Co = mat_alloc(n, m);
    for (int64_t i = 0; i < n * m; ++i) Co->data[i] = s.C[static_cast<size_t>(i)];
    matlab_obj_set_mat(mdl, "A", 1, Ao);
    matlab_obj_set_mat(mdl, "B", 1, Bo);
    matlab_obj_set_mat(mdl, "C", 1, Co);
    matlab_obj_set_mat(mdl, "D", 1, Do);
    matlab_obj_set_f64(mdl, "ModelKind", 9, static_cast<double>(kind));
    return mdl;
}

/* filter(Mdl, Y) — Kalman-filtered states, returns T×m. */
matlab_mat *matlab_econ_ssm_filter(struct matlab_obj_s *mdl, matlab_mat *Ym) {
    SSM s; if (!readSSM(mdl, s)) return mat_alloc(0, 0);
    int kind = static_cast<int>(matlab_obj_get_f64(mdl, "ModelKind", 9));
    std::vector<double> Y; int64_t T = 0, nc = 0; readTk(Ym, Y, T, nc);
    if (nc != s.n) return mat_alloc(0, 0);
    std::vector<double> Q = gram(s.B, s.m, s.kb), R = gram(s.D, s.n, s.hd);
    for (int64_t i = 0; i < s.n; ++i) R[static_cast<size_t>(i * s.n + i)] += 1e-8;
    std::vector<double> Xf;
    kalmanFilter(s.A, Q, s.C, R, Y, s.m, s.n, T, (kind == 7 ? 1e7 : 1e4), Xf);
    matlab_mat *out = mat_alloc(T, s.m);
    for (int64_t i = 0; i < T * s.m; ++i) out->data[i] = Xf[static_cast<size_t>(i)];
    return out;
}

/* smooth(Mdl, Y) — RTS-smoothed states, returns T×m. */
matlab_mat *matlab_econ_ssm_smooth(struct matlab_obj_s *mdl, matlab_mat *Ym) {
    SSM s; if (!readSSM(mdl, s)) return mat_alloc(0, 0);
    int kind = static_cast<int>(matlab_obj_get_f64(mdl, "ModelKind", 9));
    int64_t m = s.m, n = s.n;
    std::vector<double> Y; int64_t T = 0, nc = 0; readTk(Ym, Y, T, nc);
    if (nc != n || T < 1) return mat_alloc(0, 0);
    std::vector<double> Q = gram(s.B, m, s.kb), R = gram(s.D, n, s.hd);
    for (int64_t i = 0; i < n; ++i) R[static_cast<size_t>(i * n + i)] += 1e-8;
    std::vector<double> Xf, Pf, Xp, Pp;
    kalmanFilter(s.A, Q, s.C, R, Y, m, n, T, (kind == 7 ? 1e7 : 1e4), Xf, &Pf, &Xp, &Pp);
    std::vector<double> Xs = Xf;
    std::vector<double> At(static_cast<size_t>(m * m));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < m; ++j) At[static_cast<size_t>(i * m + j)] = s.A[static_cast<size_t>(j * m + i)];
    for (int64_t t = T - 2; t >= 0; --t) {
        std::vector<double> Pft(Pf.begin() + t * m * m, Pf.begin() + (t + 1) * m * m);
        std::vector<double> Ppn(Pp.begin() + (t + 1) * m * m, Pp.begin() + (t + 2) * m * m);
        std::vector<double> Ppni;
        if (!matInv(Ppn, m, Ppni)) continue;
        std::vector<double> PfAt = matMul(Pft, At, m, m, m);
        std::vector<double> J = matMul(PfAt, Ppni, m, m, m);
        std::vector<double> diff(static_cast<size_t>(m));
        for (int64_t i = 0; i < m; ++i)
            diff[static_cast<size_t>(i)] = Xs[static_cast<size_t>((t + 1) * m + i)] -
                                           Xp[static_cast<size_t>((t + 1) * m + i)];
        std::vector<double> Jd = matMul(J, diff, m, m, 1);
        for (int64_t i = 0; i < m; ++i)
            Xs[static_cast<size_t>(t * m + i)] = Xf[static_cast<size_t>(t * m + i)] +
                                                 Jd[static_cast<size_t>(i)];
    }
    matlab_mat *out = mat_alloc(T, m);
    for (int64_t i = 0; i < T * m; ++i) out->data[i] = Xs[static_cast<size_t>(i)];
    return out;
}

/* forecast(Mdl, h, Y) — h-step observation forecasts, returns h×n. */
matlab_mat *matlab_econ_ssm_forecast(struct matlab_obj_s *mdl, double hh,
                                     matlab_mat *Ym) {
    SSM s; if (!readSSM(mdl, s)) return mat_alloc(0, 0);
    int H = static_cast<int>(hh);
    if (H < 1) return mat_alloc(0, 0);
    int kind = static_cast<int>(matlab_obj_get_f64(mdl, "ModelKind", 9));
    int64_t m = s.m, n = s.n;
    std::vector<double> Y; int64_t T = 0, nc = 0; readTk(Ym, Y, T, nc);
    if (nc != n) return mat_alloc(0, 0);
    std::vector<double> Q = gram(s.B, m, s.kb), R = gram(s.D, n, s.hd);
    for (int64_t i = 0; i < n; ++i) R[static_cast<size_t>(i * n + i)] += 1e-8;
    std::vector<double> Xf;
    kalmanFilter(s.A, Q, s.C, R, Y, m, n, T, (kind == 7 ? 1e7 : 1e4), Xf);
    std::vector<double> x(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) x[static_cast<size_t>(i)] = Xf[static_cast<size_t>((T - 1) * m + i)];
    matlab_mat *out = mat_alloc(H, n);
    for (int h = 0; h < H; ++h) {
        x = matMul(s.A, x, m, m, 1);
        std::vector<double> y = matMul(s.C, x, n, m, 1);
        for (int64_t i = 0; i < n; ++i) out->data[h * n + i] = y[static_cast<size_t>(i)];
    }
    return out;
}

} // extern "C"

/* ============================================================================
 * §T6 — Bayesian linear regression (bayeslm) + Markov chains (dtmc)
 * ----------------------------------------------------------------------------
 * bayeslm: conjugate/diffuse Normal-Inverse-Gamma posterior — the posterior
 * mean of beta under a diffuse prior is the OLS estimate; the posterior
 * scale is the residual variance.  dtmc: discrete-time Markov chain over a
 * transition matrix — stationary distribution via power iteration; path
 * simulation via the cumulative row distributions.
 * ==========================================================================*/

extern "C" {

/* estimate(Mdl, X, y) — Bayesian linear regression posterior (diffuse prior
 * => posterior mean = OLS).  Mutates the receiver in place. */
struct matlab_obj_s *matlab_econ_bayeslm_estimate(struct matlab_obj_s *mdl,
                                                  matlab_mat *Xm,
                                                  matlab_mat *Ym) {
    if (!mdl || !Xm || !Xm->data) return mdl;
    int64_t n = Xm->rows, p = Xm->cols;
    std::vector<double> X(static_cast<size_t>(n * p));
    for (int64_t i = 0; i < n * p; ++i) X[static_cast<size_t>(i)] = Xm->data[i];
    std::vector<double> y = vecOf(Ym);
    if (static_cast<int64_t>(y.size()) != n) return mdl;
    std::vector<double> beta;
    if (!ols(X, y, n, p, beta)) return mdl;
    double sse = 0.0;
    for (int64_t t = 0; t < n; ++t) {
        double yhat = 0.0;
        for (int64_t j = 0; j < p; ++j)
            yhat += X[static_cast<size_t>(t * p + j)] * beta[static_cast<size_t>(j)];
        double e = y[static_cast<size_t>(t)] - yhat;
        sse += e * e;
    }
    double sigma2 = (n > p) ? sse / static_cast<double>(n - p) : sse;
    matlab_mat *bm = mat_alloc(p, 1);
    for (int64_t j = 0; j < p; ++j) bm->data[j] = beta[static_cast<size_t>(j)];
    matlab_obj_set_mat(mdl, "Beta", 4, bm);
    matlab_obj_set_f64(mdl, "Sigma2", 6, sigma2);
    matlab_obj_set_f64(mdl, "NumPredictors", 13, static_cast<double>(p));
    matlab_obj_set_f64(mdl, "ModelKind", 9, 9.0);
    return mdl;
}

/* forecast(Mdl, XNew) — posterior-mean prediction XNew * Beta. */
matlab_mat *matlab_econ_bayeslm_forecast(struct matlab_obj_s *mdl,
                                         matlab_mat *Xm) {
    if (!mdl || !Xm || !Xm->data) return mat_alloc(0, 0);
    matlab_mat *bm = matlab_obj_get_mat(mdl, "Beta", 4);
    if (!bm) return mat_alloc(0, 0);
    int64_t n = Xm->rows, p = Xm->cols;
    int64_t bp = bm->rows * bm->cols;
    if (bp != p) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(n, 1);
    for (int64_t t = 0; t < n; ++t) {
        double s = 0.0;
        for (int64_t j = 0; j < p; ++j)
            s += Xm->data[t * p + j] * bm->data[j];
        out->data[t] = s;
    }
    return out;
}

/* asymptotics(mc) — stationary distribution of the Markov chain, 1 x k.
 * Power iteration on the row-stochastic transition matrix. */
matlab_mat *matlab_econ_dtmc_asymptotics(struct matlab_obj_s *mc) {
    if (!mc) return mat_alloc(0, 0);
    matlab_mat *Pm = matlab_obj_get_mat(mc, "P", 1);
    if (!Pm || !Pm->data) return mat_alloc(0, 0);
    int64_t k = Pm->rows;
    if (k < 1 || Pm->cols != k) return mat_alloc(0, 0);
    std::vector<double> pi(static_cast<size_t>(k), 1.0 / static_cast<double>(k));
    std::vector<double> nx(static_cast<size_t>(k), 0.0);
    for (int it = 0; it < 2000; ++it) {
        for (int64_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < k; ++i)
                s += pi[static_cast<size_t>(i)] * Pm->data[i * k + j];
            nx[static_cast<size_t>(j)] = s;
        }
        double diff = 0.0, sum = 0.0;
        for (int64_t j = 0; j < k; ++j) sum += nx[static_cast<size_t>(j)];
        if (sum > 0.0) for (int64_t j = 0; j < k; ++j) nx[static_cast<size_t>(j)] /= sum;
        for (int64_t j = 0; j < k; ++j)
            diff += std::fabs(nx[static_cast<size_t>(j)] - pi[static_cast<size_t>(j)]);
        pi = nx;
        if (diff < 1e-14) break;
    }
    matlab_mat *out = mat_alloc(1, k);
    for (int64_t j = 0; j < k; ++j) out->data[j] = pi[static_cast<size_t>(j)];
    return out;
}

/* simulate(mc, numSteps) — a state path (numSteps+1 x 1, 1-based states),
 * starting from state 1, driven by the row cumulative distributions. */
matlab_mat *matlab_econ_dtmc_simulate(struct matlab_obj_s *mc, double ns) {
    if (!mc) return mat_alloc(0, 0);
    int n = static_cast<int>(ns);
    if (n < 1) return mat_alloc(0, 0);
    matlab_mat *Pm = matlab_obj_get_mat(mc, "P", 1);
    if (!Pm || !Pm->data) return mat_alloc(0, 0);
    int64_t k = Pm->rows;
    if (k < 1 || Pm->cols != k) return mat_alloc(0, 0);
    Lcg rng{0xC0FFEE1234ULL};
    matlab_mat *out = mat_alloc(n + 1, 1);
    int64_t state = 0;            /* internal 0-based; emit 1-based */
    out->data[0] = 1.0;
    for (int t = 1; t <= n; ++t) {
        double u = rng.next();
        double cum = 0.0;
        int64_t nextState = k - 1;
        for (int64_t j = 0; j < k; ++j) {
            cum += Pm->data[state * k + j];
            if (u <= cum) { nextState = j; break; }
        }
        state = nextState;
        out->data[t] = static_cast<double>(state + 1);
    }
    return out;
}

} // extern "C"
