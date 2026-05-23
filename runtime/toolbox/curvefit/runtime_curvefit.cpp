/* ============================================================================
 * runtime_curvefit.cpp — Curve Fitting Toolbox runtime
 * ----------------------------------------------------------------------------
 * Tier-1: the universal "fit me a polynomial and tell me how good it is" loop.
 *   fit(x, y, 'polyN')  → a populated `cfit` object (alloc-then-populate),
 *   feval(f, xq) / f(xq) → evaluate the fitted model,
 *   [f, gof] / [f, gof, output] = fit(...) → goodness-of-fit + output structs,
 *   coeffvalues(f)      → the fitted coefficient vector,
 *   disp(f)             → the MATLAB-faithful model block.
 *
 * Polynomial fits ride the shipped `matlab_polyfit` / `matlab_polyval`
 * (Vandermonde QR-LS).  Center-and-scale is on by default for conditioning
 * (cdate-style predictors at degree >1 are otherwise hopeless): the object
 * carries Mu / Sigma and `feval` rescales the query points before evaluating.
 * No external dependency — every core is hand-coded over the shipped numeric
 * base, matching the project precedent (Stats / Image / Ident).
 *
 * Companion classdef (`cfit` / `fittype` / `fitoptions`) lives in
 * curvefit_classdefs.m; this TU holds the numeric cores those methods call.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

/* Shipped numeric base reused by the fit cores. */
extern "C" matlab_mat *matlab_polyfit(matlab_mat *x, matlab_mat *y, double n);
extern "C" matlab_mat *matlab_polyval(matlab_mat *p, matlab_mat *x);

/* Object accessors (alloc-then-populate, class-pinned dispatch). */
extern "C" double matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void   matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);

/* ===== file-scope helpers (C++ linkage) ================================== */

/* matlab_string layout (matches runtime/matlab_runtime.cpp). */
struct cf_string_s { char *data; int64_t len; };

static std::string cf_sstr(const void *s) {
    if (!s) return std::string();
    const cf_string_s *p = reinterpret_cast<const cf_string_s *>(s);
    if (!p->data || p->len <= 0) return std::string();
    return std::string(p->data, p->data + p->len);
}

/* Flatten any matlab_mat (row / column / full) into a contiguous vector in
 * row-major order — the fit cores treat the data as a 1-D sample. */
static std::vector<double> cf_flat(const matlab_mat *m) {
    std::vector<double> v;
    if (!m || !m->data) return v;
    int64_t n = m->rows * m->cols;
    v.resize(static_cast<size_t>(n < 0 ? 0 : n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = m->data[i];
    return v;
}

static double cf_mean(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

/* Sample standard deviation (N-1), matching MATLAB's default `std`. */
static double cf_std(const std::vector<double> &v, double mu) {
    if (v.size() < 2) return 1.0;
    double s = 0.0;
    for (double x : v) s += (x - mu) * (x - mu);
    double sd = sqrt(s / static_cast<double>(v.size() - 1));
    return (sd > 0.0) ? sd : 1.0;
}

/* Wrap a vector of doubles as a fresh 1×n matlab_mat (row vector). */
static matlab_mat *cf_rowmat(const std::vector<double> &v) {
    matlab_mat *m = mat_alloc(1, static_cast<int64_t>(v.size()));
    for (size_t i = 0; i < v.size(); ++i) m->data[i] = v[i];
    return m;
}

/* Parse a 'polyN' model tag → degree N (1..9).  Returns -1 if not poly. */
static int cf_poly_degree(const std::string &tag) {
    if (tag.size() == 5 && tag.compare(0, 4, "poly") == 0) {
        char d = tag[4];
        if (d >= '1' && d <= '9') return d - '0';
    }
    return -1;
}

/* Evaluate the stored model at query points (already-allocated obj). */
static matlab_mat *cf_eval_poly(matlab_obj *obj, const matlab_mat *xq) {
    double mu  = matlab_obj_get_f64(obj, "Mu", 2);
    double sig = matlab_obj_get_f64(obj, "Sigma", 5);
    if (sig == 0.0) sig = 1.0;
    matlab_mat *coeffs = matlab_obj_get_mat(obj, "Coeffs", 6);
    /* Scale the query points the same way the fit was conditioned. */
    std::vector<double> xs = cf_flat(xq);
    for (double &x : xs) x = (x - mu) / sig;
    matlab_mat *xqs = mat_alloc(xq ? xq->rows : 1, xq ? xq->cols : 1);
    for (size_t i = 0; i < xs.size(); ++i) xqs->data[i] = xs[i];
    matlab_mat *yq = matlab_polyval(coeffs, xqs);
    return yq ? yq : mat_alloc(0, 0);
}

static double cf_model_value(int id, const double *p, int k, double x);   /* fwd */

/* Evaluate a fitted nonlinear model at query points (raw domain). */
static matlab_mat *cf_eval_nonlinear(matlab_obj *obj, int id, const matlab_mat *xq) {
    matlab_mat *coeffs = matlab_obj_get_mat(obj, "Coeffs", 6);
    std::vector<double> p = cf_flat(coeffs);
    int k = static_cast<int>(p.size());
    matlab_mat *out = mat_alloc(xq ? xq->rows : 1, xq ? xq->cols : 1);
    int64_t nq = xq ? xq->rows * xq->cols : 0;
    for (int64_t i = 0; i < nq; ++i)
        out->data[i] = cf_model_value(id, p.data(), k, xq->data[i]);
    return out;
}

/* ===== Tier-2: nonlinear library models (hand-coded Levenberg-Marquardt) ==
 * Each named family carries a closed-form value + analytic gradient, so the
 * LM core needs no finite differences and no function-handle ABI.  Model ids:
 *   2 = exp1 (a*e^{bx}) · 3 = exp2 (a*e^{bx}+c*e^{dx}) ·
 *   4 = power1 (a*x^b)  · 5 = power2 (a*x^b+c) ·
 *   6 = gaussN  (Σ aᵢ*e^{-((x-bᵢ)/cᵢ)²},      3N params) ·
 *   7 = sinN    (Σ aᵢ*sin(bᵢ*x+cᵢ),           3N params) ·
 *   8 = fourierN(a0 + Σ aₙcos(nωx)+bₙsin(nωx), 2N+2 params, ω is the last).
 * Polynomial stays model id 1 (Tier-1).  Coeffs are stored in the *raw*
 * domain (no center-and-scale — that is a polynomial conditioning trick). */

/* Trailing-integer parse: "gauss3" with prefix "gauss" → 3 (else -1). */
static int cf_suffix_n(const std::string &tag, const char *prefix) {
    size_t pl = strlen(prefix);
    if (tag.size() != pl + 1 || tag.compare(0, pl, prefix) != 0) return -1;
    char d = tag[pl];
    return (d >= '1' && d <= '8') ? d - '0' : -1;
}

/* Parse a nonlinear model tag → (id, ncoef).  Returns id 0 if not nonlinear. */
static int cf_nl_model(const std::string &tag, int &ncoef) {
    if (tag == "exp1")   { ncoef = 2; return 2; }
    if (tag == "exp2")   { ncoef = 4; return 3; }
    if (tag == "power1") { ncoef = 2; return 4; }
    if (tag == "power2") { ncoef = 3; return 5; }
    int n;
    if ((n = cf_suffix_n(tag, "gauss"))   > 0) { ncoef = 3 * n;     return 6; }
    if ((n = cf_suffix_n(tag, "sin"))     > 0) { ncoef = 3 * n;     return 7; }
    if ((n = cf_suffix_n(tag, "fourier")) > 0) { ncoef = 2 * n + 2; return 8; }
    ncoef = 0;
    return 0;
}

/* Model value at x for parameter vector p (k = number of params). */
static double cf_model_value(int id, const double *p, int k, double x) {
    switch (id) {
        case 2: return p[0] * exp(p[1] * x);
        case 3: return p[0] * exp(p[1] * x) + p[2] * exp(p[3] * x);
        case 4: return p[0] * pow(x, p[1]);
        case 5: return p[0] * pow(x, p[1]) + p[2];
        case 6: {                                           /* gaussN */
            double s = 0.0;
            for (int i = 0; i + 2 < k; i += 3) {
                double u = (x - p[i + 1]) / p[i + 2];
                s += p[i] * exp(-u * u);
            }
            return s;
        }
        case 7: {                                           /* sinN */
            double s = 0.0;
            for (int i = 0; i + 2 < k; i += 3) s += p[i] * sin(p[i + 1] * x + p[i + 2]);
            return s;
        }
        case 8: {                                           /* fourierN */
            int N = (k - 2) / 2; double w = p[k - 1];
            double s = p[0];
            for (int i = 1; i <= N; ++i)
                s += p[2 * i - 1] * cos(i * w * x) + p[2 * i] * sin(i * w * x);
            return s;
        }
        default: return 0.0;
    }
}

/* Analytic gradient d(value)/d(p[j]) at x — writes k entries into g. */
static void cf_model_grad(int id, const double *p, int k, double x, double *g) {
    switch (id) {
        case 2: { double e = exp(p[1] * x); g[0] = e; g[1] = p[0] * x * e; break; }
        case 3: {
            double e1 = exp(p[1] * x), e2 = exp(p[3] * x);
            g[0] = e1; g[1] = p[0] * x * e1; g[2] = e2; g[3] = p[2] * x * e2; break;
        }
        case 4: { double xb = pow(x, p[1]); g[0] = xb;
                  g[1] = (x > 0.0) ? p[0] * xb * log(x) : 0.0; break; }
        case 5: { double xb = pow(x, p[1]); g[0] = xb;
                  g[1] = (x > 0.0) ? p[0] * xb * log(x) : 0.0; g[2] = 1.0; break; }
        case 6: {                                           /* gaussN */
            for (int i = 0; i + 2 < k; i += 3) {
                double u = (x - p[i + 1]) / p[i + 2], e = exp(-u * u);
                g[i] = e; g[i + 1] = p[i] * e * 2.0 * u / p[i + 2];
                g[i + 2] = p[i] * e * 2.0 * u * u / p[i + 2];
            }
            break;
        }
        case 7: {                                           /* sinN */
            for (int i = 0; i + 2 < k; i += 3) {
                double th = p[i + 1] * x + p[i + 2];
                g[i] = sin(th); g[i + 1] = p[i] * cos(th) * x; g[i + 2] = p[i] * cos(th);
            }
            break;
        }
        case 8: {                                           /* fourierN */
            int N = (k - 2) / 2; double w = p[k - 1];
            g[0] = 1.0; double dw = 0.0;
            for (int i = 1; i <= N; ++i) {
                double cc = cos(i * w * x), ss = sin(i * w * x);
                g[2 * i - 1] = cc; g[2 * i] = ss;
                dw += i * x * (p[2 * i] * cc - p[2 * i - 1] * ss);
            }
            g[k - 1] = dw;
            break;
        }
        default: break;
    }
}

/* Solve the small k×k system A x = b (k ≤ 4) by Gaussian elimination with
 * partial pivoting.  A is row-major k×k (modified in place).  Returns false
 * on a singular system. */
static bool cf_solve(std::vector<double> &A, std::vector<double> &b, int k) {
    for (int col = 0; col < k; ++col) {
        int piv = col; double best = fabs(A[static_cast<size_t>(col * k + col)]);
        for (int r = col + 1; r < k; ++r) {
            double v = fabs(A[static_cast<size_t>(r * k + col)]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return false;
        if (piv != col) {
            for (int c = 0; c < k; ++c)
                std::swap(A[static_cast<size_t>(col * k + c)], A[static_cast<size_t>(piv * k + c)]);
            std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(piv)]);
        }
        double d = A[static_cast<size_t>(col * k + col)];
        for (int r = col + 1; r < k; ++r) {
            double f = A[static_cast<size_t>(r * k + col)] / d;
            for (int c = col; c < k; ++c) A[static_cast<size_t>(r * k + c)] -= f * A[static_cast<size_t>(col * k + c)];
            b[static_cast<size_t>(r)] -= f * b[static_cast<size_t>(col)];
        }
    }
    for (int r = k - 1; r >= 0; --r) {
        double s = b[static_cast<size_t>(r)];
        for (int c = r + 1; c < k; ++c) s -= A[static_cast<size_t>(r * k + c)] * b[static_cast<size_t>(c)];
        b[static_cast<size_t>(r)] = s / A[static_cast<size_t>(r * k + r)];
    }
    return true;
}

/* Levenberg-Marquardt fit of model `id` to (x,y) starting from p (in/out).
 * lb/ub may be null (unbounded); finite entries box-project each step.
 * w is a per-point weight vector (empty = unweighted): the normal equations
 * use Σ wᵢ·gᵢgᵢᵀ and the cost is Σ wᵢ·eᵢ² — this carries both `Weights` and
 * the robust-IRLS reweighting. */
static void cf_lm(int id, const std::vector<double> &x, const std::vector<double> &y,
                  std::vector<double> &p, int k, const double *lb, const double *ub,
                  const std::vector<double> &w) {
    int64_t n = static_cast<int64_t>(x.size());
    auto wt = [&](int64_t i) -> double { return w.empty() ? 1.0 : w[static_cast<size_t>(i)]; };
    auto cost = [&](const std::vector<double> &pp) -> double {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double e = y[static_cast<size_t>(i)] - cf_model_value(id, pp.data(), k, x[static_cast<size_t>(i)]);
            s += wt(i) * e * e;
        }
        return s;
    };
    double lambda = 1e-3;
    double c = cost(p);
    for (int iter = 0; iter < 200; ++iter) {
        std::vector<double> H(static_cast<size_t>(k * k), 0.0), g(static_cast<size_t>(k), 0.0), gr(static_cast<size_t>(k));
        for (int64_t i = 0; i < n; ++i) {
            double e = y[static_cast<size_t>(i)] - cf_model_value(id, p.data(), k, x[static_cast<size_t>(i)]);
            cf_model_grad(id, p.data(), k, x[static_cast<size_t>(i)], gr.data());
            double wi = wt(i);
            for (int a = 0; a < k; ++a) {
                g[static_cast<size_t>(a)] += wi * gr[static_cast<size_t>(a)] * e;
                for (int bb = 0; bb < k; ++bb)
                    H[static_cast<size_t>(a * k + bb)] += wi * gr[static_cast<size_t>(a)] * gr[static_cast<size_t>(bb)];
            }
        }
        bool improved = false;
        for (int tries = 0; tries < 12 && !improved; ++tries) {
            std::vector<double> A = H, dp = g;
            for (int a = 0; a < k; ++a) A[static_cast<size_t>(a * k + a)] += lambda * (H[static_cast<size_t>(a * k + a)] + 1e-12);
            if (!cf_solve(A, dp, k)) { lambda *= 10.0; continue; }
            std::vector<double> pn(p);
            for (int a = 0; a < k; ++a) {
                pn[static_cast<size_t>(a)] += dp[static_cast<size_t>(a)];
                if (lb && pn[static_cast<size_t>(a)] < lb[a]) pn[static_cast<size_t>(a)] = lb[a];
                if (ub && pn[static_cast<size_t>(a)] > ub[a]) pn[static_cast<size_t>(a)] = ub[a];
            }
            double cn = cost(pn);
            if (cn < c) { p = pn; c = cn; lambda *= 0.3; improved = true; }
            else lambda *= 10.0;
        }
        if (!improved) break;
        if (lambda > 1e12) break;
    }
}

/* Auto start-point heuristics per family (writes k entries into p). */
static void cf_startpoint(int id, const std::vector<double> &x, const std::vector<double> &y,
                          int k, std::vector<double> &p) {
    p.assign(static_cast<size_t>(k), 1.0);
    int64_t n = static_cast<int64_t>(x.size());
    if (n < 1) return;
    /* log-linear seed: regress log|y| (or log y over log x) on a basis. */
    auto linreg = [&](const std::vector<double> &u, const std::vector<double> &v,
                      double &slope, double &intercept) {
        double su = 0, sv = 0, suu = 0, suv = 0; int64_t m = static_cast<int64_t>(u.size());
        for (int64_t i = 0; i < m; ++i) { su += u[static_cast<size_t>(i)]; sv += v[static_cast<size_t>(i)];
            suu += u[static_cast<size_t>(i)] * u[static_cast<size_t>(i)]; suv += u[static_cast<size_t>(i)] * v[static_cast<size_t>(i)]; }
        double den = m * suu - su * su;
        slope = (fabs(den) > 1e-300) ? (m * suv - su * sv) / den : 0.0;
        intercept = (sv - slope * su) / static_cast<double>(m);
    };
    if (id == 2 || id == 3) {                       /* exp1 / exp2: log-linear */
        std::vector<double> lx, ly;
        for (int64_t i = 0; i < n; ++i)
            if (y[static_cast<size_t>(i)] > 0.0) { lx.push_back(x[static_cast<size_t>(i)]); ly.push_back(log(y[static_cast<size_t>(i)])); }
        double b = 0, a = 0;
        if (lx.size() >= 2) { linreg(lx, ly, b, a); p[0] = exp(a); p[1] = b; }
        else { p[0] = y[0]; p[1] = 0.0; }
        if (id == 3) { p[0] *= 0.5; p[2] = p[0]; p[3] = p[1] * 3.0; }   /* split into two rates */
    } else if (id == 4 || id == 5) {                /* power1 / power2: log-log */
        std::vector<double> lx, ly;
        for (int64_t i = 0; i < n; ++i)
            if (x[static_cast<size_t>(i)] > 0.0 && y[static_cast<size_t>(i)] > 0.0) {
                lx.push_back(log(x[static_cast<size_t>(i)])); ly.push_back(log(y[static_cast<size_t>(i)])); }
        double b = 1, a = 0;
        if (lx.size() >= 2) { linreg(lx, ly, b, a); p[0] = exp(a); p[1] = b; }
        if (id == 5) p[2] = 0.0;
    } else {
        /* x-range + y stats shared by the multi-term seeds below. */
        double xmin = x[0], xmax = x[0], ymax = y[0], ymin = y[0], ybar = 0.0;
        int64_t imax = 0;
        for (int64_t i = 0; i < n; ++i) {
            double xi = x[static_cast<size_t>(i)], yi = y[static_cast<size_t>(i)];
            if (xi < xmin) xmin = xi; if (xi > xmax) xmax = xi;
            if (yi > ymax) { ymax = yi; imax = i; }
            if (yi < ymin) ymin = yi;
            ybar += yi;
        }
        ybar /= static_cast<double>(n);
        double span = (xmax > xmin) ? (xmax - xmin) : 1.0;
        if (id == 6) {                               /* gaussN: peaks spread across x */
            int N = k / 3;
            for (int i = 0; i < N; ++i) {
                p[static_cast<size_t>(3 * i)]     = ymax;
                p[static_cast<size_t>(3 * i + 1)] = (N == 1) ? x[static_cast<size_t>(imax)]
                                                            : xmin + (i + 0.5) * span / N;
                p[static_cast<size_t>(3 * i + 2)] = span / (2.0 * N);
            }
        } else if (id == 7) {                        /* sinN: harmonics of a base freq */
            int N = k / 3;
            double w0 = 2.0 * M_PI / span;           /* ~one period over the data span */
            for (int i = 0; i < N; ++i) {
                p[static_cast<size_t>(3 * i)]     = (ymax - ymin) / 2.0;
                p[static_cast<size_t>(3 * i + 1)] = (i + 1) * w0;
                p[static_cast<size_t>(3 * i + 2)] = 0.0;
            }
        } else if (id == 8) {                        /* fourierN: a0 = mean, ω from span */
            int N = (k - 2) / 2;
            p[0] = ybar;
            for (int i = 1; i <= N; ++i) {
                p[static_cast<size_t>(2 * i - 1)] = 0.0;
                p[static_cast<size_t>(2 * i)]     = 0.0;
            }
            p[static_cast<size_t>(k - 1)] = 2.0 * M_PI / span;
        }
    }
}

/* Run a nonlinear fit; populate the cfit obj.  `opts`-supplied start point /
 * bounds / weights (if non-null/non-empty) override the auto heuristic.
 * robust: 0 = off, 1 = bisquare, 2 = LAR — wraps the LM in an IRLS loop. */
static void cf_fit_nonlinear(matlab_obj *obj, int id, int ncoef,
                             const std::vector<double> &xv, const std::vector<double> &yv,
                             const double *sp, const double *lb, const double *ub,
                             const std::vector<double> &weights, int robust) {
    int64_t n = static_cast<int64_t>(xv.size() < yv.size() ? xv.size() : yv.size());
    std::vector<double> x(xv.begin(), xv.begin() + n), y(yv.begin(), yv.begin() + n);
    std::vector<double> p;
    cf_startpoint(id, x, y, ncoef, p);
    if (sp) for (int a = 0; a < ncoef; ++a) p[static_cast<size_t>(a)] = sp[a];

    /* base weights (Weights option, or all-ones). */
    std::vector<double> w0;
    if (!weights.empty()) { w0 = weights; w0.resize(static_cast<size_t>(n), 1.0); }
    std::vector<double> w = w0;
    cf_lm(id, x, y, p, ncoef, lb, ub, w);

    /* Robust IRLS: recompute per-point weights from the residual MAD scale. */
    if (robust > 0) {
        for (int it = 0; it < 6; ++it) {
            std::vector<double> ar(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i)
                ar[static_cast<size_t>(i)] = fabs(y[static_cast<size_t>(i)] - cf_model_value(id, p.data(), ncoef, x[static_cast<size_t>(i)]));
            std::vector<double> srt = ar; std::sort(srt.begin(), srt.end());
            double mad = srt[static_cast<size_t>(n / 2)];
            double s = (mad > 1e-12) ? mad / 0.6745 : 1.0;
            w.assign(static_cast<size_t>(n), 1.0);
            double K = (robust == 1) ? 4.685 : 1.0;       /* bisquare tuning */
            for (int64_t i = 0; i < n; ++i) {
                double r = ar[static_cast<size_t>(i)] / (K * s);
                double wi;
                if (robust == 1) wi = (r < 1.0) ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
                else             wi = 1.0 / (ar[static_cast<size_t>(i)] + 1e-6);   /* LAR ~ L1 */
                if (!w0.empty()) wi *= w0[static_cast<size_t>(i)];
                w[static_cast<size_t>(i)] = wi;
            }
            cf_lm(id, x, y, p, ncoef, lb, ub, w);
        }
    }

    matlab_mat *coeffs = mat_alloc(1, ncoef);
    for (int a = 0; a < ncoef; ++a) coeffs->data[a] = p[static_cast<size_t>(a)];
    double ybar = cf_mean(y), sse = 0.0, sst = 0.0;
    matlab_mat *resid = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) {
        double e = y[static_cast<size_t>(i)] - cf_model_value(id, p.data(), ncoef, x[static_cast<size_t>(i)]);
        resid->data[i] = e; sse += e * e;
        sst += (y[static_cast<size_t>(i)] - ybar) * (y[static_cast<size_t>(i)] - ybar);
    }
    double dfe = static_cast<double>(n - ncoef);
    double r2 = (sst > 0.0) ? 1.0 - sse / sst : (sse == 0.0 ? 1.0 : 0.0);
    double r2adj = (sst > 0.0 && dfe > 0.0)
                       ? 1.0 - (sse / dfe) / (sst / static_cast<double>(n - 1)) : r2;
    double rmse = (dfe > 0.0) ? sqrt(sse / dfe) : 0.0;
    matlab_obj_set_f64(obj, "ModelType", 9, static_cast<double>(id));
    matlab_obj_set_f64(obj, "Degree", 6, 0.0);
    matlab_obj_set_mat(obj, "Coeffs", 6, coeffs);
    matlab_obj_set_f64(obj, "Mu", 2, 0.0);
    matlab_obj_set_f64(obj, "Sigma", 5, 1.0);
    matlab_obj_set_f64(obj, "NumObs", 6, static_cast<double>(n));
    matlab_obj_set_f64(obj, "NumCoeffs", 9, static_cast<double>(ncoef));
    matlab_obj_set_f64(obj, "SSE", 3, sse);
    matlab_obj_set_f64(obj, "Rsquare", 7, r2);
    matlab_obj_set_f64(obj, "DFE", 3, dfe);
    matlab_obj_set_f64(obj, "AdjRsquare", 10, r2adj);
    matlab_obj_set_f64(obj, "RMSE", 4, rmse);
    matlab_obj_set_mat(obj, "Resid", 5, resid);
}

/* ===== public entry points (dispatched from Lowering.cpp) ================ */

extern "C" {

/* fit(x, y, 'polyN') — populate a pre-allocated cfit shell.
 * `model` arrives as a matlab_string* (the pde_table const_char coercion).
 * Returns a dummy empty matrix (the result is emitted with a None type and
 * ignored — the populated `obj` is the real product), mirroring fitlm_init. */
matlab_mat *matlab_curvefit_fit(matlab_obj *obj, matlab_mat *x, matlab_mat *y,
                                void *model) {
    if (!obj) return mat_alloc(0, 0);
    std::string tag = cf_sstr(model);

    /* Nonlinear library model? (exp / power / gauss / sin / fourier) — LM. */
    int nlncoef = 0;
    int nlid = cf_nl_model(tag, nlncoef);
    if (nlid > 0) {
        cf_fit_nonlinear(obj, nlid, nlncoef, cf_flat(x), cf_flat(y),
                         nullptr, nullptr, nullptr, std::vector<double>(), 0);
        return mat_alloc(0, 0);
    }

    int deg = cf_poly_degree(tag);
    if (deg < 0) deg = 1;                       /* default to a line */

    std::vector<double> xv = cf_flat(x);
    std::vector<double> yv = cf_flat(y);
    int64_t n = static_cast<int64_t>(xv.size() < yv.size() ? xv.size() : yv.size());
    if (n < 1) n = 0;

    /* Center-and-scale the predictor for conditioning. */
    double mu  = cf_mean(xv);
    double sig = cf_std(xv, mu);
    std::vector<double> xs(static_cast<size_t>(n));
    std::vector<double> ys(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        xs[static_cast<size_t>(i)] = (xv[static_cast<size_t>(i)] - mu) / sig;
        ys[static_cast<size_t>(i)] = yv[static_cast<size_t>(i)];
    }
    /* Degree is capped at n-1 so the Vandermonde stays full-rank. */
    if (deg > n - 1 && n >= 1) deg = static_cast<int>(n - 1);
    if (deg < 0) deg = 0;

    matlab_mat *xsm = cf_rowmat(xs);
    matlab_mat *ysm = cf_rowmat(ys);
    matlab_mat *coeffs = matlab_polyfit(xsm, ysm, static_cast<double>(deg));
    if (!coeffs) coeffs = mat_alloc(1, 1);

    /* Residuals + goodness-of-fit on the original-scale response. */
    matlab_mat *yhat = matlab_polyval(coeffs, xsm);
    double ybar = cf_mean(ys);
    double sse = 0.0, sst = 0.0;
    matlab_mat *resid = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) {
        double yi = ys[static_cast<size_t>(i)];
        double fi = (yhat && i < yhat->rows * yhat->cols) ? yhat->data[i] : 0.0;
        double e = yi - fi;
        resid->data[i] = e;
        sse += e * e;
        sst += (yi - ybar) * (yi - ybar);
    }
    int ncoef = static_cast<int>(coeffs->rows * coeffs->cols);
    double dfe = static_cast<double>(n - ncoef);
    double r2 = (sst > 0.0) ? 1.0 - sse / sst : (sse == 0.0 ? 1.0 : 0.0);
    double r2adj = (sst > 0.0 && dfe > 0.0)
                       ? 1.0 - (sse / dfe) / (sst / static_cast<double>(n - 1))
                       : r2;
    double rmse = (dfe > 0.0) ? sqrt(sse / dfe) : 0.0;

    matlab_obj_set_f64(obj, "ModelType", 9, 1.0);          /* 1 = polynomial */
    matlab_obj_set_f64(obj, "Degree", 6, static_cast<double>(deg));
    matlab_obj_set_mat(obj, "Coeffs", 6, coeffs);
    matlab_obj_set_f64(obj, "Mu", 2, mu);
    matlab_obj_set_f64(obj, "Sigma", 5, sig);
    matlab_obj_set_f64(obj, "NumObs", 6, static_cast<double>(n));
    matlab_obj_set_f64(obj, "NumCoeffs", 9, static_cast<double>(ncoef));
    matlab_obj_set_f64(obj, "SSE", 3, sse);
    matlab_obj_set_f64(obj, "Rsquare", 7, r2);
    matlab_obj_set_f64(obj, "DFE", 3, dfe);
    matlab_obj_set_f64(obj, "AdjRsquare", 10, r2adj);
    matlab_obj_set_f64(obj, "RMSE", 4, rmse);
    matlab_obj_set_mat(obj, "Resid", 5, resid);
    return mat_alloc(0, 0);
}

/* fit(x, y, model, opts) — like matlab_curvefit_fit but the trailing
 * `fitoptions` object supplies StartPoint / Lower / Upper / Weights / Robust
 * for nonlinear families.  Polynomial models ignore the carrier (delegate). */
matlab_mat *matlab_curvefit_fit_opts(matlab_obj *obj, matlab_mat *x, matlab_mat *y,
                                     void *model, matlab_obj *opts) {
    if (!obj) return mat_alloc(0, 0);
    std::string tag = cf_sstr(model);
    int nlncoef = 0;
    int nlid = cf_nl_model(tag, nlncoef);
    if (nlid == 0)                                      /* polynomial: no opts surface yet */
        return matlab_curvefit_fit(obj, x, y, model);

    std::vector<double> sp, lb, ub, weights;
    int robust = 0;
    if (opts) {
        sp      = cf_flat(matlab_obj_get_mat(opts, "StartPoint", 10));
        lb      = cf_flat(matlab_obj_get_mat(opts, "Lower", 5));
        ub      = cf_flat(matlab_obj_get_mat(opts, "Upper", 5));
        weights = cf_flat(matlab_obj_get_mat(opts, "Weights", 7));
        robust  = static_cast<int>(matlab_obj_get_f64(opts, "RobustCode", 10));
    }
    const double *spp = (static_cast<int>(sp.size()) == nlncoef) ? sp.data() : nullptr;
    const double *lbp = (static_cast<int>(lb.size()) == nlncoef) ? lb.data() : nullptr;
    const double *ubp = (static_cast<int>(ub.size()) == nlncoef) ? ub.data() : nullptr;
    cf_fit_nonlinear(obj, nlid, nlncoef, cf_flat(x), cf_flat(y),
                     spp, lbp, ubp, weights, robust);
    return mat_alloc(0, 0);
}

/* feval(f, xq) / f(xq) — evaluate the fitted model at query points. */
matlab_mat *matlab_curvefit_feval(matlab_obj *obj, matlab_mat *xq) {
    if (!obj) return mat_alloc(0, 0);
    int mtype = static_cast<int>(matlab_obj_get_f64(obj, "ModelType", 9));
    if (mtype == 1) return cf_eval_poly(obj, xq);          /* polynomial */
    return cf_eval_nonlinear(obj, mtype, xq);              /* exp/power/gauss */
}

/* coeffvalues(f) — the fitted coefficient row vector (a fresh copy). */
matlab_mat *matlab_curvefit_coeffvalues(matlab_obj *obj) {
    if (!obj) return mat_alloc(0, 0);
    matlab_mat *c = matlab_obj_get_mat(obj, "Coeffs", 6);
    if (!c) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(c->rows, c->cols);
    int64_t n = c->rows * c->cols;
    for (int64_t i = 0; i < n; ++i) out->data[i] = c->data[i];
    return out;
}

/* [~, gof] = fit(...) — the goodness-of-fit struct (read back from obj). */
matlab_mat *matlab_curvefit_gof(matlab_obj *obj) {
    matlab_struct *s = matlab_struct_new();
    if (obj) {
        matlab_struct_set_f64(s, "sse", 3, matlab_obj_get_f64(obj, "SSE", 3));
        matlab_struct_set_f64(s, "rsquare", 7, matlab_obj_get_f64(obj, "Rsquare", 7));
        matlab_struct_set_f64(s, "dfe", 3, matlab_obj_get_f64(obj, "DFE", 3));
        matlab_struct_set_f64(s, "adjrsquare", 10, matlab_obj_get_f64(obj, "AdjRsquare", 10));
        matlab_struct_set_f64(s, "rmse", 4, matlab_obj_get_f64(obj, "RMSE", 4));
    }
    return reinterpret_cast<matlab_mat *>(s);
}

/* [~, ~, output] = fit(...) — the fit-output struct. */
matlab_mat *matlab_curvefit_output(matlab_obj *obj) {
    matlab_struct *s = matlab_struct_new();
    if (obj) {
        matlab_struct_set_f64(s, "numobs", 6, matlab_obj_get_f64(obj, "NumObs", 6));
        matlab_struct_set_f64(s, "numparam", 8, matlab_obj_get_f64(obj, "NumCoeffs", 9));
        matlab_mat *resid = matlab_obj_get_mat(obj, "Resid", 5);
        if (resid) matlab_struct_set_mat(s, "residuals", 9, resid);
        matlab_struct_set_f64(s, "exitflag", 8, 1.0);
        matlab_struct_set_f64(s, "iterations", 10, 0.0);
    }
    return reinterpret_cast<matlab_mat *>(s);
}

/* disp(f) — the MATLAB-faithful model block.
 *   General model PolyN:
 *      f(x) = p1*x^N + ... + pN1
 *      where x is normalized by mean MU and std SIGMA
 *   Coefficients:
 *      p1 = ...
 * Returns nothing (None-typed at the call site). */
matlab_mat *matlab_curvefit_disp(matlab_obj *obj) {
    if (!obj) return mat_alloc(0, 0);
    int mtype = static_cast<int>(matlab_obj_get_f64(obj, "ModelType", 9));
    int deg = static_cast<int>(matlab_obj_get_f64(obj, "Degree", 6));
    double mu  = matlab_obj_get_f64(obj, "Mu", 2);
    double sig = matlab_obj_get_f64(obj, "Sigma", 5);
    matlab_mat *c = matlab_obj_get_mat(obj, "Coeffs", 6);
    int nc = c ? static_cast<int>(c->rows * c->cols) : 0;

    pthread_mutex_lock(&matlab_io_mutex);
    /* Two-coeff exp/power families: fixed a/b/c/d names + formula. */
    if (mtype >= 2 && mtype <= 5) {
        static const char *NL_NAME[6] = { "", "", "Exp1", "Exp2", "Power1", "Power2" };
        static const char *NL_FORM[6] = { "", "",
            "a*exp(b*x)", "a*exp(b*x) + c*exp(d*x)", "a*x^b", "a*x^b + c" };
        printf("     General model %s:\n", NL_NAME[mtype]);
        printf("       f(x) = %s\n", NL_FORM[mtype]);
        printf("     Coefficients:\n");
        const char *names = "abcd";
        for (int i = 0; i < nc && i < 4; ++i)
            printf("       %c = %.6g\n", names[i], c->data[i]);
        pthread_mutex_unlock(&matlab_io_mutex);
        return mat_alloc(0, 0);
    }
    /* Multi-term gaussN / sinN: 3 coeffs per term (aᵢ/bᵢ/cᵢ). */
    if (mtype == 6 || mtype == 7) {
        int N = nc / 3;
        printf("     General model %s%d:\n", mtype == 6 ? "Gauss" : "Sin", N);
        printf("       f(x) = %s\n", mtype == 6
            ? "sum_i a_i*exp(-((x-b_i)/c_i)^2)" : "sum_i a_i*sin(b_i*x+c_i)");
        printf("     Coefficients:\n");
        for (int i = 0; i < N; ++i) {
            printf("       a%d = %.6g\n", i + 1, c->data[3 * i]);
            printf("       b%d = %.6g\n", i + 1, c->data[3 * i + 1]);
            printf("       c%d = %.6g\n", i + 1, c->data[3 * i + 2]);
        }
        pthread_mutex_unlock(&matlab_io_mutex);
        return mat_alloc(0, 0);
    }
    /* fourierN: a0, (aᵢ,bᵢ) pairs, then ω. */
    if (mtype == 8) {
        int N = (nc - 2) / 2;
        printf("     General model Fourier%d:\n", N);
        printf("       f(x) = a0 + sum_n a_n*cos(n*w*x) + b_n*sin(n*w*x)\n");
        printf("     Coefficients:\n");
        printf("       a0 = %.6g\n", c->data[0]);
        for (int i = 1; i <= N; ++i) {
            printf("       a%d = %.6g\n", i, c->data[2 * i - 1]);
            printf("       b%d = %.6g\n", i, c->data[2 * i]);
        }
        printf("       w = %.6g\n", c->data[nc - 1]);
        pthread_mutex_unlock(&matlab_io_mutex);
        return mat_alloc(0, 0);
    }
    printf("     General model Poly%d:\n", deg);
    printf("       f(x) = ");
    for (int i = 0; i < nc; ++i) {
        int power = deg - i;
        if (i > 0) printf(" + ");
        printf("p%d", i + 1);
        if (power >= 2)      printf("*x^%d", power);
        else if (power == 1) printf("*x");
    }
    printf("\n");
    if (sig != 1.0 || mu != 0.0)
        printf("       where x is normalized by mean %.6g and std %.6g\n", mu, sig);
    printf("     Coefficients:\n");
    for (int i = 0; i < nc; ++i)
        printf("       p%d = %.6g\n", i + 1, c->data[i]);
    pthread_mutex_unlock(&matlab_io_mutex);
    return mat_alloc(0, 0);
}

}  /* extern "C" */
