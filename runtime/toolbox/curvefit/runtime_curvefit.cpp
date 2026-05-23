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
extern "C" matlab_mat *matlab_sgolayfilt(matlab_mat *x, double k, double f);

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

/* Deep-copy a matlab_mat (preserving shape). */
static matlab_mat *cf_copymat(const matlab_mat *m) {
    if (!m) return mat_alloc(0, 0);
    matlab_mat *o = mat_alloc(m->rows, m->cols);
    int64_t n = m->rows * m->cols;
    for (int64_t i = 0; i < n; ++i) o->data[i] = m->data[i];
    return o;
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
    matlab_obj_set_mat(obj, "Xdata", 5, cf_rowmat(x));   /* for confint / differentiate */
}

/* ===== Tier-3: custom equations + fit postprocessing ===================== *
 * Custom models (id 100) carry the user equation string; a small recursive-
 * descent evaluator computes the value, and the LM uses a finite-difference
 * Jacobian (no analytic derivative of an arbitrary expression).  confint
 * needs a Student-t inverse, replicated here (the stats `stinv` is static). */

extern "C" struct matlab_string_s *matlab_string_from_literal(const char *src, int64_t n);
extern "C" void   matlab_obj_set_string(matlab_obj *o, const char *name, int64_t len, void *str);
extern "C" void  *matlab_obj_get_string(matlab_obj *o, const char *name, int64_t len);

/* --- Student-t CDF + inverse (replicated from runtime_stats stinv) ------- */
static double cf_betacf(double a, double b, double x) {
    const int MAXIT = 200; const double EPS = 1e-15, FPMIN = 1e-300;
    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d; double h = d;
    for (int m = 1; m <= MAXIT; ++m) {
        double mm = static_cast<double>(m), m2 = 2.0 * mm;
        double aa = mm * (b - mm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; double del = d * c; h *= del;
        if (fabs(del - 1.0) < EPS) break;
    }
    return h;
}
static double cf_betai(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) +
                    a * log(x) + b * log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) return bt * cf_betacf(a, b, x) / a;
    return 1.0 - bt * cf_betacf(b, a, 1.0 - x) / b;
}
static double cf_tcdf(double t, double nu) {
    double x = nu / (nu + t * t);
    double ib = 0.5 * cf_betai(nu / 2.0, 0.5, x);
    return (t > 0.0) ? 1.0 - ib : ib;
}
static double cf_tinv(double p, double nu) {
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return INFINITY;
    double lo = -1e4, hi = 1e4;
    for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        if (cf_tcdf(mid, nu) < p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

/* --- recursive-descent evaluator for a custom equation string ----------- */
static bool cf_is_func(const std::string &id) {
    static const char *F[] = {"exp", "log", "log10", "sqrt", "sin", "cos",
                              "tan", "tanh", "abs", "sinh", "cosh", "atan", nullptr};
    for (int i = 0; F[i]; ++i) if (id == F[i]) return true;
    return false;
}
struct CfEval {
    const char *s; size_t pos, n;
    const std::vector<std::string> &nm; const double *vals; double xval;
    CfEval(const std::string &src, const std::vector<std::string> &names,
           const double *v, double x)
        : s(src.c_str()), pos(0), n(src.size()), nm(names), vals(v), xval(x) {}
    void skip() { while (pos < n && (s[pos] == ' ' || s[pos] == '\t')) ++pos; }
    double parse() { return expr(); }
    double expr() {
        double v = term();
        for (;;) { skip(); char c = (pos < n) ? s[pos] : '\0';
            if (c == '+') { ++pos; v += term(); }
            else if (c == '-') { ++pos; v -= term(); }
            else break; }
        return v;
    }
    double term() {
        double v = powf_();
        for (;;) { skip(); char c = (pos < n) ? s[pos] : '\0';
            if (c == '*') { ++pos; v *= powf_(); }
            else if (c == '/') { ++pos; v /= powf_(); }
            else break; }
        return v;
    }
    double powf_() {
        double v = unary();
        skip();
        if (pos < n && (s[pos] == '^' || (s[pos] == '.' && pos + 1 < n && s[pos + 1] == '^'))) {
            pos += (s[pos] == '.') ? 2 : 1;
            return pow(v, powf_());
        }
        return v;
    }
    double unary() {
        skip();
        if (pos < n && s[pos] == '-') { ++pos; return -unary(); }
        if (pos < n && s[pos] == '+') { ++pos; return unary(); }
        return atom();
    }
    double atom() {
        skip();
        if (pos < n && s[pos] == '(') { ++pos; double v = expr(); skip();
            if (pos < n && s[pos] == ')') ++pos; return v; }
        if (pos < n && (isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.')) {
            char *end = nullptr; double v = strtod(s + pos, &end);
            pos = static_cast<size_t>(end - s); return v;
        }
        size_t st = pos;
        while (pos < n && (isalnum(static_cast<unsigned char>(s[pos])) || s[pos] == '_')) ++pos;
        std::string id(s + st, s + pos);
        skip();
        if (pos < n && s[pos] == '(') {
            ++pos; double a = expr(); skip(); if (pos < n && s[pos] == ')') ++pos;
            if (id == "exp") return exp(a);   if (id == "log") return log(a);
            if (id == "log10") return log10(a); if (id == "sqrt") return sqrt(a);
            if (id == "sin") return sin(a);   if (id == "cos") return cos(a);
            if (id == "tan") return tan(a);   if (id == "tanh") return tanh(a);
            if (id == "abs") return fabs(a);  if (id == "sinh") return sinh(a);
            if (id == "cosh") return cosh(a); if (id == "atan") return atan(a);
            return 0.0;
        }
        if (id == "x") return xval;
        if (id == "pi") return M_PI;
        if (id == "e") return M_E;
        for (size_t i = 0; i < nm.size(); ++i) if (nm[i] == id) return vals[i];
        return 0.0;
    }
};
static double cf_expr_eval(const std::string &src, const std::vector<std::string> &nm,
                          const double *vals, double x) {
    CfEval e(src, nm, vals, x); return e.parse();
}
/* Coefficient names = identifiers that aren't x/pi/e or a function call,
 * collected unique + sorted (MATLAB lists coefficients alphabetically). */
static std::vector<std::string> cf_extract_coeffs(const std::string &src) {
    std::vector<std::string> out; size_t i = 0, n = src.size();
    while (i < n) {
        if (isalpha(static_cast<unsigned char>(src[i])) || src[i] == '_') {
            size_t st = i;
            while (i < n && (isalnum(static_cast<unsigned char>(src[i])) || src[i] == '_')) ++i;
            std::string id = src.substr(st, i - st);
            size_t j = i; while (j < n && (src[j] == ' ' || src[j] == '\t')) ++j;
            bool isfn = (j < n && src[j] == '(');
            if (!isfn && id != "x" && id != "pi" && id != "e" && !cf_is_func(id)) {
                bool dup = false; for (const auto &c : out) if (c == id) { dup = true; break; }
                if (!dup) out.push_back(id);
            }
        } else ++i;
    }
    std::sort(out.begin(), out.end());
    return out;
}
/* Finite-difference Levenberg-Marquardt for a custom equation. */
static void cf_lm_custom(const std::string &eqn, const std::vector<std::string> &nm,
                         const std::vector<double> &x, const std::vector<double> &y,
                         std::vector<double> &p) {
    int k = static_cast<int>(p.size());
    int64_t n = static_cast<int64_t>(x.size());
    auto cost = [&](const std::vector<double> &pp) -> double {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double e = y[static_cast<size_t>(i)] - cf_expr_eval(eqn, nm, pp.data(), x[static_cast<size_t>(i)]);
            s += e * e;
        }
        return s;
    };
    double lambda = 1e-3, c = cost(p);
    for (int iter = 0; iter < 200; ++iter) {
        std::vector<double> H(static_cast<size_t>(k * k), 0.0), g(static_cast<size_t>(k), 0.0), gr(static_cast<size_t>(k));
        for (int64_t i = 0; i < n; ++i) {
            double f0 = cf_expr_eval(eqn, nm, p.data(), x[static_cast<size_t>(i)]);
            double e = y[static_cast<size_t>(i)] - f0;
            for (int a = 0; a < k; ++a) {
                double h = 1e-6 * (fabs(p[static_cast<size_t>(a)]) + 1e-6);
                std::vector<double> pp = p; pp[static_cast<size_t>(a)] += h;
                gr[static_cast<size_t>(a)] = (cf_expr_eval(eqn, nm, pp.data(), x[static_cast<size_t>(i)]) - f0) / h;
            }
            for (int a = 0; a < k; ++a) {
                g[static_cast<size_t>(a)] += gr[static_cast<size_t>(a)] * e;
                for (int bb = 0; bb < k; ++bb)
                    H[static_cast<size_t>(a * k + bb)] += gr[static_cast<size_t>(a)] * gr[static_cast<size_t>(bb)];
            }
        }
        bool improved = false;
        for (int tries = 0; tries < 12 && !improved; ++tries) {
            std::vector<double> A = H, dp = g;
            for (int a = 0; a < k; ++a) A[static_cast<size_t>(a * k + a)] += lambda * (H[static_cast<size_t>(a * k + a)] + 1e-12);
            if (!cf_solve(A, dp, k)) { lambda *= 10.0; continue; }
            std::vector<double> pn = p;
            for (int a = 0; a < k; ++a) pn[static_cast<size_t>(a)] += dp[static_cast<size_t>(a)];
            double cn = cost(pn);
            if (cn < c) { p = pn; c = cn; lambda *= 0.3; improved = true; }
            else lambda *= 10.0;
        }
        if (!improved) break;
        if (lambda > 1e12) break;
    }
}
/* Read the model row of coeffs from the cfit obj into a vector. */
static std::vector<double> cf_obj_coeffs(matlab_obj *obj) {
    return cf_flat(matlab_obj_get_mat(obj, "Coeffs", 6));
}
/* Build the n×k Jacobian J[i*k+a] = ∂model/∂p_a at the stored Xdata, for any
 * model type (poly via scaled-Vandermonde, library via grad, custom via FD). */
static std::vector<double> cf_jacobian(matlab_obj *obj, int mtype, int &nout, int &kout) {
    std::vector<double> p = cf_obj_coeffs(obj);
    std::vector<double> x = cf_flat(matlab_obj_get_mat(obj, "Xdata", 5));
    int k = static_cast<int>(p.size());
    int64_t n = static_cast<int64_t>(x.size());
    nout = static_cast<int>(n); kout = k;
    std::vector<double> J(static_cast<size_t>(n) * static_cast<size_t>(k), 0.0);
    if (mtype == 1) {                                    /* polynomial: scaled Vandermonde */
        double mu = matlab_obj_get_f64(obj, "Mu", 2), sig = matlab_obj_get_f64(obj, "Sigma", 5);
        if (sig == 0.0) sig = 1.0;
        for (int64_t i = 0; i < n; ++i) {
            double xs = (x[static_cast<size_t>(i)] - mu) / sig, xp = 1.0;
            for (int a = k - 1; a >= 0; --a) {           /* highest power first */
                J[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(a)] = xp; xp *= xs;
            }
        }
    } else if (mtype >= 2 && mtype <= 8) {               /* library: analytic grad */
        std::vector<double> gr(static_cast<size_t>(k));
        for (int64_t i = 0; i < n; ++i) {
            cf_model_grad(mtype, p.data(), k, x[static_cast<size_t>(i)], gr.data());
            for (int a = 0; a < k; ++a)
                J[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(a)] = gr[static_cast<size_t>(a)];
        }
    } else {                                             /* custom: finite difference */
        std::string eqn = cf_sstr(matlab_obj_get_string(obj, "Expr", 4));
        std::vector<std::string> nm = cf_extract_coeffs(eqn);
        for (int64_t i = 0; i < n; ++i) {
            double f0 = cf_expr_eval(eqn, nm, p.data(), x[static_cast<size_t>(i)]);
            for (int a = 0; a < k; ++a) {
                double h = 1e-6 * (fabs(p[static_cast<size_t>(a)]) + 1e-6);
                std::vector<double> pp = p; pp[static_cast<size_t>(a)] += h;
                J[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(a)] =
                    (cf_expr_eval(eqn, nm, pp.data(), x[static_cast<size_t>(i)]) - f0) / h;
            }
        }
    }
    return J;
}

/* ===== Tier-4/6: interpolation + cubic-spline core ======================= *
 * A self-contained interpolation kernel (linear / nearest / pchip / spline)
 * shared by the interpolant cfit types (Tier-4), `smooth`, the smoothing
 * spline (`csaps`), and the ppform `spline`/`fnval` family (Tier-6).  All
 * grids are assumed sorted ascending in x. */

static int cf_find_interval(const std::vector<double> &x, double xq) {
    int n = static_cast<int>(x.size());
    if (n < 2) return 0;
    if (xq <= x[0]) return 0;
    if (xq >= x[static_cast<size_t>(n - 1)]) return n - 2;
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x[static_cast<size_t>(mid)] <= xq) lo = mid; else hi = mid;
    }
    return lo;
}

/* Not-a-knot cubic-spline second derivatives M (matches MATLAB `spline`). */
static std::vector<double> cf_spline_M(const std::vector<double> &x, const std::vector<double> &y) {
    int n = static_cast<int>(x.size());
    std::vector<double> M(static_cast<size_t>(n), 0.0);
    if (n < 3) return M;
    std::vector<double> h(static_cast<size_t>(n - 1));
    for (int i = 0; i < n - 1; ++i) h[static_cast<size_t>(i)] = x[static_cast<size_t>(i + 1)] - x[static_cast<size_t>(i)];
    std::vector<double> A(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0), b(static_cast<size_t>(n), 0.0);
    A[0 * static_cast<size_t>(n) + 0] = h[1];
    A[0 * static_cast<size_t>(n) + 1] = -(h[0] + h[1]);
    A[0 * static_cast<size_t>(n) + 2] = h[0];
    for (int i = 1; i < n - 1; ++i) {
        size_t r = static_cast<size_t>(i) * static_cast<size_t>(n);
        A[r + static_cast<size_t>(i - 1)] = h[static_cast<size_t>(i - 1)];
        A[r + static_cast<size_t>(i)]     = 2.0 * (h[static_cast<size_t>(i - 1)] + h[static_cast<size_t>(i)]);
        A[r + static_cast<size_t>(i + 1)] = h[static_cast<size_t>(i)];
        b[static_cast<size_t>(i)] = 6.0 * ((y[static_cast<size_t>(i + 1)] - y[static_cast<size_t>(i)]) / h[static_cast<size_t>(i)] -
                                           (y[static_cast<size_t>(i)] - y[static_cast<size_t>(i - 1)]) / h[static_cast<size_t>(i - 1)]);
    }
    size_t rl = static_cast<size_t>(n - 1) * static_cast<size_t>(n);
    A[rl + static_cast<size_t>(n - 3)] = h[static_cast<size_t>(n - 2)];
    A[rl + static_cast<size_t>(n - 2)] = -(h[static_cast<size_t>(n - 3)] + h[static_cast<size_t>(n - 2)]);
    A[rl + static_cast<size_t>(n - 1)] = h[static_cast<size_t>(n - 3)];
    cf_solve(A, b, n);
    return b;
}
static double cf_spline_at(const std::vector<double> &x, const std::vector<double> &y,
                           const std::vector<double> &M, double xq) {
    int i = cf_find_interval(x, xq);
    double h = x[static_cast<size_t>(i + 1)] - x[static_cast<size_t>(i)];
    if (h == 0.0) return y[static_cast<size_t>(i)];
    double t = xq - x[static_cast<size_t>(i)], a = x[static_cast<size_t>(i + 1)] - xq;
    return (M[static_cast<size_t>(i)] * a * a * a + M[static_cast<size_t>(i + 1)] * t * t * t) / (6.0 * h)
         + (y[static_cast<size_t>(i)] / h - M[static_cast<size_t>(i)] * h / 6.0) * a
         + (y[static_cast<size_t>(i + 1)] / h - M[static_cast<size_t>(i + 1)] * h / 6.0) * t;
}

/* pchip monotone-cubic Hermite slopes (Fritsch-Carlson). */
static std::vector<double> cf_pchip_slopes(const std::vector<double> &x, const std::vector<double> &y) {
    int n = static_cast<int>(x.size());
    std::vector<double> d(static_cast<size_t>(n), 0.0);
    if (n < 2) return d;
    std::vector<double> h(static_cast<size_t>(n - 1)), del(static_cast<size_t>(n - 1));
    for (int i = 0; i < n - 1; ++i) {
        h[static_cast<size_t>(i)] = x[static_cast<size_t>(i + 1)] - x[static_cast<size_t>(i)];
        del[static_cast<size_t>(i)] = (y[static_cast<size_t>(i + 1)] - y[static_cast<size_t>(i)]) / h[static_cast<size_t>(i)];
    }
    for (int i = 1; i < n - 1; ++i) {
        double dl = del[static_cast<size_t>(i - 1)], dr = del[static_cast<size_t>(i)];
        if (dl * dr > 0.0) {
            double w1 = 2.0 * h[static_cast<size_t>(i)] + h[static_cast<size_t>(i - 1)];
            double w2 = h[static_cast<size_t>(i)] + 2.0 * h[static_cast<size_t>(i - 1)];
            d[static_cast<size_t>(i)] = (w1 + w2) / (w1 / dl + w2 / dr);
        }
    }
    d[0] = del[0];
    d[static_cast<size_t>(n - 1)] = del[static_cast<size_t>(n - 2)];
    return d;
}
static double cf_pchip_at(const std::vector<double> &x, const std::vector<double> &y,
                          const std::vector<double> &d, double xq) {
    int i = cf_find_interval(x, xq);
    double h = x[static_cast<size_t>(i + 1)] - x[static_cast<size_t>(i)];
    if (h == 0.0) return y[static_cast<size_t>(i)];
    double t = (xq - x[static_cast<size_t>(i)]) / h;
    double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0, h10 = t * t * t - 2.0 * t * t + t;
    double h01 = -2.0 * t * t * t + 3.0 * t * t,       h11 = t * t * t - t * t;
    return h00 * y[static_cast<size_t>(i)] + h10 * h * d[static_cast<size_t>(i)]
         + h01 * y[static_cast<size_t>(i + 1)] + h11 * h * d[static_cast<size_t>(i + 1)];
}

/* Dispatch interpolation by mode (1 linear, 2 nearest, 3 pchip, 4 spline). */
static matlab_mat *cf_interp(int mode, const std::vector<double> &x, const std::vector<double> &y,
                             const matlab_mat *xq) {
    int64_t nq = xq ? xq->rows * xq->cols : 0;
    matlab_mat *out = mat_alloc(xq ? xq->rows : 1, xq ? xq->cols : 1);
    std::vector<double> M, d;
    if (mode == 4) M = cf_spline_M(x, y);
    if (mode == 3) d = cf_pchip_slopes(x, y);
    for (int64_t q = 0; q < nq; ++q) {
        double xv = xq->data[q], r;
        int i = cf_find_interval(x, xv);
        if (mode == 1) {                                  /* linear */
            double h = x[static_cast<size_t>(i + 1)] - x[static_cast<size_t>(i)];
            r = (h == 0.0) ? y[static_cast<size_t>(i)]
              : y[static_cast<size_t>(i)] + (y[static_cast<size_t>(i + 1)] - y[static_cast<size_t>(i)]) * (xv - x[static_cast<size_t>(i)]) / h;
        } else if (mode == 2) {                           /* nearest */
            r = (xv - x[static_cast<size_t>(i)] <= x[static_cast<size_t>(i + 1)] - xv)
              ? y[static_cast<size_t>(i)] : y[static_cast<size_t>(i + 1)];
        } else if (mode == 3) {                           /* pchip */
            r = cf_pchip_at(x, y, d, xv);
        } else {                                          /* spline */
            r = cf_spline_at(x, y, M, xv);
        }
        out->data[q] = r;
    }
    return out;
}

/* Local-regression smoother (lowess deg 1 / loess deg 2), optionally robust. */
static std::vector<double> cf_lowess(const std::vector<double> &x, const std::vector<double> &y,
                                     int span, int deg, bool robust) {
    int64_t n = static_cast<int64_t>(x.size());
    std::vector<double> out(static_cast<size_t>(n)), rw(static_cast<size_t>(n), 1.0);
    int half = span / 2; if (half < 1) half = 1;
    int passes = robust ? 3 : 1;
    for (int pass = 0; pass < passes; ++pass) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t lo = i - half, hi = i + half;
            if (lo < 0) lo = 0; if (hi > n - 1) hi = n - 1;
            double xi = x[static_cast<size_t>(i)], dmax = 1e-12;
            for (int64_t j = lo; j <= hi; ++j) dmax = std::max(dmax, fabs(x[static_cast<size_t>(j)] - xi));
            int k = deg + 1;
            std::vector<double> A(static_cast<size_t>(k * k), 0.0), bvec(static_cast<size_t>(k), 0.0);
            for (int64_t j = lo; j <= hi; ++j) {
                double u = fabs(x[static_cast<size_t>(j)] - xi) / dmax;
                double w = (u < 1.0) ? pow(1.0 - u * u * u, 3.0) : 0.0;   /* tricube */
                w *= rw[static_cast<size_t>(j)];
                double xp = 1.0, dx = x[static_cast<size_t>(j)] - xi;
                std::vector<double> basis(static_cast<size_t>(k));
                for (int a = 0; a < k; ++a) { basis[static_cast<size_t>(a)] = xp; xp *= dx; }
                for (int a = 0; a < k; ++a) {
                    bvec[static_cast<size_t>(a)] += w * basis[static_cast<size_t>(a)] * y[static_cast<size_t>(j)];
                    for (int b = 0; b < k; ++b)
                        A[static_cast<size_t>(a * k + b)] += w * basis[static_cast<size_t>(a)] * basis[static_cast<size_t>(b)];
                }
            }
            if (cf_solve(A, bvec, k)) out[static_cast<size_t>(i)] = bvec[0];   /* value at dx=0 = intercept */
            else out[static_cast<size_t>(i)] = y[static_cast<size_t>(i)];
        }
        if (robust && pass + 1 < passes) {                 /* bisquare reweight on residuals */
            std::vector<double> ar(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) ar[static_cast<size_t>(i)] = fabs(y[static_cast<size_t>(i)] - out[static_cast<size_t>(i)]);
            std::vector<double> s = ar; std::sort(s.begin(), s.end());
            double mad = s[static_cast<size_t>(n / 2)]; double sc = (mad > 1e-12) ? 6.0 * mad : 1.0;
            for (int64_t i = 0; i < n; ++i) {
                double u = ar[static_cast<size_t>(i)] / sc;
                rw[static_cast<size_t>(i)] = (u < 1.0) ? (1.0 - u * u) * (1.0 - u * u) : 0.0;
            }
        }
    }
    return out;
}

/* Reinsch cubic smoothing spline: smoothed values g at the data sites.
 * Minimises Σ(yᵢ−gᵢ)² + λ∫g″² with λ = (1−p)/p; g is then a natural cubic
 * spline through (x, g) for off-knot evaluation. */
static std::vector<double> cf_smooth_spline(const std::vector<double> &x, const std::vector<double> &y,
                                            double p) {
    int n = static_cast<int>(x.size());
    if (n < 3 || p >= 1.0) return y;
    if (p < 0.0) p = 0.0;
    double lambda = (p > 0.0) ? (1.0 - p) / p : 1e12;
    std::vector<double> h(static_cast<size_t>(n - 1));
    for (int i = 0; i < n - 1; ++i) h[static_cast<size_t>(i)] = x[static_cast<size_t>(i + 1)] - x[static_cast<size_t>(i)];
    int m = n - 2;                                          /* interior knots */
    /* Q is n×m; R is m×m tridiagonal.  Solve (R + λ QᵀQ) γ = Qᵀy, g = y − λ Q γ. */
    auto Q = [&](int i, int j) -> double {                 /* i in [0,n), j in [0,m) */
        int c = j + 1;                                     /* column maps to interior knot c */
        if (i == c - 1) return 1.0 / h[static_cast<size_t>(c - 1)];
        if (i == c)     return -(1.0 / h[static_cast<size_t>(c - 1)] + 1.0 / h[static_cast<size_t>(c)]);
        if (i == c + 1) return 1.0 / h[static_cast<size_t>(c)];
        return 0.0;
    };
    std::vector<double> A(static_cast<size_t>(m * m), 0.0), rhs(static_cast<size_t>(m), 0.0);
    for (int j = 0; j < m; ++j) {
        int c = j + 1;
        for (int i = 0; i < n; ++i) rhs[static_cast<size_t>(j)] += Q(i, j) * y[static_cast<size_t>(i)];
        for (int l = 0; l < m; ++l) {                      /* (QᵀQ)[j,l] */
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += Q(i, j) * Q(i, l);
            A[static_cast<size_t>(j * m + l)] += lambda * s;
        }
        /* R[j,j], R[j,j±1] */
        A[static_cast<size_t>(j * m + j)] += (h[static_cast<size_t>(c - 1)] + h[static_cast<size_t>(c)]) / 3.0;
        if (j > 0)     A[static_cast<size_t>(j * m + (j - 1))] += h[static_cast<size_t>(c - 1)] / 6.0;
        if (j < m - 1) A[static_cast<size_t>(j * m + (j + 1))] += h[static_cast<size_t>(c)] / 6.0;
    }
    cf_solve(A, rhs, m);                                    /* rhs now holds γ */
    std::vector<double> g(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        double qg = 0.0;
        for (int j = 0; j < m; ++j) qg += Q(i, j) * rhs[static_cast<size_t>(j)];
        g[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] - lambda * qg;
    }
    return g;
}

/* ===== Tier-5: surface fitting (polynomial surfaces → sfit) ============== *
 * polyNM means degree N in x, M in y, with total degree ≤ max(N,M) — the
 * MATLAB convention.  The bivariate design matrix is solved by the normal
 * equations; the sfit object stores N, M and the coefficient vector. */

/* Term exponents (i,j) for polyNM, in MATLAB's order (ascending total degree,
 * then ascending j). */
static std::vector<std::pair<int,int>> cf_surf_terms(int N, int M) {
    std::vector<std::pair<int,int>> t;
    int tot = (N > M) ? N : M;
    for (int s = 0; s <= tot; ++s)
        for (int i = s; i >= 0; --i) {
            int j = s - i;
            if (i <= N && j <= M) t.push_back(std::make_pair(i, j));
        }
    return t;
}

/* ===== Tier-6: ppform spline layer (spline / pchip / fnval / fnder / fnint) =
 * A piecewise-polynomial (ppform) object carries Breaks (knots), a Pieces×Order
 * Coefs matrix (each row highest-power-first in the local coordinate x−break),
 * Pieces and Order.  fnval evaluates, fnder/fnint produce new ppforms. */

static int cf_pp_piece(const std::vector<double> &br, double xq) {
    int np = static_cast<int>(br.size()) - 1;
    if (np < 1) return 0;
    if (xq <= br[0]) return 0;
    if (xq >= br[static_cast<size_t>(np)]) return np - 1;
    int lo = 0, hi = np;
    while (hi - lo > 1) { int m = (lo + hi) / 2; if (br[static_cast<size_t>(m)] <= xq) lo = m; else hi = m; }
    return lo;
}
static double cf_pp_eval(const std::vector<double> &br, const std::vector<double> &cf,
                         int order, double xq) {
    int i = cf_pp_piece(br, xq);
    double t = xq - br[static_cast<size_t>(i)], v = 0.0;
    for (int a = 0; a < order; ++a) v = v * t + cf[static_cast<size_t>(i * order + a)];
    return v;
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

    /* Interpolant fit types (id 200): the cfit stores (x, y) + the interp
     * mode and feval routes through the interpolation kernel. */
    int imode = 0;
    if (tag == "linearinterp")       imode = 1;
    else if (tag == "nearestinterp") imode = 2;
    else if (tag == "pchipinterp" || tag == "cubicinterp") imode = 3;
    else if (tag == "splineinterp")  imode = 4;
    if (imode > 0) {
        int64_t nn = static_cast<int64_t>(cf_flat(x).size());
        matlab_obj_set_f64(obj, "ModelType", 9, 200.0);
        matlab_obj_set_f64(obj, "Degree", 6, static_cast<double>(imode));   /* InterpMode */
        matlab_obj_set_mat(obj, "Xdata", 5, cf_copymat(x));
        matlab_obj_set_mat(obj, "Coeffs", 6, cf_copymat(y));                /* stores y */
        matlab_obj_set_f64(obj, "NumObs", 6, static_cast<double>(nn));
        matlab_obj_set_f64(obj, "NumCoeffs", 9, static_cast<double>(nn));
        matlab_obj_set_f64(obj, "SSE", 3, 0.0);
        matlab_obj_set_f64(obj, "Rsquare", 7, 1.0);                         /* passes through */
        matlab_obj_set_f64(obj, "DFE", 3, 0.0);
        matlab_obj_set_f64(obj, "AdjRsquare", 10, 1.0);
        matlab_obj_set_f64(obj, "RMSE", 4, 0.0);
        return mat_alloc(0, 0);
    }

    /* Smoothing spline (id 201): Reinsch smoothing spline; the smoothed
     * values are stored and feval interpolates them with a natural cubic. */
    if (tag == "smoothingspline") {
        std::vector<double> xv = cf_flat(x), yv = cf_flat(y);
        int64_t nn = static_cast<int64_t>(xv.size() < yv.size() ? xv.size() : yv.size());
        xv.resize(static_cast<size_t>(nn)); yv.resize(static_cast<size_t>(nn));
        std::vector<double> g = cf_smooth_spline(xv, yv, 0.9);              /* default param */
        double sse = 0.0;
        for (int64_t i = 0; i < nn; ++i) { double e = yv[static_cast<size_t>(i)] - g[static_cast<size_t>(i)]; sse += e * e; }
        matlab_obj_set_f64(obj, "ModelType", 9, 201.0);
        matlab_obj_set_mat(obj, "Xdata", 5, cf_rowmat(xv));
        matlab_obj_set_mat(obj, "Coeffs", 6, cf_rowmat(g));                 /* smoothed values */
        matlab_obj_set_f64(obj, "NumObs", 6, static_cast<double>(nn));
        matlab_obj_set_f64(obj, "NumCoeffs", 9, static_cast<double>(nn));
        matlab_obj_set_f64(obj, "SSE", 3, sse);
        matlab_obj_set_f64(obj, "Rsquare", 7, 0.0);
        matlab_obj_set_f64(obj, "DFE", 3, 0.0);
        matlab_obj_set_f64(obj, "RMSE", 4, (nn > 0) ? sqrt(sse / static_cast<double>(nn)) : 0.0);
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
    matlab_obj_set_mat(obj, "Xdata", 5, cf_copymat(x));    /* for confint / differentiate */
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
    if (mtype >= 2 && mtype <= 8) return cf_eval_nonlinear(obj, mtype, xq);
    if (mtype == 200) {                                    /* interpolant */
        int imode = static_cast<int>(matlab_obj_get_f64(obj, "Degree", 6));
        std::vector<double> xv = cf_flat(matlab_obj_get_mat(obj, "Xdata", 5));
        std::vector<double> yv = cf_flat(matlab_obj_get_mat(obj, "Coeffs", 6));
        return cf_interp(imode, xv, yv, xq);
    }
    if (mtype == 201) {                                    /* smoothing spline (cubic of g) */
        std::vector<double> xv = cf_flat(matlab_obj_get_mat(obj, "Xdata", 5));
        std::vector<double> gv = cf_flat(matlab_obj_get_mat(obj, "Coeffs", 6));
        return cf_interp(4, xv, gv, xq);
    }
    /* custom equation (id 100): evaluate the stored expression. */
    std::string eqn = cf_sstr(matlab_obj_get_string(obj, "Expr", 4));
    std::vector<std::string> nm = cf_extract_coeffs(eqn);
    std::vector<double> p = cf_obj_coeffs(obj);
    matlab_mat *out = mat_alloc(xq ? xq->rows : 1, xq ? xq->cols : 1);
    int64_t nq = xq ? xq->rows * xq->cols : 0;
    for (int64_t i = 0; i < nq; ++i)
        out->data[i] = cf_expr_eval(eqn, nm, p.data(), xq->data[i]);
    return out;
}

/* fittype('a*exp(-b*x)+c') — store the equation string + coeff count on the
 * fittype shell.  Coeff names are re-extracted by the fitter / evaluator. */
matlab_mat *matlab_curvefit_fittype_init(matlab_obj *ft, void *exprstr) {
    if (!ft) return mat_alloc(0, 0);
    std::string eqn = cf_sstr(exprstr);
    matlab_obj_set_string(ft, "Expr", 4, exprstr);
    matlab_obj_set_f64(ft, "NumCoeffs", 9, static_cast<double>(cf_extract_coeffs(eqn).size()));
    matlab_obj_set_f64(ft, "ModelType", 9, 100.0);
    return mat_alloc(0, 0);
}

/* fit(x, y, fittypeObj) — custom-equation nonlinear fit (finite-diff LM). */
matlab_mat *matlab_curvefit_fit_custom(matlab_obj *obj, matlab_mat *x, matlab_mat *y,
                                       matlab_obj *ft) {
    if (!obj || !ft) return mat_alloc(0, 0);
    std::string eqn = cf_sstr(matlab_obj_get_string(ft, "Expr", 4));
    std::vector<std::string> nm = cf_extract_coeffs(eqn);
    int k = static_cast<int>(nm.size());
    if (k < 1) k = 1;
    std::vector<double> xv = cf_flat(x), yv = cf_flat(y);
    int64_t n = static_cast<int64_t>(xv.size() < yv.size() ? xv.size() : yv.size());
    std::vector<double> xx(xv.begin(), xv.begin() + n), yy(yv.begin(), yv.begin() + n);
    /* Custom equations have no closed-form start point, so multistart: run the
     * finite-diff LM from a spread of deterministic seeds and keep the best
     * SSE (structured all-±1 seeds first, then an LCG spread over [-5,5]). */
    auto sse_of = [&](const std::vector<double> &pp) -> double {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double e = yy[static_cast<size_t>(i)] - cf_expr_eval(eqn, nm, pp.data(), xx[static_cast<size_t>(i)]);
            s += e * e;
        }
        return s;
    };
    std::vector<double> p(static_cast<size_t>(k), 1.0), best;
    double bestcost = INFINITY;
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nextu = [&]() -> double {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((rng >> 11) & 0xFFFFF) / static_cast<double>(0x100000);
    };
    for (int trial = 0; trial < 32; ++trial) {
        std::vector<double> p0(static_cast<size_t>(k));
        if (trial == 0)      for (int a = 0; a < k; ++a) p0[static_cast<size_t>(a)] = 1.0;
        else if (trial == 1) for (int a = 0; a < k; ++a) p0[static_cast<size_t>(a)] = -1.0;
        else                 for (int a = 0; a < k; ++a) p0[static_cast<size_t>(a)] = nextu() * 10.0 - 5.0;
        cf_lm_custom(eqn, nm, xx, yy, p0);
        double cst = sse_of(p0);
        if (cst < bestcost) { bestcost = cst; best = p0; }
    }
    p = best.empty() ? p : best;

    matlab_mat *coeffs = mat_alloc(1, k);
    for (int a = 0; a < k; ++a) coeffs->data[a] = p[static_cast<size_t>(a)];
    double ybar = cf_mean(yy), sse = 0.0, sst = 0.0;
    matlab_mat *resid = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) {
        double e = yy[static_cast<size_t>(i)] - cf_expr_eval(eqn, nm, p.data(), xx[static_cast<size_t>(i)]);
        resid->data[i] = e; sse += e * e;
        sst += (yy[static_cast<size_t>(i)] - ybar) * (yy[static_cast<size_t>(i)] - ybar);
    }
    double dfe = static_cast<double>(n - k);
    double r2 = (sst > 0.0) ? 1.0 - sse / sst : (sse == 0.0 ? 1.0 : 0.0);
    double r2adj = (sst > 0.0 && dfe > 0.0)
                       ? 1.0 - (sse / dfe) / (sst / static_cast<double>(n - 1)) : r2;
    double rmse = (dfe > 0.0) ? sqrt(sse / dfe) : 0.0;
    matlab_obj_set_f64(obj, "ModelType", 9, 100.0);
    matlab_obj_set_f64(obj, "Degree", 6, 0.0);
    matlab_obj_set_mat(obj, "Coeffs", 6, coeffs);
    matlab_obj_set_string(obj, "Expr", 4, matlab_obj_get_string(ft, "Expr", 4));
    matlab_obj_set_f64(obj, "Mu", 2, 0.0);
    matlab_obj_set_f64(obj, "Sigma", 5, 1.0);
    matlab_obj_set_f64(obj, "NumObs", 6, static_cast<double>(n));
    matlab_obj_set_f64(obj, "NumCoeffs", 9, static_cast<double>(k));
    matlab_obj_set_f64(obj, "SSE", 3, sse);
    matlab_obj_set_f64(obj, "Rsquare", 7, r2);
    matlab_obj_set_f64(obj, "DFE", 3, dfe);
    matlab_obj_set_f64(obj, "AdjRsquare", 10, r2adj);
    matlab_obj_set_f64(obj, "RMSE", 4, rmse);
    matlab_obj_set_mat(obj, "Resid", 5, resid);
    matlab_obj_set_mat(obj, "Xdata", 5, cf_copymat(x));
    return mat_alloc(0, 0);
}

/* confint(f, level) — coefficient confidence intervals (2×k: [lower; upper]).
 * Σ = (SSE/dfe)·(JᵀJ)⁻¹; CI = b ± t_{dfe,1-α/2}·√Σ_aa. */
matlab_mat *matlab_curvefit_confint(matlab_obj *obj, double level) {
    if (!obj) return mat_alloc(0, 0);
    if (level <= 0.0 || level >= 1.0) level = 0.95;
    int mtype = static_cast<int>(matlab_obj_get_f64(obj, "ModelType", 9));
    int n = 0, k = 0;
    std::vector<double> J = cf_jacobian(obj, mtype, n, k);
    std::vector<double> p = cf_obj_coeffs(obj);
    double sse = matlab_obj_get_f64(obj, "SSE", 3);
    double dfe = matlab_obj_get_f64(obj, "DFE", 3);
    double s2 = (dfe > 0.0) ? sse / dfe : 0.0;
    /* JᵀJ (k×k). */
    std::vector<double> JtJ(static_cast<size_t>(k * k), 0.0);
    for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b) {
            double s = 0.0;
            for (int i = 0; i < n; ++i)
                s += J[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(a)] *
                     J[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(b)];
            JtJ[static_cast<size_t>(a * k + b)] = s;
        }
    /* invert JᵀJ by solving against each identity column. */
    matlab_mat *out = mat_alloc(2, k);
    double t = (dfe > 0.0) ? cf_tinv(1.0 - (1.0 - level) / 2.0, dfe) : 0.0;
    for (int a = 0; a < k; ++a) {
        std::vector<double> A = JtJ, e(static_cast<size_t>(k), 0.0);
        e[static_cast<size_t>(a)] = 1.0;
        double se = 0.0;
        if (cf_solve(A, e, k)) {
            double var = s2 * e[static_cast<size_t>(a)];     /* (JᵀJ)⁻¹_aa · σ² */
            se = (var > 0.0) ? sqrt(var) : 0.0;
        }
        out->data[a]     = p[static_cast<size_t>(a)] - t * se;   /* row 0: lower */
        out->data[k + a] = p[static_cast<size_t>(a)] + t * se;   /* row 1: upper */
    }
    return out;
}

/* differentiate(f, xq) — derivative values at xq.  Polynomial via the
 * coefficient derivative; everything else via a central finite difference
 * of feval (uniform, robust, and exact-to-tolerance for smooth models). */
matlab_mat *matlab_curvefit_differentiate(matlab_obj *obj, matlab_mat *xq) {
    if (!obj) return mat_alloc(0, 0);
    int64_t nq = xq ? xq->rows * xq->cols : 0;
    matlab_mat *out = mat_alloc(xq ? xq->rows : 1, xq ? xq->cols : 1);
    for (int64_t i = 0; i < nq; ++i) {
        double xi = xq->data[i];
        double h = 1e-5 * (fabs(xi) + 1.0);
        matlab_mat xp; double dp = xi + h; xp.data = &dp; xp.rows = 1; xp.cols = 1;
        matlab_mat xm; double dm = xi - h; xm.data = &dm; xm.rows = 1; xm.cols = 1;
        matlab_mat *yp = matlab_curvefit_feval(obj, &xp);
        matlab_mat *ym = matlab_curvefit_feval(obj, &xm);
        out->data[i] = (yp->data[0] - ym->data[0]) / (2.0 * h);
    }
    return out;
}

/* integrate(f, xq) — cumulative integral from xq(1) via the trapezoid rule
 * over the (assumed sorted) query grid. */
matlab_mat *matlab_curvefit_integrate(matlab_obj *obj, matlab_mat *xq) {
    if (!obj) return mat_alloc(0, 0);
    int64_t nq = xq ? xq->rows * xq->cols : 0;
    matlab_mat *yv = matlab_curvefit_feval(obj, xq);
    matlab_mat *out = mat_alloc(xq ? xq->rows : 1, xq ? xq->cols : 1);
    double acc = 0.0;
    if (nq > 0) out->data[0] = 0.0;
    for (int64_t i = 1; i < nq; ++i) {
        double dx = xq->data[i] - xq->data[i - 1];
        acc += 0.5 * (yv->data[i] + yv->data[i - 1]) * dx;
        out->data[i] = acc;
    }
    return out;
}

/* numcoeffs(f) — the coefficient count as a 1×1. */
matlab_mat *matlab_curvefit_numcoeffs(matlab_obj *obj) {
    matlab_mat *m = mat_alloc(1, 1);
    m->data[0] = obj ? matlab_obj_get_f64(obj, "NumCoeffs", 9) : 0.0;
    return m;
}

/* formula(f) — the model formula as a char row.  Custom carries its Expr;
 * library/poly get a canonical formula string. */
matlab_mat *matlab_curvefit_formula(matlab_obj *obj) {
    if (!obj) return reinterpret_cast<matlab_mat *>(matlab_string_from_literal("", 0));
    int mtype = static_cast<int>(matlab_obj_get_f64(obj, "ModelType", 9));
    std::string f;
    if (mtype == 100) {
        f = cf_sstr(matlab_obj_get_string(obj, "Expr", 4));
    } else if (mtype == 1) {
        int deg = static_cast<int>(matlab_obj_get_f64(obj, "Degree", 6));
        for (int i = 0; i <= deg; ++i) {
            int power = deg - i;
            if (i > 0) f += " + ";
            f += "p" + std::to_string(i + 1);
            if (power >= 2) f += "*x^" + std::to_string(power);
            else if (power == 1) f += "*x";
        }
    } else {
        static const char *FM[9] = { "", "", "a*exp(b*x)", "a*exp(b*x)+c*exp(d*x)",
            "a*x^b", "a*x^b+c", "a*exp(-((x-b)/c)^2)",
            "a*sin(b*x+c)", "a0+a1*cos(w*x)+b1*sin(w*x)" };
        f = (mtype >= 2 && mtype <= 8) ? FM[mtype] : "";
    }
    return reinterpret_cast<matlab_mat *>(
        matlab_string_from_literal(f.c_str(), static_cast<int64_t>(f.size())));
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
    /* Custom equation (id 100): the stored expression + named coefficients. */
    if (mtype == 100) {
        std::string eqn = cf_sstr(matlab_obj_get_string(obj, "Expr", 4));
        std::vector<std::string> nm = cf_extract_coeffs(eqn);
        printf("     General model:\n");
        printf("       f(x) = %s\n", eqn.c_str());
        printf("     Coefficients:\n");
        for (size_t i = 0; i < nm.size() && static_cast<int>(i) < nc; ++i)
            printf("       %s = %.6g\n", nm[i].c_str(), c->data[i]);
        pthread_mutex_unlock(&matlab_io_mutex);
        return mat_alloc(0, 0);
    }
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

/* smooth(y[, span][, method]) — column smoother.  Methods: 'moving' (default,
 * a shrinking symmetric window), 'lowess'/'loess' (local deg-1/2 regression),
 * 'rlowess'/'rloess' (robust), 'sgolay' (Savitzky-Golay via the shipped
 * sgolayfilt).  Index positions 1..n serve as the predictor. */
static matlab_mat *cf_smooth_run(matlab_mat *ymat, int span, const std::string &method) {
    std::vector<double> y = cf_flat(ymat);
    int64_t n = static_cast<int64_t>(y.size());
    if (span < 1) span = 5;
    if ((span % 2) == 0) span += 1;                         /* odd window */
    std::vector<double> out(static_cast<size_t>(n));
    if (method.empty() || method == "moving") {
        int half = span / 2;
        for (int64_t i = 0; i < n; ++i) {
            int64_t w = half;
            if (i < w) w = i;
            if (n - 1 - i < w) w = n - 1 - i;
            double s = 0.0; int64_t cnt = 0;
            for (int64_t j = i - w; j <= i + w; ++j) { s += y[static_cast<size_t>(j)]; ++cnt; }
            out[static_cast<size_t>(i)] = (cnt > 0) ? s / static_cast<double>(cnt) : y[static_cast<size_t>(i)];
        }
    } else if (method == "sgolay") {
        matlab_mat *r = matlab_sgolayfilt(ymat, 2.0, static_cast<double>(span));
        return r ? r : cf_copymat(ymat);
    } else {
        std::vector<double> x(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) x[static_cast<size_t>(i)] = static_cast<double>(i + 1);
        int deg = (method == "loess" || method == "rloess") ? 2 : 1;
        bool rob = (method == "rlowess" || method == "rloess");
        out = cf_lowess(x, y, span, deg, rob);
    }
    matlab_mat *o = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) o->data[i] = out[static_cast<size_t>(i)];
    return o;
}
matlab_mat *matlab_curvefit_smooth1(matlab_mat *y) { return cf_smooth_run(y, 5, ""); }
matlab_mat *matlab_curvefit_smooth2(matlab_mat *y, matlab_mat *span) {
    int s = span ? static_cast<int>(span->data[0]) : 5;
    return cf_smooth_run(y, s, "");
}
matlab_mat *matlab_curvefit_smooth3(matlab_mat *y, matlab_mat *span, void *method) {
    int s = span ? static_cast<int>(span->data[0]) : 5;
    return cf_smooth_run(y, s, cf_sstr(method));
}

/* spline(x,y) / pchip(x,y) — build a cubic ppform.  kind: 0 = not-a-knot
 * cubic spline, 1 = pchip monotone Hermite. */
matlab_mat *matlab_curvefit_spline_init(matlab_obj *obj, matlab_mat *x, matlab_mat *y, double kind) {
    if (!obj) return mat_alloc(0, 0);
    std::vector<double> xv = cf_flat(x), yv = cf_flat(y);
    int n = static_cast<int>(xv.size() < yv.size() ? xv.size() : yv.size());
    xv.resize(static_cast<size_t>(n)); yv.resize(static_cast<size_t>(n));
    int np = (n >= 2) ? n - 1 : 0, order = 4;
    matlab_mat *coefs = mat_alloc(np, order);
    if (static_cast<int>(kind) == 1) {                  /* pchip Hermite */
        std::vector<double> d = cf_pchip_slopes(xv, yv);
        for (int i = 0; i < np; ++i) {
            double h = xv[static_cast<size_t>(i + 1)] - xv[static_cast<size_t>(i)];
            double y0 = yv[static_cast<size_t>(i)], y1 = yv[static_cast<size_t>(i + 1)];
            double m0 = d[static_cast<size_t>(i)], m1 = d[static_cast<size_t>(i + 1)];
            double dl = (y1 - y0) / h;
            coefs->data[i * order + 0] = (m0 + m1 - 2.0 * dl) / (h * h);   /* c3 */
            coefs->data[i * order + 1] = (3.0 * dl - 2.0 * m0 - m1) / h;   /* c2 */
            coefs->data[i * order + 2] = m0;                              /* c1 */
            coefs->data[i * order + 3] = y0;                              /* c0 */
        }
    } else {                                            /* not-a-knot cubic spline */
        std::vector<double> M = cf_spline_M(xv, yv);
        for (int i = 0; i < np; ++i) {
            double h = xv[static_cast<size_t>(i + 1)] - xv[static_cast<size_t>(i)];
            double y0 = yv[static_cast<size_t>(i)], y1 = yv[static_cast<size_t>(i + 1)];
            double Mi = M[static_cast<size_t>(i)], Mj = M[static_cast<size_t>(i + 1)];
            coefs->data[i * order + 0] = (Mj - Mi) / (6.0 * h);                       /* c3 */
            coefs->data[i * order + 1] = Mi / 2.0;                                    /* c2 */
            coefs->data[i * order + 2] = (y1 - y0) / h - h * (2.0 * Mi + Mj) / 6.0;   /* c1 */
            coefs->data[i * order + 3] = y0;                                          /* c0 */
        }
    }
    matlab_obj_set_mat(obj, "Breaks", 6, cf_rowmat(xv));
    matlab_obj_set_mat(obj, "Coefs", 5, coefs);
    matlab_obj_set_f64(obj, "Pieces", 6, static_cast<double>(np));
    matlab_obj_set_f64(obj, "Order", 5, static_cast<double>(order));
    return mat_alloc(0, 0);
}

/* ppmak(breaks, coefs) — build a ppform from explicit parts. */
matlab_mat *matlab_curvefit_ppmak_init(matlab_obj *obj, matlab_mat *breaks, matlab_mat *coefs) {
    if (!obj) return mat_alloc(0, 0);
    std::vector<double> br = cf_flat(breaks);
    int np = static_cast<int>(br.size()) - 1; if (np < 1) np = 1;
    int order = (coefs && np > 0) ? static_cast<int>(coefs->rows * coefs->cols / np) : 1;
    matlab_obj_set_mat(obj, "Breaks", 6, cf_copymat(breaks));
    matlab_obj_set_mat(obj, "Coefs", 5, cf_copymat(coefs));
    matlab_obj_set_f64(obj, "Pieces", 6, static_cast<double>(np));
    matlab_obj_set_f64(obj, "Order", 5, static_cast<double>(order));
    return mat_alloc(0, 0);
}

/* fnval(pp, xx) — evaluate the piecewise polynomial at xx. */
matlab_mat *matlab_curvefit_fnval(matlab_obj *pp, matlab_mat *xx) {
    if (!pp) return mat_alloc(0, 0);
    std::vector<double> br = cf_flat(matlab_obj_get_mat(pp, "Breaks", 6));
    std::vector<double> cf = cf_flat(matlab_obj_get_mat(pp, "Coefs", 5));
    int order = static_cast<int>(matlab_obj_get_f64(pp, "Order", 5));
    int64_t nq = xx ? xx->rows * xx->cols : 0;
    matlab_mat *out = mat_alloc(xx ? xx->rows : 1, xx ? xx->cols : 1);
    for (int64_t q = 0; q < nq; ++q) out->data[q] = cf_pp_eval(br, cf, order, xx->data[q]);
    return out;
}

/* fnder(pp) — differentiate each piece (order drops by one). */
matlab_mat *matlab_curvefit_fnder_init(matlab_obj *obj, matlab_obj *pp) {
    if (!obj || !pp) return mat_alloc(0, 0);
    matlab_mat *br = matlab_obj_get_mat(pp, "Breaks", 6);
    std::vector<double> cf = cf_flat(matlab_obj_get_mat(pp, "Coefs", 5));
    int order = static_cast<int>(matlab_obj_get_f64(pp, "Order", 5));
    int np = static_cast<int>(matlab_obj_get_f64(pp, "Pieces", 6));
    int no = (order > 1) ? order - 1 : 1;
    matlab_mat *nc = mat_alloc(np, no);
    for (int i = 0; i < np; ++i)
        for (int a = 0; a < no; ++a) {                  /* d/dt of c_{a}*t^{order-1-a} */
            double power = static_cast<double>(order - 1 - a);
            nc->data[i * no + a] = (order > 1) ? cf[static_cast<size_t>(i * order + a)] * power : 0.0;
        }
    matlab_obj_set_mat(obj, "Breaks", 6, cf_copymat(br));
    matlab_obj_set_mat(obj, "Coefs", 5, nc);
    matlab_obj_set_f64(obj, "Pieces", 6, static_cast<double>(np));
    matlab_obj_set_f64(obj, "Order", 5, static_cast<double>(no));
    return mat_alloc(0, 0);
}

/* fnint(pp) — antiderivative (order rises by one), continuous across pieces. */
matlab_mat *matlab_curvefit_fnint_init(matlab_obj *obj, matlab_obj *pp) {
    if (!obj || !pp) return mat_alloc(0, 0);
    std::vector<double> br = cf_flat(matlab_obj_get_mat(pp, "Breaks", 6));
    std::vector<double> cf = cf_flat(matlab_obj_get_mat(pp, "Coefs", 5));
    int order = static_cast<int>(matlab_obj_get_f64(pp, "Order", 5));
    int np = static_cast<int>(matlab_obj_get_f64(pp, "Pieces", 6));
    int no = order + 1;
    matlab_mat *nc = mat_alloc(np, no);
    double carry = 0.0;
    for (int i = 0; i < np; ++i) {
        for (int a = 0; a < order; ++a) {               /* integral of c_a*t^{order-1-a} */
            double power = static_cast<double>(order - 1 - a);
            nc->data[i * no + a] = cf[static_cast<size_t>(i * order + a)] / (power + 1.0);
        }
        nc->data[i * no + (no - 1)] = carry;            /* constant = value at left break */
        double h = br[static_cast<size_t>(i + 1)] - br[static_cast<size_t>(i)], v = 0.0;
        for (int a = 0; a < no; ++a) v = v * h + nc->data[i * no + a];   /* value at right break */
        carry = v;
    }
    matlab_obj_set_mat(obj, "Breaks", 6, cf_rowmat(br));
    matlab_obj_set_mat(obj, "Coefs", 5, nc);
    matlab_obj_set_f64(obj, "Pieces", 6, static_cast<double>(np));
    matlab_obj_set_f64(obj, "Order", 5, static_cast<double>(no));
    return mat_alloc(0, 0);
}

/* fnbrk(pp, part) — extract a ppform part ('breaks' / 'coefs' / 'pieces' /
 * 'order'); returns a matrix. */
matlab_mat *matlab_curvefit_fnbrk(matlab_obj *pp, void *partstr) {
    if (!pp) return mat_alloc(0, 0);
    std::string part = cf_sstr(partstr);
    if (part == "breaks") return cf_copymat(matlab_obj_get_mat(pp, "Breaks", 6));
    if (part == "coefs" || part == "coefficients") return cf_copymat(matlab_obj_get_mat(pp, "Coefs", 5));
    matlab_mat *m = mat_alloc(1, 1);
    m->data[0] = (part == "order") ? matlab_obj_get_f64(pp, "Order", 5)
                                   : matlab_obj_get_f64(pp, "Pieces", 6);
    return m;
}

/* fit([x y], z, 'polyNM') — bivariate polynomial surface → sfit. */
matlab_mat *matlab_curvefit_fit_surface(matlab_obj *obj, matlab_mat *xy, matlab_mat *z,
                                        void *tagstr) {
    if (!obj) return mat_alloc(0, 0);
    std::string tag = cf_sstr(tagstr);
    int N = 1, M = 1;
    if (tag.size() == 6 && tag.compare(0, 4, "poly") == 0) { N = tag[4] - '0'; M = tag[5] - '0'; }
    /* split the predictor columns (xy is npts×2, row-major). */
    int64_t npts = (xy && xy->cols == 2) ? xy->rows : (xy ? xy->rows * xy->cols / 2 : 0);
    std::vector<double> xs(static_cast<size_t>(npts)), ys(static_cast<size_t>(npts));
    for (int64_t p = 0; p < npts; ++p) {
        xs[static_cast<size_t>(p)] = xy->data[p * 2 + 0];
        ys[static_cast<size_t>(p)] = xy->data[p * 2 + 1];
    }
    std::vector<double> zz = cf_flat(z);
    std::vector<std::pair<int,int>> terms = cf_surf_terms(N, M);
    int k = static_cast<int>(terms.size());
    /* normal equations DᵀD c = Dᵀz, accumulated row-by-row. */
    std::vector<double> A(static_cast<size_t>(k * k), 0.0), b(static_cast<size_t>(k), 0.0);
    std::vector<double> row(static_cast<size_t>(k));
    for (int64_t p = 0; p < npts; ++p) {
        for (int a = 0; a < k; ++a)
            row[static_cast<size_t>(a)] = pow(xs[static_cast<size_t>(p)], terms[static_cast<size_t>(a)].first) *
                                          pow(ys[static_cast<size_t>(p)], terms[static_cast<size_t>(a)].second);
        for (int a = 0; a < k; ++a) {
            b[static_cast<size_t>(a)] += row[static_cast<size_t>(a)] * zz[static_cast<size_t>(p)];
            for (int c = 0; c < k; ++c)
                A[static_cast<size_t>(a * k + c)] += row[static_cast<size_t>(a)] * row[static_cast<size_t>(c)];
        }
    }
    cf_solve(A, b, k);                                  /* b now holds coeffs */
    matlab_mat *coeffs = mat_alloc(1, k);
    for (int a = 0; a < k; ++a) coeffs->data[a] = b[static_cast<size_t>(a)];
    /* goodness of fit. */
    double zbar = cf_mean(zz), sse = 0.0, sst = 0.0;
    for (int64_t p = 0; p < npts; ++p) {
        double zh = 0.0;
        for (int a = 0; a < k; ++a)
            zh += b[static_cast<size_t>(a)] * pow(xs[static_cast<size_t>(p)], terms[static_cast<size_t>(a)].first) *
                  pow(ys[static_cast<size_t>(p)], terms[static_cast<size_t>(a)].second);
        double e = zz[static_cast<size_t>(p)] - zh;
        sse += e * e; sst += (zz[static_cast<size_t>(p)] - zbar) * (zz[static_cast<size_t>(p)] - zbar);
    }
    double dfe = static_cast<double>(npts - k);
    double r2 = (sst > 0.0) ? 1.0 - sse / sst : (sse == 0.0 ? 1.0 : 0.0);
    matlab_obj_set_f64(obj, "ModelType", 9, 300.0);
    matlab_obj_set_f64(obj, "DegN", 4, static_cast<double>(N));
    matlab_obj_set_f64(obj, "DegM", 4, static_cast<double>(M));
    matlab_obj_set_mat(obj, "Coeffs", 6, coeffs);
    matlab_obj_set_f64(obj, "NumObs", 6, static_cast<double>(npts));
    matlab_obj_set_f64(obj, "NumCoeffs", 9, static_cast<double>(k));
    matlab_obj_set_f64(obj, "SSE", 3, sse);
    matlab_obj_set_f64(obj, "Rsquare", 7, r2);
    matlab_obj_set_f64(obj, "DFE", 3, dfe);
    matlab_obj_set_f64(obj, "AdjRsquare", 10,
        (sst > 0.0 && dfe > 0.0) ? 1.0 - (sse / dfe) / (sst / static_cast<double>(npts - 1)) : r2);
    matlab_obj_set_f64(obj, "RMSE", 4, (dfe > 0.0) ? sqrt(sse / dfe) : 0.0);
    return mat_alloc(0, 0);
}

/* feval(sf, xq, yq) / sf(xq, yq) — evaluate the fitted surface. */
matlab_mat *matlab_curvefit_sfeval(matlab_obj *obj, matlab_mat *xq, matlab_mat *yq) {
    if (!obj) return mat_alloc(0, 0);
    int N = static_cast<int>(matlab_obj_get_f64(obj, "DegN", 4));
    int M = static_cast<int>(matlab_obj_get_f64(obj, "DegM", 4));
    std::vector<double> c = cf_obj_coeffs(obj);
    std::vector<std::pair<int,int>> terms = cf_surf_terms(N, M);
    int k = static_cast<int>(terms.size());
    int64_t nq = xq ? xq->rows * xq->cols : 0;
    matlab_mat *out = mat_alloc(xq ? xq->rows : 1, xq ? xq->cols : 1);
    for (int64_t q = 0; q < nq; ++q) {
        double xv = xq->data[q], yv = (yq && q < yq->rows * yq->cols) ? yq->data[q] : 0.0, s = 0.0;
        for (int a = 0; a < k && a < static_cast<int>(c.size()); ++a)
            s += c[static_cast<size_t>(a)] * pow(xv, terms[static_cast<size_t>(a)].first) *
                 pow(yv, terms[static_cast<size_t>(a)].second);
        out->data[q] = s;
    }
    return out;
}

/* csaps(x, y, p, xx) — cubic smoothing spline evaluated at xx. */
matlab_mat *matlab_curvefit_csaps(matlab_mat *x, matlab_mat *y, matlab_mat *p, matlab_mat *xx) {
    std::vector<double> xv = cf_flat(x), yv = cf_flat(y);
    int64_t n = static_cast<int64_t>(xv.size() < yv.size() ? xv.size() : yv.size());
    xv.resize(static_cast<size_t>(n)); yv.resize(static_cast<size_t>(n));
    double pp = p ? p->data[0] : 0.9;
    std::vector<double> g = cf_smooth_spline(xv, yv, pp);
    return cf_interp(4, xv, g, xx);
}

}  /* extern "C" */
