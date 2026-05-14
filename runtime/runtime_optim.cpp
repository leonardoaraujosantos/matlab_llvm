/* runtime_optim.cpp — Optimization Toolbox runtime.
 *
 * See docs/optim_toolbox_roadmap.md for the full surface.  This file
 * provides the function-form numerical core that the MATLAB-side
 * solvers (`fzero`, `fminbnd`, `fminunc`, `fmincon`, `linprog`, …)
 * resolve to.  The problem-based classdef API (`optimvar`,
 * `optimproblem`, `solve`) is layered on later (Tier-4); the runtime
 * entries here are the stable ABI the MLIR lowering targets.
 *
 * Tier-1 (smallest end-to-end Optim loop) — all single-return:
 *   matlab_optim_fzero       — scalar root, Brent + bracket-expansion
 *   matlab_optim_fzero_iv    — scalar root, bracket given as 2-vector
 *   matlab_optim_fminbnd     — 1-D minimiser, Brent (golden + parabolic)
 *   matlab_optim_fminsearch  — N-D minimiser, Nelder–Mead simplex
 *   matlab_optim_fminunc     — N-D minimiser, BFGS + FD gradient
 *   matlab_optim_linprog     — LP, dense 2-phase simplex (7-arg form)
 *   matlab_optim_linprog3    — LP, 3-arg convenience form
 *   matlab_optim_lsqnonneg   — non-negative least squares (Lawson–Hanson)
 *   matlab_optim_fsolve_scalar — scalar equation, Newton + Brent fallback
 *
 * Tier-2 (constrained + nonlinear least squares) — two shared cores:
 *   matlab_optim_fmincon     — general constrained NL, augmented Lagrangian
 *   matlab_optim_quadprog    — convex QP via the augmented-Lagrangian core
 *   matlab_optim_lsqlin      — constrained linear LS via the same core
 *   matlab_optim_lsqnonlin   — nonlinear LS, Levenberg-Marquardt
 *   matlab_optim_lsqcurvefit — curve fitting, LM over fun(x,xdata) − ydata
 *   matlab_optim_fsolve      — N-D nonlinear system, LM on ‖F(x)‖²
 *
 * Tier-3 (MILP, cone, minimax, semi-infinite) — all reformulations on
 * top of the Tier-1/2 cores:
 *   matlab_optim_intlinprog  — mixed-integer LP, branch-and-bound
 *   matlab_optim_fminimax    — minimax, epigraph reformulation
 *   matlab_optim_fgoalattain — goal attainment, epigraph reformulation
 *   matlab_optim_coneprog    — second-order cone (single cone), SOCP
 *   matlab_optim_fseminf     — semi-infinite, outer-approximation grid
 *
 * Function handles reach the runtime through the standard `void *`
 * cast-to-typed-pointer ABI.  Handle shapes used:
 *   scalar objective  : double (*)(double)
 *   vector objective  : double (*)(matlab_mat *)              (fminunc, fmincon obj)
 *   vector→vector     : matlab_mat *(*)(matlab_mat *)         (nonlcon, lsqnonlin, fsolve)
 *   two-arg vector    : matlab_mat *(*)(matlab_mat *, matlab_mat *)  (lsqcurvefit model)
 *
 * No external dependencies — every algorithm is hand-coded, matching
 * the project's LAPACK-style precedent.  Tier-1/2 use solver defaults;
 * the `[x,fval,exitflag,output]` multi-return surface and the
 * `optimoptions` / `optimset` option objects are deferred (the runtime
 * accepts plain MATLAB structs for options, the odeset precedent).
 */

#include "runtime_internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {

/* User function-handle ABIs. */
typedef double (*matlab_optim_scalar_fn)(double);
typedef double (*matlab_optim_vector_fn)(matlab_mat *);

/* ===================================================================
 * Shared helpers
 * =================================================================== */

/* Treat a null / 0×0 / empty matrix as "argument absent" — the MATLAB
 * convention of passing `[]` for an optional solver argument. */
static int mat_absent(const matlab_mat *m) {
    return !m || m->rows * m->cols == 0;
}

/* Number of elements, robust to null. */
static int64_t mat_numel(const matlab_mat *m) {
    return m ? m->rows * m->cols : 0;
}

/* Solve A x = b for a dense n×n system via Gaussian elimination with
 * partial pivoting.  A and b are overwritten; the solution is written
 * back into b.  Returns 0 on success, 1 if A is singular.            */
static int solve_dense_gepp(double *A, double *b, int n) {
    for (int k = 0; k < n; ++k) {
        /* Pivot: largest |A[i][k]| for i >= k. */
        int piv = k;
        double best = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double v = fabs(A[i * n + k]);
            if (v > best) { best = v; piv = i; }
        }
        if (best < 1e-300) return 1;  /* singular */
        if (piv != k) {
            for (int j = 0; j < n; ++j) {
                double t = A[k * n + j]; A[k * n + j] = A[piv * n + j];
                A[piv * n + j] = t;
            }
            double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        double akk = A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i * n + k] / akk;
            if (factor == 0.0) continue;
            for (int j = k; j < n; ++j) A[i * n + j] -= factor * A[k * n + j];
            b[i] -= factor * b[k];
        }
    }
    /* Back substitution. */
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
        b[i] = s / A[i * n + i];
    }
    return 0;
}

/* ===================================================================
 * fzero — scalar root finding (Brent's method)
 * =================================================================== */

/* van Wijngaarden–Dekker–Brent (1973).  Given [a, b] with
 * f(a)·f(b) ≤ 0, drives b → root via a mix of inverse-quadratic
 * interpolation, secant, and bisection, with safeguards that fall
 * back to bisection when a candidate step would be worse than half
 * the bracket.  Returns the best estimate when |f(b)| < tol_f or
 * |b−a| < tol_x; on max-iter returns the current best b.            */
static double brent_root(matlab_optim_scalar_fn f,
                         double a, double fa,
                         double b, double fb,
                         double tol_x, double tol_f, int max_iter) {
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fabs(fa) < fabs(fb)) {
        double t = a;  a  = b;  b  = t;
        t = fa;       fa = fb; fb = t;
    }
    double c  = a;
    double fc = fa;
    double d  = b - a;
    int mflag = 1;

    for (int it = 0; it < max_iter; ++it) {
        if (fabs(fb) <= tol_f) return b;
        if (fabs(b - a) <= tol_x + 4.0 * DBL_EPSILON * fabs(b)) return b;

        double s;
        if (fa != fc && fb != fc) {
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }

        double lo = (3.0 * a + b) * 0.25;
        double hi = b;
        if (lo > hi) { double t = lo; lo = hi; hi = t; }
        double bc = fabs(b - c);
        double cd = fabs(c - d);
        int use_bisect =
            (s < lo) || (s > hi) ||
            (mflag  && fabs(s - b) >= 0.5 * bc) ||
            (!mflag && fabs(s - b) >= 0.5 * cd) ||
            (mflag  && bc < tol_x) ||
            (!mflag && cd < tol_x);
        if (use_bisect) { s = 0.5 * (a + b); mflag = 1; }
        else            { mflag = 0; }

        double fs = f(s);
        d  = c;
        c  = b;
        fc = fb;
        if (fa * fs < 0.0) { b = s; fb = fs; }
        else               { a = s; fa = fs; }
        if (fabs(fa) < fabs(fb)) {
            double t = a;  a  = b;  b  = t;
            t = fa;       fa = fb; fb = t;
        }
    }
    return b;
}

/* Probe x0±dx with dx growing geometrically (×√2) until f changes
 * sign, mirroring MATLAB's scalar-guess fzero behaviour.  Returns 0
 * on success (bracket in out_*), 1 on failure.                       */
static int expand_bracket(matlab_optim_scalar_fn f, double x0,
                          double *out_a, double *out_fa,
                          double *out_b, double *out_fb) {
    double f0 = f(x0);
    if (f0 == 0.0) {
        *out_a = *out_b = x0;
        *out_fa = *out_fb = 0.0;
        return 0;
    }
    double dx = fabs(x0) * 0.02;
    if (dx < 1.0e-4) dx = 1.0e-4;
    for (int it = 0; it < 50; ++it) {
        double a = x0 - dx, fa = f(a);
        if (fa == 0.0) { *out_a = *out_b = a; *out_fa = *out_fb = 0.0; return 0; }
        if (fa * f0 < 0.0) { *out_a = a; *out_fa = fa; *out_b = x0; *out_fb = f0; return 0; }
        double b = x0 + dx, fb = f(b);
        if (fb == 0.0) { *out_a = *out_b = b; *out_fa = *out_fb = 0.0; return 0; }
        if (fb * f0 < 0.0) { *out_a = x0; *out_fa = f0; *out_b = b; *out_fb = fb; return 0; }
        dx *= 1.41421356237309504880;
    }
    return 1;
}

/* `x = fzero(@fn, x0)` — scalar initial-guess form. */
double matlab_optim_fzero(void *fn_p, double x0) {
    if (!fn_p) return (double)NAN;
    matlab_optim_scalar_fn f = (matlab_optim_scalar_fn)fn_p;
    double a, fa, b, fb;
    if (expand_bracket(f, x0, &a, &fa, &b, &fb) != 0) return (double)NAN;
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    return brent_root(f, a, fa, b, fb, 1.0e-12, 1.0e-14, 100);
}

/* `x = fzero(@fn, [a b])` — bracket form. */
double matlab_optim_fzero_iv(void *fn_p, matlab_mat *iv) {
    if (!fn_p || !iv) return (double)NAN;
    int64_t n = iv->rows * iv->cols;
    if (n < 2) return (double)NAN;
    matlab_optim_scalar_fn f = (matlab_optim_scalar_fn)fn_p;
    double a  = iv->data[0];
    double b  = iv->data[1];
    double fa = f(a);
    double fb = f(b);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa * fb > 0.0) return (double)NAN;
    return brent_root(f, a, fa, b, fb, 1.0e-12, 1.0e-14, 100);
}

/* ===================================================================
 * fminbnd — 1-D minimisation (Brent: golden section + parabolic)
 * =================================================================== */

/* Brent's `localmin` (1973): combines golden-section search with
 * parabolic interpolation through the three best points.  Parabolic
 * steps are taken only when they fall inside the bracket and shrink
 * it by at least half the second-to-last step; otherwise the golden
 * step is used.  Converges super-linearly on smooth unimodal f.      */
double matlab_optim_fminbnd(void *fn_p, double lo, double hi) {
    if (!fn_p) return (double)NAN;
    matlab_optim_scalar_fn f = (matlab_optim_scalar_fn)fn_p;
    if (lo > hi) { double t = lo; lo = hi; hi = t; }

    const double C = 0.5 * (3.0 - sqrt(5.0));  /* golden ratio ≈ 0.381966 */
    double a = lo, b = hi;
    double x = a + C * (b - a);
    double w = x, v = x;
    double fx = f(x), fw = fx, fv = fx;
    double d = 0.0, e = 0.0;
    const double tol = 1.0e-10;
    const int max_iter = 500;

    for (int it = 0; it < max_iter; ++it) {
        double m = 0.5 * (a + b);
        double tol1 = tol * fabs(x) + 1.0e-12;
        double tol2 = 2.0 * tol1;
        if (fabs(x - m) <= tol2 - 0.5 * (b - a)) break;

        int use_golden = 1;
        if (fabs(e) > tol1) {
            /* Fit a parabola through (x,fx), (w,fw), (v,fv). */
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = fabs(q);
            double etemp = e;
            e = d;
            if (fabs(p) < fabs(0.5 * q * etemp) &&
                p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                double u = x + d;
                if (u - a < tol2 || b - u < tol2)
                    d = (m > x) ? tol1 : -tol1;
                use_golden = 0;
            }
        }
        if (use_golden) {
            e = (x >= m) ? (a - x) : (b - x);
            d = C * e;
        }
        double u = (fabs(d) >= tol1) ? (x + d)
                                     : (x + ((d > 0.0) ? tol1 : -tol1));
        double fu = f(u);
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w;  fv = fw;
            w = x;  fw = fx;
            x = u;  fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w;  fv = fw;
                w = u;  fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;  fv = fu;
            }
        }
    }
    return x;
}

/* ===================================================================
 * fminsearch — N-D minimisation (Nelder–Mead simplex)
 * =================================================================== */

/* Evaluate a vector objective at a std::vector point. */
static double nm_eval(matlab_optim_vector_fn f, const std::vector<double> &v) {
    int n = (int)v.size();
    matlab_mat *m = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) m->data[i] = v[i];
    double r = f(m);
    free(m->data);
    free(m);
    return r;
}

/* Nelder–Mead (1965) downhill simplex with the standard
 * reflection/expansion/contraction/shrink coefficients.  The initial
 * simplex offsets each coordinate of x0 by 5 % (or 0.00025 if the
 * coordinate is zero), matching MATLAB's fminsearch construction.    */
matlab_mat *matlab_optim_fminsearch(void *fn_p, matlab_mat *x0) {
    if (!fn_p || !x0) return mat_alloc(0, 0);
    matlab_optim_vector_fn f = (matlab_optim_vector_fn)fn_p;
    int n = (int)(x0->rows * x0->cols);
    if (n < 1) return mat_alloc(0, 0);

    /* Simplex: n+1 vertices. */
    std::vector<std::vector<double>> S(n + 1, std::vector<double>(n));
    std::vector<double> fS(n + 1);
    for (int i = 0; i < n; ++i) S[0][i] = x0->data[i];
    fS[0] = nm_eval(f, S[0]);
    for (int j = 1; j <= n; ++j) {
        S[j] = S[0];
        double xj = S[0][j - 1];
        S[j][j - 1] = (xj != 0.0) ? (xj * 1.05) : 0.00025;
        fS[j] = nm_eval(f, S[j]);
    }

    const double rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5;
    const double tol_x = 1.0e-8, tol_f = 1.0e-8;
    const int max_iter = 200 * n + 200;

    std::vector<int> ord(n + 1);
    std::vector<double> c(n), xr(n), xe(n), xc(n);

    for (int it = 0; it < max_iter; ++it) {
        /* Sort vertices best → worst by objective value. */
        for (int i = 0; i <= n; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(),
                  [&](int p, int q) { return fS[p] < fS[q]; });
        {
            std::vector<std::vector<double>> Snew(n + 1);
            std::vector<double> fSnew(n + 1);
            for (int i = 0; i <= n; ++i) { Snew[i] = S[ord[i]]; fSnew[i] = fS[ord[i]]; }
            S.swap(Snew);
            fS.swap(fSnew);
        }

        /* Convergence: simplex small in both f and x. */
        double fspread = fabs(fS[n] - fS[0]);
        double xspread = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 1; j <= n; ++j)
                xspread = std::max(xspread, fabs(S[j][i] - S[0][i]));
        if (fspread <= tol_f && xspread <= tol_x) break;

        /* Centroid of the best n vertices (all but the worst). */
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += S[j][i];
            c[i] = s / n;
        }

        /* Reflection. */
        for (int i = 0; i < n; ++i) xr[i] = c[i] + rho * (c[i] - S[n][i]);
        double fr = nm_eval(f, xr);

        if (fr < fS[0]) {
            /* Expansion. */
            for (int i = 0; i < n; ++i) xe[i] = c[i] + chi * (xr[i] - c[i]);
            double fe = nm_eval(f, xe);
            if (fe < fr) { S[n] = xe; fS[n] = fe; }
            else         { S[n] = xr; fS[n] = fr; }
        } else if (fr < fS[n - 1]) {
            /* Accept reflection. */
            S[n] = xr; fS[n] = fr;
        } else {
            /* Contraction. */
            int shrink = 0;
            if (fr < fS[n]) {
                /* Outside contraction. */
                for (int i = 0; i < n; ++i) xc[i] = c[i] + psi * (xr[i] - c[i]);
                double fc = nm_eval(f, xc);
                if (fc <= fr) { S[n] = xc; fS[n] = fc; }
                else          { shrink = 1; }
            } else {
                /* Inside contraction. */
                for (int i = 0; i < n; ++i) xc[i] = c[i] - psi * (c[i] - S[n][i]);
                double fc = nm_eval(f, xc);
                if (fc < fS[n]) { S[n] = xc; fS[n] = fc; }
                else            { shrink = 1; }
            }
            if (shrink) {
                for (int j = 1; j <= n; ++j) {
                    for (int i = 0; i < n; ++i)
                        S[j][i] = S[0][i] + sigma * (S[j][i] - S[0][i]);
                    fS[j] = nm_eval(f, S[j]);
                }
            }
        }
    }

    /* Return the best vertex. */
    int best = 0;
    for (int i = 1; i <= n; ++i) if (fS[i] < fS[best]) best = i;
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = S[best][i];
    return out;
}

/* ===================================================================
 * fminunc — N-D unconstrained minimisation (BFGS quasi-Newton)
 * =================================================================== */

/* BFGS with a finite-difference gradient and a backtracking Armijo
 * line search.  The inverse Hessian H is maintained directly (no
 * linear solve per step) and updated by the rank-2 BFGS formula
 *   H⁺ = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ,   ρ = 1/(sᵀy).
 * The search direction is p = −H g; if it is not a descent direction
 * (curvature lost to FD noise) H resets to the identity.            */
matlab_mat *matlab_optim_fminunc(void *fn_p, matlab_mat *x0) {
    if (!fn_p || !x0) return mat_alloc(0, 0);
    matlab_optim_vector_fn f = (matlab_optim_vector_fn)fn_p;
    int n = (int)(x0->rows * x0->cols);
    if (n < 1) return mat_alloc(0, 0);

    std::vector<double> x(n), g(n), gnew(n), p(n), xnew(n), s(n), y(n), Hy(n);
    std::vector<double> H((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) x[i] = x0->data[i];
    for (int i = 0; i < n; ++i) H[(size_t)i * n + i] = 1.0;

    auto eval = [&](const std::vector<double> &v) -> double {
        return nm_eval(f, v);
    };
    /* Forward finite-difference gradient. */
    auto grad = [&](const std::vector<double> &v, double fv,
                    std::vector<double> &gout) {
        std::vector<double> vp = v;
        for (int i = 0; i < n; ++i) {
            double h = 1.0e-7 * (fabs(v[i]) + 1.0);
            vp[i] = v[i] + h;
            double fp = eval(vp);
            gout[i] = (fp - fv) / h;
            vp[i] = v[i];
        }
    };

    double fx = eval(x);
    grad(x, fx, g);

    const double tol_g = 1.0e-6, tol_x = 1.0e-12;
    const int max_iter = 400;

    for (int it = 0; it < max_iter; ++it) {
        double gn = 0.0;
        for (int i = 0; i < n; ++i) gn = std::max(gn, fabs(g[i]));
        if (gn < tol_g) break;

        /* Search direction p = −H g. */
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) sum += H[(size_t)i * n + j] * g[j];
            p[i] = -sum;
        }
        double slope = 0.0;
        for (int i = 0; i < n; ++i) slope += g[i] * p[i];
        if (slope >= 0.0) {
            /* Not a descent direction — reset to steepest descent. */
            for (size_t k = 0; k < H.size(); ++k) H[k] = 0.0;
            for (int i = 0; i < n; ++i) H[(size_t)i * n + i] = 1.0;
            for (int i = 0; i < n; ++i) p[i] = -g[i];
            slope = 0.0;
            for (int i = 0; i < n; ++i) slope += g[i] * p[i];
        }

        /* Backtracking Armijo line search. */
        double alpha = 1.0;
        const double c1 = 1.0e-4;
        double fnew = fx;
        int ls_ok = 0;
        for (int ls = 0; ls < 60; ++ls) {
            for (int i = 0; i < n; ++i) xnew[i] = x[i] + alpha * p[i];
            fnew = eval(xnew);
            if (fnew <= fx + c1 * alpha * slope) { ls_ok = 1; break; }
            alpha *= 0.5;
        }
        if (!ls_ok) break;  /* cannot make further progress */

        grad(xnew, fnew, gnew);
        for (int i = 0; i < n; ++i) { s[i] = xnew[i] - x[i]; y[i] = gnew[i] - g[i]; }
        double sy = 0.0;
        for (int i = 0; i < n; ++i) sy += s[i] * y[i];
        if (sy > 1.0e-12) {
            double rho = 1.0 / sy;
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) sum += H[(size_t)i * n + j] * y[j];
                Hy[i] = sum;
            }
            double yHy = 0.0;
            for (int i = 0; i < n; ++i) yHy += y[i] * Hy[i];
            double coef = rho * (rho * yHy + 1.0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    H[(size_t)i * n + j] +=
                        -rho * (s[i] * Hy[j] + Hy[i] * s[j])
                        + coef * s[i] * s[j];
        }

        double sn = 0.0;
        for (int i = 0; i < n; ++i) sn = std::max(sn, fabs(s[i]));
        x.swap(xnew);
        fx = fnew;
        g.swap(gnew);
        if (sn < tol_x) break;
    }

    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = x[i];
    return out;
}

/* ===================================================================
 * linprog — linear programming (dense 2-phase simplex)
 * =================================================================== */

/* Solve  min fᵀx  s.t.  A x ≤ b,  Aeq x = beq,  lb ≤ x ≤ ub.
 *
 * Tier-1 contract: lower bounds default to 0 when `lb` is absent and
 * must be finite when supplied (the variable is shifted x' = x − lb so
 * x' ≥ 0).  Finite upper bounds become extra `x' ≤ ub − lb` rows; Inf
 * entries of ub are ignored.  The shifted problem is put in standard
 * form (equalities + non-negative vars), and a textbook dense 2-phase
 * tableau simplex is run: phase 1 minimises the sum of artificials to
 * find a feasible basis, phase 2 minimises fᵀx'.  Returns a 0×0 matrix
 * when the problem is infeasible or unbounded.
 *
 * Adequate for the dense ≤ ~200-variable problems Tier-1 targets; the
 * sparse interior-point path is Tier-2 (see the roadmap).            */
static matlab_mat *linprog_core(matlab_mat *f, matlab_mat *A, matlab_mat *b,
                                matlab_mat *Aeq, matlab_mat *beq,
                                matlab_mat *lb, matlab_mat *ub) {
    if (mat_absent(f)) return mat_alloc(0, 0);
    int n = (int)mat_numel(f);

    /* Lower bounds (default 0). */
    std::vector<double> L(n, 0.0);
    if (!mat_absent(lb)) {
        if ((int)mat_numel(lb) != n) return mat_alloc(0, 0);
        for (int i = 0; i < n; ++i) {
            L[i] = lb->data[i];
            if (!isfinite(L[i])) return mat_alloc(0, 0);  /* Tier-1: finite lb */
        }
    }

    /* Build constraint rows of the shifted problem (variable x' = x − L).
     * Each row carries: coefficient vector, rhs, kind.
     *   kind 0 : a·x' ≤ rhs           (slack)
     *   kind 1 : a·x' ≥ rhs  (rhs≥0)  (surplus + artificial)
     *   kind 2 : a·x' = rhs           (artificial)
     * RHS is normalised non-negative; a ≤-row with negative rhs flips
     * to a ≥-row.                                                    */
    struct Row { std::vector<double> a; double rhs; int kind; };
    std::vector<Row> rows;

    auto add_le = [&](const double *coef, double rhs) {
        Row r;
        r.a.assign(coef, coef + n);
        r.rhs = rhs;
        r.kind = 0;
        if (r.rhs < 0.0) {
            for (int i = 0; i < n; ++i) r.a[i] = -r.a[i];
            r.rhs = -r.rhs;
            r.kind = 1;
        }
        rows.push_back(std::move(r));
    };
    auto add_eq = [&](const double *coef, double rhs) {
        Row r;
        r.a.assign(coef, coef + n);
        r.rhs = rhs;
        r.kind = 2;
        if (r.rhs < 0.0) {
            for (int i = 0; i < n; ++i) r.a[i] = -r.a[i];
            r.rhs = -r.rhs;
        }
        rows.push_back(std::move(r));
    };

    /* A x ≤ b   →   A x' ≤ b − A L. */
    if (!mat_absent(A)) {
        int m1 = (int)A->rows;
        if ((int)A->cols != n || (int)mat_numel(b) != m1) return mat_alloc(0, 0);
        std::vector<double> coef(n);
        for (int i = 0; i < m1; ++i) {
            double shift = 0.0;
            for (int j = 0; j < n; ++j) {
                coef[j] = A->data[(size_t)i * n + j];
                shift += coef[j] * L[j];
            }
            add_le(coef.data(), b->data[i] - shift);
        }
    }
    /* Aeq x = beq   →   Aeq x' = beq − Aeq L. */
    if (!mat_absent(Aeq)) {
        int m2 = (int)Aeq->rows;
        if ((int)Aeq->cols != n || (int)mat_numel(beq) != m2) return mat_alloc(0, 0);
        std::vector<double> coef(n);
        for (int i = 0; i < m2; ++i) {
            double shift = 0.0;
            for (int j = 0; j < n; ++j) {
                coef[j] = Aeq->data[(size_t)i * n + j];
                shift += coef[j] * L[j];
            }
            add_eq(coef.data(), beq->data[i] - shift);
        }
    }
    /* Finite ub   →   x' ≤ ub − L. */
    if (!mat_absent(ub)) {
        if ((int)mat_numel(ub) != n) return mat_alloc(0, 0);
        std::vector<double> coef(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double u = ub->data[i];
            if (!isfinite(u)) continue;
            coef[i] = 1.0;
            add_le(coef.data(), u - L[i]);
            coef[i] = 0.0;
        }
    }

    int R = (int)rows.size();
    /* Column layout: [n structural][slacks/surplus][artificials][RHS]. */
    int nSlack = 0, nArtif = 0;
    for (auto &r : rows) {
        if (r.kind == 0) nSlack += 1;
        else if (r.kind == 1) { nSlack += 1; nArtif += 1; }
        else nArtif += 1;
    }
    int W = n + nSlack + nArtif;  /* columns excluding RHS */
    int artBase = n + nSlack;

    std::vector<double> T((size_t)R * (W + 1), 0.0);
    std::vector<int> basis(R, -1);
    int sCol = n, aCol = artBase;
    for (int i = 0; i < R; ++i) {
        Row &r = rows[i];
        double *Trow = &T[(size_t)i * (W + 1)];
        for (int j = 0; j < n; ++j) Trow[j] = r.a[j];
        Trow[W] = r.rhs;
        if (r.kind == 0) {
            Trow[sCol] = 1.0;
            basis[i] = sCol;
            sCol += 1;
        } else if (r.kind == 1) {
            Trow[sCol] = -1.0;        /* surplus */
            Trow[aCol] = 1.0;         /* artificial */
            basis[i] = aCol;
            sCol += 1;
            aCol += 1;
        } else {
            Trow[aCol] = 1.0;
            basis[i] = aCol;
            aCol += 1;
        }
    }

    const double EPS = 1.0e-9;
    const int MAX_PIVOTS = 20000;

    /* One simplex phase: cost[j] is the objective coefficient of column
     * j; pivots until no column has a negative reduced cost.          */
    auto run_phase = [&](const std::vector<double> &cost) -> int {
        for (int iter = 0; iter < MAX_PIVOTS; ++iter) {
            /* Reduced costs rc[j] = cost[j] − cost_B · T[:,j]. */
            int enter = -1;
            double best = -EPS;
            for (int j = 0; j < W; ++j) {
                double z = 0.0;
                for (int i = 0; i < R; ++i)
                    z += cost[basis[i]] * T[(size_t)i * (W + 1) + j];
                double rc = cost[j] - z;
                if (rc < best) { best = rc; enter = j; }
            }
            if (enter < 0) return 0;  /* optimal */

            /* Ratio test. */
            int leave = -1;
            double minRatio = 0.0;
            for (int i = 0; i < R; ++i) {
                double aij = T[(size_t)i * (W + 1) + enter];
                if (aij > EPS) {
                    double ratio = T[(size_t)i * (W + 1) + W] / aij;
                    if (leave < 0 || ratio < minRatio - 1e-12) {
                        minRatio = ratio;
                        leave = i;
                    }
                }
            }
            if (leave < 0) return 1;  /* unbounded */

            /* Pivot on (leave, enter). */
            double piv = T[(size_t)leave * (W + 1) + enter];
            double *Lrow = &T[(size_t)leave * (W + 1)];
            for (int j = 0; j <= W; ++j) Lrow[j] /= piv;
            for (int i = 0; i < R; ++i) {
                if (i == leave) continue;
                double factor = T[(size_t)i * (W + 1) + enter];
                if (factor == 0.0) continue;
                double *Trow = &T[(size_t)i * (W + 1)];
                for (int j = 0; j <= W; ++j) Trow[j] -= factor * Lrow[j];
            }
            basis[leave] = enter;
        }
        return 2;  /* iteration cap hit */
    };

    /* Phase 1: minimise sum of artificials. */
    if (nArtif > 0) {
        std::vector<double> cost1(W, 0.0);
        for (int j = artBase; j < W; ++j) cost1[j] = 1.0;
        int st = run_phase(cost1);
        if (st != 0) return mat_alloc(0, 0);
        double infeas = 0.0;
        for (int i = 0; i < R; ++i)
            if (basis[i] >= artBase)
                infeas += fabs(T[(size_t)i * (W + 1) + W]);
        if (infeas > 1.0e-7) return mat_alloc(0, 0);  /* infeasible */
    }

    /* Phase 2: minimise fᵀx'.  Artificial columns are pinned out by
     * giving them a prohibitive cost. */
    std::vector<double> cost2(W, 0.0);
    for (int j = 0; j < n; ++j) cost2[j] = f->data[j];
    for (int j = artBase; j < W; ++j) cost2[j] = 1.0e15;
    int st2 = run_phase(cost2);
    if (st2 != 0) return mat_alloc(0, 0);

    /* Recover x' from the basic variables, then un-shift x = x' + L. */
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < R; ++i) {
        int bv = basis[i];
        if (bv >= 0 && bv < n)
            out->data[bv] = T[(size_t)i * (W + 1) + W];
    }
    for (int j = 0; j < n; ++j) out->data[j] += L[j];
    return out;
}

/* `x = linprog(f, A, b, Aeq, beq, lb, ub)` — full 7-argument form. */
matlab_mat *matlab_optim_linprog(matlab_mat *f, matlab_mat *A, matlab_mat *b,
                                 matlab_mat *Aeq, matlab_mat *beq,
                                 matlab_mat *lb, matlab_mat *ub) {
    return linprog_core(f, A, b, Aeq, beq, lb, ub);
}

/* `x = linprog(f, A, b)` — 3-argument convenience form (default
 * lower bound 0, no equalities, no upper bounds). */
matlab_mat *matlab_optim_linprog3(matlab_mat *f, matlab_mat *A, matlab_mat *b) {
    return linprog_core(f, A, b, NULL, NULL, NULL, NULL);
}

/* ===================================================================
 * lsqnonneg — non-negative least squares (Lawson–Hanson)
 * =================================================================== */

/* Solve  min ‖C x − d‖²  s.t.  x ≥ 0  via the Lawson–Hanson (1974)
 * active-set algorithm.  Columns move between the active set Z (pinned
 * at zero) and the passive set P (free); each major step solves the
 * unconstrained least-squares sub-problem on P through the normal
 * equations, and an inner loop reins any negative component back onto
 * its bound.                                                          */
matlab_mat *matlab_optim_lsqnonneg(matlab_mat *C, matlab_mat *d) {
    if (mat_absent(C) || mat_absent(d)) return mat_alloc(0, 0);
    int m = (int)C->rows;
    int n = (int)C->cols;
    if ((int)mat_numel(d) != m) return mat_alloc(0, 0);

    std::vector<double> x(n, 0.0);
    std::vector<int> inP(n, 0);          /* 1 ⇒ column in passive set */
    std::vector<double> w(n);            /* dual / gradient vector    */
    std::vector<double> resid(m);

    const double tol = 1.0e-12;
    const int max_outer = 3 * n + 10;

    auto compute_w = [&]() {
        /* resid = d − C x */
        for (int i = 0; i < m; ++i) {
            double s = d->data[i];
            for (int j = 0; j < n; ++j) s -= C->data[(size_t)i * n + j] * x[j];
            resid[i] = s;
        }
        /* w = Cᵀ resid */
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += C->data[(size_t)i * n + j] * resid[i];
            w[j] = s;
        }
    };

    for (int outer = 0; outer < max_outer; ++outer) {
        compute_w();
        /* Pick the active column with the largest dual; stop if none
         * promises a decrease. */
        int t = -1;
        double wmax = tol;
        for (int j = 0; j < n; ++j)
            if (!inP[j] && w[j] > wmax) { wmax = w[j]; t = j; }
        if (t < 0) break;
        inP[t] = 1;

        for (int inner = 0; inner < 3 * n + 10; ++inner) {
            /* Indices currently in the passive set. */
            std::vector<int> P;
            for (int j = 0; j < n; ++j) if (inP[j]) P.push_back(j);
            int np = (int)P.size();
            if (np == 0) break;

            /* Normal equations (CₚᵀCₚ) z = Cₚᵀ d on the passive set. */
            std::vector<double> M((size_t)np * np, 0.0);
            std::vector<double> rhs(np, 0.0);
            for (int a = 0; a < np; ++a) {
                int ja = P[a];
                for (int bcol = 0; bcol < np; ++bcol) {
                    int jb = P[bcol];
                    double s = 0.0;
                    for (int i = 0; i < m; ++i)
                        s += C->data[(size_t)i * n + ja] * C->data[(size_t)i * n + jb];
                    M[(size_t)a * np + bcol] = s;
                }
                double s = 0.0;
                for (int i = 0; i < m; ++i)
                    s += C->data[(size_t)i * n + ja] * d->data[i];
                rhs[a] = s;
            }
            std::vector<double> z(n, 0.0);
            if (solve_dense_gepp(M.data(), rhs.data(), np) != 0) {
                /* Singular passive set — accept current x. */
                inner = 1 << 30;
                break;
            }
            for (int a = 0; a < np; ++a) z[P[a]] = rhs[a];

            /* All passive components positive ⇒ this z is the step. */
            double zmin = 1.0;
            for (int a = 0; a < np; ++a) zmin = std::min(zmin, z[P[a]]);
            if (zmin > 0.0) {
                for (int a = 0; a < np; ++a) x[P[a]] = z[P[a]];
                break;
            }

            /* Otherwise move x toward z until the first component hits 0. */
            double alpha = 1.0;
            for (int a = 0; a < np; ++a) {
                int j = P[a];
                if (z[j] <= 0.0) {
                    double denom = x[j] - z[j];
                    if (denom > 1.0e-300) {
                        double ratio = x[j] / denom;
                        if (ratio < alpha) alpha = ratio;
                    }
                }
            }
            for (int j = 0; j < n; ++j)
                if (inP[j]) x[j] += alpha * (z[j] - x[j]);
            /* Drop columns that reached zero. */
            for (int a = 0; a < np; ++a) {
                int j = P[a];
                if (x[j] <= tol) { x[j] = 0.0; inP[j] = 0; }
            }
        }
    }

    matlab_mat *out = mat_alloc(n, 1);
    for (int j = 0; j < n; ++j) out->data[j] = x[j];
    return out;
}

/* ===================================================================
 * fsolve — scalar nonlinear equation (Newton + Brent fallback)
 * =================================================================== */

/* `x = fsolve(@fn, x0)` for a scalar equation f(x) = 0.  Newton's
 * method with a forward finite-difference derivative; if the iteration
 * stalls (flat derivative or divergence) it falls back to the same
 * bracket-expansion + Brent path fzero uses.                          */
double matlab_optim_fsolve_scalar(void *fn_p, double x0) {
    if (!fn_p) return (double)NAN;
    matlab_optim_scalar_fn f = (matlab_optim_scalar_fn)fn_p;

    double x = x0;
    for (int it = 0; it < 100; ++it) {
        double fx = f(x);
        if (fabs(fx) < 1.0e-12) return x;
        double h = 1.0e-7 * (fabs(x) + 1.0);
        double dfx = (f(x + h) - fx) / h;
        if (fabs(dfx) < 1.0e-14) break;
        double xnew = x - fx / dfx;
        if (!isfinite(xnew)) break;
        if (fabs(xnew - x) < 1.0e-14 * (fabs(x) + 1.0)) { x = xnew; break; }
        x = xnew;
    }
    if (fabs(f(x)) < 1.0e-9) return x;

    /* Newton stalled — fall back to bracketing + Brent. */
    double a, fa, b, fb;
    if (expand_bracket(f, x0, &a, &fa, &b, &fb) == 0) {
        if (fa == 0.0) return a;
        if (fb == 0.0) return b;
        return brent_root(f, a, fa, b, fb, 1.0e-12, 1.0e-14, 100);
    }
    return x;  /* best effort */
}

/* ===================================================================
 * Tier-2 — constrained optimisation + nonlinear least squares
 *
 * Two hand-coded cores serve every Tier-2 solver:
 *   al_minimize — augmented-Lagrangian method with a bound-projected
 *                 BFGS inner solver.  Backs fmincon / quadprog / lsqlin.
 *   lm_solve    — Levenberg-Marquardt for nonlinear least squares.
 *                 Backs lsqnonlin / lsqcurvefit / fsolve (N-D).
 * See docs/optim_toolbox_roadmap.md §3.
 * =================================================================== */

/* Raw-pointer objective handle eval: y = f(x), x an n-vector. */
static double obj_eval_raw(matlab_optim_vector_fn f, const double *x, int n) {
    matlab_mat *m = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) m->data[i] = x[i];
    double r = f(m);
    free(m->data);
    free(m);
    return r;
}

/* Vector-valued handle eval: c = f(x), returns the result as a
 * std::vector (length = numel of the returned matrix). */
typedef matlab_mat *(*matlab_optim_vecout_fn)(matlab_mat *);
static std::vector<double> vecout_eval_raw(matlab_optim_vecout_fn f,
                                           const double *x, int n) {
    matlab_mat *m = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) m->data[i] = x[i];
    matlab_mat *r = f(m);
    free(m->data);
    free(m);
    std::vector<double> out;
    if (r) {
        int k = (int)(r->rows * r->cols);
        out.assign(r->data, r->data + k);
        free(r->data);
        free(r);
    }
    return out;
}

/* Curve-fit model handle eval — used by lsqcurvefit, whose model
 * function takes (params, xdata).  The model is called once per data
 * point with a *scalar* abscissa, so the handle ABI is
 * double(*)(matlab_mat *params, double t): the `t` argument stays a
 * plain f64, which keeps element-wise model expressions
 * (`x(1)*exp(-x(2)*t)`) lowering cleanly.  This is invisible for the
 * element-wise models curve fitting uses in practice. */
typedef double (*matlab_optim_curve_fn)(matlab_mat *, double);
static std::vector<double> curve_eval_raw(matlab_optim_curve_fn f,
                                          const double *x, int n,
                                          matlab_mat *xdata) {
    matlab_mat *m = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) m->data[i] = x[i];
    int md = (int)(xdata->rows * xdata->cols);
    std::vector<double> out(md);
    for (int i = 0; i < md; ++i) out[i] = f(m, xdata->data[i]);
    free(m->data);
    free(m);
    return out;
}

typedef std::function<double(const std::vector<double> &,
                             std::vector<double> *)> AlObjFn;
typedef std::function<std::vector<double>(const std::vector<double> &)> VecFn;

/* Project x onto the box [L, U] in place. */
static void box_project(std::vector<double> &x,
                        const std::vector<double> &L,
                        const std::vector<double> &U) {
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] < L[i]) x[i] = L[i];
        if (x[i] > U[i]) x[i] = U[i];
    }
}

/* Bound-projected BFGS: minimise phi(x) over the box [L, U].  phi must
 * return the value and, when `grad` is non-null, fill the gradient.
 * The inverse Hessian is maintained directly; the search direction is
 * projected via a backtracking line search whose sufficient-decrease
 * test uses the *actual* (projected) step.  Stops on a small projected
 * gradient or a vanishing step.                                      */
typedef std::function<double(const std::vector<double> &,
                             std::vector<double> *)> PhiFn;
static void inner_pbfgs(const PhiFn &phi, std::vector<double> &x,
                        const std::vector<double> &L,
                        const std::vector<double> &U, int max_iter) {
    int n = (int)x.size();
    box_project(x, L, U);
    std::vector<double> g(n), gnew(n), p(n), xnew(n), s(n), y(n), Hy(n);
    std::vector<double> H((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) H[(size_t)i * n + i] = 1.0;

    double f = phi(x, &g);
    for (int it = 0; it < max_iter; ++it) {
        /* Projected-gradient stopping test. */
        double pgn = 0.0;
        for (int i = 0; i < n; ++i) {
            double xi = x[i] - g[i];
            if (xi < L[i]) xi = L[i];
            if (xi > U[i]) xi = U[i];
            pgn = std::max(pgn, fabs(xi - x[i]));
        }
        if (pgn < 1.0e-9) break;

        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) sum += H[(size_t)i * n + j] * g[j];
            p[i] = -sum;
        }
        double slope = 0.0;
        for (int i = 0; i < n; ++i) slope += g[i] * p[i];
        if (slope >= 0.0) {
            for (size_t k = 0; k < H.size(); ++k) H[k] = 0.0;
            for (int i = 0; i < n; ++i) H[(size_t)i * n + i] = 1.0;
            for (int i = 0; i < n; ++i) p[i] = -g[i];
        }

        double alpha = 1.0;
        const double c1 = 1.0e-4;
        double fnew = f;
        int ok = 0;
        for (int ls = 0; ls < 50; ++ls) {
            for (int i = 0; i < n; ++i) {
                double xi = x[i] + alpha * p[i];
                if (xi < L[i]) xi = L[i];
                if (xi > U[i]) xi = U[i];
                xnew[i] = xi;
            }
            fnew = phi(xnew, nullptr);
            double gs = 0.0;
            for (int i = 0; i < n; ++i) gs += g[i] * (xnew[i] - x[i]);
            if (fnew <= f + c1 * gs) { ok = 1; break; }
            alpha *= 0.5;
        }
        if (!ok) break;

        fnew = phi(xnew, &gnew);
        for (int i = 0; i < n; ++i) { s[i] = xnew[i] - x[i]; y[i] = gnew[i] - g[i]; }
        double sy = 0.0;
        for (int i = 0; i < n; ++i) sy += s[i] * y[i];
        if (sy > 1.0e-12) {
            double rho = 1.0 / sy;
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) sum += H[(size_t)i * n + j] * y[j];
                Hy[i] = sum;
            }
            double yHy = 0.0;
            for (int i = 0; i < n; ++i) yHy += y[i] * Hy[i];
            double coef = rho * (rho * yHy + 1.0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    H[(size_t)i * n + j] +=
                        -rho * (s[i] * Hy[j] + Hy[i] * s[j])
                        + coef * s[i] * s[j];
        }
        double sn = 0.0;
        for (int i = 0; i < n; ++i) sn = std::max(sn, fabs(s[i]));
        x.swap(xnew);
        f = fnew;
        g.swap(gnew);
        if (sn < 1.0e-12) break;
    }
}

/* Augmented-Lagrangian method for
 *   min f(x)  s.t.  A x ≤ b,  Aeq x = beq,  lb ≤ x ≤ ub,  c(x) ≤ 0.
 *
 * Bounds are enforced by projection inside the inner solve; linear and
 * nonlinear (in)equalities are folded into the Powell-Hestenes-
 * Rockafellar augmented Lagrangian
 *   Φ = f + Σ_eq[λ h + (μ/2)h²] + Σ_ineq (1/2μ)[max(0,σ+μg)² − σ²].
 * The outer loop updates the multipliers (λ, σ) and ramps μ whenever
 * the constraint violation fails to fall fast enough.  Nonlinear
 * constraint gradients are obtained by forward finite differences.
 * Tier-2 handles nonlinear *inequalities* only; nonlinear equalities
 * are deferred (see the roadmap).                                     */
static std::vector<double> al_minimize(const AlObjFn &obj,
                                       const std::vector<double> &x0,
                                       const matlab_mat *A, const matlab_mat *b,
                                       const matlab_mat *Aeq, const matlab_mat *beq,
                                       const matlab_mat *lb, const matlab_mat *ub,
                                       const VecFn &nonlcon) {
    int n = (int)x0.size();

    std::vector<double> L(n, -INFINITY), U(n, INFINITY);
    if (!mat_absent(lb) && (int)mat_numel(lb) == n)
        for (int i = 0; i < n; ++i) L[i] = lb->data[i];
    if (!mat_absent(ub) && (int)mat_numel(ub) == n)
        for (int i = 0; i < n; ++i) U[i] = ub->data[i];

    /* Linear inequality / equality row counts. */
    int mInA = (!mat_absent(A) && (int)A->cols == n) ? (int)A->rows : 0;
    int mEq  = (!mat_absent(Aeq) && (int)Aeq->cols == n) ? (int)Aeq->rows : 0;

    std::vector<double> x = x0;
    box_project(x, L, U);

    /* Probe the nonlinear constraint count once. */
    int mNL = 0;
    bool haveNL = (bool)nonlcon;
    if (haveNL) mNL = (int)nonlcon(x).size();
    int mIneq = mInA + mNL;

    std::vector<double> lambda(mEq, 0.0);   /* equality multipliers   */
    std::vector<double> sigma(mIneq, 0.0);  /* inequality multipliers */
    double mu = 10.0;

    /* Φ(x): augmented-Lagrangian value, optional gradient. */
    auto phi = [&](const std::vector<double> &xv,
                   std::vector<double> *grad) -> double {
        std::vector<double> gf;
        double val = obj(xv, grad ? &gf : nullptr);
        if (grad) { grad->assign(n, 0.0); for (int i = 0; i < n; ++i) (*grad)[i] = gf[i]; }

        /* Linear equalities. */
        for (int i = 0; i < mEq; ++i) {
            double hi = -beq->data[i];
            for (int j = 0; j < n; ++j) hi += Aeq->data[(size_t)i * n + j] * xv[j];
            val += lambda[i] * hi + 0.5 * mu * hi * hi;
            if (grad) {
                double coef = lambda[i] + mu * hi;
                for (int j = 0; j < n; ++j)
                    (*grad)[j] += coef * Aeq->data[(size_t)i * n + j];
            }
        }
        /* Linear inequalities. */
        for (int i = 0; i < mInA; ++i) {
            double gi = -b->data[i];
            for (int j = 0; j < n; ++j) gi += A->data[(size_t)i * n + j] * xv[j];
            double sv = sigma[i] + mu * gi;
            if (sv > 0.0) {
                val += (0.5 / mu) * (sv * sv - sigma[i] * sigma[i]);
                if (grad)
                    for (int j = 0; j < n; ++j)
                        (*grad)[j] += sv * A->data[(size_t)i * n + j];
            } else {
                val += (0.5 / mu) * (-sigma[i] * sigma[i]);
            }
        }
        /* Nonlinear inequalities. */
        if (haveNL && mNL > 0) {
            std::vector<double> c = nonlcon(xv);
            int mc = std::min((int)c.size(), mNL);
            /* Finite-difference Jacobian, only when a gradient is needed. */
            std::vector<double> Jc;
            if (grad) {
                Jc.assign((size_t)mc * n, 0.0);
                std::vector<double> xp = xv;
                for (int j = 0; j < n; ++j) {
                    double h = 1.0e-7 * (fabs(xv[j]) + 1.0);
                    xp[j] = xv[j] + h;
                    std::vector<double> cp = nonlcon(xp);
                    xp[j] = xv[j];
                    for (int k = 0; k < mc; ++k)
                        Jc[(size_t)k * n + j] = (cp[k] - c[k]) / h;
                }
            }
            for (int k = 0; k < mc; ++k) {
                int idx = mInA + k;
                double sv = sigma[idx] + mu * c[k];
                if (sv > 0.0) {
                    val += (0.5 / mu) * (sv * sv - sigma[idx] * sigma[idx]);
                    if (grad)
                        for (int j = 0; j < n; ++j)
                            (*grad)[j] += sv * Jc[(size_t)k * n + j];
                } else {
                    val += (0.5 / mu) * (-sigma[idx] * sigma[idx]);
                }
            }
        }
        return val;
    };

    const int max_outer = 60;
    double prev_viol = INFINITY;
    for (int outer = 0; outer < max_outer; ++outer) {
        inner_pbfgs(phi, x, L, U, 400);

        /* Constraint values + violation. */
        std::vector<double> hcur(mEq), gcur(mIneq);
        double viol = 0.0;
        for (int i = 0; i < mEq; ++i) {
            double hi = -beq->data[i];
            for (int j = 0; j < n; ++j) hi += Aeq->data[(size_t)i * n + j] * x[j];
            hcur[i] = hi;
            viol = std::max(viol, fabs(hi));
        }
        for (int i = 0; i < mInA; ++i) {
            double gi = -b->data[i];
            for (int j = 0; j < n; ++j) gi += A->data[(size_t)i * n + j] * x[j];
            gcur[i] = gi;
            viol = std::max(viol, gi);
        }
        if (haveNL && mNL > 0) {
            std::vector<double> c = nonlcon(x);
            for (int k = 0; k < mNL && k < (int)c.size(); ++k) {
                gcur[mInA + k] = c[k];
                viol = std::max(viol, c[k]);
            }
        }

        if (viol < 1.0e-8 && outer > 0) break;

        /* Multiplier updates. */
        for (int i = 0; i < mEq; ++i) lambda[i] += mu * hcur[i];
        for (int i = 0; i < mIneq; ++i)
            sigma[i] = std::max(0.0, sigma[i] + mu * gcur[i]);

        /* Penalty ramp when the violation is not shrinking fast enough. */
        if (viol > 0.25 * prev_viol && mu < 1.0e8) mu *= 5.0;
        prev_viol = viol;
    }
    return x;
}

/* Levenberg-Marquardt for nonlinear least squares: minimise ‖r(x)‖²
 * where r is supplied as a residual function.  The Jacobian is formed
 * by forward finite differences; each step solves the damped normal
 * equations (JᵀJ + λ·diag(JᵀJ)) p = −Jᵀr, with λ shrinking on a
 * successful step and growing otherwise.  Bounds, when present, are
 * enforced by projecting the trial point (adequate for the Tier-2
 * gating problems; the trust-region-reflective treatment is Tier-3). */
static std::vector<double> lm_solve(const VecFn &residual,
                                    const std::vector<double> &x0,
                                    const matlab_mat *lb, const matlab_mat *ub) {
    int n = (int)x0.size();
    std::vector<double> L(n, -INFINITY), U(n, INFINITY);
    if (!mat_absent(lb) && (int)mat_numel(lb) == n)
        for (int i = 0; i < n; ++i) L[i] = lb->data[i];
    if (!mat_absent(ub) && (int)mat_numel(ub) == n)
        for (int i = 0; i < n; ++i) U[i] = ub->data[i];

    std::vector<double> x = x0;
    box_project(x, L, U);
    std::vector<double> r = residual(x);
    int m = (int)r.size();
    if (m == 0) return x;

    auto sumsq = [](const std::vector<double> &v) {
        double s = 0.0;
        for (double e : v) s += e * e;
        return s;
    };
    double cost = sumsq(r);
    double lambda = 1.0e-3;
    const int max_iter = 200;

    for (int it = 0; it < max_iter; ++it) {
        /* Forward-difference Jacobian J (m × n). */
        std::vector<double> J((size_t)m * n, 0.0);
        std::vector<double> xp = x;
        for (int j = 0; j < n; ++j) {
            double h = 1.0e-7 * (fabs(x[j]) + 1.0);
            xp[j] = x[j] + h;
            std::vector<double> rp = residual(xp);
            xp[j] = x[j];
            for (int i = 0; i < m && i < (int)rp.size(); ++i)
                J[(size_t)i * n + j] = (rp[i] - r[i]) / h;
        }
        /* Normal-equation pieces: JtJ (n × n), g = Jᵀr (n). */
        std::vector<double> JtJ((size_t)n * n, 0.0), g(n, 0.0);
        for (int a = 0; a < n; ++a) {
            for (int bb = 0; bb < n; ++bb) {
                double s = 0.0;
                for (int i = 0; i < m; ++i)
                    s += J[(size_t)i * n + a] * J[(size_t)i * n + bb];
                JtJ[(size_t)a * n + bb] = s;
            }
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += J[(size_t)i * n + a] * r[i];
            g[a] = s;
        }
        double gnorm = 0.0;
        for (int j = 0; j < n; ++j) gnorm = std::max(gnorm, fabs(g[j]));
        if (gnorm < 1.0e-10) break;

        /* Inner loop: adjust λ until a step decreases the cost. */
        int accepted = 0;
        for (int tries = 0; tries < 30; ++tries) {
            std::vector<double> M((size_t)n * n), rhs(n);
            for (int a = 0; a < n; ++a) {
                for (int bb = 0; bb < n; ++bb)
                    M[(size_t)a * n + bb] = JtJ[(size_t)a * n + bb];
                M[(size_t)a * n + a] += lambda * (JtJ[(size_t)a * n + a] + 1.0e-12);
                rhs[a] = -g[a];
            }
            if (solve_dense_gepp(M.data(), rhs.data(), n) != 0) {
                lambda *= 3.0;
                continue;
            }
            std::vector<double> xt(n);
            for (int j = 0; j < n; ++j) {
                double v = x[j] + rhs[j];
                if (v < L[j]) v = L[j];
                if (v > U[j]) v = U[j];
                xt[j] = v;
            }
            std::vector<double> rt = residual(xt);
            double ct = sumsq(rt);
            if (ct < cost) {
                double step = 0.0;
                for (int j = 0; j < n; ++j) step = std::max(step, fabs(xt[j] - x[j]));
                x.swap(xt);
                r.swap(rt);
                cost = ct;
                lambda = std::max(lambda / 3.0, 1.0e-12);
                accepted = 1;
                if (step < 1.0e-12) accepted = 2;
                break;
            }
            lambda *= 3.0;
            if (lambda > 1.0e12) { accepted = 0; break; }
        }
        if (accepted == 0 || accepted == 2) break;
    }
    return x;
}

/* --- fmincon — general constrained nonlinear minimisation --------- *
 *
 * `x = fmincon(@fun, x0, A, b, Aeq, beq, lb, ub, @nonlcon)`.  The
 * lowering always passes the full 9-slot ABI, padding absent matrix
 * arguments with a null pointer and absent handles with null.  The
 * objective handle has the vector ABI double(*)(matlab_mat*); the
 * optional nonlcon handle returns the column vector of nonlinear
 * inequality constraint values c(x) ≤ 0.                              */
matlab_mat *matlab_optim_fmincon(void *obj_p, matlab_mat *x0,
                                 matlab_mat *A, matlab_mat *b,
                                 matlab_mat *Aeq, matlab_mat *beq,
                                 matlab_mat *lb, matlab_mat *ub,
                                 void *nonlcon_p) {
    if (!obj_p || mat_absent(x0)) return mat_alloc(0, 0);
    int n = (int)mat_numel(x0);
    matlab_optim_vector_fn fobj = (matlab_optim_vector_fn)obj_p;

    AlObjFn obj = [fobj, n](const std::vector<double> &xv,
                            std::vector<double> *grad) -> double {
        double fx = obj_eval_raw(fobj, xv.data(), n);
        if (grad) {
            grad->assign(n, 0.0);
            std::vector<double> xp = xv;
            for (int i = 0; i < n; ++i) {
                double h = 1.0e-7 * (fabs(xv[i]) + 1.0);
                xp[i] = xv[i] + h;
                double fp = obj_eval_raw(fobj, xp.data(), n);
                xp[i] = xv[i];
                (*grad)[i] = (fp - fx) / h;
            }
        }
        return fx;
    };

    VecFn nonlcon;
    if (nonlcon_p) {
        matlab_optim_vecout_fn fc = (matlab_optim_vecout_fn)nonlcon_p;
        nonlcon = [fc, n](const std::vector<double> &xv) -> std::vector<double> {
            return vecout_eval_raw(fc, xv.data(), n);
        };
    }

    std::vector<double> x0v(x0->data, x0->data + n);
    std::vector<double> sol =
        al_minimize(obj, x0v, A, b, Aeq, beq, lb, ub, nonlcon);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = sol[i];
    return out;
}

/* --- quadprog — convex quadratic programming --------------------- *
 *
 * `x = quadprog(H, f, A, b, Aeq, beq, lb, ub)`.  Routed through the
 * augmented-Lagrangian core with the analytic quadratic objective
 * ½xᵀHx + fᵀx (gradient Hx + f).  The starting point is the origin
 * projected onto the bounds.                                         */
matlab_mat *matlab_optim_quadprog(matlab_mat *H, matlab_mat *fvec,
                                  matlab_mat *A, matlab_mat *b,
                                  matlab_mat *Aeq, matlab_mat *beq,
                                  matlab_mat *lb, matlab_mat *ub) {
    if (mat_absent(H) || mat_absent(fvec)) return mat_alloc(0, 0);
    int n = (int)mat_numel(fvec);
    if ((int)H->rows != n || (int)H->cols != n) return mat_alloc(0, 0);

    AlObjFn obj = [H, fvec, n](const std::vector<double> &xv,
                               std::vector<double> *grad) -> double {
        std::vector<double> Hx(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += H->data[(size_t)i * n + j] * xv[j];
            Hx[i] = s;
        }
        double val = 0.0;
        for (int i = 0; i < n; ++i) val += 0.5 * xv[i] * Hx[i] + fvec->data[i] * xv[i];
        if (grad) {
            grad->assign(n, 0.0);
            for (int i = 0; i < n; ++i) (*grad)[i] = Hx[i] + fvec->data[i];
        }
        return val;
    };

    std::vector<double> x0(n, 0.0);
    VecFn none;
    std::vector<double> sol = al_minimize(obj, x0, A, b, Aeq, beq, lb, ub, none);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = sol[i];
    return out;
}

/* --- lsqlin — constrained linear least squares ------------------- *
 *
 * `x = lsqlin(C, d, A, b, Aeq, beq, lb, ub)`.  Minimises ½‖Cx − d‖²
 * subject to the linear constraints, again through the augmented-
 * Lagrangian core (gradient Cᵀ(Cx − d)).                              */
matlab_mat *matlab_optim_lsqlin(matlab_mat *C, matlab_mat *d,
                                matlab_mat *A, matlab_mat *b,
                                matlab_mat *Aeq, matlab_mat *beq,
                                matlab_mat *lb, matlab_mat *ub) {
    if (mat_absent(C) || mat_absent(d)) return mat_alloc(0, 0);
    int m = (int)C->rows;
    int n = (int)C->cols;
    if ((int)mat_numel(d) != m) return mat_alloc(0, 0);

    AlObjFn obj = [C, d, m, n](const std::vector<double> &xv,
                               std::vector<double> *grad) -> double {
        std::vector<double> r(m);
        for (int i = 0; i < m; ++i) {
            double s = -d->data[i];
            for (int j = 0; j < n; ++j) s += C->data[(size_t)i * n + j] * xv[j];
            r[i] = s;
        }
        double val = 0.0;
        for (int i = 0; i < m; ++i) val += 0.5 * r[i] * r[i];
        if (grad) {
            grad->assign(n, 0.0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    (*grad)[j] += C->data[(size_t)i * n + j] * r[i];
        }
        return val;
    };

    std::vector<double> x0(n, 0.0);
    VecFn none;
    std::vector<double> sol = al_minimize(obj, x0, A, b, Aeq, beq, lb, ub, none);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = sol[i];
    return out;
}

/* --- lsqnonlin — nonlinear least squares ------------------------- *
 *
 * `x = lsqnonlin(@fun, x0)` or `lsqnonlin(@fun, x0, lb, ub)`.  The
 * objective handle returns the residual vector r(x); the solver
 * minimises ‖r(x)‖² by Levenberg-Marquardt.                           */
matlab_mat *matlab_optim_lsqnonlin(void *fun_p, matlab_mat *x0,
                                   matlab_mat *lb, matlab_mat *ub) {
    if (!fun_p || mat_absent(x0)) return mat_alloc(0, 0);
    int n = (int)mat_numel(x0);
    matlab_optim_vecout_fn fr = (matlab_optim_vecout_fn)fun_p;
    VecFn residual = [fr, n](const std::vector<double> &xv) -> std::vector<double> {
        return vecout_eval_raw(fr, xv.data(), n);
    };
    std::vector<double> x0v(x0->data, x0->data + n);
    std::vector<double> sol = lm_solve(residual, x0v, lb, ub);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = sol[i];
    return out;
}

/* --- lsqcurvefit — nonlinear curve fitting ----------------------- *
 *
 * `x = lsqcurvefit(@fun, x0, xdata, ydata)`.  The model handle
 * `fun(params, xdata)` returns predicted values; the residual is
 * fun(x, xdata) − ydata and the fit runs through Levenberg-Marquardt. */
matlab_mat *matlab_optim_lsqcurvefit(void *fun_p, matlab_mat *x0,
                                     matlab_mat *xdata, matlab_mat *ydata) {
    if (!fun_p || mat_absent(x0) || mat_absent(xdata) || mat_absent(ydata))
        return mat_alloc(0, 0);
    int n = (int)mat_numel(x0);
    int m = (int)mat_numel(ydata);
    matlab_optim_curve_fn fc = (matlab_optim_curve_fn)fun_p;
    VecFn residual = [fc, n, m, xdata, ydata](const std::vector<double> &xv)
        -> std::vector<double> {
        std::vector<double> yp = curve_eval_raw(fc, xv.data(), n, xdata);
        std::vector<double> r(m, 0.0);
        for (int i = 0; i < m && i < (int)yp.size(); ++i)
            r[i] = yp[i] - ydata->data[i];
        return r;
    };
    std::vector<double> x0v(x0->data, x0->data + n);
    std::vector<double> sol = lm_solve(residual, x0v, NULL, NULL);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = sol[i];
    return out;
}

/* --- fsolve (N-D) — nonlinear system of equations ---------------- *
 *
 * `x = fsolve(@fun, x0)` where x0 is a vector and fun returns the
 * residual vector F(x).  Solved by Levenberg-Marquardt on ‖F(x)‖²,
 * which behaves well for square, over- and under-determined systems.
 * The scalar form (x0 a scalar) is handled by matlab_optim_fsolve_scalar. */
matlab_mat *matlab_optim_fsolve(void *fun_p, matlab_mat *x0) {
    if (!fun_p || mat_absent(x0)) return mat_alloc(0, 0);
    int n = (int)mat_numel(x0);
    matlab_optim_vecout_fn ff = (matlab_optim_vecout_fn)fun_p;
    VecFn residual = [ff, n](const std::vector<double> &xv) -> std::vector<double> {
        return vecout_eval_raw(ff, xv.data(), n);
    };
    std::vector<double> x0v(x0->data, x0->data + n);
    std::vector<double> sol = lm_solve(residual, x0v, NULL, NULL);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = sol[i];
    return out;
}

/* ===================================================================
 * Tier-3 — MILP, second-order cone, minimax / goal-attainment, and
 * semi-infinite programming.  See docs/optim_toolbox_roadmap.md §4.
 *
 * Every Tier-3 solver is a reformulation on top of the Tier-1/2 cores:
 *   intlinprog  — branch-and-bound over linprog_core
 *   fminimax    — epigraph reformulation through al_minimize
 *   fgoalattain — epigraph reformulation through al_minimize
 *   coneprog    — SOC constraint as a nonlinear inequality, al_minimize
 *   fseminf     — outer-approximation w-grid sampling + al_minimize
 * =================================================================== */

/* Build a column matlab_mat from a std::vector. */
static matlab_mat *vec_to_colmat(const std::vector<double> &v) {
    matlab_mat *m = mat_alloc((int64_t)v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) m->data[i] = v[i];
    return m;
}

/* Return M with `extra` zero columns appended (m×n → m×(n+extra)).
 * A null / absent M yields null (still "absent"). */
static matlab_mat *pad_cols(const matlab_mat *M, int extra) {
    if (mat_absent(M)) return NULL;
    int m = (int)M->rows, n = (int)M->cols;
    matlab_mat *out = mat_alloc(m, n + extra);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            out->data[(size_t)i * (n + extra) + j] = M->data[(size_t)i * n + j];
    return out;  /* the extra columns stay zero (mat_alloc calloc's) */
}

static void free_mat(matlab_mat *m) {
    if (m) { free(m->data); free(m); }
}

/* --- intlinprog — mixed-integer linear programming --------------- *
 *
 * `x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub)`.  Depth-first
 * branch-and-bound: each node solves the LP relaxation with the dense
 * 2-phase simplex (linprog_core) under tightened bounds, prunes by
 * the incumbent objective, and branches on the most-fractional
 * integer variable into the `x_j ≤ ⌊x_j⌋` and `x_j ≥ ⌈x_j⌉` children.
 * `intcon` lists the 1-based indices of the integer-restricted
 * variables.  Adequate for the small dense MILPs Tier-3 targets.    */
matlab_mat *matlab_optim_intlinprog(matlab_mat *f, matlab_mat *intcon,
                                    matlab_mat *A, matlab_mat *b,
                                    matlab_mat *Aeq, matlab_mat *beq,
                                    matlab_mat *lb, matlab_mat *ub) {
    if (mat_absent(f)) return mat_alloc(0, 0);
    int n = (int)mat_numel(f);

    std::vector<int> ints;
    if (!mat_absent(intcon)) {
        int ni = (int)mat_numel(intcon);
        for (int i = 0; i < ni; ++i) {
            int idx = (int)llround(intcon->data[i]) - 1;
            if (idx >= 0 && idx < n) ints.push_back(idx);
        }
    }

    std::vector<double> L0(n, 0.0), U0(n, INFINITY);
    if (!mat_absent(lb) && (int)mat_numel(lb) == n)
        for (int i = 0; i < n; ++i) L0[i] = lb->data[i];
    if (!mat_absent(ub) && (int)mat_numel(ub) == n)
        for (int i = 0; i < n; ++i) U0[i] = ub->data[i];

    auto solveNode = [&](const std::vector<double> &L,
                         const std::vector<double> &U,
                         std::vector<double> &xout, double &objout) -> bool {
        matlab_mat *Lm = vec_to_colmat(L);
        matlab_mat *Um = vec_to_colmat(U);
        matlab_mat *xr = linprog_core(f, A, b, Aeq, beq, Lm, Um);
        free_mat(Lm);
        free_mat(Um);
        if (!xr || xr->rows * xr->cols == 0) { free_mat(xr); return false; }
        xout.assign(xr->data, xr->data + n);
        free_mat(xr);
        objout = 0.0;
        for (int i = 0; i < n; ++i) objout += f->data[i] * xout[i];
        return true;
    };

    std::vector<double> incumbent;
    double incumbentObj = INFINITY;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> stack;
    stack.push_back({L0, U0});
    const int MAX_NODES = 200000;
    const double INT_TOL = 1.0e-6;
    int nodes = 0;

    while (!stack.empty() && nodes < MAX_NODES) {
        auto node = stack.back();
        stack.pop_back();
        ++nodes;

        std::vector<double> x;
        double obj;
        if (!solveNode(node.first, node.second, x, obj)) continue;  /* infeasible */
        if (obj >= incumbentObj - 1.0e-9) continue;                 /* bound prune */

        int branchVar = -1;
        double worstFrac = INT_TOL;
        for (int idx : ints) {
            double fr = x[idx] - floor(x[idx]);
            double dist = std::min(fr, 1.0 - fr);
            if (dist > worstFrac) { worstFrac = dist; branchVar = idx; }
        }
        if (branchVar < 0) {
            /* Integer-feasible — snap the integer components and record. */
            if (obj < incumbentObj) {
                incumbentObj = obj;
                incumbent = x;
                for (int idx : ints) incumbent[idx] = floor(incumbent[idx] + 0.5);
            }
            continue;
        }

        double xv = x[branchVar];
        std::vector<double> cL = node.first, cU = node.second;
        cU[branchVar] = floor(xv);
        if (cU[branchVar] >= cL[branchVar] - 1.0e-9)
            stack.push_back({cL, cU});
        std::vector<double> dL = node.first, dU = node.second;
        dL[branchVar] = ceil(xv);
        if (!isfinite(dU[branchVar]) || dU[branchVar] >= dL[branchVar] - 1.0e-9)
            stack.push_back({dL, dU});
    }

    if (incumbent.empty()) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = incumbent[i];
    return out;
}

/* --- fminimax — minimax optimisation ----------------------------- *
 *
 * `x = fminimax(@fun, x0, A, b, Aeq, beq, lb, ub)`.  Minimises
 * max_i F_i(x) via the standard epigraph reformulation: with the
 * augmented variable z = [x; γ], minimise γ subject to
 * F_i(x) − γ ≤ 0 (a nonlinear inequality) and the linear
 * constraints (padded with a zero column for γ).                    */
matlab_mat *matlab_optim_fminimax(void *fun_p, matlab_mat *x0,
                                  matlab_mat *A, matlab_mat *b,
                                  matlab_mat *Aeq, matlab_mat *beq,
                                  matlab_mat *lb, matlab_mat *ub) {
    if (!fun_p || mat_absent(x0)) return mat_alloc(0, 0);
    int n = (int)mat_numel(x0);
    matlab_optim_vecout_fn fF = (matlab_optim_vecout_fn)fun_p;

    AlObjFn obj = [n](const std::vector<double> &z,
                      std::vector<double> *grad) -> double {
        if (grad) { grad->assign(n + 1, 0.0); (*grad)[n] = 1.0; }
        return z[n];
    };
    VecFn nonlcon = [fF, n](const std::vector<double> &z) -> std::vector<double> {
        std::vector<double> F = vecout_eval_raw(fF, z.data(), n);
        for (double &v : F) v -= z[n];
        return F;
    };

    std::vector<double> Fx0 = vecout_eval_raw(fF, x0->data, n);
    double g0 = -INFINITY;
    for (double v : Fx0) g0 = std::max(g0, v);
    if (!isfinite(g0)) g0 = 0.0;
    std::vector<double> z0(n + 1);
    for (int i = 0; i < n; ++i) z0[i] = x0->data[i];
    z0[n] = g0;

    matlab_mat *Apad = pad_cols(A, 1);
    matlab_mat *Aeqpad = pad_cols(Aeq, 1);
    matlab_mat *lbpad = NULL, *ubpad = NULL;
    if (!mat_absent(lb) && (int)mat_numel(lb) == n) {
        std::vector<double> v(lb->data, lb->data + n);
        v.push_back(-INFINITY);
        lbpad = vec_to_colmat(v);
    }
    if (!mat_absent(ub) && (int)mat_numel(ub) == n) {
        std::vector<double> v(ub->data, ub->data + n);
        v.push_back(INFINITY);
        ubpad = vec_to_colmat(v);
    }

    std::vector<double> zsol =
        al_minimize(obj, z0, Apad, b, Aeqpad, beq, lbpad, ubpad, nonlcon);

    free_mat(Apad);
    free_mat(Aeqpad);
    free_mat(lbpad);
    free_mat(ubpad);

    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = zsol[i];
    return out;
}

/* --- fgoalattain — multiobjective goal attainment ---------------- *
 *
 * `x = fgoalattain(@fun, x0, goal, weight, A, b, Aeq, beq, lb, ub)`.
 * With z = [x; γ], minimises γ subject to
 * F_i(x) − weight_i·γ ≤ goal_i and the linear constraints.           */
matlab_mat *matlab_optim_fgoalattain(void *fun_p, matlab_mat *x0,
                                     matlab_mat *goal, matlab_mat *weight,
                                     matlab_mat *A, matlab_mat *b,
                                     matlab_mat *Aeq, matlab_mat *beq,
                                     matlab_mat *lb, matlab_mat *ub) {
    if (!fun_p || mat_absent(x0) || mat_absent(goal)) return mat_alloc(0, 0);
    int n = (int)mat_numel(x0);
    int m = (int)mat_numel(goal);
    matlab_optim_vecout_fn fF = (matlab_optim_vecout_fn)fun_p;

    std::vector<double> gvec(goal->data, goal->data + m);
    std::vector<double> wvec(m, 1.0);
    if (!mat_absent(weight) && (int)mat_numel(weight) == m)
        for (int i = 0; i < m; ++i) wvec[i] = weight->data[i];

    AlObjFn obj = [n](const std::vector<double> &z,
                      std::vector<double> *grad) -> double {
        if (grad) { grad->assign(n + 1, 0.0); (*grad)[n] = 1.0; }
        return z[n];
    };
    VecFn nonlcon = [fF, n, m, gvec, wvec](const std::vector<double> &z)
        -> std::vector<double> {
        std::vector<double> F = vecout_eval_raw(fF, z.data(), n);
        std::vector<double> c(m, 0.0);
        for (int i = 0; i < m && i < (int)F.size(); ++i)
            c[i] = F[i] - wvec[i] * z[n] - gvec[i];
        return c;
    };

    /* γ initial guess: the worst attainment factor at x0. */
    std::vector<double> Fx0 = vecout_eval_raw(fF, x0->data, n);
    double g0 = 0.0;
    for (int i = 0; i < m && i < (int)Fx0.size(); ++i) {
        double w = (fabs(wvec[i]) > 1.0e-12) ? wvec[i] : 1.0;
        g0 = std::max(g0, (Fx0[i] - gvec[i]) / w);
    }
    std::vector<double> z0(n + 1);
    for (int i = 0; i < n; ++i) z0[i] = x0->data[i];
    z0[n] = g0;

    matlab_mat *Apad = pad_cols(A, 1);
    matlab_mat *Aeqpad = pad_cols(Aeq, 1);
    matlab_mat *lbpad = NULL, *ubpad = NULL;
    if (!mat_absent(lb) && (int)mat_numel(lb) == n) {
        std::vector<double> v(lb->data, lb->data + n);
        v.push_back(-INFINITY);
        lbpad = vec_to_colmat(v);
    }
    if (!mat_absent(ub) && (int)mat_numel(ub) == n) {
        std::vector<double> v(ub->data, ub->data + n);
        v.push_back(INFINITY);
        ubpad = vec_to_colmat(v);
    }

    std::vector<double> zsol =
        al_minimize(obj, z0, Apad, b, Aeqpad, beq, lbpad, ubpad, nonlcon);

    free_mat(Apad);
    free_mat(Aeqpad);
    free_mat(lbpad);
    free_mat(ubpad);

    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = zsol[i];
    return out;
}

/* --- coneprog — second-order cone programming -------------------- *
 *
 * `x = coneprog(f, Asc, bsc, dsc, gamma, A, b, Aeq, beq, lb, ub)`.
 * Minimises fᵀx subject to the second-order cone constraint
 * ‖Asc·x + bsc‖ ≤ dscᵀx + gamma plus the linear constraints.  The
 * cone is handled as a single nonlinear inequality
 * ‖Asc·x + bsc‖ − (dscᵀx + gamma) ≤ 0 routed through al_minimize.
 * Tier-3 supports a single cone (the common SOCP shape); multi-cone
 * problems are deferred — see the roadmap.                          */
matlab_mat *matlab_optim_coneprog(matlab_mat *f, matlab_mat *Asc,
                                  matlab_mat *bsc, matlab_mat *dsc,
                                  matlab_mat *gamma_sc, matlab_mat *A,
                                  matlab_mat *b, matlab_mat *Aeq,
                                  matlab_mat *beq, matlab_mat *lb,
                                  matlab_mat *ub) {
    if (mat_absent(f)) return mat_alloc(0, 0);
    int n = (int)mat_numel(f);

    AlObjFn obj = [f, n](const std::vector<double> &x,
                         std::vector<double> *grad) -> double {
        double v = 0.0;
        for (int i = 0; i < n; ++i) v += f->data[i] * x[i];
        if (grad) {
            grad->assign(n, 0.0);
            for (int i = 0; i < n; ++i) (*grad)[i] = f->data[i];
        }
        return v;
    };

    double gam = mat_absent(gamma_sc) ? 0.0 : gamma_sc->data[0];
    VecFn nonlcon;
    if (!mat_absent(Asc)) {
        int mc = (int)Asc->rows;
        nonlcon = [Asc, bsc, dsc, gam, n, mc](const std::vector<double> &x)
            -> std::vector<double> {
            double nrm = 0.0;
            for (int i = 0; i < mc; ++i) {
                double r = mat_absent(bsc) ? 0.0 : bsc->data[i];
                for (int j = 0; j < n; ++j)
                    r += Asc->data[(size_t)i * n + j] * x[j];
                nrm += r * r;
            }
            nrm = sqrt(nrm);
            double rhs = gam;
            if (!mat_absent(dsc))
                for (int j = 0; j < n; ++j) rhs += dsc->data[j] * x[j];
            return std::vector<double>{nrm - rhs};
        };
    }

    std::vector<double> x0(n, 0.0);
    std::vector<double> sol = al_minimize(obj, x0, A, b, Aeq, beq, lb, ub, nonlcon);
    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = sol[i];
    return out;
}

/* --- fseminf — semi-infinite programming ------------------------- *
 *
 * `x = fseminf(@fun, x0, @seminfcon, lb, ub)`.  Minimises fun(x)
 * subject to the semi-infinite constraint phi(x, w) ≤ 0 for every w
 * in [0, 1].  Tier-3 supports a single semi-infinite constraint whose
 * handle has the per-point ABI double(*)(matlab_mat*, double).  The
 * solver runs an outer-approximation loop: minimise over the current
 * finite set of sampled w-points, then add the most-violating w from
 * a fine grid; iterate until the worst violation is within tolerance.
 * The full MATLAB seminfcon multi-output ABI is deferred.            */
matlab_mat *matlab_optim_fseminf(void *fun_p, matlab_mat *x0,
                                 void *seminfcon_p, matlab_mat *lb,
                                 matlab_mat *ub) {
    if (!fun_p || mat_absent(x0) || !seminfcon_p) return mat_alloc(0, 0);
    int n = (int)mat_numel(x0);
    matlab_optim_vector_fn fobj = (matlab_optim_vector_fn)fun_p;
    matlab_optim_curve_fn fcon = (matlab_optim_curve_fn)seminfcon_p;

    AlObjFn obj = [fobj, n](const std::vector<double> &x,
                            std::vector<double> *grad) -> double {
        double fx = obj_eval_raw(fobj, x.data(), n);
        if (grad) {
            grad->assign(n, 0.0);
            std::vector<double> xp = x;
            for (int i = 0; i < n; ++i) {
                double h = 1.0e-7 * (fabs(x[i]) + 1.0);
                xp[i] = x[i] + h;
                double fp = obj_eval_raw(fobj, xp.data(), n);
                xp[i] = x[i];
                (*grad)[i] = (fp - fx) / h;
            }
        }
        return fx;
    };

    std::vector<double> x(x0->data, x0->data + n);
    std::vector<double> Wset;
    const int GRID = 50;
    const int MAX_OUTER = 25;

    for (int outer = 0; outer < MAX_OUTER; ++outer) {
        VecFn nonlcon;
        if (!Wset.empty()) {
            std::vector<double> Wcopy = Wset;
            nonlcon = [fcon, n, Wcopy](const std::vector<double> &xv)
                -> std::vector<double> {
                matlab_mat *m = mat_alloc(n, 1);
                for (int i = 0; i < n; ++i) m->data[i] = xv[i];
                std::vector<double> out(Wcopy.size());
                for (size_t k = 0; k < Wcopy.size(); ++k)
                    out[k] = fcon(m, Wcopy[k]);
                free_mat(m);
                return out;
            };
        }
        x = al_minimize(obj, x, NULL, NULL, NULL, NULL, lb, ub, nonlcon);

        /* Find the most-violating sampling point on a fine grid. */
        matlab_mat *m = mat_alloc(n, 1);
        for (int i = 0; i < n; ++i) m->data[i] = x[i];
        double worstW = 0.0, worstV = -INFINITY;
        for (int g = 0; g <= GRID; ++g) {
            double w = (double)g / GRID;
            double v = fcon(m, w);
            if (v > worstV) { worstV = v; worstW = w; }
        }
        free_mat(m);
        if (worstV <= 1.0e-6) break;
        Wset.push_back(worstW);
    }

    matlab_mat *out = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) out->data[i] = x[i];
    return out;
}

/* ===================================================================
 * Tier-4 — Problem-based optimisation (expression-DAG runtime)
 *
 * See docs/optim_toolbox_roadmap.md §5.  The problem-based API
 * (`optimvar` / operator-overloaded expressions / `optimproblem` /
 * `solve`) is backed by a global scalar expression DAG built in this
 * runtime.  The thin classdef layer in `runtime/optim_classdefs.m`
 * forwards every operator to a `matlab_optim_pb_*` builder here; each
 * builder appends a node and returns its id (as an f64).  `solve`
 * reduces the DAG: linear problems route to `linprog_core` /
 * `matlab_optim_intlinprog`, everything else to `al_minimize` with a
 * DAG-evaluation objective + nonlinear-constraint closure.
 *
 * Tier-4 scope: **scalar** optimisation variables.  Vector/matrix
 * `optimvar`, `eqnproblem`, `show`/`write`, `prob2struct` are deferred
 * — see the roadmap.
 * =================================================================== */

/* matlab_string ABI (mirrors the local declaration in runtime_pde.cpp). */
struct matlab_string_local { char *data; int64_t len; };

/* Cross-TU struct builders (defined in matlab_runtime.cpp). */
matlab_struct *matlab_struct_new(void);
void matlab_struct_set_f64(matlab_struct *s, const char *name,
                           int64_t len, double v);

enum PBKind {
    PBK_VAR, PBK_CONST, PBK_ADD, PBK_SUB, PBK_NEG,
    PBK_MUL, PBK_DIV, PBK_POW, PBK_LE, PBK_GE, PBK_EQ
};
struct PBVar { int is_int; double lb, ub; };
struct PBNode { int kind; int a; int b; double cval; int var; };

static std::vector<PBVar> g_pb_vars;
static std::vector<PBNode> g_pb_nodes;

static int pb_node(int kind, int a, int b, double cval, int var) {
    g_pb_nodes.push_back(PBNode{kind, a, b, cval, var});
    return (int)g_pb_nodes.size() - 1;
}

/* --- DAG builders (each returns a node id as an f64) ------------- */

/* `optimvar` → a scalar VAR node.  `is_int` selects integer vs
 * continuous.  Tier-4 supports scalar variables only; bounds are
 * expressed as ordinary `x >= lo` / `x <= hi` constraints.  The
 * variable's name is decorative in MATLAB and is not threaded through
 * the runtime — `solve` returns the solution as a plain column vector
 * in variable-creation order. */
double matlab_optim_pb_var(double is_int) {
    PBVar v;
    v.is_int = (int)is_int;
    v.lb = -INFINITY;
    v.ub = INFINITY;
    g_pb_vars.push_back(v);
    return (double)pb_node(PBK_VAR, -1, -1, 0.0, (int)g_pb_vars.size() - 1);
}
double matlab_optim_pb_const(double v) {
    return (double)pb_node(PBK_CONST, -1, -1, v, -1);
}
double matlab_optim_pb_add(double a, double b) {
    return (double)pb_node(PBK_ADD, (int)a, (int)b, 0.0, -1);
}
double matlab_optim_pb_sub(double a, double b) {
    return (double)pb_node(PBK_SUB, (int)a, (int)b, 0.0, -1);
}
double matlab_optim_pb_neg(double a) {
    return (double)pb_node(PBK_NEG, (int)a, -1, 0.0, -1);
}
double matlab_optim_pb_mul(double a, double b) {
    return (double)pb_node(PBK_MUL, (int)a, (int)b, 0.0, -1);
}
double matlab_optim_pb_div(double a, double b) {
    return (double)pb_node(PBK_DIV, (int)a, (int)b, 0.0, -1);
}
double matlab_optim_pb_pow(double a, double b) {
    return (double)pb_node(PBK_POW, (int)a, (int)b, 0.0, -1);
}
double matlab_optim_pb_le(double a, double b) {
    return (double)pb_node(PBK_LE, (int)a, (int)b, 0.0, -1);
}
double matlab_optim_pb_ge(double a, double b) {
    return (double)pb_node(PBK_GE, (int)a, (int)b, 0.0, -1);
}
double matlab_optim_pb_eq(double a, double b) {
    return (double)pb_node(PBK_EQ, (int)a, (int)b, 0.0, -1);
}
/* --- classdef-object field readers ------------------------------ *
 * `matlab_obj` and `matlab_struct` share the leading field layout
 * (see runtime_internal.h), so a single struct-shaped reader works on
 * both.  The problem-based classdef objects store their state in
 * named fields: an OptimizationExpression / OptimizationConstraint
 * carries an `Id` f64 (a DAG node id); an OptimizationProblem carries
 * `Objective` (ptr to an expression object), `Constraints` (a nested
 * struct of constraint objects), and an optional `Maximize` f64. */
static int pb_find_field(void *o, const char *name) {
    matlab_struct_s *s = (matlab_struct_s *)o;
    if (!s) return -1;
    for (int i = 0; i < s->nfields; ++i)
        if (s->names[i] && strcmp(s->names[i], name) == 0) return i;
    return -1;
}
static double pb_field_f64(void *o, const char *name) {
    int i = pb_find_field(o, name);
    if (i < 0) return 0.0;
    return ((matlab_struct_s *)o)->f64_vals[i];
}
static void *pb_field_ptr(void *o, const char *name) {
    int i = pb_find_field(o, name);
    if (i < 0) return NULL;
    return ((matlab_struct_s *)o)->ptr_vals[i];
}

/* Per-`solve` remap: a problem only ranges over the variables that
 * actually appear in its objective + constraints, not every variable
 * ever created.  `solve` fills `g_var_remap` (global var index →
 * dense local index, or −1 when the variable is not in this problem)
 * and `g_pb_nv` (the local variable count) before walking the DAG. */
static std::vector<int> g_var_remap;
static int g_pb_nv = 0;

/* Collect every variable index referenced under node `nid`. */
static void pb_collect_vars(int nid, std::vector<char> &used) {
    if (nid < 0 || nid >= (int)g_pb_nodes.size()) return;
    const PBNode &n = g_pb_nodes[nid];
    if (n.kind == PBK_VAR) {
        if (n.var >= 0 && n.var < (int)used.size()) used[n.var] = 1;
        return;
    }
    if (n.a >= 0) pb_collect_vars(n.a, used);
    if (n.b >= 0) pb_collect_vars(n.b, used);
}

/* --- DAG numeric evaluation -------------------------------------- *
 * Evaluates node `nid` at the local variable vector `x` (indexed by
 * `g_var_remap`).  Relation nodes (LE/GE/EQ) evaluate to the canonical
 * `g(x)` whose feasible form is `g(x) ≤ 0` (LE/GE) or `g(x) = 0` (EQ). */
static double pb_eval(int nid, const double *x) {
    if (nid < 0 || nid >= (int)g_pb_nodes.size()) return 0.0;
    const PBNode &n = g_pb_nodes[nid];
    switch (n.kind) {
        case PBK_VAR: {
            int li = (n.var >= 0 && n.var < (int)g_var_remap.size())
                         ? g_var_remap[n.var] : -1;
            return (li >= 0) ? x[li] : 0.0;
        }
        case PBK_CONST: return n.cval;
        case PBK_ADD:   return pb_eval(n.a, x) + pb_eval(n.b, x);
        case PBK_SUB:   return pb_eval(n.a, x) - pb_eval(n.b, x);
        case PBK_NEG:   return -pb_eval(n.a, x);
        case PBK_MUL:   return pb_eval(n.a, x) * pb_eval(n.b, x);
        case PBK_DIV:   return pb_eval(n.a, x) / pb_eval(n.b, x);
        case PBK_POW:   return pow(pb_eval(n.a, x), pb_eval(n.b, x));
        case PBK_LE:    return pb_eval(n.a, x) - pb_eval(n.b, x);  /* a-b ≤ 0 */
        case PBK_GE:    return pb_eval(n.b, x) - pb_eval(n.a, x);  /* b-a ≤ 0 */
        case PBK_EQ:    return pb_eval(n.a, x) - pb_eval(n.b, x);  /* a-b = 0 */
    }
    return 0.0;
}

/* --- DAG linear reduction --------------------------------------- *
 * If node `nid` is an affine function of the variables, fills
 * `c0` + `lin` (length = #vars) and returns true.  Relation nodes
 * reduce to their canonical `g(x)` form. */
static bool pb_reduce_linear(int nid, double &c0, std::vector<double> &lin) {
    int nv = g_pb_nv;
    if (nid < 0 || nid >= (int)g_pb_nodes.size()) return false;
    const PBNode &n = g_pb_nodes[nid];
    auto allZero = [](const std::vector<double> &v) {
        for (double e : v) if (e != 0.0) return false;
        return true;
    };
    switch (n.kind) {
        case PBK_CONST:
            c0 = n.cval; lin.assign(nv, 0.0); return true;
        case PBK_VAR: {
            c0 = 0.0; lin.assign(nv, 0.0);
            int li = (n.var >= 0 && n.var < (int)g_var_remap.size())
                         ? g_var_remap[n.var] : -1;
            if (li >= 0) lin[li] = 1.0;
            return true;
        }
        case PBK_NEG: {
            double ca; std::vector<double> la;
            if (!pb_reduce_linear(n.a, ca, la)) return false;
            c0 = -ca; lin.assign(nv, 0.0);
            for (int i = 0; i < nv; ++i) lin[i] = -la[i];
            return true;
        }
        case PBK_ADD: case PBK_SUB:
        case PBK_LE:  case PBK_EQ:  case PBK_GE: {
            double ca, cb; std::vector<double> la, lb;
            if (!pb_reduce_linear(n.a, ca, la)) return false;
            if (!pb_reduce_linear(n.b, cb, lb)) return false;
            lin.assign(nv, 0.0);
            if (n.kind == PBK_ADD) {
                c0 = ca + cb;
                for (int i = 0; i < nv; ++i) lin[i] = la[i] + lb[i];
            } else if (n.kind == PBK_GE) {  /* canonical g = b - a */
                c0 = cb - ca;
                for (int i = 0; i < nv; ++i) lin[i] = lb[i] - la[i];
            } else {                        /* SUB / LE / EQ: g = a - b */
                c0 = ca - cb;
                for (int i = 0; i < nv; ++i) lin[i] = la[i] - lb[i];
            }
            return true;
        }
        case PBK_MUL: {
            double ca, cb; std::vector<double> la, lb;
            if (!pb_reduce_linear(n.a, ca, la)) return false;
            if (!pb_reduce_linear(n.b, cb, lb)) return false;
            if (allZero(la)) {              /* const · linear */
                c0 = ca * cb; lin.assign(nv, 0.0);
                for (int i = 0; i < nv; ++i) lin[i] = ca * lb[i];
                return true;
            }
            if (allZero(lb)) {              /* linear · const */
                c0 = ca * cb; lin.assign(nv, 0.0);
                for (int i = 0; i < nv; ++i) lin[i] = la[i] * cb;
                return true;
            }
            return false;                   /* var · var → quadratic */
        }
        case PBK_DIV: {
            double ca, cb; std::vector<double> la, lb;
            if (!pb_reduce_linear(n.a, ca, la)) return false;
            if (!pb_reduce_linear(n.b, cb, lb)) return false;
            if (!allZero(lb) || cb == 0.0) return false;  /* divisor must be const */
            c0 = ca / cb; lin.assign(nv, 0.0);
            for (int i = 0; i < nv; ++i) lin[i] = la[i] / cb;
            return true;
        }
        case PBK_POW:
        default:
            return false;
    }
}

/* --- solve ------------------------------------------------------- *
 *
 * `sol = solve(prob)` — reads the OptimizationProblem classdef object,
 * reduces its objective + constraints, and dispatches.  Linear
 * objective + linear constraints route to the dense simplex
 * (`linprog_core`) or, with integer variables, the branch-and-bound
 * MILP solver; anything else goes through the augmented-Lagrangian
 * core with a DAG-evaluation objective and a nonlinear-constraint
 * closure.  Returns the solution as a column vector in
 * variable-creation order.  `prob` is the classdef object:
 * `Objective` is a ptr to an OptimizationExpression (its `Id` field
 * is the objective node), `Constraints` is a nested struct of
 * constraint node ids, and an optional `Maximize` f64 selects the
 * objective sense. */
matlab_mat *matlab_optim_pb_solve(void *prob_obj) {
    if (!prob_obj || g_pb_vars.empty()) return mat_alloc(0, 0);

    /* Objective node id (from prob.Objective's `Id` field). */
    int objn = -1;
    void *objExpr = pb_field_ptr(prob_obj, "Objective");
    if (objExpr) objn = (int)pb_field_f64(objExpr, "Id");
    int maximize = (int)pb_field_f64(prob_obj, "Maximize");

    /* Constraint node ids — walk the nested Constraints struct. */
    std::vector<int> consIds;
    void *consS = pb_field_ptr(prob_obj, "Constraints");
    if (consS) {
        matlab_struct_s *cs = (matlab_struct_s *)consS;
        for (int i = 0; i < cs->nfields; ++i) {
            if (cs->kinds[i] == 1 && cs->ptr_vals[i]) {
                /* a constraint object → its `Id` field is the node id */
                consIds.push_back((int)pb_field_f64(cs->ptr_vals[i], "Id"));
            } else if (cs->kinds[i] == 0) {
                /* a bare node-id f64 stored directly */
                consIds.push_back((int)cs->f64_vals[i]);
            }
        }
    }

    /* A problem ranges only over the variables that actually appear in
     * its objective + constraints — not every variable ever created
     * (a script may build several independent problems).  Collect that
     * set and build the global→local remap. */
    std::vector<char> usedFlag(g_pb_vars.size(), 0);
    pb_collect_vars(objn, usedFlag);
    for (int cnid : consIds) pb_collect_vars(cnid, usedFlag);
    std::vector<int> usedVars;
    for (int i = 0; i < (int)g_pb_vars.size(); ++i)
        if (usedFlag[i]) usedVars.push_back(i);
    int nv = (int)usedVars.size();
    if (nv == 0) return mat_alloc(0, 0);
    g_var_remap.assign(g_pb_vars.size(), -1);
    for (int li = 0; li < nv; ++li) g_var_remap[usedVars[li]] = li;
    g_pb_nv = nv;

    std::vector<double> L(nv), U(nv);
    bool anyInt = false;
    for (int li = 0; li < nv; ++li) {
        const PBVar &v = g_pb_vars[usedVars[li]];
        L[li] = v.lb;
        U[li] = v.ub;
        if (v.is_int) anyInt = true;
    }

    /* Reduce constraints: linear ones into A/b/Aeq/beq row lists, the
     * rest into a nonlinear-inequality node list. */
    std::vector<std::vector<double>> Arows, Aeqrows;
    std::vector<double> brhs, beqrhs;
    std::vector<int> nlIneq;
    bool allConsLinear = true;
    for (int cnid : consIds) {
        if (cnid < 0 || cnid >= (int)g_pb_nodes.size()) continue;
        int kind = g_pb_nodes[cnid].kind;
        double c0; std::vector<double> lin;
        if (pb_reduce_linear(cnid, c0, lin)) {
            /* canonical: g(x) = lin·x + c0  (≤ 0 for LE/GE, = 0 for EQ) */
            if (kind == PBK_EQ) { Aeqrows.push_back(lin); beqrhs.push_back(-c0); }
            else                { Arows.push_back(lin);   brhs.push_back(-c0);  }
        } else {
            allConsLinear = false;
            if (kind != PBK_EQ) nlIneq.push_back(cnid);
            /* nonlinear equality constraints are deferred (see roadmap) */
        }
    }

    double objC0 = 0.0;
    std::vector<double> objLin;
    bool objLinear = (objn >= 0) && pb_reduce_linear(objn, objC0, objLin);
    double sense = maximize ? -1.0 : 1.0;

    auto rows_to_mat = [](const std::vector<std::vector<double>> &R,
                          int ncols) -> matlab_mat * {
        if (R.empty()) return NULL;
        matlab_mat *m = mat_alloc((int64_t)R.size(), ncols);
        for (size_t i = 0; i < R.size(); ++i)
            for (int j = 0; j < ncols; ++j)
                m->data[i * ncols + j] = R[i][j];
        return m;
    };
    auto vec_to_mat = [](const std::vector<double> &v) -> matlab_mat * {
        if (v.empty()) return NULL;
        matlab_mat *m = mat_alloc((int64_t)v.size(), 1);
        for (size_t i = 0; i < v.size(); ++i) m->data[i] = v[i];
        return m;
    };

    matlab_mat *Am   = rows_to_mat(Arows, nv);
    matlab_mat *bm   = vec_to_mat(brhs);
    matlab_mat *Aeqm = rows_to_mat(Aeqrows, nv);
    matlab_mat *beqm = vec_to_mat(beqrhs);
    matlab_mat *lbm  = mat_alloc(nv, 1);
    matlab_mat *ubm  = mat_alloc(nv, 1);
    for (int i = 0; i < nv; ++i) { lbm->data[i] = L[i]; ubm->data[i] = U[i]; }

    std::vector<double> sol(nv, 0.0);

    if (objLinear && allConsLinear && anyInt) {
        /* MILP — linear objective + linear constraints + integers.
         * `linprog_core` (inside the branch-and-bound) needs finite
         * bounds, so clamp the default ±Inf to a wide finite box. */
        matlab_mat *fm = mat_alloc(nv, 1);
        for (int i = 0; i < nv; ++i) fm->data[i] = sense * objLin[i];
        matlab_mat *lbc = mat_alloc(nv, 1);
        matlab_mat *ubc = mat_alloc(nv, 1);
        for (int i = 0; i < nv; ++i) {
            lbc->data[i] = isfinite(L[i]) ? L[i] : -1.0e9;
            ubc->data[i] = isfinite(U[i]) ? U[i] :  1.0e9;
        }
        std::vector<double> idx;
        for (int li = 0; li < nv; ++li)
            if (g_pb_vars[usedVars[li]].is_int)
                idx.push_back((double)(li + 1));
        matlab_mat *intcon = vec_to_mat(idx);
        matlab_mat *r = matlab_optim_intlinprog(fm, intcon, Am, bm,
                                                Aeqm, beqm, lbc, ubc);
        if (r && r->rows * r->cols == nv)
            for (int i = 0; i < nv; ++i) sol[i] = r->data[i];
        free_mat(fm);
        free_mat(lbc);
        free_mat(ubc);
        free_mat(intcon);
        free_mat(r);
    } else {
        /* General path — augmented Lagrangian with a DAG-evaluation
         * objective and a DAG-evaluation nonlinear-constraint closure.
         * This serves LP (linear objective evaluated through the DAG),
         * QP, and fully nonlinear problems uniformly; the AL inner
         * solver handles ±Inf bounds (projection is the identity). */
        int objCopy = objn;
        double senseCopy = sense;
        AlObjFn obj = [objCopy, senseCopy, nv](const std::vector<double> &z,
                                               std::vector<double> *grad)
            -> double {
            if (objCopy < 0) {
                if (grad) grad->assign(nv, 0.0);
                return 0.0;
            }
            double f = senseCopy * pb_eval(objCopy, z.data());
            if (grad) {
                grad->assign(nv, 0.0);
                std::vector<double> zp = z;
                for (int i = 0; i < nv; ++i) {
                    double h = 1.0e-7 * (fabs(z[i]) + 1.0);
                    zp[i] = z[i] + h;
                    double fp = senseCopy * pb_eval(objCopy, zp.data());
                    zp[i] = z[i];
                    (*grad)[i] = (fp - f) / h;
                }
            }
            return f;
        };
        VecFn nonlcon;
        if (!nlIneq.empty()) {
            std::vector<int> nlCopy = nlIneq;
            nonlcon = [nlCopy](const std::vector<double> &z)
                -> std::vector<double> {
                std::vector<double> c(nlCopy.size());
                for (size_t k = 0; k < nlCopy.size(); ++k)
                    c[k] = pb_eval(nlCopy[k], z.data());
                return c;
            };
        }
        std::vector<double> x0(nv, 0.0);
        sol = al_minimize(obj, x0, Am, bm, Aeqm, beqm, lbm, ubm, nonlcon);
    }

    free_mat(Am);
    free_mat(bm);
    free_mat(Aeqm);
    free_mat(beqm);
    free_mat(lbm);
    free_mat(ubm);

    matlab_mat *out = mat_alloc(nv, 1);
    for (int i = 0; i < nv; ++i) out->data[i] = sol[i];
    return out;
}

/* --- solve (equation problem) ------------------------------------ *
 *
 * `sol = solve(prob)` for an EquationProblem — reads the `Equations`
 * nested struct off the classdef object, collects the equation
 * residual nodes (each `lhs == rhs` node evaluates to its canonical
 * `lhs − rhs`), and solves the system F(x) = 0 by Levenberg-Marquardt
 * (`lm_solve`).  Returns the solution as a column vector in
 * variable-creation order.  Tier-5; see docs/optim_toolbox_roadmap.md
 * §6. */
matlab_mat *matlab_optim_pb_solve_eqn(void *prob_obj) {
    if (!prob_obj || g_pb_vars.empty()) return mat_alloc(0, 0);

    /* Equation node ids — walk the nested Equations struct. */
    std::vector<int> eqIds;
    void *eqS = pb_field_ptr(prob_obj, "Equations");
    if (eqS) {
        matlab_struct_s *es = (matlab_struct_s *)eqS;
        for (int i = 0; i < es->nfields; ++i) {
            if (es->kinds[i] == 1 && es->ptr_vals[i])
                eqIds.push_back((int)pb_field_f64(es->ptr_vals[i], "Id"));
            else if (es->kinds[i] == 0)
                eqIds.push_back((int)es->f64_vals[i]);
        }
    }
    if (eqIds.empty()) return mat_alloc(0, 0);

    /* Collect referenced variables → global→local remap. */
    std::vector<char> usedFlag(g_pb_vars.size(), 0);
    for (int eid : eqIds) pb_collect_vars(eid, usedFlag);
    std::vector<int> usedVars;
    for (int i = 0; i < (int)g_pb_vars.size(); ++i)
        if (usedFlag[i]) usedVars.push_back(i);
    int nv = (int)usedVars.size();
    if (nv == 0) return mat_alloc(0, 0);
    g_var_remap.assign(g_pb_vars.size(), -1);
    for (int li = 0; li < nv; ++li) g_var_remap[usedVars[li]] = li;
    g_pb_nv = nv;

    /* Residual vector: each equation node evaluates to lhs − rhs, so
     * F(x) = 0 is exactly the system being solved. */
    std::vector<int> eqCopy = eqIds;
    VecFn residual = [eqCopy](const std::vector<double> &z)
        -> std::vector<double> {
        std::vector<double> r(eqCopy.size());
        for (size_t k = 0; k < eqCopy.size(); ++k)
            r[k] = pb_eval(eqCopy[k], z.data());
        return r;
    };
    std::vector<double> x0(nv, 0.0);
    std::vector<double> sol = lm_solve(residual, x0, NULL, NULL);

    matlab_mat *out = mat_alloc(nv, 1);
    for (int i = 0; i < nv; ++i) out->data[i] = sol[i];
    return out;
}

}  /* extern "C" */
