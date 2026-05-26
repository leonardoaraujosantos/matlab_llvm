/* ============================================================================
 * runtime_finance.cpp — Financial Toolbox runtime
 * ----------------------------------------------------------------------------
 * Tier-1: dates + cash flows + depreciation + bond pricing + technical
 * indicators (function form).  Tier-2: performance metrics + Black-Scholes
 * Greeks.  Tier-3: Portfolio classdef numeric kernels.
 *
 * Per docs/financial_toolbox_roadmap.md.  Everything here is closed-form
 * arithmetic over the shipped numeric base — no external dependency.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* The scalar datetime / mat allocators / epoch_to_civil / civil_to_epoch
 * helpers live in matlab_runtime.cpp.  Forward-declare what we need so
 * this TU compiles standalone. */
extern "C" {
    struct matlab_datetime_s;
    typedef struct matlab_datetime_s matlab_datetime;
    matlab_datetime *matlab_datetime_ymd(double y, double m, double d);
    /* Underlying layout shared with matlab_runtime.cpp. We need the
     * `seconds` field for date arithmetic; keep the definition in sync. */
}
struct matlab_datetime_s { double seconds; };

extern "C" {
    matlab_mat *mat_alloc(int64_t r, int64_t c);
}

/* epoch <-> civil — re-implement here so we don't depend on a static
 * helper in matlab_runtime.cpp.  These are Howard Hinnant's date-civil
 * algorithms, the same form the core runtime uses. */
static void fin_epoch_to_civil(double secs, int *y, int *m, int *d) {
    int64_t day = static_cast<int64_t>(floor(secs / 86400.0));
    day += 719468;
    int64_t era = (day >= 0 ? day : day - 146096) / 146097;
    int64_t doe = day - era * 146097;
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t Y = yoe + era * 400;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int64_t mp = (5 * doy + 2) / 153;
    int64_t D = doy - (153 * mp + 2) / 5 + 1;
    int64_t M = mp < 10 ? mp + 3 : mp - 9;
    if (M <= 2) Y++;
    *y = static_cast<int>(Y); *m = static_cast<int>(M); *d = static_cast<int>(D);
}
static double fin_civil_to_epoch(int y, int m, int d) {
    int Y = m <= 2 ? y - 1 : y;
    int era = (Y >= 0 ? Y : Y - 399) / 400;
    int yoe = Y - era * 400;
    int doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    int64_t day = static_cast<int64_t>(era) * 146097 + doe - 719468;
    return static_cast<double>(day) * 86400.0;
}
static int fin_weekday(double secs) {
    /* Unix epoch (1970-01-01) was a Thursday (=4). 0=Sun..6=Sat. */
    int64_t day = static_cast<int64_t>(floor(secs / 86400.0));
    int wd = static_cast<int>((day + 4) % 7);
    if (wd < 0) wd += 7;
    return wd;
}
static bool fin_is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

/* ============================================================================
 * §T1.1 — Date arithmetic
 * ==========================================================================*/

/* yearfrac(d1, d2, basis) — fractional years between d1 and d2 under the
 * given day-count basis.  Supported basis codes:
 *   0 = actual/actual  (days / 365.25 — standard approximation)
 *   1 = 30/360 SIA     (Bond Basis)
 *   2 = actual/360
 *   3 = actual/365
 *   6 = 30/360E        (Eurobond Basis)
 *   12 = BUS/252       (business days / 252)
 * Other codes fall back to actual/365.                                 */
static double yearfrac_core(double s1, double s2, int basis) {
    int y1, m1, d1, y2, m2, d2;
    fin_epoch_to_civil(s1, &y1, &m1, &d1);
    fin_epoch_to_civil(s2, &y2, &m2, &d2);
    double actual_days = (s2 - s1) / 86400.0;
    switch (basis) {
        case 0: return actual_days / 365.25;
        case 1: {
            int dd1 = d1, dd2 = d2;
            if (dd1 == 31) dd1 = 30;
            if (dd2 == 31 && dd1 == 30) dd2 = 30;
            double days = 360.0*(y2 - y1) + 30.0*(m2 - m1) + (dd2 - dd1);
            return days / 360.0;
        }
        case 2: return actual_days / 360.0;
        case 3: return actual_days / 365.0;
        case 6: {
            int dd1 = d1, dd2 = d2;
            if (dd1 == 31) dd1 = 30;
            if (dd2 == 31) dd2 = 30;
            double days = 360.0*(y2 - y1) + 30.0*(m2 - m1) + (dd2 - dd1);
            return days / 360.0;
        }
        case 12: {
            /* Business days between d1 and d2, divided by 252.  Walks
             * day-by-day skipping weekends; OK for spans up to a year. */
            double sign = s2 >= s1 ? 1.0 : -1.0;
            double lo = sign > 0 ? s1 : s2;
            double hi = sign > 0 ? s2 : s1;
            int64_t n = 0;
            for (double t = lo; t < hi; t += 86400.0) {
                int wd = fin_weekday(t);
                if (wd != 0 && wd != 6) n++;
            }
            return sign * static_cast<double>(n) / 252.0;
        }
    }
    return actual_days / 365.0;
}
extern "C" double matlab_yearfrac(matlab_datetime *a, matlab_datetime *b,
                                   double basis) {
    if (!a || !b) return 0.0;
    return yearfrac_core(a->seconds, b->seconds, static_cast<int>(basis));
}

/* daysdif(d1, d2, basis) — days under the basis convention. */
extern "C" double matlab_daysdif(matlab_datetime *a, matlab_datetime *b,
                                  double basis) {
    if (!a || !b) return 0.0;
    int B = static_cast<int>(basis);
    double actual = (b->seconds - a->seconds) / 86400.0;
    if (B == 1 || B == 6) {
        int y1, m1, d1, y2, m2, d2;
        fin_epoch_to_civil(a->seconds, &y1, &m1, &d1);
        fin_epoch_to_civil(b->seconds, &y2, &m2, &d2);
        int dd1 = d1, dd2 = d2;
        if (dd1 == 31) dd1 = 30;
        if (B == 6 && dd2 == 31) dd2 = 30;
        if (B == 1 && dd2 == 31 && dd1 == 30) dd2 = 30;
        return 360.0*(y2 - y1) + 30.0*(m2 - m1) + (dd2 - dd1);
    }
    return actual;
}
extern "C" double matlab_days360(matlab_datetime *a, matlab_datetime *b) {
    return matlab_daysdif(a, b, 1.0);
}
extern "C" double matlab_days365(matlab_datetime *a, matlab_datetime *b) {
    if (!a || !b) return 0.0;
    return (b->seconds - a->seconds) / 86400.0;
}
extern "C" double matlab_daysact(matlab_datetime *a, matlab_datetime *b) {
    return matlab_days365(a, b);
}

/* daysadd(d, n[, basis]) — add n days under the basis convention. */
extern "C" matlab_datetime *matlab_daysadd(matlab_datetime *a, double n,
                                            double /*basis*/) {
    matlab_datetime *r = reinterpret_cast<matlab_datetime *>(calloc(1, sizeof(*r)));
    r->seconds = (a ? a->seconds : 0.0) + n * 86400.0;
    return r;
}

/* isbusday(d) — 1.0 if d is Mon-Fri, 0.0 otherwise.  Simple weekday
 * calendar; full holiday list ships later. */
extern "C" double matlab_isbusday(matlab_datetime *d) {
    if (!d) return 0.0;
    int wd = fin_weekday(d->seconds);
    return (wd != 0 && wd != 6) ? 1.0 : 0.0;
}
/* busdate(d[, dir]) — next (dir > 0) or previous (dir < 0) business day. */
extern "C" matlab_datetime *matlab_busdate(matlab_datetime *d, double dir) {
    matlab_datetime *r = reinterpret_cast<matlab_datetime *>(calloc(1, sizeof(*r)));
    if (!d) return r;
    double step = dir < 0 ? -86400.0 : 86400.0;
    double t = d->seconds + step;
    for (int i = 0; i < 7; ++i) {
        int wd = fin_weekday(t);
        if (wd != 0 && wd != 6) { r->seconds = t; return r; }
        t += step;
    }
    r->seconds = t;
    return r;
}

/* eomdate(y, m) — last calendar day of the given (year, month). */
extern "C" matlab_datetime *matlab_eomdate(double y, double m) {
    matlab_datetime *r = reinterpret_cast<matlab_datetime *>(calloc(1, sizeof(*r)));
    int Y = static_cast<int>(y), M = static_cast<int>(m);
    static const int mdays[] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
    int mi = (M - 1) % 12; if (mi < 0) mi += 12;
    int d = mdays[mi];
    if (mi == 1 && fin_is_leap(Y)) d = 29;
    r->seconds = fin_civil_to_epoch(Y, M, d);
    return r;
}

/* lweekdate(weekday, y, m) — last weekday-of-month for that month.
 * weekday: 1=Sun..7=Sat (MATLAB convention).                       */
extern "C" matlab_datetime *matlab_lweekdate(double weekday, double y,
                                              double m) {
    matlab_datetime *eom = matlab_eomdate(y, m);
    int wd_target = static_cast<int>(weekday) - 1;       /* 0=Sun..6=Sat */
    if (wd_target < 0) wd_target = 0;
    if (wd_target > 6) wd_target = 6;
    double t = eom->seconds;
    free(eom);
    int wd = fin_weekday(t);
    int back = (wd - wd_target + 7) % 7;
    matlab_datetime *r = reinterpret_cast<matlab_datetime *>(calloc(1, sizeof(*r)));
    r->seconds = t - back * 86400.0;
    return r;
}
extern "C" matlab_datetime *matlab_fweekdate(double weekday, double y,
                                              double m) {
    int Y = static_cast<int>(y), M = static_cast<int>(m);
    double first = fin_civil_to_epoch(Y, M, 1);
    int wd_target = static_cast<int>(weekday) - 1;
    if (wd_target < 0) wd_target = 0;
    if (wd_target > 6) wd_target = 6;
    int wd = fin_weekday(first);
    int fwd = (wd_target - wd + 7) % 7;
    matlab_datetime *r = reinterpret_cast<matlab_datetime *>(calloc(1, sizeof(*r)));
    r->seconds = first + fwd * 86400.0;
    return r;
}

/* MATLAB <-> Excel date number conversion.  MATLAB serial date number
 * counts days from 1-Jan-0000 (MATLAB convention); Excel from 31-Dec-1899
 * (with the Lotus leap-bug — Feb 29 1900 is treated as real).  We use
 * the simple unix-epoch offset.                                       */
extern "C" double matlab_m2xdate(double mdate) {
    /* MATLAB date 1 = 1-Jan-0000; Excel date 1 = 1-Jan-1900.  Offset
     * = 693960 (MATLAB days to 1-Jan-1900).                          */
    return mdate - 693960.0;
}
extern "C" double matlab_x2mdate(double xdate) {
    return xdate + 693960.0;
}

/* ============================================================================
 * §T1.2 — Cash flow / time-value-of-money
 * ==========================================================================*/

/* pvfix(rate, periods, payment[, futurevalue][, due]) — PV of an
 * annuity.  due = 0 (end-of-period, default) or 1 (begin-of-period).
 * Sign convention: positive payment is outflow, PV is inflow. */
static double pv_ann_core(double rate, double n, double pmt, double fv,
                           int due) {
    if (rate == 0.0) return -pmt * n - fv;
    double pvifa = (1.0 - pow(1.0 + rate, -n)) / rate;
    if (due) pvifa *= (1.0 + rate);
    return -(pmt * pvifa + fv * pow(1.0 + rate, -n));
}
static double fv_ann_core(double rate, double n, double pmt, double pv,
                           int due) {
    if (rate == 0.0) return -(pv + pmt * n);
    double fvifa = (pow(1.0 + rate, n) - 1.0) / rate;
    if (due) fvifa *= (1.0 + rate);
    return -(pv * pow(1.0 + rate, n) + pmt * fvifa);
}
extern "C" double matlab_pvfix(double rate, double n, double pmt) {
    return pv_ann_core(rate, n, pmt, 0.0, 0);
}
extern "C" double matlab_fvfix(double rate, double n, double pmt) {
    return fv_ann_core(rate, n, pmt, 0.0, 0);
}

/* pvvar(cashflow, rate) — present value of a vector of cash flows
 * discounted at `rate` per period.  cashflow(1) is at t=0.           */
extern "C" double matlab_pvvar(matlab_mat *cf, double rate) {
    if (!cf || !cf->data) return 0.0;
    int64_t n = cf->rows * cf->cols;
    double pv = 0.0;
    double disc = 1.0;
    double oneplus = 1.0 + rate;
    for (int64_t i = 0; i < n; ++i) {
        pv += cf->data[i] / disc;
        disc *= oneplus;
    }
    return pv;
}
extern "C" double matlab_fvvar(matlab_mat *cf, double rate) {
    if (!cf || !cf->data) return 0.0;
    int64_t n = cf->rows * cf->cols;
    double fv = 0.0;
    double oneplus = 1.0 + rate;
    for (int64_t i = 0; i < n; ++i) {
        fv += cf->data[i] * pow(oneplus, static_cast<double>(n - 1 - i));
    }
    return fv;
}

/* irr(cashflow) — Newton-Raphson iteration on NPV(rate). */
extern "C" double matlab_irr(matlab_mat *cf) {
    if (!cf || !cf->data) return NAN;
    int64_t n = cf->rows * cf->cols;
    double r = 0.1;
    for (int it = 0; it < 100; ++it) {
        double npv = 0.0, dnpv = 0.0;
        double oneplus = 1.0 + r;
        double disc = 1.0;
        for (int64_t i = 0; i < n; ++i) {
            npv  += cf->data[i] / disc;
            dnpv -= static_cast<double>(i) * cf->data[i] / (disc * oneplus);
            disc *= oneplus;
        }
        if (fabs(dnpv) < 1e-12) break;
        double dr = npv / dnpv;
        r -= dr;
        if (fabs(dr) < 1e-10) return r;
        if (r <= -0.999) r = -0.999;
    }
    return r;
}

/* payper(rate, n, pv[, fv][, due]) — periodic payment on a loan
 * with PV = `pv`.  Defaults: fv=0, due=0.                             */
extern "C" double matlab_payper(double rate, double n, double pv) {
    if (rate == 0.0) return -pv / n;
    return -pv * rate / (1.0 - pow(1.0 + rate, -n));
}

/* amortize(rate, n, pv) — returns an n×4 matlab_mat with per-period:
 *   col 1: principal payment
 *   col 2: interest payment
 *   col 3: remaining balance after the payment
 *   col 4: cumulative interest paid
 * The runtime stores matlab_mat row-major (data[r*cols + c]).          */
extern "C" matlab_mat *matlab_amortize(double rate, double n, double pv) {
    int64_t N = static_cast<int64_t>(n);
    matlab_mat *m = mat_alloc(N, 4);
    if (!m || !m->data) return m;
    double pmt = -matlab_payper(rate, n, pv);   /* positive outflow */
    double bal = pv;
    double cumi = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        double interest = bal * rate;
        double principal = pmt - interest;
        bal -= principal;
        cumi += interest;
        m->data[i*4 + 0] = principal;
        m->data[i*4 + 1] = interest;
        m->data[i*4 + 2] = bal;
        m->data[i*4 + 3] = cumi;
    }
    return m;
}

/* nomrr / effrr — nominal <-> effective rate. */
extern "C" double matlab_nomrr(double eff, double n) {
    return n * (pow(1.0 + eff, 1.0 / n) - 1.0);
}
extern "C" double matlab_effrr(double nom, double n) {
    return pow(1.0 + nom / n, n) - 1.0;
}

/* ============================================================================
 * §T1.3 — Depreciation
 * ==========================================================================*/

/* depstln(cost, salvage, life) — straight-line.  Returns a 1xN vector
 * of equal periodic depreciation amounts.                              */
extern "C" matlab_mat *matlab_depstln(double cost, double salvage,
                                       double life) {
    int64_t N = static_cast<int64_t>(life);
    matlab_mat *m = mat_alloc(1, N);
    if (!m || !m->data || N <= 0) return m;
    double amt = (cost - salvage) / static_cast<double>(N);
    for (int64_t i = 0; i < N; ++i) m->data[i] = amt;
    return m;
}

/* depsoyd(cost, salvage, life) — sum-of-years-digits. */
extern "C" matlab_mat *matlab_depsoyd(double cost, double salvage,
                                       double life) {
    int64_t N = static_cast<int64_t>(life);
    matlab_mat *m = mat_alloc(1, N);
    if (!m || !m->data || N <= 0) return m;
    double sum = static_cast<double>(N) * (static_cast<double>(N) + 1.0) / 2.0;
    double base = cost - salvage;
    for (int64_t i = 0; i < N; ++i) {
        m->data[i] = base * static_cast<double>(N - i) / sum;
    }
    return m;
}

/* ============================================================================
 * §T3 — Portfolio classdef numeric kernels (Mean-Variance)
 *
 * The Portfolio object holds AssetMean (N×1), AssetCovar (N×N), bounds,
 * budget, and risk-free rate. We read these via matlab_obj_get_mat /
 * _get_f64 — declared in matlab_runtime.h.
 *
 * The frontier sweep solves N quadratic programs of the form
 *   min  w' C w   s.t.  m'w = r_target,  Σw = 1,  lb ≤ w ≤ ub
 * For the simple long-only fully-invested case we use a closed-form
 * 2-point sweep (corner -> max-Sharpe) plus a linear sweep between
 * the min-variance and max-return endpoints. This is the standard
 * Lagrangian solution when bounds are inactive; bound-active problems
 * defer to a small projected-gradient loop.
 * ==========================================================================*/

extern "C" double matlab_obj_get_f64(struct matlab_obj_s *o,
                                      const char *name, int64_t len);
extern "C" matlab_mat *matlab_obj_get_mat(struct matlab_obj_s *o,
                                           const char *name, int64_t len);
extern "C" void matlab_obj_set_f64(struct matlab_obj_s *o,
                                    const char *name, int64_t len, double v);

/* Forward declarations for the runtime-polymorphic dispatch. The
 * shared method names estimateFrontier / estimatePortRisk /
 * setDefaultConstraints route on the object's RiskKind discriminant
 * (0 = mean-variance, 1 = CVaR, 2 = MAD) so a single Spec-table
 * entry serves all three Portfolio classes. */
extern "C" matlab_mat *matlab_portfoliocvar_estimate_frontier(
        struct matlab_obj_s *p, double n_pts);
extern "C" double matlab_portfoliocvar_estimate_port_risk(
        struct matlab_obj_s *p, matlab_mat *w);
extern "C" struct matlab_obj_s *matlab_portfoliocvar_set_default(
        struct matlab_obj_s *p);
extern "C" matlab_mat *matlab_portfoliomad_estimate_frontier(
        struct matlab_obj_s *p, double n_pts);
extern "C" double matlab_portfoliomad_estimate_port_risk(
        struct matlab_obj_s *p, matlab_mat *w);
extern "C" void matlab_obj_set_mat(struct matlab_obj_s *o,
                                    const char *name, int64_t len,
                                    matlab_mat *m);

/* Setter helpers: each takes the Portfolio object plus the new value(s),
 * writes properties, and returns the object pointer for chaining. */
extern "C" struct matlab_obj_s *matlab_portfolio_set_asset_moments(
        struct matlab_obj_s *p, matlab_mat *m, matlab_mat *C) {
    if (!p) return p;
    matlab_obj_set_mat(p, "AssetMean",  9,  m);
    matlab_obj_set_mat(p, "AssetCovar", 10, C);
    if (m) matlab_obj_set_f64(p, "NumAssets", 9,
                               static_cast<double>(m->rows * m->cols));
    return p;
}
extern "C" struct matlab_obj_s *matlab_portfolio_set_bounds(
        struct matlab_obj_s *p, matlab_mat *lb, matlab_mat *ub) {
    if (!p) return p;
    matlab_obj_set_mat(p, "LowerBound", 10, lb);
    matlab_obj_set_mat(p, "UpperBound", 10, ub);
    return p;
}
extern "C" struct matlab_obj_s *matlab_portfolio_set_budget(
        struct matlab_obj_s *p, double lo, double hi) {
    if (!p) return p;
    matlab_obj_set_f64(p, "LowerBudget", 11, lo);
    matlab_obj_set_f64(p, "UpperBudget", 11, hi);
    return p;
}

/* setDefaultConstraints — long-only, fully invested.  N comes from
 * AssetMean which the user must have set first.                        */
extern "C" struct matlab_obj_s *matlab_portfolio_set_default_constraints(
        struct matlab_obj_s *p) {
    if (!p) return p;
    double kind = matlab_obj_get_f64(p, "RiskKind", 8);
    if (kind == 1.0 || kind == 2.0)
        return matlab_portfoliocvar_set_default(p);
    matlab_mat *m = matlab_obj_get_mat(p, "AssetMean", 9);
    if (!m || !m->data) return p;
    int64_t N = m->rows * m->cols;
    matlab_mat *lb = mat_alloc(N, 1);
    matlab_mat *ub = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) {
        lb->data[i] = 0.0;
        ub->data[i] = 1.0;
    }
    matlab_obj_set_mat(p, "LowerBound", 10, lb);
    matlab_obj_set_mat(p, "UpperBound", 10, ub);
    matlab_obj_set_f64(p, "LowerBudget", 11, 1.0);
    matlab_obj_set_f64(p, "UpperBudget", 11, 1.0);
    return p;
}

/* Helpers ---------------------------------------------------------------- */
static void matvec(const double *A, const double *x, double *y,
                    int64_t n) {
    /* A is row-major n×n; y = A x. */
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < n; ++j) s += A[i*n + j] * x[j];
        y[i] = s;
    }
}
static double dot(const double *x, const double *y, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

/* Solve a small 2×2 linear system. */
static bool solve2x2(double a, double b, double c, double d,
                      double rhs1, double rhs2,
                      double *out_x, double *out_y) {
    double det = a * d - b * c;
    if (fabs(det) < 1e-20) return false;
    *out_x = ( d * rhs1 - b * rhs2) / det;
    *out_y = (-c * rhs1 + a * rhs2) / det;
    return true;
}

/* Markowitz unconstrained (long/short, fully invested) frontier
 * weights for a target return r.  Classical Lagrangian:
 *   Let z1 = C^{-1} 1, z2 = C^{-1} m.
 *   A = 1' z1, B = 1' z2 = m' z1, D = m' z2.
 *   w(r) = (D - r B) / (A D - B^2) * z1 + (r A - B) / (A D - B^2) * z2
 * We solve the linear systems with a small Cholesky / forward-back-
 * substitution since C is positive-definite.
 *
 * In-place Cholesky decomposition: replaces lower triangle with L
 * such that C = L L'. Returns false on numerical failure.            */
static bool chol_factor(double *L, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            double s = L[i*n + j];
            for (int64_t k = 0; k < j; ++k) s -= L[i*n + k] * L[j*n + k];
            if (i == j) {
                if (s <= 0.0) return false;
                L[i*n + j] = sqrt(s);
            } else {
                L[i*n + j] = s / L[j*n + j];
            }
        }
        for (int64_t j = i + 1; j < n; ++j) L[i*n + j] = 0.0;
    }
    return true;
}
static void chol_solve(const double *L, const double *b, double *x,
                        int64_t n) {
    /* Forward: L y = b. */
    std::vector<double> y(n);
    for (int64_t i = 0; i < n; ++i) {
        double s = b[i];
        for (int64_t k = 0; k < i; ++k) s -= L[i*n + k] * y[k];
        y[i] = s / L[i*n + i];
    }
    /* Backward: L' x = y. */
    for (int64_t i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int64_t k = i + 1; k < n; ++k) s -= L[k*n + i] * x[k];
        x[i] = s / L[i*n + i];
    }
}

/* Project weights onto the box [lb, ub] then renormalise to satisfy
 * sum == target_budget. Iterates a few rounds; usually converges in
 * 2-3 passes for moderate bounds. */
static void project_to_bounds_budget(double *w, const double *lb,
                                      const double *ub, double budget,
                                      int64_t n) {
    for (int it = 0; it < 20; ++it) {
        for (int64_t i = 0; i < n; ++i) {
            if (w[i] < lb[i]) w[i] = lb[i];
            if (w[i] > ub[i]) w[i] = ub[i];
        }
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += w[i];
        if (fabs(s - budget) < 1e-12) return;
        /* Adjust unbounded entries to absorb the residual. */
        int64_t free_n = 0;
        for (int64_t i = 0; i < n; ++i)
            if (w[i] > lb[i] + 1e-12 && w[i] < ub[i] - 1e-12) free_n++;
        if (free_n == 0) {
            /* Distribute uniformly. */
            double delta = (budget - s) / static_cast<double>(n);
            for (int64_t i = 0; i < n; ++i) w[i] += delta;
        } else {
            double delta = (budget - s) / static_cast<double>(free_n);
            for (int64_t i = 0; i < n; ++i)
                if (w[i] > lb[i] + 1e-12 && w[i] < ub[i] - 1e-12)
                    w[i] += delta;
        }
    }
}

/* Compute frontier weights for a target return. Outputs Nx1 vector. */
static void portfolio_weights_for_return(const double *m, const double *C,
                                          const double *lb, const double *ub,
                                          double budget, double r_target,
                                          int64_t n, double *w_out) {
    /* Build a chol factor of C. */
    std::vector<double> L(n*n);
    for (int64_t i = 0; i < n*n; ++i) L[i] = C[i];
    if (!chol_factor(L.data(), n)) {
        /* Regularise by adding 1e-8 * I. */
        for (int64_t i = 0; i < n*n; ++i) L[i] = C[i];
        for (int64_t i = 0; i < n; ++i) L[i*n + i] += 1e-8;
        chol_factor(L.data(), n);
    }
    /* z1 = C^-1 * 1, z2 = C^-1 * m */
    std::vector<double> ones(n, 1.0);
    std::vector<double> z1(n), z2(n);
    chol_solve(L.data(), ones.data(), z1.data(), n);
    chol_solve(L.data(), m, z2.data(), n);
    double A = 0.0, B = 0.0, D = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        A += z1[i];
        B += m[i] * z1[i];
        D += m[i] * z2[i];
    }
    double det = A * D - B * B;
    if (fabs(det) < 1e-20) {
        /* Degenerate: equal-weight. */
        for (int64_t i = 0; i < n; ++i) w_out[i] = budget / static_cast<double>(n);
        return;
    }
    double a = (D - r_target * B) / det;
    double b = (r_target * A - B) / det;
    for (int64_t i = 0; i < n; ++i) w_out[i] = a * z1[i] + b * z2[i];
    /* Project to bounds + budget (if bounds active). */
    project_to_bounds_budget(w_out, lb, ub, budget, n);
}

/* matlab_portfolio_set_default(P) — populate bounds (long-only)
 * and budget (fully invested = 1) based on NumAssets. Returns the
 * object pointer for chaining. */
extern "C" struct matlab_obj_s *matlab_portfolio_set_default(
        struct matlab_obj_s *p) {
    /* No-op stub — the .m setter does the actual property writes. */
    return p;
}

/* matlab_portfolio_estimate_frontier(P, n) — return an n_assets × n
 * weight matrix sweeping target return from min-variance to max-mean.  */
extern "C" matlab_mat *matlab_portfolio_estimate_frontier(
        struct matlab_obj_s *p, double n_pts) {
    if (!p) return mat_alloc(0, 0);
    double kind = matlab_obj_get_f64(p, "RiskKind", 8);
    if (kind == 1.0) return matlab_portfoliocvar_estimate_frontier(p, n_pts);
    if (kind == 2.0) return matlab_portfoliomad_estimate_frontier(p, n_pts);
    matlab_mat *m_mean   = matlab_obj_get_mat(p, "AssetMean",  9);
    matlab_mat *m_cov    = matlab_obj_get_mat(p, "AssetCovar", 10);
    matlab_mat *m_lb     = matlab_obj_get_mat(p, "LowerBound", 10);
    matlab_mat *m_ub     = matlab_obj_get_mat(p, "UpperBound", 10);
    double budget = matlab_obj_get_f64(p, "LowerBudget", 11);
    if (!m_mean || !m_mean->data || !m_cov || !m_cov->data)
        return mat_alloc(0, 0);
    int64_t N = m_mean->rows * m_mean->cols;
    int64_t K = static_cast<int64_t>(n_pts);
    if (K < 2) K = 20;
    matlab_mat *W = mat_alloc(N, K);
    /* min/max returns within bounds. Use the raw min/max of AssetMean
     * as a simple range; a tighter range uses the bounded-LP answer. */
    double r_min = INFINITY, r_max = -INFINITY;
    for (int64_t i = 0; i < N; ++i) {
        double v = m_mean->data[i];
        if (v < r_min) r_min = v;
        if (v > r_max) r_max = v;
    }
    std::vector<double> lb(N, 0.0), ub(N, 1.0);
    if (m_lb && m_lb->data && (m_lb->rows * m_lb->cols) == N) {
        for (int64_t i = 0; i < N; ++i) lb[i] = m_lb->data[i];
    }
    if (m_ub && m_ub->data && (m_ub->rows * m_ub->cols) == N) {
        for (int64_t i = 0; i < N; ++i) ub[i] = m_ub->data[i];
    }
    std::vector<double> w(N);
    for (int64_t k = 0; k < K; ++k) {
        double t = K == 1 ? 0.0
                          : static_cast<double>(k) / static_cast<double>(K - 1);
        double r = r_min + t * (r_max - r_min);
        portfolio_weights_for_return(m_mean->data, m_cov->data,
                                      lb.data(), ub.data(),
                                      budget, r, N, w.data());
        /* Store column k of W (W is N×K, row-major). */
        for (int64_t i = 0; i < N; ++i) W->data[i*K + k] = w[i];
    }
    return W;
}

/* matlab_portfolio_estimate_port_moments(P, w) -> 1x2 [risk, return].
 * Risk = sqrt(w' C w); return = m' w.                                  */
extern "C" matlab_mat *matlab_portfolio_estimate_port_moments(
        struct matlab_obj_s *p, matlab_mat *w) {
    matlab_mat *out = mat_alloc(1, 2);
    if (!p || !w || !w->data) return out;
    matlab_mat *m_mean = matlab_obj_get_mat(p, "AssetMean",  9);
    matlab_mat *m_cov  = matlab_obj_get_mat(p, "AssetCovar", 10);
    if (!m_mean || !m_cov || !m_mean->data || !m_cov->data) return out;
    int64_t N = m_mean->rows * m_mean->cols;
    if ((w->rows * w->cols) != N) return out;
    std::vector<double> Cw(N);
    matvec(m_cov->data, w->data, Cw.data(), N);
    double risk = sqrt(dot(w->data, Cw.data(), N));
    double ret  = dot(m_mean->data, w->data, N);
    out->data[0] = risk;
    out->data[1] = ret;
    return out;
}

extern "C" double matlab_portfolio_estimate_port_return(
        struct matlab_obj_s *p, matlab_mat *w) {
    if (!p || !w || !w->data) return 0.0;
    matlab_mat *m_mean = matlab_obj_get_mat(p, "AssetMean", 9);
    if (!m_mean || !m_mean->data) return 0.0;
    int64_t N = m_mean->rows * m_mean->cols;
    if ((w->rows * w->cols) != N) return 0.0;
    return dot(m_mean->data, w->data, N);
}
extern "C" double matlab_portfolio_estimate_port_risk(
        struct matlab_obj_s *p, matlab_mat *w) {
    if (!p || !w || !w->data) return 0.0;
    double kind = matlab_obj_get_f64(p, "RiskKind", 8);
    if (kind == 1.0) return matlab_portfoliocvar_estimate_port_risk(p, w);
    if (kind == 2.0) return matlab_portfoliomad_estimate_port_risk(p, w);
    matlab_mat *m_cov = matlab_obj_get_mat(p, "AssetCovar", 10);
    if (!m_cov || !m_cov->data) return 0.0;
    int64_t N = w->rows * w->cols;
    std::vector<double> Cw(N);
    matvec(m_cov->data, w->data, Cw.data(), N);
    return sqrt(dot(w->data, Cw.data(), N));
}

/* matlab_portfolio_estimate_max_sharpe(P) — Nx1 weights at tangency.
 * Closed-form: w ∝ C^-1 (m - rf*1), then renormalise. Bounds active
 * via projection. */
extern "C" matlab_mat *matlab_portfolio_estimate_max_sharpe(
        struct matlab_obj_s *p) {
    if (!p) return mat_alloc(0, 0);
    matlab_mat *m_mean = matlab_obj_get_mat(p, "AssetMean",  9);
    matlab_mat *m_cov  = matlab_obj_get_mat(p, "AssetCovar", 10);
    matlab_mat *m_lb   = matlab_obj_get_mat(p, "LowerBound", 10);
    matlab_mat *m_ub   = matlab_obj_get_mat(p, "UpperBound", 10);
    double rf = matlab_obj_get_f64(p, "RiskFreeRate", 12);
    double budget = matlab_obj_get_f64(p, "LowerBudget", 11);
    if (!m_mean || !m_cov || !m_mean->data || !m_cov->data)
        return mat_alloc(0, 0);
    int64_t N = m_mean->rows * m_mean->cols;
    matlab_mat *w = mat_alloc(N, 1);
    std::vector<double> excess(N);
    for (int64_t i = 0; i < N; ++i) excess[i] = m_mean->data[i] - rf;
    std::vector<double> L(N*N);
    for (int64_t i = 0; i < N*N; ++i) L[i] = m_cov->data[i];
    if (!chol_factor(L.data(), N)) {
        for (int64_t i = 0; i < N*N; ++i) L[i] = m_cov->data[i];
        for (int64_t i = 0; i < N; ++i) L[i*N + i] += 1e-8;
        chol_factor(L.data(), N);
    }
    std::vector<double> tmp(N);
    chol_solve(L.data(), excess.data(), tmp.data(), N);
    double s = 0.0;
    for (int64_t i = 0; i < N; ++i) s += tmp[i];
    if (fabs(s) < 1e-20) {
        for (int64_t i = 0; i < N; ++i) w->data[i] = budget / static_cast<double>(N);
        return w;
    }
    for (int64_t i = 0; i < N; ++i) w->data[i] = tmp[i] * budget / s;
    std::vector<double> lb(N, 0.0), ub(N, 1.0);
    if (m_lb && m_lb->data && (m_lb->rows * m_lb->cols) == N)
        for (int64_t i = 0; i < N; ++i) lb[i] = m_lb->data[i];
    if (m_ub && m_ub->data && (m_ub->rows * m_ub->cols) == N)
        for (int64_t i = 0; i < N; ++i) ub[i] = m_ub->data[i];
    project_to_bounds_budget(w->data, lb.data(), ub.data(), budget, N);
    return w;
}

/* matlab_portfolio_estimate_frontier_by_return(P, r) — Nx1. */
extern "C" matlab_mat *matlab_portfolio_estimate_frontier_by_return(
        struct matlab_obj_s *p, double r) {
    if (!p) return mat_alloc(0, 0);
    matlab_mat *m_mean = matlab_obj_get_mat(p, "AssetMean",  9);
    matlab_mat *m_cov  = matlab_obj_get_mat(p, "AssetCovar", 10);
    matlab_mat *m_lb   = matlab_obj_get_mat(p, "LowerBound", 10);
    matlab_mat *m_ub   = matlab_obj_get_mat(p, "UpperBound", 10);
    double budget = matlab_obj_get_f64(p, "LowerBudget", 11);
    if (!m_mean || !m_cov || !m_mean->data || !m_cov->data)
        return mat_alloc(0, 0);
    int64_t N = m_mean->rows * m_mean->cols;
    matlab_mat *w = mat_alloc(N, 1);
    std::vector<double> lb(N, 0.0), ub(N, 1.0);
    if (m_lb && m_lb->data && (m_lb->rows * m_lb->cols) == N)
        for (int64_t i = 0; i < N; ++i) lb[i] = m_lb->data[i];
    if (m_ub && m_ub->data && (m_ub->rows * m_ub->cols) == N)
        for (int64_t i = 0; i < N; ++i) ub[i] = m_ub->data[i];
    portfolio_weights_for_return(m_mean->data, m_cov->data,
                                  lb.data(), ub.data(),
                                  budget, r, N, w->data);
    return w;
}

/* matlab_portfolio_estimate_asset_moments(P, X) — sample mean +
 * covariance of return matrix X (T rows × N cols). Returns Nx(N+1)
 * with the mean in column 0 and the covariance in columns 1..N. */
extern "C" matlab_mat *matlab_portfolio_estimate_asset_moments(
        struct matlab_obj_s * /*p*/, matlab_mat *X) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int64_t T = X->rows, N = X->cols;
    matlab_mat *out = mat_alloc(N, N + 1);
    std::vector<double> mean(N, 0.0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < N; ++j) mean[j] += X->data[t*N + j];
    for (int64_t j = 0; j < N; ++j) mean[j] /= static_cast<double>(T);
    for (int64_t j = 0; j < N; ++j) out->data[j*(N+1) + 0] = mean[j];
    /* Sample covariance: 1/(T-1) Σ (x_t - m)(x_t - m)'. */
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (int64_t t = 0; t < T; ++t) {
                double di = X->data[t*N + i] - mean[i];
                double dj = X->data[t*N + j] - mean[j];
                s += di * dj;
            }
            out->data[i*(N+1) + (1 + j)] = s / static_cast<double>(T - 1);
        }
    }
    return out;
}

/* ----- Tier-7 §1: close the half-wired Portfolio methods ----- */

/* Read the universe (mean, cov, bounds, budget) into caller buffers.
 * Returns N, or 0 on failure. */
static int64_t portfolio_read(struct matlab_obj_s *p,
                               const double **mean, const double **cov,
                               std::vector<double> &lb, std::vector<double> &ub,
                               double *budget) {
    matlab_mat *m_mean = matlab_obj_get_mat(p, "AssetMean",  9);
    matlab_mat *m_cov  = matlab_obj_get_mat(p, "AssetCovar", 10);
    matlab_mat *m_lb   = matlab_obj_get_mat(p, "LowerBound", 10);
    matlab_mat *m_ub   = matlab_obj_get_mat(p, "UpperBound", 10);
    *budget = matlab_obj_get_f64(p, "LowerBudget", 11);
    if (*budget <= 0.0) *budget = 1.0;
    if (!m_mean || !m_mean->data || !m_cov || !m_cov->data) return 0;
    int64_t N = m_mean->rows * m_mean->cols;
    *mean = m_mean->data;
    *cov  = m_cov->data;
    lb.assign(static_cast<size_t>(N), 0.0);
    ub.assign(static_cast<size_t>(N), 1.0);
    if (m_lb && m_lb->data && (m_lb->rows*m_lb->cols) == N)
        for (int64_t i = 0; i < N; ++i) lb[static_cast<size_t>(i)] = m_lb->data[i];
    if (m_ub && m_ub->data && (m_ub->rows*m_ub->cols) == N)
        for (int64_t i = 0; i < N; ++i) ub[static_cast<size_t>(i)] = m_ub->data[i];
    return N;
}

static double port_risk(const double *cov, const double *w, int64_t N) {
    std::vector<double> Cw(static_cast<size_t>(N));
    matvec(cov, w, Cw.data(), N);
    return sqrt(dot(w, Cw.data(), N));
}

/* estimateBounds(P) -> 1x2 [minReturn, maxReturn] attainable on the
 * frontier. minReturn = return of the min-variance portfolio;
 * maxReturn = max asset mean (the all-in-best-asset corner). */
extern "C" matlab_mat *matlab_portfolio_estimate_bounds(
        struct matlab_obj_s *p) {
    matlab_mat *out = mat_alloc(1, 2);
    if (!p) return out;
    const double *mean = nullptr, *cov = nullptr; double budget = 1.0;
    std::vector<double> lb, ub;
    int64_t N = portfolio_read(p, &mean, &cov, lb, ub, &budget);
    if (N == 0) return out;
    double r_min_asset = mean[0], r_max_asset = mean[0];
    for (int64_t i = 1; i < N; ++i) {
        if (mean[i] < r_min_asset) r_min_asset = mean[i];
        if (mean[i] > r_max_asset) r_max_asset = mean[i];
    }
    /* min-variance portfolio return: weights for the lowest target. */
    std::vector<double> w(static_cast<size_t>(N));
    portfolio_weights_for_return(mean, cov, lb.data(), ub.data(),
                                  budget, r_min_asset, N, w.data());
    out->data[0] = dot(mean, w.data(), N);   /* min-var portfolio return */
    out->data[1] = r_max_asset * budget;     /* max attainable */
    return out;
}

/* estimateFrontierByRisk(P, targetSigma) -> Nx1 weights whose risk is
 * closest to targetSigma. Bisects on the return target (risk is
 * monotone in return along the efficient frontier above min-var). */
extern "C" matlab_mat *matlab_portfolio_estimate_frontier_by_risk(
        struct matlab_obj_s *p, double target_sigma) {
    if (!p) return mat_alloc(0, 0);
    const double *mean = nullptr, *cov = nullptr; double budget = 1.0;
    std::vector<double> lb, ub;
    int64_t N = portfolio_read(p, &mean, &cov, lb, ub, &budget);
    matlab_mat *w = mat_alloc(N, 1);
    if (N == 0) return w;
    double r_lo = mean[0], r_hi = mean[0];
    for (int64_t i = 1; i < N; ++i) {
        if (mean[i] < r_lo) r_lo = mean[i];
        if (mean[i] > r_hi) r_hi = mean[i];
    }
    std::vector<double> wt(static_cast<size_t>(N));
    for (int it = 0; it < 60; ++it) {
        double r = 0.5 * (r_lo + r_hi);
        portfolio_weights_for_return(mean, cov, lb.data(), ub.data(),
                                      budget, r, N, wt.data());
        double risk = port_risk(cov, wt.data(), N);
        if (risk < target_sigma) r_lo = r; else r_hi = r;
    }
    portfolio_weights_for_return(mean, cov, lb.data(), ub.data(),
                                  budget, 0.5*(r_lo+r_hi), N, w->data);
    return w;
}

/* estimatePortFrontier(P, K) -> Kx2 [risk, return] frontier points. */
extern "C" matlab_mat *matlab_portfolio_estimate_frontier_points(
        struct matlab_obj_s *p, double n_pts) {
    if (!p) return mat_alloc(0, 0);
    const double *mean = nullptr, *cov = nullptr; double budget = 1.0;
    std::vector<double> lb, ub;
    int64_t N = portfolio_read(p, &mean, &cov, lb, ub, &budget);
    int64_t K = static_cast<int64_t>(n_pts);
    if (K < 2) K = 20;
    matlab_mat *pts = mat_alloc(K, 2);
    if (N == 0) return pts;
    double r_min = mean[0], r_max = mean[0];
    for (int64_t i = 1; i < N; ++i) {
        if (mean[i] < r_min) r_min = mean[i];
        if (mean[i] > r_max) r_max = mean[i];
    }
    std::vector<double> w(static_cast<size_t>(N));
    for (int64_t k = 0; k < K; ++k) {
        double t = K == 1 ? 0.0 : static_cast<double>(k)/static_cast<double>(K-1);
        double r = r_min + t * (r_max - r_min);
        portfolio_weights_for_return(mean, cov, lb.data(), ub.data(),
                                      budget, r, N, w.data());
        pts->data[k*2 + 0] = port_risk(cov, w.data(), N);
        pts->data[k*2 + 1] = dot(mean, w.data(), N);
    }
    return pts;
}

/* plotFrontier(P[, K]) — render the risk/return frontier curve via the
 * Cairo backend. Returns the Kx2 points so the value is still usable.
 * The render is WITH_PLOT-guarded; without plot it's a no-op compute. */
#ifdef MATLAB_LLVM_WITH_PLOT
extern "C" void matlab_plot2(matlab_mat *x, matlab_mat *y);
#endif
extern "C" matlab_mat *matlab_portfolio_plot_frontier(
        struct matlab_obj_s *p, double n_pts) {
    matlab_mat *pts = matlab_portfolio_estimate_frontier_points(p, n_pts);
#ifdef MATLAB_LLVM_WITH_PLOT
    if (pts && pts->data && pts->rows >= 2) {
        int64_t K = pts->rows;
        matlab_mat *xr = mat_alloc(K, 1);
        matlab_mat *yr = mat_alloc(K, 1);
        for (int64_t k = 0; k < K; ++k) {
            xr->data[k] = pts->data[k*2 + 0];
            yr->data[k] = pts->data[k*2 + 1];
        }
        matlab_plot2(xr, yr);
    }
#endif
    return pts;
}

/* ============================================================================
 * §T2.1 — Investment performance metrics
 *
 * All take an N-row return matlab_mat (any layout — flattened) and a few
 * scalar parameters. Return a scalar f64.  Conventions:
 *   - sharpe / sortino / inforatio are NOT annualised here; multiply by
 *     sqrt(periodsPerYear) at the call site if you want annualised.
 * ==========================================================================*/

static void mat_stats(matlab_mat *m, double *out_mean, double *out_std,
                       int64_t *out_n) {
    double s = 0.0, s2 = 0.0;
    int64_t n = m && m->data ? m->rows * m->cols : 0;
    for (int64_t i = 0; i < n; ++i) s += m->data[i];
    double mean = n > 0 ? s / static_cast<double>(n) : 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = m->data[i] - mean;
        s2 += d * d;
    }
    double sd = n > 1 ? sqrt(s2 / static_cast<double>(n - 1)) : 0.0;
    if (out_mean) *out_mean = mean;
    if (out_std)  *out_std  = sd;
    if (out_n)    *out_n    = n;
}

extern "C" double matlab_sharpe(matlab_mat *r, double rf) {
    double mean, sd; int64_t n;
    mat_stats(r, &mean, &sd, &n);
    if (sd == 0.0 || n == 0) return 0.0;
    return (mean - rf) / sd;
}

/* sortino: same numerator as sharpe but uses downside deviation
 * (std of negative excess returns only). */
extern "C" double matlab_sortino(matlab_mat *r, double mar) {
    if (!r || !r->data) return 0.0;
    int64_t n = r->rows * r->cols;
    if (n == 0) return 0.0;
    double s = 0.0, dsd = 0.0;
    int64_t nd = 0;
    for (int64_t i = 0; i < n; ++i) s += r->data[i];
    double mean = s / static_cast<double>(n);
    for (int64_t i = 0; i < n; ++i) {
        double e = r->data[i] - mar;
        if (e < 0.0) { dsd += e * e; nd++; }
    }
    if (nd == 0) return 0.0;
    double downside = sqrt(dsd / static_cast<double>(nd));
    if (downside == 0.0) return 0.0;
    return (mean - mar) / downside;
}

/* inforatio: mean(r - b) / std(r - b). */
extern "C" double matlab_inforatio(matlab_mat *r, matlab_mat *b) {
    if (!r || !r->data || !b || !b->data) return 0.0;
    int64_t n = r->rows * r->cols;
    int64_t m = b->rows * b->cols;
    int64_t k = n < m ? n : m;
    if (k == 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < k; ++i) s += r->data[i] - b->data[i];
    double mean = s / static_cast<double>(k);
    double s2 = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        double d = (r->data[i] - b->data[i]) - mean;
        s2 += d * d;
    }
    if (k < 2) return 0.0;
    double sd = sqrt(s2 / static_cast<double>(k - 1));
    if (sd == 0.0) return 0.0;
    return mean / sd;
}

/* tracking: std of active returns r - b.  Sample stddev. */
extern "C" double matlab_tracking(matlab_mat *r, matlab_mat *b) {
    if (!r || !r->data || !b || !b->data) return 0.0;
    int64_t n = r->rows * r->cols;
    int64_t m = b->rows * b->cols;
    int64_t k = n < m ? n : m;
    if (k < 2) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < k; ++i) s += r->data[i] - b->data[i];
    double mean = s / static_cast<double>(k);
    double s2 = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        double d = (r->data[i] - b->data[i]) - mean;
        s2 += d * d;
    }
    return sqrt(s2 / static_cast<double>(k - 1));
}

/* maxdrawdown(p) — peak-to-trough max drawdown of a price/equity
 * curve.  Returns a positive fraction (0.25 = 25% drawdown). */
extern "C" double matlab_maxdrawdown(matlab_mat *p) {
    if (!p || !p->data) return 0.0;
    int64_t n = p->rows * p->cols;
    if (n == 0) return 0.0;
    double peak = p->data[0];
    double mdd = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double v = p->data[i];
        if (v > peak) peak = v;
        double dd = peak > 0.0 ? (peak - v) / peak : 0.0;
        if (dd > mdd) mdd = dd;
    }
    return mdd;
}

/* lpm(r, MAR, order) — lower partial moment.  Average of
 * max(MAR - r, 0)^order. */
extern "C" double matlab_lpm(matlab_mat *r, double mar, double order) {
    if (!r || !r->data) return 0.0;
    int64_t n = r->rows * r->cols;
    if (n == 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double diff = mar - r->data[i];
        if (diff > 0.0) s += pow(diff, order);
    }
    return s / static_cast<double>(n);
}

/* portalpha(r, rb, rf) — Jensen's alpha against a benchmark.
 * Computes beta first via cov(r,b)/var(b), then alpha = mean(r) - rf -
 * beta * (mean(b) - rf). */
extern "C" double matlab_portalpha(matlab_mat *r, matlab_mat *b, double rf) {
    if (!r || !r->data || !b || !b->data) return 0.0;
    int64_t n = r->rows * r->cols;
    int64_t m = b->rows * b->cols;
    int64_t k = n < m ? n : m;
    if (k < 2) return 0.0;
    double sr = 0.0, sb = 0.0;
    for (int64_t i = 0; i < k; ++i) { sr += r->data[i]; sb += b->data[i]; }
    double mr = sr / static_cast<double>(k);
    double mb = sb / static_cast<double>(k);
    double cov = 0.0, varb = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        double dr = r->data[i] - mr;
        double db = b->data[i] - mb;
        cov  += dr * db;
        varb += db * db;
    }
    if (varb == 0.0) return 0.0;
    double beta = cov / varb;
    return mr - rf - beta * (mb - rf);
}

/* ============================================================================
 * §T2.2 — Black-Scholes Greeks
 *
 * Closed-form European-option formulas. We pull normcdf via the math
 * library (erfc-based — same convention as Stats).
 * ==========================================================================*/

static double bs_normcdf(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}
static double bs_normpdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}
static void bs_d1d2(double S, double X, double r, double T,
                     double sigma, double *out_d1, double *out_d2) {
    double sT = sigma * sqrt(T);
    double d1 = (log(S / X) + (r + 0.5 * sigma * sigma) * T) / sT;
    double d2 = d1 - sT;
    *out_d1 = d1;
    *out_d2 = d2;
}

/* blsprice(S, X, r, T, sigma) — European call price. */
extern "C" double matlab_blsprice(double S, double X, double r,
                                   double T, double sigma) {
    double d1, d2;
    bs_d1d2(S, X, r, T, sigma, &d1, &d2);
    return S * bs_normcdf(d1) - X * exp(-r * T) * bs_normcdf(d2);
}

/* blsdelta(S, X, r, T, sigma) — Δ of a European call (N(d1)). */
extern "C" double matlab_blsdelta(double S, double X, double r,
                                    double T, double sigma) {
    double d1, d2;
    bs_d1d2(S, X, r, T, sigma, &d1, &d2);
    return bs_normcdf(d1);
}

/* blsgamma(S, X, r, T, sigma) — Γ. */
extern "C" double matlab_blsgamma(double S, double X, double r,
                                    double T, double sigma) {
    double d1, d2;
    bs_d1d2(S, X, r, T, sigma, &d1, &d2);
    return bs_normpdf(d1) / (S * sigma * sqrt(T));
}

/* blsvega — sensitivity to volatility (per unit sigma). */
extern "C" double matlab_blsvega(double S, double X, double r,
                                   double T, double sigma) {
    double d1, d2;
    bs_d1d2(S, X, r, T, sigma, &d1, &d2);
    return S * bs_normpdf(d1) * sqrt(T);
}

/* blsrho — sensitivity to interest rate (call). */
extern "C" double matlab_blsrho(double S, double X, double r,
                                  double T, double sigma) {
    double d1, d2;
    bs_d1d2(S, X, r, T, sigma, &d1, &d2);
    return X * T * exp(-r * T) * bs_normcdf(d2);
}

/* blstheta — time decay (call), per year.  Negative number. */
extern "C" double matlab_blstheta(double S, double X, double r,
                                    double T, double sigma) {
    double d1, d2;
    bs_d1d2(S, X, r, T, sigma, &d1, &d2);
    double t1 = -(S * bs_normpdf(d1) * sigma) / (2.0 * sqrt(T));
    double t2 = -r * X * exp(-r * T) * bs_normcdf(d2);
    return t1 + t2;
}

/* blslambda — elasticity = Δ * S / Price. */
extern "C" double matlab_blslambda(double S, double X, double r,
                                     double T, double sigma) {
    double price = matlab_blsprice(S, X, r, T, sigma);
    if (price == 0.0) return 0.0;
    double delta = matlab_blsdelta(S, X, r, T, sigma);
    return delta * S / price;
}

/* blsimpv(S, X, r, T, P) — implied vol via Newton-Raphson on
 * (blsprice - P) at fixed sigma. */
extern "C" double matlab_blsimpv(double S, double X, double r,
                                   double T, double P) {
    double sigma = 0.2;
    for (int it = 0; it < 100; ++it) {
        double p = matlab_blsprice(S, X, r, T, sigma);
        double vega = matlab_blsvega(S, X, r, T, sigma);
        if (vega < 1e-12) break;
        double dsigma = (p - P) / vega;
        sigma -= dsigma;
        if (sigma < 1e-6) sigma = 1e-6;
        if (sigma > 5.0)  sigma = 5.0;
        if (fabs(dsigma) < 1e-8) return sigma;
    }
    return sigma;
}

/* ============================================================================
 * §T1.5 — Returns + technical indicators (function-form over matlab_mat)
 * ==========================================================================*/

/* tick2ret(prices) — simple per-period returns: r[i] = p[i+1]/p[i] - 1.
 * Output length is N-1 for an N-row input.  Treats the input as a flat
 * column.                                                                */
extern "C" matlab_mat *matlab_tick2ret(matlab_mat *p) {
    if (!p || !p->data) return mat_alloc(0, 0);
    int64_t n = p->rows * p->cols;
    if (n < 2) return mat_alloc(0, 0);
    matlab_mat *r = mat_alloc(n - 1, 1);
    for (int64_t i = 0; i < n - 1; ++i) {
        double d = p->data[i];
        r->data[i] = d != 0.0 ? p->data[i + 1] / d - 1.0 : 0.0;
    }
    return r;
}

/* ret2tick(returns[, start]) — cumulative price series from returns.
 * Output length = N+1 (the first element is the starting price). */
extern "C" matlab_mat *matlab_ret2tick(matlab_mat *r) {
    if (!r || !r->data) return mat_alloc(0, 0);
    int64_t n = r->rows * r->cols;
    matlab_mat *p = mat_alloc(n + 1, 1);
    p->data[0] = 1.0;
    for (int64_t i = 0; i < n; ++i)
        p->data[i + 1] = p->data[i] * (1.0 + r->data[i]);
    return p;
}

/* sma(x, N) — simple moving average over a rolling window. Leading
 * rows use a growing window (mean of available samples). */
extern "C" matlab_mat *matlab_sma(matlab_mat *x, double N_) {
    if (!x || !x->data) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    int64_t N = static_cast<int64_t>(N_);
    if (N <= 0) N = 1;
    matlab_mat *y = mat_alloc(n, 1);
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc += x->data[i];
        if (i >= N) acc -= x->data[i - N];
        int64_t window = i < N ? (i + 1) : N;
        y->data[i] = acc / static_cast<double>(window);
    }
    return y;
}

/* bolling(x, N, K) — Bollinger bands: returns an Nx3 matrix
 *   col 0: middle band (N-period SMA)
 *   col 1: upper band  (middle + K * stddev)
 *   col 2: lower band  (middle - K * stddev)
 * Stddev is over the same rolling window.                              */
extern "C" matlab_mat *matlab_bolling(matlab_mat *x, double N_, double K) {
    if (!x || !x->data) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    int64_t N = static_cast<int64_t>(N_);
    if (N <= 0) N = 1;
    matlab_mat *out = mat_alloc(n, 3);
    for (int64_t i = 0; i < n; ++i) {
        int64_t s = i >= N - 1 ? i - N + 1 : 0;
        int64_t w = i - s + 1;
        double sum = 0.0;
        for (int64_t k = s; k <= i; ++k) sum += x->data[k];
        double mean = sum / static_cast<double>(w);
        double var = 0.0;
        for (int64_t k = s; k <= i; ++k) {
            double d = x->data[k] - mean;
            var += d * d;
        }
        double sd = sqrt(var / static_cast<double>(w));
        out->data[i*3 + 0] = mean;
        out->data[i*3 + 1] = mean + K * sd;
        out->data[i*3 + 2] = mean - K * sd;
    }
    return out;
}

/* rsindex(x, N) — Wilder's RSI over an N-period window.  Uses a
 * simple moving average of gains/losses (not the EWMA variant).  */
extern "C" matlab_mat *matlab_rsindex(matlab_mat *x, double N_) {
    if (!x || !x->data) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    int64_t N = static_cast<int64_t>(N_);
    if (N <= 0) N = 14;
    matlab_mat *out = mat_alloc(n, 1);
    if (n == 0) return out;
    out->data[0] = 50.0;
    /* Walk the diff series, accumulate avg gain / avg loss in a
     * rolling window. */
    for (int64_t i = 1; i < n; ++i) {
        int64_t s = i >= N ? i - N + 1 : 1;
        double gain = 0.0, loss = 0.0;
        for (int64_t k = s; k <= i; ++k) {
            double d = x->data[k] - x->data[k - 1];
            if (d > 0.0) gain += d;
            else         loss += -d;
        }
        if (loss == 0.0) {
            out->data[i] = gain == 0.0 ? 50.0 : 100.0;
        } else {
            double rs = gain / loss;
            out->data[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    return out;
}

/* ============================================================================
 * §T1.4 — Bond pricing
 *
 * Simplified API: take periods + freq directly rather than settle/maturity
 * dates + day-count basis (the full date-based form lands as a follow-on
 * once cfdates/cfamounts wire the schedule generator).  Conventions:
 *   - face value = 100
 *   - yield + coupon are annualised; per-period rates are /freq
 *   - freq defaults to 2 (semi-annual)
 * ==========================================================================*/

extern "C" double matlab_bndprice(double yield_, double coupon,
                                   double periods, double freq) {
    double y = yield_ / freq;
    double c = coupon * 100.0 / freq;
    int64_t N = static_cast<int64_t>(periods);
    if (N <= 0) return 100.0;
    double pv = 0.0;
    double disc = 1.0;
    double oneplus = 1.0 + y;
    for (int64_t t = 1; t <= N; ++t) {
        disc *= oneplus;
        pv += c / disc;
    }
    pv += 100.0 / disc;     /* terminal redemption */
    return pv;
}

extern "C" double matlab_bndyield(double price, double coupon,
                                   double periods, double freq) {
    /* Newton-Raphson on bndprice(y) - price = 0. */
    double y = coupon;   /* seed with coupon rate */
    for (int it = 0; it < 100; ++it) {
        double p = matlab_bndprice(y, coupon, periods, freq);
        double pdy = matlab_bndprice(y + 1e-6, coupon, periods, freq);
        double f = p - price;
        double df = (pdy - p) / 1e-6;
        if (fabs(df) < 1e-12) break;
        double dy = f / df;
        y -= dy;
        if (fabs(dy) < 1e-10) break;
        if (y < -0.5) y = -0.5;
        if (y > 5.0)  y = 5.0;
    }
    return y;
}

/* bnddurp: returns a 1x2 mat [Macaulay, Modified] duration in years. */
extern "C" matlab_mat *matlab_bnddurp(double yield_, double coupon,
                                       double periods, double freq) {
    matlab_mat *m = mat_alloc(1, 2);
    if (!m || !m->data) return m;
    double y = yield_ / freq;
    double c = coupon * 100.0 / freq;
    int64_t N = static_cast<int64_t>(periods);
    double pv = 0.0, weighted = 0.0;
    double disc = 1.0;
    double oneplus = 1.0 + y;
    for (int64_t t = 1; t <= N; ++t) {
        disc *= oneplus;
        double cft = (t == N ? c + 100.0 : c);
        pv += cft / disc;
        weighted += static_cast<double>(t) * cft / disc;
    }
    double macaulay_periods = weighted / pv;
    double macaulay = macaulay_periods / freq;
    double modified = macaulay / (1.0 + yield_ / freq);
    m->data[0] = macaulay;
    m->data[1] = modified;
    return m;
}

/* bnddury: duration from yield (alias to bnddurp; MATLAB exposes both for
 * symmetry with price-from-yield / yield-from-price). */
extern "C" matlab_mat *matlab_bnddury(double yield_, double coupon,
                                       double periods, double freq) {
    return matlab_bnddurp(yield_, coupon, periods, freq);
}

/* bndconvp: convexity (years²).  C = Σ t(t+1) CF / (P (1+y)²) per period;
 * convert to year² by dividing by freq². */
extern "C" double matlab_bndconvp(double yield_, double coupon,
                                   double periods, double freq) {
    double y = yield_ / freq;
    double c = coupon * 100.0 / freq;
    int64_t N = static_cast<int64_t>(periods);
    double pv = 0.0, weighted = 0.0;
    double disc = 1.0;
    double oneplus = 1.0 + y;
    for (int64_t t = 1; t <= N; ++t) {
        disc *= oneplus;
        double cft = (t == N ? c + 100.0 : c);
        pv += cft / disc;
        double td = static_cast<double>(t);
        weighted += td * (td + 1.0) * cft / disc;
    }
    double conv = weighted / (pv * (1.0 + y) * (1.0 + y));
    return conv / (freq * freq);
}

/* accrfrac(daysSinceLastCoupon, daysInPeriod) — accrued-interest fraction
 * for the bond's current coupon period.  Trivial ratio; the MATLAB form
 * derives the day counts from settle / coupon dates + basis.            */
extern "C" double matlab_accrfrac(double days_since, double days_in_period) {
    if (days_in_period <= 0.0) return 0.0;
    return days_since / days_in_period;
}

/* ============================================================================
 * §T1.4 (cont.) — Treasury bills
 *
 * T-bills quote as a discount rate.  Price from discount:
 *   P = 100 * (1 - d * t / 360)            where t = days to maturity
 * Yield (CMT/coupon equivalent):
 *   y = (100 - P) * 365 / (P * t)
 * ==========================================================================*/

extern "C" double matlab_prdisc(double discount, double days_to_maturity) {
    return 100.0 * (1.0 - discount * days_to_maturity / 360.0);
}

extern "C" double matlab_prtbill(double discount, double days_to_maturity) {
    return matlab_prdisc(discount, days_to_maturity);
}

extern "C" double matlab_ytbill(double price, double days_to_maturity) {
    if (price <= 0.0 || days_to_maturity <= 0.0) return 0.0;
    return (100.0 - price) * 365.0 / (price * days_to_maturity);
}

extern "C" double matlab_beytbill(double price, double days_to_maturity) {
    return matlab_ytbill(price, days_to_maturity);
}

/* depfixdb(cost, salvage, life, period[, month]) — fixed declining-
 * balance.  Rate = 1 - (salvage/cost)^(1/life).                         */
extern "C" matlab_mat *matlab_depfixdb(double cost, double salvage,
                                        double life) {
    int64_t N = static_cast<int64_t>(life);
    matlab_mat *m = mat_alloc(1, N);
    if (!m || !m->data || N <= 0 || cost <= 0.0) return m;
    double rate = 1.0 - pow(salvage / cost, 1.0 / static_cast<double>(N));
    double bal = cost;
    for (int64_t i = 0; i < N; ++i) {
        double d = bal * rate;
        m->data[i] = d;
        bal -= d;
    }
    return m;
}

/* ============================================================================
 * §T4.1 — Multivariate normal regression + ECM (missing data)
 *
 * ecmnmle / ecmncov estimate the mean + covariance of an N×d data matrix
 * that may contain NaN entries, via the Expectation-Conditional-
 * Maximisation (ECM) algorithm:
 *   E-step: for each row, fill missing components with their conditional
 *           expectation given the observed components (current θ).
 *   M-step: re-estimate mean (column mean of filled data) and covariance
 *           (sample cov of filled data + the summed conditional-covariance
 *           correction for the imputed blocks).
 * The correction term is what makes this a proper MLE rather than naive
 * mean-imputation — without it the covariance is biased toward zero.
 * ==========================================================================*/

/* Solve the small SPD system A x = b for the observed-block sub-matrix.
 * A is `k×k` row-major. Uses the chol_factor/chol_solve helpers above. */
static bool spd_solve(const double *A, const double *b, double *x,
                       int64_t k) {
    std::vector<double> L(static_cast<size_t>(k*k));
    for (int64_t i = 0; i < k*k; ++i) L[static_cast<size_t>(i)] = A[i];
    if (!chol_factor(L.data(), k)) {
        for (int64_t i = 0; i < k*k; ++i) L[static_cast<size_t>(i)] = A[i];
        for (int64_t i = 0; i < k; ++i) L[static_cast<size_t>(i*k + i)] += 1e-10;
        if (!chol_factor(L.data(), k)) return false;
    }
    chol_solve(L.data(), b, x, k);
    return true;
}

/* Core ECM. data is N×d row-major (NaN = missing). Writes the estimated
 * mean (length d) into `mean_out` and the d×d covariance (row-major) into
 * `cov_out`. */
static void ecm_core(const double *data, int64_t N, int64_t d,
                      double *mean_out, double *cov_out) {
    /* Initialise mean = column nanmean; cov = diagonal nanvar. */
    std::vector<double> mean(static_cast<size_t>(d), 0.0);
    std::vector<double> cov(static_cast<size_t>(d*d), 0.0);
    for (int64_t j = 0; j < d; ++j) {
        double s = 0.0; int64_t c = 0;
        for (int64_t i = 0; i < N; ++i) {
            double v = data[i*d + j];
            if (v == v) { s += v; c++; }
        }
        mean[static_cast<size_t>(j)] = c > 0 ? s / static_cast<double>(c) : 0.0;
    }
    for (int64_t j = 0; j < d; ++j) {
        double s2 = 0.0; int64_t c = 0;
        for (int64_t i = 0; i < N; ++i) {
            double v = data[i*d + j];
            if (v == v) { double dd = v - mean[static_cast<size_t>(j)]; s2 += dd*dd; c++; }
        }
        cov[static_cast<size_t>(j*d + j)] = c > 1 ? s2 / static_cast<double>(c - 1) : 1.0;
    }

    std::vector<double> filled(static_cast<size_t>(N*d));
    for (int it = 0; it < 100; ++it) {
        /* Accumulators for the M-step. */
        std::vector<double> new_mean(static_cast<size_t>(d), 0.0);
        std::vector<double> corr(static_cast<size_t>(d*d), 0.0);
        for (int64_t i = 0; i < N; ++i) {
            /* Partition this row into observed (O) and missing (M). */
            std::vector<int64_t> O, Mi;
            for (int64_t j = 0; j < d; ++j) {
                if (data[i*d + j] == data[i*d + j]) O.push_back(j);
                else Mi.push_back(j);
            }
            /* Copy observed values. */
            for (int64_t j : O) filled[static_cast<size_t>(i*d + j)] = data[i*d + j];
            if (!Mi.empty() && !O.empty()) {
                int64_t k = static_cast<int64_t>(O.size());
                int64_t mlen = static_cast<int64_t>(Mi.size());
                /* Sigma_OO (k×k), Sigma_MO (mlen×k). */
                std::vector<double> Soo(static_cast<size_t>(k*k));
                std::vector<double> dev(static_cast<size_t>(k));
                for (int64_t a = 0; a < k; ++a) {
                    dev[static_cast<size_t>(a)] = data[i*d + O[static_cast<size_t>(a)]]
                                  - mean[static_cast<size_t>(O[static_cast<size_t>(a)])];
                    for (int64_t b = 0; b < k; ++b)
                        Soo[static_cast<size_t>(a*k + b)] =
                            cov[static_cast<size_t>(O[static_cast<size_t>(a)]*d + O[static_cast<size_t>(b)])];
                }
                /* w = Sigma_OO^-1 dev. */
                std::vector<double> w(static_cast<size_t>(k), 0.0);
                spd_solve(Soo.data(), dev.data(), w.data(), k);
                /* For each missing index: cond mean = mu_M + Sigma_MO w. */
                for (int64_t r = 0; r < mlen; ++r) {
                    int64_t mr = Mi[static_cast<size_t>(r)];
                    double cm = mean[static_cast<size_t>(mr)];
                    for (int64_t a = 0; a < k; ++a)
                        cm += cov[static_cast<size_t>(mr*d + O[static_cast<size_t>(a)])] * w[static_cast<size_t>(a)];
                    filled[static_cast<size_t>(i*d + mr)] = cm;
                }
                /* Conditional covariance correction: Sigma_MM -
                 * Sigma_MO Sigma_OO^-1 Sigma_OM, added to corr block. */
                for (int64_t r = 0; r < mlen; ++r) {
                    int64_t mr = Mi[static_cast<size_t>(r)];
                    for (int64_t s = 0; s < mlen; ++s) {
                        int64_t ms = Mi[static_cast<size_t>(s)];
                        /* z = Sigma_OO^-1 Sigma_O,ms */
                        std::vector<double> rhs(static_cast<size_t>(k)), z(static_cast<size_t>(k), 0.0);
                        for (int64_t a = 0; a < k; ++a)
                            rhs[static_cast<size_t>(a)] = cov[static_cast<size_t>(O[static_cast<size_t>(a)]*d + ms)];
                        spd_solve(Soo.data(), rhs.data(), z.data(), k);
                        double red = 0.0;
                        for (int64_t a = 0; a < k; ++a)
                            red += cov[static_cast<size_t>(mr*d + O[static_cast<size_t>(a)])] * z[static_cast<size_t>(a)];
                        corr[static_cast<size_t>(mr*d + ms)] +=
                            cov[static_cast<size_t>(mr*d + ms)] - red;
                    }
                }
            } else if (!Mi.empty()) {
                /* No observed components — fill with the marginal mean. */
                for (int64_t j : Mi) filled[static_cast<size_t>(i*d + j)] = mean[static_cast<size_t>(j)];
            }
        }
        /* M-step: new mean. */
        for (int64_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < N; ++i) s += filled[static_cast<size_t>(i*d + j)];
            new_mean[static_cast<size_t>(j)] = s / static_cast<double>(N);
        }
        /* M-step: new cov = sample cov of filled + corr / N. */
        std::vector<double> new_cov(static_cast<size_t>(d*d), 0.0);
        for (int64_t a = 0; a < d; ++a) {
            for (int64_t b = 0; b < d; ++b) {
                double s = 0.0;
                for (int64_t i = 0; i < N; ++i) {
                    double da = filled[static_cast<size_t>(i*d + a)] - new_mean[static_cast<size_t>(a)];
                    double db = filled[static_cast<size_t>(i*d + b)] - new_mean[static_cast<size_t>(b)];
                    s += da * db;
                }
                new_cov[static_cast<size_t>(a*d + b)] =
                    (s + corr[static_cast<size_t>(a*d + b)]) / static_cast<double>(N);
            }
        }
        /* Convergence check on the mean. */
        double delta = 0.0;
        for (int64_t j = 0; j < d; ++j)
            delta += fabs(new_mean[static_cast<size_t>(j)] - mean[static_cast<size_t>(j)]);
        mean = new_mean;
        cov  = new_cov;
        if (delta < 1e-10) break;
    }
    for (int64_t j = 0; j < d; ++j) mean_out[j] = mean[static_cast<size_t>(j)];
    for (int64_t i = 0; i < d*d; ++i) cov_out[i] = cov[static_cast<size_t>(i)];
}

/* ecmnmle(Data) -> d×1 ECM mean estimate (NaN-aware). */
extern "C" matlab_mat *matlab_ecmnmle(matlab_mat *X) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int64_t N = X->rows, d = X->cols;
    matlab_mat *out = mat_alloc(d, 1);
    std::vector<double> cov(static_cast<size_t>(d*d));
    ecm_core(X->data, N, d, out->data, cov.data());
    return out;
}
/* ecmncov(Data) -> d×d ECM covariance estimate (NaN-aware). */
extern "C" matlab_mat *matlab_ecmncov(matlab_mat *X) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int64_t N = X->rows, d = X->cols;
    matlab_mat *out = mat_alloc(d, d);
    std::vector<double> mean(static_cast<size_t>(d));
    ecm_core(X->data, N, d, mean.data(), out->data);
    return out;
}

/* mvnrmle(Y, X) -> regression coefficients via OLS/MLE (no missing).
 * Y is N×1, X is N×p; returns p×1 beta = (X'X)^-1 X'Y.                  */
extern "C" matlab_mat *matlab_mvnrmle(matlab_mat *Y, matlab_mat *X) {
    if (!Y || !Y->data || !X || !X->data) return mat_alloc(0, 0);
    int64_t N = X->rows, p = X->cols;
    matlab_mat *beta = mat_alloc(p, 1);
    /* Normal equations: (X'X) beta = X'Y. */
    std::vector<double> XtX(static_cast<size_t>(p*p), 0.0);
    std::vector<double> XtY(static_cast<size_t>(p), 0.0);
    for (int64_t a = 0; a < p; ++a) {
        for (int64_t b = 0; b < p; ++b) {
            double s = 0.0;
            for (int64_t i = 0; i < N; ++i) s += X->data[i*p + a] * X->data[i*p + b];
            XtX[static_cast<size_t>(a*p + b)] = s;
        }
        double sy = 0.0;
        for (int64_t i = 0; i < N; ++i) sy += X->data[i*p + a] * Y->data[i];
        XtY[static_cast<size_t>(a)] = sy;
    }
    spd_solve(XtX.data(), XtY.data(), beta->data, p);
    return beta;
}

/* capm(assetReturns, marketReturns, rf) -> 2×1 [alpha; beta] for a
 * single asset.  Regress excess asset return on excess market return. */
extern "C" matlab_mat *matlab_capm(matlab_mat *asset, matlab_mat *market,
                                    double rf) {
    matlab_mat *out = mat_alloc(2, 1);
    if (!asset || !asset->data || !market || !market->data) return out;
    int64_t n = asset->rows * asset->cols;
    int64_t m = market->rows * market->cols;
    int64_t k = n < m ? n : m;
    if (k < 2) return out;
    double sx = 0.0, sy = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        sx += market->data[i] - rf;
        sy += asset->data[i]  - rf;
    }
    double mx = sx / static_cast<double>(k);
    double my = sy / static_cast<double>(k);
    double cov = 0.0, varx = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        double dx = (market->data[i] - rf) - mx;
        double dy = (asset->data[i]  - rf) - my;
        cov  += dx * dy;
        varx += dx * dx;
    }
    double beta = varx > 0.0 ? cov / varx : 0.0;
    double alpha = my - beta * mx;
    out->data[0] = alpha;
    out->data[1] = beta;
    return out;
}

/* ============================================================================
 * §T4.2 — Credit transition probabilities + CDS bootstrap
 * ==========================================================================*/

/* transprob(counts) — cohort-method transition probability matrix from a
 * square count matrix. Each row is normalised to sum to 1; an all-zero
 * row maps to the identity row (absorbing). Counts is N×N row-major. */
extern "C" matlab_mat *matlab_transprob(matlab_mat *counts) {
    if (!counts || !counts->data) return mat_alloc(0, 0);
    int64_t n = counts->rows;
    matlab_mat *P = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i) {
        double row = 0.0;
        for (int64_t j = 0; j < n; ++j) row += counts->data[i*n + j];
        if (row <= 0.0) {
            for (int64_t j = 0; j < n; ++j) P->data[i*n + j] = (i == j) ? 1.0 : 0.0;
        } else {
            for (int64_t j = 0; j < n; ++j) P->data[i*n + j] = counts->data[i*n + j] / row;
        }
    }
    return P;
}

/* cdsbootstrap(zeroRates, spreads, times, recovery) -> N×1 survival
 * probabilities. Piecewise-constant-hazard bootstrap (JP-Morgan style):
 * for each maturity, solve the par condition premium-leg == protection-
 * leg for P_i given P_1..P_{i-1}. zeroRates / spreads / times are
 * column vectors of length N; spreads are decimal (200 bp = 0.02).     */
extern "C" matlab_mat *matlab_cdsbootstrap(matlab_mat *zeroRates,
                                            matlab_mat *spreads,
                                            matlab_mat *times,
                                            double recovery) {
    if (!zeroRates || !spreads || !times ||
        !zeroRates->data || !spreads->data || !times->data)
        return mat_alloc(0, 0);
    int64_t n = times->rows * times->cols;
    matlab_mat *surv = mat_alloc(n, 1);
    double R = recovery;
    std::vector<double> P(static_cast<size_t>(n + 1));
    std::vector<double> DF(static_cast<size_t>(n + 1));
    std::vector<double> T(static_cast<size_t>(n + 1));
    P[0] = 1.0; T[0] = 0.0; DF[0] = 1.0;
    for (int64_t i = 1; i <= n; ++i) {
        T[static_cast<size_t>(i)]  = times->data[i - 1];
        DF[static_cast<size_t>(i)] = exp(-zeroRates->data[i - 1] * T[static_cast<size_t>(i)]);
    }
    for (int64_t i = 1; i <= n; ++i) {
        double s = spreads->data[i - 1];
        double premium_known = 0.0, protection_known = 0.0;
        for (int64_t j = 1; j < i; ++j) {
            double dt = T[static_cast<size_t>(j)] - T[static_cast<size_t>(j-1)];
            premium_known    += s * DF[static_cast<size_t>(j)] * dt * P[static_cast<size_t>(j)];
            protection_known += (1.0 - R) * DF[static_cast<size_t>(j)] *
                                (P[static_cast<size_t>(j-1)] - P[static_cast<size_t>(j)]);
        }
        double dti = T[static_cast<size_t>(i)] - T[static_cast<size_t>(i-1)];
        double A = s * DF[static_cast<size_t>(i)] * dti;            /* coeff of P_i in premium */
        double B = (1.0 - R) * DF[static_cast<size_t>(i)];         /* coeff in protection */
        double Pi = (protection_known + B * P[static_cast<size_t>(i-1)] - premium_known) / (A + B);
        if (Pi < 0.0) Pi = 0.0;
        if (Pi > P[static_cast<size_t>(i-1)]) Pi = P[static_cast<size_t>(i-1)];
        P[static_cast<size_t>(i)] = Pi;
        surv->data[i - 1] = Pi;
    }
    return surv;
}

/* cdsspread(hazard, recovery) — credit-triangle par spread for a flat
 * hazard rate: s ≈ hazard * (1 - recovery). */
extern "C" double matlab_cdsspread(double hazard, double recovery) {
    return hazard * (1.0 - recovery);
}

/* cdsprice(spreadMarket, spreadContract, rpv01) — mark-to-market value
 * of an existing CDS position (per unit notional, protection buyer):
 *   MTM = (s_market - s_contract) * RPV01
 * RPV01 is the risky annuity (caller-supplied or from a curve).        */
extern "C" double matlab_cdsprice(double s_market, double s_contract,
                                   double rpv01) {
    return (s_market - s_contract) * rpv01;
}

/* ============================================================================
 * §T4.3 — Credit scorecard (logistic-regression core)
 *
 * Simplified scorecard: logistic regression on the raw predictors (the
 * WoE/IV binning transform is a documented follow-on). The classdef
 * carries X (N×p predictors), Y (N×1 default flags 0/1), and the fitted
 * Beta ((p+1)×1, intercept first). fitmodel runs IRLS; probdefault and
 * score evaluate on new data.
 * ==========================================================================*/

/* IRLS logistic fit with an auto-prepended intercept. X is N×p row-major;
 * y is N×1. Writes (p+1)×1 beta (beta[0] = intercept). */
static void logistic_irls(const double *X, const double *y,
                           int64_t N, int64_t p, double *beta) {
    int64_t k = p + 1;                  /* +1 for intercept */
    for (int64_t j = 0; j < k; ++j) beta[j] = 0.0;
    std::vector<double> XtWX(static_cast<size_t>(k*k));
    std::vector<double> XtWz(static_cast<size_t>(k));
    std::vector<double> row(static_cast<size_t>(k));
    for (int it = 0; it < 50; ++it) {
        for (int64_t a = 0; a < k*k; ++a) XtWX[static_cast<size_t>(a)] = 0.0;
        for (int64_t a = 0; a < k; ++a)   XtWz[static_cast<size_t>(a)] = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            row[0] = 1.0;
            for (int64_t j = 0; j < p; ++j) row[static_cast<size_t>(j+1)] = X[i*p + j];
            double eta = 0.0;
            for (int64_t j = 0; j < k; ++j) eta += beta[j] * row[static_cast<size_t>(j)];
            double mu = 1.0 / (1.0 + exp(-eta));
            double w  = mu * (1.0 - mu);
            if (w < 1e-9) w = 1e-9;
            double z = eta + (y[i] - mu) / w;     /* working response */
            for (int64_t a = 0; a < k; ++a) {
                XtWz[static_cast<size_t>(a)] += row[static_cast<size_t>(a)] * w * z;
                for (int64_t b = 0; b < k; ++b)
                    XtWX[static_cast<size_t>(a*k + b)] += row[static_cast<size_t>(a)] * w * row[static_cast<size_t>(b)];
            }
        }
        std::vector<double> nb(static_cast<size_t>(k), 0.0);
        if (!spd_solve(XtWX.data(), XtWz.data(), nb.data(), k)) break;
        double delta = 0.0;
        for (int64_t j = 0; j < k; ++j) delta += fabs(nb[static_cast<size_t>(j)] - beta[j]);
        for (int64_t j = 0; j < k; ++j) beta[j] = nb[static_cast<size_t>(j)];
        if (delta < 1e-10) break;
    }
}

extern "C" struct matlab_obj_s *matlab_creditscorecard_fitmodel(
        struct matlab_obj_s *sc) {
    if (!sc) return sc;
    matlab_mat *X = matlab_obj_get_mat(sc, "X", 1);
    matlab_mat *Y = matlab_obj_get_mat(sc, "Y", 1);
    if (!X || !X->data || !Y || !Y->data) return sc;
    int64_t N = X->rows, p = X->cols;
    matlab_mat *beta = mat_alloc(p + 1, 1);
    logistic_irls(X->data, Y->data, N, p, beta->data);
    matlab_obj_set_mat(sc, "Beta", 4, beta);
    return sc;
}

/* probdefault(sc, Xnew) -> M×1 default probabilities (sigmoid of the
 * linear predictor). */
extern "C" matlab_mat *matlab_creditscorecard_probdefault(
        struct matlab_obj_s *sc, matlab_mat *Xnew) {
    if (!sc || !Xnew || !Xnew->data) return mat_alloc(0, 0);
    matlab_mat *beta = matlab_obj_get_mat(sc, "Beta", 4);
    if (!beta || !beta->data) return mat_alloc(0, 0);
    int64_t M = Xnew->rows, p = Xnew->cols;
    matlab_mat *pd = mat_alloc(M, 1);
    for (int64_t i = 0; i < M; ++i) {
        double eta = beta->data[0];
        for (int64_t j = 0; j < p; ++j) eta += beta->data[j + 1] * Xnew->data[i*p + j];
        pd->data[i] = 1.0 / (1.0 + exp(-eta));
    }
    return pd;
}

/* score(sc, Xnew) -> M×1 log-odds (the credit "score" before any
 * points-scaling transform). */
extern "C" matlab_mat *matlab_creditscorecard_score(
        struct matlab_obj_s *sc, matlab_mat *Xnew) {
    if (!sc || !Xnew || !Xnew->data) return mat_alloc(0, 0);
    matlab_mat *beta = matlab_obj_get_mat(sc, "Beta", 4);
    if (!beta || !beta->data) return mat_alloc(0, 0);
    int64_t M = Xnew->rows, p = Xnew->cols;
    matlab_mat *sco = mat_alloc(M, 1);
    for (int64_t i = 0; i < M; ++i) {
        double eta = beta->data[0];
        for (int64_t j = 0; j < p; ++j) eta += beta->data[j + 1] * Xnew->data[i*p + j];
        sco->data[i] = eta;
    }
    return sco;
}

/* ============================================================================
 * §T5.1 — PortfolioCVaR (scenario-based Conditional Value-at-Risk)
 *
 * The CVaR of a weight vector over S scenarios is the mean of the worst
 * (1-alpha) fraction of portfolio losses (loss = -scenario·w). This is a
 * convex function of w, so the frontier is computed by projected
 * subgradient descent rather than pulling in a full LP solver:
 *   g = -(1/k) Σ_{s in tail} scenario_s
 * with alternating projection onto the budget+box set and (for the
 * return-targeted frontier) the mean-return hyperplane.
 * ==========================================================================*/

#include <algorithm>

static double cvar_of_weights(const double *scen, int64_t S, int64_t N,
                               const double *w, double alpha,
                               double *out_var) {
    std::vector<double> loss(static_cast<size_t>(S));
    for (int64_t s = 0; s < S; ++s) {
        double r = 0.0;
        for (int64_t j = 0; j < N; ++j) r += scen[s*N + j] * w[j];
        loss[static_cast<size_t>(s)] = -r;                 /* loss = -return */
    }
    std::sort(loss.begin(), loss.end());           /* ascending */
    int64_t k = static_cast<int64_t>(ceil((1.0 - alpha) * static_cast<double>(S)));
    if (k < 1) k = 1;
    if (k > S) k = S;
    /* worst k losses are the largest k (tail of the sorted array). */
    double tail = 0.0;
    for (int64_t i = S - k; i < S; ++i) tail += loss[static_cast<size_t>(i)];
    if (out_var) *out_var = loss[static_cast<size_t>(S - k)];  /* VaR ~ quantile */
    return tail / static_cast<double>(k);
}

extern "C" double matlab_portfoliocvar_estimate_port_risk(
        struct matlab_obj_s *p, matlab_mat *w) {
    if (!p || !w || !w->data) return 0.0;
    matlab_mat *scen = matlab_obj_get_mat(p, "Scenarios", 9);
    double alpha = matlab_obj_get_f64(p, "ProbabilityLevel", 16);
    if (!scen || !scen->data) return 0.0;
    int64_t S = scen->rows, N = scen->cols;
    if ((w->rows * w->cols) != N) return 0.0;
    return cvar_of_weights(scen->data, S, N, w->data, alpha, nullptr);
}
extern "C" double matlab_portfoliocvar_estimate_port_var(
        struct matlab_obj_s *p, matlab_mat *w) {
    if (!p || !w || !w->data) return 0.0;
    matlab_mat *scen = matlab_obj_get_mat(p, "Scenarios", 9);
    double alpha = matlab_obj_get_f64(p, "ProbabilityLevel", 16);
    if (!scen || !scen->data) return 0.0;
    int64_t S = scen->rows, N = scen->cols;
    double var = 0.0;
    cvar_of_weights(scen->data, S, N, w->data, alpha, &var);
    return var;
}

/* scenario mean return per asset (column means of the scenario matrix). */
static void scenario_mean(const double *scen, int64_t S, int64_t N,
                           double *mean) {
    for (int64_t j = 0; j < N; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < S; ++i) s += scen[i*N + j];
        mean[j] = s / static_cast<double>(S);
    }
}

/* Minimise CVaR for a target return via projected subgradient. */
static void cvar_min_for_return(const double *scen, int64_t S, int64_t N,
                                 const double *mean, const double *lb,
                                 const double *ub, double budget,
                                 double r_target, bool use_return,
                                 double alpha, double *w_out) {
    for (int64_t j = 0; j < N; ++j) w_out[j] = budget / static_cast<double>(N);
    std::vector<double> loss(static_cast<size_t>(S));
    std::vector<std::pair<double,int64_t>> idx(static_cast<size_t>(S));
    double step = 0.5;
    for (int it = 0; it < 400; ++it) {
        for (int64_t s = 0; s < S; ++s) {
            double r = 0.0;
            for (int64_t j = 0; j < N; ++j) r += scen[s*N + j] * w_out[j];
            idx[static_cast<size_t>(s)] = { -r, s };
        }
        std::sort(idx.begin(), idx.end());
        int64_t k = static_cast<int64_t>(ceil((1.0 - alpha) * static_cast<double>(S)));
        if (k < 1) k = 1;
        /* subgradient = -(1/k) Σ_{tail} scenario_s */
        std::vector<double> g(static_cast<size_t>(N), 0.0);
        for (int64_t t = S - k; t < S; ++t) {
            int64_t s = idx[static_cast<size_t>(t)].second;
            for (int64_t j = 0; j < N; ++j) g[static_cast<size_t>(j)] -= scen[s*N + j];
        }
        for (int64_t j = 0; j < N; ++j) g[static_cast<size_t>(j)] /= static_cast<double>(k);
        double sk = step / (1.0 + 0.01 * it);
        for (int64_t j = 0; j < N; ++j) w_out[j] -= sk * g[static_cast<size_t>(j)];
        /* Alternating projection: return hyperplane, then budget+box. */
        for (int proj = 0; proj < 3; ++proj) {
            if (use_return) {
                double mw = 0.0, mm = 0.0;
                for (int64_t j = 0; j < N; ++j) { mw += mean[j]*w_out[j]; mm += mean[j]*mean[j]; }
                if (mm > 1e-20) {
                    double adj = (r_target - mw) / mm;
                    for (int64_t j = 0; j < N; ++j) w_out[j] += adj * mean[j];
                }
            }
            project_to_bounds_budget(w_out, lb, ub, budget, N);
        }
    }
}

extern "C" matlab_mat *matlab_portfoliocvar_estimate_frontier(
        struct matlab_obj_s *p, double n_pts) {
    if (!p) return mat_alloc(0, 0);
    matlab_mat *scen = matlab_obj_get_mat(p, "Scenarios", 9);
    matlab_mat *m_lb = matlab_obj_get_mat(p, "LowerBound", 10);
    matlab_mat *m_ub = matlab_obj_get_mat(p, "UpperBound", 10);
    double alpha = matlab_obj_get_f64(p, "ProbabilityLevel", 16);
    double budget = matlab_obj_get_f64(p, "LowerBudget", 11);
    if (budget <= 0.0) budget = 1.0;
    if (!scen || !scen->data) return mat_alloc(0, 0);
    int64_t S = scen->rows, N = scen->cols;
    int64_t K = static_cast<int64_t>(n_pts);
    if (K < 2) K = 10;
    std::vector<double> mean(static_cast<size_t>(N));
    scenario_mean(scen->data, S, N, mean.data());
    std::vector<double> lb(static_cast<size_t>(N), 0.0), ub(static_cast<size_t>(N), 1.0);
    if (m_lb && m_lb->data && (m_lb->rows*m_lb->cols) == N)
        for (int64_t i = 0; i < N; ++i) lb[static_cast<size_t>(i)] = m_lb->data[i];
    if (m_ub && m_ub->data && (m_ub->rows*m_ub->cols) == N)
        for (int64_t i = 0; i < N; ++i) ub[static_cast<size_t>(i)] = m_ub->data[i];
    double r_min = *std::min_element(mean.begin(), mean.end());
    double r_max = *std::max_element(mean.begin(), mean.end());
    matlab_mat *W = mat_alloc(N, K);
    std::vector<double> w(static_cast<size_t>(N));
    for (int64_t kk = 0; kk < K; ++kk) {
        double t = K == 1 ? 0.0 : static_cast<double>(kk)/static_cast<double>(K-1);
        double r = r_min + t * (r_max - r_min);
        cvar_min_for_return(scen->data, S, N, mean.data(), lb.data(),
                             ub.data(), budget, r, true, alpha, w.data());
        for (int64_t i = 0; i < N; ++i) W->data[i*K + kk] = w[static_cast<size_t>(i)];
    }
    return W;
}

/* Setters. */
extern "C" struct matlab_obj_s *matlab_portfoliocvar_set_scenarios(
        struct matlab_obj_s *p, matlab_mat *S) {
    if (!p) return p;
    matlab_obj_set_mat(p, "Scenarios", 9, S);
    if (S) matlab_obj_set_f64(p, "NumAssets", 9, static_cast<double>(S->cols));
    return p;
}
extern "C" struct matlab_obj_s *matlab_portfoliocvar_set_prob_level(
        struct matlab_obj_s *p, double alpha) {
    if (!p) return p;
    matlab_obj_set_f64(p, "ProbabilityLevel", 16, alpha);
    return p;
}
extern "C" struct matlab_obj_s *matlab_portfoliocvar_set_default(
        struct matlab_obj_s *p) {
    if (!p) return p;
    matlab_mat *S = matlab_obj_get_mat(p, "Scenarios", 9);
    if (!S || !S->data) return p;
    int64_t N = S->cols;
    matlab_mat *lb = mat_alloc(N, 1), *ub = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) { lb->data[i] = 0.0; ub->data[i] = 1.0; }
    matlab_obj_set_mat(p, "LowerBound", 10, lb);
    matlab_obj_set_mat(p, "UpperBound", 10, ub);
    matlab_obj_set_f64(p, "LowerBudget", 11, 1.0);
    matlab_obj_set_f64(p, "UpperBudget", 11, 1.0);
    return p;
}

/* ============================================================================
 * §T5.2 — PortfolioMAD (Mean-Absolute-Deviation, Konno-Yamazaki)
 *
 * MAD risk of a weight vector over S scenarios:
 *   MAD(w) = (1/S) Σ_s | r_s·w - mean_s(r·w) |
 * Convex in w; minimised by projected subgradient with the same
 * return-hyperplane + budget/box alternating projection as CVaR.
 * ==========================================================================*/

static double mad_of_weights(const double *scen, int64_t S, int64_t N,
                              const double *w) {
    std::vector<double> pr(static_cast<size_t>(S));
    double mean = 0.0;
    for (int64_t s = 0; s < S; ++s) {
        double r = 0.0;
        for (int64_t j = 0; j < N; ++j) r += scen[s*N + j] * w[j];
        pr[static_cast<size_t>(s)] = r;
        mean += r;
    }
    mean /= static_cast<double>(S);
    double mad = 0.0;
    for (int64_t s = 0; s < S; ++s) mad += fabs(pr[static_cast<size_t>(s)] - mean);
    return mad / static_cast<double>(S);
}

extern "C" double matlab_portfoliomad_estimate_port_risk(
        struct matlab_obj_s *p, matlab_mat *w) {
    if (!p || !w || !w->data) return 0.0;
    matlab_mat *scen = matlab_obj_get_mat(p, "Scenarios", 9);
    if (!scen || !scen->data) return 0.0;
    int64_t S = scen->rows, N = scen->cols;
    if ((w->rows * w->cols) != N) return 0.0;
    return mad_of_weights(scen->data, S, N, w->data);
}

extern "C" matlab_mat *matlab_portfoliomad_estimate_frontier(
        struct matlab_obj_s *p, double n_pts) {
    if (!p) return mat_alloc(0, 0);
    matlab_mat *scen = matlab_obj_get_mat(p, "Scenarios", 9);
    matlab_mat *m_lb = matlab_obj_get_mat(p, "LowerBound", 10);
    matlab_mat *m_ub = matlab_obj_get_mat(p, "UpperBound", 10);
    double budget = matlab_obj_get_f64(p, "LowerBudget", 11);
    if (budget <= 0.0) budget = 1.0;
    if (!scen || !scen->data) return mat_alloc(0, 0);
    int64_t S = scen->rows, N = scen->cols;
    int64_t K = static_cast<int64_t>(n_pts);
    if (K < 2) K = 10;
    std::vector<double> mean(static_cast<size_t>(N));
    scenario_mean(scen->data, S, N, mean.data());
    std::vector<double> lb(static_cast<size_t>(N), 0.0), ub(static_cast<size_t>(N), 1.0);
    if (m_lb && m_lb->data && (m_lb->rows*m_lb->cols) == N)
        for (int64_t i = 0; i < N; ++i) lb[static_cast<size_t>(i)] = m_lb->data[i];
    if (m_ub && m_ub->data && (m_ub->rows*m_ub->cols) == N)
        for (int64_t i = 0; i < N; ++i) ub[static_cast<size_t>(i)] = m_ub->data[i];
    double r_min = *std::min_element(mean.begin(), mean.end());
    double r_max = *std::max_element(mean.begin(), mean.end());
    const double *sd = scen->data;
    matlab_mat *W = mat_alloc(N, K);
    std::vector<double> w(static_cast<size_t>(N));
    std::vector<double> pr(static_cast<size_t>(S));
    for (int64_t kk = 0; kk < K; ++kk) {
        double t = K == 1 ? 0.0 : static_cast<double>(kk)/static_cast<double>(K-1);
        double r_target = r_min + t * (r_max - r_min);
        for (int64_t j = 0; j < N; ++j) w[static_cast<size_t>(j)] = budget / static_cast<double>(N);
        for (int it = 0; it < 400; ++it) {
            /* portfolio returns + mean */
            double pmean = 0.0;
            for (int64_t s = 0; s < S; ++s) {
                double rr = 0.0;
                for (int64_t j = 0; j < N; ++j) rr += sd[s*N + j] * w[static_cast<size_t>(j)];
                pr[static_cast<size_t>(s)] = rr; pmean += rr;
            }
            pmean /= static_cast<double>(S);
            /* subgradient of MAD: (1/S) Σ sign(pr_s - pmean) (scen_s - mean) */
            std::vector<double> g(static_cast<size_t>(N), 0.0);
            for (int64_t s = 0; s < S; ++s) {
                double sgn = (pr[static_cast<size_t>(s)] - pmean) >= 0.0 ? 1.0 : -1.0;
                for (int64_t j = 0; j < N; ++j)
                    g[static_cast<size_t>(j)] += sgn * (sd[s*N + j] - mean[static_cast<size_t>(j)]);
            }
            for (int64_t j = 0; j < N; ++j) g[static_cast<size_t>(j)] /= static_cast<double>(S);
            double sk = 0.5 / (1.0 + 0.01 * it);
            for (int64_t j = 0; j < N; ++j) w[static_cast<size_t>(j)] -= sk * g[static_cast<size_t>(j)];
            for (int proj = 0; proj < 3; ++proj) {
                double mw = 0.0, mm = 0.0;
                for (int64_t j = 0; j < N; ++j) { mw += mean[static_cast<size_t>(j)]*w[static_cast<size_t>(j)]; mm += mean[static_cast<size_t>(j)]*mean[static_cast<size_t>(j)]; }
                if (mm > 1e-20) {
                    double adj = (r_target - mw) / mm;
                    for (int64_t j = 0; j < N; ++j) w[static_cast<size_t>(j)] += adj * mean[static_cast<size_t>(j)];
                }
                project_to_bounds_budget(w.data(), lb.data(), ub.data(), budget, N);
            }
        }
        for (int64_t i = 0; i < N; ++i) W->data[i*K + kk] = w[static_cast<size_t>(i)];
    }
    return W;
}

extern "C" struct matlab_obj_s *matlab_portfoliomad_set_scenarios(
        struct matlab_obj_s *p, matlab_mat *S) {
    return matlab_portfoliocvar_set_scenarios(p, S);
}

/* ============================================================================
 * §T5.3 — Backtest engine (function-form)
 *
 * A leaner backtest than MATLAB's handle-driven backtestStrategy /
 * backtestEngine: backtest a FIXED target weight vector over a T×N
 * asset-return matrix, either rebalancing to target every period
 * (rebalance != 0) or buying-and-holding with weight drift
 * (rebalance == 0). Returns a (T+1)×1 equity curve starting at 1.0.
 * The handle-based rebalance-callback surface is a documented follow-on.
 * ==========================================================================*/

extern "C" matlab_mat *matlab_backtest(matlab_mat *returns, matlab_mat *weights,
                                        double rebalance) {
    if (!returns || !returns->data || !weights || !weights->data)
        return mat_alloc(0, 0);
    int64_t T = returns->rows, N = returns->cols;
    if ((weights->rows * weights->cols) != N) return mat_alloc(0, 0);
    matlab_mat *equity = mat_alloc(T + 1, 1);
    equity->data[0] = 1.0;
    bool rb = rebalance != 0.0;
    /* Per-asset holdings (dollar value); start at weights * 1.0. */
    std::vector<double> hold(static_cast<size_t>(N));
    for (int64_t j = 0; j < N; ++j) hold[static_cast<size_t>(j)] = weights->data[j];
    for (int64_t t = 0; t < T; ++t) {
        /* Apply this period's per-asset returns. */
        double total = 0.0;
        for (int64_t j = 0; j < N; ++j) {
            hold[static_cast<size_t>(j)] *= (1.0 + returns->data[t*N + j]);
            total += hold[static_cast<size_t>(j)];
        }
        equity->data[t + 1] = total;
        if (rb) {
            /* Rebalance back to target weights at the new total value. */
            for (int64_t j = 0; j < N; ++j)
                hold[static_cast<size_t>(j)] = total * weights->data[j];
        }
    }
    return equity;
}

/* backtestSummary(equity) -> 1×3 [totalReturn, annSharpe, maxDrawdown].
 * Derives per-period returns from the equity curve; annualises the
 * Sharpe by sqrt(periodsPerYear=252) as a convention. */
extern "C" matlab_mat *matlab_backtest_summary(matlab_mat *equity) {
    matlab_mat *out = mat_alloc(1, 3);
    if (!equity || !equity->data) return out;
    int64_t n = equity->rows * equity->cols;
    if (n < 2) return out;
    double total = equity->data[n-1] / equity->data[0] - 1.0;
    /* per-period returns */
    std::vector<double> r(static_cast<size_t>(n - 1));
    double s = 0.0;
    for (int64_t i = 0; i < n - 1; ++i) {
        r[static_cast<size_t>(i)] = equity->data[i+1] / equity->data[i] - 1.0;
        s += r[static_cast<size_t>(i)];
    }
    double mean = s / static_cast<double>(n - 1);
    double s2 = 0.0;
    for (int64_t i = 0; i < n - 1; ++i) {
        double d = r[static_cast<size_t>(i)] - mean;
        s2 += d * d;
    }
    double sd = (n > 2) ? sqrt(s2 / static_cast<double>(n - 2)) : 0.0;
    double sharpe = sd > 0.0 ? mean / sd * sqrt(252.0) : 0.0;
    /* max drawdown of the equity curve */
    double peak = equity->data[0], mdd = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double v = equity->data[i];
        if (v > peak) peak = v;
        double dd = peak > 0.0 ? (peak - v) / peak : 0.0;
        if (dd > mdd) mdd = dd;
    }
    out->data[0] = total;
    out->data[1] = sharpe;
    out->data[2] = mdd;
    return out;
}

/* ============================================================================
 * §T6 — Stochastic Differential Equations (Monte Carlo)
 *
 * Classdef carriers (bm / gbm / cir / hwv) hold the model parameters +
 * a ModelType discriminant (0=bm, 1=gbm, 2=cir, 3=hwv); simByEuler runs
 * the Euler-Maruyama scheme. 1-D models; the heston 2-D stochastic-vol
 * model + correlated multi-asset baskets are documented follow-ons.
 *
 * Random normals come from the shipped matlab_randn, so rng(seed) makes
 * a simulation reproducible.
 * ==========================================================================*/

extern "C" matlab_mat *matlab_randn(double m, double n);

enum { SDE_BM = 0, SDE_GBM = 1, SDE_CIR = 2, SDE_HWV = 3 };

/* simByEuler(obj, nPeriods, dt, nTrials) -> (nPeriods+1) × nTrials path
 * matrix. Each column is one simulated path; row 0 is StartState. */
extern "C" matlab_mat *matlab_sde_sim_euler(struct matlab_obj_s *obj,
                                             double nPeriods, double dt,
                                             double nTrials) {
    if (!obj) return mat_alloc(0, 0);
    int mt = static_cast<int>(matlab_obj_get_f64(obj, "ModelType", 9));
    double X0    = matlab_obj_get_f64(obj, "StartState", 10);
    double drift = matlab_obj_get_f64(obj, "Drift", 5);
    double sigma = matlab_obj_get_f64(obj, "Sigma", 5);
    double speed = matlab_obj_get_f64(obj, "Speed", 5);
    double level = matlab_obj_get_f64(obj, "Level", 5);
    int64_t P = static_cast<int64_t>(nPeriods);
    int64_t M = static_cast<int64_t>(nTrials);
    if (P <= 0 || M <= 0) return mat_alloc(0, 0);
    double sqdt = sqrt(dt);
    /* Pre-draw all normals: (P × M) row-major. */
    matlab_mat *Z = matlab_randn(static_cast<double>(P), static_cast<double>(M));
    matlab_mat *paths = mat_alloc(P + 1, M);   /* (P+1) × M */
    for (int64_t c = 0; c < M; ++c) paths->data[0*M + c] = X0;
    for (int64_t c = 0; c < M; ++c) {
        double x = X0;
        for (int64_t t = 0; t < P; ++t) {
            double z = (Z && Z->data) ? Z->data[t*M + c] : 0.0;
            double dx = 0.0;
            switch (mt) {
                case SDE_BM:
                    dx = drift * dt + sigma * sqdt * z;
                    break;
                case SDE_GBM:
                    dx = drift * x * dt + sigma * x * sqdt * z;
                    break;
                case SDE_CIR: {
                    double xc = x < 0.0 ? 0.0 : x;
                    dx = speed * (level - x) * dt + sigma * sqrt(xc) * sqdt * z;
                    break;
                }
                case SDE_HWV:
                    dx = speed * (level - x) * dt + sigma * sqdt * z;
                    break;
            }
            x += dx;
            if ((mt == SDE_CIR) && x < 0.0) x = 0.0;   /* keep CIR non-negative */
            paths->data[(t + 1)*M + c] = x;
        }
    }
    return paths;
}

/* simBySolution(obj, nPeriods, dt, nTrials) -> exact GBM transition
 * (no discretisation error): X_{t+1} = X_t exp((mu - sigma^2/2) dt +
 * sigma sqrt(dt) Z). Falls back to Euler for non-GBM models. */
extern "C" matlab_mat *matlab_sde_sim_solution(struct matlab_obj_s *obj,
                                                double nPeriods, double dt,
                                                double nTrials) {
    if (!obj) return mat_alloc(0, 0);
    int mt = static_cast<int>(matlab_obj_get_f64(obj, "ModelType", 9));
    if (mt != SDE_GBM)
        return matlab_sde_sim_euler(obj, nPeriods, dt, nTrials);
    double X0    = matlab_obj_get_f64(obj, "StartState", 10);
    double mu    = matlab_obj_get_f64(obj, "Drift", 5);
    double sigma = matlab_obj_get_f64(obj, "Sigma", 5);
    int64_t P = static_cast<int64_t>(nPeriods);
    int64_t M = static_cast<int64_t>(nTrials);
    if (P <= 0 || M <= 0) return mat_alloc(0, 0);
    double sqdt = sqrt(dt);
    double adrift = (mu - 0.5 * sigma * sigma) * dt;
    matlab_mat *Z = matlab_randn(static_cast<double>(P), static_cast<double>(M));
    matlab_mat *paths = mat_alloc(P + 1, M);
    for (int64_t c = 0; c < M; ++c) {
        double x = X0;
        paths->data[0*M + c] = x;
        for (int64_t t = 0; t < P; ++t) {
            double z = (Z && Z->data) ? Z->data[t*M + c] : 0.0;
            x *= exp(adrift + sigma * sqdt * z);
            paths->data[(t + 1)*M + c] = x;
        }
    }
    return paths;
}

/* ============================================================================
 * §T6.3 — Quasi-Monte-Carlo: Halton low-discrepancy sequence
 *
 * haltonseq(n, d) -> n×d matrix of quasi-random points in [0,1), one
 * coordinate per dimension using the radical inverse in the d-th prime
 * base. Lower star-discrepancy than pseudo-random draws, so Monte Carlo
 * estimators converge faster. Sobol is a documented follow-on.
 * ==========================================================================*/

static double radical_inverse(int64_t i, int64_t base) {
    double f = 1.0 / static_cast<double>(base);
    double r = 0.0;
    while (i > 0) {
        r += f * static_cast<double>(i % base);
        i /= base;
        f /= static_cast<double>(base);
    }
    return r;
}
extern "C" matlab_mat *matlab_haltonseq(double n_, double d_) {
    int64_t n = static_cast<int64_t>(n_);
    int64_t d = static_cast<int64_t>(d_);
    if (n <= 0 || d <= 0) return mat_alloc(0, 0);
    static const int64_t primes[] = { 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47 };
    int64_t np = static_cast<int64_t>(sizeof(primes)/sizeof(primes[0]));
    matlab_mat *out = mat_alloc(n, d);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < d; ++j)
            out->data[i*d + j] =
                radical_inverse(i + 1, primes[j < np ? j : np - 1]);
    return out;
}

/* optpricemc(terminalPrices, strike, r, T) — discounted Monte-Carlo
 * European-call price = exp(-rT) * mean(max(S_T - K, 0)). A small
 * helper so the headline avoids the elementwise max(mat, scalar)
 * lowering gap. terminalPrices is any-shape (flattened). */
extern "C" double matlab_optpricemc(matlab_mat *ST, double K,
                                     double r, double T) {
    if (!ST || !ST->data) return 0.0;
    int64_t n = ST->rows * ST->cols;
    if (n == 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double payoff = ST->data[i] - K;
        if (payoff > 0.0) s += payoff;
    }
    return exp(-r * T) * s / static_cast<double>(n);
}

/* ============================================================================
 * §T7.2 — Black-Litterman posterior expected returns
 *
 * blacklitterman(Sigma, wmkt, P, Q, tau, delta) -> N×1 posterior mean.
 *   Prior (equilibrium):  Pi = delta * Sigma * wmkt
 *   Omega = diag(P (tau Sigma) P')              (He-Litterman choice)
 *   Pi_BL = [ (tauSigma)^-1 + P' Omega^-1 P ]^-1
 *           [ (tauSigma)^-1 Pi + P' Omega^-1 Q ]
 * Matrix inverse via Cholesky-solve on identity columns.
 * ==========================================================================*/

static void spd_inv(const double *A, double *Ainv, int64_t n) {
    std::vector<double> L(static_cast<size_t>(n*n));
    for (int64_t i = 0; i < n*n; ++i) L[static_cast<size_t>(i)] = A[i];
    if (!chol_factor(L.data(), n)) {
        for (int64_t i = 0; i < n*n; ++i) L[static_cast<size_t>(i)] = A[i];
        for (int64_t i = 0; i < n; ++i) L[static_cast<size_t>(i*n + i)] += 1e-10;
        chol_factor(L.data(), n);
    }
    std::vector<double> e(static_cast<size_t>(n)), x(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t i = 0; i < n; ++i) e[static_cast<size_t>(i)] = (i == c) ? 1.0 : 0.0;
        chol_solve(L.data(), e.data(), x.data(), n);
        for (int64_t i = 0; i < n; ++i) Ainv[i*n + c] = x[static_cast<size_t>(i)];
    }
}

extern "C" matlab_mat *matlab_blacklitterman(matlab_mat *Sigma, matlab_mat *wmkt,
                                             matlab_mat *P, matlab_mat *Q,
                                             double tau, double delta);
/* Scalar-Q convenience: a single view's Q folds to an f64 literal at
 * the call site. Box it into a 1×1 matrix and delegate. */
extern "C" matlab_mat *matlab_blacklitterman_q1(matlab_mat *Sigma,
                                                matlab_mat *wmkt, matlab_mat *P,
                                                double q, double tau,
                                                double delta) {
    matlab_mat *Q = mat_alloc(1, 1);
    Q->data[0] = q;
    return matlab_blacklitterman(Sigma, wmkt, P, Q, tau, delta);
}

extern "C" matlab_mat *matlab_blacklitterman(matlab_mat *Sigma, matlab_mat *wmkt,
                                             matlab_mat *P, matlab_mat *Q,
                                             double tau, double delta) {
    if (!Sigma || !Sigma->data || !wmkt || !wmkt->data) return mat_alloc(0, 0);
    int64_t N = Sigma->rows;
    int64_t K = (P && P->data) ? P->rows : 0;
    const double *Sg = Sigma->data;
    /* Equilibrium prior Pi = delta * Sigma * wmkt. */
    std::vector<double> Pi(static_cast<size_t>(N), 0.0);
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < N; ++j) s += Sg[i*N + j] * wmkt->data[j];
        Pi[static_cast<size_t>(i)] = delta * s;
    }
    /* tauSigma + its inverse A. */
    std::vector<double> tauS(static_cast<size_t>(N*N));
    for (int64_t i = 0; i < N*N; ++i) tauS[static_cast<size_t>(i)] = tau * Sg[i];
    std::vector<double> A(static_cast<size_t>(N*N));
    spd_inv(tauS.data(), A.data(), N);
    /* M = A + P' Omega^-1 P;  rhs = A Pi + P' Omega^-1 Q. */
    std::vector<double> M(A);
    std::vector<double> rhs(static_cast<size_t>(N), 0.0);
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < N; ++j) s += A[i*N + j] * Pi[static_cast<size_t>(j)];
        rhs[static_cast<size_t>(i)] = s;
    }
    for (int64_t k = 0; k < K; ++k) {
        const double *Pk = &P->data[k*N];
        /* omega_k = Pk (tauSigma) Pk'. */
        double omega = 0.0;
        for (int64_t a = 0; a < N; ++a) {
            double tsp = 0.0;
            for (int64_t b = 0; b < N; ++b) tsp += tauS[static_cast<size_t>(a*N + b)] * Pk[b];
            omega += Pk[a] * tsp;
        }
        if (omega < 1e-12) omega = 1e-12;
        double inv_omega = 1.0 / omega;
        double Qk = Q && Q->data ? Q->data[k] : 0.0;
        for (int64_t a = 0; a < N; ++a) {
            rhs[static_cast<size_t>(a)] += inv_omega * Pk[a] * Qk;
            for (int64_t b = 0; b < N; ++b)
                M[static_cast<size_t>(a*N + b)] += inv_omega * Pk[a] * Pk[b];
        }
    }
    /* Pi_BL = M^-1 rhs. */
    matlab_mat *out = mat_alloc(N, 1);
    std::vector<double> Minv(static_cast<size_t>(N*N));
    spd_inv(M.data(), Minv.data(), N);
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < N; ++j) s += Minv[static_cast<size_t>(i*N + j)] * rhs[static_cast<size_t>(j)];
        out->data[i] = s;
    }
    return out;
}

/* ============================================================================
 * §T7.3 — Risk parity / risk budgeting
 *
 * riskparity(Sigma) -> equal-risk-contribution weights (each asset
 * contributes the same share of portfolio variance). riskbudget(Sigma,
 * b) targets an arbitrary risk-contribution budget b. Both use the
 * standard fixed-point iteration w_i <- b_i / (Sigma w)_i, renormalised
 * to sum 1, which converges to the (unique, long-only) RC solution.
 * ==========================================================================*/

static matlab_mat *risk_budget_core(const double *Sg, int64_t N,
                                     const double *b) {
    matlab_mat *w = mat_alloc(N, 1);
    std::vector<double> x(static_cast<size_t>(N), 1.0 / static_cast<double>(N));
    std::vector<double> Sx(static_cast<size_t>(N));
    for (int it = 0; it < 500; ++it) {
        matvec(Sg, x.data(), Sx.data(), N);
        double sum = 0.0;
        std::vector<double> nx(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i) {
            double denom = Sx[static_cast<size_t>(i)];
            if (denom < 1e-12) denom = 1e-12;
            nx[static_cast<size_t>(i)] = b[i] / denom;
            sum += nx[static_cast<size_t>(i)];
        }
        double delta = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            nx[static_cast<size_t>(i)] /= sum;
            delta += fabs(nx[static_cast<size_t>(i)] - x[static_cast<size_t>(i)]);
        }
        x = nx;
        if (delta < 1e-12) break;
    }
    for (int64_t i = 0; i < N; ++i) w->data[i] = x[static_cast<size_t>(i)];
    return w;
}

extern "C" matlab_mat *matlab_riskparity(matlab_mat *Sigma) {
    if (!Sigma || !Sigma->data) return mat_alloc(0, 0);
    int64_t N = Sigma->rows;
    std::vector<double> b(static_cast<size_t>(N), 1.0 / static_cast<double>(N));
    return risk_budget_core(Sigma->data, N, b.data());
}

extern "C" matlab_mat *matlab_riskbudget(matlab_mat *Sigma, matlab_mat *budget) {
    if (!Sigma || !Sigma->data || !budget || !budget->data) return mat_alloc(0, 0);
    int64_t N = Sigma->rows;
    /* Normalise the budget to sum 1. */
    std::vector<double> b(static_cast<size_t>(N));
    double s = 0.0;
    for (int64_t i = 0; i < N; ++i) { b[static_cast<size_t>(i)] = budget->data[i]; s += budget->data[i]; }
    if (s > 0.0) for (int64_t i = 0; i < N; ++i) b[static_cast<size_t>(i)] /= s;
    return risk_budget_core(Sigma->data, N, b.data());
}

/* riskcontribution(Sigma, w) -> N×1 per-asset risk-contribution shares
 * (RC_i = w_i (Sigma w)_i / (w' Sigma w), summing to 1). Lets a caller
 * verify a riskparity result has equal contributions. */
extern "C" matlab_mat *matlab_riskcontribution(matlab_mat *Sigma, matlab_mat *w) {
    if (!Sigma || !Sigma->data || !w || !w->data) return mat_alloc(0, 0);
    int64_t N = Sigma->rows;
    matlab_mat *rc = mat_alloc(N, 1);
    std::vector<double> Sw(static_cast<size_t>(N));
    matvec(Sigma->data, w->data, Sw.data(), N);
    double var = dot(w->data, Sw.data(), N);
    if (var < 1e-20) var = 1e-20;
    for (int64_t i = 0; i < N; ++i)
        rc->data[i] = w->data[i] * Sw[static_cast<size_t>(i)] / var;
    return rc;
}
