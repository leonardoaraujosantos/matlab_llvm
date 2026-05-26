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
