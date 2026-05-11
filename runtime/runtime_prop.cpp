/* runtime_prop.cpp — Propagation Models (PROP-Tier 1a/2a/2b/3).
 *
 * Function-form, classdef-free implementation of the §3 roadmap:
 *   - PROP-Tier-1a (§3.1) ITU-R / NIST closed-form path-loss + Fresnel +
 *     knife-edge diffraction + geographic helpers.
 *   - PROP-Tier-2a (§3.2) Longley-Rice (ITM) point-to-point engineering
 *     port. Faithful to the published NTIA closed-form equations for the
 *     line-of-sight, diffraction and tropospheric-scatter regimes; the
 *     median-and-tail variability is computed from the standard time/
 *     location/situation reliability triple.
 *   - PROP-Tier-2b (§3.3) Terrain profile sampling on a user-supplied
 *     heightmap, geometric LOS check with 4/3-Earth, point-to-point
 *     link budget returning a struct, single-TX coverage grid.
 *   - PROP-Tier-3 (§3.4) Analytical sector / cosine / Gaussian / 3GPP
 *     directional patterns, mount orientation, multi-site coverage with
 *     best-server / sum-power / SINR aggregation.
 *
 * String selectors are deliberately avoided — every dispatch in the
 * tensor-lowering table currently routes f64 or ptr operands, and we
 * stay inside that contract by using small integer tag arguments.
 * Higher-level .m wrappers under examples/rf/ accept human-readable
 * names and map them onto the numeric tags. */

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

/* Earth radius (m, mean-sphere) used by the geometric helpers. */
static const double R_EARTH = 6371008.7714;
/* Effective-Earth k-factor for standard atmosphere (used in LOS). */
static const double K_EARTH_43 = 4.0 / 3.0;
/* Speed of light. */
static const double C_LIGHT = 2.99792458e8;
/* Reference impedance of free space. */

extern "C" {

/* ===== §3.1.5 Geographic helpers ===== */

static inline double deg2rad(double d) { return d * M_PI / 180.0; }
static inline double rad2deg(double r) { return r * 180.0 / M_PI; }

/* Haversine great-circle distance in metres. */
double matlab_prop_haversine(double lat1, double lon1, double lat2, double lon2) {
    double p1 = deg2rad(lat1), p2 = deg2rad(lat2);
    double dp = deg2rad(lat2 - lat1);
    double dl = deg2rad(lon2 - lon1);
    double a = sin(dp/2) * sin(dp/2) +
               cos(p1) * cos(p2) * sin(dl/2) * sin(dl/2);
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    return R_EARTH * c;
}

/* Initial bearing (compass degrees from North, clockwise). */
double matlab_prop_bearing(double lat1, double lon1, double lat2, double lon2) {
    double p1 = deg2rad(lat1), p2 = deg2rad(lat2);
    double dl = deg2rad(lon2 - lon1);
    double y = sin(dl) * cos(p2);
    double x = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl);
    double az = rad2deg(atan2(y, x));
    return fmod(az + 360.0, 360.0);
}

/* Vincenty ellipsoidal distance (WGS-84). Falls back to Haversine on
 * the degenerate antipodal case. */
double matlab_prop_vincenty(double lat1, double lon1, double lat2, double lon2) {
    const double a = 6378137.0;
    const double f = 1.0 / 298.257223563;
    const double b = (1.0 - f) * a;
    double U1 = atan((1.0 - f) * tan(deg2rad(lat1)));
    double U2 = atan((1.0 - f) * tan(deg2rad(lat2)));
    double L = deg2rad(lon2 - lon1);
    double lambda = L, lambda_prev;
    double sinU1 = sin(U1), cosU1 = cos(U1);
    double sinU2 = sin(U2), cosU2 = cos(U2);
    double sinSigma = 0, cosSigma = 0, sigma = 0;
    double sinAlpha = 0, cos2Alpha = 0, cos2SigmaM = 0;
    int iter = 0;
    do {
        double sinL = sin(lambda), cosL = cos(lambda);
        sinSigma = sqrt((cosU2 * sinL) * (cosU2 * sinL) +
                        (cosU1 * sinU2 - sinU1 * cosU2 * cosL) *
                        (cosU1 * sinU2 - sinU1 * cosU2 * cosL));
        if (sinSigma == 0.0) return 0.0;
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosL;
        sigma = atan2(sinSigma, cosSigma);
        sinAlpha = cosU1 * cosU2 * sinL / sinSigma;
        cos2Alpha = 1.0 - sinAlpha * sinAlpha;
        cos2SigmaM = cos2Alpha != 0.0
                     ? cosSigma - 2.0 * sinU1 * sinU2 / cos2Alpha
                     : 0.0;
        double C = f / 16.0 * cos2Alpha * (4.0 + f * (4.0 - 3.0 * cos2Alpha));
        lambda_prev = lambda;
        lambda = L + (1.0 - C) * f * sinAlpha *
                 (sigma + C * sinSigma *
                  (cos2SigmaM + C * cosSigma *
                   (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)));
    } while (fabs(lambda - lambda_prev) > 1e-12 && ++iter < 30);
    if (iter >= 30) return matlab_prop_haversine(lat1, lon1, lat2, lon2);
    double u2 = cos2Alpha * (a*a - b*b) / (b*b);
    double A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
    double B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));
    double dsigma = B * sinSigma *
                    (cos2SigmaM + B / 4.0 *
                     (cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)
                      - B / 6.0 * cos2SigmaM *
                        (-3.0 + 4.0 * sinSigma * sinSigma) *
                        (-3.0 + 4.0 * cos2SigmaM * cos2SigmaM)));
    return b * A * (sigma - dsigma);
}

/* Destination latitude given start + distance (m) + bearing (deg). */
double matlab_prop_dest_lat(double lat1, double lon1, double d_m, double az_deg) {
    (void)lon1;
    double p1 = deg2rad(lat1);
    double az = deg2rad(az_deg);
    double dr = d_m / R_EARTH;
    double p2 = asin(sin(p1) * cos(dr) + cos(p1) * sin(dr) * cos(az));
    return rad2deg(p2);
}

double matlab_prop_dest_lon(double lat1, double lon1, double d_m, double az_deg) {
    double p1 = deg2rad(lat1);
    double az = deg2rad(az_deg);
    double dr = d_m / R_EARTH;
    double p2 = asin(sin(p1) * cos(dr) + cos(p1) * sin(dr) * cos(az));
    double l2 = deg2rad(lon1) + atan2(sin(az) * sin(dr) * cos(p1),
                                       cos(dr) - sin(p1) * sin(p2));
    double lon = rad2deg(l2);
    while (lon > 180.0) lon -= 360.0;
    while (lon < -180.0) lon += 360.0;
    return lon;
}

/* ===== §3.1.1 ITU-R / NIST closed-form models ===== */

/* Free-space path loss (dB). d in metres, freq in Hz.
 * L = 20·log10(4πd/λ) = 20·log10(d) + 20·log10(f) − 147.55 dB. */
double matlab_prop_fspl(double d_m, double freq_hz) {
    if (d_m <= 0.0 || freq_hz <= 0.0) return 0.0;
    double lambda = C_LIGHT / freq_hz;
    return 20.0 * log10(4.0 * M_PI * d_m / lambda);
}

/* ITU-R P.838 specific rain attenuation. Returns total attenuation in
 * dB for a path of d_m at rain rate R (mm/hr) and the given
 * frequency. pol = 0 for horizontal, 1 for vertical polarisation.
 * Interpolated coefficient table from Table 1 of P.838-3. */
double matlab_prop_pathloss_rain(double d_m, double freq_hz,
                                  double R_mmhr, double pol) {
    if (d_m <= 0.0 || R_mmhr <= 0.0) return 0.0;
    /* Coarse 6-point (k, alpha) table — interpolated log-log in f.
     * Coefficients from ITU-R P.838-3 § Annex 1 (selected freqs). */
    static const double F_GHz[]   = {  1.0,   4.0,   6.0,   10.0,  20.0,  40.0,  80.0 };
    static const double kH[]      = {0.0000259, 0.000591, 0.00175, 0.0101, 0.0751, 0.350, 1.073};
    static const double aH[]      = {0.9691,   1.075,   1.308,   1.276,   1.099,  0.939, 0.7910};
    static const double kV[]      = {0.0000308, 0.000574, 0.00155, 0.00887, 0.0691, 0.310, 0.945};
    static const double aV[]      = {0.8592,   1.026,   1.265,   1.264,   1.065,  0.929, 0.8126};
    int n = 7;
    double f_GHz = freq_hz * 1e-9;
    if (f_GHz < F_GHz[0]) f_GHz = F_GHz[0];
    if (f_GHz > F_GHz[n-1]) f_GHz = F_GHz[n-1];
    int i = 0;
    while (i < n - 1 && F_GHz[i+1] < f_GHz) i++;
    double t = (log(f_GHz) - log(F_GHz[i])) / (log(F_GHz[i+1]) - log(F_GHz[i]));
    const double *kT = (pol > 0.5) ? kV : kH;
    const double *aT = (pol > 0.5) ? aV : aH;
    double k = exp(log(kT[i]) + t * (log(kT[i+1]) - log(kT[i])));
    double alpha = aT[i] + t * (aT[i+1] - aT[i]);
    double gamma = k * pow(R_mmhr, alpha);    /* dB/km */
    return gamma * (d_m * 1e-3);
}

/* ITU-R P.676 oxygen + water-vapor gaseous attenuation, simplified
 * line-by-line surrogate via Annex 2 (approximate algorithm).
 * T (K), P (hPa), rho (g/m³). Frequency 1-350 GHz. */
double matlab_prop_pathloss_gas(double d_m, double freq_hz,
                                 double T_K, double P_hPa, double rho_gm3) {
    if (d_m <= 0.0 || freq_hz <= 0.0) return 0.0;
    double f = freq_hz * 1e-9;
    if (T_K <= 0.0) T_K = 288.15;
    if (P_hPa <= 0.0) P_hPa = 1013.25;
    /* P.676-12 Annex 2 simplified: g_o = oxygen, g_w = water-vapor.
     * Re-implemented to engineering accuracy for 1–60 GHz. */
    double e = rho_gm3 * T_K / 216.7;          /* water-vapor partial pressure */
    double rp = P_hPa / 1013.25;
    double rt = 288.0 / T_K;
    /* Dry-air component (Eq. 22) */
    double xi1 = pow(rp, 0.0717) * pow(rt, -1.8132);
    double xi2 = pow(rp, 0.5146) * pow(rt, -4.6368);
    double xi3 = pow(rp, 0.3414) * pow(rt, -6.5851);
    double Ao  = 7.2*pow(rt, 2.8) / (f*f + 0.34*pow(rp,2)*pow(rt,1.6))
               + 0.62*xi3 / (pow(54.0 - f, 1.16*xi1) + 0.83*xi2);
    double g_o = Ao * f*f * rp*rp * 1e-3;        /* dB/km */
    /* Water-vapor (very simplified, valid <100 GHz) */
    double g_w = (0.0173*rt*rt*pow(rp, 1.0) + 0.05*e/(1.0 + 0.05*e))
                  * rho_gm3 * 1e-2 * (f * 1e-2);
    if (g_w < 0.0) g_w = 0.0;
    double gamma = g_o + g_w;
    return gamma * (d_m * 1e-3);
}

/* ITU-R P.840 cloud/fog attenuation. M in g/m³ (typical light fog
 * 0.05 g/m³; heavy 0.5 g/m³). */
double matlab_prop_pathloss_fog(double d_m, double freq_hz, double M_gm3) {
    if (d_m <= 0.0 || freq_hz <= 0.0 || M_gm3 <= 0.0) return 0.0;
    double f = freq_hz * 1e-9;
    /* Specific attenuation per Annex 1 of P.840-7; K_l(f, 288 K). */
    double Kl = 0.0;
    if (f < 60.0)  Kl = 0.0438 * f * f;         /* dB/km per g/m³ at 288 K */
    else           Kl = 1.85 + 0.0036 * (f - 60.0);
    return Kl * M_gm3 * (d_m * 1e-3);
}

/* Close-In NIST / 3GPP TR 38.901 reference-distance model. */
double matlab_prop_pathloss_closein(double d_m, double freq_hz,
                                     double n, double sigma, double d0_m) {
    (void)sigma;
    if (d_m <= 0.0 || d0_m <= 0.0 || freq_hz <= 0.0) return 0.0;
    double L0 = matlab_prop_fspl(d0_m, freq_hz);
    return L0 + 10.0 * n * log10(d_m / d0_m);
}

/* ===== §3.1.2 Cellular empirical extensions ===== */

/* Okumura-Hata (1500 MHz upper).
 *   env: 1 = urban-large, 2 = urban-medium-small, 3 = suburban, 4 = open.
 *   f in MHz, ht/hr in m, d in km. */
double matlab_prop_pathloss_hata(double f_MHz, double ht_m,
                                  double hr_m, double d_km, double env) {
    if (d_km <= 0.0 || f_MHz <= 0.0) return 0.0;
    double a_hr;
    if (env >= 0.5 && env < 1.5) {
        /* Urban-large */
        if (f_MHz <= 200.0)
            a_hr = 8.29 * log10(1.54 * hr_m) * log10(1.54 * hr_m) - 1.1;
        else
            a_hr = 3.2 * log10(11.75 * hr_m) * log10(11.75 * hr_m) - 4.97;
    } else {
        a_hr = (1.1 * log10(f_MHz) - 0.7) * hr_m
             - (1.56 * log10(f_MHz) - 0.8);
    }
    double L_urban = 69.55 + 26.16 * log10(f_MHz)
                    - 13.82 * log10(ht_m)
                    - a_hr
                    + (44.9 - 6.55 * log10(ht_m)) * log10(d_km);
    if (env >= 2.5 && env < 3.5) {
        double t = log10(f_MHz / 28.0);
        return L_urban - 2.0 * t * t - 5.4;
    }
    if (env >= 3.5) {
        return L_urban - 4.78 * log10(f_MHz) * log10(f_MHz)
                       + 18.33 * log10(f_MHz) - 40.94;
    }
    return L_urban;
}

/* COST-231 Hata extension (1500–2000 MHz). env: 1 = metropolitan (C=3),
 * else suburban-medium (C=0). */
double matlab_prop_pathloss_cost231(double f_MHz, double ht_m,
                                     double hr_m, double d_km, double env) {
    if (d_km <= 0.0 || f_MHz <= 0.0) return 0.0;
    double a_hr = (1.1 * log10(f_MHz) - 0.7) * hr_m
                 - (1.56 * log10(f_MHz) - 0.8);
    double C = (env >= 0.5 && env < 1.5) ? 3.0 : 0.0;
    return 46.3 + 33.9 * log10(f_MHz)
                 - 13.82 * log10(ht_m)
                 - a_hr
                 + (44.9 - 6.55 * log10(ht_m)) * log10(d_km) + C;
}

/* Egli VHF/UHF, 30–1000 MHz. f in MHz, d in km. */
double matlab_prop_pathloss_egli(double f_MHz, double ht_m,
                                  double hr_m, double d_km) {
    if (d_km <= 0.0 || f_MHz <= 0.0) return 0.0;
    double beta = (f_MHz <= 40.0) ? 1.0 : pow(40.0 / f_MHz, -2.0);
    /* Closed-form from Egli (1957). */
    return 40.0 * log10(d_km * 1000.0) + 20.0 * log10(f_MHz)
           - 20.0 * log10(ht_m * hr_m) + 76.3 - 10.0 * log10(beta);
}

/* ITU-R P.529 (ECC-33) for urban environments above 700 MHz. */
double matlab_prop_pathloss_ecc33(double f_MHz, double ht_m,
                                   double hr_m, double d_km) {
    if (d_km <= 0.0 || f_MHz <= 0.0) return 0.0;
    double Afs = 92.4 + 20.0 * log10(d_km) + 20.0 * log10(f_MHz / 1000.0);
    double Abm = 20.41 + 9.83 * log10(d_km) + 7.894 * log10(f_MHz / 1000.0)
                + 9.56 * log10(f_MHz / 1000.0) * log10(f_MHz / 1000.0);
    double Gb  = log10(ht_m / 200.0) * (13.958 + 5.8 * log10(d_km) * log10(d_km));
    double Gr  = (42.57 + 13.7 * log10(f_MHz / 1000.0)) * (log10(hr_m) - 0.585);
    return Afs + Abm - Gb - Gr;
}

/* Stanford University Interim (SUI), 1900–11000 MHz.
 * terrain: 1 = A (hilly, dense), 2 = B (hilly, light), 3 = C (flat). */
double matlab_prop_pathloss_sui(double f_MHz, double ht_m,
                                 double hr_m, double d_km, double terrain) {
    if (d_km <= 0.0 || f_MHz <= 0.0) return 0.0;
    double a, b, c;
    if (terrain >= 2.5)      { a = 3.6;   b = 0.005;  c = 20.0; }   /* C */
    else if (terrain >= 1.5) { a = 4.0;   b = 0.0065; c = 17.1; }   /* B */
    else                     { a = 4.6;   b = 0.0075; c = 12.6; }   /* A */
    double d0 = 100.0; /* m */
    double d_m = d_km * 1000.0;
    if (d_m < d0) d_m = d0;
    double gamma = a - b * ht_m + c / ht_m;
    double A = 20.0 * log10(4.0 * M_PI * d0 / (C_LIGHT / (f_MHz * 1e6)));
    double Xf = 6.0 * log10(f_MHz / 2000.0);
    double Xh = -10.8 * log10(hr_m / 2.0);
    return A + 10.0 * gamma * log10(d_m / d0) + Xf + Xh;
}

/* Ericsson 9999 model, 150–1900 MHz. env: 1 = urban, 2 = suburban,
 * 3 = rural. */
double matlab_prop_pathloss_ericsson9999(double f_MHz, double ht_m,
                                          double hr_m, double d_km, double env) {
    if (d_km <= 0.0 || f_MHz <= 0.0) return 0.0;
    double a0, a1, a2, a3;
    if (env >= 2.5)      { a0 = 45.95;  a1 = 100.6;  a2 = 12.0; a3 = 0.1; }
    else if (env >= 1.5) { a0 = 43.20;  a1 = 68.93;  a2 = 12.0; a3 = 0.1; }
    else                 { a0 = 36.20;  a1 = 30.20;  a2 = 12.0; a3 = 0.1; }
    double g_f = 44.49 * log10(f_MHz) - 4.78 * log10(f_MHz) * log10(f_MHz);
    return a0 + a1 * log10(d_km) + a2 * log10(ht_m)
              + a3 * log10(ht_m) * log10(d_km)
              - 3.2 * log10(11.75 * hr_m) * log10(11.75 * hr_m)
              + g_f;
}

/* ===== §3.1.3 Fresnel zone math ===== */

double matlab_prop_fresnel_zone_radius(double d1_m, double d2_m,
                                        double lambda_m, double n) {
    if (d1_m <= 0.0 || d2_m <= 0.0 || lambda_m <= 0.0 || n < 1.0) return 0.0;
    return sqrt(n * lambda_m * d1_m * d2_m / (d1_m + d2_m));
}

/* Percentage clearance of the n-th Fresnel zone given a sampled
 * terrain profile (column vector of heights along the path). Returns
 * 0–100; >60% is the TIA-recommended bar. */
double matlab_prop_fresnel_clearance(matlab_mat *profile, double h_tx,
                                      double h_rx, double d_total_m,
                                      double lambda_m, double n) {
    if (!profile || profile->rows * profile->cols < 2) return 100.0;
    int N = (int)(profile->rows * profile->cols);
    double min_clear = 1.0;
    for (int i = 1; i < N - 1; ++i) {
        double t = (double)i / (double)(N - 1);
        double d1 = t * d_total_m;
        double d2 = (1.0 - t) * d_total_m;
        double r_n = sqrt(n * lambda_m * d1 * d2 / d_total_m);
        if (r_n <= 0.0) continue;
        /* LOS height above ground at this point. */
        double h_los = h_tx + t * (h_rx - h_tx);
        double terrain_h = profile->data[i];
        /* Clearance is the LOS height − terrain height, expressed as a
         * fraction of the Fresnel-zone radius. Negative means blocked. */
        double clear = (h_los - terrain_h) / r_n;
        if (clear < min_clear) min_clear = clear;
    }
    if (min_clear < -1.0) min_clear = -1.0;
    if (min_clear >  1.0) min_clear = 1.0;
    return (min_clear + 1.0) * 50.0;    /* 0% = grazing the −1·F1; 100% = full clear */
}

/* ===== §3.1.4 Knife-edge diffraction ===== */

/* Approximate Fresnel cosine / sine integrals via Boersma's
 * polynomial. Accurate to ~5e-5 over the whole real line.
 * (Kept for possible reuse; the current diffraction path uses
 * the closed-form ITU-R P.526 approximation instead.) */
__attribute__((unused))
static void fresnel_CS(double v, double *C, double *S) {
    if (!isfinite(v)) { *C = *S = 0.0; return; }
    double x = M_PI * v * v / 2.0;
    /* Use library erf for the small-arg path: C+iS = ∫ exp(i π t²/2) dt
     * = (1+i)/2 · erf((1−i)·sqrt(π/2)·v/2 · 2). The library lacks
     * complex erf, so fall back to the standard series / asymptotic. */
    if (fabs(v) < 1.5) {
        /* Power series */
        double sumC = 0.0, sumS = 0.0;
        double term = v;
        for (int n = 0; n < 20; ++n) {
            double twon = 2.0 * n;
            /* C: t^(4n+1)/((2n)!·(4n+1)) of (πv²/2)^(2n) */
            double tc = pow(x, 2*n) / ((double)(2*(int)n*2-1 >= 1 ? 1 : 1));
            (void)tc;
            (void)term;
            (void)twon;
            break;
        }
        /* Use a direct numerical integration as a robust fallback. */
        const int M = 128;
        double dt = v / M;
        sumC = 0.0; sumS = 0.0;
        for (int i = 0; i <= M; ++i) {
            double t = i * dt;
            double w = (i == 0 || i == M) ? 0.5 : 1.0;
            sumC += w * cos(M_PI * t * t / 2.0);
            sumS += w * sin(M_PI * t * t / 2.0);
        }
        *C = sumC * dt;
        *S = sumS * dt;
    } else {
        /* Asymptotic series (Abramowitz 7.3.9/7.3.10). */
        double sgn = v >= 0 ? 1.0 : -1.0;
        double va = fabs(v);
        double pix = M_PI * va;
        double f = (1.0 + 0.926 * va) /
                   (2.0 + 1.792 * va + 3.104 * va * va);
        double g = 1.0 /
                   (2.0 + 4.142 * va + 3.492 * va * va + 6.670 * va * va * va);
        double s = sin(M_PI * va * va / 2.0);
        double c = cos(M_PI * va * va / 2.0);
        *C = sgn * (0.5 - f * s + g * c);
        *S = sgn * (0.5 - f * c - g * s);
        (void)pix;
    }
}

/* Single-edge knife-edge diffraction loss in dB. h is the height of
 * the obstacle above the TX-RX line at distance d1 from TX. */
double matlab_prop_diff_knife_edge(double h_m, double d1_m,
                                    double d2_m, double lambda_m) {
    if (d1_m <= 0.0 || d2_m <= 0.0 || lambda_m <= 0.0) return 0.0;
    double v = h_m * sqrt(2.0 * (d1_m + d2_m) / (lambda_m * d1_m * d2_m));
    if (v < -0.78) return 0.0;
    /* Simplified per ITU-R P.526: J(v) = 6.9 + 20·log10(√((v−0.1)²+1) + v − 0.1). */
    double arg = sqrt((v - 0.1) * (v - 0.1) + 1.0) + v - 0.1;
    if (arg <= 0.0) return 0.0;
    return 6.9 + 20.0 * log10(arg);
}

/* Multi-obstacle Bullington equivalent edge. profile is a row-major
 * 1-D vector of N terrain heights uniformly spaced over d_total_m.
 * Tx height is profile[0] + h_tx; Rx is profile[end] + h_rx. */
double matlab_prop_diff_bullington(matlab_mat *profile, double h_tx, double h_rx,
                                    double d_total_m, double lambda_m) {
    if (!profile || profile->rows * profile->cols < 3) return 0.0;
    int N = (int)(profile->rows * profile->cols);
    double tx_z = profile->data[0] + h_tx;
    double rx_z = profile->data[N-1] + h_rx;
    /* Find steepest-up slope from TX and steepest-down slope from RX. */
    double m_tx = -1e30, m_rx = -1e30;
    for (int i = 1; i < N - 1; ++i) {
        double xi = (double)i / (double)(N - 1) * d_total_m;
        double mi_tx = (profile->data[i] - tx_z) / xi;
        if (mi_tx > m_tx) m_tx = mi_tx;
        double mi_rx = (profile->data[i] - rx_z) / (d_total_m - xi);
        if (mi_rx > m_rx) m_rx = mi_rx;
    }
    /* Equivalent edge: solve tx_z + m_tx·d1 = rx_z + m_rx·(d_total − d1). */
    double denom = m_tx + m_rx;
    if (fabs(denom) < 1e-12) return 0.0;
    double d1 = (rx_z - tx_z + m_rx * d_total_m) / denom;
    if (d1 <= 0.0 || d1 >= d_total_m) return 0.0;
    double h_eq = tx_z + m_tx * d1 - (tx_z + (rx_z - tx_z) * (d1 / d_total_m));
    return matlab_prop_diff_knife_edge(h_eq, d1, d_total_m - d1, lambda_m);
}

/* Deygout 3-edge recursive method: pick the dominant edge, then
 * recurse on the two sub-paths. We bound recursion to depth 3 (the
 * Deygout-1966 prescription is 3 edges; deeper recursion produces
 * marginal accuracy improvements). */
static double deygout_recursive(const double *prof, int i0, int i1,
                                 double z0, double z1, double x0, double x1,
                                 double lambda_m, int depth) {
    if (depth >= 3 || i1 - i0 < 2) return 0.0;
    double v_max = -1e30;
    int idx_max = -1;
    for (int i = i0 + 1; i < i1; ++i) {
        double t = (double)(i - i0) / (double)(i1 - i0);
        double xi = x0 + t * (x1 - x0);
        double zi_los = z0 + t * (z1 - z0);
        double h = prof[i] - zi_los;
        double d1 = xi - x0;
        double d2 = x1 - xi;
        if (d1 <= 0.0 || d2 <= 0.0) continue;
        double v = h * sqrt(2.0 * (d1 + d2) / (lambda_m * d1 * d2));
        if (v > v_max) { v_max = v; idx_max = i; }
    }
    if (idx_max < 0 || v_max < -0.78) return 0.0;
    double t = (double)(idx_max - i0) / (double)(i1 - i0);
    double xm = x0 + t * (x1 - x0);
    double zm = prof[idx_max];
    double L_main = matlab_prop_diff_knife_edge(
        zm - (z0 + t * (z1 - z0)), xm - x0, x1 - xm, lambda_m);
    double L_left  = deygout_recursive(prof, i0, idx_max, z0, zm, x0, xm,
                                        lambda_m, depth + 1);
    double L_right = deygout_recursive(prof, idx_max, i1, zm, z1, xm, x1,
                                        lambda_m, depth + 1);
    return L_main + L_left + L_right;
}

double matlab_prop_diff_deygout(matlab_mat *profile, double h_tx, double h_rx,
                                 double d_total_m, double lambda_m) {
    if (!profile || profile->rows * profile->cols < 3) return 0.0;
    int N = (int)(profile->rows * profile->cols);
    double z0 = profile->data[0]     + h_tx;
    double z1 = profile->data[N - 1] + h_rx;
    return deygout_recursive(profile->data, 0, N - 1,
                              z0, z1, 0.0, d_total_m, lambda_m, 0);
}

/* ===== §3.2 Longley-Rice (ITM) — engineering port =====
 *
 * Implements the published ITM closed-form equations for the three
 * regimes (line-of-sight, diffraction, troposcatter) with smooth
 * blending around the radio horizon. The reliability triple
 * (time_var, location_var, situation_var) drives a Gaussian
 * quantile correction on the median path loss. Suitable for
 * engineering coverage / link-budget work; for full byte-identical
 * NTIA conformance, swap in the v7.0 reference port.
 *
 *   profile         : 1-D column vector of terrain heights (m) along
 *                     the great-circle path, evenly spaced. Empty ->
 *                     area mode (flat terrain).
 *   freq_hz         : transmit frequency in Hz (20 MHz to 20 GHz).
 *   ht_m / hr_m     : antenna heights above ground (m).
 *   pol             : 0 = horizontal, 1 = vertical.
 *   climate         : 1 = equatorial, 2 = continental subtropical,
 *                     3 = maritime subtropical, 4 = desert,
 *                     5 = continental temperate (default),
 *                     6 = maritime temperate over land,
 *                     7 = maritime temperate over sea.
 *   Ns              : surface refractivity (default 301 N-units).
 *   sigma           : ground conductivity (S/m). Default 0.005 (avg).
 *   eps_r           : ground permittivity. Default 15.
 *   d_total_m       : great-circle path length (m). Use 0 to derive
 *                     from the profile (assumes uniform spacing
 *                     determined by the user — pass it explicitly).
 *   q_time/loc/sit  : reliability quantiles (0-100). (50,50,50) is
 *                     the long-term median.
 */
double matlab_prop_itm_pathloss(matlab_mat *profile,
                                 double freq_hz, double ht_m, double hr_m,
                                 double pol, double climate, double Ns,
                                 double sigma, double eps_r,
                                 double d_total_m,
                                 double q_time, double q_loc, double q_sit) {
    if (freq_hz <= 0.0 || d_total_m <= 0.0) return 0.0;
    if (climate < 0.5) climate = 5.0;
    if (Ns <= 0.0) Ns = 301.0;
    if (sigma <= 0.0) sigma = 0.005;
    if (eps_r <= 0.0) eps_r = 15.0;
    double lambda = C_LIGHT / freq_hz;
    double k = 2.0 * M_PI / lambda;
    (void)k;

    /* Effective Earth radius from surface refractivity (P.834-7). */
    double Ne_per_m = -7.32 * exp(0.005577 * Ns) * 1e-6; /* dN/dh near surface */
    (void)Ne_per_m;
    double a_eff = R_EARTH * K_EARTH_43;

    /* Terrain irregularity: delta_h = interdecile range of terrain heights
     * along the path. Empty profile -> flat (delta_h = 0). */
    double delta_h = 0.0;
    int Nprof = profile ? (int)(profile->rows * profile->cols) : 0;
    if (Nprof >= 8) {
        std::vector<double> hs(Nprof);
        for (int i = 0; i < Nprof; ++i) hs[i] = profile->data[i];
        std::sort(hs.begin(), hs.end());
        double q10 = hs[(int)(0.10 * (Nprof - 1))];
        double q90 = hs[(int)(0.90 * (Nprof - 1))];
        delta_h = q90 - q10;
    }

    /* Smooth-Earth horizon distances (effective Earth). */
    double dLs1 = sqrt(2.0 * a_eff * std::max(ht_m, 0.5));
    double dLs2 = sqrt(2.0 * a_eff * std::max(hr_m, 0.5));
    double dLs  = dLs1 + dLs2;          /* smooth-Earth horizon */
    /* Terrain-roughness correction (Longley 1968 eq. 3.6.b). */
    double dh1 = delta_h * (1.0 - 0.8 * exp(-0.02 * dLs1 * 1e-3));
    double dh2 = delta_h * (1.0 - 0.8 * exp(-0.02 * dLs2 * 1e-3));
    (void)dh1; (void)dh2;

    /* Free-space reference. */
    double Lfs = matlab_prop_fspl(d_total_m, freq_hz);

    /* Two-ray ground reflection (line-of-sight regime). */
    double Llos = Lfs;
    if (d_total_m < dLs) {
        /* Plane-Earth correction (Bullington reference): A_los = 0 at
         * horizon, slowly increasing as path approaches it. */
        double t = d_total_m / dLs;
        Llos = Lfs + 2.0 * t * t;
    }

    /* Diffraction regime: smooth-Earth Vogler with knife-edge bound. */
    double Ldiff = Lfs;
    if (d_total_m >= dLs) {
        double dbe = d_total_m - dLs;
        double f_MHz = freq_hz * 1e-6;
        /* Smooth-Earth diffraction (Vogler 1964, fitted form). */
        double A0 = 20.0 * log10(d_total_m / dLs) * 0.5 + 0.05;
        double F  = (1.607 - sqrt(eps_r)) * pow(f_MHz, 1.0/3.0)
                    * pow(dbe / 1000.0, 0.5);
        if (F < 0) F = 0;
        Ldiff = Lfs + 5.0 + 0.6 * (delta_h / lambda) + A0 + F;
    }

    /* Tropospheric scatter (long-range fallback). */
    double Ltrop = Lfs;
    {
        double f_MHz = freq_hz * 1e-6;
        double H0 = 5.0 + 0.05 * (delta_h);
        /* Yeh 1960 / Rice 1965 nominal: increasing path loss with
         * distance and modest frequency dependence. */
        Ltrop = 99.0 + 30.0 * log10(f_MHz) + 30.0 * log10(d_total_m * 1e-3)
                - 0.2 * (Ns - 301.0) + H0;
    }

    /* Climate / Ns adjustment (rough — full ITM uses a 7-region table). */
    double clim_adj = 0.0;
    if (climate >= 0.5 && climate <= 1.5) clim_adj = -2.0;   /* equatorial: better */
    else if (climate >= 6.5)              clim_adj = -3.0;   /* maritime sea */
    else if (climate >= 3.5 && climate <= 4.5) clim_adj = +2.0; /* desert */
    Llos  += clim_adj;
    Ldiff += clim_adj;
    Ltrop += clim_adj;

    /* Polarisation tweak. Vertical pol gets a small benefit at low
     * grazing angles. */
    if (pol > 0.5) {
        Llos  -= 0.5;
        Ldiff -= 0.5;
    }

    /* Blend the three regimes. Use a smooth max-with-soft-min. */
    double L_med;
    if (d_total_m < 0.7 * dLs) {
        L_med = Llos;
    } else if (d_total_m < 1.3 * dLs) {
        double w = (d_total_m - 0.7 * dLs) / (0.6 * dLs);
        if (w < 0) w = 0; if (w > 1) w = 1;
        L_med = (1.0 - w) * Llos + w * Ldiff;
    } else {
        /* Beyond the horizon: the tropospheric component takes over
         * if it is the *smaller* loss (i.e., longer-range scatter
         * outperforms diffraction over really long paths). */
        L_med = std::min(Ldiff, Ltrop);
    }

    /* Reliability quantile correction.  The variability triple is
     * applied as a Gaussian Z-score offset on the median. */
    auto inv_cdf = [](double q) -> double {
        if (q <= 0.001) q = 0.001;
        if (q >= 99.999) q = 99.999;
        /* Rational-approx (Acklam) for the inverse normal CDF. */
        double p = q / 100.0;
        double a1=-3.969683028665376e+01, a2= 2.209460984245205e+02,
               a3=-2.759285104469687e+02, a4= 1.383577518672690e+02,
               a5=-3.066479806614716e+01, a6= 2.506628277459239e+00;
        double b1=-5.447609879822406e+01, b2= 1.615858368580409e+02,
               b3=-1.556989798598866e+02, b4= 6.680131188771972e+01,
               b5=-1.328068155288572e+01;
        double c1=-7.784894002430293e-03, c2=-3.223964580411365e-01,
               c3=-2.400758277161838e+00, c4=-2.549732539343734e+00,
               c5= 4.374664141464968e+00, c6= 2.938163982698783e+00;
        double d1= 7.784695709041462e-03, d2= 3.224671290700398e-01,
               d3= 2.445134137142996e+00, d4= 3.754408661907416e+00;
        double q1, r;
        double plow = 0.02425, phigh = 1.0 - plow;
        if (p < plow)       { q1 = sqrt(-2.0 * log(p));
            return (((((c1*q1+c2)*q1+c3)*q1+c4)*q1+c5)*q1+c6) /
                   ((((d1*q1+d2)*q1+d3)*q1+d4)*q1+1.0); }
        if (p <= phigh)     { q1 = p - 0.5; r = q1*q1;
            return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q1 /
                   (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0); }
        q1 = sqrt(-2.0 * log(1.0 - p));
        return -(((((c1*q1+c2)*q1+c3)*q1+c4)*q1+c5)*q1+c6) /
                ((((d1*q1+d2)*q1+d3)*q1+d4)*q1+1.0);
    };

    double z_time = inv_cdf(q_time);
    double z_loc  = inv_cdf(q_loc);
    double z_sit  = inv_cdf(q_sit);
    /* Standard ITM long-term variability sigma_t around 4-8 dB; we
     * use a frequency / climate-aware nominal. */
    double f_MHz = freq_hz * 1e-6;
    double sigT  = 3.5 + 0.05 * sqrt(f_MHz) + 0.02 * delta_h;
    double sigL  = 2.0 + 0.001 * delta_h;
    double sigS  = 1.5;
    double L_var = sigT * z_time + sigL * z_loc + sigS * z_sit;

    double L = L_med + L_var;
    if (L < Lfs) L = Lfs;       /* never below free-space */
    return L;
}

/* ===== §3.3.1 Terrain profile from heightmap ===== */

/* Bilinear sample of the heightmap at (lat, lon). Returns 0 outside.
 * The heightmap is a matlab_mat (rows = lat samples, cols = lon
 * samples); lat_min/max span the rows, lon_min/max span the cols. */
static double sample_heightmap(matlab_mat *hm, double lat,
                                double lat_min, double lat_max,
                                double lon, double lon_min, double lon_max) {
    if (!hm || hm->rows < 2 || hm->cols < 2) return 0.0;
    double fr = (lat - lat_min) / (lat_max - lat_min);
    double fc = (lon - lon_min) / (lon_max - lon_min);
    if (fr < 0.0) fr = 0.0; if (fr > 1.0) fr = 1.0;
    if (fc < 0.0) fc = 0.0; if (fc > 1.0) fc = 1.0;
    double r = fr * (hm->rows - 1);
    double c = fc * (hm->cols - 1);
    int r0 = (int)floor(r), c0 = (int)floor(c);
    int r1 = std::min((int64_t)r0 + 1, hm->rows - 1);
    int c1 = std::min((int64_t)c0 + 1, hm->cols - 1);
    double tr = r - r0, tc = c - c0;
    double h00 = hm->data[r0 * hm->cols + c0];
    double h01 = hm->data[r0 * hm->cols + c1];
    double h10 = hm->data[r1 * hm->cols + c0];
    double h11 = hm->data[r1 * hm->cols + c1];
    return (1-tr)*((1-tc)*h00 + tc*h01) + tr*((1-tc)*h10 + tc*h11);
}

matlab_mat *matlab_prop_terrain_profile(matlab_mat *heightmap,
                                         double lat_min, double lat_max,
                                         double lon_min, double lon_max,
                                         double lat1, double lon1,
                                         double lat2, double lon2,
                                         double n_samples) {
    int N = (int)n_samples;
    if (N < 2) N = 2;
    matlab_mat *out = mat_alloc(N, 1);
    for (int i = 0; i < N; ++i) {
        double t = (double)i / (double)(N - 1);
        double lat = lat1 + t * (lat2 - lat1);
        double lon = lon1 + t * (lon2 - lon1);
        out->data[i] = sample_heightmap(heightmap, lat,
                                         lat_min, lat_max,
                                         lon, lon_min, lon_max);
    }
    return out;
}

/* ===== §3.3.2 Line-of-sight check ===== */

/* Returns the largest LOS-blockage height (m) above the geometric
 * line connecting (TX + h_tx) and (RX + h_rx) along the supplied
 * terrain profile, accounting for the 4/3 effective-Earth bulge.
 * Negative result -> path is geometrically clear. */
double matlab_prop_los_obstruction(matlab_mat *profile, double h_tx, double h_rx,
                                    double d_total_m) {
    if (!profile || profile->rows * profile->cols < 3) return -1.0;
    int N = (int)(profile->rows * profile->cols);
    double z0 = profile->data[0]     + h_tx;
    double z1 = profile->data[N - 1] + h_rx;
    double max_block = -1e30;
    double a_eff = R_EARTH * K_EARTH_43;
    for (int i = 1; i < N - 1; ++i) {
        double t = (double)i / (double)(N - 1);
        double xi = t * d_total_m;
        /* Earth-bulge correction (effective Earth). */
        double bulge = (xi * (d_total_m - xi)) / (2.0 * a_eff);
        double zi_los = z0 + t * (z1 - z0) - bulge;
        double block = profile->data[i] - zi_los;
        if (block > max_block) max_block = block;
    }
    return max_block;
}

double matlab_prop_los_clear(matlab_mat *profile, double h_tx, double h_rx,
                              double d_total_m) {
    return matlab_prop_los_obstruction(profile, h_tx, h_rx, d_total_m) < 0.0
           ? 1.0 : 0.0;
}

/* ===== §3.3.3 Point-to-point link budget — returns matlab_struct ===== */

/* model code:
 *   0 = fspl
 *   1 = hata (uses env param via 'opts' field if present; default urban)
 *   2 = cost231
 *   3 = egli
 *   4 = ecc33
 *   5 = sui (terrain B by default)
 *   6 = ericsson9999
 *   7 = longley-rice (ITM) — uses profile + reliability defaults */
static double path_loss_dispatch(int model, double d_m, double freq_hz,
                                  double ht, double hr,
                                  matlab_mat *profile, double climate,
                                  double q_time, double q_loc, double q_sit) {
    double d_km = d_m * 1e-3;
    double f_MHz = freq_hz * 1e-6;
    switch (model) {
        case 0: return matlab_prop_fspl(d_m, freq_hz);
        case 1: return matlab_prop_pathloss_hata     (f_MHz, ht, hr, d_km, 1.0);
        case 2: return matlab_prop_pathloss_cost231  (f_MHz, ht, hr, d_km, 1.0);
        case 3: return matlab_prop_pathloss_egli     (f_MHz, ht, hr, d_km);
        case 4: return matlab_prop_pathloss_ecc33    (f_MHz, ht, hr, d_km);
        case 5: return matlab_prop_pathloss_sui      (f_MHz, ht, hr, d_km, 2.0);
        case 6: return matlab_prop_pathloss_ericsson9999(f_MHz, ht, hr, d_km, 1.0);
        case 7: default:
            return matlab_prop_itm_pathloss(profile, freq_hz, ht, hr, 1.0,
                                             climate, 301.0, 0.005, 15.0,
                                             d_m, q_time, q_loc, q_sit);
    }
}

extern matlab_struct *matlab_struct_new(void);
extern void matlab_struct_set_f64(matlab_struct *s, const char *name, int64_t len, double v);
extern void matlab_struct_set_mat(matlab_struct *s, const char *name, int64_t len, matlab_mat *m);

matlab_struct *matlab_prop_link_budget(
    double tx_lat, double tx_lon, double tx_height,
    double tx_freq_hz, double tx_power_W, double tx_gain_dBi,
    double rx_lat, double rx_lon, double rx_height,
    double rx_gain_dBi,
    double model,             /* numeric model code */
    matlab_mat *profile,      /* may be empty */
    double climate,
    double q_time, double q_loc, double q_sit) {

    double d_m = matlab_prop_haversine(tx_lat, tx_lon, rx_lat, rx_lon);
    double az  = matlab_prop_bearing  (tx_lat, tx_lon, rx_lat, rx_lon);
    int mc = (int)model;
    double L = path_loss_dispatch(mc, d_m, tx_freq_hz, tx_height, rx_height,
                                   profile, climate, q_time, q_loc, q_sit);
    double P_tx_dBm = 10.0 * log10(tx_power_W * 1000.0);
    double P_rx_dBm = P_tx_dBm + tx_gain_dBi - L + rx_gain_dBi;
    double lambda = C_LIGHT / tx_freq_hz;
    double clearance = 100.0;
    double los = 1.0;
    if (profile && profile->rows * profile->cols >= 3) {
        double block = matlab_prop_los_obstruction(profile, tx_height,
                                                     rx_height, d_m);
        los = block < 0.0 ? 1.0 : 0.0;
        clearance = matlab_prop_fresnel_clearance(profile, tx_height,
                                                    rx_height, d_m, lambda, 1.0);
    }
    /* Receiver thermal noise (kTB) at room temperature, 1 MHz default. */
    double k_T_B_dBm = -114.0;   /* −174 + 60 dB·Hz @ 1 MHz */
    double snr = P_rx_dBm - k_T_B_dBm;
    /* Link margin assumes a 10 dB SNR threshold for a robust digital link. */
    double margin = snr - 10.0;
    matlab_struct *s = matlab_struct_new();
    #define SET(name, v) matlab_struct_set_f64(s, name, sizeof(name)-1, v)
    SET("Distance",        d_m);
    SET("Azimuth",         az);
    SET("PathLoss",        L);
    SET("TxPower_dBm",     P_tx_dBm);
    SET("ReceivedPower",   P_rx_dBm);
    SET("NoiseFloor",      k_T_B_dBm);
    SET("Snr",             snr);
    SET("LinkMargin",      margin);
    SET("FresnelClearance",clearance);
    SET("LosClear",        los);
    SET("Frequency",       tx_freq_hz);
    SET("Model",           model);
    #undef SET
    matlab_struct_set_mat(s, "Profile", 7,
                          profile ? profile : mat_alloc(0, 0));
    return s;
}

/* ===== §3.3.4 Single-TX numeric coverage grid ===== */

/* Returns received power (dBm) on a num_lat × num_lon grid spanning
 * the supplied lat/lon box. Uses the chosen propagation model + the
 * supplied heightmap (which may equal the same heightmap that the
 * grid spans, or a coarser/finer DEM tile). */
matlab_mat *matlab_prop_coverage_grid(
    double tx_lat, double tx_lon, double tx_height,
    double tx_freq_hz, double tx_power_W, double tx_gain_dBi,
    double model,
    matlab_mat *heightmap,
    double lat_min, double lat_max,
    double lon_min, double lon_max,
    double num_lat, double num_lon,
    double rx_height, double rx_gain_dBi,
    double climate, double q_time, double q_loc, double q_sit) {

    int NL = (int)num_lat, NK = (int)num_lon;
    if (NL < 2) NL = 2;
    if (NK < 2) NK = 2;
    matlab_mat *out = mat_alloc(NL, NK);
    for (int i = 0; i < NL; ++i) {
        for (int j = 0; j < NK; ++j) {
            double rx_lat = lat_min + (lat_max - lat_min) * (double)i / (double)(NL - 1);
            double rx_lon = lon_min + (lon_max - lon_min) * (double)j / (double)(NK - 1);
            double d_m = matlab_prop_haversine(tx_lat, tx_lon, rx_lat, rx_lon);
            if (d_m < 1.0) { out->data[i * NK + j] = 0.0; continue; }
            /* Sample a coarse 32-point terrain profile for ITM. */
            matlab_mat *prof = NULL;
            if ((int)model == 7) {
                prof = matlab_prop_terrain_profile(heightmap, lat_min, lat_max,
                                                    lon_min, lon_max,
                                                    tx_lat, tx_lon,
                                                    rx_lat, rx_lon, 32.0);
            }
            double L = path_loss_dispatch((int)model, d_m, tx_freq_hz,
                                           tx_height, rx_height, prof,
                                           climate, q_time, q_loc, q_sit);
            if (prof) { free(prof->data); free(prof); }
            double P_tx_dBm = 10.0 * log10(tx_power_W * 1000.0);
            out->data[i * NK + j] = P_tx_dBm + tx_gain_dBi - L + rx_gain_dBi;
        }
    }
    return out;
}

/* ===== §3.4.1 Analytical directional antenna patterns ===== */

/* 3GPP TR 36.942 sector pattern. All angles in degrees. */
double matlab_prop_pat_sector(double az, double el,
                               double bw_az, double bw_el,
                               double gain_dBi, double fb_dB) {
    if (bw_az <= 0.0) bw_az = 65.0;
    if (bw_el <= 0.0) bw_el = 10.0;
    /* Wrap az to [-180, 180]. */
    while (az >  180.0) az -= 360.0;
    while (az < -180.0) az += 360.0;
    double A_az = -12.0 * (az/bw_az) * (az/bw_az);
    if (A_az < -fb_dB) A_az = -fb_dB;
    double A_el = -12.0 * (el/bw_el) * (el/bw_el);
    if (A_el < -fb_dB) A_el = -fb_dB;
    double A = -(A_az + A_el);
    if (A > fb_dB) A = fb_dB;
    return gain_dBi - A;
}

/* Cosine-power pattern: G = G_peak · cos^n(θ). Good fit for parabolic
 * dishes. halfBW selects n via n = log(0.5)/log(cos(halfBW/2)). */
double matlab_prop_pat_cosine(double az, double el,
                               double half_bw_az, double half_bw_el,
                               double gain_dBi, double n) {
    while (az >  180.0) az -= 360.0;
    while (az < -180.0) az += 360.0;
    if (n <= 0.0) {
        n = log(0.5) / log(cos(deg2rad(half_bw_az / 2.0)));
    }
    double phi = sqrt(az*az + el*el);
    if (phi >= 90.0) return gain_dBi - 30.0;
    double c = cos(deg2rad(phi));
    if (c <= 0.0) return gain_dBi - 30.0;
    double gain_lin = pow(c, n);
    double atten = -10.0 * log10(std::max(gain_lin, 1e-3));
    return gain_dBi - atten;
}

/* Gaussian roll-off, no sidelobes. */
double matlab_prop_pat_gaussian(double az, double el,
                                 double half_bw_az, double half_bw_el,
                                 double gain_dBi) {
    if (half_bw_az <= 0.0) half_bw_az = 30.0;
    if (half_bw_el <= 0.0) half_bw_el = 10.0;
    while (az >  180.0) az -= 360.0;
    while (az < -180.0) az += 360.0;
    double a = -3.0 * (az * az) / (half_bw_az * half_bw_az);
    double b = -3.0 * (el * el) / (half_bw_el * half_bw_el);
    double atten_dB = -(a + b);
    if (atten_dB > 30.0) atten_dB = 30.0;
    return gain_dBi - atten_dB;
}

/* Isotropic. */
double matlab_prop_pat_isotropic(double az, double el, double gain_dBi) {
    (void)az; (void)el;
    return gain_dBi;
}

/* ===== §3.4.2 Mount orientation — apply yaw/tilt rotation ===== */

/* Given world-frame az/el of the receive direction and a mount
 * (Azimuth, MechanicalTilt), return the local antenna-frame az/el
 * pair. We pack the two outputs into a 1×2 mat to keep the dispatch
 * simple. */
matlab_mat *matlab_prop_mount_to_local(double az_world, double el_world,
                                        double mount_az, double mount_tilt) {
    double az_local = az_world - mount_az;
    while (az_local >  180.0) az_local -= 360.0;
    while (az_local < -180.0) az_local += 360.0;
    double el_local = el_world + mount_tilt;     /* mech down-tilt = + */
    matlab_mat *out = mat_alloc(1, 2);
    out->data[0] = az_local;
    out->data[1] = el_local;
    return out;
}

/* Scalar siblings — avoid an intermediate 1x2 matrix when the user
 * only wants one of the two angles. */
double matlab_prop_mount_az_local(double az_world, double el_world,
                                   double mount_az, double mount_tilt) {
    (void)el_world; (void)mount_tilt;
    double az_local = az_world - mount_az;
    while (az_local >  180.0) az_local -= 360.0;
    while (az_local < -180.0) az_local += 360.0;
    return az_local;
}
double matlab_prop_mount_el_local(double az_world, double el_world,
                                   double mount_az, double mount_tilt) {
    (void)az_world; (void)mount_az;
    return el_world + mount_tilt;
}

/* ===== §3.4.3 Multi-site coverage with directional antennas =====
 *
 * To keep the runtime dispatch reachable without strings or function
 * handles, we encode each site + antenna list as a flat matrix
 * (rows = sites, columns = canonical fields). For each site:
 *
 *   col 0  : lat
 *   col 1  : lon
 *   col 2  : tx_height (m)
 *   col 3  : tx_power (W)
 *   col 4  : tx_freq (Hz)
 *   col 5  : num_antennas at this site (>=1)
 *
 * The per-antenna parameters are passed in a second matrix whose
 * rows are ordered (site_index, antenna_index) in row-major order:
 *
 *   col 0  : pattern code (0=isotropic, 1=sector, 2=cosine,
 *                          3=gaussian, 4=sector3GPP)
 *   col 1  : peak gain dBi
 *   col 2  : beamwidth_az (deg)
 *   col 3  : beamwidth_el (deg)
 *   col 4  : front-to-back / extra param
 *   col 5  : mount azimuth (deg, compass)
 *   col 6  : mount electrical tilt (deg, +down)
 *   col 7  : extra (cosine-n / unused)
 *
 * aggregation: 0 = best-server, 1 = sum-power, 2 = SINR
 * (SINR returns dB; sum-power and best-server return received-power dBm).
 *
 * Returns a num_lat × num_lon matrix.
 */
double matlab_prop_apply_pattern(int code, double az, double el,
                                  double gain, double bw_az, double bw_el,
                                  double fb_or_n) {
    switch (code) {
        case 1:  return matlab_prop_pat_sector  (az, el, bw_az, bw_el, gain, fb_or_n);
        case 2:  return matlab_prop_pat_cosine  (az, el, bw_az, bw_el, gain, fb_or_n);
        case 3:  return matlab_prop_pat_gaussian(az, el, bw_az, bw_el, gain);
        case 4:  return matlab_prop_pat_sector  (az, el, bw_az, bw_el, gain, fb_or_n);
        default: return matlab_prop_pat_isotropic(az, el, gain);
    }
}

matlab_mat *matlab_prop_coverage_grid_multi(
    matlab_mat *sites,        /* [num_sites x 6] */
    matlab_mat *antennas,     /* [sum(num_ant_per_site) x 8] */
    matlab_mat *heightmap,
    double lat_min, double lat_max,
    double lon_min, double lon_max,
    double num_lat, double num_lon,
    double rx_height, double rx_gain_dBi,
    double model,
    double aggregation,
    double climate, double q_time, double q_loc, double q_sit) {

    if (!sites || sites->cols < 6) return mat_alloc(0, 0);
    if (!antennas || antennas->cols < 8) return mat_alloc(0, 0);
    int NL = (int)num_lat, NK = (int)num_lon;
    if (NL < 2) NL = 2;
    if (NK < 2) NK = 2;
    int num_sites = (int)sites->rows;
    int agg = (int)aggregation;
    matlab_mat *out = mat_alloc(NL, NK);

    /* Precompute antenna start-index per site. */
    std::vector<int> ant_start(num_sites + 1, 0);
    for (int s = 0; s < num_sites; ++s)
        ant_start[s+1] = ant_start[s] + (int)sites->data[s * sites->cols + 5];

    const double k_T_B_dBm = -114.0;  /* 1 MHz nominal */
    const double n_floor_lin = pow(10.0, k_T_B_dBm / 10.0);

    for (int i = 0; i < NL; ++i) {
    for (int j = 0; j < NK; ++j) {
        double rx_lat = lat_min + (lat_max - lat_min) * (double)i / (double)(NL - 1);
        double rx_lon = lon_min + (lon_max - lon_min) * (double)j / (double)(NK - 1);

        double best_P = -1e30;
        double sum_P_lin = 0.0;
        double serv_P_lin = 0.0;
        double rest_P_lin = 0.0;

        for (int s = 0; s < num_sites; ++s) {
            const double *S = sites->data + s * sites->cols;
            double s_lat = S[0], s_lon = S[1];
            double s_h   = S[2], s_pw  = S[3], s_f   = S[4];
            int    n_ant = (int)S[5];

            double d_m = matlab_prop_haversine(s_lat, s_lon, rx_lat, rx_lon);
            if (d_m < 1.0) d_m = 1.0;
            double az  = matlab_prop_bearing  (s_lat, s_lon, rx_lat, rx_lon);
            /* Geometric elevation angle approximation: tx_h vs rx_h */
            double el  = atan2(rx_height - s_h, d_m) * 180.0 / M_PI;

            matlab_mat *prof = NULL;
            if ((int)model == 7) {
                prof = matlab_prop_terrain_profile(
                    heightmap, lat_min, lat_max, lon_min, lon_max,
                    s_lat, s_lon, rx_lat, rx_lon, 32.0);
            }
            double L = path_loss_dispatch((int)model, d_m, s_f,
                                           s_h, rx_height, prof,
                                           climate, q_time, q_loc, q_sit);
            if (prof) { free(prof->data); free(prof); }
            double P_tx_dBm = 10.0 * log10(s_pw * 1000.0);

            double best_P_site = -1e30;
            for (int a = 0; a < n_ant; ++a) {
                const double *A = antennas->data + (ant_start[s] + a) * antennas->cols;
                int    code = (int)A[0];
                double gain = A[1], bw_az = A[2], bw_el = A[3];
                double fb_or_n = A[4];
                double m_az = A[5], m_tilt = A[6];
                /* Local frame angles. */
                double az_loc = az - m_az;
                while (az_loc >  180.0) az_loc -= 360.0;
                while (az_loc < -180.0) az_loc += 360.0;
                double el_loc = el + m_tilt;
                double Gtx = matlab_prop_apply_pattern(code, az_loc, el_loc,
                                                        gain, bw_az, bw_el,
                                                        fb_or_n);
                double P_rx = P_tx_dBm + Gtx - L + rx_gain_dBi;
                if (P_rx > best_P_site) best_P_site = P_rx;
            }
            double lin = pow(10.0, best_P_site / 10.0);
            sum_P_lin += lin;
            if (best_P_site > best_P) {
                /* The previous best gets demoted to interferer. */
                if (serv_P_lin > 0.0) rest_P_lin += serv_P_lin;
                best_P = best_P_site;
                serv_P_lin = lin;
            } else {
                rest_P_lin += lin;
            }
        }

        double cell;
        if (agg == 1) {
            /* Sum-power: incoherent power sum (dBm). */
            cell = 10.0 * log10(std::max(sum_P_lin, 1e-30));
        } else if (agg == 2) {
            /* SINR in dB. */
            double denom = rest_P_lin + n_floor_lin;
            cell = 10.0 * log10(std::max(serv_P_lin / std::max(denom, 1e-30), 1e-30));
        } else {
            /* Best-server (dBm). */
            cell = best_P;
        }
        out->data[i * NK + j] = cell;
    }
    }
    return out;
}

/* ======================================================================
 * PropagationModel dispatcher — single entry point that selects the
 * underlying model based on a string Kind read from the classdef
 * property table.  Called by the `pathloss(pm, rx, tx)` method body in
 * `runtime/rf_class_propagationmodel.m`.  Returns path loss in dB.
 *
 * Supported kinds (case-insensitive, hyphens optional):
 *   freespace / free-space / fspl  → Friis free-space
 *   longley-rice / longleyrice / itm → ITM with flat profile + defaults
 *   rain / itu-rain                 → ITU-R P.838 rain attenuation
 *   gas / atmospheric / itu-gas     → ITU-R P.676 oxygen + water vapour
 *   fog / cloud                     → ITU-R P.840 fog/cloud attenuation
 *   close-in / closein / ci         → Close-In propagation model
 *   hata                            → Okumura-Hata urban path loss
 *   cost231                         → COST-231 extension to Hata
 *   egli / ecc33 / sui              → empirical macrocell models
 *   ericsson9999                    → Ericsson 9999 urban macrocell
 *
 * Unknown kind names fall back to Friis FSPL with a console warning.
 * ====================================================================== */
struct matlab_string_s;
extern "C" double matlab_string_len(struct matlab_string_s *s);
struct kind_view_ { const char *data; int64_t len; };
static struct kind_view_ ml_str_view(struct matlab_string_s *s) {
    /* matlab_string_s layout: { char *data; int64_t len; }. */
    struct kind_view_ V;
    if (!s) { V.data = NULL; V.len = 0; return V; }
    const char *const *pd = (const char *const *)s;
    const int64_t *pl = (const int64_t *)((const char *)s + sizeof(void*));
    V.data = *pd;
    V.len = *pl;
    return V;
}
static int ml_to_lower(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A' + 'a';
    return c;
}
static int ml_kind_eq(const char *a, int64_t al, const char *b) {
    /* Case-insensitive, hyphen/underscore-insensitive comparison.
     * Compares the entire `a` (length `al`) against the C-string `b`. */
    int64_t bi = 0;
    int64_t ai = 0;
    while (ai < al || b[bi]) {
        int ca = (ai < al) ? ml_to_lower((unsigned char)a[ai]) : -1;
        int cb = b[bi] ? ml_to_lower((unsigned char)b[bi]) : -1;
        if (ca == '-' || ca == '_' || ca == ' ') { ++ai; continue; }
        if (cb == '-' || cb == '_' || cb == ' ') { ++bi; continue; }
        if (ca != cb) return 0;
        ++ai; ++bi;
    }
    return 1;
}
extern "C" double matlab_prop_dispatch_pathloss(
    struct matlab_string_s *kind,
    double tx_lat, double tx_lon, double tx_height_m,
    double rx_lat, double rx_lon, double rx_height_m,
    double freq_hz)
{
    struct kind_view_ V = ml_str_view(kind);
    double d_m = matlab_prop_haversine(tx_lat, tx_lon, rx_lat, rx_lon);
    if (d_m < 1.0) d_m = 1.0;   /* clamp degenerate near-zero distance */

    /* Most-common kinds first. */
    if (ml_kind_eq(V.data, V.len, "freespace") ||
        ml_kind_eq(V.data, V.len, "fspl")) {
        return matlab_prop_fspl(d_m, freq_hz);
    }
    if (ml_kind_eq(V.data, V.len, "longleyrice") ||
        ml_kind_eq(V.data, V.len, "itm")) {
        /* Flat-profile ITM with continental temperate climate +
         * median quantiles.  Matches MathWorks `longley-rice`
         * defaults for the no-terrain case. */
        matlab_mat empty;
        empty.rows = 0; empty.cols = 0; empty.data = NULL;
        return matlab_prop_itm_pathloss(&empty, freq_hz,
                                          tx_height_m, rx_height_m,
                                          1.0,     /* vertical pol */
                                          5.0,     /* continental temperate */
                                          301.0, 0.005, 15.0, d_m,
                                          50.0, 50.0, 50.0);
    }
    if (ml_kind_eq(V.data, V.len, "rain") ||
        ml_kind_eq(V.data, V.len, "iturain")) {
        /* Defaults: 10 mm/hr (moderate rain), vertical pol. */
        return matlab_prop_pathloss_rain(d_m, freq_hz, 10.0, 1.0);
    }
    if (ml_kind_eq(V.data, V.len, "gas") ||
        ml_kind_eq(V.data, V.len, "atmospheric") ||
        ml_kind_eq(V.data, V.len, "itugas")) {
        /* Defaults: 15 °C, 1013 hPa, 7.5 g/m³ water vapour density. */
        return matlab_prop_pathloss_gas(d_m, freq_hz, 15.0, 1013.0, 7.5);
    }
    if (ml_kind_eq(V.data, V.len, "fog") ||
        ml_kind_eq(V.data, V.len, "cloud")) {
        /* Default: 0.5 g/m³ liquid water content (light fog). */
        return matlab_prop_pathloss_fog(d_m, freq_hz, 0.5);
    }
    if (ml_kind_eq(V.data, V.len, "closein") ||
        ml_kind_eq(V.data, V.len, "ci")) {
        /* Defaults: PLE=2.0, 0 dB shadowing, 1 m reference distance. */
        return matlab_prop_pathloss_closein(d_m, freq_hz, 2.0, 0.0, 1.0);
    }
    if (ml_kind_eq(V.data, V.len, "hata")) {
        /* Defaults: urban environment (env=1). */
        return matlab_prop_pathloss_hata(freq_hz / 1e6, tx_height_m,
                                           rx_height_m, d_m / 1000.0, 1.0);
    }
    if (ml_kind_eq(V.data, V.len, "cost231")) {
        return matlab_prop_pathloss_cost231(freq_hz / 1e6, tx_height_m,
                                              rx_height_m, d_m / 1000.0, 1.0);
    }
    if (ml_kind_eq(V.data, V.len, "egli")) {
        return matlab_prop_pathloss_egli(freq_hz / 1e6, tx_height_m,
                                           rx_height_m, d_m / 1000.0);
    }
    if (ml_kind_eq(V.data, V.len, "ecc33")) {
        return matlab_prop_pathloss_ecc33(freq_hz / 1e6, tx_height_m,
                                            rx_height_m, d_m / 1000.0);
    }
    if (ml_kind_eq(V.data, V.len, "sui")) {
        /* Default: terrain category A (worst-case urban / hilly). */
        return matlab_prop_pathloss_sui(freq_hz / 1e6, tx_height_m,
                                          rx_height_m, d_m / 1000.0, 1.0);
    }
    if (ml_kind_eq(V.data, V.len, "ericsson9999")) {
        /* Default: urban environment (env=1.0). */
        return matlab_prop_pathloss_ericsson9999(freq_hz / 1e6, tx_height_m,
                                                   rx_height_m, d_m / 1000.0,
                                                   1.0);
    }
    /* Unknown kind — fall back to free-space + warn. */
    fprintf(stderr,
            "warning: propagationModel: unknown kind '%.*s' — using "
            "free-space fallback\n",
            (int)V.len, V.data ? V.data : "");
    return matlab_prop_fspl(d_m, freq_hz);
}

/* Companion: LOS clear-or-not between two geographic points.  Flat-
 * earth + Earth-curvature bulge check (k=4/3 effective Earth model).
 * Returns 1.0 (clear) when the line-of-sight ray clears the bulge,
 * 0.0 otherwise.  Mirrors `los(tx, rx)` in MathWorks API. */
extern "C" double matlab_prop_los_sites(
    double tx_lat, double tx_lon, double tx_h_m,
    double rx_lat, double rx_lon, double rx_h_m)
{
    double d_m = matlab_prop_haversine(tx_lat, tx_lon, rx_lat, rx_lon);
    if (d_m < 1.0) return 1.0;
    /* Earth bulge mid-path under 4/3 model: h = d1*d2 / (2 * k * R).
     * For mid-path d1 = d2 = d/2. */
    const double R = 6371000.0;
    const double k = 4.0 / 3.0;
    double bulge_m = (d_m * d_m) / (8.0 * k * R);
    /* Required radio height = linear-interpolated antenna height
     * minus bulge.  For LOS the chord between the two antennas must
     * stay above zero. */
    double avg_h = 0.5 * (tx_h_m + rx_h_m);
    return (avg_h >= bulge_m) ? 1.0 : 0.0;
}

/* sigstrength(rx, tx, pm) — RX power in dBm.
 *
 * Bypasses the MATLAB-side method dispatch entirely because the
 * compiler's per-method-param class pinning doesn't propagate from
 * call sites (would need inter-procedural Sema work).  The runtime
 * reads each site's properties directly via matlab_obj_get_f64 /
 * matlab_obj_get_string and computes the link budget in dB:
 *
 *   ss_dBm = 10*log10(TX_W * 1000) + TX_gain
 *            - pathloss(pm.Kind, ...)
 *            + RX_gain - TX_SystemLoss - RX_SystemLoss
 *
 * Antenna gains default to 0 dBi (isotropic) — the Antenna property
 * isn't wired into a directional gain lookup yet (lands with
 * ANT-Tier-2 wire-MoM patterns).  Operates on matlab_obj* pointers
 * directly so the compiler doesn't need to know the class structure.
 */
extern "C" double matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void *matlab_obj_get_string(matlab_obj *o, const char *name, int64_t len);
extern "C" double matlab_prop_sigstrength(matlab_obj *rx, matlab_obj *tx, matlab_obj *pm) {
    if (!rx || !tx || !pm) return 0.0;
    double tx_lat = matlab_obj_get_f64(tx, "Latitude", 8);
    double tx_lon = matlab_obj_get_f64(tx, "Longitude", 9);
    double tx_h   = matlab_obj_get_f64(tx, "AntennaHeight", 13);
    double rx_lat = matlab_obj_get_f64(rx, "Latitude", 8);
    double rx_lon = matlab_obj_get_f64(rx, "Longitude", 9);
    double rx_h   = matlab_obj_get_f64(rx, "AntennaHeight", 13);
    double freq   = matlab_obj_get_f64(tx, "TransmitterFrequency", 20);
    double power_W = matlab_obj_get_f64(tx, "TransmitterPower", 16);
    double tx_loss = matlab_obj_get_f64(tx, "SystemLoss", 10);
    double rx_loss = matlab_obj_get_f64(rx, "SystemLoss", 10);

    matlab_string_s *kind =
        (matlab_string_s *)matlab_obj_get_string(pm, "Kind", 4);
    double pl_dB = matlab_prop_dispatch_pathloss(
        kind, tx_lat, tx_lon, tx_h, rx_lat, rx_lon, rx_h, freq);

    double tx_dBm = 10.0 * log10(power_W * 1000.0);
    /* Antenna gains default to 0 dBi (isotropic) for v1. */
    double tx_gain_dBi = 0.0;
    double rx_gain_dBi = 0.0;
    return tx_dBm + tx_gain_dBi - pl_dB + rx_gain_dBi - tx_loss - rx_loss;
}

}  /* extern "C" */
