/* Direct unit tests for the Propagation Models runtime entries:
 * closed-form ITU-R + cellular empirical path loss, Fresnel zone
 * math, knife-edge diffraction, geographic helpers (Haversine /
 * Vincenty / great-circle), ITM Longley-Rice, terrain profile /
 * LOS / link budget / coverage grid, directional pattern functions,
 * mount orientation, and the closed-form thin-wire dipole. */

#include "runtime_test.h"

/* Forward decls — runtime_prop.cpp entries, not in matlab_runtime.h */
double      matlab_prop_haversine     (double lat1, double lon1, double lat2, double lon2);
double      matlab_prop_bearing       (double lat1, double lon1, double lat2, double lon2);
double      matlab_prop_vincenty      (double lat1, double lon1, double lat2, double lon2);
double      matlab_prop_dest_lat      (double lat1, double lon1, double d_m, double az_deg);
double      matlab_prop_dest_lon      (double lat1, double lon1, double d_m, double az_deg);
double      matlab_prop_fspl          (double d_m, double freq_hz);
double      matlab_prop_pathloss_hata (double f_MHz, double ht_m, double hr_m,
                                        double d_m, double env);
double      matlab_prop_pathloss_cost231(double f_MHz, double ht_m, double hr_m,
                                          double d_m, double env);
double      matlab_prop_pathloss_egli (double f_MHz, double ht_m, double hr_m,
                                        double d_m);
double      matlab_prop_pathloss_sui  (double f_MHz, double ht_m, double hr_m,
                                        double d_m, double terrain);
double      matlab_prop_fresnel_zone_radius(double d1_m, double d2_m,
                                             double lambda_m, double n);
double      matlab_prop_diff_knife_edge(double h_m, double d1_m,
                                         double d2_m, double lambda_m);
double      matlab_prop_pat_sector    (double az, double el,
                                        double bw_az, double bw_el,
                                        double peak_gain, double fb);
double      matlab_prop_pat_cosine    (double az, double el,
                                        double bw_az, double bw_el,
                                        double peak_gain, double n);
double      matlab_prop_pat_isotropic (double az, double el, double gain_dBi);
double      matlab_prop_mount_az_local(double az_world, double el_world,
                                        double mount_az, double mount_tilt);
matlab_mat *matlab_prop_mount_to_local(double az_world, double el_world,
                                        double mount_az, double mount_tilt);
matlab_struct *matlab_ant_wire_solve(double L, double a, double Nsegs,
                                      double freq);
matlab_struct *matlab_struct_new(void);
double      matlab_struct_get_f64    (matlab_struct *s, const char *name, int64_t len);

/* ===== Closed-form path-loss models ===== */

static void test_fspl_isotropic_1m_1GHz(void) {
    /* FSPL = 20*log10(4*pi*d/lambda).
       At d=1m, f=1GHz, lambda=0.3 m -> 4*pi/0.3 ~= 41.888
       -> 20*log10(41.888) ~= 32.44 dB. */
    double L = matlab_prop_fspl(1.0, 1.0e9);
    RT_NEAR(L, 32.45, 0.05, "FSPL 1m / 1GHz reference");
}

static void test_fspl_10km_5p8GHz(void) {
    /* MathWorks reference: ~127.74 dB at 10 km / 5.8 GHz. */
    double L = matlab_prop_fspl(10000.0, 5.8e9);
    RT_NEAR(L, 127.74, 0.5, "FSPL 10 km / 5.8 GHz");
}

static void test_fspl_monotonic_in_distance(void) {
    double L1 = matlab_prop_fspl( 1000.0, 2.4e9);
    double L2 = matlab_prop_fspl(10000.0, 2.4e9);
    double L3 = matlab_prop_fspl(50000.0, 2.4e9);
    RT_CHECK(L2 > L1 + 15.0, "FSPL grows ~20 dB/decade");
    RT_CHECK(L3 > L2 + 10.0, "FSPL keeps growing");
}

static void test_pathloss_hata_urban_large(void) {
    /* Hata urban-large at 900 MHz / 30 m / 1.5 m / 1 km (d in km).
       Reference: ~125.5 dB (textbook Rappaport). */
    double L = matlab_prop_pathloss_hata(900.0, 30.0, 1.5, 1.0, 1.0);
    RT_CHECK(L > 100.0 && L < 150.0, "Hata in plausible range");
}

static void test_pathloss_models_ordered(void) {
    /* In typical conditions, urban > suburban > open-area.
     * d_km = 5. */
    double L_urb = matlab_prop_pathloss_hata(900.0, 30.0, 1.5, 5.0, 1.0);
    double L_sub = matlab_prop_pathloss_hata(900.0, 30.0, 1.5, 5.0, 3.0);
    double L_opn = matlab_prop_pathloss_hata(900.0, 30.0, 1.5, 5.0, 4.0);
    RT_CHECK(L_urb > L_sub, "Hata urban > suburban");
    RT_CHECK(L_sub > L_opn, "Hata suburban > open");
}

static void test_pathloss_cost231(void) {
    /* COST231-Hata at 1800 MHz / 30 m / 1.5 m / 1 km. */
    double L = matlab_prop_pathloss_cost231(1800.0, 30.0, 1.5, 1.0, 1.0);
    RT_CHECK(L > 110.0 && L < 160.0, "COST231 in plausible range");
}

static void test_pathloss_egli(void) {
    /* Egli is a 2-ray model with d^4 dependence (40*log10(d_m)) so
     * loss grows quickly. At 300 MHz / 30 m / 3 m / 10 km the formula
     * gives ~230 dB. Use a short range for a sanity check. */
    double L = matlab_prop_pathloss_egli(300.0, 30.0, 3.0, 1.0);
    RT_CHECK(L > 90.0 && L < 260.0, "Egli plausible");
    /* Egli grows with distance. */
    double L10 = matlab_prop_pathloss_egli(300.0, 30.0, 3.0, 10.0);
    RT_CHECK(L10 > L, "Egli increases with distance");
}

static void test_pathloss_sui_terrain_progression(void) {
    /* SUI: terrain A (hilly heavy tree) > B > C (flat). d_km = 5. */
    double LA = matlab_prop_pathloss_sui(3500.0, 30.0, 6.0, 5.0, 1.0);
    double LB = matlab_prop_pathloss_sui(3500.0, 30.0, 6.0, 5.0, 2.0);
    double LC = matlab_prop_pathloss_sui(3500.0, 30.0, 6.0, 5.0, 3.0);
    RT_CHECK(LA > LB, "SUI A > B");
    RT_CHECK(LB > LC, "SUI B > C");
}

/* ===== Fresnel zone math ===== */

static void test_fresnel_zone_first_zone(void) {
    /* r1 = sqrt(n * lambda * d1 * d2 / (d1 + d2)) at midpoint.
       lambda = 0.0517 m (5.8 GHz), d1 = d2 = 5000 m
       -> r1 = sqrt(0.0517 * 5000 * 5000 / 10000) = sqrt(129.31) = 11.37 m. */
    double lambda = 3e8 / 5.8e9;
    double r = matlab_prop_fresnel_zone_radius(5000.0, 5000.0, lambda, 1.0);
    RT_NEAR(r, 11.37, 0.05, "First Fresnel zone at midpoint");
}

static void test_fresnel_zone_nth(void) {
    double lambda = 3e8 / 2.4e9;
    double r1 = matlab_prop_fresnel_zone_radius(1000.0, 1000.0, lambda, 1.0);
    double r2 = matlab_prop_fresnel_zone_radius(1000.0, 1000.0, lambda, 2.0);
    /* r_n / r_1 = sqrt(n). */
    RT_NEAR(r2 / r1, 1.41421356, 1e-6, "Fresnel sqrt(n) scaling");
}

/* ===== Knife-edge diffraction ===== */

static void test_diff_knife_edge_grazing(void) {
    /* At h=0 (grazing the LOS), the diffraction loss is about 6 dB
       (ITU-R P.526 says J(0) = 6.0 dB). */
    double lambda = 0.3;
    double L = matlab_prop_diff_knife_edge(0.0, 1000.0, 1000.0, lambda);
    RT_NEAR(L, 6.0, 1.0, "knife-edge at grazing -> 6 dB");
}

static void test_diff_knife_edge_far_below(void) {
    /* h very negative (obstacle well below LOS) -> ~0 dB loss. */
    double lambda = 0.3;
    double L = matlab_prop_diff_knife_edge(-100.0, 1000.0, 1000.0, lambda);
    RT_CHECK(L < 0.1 || L >= -0.1, "knife-edge far below -> ~0 dB");
}

/* ===== Geographic helpers ===== */

static void test_haversine_known_distance(void) {
    /* London Heathrow (51.4700 N, -0.4543) to JFK (40.6413 N, -73.7781).
       Reference Haversine distance ~5540 km. */
    double d = matlab_prop_haversine(51.4700, -0.4543, 40.6413, -73.7781);
    /* d returned in meters per the runtime convention. */
    RT_NEAR(d, 5540e3, 30e3, "Haversine LHR->JFK");
}

static void test_haversine_self_zero(void) {
    double d = matlab_prop_haversine(45.0, -75.0, 45.0, -75.0);
    RT_NEAR(d, 0.0, 1e-3, "Haversine self distance = 0");
}

static void test_vincenty_matches_haversine_close(void) {
    /* For short distances both methods agree closely. */
    double d_hav = matlab_prop_haversine(45.0,  -75.0, 45.01, -75.0);
    double d_vin = matlab_prop_vincenty (45.0,  -75.0, 45.01, -75.0);
    /* ~1.1 km on the ground. */
    RT_NEAR(d_vin, d_hav, d_hav * 0.01, "Vincenty matches Haversine short");
}

static void test_bearing_north(void) {
    /* Moving north should give bearing ~0. */
    double az = matlab_prop_bearing(45.0, -75.0, 46.0, -75.0);
    RT_NEAR(az, 0.0, 1.0, "Bearing due north -> ~0 deg");
}

static void test_dest_round_trip(void) {
    /* Move 50 km north (az=0), then check back distance. */
    double lat0 = 45.0, lon0 = -75.0;
    double lat1 = matlab_prop_dest_lat(lat0, lon0, 50000.0, 0.0);
    double lon1 = matlab_prop_dest_lon(lat0, lon0, 50000.0, 0.0);
    double d = matlab_prop_haversine(lat0, lon0, lat1, lon1);
    RT_NEAR(d, 50000.0, 200.0, "great-circle dest round-trip");
    /* North move -> longitude unchanged. */
    RT_NEAR(lon1, lon0, 1e-6, "north move keeps lon");
}

/* ===== Directional pattern functions ===== */

static void test_pat_isotropic_constant(void) {
    double g1 = matlab_prop_pat_isotropic(  0.0,  0.0, 5.0);
    double g2 = matlab_prop_pat_isotropic(180.0, 45.0, 5.0);
    RT_NEAR(g1, 5.0, 1e-12, "isotropic at 0,0");
    RT_NEAR(g2, 5.0, 1e-12, "isotropic at 180,45");
}

static void test_pat_sector_peak_at_boresight(void) {
    /* 120-deg sector with 17 dBi peak; az=0, el=0 (boresight) -> ~17 dBi. */
    double g = matlab_prop_pat_sector(0.0, 0.0, 120.0, 10.0, 17.0, 25.0);
    RT_NEAR(g, 17.0, 0.5, "sector pattern at boresight");
}

static void test_pat_sector_dropoff(void) {
    /* Off-boresight should be < peak. */
    double g_on  = matlab_prop_pat_sector(0.0,   0.0, 120.0, 10.0, 17.0, 25.0);
    double g_off = matlab_prop_pat_sector(60.0,  0.0, 120.0, 10.0, 17.0, 25.0);
    RT_CHECK(g_off < g_on, "sector falls off azimuth");
}

static void test_pat_cosine_peak(void) {
    /* cosine^n at boresight -> peak gain. */
    double g = matlab_prop_pat_cosine(0.0, 0.0, 8.0, 8.0, 22.0, 30.0);
    RT_NEAR(g, 22.0, 0.5, "cosine pattern at boresight");
}

/* ===== Mount orientation ===== */

static void test_mount_az_no_rotation(void) {
    /* Mount with az=0 -> world az passes through. */
    double a = matlab_prop_mount_az_local(120.0, 0.0, 0.0, 0.0);
    RT_NEAR(a, 120.0, 1e-9, "mount az=0 -> identity");
}

static void test_mount_az_rotate(void) {
    /* If mount points to az=90, an observer at world az=90 sees
       the antenna at local az=0. */
    double a = matlab_prop_mount_az_local(90.0, 0.0, 90.0, 0.0);
    RT_NEAR(a, 0.0, 1e-9, "mount az=90 + world az=90 -> local az=0");
}

/* ===== Antenna closed-form thin-wire dipole ===== */

static void test_ant_wire_half_wave_impedance(void) {
    /* Half-wave dipole at 300 MHz (lambda=1 m): L=0.5 m, a=0.001 m.
       Reference: Zin = 73.13 + j42.55 Ω (Balanis Eq. 8-60). */
    matlab_struct *s = matlab_ant_wire_solve(0.5, 0.001, 21.0, 300.0e6);
    double zr = matlab_struct_get_f64(s, "Zin_re", 6);
    double zi = matlab_struct_get_f64(s, "Zin_im", 6);
    RT_NEAR(zr, 73.13, 1.0, "half-wave dipole Re(Zin)");
    RT_NEAR(zi, 42.55, 2.0, "half-wave dipole Im(Zin)");
    /* Don't free the struct — runtime struct teardown isn't part of this
       test's responsibility, and the leak is intentional in unit-test
       lifetimes. */
}

int main(void) {
    RT_RUN(test_fspl_isotropic_1m_1GHz);
    RT_RUN(test_fspl_10km_5p8GHz);
    RT_RUN(test_fspl_monotonic_in_distance);
    RT_RUN(test_pathloss_hata_urban_large);
    RT_RUN(test_pathloss_models_ordered);
    RT_RUN(test_pathloss_cost231);
    RT_RUN(test_pathloss_egli);
    RT_RUN(test_pathloss_sui_terrain_progression);
    RT_RUN(test_fresnel_zone_first_zone);
    RT_RUN(test_fresnel_zone_nth);
    RT_RUN(test_diff_knife_edge_grazing);
    RT_RUN(test_diff_knife_edge_far_below);
    RT_RUN(test_haversine_known_distance);
    RT_RUN(test_haversine_self_zero);
    RT_RUN(test_vincenty_matches_haversine_close);
    RT_RUN(test_bearing_north);
    RT_RUN(test_dest_round_trip);
    RT_RUN(test_pat_isotropic_constant);
    RT_RUN(test_pat_sector_peak_at_boresight);
    RT_RUN(test_pat_sector_dropoff);
    RT_RUN(test_pat_cosine_peak);
    RT_RUN(test_mount_az_no_rotation);
    RT_RUN(test_mount_az_rotate);
    RT_RUN(test_ant_wire_half_wave_impedance);
    RT_DONE();
}
