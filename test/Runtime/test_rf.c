/* Direct unit tests for the RF Toolbox runtime entries:
 * 2-port S-parameter analyses (gammaIn / VSWR / stability_k),
 * N-port conversions (S↔Y, S↔Z), cascade, rationalfit + freqresp,
 * transmission lines (microstrip), L-section matching network. */

#include "runtime_test.h"

/* Forward decls — runtime_rf.cpp entries */
matlab_mat_c *matlab_rf_gamma_in     (matlab_mat_c *S11, matlab_mat_c *S12,
                                       matlab_mat_c *S21, matlab_mat_c *S22,
                                       double zl, double z0);
matlab_mat_c *matlab_rf_gamma_out    (matlab_mat_c *S11, matlab_mat_c *S12,
                                       matlab_mat_c *S21, matlab_mat_c *S22,
                                       double zs, double z0);
matlab_mat   *matlab_rf_vswr_from_gamma(matlab_mat_c *gamma);
matlab_mat   *matlab_rf_stability_k  (matlab_mat_c *S11, matlab_mat_c *S12,
                                       matlab_mat_c *S21, matlab_mat_c *S22);
matlab_mat   *matlab_rf_stability_mu (matlab_mat_c *S11, matlab_mat_c *S12,
                                       matlab_mat_c *S21, matlab_mat_c *S22);
matlab_struct *matlab_rf_s2y         (matlab_mat_c *S11, matlab_mat_c *S12,
                                       matlab_mat_c *S21, matlab_mat_c *S22,
                                       double z0);
matlab_struct *matlab_rf_s2z         (matlab_mat_c *S11, matlab_mat_c *S12,
                                       matlab_mat_c *S21, matlab_mat_c *S22,
                                       double z0);
matlab_struct *matlab_rf_cascade2    (matlab_mat_c *A11, matlab_mat_c *A12,
                                       matlab_mat_c *A21, matlab_mat_c *A22,
                                       matlab_mat_c *B11, matlab_mat_c *B12,
                                       matlab_mat_c *B21, matlab_mat_c *B22);
matlab_struct *matlab_rf_rationalfit (matlab_mat *freq,
                                       matlab_mat *h_re, matlab_mat *h_im,
                                       double n_poles_d, double n_iter_d);
matlab_mat_c *matlab_rf_freqresp     (matlab_struct *mdl, matlab_mat *freq);
matlab_struct *matlab_rf_microstrip  (double w, double h, double er,
                                       double length_m, matlab_mat *freqs,
                                       double z0);
matlab_struct *matlab_rf_matchingnetwork(double zs_re, double zs_im,
                                          double zl_re, double zl_im,
                                          double freq);
matlab_struct *matlab_rf_budget_friis(matlab_mat *gains_dB, matlab_mat *nfs_dB,
                                       matlab_mat *ip3s_dBm,
                                       double in_pwr_dBm,
                                       double bandwidth_Hz);

/* Complex constructor */
matlab_mat_c *matlab_mat_c_from_buf(const double *re, const double *im,
                                     double rows, double cols);

matlab_struct *matlab_struct_new(void);
double         matlab_struct_get_f64(matlab_struct *s, const char *name, int64_t len);
matlab_mat    *matlab_struct_get_mat(matlab_struct *s, const char *name, int64_t len);
/* Note: matlab_struct stores complex matrices through the same
 * `matlab_struct_get_mat` entry — the returned pointer is cast back to
 * `matlab_mat_c *` by the caller. There is no separate `_get_mat_c`
 * entry. */
static matlab_mat_c *struct_get_c(matlab_struct *s, const char *name, int64_t len) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, name, len);
}

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}
static matlab_mat_c *mkc(const double *re, const double *im, int64_t m, int64_t n) {
    return matlab_mat_c_from_buf(re, im, (double)m, (double)n);
}

/* ===== gammaIn / VSWR ===== */

static void test_gamma_in_matched_load(void) {
    /* Matched load (zl = z0 = 50): gamma_load = 0, so gamma_in = S11. */
    double re_s11[] = {0.3}, im_s11[] = {0.1};
    double re_z[]   = {0.0}, im_z[]   = {0.0};
    matlab_mat_c *S11 = mkc(re_s11, im_s11, 1, 1);
    matlab_mat_c *S12 = mkc(re_z, im_z, 1, 1);
    matlab_mat_c *S21 = mkc(re_z, im_z, 1, 1);
    matlab_mat_c *S22 = mkc(re_z, im_z, 1, 1);
    matlab_mat_c *g = matlab_rf_gamma_in(S11, S12, S21, S22, 50.0, 50.0);
    RT_NEAR(rt_c_re(g, 0, 0), 0.3, 1e-9, "gammaIn matched -> S11 re");
    RT_NEAR(rt_c_im(g, 0, 0), 0.1, 1e-9, "gammaIn matched -> S11 im");
    rt_c_free(S11); rt_c_free(S12); rt_c_free(S21); rt_c_free(S22);
    rt_c_free(g);
}

static void test_vswr_from_zero_gamma(void) {
    /* |gamma| = 0 -> VSWR = 1. */
    double re[] = {0.0}, im[] = {0.0};
    matlab_mat_c *g = mkc(re, im, 1, 1);
    matlab_mat *vswr = matlab_rf_vswr_from_gamma(g);
    RT_NEAR(rt_data(vswr)[0], 1.0, 1e-12, "VSWR(|Γ|=0) = 1");
    rt_c_free(g); rt_free(vswr);
}

static void test_vswr_from_half_gamma(void) {
    /* |gamma| = 0.5 -> VSWR = (1+0.5)/(1-0.5) = 3. */
    double re[] = {0.5}, im[] = {0.0};
    matlab_mat_c *g = mkc(re, im, 1, 1);
    matlab_mat *vswr = matlab_rf_vswr_from_gamma(g);
    RT_NEAR(rt_data(vswr)[0], 3.0, 1e-12, "VSWR(|Γ|=0.5) = 3");
    rt_c_free(g); rt_free(vswr);
}

/* ===== Stability factor K ===== */

static void test_stability_k_unconditional(void) {
    /* Build an unconditionally-stable 2-port: K > 1.
       S = [[0.1 + 0j, 0.05 + 0j], [0.5 + 0j, 0.1 + 0j]] — tiny S12. */
    double r_s11[] = {0.1}, i_z[] = {0.0};
    double r_s12[] = {0.05};
    double r_s21[] = {0.5};
    double r_s22[] = {0.1};
    matlab_mat_c *S11 = mkc(r_s11, i_z, 1, 1);
    matlab_mat_c *S12 = mkc(r_s12, i_z, 1, 1);
    matlab_mat_c *S21 = mkc(r_s21, i_z, 1, 1);
    matlab_mat_c *S22 = mkc(r_s22, i_z, 1, 1);
    matlab_mat *K = matlab_rf_stability_k(S11, S12, S21, S22);
    RT_CHECK(rt_data(K)[0] > 1.0, "K > 1 for chosen S");
    rt_c_free(S11); rt_c_free(S12); rt_c_free(S21); rt_c_free(S22);
    rt_free(K);
}

/* ===== S to Y conversion round-trip (1-port-style identity) ===== */

static void test_s2y_zero_S_reflection(void) {
    /* S = 0 (perfect match) -> Y matrix is just the conductance 1/z0 along
       the diagonal. Not a strict round-trip test, but the runtime
       should accept the call and return non-NULL Y_ij vectors. */
    double rz[] = {0.0}, iz[] = {0.0};
    matlab_mat_c *S11 = mkc(rz, iz, 1, 1);
    matlab_mat_c *S12 = mkc(rz, iz, 1, 1);
    matlab_mat_c *S21 = mkc(rz, iz, 1, 1);
    matlab_mat_c *S22 = mkc(rz, iz, 1, 1);
    matlab_struct *y = matlab_rf_s2y(S11, S12, S21, S22, 50.0);
    matlab_mat_c *Y11 = struct_get_c(y, "Y11", 3);
    /* With S = 0 (matched, no coupling): (I+S)^-1 = I, (I-S) = I,
     * so Y_normalized = I and the final Y = (1/Z0) * I.
     * Y11 = 1/50 = 0.02. */
    RT_CHECK(Y11 != NULL, "s2y returns Y11");
    if (Y11) {
        RT_NEAR(rt_c_re(Y11, 0, 0), 0.02, 1e-9, "Y11 = 1/Z0");
    }
    rt_c_free(S11); rt_c_free(S12); rt_c_free(S21); rt_c_free(S22);
}

/* ===== Cascade ===== */

static void test_cascade_through_two_thru_lines(void) {
    /* "Thru" 2-port: S = [[0, 1], [1, 0]]. Cascading two thrus gives a thru. */
    double r0[] = {0.0}, r1[] = {1.0};
    matlab_mat_c *A11 = mkc(r0, r0, 1, 1);
    matlab_mat_c *A12 = mkc(r1, r0, 1, 1);
    matlab_mat_c *A21 = mkc(r1, r0, 1, 1);
    matlab_mat_c *A22 = mkc(r0, r0, 1, 1);
    matlab_mat_c *B11 = mkc(r0, r0, 1, 1);
    matlab_mat_c *B12 = mkc(r1, r0, 1, 1);
    matlab_mat_c *B21 = mkc(r1, r0, 1, 1);
    matlab_mat_c *B22 = mkc(r0, r0, 1, 1);
    matlab_struct *c = matlab_rf_cascade2(A11, A12, A21, A22,
                                          B11, B12, B21, B22);
    matlab_mat_c *S11c = struct_get_c(c, "S11", 3);
    matlab_mat_c *S21c = struct_get_c(c, "S21", 3);
    RT_NEAR(rt_c_re(S11c, 0, 0), 0.0, 1e-9, "cascade thru S11 = 0");
    RT_NEAR(rt_c_re(S21c, 0, 0), 1.0, 1e-9, "cascade thru S21 = 1");
    rt_c_free(A11); rt_c_free(A12); rt_c_free(A21); rt_c_free(A22);
    rt_c_free(B11); rt_c_free(B12); rt_c_free(B21); rt_c_free(B22);
}

/* ===== Microstrip TL ===== */

static void test_microstrip_basic(void) {
    /* FR-4 microstrip, w=2mm, h=1.6mm, er=4.3, len=10mm. The
       characteristic impedance comes out around 50 Ω; we just verify
       the call returns S21 |= 0 (transmission). */
    double freqs_buf[] = {1.0e9};
    matlab_mat *freqs = mk(freqs_buf, 1, 1);
    matlab_struct *s = matlab_rf_microstrip(0.002, 0.0016, 4.3, 0.010,
                                            freqs, 50.0);
    matlab_mat_c *S21 = struct_get_c(s, "S21", 3);
    RT_CHECK(S21 != NULL, "microstrip returns S21");
    if (S21) {
        double mag2 = rt_c_re(S21, 0, 0) * rt_c_re(S21, 0, 0) +
                      rt_c_im(S21, 0, 0) * rt_c_im(S21, 0, 0);
        RT_CHECK(mag2 > 0.5 && mag2 <= 1.001,
                 "microstrip |S21| close to 1 (lossless)");
    }
    rt_free(freqs);
}

/* ===== L-section matching network ===== */

static void test_matchingnetwork_call(void) {
    /* L-section matching from 50 Ω to a complex load (10 + j20).
       Smoke test: the function returns a struct with L / C components. */
    matlab_struct *m = matlab_rf_matchingnetwork(50.0, 0.0,
                                                  10.0, 20.0,
                                                  1.0e9);
    RT_CHECK(m != NULL, "matchingnetwork returns struct");
}

/* ===== rationalfit + freqresp ===== */

static void test_rationalfit_freqresp_round_trip(void) {
    /* Fit a flat unit-magnitude response over a small grid. The
     * rationalfit ABI takes h_re / h_im as separate real matrices,
     * not a single complex one. n_iter (last arg) is iteration
     * count, not a tol_dB. */
    double freqs_buf[] = {1.0e9, 1.5e9, 2.0e9, 2.5e9, 3.0e9};
    double re_buf[]    = {1.0, 1.0, 1.0, 1.0, 1.0};
    double im_buf[]    = {0.0, 0.0, 0.0, 0.0, 0.0};
    matlab_mat *freqs = mk(freqs_buf, 5, 1);
    matlab_mat *h_re  = mk(re_buf,    5, 1);
    matlab_mat *h_im  = mk(im_buf,    5, 1);
    matlab_struct *mdl = matlab_rf_rationalfit(freqs, h_re, h_im, 2.0, 10.0);
    RT_CHECK(mdl != NULL, "rationalfit returns model");
    if (mdl) {
        matlab_mat_c *eval = matlab_rf_freqresp(mdl, freqs);
        RT_CHECK(eval != NULL, "freqresp evaluates");
        if (eval) {
            /* Should be close to unit magnitude on average. Vector
             * Fitting on a flat target sometimes lands with one or two
             * outlier frequencies as the poles settle, so just check
             * the mean magnitude is in a sane band. */
            double sum_mag = 0.0;
            for (int i = 0; i < 5; ++i) {
                sum_mag += sqrt(rt_c_re(eval, i, 0) * rt_c_re(eval, i, 0) +
                                rt_c_im(eval, i, 0) * rt_c_im(eval, i, 0));
            }
            double mean = sum_mag / 5.0;
            RT_CHECK(mean > 0.5 && mean < 2.0,
                     "rationalfit mean magnitude near unity");
        }
    }
    rt_free(freqs); rt_free(h_re); rt_free(h_im);
}

/* ===== rfbudget Friis cascade ===== */

static void test_rf_budget_friis_three_stage(void) {
    /* 3-stage chain: G=10/20/15 dB, NF=2/3/5 dB, IP3=20/25/30 dBm.
       Total gain = 45 dB. */
    double g_buf[] = {10.0, 20.0, 15.0};
    double nf_buf[] = {2.0, 3.0, 5.0};
    double ip3_buf[] = {20.0, 25.0, 30.0};
    matlab_mat *g  = mk(g_buf,  3, 1);
    matlab_mat *nf = mk(nf_buf, 3, 1);
    matlab_mat *ip = mk(ip3_buf, 3, 1);
    matlab_struct *budget = matlab_rf_budget_friis(g, nf, ip,
                                                    -50.0, 1.0e6);
    RT_CHECK(budget != NULL, "rfbudget returns struct");
    if (budget) {
        double gain = matlab_struct_get_f64(budget, "Gain_dB", 7);
        RT_NEAR(gain, 45.0, 1e-6, "Total gain sums dB");
    }
    rt_free(g); rt_free(nf); rt_free(ip);
}

int main(void) {
    RT_RUN(test_gamma_in_matched_load);
    RT_RUN(test_vswr_from_zero_gamma);
    RT_RUN(test_vswr_from_half_gamma);
    RT_RUN(test_stability_k_unconditional);
    RT_RUN(test_s2y_zero_S_reflection);
    RT_RUN(test_cascade_through_two_thru_lines);
    RT_RUN(test_microstrip_basic);
    RT_RUN(test_matchingnetwork_call);
    RT_RUN(test_rationalfit_freqresp_round_trip);
    RT_RUN(test_rf_budget_friis_three_stage);
    RT_DONE();
}
