/* Direct tests for the rng-backed constructors (matlab_rand,
 * matlab_randn). The runtime's rng_uniform / rng_normal are static
 * helpers; we exercise them indirectly through matlab_rand / matlab_randn. */

#include "runtime_test.h"

static void test_rand_shape_and_range(void) {
    matlab_mat *R = matlab_rand(4, 5);
    RT_CHECK(rt_rows(R) == 4 && rt_cols(R) == 5, "rand shape");
    double mn = 1.0, mx = 0.0;
    for (int k = 0; k < 20; ++k) {
        double v = rt_data(R)[k];
        RT_CHECK(v >= 0.0 && v <= 1.0, "rand in [0,1]");
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    /* Sanity: with 20 draws the spread must be non-trivial. */
    RT_CHECK(mx - mn > 0.05, "rand spread non-degenerate");
    rt_free(R);
}

static void test_rand_changes_between_calls(void) {
    /* Two consecutive 1x8 draws must differ in at least one element —
     * if every element matched, the rng would be stuck. */
    matlab_mat *A = matlab_rand(1, 8);
    matlab_mat *B = matlab_rand(1, 8);
    int diffs = 0;
    for (int k = 0; k < 8; ++k)
        if (rt_data(A)[k] != rt_data(B)[k]) ++diffs;
    RT_CHECK(diffs > 0, "rand advances state");
    rt_free(A); rt_free(B);
}

static void test_randn_shape_and_finite(void) {
    matlab_mat *R = matlab_randn(3, 4);
    RT_CHECK(rt_rows(R) == 3 && rt_cols(R) == 4, "randn shape");
    /* All draws must be finite. */
    for (int k = 0; k < 12; ++k) {
        double v = rt_data(R)[k];
        RT_CHECK(!isnan(v) && !isinf(v), "randn finite");
    }
    rt_free(R);
}

static void test_randn_mean_near_zero(void) {
    /* Loose statistical check: 1024 N(0,1) draws should average within
     * a few standard errors of 0. Standard error of the mean is
     * 1/sqrt(N) ≈ 0.031 for N=1024; pick 0.25 to keep this test
     * reliable across rng implementations. */
    matlab_mat *R = matlab_randn(32, 32);
    double sum = 0.0;
    for (int k = 0; k < 1024; ++k) sum += rt_data(R)[k];
    double mean = sum / 1024.0;
    RT_CHECK(fabs(mean) < 0.25, "randn mean roughly zero");
    rt_free(R);
}

int main(void) {
    fprintf(stderr, "test_rng:\n");
    RT_RUN(test_rand_shape_and_range);
    RT_RUN(test_rand_changes_between_calls);
    RT_RUN(test_randn_shape_and_finite);
    RT_RUN(test_randn_mean_near_zero);
    RT_DONE();
}
