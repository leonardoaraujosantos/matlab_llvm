/* Direct unit tests for the FFT family in runtime/runtime_complex.cpp:
 * matlab_fft_c, matlab_ifft_c, matlab_fft2_c, matlab_ifft2_c.
 *
 * Phase-1 catch-up after the Phase-4 RAII migration of fft_bluestein
 * and fft_columns_inplace. Tests cover both the radix-2 fast path
 * (power-of-two N) and the Bluestein general path (non-power-of-two)
 * so the two algorithm branches both get exercised. */

#include "runtime_test.h"

matlab_mat_c *matlab_fft_c   (void *A);
matlab_mat_c *matlab_ifft_c  (void *A);
matlab_mat_c *matlab_fft2_c  (void *A);
matlab_mat_c *matlab_ifft2_c (void *A);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* --- 1-D FFT, power-of-two N (radix-2 path) ------------------------ */
static void test_fft_constant_signal(void) {
    /* fft of [c c c c] is [4c 0 0 0] — DC bin holds the sum. */
    double a[] = {3, 3, 3, 3};
    matlab_mat *A = mk(a, 1, 4);
    matlab_mat_c *F = matlab_fft_c((void *)A);
    RT_NEAR(rt_c_re(F, 0, 0), 12.0, 1e-10, "DC bin = sum");
    RT_NEAR(rt_c_im(F, 0, 0),  0.0, 1e-10, "DC imag = 0");
    for (int k = 1; k < 4; ++k) {
        RT_NEAR(rt_c_re(F, 0, k), 0.0, 1e-10, "non-DC zero");
        RT_NEAR(rt_c_im(F, 0, k), 0.0, 1e-10, "non-DC zero im");
    }
    rt_free(A); rt_c_free(F);
}

static void test_fft_impulse_is_constant(void) {
    /* fft([1 0 0 0]) = [1 1 1 1]. */
    double a[] = {1, 0, 0, 0};
    matlab_mat *A = mk(a, 1, 4);
    matlab_mat_c *F = matlab_fft_c((void *)A);
    for (int k = 0; k < 4; ++k) {
        RT_NEAR(rt_c_re(F, 0, k), 1.0, 1e-10, "impulse FT all 1");
        RT_NEAR(rt_c_im(F, 0, k), 0.0, 1e-10, "imag zero");
    }
    rt_free(A); rt_c_free(F);
}

/* --- 1-D FFT, Bluestein (non-power-of-two N) ----------------------- */
static void test_fft_bluestein_constant(void) {
    /* N = 5 forces Bluestein. fft([c c c c c]) = [5c 0 0 0 0]. */
    double a[] = {2, 2, 2, 2, 2};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat_c *F = matlab_fft_c((void *)A);
    RT_NEAR(rt_c_re(F, 0, 0), 10.0, 1e-9, "DC bin");
    for (int k = 1; k < 5; ++k) {
        RT_NEAR(rt_c_re(F, 0, k), 0.0, 1e-9, "non-DC zero");
        RT_NEAR(rt_c_im(F, 0, k), 0.0, 1e-9, "non-DC im zero");
    }
    rt_free(A); rt_c_free(F);
}

static void test_fft_bluestein_impulse(void) {
    /* fft([1 0 0 0 0 0 0]) = [1 1 1 1 1 1 1]. N=7, Bluestein path. */
    double a[] = {1, 0, 0, 0, 0, 0, 0};
    matlab_mat *A = mk(a, 1, 7);
    matlab_mat_c *F = matlab_fft_c((void *)A);
    for (int k = 0; k < 7; ++k) {
        RT_NEAR(rt_c_re(F, 0, k), 1.0, 1e-9, "Bluestein impulse all 1");
        RT_NEAR(rt_c_im(F, 0, k), 0.0, 1e-9, "im zero");
    }
    rt_free(A); rt_c_free(F);
}

/* --- 1-D ifft ------------------------------------------------------- */
static void test_ifft_round_trip(void) {
    /* ifft(fft(x)) ≈ x. */
    double a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    matlab_mat *A = mk(a, 1, 8);
    matlab_mat_c *F = matlab_fft_c((void *)A);
    matlab_mat_c *X = matlab_ifft_c((void *)F);
    for (int k = 0; k < 8; ++k) {
        RT_NEAR(rt_c_re(X, 0, k), a[k], 1e-10, "round-trip re");
        RT_NEAR(rt_c_im(X, 0, k), 0.0,  1e-10, "round-trip im");
    }
    rt_free(A); rt_c_free(F); rt_c_free(X);
}

static void test_ifft_bluestein_round_trip(void) {
    /* Same round-trip but with N=6 (Bluestein). */
    double a[] = {1, 2, 3, 4, 5, 6};
    matlab_mat *A = mk(a, 1, 6);
    matlab_mat_c *F = matlab_fft_c((void *)A);
    matlab_mat_c *X = matlab_ifft_c((void *)F);
    for (int k = 0; k < 6; ++k) {
        RT_NEAR(rt_c_re(X, 0, k), a[k], 1e-9, "round-trip re N=6");
        RT_NEAR(rt_c_im(X, 0, k), 0.0,  1e-9, "round-trip im N=6");
    }
    rt_free(A); rt_c_free(F); rt_c_free(X);
}

/* --- 2-D FFT -------------------------------------------------------- */
static void test_fft2_constant(void) {
    /* fft2 of a constant 2x2 matrix: only the (0,0) bin is non-zero,
     * holding the sum. fft2([[1 1] [1 1]]) = [[4 0] [0 0]]. */
    double a[] = {1, 1, 1, 1};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat_c *F = matlab_fft2_c((void *)A);
    RT_NEAR(rt_c_re(F, 0, 0), 4.0, 1e-10, "DC bin");
    RT_NEAR(rt_c_re(F, 0, 1), 0.0, 1e-10, "F[0,1]");
    RT_NEAR(rt_c_re(F, 1, 0), 0.0, 1e-10, "F[1,0]");
    RT_NEAR(rt_c_re(F, 1, 1), 0.0, 1e-10, "F[1,1]");
    rt_free(A); rt_c_free(F);
}

static void test_ifft2_round_trip(void) {
    /* ifft2(fft2(A)) ≈ A. */
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat_c *F = matlab_fft2_c((void *)A);
    matlab_mat_c *X = matlab_ifft2_c((void *)F);
    RT_NEAR(rt_c_re(X, 0, 0), 1.0, 1e-10, "X[0,0]");
    RT_NEAR(rt_c_re(X, 0, 1), 2.0, 1e-10, "X[0,1]");
    RT_NEAR(rt_c_re(X, 1, 0), 3.0, 1e-10, "X[1,0]");
    RT_NEAR(rt_c_re(X, 1, 1), 4.0, 1e-10, "X[1,1]");
    rt_free(A); rt_c_free(F); rt_c_free(X);
}

/* --- column FFT (matrix input) -------------------------------------- */
static void test_fft_columnwise(void) {
    /* For a tall matrix, fft is applied column-wise. Each column of
     * [[c c] [c c]] FFTs to [2c 0]^T. */
    double a[] = {3, 5, 3, 5};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat_c *F = matlab_fft_c((void *)A);
    /* Column 0: input [3, 3] → [6, 0]. */
    RT_NEAR(rt_c_re(F, 0, 0), 6.0, 1e-10, "col0 DC");
    RT_NEAR(rt_c_re(F, 1, 0), 0.0, 1e-10, "col0 high");
    /* Column 1: input [5, 5] → [10, 0]. */
    RT_NEAR(rt_c_re(F, 0, 1), 10.0, 1e-10, "col1 DC");
    RT_NEAR(rt_c_re(F, 1, 1),  0.0, 1e-10, "col1 high");
    rt_free(A); rt_c_free(F);
}

int main(void) {
    fprintf(stderr, "test_fft:\n");
    RT_RUN(test_fft_constant_signal);
    RT_RUN(test_fft_impulse_is_constant);
    RT_RUN(test_fft_bluestein_constant);
    RT_RUN(test_fft_bluestein_impulse);
    RT_RUN(test_ifft_round_trip);
    RT_RUN(test_ifft_bluestein_round_trip);
    RT_RUN(test_fft2_constant);
    RT_RUN(test_ifft2_round_trip);
    RT_RUN(test_fft_columnwise);
    RT_DONE();
}
