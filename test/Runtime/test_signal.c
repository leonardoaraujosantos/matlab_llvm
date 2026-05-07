/* Direct unit tests for the Tier-1/2/3 signal-processing builtins:
 * conv, conv2, filter, xcorr, fftshift, ifftshift, hamming, hann,
 * blackman, upsample, downsample, diff. */

#include "runtime_test.h"

/* Forward decls — these aren't in matlab_runtime.h since the tests link
 * against the runtime TU directly. */
matlab_mat   *matlab_conv      (matlab_mat *u, matlab_mat *v);
matlab_mat   *matlab_conv2     (matlab_mat *A, matlab_mat *B);
matlab_mat   *matlab_filter    (matlab_mat *b, matlab_mat *a, matlab_mat *x);
matlab_mat   *matlab_xcorr     (matlab_mat *u, matlab_mat *v);
matlab_mat_c *matlab_fftshift_c(void *A);
matlab_mat_c *matlab_ifftshift_c(void *A);
matlab_mat   *matlab_hamming   (double n);
matlab_mat   *matlab_hann      (double n);
matlab_mat   *matlab_blackman  (double n);
matlab_mat   *matlab_rectwin   (double n);
matlab_mat   *matlab_triang    (double n);
matlab_mat   *matlab_bartlett  (double n);
matlab_mat   *matlab_barthannwin(double n);
matlab_mat   *matlab_bohmanwin (double n);
matlab_mat   *matlab_parzenwin (double n);
matlab_mat   *matlab_nuttallwin(double n);
matlab_mat   *matlab_blackmanharris(double n);
matlab_mat   *matlab_flattopwin(double n);
matlab_mat   *matlab_kaiser    (double n, double beta);
matlab_mat   *matlab_tukeywin  (double n, double r);
matlab_mat   *matlab_gausswin  (double n, double alpha);
matlab_mat   *matlab_chebwin   (double n, double r);
matlab_mat   *matlab_taylorwin (double n, double nbar, double sll);
matlab_mat   *matlab_upsample  (matlab_mat *x, double n);
matlab_mat   *matlab_downsample(matlab_mat *x, double n);
matlab_mat   *matlab_diff      (matlab_mat *A);
matlab_mat   *matlab_poly      (void *r);
matlab_mat   *matlab_polyder   (matlab_mat *p);
matlab_mat   *matlab_polyint   (matlab_mat *p);
matlab_mat   *matlab_polyint_k (matlab_mat *p, double k);
matlab_mat_c *matlab_roots     (matlab_mat *p);
matlab_mat_c *matlab_residue_r (matlab_mat *b, matlab_mat *a);
matlab_mat_c *matlab_residue_p (matlab_mat *b, matlab_mat *a);
matlab_mat   *matlab_residue_k (matlab_mat *b, matlab_mat *a);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

static void test_conv_polynomial_product(void) {
    /* (1 + 2x + 3x^2) * (1 + x) = 1 + 3x + 5x^2 + 3x^3 */
    double a[] = {1, 2, 3};
    double b[] = {1, 1};
    matlab_mat *A = mk(a, 1, 3);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *C = matlab_conv(A, B);
    RT_CHECK(rt_rows(C) == 1, "conv result is row");
    RT_CHECK(rt_cols(C) == 4, "conv length n+m-1");
    double expected[] = {1, 3, 5, 3};
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(C)[i], expected[i], 1e-12, "polynomial product");
    rt_free(A); rt_free(B); rt_free(C);
}

static void test_conv_moving_sum(void) {
    /* conv([1 2 3 4 5], [1 1 1]) = [1 3 6 9 12 9 5] */
    double a[] = {1, 2, 3, 4, 5};
    double b[] = {1, 1, 1};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *B = mk(b, 1, 3);
    matlab_mat *C = matlab_conv(A, B);
    double expected[] = {1, 3, 6, 9, 12, 9, 5};
    RT_CHECK(rt_cols(C) == 7, "len 5+3-1=7");
    for (int i = 0; i < 7; ++i)
        RT_NEAR(rt_data(C)[i], expected[i], 1e-12, "moving sum");
    rt_free(A); rt_free(B); rt_free(C);
}

static void test_conv_column_orientation(void) {
    /* If either input is a column, output is a column. */
    double a[] = {1, 2, 3};
    double b[] = {1, 1};
    matlab_mat *A = mk(a, 3, 1);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *C = matlab_conv(A, B);
    RT_CHECK(rt_cols(C) == 1, "column orientation");
    RT_CHECK(rt_rows(C) == 4, "column length");
    rt_free(A); rt_free(B); rt_free(C);
}

static void test_conv2_box_filter(void) {
    /* conv2([1 2 3; 4 5 6; 7 8 9], ones(2,2)) full-shape 4x4. */
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    double b[] = {1,1, 1,1};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *B = mk(b, 2, 2);
    matlab_mat *C = matlab_conv2(A, B);
    RT_CHECK(rt_rows(C) == 4 && rt_cols(C) == 4, "conv2 full size");
    /* C[1,1] = sum of A's top-left 2x2 = 1+2+4+5 = 12. */
    RT_NEAR(rt_at(C, 1, 1), 12.0, 1e-12, "C[1,1]");
    /* C[3,3] = A[2,2] = 9. */
    RT_NEAR(rt_at(C, 3, 3), 9.0, 1e-12, "C[3,3]");
    rt_free(A); rt_free(B); rt_free(C);
}

static void test_conv2_separable_outer_product(void) {
    /* conv2([1 1], [1; 1]) = ones(2,2). */
    double a[] = {1, 1};
    double b[] = {1, 1};
    matlab_mat *A = mk(a, 1, 2);
    matlab_mat *B = mk(b, 2, 1);
    matlab_mat *C = matlab_conv2(A, B);
    RT_CHECK(rt_rows(C) == 2 && rt_cols(C) == 2, "outer product 2x2");
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(C)[i], 1.0, 1e-12, "ones(2,2)");
    rt_free(A); rt_free(B); rt_free(C);
}

static void test_filter_moving_average(void) {
    /* 4-tap moving average on a step input. */
    double bd[] = {0.25, 0.25, 0.25, 0.25};
    double ad[] = {1, 0};
    double xd[] = {1, 1, 1, 1, 1, 1};
    matlab_mat *b = mk(bd, 1, 4);
    matlab_mat *a = mk(ad, 1, 2);
    matlab_mat *x = mk(xd, 1, 6);
    matlab_mat *y = matlab_filter(b, a, x);
    double expected[] = {0.25, 0.5, 0.75, 1.0, 1.0, 1.0};
    for (int i = 0; i < 6; ++i)
        RT_NEAR(rt_data(y)[i], expected[i], 1e-12, "moving avg ramp");
    rt_free(b); rt_free(a); rt_free(x); rt_free(y);
}

static void test_filter_iir_geometric_decay(void) {
    /* y[n] = 0.5*x[n] + 0.5*y[n-1], unit impulse → 0.5, 0.25, 0.125, ... */
    double bd[] = {0.5, 0};
    double ad[] = {1, -0.5};
    double xd[] = {1, 0, 0, 0, 0};
    matlab_mat *b = mk(bd, 1, 2);
    matlab_mat *a = mk(ad, 1, 2);
    matlab_mat *x = mk(xd, 1, 5);
    matlab_mat *y = matlab_filter(b, a, x);
    double expected[] = {0.5, 0.25, 0.125, 0.0625, 0.03125};
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_data(y)[i], expected[i], 1e-12, "IIR decay");
    rt_free(b); rt_free(a); rt_free(x); rt_free(y);
}

static void test_filter_zero_a0_returns_empty(void) {
    /* a(1) = 0 is invalid; runtime returns 0x0. */
    double bd[] = {1, 0};
    double ad[] = {0, 1};
    double xd[] = {1, 2, 3};
    matlab_mat *b = mk(bd, 1, 2);
    matlab_mat *a = mk(ad, 1, 2);
    matlab_mat *x = mk(xd, 1, 3);
    matlab_mat *y = matlab_filter(b, a, x);
    RT_CHECK(rt_rows(y) == 0 && rt_cols(y) == 0, "0x0 on a(1)=0");
    rt_free(b); rt_free(a); rt_free(x); rt_free(y);
}

static void test_xcorr_triangular_autocorr(void) {
    /* xcorr([1 1 1], [1 1 1]) = [1 2 3 2 1] (triangular). */
    double a[] = {1, 1, 1};
    matlab_mat *A = mk(a, 1, 3);
    matlab_mat *B = mk(a, 1, 3);
    matlab_mat *R = matlab_xcorr(A, B);
    double expected[] = {1, 2, 3, 2, 1};
    RT_CHECK(rt_cols(R) == 5, "len 2L-1");
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_data(R)[i], expected[i], 1e-12, "triangular autocorr");
    rt_free(A); rt_free(B); rt_free(R);
}

static void test_xcorr_unequal_lengths(void) {
    /* xcorr([1 2 3], [1 1]) — see runtime/matlab_runtime.c xcorr docstring. */
    double a[] = {1, 2, 3};
    double b[] = {1, 1};
    matlab_mat *A = mk(a, 1, 3);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *R = matlab_xcorr(A, B);
    double expected[] = {0, 1, 3, 5, 3};
    RT_CHECK(rt_cols(R) == 5, "len 2*max-1");
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_data(R)[i], expected[i], 1e-12, "unequal-length xcorr");
    rt_free(A); rt_free(B); rt_free(R);
}

static void test_fftshift_vector(void) {
    /* fftshift([0 1 2 3]) → [2 3 0 1]. Output is matlab_mat_c. */
    double a[] = {0, 1, 2, 3};
    matlab_mat *A = mk(a, 1, 4);
    matlab_mat_c *C = matlab_fftshift_c((void *)A);
    RT_CHECK(rt_c_rows(C) == 1 && rt_c_cols(C) == 4, "shape");
    double expected[] = {2, 3, 0, 1};
    for (int i = 0; i < 4; ++i) {
        RT_NEAR(rt_c_re(C, 0, i), expected[i], 1e-12, "fftshift re");
        RT_NEAR(rt_c_im(C, 0, i), 0.0,           1e-12, "fftshift im=0");
    }
    rt_free(A); rt_c_free(C);
}

static void test_ifftshift_inverts_fftshift(void) {
    double a[] = {0, 1, 2, 3, 4};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat_c *S = matlab_fftshift_c((void *)A);
    matlab_mat_c *R = matlab_ifftshift_c((void *)S);
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_c_re(R, 0, i), a[i], 1e-12, "round-trip");
    rt_free(A); rt_c_free(S); rt_c_free(R);
}

static void test_hamming_window(void) {
    matlab_mat *W = matlab_hamming(5);
    RT_CHECK(rt_rows(W) == 5 && rt_cols(W) == 1, "hamming column 5x1");
    /* Symmetric: W[0] = W[4] = 0.08, W[2] = 1.0 (centre). */
    RT_NEAR(rt_at(W, 0, 0), 0.08, 1e-12, "endpoint");
    RT_NEAR(rt_at(W, 4, 0), 0.08, 1e-12, "symmetric endpoint");
    RT_NEAR(rt_at(W, 2, 0), 1.00, 1e-12, "centre peak");
    rt_free(W);
}

static void test_hann_window(void) {
    matlab_mat *W = matlab_hann(5);
    RT_NEAR(rt_at(W, 0, 0), 0.0, 1e-12, "endpoint zero");
    RT_NEAR(rt_at(W, 4, 0), 0.0, 1e-12, "symmetric endpoint zero");
    RT_NEAR(rt_at(W, 2, 0), 1.0, 1e-12, "centre peak");
    rt_free(W);
}

static void test_blackman_window(void) {
    matlab_mat *W = matlab_blackman(5);
    /* Endpoints are ~0 (numerical residual ~1e-17 from cos ops). */
    RT_NEAR(rt_at(W, 0, 0), 0.0, 1e-10, "endpoint ~0");
    RT_NEAR(rt_at(W, 2, 0), 1.0, 1e-12, "centre peak");
    rt_free(W);
}

static void test_window_n_eq_1(void) {
    matlab_mat *W = matlab_hamming(1);
    RT_CHECK(rt_rows(W) == 1, "single-tap window");
    RT_NEAR(rt_at(W, 0, 0), 1.0, 0.0, "single tap = 1");
    rt_free(W);
}

static void test_rectwin_all_ones(void) {
    matlab_mat *W = matlab_rectwin(7);
    RT_CHECK(rt_rows(W) == 7 && rt_cols(W) == 1, "rectwin 7x1");
    for (int i = 0; i < 7; ++i)
        RT_NEAR(rt_at(W, i, 0), 1.0, 0.0, "rectwin entry");
    rt_free(W);
}

static void test_bartlett_triangular(void) {
    /* bartlett(5) = [0, 0.5, 1, 0.5, 0]. */
    matlab_mat *W = matlab_bartlett(5);
    RT_NEAR(rt_at(W, 0, 0), 0.0, 1e-12, "bartlett endpoint zero");
    RT_NEAR(rt_at(W, 1, 0), 0.5, 1e-12, "bartlett quarter");
    RT_NEAR(rt_at(W, 2, 0), 1.0, 1e-12, "bartlett peak");
    RT_NEAR(rt_at(W, 4, 0), 0.0, 1e-12, "bartlett symmetric end");
    rt_free(W);
}

static void test_triang_nonzero_endpoints(void) {
    /* triang differs from bartlett: endpoints are non-zero. */
    matlab_mat *W = matlab_triang(5);
    /* For odd n=5: w = [2/6, 4/6, 1, 4/6, 2/6]. */
    RT_NEAR(rt_at(W, 0, 0), 2.0/6.0, 1e-12, "triang odd endpoint");
    RT_NEAR(rt_at(W, 2, 0), 1.0,     1e-12, "triang odd peak");
    rt_free(W);
}

static void test_kaiser_beta0_is_rectwin(void) {
    /* kaiser(N, 0) is identically the rectangular window because
     * I_0(0) = 1 and the argument to the numerator is also zero. */
    matlab_mat *W = matlab_kaiser(8, 0.0);
    for (int i = 0; i < 8; ++i)
        RT_NEAR(rt_at(W, i, 0), 1.0, 1e-12, "kaiser beta=0 entry");
    rt_free(W);
}

static void test_kaiser_symmetric(void) {
    matlab_mat *W = matlab_kaiser(11, 6.0);
    RT_CHECK(rt_rows(W) == 11, "kaiser shape");
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_at(W, i, 0), rt_at(W, 10 - i, 0), 1e-12, "kaiser symmetric");
    /* Centre is the peak (== 1.0 because I_0(beta)/I_0(beta) = 1). */
    RT_NEAR(rt_at(W, 5, 0), 1.0, 1e-12, "kaiser centre peak");
    rt_free(W);
}

static void test_gausswin_centre_peak(void) {
    /* gausswin centred at the midpoint, value 1 there. */
    matlab_mat *W = matlab_gausswin(7, 2.5);
    RT_NEAR(rt_at(W, 3, 0), 1.0, 1e-12, "gausswin centre");
    /* Symmetric. */
    RT_NEAR(rt_at(W, 0, 0), rt_at(W, 6, 0), 1e-12, "gausswin symmetric");
    rt_free(W);
}

static void test_tukeywin_zero_is_rectwin(void) {
    /* r=0 -> rectangular. */
    matlab_mat *W = matlab_tukeywin(6, 0.0);
    for (int i = 0; i < 6; ++i)
        RT_NEAR(rt_at(W, i, 0), 1.0, 1e-12, "tukeywin r=0 entry");
    rt_free(W);
}

static void test_tukeywin_one_is_hann(void) {
    /* r=1 -> Hann window. */
    matlab_mat *W1 = matlab_tukeywin(7, 1.0);
    matlab_mat *W2 = matlab_hann(7);
    for (int i = 0; i < 7; ++i)
        RT_NEAR(rt_at(W1, i, 0), rt_at(W2, i, 0), 1e-12, "tukeywin r=1 == hann");
    rt_free(W1); rt_free(W2);
}

static void test_blackmanharris_normalisation(void) {
    /* Coefficients sum to a0 + a2 + a4 - a1 - a3 at the endpoints
     * for our convention. Just check shape + symmetry + bounded. */
    matlab_mat *W = matlab_blackmanharris(9);
    RT_CHECK(rt_rows(W) == 9, "shape");
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_at(W, i, 0), rt_at(W, 8 - i, 0), 1e-12, "bh symmetric");
    /* Centre is the peak; w[c] = a0 + a1 + a2 + a3 + a4 = 1. */
    RT_NEAR(rt_at(W, 4, 0), 1.0, 1e-9, "bh centre peak ~1");
    rt_free(W);
}

static void test_chebwin_normalised_to_one(void) {
    /* MATLAB normalises chebwin so max == 1. */
    matlab_mat *W = matlab_chebwin(11, 60.0);
    double mx = 0;
    for (int i = 0; i < 11; ++i) {
        double v = rt_at(W, i, 0);
        if (v > mx) mx = v;
    }
    RT_NEAR(mx, 1.0, 1e-12, "chebwin peak normalised");
    rt_free(W);
}

static void test_taylorwin_normalised(void) {
    matlab_mat *W = matlab_taylorwin(10, 4, -30);
    double mx = 0;
    for (int i = 0; i < 10; ++i)
        if (rt_at(W, i, 0) > mx) mx = rt_at(W, i, 0);
    RT_NEAR(mx, 1.0, 1e-12, "taylorwin peak normalised");
    rt_free(W);
}

static void test_upsample_inserts_zeros(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 1, 4);
    matlab_mat *R = matlab_upsample(A, 3);
    RT_CHECK(rt_cols(R) == 12, "len L*n");
    double expected[] = {1,0,0, 2,0,0, 3,0,0, 4,0,0};
    for (int i = 0; i < 12; ++i)
        RT_NEAR(rt_data(R)[i], expected[i], 1e-12, "upsample by 3");
    rt_free(A); rt_free(R);
}

static void test_downsample_takes_every_nth(void) {
    double a[] = {10, 20, 30, 40, 50, 60};
    matlab_mat *A = mk(a, 1, 6);
    matlab_mat *R = matlab_downsample(A, 2);
    double expected[] = {10, 30, 50};
    RT_CHECK(rt_cols(R) == 3, "len ceil(L/n)");
    for (int i = 0; i < 3; ++i)
        RT_NEAR(rt_data(R)[i], expected[i], 1e-12, "downsample by 2");
    rt_free(A); rt_free(R);
}

static void test_diff_vector(void) {
    /* diff([1 4 9 16 25]) = [3 5 7 9]. */
    double a[] = {1, 4, 9, 16, 25};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *D = matlab_diff(A);
    double expected[] = {3, 5, 7, 9};
    RT_CHECK(rt_cols(D) == 4, "len n-1");
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(D)[i], expected[i], 1e-12, "diff");
    rt_free(A); rt_free(D);
}

static void test_polyder_basic(void) {
    /* d/dx (x^3 + 2x^2 - x + 5) = 3x^2 + 4x - 1. */
    double a[] = {1, 2, -1, 5};
    matlab_mat *P = mk(a, 1, 4);
    matlab_mat *D = matlab_polyder(P);
    RT_CHECK(rt_cols(D) == 3, "polyder len n-1");
    RT_NEAR(rt_data(D)[0], 3.0,  1e-12, "polyder[0]");
    RT_NEAR(rt_data(D)[1], 4.0,  1e-12, "polyder[1]");
    RT_NEAR(rt_data(D)[2], -1.0, 1e-12, "polyder[2]");
    rt_free(P); rt_free(D);
}

static void test_polyder_constant_returns_zero(void) {
    double a[] = {7};
    matlab_mat *P = mk(a, 1, 1);
    matlab_mat *D = matlab_polyder(P);
    RT_CHECK(rt_cols(D) == 1 && rt_rows(D) == 1, "polyder of constant is [0]");
    RT_NEAR(rt_data(D)[0], 0.0, 0.0, "polyder constant");
    rt_free(P); rt_free(D);
}

static void test_polyint_basic(void) {
    /* integral(x^2) = x^3 / 3, so polyint([1 0 0]) = [1/3 0 0 0]. */
    double a[] = {1, 0, 0};
    matlab_mat *P = mk(a, 1, 3);
    matlab_mat *I = matlab_polyint(P);
    RT_CHECK(rt_cols(I) == 4, "polyint len n+1");
    RT_NEAR(rt_data(I)[0], 1.0/3.0, 1e-12, "polyint x^3 coef");
    RT_NEAR(rt_data(I)[1], 0.0,     1e-12, "polyint zero");
    RT_NEAR(rt_data(I)[2], 0.0,     1e-12, "polyint zero");
    RT_NEAR(rt_data(I)[3], 0.0,     1e-12, "polyint constant of integration");
    rt_free(P); rt_free(I);
}

static void test_polyint_k_constant(void) {
    /* integral(2x) + 7 = x^2 + 7  -->  [1 0 7]. */
    double a[] = {2, 0};
    matlab_mat *P = mk(a, 1, 2);
    matlab_mat *I = matlab_polyint_k(P, 7.0);
    RT_CHECK(rt_cols(I) == 3, "polyint_k len");
    RT_NEAR(rt_data(I)[0], 1.0, 1e-12, "x^2");
    RT_NEAR(rt_data(I)[1], 0.0, 1e-12, "x^1");
    RT_NEAR(rt_data(I)[2], 7.0, 1e-12, "constant");
    rt_free(P); rt_free(I);
}

static void test_poly_roots_round_trip(void) {
    /* poly([1 2]) = (x-1)(x-2) = x^2 - 3x + 2. */
    double a[] = {1, 2};
    matlab_mat *R = mk(a, 1, 2);
    matlab_mat *P = matlab_poly((void *)R);
    RT_CHECK(rt_cols(P) == 3 && rt_rows(P) == 1, "poly returns row 1xn+1");
    RT_NEAR(rt_data(P)[0], 1.0,  1e-12, "monic leading");
    RT_NEAR(rt_data(P)[1], -3.0, 1e-12, "p[1]");
    RT_NEAR(rt_data(P)[2], 2.0,  1e-12, "p[2]");
    rt_free(R); rt_free(P);
}

static void test_residue_distinct_poles(void) {
    /* H(s) = 1 / ((s - 1)(s - 2)). Residues at poles 1 and 2 are
     * -1 and 1. Sum is 0; product is -1. */
    double bd[] = {1};
    double ad[] = {1, -3, 2};
    matlab_mat *b = mk(bd, 1, 1);
    matlab_mat *a = mk(ad, 1, 3);
    matlab_mat_c *R = matlab_residue_r(b, a);
    matlab_mat_c *P = matlab_residue_p(b, a);
    matlab_mat   *K = matlab_residue_k(b, a);
    RT_CHECK(rt_c_rows(R) == 2 && rt_c_cols(R) == 1, "r is 2x1");
    RT_CHECK(rt_c_rows(P) == 2 && rt_c_cols(P) == 1, "p is 2x1");
    RT_CHECK(rt_rows(K) == 0 || rt_cols(K) == 0,
             "k empty when deg(b)<deg(a)");
    double r0 = rt_c_re(R, 0, 0), r1 = rt_c_re(R, 1, 0);
    double p0 = rt_c_re(P, 0, 0), p1 = rt_c_re(P, 1, 0);
    /* Order is solver-dependent; assert symmetric functions only. */
    RT_NEAR(r0 + r1, 0.0,  1e-10, "sum r");
    RT_NEAR(p0 + p1, 3.0,  1e-10, "sum p");
    RT_NEAR(r0 * r1, -1.0, 1e-10, "prod r");
    RT_NEAR(p0 * p1, 2.0,  1e-10, "prod p");
    RT_NEAR(rt_c_im(R, 0, 0), 0.0, 1e-10, "real residues");
    RT_NEAR(rt_c_im(R, 1, 0), 0.0, 1e-10, "real residues");
    rt_free(b); rt_free(a); rt_c_free(R); rt_c_free(P); rt_free(K);
}

static void test_residue_with_direct_term(void) {
    /* H(s) = (s^2 + 1) / (s - 1) = s + 1 + 2/(s - 1).
     * One pole at s = 1 with residue 2; direct term k = [1, 1]. */
    double bd[] = {1, 0, 1};
    double ad[] = {1, -1};
    matlab_mat *b = mk(bd, 1, 3);
    matlab_mat *a = mk(ad, 1, 2);
    matlab_mat_c *R = matlab_residue_r(b, a);
    matlab_mat_c *P = matlab_residue_p(b, a);
    matlab_mat   *K = matlab_residue_k(b, a);
    RT_CHECK(rt_c_rows(R) == 1, "single residue");
    RT_NEAR(rt_c_re(R, 0, 0), 2.0, 1e-10, "residue 2");
    RT_NEAR(rt_c_re(P, 0, 0), 1.0, 1e-10, "pole at 1");
    RT_CHECK(rt_cols(K) == 2, "k has 2 entries");
    RT_NEAR(rt_data(K)[0], 1.0, 1e-10, "k[0] = 1");
    RT_NEAR(rt_data(K)[1], 1.0, 1e-10, "k[1] = 1");
    rt_free(b); rt_free(a); rt_c_free(R); rt_c_free(P); rt_free(K);
}

static void test_poly_empty_is_one(void) {
    /* poly of empty vector is [1]. */
    matlab_mat *R = mk(NULL, 0, 0);
    matlab_mat *P = matlab_poly((void *)R);
    RT_CHECK(rt_rows(P) == 1 && rt_cols(P) == 1, "poly([]) = [1]");
    RT_NEAR(rt_data(P)[0], 1.0, 0.0, "scalar one");
    rt_free(R); rt_free(P);
}

static void test_diff_matrix_columnwise(void) {
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *D = matlab_diff(A);
    /* (m-1) x n = 2 x 3 with all entries = 3. */
    RT_CHECK(rt_rows(D) == 2 && rt_cols(D) == 3, "shape");
    for (int i = 0; i < 6; ++i)
        RT_NEAR(rt_data(D)[i], 3.0, 1e-12, "constant column diff");
    rt_free(A); rt_free(D);
}

int main(void) {
    fprintf(stderr, "test_signal:\n");
    RT_RUN(test_conv_polynomial_product);
    RT_RUN(test_conv_moving_sum);
    RT_RUN(test_conv_column_orientation);
    RT_RUN(test_conv2_box_filter);
    RT_RUN(test_conv2_separable_outer_product);
    RT_RUN(test_filter_moving_average);
    RT_RUN(test_filter_iir_geometric_decay);
    RT_RUN(test_filter_zero_a0_returns_empty);
    RT_RUN(test_xcorr_triangular_autocorr);
    RT_RUN(test_xcorr_unequal_lengths);
    RT_RUN(test_fftshift_vector);
    RT_RUN(test_ifftshift_inverts_fftshift);
    RT_RUN(test_hamming_window);
    RT_RUN(test_hann_window);
    RT_RUN(test_blackman_window);
    RT_RUN(test_window_n_eq_1);
    RT_RUN(test_rectwin_all_ones);
    RT_RUN(test_bartlett_triangular);
    RT_RUN(test_triang_nonzero_endpoints);
    RT_RUN(test_kaiser_beta0_is_rectwin);
    RT_RUN(test_kaiser_symmetric);
    RT_RUN(test_gausswin_centre_peak);
    RT_RUN(test_tukeywin_zero_is_rectwin);
    RT_RUN(test_tukeywin_one_is_hann);
    RT_RUN(test_blackmanharris_normalisation);
    RT_RUN(test_chebwin_normalised_to_one);
    RT_RUN(test_taylorwin_normalised);
    RT_RUN(test_upsample_inserts_zeros);
    RT_RUN(test_downsample_takes_every_nth);
    RT_RUN(test_diff_vector);
    RT_RUN(test_diff_matrix_columnwise);
    RT_RUN(test_polyder_basic);
    RT_RUN(test_polyder_constant_returns_zero);
    RT_RUN(test_polyint_basic);
    RT_RUN(test_polyint_k_constant);
    RT_RUN(test_poly_roots_round_trip);
    RT_RUN(test_poly_empty_is_one);
    RT_RUN(test_residue_distinct_poles);
    RT_RUN(test_residue_with_direct_term);
    RT_DONE();
}
