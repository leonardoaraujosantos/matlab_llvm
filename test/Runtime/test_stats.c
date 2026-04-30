/* Direct unit tests for the Tier-1/2 stats / poly / numeric-calculus
 * builtins: any, all, tril, triu, std, var, median, meshgrid, ndgrid,
 * polyval, polyfit, roots, interp1, interp2, trapz, cumtrapz, gradient. */

#include "runtime_test.h"

matlab_mat   *matlab_any         (matlab_mat *A);
matlab_mat   *matlab_all         (matlab_mat *A);
matlab_mat   *matlab_tril        (matlab_mat *A);
matlab_mat   *matlab_triu        (matlab_mat *A);
matlab_mat   *matlab_std         (matlab_mat *A);
matlab_mat   *matlab_var         (matlab_mat *A);
matlab_mat   *matlab_median      (matlab_mat *A);
matlab_mat   *matlab_meshgrid_X  (matlab_mat *x, matlab_mat *y);
matlab_mat   *matlab_meshgrid_Y  (matlab_mat *x, matlab_mat *y);
matlab_mat   *matlab_ndgrid_X    (matlab_mat *x, matlab_mat *y);
matlab_mat   *matlab_ndgrid_Y    (matlab_mat *x, matlab_mat *y);
matlab_mat   *matlab_polyval     (matlab_mat *p, matlab_mat *x);
matlab_mat   *matlab_polyfit     (matlab_mat *x, matlab_mat *y, double n);
matlab_mat_c *matlab_roots       (matlab_mat *p);
matlab_mat   *matlab_interp1     (matlab_mat *x, matlab_mat *y, matlab_mat *xi);
matlab_mat   *matlab_interp2     (matlab_mat *X, matlab_mat *Y, matlab_mat *V,
                                   matlab_mat *Xq, matlab_mat *Yq);
matlab_mat   *matlab_trapz       (matlab_mat *y);
matlab_mat   *matlab_trapz_xy    (matlab_mat *x, matlab_mat *y);
matlab_mat   *matlab_cumtrapz    (matlab_mat *y);
matlab_mat   *matlab_gradient    (matlab_mat *f);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* --- any / all ------------------------------------------------------- */
static void test_any_vector(void) {
    double a[] = {0, 0, 3, 0, 5};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *R = matlab_any(A);
    RT_NEAR(rt_at(R, 0, 0), 1.0, 0.0, "any non-zero");
    rt_free(A); rt_free(R);
}
static void test_any_all_zero(void) {
    double a[] = {0, 0, 0};
    matlab_mat *A = mk(a, 1, 3);
    matlab_mat *R = matlab_any(A);
    RT_NEAR(rt_at(R, 0, 0), 0.0, 0.0, "any of zeros");
    rt_free(A); rt_free(R);
}
static void test_all_vector(void) {
    double a[] = {1, 2, 3, 0, 5};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *R = matlab_all(A);
    RT_NEAR(rt_at(R, 0, 0), 0.0, 0.0, "has a zero");
    rt_free(A); rt_free(R);
}
static void test_all_columnwise(void) {
    /* M = [0 1 0; 0 1 1; 0 0 1] — column AND: [0 0 0]. */
    double a[] = {0,1,0, 0,1,1, 0,0,1};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *R = matlab_all(A);
    RT_CHECK(rt_rows(R) == 1 && rt_cols(R) == 3, "shape");
    RT_NEAR(rt_at(R, 0, 0), 0.0, 0.0, "col0");
    RT_NEAR(rt_at(R, 0, 1), 0.0, 0.0, "col1");
    RT_NEAR(rt_at(R, 0, 2), 0.0, 0.0, "col2");
    rt_free(A); rt_free(R);
}
static void test_any_columnwise(void) {
    double a[] = {0,1,0, 0,1,1, 0,0,1};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *R = matlab_any(A);
    RT_NEAR(rt_at(R, 0, 0), 0.0, 0.0, "col0 all-zero");
    RT_NEAR(rt_at(R, 0, 1), 1.0, 0.0, "col1 has nonzero");
    RT_NEAR(rt_at(R, 0, 2), 1.0, 0.0, "col2 has nonzero");
    rt_free(A); rt_free(R);
}

/* --- tril / triu ----------------------------------------------------- */
static void test_tril_3x3(void) {
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *L = matlab_tril(A);
    /* upper triangle should be zero */
    RT_NEAR(rt_at(L, 0, 1), 0.0, 0.0, "L[0,1]");
    RT_NEAR(rt_at(L, 0, 2), 0.0, 0.0, "L[0,2]");
    RT_NEAR(rt_at(L, 1, 2), 0.0, 0.0, "L[1,2]");
    /* lower + diagonal preserved */
    RT_NEAR(rt_at(L, 0, 0), 1.0, 0.0, "L[0,0]");
    RT_NEAR(rt_at(L, 1, 1), 5.0, 0.0, "L[1,1]");
    RT_NEAR(rt_at(L, 2, 0), 7.0, 0.0, "L[2,0]");
    rt_free(A); rt_free(L);
}
static void test_triu_3x3(void) {
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *U = matlab_triu(A);
    RT_NEAR(rt_at(U, 1, 0), 0.0, 0.0, "U[1,0]");
    RT_NEAR(rt_at(U, 2, 0), 0.0, 0.0, "U[2,0]");
    RT_NEAR(rt_at(U, 2, 1), 0.0, 0.0, "U[2,1]");
    RT_NEAR(rt_at(U, 0, 2), 3.0, 0.0, "U[0,2]");
    RT_NEAR(rt_at(U, 1, 2), 6.0, 0.0, "U[1,2]");
    rt_free(A); rt_free(U);
}

/* --- std / var / median ---------------------------------------------- */
static void test_var_sample(void) {
    /* var([1 2 3 4 5]) with N-1 normalisation = 10/4 = 2.5. */
    double a[] = {1, 2, 3, 4, 5};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *V = matlab_var(A);
    RT_NEAR(rt_at(V, 0, 0), 2.5, 1e-12, "sample var");
    rt_free(A); rt_free(V);
}
static void test_std_is_sqrt_var(void) {
    double a[] = {1, 2, 3, 4, 5};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *S = matlab_std(A);
    RT_NEAR(rt_at(S, 0, 0), sqrt(2.5), 1e-12, "std");
    rt_free(A); rt_free(S);
}
static void test_median_odd(void) {
    double a[] = {7, 3, 1, 9, 5};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *M = matlab_median(A);
    RT_NEAR(rt_at(M, 0, 0), 5.0, 0.0, "median odd");
    rt_free(A); rt_free(M);
}
static void test_median_even(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 1, 4);
    matlab_mat *M = matlab_median(A);
    RT_NEAR(rt_at(M, 0, 0), 2.5, 1e-12, "median even");
    rt_free(A); rt_free(M);
}
static void test_var_columnwise(void) {
    /* A = [1 2 3; 4 5 6; 7 8 9]. Each col is [1,4,7]/[2,5,8]/[3,6,9].
     * mean = 4/5/6, var = ((-3)^2 + 0 + 3^2) / 2 = 9 each. */
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *V = matlab_var(A);
    for (int j = 0; j < 3; ++j)
        RT_NEAR(rt_at(V, 0, j), 9.0, 1e-12, "col var");
    rt_free(A); rt_free(V);
}

/* --- meshgrid / ndgrid ----------------------------------------------- */
static void test_meshgrid_xy(void) {
    /* meshgrid([10 20 30], [1 2]):
     *   X = [10 20 30; 10 20 30],  Y = [1 1 1; 2 2 2] */
    double xd[] = {10, 20, 30};
    double yd[] = {1, 2};
    matlab_mat *x = mk(xd, 1, 3);
    matlab_mat *y = mk(yd, 1, 2);
    matlab_mat *X = matlab_meshgrid_X(x, y);
    matlab_mat *Y = matlab_meshgrid_Y(x, y);
    RT_CHECK(rt_rows(X) == 2 && rt_cols(X) == 3, "X shape");
    RT_NEAR(rt_at(X, 0, 0), 10, 0.0, "X[0,0]");
    RT_NEAR(rt_at(X, 1, 2), 30, 0.0, "X[1,2]");
    RT_NEAR(rt_at(Y, 0, 0),  1, 0.0, "Y[0,0]");
    RT_NEAR(rt_at(Y, 1, 0),  2, 0.0, "Y[1,0]");
    rt_free(x); rt_free(y); rt_free(X); rt_free(Y);
}
static void test_ndgrid_ij(void) {
    double xd[] = {10, 20, 30};
    double yd[] = {1, 2};
    matlab_mat *x = mk(xd, 1, 3);
    matlab_mat *y = mk(yd, 1, 2);
    matlab_mat *X = matlab_ndgrid_X(x, y);
    matlab_mat *Y = matlab_ndgrid_Y(x, y);
    /* ndgrid: X 3x2 with X(i,j)=x(i); Y 3x2 with Y(i,j)=y(j). */
    RT_CHECK(rt_rows(X) == 3 && rt_cols(X) == 2, "X shape");
    RT_NEAR(rt_at(X, 0, 0), 10, 0.0, "X[0,0]");
    RT_NEAR(rt_at(X, 2, 1), 30, 0.0, "X[2,1]");
    RT_NEAR(rt_at(Y, 0, 0),  1, 0.0, "Y[0,0]");
    RT_NEAR(rt_at(Y, 0, 1),  2, 0.0, "Y[0,1]");
    rt_free(x); rt_free(y); rt_free(X); rt_free(Y);
}
static void test_meshgrid_one_arg_via_NULL(void) {
    /* meshgrid(x) == meshgrid(x, x). The compiler passes NULL as y. */
    double xd[] = {1, 2};
    matlab_mat *x = mk(xd, 1, 2);
    matlab_mat *X = matlab_meshgrid_X(x, NULL);
    RT_CHECK(rt_rows(X) == 2 && rt_cols(X) == 2, "X 2x2");
    RT_NEAR(rt_at(X, 1, 1), 2, 0.0, "X[1,1]");
    rt_free(x); rt_free(X);
}

/* --- polyval / polyfit / roots --------------------------------------- */
static void test_polyval_quadratic(void) {
    /* p(x) = x^2 - 3x + 2 evaluated at x = 0..3 gives [2 0 0 2]. */
    double pd[] = {1, -3, 2};
    double xd[] = {0, 1, 2, 3};
    matlab_mat *p = mk(pd, 1, 3);
    matlab_mat *x = mk(xd, 1, 4);
    matlab_mat *Y = matlab_polyval(p, x);
    double expected[] = {2, 0, 0, 2};
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(Y)[i], expected[i], 1e-12, "polyval");
    rt_free(p); rt_free(x); rt_free(Y);
}
static void test_polyfit_recovers_quadratic(void) {
    /* Fit y = 2x^2 + 1 from 5 sample points. Expected p = [2 0 1]. */
    double xd[] = {-2, -1, 0, 1, 2};
    double yd[] = {9, 3, 1, 3, 9};
    matlab_mat *x = mk(xd, 1, 5);
    matlab_mat *y = mk(yd, 1, 5);
    matlab_mat *P = matlab_polyfit(x, y, 2);
    RT_CHECK(rt_cols(P) == 3, "deg+1 coeffs");
    RT_NEAR(rt_data(P)[0], 2.0, 1e-9, "leading coeff");
    RT_NEAR(rt_data(P)[1], 0.0, 1e-9, "linear coeff");
    RT_NEAR(rt_data(P)[2], 1.0, 1e-9, "constant");
    rt_free(x); rt_free(y); rt_free(P);
}
static void test_roots_real_quadratic(void) {
    /* roots([1 -5 6]) → 2 and 3 (any order). */
    double pd[] = {1, -5, 6};
    matlab_mat *p = mk(pd, 1, 3);
    matlab_mat_c *R = matlab_roots(p);
    RT_CHECK(rt_c_rows(R) == 2 && rt_c_cols(R) == 1, "deg roots");
    /* Find both roots regardless of ordering. */
    double r0 = rt_c_re(R, 0, 0), r1 = rt_c_re(R, 1, 0);
    int got_2 = (fabs(r0 - 2.0) < 1e-6) || (fabs(r1 - 2.0) < 1e-6);
    int got_3 = (fabs(r0 - 3.0) < 1e-6) || (fabs(r1 - 3.0) < 1e-6);
    RT_CHECK(got_2 && got_3, "found both real roots");
    /* Imag parts ≈ 0 for real roots. */
    RT_NEAR(rt_c_im(R, 0, 0), 0.0, 1e-6, "im0 ≈ 0");
    RT_NEAR(rt_c_im(R, 1, 0), 0.0, 1e-6, "im1 ≈ 0");
    rt_free(p); rt_c_free(R);
}
static void test_roots_complex_pair(void) {
    /* roots([1 0 1]) → ±i. */
    double pd[] = {1, 0, 1};
    matlab_mat *p = mk(pd, 1, 3);
    matlab_mat_c *R = matlab_roots(p);
    /* Both roots on imaginary axis, magnitudes 1. */
    for (int i = 0; i < 2; ++i) {
        RT_NEAR(rt_c_re(R, i, 0), 0.0, 1e-6, "real ≈ 0");
        RT_NEAR(fabs(rt_c_im(R, i, 0)), 1.0, 1e-6, "|imag| = 1");
    }
    rt_free(p); rt_c_free(R);
}

/* --- interp1 / interp2 ----------------------------------------------- */
static void test_interp1_linear(void) {
    /* y = x^2 sampled at integers 0..4; query mid-points. */
    double xd[] = {0, 1, 2, 3, 4};
    double yd[] = {0, 1, 4, 9, 16};
    double qd[] = {0.5, 1.5, 2.5, 3.5};
    matlab_mat *x  = mk(xd, 1, 5);
    matlab_mat *y  = mk(yd, 1, 5);
    matlab_mat *xi = mk(qd, 1, 4);
    matlab_mat *yi = matlab_interp1(x, y, xi);
    /* Linear interp between (0,0) and (1,1) at 0.5 → 0.5; (1,1)-(2,4) at 1.5 → 2.5; etc. */
    double expected[] = {0.5, 2.5, 6.5, 12.5};
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(yi)[i], expected[i], 1e-12, "linear interp");
    rt_free(x); rt_free(y); rt_free(xi); rt_free(yi);
}
static void test_interp1_out_of_range_nan(void) {
    double xd[] = {0, 1, 2};
    double yd[] = {0, 1, 4};
    double qd[] = {-1, 5};
    matlab_mat *x  = mk(xd, 1, 3);
    matlab_mat *y  = mk(yd, 1, 3);
    matlab_mat *xi = mk(qd, 1, 2);
    matlab_mat *yi = matlab_interp1(x, y, xi);
    RT_CHECK(isnan(rt_data(yi)[0]), "below range NaN");
    RT_CHECK(isnan(rt_data(yi)[1]), "above range NaN");
    rt_free(x); rt_free(y); rt_free(xi); rt_free(yi);
}
static void test_interp2_bilinear(void) {
    /* z = x + 10*y on {0,1,2}x{0,1,2}. Query (1.5, 0.5) → 6.5. */
    double xd[] = {0, 1, 2};
    double yd[] = {0, 1, 2};
    double Vd[] = {0, 1, 2,  10, 11, 12,  20, 21, 22};
    double xq[] = {1.5};
    double yq[] = {0.5};
    matlab_mat *X  = mk(xd, 1, 3);
    matlab_mat *Y  = mk(yd, 3, 1);
    matlab_mat *V  = mk(Vd, 3, 3);
    matlab_mat *Xq = mk(xq, 1, 1);
    matlab_mat *Yq = mk(yq, 1, 1);
    matlab_mat *R  = matlab_interp2(X, Y, V, Xq, Yq);
    RT_NEAR(rt_at(R, 0, 0), 6.5, 1e-12, "bilinear");
    rt_free(X); rt_free(Y); rt_free(V); rt_free(Xq); rt_free(Yq); rt_free(R);
}

/* --- trapz / cumtrapz / gradient ------------------------------------- */
static void test_trapz_unit_spacing(void) {
    /* trapz([1 2 3 4 5]) = 0.5*(1+5) + 2+3+4 = 12. */
    double yd[] = {1, 2, 3, 4, 5};
    matlab_mat *y = mk(yd, 1, 5);
    matlab_mat *I = matlab_trapz(y);
    RT_NEAR(rt_at(I, 0, 0), 12.0, 1e-12, "trapz unit");
    rt_free(y); rt_free(I);
}
static void test_trapz_xy(void) {
    /* x = 0:0.5:2, y = x.^2 → trapezoidal integral = 2.75. */
    double xd[] = {0, 0.5, 1.0, 1.5, 2.0};
    double yd[] = {0, 0.25, 1.0, 2.25, 4.0};
    matlab_mat *x = mk(xd, 1, 5);
    matlab_mat *y = mk(yd, 1, 5);
    matlab_mat *I = matlab_trapz_xy(x, y);
    RT_NEAR(rt_at(I, 0, 0), 2.75, 1e-12, "trapz(x,y)");
    rt_free(x); rt_free(y); rt_free(I);
}
static void test_cumtrapz_unit(void) {
    /* cumtrapz([1 1 1 1 1]) = [0 1 2 3 4]. */
    double yd[] = {1, 1, 1, 1, 1};
    matlab_mat *y = mk(yd, 1, 5);
    matlab_mat *C = matlab_cumtrapz(y);
    double expected[] = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_data(C)[i], expected[i], 1e-12, "cumtrapz");
    rt_free(y); rt_free(C);
}
static void test_gradient_vector(void) {
    /* gradient([1 4 9 16 25]):
     *   g[0] = 4-1=3, g[1]=(9-1)/2=4, g[2]=(16-4)/2=6, g[3]=(25-9)/2=8, g[4]=25-16=9 */
    double fd[] = {1, 4, 9, 16, 25};
    matlab_mat *F = mk(fd, 1, 5);
    matlab_mat *G = matlab_gradient(F);
    double expected[] = {3, 4, 6, 8, 9};
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_data(G)[i], expected[i], 1e-12, "gradient");
    rt_free(F); rt_free(G);
}

int main(void) {
    fprintf(stderr, "test_stats:\n");
    RT_RUN(test_any_vector);
    RT_RUN(test_any_all_zero);
    RT_RUN(test_all_vector);
    RT_RUN(test_all_columnwise);
    RT_RUN(test_any_columnwise);
    RT_RUN(test_tril_3x3);
    RT_RUN(test_triu_3x3);
    RT_RUN(test_var_sample);
    RT_RUN(test_std_is_sqrt_var);
    RT_RUN(test_median_odd);
    RT_RUN(test_median_even);
    RT_RUN(test_var_columnwise);
    RT_RUN(test_meshgrid_xy);
    RT_RUN(test_ndgrid_ij);
    RT_RUN(test_meshgrid_one_arg_via_NULL);
    RT_RUN(test_polyval_quadratic);
    RT_RUN(test_polyfit_recovers_quadratic);
    RT_RUN(test_roots_real_quadratic);
    RT_RUN(test_roots_complex_pair);
    RT_RUN(test_interp1_linear);
    RT_RUN(test_interp1_out_of_range_nan);
    RT_RUN(test_interp2_bilinear);
    RT_RUN(test_trapz_unit_spacing);
    RT_RUN(test_trapz_xy);
    RT_RUN(test_cumtrapz_unit);
    RT_RUN(test_gradient_vector);
    RT_DONE();
}
