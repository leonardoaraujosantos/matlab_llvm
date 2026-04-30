/* Direct tests for the matlab_mat_c (complex) family in
 * runtime/matlab_runtime.c — the MAT_C_BINARY-generated ops plus the
 * scalar / view helpers. These paths are notoriously under-exercised
 * by the .m integration suite. */

#include "runtime_test.h"

static matlab_mat_c *mkc(const double *re, const double *im,
                         int64_t m, int64_t n) {
    return matlab_mat_c_from_buf(re, im, (double)m, (double)n);
}

static void test_complex_scalar_ctor(void) {
    matlab_mat_c *z = matlab_complex_scalar(3.0, -4.0);
    RT_CHECK(rt_c_rows(z) == 1 && rt_c_cols(z) == 1, "scalar shape");
    RT_NEAR(rt_c_re(z, 0, 0),  3.0, 0.0, "scalar re");
    RT_NEAR(rt_c_im(z, 0, 0), -4.0, 0.0, "scalar im");
    rt_c_free(z);
}

static void test_complex_from_real(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat   *R = matlab_mat_from_buf(a, 2, 2);
    matlab_mat_c *C = matlab_mat_c_from_real(R);
    RT_CHECK(rt_c_rows(C) == 2 && rt_c_cols(C) == 2, "from_real shape");
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            RT_NEAR(rt_c_re(C, i, j), rt_at(R, i, j), 0.0, "re == real");
            RT_NEAR(rt_c_im(C, i, j), 0.0, 0.0, "im == 0");
        }
    rt_free(R); rt_c_free(C);
}

static void test_add_cc(void) {
    double ar[] = {1, 2}, ai[] = {3, 4};
    double br[] = {5, 6}, bi[] = {7, 8};
    matlab_mat_c *A = mkc(ar, ai, 1, 2);
    matlab_mat_c *B = mkc(br, bi, 1, 2);
    matlab_mat_c *C = matlab_add_cc(A, B);
    RT_NEAR(rt_c_re(C,0,0),  6.0, 0.0, "(1+3i)+(5+7i) re");
    RT_NEAR(rt_c_im(C,0,0), 10.0, 0.0, "(1+3i)+(5+7i) im");
    RT_NEAR(rt_c_re(C,0,1),  8.0, 0.0, "(2+4i)+(6+8i) re");
    RT_NEAR(rt_c_im(C,0,1), 12.0, 0.0, "(2+4i)+(6+8i) im");
    rt_c_free(A); rt_c_free(B); rt_c_free(C);
}

static void test_emul_cc(void) {
    /* (1+2i)*(3+4i) = -5 + 10i */
    double ar[] = {1}, ai[] = {2};
    double br[] = {3}, bi[] = {4};
    matlab_mat_c *A = mkc(ar, ai, 1, 1);
    matlab_mat_c *B = mkc(br, bi, 1, 1);
    matlab_mat_c *C = matlab_emul_cc(A, B);
    RT_NEAR(rt_c_re(C,0,0), -5.0, 1e-12, "(1+2i)*(3+4i) re");
    RT_NEAR(rt_c_im(C,0,0), 10.0, 1e-12, "(1+2i)*(3+4i) im");
    rt_c_free(A); rt_c_free(B); rt_c_free(C);
}

static void test_ediv_cc(void) {
    /* (1+2i)/(1-i) = (1+2i)(1+i)/2 = (-1+3i)/2 = -0.5 + 1.5i */
    double ar[] = {1}, ai[] = { 2};
    double br[] = {1}, bi[] = {-1};
    matlab_mat_c *A = mkc(ar, ai, 1, 1);
    matlab_mat_c *B = mkc(br, bi, 1, 1);
    matlab_mat_c *C = matlab_ediv_cc(A, B);
    RT_NEAR(rt_c_re(C,0,0), -0.5, 1e-12, "(1+2i)/(1-i) re");
    RT_NEAR(rt_c_im(C,0,0),  1.5, 1e-12, "(1+2i)/(1-i) im");
    rt_c_free(A); rt_c_free(B); rt_c_free(C);
}

static void test_matmul_cc(void) {
    /* A = [i, 1; 1, i],  B = same.
     * A*B = [i*i + 1*1, i*1 + 1*i; 1*i + i*1, 1*1 + i*i]
     *     = [0, 2i; 2i, 0] */
    double ar[] = {0,1, 1,0};
    double ai[] = {1,0, 0,1};
    matlab_mat_c *A = mkc(ar, ai, 2, 2);
    matlab_mat_c *B = mkc(ar, ai, 2, 2);
    matlab_mat_c *C = matlab_matmul_cc(A, B);
    RT_NEAR(rt_c_re(C,0,0), 0.0, 1e-12, "matmul[0,0] re");
    RT_NEAR(rt_c_im(C,0,0), 0.0, 1e-12, "matmul[0,0] im");
    RT_NEAR(rt_c_re(C,0,1), 0.0, 1e-12, "matmul[0,1] re");
    RT_NEAR(rt_c_im(C,0,1), 2.0, 1e-12, "matmul[0,1] im");
    RT_NEAR(rt_c_re(C,1,0), 0.0, 1e-12, "matmul[1,0] re");
    RT_NEAR(rt_c_im(C,1,0), 2.0, 1e-12, "matmul[1,0] im");
    RT_NEAR(rt_c_re(C,1,1), 0.0, 1e-12, "matmul[1,1] re");
    RT_NEAR(rt_c_im(C,1,1), 0.0, 1e-12, "matmul[1,1] im");
    rt_c_free(A); rt_c_free(B); rt_c_free(C);
}

static void test_conj_neg(void) {
    double ar[] = {1, -2}, ai[] = {3, -4};
    matlab_mat_c *A   = mkc(ar, ai, 1, 2);
    matlab_mat_c *Ac  = matlab_conj_c((void *)A);
    matlab_mat_c *An  = matlab_neg_c(A);
    /* conj(1+3i) = 1-3i ; conj(-2-4i) = -2+4i */
    RT_NEAR(rt_c_re(Ac,0,0),  1.0, 0.0, "conj re 0");
    RT_NEAR(rt_c_im(Ac,0,0), -3.0, 0.0, "conj im 0");
    RT_NEAR(rt_c_re(Ac,0,1), -2.0, 0.0, "conj re 1");
    RT_NEAR(rt_c_im(Ac,0,1),  4.0, 0.0, "conj im 1");
    /* neg(1+3i) = -1-3i */
    RT_NEAR(rt_c_re(An,0,0), -1.0, 0.0, "neg re 0");
    RT_NEAR(rt_c_im(An,0,0), -3.0, 0.0, "neg im 0");
    rt_c_free(A); rt_c_free(Ac); rt_c_free(An);
}

static void test_real_imag_abs_angle(void) {
    /* z = 3+4i  →  real=3, imag=4, abs=5, angle=atan2(4,3) */
    matlab_mat_c *z = matlab_complex_scalar(3.0, 4.0);
    matlab_mat   *re = matlab_real_c((void *)z);
    matlab_mat   *im = matlab_imag_c((void *)z);
    matlab_mat   *ab = matlab_abs_c((void *)z);
    matlab_mat   *ag = matlab_angle_c((void *)z);
    RT_NEAR(rt_at(re,0,0), 3.0, 0.0, "real");
    RT_NEAR(rt_at(im,0,0), 4.0, 0.0, "imag");
    RT_NEAR(rt_at(ab,0,0), 5.0, 1e-12, "abs");
    RT_NEAR(rt_at(ag,0,0), atan2(4.0, 3.0), 1e-12, "angle");
    rt_c_free(z); rt_free(re); rt_free(im); rt_free(ab); rt_free(ag);
}

static void test_ctranspose(void) {
    /* ctranspose conjugates and transposes. */
    double ar[] = {1, 2, 3, 4};            /* 2x2 */
    double ai[] = {1, 1, 1, 1};
    matlab_mat_c *A  = mkc(ar, ai, 2, 2);
    matlab_mat_c *Ah = matlab_ctranspose_c(A);
    /* (Ah)_ji = conj(A_ij) */
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            RT_NEAR(rt_c_re(Ah, j, i),  rt_c_re(A, i, j), 0.0, "ctrans re");
            RT_NEAR(rt_c_im(Ah, j, i), -rt_c_im(A, i, j), 0.0, "ctrans im");
        }
    rt_c_free(A); rt_c_free(Ah);
}

int main(void) {
    fprintf(stderr, "test_complex:\n");
    RT_RUN(test_complex_scalar_ctor);
    RT_RUN(test_complex_from_real);
    RT_RUN(test_add_cc);
    RT_RUN(test_emul_cc);
    RT_RUN(test_ediv_cc);
    RT_RUN(test_matmul_cc);
    RT_RUN(test_conj_neg);
    RT_RUN(test_real_imag_abs_angle);
    RT_RUN(test_ctranspose);
    RT_DONE();
}
