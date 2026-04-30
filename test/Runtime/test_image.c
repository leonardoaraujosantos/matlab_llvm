/* Direct unit tests for the Tier-3 image-processing wrappers and the
 * SVD-derived linalg helpers: imfilter, padarray, rank, cond, null, orth. */

#include "runtime_test.h"

matlab_mat *matlab_imfilter (matlab_mat *A, matlab_mat *h);
matlab_mat *matlab_padarray (matlab_mat *A, matlab_mat *padsize);
double      matlab_rank     (matlab_mat *A);
double      matlab_cond     (matlab_mat *A);
matlab_mat *matlab_null     (matlab_mat *A);
matlab_mat *matlab_orth     (matlab_mat *A);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* --- imfilter -------------------------------------------------------- */
static void test_imfilter_box_same_size(void) {
    /* 3x3 box average of [[1..9]] — output size matches input. */
    double imgd[] = {1,2,3, 4,5,6, 7,8,9};
    double hd[]   = {1.0/9, 1.0/9, 1.0/9,
                     1.0/9, 1.0/9, 1.0/9,
                     1.0/9, 1.0/9, 1.0/9};
    matlab_mat *A = mk(imgd, 3, 3);
    matlab_mat *h = mk(hd, 3, 3);
    matlab_mat *R = matlab_imfilter(A, h);
    RT_CHECK(rt_rows(R) == 3 && rt_cols(R) == 3, "same-size output");
    /* Centre cell = average of all 9 entries = 5. */
    RT_NEAR(rt_at(R, 1, 1), 5.0, 1e-12, "centre = mean");
    /* Top-left cell — only 4 in-bounds entries (1,2,4,5), rest are
     * implicit zero. Result = 12/9 = 1.333… */
    RT_NEAR(rt_at(R, 0, 0), 12.0 / 9.0, 1e-12, "top-left");
    rt_free(A); rt_free(h); rt_free(R);
}

static void test_imfilter_identity_kernel(void) {
    /* 1×1 identity kernel — output = input. */
    double imgd[] = {1, 2, 3, 4, 5, 6};
    double hd[]   = {1};
    matlab_mat *A = mk(imgd, 2, 3);
    matlab_mat *h = mk(hd, 1, 1);
    matlab_mat *R = matlab_imfilter(A, h);
    for (int i = 0; i < 6; ++i)
        RT_NEAR(rt_data(R)[i], imgd[i], 1e-12, "identity kernel");
    rt_free(A); rt_free(h); rt_free(R);
}

/* --- padarray -------------------------------------------------------- */
static void test_padarray_symmetric_zero(void) {
    /* padarray([1 2; 3 4], [1 1]) → 4x4 with zeros around the 2x2 core. */
    double imgd[] = {1, 2, 3, 4};
    double psd[]  = {1, 1};
    matlab_mat *A  = mk(imgd, 2, 2);
    matlab_mat *PS = mk(psd, 1, 2);
    matlab_mat *R  = matlab_padarray(A, PS);
    RT_CHECK(rt_rows(R) == 4 && rt_cols(R) == 4, "padded shape");
    /* corners = 0 */
    RT_NEAR(rt_at(R, 0, 0), 0.0, 0.0, "TL zero");
    RT_NEAR(rt_at(R, 3, 3), 0.0, 0.0, "BR zero");
    /* core preserved */
    RT_NEAR(rt_at(R, 1, 1), 1.0, 0.0, "core[0,0]");
    RT_NEAR(rt_at(R, 2, 2), 4.0, 0.0, "core[1,1]");
    rt_free(A); rt_free(PS); rt_free(R);
}

static void test_padarray_scalar_padsize(void) {
    /* Scalar padsize applied to both dims. */
    double imgd[] = {7};
    double psd[]  = {2};
    matlab_mat *A  = mk(imgd, 1, 1);
    matlab_mat *PS = mk(psd, 1, 1);
    matlab_mat *R  = matlab_padarray(A, PS);
    RT_CHECK(rt_rows(R) == 5 && rt_cols(R) == 5, "scalar pad both dims");
    RT_NEAR(rt_at(R, 2, 2), 7.0, 0.0, "centre preserved");
    rt_free(A); rt_free(PS); rt_free(R);
}

/* --- rank / cond ----------------------------------------------------- */
static void test_rank_full(void) {
    /* diag([1 2 3]) is rank 3. */
    double a[] = {1,0,0, 0,2,0, 0,0,3};
    matlab_mat *A = mk(a, 3, 3);
    RT_NEAR(matlab_rank(A), 3.0, 0.0, "rank diag");
    rt_free(A);
}
static void test_rank_singular(void) {
    /* [1 2 3; 4 5 6; 7 8 9] is rank 2. */
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    RT_NEAR(matlab_rank(A), 2.0, 0.0, "rank singular");
    rt_free(A);
}
static void test_cond_identity(void) {
    matlab_mat *I = matlab_eye(3, 3);
    RT_NEAR(matlab_cond(I), 1.0, 1e-12, "cond(I) = 1");
    rt_free(I);
}
static void test_cond_diag(void) {
    /* diag([1 2 3]) — σ_max/σ_min = 3/1 = 3. */
    double a[] = {1,0,0, 0,2,0, 0,0,3};
    matlab_mat *A = mk(a, 3, 3);
    RT_NEAR(matlab_cond(A), 3.0, 1e-12, "cond diag");
    rt_free(A);
}

/* --- null / orth ----------------------------------------------------- */
static void test_null_singular_matrix(void) {
    /* Null space of [1 2 3; 4 5 6; 7 8 9] is 1-dimensional. */
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *N = matlab_null(A);
    RT_CHECK(rt_rows(N) == 3, "null rows = n");
    RT_CHECK(rt_cols(N) == 1, "null dim = n - rank");
    /* A * N should be ~0. */
    matlab_mat *AN = matlab_matmul_mm(A, N);
    for (int i = 0; i < 3; ++i)
        RT_NEAR(rt_at(AN, i, 0), 0.0, 1e-10, "A*null(A) ≈ 0");
    /* N has unit-norm columns. */
    double s = 0;
    for (int i = 0; i < 3; ++i) {
        double v = rt_at(N, i, 0);
        s += v * v;
    }
    RT_NEAR(s, 1.0, 1e-10, "unit-norm null vector");
    rt_free(A); rt_free(N); rt_free(AN);
}

static void test_orth_full_rank(void) {
    /* For full-rank 3x2 input, orth returns a 3x2 orthonormal matrix. */
    double a[] = {1, 0,
                  0, 1,
                  1, 1};
    matlab_mat *A = mk(a, 3, 2);
    matlab_mat *Q = matlab_orth(A);
    RT_CHECK(rt_rows(Q) == 3 && rt_cols(Q) == 2, "orth shape");
    /* Q' * Q = I (2x2). */
    double dot00 = 0, dot11 = 0, dot01 = 0;
    for (int i = 0; i < 3; ++i) {
        dot00 += rt_at(Q, i, 0) * rt_at(Q, i, 0);
        dot11 += rt_at(Q, i, 1) * rt_at(Q, i, 1);
        dot01 += rt_at(Q, i, 0) * rt_at(Q, i, 1);
    }
    RT_NEAR(dot00, 1.0, 1e-10, "col0 unit norm");
    RT_NEAR(dot11, 1.0, 1e-10, "col1 unit norm");
    RT_NEAR(dot01, 0.0, 1e-10, "cols orthogonal");
    rt_free(A); rt_free(Q);
}

int main(void) {
    fprintf(stderr, "test_image:\n");
    RT_RUN(test_imfilter_box_same_size);
    RT_RUN(test_imfilter_identity_kernel);
    RT_RUN(test_padarray_symmetric_zero);
    RT_RUN(test_padarray_scalar_padsize);
    RT_RUN(test_rank_full);
    RT_RUN(test_rank_singular);
    RT_RUN(test_cond_identity);
    RT_RUN(test_cond_diag);
    RT_RUN(test_null_singular_matrix);
    RT_RUN(test_orth_full_rank);
    RT_DONE();
}
