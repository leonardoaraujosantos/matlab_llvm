/* Direct tests for the array-shape ops (the family currently
 * copy-pasted across fliplr/flipud/flip/rot90/repmat/reshape/permute/
 * squeeze in runtime/matlab_runtime.c). */

#include "runtime_test.h"

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

static void test_fliplr(void) {
    double a[] = {1,2,3, 4,5,6};            /* 2x3 */
    matlab_mat *A = mk(a, 2, 3);
    matlab_mat *F = matlab_fliplr(A);
    /* expected: [3 2 1; 6 5 4] */
    double expected[] = {3,2,1, 6,5,4};
    RT_CHECK(rt_rows(F) == 2 && rt_cols(F) == 3, "fliplr shape");
    for (int k = 0; k < 6; ++k)
        RT_NEAR(rt_data(F)[k], expected[k], 0.0, "fliplr value");
    rt_free(A); rt_free(F);
}

static void test_flipud(void) {
    double a[] = {1,2,3, 4,5,6};
    matlab_mat *A = mk(a, 2, 3);
    matlab_mat *F = matlab_flipud(A);
    /* expected: [4 5 6; 1 2 3] */
    double expected[] = {4,5,6, 1,2,3};
    for (int k = 0; k < 6; ++k)
        RT_NEAR(rt_data(F)[k], expected[k], 0.0, "flipud value");
    rt_free(A); rt_free(F);
}

static void test_rot90(void) {
    /* MATLAB rot90: A_new(i,j) = A(j, n-1-i)  (90° counterclockwise) */
    double a[] = {1,2, 3,4};                /* 2x2 */
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *R = matlab_rot90(A);
    /* expected: [2 4; 1 3] */
    double expected[] = {2,4, 1,3};
    RT_CHECK(rt_rows(R) == 2 && rt_cols(R) == 2, "rot90 shape");
    for (int k = 0; k < 4; ++k)
        RT_NEAR(rt_data(R)[k], expected[k], 0.0, "rot90 value");
    rt_free(A); rt_free(R);
}

static void test_repmat(void) {
    double a[] = {1, 2};                    /* 1x2 */
    matlab_mat *A = mk(a, 1, 2);
    matlab_mat *R = matlab_repmat(A, 2, 3); /* tile to 2x6 */
    RT_CHECK(rt_rows(R) == 2 && rt_cols(R) == 6, "repmat shape");
    /* row pattern: 1 2 1 2 1 2 */
    for (int j = 0; j < 6; ++j) {
        RT_NEAR(rt_at(R, 0, j), (j % 2 == 0) ? 1.0 : 2.0, 0.0, "repmat row 0");
        RT_NEAR(rt_at(R, 1, j), (j % 2 == 0) ? 1.0 : 2.0, 0.0, "repmat row 1");
    }
    rt_free(A); rt_free(R);
}

static void test_reshape_preserves_data(void) {
    double a[] = {1,2,3,4,5,6};
    matlab_mat *A = mk(a, 2, 3);
    matlab_mat *R = matlab_reshape(A, 3, 2);
    RT_CHECK(rt_rows(R) == 3 && rt_cols(R) == 2, "reshape shape");
    /* MATLAB reshape uses column-major; the runtime is row-major and
     * its reshape simply re-sees the same buffer with new dims, so the
     * raw data ordering is preserved. */
    for (int k = 0; k < 6; ++k)
        RT_NEAR(rt_data(R)[k], rt_data(A)[k], 0.0, "reshape data");
    rt_free(A); rt_free(R);
}

static void test_range_step(void) {
    matlab_mat *r = matlab_range(0.0, 0.5, 2.0);
    RT_CHECK(rt_rows(r) * rt_cols(r) == 5, "range count");
    double expected[] = {0.0, 0.5, 1.0, 1.5, 2.0};
    for (int k = 0; k < 5; ++k)
        RT_NEAR(rt_data(r)[k], expected[k], 1e-12, "range value");
    rt_free(r);
}

static void test_diag_from_vector(void) {
    /* MATLAB diag of a vector returns a diagonal matrix. */
    double v[] = {1, 2, 3};
    matlab_mat *V = mk(v, 3, 1);
    matlab_mat *D = matlab_diag(V);
    RT_CHECK(rt_rows(D) == 3 && rt_cols(D) == 3, "diag(v) shape");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(D, i, j),
                    (i == j) ? rt_data(V)[i] : 0.0,
                    0.0, "diag(v) value");
    rt_free(V); rt_free(D);
}

int main(void) {
    fprintf(stderr, "test_shape:\n");
    RT_RUN(test_fliplr);
    RT_RUN(test_flipud);
    RT_RUN(test_rot90);
    RT_RUN(test_repmat);
    RT_RUN(test_reshape_preserves_data);
    RT_RUN(test_range_step);
    RT_RUN(test_diag_from_vector);
    RT_DONE();
}
