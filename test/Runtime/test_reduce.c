/* Direct tests for reductions / scans / sort / set ops. These exercise
 * the COLWISE_REDUCE, DIM_REDUCE, CUM_SCAN macros plus sort/unique/
 * sortrows/ismember in runtime/matlab_runtime.c. */

#include "runtime_test.h"

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

static void test_sum_vector(void) {
    double a[] = {1, 2, 3, 4, 5};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *S = matlab_sum(A);
    RT_CHECK(rt_rows(S) * rt_cols(S) == 1, "sum(vec) is scalar");
    RT_NEAR(rt_data(S)[0], 15.0, 0.0, "sum value");
    rt_free(A); rt_free(S);
}

static void test_sum_matrix_columnwise(void) {
    /* sum of a 2x3 matrix is a 1x3 row of column sums (MATLAB default). */
    double a[] = {1,2,3, 4,5,6};
    matlab_mat *A = mk(a, 2, 3);
    matlab_mat *S = matlab_sum(A);
    RT_CHECK(rt_rows(S) == 1 && rt_cols(S) == 3, "sum(2x3) shape");
    RT_NEAR(rt_data(S)[0], 5.0,  0.0, "col 0");
    RT_NEAR(rt_data(S)[1], 7.0,  0.0, "col 1");
    RT_NEAR(rt_data(S)[2], 9.0,  0.0, "col 2");
    rt_free(A); rt_free(S);
}

static void test_prod_mean(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A  = mk(a, 1, 4);
    matlab_mat *P  = matlab_prod(A);
    matlab_mat *Mn = matlab_mean(A);
    RT_NEAR(rt_data(P)[0],  24.0, 0.0,    "prod");
    RT_NEAR(rt_data(Mn)[0],  2.5, 1e-12,  "mean");
    rt_free(A); rt_free(P); rt_free(Mn);
}

static void test_min_max_vector(void) {
    double a[] = {3, -1, 4, 1, -5, 9, 2, 6};
    matlab_mat *A   = mk(a, 1, 8);
    matlab_mat *Mn  = matlab_min(A);
    matlab_mat *Mx  = matlab_max(A);
    RT_NEAR(rt_data(Mn)[0], -5.0, 0.0, "min");
    RT_NEAR(rt_data(Mx)[0],  9.0, 0.0, "max");
    rt_free(A); rt_free(Mn); rt_free(Mx);
}

static void test_cumsum_cumprod(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A  = mk(a, 1, 4);
    matlab_mat *Cs = matlab_cumsum(A);
    matlab_mat *Cp = matlab_cumprod(A);
    double exp_s[] = {1, 3, 6, 10};
    double exp_p[] = {1, 2, 6, 24};
    for (int k = 0; k < 4; ++k) {
        RT_NEAR(rt_data(Cs)[k], exp_s[k], 0.0, "cumsum");
        RT_NEAR(rt_data(Cp)[k], exp_p[k], 0.0, "cumprod");
    }
    rt_free(A); rt_free(Cs); rt_free(Cp);
}

static void test_sort_ascending(void) {
    double a[] = {3, 1, 4, 1, 5, 9, 2, 6};
    matlab_mat *A = mk(a, 1, 8);
    matlab_mat *S = matlab_sort(A);
    double expected[] = {1, 1, 2, 3, 4, 5, 6, 9};
    for (int k = 0; k < 8; ++k)
        RT_NEAR(rt_data(S)[k], expected[k], 0.0, "sort asc");
    rt_free(A); rt_free(S);
}

static void test_unique_dedups_and_sorts(void) {
    double a[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    matlab_mat *A = mk(a, 1, 10);
    matlab_mat *U = matlab_unique(A);
    /* Expected: 1,2,3,4,5,6,9 — 7 unique values. */
    RT_CHECK(rt_rows(U) * rt_cols(U) == 7, "unique count");
    double expected[] = {1, 2, 3, 4, 5, 6, 9};
    for (int k = 0; k < 7; ++k)
        RT_NEAR(rt_data(U)[k], expected[k], 0.0, "unique value");
    rt_free(A); rt_free(U);
}

static void test_ismember(void) {
    double a[] = {1, 2, 3, 4, 5};
    double b[] = {2, 4, 6};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *B = mk(b, 1, 3);
    matlab_mat *M = matlab_ismember(A, B);
    /* Expected: 0 1 0 1 0 (1 if a-element is in b) */
    double expected[] = {0, 1, 0, 1, 0};
    for (int k = 0; k < 5; ++k)
        RT_NEAR(rt_data(M)[k], expected[k], 0.0, "ismember");
    rt_free(A); rt_free(B); rt_free(M);
}

int main(void) {
    fprintf(stderr, "test_reduce:\n");
    RT_RUN(test_sum_vector);
    RT_RUN(test_sum_matrix_columnwise);
    RT_RUN(test_prod_mean);
    RT_RUN(test_min_max_vector);
    RT_RUN(test_cumsum_cumprod);
    RT_RUN(test_sort_ascending);
    RT_RUN(test_unique_dedups_and_sorts);
    RT_RUN(test_ismember);
    RT_DONE();
}
