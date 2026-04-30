/* Phase-1 catch-up: direct unit tests for the linalg helpers that
 * the Phase-4 RAII migration touched (chol, lu_L, lu_U, qr_Q, qr_R,
 * pinv) plus a few of the previously 0%-covered allocators called
 * out in docs/port_runtime_2_cpp.md (kron, ismember, intersect,
 * union, setdiff, ind2sub, sortrows, repmat, linspace, find,
 * horzcat, vertcat, permute, squeeze, slice1, slice2, matpow).
 *
 * Each migration in Phase 4 is paired here so the leak-free RAII path
 * gets exercised under the assertion suite. Run with ASan to catch
 * any regressions introduced by the migrations. */

#include "runtime_test.h"

matlab_mat *matlab_chol     (matlab_mat *A);
matlab_mat *matlab_pinv     (matlab_mat *A);
matlab_mat *matlab_lu_L     (matlab_mat *A);
matlab_mat *matlab_lu_U     (matlab_mat *A);
matlab_mat *matlab_qr_Q     (matlab_mat *A);
matlab_mat *matlab_qr_R     (matlab_mat *A);
matlab_mat *matlab_kron     (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_intersect(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_union    (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_setdiff  (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_repmat   (matlab_mat *A, double m, double n);
matlab_mat *matlab_linspace (double a, double b, double n);
matlab_mat *matlab_find     (matlab_mat *A);
matlab_mat *matlab_horzcat  (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_vertcat  (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_permute  (matlab_mat *A, matlab_mat *perm);
matlab_mat *matlab_squeeze  (matlab_mat *A);
matlab_mat *matlab_slice1   (matlab_mat *A, matlab_mat *idx);
matlab_mat *matlab_slice2   (matlab_mat *A, matlab_mat *rows, matlab_mat *cols);
matlab_mat *matlab_ind2sub  (matlab_mat *A, double idx);
matlab_mat *matlab_matpow   (matlab_mat *A, double n);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* --- chol ----------------------------------------------------------- */
static void test_chol_spd_2x2(void) {
    /* SPD: A = [4 12; 12 37], R'*R = A so R = [2 6; 0 1]. */
    double a[] = {4, 12, 12, 37};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *R = matlab_chol(A);
    RT_NEAR(rt_at(R, 0, 0), 2.0, 1e-10, "R[0,0]");
    RT_NEAR(rt_at(R, 0, 1), 6.0, 1e-10, "R[0,1]");
    RT_NEAR(rt_at(R, 1, 1), 1.0, 1e-10, "R[1,1]");
    RT_NEAR(rt_at(R, 1, 0), 0.0, 1e-12, "R[1,0]=0 (upper-tri)");
    rt_free(A); rt_free(R);
}
static void test_chol_not_spd_zeros(void) {
    /* A is symmetric but NOT positive-definite: produces zero matrix. */
    double a[] = {1, 2, 2, 1};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *R = matlab_chol(A);
    /* Runtime returns an all-zero matrix; we accept any all-zero shape. */
    int64_t total = rt_rows(R) * rt_cols(R);
    int all_zero = 1;
    for (int64_t k = 0; k < total; ++k)
        if (rt_data(R)[k] != 0.0) { all_zero = 0; break; }
    RT_CHECK(all_zero, "non-SPD result is zero matrix");
    rt_free(A); rt_free(R);
    matlab_clear_error();
}

/* --- LU decomposition ----------------------------------------------- */
static void test_lu_LU_factorisation(void) {
    /* A = [4 3; 6 3]. After partial pivoting, L*U = P*A. We simply
     * verify L is unit-lower-triangular and U is upper-triangular. */
    double a[] = {4, 3, 6, 3};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *L = matlab_lu_L(A);
    matlab_mat *U = matlab_lu_U(A);
    RT_NEAR(rt_at(L, 0, 0), 1.0, 1e-12, "L unit diag");
    RT_NEAR(rt_at(L, 1, 1), 1.0, 1e-12, "L unit diag");
    RT_NEAR(rt_at(L, 0, 1), 0.0, 1e-12, "L upper zero");
    RT_NEAR(rt_at(U, 1, 0), 0.0, 1e-12, "U lower zero");
    rt_free(A); rt_free(L); rt_free(U);
}

/* --- QR decomposition ------------------------------------------------ */
static void test_qr_orthonormal_columns(void) {
    /* Q from QR(A) — columns should be orthonormal. */
    double a[] = {1, 1, 1, 0, 1, 1, 0, 0, 1};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *Q = matlab_qr_Q(A);
    /* Q'*Q ≈ I: dot products should be 1 on diagonal, 0 off. */
    for (int j = 0; j < 3; ++j) {
        double sum = 0;
        for (int i = 0; i < 3; ++i) sum += rt_at(Q, i, j) * rt_at(Q, i, j);
        RT_NEAR(sum, 1.0, 1e-10, "Q col unit norm");
    }
    /* Off-diagonal dot product = 0. */
    double dot01 = 0;
    for (int i = 0; i < 3; ++i) dot01 += rt_at(Q, i, 0) * rt_at(Q, i, 1);
    RT_NEAR(dot01, 0.0, 1e-10, "Q cols orthogonal");
    rt_free(A); rt_free(Q);
}
static void test_qr_R_upper_triangular(void) {
    double a[] = {1, 1, 0, 1, 1, 1};
    matlab_mat *A = mk(a, 3, 2);
    matlab_mat *R = matlab_qr_R(A);
    RT_NEAR(rt_at(R, 1, 0), 0.0, 1e-12, "R lower zero");
    rt_free(A); rt_free(R);
}

/* --- pinv (square fast path) ---------------------------------------- */
static void test_pinv_2x2(void) {
    double a[] = {4, 7, 2, 6};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *P = matlab_pinv(A);
    RT_NEAR(rt_at(P, 0, 0),  0.6, 1e-10, "pinv[0,0]");
    RT_NEAR(rt_at(P, 1, 1),  0.4, 1e-10, "pinv[1,1]");
    rt_free(A); rt_free(P);
}

/* --- kron ----------------------------------------------------------- */
static void test_kron_2x2(void) {
    /* [1 2] (x) [3 4] = [3 4 6 8]. */
    double a[] = {1, 2};
    double b[] = {3, 4};
    matlab_mat *A = mk(a, 1, 2);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *K = matlab_kron(A, B);
    double expected[] = {3, 4, 6, 8};
    RT_CHECK(rt_rows(K) == 1 && rt_cols(K) == 4, "kron shape");
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(K)[i], expected[i], 1e-12, "kron values");
    rt_free(A); rt_free(B); rt_free(K);
}

/* --- set ops --------------------------------------------------------- */
static void test_intersect_basic(void) {
    double a[] = {1, 2, 3, 4};
    double b[] = {3, 4, 5, 6};
    matlab_mat *A = mk(a, 1, 4);
    matlab_mat *B = mk(b, 1, 4);
    matlab_mat *I = matlab_intersect(A, B);
    /* setdiff/intersect/union return column vectors, sorted. */
    int64_t n = rt_rows(I) * rt_cols(I);
    RT_CHECK(n == 2, "intersect size");
    RT_NEAR(rt_data(I)[0], 3.0, 0.0, "intersect[0]");
    RT_NEAR(rt_data(I)[1], 4.0, 0.0, "intersect[1]");
    rt_free(A); rt_free(B); rt_free(I);
}
static void test_union_basic(void) {
    double a[] = {1, 2, 3};
    double b[] = {3, 4, 5};
    matlab_mat *A = mk(a, 1, 3);
    matlab_mat *B = mk(b, 1, 3);
    matlab_mat *U = matlab_union(A, B);
    int64_t n = rt_rows(U) * rt_cols(U);
    RT_CHECK(n == 5, "union size");
    rt_free(A); rt_free(B); rt_free(U);
}
static void test_setdiff_basic(void) {
    double a[] = {1, 2, 3, 4, 5};
    double b[] = {2, 4};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *D = matlab_setdiff(A, B);
    int64_t n = rt_rows(D) * rt_cols(D);
    RT_CHECK(n == 3, "setdiff size");
    /* Sorted: [1 3 5]. */
    RT_NEAR(rt_data(D)[0], 1.0, 0.0, "setdiff[0]");
    RT_NEAR(rt_data(D)[1], 3.0, 0.0, "setdiff[1]");
    RT_NEAR(rt_data(D)[2], 5.0, 0.0, "setdiff[2]");
    rt_free(A); rt_free(B); rt_free(D);
}

/* --- repmat / linspace / find --------------------------------------- */
static void test_repmat_2x3(void) {
    double a[] = {1, 2};
    matlab_mat *A = mk(a, 1, 2);
    matlab_mat *R = matlab_repmat(A, 2, 3);
    RT_CHECK(rt_rows(R) == 2 && rt_cols(R) == 6, "repmat shape");
    /* Each row tiled 3x: [1 2 1 2 1 2]. */
    for (int j = 0; j < 6; ++j) {
        double e = (j % 2 == 0) ? 1.0 : 2.0;
        RT_NEAR(rt_at(R, 0, j), e, 0.0, "repmat row 0");
        RT_NEAR(rt_at(R, 1, j), e, 0.0, "repmat row 1");
    }
    rt_free(A); rt_free(R);
}
static void test_linspace_endpoints(void) {
    matlab_mat *L = matlab_linspace(0, 1, 5);
    RT_CHECK(rt_cols(L) == 5, "len");
    RT_NEAR(rt_at(L, 0, 0), 0.0, 1e-12, "start");
    RT_NEAR(rt_at(L, 0, 4), 1.0, 1e-12, "end");
    RT_NEAR(rt_at(L, 0, 2), 0.5, 1e-12, "midpoint");
    rt_free(L);
}
static void test_find_nonzero_indices(void) {
    /* find(v) returns 1-based indices of non-zero entries. */
    double a[] = {0, 1, 0, 2, 3};
    matlab_mat *A = mk(a, 1, 5);
    matlab_mat *F = matlab_find(A);
    int64_t n = rt_rows(F) * rt_cols(F);
    RT_CHECK(n == 3, "find count");
    RT_NEAR(rt_data(F)[0], 2.0, 0.0, "first nonzero");
    RT_NEAR(rt_data(F)[1], 4.0, 0.0, "second");
    RT_NEAR(rt_data(F)[2], 5.0, 0.0, "third");
    rt_free(A); rt_free(F);
}

/* --- concat / shape ops --------------------------------------------- */
static void test_horzcat(void) {
    double a[] = {1, 2};
    double b[] = {3, 4};
    matlab_mat *A = mk(a, 1, 2);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *C = matlab_horzcat(A, B);
    RT_CHECK(rt_cols(C) == 4 && rt_rows(C) == 1, "horzcat shape");
    RT_NEAR(rt_at(C, 0, 3), 4.0, 0.0, "last");
    rt_free(A); rt_free(B); rt_free(C);
}
static void test_vertcat(void) {
    double a[] = {1, 2};
    double b[] = {3, 4};
    matlab_mat *A = mk(a, 1, 2);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *C = matlab_vertcat(A, B);
    RT_CHECK(rt_cols(C) == 2 && rt_rows(C) == 2, "vertcat shape");
    RT_NEAR(rt_at(C, 1, 1), 4.0, 0.0, "C[1,1]");
    rt_free(A); rt_free(B); rt_free(C);
}
static void test_squeeze_2d(void) {
    /* squeeze on a 2-D matrix is a no-op. */
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *S = matlab_squeeze(A);
    RT_CHECK(rt_rows(S) == 2 && rt_cols(S) == 2, "shape unchanged");
    rt_free(A); rt_free(S);
}

/* --- subscripting --------------------------------------------------- */
static void test_slice2_submatrix(void) {
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    /* Pick rows {1, 2}, cols {1, 3} → [1 3; 4 6]. */
    double rd[] = {1, 2};
    double cd[] = {1, 3};
    matlab_mat *RR = mk(rd, 1, 2);
    matlab_mat *CC = mk(cd, 1, 2);
    matlab_mat *S = matlab_slice2(A, RR, CC);
    RT_CHECK(rt_rows(S) == 2 && rt_cols(S) == 2, "slice2 shape");
    RT_NEAR(rt_at(S, 0, 0), 1.0, 0.0, "S[0,0]");
    RT_NEAR(rt_at(S, 0, 1), 3.0, 0.0, "S[0,1]");
    RT_NEAR(rt_at(S, 1, 0), 4.0, 0.0, "S[1,0]");
    RT_NEAR(rt_at(S, 1, 1), 6.0, 0.0, "S[1,1]");
    rt_free(A); rt_free(RR); rt_free(CC); rt_free(S);
}

/* --- matpow --------------------------------------------------------- */
static void test_matpow_3(void) {
    /* A = [[1 1] [0 1]], A^3 = [[1 3] [0 1]]. */
    double a[] = {1, 1, 0, 1};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *P = matlab_matpow(A, 3);
    RT_NEAR(rt_at(P, 0, 0), 1.0, 1e-12, "P[0,0]");
    RT_NEAR(rt_at(P, 0, 1), 3.0, 1e-12, "P[0,1]");
    RT_NEAR(rt_at(P, 1, 0), 0.0, 1e-12, "P[1,0]");
    RT_NEAR(rt_at(P, 1, 1), 1.0, 1e-12, "P[1,1]");
    rt_free(A); rt_free(P);
}
static void test_matpow_zero_returns_eye(void) {
    /* A^0 = I for any invertible A. */
    double a[] = {2, 1, 1, 3};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *P = matlab_matpow(A, 0);
    RT_NEAR(rt_at(P, 0, 0), 1.0, 1e-12, "I[0,0]");
    RT_NEAR(rt_at(P, 1, 1), 1.0, 1e-12, "I[1,1]");
    RT_NEAR(rt_at(P, 0, 1), 0.0, 1e-12, "I[0,1]");
    rt_free(A); rt_free(P);
}

int main(void) {
    fprintf(stderr, "test_more:\n");
    RT_RUN(test_chol_spd_2x2);
    RT_RUN(test_chol_not_spd_zeros);
    RT_RUN(test_lu_LU_factorisation);
    RT_RUN(test_qr_orthonormal_columns);
    RT_RUN(test_qr_R_upper_triangular);
    RT_RUN(test_pinv_2x2);
    RT_RUN(test_kron_2x2);
    RT_RUN(test_intersect_basic);
    RT_RUN(test_union_basic);
    RT_RUN(test_setdiff_basic);
    RT_RUN(test_repmat_2x3);
    RT_RUN(test_linspace_endpoints);
    RT_RUN(test_find_nonzero_indices);
    RT_RUN(test_horzcat);
    RT_RUN(test_vertcat);
    RT_RUN(test_squeeze_2d);
    RT_RUN(test_slice2_submatrix);
    RT_RUN(test_matpow_3);
    RT_RUN(test_matpow_zero_returns_eye);
    RT_DONE();
}
