/* Direct unit tests for the linear-algebra entries in
 * runtime/matlab_runtime.c. Independent of the MATLAB frontend — links
 * the runtime translation unit directly. */

#include "runtime_test.h"

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

static void test_matmul_identity_roundtrip(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *I = matlab_eye(2, 2);
    matlab_mat *C = matlab_matmul_mm(A, I);
    RT_NEAR(rt_at(C, 0, 0), 1.0, 1e-12, "A*I[0,0]");
    RT_NEAR(rt_at(C, 0, 1), 2.0, 1e-12, "A*I[0,1]");
    RT_NEAR(rt_at(C, 1, 0), 3.0, 1e-12, "A*I[1,0]");
    RT_NEAR(rt_at(C, 1, 1), 4.0, 1e-12, "A*I[1,1]");
    rt_free(A); rt_free(I); rt_free(C);
}

static void test_matmul_known_3x3(void) {
    /* A = [1 2 3; 4 5 6; 7 8 10],  B = [1 0 0; 0 2 0; 0 0 3]
     * A*B = [1 4 9; 4 10 18; 7 16 30] */
    double a[] = {1,2,3, 4,5,6, 7,8,10};
    double b[] = {1,0,0, 0,2,0, 0,0,3};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *B = mk(b, 3, 3);
    matlab_mat *C = matlab_matmul_mm(A, B);
    double expected[] = {1,4,9, 4,10,18, 7,16,30};
    for (int i = 0; i < 9; ++i)
        RT_NEAR(rt_data(C)[i], expected[i], 1e-12, "matmul 3x3");
    rt_free(A); rt_free(B); rt_free(C);
}

static void test_inv_2x2(void) {
    double a[] = {4, 7, 2, 6};            /* det = 10 */
    matlab_mat *A   = mk(a, 2, 2);
    matlab_mat *Ai  = matlab_inv(A);
    /* inv = [0.6 -0.7; -0.2 0.4] */
    RT_NEAR(rt_at(Ai, 0, 0),  0.6, 1e-12, "inv[0,0]");
    RT_NEAR(rt_at(Ai, 0, 1), -0.7, 1e-12, "inv[0,1]");
    RT_NEAR(rt_at(Ai, 1, 0), -0.2, 1e-12, "inv[1,0]");
    RT_NEAR(rt_at(Ai, 1, 1),  0.4, 1e-12, "inv[1,1]");
    /* A * inv(A) ≈ I */
    matlab_mat *I = matlab_matmul_mm(A, Ai);
    RT_NEAR(rt_at(I, 0, 0), 1.0, 1e-12, "A*inv(A)[0,0]");
    RT_NEAR(rt_at(I, 0, 1), 0.0, 1e-12, "A*inv(A)[0,1]");
    RT_NEAR(rt_at(I, 1, 0), 0.0, 1e-12, "A*inv(A)[1,0]");
    RT_NEAR(rt_at(I, 1, 1), 1.0, 1e-12, "A*inv(A)[1,1]");
    rt_free(A); rt_free(Ai); rt_free(I);
}

static void test_inv_3x3(void) {
    /* well-conditioned 3x3 */
    double a[] = {1,2,3, 0,1,4, 5,6,0};
    matlab_mat *A  = mk(a, 3, 3);
    matlab_mat *Ai = matlab_inv(A);
    matlab_mat *I  = matlab_matmul_mm(A, Ai);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(I, i, j), (i == j) ? 1.0 : 0.0, 1e-9,
                    "A*inv(A) ≈ I");
    rt_free(A); rt_free(Ai); rt_free(I);
}

static void test_det_known(void) {
    /* det([1 2; 3 4]) = -2, det([4 7; 2 6]) = 10 */
    double a[] = {1, 2, 3, 4};
    double b[] = {4, 7, 2, 6};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *B = mk(b, 2, 2);
    RT_NEAR(matlab_det(A), -2.0, 1e-12, "det 2x2 #1");
    RT_NEAR(matlab_det(B), 10.0, 1e-12, "det 2x2 #2");
    rt_free(A); rt_free(B);
}

static void test_det_3x3_singular(void) {
    /* rows are linearly dependent → det = 0 */
    double a[] = {1,2,3, 2,4,6, 1,1,1};
    matlab_mat *A = mk(a, 3, 3);
    RT_NEAR(matlab_det(A), 0.0, 1e-9, "det singular 3x3");
    rt_free(A);
}

static void test_mldivide_solves_Ax_eq_b(void) {
    /* A = [1 2; 3 4], b = [5; 11], x = A\b = [1; 2] */
    double a[] = {1, 2, 3, 4};
    double b[] = {5, 11};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *B = mk(b, 2, 1);
    matlab_mat *X = matlab_mldivide_mm(A, B);
    RT_CHECK(rt_rows(X) == 2 && rt_cols(X) == 1, "mldivide shape");
    RT_NEAR(rt_at(X, 0, 0), 1.0, 1e-9, "mldivide x0");
    RT_NEAR(rt_at(X, 1, 0), 2.0, 1e-9, "mldivide x1");
    rt_free(A); rt_free(B); rt_free(X);
}

static void test_mrdivide_solves_xA_eq_b(void) {
    /* x*A = b  →  x = b/A.  A = [1 2; 3 4], b = [5 11] (1x2),
     * inv(A) = [-2 1; 1.5 -0.5], x = b * inv(A) = [6.5 -0.5].
     * Verify: x*A = [6.5 -0.5] * [1 2; 3 4] = [5 11] ✓ */
    double a[] = {1, 2, 3, 4};
    double b[] = {5, 11};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *B = mk(b, 1, 2);
    matlab_mat *X = matlab_mrdivide_mm(B, A);
    RT_CHECK(rt_rows(X) == 1 && rt_cols(X) == 2, "mrdivide shape");
    RT_NEAR(rt_at(X, 0, 0),  6.5, 1e-9, "mrdivide x0");
    RT_NEAR(rt_at(X, 0, 1), -0.5, 1e-9, "mrdivide x1");
    /* And confirm x*A reconstructs b. */
    matlab_mat *XA = matlab_matmul_mm(X, A);
    RT_NEAR(rt_at(XA, 0, 0),  5.0, 1e-9, "x*A == b[0]");
    RT_NEAR(rt_at(XA, 0, 1), 11.0, 1e-9, "x*A == b[1]");
    rt_free(XA);
    rt_free(A); rt_free(B); rt_free(X);
}

static void test_transpose(void) {
    double a[] = {1,2,3, 4,5,6};            /* 2x3 */
    matlab_mat *A  = mk(a, 2, 3);
    matlab_mat *At = matlab_transpose(A);
    RT_CHECK(rt_rows(At) == 3 && rt_cols(At) == 2, "transpose shape");
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(At, j, i), rt_at(A, i, j), 0.0, "(A')_ji = A_ij");
    rt_free(A); rt_free(At);
}

static void test_diag_extracts(void) {
    double a[] = {1,2,3, 4,5,6, 7,8,9};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *d = matlab_diag(A);
    RT_CHECK(rt_rows(d) == 3 && rt_cols(d) == 1, "diag shape (column)");
    RT_NEAR(rt_at(d, 0, 0), 1.0, 0.0, "diag[0]");
    RT_NEAR(rt_at(d, 1, 0), 5.0, 0.0, "diag[1]");
    RT_NEAR(rt_at(d, 2, 0), 9.0, 0.0, "diag[2]");
    rt_free(A); rt_free(d);
}

static void test_eig_symmetric(void) {
    /* eig of [2 1; 1 2] = {1, 3} (order may vary); trace=4, det=3 */
    double a[] = {2, 1, 1, 2};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *e = matlab_eig(A);
    RT_CHECK(rt_rows(e) * rt_cols(e) == 2, "eig returns 2 values");
    double e0 = rt_data(e)[0], e1 = rt_data(e)[1];
    double sum = e0 + e1, prod = e0 * e1;
    RT_NEAR(sum,  4.0, 1e-9, "eig sum == trace");
    RT_NEAR(prod, 3.0, 1e-9, "eig prod == det");
    rt_free(A); rt_free(e);
}

static void test_eig_V_D_reconstructs_A(void) {
    /* For symmetric A, V*D*V' ≈ A. */
    double a[] = {2, 1, 1, 2};
    matlab_mat *A  = mk(a, 2, 2);
    matlab_mat *V  = matlab_eig_V(A);
    matlab_mat *D  = matlab_eig_D(A);
    matlab_mat *Vt = matlab_transpose(V);
    matlab_mat *VD = matlab_matmul_mm(V, D);
    matlab_mat *R  = matlab_matmul_mm(VD, Vt);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            RT_NEAR(rt_at(R, i, j), rt_at(A, i, j), 1e-9,
                    "V*D*V' ≈ A");
    rt_free(A); rt_free(V); rt_free(D); rt_free(Vt);
    rt_free(VD); rt_free(R);
}

static void test_svd_singular_values(void) {
    /* SVD of diag([3, 1]) returns singular values {3, 1}. */
    double a[] = {3, 0, 0, 1};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *s = matlab_svd(A);
    RT_CHECK(rt_rows(s) * rt_cols(s) == 2, "svd vector length");
    /* Order: descending by convention. */
    RT_NEAR(rt_data(s)[0], 3.0, 1e-9, "sigma_0");
    RT_NEAR(rt_data(s)[1], 1.0, 1e-9, "sigma_1");
    rt_free(A); rt_free(s);
}

static void test_zeros_ones_eye_magic(void) {
    matlab_mat *Z = matlab_zeros(2, 3);
    matlab_mat *O = matlab_ones(2, 3);
    matlab_mat *I = matlab_eye(3, 3);
    matlab_mat *M = matlab_magic(3);
    RT_CHECK(rt_rows(Z) == 2 && rt_cols(Z) == 3, "zeros shape");
    for (int k = 0; k < 6; ++k) RT_NEAR(rt_data(Z)[k], 0.0, 0.0, "zeros");
    for (int k = 0; k < 6; ++k) RT_NEAR(rt_data(O)[k], 1.0, 0.0, "ones");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(I, i, j), (i == j) ? 1.0 : 0.0, 0.0, "eye");
    /* magic(3): every row/col sums to 15. */
    for (int i = 0; i < 3; ++i) {
        double rs = rt_at(M,i,0) + rt_at(M,i,1) + rt_at(M,i,2);
        double cs = rt_at(M,0,i) + rt_at(M,1,i) + rt_at(M,2,i);
        RT_NEAR(rs, 15.0, 0.0, "magic row sum");
        RT_NEAR(cs, 15.0, 0.0, "magic col sum");
    }
    rt_free(Z); rt_free(O); rt_free(I); rt_free(M);
}

int main(void) {
    fprintf(stderr, "test_linalg:\n");
    RT_RUN(test_zeros_ones_eye_magic);
    RT_RUN(test_matmul_identity_roundtrip);
    RT_RUN(test_matmul_known_3x3);
    RT_RUN(test_inv_2x2);
    RT_RUN(test_inv_3x3);
    RT_RUN(test_det_known);
    RT_RUN(test_det_3x3_singular);
    RT_RUN(test_mldivide_solves_Ax_eq_b);
    RT_RUN(test_mrdivide_solves_xA_eq_b);
    RT_RUN(test_transpose);
    RT_RUN(test_diag_extracts);
    RT_RUN(test_eig_symmetric);
    RT_RUN(test_eig_V_D_reconstructs_A);
    RT_RUN(test_svd_singular_values);
    RT_DONE();
}
