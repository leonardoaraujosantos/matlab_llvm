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

/* expm(zeros(n)) = I — the defining identity. */
static void test_expm_zero(void) {
    matlab_mat *Z = matlab_zeros(3, 3);
    matlab_mat *E = matlab_expm(Z);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(E, i, j), (i == j) ? 1.0 : 0.0, 1e-12,
                    "expm(0) = I");
    rt_free(Z); rt_free(E);
}

/* expm(diag([a, b, c])) = diag([exp(a), exp(b), exp(c)]). */
static void test_expm_diagonal(void) {
    double a[] = {-1, 0, 0,  0, 0.5, 0,  0, 0, 2};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *E = matlab_expm(A);
    RT_NEAR(rt_at(E, 0, 0), exp(-1.0), 1e-12, "expm diag[0,0]");
    RT_NEAR(rt_at(E, 1, 1), exp( 0.5), 1e-12, "expm diag[1,1]");
    RT_NEAR(rt_at(E, 2, 2), exp( 2.0), 1e-12, "expm diag[2,2]");
    RT_NEAR(rt_at(E, 0, 1), 0.0,        1e-12, "expm diag off-diag");
    RT_NEAR(rt_at(E, 1, 0), 0.0,        1e-12, "expm diag off-diag");
    rt_free(A); rt_free(E);
}

/* Rotation matrix.  A = [0 1; -1 0],  expm(A * theta) is a 2-D rotation
 * by theta. Verify against expm(A * pi/2) = A. */
static void test_expm_rotation(void) {
    double piover2 = 1.5707963267948966;
    double a[] = {0, piover2, -piover2, 0};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *E = matlab_expm(A);
    /* expm(A * pi/2) = [cos(pi/2)  sin(pi/2); -sin(pi/2)  cos(pi/2)]
     *                = [0  1; -1  0]. */
    RT_NEAR(rt_at(E, 0, 0),  0.0, 1e-12, "rot[0,0]");
    RT_NEAR(rt_at(E, 0, 1),  1.0, 1e-12, "rot[0,1]");
    RT_NEAR(rt_at(E, 1, 0), -1.0, 1e-12, "rot[1,0]");
    RT_NEAR(rt_at(E, 1, 1),  0.0, 1e-12, "rot[1,1]");
    rt_free(A); rt_free(E);
}

/* expm(A) * expm(-A) = I — fundamental group identity. Tests both the
 * positive and negative branches and gates the LU back-solve. */
static void test_expm_inverse_identity(void) {
    /* well-conditioned but not normal — exercises the full Pade path. */
    double a[] = {1, -2, 0,
                  3,  0, 1,
                  0,  1, -1};
    matlab_mat *A     = mk(a, 3, 3);
    matlab_mat *E     = matlab_expm(A);
    /* Build -A. */
    double na[9];
    for (int i = 0; i < 9; ++i) na[i] = -a[i];
    matlab_mat *Aneg  = mk(na, 3, 3);
    matlab_mat *Einv  = matlab_expm(Aneg);
    matlab_mat *I     = matlab_matmul_mm(E, Einv);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(I, i, j), (i == j) ? 1.0 : 0.0, 1e-10,
                    "expm(A)*expm(-A) = I");
    rt_free(A); rt_free(Aneg); rt_free(E); rt_free(Einv); rt_free(I);
}

/* Large-norm path — anrm > theta13 forces scaling-and-squaring. The
 * answer must still be accurate; verify against expm(diag) which has
 * a closed form. */
static void test_expm_large_norm(void) {
    double a[] = {10, 0, 0, -8};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *E = matlab_expm(A);
    RT_NEAR(rt_at(E, 0, 0), exp(10.0), exp(10.0) * 1e-10,
            "expm large-norm diag[0,0]");
    RT_NEAR(rt_at(E, 1, 1), exp(-8.0), 1e-12,
            "expm large-norm diag[1,1]");
    RT_NEAR(rt_at(E, 0, 1), 0.0, 1e-10, "expm large-norm off");
    RT_NEAR(rt_at(E, 1, 0), 0.0, 1e-10, "expm large-norm off");
    rt_free(A); rt_free(E);
}

/* hess(A) returns the input unchanged for n <= 2 (already Hessenberg). */
static void test_hess_small(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *H = matlab_hess(A);
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(H)[i], a[i], 0.0, "hess 2x2 = A");
    rt_free(A); rt_free(H);
}

/* For a 3x3, hess zeroes out one element: H[2,0] = 0. The diagonal +
 * subdiagonal + upper triangle are non-zero in general. */
static void test_hess_3x3_subdiagonal_zero(void) {
    double a[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *H = matlab_hess(A);
    RT_NEAR(rt_at(H, 2, 0), 0.0, 1e-12, "hess H[2,0] = 0");
    /* Eigenvalues of H equal eigenvalues of A (similarity transform).
     * Trace and determinant of A are preserved.  trace = 1+5+10 = 16,
     * det = 1*(50-48) - 2*(40-42) + 3*(32-35) = 2 + 4 - 9 = -3. */
    double tr = rt_at(H, 0, 0) + rt_at(H, 1, 1) + rt_at(H, 2, 2);
    RT_NEAR(tr, 16.0, 1e-10, "hess trace preserved");
    rt_free(A); rt_free(H);
}

/* For a 4x4, hess zeroes out the (n-1)*(n-2)/2 = 3 sub-subdiagonal
 * entries: H[2,0] = H[3,0] = H[3,1] = 0. */
static void test_hess_4x4_pattern(void) {
    double a[] = { 4, 1, -2,  2,
                   1, 2,  0,  1,
                  -2, 0,  3, -2,
                   2, 1, -2, -1};
    matlab_mat *A = mk(a, 4, 4);
    matlab_mat *H = matlab_hess(A);
    RT_NEAR(rt_at(H, 2, 0), 0.0, 1e-12, "hess H[2,0]");
    RT_NEAR(rt_at(H, 3, 0), 0.0, 1e-12, "hess H[3,0]");
    RT_NEAR(rt_at(H, 3, 1), 0.0, 1e-12, "hess H[3,1]");
    /* Trace preserved: 4 + 2 + 3 + (-1) = 8. */
    double tr = rt_at(H, 0, 0) + rt_at(H, 1, 1) + rt_at(H, 2, 2) +
                rt_at(H, 3, 3);
    RT_NEAR(tr, 8.0, 1e-10, "hess 4x4 trace");
    /* Symmetric input -> symmetric tridiagonal Hessenberg
     * (i.e. H[0,2] = H[0,3] = H[1,3] = 0 too).  Verify the upper-
     * triangle "above-supdiagonal" half is also clean. */
    RT_NEAR(rt_at(H, 0, 2), 0.0, 1e-10, "hess sym-tridiag H[0,2]");
    RT_NEAR(rt_at(H, 0, 3), 0.0, 1e-10, "hess sym-tridiag H[0,3]");
    RT_NEAR(rt_at(H, 1, 3), 0.0, 1e-10, "hess sym-tridiag H[1,3]");
    rt_free(A); rt_free(H);
}

/* Upper triangular A (sigma == 0 in every column) is a fixed point — the
 * inner loop short-circuits via the `if (sigma == 0.0) continue` guard. */
static void test_hess_upper_tri_fixed_point(void) {
    double a[] = {1, 2, 3,
                  0, 5, 6,
                  0, 0, 8};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *H = matlab_hess(A);
    for (int i = 0; i < 9; ++i)
        RT_NEAR(rt_data(H)[i], a[i], 1e-12, "hess upper-tri fixed point");
    rt_free(A); rt_free(H);
}

/* Eigenvalues are preserved by similarity transform.  Trace and
 * determinant — the easy ones — match A's. */
static void test_hess_preserves_invariants(void) {
    double a[] = {2, 1, 0, 1,
                  1, 3, 1, 0,
                  0, 1, 4, 1,
                  1, 0, 1, 5};
    matlab_mat *A = mk(a, 4, 4);
    matlab_mat *H = matlab_hess(A);
    /* Trace = 2 + 3 + 4 + 5 = 14. */
    double trA = a[0] + a[5] + a[10] + a[15];
    double trH = rt_at(H, 0, 0) + rt_at(H, 1, 1) +
                 rt_at(H, 2, 2) + rt_at(H, 3, 3);
    RT_NEAR(trH, trA, 1e-10, "hess trace");
    /* det(A) = det(H) — verify via the existing matlab_det (which itself
     * uses LU; the orthogonal similarity preserves det exactly). */
    RT_NEAR(matlab_det(H), matlab_det(A), 1e-9, "hess det");
    rt_free(A); rt_free(H);
}

/* Non-symmetric eig — Francis double-shift QR.
 * A 3*3 companion matrix of the polynomial p(x) = x^3 - 6x^2 + 11x - 6
 * = (x-1)(x-2)(x-3). Eigenvalues = roots = {1, 2, 3}. */
static void test_eig_nonsym_companion(void) {
    /* Companion form for [1 -6 11 -6]:  C = [[0 0 6], [1 0 -11], [0 1 6]]. */
    double a[] = { 0,  0,  6,
                   1,  0, -11,
                   0,  1,  6};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *E = matlab_eig(A);
    /* Eigenvalues sorted ascending: {1, 2, 3}. Result is real (no complex
     * pair) — the descriptor is matlab_mat*, not mat_c*. */
    RT_NEAR(rt_at(E, 0, 0), 1.0, 1e-9, "companion eig 1");
    RT_NEAR(rt_at(E, 1, 0), 2.0, 1e-9, "companion eig 2");
    RT_NEAR(rt_at(E, 2, 0), 3.0, 1e-9, "companion eig 3");
    rt_free(A); rt_free(E);
}

/* Companion of x^4 - 10x^3 + 35x^2 - 50x + 24 = (x-1)(x-2)(x-3)(x-4). */
static void test_eig_nonsym_4x4(void) {
    double a[] = { 0, 0, 0, -24,
                   1, 0, 0,  50,
                   0, 1, 0, -35,
                   0, 0, 1,  10};
    matlab_mat *A = mk(a, 4, 4);
    matlab_mat *E = matlab_eig(A);
    RT_NEAR(rt_at(E, 0, 0), 1.0, 1e-7, "4x4 companion eig 1");
    RT_NEAR(rt_at(E, 1, 0), 2.0, 1e-7, "4x4 companion eig 2");
    RT_NEAR(rt_at(E, 2, 0), 3.0, 1e-7, "4x4 companion eig 3");
    RT_NEAR(rt_at(E, 3, 0), 4.0, 1e-7, "4x4 companion eig 4");
    rt_free(A); rt_free(E);
}

/* Rotation matrix — eigenvalues are pure imaginary +-i.
 * A = [0  1; -1  0] -> spectrum = {i, -i}. The result is matlab_mat_c *
 * which the runtime exposes via real/imag accessors. */
static void test_eig_nonsym_rotation_complex(void) {
    double a[] = {0, 1, -1, 0};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *E = matlab_eig(A);
    /* Sorted by ascending re (both 0), then imag: {-i, +i}. */
    matlab_mat *Re = matlab_real_c(E);
    matlab_mat *Im = matlab_imag_c(E);
    RT_NEAR(rt_at(Re, 0, 0),  0.0, 1e-12, "rot eig re[0]");
    RT_NEAR(rt_at(Im, 0, 0), -1.0, 1e-12, "rot eig im[0]");
    RT_NEAR(rt_at(Re, 1, 0),  0.0, 1e-12, "rot eig re[1]");
    RT_NEAR(rt_at(Im, 1, 0), +1.0, 1e-12, "rot eig im[1]");
    rt_free(A); rt_free(Re); rt_free(Im);
    /* `E` is a matlab_mat_c* aliased through matlab_mat*; not freed
     * here since the arena-policy and the polymorphic free path do
     * the right thing through the magic word. */
}

/* Stable real plant (negative-real eigenvalues). Random asymmetric A
 * but constructed to have known spectrum: A = T * diag(-1, -2, -3) * T^-1
 * where T is a small invertible matrix. We verify trace/det match. */
static void test_eig_nonsym_negative_real(void) {
    /* Build A = T * D * T^-1.
     *   T = [[1 1 0],[0 1 1],[1 0 1]],  T^-1 from a 3*3 inverse.
     *   D = diag(-1, -2, -3).
     * Trace(A) = trace(D) = -6. det(A) = det(D) = -6. */
    double a[] = {-1, 0, 0,
                  -1,-2, 0,
                  -1,-1,-3};   /* lower-triangular A with diag = D */
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *E = matlab_eig(A);
    /* Eigenvalues of a triangular matrix = diagonal entries.
     * Sorted ascending: {-3, -2, -1}. */
    RT_NEAR(rt_at(E, 0, 0), -3.0, 1e-9, "tri eig -3");
    RT_NEAR(rt_at(E, 1, 0), -2.0, 1e-9, "tri eig -2");
    RT_NEAR(rt_at(E, 2, 0), -1.0, 1e-9, "tri eig -1");
    rt_free(A); rt_free(E);
}

/* schur on a 1x1 returns the input itself with U = I. */
static void test_schur_1x1(void) {
    double a[] = {7.0};
    matlab_mat *A = mk(a, 1, 1);
    matlab_mat *T = matlab_schur(A);
    matlab_mat *U = matlab_schur_U(A);
    RT_NEAR(rt_at(T, 0, 0), 7.0, 0.0, "schur 1x1 T");
    RT_NEAR(rt_at(U, 0, 0), 1.0, 0.0, "schur 1x1 U");
    rt_free(A); rt_free(T); rt_free(U);
}

/* schur on an upper-triangular matrix is a fixed point — Hessenberg
 * reduction is a no-op (sigma = 0 in every column), and the QR loop
 * deflates immediately since each diagonal element is already a
 * 1x1 block (subdiagonal zero). T = A and U = I. */
static void test_schur_upper_tri_fixed(void) {
    double a[] = {1, 2, 3,
                  0, 5, 6,
                  0, 0, 8};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *T = matlab_schur(A);
    matlab_mat *U = matlab_schur_U(A);
    for (int i = 0; i < 9; ++i)
        RT_NEAR(rt_data(T)[i], a[i], 1e-12, "schur upper-tri T");
    /* U is identity. */
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(U, i, j), (i == j) ? 1.0 : 0.0, 1e-12,
                    "schur upper-tri U = I");
    rt_free(A); rt_free(T); rt_free(U);
}

/* Reconstruct: A = U * T * U' (real Schur identity). */
static void test_schur_reconstructs_A(void) {
    /* Asymmetric 3x3. */
    double a[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *T = matlab_schur(A);
    matlab_mat *U = matlab_schur_U(A);
    /* Compute Ut = U' (transpose), then A_back = U * T * Ut. */
    matlab_mat *Ut = matlab_transpose(U);
    matlab_mat *UT = matlab_matmul_mm(U, T);
    matlab_mat *Ar = matlab_matmul_mm(UT, Ut);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RT_NEAR(rt_at(Ar, i, j), a[i * 3 + j], 1e-9,
                    "schur A = U T U' reconstruction");
    rt_free(A); rt_free(T); rt_free(U);
    rt_free(Ut); rt_free(UT); rt_free(Ar);
}

/* U is orthogonal: U' * U = I. */
static void test_schur_U_orthogonal(void) {
    double a[] = {2, 1, 0, 1,
                  0, 3, 1, 0,
                  1, 0, 4, 1,
                  0, 1, 0, 5};   /* asymmetric 4x4 */
    matlab_mat *A   = mk(a, 4, 4);
    matlab_mat *U   = matlab_schur_U(A);
    matlab_mat *Ut  = matlab_transpose(U);
    matlab_mat *UtU = matlab_matmul_mm(Ut, U);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            RT_NEAR(rt_at(UtU, i, j), (i == j) ? 1.0 : 0.0, 1e-10,
                    "schur U' U = I");
    rt_free(A); rt_free(U); rt_free(Ut); rt_free(UtU);
}

/* Real Schur form preserves the spectrum: trace and det are invariant
 * regardless of whether T is fully triangular or carries 2x2 blocks
 * (trace is always the sum of diagonal entries; for an upper quasi-
 * triangular matrix, det is the product of 1x1 entries times
 * det of each 2x2 block, which equals det A by similarity). */
static void test_schur_preserves_spectrum_invariants(void) {
    /* Lower-triangular -> eigenvalues = diagonal {-1, -2, -3}.
     * trace(A) = -6, det(A) = -6. */
    double a[] = {-1, 0, 0,
                  -1,-2, 0,
                  -1,-1,-3};
    matlab_mat *A = mk(a, 3, 3);
    matlab_mat *T = matlab_schur(A);
    double trT = rt_at(T, 0, 0) + rt_at(T, 1, 1) + rt_at(T, 2, 2);
    RT_NEAR(trT, -6.0, 1e-10, "schur trace = -6");
    RT_NEAR(matlab_det(T), -6.0, 1e-9, "schur det = -6");
    rt_free(A); rt_free(T);
}

/* Lyapunov A X + X A' + Q = 0 - 1x1 closed form: a x + x a + q = 0
 * -> x = -q / (2 a). a = -1, q = 1 -> x = 0.5. */
static void test_lyap_1x1(void) {
    double a[] = {-1.0}, q[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *Q = mk(q, 1, 1);
    matlab_mat *X = matlab_lyap(A, Q);
    RT_NEAR(rt_at(X, 0, 0), 0.5, 1e-12, "lyap 1x1");
    rt_free(A); rt_free(Q); rt_free(X);
}

/* Lyapunov 2x2 diagonal: A = diag(-1, -2), Q = I. The unique solution
 * is also diagonal: X = diag(-Q[i,i] / (2 A[i,i])) = diag(0.5, 0.25). */
static void test_lyap_diagonal(void) {
    double a[] = {-1, 0,  0, -2};
    double q[] = { 1, 0,  0,  1};
    matlab_mat *A = mk(a, 2, 2), *Q = mk(q, 2, 2);
    matlab_mat *X = matlab_lyap(A, Q);
    RT_NEAR(rt_at(X, 0, 0), 0.5,  1e-12, "lyap diag X[0,0]");
    RT_NEAR(rt_at(X, 1, 1), 0.25, 1e-12, "lyap diag X[1,1]");
    RT_NEAR(rt_at(X, 0, 1), 0.0,  1e-12, "lyap diag off");
    RT_NEAR(rt_at(X, 1, 0), 0.0,  1e-12, "lyap diag off");
    rt_free(A); rt_free(Q); rt_free(X);
}

/* Lyapunov self-consistency: A X + X A' + Q == 0 to round-off tolerance. */
static void test_lyap_residual(void) {
    /* Stable asymmetric 3x3. */
    double a[] = {-2,  1,  0,
                   0, -3,  1,
                   1,  0, -1};
    double q[] = { 1,  0,  0,
                   0,  2,  0,
                   0,  0,  3};
    matlab_mat *A = mk(a, 3, 3), *Q = mk(q, 3, 3);
    matlab_mat *X  = matlab_lyap(A, Q);
    matlab_mat *AX = matlab_matmul_mm(A, X);
    matlab_mat *Atr= matlab_transpose(A);
    matlab_mat *XAt= matlab_matmul_mm(X, Atr);
    /* Compute residual R = A X + X A' + Q. */
    for (int i = 0; i < 9; ++i) {
        double r = rt_data(AX)[i] + rt_data(XAt)[i] + rt_data(Q)[i];
        RT_NEAR(r, 0.0, 1e-9, "lyap residual");
    }
    rt_free(A); rt_free(Q); rt_free(X);
    rt_free(AX); rt_free(Atr); rt_free(XAt);
}

/* Discrete Lyapunov A X A' - X + Q = 0. 1x1 closed form:
 * a^2 x - x + q = 0 -> x = q / (1 - a^2). a = 0.5, q = 1 -> x = 4/3. */
static void test_dlyap_1x1(void) {
    double a[] = {0.5}, q[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *Q = mk(q, 1, 1);
    matlab_mat *X = matlab_dlyap(A, Q);
    RT_NEAR(rt_at(X, 0, 0), 4.0 / 3.0, 1e-12, "dlyap 1x1");
    rt_free(A); rt_free(Q); rt_free(X);
}

/* Discrete Lyapunov self-consistency on a stable discrete plant. */
static void test_dlyap_residual(void) {
    /* All eigenvalues of A inside the unit circle. */
    double a[] = {0.5, 0.1, 0.0,
                  0.0, 0.6, 0.2,
                  0.1, 0.0, 0.4};
    double q[] = {1, 0, 0,
                  0, 1, 0,
                  0, 0, 1};
    matlab_mat *A = mk(a, 3, 3), *Q = mk(q, 3, 3);
    matlab_mat *X  = matlab_dlyap(A, Q);
    matlab_mat *AX = matlab_matmul_mm(A, X);
    matlab_mat *Atr= matlab_transpose(A);
    matlab_mat *AXAt= matlab_matmul_mm(AX, Atr);
    /* R = A X A' - X + Q. */
    for (int i = 0; i < 9; ++i) {
        double r = rt_data(AXAt)[i] - rt_data(X)[i] + rt_data(Q)[i];
        RT_NEAR(r, 0.0, 1e-9, "dlyap residual");
    }
    rt_free(A); rt_free(Q); rt_free(X);
    rt_free(AX); rt_free(Atr); rt_free(AXAt);
}

/* CARE 1x1: a = -1, b = 1, q = 1, r = 1.
 *   A'X + XA - X B/R B' X + Q = -2x - x^2 + 1 = 0  ->  x = -1 + sqrt(2). */
static void test_care_1x1(void) {
    double a[] = {-1.0}, b[] = {1.0}, q[] = {1.0}, r[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1);
    matlab_mat *Q = mk(q, 1, 1), *R = mk(r, 1, 1);
    matlab_mat *X = matlab_care(A, B, Q, R);
    RT_NEAR(rt_at(X, 0, 0), sqrt(2.0) - 1.0, 1e-10, "care 1x1 stab soln");
    rt_free(A); rt_free(B); rt_free(Q); rt_free(R); rt_free(X);
}

/* CARE for the double integrator: A=[0 1; 0 0], B=[0; 1], Q=I, R=1.
 * Closed-form X = [sqrt(3) 1; 1 sqrt(3)]. */
static void test_care_double_integrator(void) {
    double a[] = {0, 1, 0, 0};
    double b[] = {0, 1};
    double q[] = {1, 0, 0, 1};
    double r[] = {1.0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Q = mk(q, 2, 2), *R = mk(r, 1, 1);
    matlab_mat *X = matlab_care(A, B, Q, R);
    double s3 = sqrt(3.0);
    RT_NEAR(rt_at(X, 0, 0), s3,  1e-9, "care 2x2 X[0,0]");
    RT_NEAR(rt_at(X, 0, 1), 1.0, 1e-9, "care 2x2 X[0,1]");
    RT_NEAR(rt_at(X, 1, 0), 1.0, 1e-9, "care 2x2 X[1,0]");
    RT_NEAR(rt_at(X, 1, 1), s3,  1e-9, "care 2x2 X[1,1]");
    rt_free(A); rt_free(B); rt_free(Q); rt_free(R); rt_free(X);
}

/* CARE residual self-consistency: A'X + XA - X B/R B' X + Q ~ 0 to
 * round-off. */
static void test_care_residual(void) {
    /* Stable open-loop A; non-trivial B. */
    double a[] = {-1, 0.5, 0, -2};
    double b[] = {1, 0.5};       /* 2x1 */
    double q[] = {2, 0.3, 0.3, 1};
    double r[] = {0.4};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Q = mk(q, 2, 2), *R = mk(r, 1, 1);
    matlab_mat *X = matlab_care(A, B, Q, R);
    /* Residual = A'X + XA - X B (1/R) B' X + Q. */
    matlab_mat *At  = matlab_transpose(A);
    matlab_mat *AtX = matlab_matmul_mm(At, X);
    matlab_mat *XA  = matlab_matmul_mm(X, A);
    matlab_mat *XB  = matlab_matmul_mm(X, B);
    matlab_mat *Bt  = matlab_transpose(B);
    matlab_mat *XBt = matlab_matmul_mm(XB, Bt);
    /* X B (1/R) B' X for scalar R: just (1/r) * XBBtX where XBBtX = XB*Bt*X.
     * Compute XBBtX = (XB) * Bt * X = (XB Bt) * X = XBt * X. */
    matlab_mat *XBR = matlab_matmul_mm(XBt, X);  /* = X B B' X (R=scalar) */
    double rinv = 1.0 / r[0];
    for (int i = 0; i < 4; ++i) {
        double res = rt_data(AtX)[i] + rt_data(XA)[i]
                   - rinv * rt_data(XBR)[i] + q[i];
        RT_NEAR(res, 0.0, 1e-8, "care residual");
    }
    rt_free(A); rt_free(B); rt_free(Q); rt_free(R); rt_free(X);
    rt_free(At); rt_free(AtX); rt_free(XA);
    rt_free(XB); rt_free(Bt); rt_free(XBt); rt_free(XBR);
}

/* LQR for the double integrator: A = [0 1; 0 0], B = [0; 1], Q = I,
 * R = 1. Closed-form K = [1 sqrt(3)]. */
static void test_lqr_double_integrator(void) {
    double a[] = {0, 1, 0, 0};
    double b[] = {0, 1};
    double q[] = {1, 0, 0, 1};
    double r[] = {1.0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Q = mk(q, 2, 2), *R = mk(r, 1, 1);
    matlab_mat *K = matlab_lqr(A, B, Q, R);
    /* K is m x n = 1 x 2. */
    RT_NEAR(rt_at(K, 0, 0), 1.0,        1e-9, "lqr K[0,0]");
    RT_NEAR(rt_at(K, 0, 1), sqrt(3.0),  1e-9, "lqr K[0,1]");
    rt_free(A); rt_free(B); rt_free(Q); rt_free(R); rt_free(K);
}

/* DARE 1x1 closed form: a=0.5, b=1, q=1, r=1. The discrete Riccati
 *   a^2 X - X - a^2 X^2 / (r + b^2 X) + q = 0
 * with these numbers reduces to  X^2 - 0.25 X - 1 = 0  → X ≈ 1.131.
 * Spot-check via residual instead of the closed form; just confirm the
 * Newton-Kleinman iteration converges and the residual is round-off. */
static void test_dare_1x1(void) {
    double a[] = {0.5}, b[] = {1.0}, q[] = {1.0}, r[] = {1.0};
    matlab_mat *Ad = mk(a, 1, 1), *Bd = mk(b, 1, 1);
    matlab_mat *Q  = mk(q, 1, 1), *R  = mk(r, 1, 1);
    matlab_mat *X  = matlab_dare(Ad, Bd, Q, R);
    double x = rt_at(X, 0, 0);
    /* a^2 X - X - a^2 X^2 / (r + b^2 X) + q. */
    double res = a[0]*a[0]*x - x - a[0]*a[0]*x*x / (r[0] + b[0]*b[0]*x) + q[0];
    RT_NEAR(res, 0.0, 1e-12, "dare 1x1 residual");
    RT_CHECK(x > 0, "dare 1x1 X positive");
    rt_free(Ad); rt_free(Bd); rt_free(Q); rt_free(R); rt_free(X);
}

/* DARE residual: A' X A - X - A' X B (R + B' X B)^{-1} B' X A + Q ~ 0
 * for a Schur-stable 2x2 plant. */
static void test_dare_residual(void) {
    /* eig = [0.7, 0.4]: both inside unit disk. */
    double a[] = {0.7, 0.1, 0.0, 0.4};
    double b[] = {1.0, 0.5};         /* 2x1 */
    double q[] = {1.5, 0.2, 0.2, 1.0};
    double r[] = {0.5};
    matlab_mat *Ad = mk(a, 2, 2), *Bd = mk(b, 2, 1);
    matlab_mat *Q  = mk(q, 2, 2), *R  = mk(r, 1, 1);
    matlab_mat *X  = matlab_dare(Ad, Bd, Q, R);
    /* Build the residual via runtime mat ops (matlab_mat is opaque). */
    matlab_mat *At    = matlab_transpose(Ad);
    matlab_mat *AtX   = matlab_matmul_mm(At, X);
    matlab_mat *AtXA  = matlab_matmul_mm(AtX, Ad);
    matlab_mat *Bt    = matlab_transpose(Bd);
    matlab_mat *XB    = matlab_matmul_mm(X, Bd);
    matlab_mat *BtXB  = matlab_matmul_mm(Bt, XB);
    matlab_mat *S     = matlab_add_mm(R, BtXB);
    matlab_mat *Sinv  = matlab_inv(S);
    matlab_mat *AtXB  = matlab_matmul_mm(AtX, Bd);
    matlab_mat *AtXBSi= matlab_matmul_mm(AtXB, Sinv);
    matlab_mat *BtX   = matlab_matmul_mm(Bt, X);
    matlab_mat *BtXA  = matlab_matmul_mm(BtX, Ad);
    matlab_mat *Drop  = matlab_matmul_mm(AtXBSi, BtXA);
    /* res = AtXA - X - Drop + Q. */
    for (int i = 0; i < 4; ++i) {
        double res = rt_data(AtXA)[i] - rt_data(X)[i] - rt_data(Drop)[i] + q[i];
        RT_NEAR(res, 0.0, 1e-9, "dare residual");
    }
    rt_free(Ad); rt_free(Bd); rt_free(Q); rt_free(R);
    rt_free(X); rt_free(At); rt_free(AtX); rt_free(AtXA);
    rt_free(Bt); rt_free(XB); rt_free(BtXB); rt_free(S); rt_free(Sinv);
    rt_free(AtXB); rt_free(AtXBSi); rt_free(BtX); rt_free(BtXA); rt_free(Drop);
}

/* DLQR: K = (R + B'XB)^{-1} B'XA where X = dare(...).
 * Closed-loop Acl = Ad - Bd K must be Schur-stable: |eig(Acl)| < 1.
 * Newton-Kleinman seeded from X_0 = dlyap(Ad', Q) requires Ad already
 * Schur-stable, so this test starts from a stable Ad and verifies the
 * closed-loop also lands inside the unit disk (with smaller radius). */
static void test_dlqr_closed_loop_stable(void) {
    double a[] = {0.6, 0.1, 0.0, 0.7};   /* eig = {0.6, 0.7} */
    double b[] = {1.0, 0.5};             /* 2x1 */
    double q[] = {1, 0, 0, 1};
    double r[] = {1.0};
    matlab_mat *Ad = mk(a, 2, 2), *Bd = mk(b, 2, 1);
    matlab_mat *Q  = mk(q, 2, 2), *R  = mk(r, 1, 1);
    matlab_mat *K  = matlab_dlqr(Ad, Bd, Q, R);
    matlab_mat *BK = matlab_matmul_mm(Bd, K);
    matlab_mat *nBK= matlab_neg_m(BK);
    matlab_mat *Acl= matlab_add_mm(Ad, nBK);
    matlab_mat *e  = matlab_eig(Acl);
    /* Discrete stability: |lambda| < 1. real_c/imag_c work on either
     * real or complex eig output (imag is zero for real spectra). */
    matlab_mat *Re = matlab_real_c(e);
    matlab_mat *Im = matlab_imag_c(e);
    for (int i = 0; i < 2; ++i) {
        double re = rt_at(Re, i, 0);
        double im = rt_at(Im, i, 0);
        double mag = sqrt(re*re + im*im);
        RT_CHECK(mag < 1.0, "dlqr cl pole inside unit disk");
    }
    rt_free(Ad); rt_free(Bd); rt_free(Q); rt_free(R);
    rt_free(K); rt_free(BK); rt_free(nBK); rt_free(Acl);
    rt_free(Re); rt_free(Im);
}

/* ctrb on the canonical SISO double integrator: A=[0 1; 0 0], B=[0; 1].
 * Co = [B, A B] = [[0, 1]; [1, 0]]. Full rank → controllable. */
static void test_ctrb_double_integrator(void) {
    double a[] = {0, 1, 0, 0};
    double b[] = {0, 1};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Co = matlab_ctrb(A, B);
    /* Co is 2 x 2: [[B[0], (A B)[0]]; [B[1], (A B)[1]]] = [[0, 1]; [1, 0]]. */
    RT_NEAR(rt_at(Co, 0, 0), 0.0, 1e-15, "ctrb [0,0]");
    RT_NEAR(rt_at(Co, 0, 1), 1.0, 1e-15, "ctrb [0,1]");
    RT_NEAR(rt_at(Co, 1, 0), 1.0, 1e-15, "ctrb [1,0]");
    RT_NEAR(rt_at(Co, 1, 1), 0.0, 1e-15, "ctrb [1,1]");
    rt_free(A); rt_free(B); rt_free(Co);
}

/* obsv: A = [0 1; 0 0], C = [1 0]. Ob = [C; C A] = [[1 0]; [0 1]]. */
static void test_obsv_double_integrator(void) {
    double a[] = {0, 1, 0, 0};
    double c[] = {1, 0};
    matlab_mat *A = mk(a, 2, 2), *C = mk(c, 1, 2);
    matlab_mat *Ob = matlab_obsv(A, C);
    RT_NEAR(rt_at(Ob, 0, 0), 1.0, 1e-15, "obsv [0,0]");
    RT_NEAR(rt_at(Ob, 0, 1), 0.0, 1e-15, "obsv [0,1]");
    RT_NEAR(rt_at(Ob, 1, 0), 0.0, 1e-15, "obsv [1,0]");
    RT_NEAR(rt_at(Ob, 1, 1), 1.0, 1e-15, "obsv [1,1]");
    rt_free(A); rt_free(C); rt_free(Ob);
}

/* place puts double-integrator poles at {-1, -2}. Closed form:
 * desired char poly = (s+1)(s+2) = s^2 + 3s + 2.
 * Open-loop A has char poly s^2. So K is determined by α(A) = A^2 + 3A + 2I,
 * and via Ackermann, K = [2, 3]. */
static void test_place_double_integrator(void) {
    double a[] = {0, 1, 0, 0};
    double b[] = {0, 1};
    double p[] = {-1.0, -2.0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *P = mk(p, 2, 1);
    matlab_mat *K = matlab_place(A, B, P);
    RT_NEAR(rt_at(K, 0, 0), 2.0, 1e-12, "place K[0,0] = 2");
    RT_NEAR(rt_at(K, 0, 1), 3.0, 1e-12, "place K[0,1] = 3");
    rt_free(A); rt_free(B); rt_free(P); rt_free(K);
}

/* place: closed-loop spectrum must equal P. Verify via eig on Acl
 * for a 3x3 controllable plant with desired poles {-1, -2, -3}. */
static void test_place_pole_match(void) {
    /* Companion-form plant; A is 3x3 with B = [0; 0; 1] guarantees
     * controllability for any A. Open-loop poles arbitrary. */
    double a[] = {0, 1, 0,
                  0, 0, 1,
                  6, 11, -6};         /* eig: roots of -s^3-6s^2+11s+6 ≈ {1, 2, 3} */
    double b[] = {0, 0, 1};
    double p[] = {-1.0, -2.0, -3.0};
    matlab_mat *A = mk(a, 3, 3), *B = mk(b, 3, 1);
    matlab_mat *P = mk(p, 3, 1);
    matlab_mat *K = matlab_place(A, B, P);
    /* Acl = A - B K, eig should be {-3, -2, -1} (sorted ascending real). */
    matlab_mat *BK    = matlab_matmul_mm(B, K);
    matlab_mat *nBK   = matlab_neg_m(BK);
    matlab_mat *Acl   = matlab_add_mm(A, nBK);
    matlab_mat *e     = matlab_eig(Acl);
    matlab_mat *Re    = matlab_real_c(e);
    /* Eig sorted ascending: {-3, -2, -1}. */
    RT_NEAR(rt_at(Re, 0, 0), -3.0, 1e-9, "place pole[0]");
    RT_NEAR(rt_at(Re, 1, 0), -2.0, 1e-9, "place pole[1]");
    RT_NEAR(rt_at(Re, 2, 0), -1.0, 1e-9, "place pole[2]");
    rt_free(A); rt_free(B); rt_free(P); rt_free(K);
    rt_free(BK); rt_free(nBK); rt_free(Acl); rt_free(Re);
}

/* isstable: Hurwitz plant. A = diag(-1, -2). */
static void test_isstable_hurwitz(void) {
    double a[] = {-1, 0, 0, -2};
    matlab_mat *A = mk(a, 2, 2);
    RT_NEAR(matlab_isstable(A), 1.0, 1e-15, "isstable Hurwitz returns 1");
    rt_free(A);
}

/* isstable: unstable plant. A = [1 0; 0 -1]. */
static void test_isstable_unstable(void) {
    double a[] = {1, 0, 0, -1};
    matlab_mat *A = mk(a, 2, 2);
    RT_NEAR(matlab_isstable(A), 0.0, 1e-15, "isstable unstable returns 0");
    rt_free(A);
}

/* isstable: marginally-stable plant (eigenvalue on the imaginary axis).
 * MATLAB convention: marginal is *not* stable (zero real part fails). */
static void test_isstable_marginal(void) {
    double a[] = {0, 1, -1, 0};   /* eig = +-i */
    matlab_mat *A = mk(a, 2, 2);
    RT_NEAR(matlab_isstable(A), 0.0, 1e-15, "isstable marginal returns 0");
    rt_free(A);
}

/* damp: A = diag(-2). Single real pole at -2 → wn = 2, zeta = 1. */
static void test_damp_real_pole(void) {
    double a[] = {-2.0};
    matlab_mat *A = mk(a, 1, 1);
    matlab_mat *D = matlab_damp(A);
    RT_NEAR(rt_at(D, 0, 0), 2.0, 1e-12, "damp wn = 2");
    RT_NEAR(rt_at(D, 0, 1), 1.0, 1e-12, "damp zeta = 1");
    rt_free(A); rt_free(D);
}

/* damp: A = [0 1; -wn^2 -2*zeta*wn], wn = 2, zeta = 0.5.
 * Both eigenvalues should report wn = 2, zeta = 0.5. */
static void test_damp_underdamped(void) {
    double wn = 2.0, zeta = 0.5;
    double a[] = {0, 1, -wn*wn, -2*zeta*wn};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *D = matlab_damp(A);
    /* Two rows (one per pole); each must report wn=2, zeta=0.5. */
    RT_NEAR(rt_at(D, 0, 0), wn,   1e-9, "damp underdamped wn[0]");
    RT_NEAR(rt_at(D, 0, 1), zeta, 1e-9, "damp underdamped zeta[0]");
    RT_NEAR(rt_at(D, 1, 0), wn,   1e-9, "damp underdamped wn[1]");
    RT_NEAR(rt_at(D, 1, 1), zeta, 1e-9, "damp underdamped zeta[1]");
    rt_free(A); rt_free(D);
}

/* hsvd: balanced canonical form has Wc = Wo = diag(hsv). For a
 * diagonal balanced realization with Wc = Wo = diag(2, 1), we'd have
 * hsv = [2; 1]. Easier hand-checkable: SISO 1st-order Hurwitz plant
 *   A = -1, B = 1, C = 1, D = 0.
 *   Wc = lyap(-1, 1) = 0.5; Wo = lyap(-1, 1) = 0.5; Wc*Wo = 0.25;
 *   hsv = sqrt(0.25) = 0.5. */
static void test_hsvd_first_order(void) {
    double a[] = {-1.0}, b[] = {1.0}, c[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1), *C = mk(c, 1, 1);
    matlab_mat *H = matlab_hsvd(A, B, C);
    RT_NEAR(rt_at(H, 0, 0), 0.5, 1e-12, "hsvd 1st order = 1/2");
    rt_free(A); rt_free(B); rt_free(C); rt_free(H);
}

/* hsvd: invariance under similarity transform. T A T^-1 has same hsv.
 * Use A = -diag(1, 2), B = [1; 1], C = [1, 1]; then T = [[1, 1]; [0, 1]]
 * and verify hsv(A, B, C) == hsv(T A T^-1, T B, C T^-1). */
static void test_hsvd_similarity_invariant(void) {
    double a[] = {-1, 0, 0, -2};
    double b[] = {1, 1};
    double c[] = {1, 1};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1), *C = mk(c, 1, 2);
    matlab_mat *H1 = matlab_hsvd(A, B, C);
    /* Apply T = [[1, 1]; [0, 1]] (T^-1 = [[1, -1]; [0, 1]]). */
    double t[] = {1, 1, 0, 1};
    double tinv[] = {1, -1, 0, 1};
    matlab_mat *T = mk(t, 2, 2), *Ti = mk(tinv, 2, 2);
    matlab_mat *TA  = matlab_matmul_mm(T, A);
    matlab_mat *At  = matlab_matmul_mm(TA, Ti);
    matlab_mat *Bt  = matlab_matmul_mm(T, B);
    matlab_mat *Ct  = matlab_matmul_mm(C, Ti);
    matlab_mat *H2 = matlab_hsvd(At, Bt, Ct);
    RT_NEAR(rt_at(H1, 0, 0), rt_at(H2, 0, 0), 1e-9, "hsvd invariant [0]");
    RT_NEAR(rt_at(H1, 1, 0), rt_at(H2, 1, 0), 1e-9, "hsvd invariant [1]");
    rt_free(A); rt_free(B); rt_free(C); rt_free(H1);
    rt_free(T); rt_free(Ti); rt_free(TA); rt_free(At);
    rt_free(Bt); rt_free(Ct); rt_free(H2);
}

/* balreal_T: after balancing, both Wc and Wo equal diag(HSVs descending).
 * Use a diagonal Hurwitz plant where the answer is hand-traceable.
 * For A = -1, B = 1, C = 1: HSV = 0.5 (single-state plant). After
 * balancing, Wc_new = Wo_new = 0.5; the transform is just a scale. */
static void test_balreal_T_first_order(void) {
    double a[] = {-1.0}, b[] = {1.0}, c[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1), *C = mk(c, 1, 1);
    matlab_mat *T  = matlab_balreal_T(A, B, C);
    /* T must be 1x1 nonzero. */
    RT_CHECK(rt_rows(T) == 1 && rt_cols(T) == 1, "balreal_T 1x1 shape");
    RT_CHECK(rt_at(T, 0, 0) != 0.0, "balreal_T 1x1 nonzero");
    /* Verify the balanced gramians: T_inv = 1/T, Wc_new = T_inv^2 Wc.
     * Wc(A,B) = 1/2 for this plant (closed form lyap(-1, 1) = 1/2).
     * Balanced Wc_new should be the HSV = 0.5. So T_inv^2 * 0.5 = 0.5
     * → T = 1 (or -1).  The eigvec sign is arbitrary so check |T|=1. */
    RT_NEAR(fabs(rt_at(T, 0, 0)), 1.0, 1e-9, "balreal_T 1x1 |T| = 1");
    rt_free(A); rt_free(B); rt_free(C); rt_free(T);
}

/* balreal_T: post-balancing, Wc_new = Wo_new = diag(hsv) descending.
 * Verify by computing the new gramians explicitly. */
static void test_balreal_T_post_balanced(void) {
    /* 2-state stable plant. */
    double a[] = {-1, 0, 0, -2};
    double b[] = {1, 1};
    double c[] = {1, 1};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1), *C = mk(c, 1, 2);
    matlab_mat *T  = matlab_balreal_T(A, B, C);
    matlab_mat *Ti = matlab_inv(T);
    /* Build (A_b, B_b, C_b) = (Ti A T, Ti B, C T). */
    matlab_mat *TiA = matlab_matmul_mm(Ti, A);
    matlab_mat *Ab  = matlab_matmul_mm(TiA, T);
    matlab_mat *Bb  = matlab_matmul_mm(Ti, B);
    matlab_mat *Cb  = matlab_matmul_mm(C, T);
    matlab_mat *Wcb = matlab_gram_c(Ab, Bb);
    matlab_mat *Wob = matlab_gram_o(Ab, Cb);
    /* Wcb and Wob must be diagonal (offdiagonals near 0) and equal,
     * with the diagonals = HSVs in descending order. */
    matlab_mat *H = matlab_hsvd(A, B, C);
    RT_NEAR(rt_at(Wcb, 0, 0), rt_at(H, 0, 0), 1e-7, "Wcb[0,0] = hsv[0]");
    RT_NEAR(rt_at(Wcb, 1, 1), rt_at(H, 1, 0), 1e-7, "Wcb[1,1] = hsv[1]");
    RT_NEAR(rt_at(Wcb, 0, 1), 0.0,            1e-7, "Wcb off-diag ~ 0");
    RT_NEAR(rt_at(Wcb, 1, 0), 0.0,            1e-7, "Wcb off-diag ~ 0");
    RT_NEAR(rt_at(Wob, 0, 0), rt_at(H, 0, 0), 1e-7, "Wob[0,0] = hsv[0]");
    RT_NEAR(rt_at(Wob, 1, 1), rt_at(H, 1, 0), 1e-7, "Wob[1,1] = hsv[1]");
    RT_NEAR(rt_at(Wob, 0, 1), 0.0,            1e-7, "Wob off-diag ~ 0");
    RT_NEAR(rt_at(Wob, 1, 0), 0.0,            1e-7, "Wob off-diag ~ 0");
    rt_free(A); rt_free(B); rt_free(C); rt_free(T); rt_free(Ti);
    rt_free(TiA); rt_free(Ab); rt_free(Bb); rt_free(Cb);
    rt_free(Wcb); rt_free(Wob); rt_free(H);
}

/* balred preserves the leading state of the balanced realization.
 * For a 4-state plant with two near-zero HSVs, balred(...,2) keeps
 * the dominant block; we verify shapes and that the truncated
 * realization is still Hurwitz. */
static void test_balred_4to2_stable(void) {
    /* 4-state plant: dominant 2-state mass-spring-damper plus two
     * fast modes weakly coupled (small B/C entries). */
    double a[] = { 0,    1,     0,    0,
                  -9, -0.3,     0,    0,
                   0,    0,   -10,    0,
                   0,    0,     0,  -20};
    double b[] = {0, 1, 0.001, 0.001};
    double c[] = {1, 0, 0.01, 0.01};
    matlab_mat *A = mk(a, 4, 4);
    matlab_mat *B = mk(b, 4, 1);
    matlab_mat *C = mk(c, 1, 4);
    matlab_mat *Ar = matlab_balred_A(A, B, C, 2.0);
    matlab_mat *Br = matlab_balred_B(A, B, C, 2.0);
    matlab_mat *Cr = matlab_balred_C(A, B, C, 2.0);
    /* Shapes: Ar 2x2, Br 2x1, Cr 1x2. */
    RT_CHECK(rt_rows(Ar) == 2 && rt_cols(Ar) == 2, "balred_A 2x2");
    RT_CHECK(rt_rows(Br) == 2 && rt_cols(Br) == 1, "balred_B 2x1");
    RT_CHECK(rt_rows(Cr) == 1 && rt_cols(Cr) == 2, "balred_C 1x2");
    /* Truncated realization must still be Hurwitz (real(eig(Ar)) < 0). */
    RT_NEAR(matlab_isstable(Ar), 1.0, 1e-15, "balred Ar Hurwitz");
    rt_free(A); rt_free(B); rt_free(C); rt_free(Ar); rt_free(Br); rt_free(Cr);
}

/* balred(A, B, C, n) where n = state dim of A: should reproduce the
 * full balanced realization. Verify the truncation-equals-balanced
 * relationship by comparing gramians. */
static void test_balred_full_order_matches_balreal(void) {
    double a[] = {-1, 0, 0, -2};
    double b[] = {1, 1};
    double c[] = {1, 1};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1), *C = mk(c, 1, 2);
    /* Full-order truncation (k = n) should give the balanced realization. */
    matlab_mat *Ar = matlab_balred_A(A, B, C, 2.0);
    matlab_mat *Br = matlab_balred_B(A, B, C, 2.0);
    matlab_mat *Cr = matlab_balred_C(A, B, C, 2.0);
    /* Wc(Ar, Br) should equal diag(HSV) (the balanced gramian). */
    matlab_mat *Wcr = matlab_gram_c(Ar, Br);
    matlab_mat *H   = matlab_hsvd(A, B, C);
    RT_NEAR(rt_at(Wcr, 0, 0), rt_at(H, 0, 0), 1e-7, "full balred Wc[0,0]");
    RT_NEAR(rt_at(Wcr, 1, 1), rt_at(H, 1, 0), 1e-7, "full balred Wc[1,1]");
    rt_free(A); rt_free(B); rt_free(C);
    rt_free(Ar); rt_free(Br); rt_free(Cr); rt_free(Wcr); rt_free(H);
}

/* H2 norm: closed form for SISO 1st-order Hurwitz plant.
 *   xdot = -a x + b u, y = c x.  G(s) = bc / (s + a).
 *   ||G||_2 = b·c / sqrt(2a).
 * Pick a = 1, b = 1, c = 1 → ||G||_2 = 1/sqrt(2) ≈ 0.7071. */
static void test_norm_h2_first_order(void) {
    double a[] = {-1.0}, b[] = {1.0}, c[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1), *C = mk(c, 1, 1);
    double h2 = matlab_norm_h2(A, B, C);
    RT_NEAR(h2, 1.0 / sqrt(2.0), 1e-12, "H2 norm 1st-order");
    rt_free(A); rt_free(B); rt_free(C);
}

/* H2 norm: similarity invariance — for the same plant in different
 * coordinates, the H2 norm must agree. */
static void test_norm_h2_similarity_invariant(void) {
    double a[] = {-1, 0, 0, -2};
    double b[] = {1, 1};
    double c[] = {1, 1};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1), *C = mk(c, 1, 2);
    double h2_orig = matlab_norm_h2(A, B, C);
    /* Apply T = [[1, 1]; [0, 1]]. */
    double t[]    = {1, 1, 0, 1};
    double tinv[] = {1, -1, 0, 1};
    matlab_mat *T = mk(t, 2, 2), *Ti = mk(tinv, 2, 2);
    matlab_mat *TiA = matlab_matmul_mm(Ti, A);
    matlab_mat *At  = matlab_matmul_mm(TiA, T);
    matlab_mat *Bt  = matlab_matmul_mm(Ti, B);
    matlab_mat *Ct  = matlab_matmul_mm(C, T);
    double h2_sim = matlab_norm_h2(At, Bt, Ct);
    RT_NEAR(h2_orig, h2_sim, 1e-12, "H2 invariant under similarity");
    rt_free(A); rt_free(B); rt_free(C); rt_free(T); rt_free(Ti);
    rt_free(TiA); rt_free(At); rt_free(Bt); rt_free(Ct);
}

/* H2 norm: unstable plant returns +Inf. */
static void test_norm_h2_unstable_returns_inf(void) {
    double a[] = {1.0};   /* unstable: positive eigenvalue */
    double b[] = {1.0}, c[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1), *C = mk(c, 1, 1);
    double h2 = matlab_norm_h2(A, B, C);
    RT_CHECK(h2 > 1e10, "H2 norm of unstable plant is +Inf");
    rt_free(A); rt_free(B); rt_free(C);
}

/* dcgain_ss closed form: SISO 1st-order plant.
 *   xdot = -a x + b u, y = c x + d u → G(0) = d - c·(-a)⁻¹·b = d + bc/a.
 * Pick a = 2, b = 1, c = 3, d = 0.5 → G(0) = 0.5 + 3/2 = 2.0. */
static void test_dcgain_ss_first_order(void) {
    double a[] = {-2.0}, b[] = {1.0}, c[] = {3.0}, d[] = {0.5};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1);
    matlab_mat *C = mk(c, 1, 1), *D = mk(d, 1, 1);
    matlab_mat *K = matlab_dcgain_ss(A, B, C, D);
    RT_NEAR(rt_at(K, 0, 0), 2.0, 1e-12, "dcgain 1st-order = 2.0");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(K);
}

/* dcgain_ss for the mass-spring-damper. State-space form:
 *   xdot = [0 1; -k/m -c/m] x + [0; 1/m] u, y = [1 0] x.
 *   At s = 0 the spring dominates; G(0) = 1/k (static deflection per
 *   force). Pick m = 1, k = 4, c = 0.6 → G(0) = 1/4 = 0.25. */
static void test_dcgain_ss_msd(void) {
    double a[] = {0, 1, -4, -0.6};
    double b[] = {0, 1};
    double c[] = {1, 0};
    double d[] = {0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *C = mk(c, 1, 2), *D = mk(d, 1, 1);
    matlab_mat *K = matlab_dcgain_ss(A, B, C, D);
    RT_NEAR(rt_at(K, 0, 0), 0.25, 1e-12, "dcgain MSD = 1/k");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(K);
}

/* dcgain_ss similarity invariance: same plant in different
 * coordinates must give the same DC gain. */
static void test_dcgain_ss_similarity(void) {
    double a[] = {-1, 0.5, 0, -2};
    double b[] = {1, 0.5};
    double c[] = {1, 1};
    double d[] = {0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *C = mk(c, 1, 2), *D = mk(d, 1, 1);
    matlab_mat *K1 = matlab_dcgain_ss(A, B, C, D);
    /* T = [[1, 1]; [0, 1]], Tinv = [[1, -1]; [0, 1]]. */
    double t[]    = {1, 1, 0, 1};
    double tinv[] = {1, -1, 0, 1};
    matlab_mat *T = mk(t, 2, 2), *Ti = mk(tinv, 2, 2);
    matlab_mat *TiA  = matlab_matmul_mm(Ti, A);
    matlab_mat *Asim = matlab_matmul_mm(TiA, T);
    matlab_mat *Bsim = matlab_matmul_mm(Ti, B);
    matlab_mat *Csim = matlab_matmul_mm(C, T);
    matlab_mat *K2 = matlab_dcgain_ss(Asim, Bsim, Csim, D);
    RT_NEAR(rt_at(K1, 0, 0), rt_at(K2, 0, 0), 1e-12, "dcgain invariant");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(K1);
    rt_free(T); rt_free(Ti); rt_free(TiA); rt_free(Asim);
    rt_free(Bsim); rt_free(Csim); rt_free(K2);
}

/* kalman_L: 1×1 closed form. Plant a = -1, G = 1, C = 1, Qn = 1, Rn = 1.
 *   Dual ARE: A P + P A' - P C' Rn^-1 C P + G Qn G' = 0
 *           → -2P - P^2 + 1 = 0 → P = -1 + sqrt(2) (take positive root).
 *   L = P · C' / Rn = sqrt(2) - 1 ≈ 0.4142.
 * Estimator pole: A - L*C = -1 - (sqrt(2)-1) = -sqrt(2) (Hurwitz). */
static void test_kalman_L_first_order(void) {
    double a[] = {-1.0}, g[] = {1.0}, c[] = {1.0};
    double qn[] = {1.0}, rn[] = {1.0};
    matlab_mat *A = mk(a, 1, 1), *G = mk(g, 1, 1), *C = mk(c, 1, 1);
    matlab_mat *Qn = mk(qn, 1, 1), *Rn = mk(rn, 1, 1);
    matlab_mat *L = matlab_kalman_L(A, G, C, Qn, Rn);
    RT_NEAR(rt_at(L, 0, 0), sqrt(2.0) - 1.0, 1e-10, "kalman_L 1×1");
    rt_free(A); rt_free(G); rt_free(C); rt_free(Qn); rt_free(Rn); rt_free(L);
}

/* kalman_L: estimator (A - L*C) must be Hurwitz on a 2-state plant. */
static void test_kalman_L_estimator_stable(void) {
    /* Open-loop unstable plant. */
    double a[] = {1, 1, 0, -2};
    double g[] = {1, 0, 0, 1};   /* 2x2: process noise on each state */
    double c[] = {1, 0};         /* 1x2: measure first state only */
    double qn[] = {1, 0, 0, 1};
    double rn[] = {1.0};
    matlab_mat *A = mk(a, 2, 2), *G = mk(g, 2, 2), *C = mk(c, 1, 2);
    matlab_mat *Qn = mk(qn, 2, 2), *Rn = mk(rn, 1, 1);
    matlab_mat *L = matlab_kalman_L(A, G, C, Qn, Rn);
    /* Estimator A_est = A - L*C must be Hurwitz. */
    matlab_mat *LC = matlab_matmul_mm(L, C);
    matlab_mat *negLC = matlab_neg_m(LC);
    matlab_mat *Aest = matlab_add_mm(A, negLC);
    RT_NEAR(matlab_isstable(Aest), 1.0, 1e-15, "Kalman estimator Hurwitz");
    rt_free(A); rt_free(G); rt_free(C); rt_free(Qn); rt_free(Rn);
    rt_free(L); rt_free(LC); rt_free(negLC); rt_free(Aest);
}

/* kalmd_L: discrete estimator must be Schur-stable. */
static void test_kalmd_L_estimator_schur(void) {
    /* Schur-stable Ad. */
    double ad[] = {0.7, 0.1, 0.0, 0.4};
    double g[] = {1, 0, 0, 1};
    double c[] = {1, 0};
    double qn[] = {1, 0, 0, 1};
    double rn[] = {0.5};
    matlab_mat *Ad = mk(ad, 2, 2), *G = mk(g, 2, 2), *C = mk(c, 1, 2);
    matlab_mat *Qn = mk(qn, 2, 2), *Rn = mk(rn, 1, 1);
    matlab_mat *L = matlab_kalmd_L(Ad, G, C, Qn, Rn);
    /* Estimator Ad_est = Ad - L*C; |eig| must be < 1. */
    matlab_mat *LC = matlab_matmul_mm(L, C);
    matlab_mat *negLC = matlab_neg_m(LC);
    matlab_mat *Adest = matlab_add_mm(Ad, negLC);
    matlab_mat *e  = matlab_eig(Adest);
    matlab_mat *Re = matlab_real_c(e);
    matlab_mat *Im = matlab_imag_c(e);
    for (int i = 0; i < 2; ++i) {
        double re = rt_at(Re, i, 0);
        double im = rt_at(Im, i, 0);
        double mag = sqrt(re*re + im*im);
        RT_CHECK(mag < 1.0, "kalmd estimator Schur-stable");
    }
    rt_free(Ad); rt_free(G); rt_free(C); rt_free(Qn); rt_free(Rn);
    rt_free(L); rt_free(LC); rt_free(negLC); rt_free(Adest);
    rt_free(Re); rt_free(Im);
}

/* LQR closed-loop is Hurwitz: real(eig(A - B K)) < 0 elementwise. */
static void test_lqr_closed_loop_stable(void) {
    /* Marginally unstable plant: A = [1 1; 0 -2] (one positive eigenvalue). */
    double a[] = {1, 1, 0, -2};
    double b[] = {1, 0};
    double q[] = {1, 0, 0, 1};
    double r[] = {1.0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Q = mk(q, 2, 2), *R = mk(r, 1, 1);
    matlab_mat *K = matlab_lqr(A, B, Q, R);
    /* Acl = A - B K (2 x 2 since K is 1x2). Build via matlab arithmetic
     * to avoid touching internal layout (matlab_mat is opaque from the
     * test side). */
    matlab_mat *BK   = matlab_matmul_mm(B, K);
    matlab_mat *negBK= matlab_neg_m(BK);
    matlab_mat *Acl  = matlab_add_mm(A, negBK);
    matlab_mat *e    = matlab_eig(Acl);
    /* Closed-loop poles must have negative real part. e is real for this
     * plant (LQR places it at two real negative poles). */
    RT_CHECK(rt_at(e, 0, 0) < 0.0, "lqr cl pole 0 negative");
    RT_CHECK(rt_at(e, 1, 0) < 0.0, "lqr cl pole 1 negative");
    rt_free(A); rt_free(B); rt_free(Q); rt_free(R);
    rt_free(K); rt_free(BK); rt_free(negBK); rt_free(Acl); rt_free(e);
}

/* c2d ZOH: discretise A=[-1 0; 0 -2], B=[1; 0.5] at Ts=0.1.
 * Closed form (diagonal A): Ad = diag(exp(-0.1), exp(-0.2)),
 * Bd[i] = (1 - exp(A[i]*Ts))/(-A[i]) * B[i]:
 *   Bd[0] = (1 - exp(-0.1)) / 1.0  = 0.0951625819...
 *   Bd[1] = (1 - exp(-0.2)) / 2.0 * 0.5 = 0.0453173...  (wait: (1 - e^(-0.2))/2 = 0.090634...; * 0.5 = 0.0453173) */
static void test_c2d_diagonal(void) {
    double a[] = {-1, 0, 0, -2};
    double b[] = {1.0, 0.5};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Ad = matlab_c2d_Ad(A, B, 0.1);
    matlab_mat *Bd = matlab_c2d_Bd(A, B, 0.1);
    /* Ad diagonal entries. */
    RT_NEAR(rt_at(Ad, 0, 0), exp(-0.1), 1e-12, "c2d Ad[0,0]");
    RT_NEAR(rt_at(Ad, 1, 1), exp(-0.2), 1e-12, "c2d Ad[1,1]");
    RT_NEAR(rt_at(Ad, 0, 1), 0.0,        1e-12, "c2d Ad[0,1]");
    RT_NEAR(rt_at(Ad, 1, 0), 0.0,        1e-12, "c2d Ad[1,0]");
    /* Bd entries via closed form. */
    RT_NEAR(rt_at(Bd, 0, 0), (1.0 - exp(-0.1)) / 1.0,        1e-12, "c2d Bd[0]");
    RT_NEAR(rt_at(Bd, 1, 0), (1.0 - exp(-0.2)) / 2.0 * 0.5,  1e-12, "c2d Bd[1]");
    rt_free(A); rt_free(B); rt_free(Ad); rt_free(Bd);
}

/* c2d Ts=0 limit: Ad -> I, Bd -> 0. Edge case. */
static void test_c2d_zero_Ts(void) {
    double a[] = {1, 2, 3, 4};
    double b[] = {1, 0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Ad = matlab_c2d_Ad(A, B, 0.0);
    matlab_mat *Bd = matlab_c2d_Bd(A, B, 0.0);
    /* Ad = expm(0) = I. */
    RT_NEAR(rt_at(Ad, 0, 0), 1.0, 1e-12, "c2d Ts=0 Ad[0,0]");
    RT_NEAR(rt_at(Ad, 1, 1), 1.0, 1e-12, "c2d Ts=0 Ad[1,1]");
    RT_NEAR(rt_at(Ad, 0, 1), 0.0, 1e-12, "c2d Ts=0 Ad[0,1]");
    RT_NEAR(rt_at(Ad, 1, 0), 0.0, 1e-12, "c2d Ts=0 Ad[1,0]");
    /* Bd = 0. */
    RT_NEAR(rt_at(Bd, 0, 0), 0.0, 1e-12, "c2d Ts=0 Bd[0]");
    RT_NEAR(rt_at(Bd, 1, 0), 0.0, 1e-12, "c2d Ts=0 Bd[1]");
    rt_free(A); rt_free(B); rt_free(Ad); rt_free(Bd);
}

/* Controllability gramian: A = [-1 0; 0 -2], B = [1; 1].
 * Wc satisfies A Wc + Wc A' + B B' = 0. With diagonal A:
 *   Wc[i, j] = (B B')[i, j] / (-A[i, i] - A[j, j]).
 * For B B' = [1 1; 1 1]:
 *   Wc = [[1 / 2,    1 / 3],
 *         [1 / 3,    1 / 4]]. */
static void test_gram_c_diagonal(void) {
    double a[] = {-1, 0, 0, -2};
    double b[] = {1, 1};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *Wc = matlab_gram_c(A, B);
    RT_NEAR(rt_at(Wc, 0, 0), 0.5,         1e-10, "gram_c diag [0,0]");
    RT_NEAR(rt_at(Wc, 0, 1), 1.0 / 3.0,   1e-10, "gram_c diag [0,1]");
    RT_NEAR(rt_at(Wc, 1, 0), 1.0 / 3.0,   1e-10, "gram_c diag [1,0]");
    RT_NEAR(rt_at(Wc, 1, 1), 0.25,        1e-10, "gram_c diag [1,1]");
    rt_free(A); rt_free(B); rt_free(Wc);
}

/* Observability gramian: A = [-1 0; 0 -2], C = [1 1].
 * Wo[i,j] = (C' C)[i,j] / (-A[i,i] - A[j,j]) (same arithmetic). */
static void test_gram_o_diagonal(void) {
    double a[] = {-1, 0, 0, -2};
    double c[] = {1, 1};
    matlab_mat *A = mk(a, 2, 2), *C = mk(c, 1, 2);
    matlab_mat *Wo = matlab_gram_o(A, C);
    RT_NEAR(rt_at(Wo, 0, 0), 0.5,         1e-10, "gram_o diag [0,0]");
    RT_NEAR(rt_at(Wo, 0, 1), 1.0 / 3.0,   1e-10, "gram_o diag [0,1]");
    RT_NEAR(rt_at(Wo, 1, 1), 0.25,        1e-10, "gram_o diag [1,1]");
    rt_free(A); rt_free(C); rt_free(Wo);
}

/* Step response of a first-order lowpass. y(t) = 1 - exp(-t/tau).
 * Continuous form:  xdot = -(1/tau) x + (1/tau) u, y = x.
 *   tau = 0.5, dt = 0.05, N = 10.
 *   y[k] = 1 - exp(-k * dt / tau) at sample k after rebuild...
 * Wait — relaxed initial state means y[0] = 0. y[k] is the value AT
 * sample k, just AFTER updating x with the input applied during step
 * k-1. With u = 1 from k=0:
 *   y[0] = C x[0] + D u = 0 (since D = 0, x[0] = 0).
 *   x[1] = Ad x[0] + Bd u = Bd.
 *   y[1] = C x[1] = Bd.
 * For the closed-form  y(k * dt) = 1 - exp(-k * dt / tau):
 *   Bd = 1 - exp(-dt / tau)  (matches y[1]).
 *   Ad = exp(-dt / tau).
 *   Recurrence  x[k+1] = Ad x[k] + Bd ones to closed form:
 *     y[k] = 1 - Ad^k = 1 - exp(-k*dt/tau).  */
static void test_step_ss_first_order(void) {
    double tau = 0.5;
    double a[] = {-1.0 / tau};      /* A = -1 / tau */
    double b[] = { 1.0 / tau};      /* B =  1 / tau */
    double c[] = { 1.0};            /* C = 1 */
    double d[] = { 0.0};            /* D = 0 */
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1);
    matlab_mat *C = mk(c, 1, 1), *D = mk(d, 1, 1);
    double dt = 0.05;
    int N = 10;
    matlab_mat *y = matlab_step_ss(A, B, C, D, dt, (double)N);
    /* Verify y is N x 1, values match the closed form. */
    for (int k = 0; k < N; ++k) {
        double expected = 1.0 - exp(-k * dt / tau);
        RT_NEAR(rt_at(y, k, 0), expected, 1e-10, "step_ss first order");
    }
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(y);
}

/* Step response with Hurwitz 2x2 plant — sanity check the recurrence is
 * stable and the steady-state value matches  y_ss = -C A^{-1} B. */
static void test_step_ss_steady_state(void) {
    /* Stable 2x2: A = [-1 1; 0 -2], B = [0; 1], C = [1 0], D = 0.
     * Steady state:  y_ss = -C A^{-1} B = 0.5. */
    double a[] = {-1, 1, 0, -2};
    double b[] = {0, 1};
    double c[] = {1, 0};
    double d[] = {0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *C = mk(c, 1, 2), *D = mk(d, 1, 1);
    matlab_mat *y = matlab_step_ss(A, B, C, D, 0.1, 200.0);
    /* y(t = 20) should be very close to 0.5. */
    RT_NEAR(rt_at(y, 199, 0), 0.5, 1e-3, "step_ss steady state");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(y);
}

/* bode_ss for first-order lowpass H(s) = 1/(s+1). At w=1: |H| = 1/sqrt(2),
 * phase = -45 degrees. At w=0: |H| = 1, phase = 0. */
static void test_bode_ss_first_order(void) {
    double a[] = {-1.0};
    double b[] = { 1.0};
    double c[] = { 1.0};
    double d[] = { 0.0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1);
    matlab_mat *C = mk(c, 1, 1), *D = mk(d, 1, 1);
    /* w = [0.0, 1.0, 10.0] */
    double wd[] = {0.0, 1.0, 10.0};
    matlab_mat *w = mk(wd, 3, 1);
    matlab_mat *mag   = matlab_bode_ss_mag  (A, B, C, D, w);
    matlab_mat *phase = matlab_bode_ss_phase(A, B, C, D, w);
    /* DC: |H(0)| = 1, phase = 0. */
    RT_NEAR(rt_at(mag,   0, 0), 1.0,                1e-12, "bode |H(0)|");
    RT_NEAR(rt_at(phase, 0, 0), 0.0,                1e-12, "bode phase(0)");
    /* w = 1: |H| = 1/sqrt(2), phase = -45. */
    RT_NEAR(rt_at(mag,   1, 0), 1.0 / sqrt(2.0),    1e-10, "bode |H(1)|");
    RT_NEAR(rt_at(phase, 1, 0), -45.0,              1e-10, "bode phase(1)");
    /* w = 10: |H| = 1/sqrt(101) ~ 0.0995, phase ~ -84.29 degrees. */
    RT_NEAR(rt_at(mag,   2, 0), 1.0 / sqrt(101.0),  1e-10, "bode |H(10)|");
    RT_NEAR(rt_at(phase, 2, 0), atan2(-10.0, 1.0) * 180.0 / M_PI,
            1e-10, "bode phase(10)");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D);
    rt_free(w); rt_free(mag); rt_free(phase);
}

/* bode_ss for double integrator H(s) = 1/s^2. H(jw) = -1/w^2 (real,
 * negative). Magnitude is 1/w^2; principal-value phase from atan2(0, -)
 * is +180 degrees (equivalent to -180 mod 360). */
static void test_bode_ss_double_integrator(void) {
    double a[] = {0, 1, 0, 0};
    double b[] = {0, 1};
    double c[] = {1, 0};
    double d[] = {0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *C = mk(c, 1, 2), *D = mk(d, 1, 1);
    double wd[] = {0.5, 1.0, 2.0};
    matlab_mat *w = mk(wd, 3, 1);
    matlab_mat *mag   = matlab_bode_ss_mag  (A, B, C, D, w);
    matlab_mat *phase = matlab_bode_ss_phase(A, B, C, D, w);
    for (int k = 0; k < 3; ++k) {
        RT_NEAR(rt_at(mag,   k, 0), 1.0 / (wd[k] * wd[k]), 1e-10,
                "double-int |H(w)|");
        /* Phase is +180 (principal value of atan2(0, -|x|)). */
        RT_NEAR(fabs(rt_at(phase, k, 0)), 180.0, 1e-10, "double-int |phase|");
    }
    rt_free(A); rt_free(B); rt_free(C); rt_free(D);
    rt_free(w); rt_free(mag); rt_free(phase);
}

/* lsim_ss with constant unit input matches step_ss exactly. */
static void test_lsim_ss_matches_step(void) {
    /* First-order lowpass: tau = 0.5. */
    double tau = 0.5;
    double a[] = {-1.0/tau};
    double b[] = { 1.0/tau};
    double c[] = { 1.0};
    double d[] = { 0.0};
    int N = 8;
    double dt = 0.05;
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1);
    matlab_mat *C = mk(c, 1, 1), *D = mk(d, 1, 1);
    /* Build u as N x 1 of ones. */
    double u_data[16];   /* room for up to 16 samples */
    for (int i = 0; i < N; ++i) u_data[i] = 1.0;
    matlab_mat *u = mk(u_data, N, 1);
    matlab_mat *y_lsim = matlab_lsim_ss(A, B, C, D, u, dt);
    matlab_mat *y_step = matlab_step_ss(A, B, C, D, dt, (double)N);
    for (int k = 0; k < N; ++k) {
        RT_NEAR(rt_at(y_lsim, k, 0), rt_at(y_step, k, 0), 1e-12,
                "lsim_ss matches step_ss for u = 1");
    }
    rt_free(A); rt_free(B); rt_free(C); rt_free(D);
    rt_free(u); rt_free(y_lsim); rt_free(y_step);
}

/* lsim_ss with zero input from relaxed initial state returns zeros. */
static void test_lsim_ss_zero_input(void) {
    double a[] = {-1};
    double b[] = { 1};
    double c[] = { 1};
    double d[] = { 0};
    int N = 5;
    double u_data[5] = {0, 0, 0, 0, 0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1);
    matlab_mat *C = mk(c, 1, 1), *D = mk(d, 1, 1);
    matlab_mat *u = mk(u_data, N, 1);
    matlab_mat *y = matlab_lsim_ss(A, B, C, D, u, 0.1);
    for (int k = 0; k < N; ++k)
        RT_NEAR(rt_at(y, k, 0), 0.0, 1e-12, "lsim_ss zero in -> zero out");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(u); rt_free(y);
}

/* gain_margin for first-order lowpass H(s) = 1/(s+1) is +Inf
 * (phase asymptotes to -90 degrees, never reaches -180). */
static void test_gain_margin_first_order(void) {
    double a[] = {-1};
    double b[] = { 1};
    double c[] = { 1};
    double d[] = { 0};
    matlab_mat *A = mk(a, 1, 1), *B = mk(b, 1, 1);
    matlab_mat *C = mk(c, 1, 1), *D = mk(d, 1, 1);
    /* w grid 0.01 .. 100 in 9 points. */
    double wd[9];
    for (int i = 0; i < 9; ++i) wd[i] = pow(10.0, -2.0 + 0.5 * i);
    matlab_mat *w = mk(wd, 9, 1);
    double Gm = matlab_gain_margin(A, B, C, D, w);
    /* +Inf since first-order can never reach -180 phase. */
    RT_CHECK(Gm > 1e10, "gain_margin = Inf for first-order");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(w);
}

/* phase_margin for L(s) = 4 / (s*(s+2)) — gain crossover near
 * sqrt(2) rad/s, phase margin = 60 degrees (closed form). */
static void test_phase_margin_type1(void) {
    /* State-space realisation of L(s) = 4 / (s*(s+2)).
     * SS form (controllable canonical):
     *   xdot = [0 1; 0 -2] x + [0; 1] u, y = [4 0] x. */
    double a[] = {0, 1, 0, -2};
    double b[] = {0, 1};
    double c[] = {4, 0};
    double d[] = {0};
    matlab_mat *A = mk(a, 2, 2), *B = mk(b, 2, 1);
    matlab_mat *C = mk(c, 1, 2), *D = mk(d, 1, 1);
    /* Dense w grid around the gain crossover wc = sqrt(2) ~= 1.414. */
    double wd[400];
    for (int i = 0; i < 400; ++i) wd[i] = 0.1 + 0.01 * i;
    matlab_mat *w = mk(wd, 400, 1);
    double Pm = matlab_phase_margin(A, B, C, D, w);
    /* Closed form:  |L(jwc)| = 4 / (wc * sqrt(wc^2 + 4)) = 1
     *               wc^4 + 4 wc^2 - 16 = 0   ->   wc^2 = 2(sqrt(5) - 1)
     *               wc = sqrt(2(sqrt(5)-1)) ~ 1.5723.
     *               phase(L(jwc)) = -90 - atan(wc/2) ~ -128.16 deg.
     *               Pm = 180 - 128.16 ~ 51.84 deg. */
    RT_NEAR(Pm, 51.8273, 1e-2, "phase margin type-1");
    rt_free(A); rt_free(B); rt_free(C); rt_free(D); rt_free(w);
}

/* bode_tf for H(s) = 1/(s+1): b = [1], a = [1, 1]. Same closed forms
 * as the bode_ss test of the first-order plant. */
static void test_bode_tf_first_order(void) {
    double bd[] = {1.0};
    double ad[] = {1.0, 1.0};
    matlab_mat *b = mk(bd, 1, 1), *a = mk(ad, 2, 1);
    double wd[] = {0.0, 1.0, 10.0};
    matlab_mat *w = mk(wd, 3, 1);
    matlab_mat *mag   = matlab_bode_tf_mag  (b, a, w);
    matlab_mat *phase = matlab_bode_tf_phase(b, a, w);
    RT_NEAR(rt_at(mag,   0, 0), 1.0,                1e-12, "tf |H(0)|");
    RT_NEAR(rt_at(phase, 0, 0), 0.0,                1e-12, "tf phase(0)");
    RT_NEAR(rt_at(mag,   1, 0), 1.0 / sqrt(2.0),    1e-10, "tf |H(1)|");
    RT_NEAR(rt_at(phase, 1, 0), -45.0,              1e-10, "tf phase(1)");
    RT_NEAR(rt_at(mag,   2, 0), 1.0 / sqrt(101.0),  1e-10, "tf |H(10)|");
    rt_free(b); rt_free(a); rt_free(w); rt_free(mag); rt_free(phase);
}

/* bode_tf for H(s) = (s+1)/(s^2 + 2s + 1) = 1/(s+1) — pole-zero
 * cancellation. Should give the same response as the first-order test. */
static void test_bode_tf_pz_cancel(void) {
    double bd[] = {1.0, 1.0};
    double ad[] = {1.0, 2.0, 1.0};
    matlab_mat *b = mk(bd, 2, 1), *a = mk(ad, 3, 1);
    double wd[] = {0.0, 1.0, 10.0};
    matlab_mat *w = mk(wd, 3, 1);
    matlab_mat *mag = matlab_bode_tf_mag(b, a, w);
    /* H is mathematically equal to 1/(s+1) after pole-zero cancellation. */
    RT_NEAR(rt_at(mag, 0, 0), 1.0,             1e-12, "pz |H(0)|");
    RT_NEAR(rt_at(mag, 1, 0), 1.0 / sqrt(2.0), 1e-10, "pz |H(1)|");
    RT_NEAR(rt_at(mag, 2, 0), 1.0 / sqrt(101), 1e-10, "pz |H(10)|");
    rt_free(b); rt_free(a); rt_free(w); rt_free(mag);
}

/* bode_tf and bode_ss must agree for the same plant.
 *   H(s) = 1/(s+1)  vs.  ss(A=-1, B=1, C=1, D=0). */
static void test_bode_tf_matches_ss(void) {
    /* TF form. */
    double bd[] = {1.0};
    double ad[] = {1.0, 1.0};
    matlab_mat *b = mk(bd, 1, 1), *a = mk(ad, 2, 1);
    /* SS form. */
    double Aa[] = {-1.0}, Bb[] = {1.0}, Cc[] = {1.0}, Dd[] = {0.0};
    matlab_mat *A = mk(Aa, 1, 1), *B = mk(Bb, 1, 1);
    matlab_mat *C = mk(Cc, 1, 1), *D = mk(Dd, 1, 1);
    /* Common w grid. */
    double wd[] = {0.1, 0.5, 1.0, 2.0, 10.0};
    matlab_mat *w = mk(wd, 5, 1);
    matlab_mat *m_tf = matlab_bode_tf_mag  (b, a, w);
    matlab_mat *m_ss = matlab_bode_ss_mag  (A, B, C, D, w);
    matlab_mat *p_tf = matlab_bode_tf_phase(b, a, w);
    matlab_mat *p_ss = matlab_bode_ss_phase(A, B, C, D, w);
    for (int k = 0; k < 5; ++k) {
        RT_NEAR(rt_at(m_tf, k, 0), rt_at(m_ss, k, 0), 1e-10,
                "bode_tf vs bode_ss mag");
        RT_NEAR(rt_at(p_tf, k, 0), rt_at(p_ss, k, 0), 1e-10,
                "bode_tf vs bode_ss phase");
    }
    rt_free(b); rt_free(a); rt_free(A); rt_free(B); rt_free(C); rt_free(D);
    rt_free(w); rt_free(m_tf); rt_free(m_ss); rt_free(p_tf); rt_free(p_ss);
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
    RT_RUN(test_expm_zero);
    RT_RUN(test_expm_diagonal);
    RT_RUN(test_expm_rotation);
    RT_RUN(test_expm_inverse_identity);
    RT_RUN(test_expm_large_norm);
    RT_RUN(test_hess_small);
    RT_RUN(test_hess_3x3_subdiagonal_zero);
    RT_RUN(test_hess_4x4_pattern);
    RT_RUN(test_hess_upper_tri_fixed_point);
    RT_RUN(test_hess_preserves_invariants);
    RT_RUN(test_eig_nonsym_companion);
    RT_RUN(test_eig_nonsym_4x4);
    RT_RUN(test_eig_nonsym_rotation_complex);
    RT_RUN(test_eig_nonsym_negative_real);
    RT_RUN(test_schur_1x1);
    RT_RUN(test_schur_upper_tri_fixed);
    RT_RUN(test_schur_reconstructs_A);
    RT_RUN(test_schur_U_orthogonal);
    RT_RUN(test_schur_preserves_spectrum_invariants);
    RT_RUN(test_lyap_1x1);
    RT_RUN(test_lyap_diagonal);
    RT_RUN(test_lyap_residual);
    RT_RUN(test_dlyap_1x1);
    RT_RUN(test_dlyap_residual);
    RT_RUN(test_care_1x1);
    RT_RUN(test_care_double_integrator);
    RT_RUN(test_care_residual);
    RT_RUN(test_lqr_double_integrator);
    RT_RUN(test_lqr_closed_loop_stable);
    RT_RUN(test_dare_1x1);
    RT_RUN(test_dare_residual);
    RT_RUN(test_dlqr_closed_loop_stable);
    RT_RUN(test_ctrb_double_integrator);
    RT_RUN(test_obsv_double_integrator);
    RT_RUN(test_place_double_integrator);
    RT_RUN(test_place_pole_match);
    RT_RUN(test_isstable_hurwitz);
    RT_RUN(test_isstable_unstable);
    RT_RUN(test_isstable_marginal);
    RT_RUN(test_damp_real_pole);
    RT_RUN(test_damp_underdamped);
    RT_RUN(test_hsvd_first_order);
    RT_RUN(test_hsvd_similarity_invariant);
    RT_RUN(test_balreal_T_first_order);
    RT_RUN(test_balreal_T_post_balanced);
    RT_RUN(test_balred_4to2_stable);
    RT_RUN(test_balred_full_order_matches_balreal);
    RT_RUN(test_norm_h2_first_order);
    RT_RUN(test_norm_h2_similarity_invariant);
    RT_RUN(test_norm_h2_unstable_returns_inf);
    RT_RUN(test_dcgain_ss_first_order);
    RT_RUN(test_dcgain_ss_msd);
    RT_RUN(test_dcgain_ss_similarity);
    RT_RUN(test_kalman_L_first_order);
    RT_RUN(test_kalman_L_estimator_stable);
    RT_RUN(test_kalmd_L_estimator_schur);
    RT_RUN(test_c2d_diagonal);
    RT_RUN(test_c2d_zero_Ts);
    RT_RUN(test_gram_c_diagonal);
    RT_RUN(test_gram_o_diagonal);
    RT_RUN(test_step_ss_first_order);
    RT_RUN(test_step_ss_steady_state);
    RT_RUN(test_bode_ss_first_order);
    RT_RUN(test_bode_ss_double_integrator);
    RT_RUN(test_lsim_ss_matches_step);
    RT_RUN(test_lsim_ss_zero_input);
    RT_RUN(test_gain_margin_first_order);
    RT_RUN(test_phase_margin_type1);
    RT_RUN(test_bode_tf_first_order);
    RT_RUN(test_bode_tf_pz_cancel);
    RT_RUN(test_bode_tf_matches_ss);
    RT_DONE();
}
