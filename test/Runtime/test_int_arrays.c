/* Phase 1.1.A — native integer matrix descriptors.
 *
 * Exercises matlab_mat_u8 and matlab_mat_i32 storage primitives:
 * constructors (zeros / ones / eye / from_buf / from_scalar),
 * predicates, scalar + slice indexing, set, fill, concat (row + col),
 * disp. Matches the i64/u64 test surface in test_fi_arrays.c.
 *
 * Saturating arithmetic + casts land in Phase 1.1.B with their own
 * test file. */

#include "runtime_test.h"

/* Test-only mirror of matlab_runtime.cpp:matlab_mat_u8 / matlab_mat_i32.
 * The runtime exposes opaque pointers; tests need to read .data, .rows,
 * .cols. Keep this in sync with the runtime layout. */
struct rt_mat_u8_layout {
    uint8_t *data;
    int64_t  rows;
    int64_t  cols;
};
struct rt_mat_i32_layout {
    int32_t *data;
    int64_t  rows;
    int64_t  cols;
};

/* ===================== uint8 ===================== */

static void test_u8_zeros_and_predicates(void) {
    matlab_mat_u8 *A = matlab_mat_u8_zeros(3, 4);
    RT_NEAR(matlab_mat_u8_numel(A),     12.0, 0.0, "u8 numel");
    RT_NEAR(matlab_mat_u8_size_dim(A, 1), 3.0, 0.0, "u8 size dim1");
    RT_NEAR(matlab_mat_u8_size_dim(A, 2), 4.0, 0.0, "u8 size dim2");
    RT_NEAR((double)matlab_mat_u8_rows(A), 3.0, 0.0, "u8 rows");
    RT_NEAR((double)matlab_mat_u8_cols(A), 4.0, 0.0, "u8 cols");
    RT_NEAR(matlab_mat_u8_length(A),     4.0, 0.0, "u8 length");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(A, 1, 1), 0.0, 0.0, "u8 [1,1]=0");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(A, 3, 4), 0.0, 0.0, "u8 [3,4]=0");
}

static void test_u8_ones_and_eye(void) {
    matlab_mat_u8 *O = matlab_mat_u8_ones(2, 3);
    for (int i = 1; i <= 6; ++i)
        RT_NEAR((double)matlab_mat_u8_subscript1_s(O, i), 1.0, 0.0, "u8 ones");
    matlab_mat_u8 *I = matlab_mat_u8_eye(3, 3);
    RT_NEAR((double)matlab_mat_u8_subscript2_s(I, 1, 1), 1.0, 0.0, "u8 eye[1,1]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(I, 2, 2), 1.0, 0.0, "u8 eye[2,2]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(I, 3, 3), 1.0, 0.0, "u8 eye[3,3]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(I, 1, 2), 0.0, 0.0, "u8 eye[1,2]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(I, 2, 1), 0.0, 0.0, "u8 eye[2,1]");
}

static void test_u8_from_buf_subscript(void) {
    uint8_t b[] = {10, 20, 30, 40, 50, 60};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(b, 2, 3);
    RT_NEAR((double)matlab_mat_u8_subscript2_s(A, 1, 1), 10.0, 0.0, "u8 [1,1]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(A, 1, 3), 30.0, 0.0, "u8 [1,3]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(A, 2, 2), 50.0, 0.0, "u8 [2,2]");
    /* Row-major linear: index 4 is row 2 col 1 = 40. */
    RT_NEAR((double)matlab_mat_u8_subscript1_s(A, 4), 40.0, 0.0, "u8 lin[4]");
}

static void test_u8_from_scalar(void) {
    matlab_mat_u8 *S = matlab_mat_u8_from_scalar(200);
    RT_NEAR(matlab_mat_u8_numel(S), 1.0, 0.0, "u8 scalar numel");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(S, 1, 1), 200.0, 0.0, "u8 scalar val");
}

static void test_u8_set_and_fill(void) {
    matlab_mat_u8 *A = matlab_mat_u8_zeros(2, 3);
    matlab_mat_u8_set1_s(A, 3, 99);
    RT_NEAR((double)matlab_mat_u8_subscript1_s(A, 3), 99.0, 0.0, "u8 set1");
    matlab_mat_u8_set2_s(A, 2, 2, 77);
    RT_NEAR((double)matlab_mat_u8_subscript2_s(A, 2, 2), 77.0, 0.0, "u8 set2");
    matlab_mat_u8_fill(A, 7);
    for (int i = 1; i <= 6; ++i)
        RT_NEAR((double)matlab_mat_u8_subscript1_s(A, i), 7.0, 0.0, "u8 fill");
}

static void test_u8_slice1_and_slice2(void) {
    uint8_t b[] = {10, 20, 30, 40};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(b, 1, 4); /* row vector */
    /* slice1 with idx = [2 4]. */
    double idx[] = {2, 4};
    matlab_mat *I = matlab_mat_from_buf(idx, 1.0, 2.0);
    matlab_mat_u8 *R = matlab_mat_u8_slice1(A, I);
    RT_NEAR(matlab_mat_u8_numel(R), 2.0, 0.0, "u8 slice1 numel");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 1), 20.0, 0.0, "u8 slice1[1]");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 2), 40.0, 0.0, "u8 slice1[2]");

    /* slice2 — pull rows [1,2], cols [1,3] from a 2x3 matrix. */
    uint8_t b2[] = {1, 2, 3, 4, 5, 6};
    matlab_mat_u8 *M = matlab_mat_u8_from_buf(b2, 2, 3);
    double rs[] = {1, 2}, cs[] = {1, 3};
    matlab_mat *R2r = matlab_mat_from_buf(rs, 1.0, 2.0);
    matlab_mat *R2c = matlab_mat_from_buf(cs, 1.0, 2.0);
    matlab_mat_u8 *S = matlab_mat_u8_slice2(M, R2r, R2c);
    RT_NEAR(matlab_mat_u8_numel(S), 4.0, 0.0, "u8 slice2 numel");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(S, 1, 1), 1.0, 0.0, "u8 slice2[1,1]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(S, 1, 2), 3.0, 0.0, "u8 slice2[1,2]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(S, 2, 1), 4.0, 0.0, "u8 slice2[2,1]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(S, 2, 2), 6.0, 0.0, "u8 slice2[2,2]");
}

static void test_u8_concat(void) {
    uint8_t a[] = {1, 2, 3}, b[] = {4, 5};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(a, 1, 3);
    matlab_mat_u8 *B = matlab_mat_u8_from_buf(b, 1, 2);
    matlab_mat_u8 *R = matlab_mat_u8_concat_row(A, B);
    RT_NEAR(matlab_mat_u8_numel(R), 5.0, 0.0, "u8 row-concat numel");
    for (int i = 1; i <= 5; ++i)
        RT_NEAR((double)matlab_mat_u8_subscript1_s(R, i), (double)i, 0.0,
                "u8 row-concat val");

    /* Col concat: stack two 2x3 matrices to get 4x3. */
    uint8_t m1[] = {1, 2, 3, 4, 5, 6}, m2[] = {7, 8, 9, 10, 11, 12};
    matlab_mat_u8 *M1 = matlab_mat_u8_from_buf(m1, 2, 3);
    matlab_mat_u8 *M2 = matlab_mat_u8_from_buf(m2, 2, 3);
    matlab_mat_u8 *C = matlab_mat_u8_concat_col(M1, M2);
    RT_NEAR((double)matlab_mat_u8_rows(C), 4.0, 0.0, "u8 col-concat rows");
    RT_NEAR((double)matlab_mat_u8_cols(C), 3.0, 0.0, "u8 col-concat cols");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(C, 1, 1), 1.0,  0.0, "u8 col[1,1]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(C, 2, 3), 6.0,  0.0, "u8 col[2,3]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(C, 3, 1), 7.0,  0.0, "u8 col[3,1]");
    RT_NEAR((double)matlab_mat_u8_subscript2_s(C, 4, 3), 12.0, 0.0, "u8 col[4,3]");
}

/* ===================== int32 ===================== */

static void test_i32_zeros_and_predicates(void) {
    matlab_mat_i32 *A = matlab_mat_i32_zeros(3, 4);
    RT_NEAR(matlab_mat_i32_numel(A),      12.0, 0.0, "i32 numel");
    RT_NEAR(matlab_mat_i32_size_dim(A, 1), 3.0, 0.0, "i32 dim1");
    RT_NEAR(matlab_mat_i32_size_dim(A, 2), 4.0, 0.0, "i32 dim2");
    RT_NEAR((double)matlab_mat_i32_rows(A), 3.0, 0.0, "i32 rows");
    RT_NEAR((double)matlab_mat_i32_cols(A), 4.0, 0.0, "i32 cols");
    RT_NEAR(matlab_mat_i32_length(A),       4.0, 0.0, "i32 length");
    RT_NEAR((double)matlab_mat_i32_subscript2_s(A, 1, 1), 0.0, 0.0, "i32 [1,1]");
}

static void test_i32_ones_and_eye(void) {
    matlab_mat_i32 *O = matlab_mat_i32_ones(2, 3);
    for (int i = 1; i <= 6; ++i)
        RT_NEAR((double)matlab_mat_i32_subscript1_s(O, i), 1.0, 0.0, "i32 ones");
    matlab_mat_i32 *I = matlab_mat_i32_eye(3, 3);
    RT_NEAR((double)matlab_mat_i32_subscript2_s(I, 1, 1), 1.0, 0.0, "i32 eye[1,1]");
    RT_NEAR((double)matlab_mat_i32_subscript2_s(I, 2, 1), 0.0, 0.0, "i32 eye[2,1]");
}

static void test_i32_from_buf_subscript(void) {
    int32_t b[] = {-100000, 200000, -300000, 400000, -500000, 600000};
    matlab_mat_i32 *A = matlab_mat_i32_from_buf(b, 2, 3);
    RT_NEAR((double)matlab_mat_i32_subscript2_s(A, 1, 1), -100000.0, 0.0, "i32 [1,1]");
    RT_NEAR((double)matlab_mat_i32_subscript2_s(A, 2, 3),  600000.0, 0.0, "i32 [2,3]");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(A, 4),     400000.0, 0.0, "i32 lin[4]");
}

static void test_i32_from_scalar(void) {
    matlab_mat_i32 *S = matlab_mat_i32_from_scalar(-2147483647);
    RT_NEAR(matlab_mat_i32_numel(S), 1.0, 0.0, "i32 scalar numel");
    RT_NEAR((double)matlab_mat_i32_subscript2_s(S, 1, 1),
            -2147483647.0, 0.0, "i32 scalar val");
}

static void test_i32_set_and_fill(void) {
    matlab_mat_i32 *A = matlab_mat_i32_zeros(2, 3);
    matlab_mat_i32_set1_s(A, 3, 999999);
    RT_NEAR((double)matlab_mat_i32_subscript1_s(A, 3), 999999.0, 0.0, "i32 set1");
    matlab_mat_i32_set2_s(A, 2, 2, -77777);
    RT_NEAR((double)matlab_mat_i32_subscript2_s(A, 2, 2), -77777.0, 0.0, "i32 set2");
    matlab_mat_i32_fill(A, 42);
    for (int i = 1; i <= 6; ++i)
        RT_NEAR((double)matlab_mat_i32_subscript1_s(A, i), 42.0, 0.0, "i32 fill");
}

static void test_i32_slice1_and_slice2(void) {
    int32_t b[] = {10, 20, 30, 40};
    matlab_mat_i32 *A = matlab_mat_i32_from_buf(b, 1, 4);
    double idx[] = {2, 4};
    matlab_mat *I = matlab_mat_from_buf(idx, 1.0, 2.0);
    matlab_mat_i32 *R = matlab_mat_i32_slice1(A, I);
    RT_NEAR(matlab_mat_i32_numel(R), 2.0, 0.0, "i32 slice1 numel");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 1), 20.0, 0.0, "i32 slice1[1]");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 2), 40.0, 0.0, "i32 slice1[2]");

    int32_t b2[] = {1, 2, 3, 4, 5, 6};
    matlab_mat_i32 *M = matlab_mat_i32_from_buf(b2, 2, 3);
    double rs[] = {1, 2}, cs[] = {1, 3};
    matlab_mat *Rr = matlab_mat_from_buf(rs, 1.0, 2.0);
    matlab_mat *Rc = matlab_mat_from_buf(cs, 1.0, 2.0);
    matlab_mat_i32 *S = matlab_mat_i32_slice2(M, Rr, Rc);
    RT_NEAR((double)matlab_mat_i32_subscript2_s(S, 1, 1), 1.0, 0.0, "i32 slice2[1,1]");
    RT_NEAR((double)matlab_mat_i32_subscript2_s(S, 2, 2), 6.0, 0.0, "i32 slice2[2,2]");
}

static void test_i32_concat(void) {
    int32_t a[] = {1, 2, 3}, b[] = {4, 5};
    matlab_mat_i32 *A = matlab_mat_i32_from_buf(a, 1, 3);
    matlab_mat_i32 *B = matlab_mat_i32_from_buf(b, 1, 2);
    matlab_mat_i32 *R = matlab_mat_i32_concat_row(A, B);
    RT_NEAR(matlab_mat_i32_numel(R), 5.0, 0.0, "i32 row-concat numel");
    for (int i = 1; i <= 5; ++i)
        RT_NEAR((double)matlab_mat_i32_subscript1_s(R, i), (double)i, 0.0,
                "i32 row-concat val");

    int32_t m1[] = {1, 2, 3, 4, 5, 6}, m2[] = {7, 8, 9, 10, 11, 12};
    matlab_mat_i32 *M1 = matlab_mat_i32_from_buf(m1, 2, 3);
    matlab_mat_i32 *M2 = matlab_mat_i32_from_buf(m2, 2, 3);
    matlab_mat_i32 *C = matlab_mat_i32_concat_col(M1, M2);
    RT_NEAR((double)matlab_mat_i32_rows(C), 4.0, 0.0, "i32 col-concat rows");
    RT_NEAR((double)matlab_mat_i32_subscript2_s(C, 4, 3), 12.0, 0.0, "i32 col[4,3]");
}

/* ============== Edge cases shared by both lanes ============== */

static void test_int_array_zero_dim(void) {
    /* Zero-row / zero-col allocations must be safe. */
    matlab_mat_u8  *A = matlab_mat_u8_zeros(0, 0);
    matlab_mat_i32 *B = matlab_mat_i32_zeros(0, 5);
    RT_NEAR(matlab_mat_u8_numel(A),  0.0, 0.0, "u8 0x0 numel");
    RT_NEAR(matlab_mat_i32_numel(B), 0.0, 0.0, "i32 0x5 numel");
    /* Subscript on empty → 0. */
    RT_NEAR((double)matlab_mat_u8_subscript1_s(A, 1),  0.0, 0.0, "u8 empty sub");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(B, 1), 0.0, 0.0, "i32 empty sub");
}

static void test_int_array_null_safety(void) {
    /* All public helpers tolerate NULL inputs (matches matlab_mat_i64
     * conventions — the JIT can pass an uninitialised slot through). */
    RT_NEAR(matlab_mat_u8_numel(NULL),  0.0, 0.0, "u8 null numel");
    RT_NEAR(matlab_mat_i32_numel(NULL), 0.0, 0.0, "i32 null numel");
    RT_NEAR(matlab_mat_u8_length(NULL), 0.0, 0.0, "u8 null length");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(NULL, 1),  0.0, 0.0, "u8 null sub");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(NULL, 1), 0.0, 0.0, "i32 null sub");
    /* Stores on NULL must not crash. */
    matlab_mat_u8_set1_s(NULL,  1, 7);
    matlab_mat_i32_set1_s(NULL, 1, 7);
    matlab_mat_u8_fill(NULL,  9);
    matlab_mat_i32_fill(NULL, 9);
    RT_CHECK(1, "null fill survived");
}

/* ===================== Phase 1.1.B: arith / cmp / cast ===================== */

/* Casts */

static void test_cast_u8_from_double_saturates(void) {
    /* In: [-1, 0, 100, 255, 256, 1e9, NaN]; Expect: [0, 0, 100, 255, 255, 255, 0] */
    double in[] = {-1.0, 0.0, 100.0, 255.0, 256.0, 1e9, 0.0/0.0};
    matlab_mat *D = matlab_mat_from_buf(in, 1.0, 7.0);
    matlab_mat_u8 *R = matlab_mat_u8_from_double(D);
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 1),   0.0, 0.0, "u8 cast -1");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 2),   0.0, 0.0, "u8 cast 0");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 3), 100.0, 0.0, "u8 cast 100");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 4), 255.0, 0.0, "u8 cast 255");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 5), 255.0, 0.0, "u8 cast 256→255");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 6), 255.0, 0.0, "u8 cast 1e9→255");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 7),   0.0, 0.0, "u8 cast NaN→0");
}

static void test_cast_i32_from_double_saturates(void) {
    /* MATLAB rounds half-away-from-zero: 2.5 → 3, -2.5 → -3. */
    double in[] = {0.0, 2.5, -2.5, 1e10, -1e10, 0.0/0.0};
    matlab_mat *D = matlab_mat_from_buf(in, 1.0, 6.0);
    matlab_mat_i32 *R = matlab_mat_i32_from_double(D);
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 1),       0.0, 0.0, "i32 cast 0");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 2),       3.0, 0.0, "i32 cast 2.5→3");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 3),      -3.0, 0.0, "i32 cast -2.5→-3");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 4),  2147483647.0, 0.0, "i32 cast 1e10");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 5), -2147483648.0, 0.0, "i32 cast -1e10");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 6),       0.0, 0.0, "i32 cast NaN→0");
}

static void test_cast_to_double(void) {
    uint8_t bu[] = {0, 100, 255};
    matlab_mat_u8 *U = matlab_mat_u8_from_buf(bu, 1, 3);
    matlab_mat *DU = matlab_mat_u8_to_double(U);
    RT_NEAR(rt_at(DU, 0, 0),   0.0, 0.0, "u8→d 0");
    RT_NEAR(rt_at(DU, 0, 1), 100.0, 0.0, "u8→d 100");
    RT_NEAR(rt_at(DU, 0, 2), 255.0, 0.0, "u8→d 255");

    int32_t bi[] = {-2147483647, 0, 2147483647};
    matlab_mat_i32 *I = matlab_mat_i32_from_buf(bi, 1, 3);
    matlab_mat *DI = matlab_mat_i32_to_double(I);
    RT_NEAR(rt_at(DI, 0, 0), -2147483647.0, 0.0, "i32→d min");
    RT_NEAR(rt_at(DI, 0, 1),           0.0, 0.0, "i32→d 0");
    RT_NEAR(rt_at(DI, 0, 2),  2147483647.0, 0.0, "i32→d max");
}

static void test_cast_cross(void) {
    int32_t bi[] = {-1, 0, 100, 255, 300};
    matlab_mat_i32 *I = matlab_mat_i32_from_buf(bi, 1, 5);
    matlab_mat_u8 *U = matlab_mat_u8_from_i32(I);
    RT_NEAR((double)matlab_mat_u8_subscript1_s(U, 1),   0.0, 0.0, "i32→u8 -1");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(U, 4), 255.0, 0.0, "i32→u8 255");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(U, 5), 255.0, 0.0, "i32→u8 300→sat");

    uint8_t bu[] = {0, 100, 255};
    matlab_mat_u8 *U2 = matlab_mat_u8_from_buf(bu, 1, 3);
    matlab_mat_i32 *I2 = matlab_mat_i32_from_u8(U2);
    RT_NEAR((double)matlab_mat_i32_subscript1_s(I2, 1),   0.0, 0.0, "u8→i32 0");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(I2, 3), 255.0, 0.0, "u8→i32 255");
}

/* Arithmetic — saturation */

static void test_u8_add_saturates(void) {
    uint8_t a[] = {100, 200, 50}, b[] = {100, 100, 50};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(a, 1, 3);
    matlab_mat_u8 *B = matlab_mat_u8_from_buf(b, 1, 3);
    matlab_mat_u8 *R = matlab_mat_u8_add_mm(A, B);
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 1), 200.0, 0.0, "u8 100+100");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 2), 255.0, 0.0, "u8 200+100→sat");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 3), 100.0, 0.0, "u8 50+50");

    matlab_mat_u8 *Rs = matlab_mat_u8_add_ms(A, 60);
    RT_NEAR((double)matlab_mat_u8_subscript1_s(Rs, 2), 255.0, 0.0, "u8 200+60→sat");
}

static void test_u8_sub_saturates(void) {
    uint8_t a[] = {10, 100}, b[] = {20, 50};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(a, 1, 2);
    matlab_mat_u8 *B = matlab_mat_u8_from_buf(b, 1, 2);
    matlab_mat_u8 *R = matlab_mat_u8_sub_mm(A, B);
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 1),  0.0, 0.0, "u8 10-20→0");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(R, 2), 50.0, 0.0, "u8 100-50");
}

static void test_u8_mul_div(void) {
    uint8_t a[] = {10, 50, 100}, b[] = {10, 6, 3};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(a, 1, 3);
    matlab_mat_u8 *B = matlab_mat_u8_from_buf(b, 1, 3);
    matlab_mat_u8 *M = matlab_mat_u8_emul_mm(A, B);
    RT_NEAR((double)matlab_mat_u8_subscript1_s(M, 1), 100.0, 0.0, "u8 10*10");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(M, 2), 255.0, 0.0, "u8 50*6→sat");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(M, 3), 255.0, 0.0, "u8 100*3→sat");

    matlab_mat_u8 *D = matlab_mat_u8_ediv_mm(A, B);
    /* 10/10=1, 50/6 = 8.33 → round to 8, 100/3 = 33.33 → round to 33. */
    RT_NEAR((double)matlab_mat_u8_subscript1_s(D, 1),  1.0, 0.0, "u8 10/10");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(D, 2),  8.0, 0.0, "u8 50/6→8");
    RT_NEAR((double)matlab_mat_u8_subscript1_s(D, 3), 33.0, 0.0, "u8 100/3→33");
}

static void test_i32_add_saturates(void) {
    int32_t a[] = {2147483640, -2147483640, 100};
    int32_t b[] = {        20,         -20,  50};
    matlab_mat_i32 *A = matlab_mat_i32_from_buf(a, 1, 3);
    matlab_mat_i32 *B = matlab_mat_i32_from_buf(b, 1, 3);
    matlab_mat_i32 *R = matlab_mat_i32_add_mm(A, B);
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 1),  2147483647.0, 0.0,
            "i32 near-max + 20→sat");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 2), -2147483648.0, 0.0,
            "i32 near-min + -20→sat");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(R, 3),         150.0, 0.0,
            "i32 100+50");
}

static void test_i32_mul_div(void) {
    int32_t a[] = {100000, 1000000000, -7};
    int32_t b[] = {100000,         10,  2};
    matlab_mat_i32 *A = matlab_mat_i32_from_buf(a, 1, 3);
    matlab_mat_i32 *B = matlab_mat_i32_from_buf(b, 1, 3);
    matlab_mat_i32 *M = matlab_mat_i32_emul_mm(A, B);
    RT_NEAR((double)matlab_mat_i32_subscript1_s(M, 1), 1.0e10 > 2147483647.0
                                                        ? 2147483647.0 : 1.0e10,
            0.0, "i32 100k*100k→sat");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(M, 2), 2147483647.0, 0.0,
            "i32 1e9*10→sat");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(M, 3),       -14.0, 0.0,
            "i32 -7*2");

    matlab_mat_i32 *D = matlab_mat_i32_ediv_mm(A, B);
    /* 100000/100000=1, 1e9/10=1e8, -7/2 = -3.5 → -4 (away from zero). */
    RT_NEAR((double)matlab_mat_i32_subscript1_s(D, 1),         1.0, 0.0, "i32 1");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(D, 2), 100000000.0, 0.0, "i32 1e9/10");
    RT_NEAR((double)matlab_mat_i32_subscript1_s(D, 3),        -4.0, 0.0, "i32 -7/2");
}

/* Comparisons — produce 0/1 doubles in a regular matlab_mat. */

static void test_u8_cmp_returns_logical(void) {
    uint8_t a[] = {1, 5, 10}, b[] = {3, 5, 7};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(a, 1, 3);
    matlab_mat_u8 *B = matlab_mat_u8_from_buf(b, 1, 3);
    matlab_mat *Gt = matlab_mat_u8_gt_mm(A, B);
    RT_NEAR(rt_at(Gt, 0, 0), 0.0, 0.0, "u8 1>3→0");
    RT_NEAR(rt_at(Gt, 0, 1), 0.0, 0.0, "u8 5>5→0");
    RT_NEAR(rt_at(Gt, 0, 2), 1.0, 0.0, "u8 10>7→1");

    matlab_mat *Eq = matlab_mat_u8_eq_mm(A, B);
    RT_NEAR(rt_at(Eq, 0, 0), 0.0, 0.0, "u8 1==3→0");
    RT_NEAR(rt_at(Eq, 0, 1), 1.0, 0.0, "u8 5==5→1");

    /* scalar form */
    matlab_mat *Le = matlab_mat_u8_le_ms(A, 5);
    RT_NEAR(rt_at(Le, 0, 0), 1.0, 0.0, "u8 1<=5");
    RT_NEAR(rt_at(Le, 0, 1), 1.0, 0.0, "u8 5<=5");
    RT_NEAR(rt_at(Le, 0, 2), 0.0, 0.0, "u8 10<=5→0");
}

static void test_i32_cmp_returns_logical(void) {
    int32_t a[] = {-1000, 0, 999999};
    matlab_mat_i32 *A = matlab_mat_i32_from_buf(a, 1, 3);
    matlab_mat *Lt = matlab_mat_i32_lt_ms(A, 0);
    RT_NEAR(rt_at(Lt, 0, 0), 1.0, 0.0, "i32 -1000<0");
    RT_NEAR(rt_at(Lt, 0, 1), 0.0, 0.0, "i32 0<0");
    RT_NEAR(rt_at(Lt, 0, 2), 0.0, 0.0, "i32 999999<0");

    matlab_mat *Ne = matlab_mat_i32_ne_ms(A, 0);
    RT_NEAR(rt_at(Ne, 0, 0), 1.0, 0.0, "i32 -1000!=0");
    RT_NEAR(rt_at(Ne, 0, 1), 0.0, 0.0, "i32 0!=0");
    RT_NEAR(rt_at(Ne, 0, 2), 1.0, 0.0, "i32 999999!=0");
}

/* Reductions */

static void test_u8_reductions(void) {
    uint8_t a[] = {10, 250, 30};
    matlab_mat_u8 *A = matlab_mat_u8_from_buf(a, 1, 3);
    /* sum saturates: 10+250 already 260→255, then +30 still 255. */
    RT_NEAR((double)matlab_mat_u8_sum(A), 255.0, 0.0, "u8 sum sat");
    /* mean: (10+250+30)/3 = 96.67 → 97. */
    RT_NEAR((double)matlab_mat_u8_mean(A), 97.0, 0.0, "u8 mean");
    RT_NEAR((double)matlab_mat_u8_min(A),  10.0, 0.0, "u8 min");
    RT_NEAR((double)matlab_mat_u8_max(A), 250.0, 0.0, "u8 max");
}

static void test_i32_reductions(void) {
    int32_t a[] = {-100, 0, 100, 500};
    matlab_mat_i32 *A = matlab_mat_i32_from_buf(a, 1, 4);
    RT_NEAR((double)matlab_mat_i32_sum(A),  500.0, 0.0, "i32 sum");
    RT_NEAR((double)matlab_mat_i32_mean(A), 125.0, 0.0, "i32 mean");
    RT_NEAR((double)matlab_mat_i32_min(A), -100.0, 0.0, "i32 min");
    RT_NEAR((double)matlab_mat_i32_max(A),  500.0, 0.0, "i32 max");
}

int main(void) {
    fprintf(stderr, "test_int_arrays:\n");
    /* 1.1.A — descriptors / storage */
    RT_RUN(test_u8_zeros_and_predicates);
    RT_RUN(test_u8_ones_and_eye);
    RT_RUN(test_u8_from_buf_subscript);
    RT_RUN(test_u8_from_scalar);
    RT_RUN(test_u8_set_and_fill);
    RT_RUN(test_u8_slice1_and_slice2);
    RT_RUN(test_u8_concat);
    RT_RUN(test_i32_zeros_and_predicates);
    RT_RUN(test_i32_ones_and_eye);
    RT_RUN(test_i32_from_buf_subscript);
    RT_RUN(test_i32_from_scalar);
    RT_RUN(test_i32_set_and_fill);
    RT_RUN(test_i32_slice1_and_slice2);
    RT_RUN(test_i32_concat);
    RT_RUN(test_int_array_zero_dim);
    RT_RUN(test_int_array_null_safety);
    /* 1.1.B — casts / arith / cmp / reductions */
    RT_RUN(test_cast_u8_from_double_saturates);
    RT_RUN(test_cast_i32_from_double_saturates);
    RT_RUN(test_cast_to_double);
    RT_RUN(test_cast_cross);
    RT_RUN(test_u8_add_saturates);
    RT_RUN(test_u8_sub_saturates);
    RT_RUN(test_u8_mul_div);
    RT_RUN(test_i32_add_saturates);
    RT_RUN(test_i32_mul_div);
    RT_RUN(test_u8_cmp_returns_logical);
    RT_RUN(test_i32_cmp_returns_logical);
    RT_RUN(test_u8_reductions);
    RT_RUN(test_i32_reductions);
    RT_DONE();
}
