/* Phase-1 catch-up: elementwise binary ops + comparisons.
 *
 * Targets the macro-generated families:
 *   matlab_(add|sub|emul|ediv|epow)_(mm|ms|sm)
 *   matlab_(gt|ge|lt|le|eq|ne)_(mm|ms|sm)
 *
 * Each function gets a single shape-3 case so any regression in the
 * macro expansion or polymorphic-dispatch wrapper surfaces immediately.
 * Direct unit tests for these matter even though they're macro-generated:
 * a typo in the macro body would silently corrupt every variant. */

#include "runtime_test.h"

/* Binary ops (mm/ms/sm). The mm forms take void* for polymorphic
 * complex dispatch but we always pass real matrices in this suite. */
matlab_mat *matlab_add_mm (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_sub_mm (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_emul_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ediv_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_epow_mm(matlab_mat *A, matlab_mat *B);

matlab_mat *matlab_add_ms (matlab_mat *A, double s);
matlab_mat *matlab_sub_ms (matlab_mat *A, double s);
matlab_mat *matlab_emul_ms(matlab_mat *A, double s);
matlab_mat *matlab_ediv_ms(matlab_mat *A, double s);
matlab_mat *matlab_epow_ms(matlab_mat *A, double s);

matlab_mat *matlab_add_sm (double s, matlab_mat *A);
matlab_mat *matlab_sub_sm (double s, matlab_mat *A);
matlab_mat *matlab_emul_sm(double s, matlab_mat *A);
matlab_mat *matlab_ediv_sm(double s, matlab_mat *A);
matlab_mat *matlab_epow_sm(double s, matlab_mat *A);

matlab_mat *matlab_neg_m (matlab_mat *A);

/* Comparisons (mm/ms/sm). All return 0/1 matrices. */
matlab_mat *matlab_gt_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ge_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_lt_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_le_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_eq_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ne_mm(matlab_mat *A, matlab_mat *B);

matlab_mat *matlab_gt_ms(matlab_mat *A, double s);
matlab_mat *matlab_ge_ms(matlab_mat *A, double s);
matlab_mat *matlab_lt_ms(matlab_mat *A, double s);
matlab_mat *matlab_le_ms(matlab_mat *A, double s);
matlab_mat *matlab_eq_ms(matlab_mat *A, double s);
matlab_mat *matlab_ne_ms(matlab_mat *A, double s);

matlab_mat *matlab_gt_sm(double s, matlab_mat *A);
matlab_mat *matlab_ge_sm(double s, matlab_mat *A);
matlab_mat *matlab_lt_sm(double s, matlab_mat *A);
matlab_mat *matlab_le_sm(double s, matlab_mat *A);
matlab_mat *matlab_eq_sm(double s, matlab_mat *A);
matlab_mat *matlab_ne_sm(double s, matlab_mat *A);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* --- mm binary ops -------------------------------------------------- */
static void test_binary_mm(void) {
    double a[] = {1, 2, 3, 4};
    double b[] = {2, 2, 2, 2};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *B = mk(b, 2, 2);

    matlab_mat *r;

    r = matlab_add_mm(A, B);
    RT_NEAR(rt_at(r, 0, 0), 3.0, 1e-12, "add_mm[0,0]");
    RT_NEAR(rt_at(r, 1, 1), 6.0, 1e-12, "add_mm[1,1]");

    r = matlab_sub_mm(A, B);
    RT_NEAR(rt_at(r, 0, 0), -1.0, 1e-12, "sub_mm[0,0]");
    RT_NEAR(rt_at(r, 1, 1),  2.0, 1e-12, "sub_mm[1,1]");

    r = matlab_emul_mm(A, B);
    RT_NEAR(rt_at(r, 0, 1), 4.0, 1e-12, "emul_mm");

    r = matlab_ediv_mm(A, B);
    RT_NEAR(rt_at(r, 1, 0), 1.5, 1e-12, "ediv_mm");

    r = matlab_epow_mm(A, B);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 1e-12, "epow_mm 1^2");
    RT_NEAR(rt_at(r, 1, 1), 16.0, 1e-12, "epow_mm 4^2");
}

/* --- ms (matrix, scalar) ------------------------------------------- */
static void test_binary_ms(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r;

    r = matlab_add_ms(A, 10.0);
    RT_NEAR(rt_at(r, 0, 0), 11.0, 1e-12, "add_ms");

    r = matlab_sub_ms(A, 1.0);
    RT_NEAR(rt_at(r, 1, 1), 3.0, 1e-12, "sub_ms");

    r = matlab_emul_ms(A, 3.0);
    RT_NEAR(rt_at(r, 0, 1), 6.0, 1e-12, "emul_ms");

    r = matlab_ediv_ms(A, 2.0);
    RT_NEAR(rt_at(r, 1, 0), 1.5, 1e-12, "ediv_ms");

    r = matlab_epow_ms(A, 2.0);
    RT_NEAR(rt_at(r, 1, 1), 16.0, 1e-12, "epow_ms");
}

/* --- sm (scalar, matrix) ------------------------------------------- */
static void test_binary_sm(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r;

    r = matlab_add_sm(10.0, A);
    RT_NEAR(rt_at(r, 1, 1), 14.0, 1e-12, "add_sm");

    r = matlab_sub_sm(10.0, A);
    /* 10 - 4 = 6 at [1,1]. */
    RT_NEAR(rt_at(r, 1, 1), 6.0, 1e-12, "sub_sm");

    r = matlab_emul_sm(3.0, A);
    RT_NEAR(rt_at(r, 0, 1), 6.0, 1e-12, "emul_sm");

    r = matlab_ediv_sm(12.0, A);
    /* 12/3 = 4 at [1,0]. */
    RT_NEAR(rt_at(r, 1, 0), 4.0, 1e-12, "ediv_sm");

    r = matlab_epow_sm(2.0, A);
    /* 2^4 = 16 at [1,1]. */
    RT_NEAR(rt_at(r, 1, 1), 16.0, 1e-12, "epow_sm");
}

/* --- unary neg ------------------------------------------------------ */
static void test_neg_m(void) {
    double a[] = {1, -2, 3, -4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r = matlab_neg_m(A);
    RT_NEAR(rt_at(r, 0, 0), -1.0, 1e-12, "neg [0,0]");
    RT_NEAR(rt_at(r, 0, 1),  2.0, 1e-12, "neg [0,1]");
    RT_NEAR(rt_at(r, 1, 1),  4.0, 1e-12, "neg [1,1]");
}

/* --- mm comparisons ------------------------------------------------- */
static void test_compare_mm(void) {
    double a[] = {1, 2, 3, 4};
    double b[] = {2, 2, 2, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *B = mk(b, 2, 2);
    matlab_mat *r;

    r = matlab_gt_mm(A, B);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 0.0, "gt 1>2");
    RT_NEAR(rt_at(r, 1, 0), 1.0, 0.0, "gt 3>2");

    r = matlab_ge_mm(A, B);
    RT_NEAR(rt_at(r, 1, 1), 1.0, 0.0, "ge 4>=4");

    r = matlab_lt_mm(A, B);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 0.0, "lt 1<2");
    RT_NEAR(rt_at(r, 1, 1), 0.0, 0.0, "lt 4<4");

    r = matlab_le_mm(A, B);
    RT_NEAR(rt_at(r, 1, 1), 1.0, 0.0, "le 4<=4");

    r = matlab_eq_mm(A, B);
    RT_NEAR(rt_at(r, 0, 1), 1.0, 0.0, "eq 2==2");
    RT_NEAR(rt_at(r, 1, 1), 1.0, 0.0, "eq 4==4");

    r = matlab_ne_mm(A, B);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 0.0, "ne 1!=2");
    RT_NEAR(rt_at(r, 1, 1), 0.0, 0.0, "ne 4!=4 false");
}

/* --- ms comparisons (matrix-scalar) -------------------------------- */
static void test_compare_ms(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r;

    r = matlab_gt_ms(A, 2.0);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 0.0, "gt 1>2");
    RT_NEAR(rt_at(r, 1, 1), 1.0, 0.0, "gt 4>2");

    r = matlab_ge_ms(A, 2.0);
    RT_NEAR(rt_at(r, 0, 1), 1.0, 0.0, "ge 2>=2");

    r = matlab_lt_ms(A, 3.0);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 0.0, "lt 1<3");
    RT_NEAR(rt_at(r, 1, 0), 0.0, 0.0, "lt 3<3 false");

    r = matlab_le_ms(A, 3.0);
    RT_NEAR(rt_at(r, 1, 0), 1.0, 0.0, "le 3<=3");

    r = matlab_eq_ms(A, 3.0);
    RT_NEAR(rt_at(r, 1, 0), 1.0, 0.0, "eq 3==3");

    r = matlab_ne_ms(A, 0.0);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 0.0, "ne 1!=0");
}

/* --- sm comparisons (scalar-matrix) -------------------------------- */
static void test_compare_sm(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r;

    r = matlab_gt_sm(3.0, A);
    /* 3 > 1, 3 > 2, 3 > 3 false, 3 > 4 false. */
    RT_NEAR(rt_at(r, 0, 0), 1.0, 0.0, "gt 3>1");
    RT_NEAR(rt_at(r, 1, 0), 0.0, 0.0, "gt 3>3 false");

    r = matlab_ge_sm(3.0, A);
    RT_NEAR(rt_at(r, 1, 0), 1.0, 0.0, "ge 3>=3");

    r = matlab_lt_sm(3.0, A);
    RT_NEAR(rt_at(r, 1, 1), 1.0, 0.0, "lt 3<4");

    r = matlab_le_sm(3.0, A);
    RT_NEAR(rt_at(r, 1, 0), 1.0, 0.0, "le 3<=3");

    r = matlab_eq_sm(2.0, A);
    RT_NEAR(rt_at(r, 0, 1), 1.0, 0.0, "eq 2==2");

    r = matlab_ne_sm(2.0, A);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 0.0, "ne 2!=1");
}

int main(void) {
    fprintf(stderr, "test_elementwise:\n");
    RT_RUN(test_binary_mm);
    RT_RUN(test_binary_ms);
    RT_RUN(test_binary_sm);
    RT_RUN(test_neg_m);
    RT_RUN(test_compare_mm);
    RT_RUN(test_compare_ms);
    RT_RUN(test_compare_sm);
    RT_DONE();
}
