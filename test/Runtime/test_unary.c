/* Phase-1 catch-up: unary scalar + matrix math, plus type-cast
 * helpers (int8_s..uint64_s, double_s, logical_s).
 *
 * Targets the macro families:
 *   matlab_(exp|log|log2|log10|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|sqrt|abs|sign)_(s|m)
 *   matlab_(floor|ceil|round|fix)_(s|m)
 *   matlab_(int{8,16,32,64}_s|uint{8,16,32,64}_s|double_s|single_s|logical_s)
 *   matlab_atan2_(s|m), matlab_mod_s, matlab_rem_s
 *
 * Plus a few miscellaneous primitives that previously had 0% line
 * coverage: isempty, isequal, length, numel, ndims, end_of_dim,
 * mat_truth, mat_from_scalar, empty_mat, neg_c. */

#include "runtime_test.h"

double matlab_exp_s   (double x);
double matlab_log_s   (double x);
double matlab_log2_s  (double x);
double matlab_log10_s (double x);
double matlab_sin_s   (double x);
double matlab_cos_s   (double x);
double matlab_tan_s   (double x);
double matlab_asin_s  (double x);
double matlab_acos_s  (double x);
double matlab_atan_s  (double x);
double matlab_sinh_s  (double x);
double matlab_cosh_s  (double x);
double matlab_tanh_s  (double x);
double matlab_sqrt_s  (double x);
double matlab_abs_s   (double x);
double matlab_sign_s  (double x);
double matlab_floor_s (double x);
double matlab_ceil_s  (double x);
double matlab_round_s (double x);
double matlab_fix_s   (double x);
double matlab_atan2_s (double y, double x);
double matlab_mod_s   (double a, double b);
double matlab_rem_s   (double a, double b);

double matlab_int8_s   (double x);
double matlab_int16_s  (double x);
double matlab_int32_s  (double x);
double matlab_int64_s  (double x);
double matlab_uint8_s  (double x);
double matlab_uint16_s (double x);
double matlab_uint32_s (double x);
double matlab_uint64_s (double x);
double matlab_double_s (double x);
double matlab_single_s (double x);
double matlab_logical_s(double x);

matlab_mat *matlab_exp_m  (matlab_mat *A);
matlab_mat *matlab_log_m  (matlab_mat *A);
matlab_mat *matlab_log2_m (matlab_mat *A);
matlab_mat *matlab_log10_m(matlab_mat *A);
matlab_mat *matlab_sin_m  (matlab_mat *A);
matlab_mat *matlab_cos_m  (matlab_mat *A);
matlab_mat *matlab_tan_m  (matlab_mat *A);
matlab_mat *matlab_asin_m (matlab_mat *A);
matlab_mat *matlab_acos_m (matlab_mat *A);
matlab_mat *matlab_atan_m (matlab_mat *A);
matlab_mat *matlab_sinh_m (matlab_mat *A);
matlab_mat *matlab_cosh_m (matlab_mat *A);
matlab_mat *matlab_tanh_m (matlab_mat *A);
matlab_mat *matlab_sqrt_m (matlab_mat *A);
matlab_mat *matlab_abs_m  (matlab_mat *A);
matlab_mat *matlab_sign_m (matlab_mat *A);
matlab_mat *matlab_floor_m(matlab_mat *A);
matlab_mat *matlab_ceil_m (matlab_mat *A);
matlab_mat *matlab_round_m(matlab_mat *A);
matlab_mat *matlab_fix_m  (matlab_mat *A);
matlab_mat *matlab_atan2_m(matlab_mat *A, matlab_mat *B);

matlab_mat   *matlab_empty_mat(void);
matlab_mat   *matlab_mat_from_scalar(double x);
int8_t        matlab_mat_truth(matlab_mat *A);
double        matlab_isempty(matlab_mat *A);
double        matlab_isequal(matlab_mat *A, matlab_mat *B);
double        matlab_length(matlab_mat *A);
double        matlab_numel(matlab_mat *A);
double        matlab_ndims(matlab_mat *A);
double        matlab_size_dim(matlab_mat *A, double dim);
double        matlab_end_of_dim(matlab_mat *A, double dim);

matlab_mat_c *matlab_neg_c(matlab_mat_c *A);
matlab_mat_c *matlab_complex_scalar(double re, double im);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

#define M_PI_D 3.14159265358979323846

/* --- scalar math ---------------------------------------------------- */
static void test_scalar_exp_log_sqrt(void) {
    RT_NEAR(matlab_exp_s(0.0),     1.0,        1e-12, "exp(0)");
    RT_NEAR(matlab_exp_s(1.0),     2.718281828, 1e-9, "exp(1)");
    RT_NEAR(matlab_log_s(1.0),     0.0,        1e-12, "log(1)");
    RT_NEAR(matlab_log_s(2.718281828459045), 1.0, 1e-12, "log(e)");
    RT_NEAR(matlab_log2_s(8.0),    3.0,        1e-12, "log2(8)");
    RT_NEAR(matlab_log10_s(1000.0), 3.0,       1e-12, "log10(1000)");
    RT_NEAR(matlab_sqrt_s(16.0),   4.0,        1e-12, "sqrt(16)");
}

static void test_scalar_trig(void) {
    RT_NEAR(matlab_sin_s(0.0),     0.0,        1e-12, "sin(0)");
    RT_NEAR(matlab_cos_s(0.0),     1.0,        1e-12, "cos(0)");
    RT_NEAR(matlab_tan_s(0.0),     0.0,        1e-12, "tan(0)");
    RT_NEAR(matlab_sin_s(M_PI_D / 2), 1.0,     1e-12, "sin(pi/2)");
    RT_NEAR(matlab_asin_s(1.0),    M_PI_D / 2, 1e-12, "asin(1)");
    RT_NEAR(matlab_acos_s(1.0),    0.0,        1e-12, "acos(1)");
    RT_NEAR(matlab_atan_s(1.0),    M_PI_D / 4, 1e-12, "atan(1)");
    RT_NEAR(matlab_sinh_s(0.0),    0.0,        1e-12, "sinh(0)");
    RT_NEAR(matlab_cosh_s(0.0),    1.0,        1e-12, "cosh(0)");
    RT_NEAR(matlab_tanh_s(0.0),    0.0,        1e-12, "tanh(0)");
}

static void test_scalar_round_family(void) {
    RT_NEAR(matlab_floor_s( 1.7),  1.0, 0.0, "floor 1.7");
    RT_NEAR(matlab_floor_s(-1.7), -2.0, 0.0, "floor -1.7");
    RT_NEAR(matlab_ceil_s ( 1.2),  2.0, 0.0, "ceil 1.2");
    RT_NEAR(matlab_ceil_s (-1.2), -1.0, 0.0, "ceil -1.2");
    RT_NEAR(matlab_round_s( 1.5),  2.0, 0.0, "round 1.5");
    RT_NEAR(matlab_round_s(-1.5), -2.0, 0.0, "round -1.5");
    RT_NEAR(matlab_fix_s  ( 1.7),  1.0, 0.0, "fix 1.7 (toward 0)");
    RT_NEAR(matlab_fix_s  (-1.7), -1.0, 0.0, "fix -1.7");
}

static void test_scalar_misc(void) {
    RT_NEAR(matlab_abs_s(-3.0),  3.0, 0.0, "abs -3");
    RT_NEAR(matlab_sign_s(-3.0), -1.0, 0.0, "sign -3");
    RT_NEAR(matlab_sign_s( 3.0),  1.0, 0.0, "sign +3");
    RT_NEAR(matlab_sign_s( 0.0),  0.0, 0.0, "sign 0");
    RT_NEAR(matlab_atan2_s(1.0, 1.0), M_PI_D / 4, 1e-12, "atan2(1,1)");
    RT_NEAR(matlab_mod_s(7.0, 3.0),  1.0, 1e-12, "mod 7,3");
    RT_NEAR(matlab_rem_s(7.0, 3.0),  1.0, 1e-12, "rem 7,3");
    /* mod and rem differ for negatives. */
    RT_NEAR(matlab_mod_s(-1.0, 3.0), 2.0, 1e-12, "mod -1,3");
    RT_NEAR(matlab_rem_s(-1.0, 3.0), -1.0, 1e-12, "rem -1,3");
}

/* --- type-cast helpers --------------------------------------------- */
static void test_int_casts_clamp(void) {
    /* int8 range = [-128, 127]. The runtime uses trunc (toward zero),
     * not round-half-to-even — int8(5.5) = 5, not 6. */
    RT_NEAR(matlab_int8_s ( 200.0),  127.0, 0.0, "int8 sat high");
    RT_NEAR(matlab_int8_s (-200.0), -128.0, 0.0, "int8 sat low");
    RT_NEAR(matlab_int8_s (   5.5),   5.0,  0.0, "int8 trunc 5.5");

    RT_NEAR(matlab_int16_s(40000.0), 32767.0, 0.0, "int16 sat high");
    RT_NEAR(matlab_int32_s(1.0e10), 2147483647.0, 1.0, "int32 sat high");
    RT_NEAR(matlab_int64_s(-3.5),  -3.0, 0.0, "int64 trunc -3.5");

    RT_NEAR(matlab_uint8_s ( 200.0),  200.0, 0.0, "uint8 in range");
    RT_NEAR(matlab_uint8_s (-1.0),    0.0,   0.0, "uint8 sat low");
    RT_NEAR(matlab_uint8_s ( 300.0),  255.0, 0.0, "uint8 sat high");
    RT_NEAR(matlab_uint16_s(70000.0), 65535.0, 0.0, "uint16 sat");
    RT_NEAR(matlab_uint32_s(-1.0),     0.0,   0.0, "uint32 sat");
    RT_NEAR(matlab_uint64_s(-1.0),     0.0,   0.0, "uint64 sat");

    RT_NEAR(matlab_double_s ( 3.14), 3.14, 1e-12, "double identity");
    RT_NEAR(matlab_single_s ( 3.0),  3.0,  1e-7,  "single approx");
    RT_NEAR(matlab_logical_s( 5.0),  1.0,  0.0,   "logical nonzero");
    RT_NEAR(matlab_logical_s( 0.0),  0.0,  0.0,   "logical zero");
}

/* --- matrix unary ops ---------------------------------------------- */
static void test_matrix_unary(void) {
    double a[]  = {1, 4, 9, 16};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r;

    r = matlab_exp_m(A);
    RT_NEAR(rt_at(r, 0, 0), matlab_exp_s(1.0), 1e-9, "exp_m");

    r = matlab_log_m(A);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "log(1)=0");
    RT_NEAR(rt_at(r, 0, 1), matlab_log_s(4.0), 1e-12, "log(4)");

    r = matlab_log2_m(A);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "log2(1)");
    RT_NEAR(rt_at(r, 1, 1), 4.0, 1e-12, "log2(16)");

    r = matlab_log10_m(A);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "log10(1)");

    r = matlab_sqrt_m(A);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 1e-12, "sqrt(1)");
    RT_NEAR(rt_at(r, 1, 1), 4.0, 1e-12, "sqrt(16)");

    r = matlab_abs_m(A);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 1e-12, "abs(+x)");
}

static void test_matrix_trig(void) {
    double a[] = {0, M_PI_D / 2, M_PI_D, 0};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r;

    r = matlab_sin_m(A);
    RT_NEAR(rt_at(r, 0, 1), 1.0, 1e-12, "sin(pi/2)");

    r = matlab_cos_m(A);
    RT_NEAR(rt_at(r, 0, 0), 1.0, 1e-12, "cos(0)");

    r = matlab_tan_m(A);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "tan(0)");

    /* Domain inputs for asin/acos in [-1, 1]. */
    double b[] = {0, 0.5, -0.5, 1.0};
    matlab_mat *B = mk(b, 2, 2);
    r = matlab_asin_m(B);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "asin(0)");

    r = matlab_acos_m(B);
    RT_NEAR(rt_at(r, 0, 0), M_PI_D / 2, 1e-12, "acos(0)");

    r = matlab_atan_m(B);
    RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "atan(0)");

    /* Hyperbolic family — origin behaviour. */
    double zeros[] = {0, 0, 0, 0};
    matlab_mat *Z = mk(zeros, 2, 2);
    r = matlab_sinh_m(Z); RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "sinh(0)");
    r = matlab_cosh_m(Z); RT_NEAR(rt_at(r, 0, 0), 1.0, 1e-12, "cosh(0)");
    r = matlab_tanh_m(Z); RT_NEAR(rt_at(r, 0, 0), 0.0, 1e-12, "tanh(0)");
}

static void test_matrix_round_and_sign(void) {
    double a[] = {1.7, -1.7, 1.5, -0.3};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *r;

    r = matlab_floor_m(A);
    RT_NEAR(rt_at(r, 0, 0),  1.0, 0.0, "floor 1.7");
    RT_NEAR(rt_at(r, 0, 1), -2.0, 0.0, "floor -1.7");

    r = matlab_ceil_m(A);
    RT_NEAR(rt_at(r, 0, 0), 2.0, 0.0, "ceil 1.7");

    r = matlab_round_m(A);
    RT_NEAR(rt_at(r, 1, 0), 2.0, 0.0, "round 1.5");

    r = matlab_fix_m(A);
    RT_NEAR(rt_at(r, 0, 1), -1.0, 0.0, "fix -1.7");

    r = matlab_sign_m(A);
    RT_NEAR(rt_at(r, 0, 0),  1.0, 0.0, "sign +1.7");
    RT_NEAR(rt_at(r, 0, 1), -1.0, 0.0, "sign -1.7");
    RT_NEAR(rt_at(r, 1, 1), -1.0, 0.0, "sign -0.3");
}

static void test_matrix_atan2(void) {
    double a[] = {1, 0, 1, 0};
    double b[] = {1, 1, 0, 0};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *B = mk(b, 2, 2);
    matlab_mat *r = matlab_atan2_m(A, B);
    RT_NEAR(rt_at(r, 0, 0), M_PI_D / 4, 1e-12, "atan2(1,1)");
    RT_NEAR(rt_at(r, 0, 1), 0.0,        1e-12, "atan2(0,1)");
}

/* --- predicates / shape queries ----------------------------------- */
static void test_predicates(void) {
    double a[] = {1, 2, 3, 4};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *E = matlab_empty_mat();
    matlab_mat *S = matlab_mat_from_scalar(7.0);

    RT_NEAR(matlab_isempty(A), 0.0, 0.0, "non-empty");
    RT_NEAR(matlab_isempty(E), 1.0, 0.0, "empty");
    RT_NEAR(matlab_length(A),  2.0, 0.0, "length 2x2");
    RT_NEAR(matlab_numel(A),   4.0, 0.0, "numel");
    RT_NEAR(matlab_ndims(A),   2.0, 0.0, "ndims");
    RT_NEAR(matlab_size_dim(A, 1.0), 2.0, 0.0, "size dim1");
    RT_NEAR(matlab_size_dim(A, 2.0), 2.0, 0.0, "size dim2");
    RT_NEAR(matlab_end_of_dim(A, 1.0), 2.0, 0.0, "end dim1");

    /* mat_from_scalar yields 1x1. */
    RT_NEAR(matlab_numel(S), 1.0, 0.0, "scalar numel");
    RT_NEAR(rt_at(S, 0, 0), 7.0, 0.0, "scalar value");

    /* mat_truth: nonzero non-empty → 1, zero or empty → 0. */
    RT_NEAR((double)matlab_mat_truth(A), 1.0, 0.0, "truth nonzero");
    double z[] = {0, 0, 0, 0};
    matlab_mat *Z = mk(z, 2, 2);
    RT_NEAR((double)matlab_mat_truth(Z), 0.0, 0.0, "truth all zeros");
    RT_NEAR((double)matlab_mat_truth(E), 0.0, 0.0, "truth empty");
}

static void test_isequal(void) {
    double a[] = {1, 2, 3, 4};
    double b[] = {1, 2, 3, 4};
    double c[] = {1, 2, 3, 5};
    matlab_mat *A = mk(a, 2, 2);
    matlab_mat *B = mk(b, 2, 2);
    matlab_mat *C = mk(c, 2, 2);
    RT_NEAR(matlab_isequal(A, B), 1.0, 0.0, "equal");
    RT_NEAR(matlab_isequal(A, C), 0.0, 0.0, "not equal");
}

/* --- complex helpers ---------------------------------------------- */
static void test_complex_helpers(void) {
    /* complex_scalar + neg_c. */
    matlab_mat_c *Z = matlab_complex_scalar(3.0, 4.0);
    RT_NEAR(rt_c_re(Z, 0, 0),  3.0, 1e-12, "Z.re");
    RT_NEAR(rt_c_im(Z, 0, 0),  4.0, 1e-12, "Z.im");
    matlab_mat_c *N = matlab_neg_c(Z);
    RT_NEAR(rt_c_re(N, 0, 0), -3.0, 1e-12, "neg.re");
    RT_NEAR(rt_c_im(N, 0, 0), -4.0, 1e-12, "neg.im");
}

int main(void) {
    fprintf(stderr, "test_unary:\n");
    RT_RUN(test_scalar_exp_log_sqrt);
    RT_RUN(test_scalar_trig);
    RT_RUN(test_scalar_round_family);
    RT_RUN(test_scalar_misc);
    RT_RUN(test_int_casts_clamp);
    RT_RUN(test_matrix_unary);
    RT_RUN(test_matrix_trig);
    RT_RUN(test_matrix_round_and_sign);
    RT_RUN(test_matrix_atan2);
    RT_RUN(test_predicates);
    RT_RUN(test_isequal);
    RT_RUN(test_complex_helpers);
    RT_DONE();
}
