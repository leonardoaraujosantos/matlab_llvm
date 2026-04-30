/* Phase-1 catch-up: matlab_mat_i64 / matlab_mat_u64 array family.
 *
 * Targets the fi-array primitives generated for fixed-point integer
 * matrices: zeros / from_buf / from_scalar constructors, predicates
 * (length / numel / size_dim / rows / cols), scalar + slice index,
 * in-place set, fill, sum, concat. Every entry of this family was
 * 0%-line-covered before this file landed. */

#include "runtime_test.h"

matlab_mat_i64 *matlab_mat_i64_zeros        (double rows, double cols);
matlab_mat_i64 *matlab_mat_i64_from_buf     (const int64_t *buf, double r, double c);
matlab_mat_i64 *matlab_mat_i64_from_scalar  (int64_t v);
double          matlab_mat_i64_length       (matlab_mat_i64 *A);
double          matlab_mat_i64_numel        (matlab_mat_i64 *A);
double          matlab_mat_i64_size_dim     (matlab_mat_i64 *A, double d);
int64_t         matlab_mat_i64_rows         (matlab_mat_i64 *A);
int64_t         matlab_mat_i64_cols         (matlab_mat_i64 *A);
int64_t         matlab_mat_i64_subscript1_s (matlab_mat_i64 *A, double i);
int64_t         matlab_mat_i64_subscript2_s (matlab_mat_i64 *A, double i, double j);
matlab_mat_i64 *matlab_mat_i64_slice1       (matlab_mat_i64 *A, matlab_mat *idx);
void            matlab_mat_i64_set1_s       (matlab_mat_i64 *A, double i, int64_t v);
void            matlab_mat_i64_fill         (matlab_mat_i64 *A, int64_t v);
matlab_mat_i64 *matlab_mat_i64_concat_row   (matlab_mat_i64 *A, matlab_mat_i64 *B);
int64_t         matlab_mat_i64_sum          (matlab_mat_i64 *A);

matlab_mat_u64 *matlab_mat_u64_zeros        (double rows, double cols);
matlab_mat_u64 *matlab_mat_u64_from_buf     (const uint64_t *buf, double r, double c);
matlab_mat_u64 *matlab_mat_u64_from_scalar  (uint64_t v);
double          matlab_mat_u64_length       (matlab_mat_u64 *A);
double          matlab_mat_u64_numel        (matlab_mat_u64 *A);
double          matlab_mat_u64_size_dim     (matlab_mat_u64 *A, double d);
uint64_t        matlab_mat_u64_subscript1_s (matlab_mat_u64 *A, double i);
uint64_t        matlab_mat_u64_subscript2_s (matlab_mat_u64 *A, double i, double j);
matlab_mat_u64 *matlab_mat_u64_slice1       (matlab_mat_u64 *A, matlab_mat *idx);
void            matlab_mat_u64_set1_s       (matlab_mat_u64 *A, double i, uint64_t v);
void            matlab_mat_u64_fill         (matlab_mat_u64 *A, uint64_t v);
matlab_mat_u64 *matlab_mat_u64_concat_row   (matlab_mat_u64 *A, matlab_mat_u64 *B);
uint64_t        matlab_mat_u64_sum          (matlab_mat_u64 *A);

/* --- signed (int64) ----------------------------------------------- */
static void test_i64_zeros_and_predicates(void) {
    matlab_mat_i64 *A = matlab_mat_i64_zeros(3, 4);
    RT_NEAR(matlab_mat_i64_numel(A),     12.0, 0.0, "numel");
    RT_NEAR(matlab_mat_i64_size_dim(A, 1), 3.0, 0.0, "size dim1");
    RT_NEAR(matlab_mat_i64_size_dim(A, 2), 4.0, 0.0, "size dim2");
    RT_NEAR((double)matlab_mat_i64_rows(A), 3.0, 0.0, "rows");
    RT_NEAR((double)matlab_mat_i64_cols(A), 4.0, 0.0, "cols");
    /* length(A) = max(rows, cols) per MATLAB convention. */
    RT_NEAR(matlab_mat_i64_length(A),    4.0, 0.0, "length");
    /* All zeros. */
    RT_NEAR((double)matlab_mat_i64_subscript2_s(A, 1, 1), 0.0, 0.0, "[1,1]=0");
}

static void test_i64_from_buf_subscript(void) {
    int64_t b[] = {10, 20, 30, 40, 50, 60};
    matlab_mat_i64 *A = matlab_mat_i64_from_buf(b, 2, 3);
    /* MATLAB 1-based subscript. */
    RT_NEAR((double)matlab_mat_i64_subscript2_s(A, 1, 1), 10.0, 0.0, "[1,1]");
    RT_NEAR((double)matlab_mat_i64_subscript2_s(A, 1, 3), 30.0, 0.0, "[1,3]");
    RT_NEAR((double)matlab_mat_i64_subscript2_s(A, 2, 2), 50.0, 0.0, "[2,2]");
    /* 1-D linear subscript: column-major would give 40 for index 4,
     * row-major would give 40 for index 4. The runtime uses row-major. */
    RT_NEAR((double)matlab_mat_i64_subscript1_s(A, 4), 40.0, 0.0, "linear [4]");
}

static void test_i64_from_scalar(void) {
    matlab_mat_i64 *S = matlab_mat_i64_from_scalar(-42);
    RT_NEAR(matlab_mat_i64_numel(S),    1.0, 0.0, "scalar numel");
    RT_NEAR((double)matlab_mat_i64_subscript2_s(S, 1, 1), -42.0, 0.0, "value");
}

static void test_i64_set_and_fill(void) {
    matlab_mat_i64 *A = matlab_mat_i64_zeros(2, 3);
    matlab_mat_i64_set1_s(A, 3, 99);
    RT_NEAR((double)matlab_mat_i64_subscript1_s(A, 3), 99.0, 0.0, "set1");
    matlab_mat_i64_fill(A, 7);
    for (int i = 1; i <= 6; ++i)
        RT_NEAR((double)matlab_mat_i64_subscript1_s(A, i), 7.0, 0.0, "fill");
}

static void test_i64_sum(void) {
    int64_t b[] = {1, 2, 3, 4, 5};
    matlab_mat_i64 *A = matlab_mat_i64_from_buf(b, 1, 5);
    RT_NEAR((double)matlab_mat_i64_sum(A), 15.0, 0.0, "sum 1..5");
}

static void test_i64_concat_row(void) {
    int64_t a[] = {1, 2, 3};
    int64_t b[] = {4, 5};
    matlab_mat_i64 *A = matlab_mat_i64_from_buf(a, 1, 3);
    matlab_mat_i64 *B = matlab_mat_i64_from_buf(b, 1, 2);
    matlab_mat_i64 *C = matlab_mat_i64_concat_row(A, B);
    RT_NEAR(matlab_mat_i64_numel(C), 5.0, 0.0, "concat numel");
    RT_NEAR((double)matlab_mat_i64_subscript1_s(C, 1), 1.0, 0.0, "[1]");
    RT_NEAR((double)matlab_mat_i64_subscript1_s(C, 5), 5.0, 0.0, "[5]");
}

static void test_i64_slice1(void) {
    int64_t b[] = {10, 20, 30, 40, 50};
    matlab_mat_i64 *A = matlab_mat_i64_from_buf(b, 1, 5);
    /* Pick indices [2, 4]. */
    double idx[] = {2, 4};
    matlab_mat *I = matlab_mat_from_buf(idx, 1.0, 2.0);
    matlab_mat_i64 *S = matlab_mat_i64_slice1(A, I);
    RT_NEAR(matlab_mat_i64_numel(S), 2.0, 0.0, "slice numel");
    RT_NEAR((double)matlab_mat_i64_subscript1_s(S, 1), 20.0, 0.0, "slice[1]");
    RT_NEAR((double)matlab_mat_i64_subscript1_s(S, 2), 40.0, 0.0, "slice[2]");
}

/* --- unsigned (uint64) -------------------------------------------- */
static void test_u64_zeros_and_predicates(void) {
    matlab_mat_u64 *A = matlab_mat_u64_zeros(2, 5);
    RT_NEAR(matlab_mat_u64_numel(A),    10.0, 0.0, "u numel");
    RT_NEAR(matlab_mat_u64_size_dim(A, 1), 2.0, 0.0, "u size dim1");
    RT_NEAR(matlab_mat_u64_size_dim(A, 2), 5.0, 0.0, "u size dim2");
    RT_NEAR(matlab_mat_u64_length(A),    5.0, 0.0, "u length");
}

static void test_u64_from_buf_subscript(void) {
    uint64_t b[] = {10, 20, 30, 40, 50, 60};
    matlab_mat_u64 *A = matlab_mat_u64_from_buf(b, 2, 3);
    RT_NEAR((double)matlab_mat_u64_subscript2_s(A, 1, 1), 10.0, 0.0, "u [1,1]");
    RT_NEAR((double)matlab_mat_u64_subscript2_s(A, 2, 2), 50.0, 0.0, "u [2,2]");
    RT_NEAR((double)matlab_mat_u64_subscript1_s(A, 6),    60.0, 0.0, "u linear");
}

static void test_u64_set_fill_concat_sum(void) {
    matlab_mat_u64 *A = matlab_mat_u64_zeros(1, 3);
    matlab_mat_u64_set1_s(A, 2, 7);
    RT_NEAR((double)matlab_mat_u64_subscript1_s(A, 2), 7.0, 0.0, "u set");
    matlab_mat_u64_fill(A, 4);
    RT_NEAR((double)matlab_mat_u64_subscript1_s(A, 1), 4.0, 0.0, "u fill[1]");
    RT_NEAR((double)matlab_mat_u64_subscript1_s(A, 3), 4.0, 0.0, "u fill[3]");

    uint64_t b[] = {1, 2};
    matlab_mat_u64 *B = matlab_mat_u64_from_buf(b, 1, 2);
    matlab_mat_u64 *C = matlab_mat_u64_concat_row(A, B);
    RT_NEAR(matlab_mat_u64_numel(C), 5.0, 0.0, "u concat numel");
    /* sum [4 4 4 1 2] = 15 */
    RT_NEAR((double)matlab_mat_u64_sum(C), 15.0, 0.0, "u sum");
}

static void test_u64_from_scalar_and_slice1(void) {
    matlab_mat_u64 *S = matlab_mat_u64_from_scalar(99);
    RT_NEAR((double)matlab_mat_u64_subscript1_s(S, 1), 99.0, 0.0, "u scalar");

    uint64_t b[] = {10, 20, 30, 40, 50};
    matlab_mat_u64 *A = matlab_mat_u64_from_buf(b, 1, 5);
    double idx[] = {3};
    matlab_mat *I = matlab_mat_from_buf(idx, 1.0, 1.0);
    matlab_mat_u64 *Sl = matlab_mat_u64_slice1(A, I);
    RT_NEAR(matlab_mat_u64_numel(Sl), 1.0, 0.0, "u slice numel");
    RT_NEAR((double)matlab_mat_u64_subscript1_s(Sl, 1), 30.0, 0.0, "u slice val");
}

int main(void) {
    fprintf(stderr, "test_fi_arrays:\n");
    RT_RUN(test_i64_zeros_and_predicates);
    RT_RUN(test_i64_from_buf_subscript);
    RT_RUN(test_i64_from_scalar);
    RT_RUN(test_i64_set_and_fill);
    RT_RUN(test_i64_sum);
    RT_RUN(test_i64_concat_row);
    RT_RUN(test_i64_slice1);
    RT_RUN(test_u64_zeros_and_predicates);
    RT_RUN(test_u64_from_buf_subscript);
    RT_RUN(test_u64_set_fill_concat_sum);
    RT_RUN(test_u64_from_scalar_and_slice1);
    RT_DONE();
}
