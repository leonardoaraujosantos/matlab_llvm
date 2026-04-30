/* Tiny dependency-free assertion macros for the direct runtime suite.
 * Each test_*.c file links runtime/matlab_runtime.c directly (no JIT, no
 * MATLAB frontend). Failures are accumulated and the process exits with
 * non-zero on any miss.
 *
 * The matlab_mat / matlab_mat_c layouts mirror runtime/matlab_runtime.c
 * verbatim. Public ABI clients are not allowed to peek at these fields,
 * but the runtime's own tests are. If the layout ever changes, mirror
 * the change here. */

#ifndef RUNTIME_TEST_H
#define RUNTIME_TEST_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matlab_runtime.h"

/* Functions defined in runtime/matlab_runtime.c but not present in
 * matlab_runtime.h (the public header is scoped to the JIT-emitted
 * ABI; some entries are reached only through the lowering layer's
 * name strings). The runtime tests need them, so forward-declare
 * locally. If the runtime header gains these declarations later,
 * delete the duplicates here. */
matlab_mat *matlab_fliplr   (matlab_mat *A);
matlab_mat *matlab_flipud   (matlab_mat *A);
matlab_mat *matlab_flip     (matlab_mat *A);
matlab_mat *matlab_rot90    (matlab_mat *A);
matlab_mat *matlab_cumsum   (matlab_mat *A);
matlab_mat *matlab_cumprod  (matlab_mat *A);
matlab_mat *matlab_sort     (matlab_mat *A);
matlab_mat *matlab_unique   (matlab_mat *A);
matlab_mat *matlab_ismember (matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_sum      (matlab_mat *A);
matlab_mat *matlab_prod     (matlab_mat *A);
matlab_mat *matlab_mean     (matlab_mat *A);
matlab_mat *matlab_min      (matlab_mat *A);
matlab_mat *matlab_max      (matlab_mat *A);

/* Mirrors the layout in runtime/matlab_runtime.c (line ~187). Tests use
 * this to read elements out of returned matrices. Changing the runtime
 * layout requires updating this header. */
struct rt_test_mat_layout {
    double *data;
    int64_t rows;
    int64_t cols;
};

/* Mirrors `struct matlab_mat_c` in runtime/matlab_runtime.c (line ~6488). */
struct rt_test_matc_layout {
    uint32_t magic;
    uint32_t _pad;
    double *re;
    double *im;
    int64_t rows;
    int64_t cols;
};

static inline double  rt_at (matlab_mat *A, int64_t i, int64_t j) {
    struct rt_test_mat_layout *m = (struct rt_test_mat_layout *)A;
    return m->data[i * m->cols + j];
}
static inline int64_t rt_rows(matlab_mat *A) {
    return ((struct rt_test_mat_layout *)A)->rows;
}
static inline int64_t rt_cols(matlab_mat *A) {
    return ((struct rt_test_mat_layout *)A)->cols;
}
static inline double *rt_data(matlab_mat *A) {
    return ((struct rt_test_mat_layout *)A)->data;
}
static inline void rt_free(matlab_mat *A) {
    if (!A) return;
    struct rt_test_mat_layout *m = (struct rt_test_mat_layout *)A;
    free(m->data); free(m);
}

static inline double  rt_c_re(matlab_mat_c *A, int64_t i, int64_t j) {
    struct rt_test_matc_layout *m = (struct rt_test_matc_layout *)A;
    return m->re[i * m->cols + j];
}
static inline double  rt_c_im(matlab_mat_c *A, int64_t i, int64_t j) {
    struct rt_test_matc_layout *m = (struct rt_test_matc_layout *)A;
    return m->im[i * m->cols + j];
}
static inline int64_t rt_c_rows(matlab_mat_c *A) {
    return ((struct rt_test_matc_layout *)A)->rows;
}
static inline int64_t rt_c_cols(matlab_mat_c *A) {
    return ((struct rt_test_matc_layout *)A)->cols;
}
static inline void rt_c_free(matlab_mat_c *A) {
    if (!A) return;
    struct rt_test_matc_layout *m = (struct rt_test_matc_layout *)A;
    free(m->re); free(m->im); free(m);
}

static int rt_failures = 0;
static int rt_total    = 0;

#define RT_RUN(name)                                                       \
    do {                                                                   \
        fprintf(stderr, "  %s\n", #name);                                  \
        name();                                                            \
    } while (0)

#define RT_CHECK(cond, msg)                                                \
    do {                                                                   \
        ++rt_total;                                                        \
        if (!(cond)) {                                                     \
            ++rt_failures;                                                 \
            fprintf(stderr, "    FAIL %s:%d: %s — %s\n",                   \
                    __FILE__, __LINE__, #cond, (msg));                     \
        }                                                                  \
    } while (0)

#define RT_NEAR(actual, expected, tol, msg)                                \
    do {                                                                   \
        ++rt_total;                                                        \
        double _a = (actual), _b = (expected);                             \
        if (!(fabs(_a - _b) <= (tol))) {                                   \
            ++rt_failures;                                                 \
            fprintf(stderr,                                                \
                    "    FAIL %s:%d: |%g - %g| > %g — %s\n",               \
                    __FILE__, __LINE__, _a, _b, (double)(tol), (msg));     \
        }                                                                  \
    } while (0)

#define RT_DONE()                                                          \
    do {                                                                   \
        fprintf(stderr, "%s: %d/%d passed\n", __FILE__,                    \
                rt_total - rt_failures, rt_total);                         \
        return rt_failures == 0 ? 0 : 1;                                   \
    } while (0)

#endif
