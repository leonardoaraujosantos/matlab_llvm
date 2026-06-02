/* Phase-1 catch-up: direct unit tests for the struct + cell helpers
 * the workspace mirror and field-access lowering depend on. None of
 * these had direct test coverage before; the .m integration suite
 * exercises them via lowered code, but a unit-level assertion suite
 * catches regressions in the struct field-find / reserve / dedup
 * primitives faster. */

#include "runtime_test.h"

matlab_struct *matlab_struct_new(void);
void           matlab_struct_set_f64(matlab_struct *s, const char *name,
                                      int64_t len, double v);
void           matlab_struct_set_mat(matlab_struct *s, const char *name,
                                      int64_t len, matlab_mat *m);
double         matlab_struct_get_f64(matlab_struct *s, const char *name,
                                      int64_t len);
matlab_mat    *matlab_struct_get_mat(matlab_struct *s, const char *name,
                                      int64_t len);
double         matlab_struct_has_field(matlab_struct *s, const char *name,
                                        int64_t len);
matlab_struct *matlab_struct_get_child_struct(matlab_struct *s,
                                              const char *name, int64_t len);
matlab_struct *matlab_struct_rmfield(matlab_struct *s, const char *name,
                                      int64_t len);
/* #128: property read off an empty-matrix value. */
matlab_mat    *matlab_obj_get_mat(void *o, const char *name, int64_t len);

matlab_cell   *matlab_cell_new(double n);
void           matlab_cell_set_f64(matlab_cell *c, double i1, double v);
void           matlab_cell_set_mat(matlab_cell *c, double i1, matlab_mat *m);
double         matlab_cell_get_f64(matlab_cell *c, double i1);
matlab_mat    *matlab_cell_get_mat(matlab_cell *c, double i1);
double         matlab_cell_numel(matlab_cell *c);
double         matlab_iscell(matlab_cell *c);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* --- struct -------------------------------------------------------- */
static void test_struct_set_get_f64(void) {
    matlab_struct *s = matlab_struct_new();
    matlab_struct_set_f64(s, "x", 1, 42.0);
    RT_NEAR(matlab_struct_get_f64(s, "x", 1), 42.0, 0.0, "round-trip f64");
    /* Missing field: 0.0 by convention. */
    RT_NEAR(matlab_struct_get_f64(s, "missing", 7), 0.0, 0.0, "missing");
}

static void test_struct_overwrite_field(void) {
    /* Setting the same field twice replaces, not appends. */
    matlab_struct *s = matlab_struct_new();
    matlab_struct_set_f64(s, "k", 1, 1.0);
    matlab_struct_set_f64(s, "k", 1, 2.0);
    RT_NEAR(matlab_struct_get_f64(s, "k", 1), 2.0, 0.0, "overwrite");
}

static void test_struct_set_get_mat(void) {
    matlab_struct *s = matlab_struct_new();
    double a[] = {1, 2, 3, 4};
    matlab_mat *M = mk(a, 2, 2);
    matlab_struct_set_mat(s, "m", 1, M);
    matlab_mat *out = matlab_struct_get_mat(s, "m", 1);
    RT_CHECK(out != NULL, "mat round-trip");
    RT_NEAR(rt_at(out, 1, 1), 4.0, 1e-12, "mat values");
}

static void test_struct_has_field(void) {
    matlab_struct *s = matlab_struct_new();
    matlab_struct_set_f64(s, "present", 7, 1.0);
    RT_NEAR(matlab_struct_has_field(s, "present", 7), 1.0, 0.0, "exists");
    RT_NEAR(matlab_struct_has_field(s, "absent", 6),  0.0, 0.0, "absent");
}

static void test_struct_rmfield(void) {
    matlab_struct *s = matlab_struct_new();
    matlab_struct_set_f64(s, "a", 1, 1.0);
    matlab_struct_set_f64(s, "b", 1, 2.0);
    matlab_struct_set_f64(s, "c", 1, 3.0);
    matlab_struct_rmfield(s, "b", 1);
    RT_NEAR(matlab_struct_has_field(s, "a", 1), 1.0, 0.0, "a kept");
    RT_NEAR(matlab_struct_has_field(s, "b", 1), 0.0, 0.0, "b removed");
    RT_NEAR(matlab_struct_has_field(s, "c", 1), 1.0, 0.0, "c kept");
    /* Removing a non-existent field is a no-op. */
    matlab_struct_rmfield(s, "missing", 7);
    RT_NEAR(matlab_struct_has_field(s, "a", 1), 1.0, 0.0, "still a");
}

static void test_struct_get_child_struct(void) {
    /* Access s.a.b.c — three levels deep. Each call materialises an
     * intermediate child struct on first touch. */
    matlab_struct *s = matlab_struct_new();
    matlab_struct *a = matlab_struct_get_child_struct(s, "a", 1);
    matlab_struct_set_f64(a, "x", 1, 7.0);
    /* Re-accessing returns the same child (so the value persists). */
    matlab_struct *a2 = matlab_struct_get_child_struct(s, "a", 1);
    RT_NEAR(matlab_struct_get_f64(a2, "x", 1), 7.0, 0.0, "nested f64");
    RT_CHECK(a == a2, "same child struct on re-access");
}

static void test_struct_grows_past_initial_capacity(void) {
    /* Initial capacity is 4; add 16 fields and verify they all
     * round-trip. Exercises the doubling-capacity branch in
     * struct_reserve. */
    matlab_struct *s = matlab_struct_new();
    for (int i = 0; i < 16; ++i) {
        char name[8];
        int n = snprintf(name, sizeof(name), "f%d", i);
        matlab_struct_set_f64(s, name, n, (double)i);
    }
    for (int i = 0; i < 16; ++i) {
        char name[8];
        int n = snprintf(name, sizeof(name), "f%d", i);
        RT_NEAR(matlab_struct_get_f64(s, name, n), (double)i, 0.0,
                "field round-trip past initial capacity");
    }
}

/* --- cell ---------------------------------------------------------- */
static void test_cell_set_get_f64(void) {
    matlab_cell *c = matlab_cell_new(4);
    matlab_cell_set_f64(c, 1, 10.0);
    matlab_cell_set_f64(c, 2, 20.0);
    matlab_cell_set_f64(c, 3, 30.0);
    matlab_cell_set_f64(c, 4, 40.0);
    RT_NEAR(matlab_cell_get_f64(c, 1), 10.0, 0.0, "c{1}");
    RT_NEAR(matlab_cell_get_f64(c, 4), 40.0, 0.0, "c{4}");
}

static void test_cell_set_get_mat(void) {
    matlab_cell *c = matlab_cell_new(2);
    double a[] = {1, 2};
    matlab_mat *M = mk(a, 1, 2);
    matlab_cell_set_mat(c, 1, M);
    matlab_mat *out = matlab_cell_get_mat(c, 1);
    RT_NEAR(rt_at(out, 0, 0), 1.0, 1e-12, "cell mat[0]");
    RT_NEAR(rt_at(out, 0, 1), 2.0, 1e-12, "cell mat[1]");
}

static void test_cell_numel(void) {
    /* numel tracks the high-water mark of indexed positions, not the
     * initial capacity. matlab_cell_new(7) gives an empty cell; numel
     * grows as indices are written. */
    matlab_cell *c = matlab_cell_new(7);
    matlab_cell_set_f64(c, 5, 1.0);
    RT_NEAR(matlab_cell_numel(c), 5.0, 0.0, "numel after set");
    matlab_cell_set_f64(c, 7, 2.0);
    RT_NEAR(matlab_cell_numel(c), 7.0, 0.0, "numel grows to high index");
}

static void test_iscell(void) {
    matlab_cell *c = matlab_cell_new(3);
    RT_NEAR(matlab_iscell(c), 1.0, 0.0, "real cell");
    RT_NEAR(matlab_iscell(NULL), 0.0, 0.0, "null is not a cell");
}

static void test_cell_grows_past_initial_capacity(void) {
    /* Cells start at small capacity and grow on demand. */
    matlab_cell *c = matlab_cell_new(20);
    for (int i = 1; i <= 20; ++i) matlab_cell_set_f64(c, (double)i, (double)i * 0.5);
    for (int i = 1; i <= 20; ++i)
        RT_NEAR(matlab_cell_get_f64(c, (double)i), (double)i * 0.5, 1e-12,
                "cell round-trip across grow");
}

/* #128: a property read off an EMPTY matrix (the non-NULL empty matlab_mat
 * that matlab_struct_get_mat returns for a missing field, e.g.
 * `RF.ModeShapes.Magnitude` where ModeShapes is absent) must not be walked as
 * an obj.  Before the fix, the empty mat's zero `rows` word landed at the
 * struct `names` pointer offset, so struct_find_field dereferenced
 * ((char**)NULL)[i] and crashed at a near-NULL address. */
static void test_obj_get_mat_on_empty(void) {
    double dummy = 0.0;
    matlab_mat *empty = mk(&dummy, 0, 0);   /* rows==0, cols==0 by construction */
    matlab_mat *r = matlab_obj_get_mat((void *)empty, "Magnitude", 9);
    RT_CHECK(r != NULL, "obj prop read on empty mat returns non-null (no crash)");
}

int main(void) {
    fprintf(stderr, "test_struct_cell:\n");
    RT_RUN(test_struct_set_get_f64);
    RT_RUN(test_obj_get_mat_on_empty);
    RT_RUN(test_struct_overwrite_field);
    RT_RUN(test_struct_set_get_mat);
    RT_RUN(test_struct_has_field);
    RT_RUN(test_struct_rmfield);
    RT_RUN(test_struct_get_child_struct);
    RT_RUN(test_struct_grows_past_initial_capacity);
    RT_RUN(test_cell_set_get_f64);
    RT_RUN(test_cell_set_get_mat);
    RT_RUN(test_cell_numel);
    RT_RUN(test_iscell);
    RT_RUN(test_cell_grows_past_initial_capacity);
    RT_DONE();
}
