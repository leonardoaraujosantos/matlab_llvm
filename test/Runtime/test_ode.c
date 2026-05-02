/* Direct unit tests for the matlab_ode45_* / matlab_ode23_* runtime
 * entries. No JIT / no compiler frontend — exercises the integration
 * loop, dense output, cache, and odeset path against analytic
 * solutions. */

#include "runtime_test.h"

/* Forward decls for entries declared in matlab_runtime.h, repeated
 * here for clarity at the call sites below. */
typedef double (*ode_rhs_t)(double, double);
matlab_mat *matlab_ode45_t(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode45_y(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23_t(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23_y(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode45_t_opts(ode_rhs_t f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);
matlab_mat *matlab_ode45_y_opts(ode_rhs_t f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);

/* Test RHSes. dy/dt = -y has analytic solution y(t) = y0 * exp(-t). */
static double rhs_decay(double t, double y) { (void)t; return -y; }
/* dy/dt = 1 has analytic y(t) = y0 + t. */
static double rhs_one(double t, double y) { (void)t; (void)y; return 1.0; }
/* dy/dt = 0 — constant solution. */
static double rhs_zero(double t, double y) { (void)t; (void)y; return 0.0; }

static matlab_mat *mk_tspan(double a, double b) {
    double buf[2] = {a, b};
    return matlab_mat_from_buf(buf, 1.0, 2.0);
}

static double last(matlab_mat *m) {
    int64_t n = rt_rows(m) * rt_cols(m);
    return rt_data(m)[n - 1];
}

/* ---- ode45 forward ---- */
static void test_ode45_forward(void) {
    matlab_mat *ts = mk_tspan(0.0, 1.0);
    matlab_mat *T = matlab_ode45_t(rhs_decay, ts, 1.0);
    matlab_mat *Y = matlab_ode45_y(rhs_decay, ts, 1.0);
    /* End-of-grid lands exactly at tf. */
    RT_NEAR(last(T), 1.0, 1e-12, "ode45 t(end) == tf");
    /* y(1) = exp(-1) ≈ 0.3679. rtol = 1e-3 → tolerance ~1e-3. */
    RT_NEAR(last(Y), 0.36787944117, 1e-3, "ode45 y(1) tracks exp(-t)");
    RT_CHECK(rt_rows(Y) == rt_rows(T), "ode45 t and y same length");
    RT_CHECK(rt_cols(T) == 1 && rt_cols(Y) == 1, "ode45 columns");
    /* Refine = 4 → expect ≥ 4 dense samples per accepted step plus
     * the seed. For dy/dt = -y on [0,1] the integrator typically
     * accepts ~3–5 steps; ≥ 10 is a safe lower bound that still
     * confirms dense output is actually emitting interior samples. */
    RT_CHECK(rt_rows(T) >= 10, "ode45 dense output count");
}

/* ---- ode45 backward ---- */
static void test_ode45_backward(void) {
    matlab_mat *ts = mk_tspan(1.0, 0.0);
    matlab_mat *T = matlab_ode45_t(rhs_decay, ts, 0.36787944117);
    matlab_mat *Y = matlab_ode45_y(rhs_decay, ts, 0.36787944117);
    /* Grid spans 1 → 0; the LAST emitted t is 0. */
    RT_NEAR(rt_data(T)[0], 1.0, 1e-12, "ode45 backward t(1) == t1");
    RT_NEAR(last(T), 0.0, 1e-12, "ode45 backward t(end) == t0");
    /* Round-trip recovery: y(0) ≈ 1.0. */
    RT_NEAR(last(Y), 1.0, 1e-3, "ode45 backward recovers y(0)");
}

/* ---- constant RHS ---- */
static void test_ode45_constant(void) {
    matlab_mat *ts = mk_tspan(0.0, 5.0);
    matlab_mat *T = matlab_ode45_t(rhs_one, ts, 0.0);
    matlab_mat *Y = matlab_ode45_y(rhs_one, ts, 0.0);
    /* dy/dt = 1, y(0) = 0 → y(5) = 5. */
    RT_NEAR(last(T), 5.0, 1e-12, "constant-RHS endpoint");
    RT_NEAR(last(Y), 5.0, 1e-9, "y(5) = 5 exact for linear");
    /* Zero-RHS variant. */
    matlab_mat *Y0 = matlab_ode45_y(rhs_zero, ts, 7.5);
    RT_NEAR(last(Y0), 7.5, 1e-12, "zero RHS preserves y0");
}

/* ---- ode23 ---- */
static void test_ode23_forward(void) {
    matlab_mat *ts = mk_tspan(0.0, 1.0);
    matlab_mat *Y = matlab_ode23_y(rhs_decay, ts, 1.0);
    matlab_mat *T = matlab_ode23_t(rhs_decay, ts, 1.0);
    RT_NEAR(last(T), 1.0, 1e-12, "ode23 t(end) == tf");
    RT_NEAR(last(Y), 0.36787944117, 5e-3, "ode23 y(1) tracks exp(-t)");
}

/* ---- odeset RelTol / AbsTol ---- */
static void test_ode45_opts_tol(void) {
    matlab_mat *ts = mk_tspan(0.0, 5.0);
    matlab_struct *opts = matlab_struct_new();
    matlab_struct_set_f64(opts, "RelTol", 6, 1e-9);
    matlab_struct_set_f64(opts, "AbsTol", 6, 1e-12);
    matlab_mat *Y = matlab_ode45_y_opts(rhs_decay, ts, 1.0, opts);
    /* exp(-5) ≈ 6.7379e-3. With rtol=1e-9, error well below 1e-7. */
    RT_NEAR(last(Y), 6.737946999e-3, 1e-7, "tight RelTol -> high accuracy");

    /* Loose: no opts struct, defaults rtol=1e-3. */
    matlab_mat *Yloose = matlab_ode45_y(rhs_decay, ts, 1.0);
    /* Both end at exp(-5); just confirm coarse agreement. */
    RT_NEAR(last(Yloose), 6.737946999e-3, 1e-3, "default RelTol coarse");
}

/* ---- odeset MaxStep ---- */
static void test_ode45_opts_maxstep(void) {
    matlab_mat *ts = mk_tspan(0.0, 5.0);
    matlab_mat *Tdef = matlab_ode45_t(rhs_decay, ts, 1.0);
    int64_t n_def = rt_rows(Tdef);

    matlab_struct *opts = matlab_struct_new();
    matlab_struct_set_f64(opts, "MaxStep", 7, 0.05);
    matlab_mat *Tcap = matlab_ode45_t_opts(rhs_decay, ts, 1.0, opts);
    int64_t n_cap = rt_rows(Tcap);
    /* MaxStep = 0.05 over [0,5] forces ≥ 100 accepted steps; the
     * default takes ~10. Cap output count must be much larger. */
    RT_CHECK(n_cap > 5 * n_def, "MaxStep forces more output points");
}

/* ---- user-specified output grid ---- */
static void test_ode45_user_grid(void) {
    double grid[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    matlab_mat *ts = matlab_mat_from_buf(grid, 1.0, 6.0);
    matlab_mat *T = matlab_ode45_t(rhs_decay, ts, 1.0);
    matlab_mat *Y = matlab_ode45_y(rhs_decay, ts, 1.0);
    RT_CHECK(rt_rows(T) == 6, "user-grid t length matches input");
    RT_CHECK(rt_rows(Y) == 6, "user-grid y length matches input");
    for (int i = 0; i < 6; ++i)
        RT_NEAR(rt_data(T)[i], grid[i], 1e-12, "user-grid t entry exact");
    RT_NEAR(rt_data(Y)[0], 1.0, 1e-12, "y(0) seed");
    RT_NEAR(rt_data(Y)[3], 0.0497870684, 1e-3, "y(3) ~ exp(-3)");
    RT_NEAR(rt_data(Y)[5], 0.00673794700, 1e-4, "y(5) ~ exp(-5)");
}

/* ---- cache hit: paired _t / _y same args ---- */
static void test_ode45_cache(void) {
    matlab_mat *ts = mk_tspan(0.0, 1.0);
    matlab_mat *T1 = matlab_ode45_t(rhs_decay, ts, 1.0);
    matlab_mat *Y1 = matlab_ode45_y(rhs_decay, ts, 1.0);
    /* Without the cache the second call would re-integrate; with
     * the cache it returns the paired column. Both must be the same
     * length. */
    RT_CHECK(rt_rows(T1) == rt_rows(Y1), "cache pairs T and Y same length");
    /* Issue a third unrelated call to bust the cache, then re-pair. */
    matlab_mat *Tother = matlab_ode45_t(rhs_one, ts, 0.0);
    (void)Tother;
    matlab_mat *T2 = matlab_ode45_t(rhs_decay, ts, 1.0);
    matlab_mat *Y2 = matlab_ode45_y(rhs_decay, ts, 1.0);
    /* Re-paired call: same data as first. */
    RT_NEAR(last(T2), last(T1), 1e-15, "cache miss + repair: T match");
    RT_NEAR(last(Y2), last(Y1), 1e-15, "cache miss + repair: Y match");
}

int main(void) {
    fprintf(stderr, "test_ode:\n");
    RT_RUN(test_ode45_forward);
    RT_RUN(test_ode45_backward);
    RT_RUN(test_ode45_constant);
    RT_RUN(test_ode23_forward);
    RT_RUN(test_ode45_opts_tol);
    RT_RUN(test_ode45_opts_maxstep);
    RT_RUN(test_ode45_user_grid);
    RT_RUN(test_ode45_cache);
    RT_DONE();
}
