/* Direct unit tests for the matlab_ode45_* / matlab_ode23_* runtime
 * entries. No JIT / no compiler frontend — exercises the integration
 * loop, dense output, cache, and odeset path against analytic
 * solutions. */

#include "runtime_test.h"

/* Forward decls for entries declared in matlab_runtime.h, repeated
 * here for clarity at the call sites below. */
typedef double (*ode_rhs_t)(double, double);
typedef matlab_mat *(*ode_rhs_v_t)(double, matlab_mat *);
matlab_mat *matlab_ode45_t(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode45_y(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23_t(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23_y(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode45_t_opts(ode_rhs_t f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);
matlab_mat *matlab_ode45_y_opts(ode_rhs_t f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);
matlab_mat *matlab_ode45_v_t(ode_rhs_v_t f, matlab_mat *tspan,
                              matlab_mat *y0);
matlab_mat *matlab_ode45_v_y(ode_rhs_v_t f, matlab_mat *tspan,
                              matlab_mat *y0);
matlab_mat *matlab_ode23_v_y(ode_rhs_v_t f, matlab_mat *tspan,
                              matlab_mat *y0);
matlab_mat *matlab_ode23s_t(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23s_y(ode_rhs_t f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23s_v_y(ode_rhs_v_t f, matlab_mat *tspan,
                               matlab_mat *y0);

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

/* ---- vector-y oscillator ---- */
/* dy/dt = [-y(2); y(1)] — linear oscillator. y(t) = [cos(t); sin(t)] for
 * y(0) = [1; 0]. At t = 2π the state returns to the initial. */
static matlab_mat *rhs_oscillator(double t, matlab_mat *y) {
    (void)t;
    /* matlab_mat is opaque in the public header; build the result via
     * matlab_mat_from_buf and the rt_test layout for read access. */
    double buf[2];
    buf[0] = -rt_data(y)[1];
    buf[1] =  rt_data(y)[0];
    return matlab_mat_from_buf(buf, 2.0, 1.0);
}

static void test_ode45_vector_oscillator(void) {
    double y0buf[2] = {1.0, 0.0};
    matlab_mat *y0 = matlab_mat_from_buf(y0buf, 2.0, 1.0);
    double tsbuf[2] = {0.0, 6.283185307179586};   /* 0 to 2π */
    matlab_mat *ts = matlab_mat_from_buf(tsbuf, 1.0, 2.0);

    matlab_mat *T = matlab_ode45_v_t(rhs_oscillator, ts, y0);
    matlab_mat *Y = matlab_ode45_v_y(rhs_oscillator, ts, y0);
    int64_t N = rt_rows(T);
    RT_CHECK(N >= 5, "oscillator emits multiple steps");
    RT_NEAR(rt_data(T)[0], 0.0, 1e-12, "t(1) == 0");
    RT_NEAR(rt_data(T)[N-1], 6.283185307179586, 1e-12, "t(end) == 2π");
    /* Y is row-major NxD with D = 2: Y[i*2 + 0] = cos(t_i), [+1] = sin(t_i). */
    RT_NEAR(rt_data(Y)[0], 1.0, 1e-12, "y(0,1) seed cos(0) = 1");
    RT_NEAR(rt_data(Y)[1], 0.0, 1e-12, "y(0,2) seed sin(0) = 0");
    /* At t = 2π, expect cos = 1, sin = 0 (within rtol = 1e-3). */
    RT_NEAR(rt_data(Y)[(N-1)*2 + 0],  1.0, 5e-3, "y(end,1) ~ cos(2π)");
    RT_NEAR(rt_data(Y)[(N-1)*2 + 1],  0.0, 5e-3, "y(end,2) ~ sin(2π)");
}

static void test_ode23_vector_oscillator(void) {
    double y0buf[2] = {1.0, 0.0};
    matlab_mat *y0 = matlab_mat_from_buf(y0buf, 2.0, 1.0);
    double tsbuf[2] = {0.0, 6.283185307179586};
    matlab_mat *ts = matlab_mat_from_buf(tsbuf, 1.0, 2.0);
    matlab_mat *Y = matlab_ode23_v_y(rhs_oscillator, ts, y0);
    int64_t N = rt_rows(Y);
    /* ode23 is lower order — looser tolerance. */
    RT_NEAR(rt_data(Y)[(N-1)*2 + 0], 1.0, 5e-2, "ode23 y(end,1)");
    RT_NEAR(rt_data(Y)[(N-1)*2 + 1], 0.0, 5e-2, "ode23 y(end,2)");
}

/* ---- ode23s scalar (stiff) ---- */
static double rhs_stiff_decay(double t, double y) { (void)t; return -100.0 * y; }

static void test_ode23s_scalar_stiff(void) {
    matlab_mat *ts = mk_tspan(0.0, 1.0);
    matlab_mat *T = matlab_ode23s_t(rhs_stiff_decay, ts, 1.0);
    matlab_mat *Y = matlab_ode23s_y(rhs_stiff_decay, ts, 1.0);
    int64_t N = rt_rows(T);
    /* ode23s should converge for dy/dt = -100*y in tens of steps. */
    RT_CHECK(N > 5,  "ode23s steady-state run produced output");
    RT_CHECK(N < 80, "ode23s used reasonable step count on stiff problem");
    /* y(1) ≈ exp(-100) ≈ 3.7e-44 — effectively 0. */
    RT_NEAR(last(Y), 0.0, 1e-6, "ode23s reaches stiff steady state");
}

/* ---- ode23s vector (Robertson) ---- */
static matlab_mat *rhs_robertson(double t, matlab_mat *y) {
    (void)t;
    double y1 = rt_data(y)[0], y2 = rt_data(y)[1], y3 = rt_data(y)[2];
    double buf[3];
    buf[0] = -0.04*y1 + 1e4*y2*y3;
    buf[1] =  0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2;
    buf[2] =                        3e7*y2*y2;
    return matlab_mat_from_buf(buf, 3.0, 1.0);
}

static void test_ode23s_vector_robertson(void) {
    double y0buf[3] = {1.0, 0.0, 0.0};
    matlab_mat *y0 = matlab_mat_from_buf(y0buf, 3.0, 1.0);
    matlab_mat *ts = mk_tspan(0.0, 1.0);
    matlab_mat *Y = matlab_ode23s_v_y(rhs_robertson, ts, y0);
    int64_t N = rt_rows(Y);
    RT_CHECK(N > 3,    "Robertson stiff system integrates");
    /* Conservation: y1 + y2 + y3 = 1 throughout. */
    double total = rt_data(Y)[(N-1)*3 + 0]
                 + rt_data(Y)[(N-1)*3 + 1]
                 + rt_data(Y)[(N-1)*3 + 2];
    RT_NEAR(total, 1.0, 1e-6, "Robertson mass conserved");
    /* y2 is a fast transient — small at t=1 (steady-state ~1e-5). */
    RT_CHECK(rt_data(Y)[(N-1)*3 + 1] < 1e-3, "Robertson y2 decayed");
}

/* ---- 3-return form: stats struct ---- */
matlab_struct *matlab_ode45_stats(ode_rhs_t f, matlab_mat *tspan, double y0);

static void test_ode45_stats_struct(void) {
    matlab_mat *ts = mk_tspan(0.0, 1.0);
    matlab_mat *T = matlab_ode45_t(rhs_decay, ts, 1.0);
    matlab_mat *Y = matlab_ode45_y(rhs_decay, ts, 1.0);
    matlab_struct *s = matlab_ode45_stats(rhs_decay, ts, 1.0);
    /* The cache means the second & third calls reuse the first's solve. */
    int64_t n = (int64_t)matlab_struct_get_f64(s, "nsteps", 6);
    int64_t fail = (int64_t)matlab_struct_get_f64(s, "nfailed", 7);
    int64_t fev = (int64_t)matlab_struct_get_f64(s, "nfevals", 7);
    RT_CHECK(n > 0, "stats.nsteps positive");
    RT_CHECK(fail >= 0, "stats.nfailed non-negative");
    RT_CHECK(fev > n, "stats.nfevals greater than step count");
    /* RK45 does 6 fevals per step plus an initial k1 (FSAL keeps subsequent
     * step starts free). With ~5 steps, ~31 fevals — bounds. */
    RT_CHECK(fev >= 6 * n, "stats.nfevals >= 6 * nsteps");
    /* T, Y untouched. */
    RT_CHECK(rt_rows(T) == rt_rows(Y), "T and Y same length after stats call");
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
    RT_RUN(test_ode45_vector_oscillator);
    RT_RUN(test_ode23_vector_oscillator);
    RT_RUN(test_ode23s_scalar_stiff);
    RT_RUN(test_ode23s_vector_robertson);
    RT_RUN(test_ode45_stats_struct);
    RT_RUN(test_ode45_cache);
    RT_DONE();
}
