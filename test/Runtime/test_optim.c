/* Direct unit tests for the Optimization Toolbox runtime entries:
 * Tier-1 (fzero / fminbnd / fminsearch / fminunc BFGS / linprog /
 * lsqnonneg / fsolve), Tier-2 (fmincon / quadprog / lsqlin /
 * lsqnonlin / lsqcurvefit), Tier-3 (intlinprog).
 *
 * Mirrors the Sema-level builtin dispatch — each entry is exercised
 * with a small textbook problem whose closed-form answer is known. */

#include "runtime_test.h"

/* Forward decls — runtime_optim.cpp entries, not in matlab_runtime.h */
double      matlab_optim_fzero       (void *fn_p, double x0);
double      matlab_optim_fminbnd     (void *fn_p, double lo, double hi);
matlab_mat *matlab_optim_fminsearch  (void *fn_p, matlab_mat *x0);
matlab_mat *matlab_optim_fminunc     (void *fn_p, matlab_mat *x0);
matlab_mat *matlab_optim_linprog     (matlab_mat *f, matlab_mat *A, matlab_mat *b,
                                       matlab_mat *Aeq, matlab_mat *beq,
                                       matlab_mat *lb, matlab_mat *ub);
matlab_mat *matlab_optim_linprog3    (matlab_mat *f, matlab_mat *A, matlab_mat *b);
matlab_mat *matlab_optim_lsqnonneg   (matlab_mat *C, matlab_mat *d);
double      matlab_optim_fsolve_scalar(void *fn_p, double x0);
matlab_mat *matlab_optim_fsolve      (void *fun_p, matlab_mat *x0);
matlab_mat *matlab_optim_fmincon     (void *obj_p, matlab_mat *x0,
                                       matlab_mat *A, matlab_mat *b,
                                       matlab_mat *Aeq, matlab_mat *beq,
                                       matlab_mat *lb, matlab_mat *ub,
                                       void *nonlcon_p);
matlab_mat *matlab_optim_quadprog    (matlab_mat *H, matlab_mat *fvec,
                                       matlab_mat *A, matlab_mat *b,
                                       matlab_mat *Aeq, matlab_mat *beq,
                                       matlab_mat *lb, matlab_mat *ub);
matlab_mat *matlab_optim_lsqlin      (matlab_mat *C, matlab_mat *d,
                                       matlab_mat *A, matlab_mat *b,
                                       matlab_mat *Aeq, matlab_mat *beq,
                                       matlab_mat *lb, matlab_mat *ub);
matlab_mat *matlab_optim_lsqnonlin   (void *fun_p, matlab_mat *x0,
                                       matlab_mat *lb, matlab_mat *ub);
matlab_mat *matlab_optim_lsqcurvefit (void *fun_p, matlab_mat *x0,
                                       matlab_mat *xdata, matlab_mat *ydata);
matlab_mat *matlab_optim_intlinprog  (matlab_mat *f, matlab_mat *intcon,
                                       matlab_mat *A, matlab_mat *b,
                                       matlab_mat *Aeq, matlab_mat *beq,
                                       matlab_mat *lb, matlab_mat *ub);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* ===== Tier-1: fzero ===== */

static double parabola_minus_two(double x) { return x * x - 2.0; }

static void test_fzero_sqrt2(void) {
    double r = matlab_optim_fzero((void *)parabola_minus_two, 1.0);
    RT_NEAR(r, 1.4142135623730951, 1e-8, "fzero on x^2 - 2 -> sqrt(2)");
}

static double cubic_three_roots(double x) { return x * x * x - x; }

static void test_fzero_cubic_near_zero(void) {
    /* x^3 - x has roots at -1, 0, +1. Brent from x0=0.1 should find 0. */
    double r = matlab_optim_fzero((void *)cubic_three_roots, 0.1);
    RT_NEAR(r, 0.0, 1e-6, "fzero on x^3-x near zero -> 0");
}

/* ===== Tier-1: fminbnd ===== */

static double parabola_minus_three(double x) {
    return (x - 3.0) * (x - 3.0) + 1.0;
}

static void test_fminbnd_parabola(void) {
    /* min of (x-3)^2 + 1 on [0, 5] is x* = 3. */
    double xmin = matlab_optim_fminbnd((void *)parabola_minus_three, 0.0, 5.0);
    RT_NEAR(xmin, 3.0, 1e-5, "fminbnd on (x-3)^2 + 1");
}

/* ===== Tier-1: fminsearch ===== */

static double rosenbrock_2d(matlab_mat *x) {
    double x1 = rt_data(x)[0], x2 = rt_data(x)[1];
    double a = 1.0 - x1, b = x2 - x1 * x1;
    return a * a + 100.0 * b * b;
}

static void test_fminsearch_rosenbrock(void) {
    double x0buf[] = {-1.2, 1.0};
    matlab_mat *x0 = mk(x0buf, 2, 1);
    matlab_mat *r = matlab_optim_fminsearch((void *)rosenbrock_2d, x0);
    RT_NEAR(rt_data(r)[0], 1.0, 1e-2, "fminsearch Rosenbrock x1");
    RT_NEAR(rt_data(r)[1], 1.0, 1e-2, "fminsearch Rosenbrock x2");
    rt_free(x0); rt_free(r);
}

/* ===== Tier-1: fminunc (BFGS) ===== */

static double quadratic_2d(matlab_mat *x) {
    /* f(x) = (x1 - 2)^2 + (x2 + 1)^2; min at (2, -1). */
    double x1 = rt_data(x)[0], x2 = rt_data(x)[1];
    return (x1 - 2.0) * (x1 - 2.0) + (x2 + 1.0) * (x2 + 1.0);
}

static void test_fminunc_quadratic(void) {
    double x0buf[] = {0.0, 0.0};
    matlab_mat *x0 = mk(x0buf, 2, 1);
    matlab_mat *r = matlab_optim_fminunc((void *)quadratic_2d, x0);
    RT_NEAR(rt_data(r)[0],  2.0, 1e-4, "fminunc BFGS x1");
    RT_NEAR(rt_data(r)[1], -1.0, 1e-4, "fminunc BFGS x2");
    rt_free(x0); rt_free(r);
}

/* ===== Tier-1: linprog ===== */

static void test_linprog_simple(void) {
    /* min  -x1 - x2
       s.t.   x1 + x2 <= 1
              x1, x2 >= 0
       Optimal: x* = (1, 0) or (0, 1) — both feasible vertices; LP
       picks one. We assert the sum is at the boundary. */
    double fbuf[] = {-1, -1};
    double Abuf[] = {1, 1};
    double bbuf[] = {1};
    matlab_mat *f = mk(fbuf, 2, 1);
    matlab_mat *A = mk(Abuf, 1, 2);
    matlab_mat *b = mk(bbuf, 1, 1);
    matlab_mat *r = matlab_optim_linprog3(f, A, b);
    RT_CHECK(rt_rows(r) * rt_cols(r) >= 2, "linprog returned x");
    double s = rt_data(r)[0] + rt_data(r)[1];
    RT_NEAR(s, 1.0, 1e-5, "linprog reaches x1+x2=1 boundary");
    RT_CHECK(rt_data(r)[0] >= -1e-6 && rt_data(r)[1] >= -1e-6,
             "linprog respects x>=0");
    rt_free(f); rt_free(A); rt_free(b); rt_free(r);
}

/* ===== Tier-1: lsqnonneg ===== */

static void test_lsqnonneg_overdetermined(void) {
    /* C = [1 0; 0 1; 1 1]; d = [2; 3; 5]
       Unconstrained LS gives (2, 3); >= 0 so NNLS returns the same. */
    double Cbuf[] = {1, 0,
                     0, 1,
                     1, 1};
    double dbuf[] = {2, 3, 5};
    matlab_mat *C = mk(Cbuf, 3, 2);
    matlab_mat *d = mk(dbuf, 3, 1);
    matlab_mat *r = matlab_optim_lsqnonneg(C, d);
    RT_NEAR(rt_data(r)[0], 2.0, 1e-6, "lsqnonneg x1");
    RT_NEAR(rt_data(r)[1], 3.0, 1e-6, "lsqnonneg x2");
    rt_free(C); rt_free(d); rt_free(r);
}

static void test_lsqnonneg_negative_unconstrained(void) {
    /* C = [1; 1]; d = [-2; -2]. Unconstrained gives x = -2; NNLS clamps to 0. */
    double Cbuf[] = {1, 1};
    double dbuf[] = {-2, -2};
    matlab_mat *C = mk(Cbuf, 2, 1);
    matlab_mat *d = mk(dbuf, 2, 1);
    matlab_mat *r = matlab_optim_lsqnonneg(C, d);
    RT_NEAR(rt_data(r)[0], 0.0, 1e-9, "lsqnonneg clamps to zero");
    rt_free(C); rt_free(d); rt_free(r);
}

/* ===== Tier-1: fsolve scalar ===== */

static double cubic_one_real_root(double x) {
    return x * x * x + x - 5.0;
}

static void test_fsolve_scalar(void) {
    /* x^3 + x - 5 = 0 has one real root ~1.515980357. */
    double r = matlab_optim_fsolve_scalar((void *)cubic_one_real_root, 1.0);
    RT_NEAR(r, 1.515980357, 1e-6, "fsolve scalar cubic");
}

/* ===== Tier-2: quadprog ===== */

static void test_quadprog_unconstrained(void) {
    /* min  0.5 x' H x + f' x, H = [[2, 0]; [0, 2]], f = [-2; -6]
       Closed form: x = -H^-1 f = [1; 3]. */
    double Hbuf[] = {2, 0, 0, 2};
    double fbuf[] = {-2, -6};
    matlab_mat *H = mk(Hbuf, 2, 2);
    matlab_mat *fv = mk(fbuf, 2, 1);
    matlab_mat *empty = matlab_mat_from_buf(NULL, 0, 0);
    matlab_mat *r = matlab_optim_quadprog(H, fv, empty, empty, empty,
                                          empty, empty, empty);
    RT_NEAR(rt_data(r)[0], 1.0, 1e-4, "quadprog x1");
    RT_NEAR(rt_data(r)[1], 3.0, 1e-4, "quadprog x2");
    rt_free(H); rt_free(fv); rt_free(empty); rt_free(r);
}

/* ===== Tier-2: lsqlin ===== */

static void test_lsqlin_overdetermined(void) {
    /* Same as lsqnonneg test but without constraints — should give
       the same answer since both vars are positive. */
    double Cbuf[] = {1, 0,
                     0, 1,
                     1, 1};
    double dbuf[] = {2, 3, 5};
    matlab_mat *C = mk(Cbuf, 3, 2);
    matlab_mat *d = mk(dbuf, 3, 1);
    matlab_mat *empty = matlab_mat_from_buf(NULL, 0, 0);
    matlab_mat *r = matlab_optim_lsqlin(C, d, empty, empty, empty,
                                        empty, empty, empty);
    RT_NEAR(rt_data(r)[0], 2.0, 1e-3, "lsqlin x1");
    RT_NEAR(rt_data(r)[1], 3.0, 1e-3, "lsqlin x2");
    rt_free(C); rt_free(d); rt_free(empty); rt_free(r);
}

/* ===== Tier-2: fmincon (unconstrained as a constraint-set-empty smoke) ===== */

static void test_fmincon_no_constraints(void) {
    double x0buf[] = {0.0, 0.0};
    matlab_mat *x0 = mk(x0buf, 2, 1);
    matlab_mat *empty = matlab_mat_from_buf(NULL, 0, 0);
    matlab_mat *r = matlab_optim_fmincon((void *)quadratic_2d, x0,
                                         empty, empty, empty, empty,
                                         empty, empty, NULL);
    RT_NEAR(rt_data(r)[0],  2.0, 1e-2, "fmincon (no cons) x1");
    RT_NEAR(rt_data(r)[1], -1.0, 1e-2, "fmincon (no cons) x2");
    rt_free(x0); rt_free(empty); rt_free(r);
}

/* ===== Tier-2: lsqnonlin ===== */

static matlab_mat *residual_two_eqs(matlab_mat *x) {
    /* r1 = x1 - 2, r2 = x2 + 1. Min at (2, -1). */
    double r1 = rt_data(x)[0] - 2.0;
    double r2 = rt_data(x)[1] + 1.0;
    double buf[2] = {r1, r2};
    return matlab_mat_from_buf(buf, 2.0, 1.0);
}

static void test_lsqnonlin_quadratic(void) {
    double x0buf[] = {0.0, 0.0};
    matlab_mat *x0 = mk(x0buf, 2, 1);
    matlab_mat *empty = matlab_mat_from_buf(NULL, 0, 0);
    matlab_mat *r = matlab_optim_lsqnonlin((void *)residual_two_eqs, x0,
                                           empty, empty);
    RT_NEAR(rt_data(r)[0],  2.0, 1e-3, "lsqnonlin x1");
    RT_NEAR(rt_data(r)[1], -1.0, 1e-3, "lsqnonlin x2");
    rt_free(x0); rt_free(empty); rt_free(r);
}

/* ===== Tier-3: intlinprog ===== */

static void test_intlinprog_simple(void) {
    /* min -x1 - 2*x2  s.t.  x1 + x2 <= 4, x1, x2 in {0, 1, 2, 3, 4}.
       Optimal: x1 = 0, x2 = 4 (objective -8). */
    double fbuf[] = {-1, -2};
    double Abuf[] = {1, 1};
    double bbuf[] = {4};
    double ibuf[] = {1, 2};       /* both vars integer (1-based indices) */
    double lbbuf[] = {0, 0};
    double ubbuf[] = {4, 4};
    matlab_mat *f      = mk(fbuf, 2, 1);
    matlab_mat *intcon = mk(ibuf, 2, 1);
    matlab_mat *A      = mk(Abuf, 1, 2);
    matlab_mat *b      = mk(bbuf, 1, 1);
    matlab_mat *empty  = matlab_mat_from_buf(NULL, 0, 0);
    matlab_mat *lb     = mk(lbbuf, 2, 1);
    matlab_mat *ub     = mk(ubbuf, 2, 1);
    matlab_mat *r = matlab_optim_intlinprog(f, intcon, A, b,
                                            empty, empty, lb, ub);
    /* Optimum is (0, 4); allow either rounding direction. */
    double x1 = rt_data(r)[0], x2 = rt_data(r)[1];
    RT_NEAR(x1, 0.0, 1e-6, "intlinprog x1");
    RT_NEAR(x2, 4.0, 1e-6, "intlinprog x2");
    rt_free(f); rt_free(intcon); rt_free(A); rt_free(b);
    rt_free(empty); rt_free(lb); rt_free(ub); rt_free(r);
}

int main(void) {
    RT_RUN(test_fzero_sqrt2);
    RT_RUN(test_fzero_cubic_near_zero);
    RT_RUN(test_fminbnd_parabola);
    RT_RUN(test_fminsearch_rosenbrock);
    RT_RUN(test_fminunc_quadratic);
    RT_RUN(test_linprog_simple);
    RT_RUN(test_lsqnonneg_overdetermined);
    RT_RUN(test_lsqnonneg_negative_unconstrained);
    RT_RUN(test_fsolve_scalar);
    RT_RUN(test_quadprog_unconstrained);
    RT_RUN(test_lsqlin_overdetermined);
    RT_RUN(test_fmincon_no_constraints);
    RT_RUN(test_lsqnonlin_quadratic);
    RT_RUN(test_intlinprog_simple);
    RT_DONE();
}
