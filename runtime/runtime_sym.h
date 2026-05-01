#ifndef MATLAB_RUNTIME_SYM_H
#define MATLAB_RUNTIME_SYM_H

/* Symbolic Math Toolbox runtime — Phase A.
 *
 * Opaque matlab_sym wraps a sympp::Expr. Generated code (LLVM IR / C /
 * C++ via -emit-c) only sees the C ABI: pointers in, pointers out.
 *
 * Lifecycle: every constructor returns a heap-owned matlab_sym* that
 * the workspace store / value-tracker is responsible for freeing via
 * matlab_sym_free. Operators that take matlab_sym* arguments do NOT
 * consume them (the caller still owns the inputs); they return a fresh
 * matlab_sym*.
 *
 * Thread safety: each matlab_sym is independently allocated and refers
 * to an immutable SymPP Expr. Operators are pure functions. The
 * assumption side-table (assume / assumeAlso / clearAssumptions) is
 * SymPP-side-effecting and shares the lock SymPP uses internally.
 *
 * Phase A surface (matches MATLAB Symbolic Math Toolbox UG, R2026a):
 *   syms / sym / str2sym
 *   + - * / ^ unary-minus
 *   diff, int (definite/indefinite), simplify, expand, factor, subs
 *   solve(eq, var)  — single equation, single variable
 *   double, disp, str (for REPL/DAP), latex, ccode, pretty
 *
 * Phases B/C (deferred): symbolic matrices, taylor/limit, dsolve,
 * pdsolve, transforms, vpa, assume, matlabFunction, matrix linsolve.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct matlab_sym_s matlab_sym;

/* --- Construction --------------------------------------------------------- */

/* sym("x") — bare identifier becomes a Symbol.
 * sym("a*x^2 + b") — operator/whitespace triggers parse. */
matlab_sym *matlab_sym_from_str(const char *s, int64_t n);

/* str2sym(str) — explicit parse, no identifier shortcut. */
matlab_sym *matlab_sym_str2sym(const char *s, int64_t n);

/* sym(value) — exact integer when v is integral, else Float. */
matlab_sym *matlab_sym_from_double(double v);
matlab_sym *matlab_sym_from_i64(int64_t v);

/* syms x — names a fresh symbol with no assumptions. Equivalent to
 * matlab_sym_from_str on a bare identifier but skips the heuristic. */
matlab_sym *matlab_sym_named(const char *s, int64_t n);

/* --- Algebra -------------------------------------------------------------- */

matlab_sym *matlab_sym_add(const matlab_sym *a, const matlab_sym *b);
matlab_sym *matlab_sym_sub(const matlab_sym *a, const matlab_sym *b);
matlab_sym *matlab_sym_mul(const matlab_sym *a, const matlab_sym *b);
matlab_sym *matlab_sym_div(const matlab_sym *a, const matlab_sym *b);
matlab_sym *matlab_sym_pow(const matlab_sym *a, const matlab_sym *b);
matlab_sym *matlab_sym_neg(const matlab_sym *a);

/* Mixed-mode arithmetic (sym op double) — saves an alloc for the
 * common case f = x + 1, lowering doesn't have to box the literal. */
matlab_sym *matlab_sym_add_d(const matlab_sym *a, double b);
matlab_sym *matlab_sym_sub_d(const matlab_sym *a, double b);
matlab_sym *matlab_sym_mul_d(const matlab_sym *a, double b);
matlab_sym *matlab_sym_div_d(const matlab_sym *a, double b);
matlab_sym *matlab_sym_pow_d(const matlab_sym *a, double b);
matlab_sym *matlab_sym_d_sub(double a, const matlab_sym *b);
matlab_sym *matlab_sym_d_div(double a, const matlab_sym *b);
matlab_sym *matlab_sym_d_pow(double a, const matlab_sym *b);

/* eq(a, b) — symbolic equality (used by solve(f == 0, x)).
 * Returns a sym holding Eq(a, b). */
matlab_sym *matlab_sym_eq(const matlab_sym *a, const matlab_sym *b);
matlab_sym *matlab_sym_eq_d(const matlab_sym *a, double b);

/* Elementary-function overloads — when sin/cos/exp/... is called with a
 * sym argument, dispatch lands here. SymPP's free functions return Expr
 * by argument-dependent lookup; we wrap a fixed set under stable names. */
matlab_sym *matlab_sym_sin(const matlab_sym *a);
matlab_sym *matlab_sym_cos(const matlab_sym *a);
matlab_sym *matlab_sym_tan(const matlab_sym *a);
matlab_sym *matlab_sym_asin(const matlab_sym *a);
matlab_sym *matlab_sym_acos(const matlab_sym *a);
matlab_sym *matlab_sym_atan(const matlab_sym *a);
matlab_sym *matlab_sym_sinh(const matlab_sym *a);
matlab_sym *matlab_sym_cosh(const matlab_sym *a);
matlab_sym *matlab_sym_tanh(const matlab_sym *a);
matlab_sym *matlab_sym_exp(const matlab_sym *a);
matlab_sym *matlab_sym_log(const matlab_sym *a);
matlab_sym *matlab_sym_sqrt(const matlab_sym *a);
matlab_sym *matlab_sym_abs(const matlab_sym *a);

/* --- Calculus ------------------------------------------------------------- */

matlab_sym *matlab_sym_diff(const matlab_sym *f, const matlab_sym *var);
matlab_sym *matlab_sym_diff_n(const matlab_sym *f, const matlab_sym *var,
                              int64_t n);

/* Indefinite integral. */
matlab_sym *matlab_sym_int(const matlab_sym *f, const matlab_sym *var);

/* Definite integral over [a, b]. */
matlab_sym *matlab_sym_int_def(const matlab_sym *f, const matlab_sym *var,
                               const matlab_sym *a, const matlab_sym *b);

/* --- Manipulation --------------------------------------------------------- */

matlab_sym *matlab_sym_simplify(const matlab_sym *e);
matlab_sym *matlab_sym_expand(const matlab_sym *e);
matlab_sym *matlab_sym_factor(const matlab_sym *e, const matlab_sym *var);
matlab_sym *matlab_sym_subs(const matlab_sym *e,
                            const matlab_sym *old_e,
                            const matlab_sym *new_e);

/* --- Solver --------------------------------------------------------------- */

/* solve(f == 0, x) and solve(f, x) both route here. The shape mirrors
 * MATLAB's solve return: a flat array of root sym*s, count via *n_out.
 * Caller owns each result and must matlab_sym_free them.
 *
 * If matlab_sym_eq(...) was used to build f, the runtime extracts
 * lhs - rhs before forwarding. Otherwise f is treated as the LHS with
 * implicit RHS = 0 (MATLAB's `solve(expr)` shape). */
matlab_sym **matlab_sym_solve(const matlab_sym *eq,
                              const matlab_sym *var,
                              int64_t *n_out);

/* solve(...) for the language layer — routes to matlab_sym_solve and
 * collapses the result list into a single sym. When zero roots: returns
 * a sym holding the literal symbol "NoSolution". When one root: that
 * root. When multiple: a sym whose pretty form is "[r1, r2, ...]".
 * Phase A simplification — proper sym vector returns land in Phase B. */
matlab_sym *matlab_sym_solve_one(const matlab_sym *eq, const matlab_sym *var);

/* --- Conversion / display ------------------------------------------------- */

/* double(sym) — numeric evaluation. NaN if the expression still has
 * free symbols. */
double matlab_sym_double(const matlab_sym *e);

/* disp(sym) — prints the pretty form to stdout, no trailing newline
 * before '\n' (matches matlab_disp_* shape). */
void matlab_sym_disp(const matlab_sym *e);

/* str(sym) — pretty-printed expression text. Returns a malloc'd
 * NUL-terminated buffer; caller owns. *len_out gets the length
 * excluding NUL (or 0 if e is null). Used by REPL workspace + DAP
 * variable formatter. */
char *matlab_sym_str(const matlab_sym *e, int64_t *len_out);
char *matlab_sym_latex(const matlab_sym *e, int64_t *len_out);
char *matlab_sym_ccode(const matlab_sym *e, int64_t *len_out);

/* --- Lifecycle ------------------------------------------------------------ */

void matlab_sym_free(matlab_sym *e);

/* Workspace setter — kind=7 in the matlab_ws table. Routes through
 * the same undo-log machinery as matlab_ws_set_obj. */
void matlab_ws_set_sym(const char *name, int64_t name_len, matlab_sym *e);
matlab_sym *matlab_ws_get_sym(const char *name, int64_t name_len);

/* DAP introspection — read-only, returns NULL on miss. */
const char *matlab_dbg_sym_str(const matlab_sym *e, int64_t *len_out);

/* --- Phase B: assumptions ------------------------------------------------- */

/* assume(x, "positive") — register an assumption on x's name. The
 * SymPP side-table is keyed by name (Expr->str()), so the matlab_sym
 * argument is consumed only to extract its name; the registered mask
 * applies to every same-named Symbol going forward. After assume(),
 * the runtime returns a fresh sym carrying the mask — caller
 * (lowering's AssignStmt) writes it back into the binding.
 *
 * Property strings: "real", "rational", "integer", "positive",
 * "negative", "zero", "nonzero", "nonnegative", "nonpositive",
 * "finite". Anything else throws std::runtime_error. */
matlab_sym *matlab_sym_assume(const matlab_sym *x,
                              const char *prop, int64_t prop_len);
matlab_sym *matlab_sym_assumeAlso(const matlab_sym *x,
                                  const char *prop, int64_t prop_len);

/* assumptions(x) — returns a malloc'd, comma-joined list of property
 * names ("real,positive"), or an empty string if none. *len_out gets
 * the length excluding NUL. Caller frees via free(). */
char *matlab_sym_assumptions(const matlab_sym *x, int64_t *len_out);

/* clearAssumptions(x) — drop the side-table entry; returns a fresh
 * sym carrying no assumptions, again to be written back. */
matlab_sym *matlab_sym_clearAssumptions(const matlab_sym *x);

/* --- Phase B: vpa / taylor / limit --------------------------------------- */

matlab_sym *matlab_sym_vpa(const matlab_sym *e, int64_t dps);
matlab_sym *matlab_sym_taylor(const matlab_sym *f, const matlab_sym *var,
                              const matlab_sym *a, int64_t n);
matlab_sym *matlab_sym_limit(const matlab_sym *f, const matlab_sym *var,
                             const matlab_sym *target);

/* --- Phase B: dsolve / pdsolve ------------------------------------------- */

/* dsolve(eq, y, yp, x) — first-order ODE.
 *  - eq:  expression in {y, yp, x} representing eq = 0
 *  - y:   the unknown function symbol (not the AppliedFunction y(x))
 *  - yp:  symbol standing in for y'
 *  - x:   independent variable
 * Returns the general solution as a sym, or an unevaluated Dsolve(...)
 * marker if no strategy matches. SymPP's facade has no 3-arg form
 * because deriving the order from `eq` requires lifting a SymPP
 * derivative-symbol convention into the AST.
 *
 * dsolve_2(eq, y, yp, ypp, x) — second-order, auto-classified into
 * constant-coefficient or Cauchy-Euler shape. */
matlab_sym *matlab_sym_dsolve(const matlab_sym *eq, const matlab_sym *y,
                              const matlab_sym *yp, const matlab_sym *x);
matlab_sym *matlab_sym_dsolve_2(const matlab_sym *eq, const matlab_sym *y,
                                const matlab_sym *yp, const matlab_sym *ypp,
                                const matlab_sym *x);

/* pdsolve(a, b, c, x, y) — first-order linear PDE
 * a(x,y)*u_x + b(x,y)*u_y = c(x,y). */
matlab_sym *matlab_sym_pdsolve(const matlab_sym *a, const matlab_sym *b,
                               const matlab_sym *c,
                               const matlab_sym *x, const matlab_sym *y);

/* pdsolve_heat / pdsolve_wave — convenience entries for the named PDEs. */
matlab_sym *matlab_sym_pdsolve_heat(const matlab_sym *k,
                                    const matlab_sym *lambda,
                                    const matlab_sym *x,
                                    const matlab_sym *t);
matlab_sym *matlab_sym_pdsolve_wave(const matlab_sym *c,
                                    const matlab_sym *x,
                                    const matlab_sym *t);

/* --- Phase C: integral transforms ---------------------------------------- */

matlab_sym *matlab_sym_laplace(const matlab_sym *f,
                               const matlab_sym *t, const matlab_sym *s);
matlab_sym *matlab_sym_ilaplace(const matlab_sym *F,
                                const matlab_sym *s, const matlab_sym *t);
matlab_sym *matlab_sym_fourier(const matlab_sym *f,
                               const matlab_sym *t, const matlab_sym *w);
matlab_sym *matlab_sym_ifourier(const matlab_sym *F,
                                const matlab_sym *w, const matlab_sym *t);
matlab_sym *matlab_sym_ztrans(const matlab_sym *f,
                              const matlab_sym *n, const matlab_sym *z);
matlab_sym *matlab_sym_iztrans(const matlab_sym *F,
                               const matlab_sym *z, const matlab_sym *n);

#ifdef __cplusplus
}
#endif

#endif /* MATLAB_RUNTIME_SYM_H */
