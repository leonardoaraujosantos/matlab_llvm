/* Symbolic Math Toolbox runtime — Phase A.
 *
 * Wraps SymPP's MATLAB facade (sympp::matlab::*) behind a pure C ABI
 * declared in runtime_sym.h. Generated code never sees a SymPP type.
 *
 * Build gate: only compiled when MATLAB_LLVM_WITH_SYM is defined; the
 * top-level CMakeLists wires that automatically when SymPP is found.
 * When the gate is off, generated code that calls these symbols would
 * fail to link — same diagnosis story as any other off-by-default
 * runtime feature. */

#include "runtime_sym.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include <mpfr.h>

#include <sympp/functions/exponential.hpp>
#include <sympp/functions/hyperbolic.hpp>
#include <sympp/functions/miscellaneous.hpp>
#include <sympp/functions/trigonometric.hpp>

#include <sympp/core/basic.hpp>
#include <sympp/core/boolean.hpp>
#include <sympp/core/float.hpp>
#include <sympp/core/integer.hpp>
#include <sympp/core/operators.hpp>
#include <sympp/core/symbol.hpp>
#include <sympp/core/type_id.hpp>
#include <sympp/matlab/matlab.hpp>
#include <sympp/matrices/matrix.hpp>
#include <sympp/printing/printing.hpp>
#include <sympp/solvers/solve.hpp>
#include <sympp/ode/dsolve.hpp>

namespace {

/* matlab_sym_s holds a sympp::Expr by value. SymPP's Expr is itself a
 * shared_ptr<const Basic>, so copies are cheap and the wrapper does
 * not need its own refcount. */
struct SymBox {
    sympp::Expr expr;
};

/* Heap-owned wrapper around sympp::Matrix. Same lifecycle model as
 * SymBox: the matlab_symmat C ABI hands out new MatBox*'s and the
 * caller frees via matlab_symmat_free. */
struct MatBox {
    sympp::Matrix mat;
};

inline matlab_symmat *boxm(sympp::Matrix m) {
    auto *b = new MatBox{std::move(m)};
    return reinterpret_cast<matlab_symmat *>(b);
}
inline const sympp::Matrix &unboxm(const matlab_symmat *p) {
    return reinterpret_cast<const MatBox *>(p)->mat;
}
inline sympp::Matrix &unboxm_mut(matlab_symmat *p) {
    return reinterpret_cast<MatBox *>(p)->mat;
}

inline matlab_sym *box(sympp::Expr e) {
    auto *b = new SymBox{std::move(e)};
    return reinterpret_cast<matlab_sym *>(b);
}

inline const sympp::Expr &unbox(const matlab_sym *p) {
    return reinterpret_cast<const SymBox *>(p)->expr;
}

inline sympp::Expr d2expr(double v) {
    /* Match MATLAB's sym(double): when v is integral, build an Integer
     * exactly; otherwise a Float (loses no precision worse than double
     * input did). */
    if (v == static_cast<double>(static_cast<long long>(v))) {
        return sympp::integer(static_cast<long>(v));
    }
    return sympp::float_value(v);
}

inline char *dup_to_c(const std::string &s, int64_t *len_out) {
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    if (!out) {
        if (len_out) *len_out = 0;
        return nullptr;
    }
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    if (len_out) *len_out = static_cast<int64_t>(s.size());
    return out;
}

inline std::string borrow(const char *s, int64_t n) {
    if (!s || n <= 0) return {};
    return std::string(s, static_cast<size_t>(n));
}

}  // namespace

extern "C" {

/* --- Construction --------------------------------------------------------- */

matlab_sym *matlab_sym_from_str(const char *s, int64_t n) {
    return box(sympp::matlab::sym(borrow(s, n)));
}

matlab_sym *matlab_sym_str2sym(const char *s, int64_t n) {
    return box(sympp::matlab::str2sym(borrow(s, n)));
}

matlab_sym *matlab_sym_from_double(double v) { return box(d2expr(v)); }

matlab_sym *matlab_sym_from_i64(int64_t v) {
    return box(sympp::integer(static_cast<long>(v)));
}

matlab_sym *matlab_sym_named(const char *s, int64_t n) {
    return box(sympp::symbol(borrow(s, n)));
}

/* --- Algebra -------------------------------------------------------------- */

matlab_sym *matlab_sym_add(const matlab_sym *a, const matlab_sym *b) {
    return box(unbox(a) + unbox(b));
}
matlab_sym *matlab_sym_sub(const matlab_sym *a, const matlab_sym *b) {
    return box(unbox(a) - unbox(b));
}
matlab_sym *matlab_sym_mul(const matlab_sym *a, const matlab_sym *b) {
    return box(unbox(a) * unbox(b));
}
matlab_sym *matlab_sym_div(const matlab_sym *a, const matlab_sym *b) {
    return box(unbox(a) / unbox(b));
}
matlab_sym *matlab_sym_pow(const matlab_sym *a, const matlab_sym *b) {
    return box(sympp::pow(unbox(a), unbox(b)));
}
matlab_sym *matlab_sym_neg(const matlab_sym *a) { return box(-unbox(a)); }

matlab_sym *matlab_sym_add_d(const matlab_sym *a, double b) {
    return box(unbox(a) + d2expr(b));
}
matlab_sym *matlab_sym_sub_d(const matlab_sym *a, double b) {
    return box(unbox(a) - d2expr(b));
}
matlab_sym *matlab_sym_mul_d(const matlab_sym *a, double b) {
    return box(unbox(a) * d2expr(b));
}
matlab_sym *matlab_sym_div_d(const matlab_sym *a, double b) {
    return box(unbox(a) / d2expr(b));
}
matlab_sym *matlab_sym_pow_d(const matlab_sym *a, double b) {
    return box(sympp::pow(unbox(a), d2expr(b)));
}
matlab_sym *matlab_sym_d_sub(double a, const matlab_sym *b) {
    return box(d2expr(a) - unbox(b));
}
matlab_sym *matlab_sym_d_div(double a, const matlab_sym *b) {
    return box(d2expr(a) / unbox(b));
}
matlab_sym *matlab_sym_d_pow(double a, const matlab_sym *b) {
    return box(sympp::pow(d2expr(a), unbox(b)));
}

matlab_sym *matlab_sym_eq(const matlab_sym *a, const matlab_sym *b) {
    return box(sympp::eq(unbox(a), unbox(b)));
}
matlab_sym *matlab_sym_eq_d(const matlab_sym *a, double b) {
    return box(sympp::eq(unbox(a), d2expr(b)));
}

/* Elementary-function overloads. Each maps to the same-named SymPP free
 * function (declared in functions/trigonometric.hpp / hyperbolic.hpp /
 * exponential.hpp / miscellaneous.hpp). */
matlab_sym *matlab_sym_sin(const matlab_sym *a) { return box(sympp::sin(unbox(a))); }
matlab_sym *matlab_sym_cos(const matlab_sym *a) { return box(sympp::cos(unbox(a))); }
matlab_sym *matlab_sym_tan(const matlab_sym *a) { return box(sympp::tan(unbox(a))); }
matlab_sym *matlab_sym_asin(const matlab_sym *a) { return box(sympp::asin(unbox(a))); }
matlab_sym *matlab_sym_acos(const matlab_sym *a) { return box(sympp::acos(unbox(a))); }
matlab_sym *matlab_sym_atan(const matlab_sym *a) { return box(sympp::atan(unbox(a))); }
matlab_sym *matlab_sym_sinh(const matlab_sym *a) { return box(sympp::sinh(unbox(a))); }
matlab_sym *matlab_sym_cosh(const matlab_sym *a) { return box(sympp::cosh(unbox(a))); }
matlab_sym *matlab_sym_tanh(const matlab_sym *a) { return box(sympp::tanh(unbox(a))); }
matlab_sym *matlab_sym_exp(const matlab_sym *a) { return box(sympp::exp(unbox(a))); }
matlab_sym *matlab_sym_log(const matlab_sym *a) { return box(sympp::log(unbox(a))); }
matlab_sym *matlab_sym_sqrt(const matlab_sym *a) { return box(sympp::sqrt(unbox(a))); }
matlab_sym *matlab_sym_abs(const matlab_sym *a) { return box(sympp::abs(unbox(a))); }

/* --- Calculus ------------------------------------------------------------- */

matlab_sym *matlab_sym_diff(const matlab_sym *f, const matlab_sym *var) {
    return box(sympp::matlab::diff(unbox(f), unbox(var)));
}
matlab_sym *matlab_sym_diff_n(const matlab_sym *f, const matlab_sym *var,
                              int64_t n) {
    if (n < 0) n = 0;
    return box(sympp::matlab::diff(unbox(f), unbox(var),
                                      static_cast<std::size_t>(n)));
}
matlab_sym *matlab_sym_int(const matlab_sym *f, const matlab_sym *var) {
    return box(sympp::matlab::Int(unbox(f), unbox(var)));
}
matlab_sym *matlab_sym_int_def(const matlab_sym *f, const matlab_sym *var,
                               const matlab_sym *a, const matlab_sym *b) {
    return box(sympp::matlab::Int(unbox(f), unbox(var), unbox(a), unbox(b)));
}

/* --- Manipulation --------------------------------------------------------- */

matlab_sym *matlab_sym_simplify(const matlab_sym *e) {
    return box(sympp::matlab::simplify(unbox(e)));
}
matlab_sym *matlab_sym_expand(const matlab_sym *e) {
    return box(sympp::matlab::expand(unbox(e)));
}
matlab_sym *matlab_sym_factor(const matlab_sym *e, const matlab_sym *var) {
    return box(sympp::matlab::factor(unbox(e), unbox(var)));
}
matlab_sym *matlab_sym_subs(const matlab_sym *e, const matlab_sym *old_e,
                            const matlab_sym *new_e) {
    return box(sympp::matlab::subs(unbox(e), unbox(old_e), unbox(new_e)));
}

/* --- Solver --------------------------------------------------------------- */

matlab_sym *matlab_sym_solve_one(const matlab_sym *eq, const matlab_sym *var) {
    int64_t n = 0;
    matlab_sym **roots = matlab_sym_solve(eq, var, &n);
    if (n == 0) {
        if (roots) std::free(roots);
        /* Match MATLAB's "no solution" display: an empty sym whose
         * pretty form is the empty string. The caller can detect this
         * at the language level via length(r) == 0 in Phase B. */
        return matlab_sym_named("NoSolution", 10);
    }
    if (n == 1) {
        matlab_sym *only = roots[0];
        std::free(roots);
        return only;
    }
    /* Multi-root: build "[r1, r2, ...]" as a single sym whose pretty
     * form is human-readable. Phase B replaces this with a real symbolic
     * column-vector return. */
    std::string buf = "[";
    for (int64_t i = 0; i < n; ++i) {
        if (i) buf += ", ";
        buf += sympp::printing::pretty(unbox(roots[i]));
        matlab_sym_free(roots[i]);
    }
    buf += "]";
    std::free(roots);
    /* Wrap the rendered text in a Symbol so the value still has a
     * matlab_sym type. The pretty-printer will surface the brackets. */
    return box(sympp::symbol(buf));
}

matlab_symmat *matlab_sym_solve_sys(const matlab_sym *const *eqs, int64_t n_eqs,
                                    const matlab_sym *const *vars, int64_t n_vars) {
    std::vector<sympp::Expr> ev, vv;
    ev.reserve(static_cast<size_t>(n_eqs > 0 ? n_eqs : 0));
    vv.reserve(static_cast<size_t>(n_vars > 0 ? n_vars : 0));
    for (int64_t i = 0; i < n_eqs; ++i) {
        /* Like the single-eq path: lift Relational to lhs - rhs. */
        const auto &E = unbox(eqs[i]);
        if (E && E->type_id() == sympp::TypeId::Relational) {
            const auto *rel = static_cast<const sympp::Relational *>(E.get());
            ev.push_back(rel->lhs() - rel->rhs());
        } else {
            ev.push_back(E);
        }
    }
    for (int64_t i = 0; i < n_vars; ++i) vv.push_back(unbox(vars[i]));
    /* Pick the algorithm: 2x2 polynomial → resultant nonlinsolve;
     * everything else → Gröbner triangularisation. Both return a list
     * of joint solution vectors (each vector has n_vars entries). */
    std::vector<std::vector<sympp::Expr>> sols;
    if (n_eqs == 2 && n_vars == 2) {
        sols = sympp::nonlinsolve(ev, vv);
    } else {
        sols = sympp::nonlinsolve_groebner(ev, vv);
    }
    auto k = sols.size();
    auto n = static_cast<size_t>(n_vars > 0 ? n_vars : 0);
    sympp::Matrix M(k, n);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n && j < sols[i].size(); ++j) {
            M.set(i, j, sols[i][j]);
        }
    }
    return boxm(std::move(M));
}

matlab_symmat *matlab_sym_solve_2x2(const matlab_sym *eq1, const matlab_sym *eq2,
                                    const matlab_sym *var1, const matlab_sym *var2) {
    const matlab_sym *eqs[2] = {eq1, eq2};
    const matlab_sym *vars[2] = {var1, var2};
    return matlab_sym_solve_sys(eqs, 2, vars, 2);
}
matlab_symmat *matlab_sym_solve_3x3(const matlab_sym *eq1, const matlab_sym *eq2,
                                    const matlab_sym *eq3,
                                    const matlab_sym *var1, const matlab_sym *var2,
                                    const matlab_sym *var3) {
    const matlab_sym *eqs[3] = {eq1, eq2, eq3};
    const matlab_sym *vars[3] = {var1, var2, var3};
    return matlab_sym_solve_sys(eqs, 3, vars, 3);
}

matlab_sym **matlab_sym_solve(const matlab_sym *eq, const matlab_sym *var,
                              int64_t *n_out) {
    /* MATLAB's solve(f == 0, x) passes a Relational. SymPP's solve takes
     * an expression treated as the LHS with implicit RHS = 0, so when we
     * see a Relational, lift its (lhs - rhs) and forward. Anything else
     * is treated as the LHS directly. */
    sympp::Expr lhs;
    const auto &E = unbox(eq);
    if (E && E->type_id() == sympp::TypeId::Relational) {
        const auto *rel = static_cast<const sympp::Relational *>(E.get());
        lhs = rel->lhs() - rel->rhs();
    } else {
        lhs = E;
    }
    auto roots = sympp::matlab::solve(lhs, unbox(var));
    auto n = static_cast<int64_t>(roots.size());
    if (n_out) *n_out = n;
    if (n == 0) return nullptr;
    auto **out = static_cast<matlab_sym **>(
        std::malloc(static_cast<size_t>(n) * sizeof(matlab_sym *)));
    if (!out) {
        if (n_out) *n_out = 0;
        return nullptr;
    }
    for (int64_t i = 0; i < n; ++i) out[i] = box(std::move(roots[i]));
    return out;
}

/* --- Conversion / display ------------------------------------------------- */

double matlab_sym_double(const matlab_sym *e) {
    auto numeric = sympp::evalf(unbox(e), 17);
    if (numeric && numeric->type_id() == sympp::TypeId::Float) {
        const auto *f = static_cast<const sympp::Float *>(numeric.get());
        return mpfr_get_d(f->value(), MPFR_RNDN);
    }
    if (numeric && numeric->type_id() == sympp::TypeId::Integer) {
        const auto *i = static_cast<const sympp::Integer *>(numeric.get());
        return static_cast<double>(i->to_long());
    }
    /* Free symbol still present — caller probably expected a closed form. */
    return std::numeric_limits<double>::quiet_NaN();
}

void matlab_sym_disp(const matlab_sym *e) {
    if (!e) {
        std::printf("\n");
        return;
    }
    auto s = sympp::printing::pretty(unbox(e));
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::printf("\n");
}

char *matlab_sym_str(const matlab_sym *e, int64_t *len_out) {
    if (!e) return dup_to_c({}, len_out);
    return dup_to_c(sympp::printing::pretty(unbox(e)), len_out);
}
char *matlab_sym_latex(const matlab_sym *e, int64_t *len_out) {
    if (!e) return dup_to_c({}, len_out);
    return dup_to_c(sympp::printing::latex(unbox(e)), len_out);
}
char *matlab_sym_ccode(const matlab_sym *e, int64_t *len_out) {
    if (!e) return dup_to_c({}, len_out);
    return dup_to_c(sympp::printing::ccode(unbox(e)), len_out);
}

/* --- Lifecycle ------------------------------------------------------------ */

void matlab_sym_free(matlab_sym *e) {
    if (!e) return;
    delete reinterpret_cast<SymBox *>(e);
}

/* --- Phase B: assumptions -------------------------------------------------
 *
 * MATLAB's assume() mutates the named symbol. SymPP can't mutate (Symbols
 * are interned by name+mask), so the facade returns a fresh sym carrying
 * the new mask and the runtime returns it as the new binding value. The
 * SymPP side-table records the assumption for future re-creations of the
 * same name; refresh() rebuilds a Symbol from the current side-table. */

matlab_sym *matlab_sym_assume(const matlab_sym *x,
                              const char *prop, int64_t prop_len) {
    sympp::matlab::assume(unbox(x), borrow(prop, prop_len));
    return box(sympp::matlab::refresh(unbox(x)));
}
matlab_sym *matlab_sym_assumeAlso(const matlab_sym *x,
                                  const char *prop, int64_t prop_len) {
    sympp::matlab::assumeAlso(unbox(x), borrow(prop, prop_len));
    return box(sympp::matlab::refresh(unbox(x)));
}
char *matlab_sym_assumptions(const matlab_sym *x, int64_t *len_out) {
    auto props = sympp::matlab::assumptions(unbox(x));
    std::string joined;
    for (size_t i = 0; i < props.size(); ++i) {
        if (i) joined += ',';
        joined += props[i];
    }
    return dup_to_c(joined, len_out);
}
matlab_sym *matlab_sym_clearAssumptions(const matlab_sym *x) {
    sympp::matlab::clearAssumptions(unbox(x));
    /* Return a fresh symbol with no assumptions; SymPP's interning
     * gives us back the bare symbol() since the mask is now empty. */
    return box(sympp::symbol(unbox(x)->str()));
}

/* --- Phase B: vpa / taylor / limit ---------------------------------------- */

matlab_sym *matlab_sym_vpa(const matlab_sym *e, int64_t dps) {
    return box(sympp::matlab::vpa(unbox(e), static_cast<int>(dps > 0 ? dps : 32)));
}
matlab_sym *matlab_sym_taylor(const matlab_sym *f, const matlab_sym *var,
                              const matlab_sym *a, int64_t n) {
    return box(sympp::matlab::taylor(unbox(f), unbox(var), unbox(a),
                                       static_cast<std::size_t>(n > 0 ? n : 6)));
}
matlab_sym *matlab_sym_limit(const matlab_sym *f, const matlab_sym *var,
                             const matlab_sym *target) {
    return box(sympp::matlab::limit(unbox(f), unbox(var), unbox(target)));
}

/* --- Phase B: dsolve / pdsolve -------------------------------------------- */

matlab_sym *matlab_sym_dsolve(const matlab_sym *eq, const matlab_sym *y,
                              const matlab_sym *yp, const matlab_sym *x) {
    return box(sympp::matlab::dsolve(unbox(eq), unbox(y), unbox(yp), unbox(x)));
}
matlab_sym *matlab_sym_dsolve_2(const matlab_sym *eq, const matlab_sym *y,
                                const matlab_sym *yp, const matlab_sym *ypp,
                                const matlab_sym *x) {
    return box(sympp::matlab::dsolve(unbox(eq), unbox(y), unbox(yp),
                                       unbox(ypp), unbox(x)));
}

matlab_sym *matlab_sym_pdsolve(const matlab_sym *a, const matlab_sym *b,
                               const matlab_sym *c,
                               const matlab_sym *x, const matlab_sym *y) {
    return box(sympp::matlab::pdsolve(unbox(a), unbox(b), unbox(c),
                                        unbox(x), unbox(y)));
}
matlab_sym *matlab_sym_pdsolve_heat(const matlab_sym *k,
                                    const matlab_sym *lambda,
                                    const matlab_sym *x,
                                    const matlab_sym *t) {
    return box(sympp::matlab::pdsolve_heat(unbox(k), unbox(lambda),
                                             unbox(x), unbox(t)));
}
matlab_sym *matlab_sym_pdsolve_wave(const matlab_sym *c,
                                    const matlab_sym *x,
                                    const matlab_sym *t) {
    return box(sympp::matlab::pdsolve_wave(unbox(c), unbox(x), unbox(t)));
}

/* --- Phase 6.1: symbolic matrices ----------------------------------------- */

matlab_symmat *matlab_symmat_new(int64_t rows, int64_t cols,
                                 const matlab_sym *const *data) {
    if (rows < 0) rows = 0;
    if (cols < 0) cols = 0;
    auto r = static_cast<std::size_t>(rows);
    auto c = static_cast<std::size_t>(cols);
    std::vector<sympp::Expr> flat;
    flat.reserve(r * c);
    for (std::size_t i = 0; i < r * c; ++i)
        flat.push_back(data && data[i] ? unbox(data[i]) : sympp::integer(0));
    return boxm(sympp::Matrix(r, c, std::move(flat)));
}

matlab_symmat *matlab_symmat_zeros(int64_t rows, int64_t cols) {
    if (rows < 0) rows = 0;
    if (cols < 0) cols = 0;
    return boxm(sympp::Matrix::zeros(static_cast<std::size_t>(rows),
                                       static_cast<std::size_t>(cols)));
}

matlab_symmat *matlab_symmat_eye(int64_t n) {
    if (n < 0) n = 0;
    return boxm(sympp::Matrix::identity(static_cast<std::size_t>(n)));
}

int64_t matlab_symmat_rows(const matlab_symmat *m) {
    return m ? static_cast<int64_t>(unboxm(m).rows()) : 0;
}
int64_t matlab_symmat_cols(const matlab_symmat *m) {
    return m ? static_cast<int64_t>(unboxm(m).cols()) : 0;
}

matlab_sym *matlab_symmat_get(const matlab_symmat *m, int64_t r, int64_t c) {
    if (!m) return nullptr;
    return box(unboxm(m).at(static_cast<std::size_t>(r),
                              static_cast<std::size_t>(c)));
}
void matlab_symmat_set(matlab_symmat *m, int64_t r, int64_t c,
                       const matlab_sym *v) {
    if (!m || !v) return;
    unboxm_mut(m).set(static_cast<std::size_t>(r),
                       static_cast<std::size_t>(c), unbox(v));
}

matlab_symmat *matlab_symmat_add(const matlab_symmat *a, const matlab_symmat *b) {
    return boxm(unboxm(a) + unboxm(b));
}
matlab_symmat *matlab_symmat_sub(const matlab_symmat *a, const matlab_symmat *b) {
    return boxm(unboxm(a) - unboxm(b));
}
matlab_symmat *matlab_symmat_mul(const matlab_symmat *a, const matlab_symmat *b) {
    return boxm(unboxm(a) * unboxm(b));
}
matlab_symmat *matlab_symmat_scalar_mul(const matlab_symmat *a,
                                        const matlab_sym *s) {
    return boxm(unboxm(a).scalar_mul(unbox(s)));
}
matlab_symmat *matlab_symmat_transpose(const matlab_symmat *a) {
    return boxm(unboxm(a).transpose());
}
matlab_symmat *matlab_symmat_inverse(const matlab_symmat *a) {
    return boxm(unboxm(a).inverse());
}
matlab_sym *matlab_symmat_det(const matlab_symmat *a) {
    return box(unboxm(a).det());
}
matlab_sym *matlab_symmat_trace(const matlab_symmat *a) {
    return box(unboxm(a).trace());
}
int64_t matlab_symmat_rank(const matlab_symmat *a) {
    return a ? static_cast<int64_t>(unboxm(a).rank()) : 0;
}

matlab_sym **matlab_symmat_eigenvals(const matlab_symmat *a, int64_t *n_out) {
    auto eig = unboxm(a).eigenvals();
    auto k = static_cast<int64_t>(eig.size());
    if (n_out) *n_out = k;
    if (k == 0) return nullptr;
    auto **out = static_cast<matlab_sym **>(
        std::malloc(static_cast<size_t>(k) * sizeof(matlab_sym *)));
    if (!out) {
        if (n_out) *n_out = 0;
        return nullptr;
    }
    for (int64_t i = 0; i < k; ++i) out[i] = box(std::move(eig[i]));
    return out;
}

int matlab_symmat_lu(const matlab_symmat *a,
                     matlab_symmat **L_out, matlab_symmat **U_out) {
    try {
        auto pr = unboxm(a).lu();
        if (L_out) *L_out = boxm(std::move(pr.first));
        if (U_out) *U_out = boxm(std::move(pr.second));
        return 0;
    } catch (...) {
        if (L_out) *L_out = nullptr;
        if (U_out) *U_out = nullptr;
        return 1;
    }
}
int matlab_symmat_qr(const matlab_symmat *a,
                     matlab_symmat **Q_out, matlab_symmat **R_out) {
    try {
        auto pr = unboxm(a).qr();
        if (Q_out) *Q_out = boxm(std::move(pr.first));
        if (R_out) *R_out = boxm(std::move(pr.second));
        return 0;
    } catch (...) {
        if (Q_out) *Q_out = nullptr;
        if (R_out) *R_out = nullptr;
        return 1;
    }
}
matlab_symmat *matlab_symmat_cholesky(const matlab_symmat *a) {
    try { return boxm(unboxm(a).cholesky()); }
    catch (...) { return nullptr; }
}

matlab_symmat *matlab_symmat_linsolve(const matlab_symmat *A,
                                      const matlab_symmat *b) {
    return boxm(sympp::linsolve(unboxm(A), unboxm(b)));
}

matlab_symmat *matlab_symmat_dsolve_system(const matlab_symmat *A,
                                           const matlab_sym *x) {
    return boxm(sympp::dsolve_system(unboxm(A), unbox(x)));
}

void matlab_symmat_disp(const matlab_symmat *m) {
    if (!m) { std::printf("\n"); return; }
    auto s = unboxm(m).str();
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::printf("\n");
}
char *matlab_symmat_str(const matlab_symmat *m, int64_t *len_out) {
    if (!m) return dup_to_c({}, len_out);
    return dup_to_c(unboxm(m).str(), len_out);
}
void matlab_symmat_free(matlab_symmat *m) {
    if (!m) return;
    delete reinterpret_cast<MatBox *>(m);
}

/* --- Phase 6.1: easy-win extras ------------------------------------------- */

matlab_sym *matlab_sym_nsolve(const matlab_sym *eq, const matlab_sym *var,
                              const matlab_sym *x0, int64_t dps) {
    return box(sympp::nsolve(unbox(eq), unbox(var), unbox(x0),
                              static_cast<int>(dps > 0 ? dps : 15)));
}

matlab_sym *matlab_sym_vpasolve(const matlab_sym *eq, const matlab_sym *var,
                                const matlab_sym *x0, int64_t dps) {
    /* MATLAB's vpasolve is a numeric Newton solve at variable precision —
     * SymPP's nsolve is the closest match. The default dps is 32 (matches
     * MATLAB's default vpa precision). */
    return matlab_sym_nsolve(eq, var, x0, dps > 0 ? dps : 32);
}

matlab_sym *matlab_sym_rsolve(const matlab_sym *const *coeffs, int64_t n_coef,
                              const matlab_sym *n) {
    std::vector<sympp::Expr> coef_vec;
    coef_vec.reserve(static_cast<size_t>(n_coef > 0 ? n_coef : 0));
    for (int64_t i = 0; i < n_coef; ++i) coef_vec.push_back(unbox(coeffs[i]));
    return box(sympp::rsolve(coef_vec, unbox(n)));
}

matlab_sym *matlab_sym_checkodesol(const matlab_sym *eq, const matlab_sym *sol,
                                   const matlab_sym *y, const matlab_sym *yp,
                                   const matlab_sym *x) {
    return box(sympp::checkodesol(unbox(eq), unbox(sol), unbox(y), unbox(yp),
                                    unbox(x)));
}

/* Pair-up two parallel matlab_sym* arrays into the SymPP IVP shape.
 * Implementation detail of dsolve_ivp / apply_ivp; declared static so
 * the C-linkage diagnostic doesn't fire on its non-C return type. */
static std::vector<std::pair<sympp::Expr, sympp::Expr>>
collect_ivp_pairs_impl(int64_t n, const matlab_sym *const *xs,
                       const matlab_sym *const *ys) {
    std::vector<std::pair<sympp::Expr, sympp::Expr>> out;
    out.reserve(static_cast<size_t>(n > 0 ? n : 0));
    for (int64_t i = 0; i < n; ++i)
        out.emplace_back(unbox(xs[i]), unbox(ys[i]));
    return out;
}

matlab_sym *matlab_sym_dsolve_ivp(const matlab_sym *eq, const matlab_sym *y,
                                  const matlab_sym *yp, const matlab_sym *x,
                                  int64_t n_conds,
                                  const matlab_sym *const *x_vals,
                                  const matlab_sym *const *y_vals) {
    auto conds = collect_ivp_pairs_impl(n_conds, x_vals, y_vals);
    return box(sympp::matlab::dsolve_ivp(unbox(eq), unbox(y), unbox(yp),
                                            unbox(x), conds));
}

matlab_sym *matlab_sym_apply_ivp(const matlab_sym *general_solution,
                                 const matlab_sym *x, int64_t n_conds,
                                 const matlab_sym *const *x_vals,
                                 const matlab_sym *const *y_vals) {
    auto conds = collect_ivp_pairs_impl(n_conds, x_vals, y_vals);
    return box(sympp::apply_ivp(unbox(general_solution), unbox(x), conds));
}

matlab_sym *matlab_sym_dsolve_ivp_1(const matlab_sym *eq, const matlab_sym *y,
                                    const matlab_sym *yp, const matlab_sym *x,
                                    const matlab_sym *x0, const matlab_sym *y0) {
    const matlab_sym *xs[1] = {x0};
    const matlab_sym *ys[1] = {y0};
    return matlab_sym_dsolve_ivp(eq, y, yp, x, 1, xs, ys);
}
matlab_sym *matlab_sym_apply_ivp_1(const matlab_sym *general_solution,
                                   const matlab_sym *x,
                                   const matlab_sym *x0, const matlab_sym *y0) {
    const matlab_sym *xs[1] = {x0};
    const matlab_sym *ys[1] = {y0};
    return matlab_sym_apply_ivp(general_solution, x, 1, xs, ys);
}

matlab_sym *matlab_sym_solve_inequality(const matlab_sym *lhs,
                                        const matlab_sym *rhs,
                                        const char *op, int64_t op_len,
                                        const matlab_sym *var) {
    /* The SymPP entry takes the relation as an Expr (Symbol) carrying
     * the operator string — the call site builds that from the borrowed
     * char buffer. Result is a SetPtr; we render its str() into a
     * Symbol so the matlab_sym layer stays uniform. */
    sympp::Expr op_sym = sympp::symbol(borrow(op, op_len));
    sympp::SetPtr s = sympp::solve_univariate_inequality(
        unbox(lhs), op_sym, unbox(rhs), unbox(var));
    return box(sympp::symbol(s ? s->str() : std::string("EmptySet")));
}

matlab_sym *matlab_sym_reduce_inequalities(const matlab_sym *rel,
                                           const matlab_sym *var) {
    sympp::SetPtr s = sympp::reduce_inequalities(unbox(rel), unbox(var));
    return box(sympp::symbol(s ? s->str() : std::string("EmptySet")));
}

matlab_sym **matlab_sym_groebner(const matlab_sym *const *eqs, int64_t n_eqs,
                                 const matlab_sym *const *vars, int64_t n_vars,
                                 int64_t *n_basis_out) {
    std::vector<sympp::Expr> eq_vec, var_vec;
    eq_vec.reserve(static_cast<size_t>(n_eqs > 0 ? n_eqs : 0));
    var_vec.reserve(static_cast<size_t>(n_vars > 0 ? n_vars : 0));
    for (int64_t i = 0; i < n_eqs; ++i) eq_vec.push_back(unbox(eqs[i]));
    for (int64_t i = 0; i < n_vars; ++i) var_vec.push_back(unbox(vars[i]));
    auto basis = sympp::groebner(eq_vec, var_vec);
    auto k = static_cast<int64_t>(basis.size());
    if (n_basis_out) *n_basis_out = k;
    if (k == 0) return nullptr;
    auto **out = static_cast<matlab_sym **>(
        std::malloc(static_cast<size_t>(k) * sizeof(matlab_sym *)));
    if (!out) {
        if (n_basis_out) *n_basis_out = 0;
        return nullptr;
    }
    for (int64_t i = 0; i < k; ++i) out[i] = box(std::move(basis[i]));
    return out;
}

int matlab_sym_linear_diophantine(const matlab_sym *a, const matlab_sym *b,
                                  const matlab_sym *c,
                                  matlab_sym **x_out, matlab_sym **y_out) {
    auto pair = sympp::linear_diophantine(unbox(a), unbox(b), unbox(c));
    if (!pair) {
        if (x_out) *x_out = nullptr;
        if (y_out) *y_out = nullptr;
        return 0;
    }
    if (x_out) *x_out = box(pair->first);
    if (y_out) *y_out = box(pair->second);
    return 1;
}

matlab_sym **matlab_sym_pythagorean_triples(int64_t max_z,
                                            int64_t *n_triples_out) {
    auto triples = sympp::pythagorean_triples(static_cast<long>(max_z));
    auto k = static_cast<int64_t>(triples.size());
    if (n_triples_out) *n_triples_out = k;
    if (k == 0) return nullptr;
    auto **out = static_cast<matlab_sym **>(
        std::malloc(static_cast<size_t>(k) * 3 * sizeof(matlab_sym *)));
    if (!out) {
        if (n_triples_out) *n_triples_out = 0;
        return nullptr;
    }
    for (int64_t i = 0; i < k; ++i) {
        out[3 * i + 0] = box(triples[i][0]);
        out[3 * i + 1] = box(triples[i][1]);
        out[3 * i + 2] = box(triples[i][2]);
    }
    return out;
}

/* --- Phase C: integral transforms ----------------------------------------- */

matlab_sym *matlab_sym_laplace(const matlab_sym *f,
                               const matlab_sym *t, const matlab_sym *s) {
    return box(sympp::matlab::laplace(unbox(f), unbox(t), unbox(s)));
}
matlab_sym *matlab_sym_ilaplace(const matlab_sym *F,
                                const matlab_sym *s, const matlab_sym *t) {
    return box(sympp::matlab::ilaplace(unbox(F), unbox(s), unbox(t)));
}
matlab_sym *matlab_sym_fourier(const matlab_sym *f,
                               const matlab_sym *t, const matlab_sym *w) {
    return box(sympp::matlab::fourier(unbox(f), unbox(t), unbox(w)));
}
matlab_sym *matlab_sym_ifourier(const matlab_sym *F,
                                const matlab_sym *w, const matlab_sym *t) {
    return box(sympp::matlab::ifourier(unbox(F), unbox(w), unbox(t)));
}
matlab_sym *matlab_sym_ztrans(const matlab_sym *f,
                              const matlab_sym *n, const matlab_sym *z) {
    return box(sympp::matlab::ztrans(unbox(f), unbox(n), unbox(z)));
}
matlab_sym *matlab_sym_iztrans(const matlab_sym *F,
                               const matlab_sym *z, const matlab_sym *n) {
    return box(sympp::matlab::iztrans(unbox(F), unbox(z), unbox(n)));
}

}  // extern "C"
