// runtime/toolbox/sym/runtime_sym_stub.cpp — dlopen forwarding stubs.
//
// #50 Phase 3.  Provides the matlab_sym_* / matlab_symmat_* C ABI as
// stub wrappers that dlopen libmatlab_sym.dylib on first call and
// forward via dlsym.  matlabc links these stubs; they're force-loaded
// into MatlabRuntime so the JIT can see them, but the actual libsympp
// only loads when a sym `.m` invokes a sym builtin.
//
// This file is AUTO-GENERATED from runtime_sym.h.  Edit the generator
// (scripts/gen_sym_stub.py), not this file.
//
// Pre-#50 Phase 3, matlabc statically linked libsympp via the
// SymPP::sympp CMake target — every matlabc binary on disk carried
// an LC_LOAD_DYLIB for libsympp regardless of whether the running
// program used sym.  After this phase the load is lazy.

#include "runtime_sym.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace {

void *libsym = nullptr;
std::atomic_flag init_lock = ATOMIC_FLAG_INIT;

void ensure_loaded() {
  if (libsym) return;
  while (init_lock.test_and_set(std::memory_order_acquire)) { /* spin */ }
  if (!libsym) {
    static const char *candidates[] = {
        /* In-tree development build: alongside matlabc. */
        "@executable_path/libmatlab_sym.dylib",
        "@executable_path/../lib/libmatlab_sym.dylib",
        "$ORIGIN/libmatlab_sym.so",
        "$ORIGIN/../lib/libmatlab_sym.so",
        /* Installed locations. */
        "/opt/homebrew/lib/libmatlab_sym.dylib",
        "/usr/local/lib/libmatlab_sym.dylib",
        "/usr/lib/libmatlab_sym.so",
        "/usr/local/lib/libmatlab_sym.so",
        /* Bare names — let the dynamic linker search its default path. */
        "libmatlab_sym.dylib",
        "libmatlab_sym.so",
        nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
      libsym = dlopen(candidates[i], RTLD_LAZY | RTLD_GLOBAL);
      if (libsym) break;
    }
    if (!libsym) {
      std::fprintf(stderr,
          "Symbolic Math Toolbox backend (libmatlab_sym) not found.\n"
          "  Build matlabc with -DMATLAB_LLVM_WITH_SYM=ON to produce the\n"
          "  shared library alongside the matlabc binary.\n"
          "  (dlerror: %s)\n",
          dlerror());
      std::exit(1);
    }
  }
  init_lock.clear(std::memory_order_release);
}

void *resolve(const char *name) {
  ensure_loaded();
  void *sym = dlsym(libsym, name);
  if (!sym) {
    std::fprintf(stderr,
        "Symbolic Math Toolbox: symbol '%s' not found in libmatlab_sym (%s)\n",
        name, dlerror());
    std::exit(1);
  }
  return sym;
}

}  // namespace

extern "C" {

matlab_sym * matlab_sym_from_str(const char *s, int64_t n) {
  using FnT = matlab_sym * (*)(const char *s, int64_t n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_from_str"));
  return fp(s, n);
}

matlab_sym * matlab_sym_str2sym(const char *s, int64_t n) {
  using FnT = matlab_sym * (*)(const char *s, int64_t n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_str2sym"));
  return fp(s, n);
}

matlab_sym * matlab_sym_from_double(double v) {
  using FnT = matlab_sym * (*)(double v);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_from_double"));
  return fp(v);
}

matlab_sym * matlab_sym_from_i64(int64_t v) {
  using FnT = matlab_sym * (*)(int64_t v);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_from_i64"));
  return fp(v);
}

matlab_sym * matlab_sym_named(const char *s, int64_t n) {
  using FnT = matlab_sym * (*)(const char *s, int64_t n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_named"));
  return fp(s, n);
}

matlab_sym * matlab_sym_add(const matlab_sym *a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_add"));
  return fp(a, b);
}

matlab_sym * matlab_sym_sub(const matlab_sym *a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_sub"));
  return fp(a, b);
}

matlab_sym * matlab_sym_mul(const matlab_sym *a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_mul"));
  return fp(a, b);
}

matlab_sym * matlab_sym_div(const matlab_sym *a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_div"));
  return fp(a, b);
}

matlab_sym * matlab_sym_pow(const matlab_sym *a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_pow"));
  return fp(a, b);
}

matlab_sym * matlab_sym_neg(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_neg"));
  return fp(a);
}

matlab_sym * matlab_sym_add_d(const matlab_sym *a, double b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, double b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_add_d"));
  return fp(a, b);
}

matlab_sym * matlab_sym_sub_d(const matlab_sym *a, double b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, double b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_sub_d"));
  return fp(a, b);
}

matlab_sym * matlab_sym_mul_d(const matlab_sym *a, double b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, double b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_mul_d"));
  return fp(a, b);
}

matlab_sym * matlab_sym_div_d(const matlab_sym *a, double b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, double b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_div_d"));
  return fp(a, b);
}

matlab_sym * matlab_sym_pow_d(const matlab_sym *a, double b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, double b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_pow_d"));
  return fp(a, b);
}

matlab_sym * matlab_sym_d_sub(double a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(double a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_d_sub"));
  return fp(a, b);
}

matlab_sym * matlab_sym_d_div(double a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(double a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_d_div"));
  return fp(a, b);
}

matlab_sym * matlab_sym_d_pow(double a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(double a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_d_pow"));
  return fp(a, b);
}

matlab_sym * matlab_sym_eq(const matlab_sym *a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_eq"));
  return fp(a, b);
}

matlab_sym * matlab_sym_eq_d(const matlab_sym *a, double b) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, double b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_eq_d"));
  return fp(a, b);
}

matlab_sym * matlab_sym_sin(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_sin"));
  return fp(a);
}

matlab_sym * matlab_sym_cos(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_cos"));
  return fp(a);
}

matlab_sym * matlab_sym_tan(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_tan"));
  return fp(a);
}

matlab_sym * matlab_sym_asin(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_asin"));
  return fp(a);
}

matlab_sym * matlab_sym_acos(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_acos"));
  return fp(a);
}

matlab_sym * matlab_sym_atan(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_atan"));
  return fp(a);
}

matlab_sym * matlab_sym_sinh(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_sinh"));
  return fp(a);
}

matlab_sym * matlab_sym_cosh(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_cosh"));
  return fp(a);
}

matlab_sym * matlab_sym_tanh(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_tanh"));
  return fp(a);
}

matlab_sym * matlab_sym_exp(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_exp"));
  return fp(a);
}

matlab_sym * matlab_sym_log(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_log"));
  return fp(a);
}

matlab_sym * matlab_sym_sqrt(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_sqrt"));
  return fp(a);
}

matlab_sym * matlab_sym_abs(const matlab_sym *a) {
  using FnT = matlab_sym * (*)(const matlab_sym *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_abs"));
  return fp(a);
}

matlab_sym * matlab_sym_diff(const matlab_sym *f, const matlab_sym *var) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *var);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_diff"));
  return fp(f, var);
}

matlab_sym * matlab_sym_diff_n(const matlab_sym *f, const matlab_sym *var, int64_t n) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *var, int64_t n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_diff_n"));
  return fp(f, var, n);
}

matlab_sym * matlab_sym_int(const matlab_sym *f, const matlab_sym *var) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *var);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_int"));
  return fp(f, var);
}

matlab_sym * matlab_sym_int_def(const matlab_sym *f, const matlab_sym *var, const matlab_sym *a, const matlab_sym *b) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *var, const matlab_sym *a, const matlab_sym *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_int_def"));
  return fp(f, var, a, b);
}

matlab_sym * matlab_sym_simplify(const matlab_sym *e) {
  using FnT = matlab_sym * (*)(const matlab_sym *e);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_simplify"));
  return fp(e);
}

matlab_sym * matlab_sym_expand(const matlab_sym *e) {
  using FnT = matlab_sym * (*)(const matlab_sym *e);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_expand"));
  return fp(e);
}

matlab_sym * matlab_sym_factor(const matlab_sym *e, const matlab_sym *var) {
  using FnT = matlab_sym * (*)(const matlab_sym *e, const matlab_sym *var);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_factor"));
  return fp(e, var);
}

matlab_sym * matlab_sym_subs(const matlab_sym *e, const matlab_sym *old_e, const matlab_sym *new_e) {
  using FnT = matlab_sym * (*)(const matlab_sym *e, const matlab_sym *old_e, const matlab_sym *new_e);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_subs"));
  return fp(e, old_e, new_e);
}

matlab_sym ** matlab_sym_solve(const matlab_sym *eq, const matlab_sym *var, int64_t *n_out) {
  using FnT = matlab_sym ** (*)(const matlab_sym *eq, const matlab_sym *var, int64_t *n_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_solve"));
  return fp(eq, var, n_out);
}

matlab_sym * matlab_sym_solve_one(const matlab_sym *eq, const matlab_sym *var) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *var);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_solve_one"));
  return fp(eq, var);
}

matlab_symmat * matlab_sym_solve_sys(const matlab_sym *const *eqs, int64_t n_eqs, const matlab_sym *const *vars, int64_t n_vars) {
  using FnT = matlab_symmat * (*)(const matlab_sym *const *eqs, int64_t n_eqs, const matlab_sym *const *vars, int64_t n_vars);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_solve_sys"));
  return fp(eqs, n_eqs, vars, n_vars);
}

matlab_symmat * matlab_sym_solve_2x2(const matlab_sym *eq1, const matlab_sym *eq2, const matlab_sym *var1, const matlab_sym *var2) {
  using FnT = matlab_symmat * (*)(const matlab_sym *eq1, const matlab_sym *eq2, const matlab_sym *var1, const matlab_sym *var2);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_solve_2x2"));
  return fp(eq1, eq2, var1, var2);
}

matlab_symmat * matlab_sym_solve_3x3(const matlab_sym *eq1, const matlab_sym *eq2, const matlab_sym *eq3, const matlab_sym *var1, const matlab_sym *var2, const matlab_sym *var3) {
  using FnT = matlab_symmat * (*)(const matlab_sym *eq1, const matlab_sym *eq2, const matlab_sym *eq3, const matlab_sym *var1, const matlab_sym *var2, const matlab_sym *var3);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_solve_3x3"));
  return fp(eq1, eq2, eq3, var1, var2, var3);
}

double matlab_sym_double(const matlab_sym *e) {
  using FnT = double (*)(const matlab_sym *e);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_double"));
  return fp(e);
}

void matlab_sym_disp(const matlab_sym *e) {
  using FnT = void (*)(const matlab_sym *e);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_disp"));
  fp(e);
}

char * matlab_sym_str(const matlab_sym *e, int64_t *len_out) {
  using FnT = char * (*)(const matlab_sym *e, int64_t *len_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_str"));
  return fp(e, len_out);
}

char * matlab_sym_latex(const matlab_sym *e, int64_t *len_out) {
  using FnT = char * (*)(const matlab_sym *e, int64_t *len_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_latex"));
  return fp(e, len_out);
}

char * matlab_sym_ccode(const matlab_sym *e, int64_t *len_out) {
  using FnT = char * (*)(const matlab_sym *e, int64_t *len_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_ccode"));
  return fp(e, len_out);
}

void matlab_sym_free(matlab_sym *e) {
  using FnT = void (*)(matlab_sym *e);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_free"));
  fp(e);
}

matlab_sym * matlab_sym_assume(const matlab_sym *x, const char *prop, int64_t prop_len) {
  using FnT = matlab_sym * (*)(const matlab_sym *x, const char *prop, int64_t prop_len);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_assume"));
  return fp(x, prop, prop_len);
}

char * matlab_sym_assumptions(const matlab_sym *x, int64_t *len_out) {
  using FnT = char * (*)(const matlab_sym *x, int64_t *len_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_assumptions"));
  return fp(x, len_out);
}

matlab_sym * matlab_sym_vpa(const matlab_sym *e, int64_t dps) {
  using FnT = matlab_sym * (*)(const matlab_sym *e, int64_t dps);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_vpa"));
  return fp(e, dps);
}

matlab_sym * matlab_sym_taylor(const matlab_sym *f, const matlab_sym *var, const matlab_sym *a, int64_t n) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *var, const matlab_sym *a, int64_t n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_taylor"));
  return fp(f, var, a, n);
}

matlab_sym * matlab_sym_limit(const matlab_sym *f, const matlab_sym *var, const matlab_sym *target) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *var, const matlab_sym *target);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_limit"));
  return fp(f, var, target);
}

matlab_sym * matlab_sym_dsolve(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_dsolve"));
  return fp(eq, y, yp, x);
}

matlab_sym * matlab_sym_dsolve_2(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *ypp, const matlab_sym *x) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *ypp, const matlab_sym *x);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_dsolve_2"));
  return fp(eq, y, yp, ypp, x);
}

matlab_sym * matlab_sym_pdsolve(const matlab_sym *a, const matlab_sym *b, const matlab_sym *c, const matlab_sym *x, const matlab_sym *y) {
  using FnT = matlab_sym * (*)(const matlab_sym *a, const matlab_sym *b, const matlab_sym *c, const matlab_sym *x, const matlab_sym *y);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_pdsolve"));
  return fp(a, b, c, x, y);
}

matlab_sym * matlab_sym_pdsolve_heat(const matlab_sym *k, const matlab_sym *lambda, const matlab_sym *x, const matlab_sym *t) {
  using FnT = matlab_sym * (*)(const matlab_sym *k, const matlab_sym *lambda, const matlab_sym *x, const matlab_sym *t);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_pdsolve_heat"));
  return fp(k, lambda, x, t);
}

matlab_sym * matlab_sym_pdsolve_wave(const matlab_sym *c, const matlab_sym *x, const matlab_sym *t) {
  using FnT = matlab_sym * (*)(const matlab_sym *c, const matlab_sym *x, const matlab_sym *t);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_pdsolve_wave"));
  return fp(c, x, t);
}

matlab_symmat * matlab_symmat_new(int64_t rows, int64_t cols, const matlab_sym *const *data) {
  using FnT = matlab_symmat * (*)(int64_t rows, int64_t cols, const matlab_sym *const *data);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_new"));
  return fp(rows, cols, data);
}

matlab_symmat * matlab_symmat_zeros(int64_t rows, int64_t cols) {
  using FnT = matlab_symmat * (*)(int64_t rows, int64_t cols);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_zeros"));
  return fp(rows, cols);
}

matlab_symmat * matlab_symmat_eye(int64_t n) {
  using FnT = matlab_symmat * (*)(int64_t n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_eye"));
  return fp(n);
}

int64_t matlab_symmat_rows(const matlab_symmat *m) {
  using FnT = int64_t (*)(const matlab_symmat *m);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_rows"));
  return fp(m);
}

int64_t matlab_symmat_cols(const matlab_symmat *m) {
  using FnT = int64_t (*)(const matlab_symmat *m);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_cols"));
  return fp(m);
}

matlab_sym * matlab_symmat_get(const matlab_symmat *m, int64_t r, int64_t c) {
  using FnT = matlab_sym * (*)(const matlab_symmat *m, int64_t r, int64_t c);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_get"));
  return fp(m, r, c);
}

void matlab_symmat_set(matlab_symmat *m, int64_t r, int64_t c, const matlab_sym *v) {
  using FnT = void (*)(matlab_symmat *m, int64_t r, int64_t c, const matlab_sym *v);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_set"));
  fp(m, r, c, v);
}

matlab_symmat * matlab_symmat_add(const matlab_symmat *a, const matlab_symmat *b) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *a, const matlab_symmat *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_add"));
  return fp(a, b);
}

matlab_symmat * matlab_symmat_sub(const matlab_symmat *a, const matlab_symmat *b) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *a, const matlab_symmat *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_sub"));
  return fp(a, b);
}

matlab_symmat * matlab_symmat_mul(const matlab_symmat *a, const matlab_symmat *b) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *a, const matlab_symmat *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_mul"));
  return fp(a, b);
}

matlab_symmat * matlab_symmat_scalar_mul(const matlab_symmat *a, const matlab_sym *s) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *a, const matlab_sym *s);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_scalar_mul"));
  return fp(a, s);
}

matlab_symmat * matlab_symmat_transpose(const matlab_symmat *a) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_transpose"));
  return fp(a);
}

matlab_symmat * matlab_symmat_inverse(const matlab_symmat *a) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_inverse"));
  return fp(a);
}

matlab_sym * matlab_symmat_det(const matlab_symmat *a) {
  using FnT = matlab_sym * (*)(const matlab_symmat *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_det"));
  return fp(a);
}

matlab_sym * matlab_symmat_trace(const matlab_symmat *a) {
  using FnT = matlab_sym * (*)(const matlab_symmat *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_trace"));
  return fp(a);
}

int64_t matlab_symmat_rank(const matlab_symmat *a) {
  using FnT = int64_t (*)(const matlab_symmat *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_rank"));
  return fp(a);
}

matlab_sym ** matlab_symmat_eigenvals(const matlab_symmat *a, int64_t *n_out) {
  using FnT = matlab_sym ** (*)(const matlab_symmat *a, int64_t *n_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_eigenvals"));
  return fp(a, n_out);
}

int matlab_symmat_lu(const matlab_symmat *a, matlab_symmat **L_out, matlab_symmat **U_out) {
  using FnT = int (*)(const matlab_symmat *a, matlab_symmat **L_out, matlab_symmat **U_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_lu"));
  return fp(a, L_out, U_out);
}

int matlab_symmat_qr(const matlab_symmat *a, matlab_symmat **Q_out, matlab_symmat **R_out) {
  using FnT = int (*)(const matlab_symmat *a, matlab_symmat **Q_out, matlab_symmat **R_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_qr"));
  return fp(a, Q_out, R_out);
}

matlab_symmat * matlab_symmat_cholesky(const matlab_symmat *a) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *a);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_cholesky"));
  return fp(a);
}

matlab_symmat * matlab_symmat_linsolve(const matlab_symmat *A, const matlab_symmat *b) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *A, const matlab_symmat *b);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_linsolve"));
  return fp(A, b);
}

matlab_symmat * matlab_symmat_dsolve_system(const matlab_symmat *A, const matlab_sym *x) {
  using FnT = matlab_symmat * (*)(const matlab_symmat *A, const matlab_sym *x);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_dsolve_system"));
  return fp(A, x);
}

void matlab_symmat_disp(const matlab_symmat *m) {
  using FnT = void (*)(const matlab_symmat *m);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_disp"));
  fp(m);
}

char * matlab_symmat_str(const matlab_symmat *m, int64_t *len_out) {
  using FnT = char * (*)(const matlab_symmat *m, int64_t *len_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_str"));
  return fp(m, len_out);
}

void matlab_symmat_free(matlab_symmat *m) {
  using FnT = void (*)(matlab_symmat *m);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_symmat_free"));
  fp(m);
}

matlab_sym * matlab_sym_nsolve(const matlab_sym *eq, const matlab_sym *var, const matlab_sym *x0, int64_t dps) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *var, const matlab_sym *x0, int64_t dps);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_nsolve"));
  return fp(eq, var, x0, dps);
}

matlab_sym * matlab_sym_vpasolve(const matlab_sym *eq, const matlab_sym *var, const matlab_sym *x0, int64_t dps) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *var, const matlab_sym *x0, int64_t dps);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_vpasolve"));
  return fp(eq, var, x0, dps);
}

matlab_sym * matlab_sym_rsolve(const matlab_sym *const *coeffs, int64_t n_coef, const matlab_sym *n) {
  using FnT = matlab_sym * (*)(const matlab_sym *const *coeffs, int64_t n_coef, const matlab_sym *n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_rsolve"));
  return fp(coeffs, n_coef, n);
}

matlab_sym * matlab_sym_checkodesol(const matlab_sym *eq, const matlab_sym *sol, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *sol, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_checkodesol"));
  return fp(eq, sol, y, yp, x);
}

matlab_sym * matlab_sym_dsolve_ivp(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x, int64_t n_conds, const matlab_sym *const *x_vals, const matlab_sym *const *y_vals) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x, int64_t n_conds, const matlab_sym *const *x_vals, const matlab_sym *const *y_vals);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_dsolve_ivp"));
  return fp(eq, y, yp, x, n_conds, x_vals, y_vals);
}

matlab_sym * matlab_sym_apply_ivp(const matlab_sym *general_solution, const matlab_sym *x, int64_t n_conds, const matlab_sym *const *x_vals, const matlab_sym *const *y_vals) {
  using FnT = matlab_sym * (*)(const matlab_sym *general_solution, const matlab_sym *x, int64_t n_conds, const matlab_sym *const *x_vals, const matlab_sym *const *y_vals);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_apply_ivp"));
  return fp(general_solution, x, n_conds, x_vals, y_vals);
}

matlab_sym * matlab_sym_dsolve_ivp_1(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x, const matlab_sym *x0, const matlab_sym *y0) {
  using FnT = matlab_sym * (*)(const matlab_sym *eq, const matlab_sym *y, const matlab_sym *yp, const matlab_sym *x, const matlab_sym *x0, const matlab_sym *y0);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_dsolve_ivp_1"));
  return fp(eq, y, yp, x, x0, y0);
}

matlab_sym * matlab_sym_apply_ivp_1(const matlab_sym *general_solution, const matlab_sym *x, const matlab_sym *x0, const matlab_sym *y0) {
  using FnT = matlab_sym * (*)(const matlab_sym *general_solution, const matlab_sym *x, const matlab_sym *x0, const matlab_sym *y0);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_apply_ivp_1"));
  return fp(general_solution, x, x0, y0);
}

matlab_sym * matlab_sym_solve_inequality(const matlab_sym *lhs, const matlab_sym *rhs, const char *op, int64_t op_len, const matlab_sym *var) {
  using FnT = matlab_sym * (*)(const matlab_sym *lhs, const matlab_sym *rhs, const char *op, int64_t op_len, const matlab_sym *var);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_solve_inequality"));
  return fp(lhs, rhs, op, op_len, var);
}

matlab_sym * matlab_sym_reduce_inequalities(const matlab_sym *rel, const matlab_sym *var) {
  using FnT = matlab_sym * (*)(const matlab_sym *rel, const matlab_sym *var);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_reduce_inequalities"));
  return fp(rel, var);
}

matlab_sym ** matlab_sym_groebner(const matlab_sym *const *eqs, int64_t n_eqs, const matlab_sym *const *vars, int64_t n_vars, int64_t *n_basis_out) {
  using FnT = matlab_sym ** (*)(const matlab_sym *const *eqs, int64_t n_eqs, const matlab_sym *const *vars, int64_t n_vars, int64_t *n_basis_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_groebner"));
  return fp(eqs, n_eqs, vars, n_vars, n_basis_out);
}

int matlab_sym_linear_diophantine(const matlab_sym *a, const matlab_sym *b, const matlab_sym *c, matlab_sym **x_out, matlab_sym **y_out) {
  using FnT = int (*)(const matlab_sym *a, const matlab_sym *b, const matlab_sym *c, matlab_sym **x_out, matlab_sym **y_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_linear_diophantine"));
  return fp(a, b, c, x_out, y_out);
}

matlab_sym ** matlab_sym_pythagorean_triples(int64_t max_z, int64_t *n_triples_out) {
  using FnT = matlab_sym ** (*)(int64_t max_z, int64_t *n_triples_out);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_pythagorean_triples"));
  return fp(max_z, n_triples_out);
}

matlab_sym * matlab_sym_laplace(const matlab_sym *f, const matlab_sym *t, const matlab_sym *s) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *t, const matlab_sym *s);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_laplace"));
  return fp(f, t, s);
}

matlab_sym * matlab_sym_ilaplace(const matlab_sym *F, const matlab_sym *s, const matlab_sym *t) {
  using FnT = matlab_sym * (*)(const matlab_sym *F, const matlab_sym *s, const matlab_sym *t);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_ilaplace"));
  return fp(F, s, t);
}

matlab_sym * matlab_sym_fourier(const matlab_sym *f, const matlab_sym *t, const matlab_sym *w) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *t, const matlab_sym *w);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_fourier"));
  return fp(f, t, w);
}

matlab_sym * matlab_sym_ifourier(const matlab_sym *F, const matlab_sym *w, const matlab_sym *t) {
  using FnT = matlab_sym * (*)(const matlab_sym *F, const matlab_sym *w, const matlab_sym *t);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_ifourier"));
  return fp(F, w, t);
}

matlab_sym * matlab_sym_ztrans(const matlab_sym *f, const matlab_sym *n, const matlab_sym *z) {
  using FnT = matlab_sym * (*)(const matlab_sym *f, const matlab_sym *n, const matlab_sym *z);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_ztrans"));
  return fp(f, n, z);
}

matlab_sym * matlab_sym_iztrans(const matlab_sym *F, const matlab_sym *z, const matlab_sym *n) {
  using FnT = matlab_sym * (*)(const matlab_sym *F, const matlab_sym *z, const matlab_sym *n);
  static FnT fp = nullptr;
  if (!fp) fp = reinterpret_cast<FnT>(resolve("matlab_sym_iztrans"));
  return fp(F, z, n);
}

}  // extern "C"
