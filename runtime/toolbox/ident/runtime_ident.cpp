/* runtime_ident.cpp — System Identification Toolbox runtime, Tier-1.
 *
 * See docs/ident_toolbox_roadmap.md for the full surface.  Tier-1 is
 * the smallest end-to-end loop: the `iddata` container, linear ARX /
 * AR estimation, `sim` / `predict` simulation, `compare` validation
 * (NRMSE fit %), the fpe / aic / goodnessOfFit quality metrics, and
 * the idpoly → ss / tf conversion that lets the existing Control
 * System Toolbox plots (bode / step / pzmap) light up.
 *
 * The MATLAB-side `ident_classdefs.m` holds only the `iddata` and
 * `idpoly` property bags; every estimator entry point is dispatched
 * from Lowering.cpp's class-pinned-first-arg path directly to the
 * `matlab_ident_*` C-ABI entries below (same pattern as MPC's mpcmove
 * / sim), so the regressor assembly through the QR least-squares solve
 * runs entirely in this TU.
 *
 * ARX is, per the User's Guide, "least-squares estimation using
 * QR-factorization for overdetermined linear equations" — built here
 * as a regressor matrix Φ and a right-hand side Y handed to the
 * shared `matlab_mldivide_mm` (which performs the LS solve for tall A).
 *
 * Tier-1 carve-downs (deferred to Tier-2 / 3):
 *   - SISO only (single output / single input column).
 *   - ARX / AR only; the iterative PEM estimators (armax / oe / bj)
 *     and idss / idtf / idproc land in Tier-2 / 3.
 *   - compare uses simulation (input present) or 1-step prediction
 *     (time series); the K-step predictor with arbitrary K is Tier-2.
 */

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <utility>
#include <vector>

/* matlab_obj_* helpers — defined in runtime/matlab_runtime.cpp but not
 * part of the public matlab_runtime.h surface.  Same forward-decl
 * pattern as runtime/toolbox/mpc/runtime_mpc.cpp. */
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);

/* Optim's Levenberg-Marquardt nonlinear least-squares — the PEM engine
 * for armax / oe / bj.  `fun_p` is a matlab_optim_vecout_fn
 * (matlab_mat *(*)(matlab_mat *)); we pass our file-static
 * ident_pem_residual.  lb / ub may be null (unconstrained). */
extern "C" matlab_mat *matlab_optim_lsqnonlin(void *fun_p, matlab_mat *x0,
                                              matlab_mat *lb, matlab_mat *ub);

/* Symmetric eigendecomposition — the SVD of the ERA block-Hankel is taken
 * via the Gram matrix HᵀH (symmetric PSD), whose eigenvectors are V and
 * whose eigenvalues are σ².  Declared in matlab_runtime.h. */
extern "C" matlab_mat *matlab_eig_V(matlab_mat *A);
extern "C" matlab_mat *matlab_eig_D(matlab_mat *A);

extern "C" {

/* ---------------------------------------------------------------- */
/* File-local helpers                                               */
/* ---------------------------------------------------------------- */

static int64_t mat_len(const matlab_mat *m) {
    return m ? m->rows * m->cols : 0;
}

/* Build a 1×n row vector from a std::vector<double>. */
static matlab_mat *row_from_vec(const std::vector<double> &v) {
    matlab_mat *M = mat_alloc(1, static_cast<int64_t>(v.size()));
    for (size_t i = 0; i < v.size(); ++i) M->data[i] = v[i];
    return M;
}

/* Variance of a residual vector (1/N · Σ e²). */
static double resid_var(const std::vector<double> &e) {
    if (e.empty()) return 0.0;
    double s = 0.0;
    for (double x : e) s += x * x;
    return s / static_cast<double>(e.size());
}

/* ---------------------------------------------------------------- */
/* arx_ls — shared ARX (regularized) normal-equations solve         */
/*                                                                  */
/*   θ = (ΦᵀΦ + λI)⁻¹ Φᵀy   (λ = 0 → ordinary LS)                  */
/*                                                                  */
/* Fills a_out[0..na-1] (the A-polynomial tail a1..a_na) and        */
/* b_out[0..nb-1] (the B taps b1..b_nb).  When gram_out / rows_out  */
/* are non-null, also returns the regularized Gram matrix and the   */
/* number of regression rows — used by `getcov` for the parameter   */
/* covariance V · (ΦᵀΦ+λI)⁻¹.  Returns false when the data is too   */
/* short.                                                            */
/* ---------------------------------------------------------------- */
static bool arx_ls(const double *y, const double *u, int64_t N,
                   int na, int nb, int nk, double lambda,
                   std::vector<double> &a_out, std::vector<double> &b_out,
                   matlab_mat **gram_out, int64_t *rows_out) {
    if (!u) nb = 0;
    int np = na + nb;
    if (np == 0) return false;
    int t0 = na;
    if (nb > 0 && nk + nb - 1 > t0) t0 = nk + nb - 1;
    int64_t rows = N - t0;
    if (rows < np) return false;
    matlab_mat *Phi = mat_alloc(rows, np);
    matlab_mat *Rhs = mat_alloc(rows, 1);
    for (int64_t i = 0; i < rows; ++i) {
        int64_t t = t0 + i;
        Rhs->data[i] = y[t];
        int c = 0;
        for (int j = 1; j <= na; ++j) Phi->data[i * np + (c++)] = -y[t - j];
        for (int k = 0; k < nb; ++k)  Phi->data[i * np + (c++)] = u[t - nk - k];
    }
    matlab_mat *PhiT  = matlab_transpose(Phi);
    matlab_mat *Gram  = matlab_matmul_mm(PhiT, Phi);
    if (lambda > 0.0)
        for (int i = 0; i < np; ++i) Gram->data[i * np + i] += lambda;
    matlab_mat *PtY   = matlab_matmul_mm(PhiT, Rhs);
    matlab_mat *theta = matlab_mldivide_mm(Gram, PtY);
    if (!theta || mat_len(theta) < np) return false;
    a_out.assign(static_cast<size_t>(na), 0.0);
    for (int j = 0; j < na; ++j) a_out[static_cast<size_t>(j)] = theta->data[j];
    b_out.assign(static_cast<size_t>(nb), 0.0);
    for (int k = 0; k < nb; ++k) b_out[static_cast<size_t>(k)] = theta->data[na + k];
    if (gram_out) *gram_out = Gram;
    if (rows_out) *rows_out = rows;
    return true;
}

/* Thin 5-arg wrapper for existing callers (arx ordinary LS, no Gram). */
static bool arx_ls(const double *y, const double *u, int64_t N,
                   int na, int nb, int nk,
                   std::vector<double> &a_out, std::vector<double> &b_out) {
    return arx_ls(y, u, N, na, nb, nk, 0.0, a_out, b_out, nullptr, nullptr);
}

/* ---------------------------------------------------------------- */
/* compute_pe — the general one-step prediction error               */
/*                                                                  */
/* Every linear polynomial model A·y = (B/F)·u + (C/D)·e shares one */
/* predictor; the prediction error is                               */
/*                                                                  */
/*   uf = (B/F)·u                                                   */
/*   w  = A·y − uf                                                  */
/*   e  = (D/C)·w                                                   */
/*                                                                  */
/* ARX (C=D=F=1), ARMAX (F=D=1), OE (A=C=D=1) and BJ (A=1) are all  */
/* special cases.  Polynomials arrive full: A=[1 a…], B=[0…0(nk) b…],*/
/* C=[1 c…], D=[1 d…], F=[1 f…].  Zero initial conditions.  `u` may  */
/* be empty (time series → uf = 0).                                 */
/* ---------------------------------------------------------------- */
static std::vector<double> compute_pe(const std::vector<double> &A,
                                      const std::vector<double> &B,
                                      const std::vector<double> &C,
                                      const std::vector<double> &D,
                                      const std::vector<double> &F,
                                      const std::vector<double> &y,
                                      const std::vector<double> &u) {
    int64_t N = static_cast<int64_t>(y.size());
    bool hasU = !u.empty() && B.size() > 0;
    std::vector<double> uf(static_cast<size_t>(N), 0.0);   /* (B/F)·u   */
    std::vector<double> w(static_cast<size_t>(N), 0.0);    /* A·y − uf  */
    std::vector<double> e(static_cast<size_t>(N), 0.0);    /* (D/C)·w   */
    for (int64_t t = 0; t < N; ++t) {
        /* uf = (B/F)·u :  uf[t] = Σ B[k]u[t-k] − Σ_{k≥1} F[k]uf[t-k] */
        double uft = 0.0;
        if (hasU) {
            for (size_t k = 0; k < B.size(); ++k)
                if (t - static_cast<int64_t>(k) >= 0)
                    uft += B[k] * u[static_cast<size_t>(t - static_cast<int64_t>(k))];
            for (size_t k = 1; k < F.size(); ++k)
                if (t - static_cast<int64_t>(k) >= 0)
                    uft -= F[k] * uf[static_cast<size_t>(t - static_cast<int64_t>(k))];
        }
        uf[static_cast<size_t>(t)] = uft;
        /* w = A·y − uf :  w[t] = Σ A[k]y[t-k] − uf[t]  (A[0] = 1) */
        double wt = 0.0;
        for (size_t k = 0; k < A.size(); ++k)
            if (t - static_cast<int64_t>(k) >= 0)
                wt += A[k] * y[static_cast<size_t>(t - static_cast<int64_t>(k))];
        wt -= uft;
        w[static_cast<size_t>(t)] = wt;
        /* e = (D/C)·w :  e[t] = Σ D[k]w[t-k] − Σ_{k≥1} C[k]e[t-k]  (D[0]=1) */
        double et = 0.0;
        for (size_t k = 0; k < D.size(); ++k)
            if (t - static_cast<int64_t>(k) >= 0)
                et += D[k] * w[static_cast<size_t>(t - static_cast<int64_t>(k))];
        for (size_t k = 1; k < C.size(); ++k)
            if (t - static_cast<int64_t>(k) >= 0)
                et -= C[k] * e[static_cast<size_t>(t - static_cast<int64_t>(k))];
        e[static_cast<size_t>(t)] = et;
    }
    return e;
}

/* Read a model polynomial property into a std::vector (defaults to the
 * monic [1] when the field is absent/empty). */
static std::vector<double> model_poly(matlab_obj *m, const char *name,
                                      int64_t len, bool monic_default) {
    matlab_mat *p = matlab_obj_get_mat(m, name, len);
    std::vector<double> v;
    if (p && mat_len(p) > 0) {
        v.assign(p->data, p->data + mat_len(p));
    } else {
        v.push_back(monic_default ? 1.0 : 0.0);
    }
    return v;
}

/* Polynomial convolution (product) — c(q) = a(q)·b(q). */
static std::vector<double> poly_conv(const std::vector<double> &a,
                                     const std::vector<double> &b) {
    if (a.empty() || b.empty()) return {1.0};
    std::vector<double> c(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            c[i + j] += a[i] * b[j];
    return c;
}

/* ---------------------------------------------------------------- */
/* matlab_ident_arx — ARX least-squares estimation                  */
/*                                                                  */
/*   A(q) y(t) = B(q) u(t) + e(t)                                   */
/*   orders = [na, nb, nk]  (SISO)                                  */
/*                                                                  */
/* Regression rows:                                                 */
/*   y(t) = -a1 y(t-1) − … − a_na y(t-na)                           */
/*          + b1 u(t-nk) + … + b_nb u(t-nk-nb+1) + e(t)             */
/* θ = [a1 … a_na, b1 … b_nb] solved via QR LS (matlab_mldivide_mm).*/
/* Populates the pre-allocated `idpoly` obj and returns a dummy.    */
/* ---------------------------------------------------------------- */
static matlab_mat *ident_arx_impl(void *model_v, void *data_v,
                                  matlab_mat *orders, double lambda) {
    if (!model_v || !data_v || !orders) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y) return mat_alloc(0, 0);
    int64_t N = mat_len(Y);
    int na = static_cast<int>(orders->data[0]);
    int nb = (mat_len(orders) >= 2) ? static_cast<int>(orders->data[1]) : 0;
    int nk = (mat_len(orders) >= 3) ? static_cast<int>(orders->data[2]) : 0;
    if (na < 0) na = 0; if (nb < 0) nb = 0; if (nk < 0) nk = 0;
    if (!U || mat_len(U) == 0) nb = 0;          /* no input → AR-by-LS */
    const double *u = (U && mat_len(U) > 0) ? U->data : nullptr;
    std::vector<double> a, b;
    matlab_mat *Gram = nullptr;
    int64_t rows = 0;
    bool ok = arx_ls(Y->data, u, N, na, nb, nk, lambda, a, b, &Gram, &rows);
    if (!ok) {
        matlab_mat *A1 = mat_alloc(1, 1); A1->data[0] = 1.0;
        matlab_obj_set_mat(model, "A", 1, A1);
        matlab_obj_set_f64(model, "Ts", 2, Ts);
        return mat_alloc(0, 0);
    }
    std::vector<double> Avec(static_cast<size_t>(na + 1), 0.0);
    Avec[0] = 1.0;
    for (int j = 0; j < na; ++j) Avec[static_cast<size_t>(j + 1)] = a[static_cast<size_t>(j)];
    std::vector<double> Bvec(static_cast<size_t>(nk + nb), 0.0);
    for (int k = 0; k < nb; ++k) Bvec[static_cast<size_t>(nk + k)] = b[static_cast<size_t>(k)];
    if (Bvec.empty()) Bvec.push_back(0.0);
    std::vector<double> y(Y->data, Y->data + N);
    std::vector<double> uv;
    if (u) uv.assign(u, u + N);
    std::vector<double> one(1, 1.0);
    std::vector<double> e = compute_pe(Avec, Bvec, one, one, one, y, uv);
    double V = resid_var(e);
    matlab_obj_set_mat(model, "A", 1, row_from_vec(Avec));
    matlab_obj_set_mat(model, "B", 1, row_from_vec(Bvec));
    matlab_mat *nkm = mat_alloc(1, 1); nkm->data[0] = nk;
    matlab_obj_set_mat(model, "nk", 2, nkm);
    if (Gram) matlab_obj_set_mat(model, "Gram", 4, Gram);    /* T6: getcov */
    matlab_obj_set_f64(model, "Ts", 2, Ts);
    matlab_obj_set_f64(model, "NoiseVariance", 13, V);
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(na + nb));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(rows));
    matlab_obj_set_f64(model, "Lambda", 6, lambda);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_ident_arx(void *model_v, void *data_v, matlab_mat *orders) {
    return ident_arx_impl(model_v, data_v, orders, 0.0);
}

/* Tier-6 — 4-arg regularized arx: arx(data, orders, opt) where opt is an
 * arxOptions whose `Regularization` is read here as scalar lambda. */
matlab_mat *matlab_ident_arx_reg(void *model_v, void *data_v,
                                 matlab_mat *orders, double lambda) {
    if (lambda < 0.0) lambda = 0.0;
    return ident_arx_impl(model_v, data_v, orders, lambda);
}

/* ---------------------------------------------------------------- */
/* Tier-6 — parameter introspection: getcov / getpvec / setpvec     */
/*                                                                  */
/* getcov returns V · (ΦᵀΦ + λI)⁻¹ — the standard asymptotic LS     */
/* parameter covariance — using the Gram matrix arx stashed on the  */
/* model.  Returns 0×0 if no Gram was cached (e.g. for PEM fits,    */
/* whose covariance is a Fisher-information follow-on).             */
/* ---------------------------------------------------------------- */
extern matlab_mat *matlab_inv(matlab_mat *A);

matlab_mat *matlab_ident_getcov(void *model_v) {
    if (!model_v) return mat_alloc(0, 0);
    matlab_obj *m = reinterpret_cast<matlab_obj *>(model_v);
    matlab_mat *G = matlab_obj_get_mat(m, "Gram", 4);
    if (!G || G->rows == 0) return mat_alloc(0, 0);
    double V = matlab_obj_get_f64(m, "NoiseVariance", 13);
    matlab_mat *Inv = matlab_inv(G);
    if (!Inv || Inv->rows == 0) return mat_alloc(0, 0);
    int n = static_cast<int>(Inv->rows);
    matlab_mat *Cov = mat_alloc(n, n);
    for (int i = 0; i < n * n; ++i) Cov->data[i] = V * Inv->data[i];
    return Cov;
}

/* getpvec(idpoly) — reconstruct θ = [a1..a_na, b_{nk+1}..b_{nk+nb}].
 * Reads na from A's length and the nk index from the model. */
matlab_mat *matlab_ident_getpvec(void *model_v) {
    if (!model_v) return mat_alloc(0, 0);
    matlab_obj *m = reinterpret_cast<matlab_obj *>(model_v);
    matlab_mat *A  = matlab_obj_get_mat(m, "A", 1);
    matlab_mat *B  = matlab_obj_get_mat(m, "B", 1);
    matlab_mat *NK = matlab_obj_get_mat(m, "nk", 2);
    int na = A ? static_cast<int>(mat_len(A)) - 1 : 0;
    int nk = (NK && mat_len(NK) > 0) ? static_cast<int>(NK->data[0]) : 0;
    int nbtot = B ? static_cast<int>(mat_len(B)) : 0;
    int nb = nbtot - nk; if (nb < 0) nb = 0;
    int np = na + nb;
    if (np <= 0) return mat_alloc(0, 0);
    matlab_mat *p = mat_alloc(np, 1);
    for (int j = 0; j < na; ++j) p->data[j] = A->data[j + 1];
    for (int k = 0; k < nb; ++k) p->data[na + k] = B->data[nk + k];
    return p;
}

/* setpvec(idpoly, theta) — write θ back into A/B (preserves nk and the
 * monic A(1) = 1).  Length mismatches truncate / zero-pad. */
matlab_mat *matlab_ident_setpvec(void *model_v, matlab_mat *theta) {
    if (!model_v || !theta) return mat_alloc(0, 0);
    matlab_obj *m = reinterpret_cast<matlab_obj *>(model_v);
    matlab_mat *A  = matlab_obj_get_mat(m, "A", 1);
    matlab_mat *B  = matlab_obj_get_mat(m, "B", 1);
    matlab_mat *NK = matlab_obj_get_mat(m, "nk", 2);
    int na = A ? static_cast<int>(mat_len(A)) - 1 : 0;
    int nk = (NK && mat_len(NK) > 0) ? static_cast<int>(NK->data[0]) : 0;
    int nbtot = B ? static_cast<int>(mat_len(B)) : 0;
    int nb = nbtot - nk; if (nb < 0) nb = 0;
    int np = na + nb;
    int avail = static_cast<int>(mat_len(theta));
    std::vector<double> Avec(static_cast<size_t>(na + 1), 0.0);
    Avec[0] = 1.0;
    for (int j = 0; j < na && j < avail; ++j) Avec[static_cast<size_t>(j + 1)] = theta->data[j];
    std::vector<double> Bvec(static_cast<size_t>(nk + nb), 0.0);
    for (int k = 0; k < nb && (na + k) < avail; ++k)
        Bvec[static_cast<size_t>(nk + k)] = theta->data[na + k];
    if (Bvec.empty()) Bvec.push_back(0.0);
    matlab_obj_set_mat(m, "A", 1, row_from_vec(Avec));
    matlab_obj_set_mat(m, "B", 1, row_from_vec(Bvec));
    (void)np;
    return mat_alloc(0, 0);
}

/* ---------------------------------------------------------------- */
/* matlab_ident_ar — AR time-series estimation (Yule-Walker)        */
/*                                                                  */
/* Reuses the shipped matlab_aryule (Levinson-Durbin on the biased  */
/* autocorrelation).  Populates idpoly.A = [1 a1 … a_na], B = 0.    */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_ar(void *model_v, void *data_v, double na_d) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);

    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y) return mat_alloc(0, 0);
    int na = static_cast<int>(na_d);
    if (na < 1) na = 1;

    matlab_mat *A = matlab_aryule(Y, static_cast<double>(na));   /* 1×(na+1) */

    /* Residual variance: e = filter(A, 1, y) (the AR innovations). */
    matlab_mat *one = mat_alloc(1, 1); one->data[0] = 1.0;
    matlab_mat *E = matlab_filter(A, one, Y);
    int64_t N = mat_len(E);
    std::vector<double> e;
    e.reserve(static_cast<size_t>(N > na ? N - na : 0));
    for (int64_t i = na; i < N; ++i) e.push_back(E->data[i]);
    double V = resid_var(e);

    matlab_mat *B0 = mat_alloc(1, 1);        /* AR has no input */
    matlab_obj_set_mat(model, "A", 1, A);
    matlab_obj_set_mat(model, "B", 1, B0);
    matlab_obj_set_f64(model, "Ts", 2, Ts);
    matlab_obj_set_f64(model, "NoiseVariance", 13, V);
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(na));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(N > na ? N - na : 0));
    return mat_alloc(0, 0);
}

/* ================================================================ */
/* Tier-2 — prediction-error estimators (armax / oe / bj)           */
/*                                                                  */
/* Each minimises ‖e(θ)‖² over the polynomial coefficients with the */
/* shared Levenberg-Marquardt `matlab_optim_lsqnonlin`.  The data    */
/* (y, u) and the order spec ride in a thread-local context that the */
/* file-static residual callback reads each evaluation — the same    */
/* pattern nlmpc uses to feed extra parameters through the void* ABI.*/
/* ================================================================ */
struct IdentPemContext {
    std::vector<double> y, u;
    int na, nb, nc, nd, nf, nk;
    int type;            /* 0 = armax, 1 = oe, 2 = bj */
};
static thread_local IdentPemContext g_pem;

/* Assemble the five full polynomials A/B/C/D/F from a packed θ. */
static void pem_unpack(const double *th, const IdentPemContext &c,
                       std::vector<double> &A, std::vector<double> &B,
                       std::vector<double> &C, std::vector<double> &D,
                       std::vector<double> &F) {
    A.assign(1, 1.0);
    B.assign(static_cast<size_t>(c.nk), 0.0);   /* nk leading zeros */
    C.assign(1, 1.0);
    D.assign(1, 1.0);
    F.assign(1, 1.0);
    int idx = 0;
    if (c.type == 0) {                  /* armax: a, b, c */
        for (int i = 0; i < c.na; ++i) A.push_back(th[idx++]);
        for (int i = 0; i < c.nb; ++i) B.push_back(th[idx++]);
        for (int i = 0; i < c.nc; ++i) C.push_back(th[idx++]);
    } else if (c.type == 1) {           /* oe: b, f */
        for (int i = 0; i < c.nb; ++i) B.push_back(th[idx++]);
        for (int i = 0; i < c.nf; ++i) F.push_back(th[idx++]);
    } else {                            /* bj: b, c, d, f */
        for (int i = 0; i < c.nb; ++i) B.push_back(th[idx++]);
        for (int i = 0; i < c.nc; ++i) C.push_back(th[idx++]);
        for (int i = 0; i < c.nd; ++i) D.push_back(th[idx++]);
        for (int i = 0; i < c.nf; ++i) F.push_back(th[idx++]);
    }
    if (B.empty()) B.push_back(0.0);
}

/* matlab_optim_vecout_fn — θ (n×1) → residual vector e. */
static matlab_mat *ident_pem_residual(matlab_mat *theta) {
    if (!theta) return mat_alloc(0, 0);
    std::vector<double> A, B, C, D, F;
    pem_unpack(theta->data, g_pem, A, B, C, D, F);
    std::vector<double> e = compute_pe(A, B, C, D, F, g_pem.y, g_pem.u);
    matlab_mat *out = mat_alloc(static_cast<int64_t>(e.size()), 1);
    for (size_t i = 0; i < e.size(); ++i) out->data[i] = e[i];
    return out;
}

/* Shared driver: parse orders, initialise from an ARX fit, run LM,
 * write the optimised A/B/C/D/F back onto the idpoly. */
static matlab_mat *pem_estimate(void *model_v, void *data_v,
                                matlab_mat *orders, int type) {
    if (!model_v || !data_v || !orders) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y) return mat_alloc(0, 0);
    int64_t N = mat_len(Y);

    const double *od = orders->data;
    int no = static_cast<int>(mat_len(orders));
    int na = 0, nb = 0, nc = 0, nd = 0, nf = 0, nk = 0;
    if (type == 0) {            /* armax [na nb nc nk] */
        na = static_cast<int>(od[0]);
        nb = no > 1 ? static_cast<int>(od[1]) : 0;
        nc = no > 2 ? static_cast<int>(od[2]) : 0;
        nk = no > 3 ? static_cast<int>(od[3]) : 1;
    } else if (type == 1) {     /* oe [nb nf nk] */
        nb = static_cast<int>(od[0]);
        nf = no > 1 ? static_cast<int>(od[1]) : 0;
        nk = no > 2 ? static_cast<int>(od[2]) : 1;
    } else {                    /* bj [nb nc nd nf nk] */
        nb = static_cast<int>(od[0]);
        nc = no > 1 ? static_cast<int>(od[1]) : 0;
        nd = no > 2 ? static_cast<int>(od[2]) : 0;
        nf = no > 3 ? static_cast<int>(od[3]) : 0;
        nk = no > 4 ? static_cast<int>(od[4]) : 1;
    }

    g_pem.y.assign(Y->data, Y->data + N);
    g_pem.u.clear();
    if (U && mat_len(U) > 0) g_pem.u.assign(U->data, U->data + mat_len(U));
    g_pem.na = na; g_pem.nb = nb; g_pem.nc = nc;
    g_pem.nd = nd; g_pem.nf = nf; g_pem.nk = nk; g_pem.type = type;

    /* Initialise from an ARX least-squares fit (na_init = na for armax,
     * nf for oe/bj — that gives a denominator estimate to seed A or F). */
    int na_init = (type == 0) ? na : nf;
    std::vector<double> a0, b0;
    bool ok = arx_ls(g_pem.y.data(),
                     g_pem.u.empty() ? nullptr : g_pem.u.data(),
                     N, na_init, nb, nk, a0, b0);
    if (!ok) { a0.assign(static_cast<size_t>(na_init), 0.0);
               b0.assign(static_cast<size_t>(nb), 0.0); }

    std::vector<double> x0v;
    int na0 = static_cast<int>(a0.size());
    int nb0 = static_cast<int>(b0.size());
    if (type == 0) {                    /* [a, b, c] */
        for (int i = 0; i < na; ++i) x0v.push_back(i < na0 ? a0[i] : 0.0);
        for (int i = 0; i < nb; ++i) x0v.push_back(i < nb0 ? b0[i] : 0.0);
        for (int i = 0; i < nc; ++i) x0v.push_back(0.0);
    } else if (type == 1) {             /* [b, f] */
        for (int i = 0; i < nb; ++i) x0v.push_back(i < nb0 ? b0[i] : 0.0);
        for (int i = 0; i < nf; ++i) x0v.push_back(i < na0 ? a0[i] : 0.0);
    } else {                            /* [b, c, d, f] */
        for (int i = 0; i < nb; ++i) x0v.push_back(i < nb0 ? b0[i] : 0.0);
        for (int i = 0; i < nc; ++i) x0v.push_back(0.0);
        for (int i = 0; i < nd; ++i) x0v.push_back(0.0);
        for (int i = 0; i < nf; ++i) x0v.push_back(i < na0 ? a0[i] : 0.0);
    }
    int np = static_cast<int>(x0v.size());
    if (np == 0) return mat_alloc(0, 0);

    matlab_mat *x0 = mat_alloc(np, 1);
    for (int i = 0; i < np; ++i) x0->data[i] = x0v[i];
    matlab_mat *sol = matlab_optim_lsqnonlin(
        reinterpret_cast<void *>(&ident_pem_residual), x0, nullptr, nullptr);
    const double *th = (sol && mat_len(sol) >= np) ? sol->data : x0v.data();

    std::vector<double> A, B, C, D, F;
    pem_unpack(th, g_pem, A, B, C, D, F);
    std::vector<double> e = compute_pe(A, B, C, D, F, g_pem.y, g_pem.u);
    double V = resid_var(e);

    matlab_obj_set_mat(model, "A", 1, row_from_vec(A));
    matlab_obj_set_mat(model, "B", 1, row_from_vec(B));
    matlab_obj_set_mat(model, "C", 1, row_from_vec(C));
    matlab_obj_set_mat(model, "D", 1, row_from_vec(D));
    matlab_obj_set_mat(model, "F", 1, row_from_vec(F));
    matlab_mat *nkm = mat_alloc(1, 1); nkm->data[0] = nk;
    matlab_obj_set_mat(model, "nk", 2, nkm);
    matlab_obj_set_f64(model, "Ts", 2, Ts);
    matlab_obj_set_f64(model, "NoiseVariance", 13, V);
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(np));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(N));
    return mat_alloc(0, 0);
}

matlab_mat *matlab_ident_armax(void *model_v, void *data_v, matlab_mat *orders) {
    return pem_estimate(model_v, data_v, orders, 0);
}
matlab_mat *matlab_ident_oe(void *model_v, void *data_v, matlab_mat *orders) {
    return pem_estimate(model_v, data_v, orders, 1);
}
matlab_mat *matlab_ident_bj(void *model_v, void *data_v, matlab_mat *orders) {
    return pem_estimate(model_v, data_v, orders, 2);
}

/* ---------------------------------------------------------------- */
/* matlab_ident_tfest — transfer-function estimation                */
/*                                                                  */
/* A discrete TF y = B(q)/F(q)·u is exactly the Output-Error model, */
/* so `tfest(data, np, nz)` maps to `oe(data, [nz+1, np, 0])`.  The */
/* result is returned in idpoly form (B = numerator, F = denominator,*/
/* A = C = D = 1); sim / compare / ss / tf / fpe all work on it      */
/* unchanged.  A dedicated idtf class with Numerator/Denominator      */
/* naming is a fidelity follow-on.                                  */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_tfest(void *model_v, void *data_v,
                               double np_d, double nz_d) {
    int np = static_cast<int>(np_d);
    int nz = static_cast<int>(nz_d);
    if (np < 1) np = 1;
    if (nz < 0) nz = 0;
    matlab_mat *ord = mat_alloc(1, 3);
    ord->data[0] = nz + 1;   /* nb */
    ord->data[1] = np;       /* nf */
    ord->data[2] = 0;        /* nk */
    return pem_estimate(model_v, data_v, ord, 1);   /* OE structure */
}

/* ---------------------------------------------------------------- */
/* matlab_ident_iv4 — instrumental-variable ARX estimation          */
/*                                                                  */
/* Unlike ARX least-squares (which assumes white equation noise),   */
/* the IV method is insensitive to noise colour: it replaces the    */
/* regressor's lagged outputs with *instruments* — the noise-free   */
/* simulated output x̂ from a first-stage ARX model — so the noise   */
/* no longer biases the normal equations.  Solve (ZᵀΦ)·θ = ZᵀY.     */
/* Tier-2 ships the IV core; the full 4-stage prefiltered IV4 (with  */
/* an AR noise prefilter) is a numerical follow-on.                 */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_iv4(void *model_v, void *data_v, matlab_mat *orders) {
    if (!model_v || !data_v || !orders) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y || !U) return mat_alloc(0, 0);
    int64_t N = mat_len(Y);
    const double *y = Y->data;
    const double *u = U->data;
    int na = static_cast<int>(orders->data[0]);
    int nb = (mat_len(orders) >= 2) ? static_cast<int>(orders->data[1]) : 0;
    int nk = (mat_len(orders) >= 3) ? static_cast<int>(orders->data[2]) : 0;

    /* Stage 1 — ordinary ARX to seed the instruments. */
    std::vector<double> a1, b1;
    if (!arx_ls(y, u, N, na, nb, nk, a1, b1)) return mat_alloc(0, 0);
    std::vector<double> A1(1, 1.0);
    for (double v : a1) A1.push_back(v);
    std::vector<double> B1(static_cast<size_t>(nk), 0.0);
    for (double v : b1) B1.push_back(v);

    /* x̂ = B1/A1 · u  (noise-free simulated output → the instruments). */
    std::vector<double> xhat(static_cast<size_t>(N), 0.0);
    for (int64_t t = 0; t < N; ++t) {
        double v = 0.0;
        for (size_t k = 0; k < B1.size(); ++k)
            if (t - static_cast<int64_t>(k) >= 0) v += B1[k] * u[t - static_cast<int64_t>(k)];
        for (size_t k = 1; k < A1.size(); ++k)
            if (t - static_cast<int64_t>(k) >= 0) v -= A1[k] * xhat[static_cast<size_t>(t - static_cast<int64_t>(k))];
        xhat[static_cast<size_t>(t)] = v;
    }

    int np = na + nb;
    int t0 = na;
    if (nb > 0 && nk + nb - 1 > t0) t0 = nk + nb - 1;
    int64_t rows = N - t0;
    if (rows < np || np == 0) return mat_alloc(0, 0);
    matlab_mat *Phi = mat_alloc(rows, np);   /* regressors (use y)       */
    matlab_mat *Z   = mat_alloc(rows, np);   /* instruments (use x̂)      */
    matlab_mat *Rhs = mat_alloc(rows, 1);
    for (int64_t i = 0; i < rows; ++i) {
        int64_t t = t0 + i;
        Rhs->data[i] = y[t];
        int c = 0;
        for (int j = 1; j <= na; ++j) {
            Phi->data[i * np + c] = -y[t - j];
            Z->data[i * np + c]   = -xhat[static_cast<size_t>(t - j)];
            ++c;
        }
        for (int k = 0; k < nb; ++k) {
            double val = u[t - nk - k];
            Phi->data[i * np + c] = val;
            Z->data[i * np + c]   = val;
            ++c;
        }
    }
    matlab_mat *ZT     = matlab_transpose(Z);
    matlab_mat *ZtPhi  = matlab_matmul_mm(ZT, Phi);     /* np × np */
    matlab_mat *ZtY    = matlab_matmul_mm(ZT, Rhs);     /* np × 1  */
    matlab_mat *theta  = matlab_mldivide_mm(ZtPhi, ZtY);
    if (!theta || mat_len(theta) < np) return mat_alloc(0, 0);

    std::vector<double> A(1, 1.0);
    for (int j = 0; j < na; ++j) A.push_back(theta->data[j]);
    std::vector<double> B(static_cast<size_t>(nk), 0.0);
    for (int k = 0; k < nb; ++k) B.push_back(theta->data[na + k]);
    if (B.empty()) B.push_back(0.0);
    std::vector<double> one(1, 1.0);
    std::vector<double> yv(y, y + N), uv(u, u + N);
    std::vector<double> e = compute_pe(A, B, one, one, one, yv, uv);
    double V = resid_var(e);

    matlab_obj_set_mat(model, "A", 1, row_from_vec(A));
    matlab_obj_set_mat(model, "B", 1, row_from_vec(B));
    matlab_mat *nkm = mat_alloc(1, 1); nkm->data[0] = nk;
    matlab_obj_set_mat(model, "nk", 2, nkm);
    matlab_obj_set_f64(model, "Ts", 2, Ts);
    matlab_obj_set_f64(model, "NoiseVariance", 13, V);
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(np));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(rows));
    return mat_alloc(0, 0);
}

/* ---------------------------------------------------------------- */
/* matlab_ident_delayest — estimate the input transport delay nk    */
/*                                                                  */
/* Fits ARX[na nb nk] over a range of nk (na = nb = 2) and returns  */
/* the delay that minimises the residual variance.                  */
/* ---------------------------------------------------------------- */
double matlab_ident_delayest(void *data_v) {
    if (!data_v) return 0.0;
    matlab_obj *data = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    if (!Y || !U) return 0.0;
    int64_t N = mat_len(Y);
    std::vector<double> yv(Y->data, Y->data + N);
    std::vector<double> uv(U->data, U->data + mat_len(U));
    const int na = 2, nb = 2;
    int nkmax = 10;
    if (nkmax > static_cast<int>(N) / 4) nkmax = static_cast<int>(N) / 4;
    int best_nk = 0;
    double bestV = 1e300;
    for (int nk = 0; nk <= nkmax; ++nk) {
        std::vector<double> a, b;
        if (!arx_ls(yv.data(), uv.data(), N, na, nb, nk, a, b)) continue;
        std::vector<double> A(1, 1.0); for (double v : a) A.push_back(v);
        std::vector<double> B(static_cast<size_t>(nk), 0.0); for (double v : b) B.push_back(v);
        std::vector<double> one(1, 1.0);
        std::vector<double> e = compute_pe(A, B, one, one, one, yv, uv);
        double V = resid_var(e);
        if (V < bestV) { bestV = V; best_nk = nk; }
    }
    return static_cast<double>(best_nk);
}

/* ---------------------------------------------------------------- */
/* matlab_ident_pe — prediction-error vector for any idpoly         */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_pe(void *model_v, void *data_v) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    if (!Y) return mat_alloc(0, 0);
    std::vector<double> y(Y->data, Y->data + mat_len(Y));
    std::vector<double> u;
    if (U && mat_len(U) > 0) u.assign(U->data, U->data + mat_len(U));
    std::vector<double> A = model_poly(model, "A", 1, true);
    std::vector<double> B = model_poly(model, "B", 1, false);
    std::vector<double> C = model_poly(model, "C", 1, true);
    std::vector<double> D = model_poly(model, "D", 1, true);
    std::vector<double> F = model_poly(model, "F", 1, true);
    std::vector<double> e = compute_pe(A, B, C, D, F, y, u);
    matlab_mat *out = mat_alloc(static_cast<int64_t>(e.size()), 1);
    for (size_t i = 0; i < e.size(); ++i) out->data[i] = e[i];
    return out;
}

/* ---------------------------------------------------------------- */
/* matlab_ident_resid — residual whiteness diagnostic               */
/*                                                                  */
/* Returns a 2×1 vector [maxAutoCorr; maxCrossCorr]: the largest    */
/* magnitude of the normalised residual autocorrelation over lags   */
/* 1..M and of the residual/input cross-correlation over lags 0..M. */
/* Both small (within the ~1.96/√N band) ⇒ the residuals are white  */
/* and uncorrelated with the input — a validated model.  This is the */
/* scalar content of MATLAB's resid plot (the plot itself is a       */
/* carve-down).                                                      */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_resid(void *model_v, void *data_v) {
    matlab_mat *E = matlab_ident_pe(model_v, data_v);
    int64_t N = mat_len(E);
    matlab_mat *out = mat_alloc(2, 1);
    if (N < 2) { return out; }
    matlab_obj *data = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    const double *e = E->data;
    double mean = 0.0;
    for (int64_t i = 0; i < N; ++i) mean += e[i];
    mean /= static_cast<double>(N);
    double r0 = 0.0;
    for (int64_t i = 0; i < N; ++i) r0 += (e[i] - mean) * (e[i] - mean);
    if (r0 <= 0.0) r0 = 1.0;
    int M = static_cast<int>(N / 4); if (M > 20) M = 20;
    double maxauto = 0.0;
    for (int lag = 1; lag <= M; ++lag) {
        double s = 0.0;
        for (int64_t i = lag; i < N; ++i) s += (e[i] - mean) * (e[i - lag] - mean);
        double r = std::fabs(s / r0);
        if (r > maxauto) maxauto = r;
    }
    double maxcross = 0.0;
    if (U && mat_len(U) >= N) {
        const double *u = U->data;
        double um = 0.0;
        for (int64_t i = 0; i < N; ++i) um += u[i];
        um /= static_cast<double>(N);
        double su = 0.0;
        for (int64_t i = 0; i < N; ++i) su += (u[i] - um) * (u[i] - um);
        double denom = std::sqrt((r0) * (su > 0 ? su : 1.0));
        for (int lag = 0; lag <= M; ++lag) {
            double s = 0.0;
            for (int64_t i = lag; i < N; ++i) s += (e[i] - mean) * (u[i - lag] - um);
            double r = std::fabs(s / denom);
            if (r > maxcross) maxcross = r;
        }
    }
    out->data[0] = maxauto;
    out->data[1] = maxcross;
    return out;
}

/* ---------------------------------------------------------------- */
/* matlab_ident_sim — deterministic simulation  y = B/(A·F) · u     */
/*                                                                  */
/* General across ARX/ARMAX (F=1 → B/A) and OE/BJ (A=1 → B/F).  The */
/* noise polynomials C/D play no part in a deterministic sim.       */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_sim(void *model_v, matlab_mat *u) {
    if (!model_v || !u) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    std::vector<double> A = model_poly(model, "A", 1, true);
    std::vector<double> B = model_poly(model, "B", 1, false);
    std::vector<double> F = model_poly(model, "F", 1, true);
    std::vector<double> den = poly_conv(A, F);    /* A·F, monic */
    int64_t N = mat_len(u);
    const double *uu = u->data;
    matlab_mat *out = mat_alloc(N, 1);
    for (int64_t t = 0; t < N; ++t) {
        double v = 0.0;
        for (size_t k = 0; k < B.size(); ++k)
            if (t - static_cast<int64_t>(k) >= 0)
                v += B[k] * uu[t - static_cast<int64_t>(k)];
        for (size_t k = 1; k < den.size(); ++k)
            if (t - static_cast<int64_t>(k) >= 0)
                v -= den[k] * out->data[t - static_cast<int64_t>(k)];
        out->data[t] = v;
    }
    return out;
}

/* ---------------------------------------------------------------- */
/* matlab_ident_predict — K-step-ahead predictor                    */
/*                                                                  */
/* The 1-step predictor is ŷ = y − e, where e is the general        */
/* prediction error (compute_pe) — correct for every structure.  A  */
/* huge K (≥ 1e6, the `Inf` idiom) falls back to pure simulation.   */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_predict(void *model_v, void *data_v, double K) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    if (!Y) return mat_alloc(0, 0);

    std::vector<double> y(Y->data, Y->data + mat_len(Y));
    std::vector<double> u;
    if (U && mat_len(U) > 0) u.assign(U->data, U->data + mat_len(U));
    if (K >= 1e6 && !u.empty()) return matlab_ident_sim(model_v, U);  /* Inf ≡ sim */

    std::vector<double> A = model_poly(model, "A", 1, true);
    std::vector<double> B = model_poly(model, "B", 1, false);
    std::vector<double> C = model_poly(model, "C", 1, true);
    std::vector<double> D = model_poly(model, "D", 1, true);
    std::vector<double> F = model_poly(model, "F", 1, true);
    std::vector<double> e = compute_pe(A, B, C, D, F, y, u);
    int64_t N = static_cast<int64_t>(y.size());
    matlab_mat *Yhat = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) Yhat->data[i] = y[i] - e[i];
    return Yhat;
}

/* NRMSE fit percentage: 100·(1 − ‖y − ŷ‖ / ‖y − mean(y)‖). */
static double nrmse_fit(const double *y, const double *yh, int64_t N) {
    if (N <= 0) return 0.0;
    double mean = 0.0;
    for (int64_t i = 0; i < N; ++i) mean += y[i];
    mean /= static_cast<double>(N);
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        double d = y[i] - yh[i];
        double m = y[i] - mean;
        num += d * d;
        den += m * m;
    }
    if (den <= 0.0) return 0.0;
    return 100.0 * (1.0 - sqrt(num) / sqrt(den));
}

/* ---------------------------------------------------------------- */
/* matlab_ident_compare — NRMSE fit % of model against data         */
/*                                                                  */
/* Uses simulation when the data carries an input, else the 1-step  */
/* predictor (time series).  Returns the fit percentage as a scalar.*/
/* ---------------------------------------------------------------- */
double matlab_ident_compare(void *data_v, void *model_v) {
    if (!data_v || !model_v) return 0.0;
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    if (!Y) return 0.0;
    int64_t N = mat_len(Y);

    matlab_mat *Yhat;
    if (U && mat_len(U) > 0)
        Yhat = matlab_ident_sim(model_v, U);
    else
        Yhat = matlab_ident_predict(model_v, data_v, 1.0);
    if (!Yhat) return 0.0;
    int64_t M = mat_len(Yhat);
    if (M < N) N = M;
    return nrmse_fit(Y->data, Yhat->data, N);
}

/* ---------------------------------------------------------------- */
/* matlab_ident_goodness — NRMSE cost (MATLAB goodnessOfFit form)   */
/* Returns ‖xref − x‖ / ‖xref − mean(xref)‖  (0 = perfect).         */
/* ---------------------------------------------------------------- */
double matlab_ident_goodness(matlab_mat *x, matlab_mat *xref) {
    if (!x || !xref) return 0.0;
    int64_t N = mat_len(xref);
    int64_t M = mat_len(x);
    if (M < N) N = M;
    double fit = nrmse_fit(xref->data, x->data, N);
    return 1.0 - fit / 100.0;   /* invert the percentage back to NRMSE cost */
}

/* ---------------------------------------------------------------- */
/* Quality metrics: fpe / aic from the cached loss + parameter count */
/* ---------------------------------------------------------------- */
double matlab_ident_fpe(void *model_v) {
    if (!model_v) return 0.0;
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    double V = matlab_obj_get_f64(model, "NoiseVariance", 13);
    double d = matlab_obj_get_f64(model, "Np", 2);
    double N = matlab_obj_get_f64(model, "Ns", 2);
    if (N <= d) return V;
    return V * (N + d) / (N - d);
}

double matlab_ident_aic(void *model_v) {
    if (!model_v) return 0.0;
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    double V = matlab_obj_get_f64(model, "NoiseVariance", 13);
    double d = matlab_obj_get_f64(model, "Np", 2);
    double N = matlab_obj_get_f64(model, "Ns", 2);
    if (V <= 0.0 || N <= 0.0) return 0.0;
    return N * log(V) + 2.0 * d;
}

/* ================================================================ */
/* Tier-3 — subspace state-space identification (n4sid / ssest)     */
/*                                                                  */
/* SISO realization via Ho-Kalman / ERA on Markov parameters:        */
/*   1. estimate the impulse response g(0..2s) by a high-order FIR    */
/*      least-squares fit (g(0)=D, g(k)=C·A^{k-1}·B);                 */
/*   2. block-Hankel H[i][j]=g(1+i+j), shift Hs[i][j]=g(2+i+j);      */
/*   3. SVD of H via the Gram HᵀH (symmetric eig): keep nx modes;     */
/*   4. A = Σ^{-½}Uₙᵀ·Hs·Vₙ·Σ^{-½}, C = (UₙΣ^{½})(row 0),            */
/*      B = (Σ^{½}Vₙᵀ)(col 0), D = g(0).                             */
/*                                                                  */
/* This is the impulse-response realization member of the subspace   */
/* family; n4sid and ssest share it (the projection-based N4SID and  */
/* a PEM refinement of the subspace seed are follow-ons).  Any valid */
/* realization is equivalent up to similarity, which is all the      */
/* downstream ss()/mpc() consumers need.                            */
/* ---------------------------------------------------------------- */
static std::vector<double> fir_markov(const double *y, const double *u,
                                      int64_t N, int L) {
    int np = L + 1;
    int t0 = L;
    int64_t rows = N - t0;
    std::vector<double> g(static_cast<size_t>(np), 0.0);
    if (rows < np) return g;
    matlab_mat *Phi = mat_alloc(rows, np);
    matlab_mat *Rhs = mat_alloc(rows, 1);
    for (int64_t i = 0; i < rows; ++i) {
        int64_t t = t0 + i;
        Rhs->data[i] = y[t];
        for (int k = 0; k <= L; ++k) Phi->data[i * np + k] = u[t - k];
    }
    matlab_mat *PhiT = matlab_transpose(Phi);
    matlab_mat *Gram = matlab_matmul_mm(PhiT, Phi);
    matlab_mat *PtY  = matlab_matmul_mm(PhiT, Rhs);
    matlab_mat *th   = matlab_mldivide_mm(Gram, PtY);
    if (th && mat_len(th) >= np)
        for (int k = 0; k < np; ++k) g[static_cast<size_t>(k)] = th->data[k];
    return g;
}

static bool era_realize(const std::vector<double> &g, int nx,
                        matlab_mat **Aout, matlab_mat **Bout,
                        matlab_mat **Cout, double *Dout) {
    int ng = static_cast<int>(g.size()) - 1;
    int s = ng / 2;                       /* block rows; need g up to 2s */
    if (s < 1) return false;
    if (nx > s) nx = s;
    if (nx < 1) nx = 1;
    matlab_mat *H  = mat_alloc(s, s);
    matlab_mat *Hs = mat_alloc(s, s);
    for (int i = 0; i < s; ++i)
        for (int j = 0; j < s; ++j) {
            H->data[i * s + j]  = g[static_cast<size_t>(1 + i + j)];
            Hs->data[i * s + j] = g[static_cast<size_t>(2 + i + j)];
        }
    /* SVD via Gram: G = HᵀH = V·Σ²·Vᵀ. */
    matlab_mat *HT = matlab_transpose(H);
    matlab_mat *G  = matlab_matmul_mm(HT, H);
    matlab_mat *V  = matlab_eig_V(G);
    matlab_mat *Dg = matlab_eig_D(G);
    if (!V || !Dg || V->rows != s || Dg->rows != s) return false;
    std::vector<std::pair<double, int>> ev(static_cast<size_t>(s));
    for (int i = 0; i < s; ++i) ev[static_cast<size_t>(i)] = {Dg->data[i * s + i], i};
    std::sort(ev.begin(), ev.end(),
              [](const std::pair<double,int> &a, const std::pair<double,int> &b) {
                  return a.first > b.first;
              });
    std::vector<double> sig(static_cast<size_t>(nx));
    matlab_mat *Vn = mat_alloc(s, nx);
    for (int c = 0; c < nx; ++c) {
        double lam = ev[static_cast<size_t>(c)].first;
        if (lam < 0.0) lam = 0.0;
        sig[static_cast<size_t>(c)] = sqrt(lam);
        int col = ev[static_cast<size_t>(c)].second;
        for (int i = 0; i < s; ++i) Vn->data[i * nx + c] = V->data[i * s + col];
    }
    /* Uₙ = H·Vₙ·Σ⁻¹ */
    matlab_mat *HVn = matlab_matmul_mm(H, Vn);
    matlab_mat *Un  = mat_alloc(s, nx);
    for (int i = 0; i < s; ++i)
        for (int c = 0; c < nx; ++c) {
            double sc = (sig[static_cast<size_t>(c)] > 1e-12) ? 1.0 / sig[static_cast<size_t>(c)] : 0.0;
            Un->data[i * nx + c] = HVn->data[i * nx + c] * sc;
        }
    /* A = Σ^{-½}·Uₙᵀ·Hs·Vₙ·Σ^{-½} */
    matlab_mat *UnT   = matlab_transpose(Un);
    matlab_mat *UnTHs = matlab_matmul_mm(UnT, Hs);
    matlab_mat *M     = matlab_matmul_mm(UnTHs, Vn);
    matlab_mat *A = mat_alloc(nx, nx);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j) {
            double si = (sig[static_cast<size_t>(i)] > 1e-12) ? 1.0 / sqrt(sig[static_cast<size_t>(i)]) : 0.0;
            double sj = (sig[static_cast<size_t>(j)] > 1e-12) ? 1.0 / sqrt(sig[static_cast<size_t>(j)]) : 0.0;
            A->data[i * nx + j] = si * M->data[i * nx + j] * sj;
        }
    /* C = (Uₙ·Σ^{½}) row 0 ; B = (Σ^{½}·Vₙᵀ) col 0 */
    matlab_mat *C = mat_alloc(1, nx);
    matlab_mat *B = mat_alloc(nx, 1);
    for (int j = 0; j < nx; ++j) C->data[j] = Un->data[0 * nx + j] * sqrt(sig[static_cast<size_t>(j)]);
    for (int i = 0; i < nx; ++i) B->data[i] = sqrt(sig[static_cast<size_t>(i)]) * Vn->data[0 * nx + i];
    *Aout = A; *Bout = B; *Cout = C; *Dout = g[0];
    return true;
}

/* Shared driver for n4sid / ssest. */
static matlab_mat *subspace_estimate(void *model_v, void *data_v, double nx_d) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y || !U) return mat_alloc(0, 0);
    int64_t N = mat_len(Y);
    int nx = static_cast<int>(nx_d);
    if (nx < 1) nx = 1;
    int s = nx + 10;                    /* block rows (≥ nx, margin)   */
    int L = 2 * s;                      /* Markov params g(0..2s)      */
    if (L > static_cast<int>(N) / 2) { L = static_cast<int>(N) / 2; s = L / 2; }
    if (s <= nx) { nx = s - 1; if (nx < 1) nx = 1; }

    std::vector<double> g = fir_markov(Y->data, U->data, N, L);
    matlab_mat *A = nullptr, *B = nullptr, *C = nullptr;
    double D = 0.0;
    if (!era_realize(g, nx, &A, &B, &C, &D)) return mat_alloc(0, 0);
    int nxr = static_cast<int>(A->rows);

    matlab_mat *Dm = mat_alloc(1, 1); Dm->data[0] = D;
    matlab_mat *K  = mat_alloc(nxr, 1);     /* zeros — innovations gain follow-on */
    matlab_mat *x0 = mat_alloc(nxr, 1);
    matlab_obj_set_mat(model, "A", 1, A);
    matlab_obj_set_mat(model, "B", 1, B);
    matlab_obj_set_mat(model, "C", 1, C);
    matlab_obj_set_mat(model, "D", 1, Dm);
    matlab_obj_set_mat(model, "K", 1, K);
    matlab_obj_set_mat(model, "x0", 2, x0);
    matlab_obj_set_f64(model, "Ts", 2, Ts);
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(nxr * nxr + 2 * nxr + 1));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(N));
    return mat_alloc(0, 0);
}

matlab_mat *matlab_ident_n4sid(void *model_v, void *data_v, double nx) {
    return subspace_estimate(model_v, data_v, nx);
}
matlab_mat *matlab_ident_ssest(void *model_v, void *data_v, double nx) {
    return subspace_estimate(model_v, data_v, nx);
}

/* ---------------------------------------------------------------- */
/* State-space simulation / validation for idss                     */
/*   x(t+1) = A x(t) + B u(t),   y(t) = C x(t) + D u(t)             */
/* ---------------------------------------------------------------- */
static std::vector<double> sim_ss_core(matlab_obj *model,
                                       const std::vector<double> &u, int64_t N) {
    matlab_mat *A = matlab_obj_get_mat(model, "A", 1);
    matlab_mat *B = matlab_obj_get_mat(model, "B", 1);
    matlab_mat *C = matlab_obj_get_mat(model, "C", 1);
    matlab_mat *D = matlab_obj_get_mat(model, "D", 1);
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    if (!A || !B || !C) return y;
    int nx = static_cast<int>(A->rows);
    double d0 = (D && mat_len(D) > 0) ? D->data[0] : 0.0;
    std::vector<double> x(static_cast<size_t>(nx), 0.0);
    for (int64_t t = 0; t < N; ++t) {
        double ut = (t < static_cast<int64_t>(u.size())) ? u[static_cast<size_t>(t)] : 0.0;
        double yt = d0 * ut;
        for (int i = 0; i < nx; ++i) yt += C->data[i] * x[static_cast<size_t>(i)];
        y[static_cast<size_t>(t)] = yt;
        std::vector<double> xn(static_cast<size_t>(nx), 0.0);
        for (int i = 0; i < nx; ++i) {
            double v = B->data[i] * ut;
            for (int j = 0; j < nx; ++j) v += A->data[i * nx + j] * x[static_cast<size_t>(j)];
            xn[static_cast<size_t>(i)] = v;
        }
        x.swap(xn);
    }
    return y;
}

matlab_mat *matlab_ident_sim_ss(void *model_v, matlab_mat *u) {
    if (!model_v || !u) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    int64_t N = mat_len(u);
    std::vector<double> uv(u->data, u->data + N);
    std::vector<double> y = sim_ss_core(model, uv, N);
    matlab_mat *out = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) out->data[i] = y[static_cast<size_t>(i)];
    return out;
}

double matlab_ident_compare_ss(void *data_v, void *model_v) {
    if (!data_v || !model_v) return 0.0;
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    if (!Y || !U) return 0.0;
    int64_t N = mat_len(Y);
    std::vector<double> uv(U->data, U->data + mat_len(U));
    std::vector<double> yh = sim_ss_core(model, uv, N);
    return nrmse_fit(Y->data, yh.data(), N);
}

/* ss(idss) blocks — Lowering assembles ss__ss(A,B,C,D,Ts).  These are
 * trivial pass-throughs of the stored matrices (the model already *is*
 * a realization), used by the constructor-path ss(model) interception. */
matlab_mat *matlab_ident_ss_A(void *model_v) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(model_v), "A", 1);
}
matlab_mat *matlab_ident_ss_B(void *model_v) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(model_v), "B", 1);
}
matlab_mat *matlab_ident_ss_C(void *model_v) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(model_v), "C", 1);
}
matlab_mat *matlab_ident_ss_D(void *model_v) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(model_v), "D", 1);
}

/* ================================================================ */
/* Tier-4 — linear grey-box identification (idgrey / greyest)       */
/*                                                                  */
/* The user supplies a structure function that maps a physical      */
/* parameter vector to the continuous-time realization, packed as a */
/* single matrix M = [A B; C D] (SISO: (nx+1)×(nx+1)) — the same    */
/* matlab_mat*(*)(matlab_mat*) ABI the nlmpc StateFcn uses.  greyest */
/* discretizes (ZOH c2d) at the data's Ts and PEM-fits the          */
/* parameters with the shipped lsqnonlin.                           */
/* ---------------------------------------------------------------- */
extern matlab_mat *matlab_c2d_Ad(matlab_mat *A, matlab_mat *B, double Ts);
extern matlab_mat *matlab_c2d_Bd(matlab_mat *A, matlab_mat *B, double Ts);

typedef matlab_mat *(*grey_struct_fn)(matlab_mat *);

struct GreyContext {
    std::vector<double> y, u;
    double Ts;
    int nx;
    grey_struct_fn struct_fn;
};
static thread_local GreyContext g_grey;

/* Unpack the packed continuous M = [A B; C D] (SISO), discretize, and
 * simulate; returns the prediction error e = y − ŷ.  Shared by the LM
 * residual and the final realization. */
static matlab_mat *grey_eval(matlab_mat *par, matlab_mat **AdOut,
                             matlab_mat **BdOut, std::vector<double> *CcOut,
                             double *DcOut) {
    GreyContext &c = g_grey;
    if (!par || !c.struct_fn) return mat_alloc(0, 0);
    matlab_mat *M = c.struct_fn(par);
    int nx = c.nx, dim = nx + 1;
    if (!M || M->rows < dim || M->cols < dim) return mat_alloc(0, 0);
    matlab_mat *Ac = mat_alloc(nx, nx);
    matlab_mat *Bc = mat_alloc(nx, 1);
    std::vector<double> Cc(static_cast<size_t>(nx));
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nx; ++j) Ac->data[i * nx + j] = M->data[i * dim + j];
        Bc->data[i] = M->data[i * dim + nx];
    }
    for (int j = 0; j < nx; ++j) Cc[static_cast<size_t>(j)] = M->data[nx * dim + j];
    double Dc = M->data[nx * dim + nx];
    matlab_mat *Ad = matlab_c2d_Ad(Ac, Bc, c.Ts);
    matlab_mat *Bd = matlab_c2d_Bd(Ac, Bc, c.Ts);
    if (!Ad || !Bd) return mat_alloc(0, 0);
    int64_t N = static_cast<int64_t>(c.y.size());
    std::vector<double> x(static_cast<size_t>(nx), 0.0);
    matlab_mat *e = mat_alloc(N, 1);
    for (int64_t t = 0; t < N; ++t) {
        double ut = (t < static_cast<int64_t>(c.u.size())) ? c.u[static_cast<size_t>(t)] : 0.0;
        double yh = Dc * ut;
        for (int i = 0; i < nx; ++i) yh += Cc[static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
        e->data[t] = c.y[static_cast<size_t>(t)] - yh;
        std::vector<double> xn(static_cast<size_t>(nx), 0.0);
        for (int i = 0; i < nx; ++i) {
            double v = Bd->data[i] * ut;
            for (int j = 0; j < nx; ++j) v += Ad->data[i * nx + j] * x[static_cast<size_t>(j)];
            xn[static_cast<size_t>(i)] = v;
        }
        x.swap(xn);
    }
    if (AdOut) *AdOut = Ad;
    if (BdOut) *BdOut = Bd;
    if (CcOut) *CcOut = Cc;
    if (DcOut) *DcOut = Dc;
    return e;
}

static matlab_mat *grey_residual(matlab_mat *par) {
    return grey_eval(par, nullptr, nullptr, nullptr, nullptr);
}

matlab_mat *matlab_ident_greyest(void *model_v, void *data_v, matlab_mat *par0,
                                 void *fn_ptr, double nx_d) {
    if (!model_v || !data_v || !par0 || !fn_ptr) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y || !U) return mat_alloc(0, 0);
    int64_t N = mat_len(Y);
    g_grey.y.assign(Y->data, Y->data + N);
    g_grey.u.assign(U->data, U->data + mat_len(U));
    g_grey.Ts = (Ts > 0.0) ? Ts : 1.0;
    g_grey.nx = static_cast<int>(nx_d);
    g_grey.struct_fn = reinterpret_cast<grey_struct_fn>(fn_ptr);

    matlab_mat *par = matlab_optim_lsqnonlin(
        reinterpret_cast<void *>(&grey_residual), par0, nullptr, nullptr);
    if (!par || mat_len(par) < mat_len(par0)) par = par0;

    matlab_mat *Ad = nullptr, *Bd = nullptr;
    std::vector<double> Cc;
    double Dc = 0.0;
    matlab_mat *e = grey_eval(par, &Ad, &Bd, &Cc, &Dc);
    int nx = g_grey.nx;
    std::vector<double> ev(e->data, e->data + mat_len(e));
    double V = resid_var(ev);

    matlab_mat *Cm = mat_alloc(1, nx);
    for (int j = 0; j < nx; ++j) Cm->data[j] = (j < static_cast<int>(Cc.size())) ? Cc[static_cast<size_t>(j)] : 0.0;
    matlab_mat *Dm = mat_alloc(1, 1); Dm->data[0] = Dc;
    matlab_obj_set_mat(model, "Parameters", 10, par);
    if (Ad) matlab_obj_set_mat(model, "A", 1, Ad);
    if (Bd) matlab_obj_set_mat(model, "B", 1, Bd);
    matlab_obj_set_mat(model, "C", 1, Cm);
    matlab_obj_set_mat(model, "D", 1, Dm);
    matlab_obj_set_f64(model, "Ts", 2, g_grey.Ts);
    matlab_obj_set_f64(model, "NoiseVariance", 13, V);
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(mat_len(par)));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(N));
    return mat_alloc(0, 0);
}

/* ---------------------------------------------------------------- */
/* matlab_ident_impulseest — non-parametric impulse-response model   */
/*                                                                  */
/* The first L+1 Markov parameters g(0..L) via the same FIR least-   */
/* squares used by the subspace core.  Returned as an idpoly FIR     */
/* model (A = 1, B = [g0 … gL]); sim / step then reproduce the       */
/* impulse / step response.                                          */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_impulseest(void *model_v, void *data_v, double L_d) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y || !U) return mat_alloc(0, 0);
    int64_t N = mat_len(Y);
    int L = static_cast<int>(L_d);
    if (L < 1) L = 1;
    std::vector<double> g = fir_markov(Y->data, U->data, N, L);
    std::vector<double> A(1, 1.0);
    std::vector<double> yv(Y->data, Y->data + N);
    std::vector<double> uv(U->data, U->data + mat_len(U));
    std::vector<double> one(1, 1.0);
    std::vector<double> e = compute_pe(A, g, one, one, one, yv, uv);
    matlab_obj_set_mat(model, "A", 1, row_from_vec(A));
    matlab_obj_set_mat(model, "B", 1, row_from_vec(g));
    matlab_mat *nkm = mat_alloc(1, 1); nkm->data[0] = 0;
    matlab_obj_set_mat(model, "nk", 2, nkm);
    matlab_obj_set_f64(model, "Ts", 2, Ts);
    matlab_obj_set_f64(model, "NoiseVariance", 13, resid_var(e));
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(L + 1));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(N));
    return mat_alloc(0, 0);
}

/* ---------------------------------------------------------------- */
/* matlab_ident_forecast — K-step-ahead forecast past the data       */
/*                                                                  */
/* Rolls the deterministic A/B recursion forward with future noise   */
/* e = 0 (minimum-variance) and future input u = 0, using actual     */
/* past outputs in-sample and forecasted values out-of-sample.       */
/* The natural AR/ARX time-series forecast; returns the K future     */
/* outputs.  (ARMAX C/D filtering of future e is a follow-on.)       */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_ident_forecast(void *model_v, void *data_v, double K_d) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    if (!Y) return mat_alloc(0, 0);
    int K = static_cast<int>(K_d);
    if (K < 1) K = 1;
    std::vector<double> A = model_poly(model, "A", 1, true);
    std::vector<double> B = model_poly(model, "B", 1, false);
    int64_t N = mat_len(Y);
    int na = static_cast<int>(A.size()) - 1;
    int nb = static_cast<int>(B.size());
    const double *u = (U && mat_len(U) > 0) ? U->data : nullptr;
    int64_t Nu = mat_len(U);
    std::vector<double> yext(static_cast<size_t>(N + K), 0.0);
    for (int64_t i = 0; i < N; ++i) yext[static_cast<size_t>(i)] = Y->data[i];
    for (int64_t t = N; t < N + K; ++t) {
        double yh = 0.0;
        for (int k = 1; k <= na; ++k)
            if (t - k >= 0) yh += -A[static_cast<size_t>(k)] * yext[static_cast<size_t>(t - k)];
        if (u)
            for (int k = 0; k < nb; ++k)
                if (t - k >= 0 && t - k < Nu)
                    yh += B[static_cast<size_t>(k)] * u[t - k];
        yext[static_cast<size_t>(t)] = yh;
    }
    matlab_mat *out = mat_alloc(K, 1);
    for (int i = 0; i < K; ++i) out->data[i] = yext[static_cast<size_t>(N + i)];
    return out;
}

/* ================================================================ */
/* Tier-4 — non-parametric frequency response (etfe / spa)          */
/*                                                                  */
/* Empirical transfer-function estimate G(e^{jω}) = Y(ω)/U(ω) via   */
/* the shipped complex FFT.  Stored on an idfrd as Frequency (rad/s)*/
/* + magnitude + phase.  `spa` frequency-smooths the cross/auto      */
/* spectra (a Blackman-Tukey-style estimate) before the ratio.      */
/* ---------------------------------------------------------------- */
extern matlab_mat_c *matlab_fft_c(void *A);

/* Column vector (n×1) — idfrd Frequency / response arrays are columns. */
static matlab_mat *col_from_vec(const std::vector<double> &v) {
    matlab_mat *M = mat_alloc(static_cast<int64_t>(v.size()), 1);
    for (size_t i = 0; i < v.size(); ++i) M->data[i] = v[i];
    return M;
}

static void freqresp_store(matlab_obj *model, const std::vector<double> &freq,
                           const std::vector<double> &mag,
                           const std::vector<double> &phase, double Ts) {
    matlab_obj_set_mat(model, "Frequency", 9, col_from_vec(freq));
    matlab_obj_set_mat(model, "ResponseMag", 11, col_from_vec(mag));
    matlab_obj_set_mat(model, "ResponsePhase", 13, col_from_vec(phase));
    matlab_obj_set_f64(model, "Ts", 2, Ts);
}

/* Shared FFT of the I/O records → one-sided complex spectra Yf, Uf. */
static int freqresp_fft(matlab_obj *data, matlab_mat_c **Yf, matlab_mat_c **Uf,
                        int64_t *N_out, double *Ts_out) {
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    if (!Y || !U || mat_len(U) == 0) return 0;
    *N_out  = mat_len(Y);
    *Ts_out = matlab_obj_get_f64(data, "Ts", 2);
    *Yf = matlab_fft_c(reinterpret_cast<void *>(Y));
    *Uf = matlab_fft_c(reinterpret_cast<void *>(U));
    return (*Yf && *Uf) ? 1 : 0;
}

matlab_mat *matlab_ident_etfe(void *model_v, void *data_v) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat_c *Yf = nullptr, *Uf = nullptr;
    int64_t N = 0; double Ts = 1.0;
    if (!freqresp_fft(data, &Yf, &Uf, &N, &Ts)) return mat_alloc(0, 0);
    if (Ts <= 0.0) Ts = 1.0;
    int64_t Nf = N / 2 + 1;
    std::vector<double> freq(static_cast<size_t>(Nf)), mag(static_cast<size_t>(Nf)),
                        phase(static_cast<size_t>(Nf));
    for (int64_t k = 0; k < Nf; ++k) {
        double ur = Uf->re[k], ui = Uf->im[k];
        double yr = Yf->re[k], yi = Yf->im[k];
        double den = ur * ur + ui * ui;
        double gr = 0.0, gi = 0.0;
        if (den > 1e-300) { gr = (yr * ur + yi * ui) / den; gi = (yi * ur - yr * ui) / den; }
        mag[static_cast<size_t>(k)]   = sqrt(gr * gr + gi * gi);
        phase[static_cast<size_t>(k)] = atan2(gi, gr);
        freq[static_cast<size_t>(k)]  = 2.0 * M_PI * static_cast<double>(k) /
                                        (static_cast<double>(N) * Ts);
    }
    freqresp_store(model, freq, mag, phase, Ts);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_ident_spa(void *model_v, void *data_v) {
    if (!model_v || !data_v) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat_c *Yf = nullptr, *Uf = nullptr;
    int64_t N = 0; double Ts = 1.0;
    if (!freqresp_fft(data, &Yf, &Uf, &N, &Ts)) return mat_alloc(0, 0);
    if (Ts <= 0.0) Ts = 1.0;
    int64_t Nf = N / 2 + 1;
    /* Cross-spectrum Φyu = conj(U)·Y and auto-spectrum Φuu = |U|², then
     * frequency-smooth both with a moving average before the ratio. */
    std::vector<double> pyr(static_cast<size_t>(Nf)), pyi(static_cast<size_t>(Nf)),
                        puu(static_cast<size_t>(Nf));
    for (int64_t k = 0; k < Nf; ++k) {
        double ur = Uf->re[k], ui = Uf->im[k], yr = Yf->re[k], yi = Yf->im[k];
        pyr[static_cast<size_t>(k)] = ur * yr + ui * yi;   /* Re(conj(U)·Y) */
        pyi[static_cast<size_t>(k)] = ur * yi - ui * yr;   /* Im(conj(U)·Y) */
        puu[static_cast<size_t>(k)] = ur * ur + ui * ui;
    }
    int win = 4;                                   /* ± half-width */
    std::vector<double> freq(static_cast<size_t>(Nf)), mag(static_cast<size_t>(Nf)),
                        phase(static_cast<size_t>(Nf));
    for (int64_t k = 0; k < Nf; ++k) {
        double sr = 0.0, si = 0.0, su = 0.0;
        int64_t lo = k - win < 0 ? 0 : k - win;
        int64_t hi = k + win >= Nf ? Nf - 1 : k + win;
        for (int64_t j = lo; j <= hi; ++j) {
            sr += pyr[static_cast<size_t>(j)];
            si += pyi[static_cast<size_t>(j)];
            su += puu[static_cast<size_t>(j)];
        }
        double gr = 0.0, gi = 0.0;
        if (su > 1e-300) { gr = sr / su; gi = si / su; }
        mag[static_cast<size_t>(k)]   = sqrt(gr * gr + gi * gi);
        phase[static_cast<size_t>(k)] = atan2(gi, gr);
        freq[static_cast<size_t>(k)]  = 2.0 * M_PI * static_cast<double>(k) /
                                        (static_cast<double>(N) * Ts);
    }
    freqresp_store(model, freq, mag, phase, Ts);
    return mat_alloc(0, 0);
}

/* ================================================================ */
/* Tier-5 — nonlinear state estimation (EKF / UKF)                  */
/*                                                                  */
/* The dynamic Kalman filtering loop — the only fully-new numeric    */
/* subsystem (the shipped CST kalman is steady-state-gain only).     */
/* The user supplies a discrete state-transition StateFcn f(x) and a */
/* measurement MeasFcn h(x) (single-arg matlab_mat handles, the      */
/* nlmpc/greyest ABI; scalar measurement).  Handles are passed at    */
/* predict/correct time (they don't round-trip through obj storage). */
/* The filter object carries mutable State + StateCovariance.        */
/* ---------------------------------------------------------------- */
extern matlab_mat *matlab_chol(matlab_mat *A);

typedef matlab_mat *(*ident_vec_fn)(matlab_mat *);

/* Populate a (zero-arg-allocated) EKF/UKF shell from x0/P0/Q/R. */
matlab_mat *matlab_ident_ekf_init(void *obj_v, matlab_mat *x0, matlab_mat *P0,
                                  matlab_mat *Q, matlab_mat *R) {
    if (!obj_v) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    if (x0) matlab_obj_set_mat(o, "State", 5, x0);
    if (P0) matlab_obj_set_mat(o, "StateCovariance", 15, P0);
    if (Q)  matlab_obj_set_mat(o, "ProcessNoise", 12, Q);
    if (R)  matlab_obj_set_mat(o, "MeasurementNoise", 16, R);
    return mat_alloc(0, 0);
}

/* Finite-difference Jacobian of f (nx-in, m-out) at x; also returns fx. */
static matlab_mat *fd_jac(ident_vec_fn f, matlab_mat *x, matlab_mat **fx_out) {
    int nx = static_cast<int>(x->rows);
    matlab_mat *fx = f(x);
    int m = static_cast<int>(mat_len(fx));
    matlab_mat *J = mat_alloc(m, nx);
    for (int j = 0; j < nx; ++j) {
        matlab_mat *xp = mat_alloc(nx, 1);
        for (int i = 0; i < nx; ++i) xp->data[i] = x->data[i];
        double dx = 1e-6 * (fabs(x->data[j]) + 1e-6);
        xp->data[j] += dx;
        matlab_mat *fp = f(xp);
        for (int i = 0; i < m; ++i) J->data[i * nx + j] = (fp->data[i] - fx->data[i]) / dx;
    }
    *fx_out = fx;
    return J;
}

matlab_mat *matlab_ident_ekf_predict(void *obj_v, void *f_ptr) {
    if (!obj_v || !f_ptr) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    ident_vec_fn f = reinterpret_cast<ident_vec_fn>(f_ptr);
    matlab_mat *x = matlab_obj_get_mat(o, "State", 5);
    matlab_mat *P = matlab_obj_get_mat(o, "StateCovariance", 15);
    matlab_mat *Q = matlab_obj_get_mat(o, "ProcessNoise", 12);
    if (!x || !P) return mat_alloc(0, 0);
    int nx = static_cast<int>(x->rows);
    matlab_mat *fx = nullptr;
    matlab_mat *F = fd_jac(f, x, &fx);              /* nx × nx */
    matlab_mat *Ft = matlab_transpose(F);
    matlab_mat *FP = matlab_matmul_mm(F, P);
    matlab_mat *FPFt = matlab_matmul_mm(FP, Ft);    /* nx × nx */
    matlab_mat *Pn = mat_alloc(nx, nx);
    for (int i = 0; i < nx * nx; ++i)
        Pn->data[i] = FPFt->data[i] + (Q ? Q->data[i] : 0.0);
    matlab_obj_set_mat(o, "State", 5, fx);
    matlab_obj_set_mat(o, "StateCovariance", 15, Pn);
    return fx;
}

matlab_mat *matlab_ident_ekf_correct(void *obj_v, void *h_ptr, double y) {
    if (!obj_v || !h_ptr) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    ident_vec_fn h = reinterpret_cast<ident_vec_fn>(h_ptr);
    matlab_mat *x = matlab_obj_get_mat(o, "State", 5);
    matlab_mat *P = matlab_obj_get_mat(o, "StateCovariance", 15);
    matlab_mat *R = matlab_obj_get_mat(o, "MeasurementNoise", 16);
    if (!x || !P) return mat_alloc(0, 0);
    int nx = static_cast<int>(x->rows);
    matlab_mat *hx = nullptr;
    matlab_mat *H = fd_jac(h, x, &hx);              /* 1 × nx (scalar meas) */
    matlab_mat *Ht = matlab_transpose(H);          /* nx × 1 */
    matlab_mat *HP = matlab_matmul_mm(H, P);       /* 1 × nx */
    matlab_mat *HPHt = matlab_matmul_mm(HP, Ht);   /* 1 × 1 */
    double S = HPHt->data[0] + (R ? R->data[0] : 0.0);
    if (S == 0.0) S = 1e-12;
    matlab_mat *PHt = matlab_matmul_mm(P, Ht);     /* nx × 1 = K·S */
    double innov = y - hx->data[0];
    matlab_mat *xn = mat_alloc(nx, 1);
    for (int i = 0; i < nx; ++i) xn->data[i] = x->data[i] + (PHt->data[i] / S) * innov;
    /* P_new = P − K·(H·P) = P − (PHt/S)·HP */
    matlab_mat *Pn = mat_alloc(nx, nx);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j)
            Pn->data[i * nx + j] = P->data[i * nx + j] - (PHt->data[i] / S) * HP->data[j];
    matlab_obj_set_mat(o, "State", 5, xn);
    matlab_obj_set_mat(o, "StateCovariance", 15, Pn);
    return xn;
}

/* UKF — sigma-point (Julier scaled unscented transform, α=1).  Generate
 * 2·nx+1 points (point-major: pt p, comp k at Xs[p*nx+k]) + weights. */
static bool ukf_sigma(matlab_mat *x, matlab_mat *P, int nx,
                      std::vector<double> &Xs, std::vector<double> &Wm,
                      std::vector<double> &Wc) {
    double kappa = (nx < 3) ? (3.0 - nx) : 0.0;
    double lambda = kappa;                 /* α = 1 → λ = κ */
    double c = nx + lambda;
    matlab_mat *cP = mat_alloc(nx, nx);
    for (int i = 0; i < nx * nx; ++i) cP->data[i] = c * P->data[i];
    matlab_mat *R = matlab_chol(cP);       /* upper, RᵀR = c·P */
    if (!R || R->rows != nx) return false;
    int npts = 2 * nx + 1;
    Xs.assign(static_cast<size_t>(nx * npts), 0.0);
    for (int k = 0; k < nx; ++k) Xs[static_cast<size_t>(k)] = x->data[k];
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nx; ++k) {
            double v = R->data[i * nx + k];          /* row i of R = √-col */
            Xs[static_cast<size_t>((i + 1) * nx + k)]        = x->data[k] + v;
            Xs[static_cast<size_t>((i + 1 + nx) * nx + k)]   = x->data[k] - v;
        }
    Wm.assign(static_cast<size_t>(npts), 0.0);
    Wc.assign(static_cast<size_t>(npts), 0.0);
    Wm[0] = lambda / c;
    Wc[0] = lambda / c + 2.0;              /* (1 − α² + β), α=1, β=2 → β */
    for (int p = 1; p < npts; ++p) { Wm[static_cast<size_t>(p)] = 1.0 / (2.0 * c);
                                     Wc[static_cast<size_t>(p)] = 1.0 / (2.0 * c); }
    return true;
}

matlab_mat *matlab_ident_ukf_predict(void *obj_v, void *f_ptr) {
    if (!obj_v || !f_ptr) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    ident_vec_fn f = reinterpret_cast<ident_vec_fn>(f_ptr);
    matlab_mat *x = matlab_obj_get_mat(o, "State", 5);
    matlab_mat *P = matlab_obj_get_mat(o, "StateCovariance", 15);
    matlab_mat *Q = matlab_obj_get_mat(o, "ProcessNoise", 12);
    if (!x || !P) return mat_alloc(0, 0);
    int nx = static_cast<int>(x->rows);
    std::vector<double> Xs, Wm, Wc;
    if (!ukf_sigma(x, P, nx, Xs, Wm, Wc)) return mat_alloc(0, 0);
    int npts = 2 * nx + 1;
    std::vector<double> Ys(static_cast<size_t>(nx * npts), 0.0);
    for (int p = 0; p < npts; ++p) {
        matlab_mat *xp = mat_alloc(nx, 1);
        for (int k = 0; k < nx; ++k) xp->data[k] = Xs[static_cast<size_t>(p * nx + k)];
        matlab_mat *yp = f(xp);
        for (int k = 0; k < nx; ++k) Ys[static_cast<size_t>(p * nx + k)] = yp->data[k];
    }
    matlab_mat *xpred = mat_alloc(nx, 1);
    for (int p = 0; p < npts; ++p)
        for (int k = 0; k < nx; ++k)
            xpred->data[k] += Wm[static_cast<size_t>(p)] * Ys[static_cast<size_t>(p * nx + k)];
    matlab_mat *Pp = mat_alloc(nx, nx);
    for (int p = 0; p < npts; ++p)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < nx; ++j)
                Pp->data[i * nx + j] += Wc[static_cast<size_t>(p)] *
                    (Ys[static_cast<size_t>(p * nx + i)] - xpred->data[i]) *
                    (Ys[static_cast<size_t>(p * nx + j)] - xpred->data[j]);
    for (int idx = 0; idx < nx * nx; ++idx) Pp->data[idx] += Q ? Q->data[idx] : 0.0;
    matlab_obj_set_mat(o, "State", 5, xpred);
    matlab_obj_set_mat(o, "StateCovariance", 15, Pp);
    return xpred;
}

matlab_mat *matlab_ident_ukf_correct(void *obj_v, void *h_ptr, double y) {
    if (!obj_v || !h_ptr) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    ident_vec_fn h = reinterpret_cast<ident_vec_fn>(h_ptr);
    matlab_mat *x = matlab_obj_get_mat(o, "State", 5);
    matlab_mat *P = matlab_obj_get_mat(o, "StateCovariance", 15);
    matlab_mat *R = matlab_obj_get_mat(o, "MeasurementNoise", 16);
    if (!x || !P) return mat_alloc(0, 0);
    int nx = static_cast<int>(x->rows);
    std::vector<double> Xs, Wm, Wc;
    if (!ukf_sigma(x, P, nx, Xs, Wm, Wc)) return mat_alloc(0, 0);
    int npts = 2 * nx + 1;
    std::vector<double> Z(static_cast<size_t>(npts), 0.0);
    for (int p = 0; p < npts; ++p) {
        matlab_mat *xp = mat_alloc(nx, 1);
        for (int k = 0; k < nx; ++k) xp->data[k] = Xs[static_cast<size_t>(p * nx + k)];
        matlab_mat *zp = h(xp);
        Z[static_cast<size_t>(p)] = zp->data[0];
    }
    double yhat = 0.0;
    for (int p = 0; p < npts; ++p) yhat += Wm[static_cast<size_t>(p)] * Z[static_cast<size_t>(p)];
    double S = R ? R->data[0] : 0.0;
    for (int p = 0; p < npts; ++p)
        S += Wc[static_cast<size_t>(p)] * (Z[static_cast<size_t>(p)] - yhat) *
             (Z[static_cast<size_t>(p)] - yhat);
    if (S == 0.0) S = 1e-12;
    std::vector<double> Pxy(static_cast<size_t>(nx), 0.0);
    for (int p = 0; p < npts; ++p)
        for (int k = 0; k < nx; ++k)
            Pxy[static_cast<size_t>(k)] += Wc[static_cast<size_t>(p)] *
                (Xs[static_cast<size_t>(p * nx + k)] - x->data[k]) *
                (Z[static_cast<size_t>(p)] - yhat);
    matlab_mat *xn = mat_alloc(nx, 1);
    for (int k = 0; k < nx; ++k) xn->data[k] = x->data[k] + (Pxy[static_cast<size_t>(k)] / S) * (y - yhat);
    matlab_mat *Pn = mat_alloc(nx, nx);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j)
            Pn->data[i * nx + j] = P->data[i * nx + j] -
                Pxy[static_cast<size_t>(i)] * Pxy[static_cast<size_t>(j)] / S;
    matlab_obj_set_mat(o, "State", 5, xn);
    matlab_obj_set_mat(o, "StateCovariance", 15, Pn);
    return xn;
}

/* ================================================================ */
/* Tier-5 — nonlinear grey-box (nlgreyest)                          */
/*                                                                  */
/* The user supplies a continuous ODE right-hand side StateFcn whose */
/* single packed argument is z = [x; u; par] (nx + nu + npar); it    */
/* returns dx/dt (nx×1).  nlgreyest rolls the ODE out (Euler) and    */
/* PEM-fits the physical parameters with lsqnonlin.  Output y = x₁.  */
/* ---------------------------------------------------------------- */
struct NlGreyContext {
    std::vector<double> y, u;
    double Ts;
    int nx;
    grey_struct_fn state_fn;     /* z=[x;u;par] -> dxdt */
};
static thread_local NlGreyContext g_nlgrey;

static matlab_mat *nlgrey_residual(matlab_mat *par) {
    NlGreyContext &c = g_nlgrey;
    if (!par || !c.state_fn) return mat_alloc(0, 0);
    int nx = c.nx;
    int npar = static_cast<int>(mat_len(par));
    int64_t N = static_cast<int64_t>(c.y.size());
    int zdim = nx + 1 + npar;          /* [x; u; par] (SISO: nu=1) */
    std::vector<double> x(static_cast<size_t>(nx), 0.0);
    matlab_mat *e = mat_alloc(N, 1);
    for (int64_t t = 0; t < N; ++t) {
        double ut = (t < static_cast<int64_t>(c.u.size())) ? c.u[static_cast<size_t>(t)] : 0.0;
        e->data[t] = c.y[static_cast<size_t>(t)] - x[0];   /* y = x₁ */
        matlab_mat *z = mat_alloc(zdim, 1);
        for (int i = 0; i < nx; ++i) z->data[i] = x[static_cast<size_t>(i)];
        z->data[nx] = ut;
        for (int i = 0; i < npar; ++i) z->data[nx + 1 + i] = par->data[i];
        matlab_mat *dx = c.state_fn(z);
        if (dx && mat_len(dx) >= nx)
            for (int i = 0; i < nx; ++i) x[static_cast<size_t>(i)] += c.Ts * dx->data[i];
    }
    return e;
}

matlab_mat *matlab_ident_nlgreyest(void *model_v, void *data_v, matlab_mat *par0,
                                   void *fn_ptr, double nx_d) {
    if (!model_v || !data_v || !par0 || !fn_ptr) return mat_alloc(0, 0);
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_obj *data  = reinterpret_cast<matlab_obj *>(data_v);
    matlab_mat *Y = matlab_obj_get_mat(data, "OutputData", 10);
    matlab_mat *U = matlab_obj_get_mat(data, "InputData", 9);
    double Ts = matlab_obj_get_f64(data, "Ts", 2);
    if (!Y) return mat_alloc(0, 0);
    int64_t N = mat_len(Y);
    g_nlgrey.y.assign(Y->data, Y->data + N);
    g_nlgrey.u.clear();
    if (U && mat_len(U) > 0) g_nlgrey.u.assign(U->data, U->data + mat_len(U));
    g_nlgrey.Ts = (Ts > 0.0) ? Ts : 1.0;
    g_nlgrey.nx = static_cast<int>(nx_d);
    g_nlgrey.state_fn = reinterpret_cast<grey_struct_fn>(fn_ptr);

    matlab_mat *par = matlab_optim_lsqnonlin(
        reinterpret_cast<void *>(&nlgrey_residual), par0, nullptr, nullptr);
    if (!par || mat_len(par) < mat_len(par0)) par = par0;
    matlab_mat *e = nlgrey_residual(par);
    std::vector<double> ev(e->data, e->data + mat_len(e));
    matlab_obj_set_mat(model, "Parameters", 10, par);
    matlab_obj_set_f64(model, "Ts", 2, g_nlgrey.Ts);
    matlab_obj_set_f64(model, "NoiseVariance", 13, resid_var(ev));
    matlab_obj_set_f64(model, "Np", 2, static_cast<double>(mat_len(par)));
    matlab_obj_set_f64(model, "Ns", 2, static_cast<double>(N));
    return mat_alloc(0, 0);
}

/* ================================================================ */
/* Tier-5 — recursive (online) estimation: forgetting-factor RLS    */
/*                                                                  */
/* θ(t) = θ(t-1) + K(t)·(y − φᵀθ),  K = Pφ/(λ + φᵀPφ),             */
/* P(t) = (P − KφᵀP)/λ.  recursiveLS takes the regressor φ from the */
/* user each step; recursiveARX builds φ from a buffered I/O history */
/* so it tracks time-varying ARX dynamics sample-by-sample.         */
/* ---------------------------------------------------------------- */
static void rls_update(double *theta, double *P, const double *phi,
                       int np, double y, double lambda) {
    std::vector<double> Pphi(static_cast<size_t>(np), 0.0);
    for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j) Pphi[static_cast<size_t>(i)] += P[i * np + j] * phi[j];
    double phiPphi = 0.0, yhat = 0.0;
    for (int i = 0; i < np; ++i) { phiPphi += phi[i] * Pphi[static_cast<size_t>(i)]; yhat += phi[i] * theta[i]; }
    double denom = lambda + phiPphi;
    if (denom == 0.0) denom = 1e-12;
    double err = y - yhat;
    for (int i = 0; i < np; ++i) theta[i] += (Pphi[static_cast<size_t>(i)] / denom) * err;
    for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j)
            P[i * np + j] = (P[i * np + j] - Pphi[static_cast<size_t>(i)] * Pphi[static_cast<size_t>(j)] / denom) / lambda;
}

/* recursiveLS — generic linear-regression RLS (user supplies φ). */
matlab_mat *matlab_ident_rls_init(void *obj_v, double np_d) {
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int np = static_cast<int>(np_d); if (np < 1) np = 1;
    matlab_mat *th = mat_alloc(np, 1);
    matlab_mat *P  = mat_alloc(np, np);
    for (int i = 0; i < np; ++i) P->data[i * np + i] = 1e4;
    matlab_obj_set_mat(o, "Parameters", 10, th);
    matlab_obj_set_mat(o, "Covariance", 10, P);
    matlab_obj_set_f64(o, "ForgettingFactor", 16, 1.0);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_ident_rls_step(void *obj_v, double y, matlab_mat *H) {
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    matlab_mat *th = matlab_obj_get_mat(o, "Parameters", 10);
    matlab_mat *P  = matlab_obj_get_mat(o, "Covariance", 10);
    double lambda  = matlab_obj_get_f64(o, "ForgettingFactor", 16);
    if (!th || !P || !H) return th ? th : mat_alloc(0, 0);
    int np = static_cast<int>(th->rows);
    if (mat_len(H) < np) return th;
    if (lambda <= 0.0) lambda = 1.0;
    rls_update(th->data, P->data, H->data, np, y, lambda);
    return th;
}

/* recursiveARX — RLS with an internally-buffered ARX regressor. */
matlab_mat *matlab_ident_rarx_init(void *obj_v, matlab_mat *orders) {
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    if (!orders) return mat_alloc(0, 0);
    int na = static_cast<int>(orders->data[0]);
    int nb = (mat_len(orders) >= 2) ? static_cast<int>(orders->data[1]) : 0;
    int nk = (mat_len(orders) >= 3) ? static_cast<int>(orders->data[2]) : 1;
    if (na < 0) na = 0; if (nb < 0) nb = 0; if (nk < 0) nk = 0;
    int np = na + nb; if (np < 1) np = 1;
    int L = nk + nb + 1;
    matlab_mat *th = mat_alloc(np, 1);
    matlab_mat *P  = mat_alloc(np, np);
    for (int i = 0; i < np; ++i) P->data[i * np + i] = 1e4;
    matlab_obj_set_mat(o, "Parameters", 10, th);
    matlab_obj_set_mat(o, "Covariance", 10, P);
    matlab_obj_set_f64(o, "ForgettingFactor", 16, 1.0);
    matlab_obj_set_f64(o, "na", 2, na);
    matlab_obj_set_f64(o, "nb", 2, nb);
    matlab_obj_set_f64(o, "nk", 2, nk);
    matlab_obj_set_mat(o, "yhist", 5, mat_alloc(na > 0 ? na : 1, 1));
    matlab_obj_set_mat(o, "uhist", 5, mat_alloc(L, 1));
    return mat_alloc(0, 0);
}

matlab_mat *matlab_ident_rarx_step(void *obj_v, double y, double u) {
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    matlab_mat *th = matlab_obj_get_mat(o, "Parameters", 10);
    matlab_mat *P  = matlab_obj_get_mat(o, "Covariance", 10);
    double lambda  = matlab_obj_get_f64(o, "ForgettingFactor", 16);
    int na = static_cast<int>(matlab_obj_get_f64(o, "na", 2));
    int nb = static_cast<int>(matlab_obj_get_f64(o, "nb", 2));
    int nk = static_cast<int>(matlab_obj_get_f64(o, "nk", 2));
    matlab_mat *yh = matlab_obj_get_mat(o, "yhist", 5);
    matlab_mat *uh = matlab_obj_get_mat(o, "uhist", 5);
    if (!th || !P || !yh || !uh) return th ? th : mat_alloc(0, 0);
    int np = na + nb;
    if (lambda <= 0.0) lambda = 1.0;
    /* φ = [−y(t-1) … −y(t-na), u(t-nk) … u(t-nk-nb+1)]. */
    std::vector<double> phi(static_cast<size_t>(np), 0.0);
    for (int j = 0; j < na; ++j) phi[static_cast<size_t>(j)] = -yh->data[j];
    /* ucur[0]=u(t), ucur[1]=u(t-1)… → ucur = [u; uhist]. */
    int L = static_cast<int>(mat_len(uh));
    for (int j = 0; j < nb; ++j) {
        int idx = nk + j;             /* index into ucur */
        double uv = (idx == 0) ? u : ((idx - 1 < L) ? uh->data[idx - 1] : 0.0);
        phi[static_cast<size_t>(na + j)] = uv;
    }
    rls_update(th->data, P->data, phi.data(), np, y, lambda);
    /* Shift the buffers: newest first. */
    for (int j = na - 1; j > 0; --j) yh->data[j] = yh->data[j - 1];
    if (na > 0) yh->data[0] = y;
    for (int j = L - 1; j > 0; --j) uh->data[j] = uh->data[j - 1];
    if (L > 0) uh->data[0] = u;
    return th;
}

/* ---------------------------------------------------------------- */
/* idpoly → ss controllable-canonical form (discrete, carries Ts).  */
/* Returns A / B / C / D blocks separately; the Lowering.cpp site    */
/* assembles ss__ss(A, B, C, D, Ts).                                 */
/*                                                                  */
/*   A(q) = [1 a1 … an],  B(q) padded to bb = [b0 b1 … bn]          */
/*   Acc row0 = [-a1 … -an]; sub-diagonal ones                      */
/*   Bcc = e1 ; Ccc[j] = b_{j+1} − b0·a_{j+1} ; Dcc = b0           */
/* ---------------------------------------------------------------- */
static void poly2ss_dims(matlab_obj *model, int *n_out) {
    matlab_mat *A = matlab_obj_get_mat(model, "A", 1);
    int n = A ? static_cast<int>(mat_len(A)) - 1 : 0;
    if (n < 1) n = 1;
    *n_out = n;
}

matlab_mat *matlab_ident_poly2ss_A(void *model_v) {
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_mat *A = matlab_obj_get_mat(model, "A", 1);
    int n; poly2ss_dims(model, &n);
    matlab_mat *Acc = mat_alloc(n, n);
    for (int j = 0; j < n; ++j)
        Acc->data[0 * n + j] = (A && j + 1 < mat_len(A)) ? -A->data[j + 1] : 0.0;
    for (int i = 1; i < n; ++i) Acc->data[i * n + (i - 1)] = 1.0;
    return Acc;
}

matlab_mat *matlab_ident_poly2ss_B(void *model_v) {
    int n; poly2ss_dims(reinterpret_cast<matlab_obj *>(model_v), &n);
    matlab_mat *Bcc = mat_alloc(n, 1);
    Bcc->data[0] = 1.0;
    return Bcc;
}

matlab_mat *matlab_ident_poly2ss_C(void *model_v) {
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_mat *A = matlab_obj_get_mat(model, "A", 1);
    matlab_mat *B = matlab_obj_get_mat(model, "B", 1);
    int n; poly2ss_dims(model, &n);
    double b0 = (B && mat_len(B) > 0) ? B->data[0] : 0.0;
    matlab_mat *Ccc = mat_alloc(1, n);
    for (int j = 0; j < n; ++j) {
        double bj1 = (B && j + 1 < mat_len(B)) ? B->data[j + 1] : 0.0;
        double aj1 = (A && j + 1 < mat_len(A)) ? A->data[j + 1] : 0.0;
        Ccc->data[j] = bj1 - b0 * aj1;
    }
    return Ccc;
}

matlab_mat *matlab_ident_poly2ss_D(void *model_v) {
    matlab_obj *model = reinterpret_cast<matlab_obj *>(model_v);
    matlab_mat *B = matlab_obj_get_mat(model, "B", 1);
    matlab_mat *D = mat_alloc(1, 1);
    D->data[0] = (B && mat_len(B) > 0) ? B->data[0] : 0.0;
    return D;
}

}  /* extern "C" */
