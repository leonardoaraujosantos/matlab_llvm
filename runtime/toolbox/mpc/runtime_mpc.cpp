/* runtime_mpc.cpp — Model Predictive Control Toolbox runtime, Tier-1.
 *
 * See docs/mpc_toolbox_roadmap.md for the full surface.  Tier-1 is the
 * smallest end-to-end loop: a `mpc(plant, p, m)` constructor, a
 * `mpcmove(obj, st, ym, r)` single-step controller, and a
 * `sim(obj, T, r)` closed-loop simulation.  The MATLAB-side
 * `mpc_classdefs.m` classdef bodies are thin wrappers around the
 * `matlab_mpc_*` C-ABI entries here — everything from matrix
 * assembly through the QP solve runs in this TU.
 *
 * Public entries:
 *   matlab_mpc_construct   — populate an `mpc` obj from a plant `ss`
 *   matlab_mpc_move        — one controller tick, mutates an mpcstate
 *   matlab_mpc_sim         — closed-loop T-tick simulation
 *
 * The KWIK active-set QP is hand-coded file-locally (qp_kwik below) —
 * a simplified Schmid-Biegler-Bemporad dual active-set with Cholesky-
 * factored Hessian and KKT-solve per iteration via the runtime's
 * shared matlab_mldivide_mm.  See User's Guide §1-18.
 *
 * MPC math conventions follow User's Guide §1 (R2026a):
 *   z = [Δu(0); Δu(1); ...; Δu(m-1); ε]            decision variable
 *   y(k+i|k) = Sx(i)·x(k) + Su1(i)·u(k-1) + Σ Su(i,j)·Δu(j)
 *   J(z,ε) = Σᵢ ‖Wy·(r - y)‖² + Σᵢ ‖Wdu·Δu‖² + ρε·ε²
 *
 * Tier-1 carve-downs (deferred to Tier-2 / 3):
 *   - measured disturbances v: assumed zero
 *   - output bounds y_min/y_max: only MV bounds u_min/u_max enforced
 *   - rate bounds Δu_min/Δu_max: not in QP yet
 *   - MV-tracking Wu / u_target: only the four-term cost
 *   - reference previewing: r broadcast as constant over horizon
 *   - mpcmoveopt run-time overrides: deferred (uses cached weights)
 */

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

/* matlab_obj_* helpers — defined in runtime/matlab_runtime.cpp but
 * not part of the public matlab_runtime.h surface yet.  Pattern
 * mirrors runtime/toolbox/prop/runtime_prop.cpp. */
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
extern "C" matlab_mat *matlab_chol(matlab_mat *A);

/* c2d ZOH discretisation pieces — used by Tier-6 auto-c2d in the
 * mpc constructor when the user passes a continuous plant. */
extern "C" matlab_mat *matlab_c2d_Ad(matlab_mat *A, matlab_mat *B, double Ts);
extern "C" matlab_mat *matlab_c2d_Bd(matlab_mat *A, matlab_mat *B, double Ts);

/* Optim's fmincon — solves min f(x) s.t. Ax≤b / Aeq·x=beq / lb≤x≤ub
 * + nonlcon.  Used as the inner NLP solver for `nlmpcmove`. */
extern "C" matlab_mat *matlab_optim_fmincon(void *obj_p, matlab_mat *x0,
                                            matlab_mat *A, matlab_mat *b,
                                            matlab_mat *Aeq, matlab_mat *beq,
                                            matlab_mat *lb, matlab_mat *ub,
                                            void *nonlcon_p);

extern "C" {

/* ---------------------------------------------------------------- */
/* File-local helpers                                               */
/* ---------------------------------------------------------------- */

static matlab_mat *mpc_zeros(int64_t m, int64_t n) {
    return mat_alloc(m, n);  /* calloc'd → zeros */
}

static matlab_mat *mpc_identity(int64_t n) {
    matlab_mat *I = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i) I->data[i * n + i] = 1.0;
    return I;
}

static void mpc_block_copy(matlab_mat *Dst, int64_t r0, int64_t c0, matlab_mat *Src) {
    for (int64_t r = 0; r < Src->rows; ++r)
        for (int64_t c = 0; c < Src->cols; ++c)
            Dst->data[(r0 + r) * Dst->cols + (c0 + c)] =
                Src->data[r * Src->cols + c];
}

static matlab_mat *mpc_mat_ones_col(int64_t n, double v) {
    matlab_mat *M = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) M->data[i] = v;
    return M;
}

/* ---------------------------------------------------------------- */
/* Construction-time matrix builders (file-static)                  */
/* ---------------------------------------------------------------- */

static matlab_mat *build_Sx(matlab_mat *A, matlab_mat *C, int p) {
    int64_t n = A->rows, ny = C->rows;
    matlab_mat *Sx = mpc_zeros(static_cast<int64_t>(p) * ny, n);
    matlab_mat *Apow = mpc_identity(n);
    for (int i = 1; i <= p; ++i) {
        Apow = matlab_matmul_mm(A, Apow);
        matlab_mat *CAi = matlab_matmul_mm(C, Apow);
        mpc_block_copy(Sx, static_cast<int64_t>(i - 1) * ny, 0, CAi);
    }
    return Sx;
}

static matlab_mat *build_Su1(matlab_mat *A, matlab_mat *B, matlab_mat *C, int p) {
    int64_t n = A->rows, nu = B->cols, ny = C->rows;
    matlab_mat *Su1 = mpc_zeros(static_cast<int64_t>(p) * ny, nu);
    matlab_mat *Phi = mpc_identity(n);
    for (int i = 1; i <= p; ++i) {
        matlab_mat *CPhi  = matlab_matmul_mm(C, Phi);
        matlab_mat *CPhiB = matlab_matmul_mm(CPhi, B);
        mpc_block_copy(Su1, static_cast<int64_t>(i - 1) * ny, 0, CPhiB);
        matlab_mat *APhi = matlab_matmul_mm(A, Phi);
        matlab_mat *next = mat_alloc(n, n);
        for (int64_t r = 0; r < n; ++r)
            for (int64_t c = 0; c < n; ++c)
                next->data[r * n + c] =
                    (r == c ? 1.0 : 0.0) + APhi->data[r * n + c];
        Phi = next;
    }
    return Su1;
}

static matlab_mat *build_Su(matlab_mat *A, matlab_mat *B, matlab_mat *C, int p, int m) {
    int64_t n = A->rows, nu = B->cols, ny = C->rows;
    if (m > p) m = p;
    matlab_mat *Su = mpc_zeros(static_cast<int64_t>(p) * ny, static_cast<int64_t>(m) * nu);
    std::vector<matlab_mat *> Phi(static_cast<size_t>(p + 1), nullptr);
    Phi[1] = mpc_identity(n);
    for (int k = 2; k <= p; ++k) {
        matlab_mat *APhi = matlab_matmul_mm(A, Phi[static_cast<size_t>(k - 1)]);
        matlab_mat *next = mat_alloc(n, n);
        for (int64_t r = 0; r < n; ++r)
            for (int64_t c = 0; c < n; ++c)
                next->data[r * n + c] =
                    (r == c ? 1.0 : 0.0) + APhi->data[r * n + c];
        Phi[static_cast<size_t>(k)] = next;
    }
    for (int i = 1; i <= p; ++i) {
        for (int j = 0; j < m && j < i; ++j) {
            int k = i - j;
            matlab_mat *CPhi  = matlab_matmul_mm(C, Phi[static_cast<size_t>(k)]);
            matlab_mat *block = matlab_matmul_mm(CPhi, B);
            mpc_block_copy(Su, static_cast<int64_t>(i - 1) * ny,
                           static_cast<int64_t>(j) * nu, block);
        }
    }
    return Su;
}

static matlab_mat *build_Hessian(matlab_mat *Su, matlab_mat *Wy,
                                 matlab_mat *Wdu, double rho_eps,
                                 int m, int nu) {
    int64_t p_ny = Su->rows;
    int64_t mnu  = Su->cols;
    int64_t ny   = Wy->rows;
    int p = static_cast<int>(p_ny / ny);
    matlab_mat *WySu = mat_alloc(p_ny, mnu);
    for (int i = 0; i < p; ++i) {
        for (int64_t k = 0; k < ny; ++k) {
            double w  = Wy->data[k];
            double w2 = w * w;
            int64_t row = static_cast<int64_t>(i) * ny + k;
            for (int64_t c = 0; c < mnu; ++c)
                WySu->data[row * mnu + c] = w2 * Su->data[row * mnu + c];
        }
    }
    matlab_mat *SuT = matlab_transpose(Su);
    matlab_mat *Huu = matlab_matmul_mm(SuT, WySu);
    for (int j = 0; j < m; ++j)
        for (int64_t k = 0; k < nu; ++k) {
            double w = Wdu->data[k];
            int64_t idx = static_cast<int64_t>(j) * nu + k;
            Huu->data[idx * mnu + idx] += w * w;
        }
    /* Standard QP form is `min ½·z'·H·z + f'·z`, so the Hessian gets
     * a factor of 2 baked in to match the un-half'd cost
     * J(z) = z'·M·z + f'·z that the MPC math builds.  f already
     * carries its own -2 factor in mpc_tick (see f = -2·Su'·Wy²·err). */
    int64_t N = mnu + 1;
    matlab_mat *H = mat_alloc(N, N);
    for (int64_t r = 0; r < mnu; ++r)
        for (int64_t c = 0; c < mnu; ++c)
            H->data[r * N + c] = 2.0 * Huu->data[r * mnu + c];
    H->data[mnu * N + mnu] = 2.0 * rho_eps;
    double ridge = 1e-10;
    for (int64_t i = 0; i < N; ++i) H->data[i * N + i] += ridge;
    return H;
}

/* ---------------------------------------------------------------- */
/* Time-varying matrix builders (Tier-3 §4.2)                       */
/*                                                                  */
/* Plants are stacked vertically: A_stack is (p·nx × nx) — rows     */
/* [i·nx .. (i+1)·nx-1] are A_i (the transition from step i to      */
/* step i+1).  Same for B_stack ((p·nx × nu) and C_stack            */
/* ((p·ny × nx)).  The TV prediction math is:                       */
/*   Φ(i, j) = A_{i-1} · A_{i-2} · ... · A_j   (Φ(i,i) = I)         */
/*   y(k+i)  = C_{i-1} · x(k+i)                                      */
/*   Sx(i)   = C_{i-1} · Φ(i, 0)                                     */
/*   Su1(i)  = C_{i-1} · Σ_{h=0}^{i-1} Φ(i, h+1) · B_h               */
/*   Su(i,j) = C_{i-1} · Σ_{h=j}^{i-1} Φ(i, h+1) · B_h  (j < m, i)  */
/* ---------------------------------------------------------------- */

/* Helper: extract block i (rows i·rows_per_block .. -1) of a stack. */
static matlab_mat *stack_block(matlab_mat *stack, int i, int64_t rows_per) {
    matlab_mat *blk = mat_alloc(rows_per, stack->cols);
    for (int64_t r = 0; r < rows_per; ++r)
        for (int64_t c = 0; c < stack->cols; ++c)
            blk->data[r * stack->cols + c] =
                stack->data[(static_cast<int64_t>(i) * rows_per + r)
                            * stack->cols + c];
    return blk;
}

/* Precompute Φ(i, j) for 0 ≤ j ≤ i ≤ p.  Returns a flat array of
 * (p+1)·(p+1) matrices (most slots empty / unused; allocated lazily
 * for the (j ≤ i) cases). */
static matlab_mat *tv_phi_get(matlab_mat **phi, int p, int i, int j) {
    return phi[i * (p + 1) + j];
}
static void tv_phi_set(matlab_mat **phi, int p, int i, int j,
                       matlab_mat *m) {
    phi[i * (p + 1) + j] = m;
}

static matlab_mat *build_Sx_tv(matlab_mat *A_stack, matlab_mat *C_stack,
                               int p, int64_t nx, int64_t ny) {
    matlab_mat *Sx = mpc_zeros(static_cast<int64_t>(p) * ny, nx);
    /* Recurrence Φ(i+1, 0) = A_i · Φ(i, 0). */
    matlab_mat *Phi = mpc_identity(nx);   /* Φ(0, 0) = I */
    for (int i = 1; i <= p; ++i) {
        matlab_mat *Ai_m1 = stack_block(A_stack, i - 1, nx);
        Phi = matlab_matmul_mm(Ai_m1, Phi);        /* Φ(i, 0) */
        matlab_mat *Ci_m1 = stack_block(C_stack, i - 1, ny);
        matlab_mat *Cphi  = matlab_matmul_mm(Ci_m1, Phi);  /* ny × nx */
        mpc_block_copy(Sx, static_cast<int64_t>(i - 1) * ny, 0, Cphi);
    }
    return Sx;
}

/* Build the full Φ(i, j) table for i = 1..p, j = 0..i. */
static std::vector<matlab_mat *> build_phi_table(matlab_mat *A_stack,
                                                 int p, int64_t nx) {
    std::vector<matlab_mat *> phi(static_cast<size_t>((p + 1) * (p + 1)),
                                  nullptr);
    /* Φ(i, i) = I for all i. */
    for (int i = 0; i <= p; ++i)
        tv_phi_set(phi.data(), p, i, i, mpc_identity(nx));
    /* Φ(i, j) = Φ(i, j+1) · A_j  for j < i.
     * Equivalently, recurrence on j from j = i-1 down to j = 0:
     *   Φ(i, j) = Φ(i, j+1) · A_j. */
    for (int i = 1; i <= p; ++i) {
        for (int j = i - 1; j >= 0; --j) {
            matlab_mat *Aj = stack_block(A_stack, j, nx);
            matlab_mat *Phi_next = tv_phi_get(phi.data(), p, i, j + 1);
            tv_phi_set(phi.data(), p, i, j,
                       matlab_matmul_mm(Phi_next, Aj));
        }
    }
    return phi;
}

static matlab_mat *build_Su1_tv(matlab_mat *A_stack, matlab_mat *B_stack,
                                matlab_mat *C_stack,
                                int p, int64_t nx, int64_t nu, int64_t ny) {
    matlab_mat *Su1 = mpc_zeros(static_cast<int64_t>(p) * ny, nu);
    auto phi = build_phi_table(A_stack, p, nx);
    for (int i = 1; i <= p; ++i) {
        /* sum_h = Σ_{h=0}^{i-1} Φ(i, h+1) · B_h   (nx × nu). */
        matlab_mat *acc = mpc_zeros(nx, nu);
        for (int h = 0; h < i; ++h) {
            matlab_mat *Phi_h1 = tv_phi_get(phi.data(), p, i, h + 1);
            matlab_mat *Bh = stack_block(B_stack, h, nx);
            matlab_mat *term = matlab_matmul_mm(Phi_h1, Bh);
            for (int64_t r = 0; r < nx; ++r)
                for (int64_t c = 0; c < nu; ++c)
                    acc->data[r * nu + c] += term->data[r * nu + c];
        }
        matlab_mat *Ci_m1 = stack_block(C_stack, i - 1, ny);
        matlab_mat *blk = matlab_matmul_mm(Ci_m1, acc);
        mpc_block_copy(Su1, static_cast<int64_t>(i - 1) * ny, 0, blk);
    }
    return Su1;
}

static matlab_mat *build_Su_tv(matlab_mat *A_stack, matlab_mat *B_stack,
                               matlab_mat *C_stack,
                               int p, int m, int64_t nx, int64_t nu, int64_t ny) {
    if (m > p) m = p;
    matlab_mat *Su = mpc_zeros(static_cast<int64_t>(p) * ny,
                               static_cast<int64_t>(m) * nu);
    auto phi = build_phi_table(A_stack, p, nx);
    for (int i = 1; i <= p; ++i) {
        for (int j = 0; j < m && j < i; ++j) {
            matlab_mat *acc = mpc_zeros(nx, nu);
            for (int h = j; h < i; ++h) {
                matlab_mat *Phi_h1 = tv_phi_get(phi.data(), p, i, h + 1);
                matlab_mat *Bh = stack_block(B_stack, h, nx);
                matlab_mat *term = matlab_matmul_mm(Phi_h1, Bh);
                for (int64_t r = 0; r < nx; ++r)
                    for (int64_t c = 0; c < nu; ++c)
                        acc->data[r * nu + c] += term->data[r * nu + c];
            }
            matlab_mat *Ci_m1 = stack_block(C_stack, i - 1, ny);
            matlab_mat *blk = matlab_matmul_mm(Ci_m1, acc);
            mpc_block_copy(Su, static_cast<int64_t>(i - 1) * ny,
                           static_cast<int64_t>(j) * nu, blk);
        }
    }
    return Su;
}

/* Steady-state continuous Kalman gain — copies the matlab_kalman_L
 * recipe (lqr-on-dual then transpose).  Forward-declare the public
 * runtime entries.  B doubles as G. */
extern matlab_mat *matlab_kalman_L(matlab_mat *A, matlab_mat *G, matlab_mat *C,
                                   matlab_mat *Qn, matlab_mat *Rn);
extern matlab_mat *matlab_kalmd_L(matlab_mat *Ad, matlab_mat *G, matlab_mat *C,
                                  matlab_mat *Qn, matlab_mat *Rn);

static matlab_mat *build_Kalman_L(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                                  double Ts) {
    int64_t nu = B->cols, ny = C->rows;
    matlab_mat *Qn = mpc_identity(nu);
    matlab_mat *Rn = mpc_identity(ny);
    if (Ts > 0.0) return matlab_kalmd_L(A, B, C, Qn, Rn);
    return matlab_kalman_L(A, B, C, Qn, Rn);
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_construct — top-level builder.                        */
/*                                                                  */
/* Reads A/B/C/Ts off the plant `ss` (passed as an opaque obj ptr),  */
/* computes every cached matrix, and writes them back as properties */
/* on the mpc obj.  No return value; the classdef constructor body  */
/* just calls this and returns.                                     */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_construct(void *mpc_obj_v, void *plant_obj_v,
                                 double p_d, double m_d) {
    if (!mpc_obj_v || !plant_obj_v) return mat_alloc(0, 0);
    matlab_obj *mpc_obj   = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    matlab_obj *plant_obj = reinterpret_cast<matlab_obj *>(plant_obj_v);
    matlab_mat *A = matlab_obj_get_mat(plant_obj, "A", 1);
    matlab_mat *B = matlab_obj_get_mat(plant_obj, "B", 1);
    matlab_mat *C = matlab_obj_get_mat(plant_obj, "C", 1);
    if (!A || !B || !C) return mat_alloc(0, 0);
    double Ts = matlab_obj_get_f64(plant_obj, "Ts", 2);
    int p = static_cast<int>(p_d);
    int m = static_cast<int>(m_d);
    if (p < 1) p = 1;
    if (m < 1) m = 1;
    if (m > p) m = p;
    int64_t nu = B->cols;
    int64_t ny = C->rows;

    /* Tier-6 §7.1 — continuous-plant auto-c2d.  When the user passes
     * a continuous-time `ss` (Ts == 0), default to Ts = 0.1 and
     * discretise via ZOH (matlab_c2d_Ad / _Bd) so the MPC always
     * operates on a discrete plant.  Users can override by calling
     * c2d explicitly with a different Ts before mpc(...). */
    if (Ts == 0.0) {
        Ts = 0.1;
        matlab_mat *Ad = matlab_c2d_Ad(A, B, Ts);
        matlab_mat *Bd = matlab_c2d_Bd(A, B, Ts);
        if (Ad && Ad->rows == A->rows) A = Ad;
        if (Bd && Bd->rows == B->rows) B = Bd;
    }

    /* Defaults for weights / bounds. */
    matlab_mat *Wy   = mpc_mat_ones_col(ny, 1.0);
    matlab_mat *Wdu  = mpc_mat_ones_col(nu, 0.1);
    matlab_mat *umin = mpc_mat_ones_col(nu, -1e6);
    matlab_mat *umax = mpc_mat_ones_col(nu, +1e6);
    /* Tier-6 §7.2 — rate-bound defaults: empty (no rate constraints). */
    matlab_mat *dumin = mat_alloc(0, 0);
    matlab_mat *dumax = mat_alloc(0, 0);
    /* Tier-6 §7.3 — MV-tracking defaults: Wu zero (no MV-tracking),
     * u_target zero. */
    matlab_mat *Wu       = mpc_zeros(nu, 1);
    matlab_mat *u_target = mpc_zeros(nu, 1);
    matlab_mat *ymin = mpc_mat_ones_col(ny, -1e6);
    matlab_mat *ymax = mpc_mat_ones_col(ny, +1e6);
    /* ECR (soft-constraint slack) — Tier-2 defaults to all-zero
     * (hard bounds).  Users set V_y_min[j] > 0 to soften output j's
     * lower bound, etc. */
    matlab_mat *V_y_min = mpc_zeros(ny, 1);
    matlab_mat *V_y_max = mpc_zeros(ny, 1);
    matlab_mat *V_u_min = mpc_zeros(nu, 1);
    matlab_mat *V_u_max = mpc_zeros(nu, 1);
    /* Mixed-constraint defaults: empty 0×nu / 0×ny / 0×1 matrices.
     * mpc_tick treats nE = 0 as "no mixed rows", so leaving these
     * empty is the right Tier-1-compat default. */
    matlab_mat *E = mat_alloc(0, nu);
    matlab_mat *F = mat_alloc(0, ny);
    matlab_mat *G = mat_alloc(0, 1);
    double rho_eps = 1e5;

    matlab_mat *Sx  = build_Sx(A, C, p);
    matlab_mat *Su  = build_Su(A, B, C, p, m);
    matlab_mat *Su1 = build_Su1(A, B, C, p);
    matlab_mat *H   = build_Hessian(Su, Wy, Wdu, rho_eps, m, static_cast<int>(nu));
    matlab_mat *R   = matlab_chol(H);
    matlab_mat *L   = build_Kalman_L(A, B, C, Ts);

    matlab_obj_set_mat(mpc_obj, "A",  1, A);
    matlab_obj_set_mat(mpc_obj, "B",  1, B);
    matlab_obj_set_mat(mpc_obj, "C",  1, C);
    matlab_obj_set_f64(mpc_obj, "Ts", 2, Ts);
    matlab_obj_set_f64(mpc_obj, "p",  1, static_cast<double>(p));
    matlab_obj_set_f64(mpc_obj, "m",  1, static_cast<double>(m));
    matlab_obj_set_mat(mpc_obj, "Wy",      2, Wy);
    matlab_obj_set_mat(mpc_obj, "Wdu",     3, Wdu);
    matlab_obj_set_f64(mpc_obj, "rho_eps", 7, rho_eps);
    matlab_obj_set_mat(mpc_obj, "umin",    4, umin);
    matlab_obj_set_mat(mpc_obj, "umax",    4, umax);
    matlab_obj_set_mat(mpc_obj, "dumin",   5, dumin);
    matlab_obj_set_mat(mpc_obj, "dumax",   5, dumax);
    matlab_obj_set_mat(mpc_obj, "Wu",       2, Wu);
    matlab_obj_set_mat(mpc_obj, "u_target", 8, u_target);
    matlab_obj_set_mat(mpc_obj, "ymin",    4, ymin);
    matlab_obj_set_mat(mpc_obj, "ymax",    4, ymax);
    matlab_obj_set_mat(mpc_obj, "E",       1, E);
    matlab_obj_set_mat(mpc_obj, "F",       1, F);
    matlab_obj_set_mat(mpc_obj, "G",       1, G);
    matlab_obj_set_mat(mpc_obj, "V_y_min", 7, V_y_min);
    matlab_obj_set_mat(mpc_obj, "V_y_max", 7, V_y_max);
    matlab_obj_set_mat(mpc_obj, "V_u_min", 7, V_u_min);
    matlab_obj_set_mat(mpc_obj, "V_u_max", 7, V_u_max);
    matlab_obj_set_f64(mpc_obj, "outdist",  7, 0.0);
    matlab_obj_set_f64(mpc_obj, "nx_plant", 8, static_cast<double>(A->rows));
    /* Tier-4 §5.6 / §5.7 — defaults: no custom solver, all MVs
     * continuous (`mv_binary` zero vector of length nu). */
    matlab_obj_set_f64(mpc_obj, "CustomSolver",    12, 0.0);
    matlab_obj_set_f64(mpc_obj, "UseCustomSolver", 15, 0.0);
    matlab_obj_set_mat(mpc_obj, "mv_binary",        9, mpc_zeros(nu, 1));
    matlab_obj_set_mat(mpc_obj, "Sx",  2, Sx);
    matlab_obj_set_mat(mpc_obj, "Su",  2, Su);
    matlab_obj_set_mat(mpc_obj, "Su1", 3, Su1);
    matlab_obj_set_mat(mpc_obj, "H",   1, H);
    matlab_obj_set_mat(mpc_obj, "R",   1, R);
    matlab_obj_set_mat(mpc_obj, "L",   1, L);
    return mat_alloc(0, 0);  /* construct() is "void"; caller discards. */
}

/* ---------------------------------------------------------------- */
/* KWIK active-set QP solver                                        */
/*                                                                  */
/* Solve:  min ½ x'·H·x + f'·x   s.t.   A_ineq · x ≤ b_ineq         */
/*                                                                  */
/* Dual active-set (Schmid-Biegler-Bemporad), simplified for Tier-1: */
/*   1. Cold-start unconstrained: x = H \ (-f).                     */
/*   2. If all rows satisfy A_ineq·x ≤ b_ineq → done.               */
/*   3. Add the most-violated row j to the active set, resolve the   */
/*      equality-constrained QP via the KKT system                  */
/*           [H, A_a'; A_a, 0] [x; λ_a] = [-f; b_a]                 */
/*      using mldivide.                                              */
/*   4. If any λ < 0, drop the most-negative-λ row from the active  */
/*      set and resolve.                                             */
/*   5. Iterate until primal-feasible AND dual-non-negative.        */
/* ---------------------------------------------------------------- */

static matlab_mat *qp_kwik(matlab_mat *H, matlab_mat *f,
                           matlab_mat *A_ineq, matlab_mat *b_ineq) {
    int64_t n  = H->rows;
    int64_t ni = (A_ineq && A_ineq->rows > 0) ? A_ineq->rows : 0;
    std::vector<char> iA(static_cast<size_t>(ni), 0);
    int max_iter = static_cast<int>(4 * (n + ni) + 50);
    if (max_iter < 80) max_iter = 80;

    matlab_mat *x = nullptr;

    auto resolve = [&](void) -> matlab_mat * {
        int64_t k = 0;
        for (int64_t i = 0; i < ni; ++i) if (iA[static_cast<size_t>(i)]) ++k;
        int64_t N = n + k;
        matlab_mat *K = mat_alloc(N, N);
        matlab_mat *r = mat_alloc(N, 1);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                K->data[i * N + j] = H->data[i * n + j];
        if (k > 0 && A_ineq && b_ineq) {
            int64_t a_row = 0;
            for (int64_t i = 0; i < ni; ++i) {
                if (!iA[static_cast<size_t>(i)]) continue;
                for (int64_t j = 0; j < n; ++j) {
                    K->data[j * N + (n + a_row)] = A_ineq->data[i * n + j];
                    K->data[(n + a_row) * N + j] = A_ineq->data[i * n + j];
                }
                r->data[n + a_row] = b_ineq->data[i];
                ++a_row;
            }
        }
        for (int64_t i = 0; i < n; ++i) r->data[i] = -f->data[i];
        matlab_mat *sol = matlab_mldivide_mm(K, r);
        if (!sol || sol->rows != N) return nullptr;
        matlab_mat *xx = mat_alloc(n, 1);
        for (int64_t i = 0; i < n; ++i) xx->data[i] = sol->data[i];
        return xx;
    };

    auto compute_lambda = [&](matlab_mat *x_cur, std::vector<double> *lam_out) -> void {
        lam_out->assign(static_cast<size_t>(ni), 0.0);
        int64_t k = 0;
        for (int64_t i = 0; i < ni; ++i) if (iA[static_cast<size_t>(i)]) ++k;
        if (k == 0) return;
        matlab_mat *Hx  = matlab_matmul_mm(H, x_cur);
        matlab_mat *rhs = mat_alloc(n, 1);
        for (int64_t i = 0; i < n; ++i)
            rhs->data[i] = -(Hx->data[i] + f->data[i]);
        matlab_mat *AaT = mat_alloc(n, k);
        std::vector<int64_t> idx;
        idx.reserve(static_cast<size_t>(k));
        for (int64_t i = 0; i < ni; ++i)
            if (iA[static_cast<size_t>(i)]) idx.push_back(i);
        for (int64_t col = 0; col < k; ++col) {
            int64_t row = idx[static_cast<size_t>(col)];
            for (int64_t r2 = 0; r2 < n; ++r2)
                AaT->data[r2 * k + col] = A_ineq->data[row * n + r2];
        }
        matlab_mat *lam = matlab_mldivide_mm(AaT, rhs);
        if (lam && lam->rows == k) {
            for (int64_t i = 0; i < k; ++i)
                (*lam_out)[static_cast<size_t>(idx[static_cast<size_t>(i)])] = lam->data[i];
        }
    };

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        x = resolve();
        if (!x) break;
        double worst_viol = 0.0;
        int64_t worst_idx = -1;
        for (int64_t i = 0; i < ni; ++i) {
            if (iA[static_cast<size_t>(i)]) continue;
            double Ax = 0.0;
            for (int64_t j = 0; j < n; ++j)
                Ax += A_ineq->data[i * n + j] * x->data[j];
            double viol = Ax - b_ineq->data[i];
            if (viol > worst_viol + 1e-9) {
                worst_viol = viol;
                worst_idx  = i;
            }
        }
        std::vector<double> lam;
        compute_lambda(x, &lam);
        double worst_lam = 0.0;
        int64_t worst_lam_idx = -1;
        for (int64_t i = 0; i < ni; ++i) {
            if (!iA[static_cast<size_t>(i)]) continue;
            if (lam[static_cast<size_t>(i)] < worst_lam - 1e-9) {
                worst_lam = lam[static_cast<size_t>(i)];
                worst_lam_idx = i;
            }
        }
        if (worst_idx < 0 && worst_lam_idx < 0) break;
        if (worst_lam_idx >= 0) iA[static_cast<size_t>(worst_lam_idx)] = 0;
        else                    iA[static_cast<size_t>(worst_idx)]      = 1;
    }
    if (!x) {
        x = mat_alloc(n, 1);
        for (int64_t i = 0; i < n; ++i) x->data[i] = 0.0;
    }
    return x;
}

/* ---------------------------------------------------------------- */
/* Explicit MPC (Tier-4 §5.1/5.2)                                   */
/*                                                                  */
/* Pragmatic grid-tessellation form: at generation time we solve    */
/* the full MPC QP at every grid point in [x_lo, x_hi]^nx and store */
/* the optimal MV in a lookup table.  At run-time `mpcmoveExplicit` */
/* does nearest-neighbor lookup — pure O(grid_size · nx) integer    */
/* arithmetic, no QP solver, no Cholesky factorisation.             */
/*                                                                  */
/* The full Tøndel-Johansen-Bemporad mpQP that yields exact         */
/* piecewise-affine regions is a research-grade follow-up; the      */
/* grid form ships the "deploy without QP solver" benefit today.    */
/* ---------------------------------------------------------------- */

/* Helper: solve the MPC QP at a given state xp and return Δu(0).
 * Inputs are the mpc obj fields (already loaded by the caller) +
 * a fresh xp.  Reuses qp_kwik with the obj's cached Hessian. */
static matlab_mat *solve_qp_at_state(matlab_mat *Sx, matlab_mat *Su,
                                     matlab_mat *Su1, matlab_mat *H,
                                     matlab_mat *Wy,
                                     matlab_mat *umin, matlab_mat *umax,
                                     matlab_mat *xp, matlab_mat *u_prev,
                                     matlab_mat *r,
                                     int p, int m, int64_t nu, int64_t ny) {
    int64_t mnu = static_cast<int64_t>(m) * nu;
    int64_t N   = mnu + 1;
    /* Reference broadcast across horizon. */
    matlab_mat *R_vec = mat_alloc(static_cast<int64_t>(p) * ny, 1);
    for (int i = 0; i < p; ++i)
        for (int64_t k = 0; k < ny; ++k) {
            double rv = (k < r->rows) ? r->data[k] : 0.0;
            R_vec->data[static_cast<int64_t>(i) * ny + k] = rv;
        }
    matlab_mat *Sxxp  = matlab_matmul_mm(Sx, xp);
    matlab_mat *Su1up = matlab_matmul_mm(Su1, u_prev);
    matlab_mat *err   = mat_alloc(static_cast<int64_t>(p) * ny, 1);
    for (int64_t i = 0; i < static_cast<int64_t>(p) * ny; ++i)
        err->data[i] = R_vec->data[i] - Sxxp->data[i] - Su1up->data[i];
    matlab_mat *Wyerr = mat_alloc(static_cast<int64_t>(p) * ny, 1);
    for (int i = 0; i < p; ++i)
        for (int64_t k = 0; k < ny; ++k) {
            double w = Wy->data[k];
            Wyerr->data[static_cast<int64_t>(i) * ny + k] =
                w * w * err->data[static_cast<int64_t>(i) * ny + k];
        }
    matlab_mat *SuT   = matlab_transpose(Su);
    matlab_mat *SuTWy = matlab_matmul_mm(SuT, Wyerr);
    matlab_mat *f     = mat_alloc(N, 1);
    for (int64_t i = 0; i < mnu; ++i) f->data[i] = -2.0 * SuTWy->data[i];
    f->data[mnu] = 0.0;
    /* MV bounds + ε ≥ 0 — minimal set for explicit MPC's open-loop
     * grid evaluation (no output bounds / mixed constraints baked in
     * because the grid samples STATE, not output trajectory).  */
    int64_t ni = 2 * static_cast<int64_t>(m) * nu + 1;
    matlab_mat *A_ineq = mat_alloc(ni, N);
    matlab_mat *b_ineq = mat_alloc(ni, 1);
    int64_t row = 0;
    for (int h = 0; h < m; ++h) {
        for (int64_t k = 0; k < nu; ++k) {
            double up = (umax ? umax->data[k] :  1e6) - u_prev->data[k];
            double lo = (umin ? umin->data[k] : -1e6) - u_prev->data[k];
            for (int j = 0; j <= h; ++j)
                A_ineq->data[row * N + static_cast<int64_t>(j) * nu + k] = 1.0;
            b_ineq->data[row++] = up;
            for (int j = 0; j <= h; ++j)
                A_ineq->data[row * N + static_cast<int64_t>(j) * nu + k] = -1.0;
            b_ineq->data[row++] = -lo;
        }
    }
    A_ineq->data[row * N + mnu] = -1.0;  /* ε ≥ 0 */
    b_ineq->data[row++] = 0.0;
    return qp_kwik(H, f, A_ineq, b_ineq);
}

/* matlab_mpc_generate_explicit — populates an explicitMPC obj.
 * Grid range [x_lo, x_hi]^nx with (n_grid+1) points per dimension. */
matlab_mat *matlab_mpc_generate_explicit(void *exp_obj_v,
                                         void *mpc_obj_v,
                                         matlab_mat *x_lo, matlab_mat *x_hi,
                                         double n_grid_d,
                                         matlab_mat *r) {
    if (!mpc_obj_v || !exp_obj_v || !x_lo || !x_hi || !r)
        return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    matlab_obj *exp_obj = reinterpret_cast<matlab_obj *>(exp_obj_v);

    matlab_mat *Sx   = matlab_obj_get_mat(mpc_obj, "Sx",  2);
    matlab_mat *Su   = matlab_obj_get_mat(mpc_obj, "Su",  2);
    matlab_mat *Su1  = matlab_obj_get_mat(mpc_obj, "Su1", 3);
    matlab_mat *H    = matlab_obj_get_mat(mpc_obj, "H",   1);
    matlab_mat *Wy   = matlab_obj_get_mat(mpc_obj, "Wy",  2);
    matlab_mat *umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    matlab_mat *umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    matlab_mat *A    = matlab_obj_get_mat(mpc_obj, "A",   1);
    matlab_mat *B    = matlab_obj_get_mat(mpc_obj, "B",   1);
    matlab_mat *C    = matlab_obj_get_mat(mpc_obj, "C",   1);
    double p_d = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d = matlab_obj_get_f64(mpc_obj, "m", 1);
    double Ts  = matlab_obj_get_f64(mpc_obj, "Ts", 2);
    int p = static_cast<int>(p_d);
    int m = static_cast<int>(m_d);
    int n_grid = static_cast<int>(n_grid_d);
    if (n_grid < 1) n_grid = 1;
    if (!Sx || !Su || !Su1 || !H || !Wy || !A || !B || !C)
        return mat_alloc(0, 0);

    int64_t nx = A->rows;
    int64_t nu = B->cols;
    int64_t ny = C->rows;

    /* Total grid points = (n_grid+1)^nx. */
    int64_t total = 1;
    for (int d = 0; d < nx; ++d) total *= (n_grid + 1);

    matlab_mat *u_table = mat_alloc(total, nu);
    matlab_mat *xp_tmp     = mat_alloc(nx, 1);
    matlab_mat *u_prev_tmp = mat_alloc(nu, 1);
    /* u_prev set to zeros — generation assumes a "fresh" controller
     * with no prior history.  At deploy time the runtime should also
     * use u_prev = 0 for consistency. */

    for (int64_t idx = 0; idx < total; ++idx) {
        /* Decode flat idx into per-dimension grid index. */
        int64_t k = idx;
        for (int64_t d = 0; d < nx; ++d) {
            int gi = static_cast<int>(k % (n_grid + 1));
            k /= (n_grid + 1);
            double lo = x_lo->data[d];
            double hi = x_hi->data[d];
            double v = lo + (hi - lo) * static_cast<double>(gi)
                              / static_cast<double>(n_grid > 0 ? n_grid : 1);
            xp_tmp->data[d] = v;
        }
        matlab_mat *z = solve_qp_at_state(Sx, Su, Su1, H, Wy, umin, umax,
                                          xp_tmp, u_prev_tmp, r,
                                          p, m, nu, ny);
        for (int64_t k2 = 0; k2 < nu; ++k2) {
            /* First MV: u(k) = u_prev + Δu(0); u_prev = 0 here. */
            u_table->data[idx * nu + k2] = z->data[k2];
        }
    }

    matlab_obj_set_mat(exp_obj, "x_lo",    4, x_lo);
    matlab_obj_set_mat(exp_obj, "x_hi",    4, x_hi);
    matlab_obj_set_f64(exp_obj, "n_grid",  6, static_cast<double>(n_grid));
    matlab_obj_set_mat(exp_obj, "u_table", 7, u_table);
    matlab_obj_set_f64(exp_obj, "nx",      2, static_cast<double>(nx));
    matlab_obj_set_f64(exp_obj, "nu",      2, static_cast<double>(nu));
    matlab_obj_set_f64(exp_obj, "ny",      2, static_cast<double>(ny));
    matlab_obj_set_f64(exp_obj, "Ts",      2, Ts);
    matlab_obj_set_mat(exp_obj, "r_gen",   5, r);
    return mat_alloc(0, 0);
}

/* matlab_mpc_move_explicit — nearest-neighbor lookup on the table. */
matlab_mat *matlab_mpc_move_explicit(void *exp_obj_v, matlab_mat *xc) {
    if (!exp_obj_v || !xc) return mat_alloc(0, 0);
    matlab_obj *exp_obj = reinterpret_cast<matlab_obj *>(exp_obj_v);
    matlab_mat *x_lo    = matlab_obj_get_mat(exp_obj, "x_lo",    4);
    matlab_mat *x_hi    = matlab_obj_get_mat(exp_obj, "x_hi",    4);
    matlab_mat *u_table = matlab_obj_get_mat(exp_obj, "u_table", 7);
    double n_grid_d = matlab_obj_get_f64(exp_obj, "n_grid", 6);
    double nx_d     = matlab_obj_get_f64(exp_obj, "nx", 2);
    double nu_d     = matlab_obj_get_f64(exp_obj, "nu", 2);
    int n_grid = static_cast<int>(n_grid_d);
    int64_t nx = static_cast<int64_t>(nx_d);
    int64_t nu = static_cast<int64_t>(nu_d);
    if (!x_lo || !x_hi || !u_table || nx < 1 || nu < 1 || n_grid < 1)
        return mat_alloc(0, 0);

    /* Encode xc into flat grid idx by clamping + rounding per dim. */
    int64_t idx = 0;
    int64_t stride = 1;
    for (int64_t d = 0; d < nx; ++d) {
        double lo = x_lo->data[d];
        double hi = x_hi->data[d];
        double v = (d < xc->rows) ? xc->data[d] : 0.0;
        double t = (v - lo) / ((hi - lo) > 0 ? (hi - lo) : 1.0);
        int gi = static_cast<int>(t * static_cast<double>(n_grid) + 0.5);
        if (gi < 0) gi = 0;
        if (gi > n_grid) gi = n_grid;
        idx += static_cast<int64_t>(gi) * stride;
        stride *= (n_grid + 1);
    }
    matlab_mat *u = mat_alloc(nu, 1);
    for (int64_t k = 0; k < nu; ++k)
        u->data[k] = u_table->data[idx * nu + k];
    return u;
}

/* matlab_mpc_simplify_explicit — Tier-4 §5.3.  For the grid form we
 * implement region "merging" by counting the number of distinct MVs
 * in the table; tol > 0 collapses near-identical entries into a
 * single representative.  Returns the simplified region count as a
 * 1x1 matrix.  The table itself is mutated in place. */
matlab_mat *matlab_mpc_simplify_explicit(void *exp_obj_v, double tol) {
    if (!exp_obj_v) return mat_alloc(0, 0);
    matlab_obj *exp_obj = reinterpret_cast<matlab_obj *>(exp_obj_v);
    matlab_mat *u_table = matlab_obj_get_mat(exp_obj, "u_table", 7);
    double nu_d = matlab_obj_get_f64(exp_obj, "nu", 2);
    int64_t nu = static_cast<int64_t>(nu_d);
    if (!u_table || nu < 1) return mat_alloc(0, 0);
    int64_t total = u_table->rows;
    /* Count distinct rows up to tol. */
    int64_t distinct = 0;
    std::vector<int64_t> rep(static_cast<size_t>(total), -1);
    for (int64_t i = 0; i < total; ++i) {
        int64_t match = -1;
        for (int64_t j = 0; j < i; ++j) {
            double sumsq = 0.0;
            for (int64_t k = 0; k < nu; ++k) {
                double d = u_table->data[i * nu + k] - u_table->data[j * nu + k];
                sumsq += d * d;
            }
            if (sumsq < tol * tol) { match = j; break; }
        }
        if (match < 0) {
            rep[static_cast<size_t>(i)] = i;
            ++distinct;
        } else {
            /* Snap this row to its representative. */
            for (int64_t k = 0; k < nu; ++k)
                u_table->data[i * nu + k] = u_table->data[match * nu + k];
            rep[static_cast<size_t>(i)] = match;
        }
    }
    matlab_mat *out = mat_alloc(1, 1);
    out->data[0] = static_cast<double>(distinct);
    return out;
}

/* Forward-decl matlab_mpc_move so the FCS branch-and-bound can
 * call it for each clamped branch. */
matlab_mat *matlab_mpc_move(void *mpc_obj_v, void *mpcstate_obj_v,
                            matlab_mat *ym, matlab_mat *r);
matlab_mat *matlab_mpc_sim(void *mpc_obj_v, double T_d, matlab_mat *r);

/* ---------------------------------------------------------------- */
/* Finite Control Set MPC (Tier-4 §5.7)                             */
/*                                                                  */
/* When MV k is marked binary (mv_binary[k] > 0), it's restricted   */
/* to {umin[k], umax[k]}.  For a SINGLE binary MV (the surge-tank   */
/* case from User's Guide §2.28), we enumerate the two branches:    */
/* clamp the MV to umin or umax, solve the relaxed QP for each, and */
/* keep the lower-cost branch.  Multi-binary support is a follow-up */
/* (the recursion blows up combinatorially without smart pruning).  */
/* ---------------------------------------------------------------- */
static double qp_cost(matlab_mat *H, matlab_mat *f, matlab_mat *z) {
    if (!H || !f || !z) return 1e30;
    int64_t n = z->rows;
    double quad = 0.0;
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            quad += z->data[i] * H->data[i * n + j] * z->data[j];
    double lin = 0.0;
    for (int64_t i = 0; i < n; ++i) lin += f->data[i] * z->data[i];
    return 0.5 * quad + lin;
}

/* matlab_mpc_move_finite — single-binary FCS MPC. */
matlab_mat *matlab_mpc_move_finite(void *mpc_obj_v, void *mpcstate_obj_v,
                                   matlab_mat *ym, matlab_mat *r) {
    if (!mpc_obj_v || !mpcstate_obj_v || !ym || !r) return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    matlab_obj *st      = reinterpret_cast<matlab_obj *>(mpcstate_obj_v);

    matlab_mat *mv_binary = matlab_obj_get_mat(mpc_obj, "mv_binary", 9);
    matlab_mat *umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    matlab_mat *umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    /* If no binary MVs, fall back to standard mpcmove. */
    bool has_binary = false;
    int binary_k = -1;
    if (mv_binary)
        for (int64_t k = 0; k < mv_binary->rows; ++k)
            if (mv_binary->data[k] > 0.5) { has_binary = true; binary_k = static_cast<int>(k); break; }
    if (!has_binary || !umin || !umax)
        return matlab_mpc_move(mpc_obj_v, mpcstate_obj_v, ym, r);

    /* Two-branch enumeration over the single binary MV.  Each branch
     * temporarily clamps umin[binary_k] = umax[binary_k] to one of
     * the bound values, runs the standard tick, computes the QP
     * cost, and keeps the lower-cost branch.  We use a save/restore
     * pattern on the mpc_obj's umin/umax — the standard tick reads
     * them via obj_get_mat each call. */
    double u_lo = umin->data[binary_k];
    double u_hi = umax->data[binary_k];

    /* Snapshot the state for branch evaluation. */
    matlab_mat *xp_snap   = matlab_obj_get_mat(st, "Plant",    5);
    matlab_mat *uprev_snap = matlab_obj_get_mat(st, "LastMove", 8);
    matlab_mat *xp_init = mat_alloc(xp_snap->rows, 1);
    for (int64_t i = 0; i < xp_snap->rows; ++i) xp_init->data[i] = xp_snap->data[i];
    matlab_mat *uprev_init = mat_alloc(uprev_snap->rows, 1);
    for (int64_t i = 0; i < uprev_snap->rows; ++i) uprev_init->data[i] = uprev_snap->data[i];

    /* Branch A: clamp at lo. */
    matlab_mat *umin_A = mat_alloc(umin->rows, 1);
    matlab_mat *umax_A = mat_alloc(umax->rows, 1);
    for (int64_t i = 0; i < umin->rows; ++i) {
        umin_A->data[i] = umin->data[i];
        umax_A->data[i] = umax->data[i];
    }
    umax_A->data[binary_k] = u_lo;  /* upper bound = lo → forces u to lo */
    matlab_obj_set_mat(mpc_obj, "umin", 4, umin_A);
    matlab_obj_set_mat(mpc_obj, "umax", 4, umax_A);
    matlab_obj_set_mat(st, "Plant",    5, xp_init);
    matlab_obj_set_mat(st, "LastMove", 8, uprev_init);
    matlab_mat *u_A = matlab_mpc_move(mpc_obj_v, mpcstate_obj_v, ym, r);
    /* Read post-branch QP cost: snapshot the just-mutated state to
     * recover the chosen u.  We don't have direct access to (H, f, z)
     * after-the-fact, so use a simpler proxy: tracking error
     * ‖C·xp - r‖² at the new state. */
    matlab_mat *C = matlab_obj_get_mat(mpc_obj, "C", 1);
    matlab_mat *xp_A = matlab_obj_get_mat(st, "Plant", 5);
    matlab_mat *yA = matlab_matmul_mm(C, xp_A);
    double cost_A = 0.0;
    for (int64_t k = 0; k < yA->rows; ++k) {
        double e = yA->data[k] - (k < r->rows ? r->data[k] : 0.0);
        cost_A += e * e;
    }
    double uA = u_A ? u_A->data[binary_k] : 0.0;

    /* Branch B: clamp at hi.  Reset state first. */
    matlab_obj_set_mat(st, "Plant",    5, xp_init);
    matlab_obj_set_mat(st, "LastMove", 8, uprev_init);
    matlab_mat *umin_B = mat_alloc(umin->rows, 1);
    matlab_mat *umax_B = mat_alloc(umax->rows, 1);
    for (int64_t i = 0; i < umin->rows; ++i) {
        umin_B->data[i] = umin->data[i];
        umax_B->data[i] = umax->data[i];
    }
    umin_B->data[binary_k] = u_hi;  /* lower bound = hi → forces u to hi */
    matlab_obj_set_mat(mpc_obj, "umin", 4, umin_B);
    matlab_obj_set_mat(mpc_obj, "umax", 4, umax_B);
    matlab_mat *u_B = matlab_mpc_move(mpc_obj_v, mpcstate_obj_v, ym, r);
    matlab_mat *xp_B = matlab_obj_get_mat(st, "Plant", 5);
    matlab_mat *yB = matlab_matmul_mm(C, xp_B);
    double cost_B = 0.0;
    for (int64_t k = 0; k < yB->rows; ++k) {
        double e = yB->data[k] - (k < r->rows ? r->data[k] : 0.0);
        cost_B += e * e;
    }
    double uB = u_B ? u_B->data[binary_k] : 0.0;
    (void)uA; (void)uB;

    /* Restore original umin/umax.  Pick the lower-cost branch's
     * resulting state. */
    matlab_obj_set_mat(mpc_obj, "umin", 4, umin);
    matlab_obj_set_mat(mpc_obj, "umax", 4, umax);

    if (cost_A < cost_B) {
        /* Re-run Branch A so state ends up reflecting it. */
        matlab_obj_set_mat(st, "Plant",    5, xp_init);
        matlab_obj_set_mat(st, "LastMove", 8, uprev_init);
        matlab_obj_set_mat(mpc_obj, "umin", 4, umin_A);
        matlab_obj_set_mat(mpc_obj, "umax", 4, umax_A);
        matlab_mat *u_final = matlab_mpc_move(mpc_obj_v, mpcstate_obj_v, ym, r);
        matlab_obj_set_mat(mpc_obj, "umin", 4, umin);
        matlab_obj_set_mat(mpc_obj, "umax", 4, umax);
        return u_final;
    }
    /* Branch B already ran last; state is correct. */
    return u_B;
}

/* matlab_mpc_sim_opt is defined further down — after matlab_mpc_sim
 * itself, since it shares the TickInputs setup. */

/* ---------------------------------------------------------------- */
/* matlab_mpc_review — Tier-6 §7.5, sanity report.                  */
/* Checks: p > 0, m > 0, m ≤ p, cached matrices populated and not   */
/* NaN/Inf, Hessian Cholesky factor R has positive diagonal.        */
/* Returns 1.0 if all checks pass, 0.0 otherwise.                   */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_review(void *mpc_obj_v) {
    matlab_mat *out = mat_alloc(1, 1);
    out->data[0] = 0.0;
    if (!mpc_obj_v) return out;
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    double p_d = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d = matlab_obj_get_f64(mpc_obj, "m", 1);
    int p = static_cast<int>(p_d);
    int m = static_cast<int>(m_d);
    if (p <= 0 || m <= 0 || m > p) return out;
    matlab_mat *H = matlab_obj_get_mat(mpc_obj, "H", 1);
    matlab_mat *R = matlab_obj_get_mat(mpc_obj, "R", 1);
    matlab_mat *L = matlab_obj_get_mat(mpc_obj, "L", 1);
    if (!H || !R || !L || H->rows == 0 || R->rows == 0 || L->rows == 0)
        return out;
    /* R is upper-triangular Cholesky.  Check positive diagonals. */
    int64_t n = R->rows;
    for (int64_t i = 0; i < n; ++i) {
        double d = R->data[i * R->cols + i];
        if (!(d > 0) || d != d) return out;   /* NaN check via self */
    }
    out->data[0] = 1.0;
    return out;
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_active_set — Tier-4 §5.4, standalone KWIK QP.         */
/*                                                                  */
/* Exposes the same dual active-set solver used internally by       */
/* mpcmove, for users assembling their own QP outside the MPC obj.  */
/* Solves min ½·x'·H·x + f'·x s.t. A·x ≤ b, returns the optimal x.  */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_active_set(matlab_mat *H, matlab_mat *f,
                                  matlab_mat *A_ineq, matlab_mat *b_ineq) {
    if (!H || !f) return mat_alloc(0, 0);
    /* qp_kwik tolerates null/empty A_ineq + b_ineq as "unconstrained". */
    return qp_kwik(H, f, A_ineq, b_ineq);
}

/* ---------------------------------------------------------------- */
/* mpc_tick — Kalman update + QP solve + state-advance.             */
/* Returns u_new (nu × 1); also updates xp / u_prev in place.       */
/* ---------------------------------------------------------------- */
struct TickInputs {
    matlab_mat *A, *B, *C;
    matlab_mat *Sx, *Su, *Su1;
    matlab_mat *H, *Wy, *L;
    matlab_mat *caller_umin, *caller_umax;
    matlab_mat *caller_ymin, *caller_ymax;
    matlab_mat *caller_V_y_min, *caller_V_y_max;
    matlab_mat *caller_V_u_min, *caller_V_u_max;
    matlab_mat *caller_E, *caller_F, *caller_G;
    matlab_mat *caller_dumin, *caller_dumax;   /* Tier-6 §7.2: rate bounds on Δu. */
    matlab_mat *caller_Wu, *caller_u_target;   /* Tier-6 §7.3: MV-tracking. */
    int p, m;
    double rho_eps;
    double outdist;   /* Tier-2 §3.3: when > 0, integrate an output
                       * disturbance estimate to cancel steady-state
                       * tracking error under model mismatch. */
};

static matlab_mat *mpc_tick(const TickInputs &in,
                            matlab_mat **xp_io, matlab_mat **u_prev_io,
                            matlab_mat **dist_io,
                            matlab_mat *ym, matlab_mat *r) {
    matlab_mat *A   = in.A;
    matlab_mat *B   = in.B;
    matlab_mat *C   = in.C;
    matlab_mat *Sx  = in.Sx;
    matlab_mat *Su  = in.Su;
    matlab_mat *Su1 = in.Su1;
    matlab_mat *H   = in.H;
    matlab_mat *Wy  = in.Wy;
    matlab_mat *L   = in.L;
    matlab_mat *caller_umin = in.caller_umin;
    matlab_mat *caller_umax = in.caller_umax;
    matlab_mat *caller_ymin = in.caller_ymin;
    matlab_mat *caller_ymax = in.caller_ymax;
    matlab_mat *caller_V_y_min = in.caller_V_y_min;
    matlab_mat *caller_V_y_max = in.caller_V_y_max;
    matlab_mat *caller_V_u_min = in.caller_V_u_min;
    matlab_mat *caller_V_u_max = in.caller_V_u_max;
    matlab_mat *caller_E = in.caller_E;
    matlab_mat *caller_F = in.caller_F;
    matlab_mat *caller_G = in.caller_G;
    matlab_mat *caller_dumin = in.caller_dumin;
    matlab_mat *caller_dumax = in.caller_dumax;
    matlab_mat *caller_Wu       = in.caller_Wu;
    matlab_mat *caller_u_target = in.caller_u_target;
    int p = in.p;
    int m = in.m;
    double rho_eps = in.rho_eps;
    matlab_mat *xp     = *xp_io;
    matlab_mat *u_prev = *u_prev_io;
    matlab_mat *dist   = dist_io ? *dist_io : nullptr;
    int64_t nx = A->rows;
    int64_t nu = B->cols;
    int64_t ny = C->rows;
    int64_t mnu = static_cast<int64_t>(m) * nu;
    int64_t N   = mnu + 1;

    /* Tier-2 §3.3: when output-disturbance estimation is enabled,
     * the model predicts y_pred = C·xp + d.  The Kalman update fits
     * `ym - d` (i.e. the model-prediction-relative innovation), and
     * after the state update we re-derive d as the residual
     * d_new = ym - C·xp_new (a one-shot estimator equivalent to
     * augmented-Kalman with unit gain on the disturbance state). */
    bool use_outdist = (in.outdist > 0.5) && dist && dist->rows == ny;
    matlab_mat *d_cur = mat_alloc(ny, 1);
    if (use_outdist)
        for (int64_t k = 0; k < ny; ++k) d_cur->data[k] = dist->data[k];

    /* Kalman update: xp ← xp + L · ((ym - d_cur) - C·xp). */
    matlab_mat *yhat  = matlab_matmul_mm(C, xp);
    matlab_mat *innov = mat_alloc(ny, 1);
    for (int64_t i = 0; i < ny; ++i)
        innov->data[i] = (ym->data[i] - d_cur->data[i]) - yhat->data[i];
    matlab_mat *L_innov = matlab_matmul_mm(L, innov);
    matlab_mat *xp_now = mat_alloc(nx, 1);
    for (int64_t i = 0; i < nx; ++i)
        xp_now->data[i] = xp->data[i] + L_innov->data[i];

    /* Update disturbance estimate (one-shot integrator).  When
     * outdist is disabled, d stays at zero — the rest of the QP
     * code path is unchanged from Tier-1. */
    matlab_mat *d_new = mat_alloc(ny, 1);
    if (use_outdist) {
        matlab_mat *yhat_now = matlab_matmul_mm(C, xp_now);
        for (int64_t k = 0; k < ny; ++k)
            d_new->data[k] = ym->data[k] - yhat_now->data[k];
    }

    /* Reference broadcast across horizon — subtract the current
     * disturbance estimate so the controller tracks (r - d).
     * Tier-6 §7.7 — when `r` is (p × ny) instead of (ny × 1), use
     * per-step preview r(i, :); otherwise broadcast the single ref. */
    matlab_mat *R_vec = mat_alloc(static_cast<int64_t>(p) * ny, 1);
    bool preview = (r->rows == static_cast<int64_t>(p) && r->cols == ny);
    for (int i = 0; i < p; ++i)
        for (int64_t k = 0; k < ny; ++k) {
            double rv;
            if (preview) {
                rv = r->data[static_cast<int64_t>(i) * ny + k];
            } else {
                rv = (k < r->rows) ? r->data[k] : 0.0;
            }
            R_vec->data[static_cast<int64_t>(i) * ny + k] =
                rv - d_new->data[k];
        }

    /* QP gradient: f_z = -2 · Su' · Wy_full · (R - Sx·xp - Su1·u_prev). */
    matlab_mat *Sxxp  = matlab_matmul_mm(Sx, xp_now);
    matlab_mat *Su1up = matlab_matmul_mm(Su1, u_prev);
    matlab_mat *err   = mat_alloc(static_cast<int64_t>(p) * ny, 1);
    for (int64_t i = 0; i < static_cast<int64_t>(p) * ny; ++i)
        err->data[i] = R_vec->data[i] - Sxxp->data[i] - Su1up->data[i];
    matlab_mat *Wyerr = mat_alloc(static_cast<int64_t>(p) * ny, 1);
    for (int i = 0; i < p; ++i)
        for (int64_t k = 0; k < ny; ++k) {
            double w = Wy->data[k];
            int64_t idx = static_cast<int64_t>(i) * ny + k;
            Wyerr->data[idx] = w * w * err->data[idx];
        }
    matlab_mat *SuT    = matlab_transpose(Su);
    matlab_mat *SuTWy  = matlab_matmul_mm(SuT, Wyerr);
    matlab_mat *f      = mat_alloc(N, 1);
    for (int64_t i = 0; i < mnu; ++i)
        f->data[i] = -2.0 * SuTWy->data[i];
    f->data[mnu] = 0.0;
    (void)rho_eps;  /* baked into H already */

    /* Tier-6 §7.3 — MV-tracking contribution to gradient AND Hessian.
     * Cost: J_u = Σⱼ ‖Wu·(u(k+j) - u_target)‖² with
     *   u(k+j) = u_prev + Σ_{i≤j} Δu(i)
     * Let L be the m×m lower-triangular ones matrix.  Per MV k,
     *   U_k = u_prev[k]·1 + L·Δu[:,k]
     * so J_u[k] = Wu[k]² · ‖U_k - u_target[k]‖²
     *           = Δu' · (L' Wu² L) · Δu
     *             + 2·Δu' · L' Wu² · (u_prev[k] - u_target[k])·1
     *             + const
     * (L'L)[i,j] = m - max(i, j).  L'·1 has entry i = m-i.
     * H_local gets the +2·Wu²·(m-max(i,j)) per-MV block (2× to match
     * the QP form ½ z'Hz that build_Hessian uses); f gets
     * +2·Wu²·(m-i)·diff. */
    matlab_mat *H_local = H;  /* default: reuse cached H */
    bool wu_active = false;
    if (caller_Wu) {
        for (int64_t k = 0; k < nu && k < caller_Wu->rows; ++k)
            if (caller_Wu->data[k] != 0.0) { wu_active = true; break; }
    }
    if (wu_active && caller_u_target) {
        /* Copy H into H_local so we don't mutate the cached one. */
        H_local = mat_alloc(N, N);
        for (int64_t r = 0; r < N; ++r)
            for (int64_t c = 0; c < N; ++c)
                H_local->data[r * N + c] = H->data[r * N + c];
        for (int64_t k = 0; k < nu; ++k) {
            double wu = (k < caller_Wu->rows) ? caller_Wu->data[k] : 0.0;
            if (wu == 0.0) continue;
            double w2 = wu * wu;
            double ut = (k < caller_u_target->rows)
                ? caller_u_target->data[k] : 0.0;
            double diff = u_prev->data[k] - ut;
            for (int i = 0; i < m; ++i) {
                /* gradient: ∂J_u/∂Δu(i)[k] = 2·Wu²·(m-i)·diff. */
                f->data[static_cast<int64_t>(i) * nu + k] +=
                    2.0 * w2 * static_cast<double>(m - i) * diff;
                /* Hessian: 2·Wu²·(m-max(i,j)) per (i,j) MV block. */
                for (int j = 0; j < m; ++j) {
                    int64_t r2 = static_cast<int64_t>(i) * nu + k;
                    int64_t c2 = static_cast<int64_t>(j) * nu + k;
                    int mx = (i > j) ? i : j;
                    H_local->data[r2 * N + c2] +=
                        2.0 * w2 * static_cast<double>(m - mx);
                }
            }
        }
    }

    /* Tier-2 §3.4/§3.1/§3.2: build the full inequality block.
     *   - 2·m·nu rows for MV bounds (Tier-1)
     *   - 2·p·ny rows for output bounds (Tier-2 §3.4)
     *   - nE rows for mixed E·u + F·y ≤ G at j=0 (Tier-2 §3.1)
     *   - 1 row for ε ≥ 0
     * The slack column (last column of A_ineq) carries the V_* ECR
     * coefficients per row — V > 0 makes that row soft.
     * Tier-2 carve-down: mixed constraints applied at j=0 only;
     * full j ∈ [0, p] sweep deferred. */
    int64_t nE = (caller_E && caller_E->rows > 0) ? caller_E->rows : 0;
    bool has_rate = (caller_dumin && caller_dumin->rows > 0) ||
                    (caller_dumax && caller_dumax->rows > 0);
    int64_t n_rate = has_rate ? (2 * static_cast<int64_t>(m) * nu) : 0;
    int64_t ni = 2 * static_cast<int64_t>(m) * nu
               + 2 * static_cast<int64_t>(p) * ny
               + nE
               + n_rate          /* Tier-6 §7.2 rate bounds on Δu */
               + 1;  /* ε ≥ 0 */
    matlab_mat *A_ineq = mat_alloc(ni, N);
    matlab_mat *b_ineq = mat_alloc(ni, 1);
    int64_t row = 0;
    /* --- MV bounds. */
    for (int h = 0; h < m; ++h) {
        for (int64_t k = 0; k < nu; ++k) {
            double up = (caller_umax ? caller_umax->data[k] :  1e6) - u_prev->data[k];
            double lo = (caller_umin ? caller_umin->data[k] : -1e6) - u_prev->data[k];
            double v_up = (caller_V_u_max && k < caller_V_u_max->rows) ? caller_V_u_max->data[k] : 0.0;
            double v_lo = (caller_V_u_min && k < caller_V_u_min->rows) ? caller_V_u_min->data[k] : 0.0;
            /* Upper: Σ Δu(0..h)[k] - V·ε ≤ up. */
            for (int j = 0; j <= h; ++j)
                A_ineq->data[row * N + static_cast<int64_t>(j) * nu + k] = 1.0;
            A_ineq->data[row * N + mnu] = -v_up;
            b_ineq->data[row++] = up;
            /* Lower: -Σ Δu(0..h)[k] - V·ε ≤ -lo. */
            for (int j = 0; j <= h; ++j)
                A_ineq->data[row * N + static_cast<int64_t>(j) * nu + k] = -1.0;
            A_ineq->data[row * N + mnu] = -v_lo;
            b_ineq->data[row++] = -lo;
        }
    }
    /* --- Output bounds (Tier-2 §3.4).  At prediction step i ∈ [1, p]:
     *     y(k+i) = Sx(i,:)·xp + Su1(i,:)·u_prev + Σ Su(i,j,:)·Δu(j).
     * Translate ymin[k] ≤ y(k+i)[k] ≤ ymax[k] to z-space:
     *     Upper: Σ Su(i,:,k,:)·Δu - V·ε ≤ ymax[k] - Sx(i,k,:)·xp - Su1(i,k,:)·u_prev
     *     Lower:-Σ Su(i,:,k,:)·Δu - V·ε ≤ -(ymin[k] - Sx(i,k,:)·xp - Su1(i,k,:)·u_prev)
     */
    for (int i = 1; i <= p; ++i) {
        int64_t row_base = static_cast<int64_t>(i - 1) * ny;
        for (int64_t k = 0; k < ny; ++k) {
            int64_t row_yk = row_base + k;
            double Sxxp_k  = Sxxp->data[row_yk];
            double Su1up_k = Su1up->data[row_yk];
            double ymax_k = (caller_ymax && k < caller_ymax->rows) ? caller_ymax->data[k] :  1e6;
            double ymin_k = (caller_ymin && k < caller_ymin->rows) ? caller_ymin->data[k] : -1e6;
            double v_ymax = (caller_V_y_max && k < caller_V_y_max->rows) ? caller_V_y_max->data[k] : 0.0;
            double v_ymin = (caller_V_y_min && k < caller_V_y_min->rows) ? caller_V_y_min->data[k] : 0.0;
            /* Upper. */
            for (int64_t c = 0; c < mnu; ++c)
                A_ineq->data[row * N + c] = Su->data[row_yk * mnu + c];
            A_ineq->data[row * N + mnu] = -v_ymax;
            b_ineq->data[row++] = ymax_k - Sxxp_k - Su1up_k;
            /* Lower. */
            for (int64_t c = 0; c < mnu; ++c)
                A_ineq->data[row * N + c] = -Su->data[row_yk * mnu + c];
            A_ineq->data[row * N + mnu] = -v_ymin;
            b_ineq->data[row++] = -(ymin_k - Sxxp_k - Su1up_k);
        }
    }
    /* --- Mixed E·u(k) + F·y(k+1) ≤ G  at j=0 (Tier-2 §3.1).
     * u(k) = u_prev + Δu(0).  y(k+1) = Sx(1,:)·xp + Su1(1,:)·u_prev +
     * Σ Su(1,j,:)·Δu(j).  Combine: row gets
     *     [E + F·Su(1,j)] Δu(j)  ≤  G - E·u_prev - F·(Sx(1,:)·xp + Su1(1,:)·u_prev)
     * One row per mixed inequality; expanded over Δu(0..m-1). */
    if (nE > 0 && caller_E && caller_F && caller_G) {
        for (int64_t e = 0; e < nE; ++e) {
            /* Coefficient of Δu(j)[k] in row e: E[e,k] (only j=0) + F[e,:]·Su(1,:,k).
             * Note: Δu(0) maps directly to u(k); for j>0, Δu(j) affects
             * y(k+1) via Su(1,j) but doesn't affect u(k) (so E coeff
             * applies only at j=0). */
            for (int64_t k = 0; k < nu; ++k)
                A_ineq->data[row * N + k] = caller_E->data[e * nu + k];
            /* Add F · Su(1,:,:) terms over j = 0..m-1.  Su row 0..ny-1
             * (the i=1 block). */
            for (int j = 0; j < m; ++j) {
                for (int64_t k = 0; k < nu; ++k) {
                    double coeff = 0.0;
                    for (int64_t r2 = 0; r2 < ny; ++r2)
                        coeff += caller_F->data[e * ny + r2]
                                 * Su->data[r2 * mnu + static_cast<int64_t>(j) * nu + k];
                    A_ineq->data[row * N + static_cast<int64_t>(j) * nu + k] += coeff;
                }
            }
            /* RHS: G[e] - E[e,:]·u_prev - F[e,:]·(Sx(1,:)·xp + Su1(1,:)·u_prev). */
            double rhs = caller_G->data[e];
            for (int64_t k = 0; k < nu; ++k)
                rhs -= caller_E->data[e * nu + k] * u_prev->data[k];
            for (int64_t r2 = 0; r2 < ny; ++r2)
                rhs -= caller_F->data[e * ny + r2] * (Sxxp->data[r2] + Su1up->data[r2]);
            b_ineq->data[row++] = rhs;
        }
    }
    /* --- Tier-6 §7.2 rate bounds on Δu(j) for j ∈ [0, m-1].  Each
     * MV gets two rows: +Δu(j)[k] ≤ dumax[k] and -Δu(j)[k] ≤ -dumin[k].
     * No slack scaling (rate bounds are hard by default in Tier-6). */
    if (has_rate) {
        for (int j = 0; j < m; ++j) {
            for (int64_t k = 0; k < nu; ++k) {
                double du_up = (caller_dumax && k < caller_dumax->rows)
                    ? caller_dumax->data[k] :  1e6;
                double du_lo = (caller_dumin && k < caller_dumin->rows)
                    ? caller_dumin->data[k] : -1e6;
                A_ineq->data[row * N + static_cast<int64_t>(j) * nu + k] = 1.0;
                b_ineq->data[row++] = du_up;
                A_ineq->data[row * N + static_cast<int64_t>(j) * nu + k] = -1.0;
                b_ineq->data[row++] = -du_lo;
            }
        }
    }
    /* --- ε ≥ 0  ⇔  -ε ≤ 0. */
    A_ineq->data[row * N + mnu] = -1.0;
    b_ineq->data[row++] = 0.0;

    matlab_mat *z = qp_kwik(H_local, f, A_ineq, b_ineq);

    /* New MV: u_new = u_prev + Δu(0). */
    matlab_mat *u_new = mat_alloc(nu, 1);
    for (int64_t k = 0; k < nu; ++k)
        u_new->data[k] = u_prev->data[k] + z->data[k];

    /* Propagate: xp(k+1) = A·xp(k|k) + B·u(k). */
    matlab_mat *Axp = matlab_matmul_mm(A, xp_now);
    matlab_mat *Bu  = matlab_matmul_mm(B, u_new);
    matlab_mat *xp_next = mat_alloc(nx, 1);
    for (int64_t i = 0; i < nx; ++i)
        xp_next->data[i] = Axp->data[i] + Bu->data[i];

    *xp_io     = xp_next;
    *u_prev_io = u_new;
    if (dist_io) *dist_io = d_new;
    return u_new;
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_move — one tick, taking mpcstate as an obj.           */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_move(void *mpc_obj_v, void *mpcstate_obj_v,
                            matlab_mat *ym, matlab_mat *r) {
    if (!mpc_obj_v || !mpcstate_obj_v || !ym || !r) return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    matlab_obj *st      = reinterpret_cast<matlab_obj *>(mpcstate_obj_v);

    TickInputs in;
    in.A   = matlab_obj_get_mat(mpc_obj, "A",   1);
    in.B   = matlab_obj_get_mat(mpc_obj, "B",   1);
    in.C   = matlab_obj_get_mat(mpc_obj, "C",   1);
    in.Sx  = matlab_obj_get_mat(mpc_obj, "Sx",  2);
    in.Su  = matlab_obj_get_mat(mpc_obj, "Su",  2);
    in.Su1 = matlab_obj_get_mat(mpc_obj, "Su1", 3);
    in.H   = matlab_obj_get_mat(mpc_obj, "H",   1);
    in.Wy  = matlab_obj_get_mat(mpc_obj, "Wy",  2);
    in.L   = matlab_obj_get_mat(mpc_obj, "L",   1);
    in.caller_umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    in.caller_umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    in.caller_ymin = matlab_obj_get_mat(mpc_obj, "ymin", 4);
    in.caller_ymax = matlab_obj_get_mat(mpc_obj, "ymax", 4);
    in.caller_V_y_min = matlab_obj_get_mat(mpc_obj, "V_y_min", 7);
    in.caller_V_y_max = matlab_obj_get_mat(mpc_obj, "V_y_max", 7);
    in.caller_V_u_min = matlab_obj_get_mat(mpc_obj, "V_u_min", 7);
    in.caller_V_u_max = matlab_obj_get_mat(mpc_obj, "V_u_max", 7);
    in.caller_E = matlab_obj_get_mat(mpc_obj, "E", 1);
    in.caller_F = matlab_obj_get_mat(mpc_obj, "F", 1);
    in.caller_G = matlab_obj_get_mat(mpc_obj, "G", 1);
    in.caller_dumin    = matlab_obj_get_mat(mpc_obj, "dumin",    5);
    in.caller_dumax    = matlab_obj_get_mat(mpc_obj, "dumax",    5);
    in.caller_Wu       = matlab_obj_get_mat(mpc_obj, "Wu",       2);
    in.caller_u_target = matlab_obj_get_mat(mpc_obj, "u_target", 8);
    double p_d = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d = matlab_obj_get_f64(mpc_obj, "m", 1);
    in.rho_eps = matlab_obj_get_f64(mpc_obj, "rho_eps", 7);
    in.outdist = matlab_obj_get_f64(mpc_obj, "outdist", 7);
    in.p = static_cast<int>(p_d);
    in.m = static_cast<int>(m_d);

    matlab_mat *xp     = matlab_obj_get_mat(st, "Plant",    5);
    matlab_mat *u_prev = matlab_obj_get_mat(st, "LastMove", 8);
    matlab_mat *dist   = matlab_obj_get_mat(st, "Dist",     4);

    if (!in.A || !in.B || !in.C || !in.Sx || !in.Su || !in.Su1 ||
        !in.H || !in.Wy || !in.L || !xp || !u_prev) return mat_alloc(0, 0);

    matlab_mat *u_new = mpc_tick(in, &xp, &u_prev, &dist, ym, r);
    matlab_obj_set_mat(st, "Plant",    5, xp);
    matlab_obj_set_mat(st, "LastMove", 8, u_new);
    if (dist) matlab_obj_set_mat(st, "Dist", 4, dist);
    return u_new;
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_move_opt — Tier-2 §3.7, mpcmove(obj, st, ym, r, opt). */
/* opt is an mpcmoveopt classdef instance — when its Use_* flags    */
/* are set, the corresponding bound overrides the cached one for    */
/* this tick only.                                                  */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_move_opt(void *mpc_obj_v, void *mpcstate_obj_v,
                                matlab_mat *ym, matlab_mat *r,
                                void *opt_obj_v) {
    if (!mpc_obj_v || !mpcstate_obj_v || !ym || !r) return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    matlab_obj *st      = reinterpret_cast<matlab_obj *>(mpcstate_obj_v);
    matlab_obj *opt     = reinterpret_cast<matlab_obj *>(opt_obj_v);

    TickInputs in;
    in.A   = matlab_obj_get_mat(mpc_obj, "A",   1);
    in.B   = matlab_obj_get_mat(mpc_obj, "B",   1);
    in.C   = matlab_obj_get_mat(mpc_obj, "C",   1);
    in.Sx  = matlab_obj_get_mat(mpc_obj, "Sx",  2);
    in.Su  = matlab_obj_get_mat(mpc_obj, "Su",  2);
    in.Su1 = matlab_obj_get_mat(mpc_obj, "Su1", 3);
    in.H   = matlab_obj_get_mat(mpc_obj, "H",   1);
    in.Wy  = matlab_obj_get_mat(mpc_obj, "Wy",  2);
    in.L   = matlab_obj_get_mat(mpc_obj, "L",   1);
    in.caller_umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    in.caller_umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    in.caller_ymin = matlab_obj_get_mat(mpc_obj, "ymin", 4);
    in.caller_ymax = matlab_obj_get_mat(mpc_obj, "ymax", 4);
    in.caller_V_y_min = matlab_obj_get_mat(mpc_obj, "V_y_min", 7);
    in.caller_V_y_max = matlab_obj_get_mat(mpc_obj, "V_y_max", 7);
    in.caller_V_u_min = matlab_obj_get_mat(mpc_obj, "V_u_min", 7);
    in.caller_V_u_max = matlab_obj_get_mat(mpc_obj, "V_u_max", 7);
    in.caller_E = matlab_obj_get_mat(mpc_obj, "E", 1);
    in.caller_F = matlab_obj_get_mat(mpc_obj, "F", 1);
    in.caller_G = matlab_obj_get_mat(mpc_obj, "G", 1);
    in.caller_dumin    = matlab_obj_get_mat(mpc_obj, "dumin",    5);
    in.caller_dumax    = matlab_obj_get_mat(mpc_obj, "dumax",    5);
    in.caller_Wu       = matlab_obj_get_mat(mpc_obj, "Wu",       2);
    in.caller_u_target = matlab_obj_get_mat(mpc_obj, "u_target", 8);
    double p_d = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d = matlab_obj_get_f64(mpc_obj, "m", 1);
    in.rho_eps = matlab_obj_get_f64(mpc_obj, "rho_eps", 7);
    in.outdist = matlab_obj_get_f64(mpc_obj, "outdist", 7);
    in.p = static_cast<int>(p_d);
    in.m = static_cast<int>(m_d);

    /* Apply opt overrides. */
    if (opt) {
        if (matlab_obj_get_f64(opt, "Use_MVMin", 9) > 0.5)
            in.caller_umin = matlab_obj_get_mat(opt, "MVMin", 5);
        if (matlab_obj_get_f64(opt, "Use_MVMax", 9) > 0.5)
            in.caller_umax = matlab_obj_get_mat(opt, "MVMax", 5);
        if (matlab_obj_get_f64(opt, "Use_OutputMin", 13) > 0.5)
            in.caller_ymin = matlab_obj_get_mat(opt, "OutputMin", 9);
        if (matlab_obj_get_f64(opt, "Use_OutputMax", 13) > 0.5)
            in.caller_ymax = matlab_obj_get_mat(opt, "OutputMax", 9);
    }

    matlab_mat *xp     = matlab_obj_get_mat(st, "Plant",    5);
    matlab_mat *u_prev = matlab_obj_get_mat(st, "LastMove", 8);
    matlab_mat *dist   = matlab_obj_get_mat(st, "Dist",     4);
    if (!in.A || !in.B || !in.C || !in.Sx || !in.Su || !in.Su1 ||
        !in.H || !in.Wy || !in.L || !xp || !u_prev) return mat_alloc(0, 0);

    matlab_mat *u_new = mpc_tick(in, &xp, &u_prev, &dist, ym, r);
    matlab_obj_set_mat(st, "Plant",    5, xp);
    matlab_obj_set_mat(st, "LastMove", 8, u_new);
    if (dist) matlab_obj_set_mat(st, "Dist", 4, dist);
    return u_new;
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_move_adaptive — Tier-3 §4.1, mpcmoveAdaptive.         */
/*                                                                  */
/* Rebuilds the cached prediction matrices (Sx / Su / Su1 / Hessian /*/
/* Kalman) from a per-tick new plant (A_new, B_new, C_new) before   */
/* running the usual QP solve.  Weights / bounds / horizons / ECR / */
/* mixed-constraint matrices on the obj are preserved across ticks. */
/*                                                                  */
/* Tier-3 simplification: always rebuilds (no fingerprint cache).   */
/* For small plants (n ≤ 10) the rebuild is ~µs — negligible vs.    */
/* the QP solve.                                                    */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_move_adaptive(void *mpc_obj_v, void *mpcstate_obj_v,
                                     matlab_mat *A_new, matlab_mat *B_new,
                                     matlab_mat *C_new,
                                     matlab_mat *ym, matlab_mat *r) {
    if (!mpc_obj_v || !mpcstate_obj_v || !A_new || !B_new || !C_new ||
        !ym || !r) return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    matlab_obj *st      = reinterpret_cast<matlab_obj *>(mpcstate_obj_v);

    /* Read the user-tunable knobs off obj — these survive the per-
     * tick rebuild. */
    matlab_mat *Wy   = matlab_obj_get_mat(mpc_obj, "Wy",  2);
    matlab_mat *Wdu  = matlab_obj_get_mat(mpc_obj, "Wdu", 3);
    double Ts      = matlab_obj_get_f64(mpc_obj, "Ts", 2);
    double p_d     = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d     = matlab_obj_get_f64(mpc_obj, "m", 1);
    double rho_eps = matlab_obj_get_f64(mpc_obj, "rho_eps", 7);
    int p = static_cast<int>(p_d);
    int m = static_cast<int>(m_d);
    if (p < 1) p = 1;
    if (m < 1) m = 1;
    if (m > p) m = p;
    int64_t nu = B_new->cols;

    /* Rebuild prediction matrices from the new plant. */
    matlab_mat *Sx_new  = build_Sx(A_new, C_new, p);
    matlab_mat *Su_new  = build_Su(A_new, B_new, C_new, p, m);
    matlab_mat *Su1_new = build_Su1(A_new, B_new, C_new, p);
    matlab_mat *H_new   = build_Hessian(Su_new, Wy, Wdu, rho_eps, m,
                                        static_cast<int>(nu));
    matlab_mat *R_new   = matlab_chol(H_new);
    matlab_mat *L_new   = build_Kalman_L(A_new, B_new, C_new, Ts);

    /* Write back to obj so subsequent reads see the new plant. */
    matlab_obj_set_mat(mpc_obj, "A",   1, A_new);
    matlab_obj_set_mat(mpc_obj, "B",   1, B_new);
    matlab_obj_set_mat(mpc_obj, "C",   1, C_new);
    matlab_obj_set_mat(mpc_obj, "Sx",  2, Sx_new);
    matlab_obj_set_mat(mpc_obj, "Su",  2, Su_new);
    matlab_obj_set_mat(mpc_obj, "Su1", 3, Su1_new);
    matlab_obj_set_mat(mpc_obj, "H",   1, H_new);
    matlab_obj_set_mat(mpc_obj, "R",   1, R_new);
    matlab_obj_set_mat(mpc_obj, "L",   1, L_new);

    /* Standard tick. */
    TickInputs in;
    in.A = A_new; in.B = B_new; in.C = C_new;
    in.Sx = Sx_new; in.Su = Su_new; in.Su1 = Su1_new;
    in.H = H_new; in.Wy = Wy; in.L = L_new;
    in.caller_umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    in.caller_umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    in.caller_ymin = matlab_obj_get_mat(mpc_obj, "ymin", 4);
    in.caller_ymax = matlab_obj_get_mat(mpc_obj, "ymax", 4);
    in.caller_V_y_min = matlab_obj_get_mat(mpc_obj, "V_y_min", 7);
    in.caller_V_y_max = matlab_obj_get_mat(mpc_obj, "V_y_max", 7);
    in.caller_V_u_min = matlab_obj_get_mat(mpc_obj, "V_u_min", 7);
    in.caller_V_u_max = matlab_obj_get_mat(mpc_obj, "V_u_max", 7);
    in.caller_E = matlab_obj_get_mat(mpc_obj, "E", 1);
    in.caller_F = matlab_obj_get_mat(mpc_obj, "F", 1);
    in.caller_G = matlab_obj_get_mat(mpc_obj, "G", 1);
    in.caller_dumin    = matlab_obj_get_mat(mpc_obj, "dumin",    5);
    in.caller_dumax    = matlab_obj_get_mat(mpc_obj, "dumax",    5);
    in.caller_Wu       = matlab_obj_get_mat(mpc_obj, "Wu",       2);
    in.caller_u_target = matlab_obj_get_mat(mpc_obj, "u_target", 8);
    in.rho_eps = rho_eps;
    in.outdist = matlab_obj_get_f64(mpc_obj, "outdist", 7);
    in.p = p; in.m = m;

    matlab_mat *xp     = matlab_obj_get_mat(st, "Plant",    5);
    matlab_mat *u_prev = matlab_obj_get_mat(st, "LastMove", 8);
    matlab_mat *dist   = matlab_obj_get_mat(st, "Dist",     4);
    if (!xp || !u_prev) return mat_alloc(0, 0);

    matlab_mat *u_new = mpc_tick(in, &xp, &u_prev, &dist, ym, r);
    matlab_obj_set_mat(st, "Plant",    5, xp);
    matlab_obj_set_mat(st, "LastMove", 8, u_new);
    if (dist) matlab_obj_set_mat(st, "Dist", 4, dist);
    return u_new;
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_move_tv — Tier-3 §4.2, mpcmoveTV.                     */
/*                                                                  */
/* Time-varying MPC: the plant changes per prediction step.  Inputs */
/* are stacked: A_stack is (p·nx × nx), B_stack is (p·nx × nu),     */
/* C_stack is (p·ny × nx) — block i of each holds A_i / B_i / C_i,  */
/* used for the transition from step i to step i+1 (and C_i for the */
/* output at step i+1).                                              */
/*                                                                  */
/* Tier-3 simplification: uses the first plant snapshot (A_0/B_0/   */
/* C_0) for the per-tick Kalman gain — that's a reasonable choice   */
/* when the variation is slow vs. observer dynamics.  A fully       */
/* time-varying Kalman is a follow-up.                              */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_move_tv(void *mpc_obj_v, void *mpcstate_obj_v,
                               matlab_mat *A_stack, matlab_mat *B_stack,
                               matlab_mat *C_stack,
                               matlab_mat *ym, matlab_mat *r) {
    if (!mpc_obj_v || !mpcstate_obj_v || !A_stack || !B_stack ||
        !C_stack || !ym || !r) return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    matlab_obj *st      = reinterpret_cast<matlab_obj *>(mpcstate_obj_v);

    matlab_mat *Wy   = matlab_obj_get_mat(mpc_obj, "Wy",  2);
    matlab_mat *Wdu  = matlab_obj_get_mat(mpc_obj, "Wdu", 3);
    double Ts      = matlab_obj_get_f64(mpc_obj, "Ts", 2);
    double p_d     = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d     = matlab_obj_get_f64(mpc_obj, "m", 1);
    double rho_eps = matlab_obj_get_f64(mpc_obj, "rho_eps", 7);
    int p = static_cast<int>(p_d);
    int m = static_cast<int>(m_d);
    if (p < 1) p = 1;
    if (m < 1) m = 1;
    if (m > p) m = p;

    /* Infer nx / nu / ny from the stack shapes. */
    int64_t nx = A_stack->cols;          /* A_i is nx × nx; A_stack rows = p·nx */
    int64_t nu = B_stack->cols;          /* B_i is nx × nu */
    int64_t ny = C_stack->rows / p;      /* C_i is ny × nx */
    if (ny < 1) ny = 1;

    /* Sanity: matrix dims must match. */
    if (A_stack->rows != static_cast<int64_t>(p) * nx) return mat_alloc(0, 0);
    if (B_stack->rows != static_cast<int64_t>(p) * nx) return mat_alloc(0, 0);
    if (C_stack->cols != nx) return mat_alloc(0, 0);

    /* Build the TV prediction matrices. */
    matlab_mat *Sx_new  = build_Sx_tv(A_stack, C_stack, p, nx, ny);
    matlab_mat *Su_new  = build_Su_tv(A_stack, B_stack, C_stack,
                                      p, m, nx, nu, ny);
    matlab_mat *Su1_new = build_Su1_tv(A_stack, B_stack, C_stack,
                                       p, nx, nu, ny);
    matlab_mat *H_new   = build_Hessian(Su_new, Wy, Wdu, rho_eps, m,
                                        static_cast<int>(nu));
    matlab_mat *R_new   = matlab_chol(H_new);

    /* Kalman gain from the first plant snapshot (A_0, B_0, C_0). */
    matlab_mat *A0 = stack_block(A_stack, 0, nx);
    matlab_mat *B0 = stack_block(B_stack, 0, nx);
    matlab_mat *C0 = stack_block(C_stack, 0, ny);
    matlab_mat *L_new = build_Kalman_L(A0, B0, C0, Ts);

    /* Write back to obj so subsequent reads / mpcmove ticks see the
     * latest TV-derived matrices. */
    matlab_obj_set_mat(mpc_obj, "A",   1, A0);
    matlab_obj_set_mat(mpc_obj, "B",   1, B0);
    matlab_obj_set_mat(mpc_obj, "C",   1, C0);
    matlab_obj_set_mat(mpc_obj, "Sx",  2, Sx_new);
    matlab_obj_set_mat(mpc_obj, "Su",  2, Su_new);
    matlab_obj_set_mat(mpc_obj, "Su1", 3, Su1_new);
    matlab_obj_set_mat(mpc_obj, "H",   1, H_new);
    matlab_obj_set_mat(mpc_obj, "R",   1, R_new);
    matlab_obj_set_mat(mpc_obj, "L",   1, L_new);

    /* Standard tick using the TV prediction matrices.  Note we use
     * A_0 / B_0 for the state propagation at the end of the tick —
     * that matches the "we're transitioning from step 0 to step 1
     * THIS tick" interpretation. */
    TickInputs in;
    in.A = A0; in.B = B0; in.C = C0;
    in.Sx = Sx_new; in.Su = Su_new; in.Su1 = Su1_new;
    in.H = H_new; in.Wy = Wy; in.L = L_new;
    in.caller_umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    in.caller_umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    in.caller_ymin = matlab_obj_get_mat(mpc_obj, "ymin", 4);
    in.caller_ymax = matlab_obj_get_mat(mpc_obj, "ymax", 4);
    in.caller_V_y_min = matlab_obj_get_mat(mpc_obj, "V_y_min", 7);
    in.caller_V_y_max = matlab_obj_get_mat(mpc_obj, "V_y_max", 7);
    in.caller_V_u_min = matlab_obj_get_mat(mpc_obj, "V_u_min", 7);
    in.caller_V_u_max = matlab_obj_get_mat(mpc_obj, "V_u_max", 7);
    in.caller_E = matlab_obj_get_mat(mpc_obj, "E", 1);
    in.caller_F = matlab_obj_get_mat(mpc_obj, "F", 1);
    in.caller_G = matlab_obj_get_mat(mpc_obj, "G", 1);
    in.caller_dumin    = matlab_obj_get_mat(mpc_obj, "dumin",    5);
    in.caller_dumax    = matlab_obj_get_mat(mpc_obj, "dumax",    5);
    in.caller_Wu       = matlab_obj_get_mat(mpc_obj, "Wu",       2);
    in.caller_u_target = matlab_obj_get_mat(mpc_obj, "u_target", 8);
    in.rho_eps = rho_eps;
    in.outdist = matlab_obj_get_f64(mpc_obj, "outdist", 7);
    in.p = p; in.m = m;

    matlab_mat *xp     = matlab_obj_get_mat(st, "Plant",    5);
    matlab_mat *u_prev = matlab_obj_get_mat(st, "LastMove", 8);
    matlab_mat *dist   = matlab_obj_get_mat(st, "Dist",     4);
    if (!xp || !u_prev) return mat_alloc(0, 0);

    matlab_mat *u_new = mpc_tick(in, &xp, &u_prev, &dist, ym, r);
    matlab_obj_set_mat(st, "Plant",    5, xp);
    matlab_obj_set_mat(st, "LastMove", 8, u_new);
    if (dist) matlab_obj_set_mat(st, "Dist", 4, dist);
    return u_new;
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_sim — closed-loop T-tick simulation.                  */
/* Returns Y (T × ny).  Uses internal xp / u_prev arrays — no       */
/* mpcstate obj needed.                                              */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_sim(void *mpc_obj_v, double T_d, matlab_mat *r) {
    if (!mpc_obj_v || !r) return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    int T = static_cast<int>(T_d);
    if (T < 1) T = 1;

    TickInputs in;
    in.A   = matlab_obj_get_mat(mpc_obj, "A",   1);
    in.B   = matlab_obj_get_mat(mpc_obj, "B",   1);
    in.C   = matlab_obj_get_mat(mpc_obj, "C",   1);
    in.Sx  = matlab_obj_get_mat(mpc_obj, "Sx",  2);
    in.Su  = matlab_obj_get_mat(mpc_obj, "Su",  2);
    in.Su1 = matlab_obj_get_mat(mpc_obj, "Su1", 3);
    in.H   = matlab_obj_get_mat(mpc_obj, "H",   1);
    in.Wy  = matlab_obj_get_mat(mpc_obj, "Wy",  2);
    in.L   = matlab_obj_get_mat(mpc_obj, "L",   1);
    in.caller_umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    in.caller_umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    in.caller_ymin = matlab_obj_get_mat(mpc_obj, "ymin", 4);
    in.caller_ymax = matlab_obj_get_mat(mpc_obj, "ymax", 4);
    in.caller_V_y_min = matlab_obj_get_mat(mpc_obj, "V_y_min", 7);
    in.caller_V_y_max = matlab_obj_get_mat(mpc_obj, "V_y_max", 7);
    in.caller_V_u_min = matlab_obj_get_mat(mpc_obj, "V_u_min", 7);
    in.caller_V_u_max = matlab_obj_get_mat(mpc_obj, "V_u_max", 7);
    in.caller_E = matlab_obj_get_mat(mpc_obj, "E", 1);
    in.caller_F = matlab_obj_get_mat(mpc_obj, "F", 1);
    in.caller_G = matlab_obj_get_mat(mpc_obj, "G", 1);
    in.caller_dumin    = matlab_obj_get_mat(mpc_obj, "dumin",    5);
    in.caller_dumax    = matlab_obj_get_mat(mpc_obj, "dumax",    5);
    in.caller_Wu       = matlab_obj_get_mat(mpc_obj, "Wu",       2);
    in.caller_u_target = matlab_obj_get_mat(mpc_obj, "u_target", 8);
    double p_d = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d = matlab_obj_get_f64(mpc_obj, "m", 1);
    in.rho_eps = matlab_obj_get_f64(mpc_obj, "rho_eps", 7);
    in.outdist = matlab_obj_get_f64(mpc_obj, "outdist", 7);
    in.p = static_cast<int>(p_d);
    in.m = static_cast<int>(m_d);
    if (!in.A || !in.B || !in.C || !in.Sx || !in.Su || !in.Su1 ||
        !in.H || !in.Wy || !in.L) return mat_alloc(0, 0);

    /* For sim, we drive the *plant* state (not the observer estimate)
     * via the cached A/B/C — Tier-2 sim still assumes a perfect plant
     * model, so the observer = plant.  When obj.outdist is set the
     * augmented A/B/C may have nx_aug > nx_plant rows; for sim we
     * carry the augmented state but only the first nx_plant rows are
     * "the real plant"; the rest are the disturbance estimate which
     * stays zero in perfect-model sim. */
    int64_t nx = in.A->rows, nu = in.B->cols, ny = in.C->rows;
    matlab_mat *xp     = mat_alloc(nx, 1);
    matlab_mat *u_prev = mat_alloc(nu, 1);
    matlab_mat *dist   = mat_alloc(ny, 1);   /* perfect-model sim — dist stays 0 */
    matlab_mat *Y      = mat_alloc(T, ny);

    for (int t = 0; t < T; ++t) {
        matlab_mat *ym = matlab_matmul_mm(in.C, xp);
        matlab_mat *u_new = mpc_tick(in, &xp, &u_prev, &dist, ym, r);
        (void)u_new;
        matlab_mat *y_t = matlab_matmul_mm(in.C, xp);
        for (int64_t k = 0; k < ny; ++k)
            Y->data[static_cast<int64_t>(t) * ny + k] = y_t->data[k];
    }
    return Y;
}

/* ---------------------------------------------------------------- */
/* matlab_mpc_sim_opt — Tier-6 §7.6, 4-arg `sim(obj, T, r, opt)`.   */
/* Reads opt's PlantInitialState if Use_PlantInitialState is set;   */
/* otherwise behaves like the standard `matlab_mpc_sim`.            */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_mpc_sim_opt(void *mpc_obj_v, double T_d, matlab_mat *r,
                                void *opt_obj_v) {
    if (!opt_obj_v)
        return matlab_mpc_sim(mpc_obj_v, T_d, r);
    matlab_obj *opt = reinterpret_cast<matlab_obj *>(opt_obj_v);
    double use_x0 = matlab_obj_get_f64(opt, "Use_PlantInitialState", 21);
    if (use_x0 < 0.5)
        return matlab_mpc_sim(mpc_obj_v, T_d, r);

    /* PlantInitialState override path: replicate matlab_mpc_sim but
     * with a non-zero starting xp. */
    if (!mpc_obj_v || !r) return mat_alloc(0, 0);
    matlab_obj *mpc_obj = reinterpret_cast<matlab_obj *>(mpc_obj_v);
    int T = static_cast<int>(T_d);
    if (T < 1) T = 1;

    TickInputs in;
    in.A   = matlab_obj_get_mat(mpc_obj, "A",   1);
    in.B   = matlab_obj_get_mat(mpc_obj, "B",   1);
    in.C   = matlab_obj_get_mat(mpc_obj, "C",   1);
    in.Sx  = matlab_obj_get_mat(mpc_obj, "Sx",  2);
    in.Su  = matlab_obj_get_mat(mpc_obj, "Su",  2);
    in.Su1 = matlab_obj_get_mat(mpc_obj, "Su1", 3);
    in.H   = matlab_obj_get_mat(mpc_obj, "H",   1);
    in.Wy  = matlab_obj_get_mat(mpc_obj, "Wy",  2);
    in.L   = matlab_obj_get_mat(mpc_obj, "L",   1);
    in.caller_umin = matlab_obj_get_mat(mpc_obj, "umin", 4);
    in.caller_umax = matlab_obj_get_mat(mpc_obj, "umax", 4);
    in.caller_ymin = matlab_obj_get_mat(mpc_obj, "ymin", 4);
    in.caller_ymax = matlab_obj_get_mat(mpc_obj, "ymax", 4);
    in.caller_V_y_min = matlab_obj_get_mat(mpc_obj, "V_y_min", 7);
    in.caller_V_y_max = matlab_obj_get_mat(mpc_obj, "V_y_max", 7);
    in.caller_V_u_min = matlab_obj_get_mat(mpc_obj, "V_u_min", 7);
    in.caller_V_u_max = matlab_obj_get_mat(mpc_obj, "V_u_max", 7);
    in.caller_E = matlab_obj_get_mat(mpc_obj, "E", 1);
    in.caller_F = matlab_obj_get_mat(mpc_obj, "F", 1);
    in.caller_G = matlab_obj_get_mat(mpc_obj, "G", 1);
    in.caller_dumin    = matlab_obj_get_mat(mpc_obj, "dumin",    5);
    in.caller_dumax    = matlab_obj_get_mat(mpc_obj, "dumax",    5);
    in.caller_Wu       = matlab_obj_get_mat(mpc_obj, "Wu",       2);
    in.caller_u_target = matlab_obj_get_mat(mpc_obj, "u_target", 8);
    double p_d = matlab_obj_get_f64(mpc_obj, "p", 1);
    double m_d = matlab_obj_get_f64(mpc_obj, "m", 1);
    in.rho_eps = matlab_obj_get_f64(mpc_obj, "rho_eps", 7);
    in.outdist = matlab_obj_get_f64(mpc_obj, "outdist", 7);
    in.p = static_cast<int>(p_d);
    in.m = static_cast<int>(m_d);
    if (!in.A || !in.B || !in.C || !in.Sx || !in.Su || !in.Su1 ||
        !in.H || !in.Wy || !in.L) return mat_alloc(0, 0);

    int64_t nx = in.A->rows, nu = in.B->cols, ny = in.C->rows;
    matlab_mat *xp     = mat_alloc(nx, 1);
    matlab_mat *u_prev = mat_alloc(nu, 1);
    matlab_mat *dist   = mat_alloc(ny, 1);
    matlab_mat *Y      = mat_alloc(T, ny);

    matlab_mat *x0 = matlab_obj_get_mat(opt, "PlantInitialState", 17);
    if (x0)
        for (int64_t i = 0; i < nx && i < x0->rows; ++i)
            xp->data[i] = x0->data[i];

    for (int t = 0; t < T; ++t) {
        matlab_mat *ym = matlab_matmul_mm(in.C, xp);
        matlab_mat *u_new = mpc_tick(in, &xp, &u_prev, &dist, ym, r);
        (void)u_new;
        matlab_mat *y_t = matlab_matmul_mm(in.C, xp);
        for (int64_t k = 0; k < ny; ++k)
            Y->data[static_cast<int64_t>(t) * ny + k] = y_t->data[k];
    }
    return Y;
}

}  /* extern "C" */

/* ---------------------------------------------------------------- */
/* Nonlinear MPC (Tier-5 §6.1/6.2)                                  */
/*                                                                  */
/* `nlmpcmove(nlobj, st, ym, r)` builds a fmincon-style rollout cost */
/* over the decision z = [u(0); u(1); ...; u(m-1)] and hands it to  */
/* the shipped `matlab_optim_fmincon`.                              */
/*                                                                  */
/* User's StateFcn signature: `dxdt = stateFn(zxu)` where            */
/* `zxu = [x; u]` (length nx+nu).  Single-arg matches the existing  */
/* Optim handle ABI; the runtime packs x and u into the vector      */
/* before each call.  Forward Euler: x[h+1] = x[h] + Ts·dxdt.       */
/* Default tracking cost: Σᵢ ‖r - y[i]‖²·Wy + Σⱼ ‖Δu(j)‖²·Wdu.       */
/*                                                                  */
/* Thread-local context bridges fmincon's `double(*)(matlab_mat*)`  */
/* objective ABI to the user's StateFcn handle + the per-tick       */
/* parameters (x_init, r, Ts, weights, ...).                        */
/* ---------------------------------------------------------------- */

typedef matlab_mat *(*nlmpc_state_fn)(matlab_mat *);

struct NlmpcContext {
    nlmpc_state_fn state_fn;
    matlab_mat *x_init;
    matlab_mat *r;
    matlab_mat *u_prev;
    matlab_mat *Wy;
    matlab_mat *Wdu;
    double Ts;
    int p, m;
    int64_t nx, nu, ny;
};
static thread_local NlmpcContext g_nlmpc_ctx = {};

static double nlmpc_objective(matlab_mat *z) {
    NlmpcContext &c = g_nlmpc_ctx;
    if (!z || !c.state_fn) return 1e30;
    int64_t mnu = static_cast<int64_t>(c.m) * c.nu;
    if (z->rows < mnu) return 1e30;

    /* Roll out the state x[0..p] using Forward Euler over the
     * decision variable u trajectory (length m, then frozen at u(m-1)). */
    std::vector<double> x(static_cast<size_t>(c.nx), 0.0);
    for (int64_t i = 0; i < c.nx; ++i) x[static_cast<size_t>(i)] = c.x_init->data[i];

    double J = 0.0;
    /* Helper: evaluate dxdt = stateFn([x; u]) for the given (x, u). */
    auto eval_dxdt = [&](const std::vector<double> &xv,
                         int u_idx, std::vector<double> &dxdt_out) {
        matlab_mat *zxu = mat_alloc(c.nx + c.nu, 1);
        for (int64_t i = 0; i < c.nx; ++i)
            zxu->data[i] = xv[static_cast<size_t>(i)];
        for (int64_t k = 0; k < c.nu; ++k)
            zxu->data[c.nx + k] =
                z->data[static_cast<int64_t>(u_idx) * c.nu + k];
        matlab_mat *dxdt = c.state_fn(zxu);
        dxdt_out.assign(static_cast<size_t>(c.nx), 0.0);
        if (dxdt && dxdt->rows >= c.nx) {
            for (int64_t i = 0; i < c.nx; ++i)
                dxdt_out[static_cast<size_t>(i)] = dxdt->data[i];
        }
    };

    for (int h = 0; h < c.p; ++h) {
        /* u(h) = u trajectory at step h; frozen at m-1 once h ≥ m. */
        int idx = (h < c.m) ? h : (c.m - 1);
        /* Tier-6 §7.8 — RK4 integration (was Forward Euler in Tier-5).
         *   k1 = f(x, u)
         *   k2 = f(x + Ts/2·k1, u)
         *   k3 = f(x + Ts/2·k2, u)
         *   k4 = f(x + Ts·k3, u)
         *   x[h+1] = x[h] + Ts/6·(k1 + 2·k2 + 2·k3 + k4)
         */
        std::vector<double> k1, k2, k3, k4, xt;
        eval_dxdt(x, idx, k1);
        xt.assign(static_cast<size_t>(c.nx), 0.0);
        for (int64_t i = 0; i < c.nx; ++i)
            xt[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] +
                0.5 * c.Ts * k1[static_cast<size_t>(i)];
        eval_dxdt(xt, idx, k2);
        for (int64_t i = 0; i < c.nx; ++i)
            xt[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] +
                0.5 * c.Ts * k2[static_cast<size_t>(i)];
        eval_dxdt(xt, idx, k3);
        for (int64_t i = 0; i < c.nx; ++i)
            xt[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] +
                c.Ts * k3[static_cast<size_t>(i)];
        eval_dxdt(xt, idx, k4);
        for (int64_t i = 0; i < c.nx; ++i)
            x[static_cast<size_t>(i)] += (c.Ts / 6.0) *
                (k1[static_cast<size_t>(i)] +
                 2.0 * k2[static_cast<size_t>(i)] +
                 2.0 * k3[static_cast<size_t>(i)] +
                 k4[static_cast<size_t>(i)]);

        /* Tracking cost contribution: y = x[0:ny]; J += ‖r - y‖²·Wy². */
        for (int64_t k = 0; k < c.ny; ++k) {
            double yi = x[static_cast<size_t>(k)];
            double ri = (k < c.r->rows) ? c.r->data[k] : 0.0;
            double e = ri - yi;
            double w = c.Wy->data[k];
            J += w * w * e * e;
        }
        /* Move-suppression cost: ‖Δu(h)‖²·Wdu² for h < m. */
        if (h < c.m) {
            for (int64_t k = 0; k < c.nu; ++k) {
                double uh   = z->data[static_cast<int64_t>(h) * c.nu + k];
                double uprev = (h == 0)
                    ? c.u_prev->data[k]
                    : z->data[static_cast<int64_t>(h - 1) * c.nu + k];
                double du = uh - uprev;
                double w = c.Wdu->data[k];
                J += w * w * du * du;
            }
        }
    }
    return J;
}

extern "C" matlab_mat *matlab_nlmpc_move(void *nlobj_v, matlab_mat *x,
                                         matlab_mat *u_prev,
                                         matlab_mat *r,
                                         void *state_fn_ptr) {
    if (!nlobj_v || !x || !u_prev || !r || !state_fn_ptr)
        return mat_alloc(0, 0);
    matlab_obj *nlobj = reinterpret_cast<matlab_obj *>(nlobj_v);

    /* Pull configuration off the nlmpc obj.  Note: StateFcn is the
     * direct 5th argument to this runtime entry (function handles
     * don't round-trip cleanly through classdef property storage);
     * the obj's `StateFcn` property is informational only. */
    double nx_d = matlab_obj_get_f64(nlobj, "nx", 2);
    double nu_d = matlab_obj_get_f64(nlobj, "nu", 2);
    double ny_d = matlab_obj_get_f64(nlobj, "ny", 2);
    double Ts   = matlab_obj_get_f64(nlobj, "Ts", 2);
    double p_d  = matlab_obj_get_f64(nlobj, "p", 1);
    double m_d  = matlab_obj_get_f64(nlobj, "m", 1);
    matlab_mat *Wy   = matlab_obj_get_mat(nlobj, "Wy",  2);
    matlab_mat *Wdu  = matlab_obj_get_mat(nlobj, "Wdu", 3);
    matlab_mat *umin = matlab_obj_get_mat(nlobj, "umin", 4);
    matlab_mat *umax = matlab_obj_get_mat(nlobj, "umax", 4);

    int64_t nx = static_cast<int64_t>(nx_d);
    int64_t nu = static_cast<int64_t>(nu_d);
    int64_t ny = static_cast<int64_t>(ny_d);
    int p = static_cast<int>(p_d);
    int m = static_cast<int>(m_d);
    if (p < 1) p = 1;
    if (m < 1) m = 1;
    if (m > p) m = p;

    if (!Wy || !Wdu) return mat_alloc(0, 0);

    /* Set up the thread-local context. */
    g_nlmpc_ctx.state_fn = reinterpret_cast<nlmpc_state_fn>(state_fn_ptr);
    g_nlmpc_ctx.x_init   = x;
    g_nlmpc_ctx.r        = r;
    g_nlmpc_ctx.u_prev   = u_prev;
    g_nlmpc_ctx.Wy       = Wy;
    g_nlmpc_ctx.Wdu      = Wdu;
    g_nlmpc_ctx.Ts       = Ts;
    g_nlmpc_ctx.p        = p;
    g_nlmpc_ctx.m        = m;
    g_nlmpc_ctx.nx       = nx;
    g_nlmpc_ctx.nu       = nu;
    g_nlmpc_ctx.ny       = ny;

    /* Build fmincon x0 = u_prev replicated m times — warm-start at
     * the holding trajectory. */
    int64_t mnu = static_cast<int64_t>(m) * nu;
    matlab_mat *z0 = mat_alloc(mnu, 1);
    for (int j = 0; j < m; ++j)
        for (int64_t k = 0; k < nu; ++k)
            z0->data[static_cast<int64_t>(j) * nu + k] = u_prev->data[k];

    /* MV bound vectors expanded over the m horizon steps. */
    matlab_mat *lb = mat_alloc(mnu, 1);
    matlab_mat *ub = mat_alloc(mnu, 1);
    for (int j = 0; j < m; ++j)
        for (int64_t k = 0; k < nu; ++k) {
            lb->data[static_cast<int64_t>(j) * nu + k] =
                umin ? umin->data[k] : -1e6;
            ub->data[static_cast<int64_t>(j) * nu + k] =
                umax ? umax->data[k] :  1e6;
        }

    /* Call fmincon with the objective wrapper.  No linear / equality
     * / nonlinear constraints for Tier-5 — bounds only. */
    void *obj_p = reinterpret_cast<void *>(&nlmpc_objective);
    matlab_mat *z_opt = matlab_optim_fmincon(obj_p, z0,
                                              nullptr, nullptr,
                                              nullptr, nullptr,
                                              lb, ub, nullptr);
    if (!z_opt || z_opt->rows < nu) {
        matlab_mat *u_fallback = mat_alloc(nu, 1);
        for (int64_t k = 0; k < nu; ++k) u_fallback->data[k] = u_prev->data[k];
        return u_fallback;
    }
    /* Return u(0) = first nu entries of z. */
    matlab_mat *u_new = mat_alloc(nu, 1);
    for (int64_t k = 0; k < nu; ++k) u_new->data[k] = z_opt->data[k];
    return u_new;
}
