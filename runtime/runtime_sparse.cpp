/* runtime_sparse.cpp — sparse-matrix infrastructure for matlab_llvm.
 *
 * This is the §10.1 sub-project of docs/pde_toolbox_roadmap.md.  Lives
 * outside runtime_pde.cpp so the descriptor is reusable for SPT / CST
 * / general sparse linear algebra later.
 *
 * Descriptor layout (CSR, 0-based internally, 1-based at the MATLAB
 * surface):
 *     matlab_sparse_mat {
 *       magic     = 0xC0FFEE05u  (sniffable polymorphic dispatch)
 *       _pad
 *       row_ptr   int64_t[m+1]    cumulative nnz per row
 *       col_idx   int64_t[nnz]    column index per stored value (0-based)
 *       vals      double[nnz]     stored values
 *       rows, cols, nnz
 *     }
 *
 * Builders take triplet (I, J, V, m, n) input — the natural format
 * FEM assembly produces.  Duplicate (i, j) coordinates are summed.
 *
 * Solvers:
 *   - `sparse_matvec`           y = S * x        (used by PCG)
 *   - `sparse_dot`              <x, y>           (PCG helper)
 *   - `sparse_pcg(S, b, tol, maxit)` PCG with Jacobi preconditioner.
 *     Returns matlab_struct { Solution, Flag, RelRes, Iter }.
 *
 * The 0xC0FFEE05 magic word lives in the runtime_internal.h header
 * (added there in this commit).  Existing matlab_disp / matlab_size
 * predicates that sniff descriptor kind use the same first-uint32
 * convention as matlab_mat_c / matlab_mat3.
 */

#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MATLAB_SPARSE_MAGIC 0xC0FFEE05u

extern "C" {

/* Same forward-decl strategy as runtime_pde.cpp — declare what we use
 * without pulling matlab_runtime.h's typedef tail in. */
matlab_struct *matlab_struct_new(void);
void   matlab_struct_set_f64(matlab_struct *s, const char *name, int64_t len, double v);
void   matlab_struct_set_mat(matlab_struct *s, const char *name, int64_t len, matlab_mat *m);
double matlab_struct_get_f64(matlab_struct *s, const char *name, int64_t len);
matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name, int64_t len);

/* --- Descriptor + allocator ----------------------------------------- */

struct matlab_sparse_mat {
    uint32_t magic;
    uint32_t _pad;
    int64_t *row_ptr;     /* (rows + 1) entries, last == nnz */
    int64_t *col_idx;     /* nnz entries */
    double  *vals;        /* nnz entries */
    int64_t  rows;
    int64_t  cols;
    int64_t  nnz;
};

static matlab_sparse_mat *sparse_alloc(int64_t m, int64_t n, int64_t nnz) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    if (nnz < 0) nnz = 0;
    matlab_sparse_mat *S = (matlab_sparse_mat *)calloc(1, sizeof(matlab_sparse_mat));
    S->magic   = MATLAB_SPARSE_MAGIC;
    S->rows    = m;
    S->cols    = n;
    S->nnz     = nnz;
    S->row_ptr = (int64_t *)calloc((size_t)(m + 1), sizeof(int64_t));
    S->col_idx = nnz ? (int64_t *)calloc((size_t)nnz, sizeof(int64_t)) : nullptr;
    S->vals    = nnz ? (double  *)calloc((size_t)nnz, sizeof(double))  : nullptr;
    return S;
}

/* Predicate: 0xC0FFEE05 sniff (matches mat_is_complex / mat_is_3d
 * idiom).  Inline so it can fold in callers. */
static inline int mat_is_sparse(const void *p) {
    if (!p) return 0;
    return *(const uint32_t *)p == MATLAB_SPARSE_MAGIC;
}

/* --- Public ABI --------------------------------------------------- */

/* sparse(I, J, V, m, n) — MATLAB-faithful triplet constructor.
 *
 * I and J are 1-based row/col index column vectors (Nx1 each).
 * V holds the values (Nx1).  m, n are the target shape.
 *
 * Duplicates (same (i, j) pair) are summed — this is what makes the
 * FEM scatter work without explicit deduplication.
 *
 * Output: matlab_sparse_mat returned as a void* (caller sniffs the
 * magic for polymorphic dispatch).
 */
void *matlab_sparse_from_triplets(matlab_mat *I, matlab_mat *J, matlab_mat *V,
                                  double m_d, double n_d) {
    if (!I || !J || !V) return nullptr;
    int64_t m  = (int64_t)m_d;
    int64_t n  = (int64_t)n_d;
    int64_t nz = I->rows * I->cols;
    if (J->rows * J->cols != nz || V->rows * V->cols != nz) return nullptr;
    if (m <= 0 || n <= 0) return nullptr;

    /* Bucket triplets by row.  We use a 3-step build:
     *   pass 1: count entries per row;
     *   pass 2: prefix-sum into row_ptr;
     *   pass 3: scatter (col_idx, vals) by row, summing duplicates.
     */
    std::vector<int64_t> count((size_t)m, 0);
    for (int64_t k = 0; k < nz; ++k) {
        int64_t r = (int64_t)I->data[k] - 1;
        if (r >= 0 && r < m) count[(size_t)r] += 1;
    }
    /* Worst-case nnz: every triplet unique.  We'll compact after. */
    auto *S = sparse_alloc(m, n, nz);
    int64_t acc = 0;
    for (int64_t r = 0; r < m; ++r) {
        S->row_ptr[r] = acc;
        acc += count[(size_t)r];
        count[(size_t)r] = S->row_ptr[r];  /* reuse as write cursor */
    }
    S->row_ptr[m] = acc;
    /* Scatter. */
    for (int64_t k = 0; k < nz; ++k) {
        int64_t r = (int64_t)I->data[k] - 1;
        int64_t c = (int64_t)J->data[k] - 1;
        if (r < 0 || r >= m || c < 0 || c >= n) continue;
        int64_t pos = count[(size_t)r]++;
        S->col_idx[pos] = c;
        S->vals[pos]    = V->data[k];
    }
    /* Per-row sort + dedupe.  For symmetric SPD assembly the duplicate
     * rate is high (every node-pair gets multiple element contributions).
     * In-place insertion sort is fine because each row is short. */
    int64_t write = 0;
    std::vector<int64_t> new_rp((size_t)(m + 1), 0);
    for (int64_t r = 0; r < m; ++r) {
        int64_t lo = S->row_ptr[r];
        int64_t hi = S->row_ptr[r + 1];
        /* Sort the [lo, hi) slice by col_idx, summing duplicates. */
        std::vector<std::pair<int64_t, double>> tmp;
        tmp.reserve((size_t)(hi - lo));
        for (int64_t k = lo; k < hi; ++k) tmp.emplace_back(S->col_idx[k], S->vals[k]);
        std::sort(tmp.begin(), tmp.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });
        new_rp[(size_t)r] = write;
        int64_t cur = -1;
        for (auto &p : tmp) {
            if (p.first == cur) {
                S->vals[write - 1] += p.second;
            } else {
                S->col_idx[write] = p.first;
                S->vals[write]    = p.second;
                cur = p.first;
                write++;
            }
        }
    }
    new_rp[(size_t)m] = write;
    memcpy(S->row_ptr, new_rp.data(), sizeof(int64_t) * (size_t)(m + 1));
    S->nnz = write;
    /* We over-allocated; the extra slots stay parked at the tail —
     * cheap and avoids a realloc round-trip. */
    return (void *)S;
}

/* sparse(S) — return S itself.  Provided so user code can write
 * `S = sparse(K)` idempotently; the FEM caller already gets a
 * sparse_mat from sparse_from_triplets. */
void *matlab_sparse_identity_wrap(void *S) {
    return S;
}

double matlab_sparse_nnz(void *Sv) {
    if (!mat_is_sparse(Sv)) return 0.0;
    return (double)((matlab_sparse_mat *)Sv)->nnz;
}
double matlab_sparse_rows(void *Sv) {
    if (!mat_is_sparse(Sv)) return 0.0;
    return (double)((matlab_sparse_mat *)Sv)->rows;
}
double matlab_sparse_cols(void *Sv) {
    if (!mat_is_sparse(Sv)) return 0.0;
    return (double)((matlab_sparse_mat *)Sv)->cols;
}

/* full(S) — convert sparse to dense matlab_mat. */
matlab_mat *matlab_sparse_full(void *Sv) {
    if (!mat_is_sparse(Sv)) return mat_alloc(0, 0);
    matlab_sparse_mat *S = (matlab_sparse_mat *)Sv;
    matlab_mat *D = mat_alloc(S->rows, S->cols);
    for (int64_t r = 0; r < S->rows; ++r) {
        int64_t lo = S->row_ptr[r];
        int64_t hi = S->row_ptr[r + 1];
        for (int64_t k = lo; k < hi; ++k) {
            int64_t c = S->col_idx[k];
            D->data[r * S->cols + c] = S->vals[k];
        }
    }
    return D;
}

/* speye(n) — n×n identity in sparse form. */
void *matlab_sparse_eye(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 0) n = 0;
    auto *S = sparse_alloc(n, n, n);
    for (int64_t i = 0; i < n; ++i) {
        S->row_ptr[i] = i;
        S->col_idx[i] = i;
        S->vals[i]    = 1.0;
    }
    S->row_ptr[n] = n;
    return (void *)S;
}

/* y = S * x.  x is a dense column vector (or n×1 matrix). */
matlab_mat *matlab_sparse_matvec(void *Sv, matlab_mat *x) {
    if (!mat_is_sparse(Sv) || !x) return mat_alloc(0, 0);
    matlab_sparse_mat *S = (matlab_sparse_mat *)Sv;
    if (S->cols != x->rows * x->cols) return mat_alloc(0, 0);
    matlab_mat *y = mat_alloc(S->rows, 1);
    for (int64_t r = 0; r < S->rows; ++r) {
        double s = 0.0;
        int64_t lo = S->row_ptr[r];
        int64_t hi = S->row_ptr[r + 1];
        for (int64_t k = lo; k < hi; ++k) s += S->vals[k] * x->data[S->col_idx[k]];
        y->data[r] = s;
    }
    return y;
}

/* Diagonal of S as a column vector (the Jacobi preconditioner's
 * source).  For asymmetric / non-square inputs, missing diagonal
 * entries are 1 (treat as a no-op preconditioner). */
matlab_mat *matlab_sparse_diag(void *Sv) {
    if (!mat_is_sparse(Sv)) return mat_alloc(0, 0);
    matlab_sparse_mat *S = (matlab_sparse_mat *)Sv;
    int64_t n = (S->rows < S->cols) ? S->rows : S->cols;
    matlab_mat *d = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) {
        d->data[i] = 1.0;  /* default if missing — leaves PCG unscaled */
        int64_t lo = S->row_ptr[i];
        int64_t hi = S->row_ptr[i + 1];
        for (int64_t k = lo; k < hi; ++k) {
            if (S->col_idx[k] == i) {
                d->data[i] = S->vals[k];
                break;
            }
        }
    }
    return d;
}

/* --- PCG with Jacobi preconditioner ---------------------------------
 *
 * Solves S · x = b for symmetric positive-definite S.  Pre-conditioner
 * M is the diagonal of S; the iteration uses M^{-1} as a vector mul.
 *
 * Returns matlab_struct {
 *   .Solution = x (n×1 dense),
 *   .Flag     = 0 (converged), 1 (max iter), 2 (singular),
 *   .RelRes   = ||b - S x|| / ||b||,
 *   .Iter     = number of iterations
 * }.
 */
matlab_struct *matlab_sparse_pcg(void *Sv, matlab_mat *b,
                                 double tol, double maxit_d) {
    matlab_struct *out = matlab_struct_new();
    if (!mat_is_sparse(Sv) || !b) {
        matlab_struct_set_mat(out, "Solution", 8, mat_alloc(0, 0));
        matlab_struct_set_f64(out, "Flag",     4, 2.0);
        matlab_struct_set_f64(out, "RelRes",   6, 0.0);
        matlab_struct_set_f64(out, "Iter",     4, 0.0);
        return out;
    }
    matlab_sparse_mat *S = (matlab_sparse_mat *)Sv;
    int64_t n = S->rows;
    if (S->cols != n) {  /* PCG is for square SPD systems */
        matlab_struct_set_mat(out, "Solution", 8, mat_alloc(0, 0));
        matlab_struct_set_f64(out, "Flag",     4, 2.0);
        return out;
    }
    if (tol <= 0) tol = 1e-8;
    int64_t maxit = (int64_t)maxit_d;
    if (maxit <= 0) maxit = (n < 1000) ? n : 1000;

    matlab_mat *diag = matlab_sparse_diag(Sv);
    /* Allow zeros on the diagonal (constrained DOFs) — Jacobi inverse
     * collapses to 1 there (no-op scaling). */
    std::vector<double> Minv((size_t)n, 1.0);
    for (int64_t i = 0; i < n; ++i) {
        double d = diag->data[i];
        Minv[(size_t)i] = (fabs(d) > 1e-30) ? 1.0 / d : 1.0;
    }

    std::vector<double> x((size_t)n, 0.0);
    std::vector<double> r((size_t)n);
    std::vector<double> z((size_t)n);
    std::vector<double> p((size_t)n);
    std::vector<double> Ap((size_t)n);

    /* r = b - S * x = b (since x=0). */
    double bnorm2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        r[(size_t)i] = b->data[i];
        bnorm2 += r[(size_t)i] * r[(size_t)i];
    }
    double bnorm = sqrt(bnorm2);
    if (bnorm == 0.0) {
        /* RHS is zero — solution is zero. */
        matlab_mat *X = mat_alloc(n, 1);
        matlab_struct_set_mat(out, "Solution", 8, X);
        matlab_struct_set_f64(out, "Flag",     4, 0.0);
        matlab_struct_set_f64(out, "RelRes",   6, 0.0);
        matlab_struct_set_f64(out, "Iter",     4, 0.0);
        return out;
    }
    /* z = M^{-1} r;  p = z;  rzold = <r, z>. */
    double rzold = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        z[(size_t)i] = Minv[(size_t)i] * r[(size_t)i];
        p[(size_t)i] = z[(size_t)i];
        rzold += r[(size_t)i] * z[(size_t)i];
    }

    int64_t iter = 0;
    int flag = 1;  /* default: maxit reached */
    double rel = 1.0;
    for (iter = 1; iter <= maxit; ++iter) {
        /* Ap = S * p. */
        for (int64_t k = 0; k < n; ++k) {
            double s = 0.0;
            int64_t lo = S->row_ptr[k];
            int64_t hi = S->row_ptr[k + 1];
            for (int64_t kk = lo; kk < hi; ++kk)
                s += S->vals[kk] * p[(size_t)S->col_idx[kk]];
            Ap[(size_t)k] = s;
        }
        double pAp = 0.0;
        for (int64_t i = 0; i < n; ++i) pAp += p[(size_t)i] * Ap[(size_t)i];
        if (fabs(pAp) < 1e-30) { flag = 2; break; }
        double alpha = rzold / pAp;

        double rnorm2 = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            x[(size_t)i] += alpha * p[(size_t)i];
            r[(size_t)i] -= alpha * Ap[(size_t)i];
            rnorm2 += r[(size_t)i] * r[(size_t)i];
        }
        rel = sqrt(rnorm2) / bnorm;
        if (rel < tol) { flag = 0; break; }

        double rznew = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            z[(size_t)i] = Minv[(size_t)i] * r[(size_t)i];
            rznew += r[(size_t)i] * z[(size_t)i];
        }
        double beta = rznew / rzold;
        for (int64_t i = 0; i < n; ++i)
            p[(size_t)i] = z[(size_t)i] + beta * p[(size_t)i];
        rzold = rznew;
    }
    if (iter > maxit) iter = maxit;

    matlab_mat *X = mat_alloc(n, 1);
    memcpy(X->data, x.data(), sizeof(double) * (size_t)n);
    matlab_struct_set_mat(out, "Solution", 8, X);
    matlab_struct_set_f64(out, "Flag",     4, (double)flag);
    matlab_struct_set_f64(out, "RelRes",   6, rel);
    matlab_struct_set_f64(out, "Iter",     4, (double)iter);
    return out;
}

/* Convenience accessors for the PCG result struct. */
matlab_mat *matlab_sparse_pcg_x(matlab_struct *r) {
    return matlab_struct_get_mat(r, "Solution", 8);
}
double matlab_sparse_pcg_flag(matlab_struct *r) {
    return matlab_struct_get_f64(r, "Flag", 4);
}
double matlab_sparse_pcg_relres(matlab_struct *r) {
    return matlab_struct_get_f64(r, "RelRes", 6);
}
double matlab_sparse_pcg_iter(matlab_struct *r) {
    return matlab_struct_get_f64(r, "Iter", 4);
}

}  /* extern "C" */
