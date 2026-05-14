/* Tiny MATLAB-runtime shim. Linked with programs produced by matlabc's
 * -emit-llvm pipeline.
 *
 * All functions use a leading `matlab_` prefix to avoid collision with libc
 * and to make the calling convention explicit to the compiler frontend.
 *
 * Built as C++ (Phase 3 of docs/port_runtime_2_cpp.md). The body keeps
 * its C structure end-to-end — no STL types in signatures, no exceptions
 * crossing the JIT boundary — but compiling under a C++ compiler unlocks
 * RAII migrations in subsequent phases. The single `extern "C"` block
 * around the entire payload preserves the exported symbol names so
 * JIT-emitted code resolves matlab_* by C name unchanged.
 */

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    /* clock_gettime / nanosleep — pause/tic/toc */
#include <unistd.h>  /* for write(2), used by matlab_err_emit_traceback_to_stderr */

#include <vector>    /* Phase-4 RAII scratch buffers */
#include <algorithm> /* std::sort — used by §4.3 medfilt1 / hampel */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "runtime_internal.h"

extern "C" {

/* A single global mutex serializes all stdout I/O so parfor bodies that call
 * disp/fprintf don't interleave mid-line. This is a tiny concession to
 * predictability; real MATLAB uses per-worker stdout aggregation.
 * Non-static: shared with runtime_debug.cpp via runtime_internal.h. */
pthread_mutex_t matlab_io_mutex = PTHREAD_MUTEX_INITIALIZER;

/* disp('text') — print a MATLAB char array (length-prefixed, no NUL). */
void matlab_disp_str(const char *s, int64_t n) {
    pthread_mutex_lock(&matlab_io_mutex);
    fwrite(s, 1, (size_t)n, stdout);
    fputc('\n', stdout);
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* disp(scalar) — MATLAB formats doubles with a leading blank line then the
 * value; we simplify to just the value plus a newline. */
void matlab_disp_f64(double v) {
    pthread_mutex_lock(&matlab_io_mutex);
    printf("%g\n", v);
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* disp(row_vector) — prints the elements on one line. */
void matlab_disp_vec_f64(const double *data, int64_t n) {
    if (n < 0) n = 0;
    pthread_mutex_lock(&matlab_io_mutex);
    for (int64_t i = 0; i < n; ++i)
        printf("   %7g", data[i]);
    putchar('\n');
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* disp(matrix) — prints each row on its own line. Data is row-major. */
void matlab_disp_mat_f64(const double *data, int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    pthread_mutex_lock(&matlab_io_mutex);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j)
            printf("   %7g", data[i * n + j]);
        putchar('\n');
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* Copy `n` bytes of `src` into `dst`, expanding MATLAB's printf-style escape
 * sequences (\n, \t, \r, \\, \', \", \0). MATLAB's fprintf is documented to
 * interpret these sequences inside the format string even when the format
 * comes from a single-quoted char literal. Returns the new length. */
static int64_t expand_escapes(char *dst, const char *src, int64_t n) {
    int64_t w = 0;
    for (int64_t i = 0; i < n; ++i) {
        char c = src[i];
        if (c != '\\' || i + 1 >= n) { dst[w++] = c; continue; }
        char e = src[++i];
        switch (e) {
            case 'n':  dst[w++] = '\n'; break;
            case 't':  dst[w++] = '\t'; break;
            case 'r':  dst[w++] = '\r'; break;
            case '\\': dst[w++] = '\\'; break;
            case '\'': dst[w++] = '\''; break;
            case '"':  dst[w++] = '"';  break;
            case '0':  dst[w++] = '\0'; break;
            default:   dst[w++] = '\\'; dst[w++] = e; break;
        }
    }
    return w;
}

/* fprintf('fmt', v) with a single f64 argument. */
void matlab_fprintf_f64(const char *fmt, int64_t n, double v) {
    if (n < 0) n = 0;
    if (n > 1023) n = 1023;
    char buf[1024];
    int64_t len = expand_escapes(buf, fmt, n);
    buf[len] = '\0';
    printf(buf, v);
}

/* fprintf('fmt') with no numeric arguments. */
void matlab_fprintf_str(const char *fmt, int64_t n) {
    if (n < 0) n = 0;
    if (n > 1023) n = 1023;
    char buf[1024];
    int64_t len = expand_escapes(buf, fmt, n);
    buf[len] = '\0';
    pthread_mutex_lock(&matlab_io_mutex);
    fputs(buf, stdout);
    pthread_mutex_unlock(&matlab_io_mutex);
}

/*
 * parfor dispatcher: spawns one pthread per iteration of start:step:end.
 * `body(iv, state)` is called for each iteration. `state` is an opaque
 * pointer the compiler uses to pass captured values (today: a packed array
 * of pointers to reduction variables). Iterations run concurrently; the
 * dispatcher blocks until all threads finish (join).
 */
typedef void (*matlab_parfor_body_t)(double iv, void *state);

struct matlab_parfor_arg {
    matlab_parfor_body_t body;
    double iv;
    void *state;
};

static void *matlab_parfor_worker(void *p) {
    struct matlab_parfor_arg *a = (struct matlab_parfor_arg *)p;
    a->body(a->iv, a->state);
    return NULL;
}

void matlab_parfor_dispatch(double start, double step, double end,
                            matlab_parfor_body_t body, void *state) {
    if (!body) return;
    if (step == 0.0) return;
    /* Count iterations using MATLAB's range length formula. */
    double diff = end - start;
    if ((step > 0 && diff < 0) || (step < 0 && diff > 0)) return;
    int64_t n = (int64_t)(diff / step) + 1;
    if (n <= 0) return;

    pthread_t *threads = (pthread_t *)malloc((size_t)n * sizeof(pthread_t));
    struct matlab_parfor_arg *args = (struct matlab_parfor_arg *)malloc(
        (size_t)n * sizeof(struct matlab_parfor_arg));
    if (!threads || !args) { free(threads); free(args); return; }

    for (int64_t i = 0; i < n; ++i) {
        args[i].body = body;
        args[i].iv = start + (double)i * step;
        args[i].state = state;
        pthread_create(&threads[i], NULL, matlab_parfor_worker, &args[i]);
    }
    for (int64_t i = 0; i < n; ++i) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    free(args);
}

/*
 * Mutex-protected floating-point add used for parfor reductions on f64
 * scalars. `*ptr += delta`, atomic w.r.t. other callers of this function
 * across threads. Not fast (global lock) but deterministic and correct.
 */
static pthread_mutex_t matlab_reduction_mutex = PTHREAD_MUTEX_INITIALIZER;

void matlab_reduce_add_f64(double *ptr, double delta) {
    pthread_mutex_lock(&matlab_reduction_mutex);
    *ptr += delta;
    pthread_mutex_unlock(&matlab_reduction_mutex);
}

/*===========================================================================
 *
 *  Matrix descriptor + math
 *
 * --------------------------------------------------------------------------
 * A `matlab_mat` is a heap-allocated row-major double matrix. Every
 * matrix-producing runtime entry allocates a fresh matlab_mat; all results
 * are leaked (programs are assumed short-lived — this is a demo runtime).
 *
 * The compiler passes matrix values around as `matlab_mat *` (i.e. `ptr` in
 * the LLVM dialect). Matrix-typed variables become stack slots of pointer
 * type (llvm.alloca !llvm.ptr).
 *===========================================================================*/

/* matlab_mat / matlab_mat_c / matlab_mat3 layouts, the magic constants,
 * mat_is_complex / mat_is_3d, and the mat_alloc / mat_c_alloc / mat3_alloc
 * declarations all live in runtime_internal.h now (Phase-2 file split).
 * The definitions for mat_alloc and matlab_disp_mat_c remain in this TU. */
void matlab_disp_mat_c(matlab_mat_c *A);

matlab_mat *mat_alloc(int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    matlab_mat *A = (matlab_mat *)calloc(1, sizeof(matlab_mat));
    A->rows = m; A->cols = n;
    A->data = (double *)calloc((size_t)(m * n + 1), sizeof(double));
    return A;
}

/* matlab_mat_from_buf: used by the compiler when materializing a literal
 * `[a b; c d]`. Takes a row-major buffer of doubles and wraps it into a
 * fresh matrix descriptor. */
matlab_mat *matlab_mat_from_buf(const double *buf, double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    matlab_mat *A = mat_alloc(rm, cn);
    memcpy(A->data, buf, (size_t)(rm * cn) * sizeof(double));
    return A;
}

/*---------- Constructors --------------------------------------------------*/

matlab_mat *matlab_zeros(double m, double n) {
    /* calloc-zeroed */
    return mat_alloc((int64_t)m, (int64_t)n);
}

matlab_mat *matlab_ones(double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    /* Phase-5: fill_mat collapses the alloc-then-loop boilerplate. */
    return matlab::runtime::fill_mat(rm, cn,
        [](int64_t, int64_t) { return 1.0; }).release();
}

matlab_mat *matlab_eye(double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    return matlab::runtime::fill_mat(rm, cn,
        [](int64_t i, int64_t j) { return (i == j) ? 1.0 : 0.0; }).release();
}

/* Siamese method for odd-order magic squares. For even n we fall back to a
 * simple 1..n² row-major fill (not a true magic square, but the shape and
 * total match). MATLAB uses three different algorithms for odd / 4k / 4k+2;
 * implementing all three is a separate exercise. */
matlab_mat *matlab_magic(double nd) {
    int64_t n = (int64_t)nd;
    if (n <= 0) n = 1;
    matlab_mat *A = mat_alloc(n, n);
    if (n % 2 == 1) {
        int64_t i = 0, j = n / 2;
        for (int64_t k = 1; k <= n * n; ++k) {
            A->data[i * n + j] = (double)k;
            int64_t ni = (i - 1 + n) % n;
            int64_t nj = (j + 1) % n;
            if (A->data[ni * n + nj] != 0.0) {
                i = (i + 1) % n;
            } else {
                i = ni; j = nj;
            }
        }
    } else {
        for (int64_t k = 0; k < n * n; ++k) A->data[k] = (double)(k + 1);
    }
    return A;
}

/*---------- Random number generators --------------------------------------
 * xorshift64 for uniform, Box-Muller for normal. Seed is fixed so tests
 * with -DMATLAB_RUNTIME_FIXED_SEED (default) produce reproducible output;
 * to randomize, link with a caller that first sets matlab_rng_state before
 * any rand/randn call.
 *--------------------------------------------------------------------------*/

uint64_t matlab_rng_state = 0x243f6a8885a308d3ULL;

static double rng_uniform(void) {
    uint64_t x = matlab_rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    matlab_rng_state = x;
    return (double)(x >> 11) / (double)(1ULL << 53);
}

static double rng_normal(void) {
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

matlab_mat *matlab_rand(double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    matlab_mat *A = mat_alloc(rm, cn);
    for (int64_t k = 0; k < rm * cn; ++k) A->data[k] = rng_uniform();
    return A;
}

matlab_mat *matlab_randn(double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    matlab_mat *A = mat_alloc(rm, cn);
    for (int64_t k = 0; k < rm * cn; ++k) A->data[k] = rng_normal();
    return A;
}

/*---------- Linear algebra (pure C, no BLAS) ------------------------------
 *
 * These routines are intentionally library-agnostic: no dependency on
 * BLAS / LAPACK. Performance is a naive O(N^3) for matmul and LU, which is
 * fine for teaching-scale inputs (few hundred rows) and keeps the runtime
 * a single, transpilable C file.
 *
 * Numerical robustness:
 *   - LU factorization uses partial row pivoting, standard and stable for
 *     well-conditioned inputs.
 *   - We don't do row scaling or iterative refinement. Inputs near
 *     singular may produce inaccurate results; we detect exact singularity
 *     (pivot magnitude below 1e-300) and return a zero-sized result.
 *
 *--------------------------------------------------------------------------*/

/* Forward decl used by mrdivide (defined in the shape-ops section below). */
matlab_mat *matlab_transpose(matlab_mat *A);

/* C = A * B. Returns a 0x0 matrix if dimensions don't match. */
matlab_mat *matlab_matmul_mm(matlab_mat *A, matlab_mat *B) {
    if (A->cols != B->rows) return mat_alloc(0, 0);
    int64_t m = A->rows, k = A->cols, n = B->cols;
    matlab_mat *C = mat_alloc(m, n);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t t = 0; t < k; ++t)
                s += A->data[i * k + t] * B->data[t * n + j];
            C->data[i * n + j] = s;
        }
    }
    return C;
}

/*
 * In-place LU factorization with partial pivoting.
 *
 * On entry:  `A` is n*n, row-major.
 * On exit:   A is overwritten with L (unit diagonal, stored strictly
 *            below diag) and U (stored on and above diag). `piv[i]` is
 *            the original row index that ended up in row i. `sign` holds
 *            the permutation sign (+1 / -1) for det().
 * Returns 0 on success, -1 on (detected) singularity.
 */
static int lu_decompose(double *A, int64_t n, int64_t *piv, int *sign) {
    *sign = 1;
    for (int64_t i = 0; i < n; ++i) piv[i] = i;
    for (int64_t k = 0; k < n; ++k) {
        /* find pivot row */
        int64_t p = k;
        double best = fabs(A[k * n + k]);
        for (int64_t i = k + 1; i < n; ++i) {
            double v = fabs(A[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        if (best < 1e-300) return -1;
        if (p != k) {
            for (int64_t j = 0; j < n; ++j) {
                double t = A[k * n + j];
                A[k * n + j] = A[p * n + j];
                A[p * n + j] = t;
            }
            int64_t tp = piv[k]; piv[k] = piv[p]; piv[p] = tp;
            *sign = -*sign;
        }
        /* eliminate */
        double pivot = A[k * n + k];
        for (int64_t i = k + 1; i < n; ++i) {
            double f = A[i * n + k] / pivot;
            A[i * n + k] = f;  /* L[i,k] stored below diag */
            for (int64_t j = k + 1; j < n; ++j)
                A[i * n + j] -= f * A[k * n + j];
        }
    }
    return 0;
}

/*
 * Solve L*y = P*b then U*x = y, given the in-place LU from lu_decompose.
 * `b` is overwritten with the solution x.
 */
static void lu_solve_column(const double *LU, int64_t n, const int64_t *piv,
                            const double *rhs, double *x) {
    /* apply permutation: y = P * rhs */
    for (int64_t i = 0; i < n; ++i) x[i] = rhs[piv[i]];
    /* forward substitution for L (unit diagonal) */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < i; ++j)
            x[i] -= LU[i * n + j] * x[j];
    /* back substitution for U */
    for (int64_t i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (int64_t j = i + 1; j < n; ++j)
            s -= LU[i * n + j] * x[j];
        x[i] = s / LU[i * n + i];
    }
}

/* inv(A): Gauss-Jordan via LU, solving A*X = I column by column.
 *
 * Phase-4 RAII exemplar — see docs/port_runtime_2_cpp.md. Replaces the
 * four manual malloc/free pairs with std::vector<double>/<int64_t>
 * scratch buffers and a MatPtr for the result descriptor. The two
 * early-exit paths (non-square A; singular factorisation) used to leak
 * LU + piv on the singular path; the RAII version is leak-free by
 * construction. The body still hands the result back as a raw
 * matlab_mat * across the C ABI via .release(), so the symbol shape
 * is unchanged. */
matlab_mat *matlab_inv(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    std::vector<double> LU(A->data, A->data + n * n);
    std::vector<int64_t> piv(n);
    int sign;
    if (lu_decompose(LU.data(), n, piv.data(), &sign) != 0)
        return mat_alloc(0, 0);
    matlab::runtime::MatPtr work = matlab::runtime::make_mat(n, n);
    std::vector<double> rhs(n), col(n);
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t i = 0; i < n; ++i) rhs[i] = (i == c) ? 1.0 : 0.0;
        lu_solve_column(LU.data(), n, piv.data(), rhs.data(), col.data());
        for (int64_t i = 0; i < n; ++i) work->data[i * n + c] = col[i];
    }
    return work.release();
}

/* A \ B: solve A*X = B (MATLAB left divide). B may have multiple columns.
 * Phase-4 RAII migration — same scratch shape as matlab_inv. */
matlab_mat *matlab_mldivide_mm(matlab_mat *A, matlab_mat *B) {
    if (!A || !B || A->rows != A->cols || A->rows != B->rows)
        return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t k = B->cols;
    std::vector<double> LU(A->data, A->data + n * n);
    std::vector<int64_t> piv(n);
    int sign;
    if (lu_decompose(LU.data(), n, piv.data(), &sign) != 0)
        return mat_alloc(0, 0);
    matlab::runtime::MatPtr work = matlab::runtime::make_mat(n, k);
    std::vector<double> rhs(n), col(n);
    for (int64_t c = 0; c < k; ++c) {
        for (int64_t i = 0; i < n; ++i) rhs[i] = B->data[i * k + c];
        lu_solve_column(LU.data(), n, piv.data(), rhs.data(), col.data());
        for (int64_t i = 0; i < n; ++i) work->data[i * k + c] = col[i];
    }
    return work.release();
}

/* A / B = (B' \ A')'. Built on top of mldivide + transpose. */
matlab_mat *matlab_mrdivide_mm(matlab_mat *A, matlab_mat *B) {
    matlab_mat *At = matlab_transpose(A);
    matlab_mat *Bt = matlab_transpose(B);
    matlab_mat *Yt = matlab_mldivide_mm(Bt, At);
    return matlab_transpose(Yt);
    /* At/Bt/Yt are intentionally leaked with the rest of the heap. */
}

/*
 * One-sided Jacobi SVD.
 *
 * Returns a column vector of the min(m,n) singular values of A, sorted in
 * descending order. Works on any m×n matrix. Algorithm:
 *
 *   Maintain a working matrix U (initially a copy of A, possibly
 *   transposed when m<n). Repeatedly sweep over column pairs (p, q) and
 *   apply a plane rotation that makes columns p and q orthogonal:
 *
 *     α = ||U[:,p]||²,  β = ||U[:,q]||²,  γ = <U[:,p], U[:,q]>
 *     ζ = (β - α) / (2γ)
 *     t = sign(ζ) / (|ζ| + sqrt(1 + ζ²))
 *     c = 1/sqrt(1+t²),  s = t·c
 *     U[:,p], U[:,q] ← c·U[:,p] - s·U[:,q],  s·U[:,p] + c·U[:,q]
 *
 *   Convergence is quadratic in the number of sweeps; 30 sweeps are plenty
 *   for any input we've tested.
 *
 *   After convergence, column norms of U are the singular values. Sort
 *   descending for MATLAB's convention.
 *
 * Full [U, S, V] decomposition is a natural extension (accumulate the
 * rotations into V, normalize U's columns), but MATLAB's scalar-return
 * form `s = svd(A)` is the more common call and all we need today.
 */
matlab_mat *matlab_svd(matlab_mat *A_in) {
    int64_t m_orig = A_in->rows, n_orig = A_in->cols;
    int64_t m = m_orig, n = n_orig;
    matlab_mat *A = A_in;
    matlab_mat *T = NULL;
    if (m < n) {
        T = matlab_transpose(A_in);
        A = T;
        m = T->rows;
        n = T->cols;
    }
    /* `U` (m×n) starts as a copy of A.
     * Phase-4 RAII: std::vector replaces the manual U + sv mallocs. */
    std::vector<double> U(A->data, A->data + m * n);

    const double eps = 1e-14;
    const int max_sweeps = 30;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int64_t p = 0; p < n - 1; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int64_t i = 0; i < m; ++i) {
                    double a = U[i * n + p];
                    double b = U[i * n + q];
                    alpha += a * a;
                    beta  += b * b;
                    gamma += a * b;
                }
                off += gamma * gamma;
                double thresh = eps * sqrt(alpha * beta);
                if (fabs(gamma) <= thresh) continue;

                double zeta = (beta - alpha) / (2.0 * gamma);
                double sign_zeta = (zeta >= 0.0) ? 1.0 : -1.0;
                double t = sign_zeta / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                for (int64_t i = 0; i < m; ++i) {
                    double up = U[i * n + p];
                    double uq = U[i * n + q];
                    U[i * n + p] = c * up - s * uq;
                    U[i * n + q] = s * up + c * uq;
                }
            }
        }
        if (off < eps * eps) break;
    }

    /* Singular values = column norms of final U. */
    std::vector<double> sv(n);
    for (int64_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            double v = U[i * n + j];
            s += v * v;
        }
        sv[j] = sqrt(s);
    }
    /* Insertion sort, descending. */
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            if (sv[j] > sv[i]) {
                double t = sv[i]; sv[i] = sv[j]; sv[j] = t;
            }
        }
    }

    int64_t k = (n_orig < m_orig) ? n_orig : m_orig;
    matlab::runtime::MatPtr S = matlab::runtime::make_mat(k, 1);
    for (int64_t i = 0; i < k; ++i) S->data[i] = sv[i];
    (void)T;  /* T is kept alive by the arena-leak policy */
    return S.release();
}

/*
 * Symmetry detection — returns 1 iff A[i,j] ≈ A[j,i] for all i, j with a
 * relative tolerance suited to floating-point round-off. Used by
 * matlab_eig to dispatch between the Jacobi (symmetric) and Francis QR
 * (non-symmetric) paths.
 */
static int matrix_is_symmetric_(const double *A, int64_t n) {
    /* Frobenius norm of A and of (A - A^T)/2 — relative tolerance. */
    double frobA = 0.0, frobAS = 0.0;
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double a = A[i * n + j];
            frobA += a * a;
            double d = a - A[j * n + i];
            frobAS += d * d;
        }
    return frobAS <= 1e-24 * frobA + 1e-300;
}

/*
 * In-place Hessenberg reduction via Householder reflections. Same
 * algorithm as matlab_hess() above; factored here as a static helper
 * so the Francis QR path can reduce its working copy without an
 * intermediate allocation.
 *
 * If U != NULL, U is initialized to the identity and post-multiplied
 * by each Householder. After the call A_orig = U H U' (Schur-form
 * relation). Pass U == NULL when the orthogonal accumulator isn't
 * needed (eig 1-return, matlab_hess).
 */
static void hessenberg_inplace_(double *H, int64_t n, double *U) {
    if (U) {
        for (int64_t i = 0; i < n * n; ++i) U[i] = 0.0;
        for (int64_t i = 0; i < n; ++i) U[i * n + i] = 1.0;
    }
    if (n <= 2) return;
    std::vector<double> v(n);
    for (int64_t k = 0; k + 2 < n; ++k) {
        double sigma = 0.0;
        for (int64_t i = k + 1; i < n; ++i) {
            double x = H[i * n + k];
            sigma += x * x;
        }
        if (sigma == 0.0) continue;
        double xk = H[(k + 1) * n + k];
        double xnorm = sqrt(sigma);
        double v0 = xk + (xk >= 0 ? xnorm : -xnorm);
        v[k + 1] = v0;
        for (int64_t i = k + 2; i < n; ++i) v[i] = H[i * n + k];
        double vnorm2 = v0 * v0 + (sigma - xk * xk);
        if (vnorm2 == 0.0) continue;
        double beta = 2.0 / vnorm2;
        for (int64_t j = k; j < n; ++j) {
            double w = 0.0;
            for (int64_t i = k + 1; i < n; ++i) w += v[i] * H[i * n + j];
            w *= beta;
            for (int64_t i = k + 1; i < n; ++i) H[i * n + j] -= v[i] * w;
        }
        for (int64_t i = 0; i < n; ++i) {
            double w = 0.0;
            for (int64_t j = k + 1; j < n; ++j) w += H[i * n + j] * v[j];
            w *= beta;
            for (int64_t j = k + 1; j < n; ++j) H[i * n + j] -= w * v[j];
        }
        /* Apply Householder from the right to U: U := U * P_k. P_k is
         * symmetric (P = I - beta v v'), so we sweep the same v over
         * U's columns k+1..n-1. */
        if (U) {
            for (int64_t i = 0; i < n; ++i) {
                double w = 0.0;
                for (int64_t j = k + 1; j < n; ++j) w += U[i * n + j] * v[j];
                w *= beta;
                for (int64_t j = k + 1; j < n; ++j) U[i * n + j] -= w * v[j];
            }
        }
        for (int64_t i = k + 2; i < n; ++i) H[i * n + k] = 0.0;
    }
}

/*
 * Francis double-shift implicit QR iteration on an upper-Hessenberg
 * matrix H. Drives H to real Schur form (block upper-triangular with
 * 1*1 and 2*2 diagonal blocks) by applying implicit double shifts and
 * chasing the resulting bulge down the diagonal. Deflation: subdiagonal
 * elements that become "small" are zeroed, splitting the active region.
 *
 * Reference: Golub & Van Loan, "Matrix Computations" (4th ed),
 * Algorithm 7.5.1 (Francis QR Step) + Algorithm 7.5.2 (driver).
 *
 * If U != NULL, post-multiply U by each Householder so that on exit
 * U H_final U' = H_initial — the orthogonal accumulator that gates
 * the schur 2-return form.
 *
 * Returns 0 on success; -1 if the iteration budget is exhausted.
 */
static int francis_qr_(double *H, int64_t n, double *U) {
    if (n <= 1) return 0;
    const int max_total_iter = 30 * (int)n;
    int total_iter = 0;
    /* Active block lives in [p, q]; we shrink it by deflation. */
    int64_t q = n - 1;
    while (q > 0 && total_iter < max_total_iter) {
        /* Find the largest k ∈ [0, q] such that H[k, k-1] is "small" —
         * that's the start of the active block. */
        int64_t p = q;
        while (p > 0) {
            double off = fabs(H[p * n + (p - 1)]);
            double diag = fabs(H[(p - 1) * n + (p - 1)]) + fabs(H[p * n + p]);
            if (off <= 1e-14 * (diag == 0.0 ? 1.0 : diag)) {
                H[p * n + (p - 1)] = 0.0;
                break;
            }
            --p;
        }
        /* Trailing 1*1 block — deflate. */
        if (p == q) { --q; continue; }
        /* Trailing 2*2 block — deflate (eigenvalues extracted later). */
        if (p == q - 1) { q -= 2; continue; }
        ++total_iter;

        /* Wilkinson double-shift from trailing 2*2:
         *   s = trace,  t = det. */
        double s = H[(q - 1) * n + (q - 1)] + H[q * n + q];
        double t = H[(q - 1) * n + (q - 1)] * H[q * n + q] -
                   H[(q - 1) * n + q] * H[q * n + (q - 1)];

        /* Exceptional "ad-hoc" shift every 10 iterations to avoid cycling
         * on stagnant matrices (Wilkinson's perturbation trick). */
        if (total_iter % 10 == 0) {
            double scale = fabs(H[q * n + (q - 1)]) +
                           fabs(H[(q - 1) * n + (q - 2)]);
            s = 1.5 * scale;
            t = scale * scale;
        }

        /* First column of (H - lambda1 I)(H - lambda2 I) where the two
         * shifts have sum s and product t, evaluated at row p. */
        double Hpp  = H[p * n + p];
        double Hpp1 = H[p * n + (p + 1)];
        double Hp1p = H[(p + 1) * n + p];
        double Hp1  = H[(p + 1) * n + (p + 1)];
        double x = Hpp * Hpp + Hpp1 * Hp1p - s * Hpp + t;
        double y = Hp1p * (Hpp + Hp1 - s);
        double z = Hp1p * H[(p + 2) * n + (p + 1)];

        /* Implicit Q step: Householder on (x, y, z) introduces a bulge,
         * then we chase it down the diagonal back to Hessenberg form. */
        for (int64_t k = p; k + 1 <= q; ++k) {
            /* Build 3-element (or 2-element near the bottom) Householder.
             * r == 3 when row k+2 is still inside [p..q]; r == 2 at the
             * final chase step where only rows {k, k+1} fit. */
            int64_t r = (k + 2 <= q) ? 3 : 2;
            double v0, v1, v2 = 0.0;
            if (k > p) {
                v0 = H[k * n + (k - 1)];
                v1 = H[(k + 1) * n + (k - 1)];
                v2 = (r == 3) ? H[(k + 2) * n + (k - 1)] : 0.0;
            } else {
                v0 = x; v1 = y; v2 = z;
            }
            double sig = v0 * v0 + v1 * v1 + v2 * v2;
            if (sig == 0.0) continue;
            double xnorm = sqrt(sig);
            double v0p = v0 + (v0 >= 0 ? xnorm : -xnorm);
            double vnorm2 = v0p * v0p + v1 * v1 + v2 * v2;
            if (vnorm2 == 0.0) continue;
            double beta = 2.0 / vnorm2;
            /* Apply reflection from the LEFT to rows {k, k+1, k+2}. */
            int64_t row_lim_lo = (k > p) ? (k - 1) : p;
            for (int64_t j = row_lim_lo; j < n; ++j) {
                double w = v0p * H[k * n + j] + v1 * H[(k + 1) * n + j];
                if (r == 3) w += v2 * H[(k + 2) * n + j];
                w *= beta;
                H[k * n + j]       -= v0p * w;
                H[(k + 1) * n + j] -= v1  * w;
                if (r == 3) H[(k + 2) * n + j] -= v2 * w;
            }
            /* Apply from the RIGHT to columns {k, k+1, k+2}. */
            int64_t col_lim_hi = (k + r + 1 < n) ? (k + r + 1) : n;
            for (int64_t i = 0; i < col_lim_hi; ++i) {
                double w = v0p * H[i * n + k] + v1 * H[i * n + (k + 1)];
                if (r == 3) w += v2 * H[i * n + (k + 2)];
                w *= beta;
                H[i * n + k]       -= v0p * w;
                H[i * n + (k + 1)] -= v1  * w;
                if (r == 3) H[i * n + (k + 2)] -= v2 * w;
            }
            /* Apply from the RIGHT to U over all n rows. */
            if (U) {
                for (int64_t i = 0; i < n; ++i) {
                    double w = v0p * U[i * n + k] + v1 * U[i * n + (k + 1)];
                    if (r == 3) w += v2 * U[i * n + (k + 2)];
                    w *= beta;
                    U[i * n + k]       -= v0p * w;
                    U[i * n + (k + 1)] -= v1  * w;
                    if (r == 3) U[i * n + (k + 2)] -= v2 * w;
                }
            }
        }
    }
    return total_iter < max_total_iter ? 0 : -1;
}

/*
 * Extract eigenvalues from a converged real Schur form. Walks the
 * diagonal: a 1*1 block (zero subdiagonal below) is a real eigenvalue;
 * a 2*2 block carries either two real eigenvalues or a complex conjugate
 * pair, depending on the discriminant of the 2*2 characteristic polynomial.
 *
 * Fills eig_re[k], eig_im[k] for k in [0, n) and returns the number of
 * complex pairs found (each contributes a non-zero imag part).
 */
static int extract_eigenvalues_(const double *H, int64_t n,
                                double *eig_re, double *eig_im) {
    int complex_pairs = 0;
    int64_t i = 0;
    while (i < n) {
        int is_2x2 = (i + 1 < n) && (H[(i + 1) * n + i] != 0.0);
        if (!is_2x2) {
            eig_re[i] = H[i * n + i];
            eig_im[i] = 0.0;
            ++i;
            continue;
        }
        double a = H[i * n + i];
        double b = H[i * n + (i + 1)];
        double c = H[(i + 1) * n + i];
        double d = H[(i + 1) * n + (i + 1)];
        double tr  = a + d;
        double det = a * d - b * c;
        double disc = tr * tr - 4.0 * det;
        if (disc >= 0.0) {
            double sq = sqrt(disc);
            eig_re[i]     = (tr + sq) * 0.5;  eig_im[i]     = 0.0;
            eig_re[i + 1] = (tr - sq) * 0.5;  eig_im[i + 1] = 0.0;
        } else {
            double sq = sqrt(-disc) * 0.5;
            eig_re[i]     = tr * 0.5;  eig_im[i]     = sq;
            eig_re[i + 1] = tr * 0.5;  eig_im[i + 1] = -sq;
            ++complex_pairs;
        }
        i += 2;
    }
    return complex_pairs;
}

matlab_mat *matlab_eig(matlab_mat *A_in) {
    if (!A_in || A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;

    /* Non-symmetric path — Francis double-shift QR on Hessenberg form.
     * Returns matlab_mat* (real) when all eigenvalues are real, or
     * matlab_mat_c* (cast back to matlab_mat*) when any complex pair
     * exists. The polymorphism rides on the magic-word convention used
     * elsewhere in the runtime (mat_is_complex). */
    if (n > 0 && !matrix_is_symmetric_(A_in->data, n)) {
        std::vector<double> H(A_in->data, A_in->data + n * n);
        hessenberg_inplace_(H.data(), n, /*U=*/nullptr);
        francis_qr_(H.data(), n, /*U=*/nullptr);
        std::vector<double> ere(n), eim(n);
        int complex_pairs = extract_eigenvalues_(H.data(), n,
                                                 ere.data(), eim.data());
        /* Sort eigenvalues by ascending real part, tie-break by imag.
         * Insertion sort — n is small for typical control plants. */
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                bool swap = (ere[j] < ere[i]) ||
                            (ere[j] == ere[i] && eim[j] < eim[i]);
                if (swap) {
                    double t;
                    t = ere[i]; ere[i] = ere[j]; ere[j] = t;
                    t = eim[i]; eim[i] = eim[j]; eim[j] = t;
                }
            }
        }
        if (complex_pairs == 0) {
            matlab::runtime::MatPtr E = matlab::runtime::make_mat(n, 1);
            for (int64_t i = 0; i < n; ++i) E->data[i] = ere[i];
            return E.release();
        }
        matlab_mat_c *Ec = mat_c_alloc(n, 1);
        for (int64_t i = 0; i < n; ++i) {
            Ec->re[i] = ere[i];
            Ec->im[i] = eim[i];
        }
        return (matlab_mat *)Ec;
    }

    /* Symmetric path — Jacobi sweep (unchanged). */
    /* Phase-4 RAII: H scratch buffer holds the symmetric working matrix. */
    std::vector<double> H(n * n);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            H[i * n + j] = 0.5 * (A_in->data[i * n + j] +
                                  A_in->data[j * n + i]);
        }
    }

    const double eps = 1e-14;
    const int max_sweeps = 50;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int64_t p = 0; p < n - 1; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                double Apq = H[p * n + q];
                off += Apq * Apq;
                if (fabs(Apq) < eps) continue;

                double App = H[p * n + p];
                double Aqq = H[q * n + q];
                double tau = (Aqq - App) / (2.0 * Apq);
                double sign_tau = (tau >= 0.0) ? 1.0 : -1.0;
                double t = sign_tau / (fabs(tau) + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                /* Diagonal update and zero the target element. */
                H[p * n + p] = App - t * Apq;
                H[q * n + q] = Aqq + t * Apq;
                H[p * n + q] = 0.0;
                H[q * n + p] = 0.0;

                /* Rotate rows/cols p and q for i ∉ {p, q}. */
                for (int64_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    double Aip = H[i * n + p];
                    double Aiq = H[i * n + q];
                    double Ip = c * Aip - s * Aiq;
                    double Iq = s * Aip + c * Aiq;
                    H[i * n + p] = Ip;
                    H[i * n + q] = Iq;
                    H[p * n + i] = Ip;
                    H[q * n + i] = Iq;
                }
            }
        }
        if (off < eps * eps) break;
    }

    matlab::runtime::MatPtr E = matlab::runtime::make_mat(n, 1);
    for (int64_t i = 0; i < n; ++i) E->data[i] = H[i * n + i];
    /* Insertion sort, ascending. */
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            if (E->data[j] < E->data[i]) {
                double t = E->data[i]; E->data[i] = E->data[j]; E->data[j] = t;
            }
        }
    }
    return E.release();
}

/* Two-return eig: [V, D] = eig(A). V has eigenvectors as columns,
 * D is a diagonal matrix of eigenvalues (ascending). Outputs packed
 * into a single heap struct that the frontend decomposes; we simply
 * expose two independent entry points that share the same Jacobi sweep.
 *
 * Both V and D arrive allocated internally and returned via out-params.
 * The frontend calls matlab_eig_V and matlab_eig_D separately when
 * nargout==2; each walks the full Jacobi sweep on its own copy so the
 * two calls are independent (simple and correct, if a bit redundant). */

/* Jacobi sweep producing eigenvalues AND eigenvectors in column-major
 * V (same shape as A). */
static void jacobi_sym(matlab_mat *A_in, double *eigvals, double *V) {
    int64_t n = A_in->rows;
    /* Phase-4 RAII: H scratch holds the symmetric working matrix. */
    std::vector<double> H(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            H[i * n + j] = 0.5 * (A_in->data[i * n + j] +
                                  A_in->data[j * n + i]);
    /* V starts as identity. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            V[i * n + j] = (i == j) ? 1.0 : 0.0;

    const double eps = 1e-14;
    const int max_sweeps = 50;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int64_t p = 0; p < n - 1; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                double Apq = H[p * n + q];
                off += Apq * Apq;
                if (fabs(Apq) < eps) continue;
                double App = H[p * n + p];
                double Aqq = H[q * n + q];
                double tau = (Aqq - App) / (2.0 * Apq);
                double sign_tau = (tau >= 0.0) ? 1.0 : -1.0;
                double t = sign_tau / (fabs(tau) + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;
                H[p * n + p] = App - t * Apq;
                H[q * n + q] = Aqq + t * Apq;
                H[p * n + q] = 0.0;
                H[q * n + p] = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    double Aip = H[i * n + p];
                    double Aiq = H[i * n + q];
                    H[i * n + p] = c * Aip - s * Aiq;
                    H[i * n + q] = s * Aip + c * Aiq;
                    H[p * n + i] = H[i * n + p];
                    H[q * n + i] = H[i * n + q];
                }
                /* Rotate V's columns p, q. */
                for (int64_t i = 0; i < n; ++i) {
                    double Vip = V[i * n + p];
                    double Viq = V[i * n + q];
                    V[i * n + p] = c * Vip - s * Viq;
                    V[i * n + q] = s * Vip + c * Viq;
                }
            }
        }
        if (off < eps * eps) break;
    }
    for (int64_t i = 0; i < n; ++i) eigvals[i] = H[i * n + i];
}

/* Internal helper for the 2-return non-symmetric `[V, D] = eig(A)`
 * shape. Computes the real Schur form U' A U = T (with the orthogonal
 * accumulator U), then back-substitutes T y_i = λ_i y_i for each
 * eigenvalue and recovers v_i = U y_i. v1 path handles the all-real-
 * eigenvalues case (1×1 Schur blocks only); a 2×2 quasi-triangular
 * block (complex conjugate pair) flips the `has_complex` flag and the
 * caller falls back to returning 0×0 — proper complex eigenvector
 * back-substitution is the follow-on. Eigenvalues are stored in
 * Schur-diagonal order (no post-sort), so V's column k matches the
 * (k, k) entry of the corresponding D. */
static int eig_VD_real_(matlab_mat *A_in, double *V_out, double *D_out) {
    int64_t n = A_in->rows;
    std::vector<double> T(A_in->data, A_in->data + n * n);
    std::vector<double> U(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) U[i * n + i] = 1.0;
    hessenberg_inplace_(T.data(), n, U.data());
    francis_qr_(T.data(), n, U.data());

    /* Detect any 2×2 quasi-triangular blocks (complex eigenvalue
     * pairs); v1 doesn't compute their eigenvectors. */
    for (int64_t i = 0; i + 1 < n; ++i)
        if (T[(i + 1) * n + i] != 0.0) return 0;

    /* Eigenvalues sit on the diagonal — copy them into D's diagonal. */
    for (int64_t i = 0; i < n * n; ++i) D_out[i] = 0.0;
    for (int64_t i = 0; i < n; ++i) D_out[i * n + i] = T[i * n + i];

    /* For each eigenvalue λ_i = T[i,i], solve (T - λ_i I) y = 0 by
     * back-substitution on the upper-triangular slice T[0..i, 0..i].
     * Set y[i] = 1, y[k] = 0 for k > i, then for k = i-1..0:
     *   (T[k,k] - λ) y[k] = -Σ_{m=k+1}^{i} T[k,m] y[m]
     *   y[k] = (Σ_{m=k+1}^{i} T[k,m] y[m]) / (λ - T[k,k]).
     * If λ - T[k,k] is too small (Schur diagonal has a near-repeated
     * eigenvalue), set y[k] = 0 and continue — the resulting V column
     * will be defective but still proportional to a valid eigenvector
     * for the dominant eigenvalue. */
    std::vector<double> y(n);
    for (int64_t i = 0; i < n; ++i) {
        double lambda = T[i * n + i];
        for (int64_t k = i + 1; k < n; ++k) y[k] = 0.0;
        y[i] = 1.0;
        for (int64_t k = i; k-- > 0;) {
            double s = 0.0;
            for (int64_t m = k + 1; m <= i; ++m)
                s += T[k * n + m] * y[m];
            double denom = lambda - T[k * n + k];
            if (std::fabs(denom) < 1e-14 * (std::fabs(lambda) + 1.0))
                y[k] = 0.0;
            else
                y[k] = s / denom;
        }
        /* v_i = U · y, but only the leading i+1 components of y are
         * non-zero, so the column accumulates U[:, 0..i] · y[0..i]. */
        for (int64_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (int64_t k = 0; k <= i; ++k) s += U[r * n + k] * y[k];
            V_out[r * n + i] = s;
        }
        /* Normalise the column to unit 2-norm for a stable answer
         * (MATLAB's eig also returns unit-norm eigenvectors). */
        double col_nrm = 0.0;
        for (int64_t r = 0; r < n; ++r)
            col_nrm += V_out[r * n + i] * V_out[r * n + i];
        col_nrm = std::sqrt(col_nrm);
        if (col_nrm > 0.0)
            for (int64_t r = 0; r < n; ++r) V_out[r * n + i] /= col_nrm;
    }
    return 1;
}

/* matlab_eig_V(A): eigenvector matrix (columns = eigenvectors).
 * Symmetric path orders columns by ascending eigenvalue (Jacobi
 * convention). Non-symmetric path keeps Schur-diagonal order so the
 * companion matlab_eig_D matches column-for-column. Returns 0×0 when
 * A has complex eigenvalue pairs (those need complex back-
 * substitution; deferred). */
matlab_mat *matlab_eig_V(matlab_mat *A_in) {
    if (!A_in || A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
    if (n == 0) return mat_alloc(0, 0);
    if (!matrix_is_symmetric_(A_in->data, n)) {
        matlab::runtime::MatPtr V = matlab::runtime::make_mat(n, n);
        std::vector<double> Dscratch(n * n);
        if (!eig_VD_real_(A_in, V->data, Dscratch.data()))
            return mat_alloc(0, 0);
        return V.release();
    }
    std::vector<double> eigvals(n);
    matlab::runtime::MatPtr V = matlab::runtime::make_mat(n, n);
    jacobi_sym(A_in, eigvals.data(), V->data);
    /* Sort V's columns by ascending eigvals (insertion sort). */
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            if (eigvals[j] < eigvals[i]) {
                double t = eigvals[i]; eigvals[i] = eigvals[j]; eigvals[j] = t;
                for (int64_t r = 0; r < n; ++r) {
                    double tmp = V->data[r * n + i];
                    V->data[r * n + i] = V->data[r * n + j];
                    V->data[r * n + j] = tmp;
                }
            }
        }
    }
    return V.release();
}

/* matlab_eig_D(A): diagonal matrix of eigenvalues. Symmetric path:
 * ascending order. Non-symmetric path: Schur-diagonal order (matches
 * matlab_eig_V's column order). 0×0 when complex pairs are present. */
matlab_mat *matlab_eig_D(matlab_mat *A_in) {
    if (!A_in || A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
    if (n == 0) return mat_alloc(0, 0);
    if (!matrix_is_symmetric_(A_in->data, n)) {
        matlab::runtime::MatPtr D = matlab::runtime::make_mat(n, n);
        std::vector<double> Vscratch(n * n);
        if (!eig_VD_real_(A_in, Vscratch.data(), D->data))
            return mat_alloc(0, 0);
        return D.release();
    }
    std::vector<double> eigvals(n);
    std::vector<double> Vtmp(n * n);
    jacobi_sym(A_in, eigvals.data(), Vtmp.data());
    /* Ascending sort of eigvals. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j)
            if (eigvals[j] < eigvals[i]) {
                double t = eigvals[i]; eigvals[i] = eigvals[j]; eigvals[j] = t;
            }
    matlab::runtime::MatPtr D = matlab::runtime::make_mat(n, n);
    for (int64_t i = 0; i < n; ++i) D->data[i * n + i] = eigvals[i];
    return D.release();
}

/* det(A): product of LU diagonal times permutation sign. */
double matlab_det(matlab_mat *A) {
    if (A->rows != A->cols) return 0.0;
    int64_t n = A->rows;
    double *LU = (double *)malloc((size_t)(n * n) * sizeof(double));
    memcpy(LU, A->data, (size_t)(n * n) * sizeof(double));
    int64_t *piv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int sign;
    double d;
    if (lu_decompose(LU, n, piv, &sign) != 0) {
        d = 0.0;
    } else {
        d = (double)sign;
        for (int64_t i = 0; i < n; ++i) d *= LU[i * n + i];
    }
    free(LU); free(piv);
    return d;
}

/*---------- Shape operations ----------------------------------------------*/

/* Phase-5: shape-op template — see runtime_internal.h. */

matlab_mat *matlab_transpose(matlab_mat *A) {
    int64_t m = A->rows, n = A->cols;
    return matlab::runtime::shape_op(n, m, [&](int64_t i, int64_t j) {
        return A->data[j * n + i];
    }).release();
}

/* diag(A): if A is a row or column vector, build an n×n matrix with A on
 * the main diagonal. Otherwise extract the main diagonal as a column.
 * Vector→matrix path is sparse (only diagonal cells nonzero); the
 * matrix→vector path is short and direct. Neither fits shape_op
 * cleanly so they stay as-is. */
matlab_mat *matlab_diag(matlab_mat *A) {
    if (A->rows == 1 || A->cols == 1) {
        int64_t n = A->rows * A->cols;
        matlab::runtime::MatPtr D = matlab::runtime::make_mat(n, n);
        for (int64_t i = 0; i < n; ++i) D->data[i * n + i] = A->data[i];
        return D.release();
    }
    int64_t d = A->rows < A->cols ? A->rows : A->cols;
    matlab::runtime::MatPtr V = matlab::runtime::make_mat(d, 1);
    for (int64_t i = 0; i < d; ++i) V->data[i] = A->data[i * A->cols + i];
    return V.release();
}

matlab_mat *matlab_reshape(matlab_mat *A, double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    if (rm * cn != A->rows * A->cols) return mat_alloc(0, 0);
    matlab_mat *B = mat_alloc(rm, cn);
    memcpy(B->data, A->data, (size_t)(rm * cn) * sizeof(double));
    return B;
}

/* Range: start:step:end materializes as a 1×N row vector. */
matlab_mat *matlab_range(double start, double step, double end) {
    if (step == 0.0) return mat_alloc(0, 0);
    double diff = end - start;
    if ((step > 0 && diff < 0) || (step < 0 && diff > 0))
        return mat_alloc(1, 0);
    int64_t n = (int64_t)(diff / step) + 1;
    matlab_mat *A = mat_alloc(1, n);
    for (int64_t i = 0; i < n; ++i) A->data[i] = start + (double)i * step;
    return A;
}

/* Phase-5: shape_op brings the 4-deep tile loop down to one
 * modulo-arithmetic lambda. */
matlab_mat *matlab_repmat(matlab_mat *A, double m, double n) {
    int64_t tm = (int64_t)m, tn = (int64_t)n;
    int64_t am = A->rows, an = A->cols;
    int64_t nr = am * tm, nc = an * tn;
    return matlab::runtime::shape_op(nr, nc, [&](int64_t i, int64_t j) {
        return A->data[(i % am) * an + (j % an)];
    }).release();
}

/*---------- Reductions ----------------------------------------------------
 *
 * MATLAB's rule for sum/min/max/mean/prod on a plain `A`:
 *   - If A is a row or column vector → reduce to a scalar (1×1 matrix).
 *   - Otherwise → column-wise reduction, result is a 1×N row vector.
 *
 *--------------------------------------------------------------------------*/

/* If A is a vector, reduce the flat sequence into a 1×1. Otherwise apply
 * `col_init` to each column and fold with `op`. The init lambdas and
 * ops are passed as macros so the resulting code inlines cleanly. */
#define COLWISE_REDUCE(NAME, INIT_EXPR, UPDATE_EXPR, FINALIZE_EXPR)       \
    matlab_mat *matlab_##NAME(matlab_mat *A) {                            \
        int64_t m = A->rows, n = A->cols;                                 \
        if (m <= 1 || n == 1) {                                           \
            int64_t total = m * n;                                        \
            double acc = INIT_EXPR;                                       \
            for (int64_t k = 0; k < total; ++k) {                         \
                double x = A->data[k];                                    \
                acc = UPDATE_EXPR;                                        \
            }                                                             \
            double result = FINALIZE_EXPR;                                \
            matlab_mat *R = mat_alloc(1, 1);                              \
            R->data[0] = total > 0 ? result : 0.0;                        \
            return R;                                                     \
        }                                                                 \
        matlab_mat *R = mat_alloc(1, n);                                  \
        for (int64_t j = 0; j < n; ++j) {                                 \
            double acc = INIT_EXPR;                                       \
            int64_t total = m;                                            \
            (void)total; /* used only by mean's FINALIZE_EXPR */          \
            for (int64_t i = 0; i < m; ++i) {                             \
                double x = A->data[i * n + j];                            \
                acc = UPDATE_EXPR;                                        \
            }                                                             \
            R->data[j] = FINALIZE_EXPR;                                   \
        }                                                                 \
        return R;                                                         \
    }

COLWISE_REDUCE(sum,  0.0,       acc + x,                    acc)
COLWISE_REDUCE(prod, 1.0,       acc * x,                    acc)
COLWISE_REDUCE(mean, 0.0,       acc + x,                    acc / (double)total)
COLWISE_REDUCE(min,  INFINITY,  (x < acc ? x : acc),        acc)
COLWISE_REDUCE(max, -INFINITY,  (x > acc ? x : acc),        acc)

#undef COLWISE_REDUCE

/* Dimension-aware reductions: sum(A, dim) etc.
 * dim==1 collapses rows (result has 1 row, A->cols cols);
 * dim==2 collapses cols (result has A->rows rows, 1 col).
 * Any other dim just returns a 1×1 with the grand total. */
#define DIM_REDUCE(NAME, INIT_EXPR, UPDATE_EXPR, FINALIZE_EXPR)           \
    matlab_mat *matlab_##NAME##_dim(matlab_mat *A, double d) {            \
        if (!A) return mat_alloc(0, 0);                                    \
        int64_t m = A->rows, n = A->cols;                                  \
        int64_t dim = (int64_t)d;                                          \
        if (dim == 1) {                                                    \
            matlab_mat *R = mat_alloc(1, n);                               \
            for (int64_t j = 0; j < n; ++j) {                              \
                double acc = INIT_EXPR;                                    \
                int64_t total = m;                                         \
                for (int64_t i = 0; i < m; ++i) {                          \
                    double x = A->data[i * n + j];                         \
                    acc = UPDATE_EXPR;                                     \
                }                                                          \
                R->data[j] = total > 0 ? (FINALIZE_EXPR) : INIT_EXPR;      \
            }                                                              \
            return R;                                                      \
        }                                                                  \
        if (dim == 2) {                                                    \
            matlab_mat *R = mat_alloc(m, 1);                               \
            for (int64_t i = 0; i < m; ++i) {                              \
                double acc = INIT_EXPR;                                    \
                int64_t total = n;                                         \
                for (int64_t j = 0; j < n; ++j) {                          \
                    double x = A->data[i * n + j];                         \
                    acc = UPDATE_EXPR;                                     \
                }                                                          \
                R->data[i] = total > 0 ? (FINALIZE_EXPR) : INIT_EXPR;      \
            }                                                              \
            return R;                                                      \
        }                                                                  \
        /* Fallback: treat as flat */                                      \
        int64_t total = m * n;                                             \
        double acc = INIT_EXPR;                                            \
        for (int64_t k = 0; k < total; ++k) {                              \
            double x = A->data[k];                                         \
            acc = UPDATE_EXPR;                                             \
        }                                                                  \
        matlab_mat *R = mat_alloc(1, 1);                                   \
        R->data[0] = total > 0 ? (FINALIZE_EXPR) : 0.0;                    \
        return R;                                                          \
    }

DIM_REDUCE(sum,  0.0,       acc + x,             acc)
DIM_REDUCE(prod, 1.0,       acc * x,             acc)
DIM_REDUCE(mean, 0.0,       acc + x,             acc / (double)total)
DIM_REDUCE(min,  INFINITY,  (x < acc ? x : acc), acc)
DIM_REDUCE(max, -INFINITY,  (x > acc ? x : acc), acc)

#undef DIM_REDUCE

/* Cumulative scans: along dim 1 by default (or along the only axis
 * when A is a row / column vector). Output has the same shape as A. */
#define CUM_SCAN(NAME, INIT_EXPR, UPDATE_EXPR)                            \
    matlab_mat *matlab_##NAME(matlab_mat *A) {                            \
        if (!A) return mat_alloc(0, 0);                                    \
        int64_t m = A->rows, n = A->cols;                                  \
        matlab_mat *R = mat_alloc(m, n);                                   \
        if (m == 1 || n == 1) {                                            \
            int64_t total = m * n;                                         \
            double acc = INIT_EXPR;                                        \
            for (int64_t k = 0; k < total; ++k) {                          \
                double x = A->data[k]; acc = UPDATE_EXPR;                  \
                R->data[k] = acc;                                          \
            }                                                              \
            return R;                                                      \
        }                                                                  \
        for (int64_t j = 0; j < n; ++j) {                                  \
            double acc = INIT_EXPR;                                        \
            for (int64_t i = 0; i < m; ++i) {                              \
                double x = A->data[i * n + j]; acc = UPDATE_EXPR;          \
                R->data[i * n + j] = acc;                                  \
            }                                                              \
        }                                                                  \
        return R;                                                          \
    }                                                                      \
    matlab_mat *matlab_##NAME##_dim(matlab_mat *A, double d) {             \
        if (!A) return mat_alloc(0, 0);                                    \
        int64_t m = A->rows, n = A->cols;                                  \
        int64_t dim = (int64_t)d;                                          \
        matlab_mat *R = mat_alloc(m, n);                                   \
        if (dim == 2) {                                                    \
            for (int64_t i = 0; i < m; ++i) {                              \
                double acc = INIT_EXPR;                                    \
                for (int64_t j = 0; j < n; ++j) {                          \
                    double x = A->data[i * n + j]; acc = UPDATE_EXPR;      \
                    R->data[i * n + j] = acc;                              \
                }                                                          \
            }                                                              \
            return R;                                                      \
        }                                                                  \
        /* dim==1 or anything else: column scans */                        \
        for (int64_t j = 0; j < n; ++j) {                                  \
            double acc = INIT_EXPR;                                        \
            for (int64_t i = 0; i < m; ++i) {                              \
                double x = A->data[i * n + j]; acc = UPDATE_EXPR;          \
                R->data[i * n + j] = acc;                                  \
            }                                                              \
        }                                                                  \
        return R;                                                          \
    }

CUM_SCAN(cumsum,  0.0, acc + x)
CUM_SCAN(cumprod, 1.0, acc * x)

#undef CUM_SCAN

/* -------- Sort / unique / set operations ----------------------------------
 *
 * All operate on doubles using the natural ordering. Results preserve
 * the shape of vectors: a 1×N input produces a 1×N result, an N×1
 * input produces an N×1 result. For 2-D matrices sort operates
 * column-wise by default (matching MATLAB's sort(A) behavior on
 * matrices).
 *--------------------------------------------------------------------------*/

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

matlab_mat *matlab_sort(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    if (m == 1 || n == 1) {
        int64_t total = m * n;
        memcpy(R->data, A->data, (size_t)total * sizeof(double));
        qsort(R->data, (size_t)total, sizeof(double), cmp_double_asc);
        return R;
    }
    /* Column-wise sort for matrices. Allocate a per-column buffer
     * once outside the loop and reuse it for every column. */
    double *col = (double *)malloc((size_t)m * sizeof(double));
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) col[i] = A->data[i * n + j];
        qsort(col, (size_t)m, sizeof(double), cmp_double_asc);
        for (int64_t i = 0; i < m; ++i) R->data[i * n + j] = col[i];
    }
    free(col);
    return R;
}

matlab_mat *matlab_unique(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t total = A->rows * A->cols;
    if (total == 0) return mat_alloc(0, 0);
    double *tmp = (double *)malloc((size_t)total * sizeof(double));
    memcpy(tmp, A->data, (size_t)total * sizeof(double));
    qsort(tmp, (size_t)total, sizeof(double), cmp_double_asc);
    int64_t u = 0;
    for (int64_t k = 0; k < total; ++k) {
        if (u == 0 || tmp[u - 1] != tmp[k]) tmp[u++] = tmp[k];
    }
    /* Preserve column-vector shape when input was a column, otherwise
     * return a row vector. MATLAB's default is column for all
     * unique() results; we keep a column to match that. */
    matlab_mat *R = mat_alloc(u, 1);
    memcpy(R->data, tmp, (size_t)u * sizeof(double));
    free(tmp);
    return R;
}

static int has_value(const double *xs, int64_t n, double v) {
    for (int64_t i = 0; i < n; ++i) if (xs[i] == v) return 1;
    return 0;
}

matlab_mat *matlab_ismember(matlab_mat *A, matlab_mat *B) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    int64_t bn = B ? B->rows * B->cols : 0;
    for (int64_t k = 0; k < m * n; ++k)
        R->data[k] = has_value(B ? B->data : NULL, bn, A->data[k]) ? 1.0 : 0.0;
    return R;
}

static matlab_mat *set_op(matlab_mat *A, matlab_mat *B, int op /*0=diff,1=inter,2=union*/) {
    int64_t an = A ? A->rows * A->cols : 0;
    int64_t bn = B ? B->rows * B->cols : 0;
    int64_t cap = an + bn;
    /* Phase-4 RAII: scratch tmp buffer; auto-freed on every return. */
    std::vector<double> tmp(cap);
    int64_t u = 0;
    if (op == 0 /* setdiff */) {
        for (int64_t i = 0; i < an; ++i)
            if (!has_value(B ? B->data : NULL, bn, A->data[i]))
                tmp[u++] = A->data[i];
    } else if (op == 1 /* intersect */) {
        for (int64_t i = 0; i < an; ++i)
            if (has_value(B ? B->data : NULL, bn, A->data[i]))
                tmp[u++] = A->data[i];
    } else /* union */ {
        for (int64_t i = 0; i < an; ++i) tmp[u++] = A->data[i];
        for (int64_t j = 0; j < bn; ++j) tmp[u++] = B->data[j];
    }
    /* Sort + dedupe to match MATLAB's unique-and-sorted output. */
    if (u > 0) {
        qsort(tmp.data(), (size_t)u, sizeof(double), cmp_double_asc);
        int64_t uu = 0;
        for (int64_t k = 0; k < u; ++k)
            if (uu == 0 || tmp[uu - 1] != tmp[k]) tmp[uu++] = tmp[k];
        u = uu;
    }
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(u, 1);
    if (u > 0) memcpy(R->data, tmp.data(), (size_t)u * sizeof(double));
    return R.release();
}

matlab_mat *matlab_setdiff(matlab_mat *A, matlab_mat *B)   { return set_op(A, B, 0); }
matlab_mat *matlab_intersect(matlab_mat *A, matlab_mat *B) { return set_op(A, B, 1); }
matlab_mat *matlab_union(matlab_mat *A, matlab_mat *B)     { return set_op(A, B, 2); }

/* sortrows(A): stable lexicographic sort on rows. Column priority is
 * left-to-right: compare row[0][0] vs row[1][0], tie-break on [1], etc. */
static const matlab_mat *sortrows_ctx;
static int cmp_row_lex(const void *a, const void *b) {
    int64_t ia = *(const int64_t *)a, ib = *(const int64_t *)b;
    const matlab_mat *M = sortrows_ctx;
    for (int64_t j = 0; j < M->cols; ++j) {
        double xa = M->data[ia * M->cols + j];
        double xb = M->data[ib * M->cols + j];
        if (xa < xb) return -1;
        if (xa > xb) return 1;
    }
    return 0;
}

matlab_mat *matlab_sortrows(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    int64_t *idx = (int64_t *)malloc((size_t)m * sizeof(int64_t));
    for (int64_t i = 0; i < m; ++i) idx[i] = i;
    sortrows_ctx = A;
    qsort(idx, (size_t)m, sizeof(int64_t), cmp_row_lex);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            R->data[i * n + j] = A->data[idx[i] * n + j];
    free(idx);
    return R;
}

/* -------- Reshape / layout tail ----------------------------------
 * horzcat / vertcat (as builtins, distinct from the [A B] / [A;B]
 * literal paths which the frontend already lowers via concat_row /
 * concat_col); permute / squeeze (2-D no-ops for most cases);
 * flip family; rot90.
 *------------------------------------------------------------------*/

/* Phase-4: horzcat/vertcat adopt MatPtr for RAII consistency. */
matlab_mat *matlab_horzcat(matlab_mat *A, matlab_mat *B) {
    if (!A) return B;
    if (!B) return A;
    if (A->rows != B->rows) return mat_alloc(0, 0);
    int64_t m = A->rows, na = A->cols, nb = B->cols;
    matlab_mat *_a = A, *_b = B;
    return matlab::runtime::shape_op(m, na + nb,
        [&](int64_t i, int64_t j) {
            return j < na ? _a->data[i * na + j]
                          : _b->data[i * nb + (j - na)];
        }).release();
}

matlab_mat *matlab_vertcat(matlab_mat *A, matlab_mat *B) {
    if (!A) return B;
    if (!B) return A;
    if (A->cols != B->cols) return mat_alloc(0, 0);
    int64_t n = A->cols, ma = A->rows, mb = B->rows;
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(ma + mb, n);
    memcpy(R->data,            A->data, (size_t)(ma * n) * sizeof(double));
    memcpy(R->data + ma * n,   B->data, (size_t)(mb * n) * sizeof(double));
    return R.release();
}

/* permute(A, [p1 p2]) for 2-D matrices. p = [1 2] is identity;
 * anything else falls back to transpose (which matches p = [2 1]).
 * Higher-rank permutations aren't modelled because we don't carry
 * N-D shape through matlab_mat. */
matlab_mat *matlab_permute(matlab_mat *A, matlab_mat *perm) {
    if (!A || !perm) return mat_alloc(0, 0);
    int64_t total = perm->rows * perm->cols;
    int Identity = (total >= 2 &&
                    perm->data[0] == 1.0 && perm->data[1] == 2.0);
    if (Identity) {
        /* Phase-5 RAII: identity copy via MatPtr. */
        matlab::runtime::MatPtr R =
            matlab::runtime::make_mat(A->rows, A->cols);
        memcpy(R->data, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
        return R.release();
    }
    return matlab_transpose(A);
}

/* squeeze(A) is a no-op for 2-D matrices — MATLAB's squeeze only
 * collapses singleton dims in higher-rank arrays, which we don't
 * model. Keeps the name available as a syntactic identity. */
matlab_mat *matlab_squeeze(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    /* Phase-5 RAII: pure copy. shape_op fits but is no faster than
     * memcpy here, so keep memcpy and just adopt MatPtr. */
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(m, n);
    memcpy(R->data, A->data, (size_t)(m * n) * sizeof(double));
    return R.release();
}

/* Phase-5: fliplr / flipud / rot90 collapse into one-line lambdas
 * via the shape_op template (runtime_internal.h). */
matlab_mat *matlab_fliplr(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    return matlab::runtime::shape_op(m, n, [&](int64_t i, int64_t j) {
        return A->data[i * n + (n - 1 - j)];
    }).release();
}

matlab_mat *matlab_flipud(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    return matlab::runtime::shape_op(m, n, [&](int64_t i, int64_t j) {
        return A->data[(m - 1 - i) * n + j];
    }).release();
}

/* flip(A) with no dim: match MATLAB — flip along the first non-
 * singleton dim. Vectors flip themselves; matrices flip rows (the
 * equivalent of flipud). */
matlab_mat *matlab_flip(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    if (A->rows == 1) return matlab_fliplr(A);
    return matlab_flipud(A);
}

/* rot90(A): counter-clockwise 90° rotation, once. Result is
 * cols-by-rows; element (i, j) of the result comes from (j, cols-1-i)
 * of the input. */
matlab_mat *matlab_rot90(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    return matlab::runtime::shape_op(n, m, [&](int64_t i, int64_t j) {
        return A->data[j * n + (n - 1 - i)];
    }).release();
}

/* Element-wise min/max of two matrices with the usual broadcast. */
matlab_mat *matlab_min_mm(matlab_mat *A, matlab_mat *B) {
    int64_t m = A->rows, n = A->cols;
    matlab_mat *C = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) {
        double a = A->data[k], b = B->data[k];
        C->data[k] = a < b ? a : b;
    }
    return C;
}
matlab_mat *matlab_max_mm(matlab_mat *A, matlab_mat *B) {
    int64_t m = A->rows, n = A->cols;
    matlab_mat *C = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) {
        double a = A->data[k], b = B->data[k];
        C->data[k] = a > b ? a : b;
    }
    return C;
}

/*---------- Shape queries ------------------------------------------------*/

/* size(A) -> 1×2 row vector [rows cols]. */
/* size / numel / length / ndims — all take `matlab_mat *` in the
 * MATLAB-facing API but must also handle `matlab_mat_c *` (complex)
 * and `matlab_mat3 *` (3-D) descriptors, since the call sites are
 * uniformly typed as ptr at the lowering layer.  Check the leading
 * magic word and read rows/cols from the correct offset for each
 * layout. */
static inline void mat_any_shape(const void *A,
                                 int64_t *out_r, int64_t *out_c) {
    if (!A) { *out_r = 0; *out_c = 0; return; }
    if (mat_is_complex(A)) {
        const matlab_mat_c *c = (const matlab_mat_c *)A;
        *out_r = c->rows; *out_c = c->cols; return;
    }
    /* Default: matlab_mat layout (data*, rows, cols, ...). */
    const matlab_mat *m = (const matlab_mat *)A;
    *out_r = m->rows; *out_c = m->cols;
}

matlab_mat *matlab_size(matlab_mat *A) {
    int64_t r, c; mat_any_shape(A, &r, &c);
    matlab_mat *R = mat_alloc(1, 2);
    R->data[0] = (double)r;
    R->data[1] = (double)c;
    return R;
}

/* size(A, dim). dim is 1-based; 1=rows, 2=cols; any other dim returns 1. */
double matlab_size_dim(matlab_mat *A, double dim) {
    int64_t r, c; mat_any_shape(A, &r, &c);
    int64_t d = (int64_t)dim;
    if (d == 1) return (double)r;
    if (d == 2) return (double)c;
    return 1.0;
}

double matlab_length(matlab_mat *A) {
    int64_t r, c; mat_any_shape(A, &r, &c);
    if (r == 0 || c == 0) return 0.0;
    return (double)(r > c ? r : c);
}

double matlab_numel(matlab_mat *A)  {
    int64_t r, c; mat_any_shape(A, &r, &c);
    return (double)(r * c);
}
double matlab_ndims(matlab_mat *A)  { (void)A; return 2.0; }

/* end-of-dim for use inside subscript expressions: `end` in A(..., end, ...)
 * resolves to size(A, dim) where `dim` is the 1-based position of the
 * argument in the subscript. */
double matlab_end_of_dim(matlab_mat *A, double dim) {
    return matlab_size_dim(A, dim);
}

/*---------- Slicing ------------------------------------------------------
 *
 * `rows` and `cols` are matlab_mat row vectors (or single-element 1×1) of
 * 1-based integer indices. A NULL pointer means "colon" — take all indices
 * along that dimension.
 *
 *--------------------------------------------------------------------------*/

/* Wrap a scalar double as a 1×1 matrix. Used by the subscript lowering
 * when one index is scalar and another is a range/colon. */
matlab_mat *matlab_mat_from_scalar(double x) {
    matlab_mat *M = mat_alloc(1, 1);
    M->data[0] = x;
    return M;
}

/* MATLAB `if M` / `while M` semantics: truthy iff M is non-empty AND
 * every element is non-zero. Used by the DAP/REPL paths where script
 * variables are workspace-backed and a scalar load comes back as a
 * 1×1 matrix pointer rather than a raw f64 — fixupIfCond emits a
 * matlab.call_builtin @matlab_mat_truth(ptr) -> i8 to coerce the
 * result back to a scalar logical for scf.if / scf.while. */
int8_t matlab_mat_truth(matlab_mat *m) {
    if (!m) return 0;
    int64_t n = m->rows * m->cols;
    if (n == 0) return 0;
    for (int64_t i = 0; i < n; ++i)
        if (m->data[i] == 0.0) return 0;
    return 1;
}

/* A(rows, cols): rank-2 slice. Result dims are the lengths of rows/cols
 * (or the base's dim if the corresponding index is NULL/colon). 1-based
 * indexing; out-of-range indices leave 0 in the output cell. */
matlab_mat *matlab_slice2(matlab_mat *A, matlab_mat *rows, matlab_mat *cols) {
    int64_t R = rows ? rows->rows * rows->cols : A->rows;
    int64_t C = cols ? cols->rows * cols->cols : A->cols;
    matlab_mat *Y = mat_alloc(R, C);
    for (int64_t i = 0; i < R; ++i) {
        int64_t ri = rows ? ((int64_t)rows->data[i] - 1) : i;
        if (ri < 0 || ri >= A->rows) continue;
        for (int64_t j = 0; j < C; ++j) {
            int64_t cj = cols ? ((int64_t)cols->data[j] - 1) : j;
            if (cj < 0 || cj >= A->cols) continue;
            Y->data[i * C + j] = A->data[ri * A->cols + cj];
        }
    }
    return Y;
}

/* A(idx): linear indexing. MATLAB uses column-major order — A(k) walks
 * down column 1, then column 2, etc. Result shape tracks the index shape.
 *
 * Logical indexing: when `idx` has the same shape as `A` (and A isn't a
 * 1x1 scalar) we interpret idx as a mask — pick elements where idx!=0,
 * walked in column-major order, return as a column vector. This is what
 * makes `A(A > 0)` work naturally. */
matlab_mat *matlab_slice1(matlab_mat *A, matlab_mat *idx) {
    int64_t m = A->rows, n = A->cols;
    if (idx && idx->rows == m && idx->cols == n && (m > 1 || n > 1)) {
        int64_t count = 0;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                if (idx->data[i * n + j] != 0.0) ++count;
        matlab_mat *Y = mat_alloc(count, 1);
        int64_t w = 0;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                if (idx->data[i * n + j] != 0.0)
                    Y->data[w++] = A->data[i * n + j];
        return Y;
    }
    int64_t N = idx ? idx->rows * idx->cols : m * n;
    int64_t outR = idx ? idx->rows : 1;
    int64_t outC = idx ? idx->cols : N;
    if (outR * outC != N) { outR = 1; outC = N; }
    matlab_mat *Y = mat_alloc(outR, outC);
    for (int64_t k = 0; k < N; ++k) {
        int64_t lin = idx ? ((int64_t)idx->data[k] - 1) : k;
        if (lin < 0 || lin >= m * n) continue;
        int64_t col = lin / m;
        int64_t row = lin - col * m;
        Y->data[k] = A->data[row * n + col];
    }
    return Y;
}

/* Empty 0×0 matrix. Used for `A = []` deallocation / `clear A`. */
matlab_mat *matlab_empty_mat(void) {
    matlab_mat *M = (matlab_mat *)calloc(1, sizeof(matlab_mat));
    M->rows = 0;
    M->cols = 0;
    M->data = NULL;
    return M;
}

/* A(rows, cols) = V. Scalar V is broadcast. NULL rows/cols = colon. */
void matlab_slice_store2(matlab_mat *A, matlab_mat *rows, matlab_mat *cols,
                         matlab_mat *V) {
    int64_t R = rows ? rows->rows * rows->cols : A->rows;
    int64_t C = cols ? cols->rows * cols->cols : A->cols;
    int bcast = (V->rows == 1 && V->cols == 1);
    for (int64_t i = 0; i < R; ++i) {
        int64_t ri = rows ? ((int64_t)rows->data[i] - 1) : i;
        if (ri < 0 || ri >= A->rows) continue;
        for (int64_t j = 0; j < C; ++j) {
            int64_t cj = cols ? ((int64_t)cols->data[j] - 1) : j;
            if (cj < 0 || cj >= A->cols) continue;
            double v;
            if (bcast) v = V->data[0];
            else if (V->rows == R && V->cols == C) v = V->data[i * C + j];
            else continue;
            A->data[ri * A->cols + cj] = v;
        }
    }
}

void matlab_slice_store2_scalar(matlab_mat *A, matlab_mat *rows,
                                matlab_mat *cols, double v) {
    int64_t R = rows ? rows->rows * rows->cols : A->rows;
    int64_t C = cols ? cols->rows * cols->cols : A->cols;
    for (int64_t i = 0; i < R; ++i) {
        int64_t ri = rows ? ((int64_t)rows->data[i] - 1) : i;
        if (ri < 0 || ri >= A->rows) continue;
        for (int64_t j = 0; j < C; ++j) {
            int64_t cj = cols ? ((int64_t)cols->data[j] - 1) : j;
            if (cj < 0 || cj >= A->cols) continue;
            A->data[ri * A->cols + cj] = v;
        }
    }
}

void matlab_slice_store1(matlab_mat *A, matlab_mat *idx, matlab_mat *V) {
    int64_t N = idx ? idx->rows * idx->cols : A->rows * A->cols;
    int64_t m = A->rows, n = A->cols;
    int bcast = (V->rows == 1 && V->cols == 1);
    for (int64_t k = 0; k < N; ++k) {
        int64_t lin = idx ? ((int64_t)idx->data[k] - 1) : k;
        if (lin < 0 || lin >= m * n) continue;
        int64_t col = lin / m;
        int64_t row = lin - col * m;
        double v;
        if (bcast) v = V->data[0];
        else if (k < V->rows * V->cols) v = V->data[k];
        else continue;
        A->data[row * n + col] = v;
    }
}

void matlab_slice_store1_scalar(matlab_mat *A, matlab_mat *idx, double v) {
    int64_t N = idx ? idx->rows * idx->cols : A->rows * A->cols;
    int64_t m = A->rows, n = A->cols;
    for (int64_t k = 0; k < N; ++k) {
        int64_t lin = idx ? ((int64_t)idx->data[k] - 1) : k;
        if (lin < 0 || lin >= m * n) continue;
        int64_t col = lin / m;
        int64_t row = lin - col * m;
        A->data[row * n + col] = v;
    }
}

/* find(A): column vector of linear (column-major, 1-based) indices of
 * non-zero elements. Very common MATLAB idiom: `find(A > 0)` gives you
 * the indices where a condition holds. */
matlab_mat *matlab_find(matlab_mat *A) {
    int64_t m = A->rows, n = A->cols;
    int64_t count = 0;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            if (A->data[i * n + j] != 0.0) ++count;
    matlab_mat *Y = mat_alloc(count, 1);
    int64_t k = 0;
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            if (A->data[i * n + j] != 0.0) {
                Y->data[k++] = (double)(j * m + i + 1);
            }
        }
    }
    return Y;
}

/* A(rows, :) = [] / A(:, cols) = [] semantics, exposed as runtime helpers
 * that a future pass can call when the frontend lowers the empty-matrix
 * assignment. */
matlab_mat *matlab_erase_rows(matlab_mat *A, matlab_mat *rows) {
    int64_t m = A->rows, n = A->cols;
    int64_t r = rows ? rows->rows * rows->cols : 0;
    char *kill = (char *)calloc((size_t)m, 1);
    for (int64_t k = 0; k < r; ++k) {
        int64_t ri = (int64_t)rows->data[k] - 1;
        if (ri >= 0 && ri < m) kill[ri] = 1;
    }
    int64_t keep = 0;
    for (int64_t i = 0; i < m; ++i) if (!kill[i]) ++keep;
    matlab_mat *Y = mat_alloc(keep, n);
    int64_t w = 0;
    for (int64_t i = 0; i < m; ++i) {
        if (kill[i]) continue;
        for (int64_t j = 0; j < n; ++j) Y->data[w * n + j] = A->data[i * n + j];
        ++w;
    }
    free(kill);
    return Y;
}

matlab_mat *matlab_erase_cols(matlab_mat *A, matlab_mat *cols) {
    int64_t m = A->rows, n = A->cols;
    int64_t c = cols ? cols->rows * cols->cols : 0;
    char *kill = (char *)calloc((size_t)n, 1);
    for (int64_t k = 0; k < c; ++k) {
        int64_t cj = (int64_t)cols->data[k] - 1;
        if (cj >= 0 && cj < n) kill[cj] = 1;
    }
    int64_t keep = 0;
    for (int64_t j = 0; j < n; ++j) if (!kill[j]) ++keep;
    matlab_mat *Y = mat_alloc(m, keep);
    for (int64_t i = 0; i < m; ++i) {
        int64_t w = 0;
        for (int64_t j = 0; j < n; ++j) {
            if (kill[j]) continue;
            Y->data[i * keep + w++] = A->data[i * n + j];
        }
    }
    free(kill);
    return Y;
}

/* Multi-arg fprintf variants for 2, 3, 4 f64 trailing args. LowerTensorOps
 * picks the matching symbol based on the call arity. Variadic C is too
 * ABI-fragile across targets; per-arity entries are the cleanest path. */
void matlab_fprintf_f64_2(const char *fmt, int64_t n, double a, double b) {
    if (n < 0) n = 0;
    if (n > 1023) n = 1023;
    char buf[1024];
    int64_t len = expand_escapes(buf, fmt, n);
    buf[len] = '\0';
    printf(buf, a, b);
}

void matlab_fprintf_f64_3(const char *fmt, int64_t n,
                          double a, double b, double c) {
    if (n < 0) n = 0;
    if (n > 1023) n = 1023;
    char buf[1024];
    int64_t len = expand_escapes(buf, fmt, n);
    buf[len] = '\0';
    printf(buf, a, b, c);
}

void matlab_fprintf_f64_4(const char *fmt, int64_t n,
                          double a, double b, double c, double d) {
    if (n < 0) n = 0;
    if (n > 1023) n = 1023;
    char buf[1024];
    int64_t len = expand_escapes(buf, fmt, n);
    buf[len] = '\0';
    printf(buf, a, b, c, d);
}

/* input(prompt): numeric-only subset. Prompt goes to stdout, read a double
 * from stdin, return it. Real MATLAB's input evals an arbitrary expression
 * and the 's' mode returns a string — both out of scope for now. */
double matlab_input_num(const char *prompt, int64_t plen) {
    if (plen > 0) {
        fwrite(prompt, 1, (size_t)plen, stdout);
        fflush(stdout);
    }
    double v = 0.0;
    if (scanf("%lf", &v) != 1) v = 0.0;
    return v;
}

/*---------- Timing & sleep ----------------------------------------------*/

/* Per-thread default tic/toc slot. INT64_MIN sentinels "tic never called",
 * which makes toc return 0.0 — matches the MATLAB convention of not warning
 * on a bare toc, while distinguishing the case from "tic'd at t=0". */
static thread_local int64_t matlab_tic_ns = INT64_MIN;

static int64_t monotonic_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

void matlab_pause(double seconds) {
    /* Match MATLAB: non-positive / NaN → return immediately. */
    if (!(seconds > 0.0)) return;
    /* Cap at ~10^9 seconds to keep the conversion to timespec defined. */
    if (seconds > 1e9) seconds = 1e9;
    struct timespec req;
    req.tv_sec  = (time_t)seconds;
    double frac = seconds - (double)req.tv_sec;
    long ns = (long)(frac * 1e9);
    if (ns < 0) ns = 0;
    if (ns > 999999999L) ns = 999999999L;
    req.tv_nsec = ns;
    /* Resume on EINTR — pthread cond signals from the debugger thread can
     * wake nanosleep early; the loop preserves the requested duration. */
    struct timespec rem;
    while (nanosleep(&req, &rem) == -1) {
        if (rem.tv_sec == 0 && rem.tv_nsec == 0) break;
        req = rem;
    }
}

void matlab_pause_keypress(void) {
    /* Non-interactive run: bail out instead of hanging on a closed stdin.
     * Matches the behaviour of MATLAB scripts running with -nodesktop in
     * a redirected pipe (effectively a no-op). */
    if (!isatty(fileno(stdin))) return;
    int c = getchar();
    (void)c;
}

void matlab_tic(void) {
    matlab_tic_ns = monotonic_now_ns();
}

double matlab_toc(void) {
    if (matlab_tic_ns == INT64_MIN) return 0.0;
    int64_t dt = monotonic_now_ns() - matlab_tic_ns;
    return (double)dt * 1e-9;
}

void matlab_toc_print(void) {
    double s = matlab_toc();
    pthread_mutex_lock(&matlab_io_mutex);
    printf("Elapsed time is %.6f seconds.\n", s);
    pthread_mutex_unlock(&matlab_io_mutex);
}

/*---------- Predicates ---------------------------------------------------*/

double matlab_isempty(matlab_mat *A) {
    return (A->rows == 0 || A->cols == 0) ? 1.0 : 0.0;
}

double matlab_isequal(matlab_mat *A, matlab_mat *B) {
    if (A->rows != B->rows || A->cols != B->cols) return 0.0;
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k)
        if (A->data[k] != B->data[k]) return 0.0;
    return 1.0;
}

/*---------- Matrix power -------------------------------------------------
 * matlab_matpow(A, n) = A^n for integer n. Uses repeated multiplication,
 * with inv(A) for negative n. Non-integer n falls back to A * A scaled
 * appropriately (not a true matrix function — for teaching scale, document
 * the limitation).
 *-------------------------------------------------------------------------*/

matlab_mat *matlab_matpow(matlab_mat *A, double n) {
    if (A->rows != A->cols) return mat_alloc(0, 0);
    int64_t ni = (int64_t)n;
    if ((double)ni != n) {
        /* Non-integer — return element-wise power as a degraded fallback.
         * Real matrix power for non-integer exponents requires eigen-
         * decomposition which we don't surface in runtime form yet. */
        int64_t total = A->rows * A->cols;
        matlab_mat *C = mat_alloc(A->rows, A->cols);
        for (int64_t k = 0; k < total; ++k) C->data[k] = pow(A->data[k], n);
        return C;
    }
    matlab_mat *base = A;
    matlab_mat *freeable_base = NULL;
    if (ni < 0) {
        freeable_base = matlab_inv(A);
        base = freeable_base;
        ni = -ni;
    }
    int64_t N = A->rows;
    /* Start with identity of the right size. */
    matlab_mat *acc = matlab_eye((double)N, (double)N);
    matlab_mat *p = base;  /* current power of base */
    while (ni > 0) {
        if (ni & 1) acc = matlab_matmul_mm(acc, p);
        ni >>= 1;
        if (ni > 0) p = matlab_matmul_mm(p, p);
    }
    (void)freeable_base;
    return acc;
}

/*-------------------------------------------------------------------------
 * Matrix exponential — expm(A).
 *
 * Scaling-and-squaring with a [13/13] Padé approximant (Higham 2005,
 * "The Scaling and Squaring Method for the Matrix Exponential
 * Revisited", SIAM J. Matrix Anal. Appl. 26(4), 1179-1193). This is
 * the workhorse algorithm scipy.linalg.expm and MATLAB's expm both
 * use as their double-precision path.
 *
 *   1. Pick s so that ||A / 2^s||_1 <= theta_13 ≈ 5.37192...
 *   2. A_s = A / 2^s.
 *   3. Compute U, V from A_s via Higham's Algorithm 10.20 (uses A_s^2,
 *      A_s^4, A_s^6 only — the rest is linear combinations + one extra
 *      mat-mat).
 *   4. Solve (V - U) * R = (V + U) for R.
 *   5. Square R s times to undo the scaling.
 *
 * Tier 1.3 of the Control System Toolbox roadmap. See
 * docs/control_toolbox_roadmap.md §2.3.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_expm(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0) return mat_alloc(0, 0);

    /* Padé [13/13] coefficients — denominators of the rational Padé
     * are obtained by replacing A with -A and reusing the same b[k]
     * (the [m/m]-Padé property). */
    static const double b[14] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0,  129060195264000.0,   10559470521600.0,
        670442572800.0,      33522128640.0,       1323241920.0,
        40840800.0,          960960.0,            16380.0,
        182.0,               1.0
    };
    /* Higham 2005 Table 2.3: maximum 1-norm at which [13/13] Padé is
     * accurate to ~unit roundoff in IEEE double. */
    static const double theta13 = 5.371920351148152;

    /* 1-norm of A (max column sum of |A_ij|). */
    double anrm = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        double col = 0.0;
        for (int64_t i = 0; i < n; ++i) col += fabs(A->data[i * n + j]);
        if (col > anrm) anrm = col;
    }

    int s = 0;
    std::vector<double> As(A->data, A->data + n * n);
    if (anrm > theta13) {
        /* s = ceil(log2(anrm / theta13)).  Use ldexp(1, k) instead of
         * `1 << k` so we don't trip the shift-overflow on very large
         * anrm. */
        double r = anrm / theta13;
        s = 0;
        while (ldexp(1.0, s + 1) < r) ++s;
        if (ldexp(1.0, s) < r) ++s;
        double scale = ldexp(1.0, -s);
        for (int64_t i = 0; i < n * n; ++i) As[i] *= scale;
    }

    auto mm = [n](const double *X, const double *Y, double *Z) {
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int64_t k = 0; k < n; ++k)
                    sum += X[i * n + k] * Y[k * n + j];
                Z[i * n + j] = sum;
            }
    };

    std::vector<double> A2(n * n), A4(n * n), A6(n * n);
    mm(As.data(), As.data(), A2.data());
    mm(A2.data(), A2.data(), A4.data());
    mm(A4.data(), A2.data(), A6.data());

    /* Algorithm 10.20 — split the polynomial in two halves so the
     * inner-most product reuses A6:
     *   U = A_s * (A6 * (b13*A6 + b11*A4 + b9*A2)
     *              + b7*A6 + b5*A4 + b3*A2 + b1*I)
     *   V =        A6 * (b12*A6 + b10*A4 + b8*A2)
     *              + b6*A6 + b4*A4 + b2*A2 + b0*I
     * Then exp(A_s) ≈ (V - U)^{-1} (V + U). */
    std::vector<double> W1(n * n), W2(n * n), Z1(n * n), Z2(n * n);
    for (int64_t i = 0; i < n * n; ++i) {
        W1[i] = b[13] * A6[i] + b[11] * A4[i] + b[9] * A2[i];
        Z1[i] = b[12] * A6[i] + b[10] * A4[i] + b[8] * A2[i];
        W2[i] = b[7]  * A6[i] + b[5]  * A4[i] + b[3] * A2[i];
        Z2[i] = b[6]  * A6[i] + b[4]  * A4[i] + b[2] * A2[i];
    }
    for (int64_t i = 0; i < n; ++i) {
        W2[i * n + i] += b[1];
        Z2[i * n + i] += b[0];
    }

    std::vector<double> tmp(n * n), W(n * n), V(n * n), U(n * n);
    mm(A6.data(), W1.data(), tmp.data());
    for (int64_t i = 0; i < n * n; ++i) W[i] = tmp[i] + W2[i];
    mm(As.data(), W.data(), U.data());
    mm(A6.data(), Z1.data(), tmp.data());
    for (int64_t i = 0; i < n * n; ++i) V[i] = tmp[i] + Z2[i];

    /* Solve (V - U) * R = (V + U) — column by column via the existing
     * pivoted-LU scratch helpers. */
    std::vector<double> LU(n * n);
    std::vector<double> RHS(n * n);
    for (int64_t i = 0; i < n * n; ++i) {
        LU[i]  = V[i] - U[i];
        RHS[i] = V[i] + U[i];
    }
    std::vector<int64_t> piv(n);
    int sgn;
    if (lu_decompose(LU.data(), n, piv.data(), &sgn) != 0)
        return mat_alloc(0, 0);

    std::vector<double> R(n * n);
    std::vector<double> rhs_col(n), x_col(n);
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t i = 0; i < n; ++i) rhs_col[i] = RHS[i * n + c];
        lu_solve_column(LU.data(), n, piv.data(), rhs_col.data(),
                        x_col.data());
        for (int64_t i = 0; i < n; ++i) R[i * n + c] = x_col[i];
    }

    /* Square R back, s times, to recover exp(A) from exp(A_s). */
    std::vector<double> Rsq(n * n);
    for (int k = 0; k < s; ++k) {
        mm(R.data(), R.data(), Rsq.data());
        std::swap(R, Rsq);
    }

    matlab_mat *out = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) out->data[i] = R[i];
    return out;
}

/*-------------------------------------------------------------------------
 * Matrix logarithm — L = logm(A).
 *
 *   logm(A) is the inverse of expm: a matrix L such that expm(L) = A.
 *   For a stable continuous-time plant whose discrete sample is Ad =
 *   expm(A·Ts), logm(Ad)/Ts recovers A — that's the d2c ZOH workflow's
 *   gating primitive.
 *
 * Algorithm: Schur-then-Parlett (Higham 2008 §11.4).
 *   1. Real Schur T = U' A U (existing matlab_schur primitives).
 *   2. log(T) for upper-triangular T computed via Parlett's recurrence:
 *      diagonal entries are scalar logs; off-diagonals propagate from the
 *      commutativity of T with any analytic function of T,
 *        F[i,j] = (T[i,j] (F[j,j] − F[i,i]) +
 *                  Σ_{k=i+1}^{j-1} (T[i,k] F[k,j] − F[i,k] T[k,j])) /
 *                 (T[j,j] − T[i,i]).
 *   3. logm(A) = U · log(T) · U'.
 *
 * Limitations of this v1 entry:
 *   - Real Schur form must come back UPPER-TRIANGULAR (all eigenvalues
 *     real). Complex conjugate pairs would land in 2×2 quasi-triangular
 *     blocks; their proper handling needs a complex-arithmetic block log
 *     plus Parlett's block-form recurrence — deferred.
 *   - All eigenvalues must be POSITIVE. A negative or zero diagonal
 *     entry would force log into the complex plane; we'd need a complex
 *     return path.
 *   - No two diagonal entries may coincide. Repeated eigenvalues make
 *     the recurrence divide by zero; the cure (Parlett's "block"
 *     algorithm with confluent Taylor expansion) is also deferred.
 * Returns 0×0 in any of those cases — same convention the other
 * decomposition primitives follow when their preconditions don't hold.
 *
 * Tier 1.3 follow-on of the Control System Toolbox roadmap; see
 * docs/control_toolbox_roadmap.md §2.3.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_logm(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0) return mat_alloc(0, 0);

    /* Schur decomposition. We need both U and T; the existing entries
     * compute each independently, so re-do the full pipeline once here
     * to get matched (U, T) without two redundant Hessenberg reductions. */
    std::vector<double> T(A->data, A->data + n * n);
    std::vector<double> U(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) U[i * n + i] = 1.0;
    hessenberg_inplace_(T.data(), n, U.data());
    francis_qr_(T.data(), n, U.data());

    /* Validate preconditions on the Schur form. */
    const double eps = 1e-12;
    for (int64_t i = 0; i < n; ++i) {
        /* Subdiagonal must be (near) zero — no 2×2 quasi-triangular blocks. */
        if (i + 1 < n) {
            double sub = T[(i + 1) * n + i];
            if (std::fabs(sub) > eps * (std::fabs(T[i * n + i]) +
                                          std::fabs(T[(i+1) * n + (i+1)]) +
                                          1.0))
                return mat_alloc(0, 0);
        }
        /* Diagonal must be strictly positive. */
        if (T[i * n + i] <= eps) return mat_alloc(0, 0);
    }
    /* Coincident diagonals would divide-by-zero in Parlett's recurrence. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j)
            if (std::fabs(T[j * n + j] - T[i * n + i]) < eps *
                (std::fabs(T[i * n + i]) + std::fabs(T[j * n + j]) + 1.0))
                return mat_alloc(0, 0);

    /* F = log(T) on the diagonal. */
    std::vector<double> F(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) F[i * n + i] = std::log(T[i * n + i]);

    /* Parlett's recurrence for off-diagonals, sweeping super-diagonals
     * from j-i = 1 upward so each F[i,j] depends only on already-filled
     * entries. */
    for (int64_t d = 1; d < n; ++d) {
        for (int64_t i = 0; i + d < n; ++i) {
            int64_t j = i + d;
            double sum = 0.0;
            for (int64_t k = i + 1; k < j; ++k)
                sum += T[i * n + k] * F[k * n + j] -
                       F[i * n + k] * T[k * n + j];
            F[i * n + j] = (T[i * n + j] *
                                (F[j * n + j] - F[i * n + i]) + sum) /
                           (T[j * n + j] - T[i * n + i]);
        }
    }

    /* logm(A) = U * F * U'. */
    auto mm = [n](const double *X, const double *Y, double *Z) {
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (int64_t k = 0; k < n; ++k)
                    s += X[i * n + k] * Y[k * n + j];
                Z[i * n + j] = s;
            }
    };
    std::vector<double> Ut(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) Ut[j * n + i] = U[i * n + j];
    std::vector<double> tmp(n * n);
    mm(U.data(), F.data(), tmp.data());
    matlab_mat *out = mat_alloc(n, n);
    mm(tmp.data(), Ut.data(), out->data);
    return out;
}

/*-------------------------------------------------------------------------
 * Hessenberg reduction — H = hess(A).
 *
 * Reduce a real n*n matrix A to upper Hessenberg form H via a sequence
 * of Householder reflections, P_k. The composite orthogonal matrix
 * P = P_0 P_1 ... P_{n-3} satisfies P' A P = H with H[i,j] = 0 for
 * i > j+1.
 *
 * Hessenberg form preserves eigenvalues (similarity transform) and is
 * the standard launch pad for the Francis double-shift QR algorithm
 * that converges to real Schur form. Direct cost: O(n^3) per call.
 *
 * Tier 1.2 of the Control System Toolbox roadmap. See
 * docs/control_toolbox_roadmap.md §2.2. The 2-return form
 * [H, P] = hess(A) is a follow-on (will use the same scratch with one
 * extra accumulator matrix; routed via separate matlab_hess_H /
 * matlab_hess_P entries mirroring the eig_V / eig_D precedent).
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_hess(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    matlab_mat *H = mat_alloc(n, n);
    if (n == 0) return H;
    for (int64_t i = 0; i < n * n; ++i) H->data[i] = A->data[i];
    if (n <= 2) return H;  /* already upper Hessenberg */

    std::vector<double> v(n);
    for (int64_t k = 0; k + 2 < n; ++k) {
        /* Build the Householder vector that zeroes H[k+2..n-1, k]. */
        double sigma = 0.0;
        for (int64_t i = k + 1; i < n; ++i) {
            double x = H->data[i * n + k];
            sigma += x * x;
        }
        if (sigma == 0.0) continue;
        double xk = H->data[(k + 1) * n + k];
        double xnorm = sqrt(sigma);
        /* Choose the sign that *adds* magnitudes (avoids cancellation). */
        double v0 = xk + (xk >= 0 ? xnorm : -xnorm);
        v[k + 1] = v0;
        for (int64_t i = k + 2; i < n; ++i) v[i] = H->data[i * n + k];
        double vnorm2 = v0 * v0 + (sigma - xk * xk);
        if (vnorm2 == 0.0) continue;
        double beta = 2.0 / vnorm2;

        /* Apply (I - beta v v^T) from the left: only rows k+1..n-1 are
         * touched. Sweep across all n columns. */
        for (int64_t j = k; j < n; ++j) {
            double w = 0.0;
            for (int64_t i = k + 1; i < n; ++i)
                w += v[i] * H->data[i * n + j];
            w *= beta;
            for (int64_t i = k + 1; i < n; ++i)
                H->data[i * n + j] -= v[i] * w;
        }
        /* Apply (I - beta v v^T) from the right: only columns k+1..n-1
         * are touched. Sweep across all n rows. */
        for (int64_t i = 0; i < n; ++i) {
            double w = 0.0;
            for (int64_t j = k + 1; j < n; ++j)
                w += H->data[i * n + j] * v[j];
            w *= beta;
            for (int64_t j = k + 1; j < n; ++j)
                H->data[i * n + j] -= w * v[j];
        }
        /* Numeric cleanup — round tiny subdiagonal residues to exact 0
         * so disp(hess(A)) prints clean zeros. The reflection has
         * already moved column k's tail into one element by
         * construction; this is just IEEE-rounding hygiene. */
        for (int64_t i = k + 2; i < n; ++i) H->data[i * n + k] = 0.0;
    }
    return H;
}

/* 2-return [H, P] = hess(A) shape — H is upper Hessenberg, P is the
 * orthogonal similarity (P' A P = H). Two entries route through the
 * same multi-return splitter pattern as eig_V/eig_D: the frontend
 * dispatches each LHS to its own helper, and both helpers redo the
 * Hessenberg reduction independently to keep the runtime stateless.
 * Cost is one extra O(n³) Householder pass — negligible compared to
 * the typical caller's downstream work. */
matlab_mat *matlab_hess_H(matlab_mat *A) { return matlab_hess(A); }

matlab_mat *matlab_hess_P(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    matlab_mat *P = mat_alloc(n, n);
    if (n == 0) return P;
    /* Initialise P to identity; the in-place pass accumulates the
     * Householder reflections into it. */
    for (int64_t i = 0; i < n; ++i) P->data[i * n + i] = 1.0;
    if (n <= 2) return P;
    std::vector<double> H(A->data, A->data + n * n);
    hessenberg_inplace_(H.data(), n, P->data);
    return P;
}

/*-------------------------------------------------------------------------
 * Real Schur decomposition — [U, T] = schur(A).
 *
 * For real A, returns an orthogonal U and a real-Schur (upper quasi-
 * triangular: 1*1 and 2*2 diagonal blocks) T such that A = U T U'.
 * Real eigenvalues of A appear on T's diagonal as 1*1 blocks; complex
 * conjugate pairs appear as 2*2 diagonal blocks (the [a b; c d] pencil
 * has tr=a+d, det=ad-bc, with disc = tr^2 - 4 det < 0).
 *
 * Implementation: Hessenberg reduce + Francis double-shift QR (the
 * same pipeline as non-symmetric matlab_eig), with the orthogonal
 * accumulator U threaded through both passes.
 *
 * Three public entries — matlab_schur returns T (1-return form);
 * matlab_schur_T / matlab_schur_U each compute both and return one for
 * the [U, T] = schur(A) shape (eig_V / eig_D precedent). The two-pass
 * computation is repeated per call rather than cached — keeps the
 * runtime stateless. The cost is two extra Householder reductions
 * worth of compute, which is negligible relative to the QR convergence.
 *
 * Tier 1.2 follow-on of the Control System Toolbox roadmap. See
 * docs/control_toolbox_roadmap.md §2.2. Gates Tier 1.4 (Bartels-Stewart
 * Lyapunov) and Tier 1.5 (ordered-Schur Riccati).
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_schur(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    matlab_mat *T = mat_alloc(n, n);
    if (n == 0) return T;
    for (int64_t i = 0; i < n * n; ++i) T->data[i] = A->data[i];
    hessenberg_inplace_(T->data, n, /*U=*/nullptr);
    francis_qr_(T->data, n, /*U=*/nullptr);
    return T;
}

matlab_mat *matlab_schur_T(matlab_mat *A) { return matlab_schur(A); }

matlab_mat *matlab_schur_U(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    matlab_mat *U = mat_alloc(n, n);
    if (n == 0) return U;
    std::vector<double> H(A->data, A->data + n * n);
    hessenberg_inplace_(H.data(), n, U->data);
    francis_qr_(H.data(), n, U->data);
    return U;
}

/* Forward decls — qr_factor is defined later in this TU. */
static void qr_factor(matlab_mat *A, matlab_mat *Q, matlab_mat *R);

/*-------------------------------------------------------------------------
 * Generalised Schur decomposition — [AA, BB, Q, Z] = qz(A, B).
 *
 *   qz reduces the matrix pencil A − λ·B to a generalised Schur form:
 *     Q · A · Z = AA   (real upper quasi-triangular: 1×1 / 2×2 blocks)
 *     Q · B · Z = BB   (real upper triangular)
 *   with Q and Z orthogonal. The generalised eigenvalues are the
 *   diagonal pairs (AA[i,i], BB[i,i]); λ_i = AA[i,i] / BB[i,i] when
 *   BB[i,i] ≠ 0, otherwise λ_i = ∞ (an "infinite" pencil eigenvalue).
 *
 * v1 implementation: layered on the existing schur + qr primitives,
 * valid when B is invertible (the common case for descriptor systems
 * with regular dynamics):
 *   1. C = B⁻¹ · A
 *   2. Real Schur of C: U' · C · U = T (upper quasi-triangular)
 *   3. M = B · U;  QR of M:  M = O · R  (O orthogonal, R upper
 *      triangular).
 *   4. Q = O',   Z = U.
 *      AA = Q · A · Z = O' · A · U = O' · (B · U) · T = R · T
 *      BB = Q · B · Z = O' · B · U = R
 *   The product R·T is upper quasi-triangular when T is.
 *   Returns 0×0 when B is singular — that path needs the proper
 *   Hessenberg-Triangular reduction + double-shift QZ iteration
 *   (Moler-Stewart 1973), which is the Tier-1.2 final follow-on for
 *   `zero(sys)` on the Rosenbrock system matrix (where B is rank-
 *   deficient by construction). Tracked in
 *   docs/control_toolbox_roadmap.md §2.2.
 *
 * Four public entries follow the schur_U / schur_T precedent — each
 * recomputes the full decomposition and returns one piece. Cost is
 * negligible relative to the typical caller's downstream work
 * (small-plant CST workflows, n = 2..10).
 *-------------------------------------------------------------------------*/
namespace {

bool qz_is_b_invertible_(matlab_mat *B) {
    if (!B || B->rows != B->cols) return false;
    int64_t n = B->rows;
    if (n == 0) return false;
    std::vector<double> LU(B->data, B->data + n * n);
    std::vector<int64_t> piv(n);
    int sgn;
    if (lu_decompose(LU.data(), n, piv.data(), &sgn) != 0) return false;
    /* Reject if any diagonal entry of LU is too small (singular
     * within roundoff). lu_decompose already errors on exact zero
     * but a rank-deficient B can squeak through with a tiny pivot;
     * compare against a Frobenius-scaled threshold. */
    double fro = 0.0;
    for (int64_t i = 0; i < n * n; ++i) fro += B->data[i] * B->data[i];
    fro = std::sqrt(fro);
    double tol = 1e-12 * (fro + 1.0);
    for (int64_t i = 0; i < n; ++i)
        if (std::fabs(LU[i * n + i]) < tol) return false;
    return true;
}

bool qz_compute_(matlab_mat *A, matlab_mat *B,
                  std::vector<double> &AA,
                  std::vector<double> &BB,
                  std::vector<double> &Q,
                  std::vector<double> &Z,
                  int64_t &n_out) {
    if (!A || !B || A->rows != A->cols || B->rows != B->cols ||
        A->rows != B->rows) return false;
    int64_t n = A->rows;
    if (n == 0) return false;
    if (!qz_is_b_invertible_(B)) return false;

    /* C = B⁻¹ · A — solve B · C = A column by column. */
    std::vector<double> LU(B->data, B->data + n * n);
    std::vector<int64_t> piv(n);
    int sgn;
    if (lu_decompose(LU.data(), n, piv.data(), &sgn) != 0) return false;
    std::vector<double> C(n * n);
    std::vector<double> rhs(n), x(n);
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t r = 0; r < n; ++r) rhs[r] = A->data[r * n + c];
        lu_solve_column(LU.data(), n, piv.data(), rhs.data(), x.data());
        for (int64_t r = 0; r < n; ++r) C[r * n + c] = x[r];
    }

    /* Real Schur of C: T = U' · C · U via the same Hessenberg + QR
     * machinery the schur entry uses. */
    std::vector<double> T = C;
    std::vector<double> U(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) U[i * n + i] = 1.0;
    hessenberg_inplace_(T.data(), n, U.data());
    francis_qr_(T.data(), n, U.data());

    /* M = B · U. */
    std::vector<double> M(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k)
                s += B->data[i * n + k] * U[k * n + j];
            M[i * n + j] = s;
        }

    /* QR of M (modified Gram-Schmidt) — returns O orthogonal and R
     * upper triangular with M = O · R. */
    matlab::runtime::MatPtr Mmat = matlab::runtime::make_mat(n, n);
    for (int64_t i = 0; i < n * n; ++i) Mmat->data[i] = M[i];
    matlab::runtime::MatPtr O = matlab::runtime::make_mat(n, n);
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(n, n);
    qr_factor(Mmat.get(), O.get(), R.get());

    /* Q = O' (transposed orthogonal so MATLAB's `Q*A*Z = AA` shape holds). */
    Q.assign(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) Q[j * n + i] = O->data[i * n + j];

    /* Z = U. */
    Z.assign(U.begin(), U.end());

    /* BB = R (upper triangular). */
    BB.assign(R->data, R->data + n * n);

    /* AA = R · T (upper quasi-triangular when T is). */
    AA.assign(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k)
                s += R->data[i * n + k] * T[k * n + j];
            AA[i * n + j] = s;
        }
    n_out = n;
    return true;
}

matlab_mat *qz_pick_(matlab_mat *A, matlab_mat *B, int which) {
    std::vector<double> AA, BB, Q, Z;
    int64_t n = 0;
    if (!qz_compute_(A, B, AA, BB, Q, Z, n)) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(n, n);
    const double *src = (which == 0) ? AA.data() :
                         (which == 1) ? BB.data() :
                         (which == 2) ? Q.data()  : Z.data();
    for (int64_t i = 0; i < n * n; ++i) out->data[i] = src[i];
    return out;
}

} // namespace

matlab_mat *matlab_qz_AA(matlab_mat *A, matlab_mat *B) {
    return qz_pick_(A, B, 0);
}
matlab_mat *matlab_qz_BB(matlab_mat *A, matlab_mat *B) {
    return qz_pick_(A, B, 1);
}
matlab_mat *matlab_qz_Q(matlab_mat *A, matlab_mat *B) {
    return qz_pick_(A, B, 2);
}
matlab_mat *matlab_qz_Z(matlab_mat *A, matlab_mat *B) {
    return qz_pick_(A, B, 3);
}

/*-------------------------------------------------------------------------
 * Generalised eigenvalue problem — `eig(A, B)` returning the
 * eigenvalues of the matrix pencil A − λB. Stages: QZ → diagonal
 * walk → quadratic on 2×2 quasi-blocks. Polymorphic real/complex
 * return (matches matlab_eig): pure-real spectrum returns a real
 * column matrix; any complex pair flips the return to
 * matlab_mat_c* (cast back to matlab_mat* for the dispatch ABI).
 *
 * 2×2 generalised eigenproblem: with AA_2 = AA[i:i+2, i:i+2] and
 * BB_2 = BB[i:i+2, i:i+2] (BB upper-triangular so BB_2 has a zero
 * sub-diagonal), det(AA_2 − λ·BB_2) = 0 expands to
 *   (b11·b22) λ²
 *     − (a11·b22 + a22·b11 − a21·b12) λ
 *     + (a11·a22 − a12·a21) = 0
 * which is a standard scalar quadratic. A zero leading coefficient
 * means an infinite eigenvalue (BB rank-deficient on the block) —
 * surfaced as NaN today.
 *
 * Tier 1 closure piece: matches the conversation in CST roadmap
 * §2.1 that the generalised `eig(A, B)` "is a small wrapper over
 * the already-shipped 4-return qz". B singular keeps the
 * `qz_compute_` 0×0 fallback (Moler-Stewart QZ remains a follow-on).
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_eig_gen(matlab_mat *A_in, matlab_mat *B_in) {
    std::vector<double> AA, BB, Q, Z;
    int64_t n = 0;
    if (!qz_compute_(A_in, B_in, AA, BB, Q, Z, n)) return mat_alloc(0, 0);

    std::vector<double> ere(n), eim(n);
    int complex_pairs = 0;
    int64_t i = 0;
    while (i < n) {
        bool is_2x2 = (i + 1 < n) &&
                      (std::fabs(AA[(i + 1) * n + i]) > 1e-12);
        if (is_2x2) {
            double a11 = AA[i * n + i],     a12 = AA[i * n + (i + 1)];
            double a21 = AA[(i + 1) * n + i], a22 = AA[(i + 1) * n + (i + 1)];
            double b11 = BB[i * n + i],     b12 = BB[i * n + (i + 1)];
            double b22 = BB[(i + 1) * n + (i + 1)];
            double Aq = b11 * b22;
            double Bq = -(a11 * b22 + a22 * b11 - a21 * b12);
            double Cq = a11 * a22 - a12 * a21;
            if (Aq == 0.0) {
                ere[i] = std::numeric_limits<double>::quiet_NaN();
                eim[i] = 0.0;
                ere[i + 1] = std::numeric_limits<double>::quiet_NaN();
                eim[i + 1] = 0.0;
            } else {
                double disc = Bq * Bq - 4.0 * Aq * Cq;
                if (disc >= 0) {
                    double sq = std::sqrt(disc);
                    ere[i]     = (-Bq + sq) / (2.0 * Aq); eim[i]     = 0.0;
                    ere[i + 1] = (-Bq - sq) / (2.0 * Aq); eim[i + 1] = 0.0;
                } else {
                    double sq = std::sqrt(-disc);
                    double re = -Bq / (2.0 * Aq);
                    double im =  sq / (2.0 * Aq);
                    ere[i]     = re; eim[i]     = im;
                    ere[i + 1] = re; eim[i + 1] = -im;
                    complex_pairs++;
                }
            }
            i += 2;
        } else {
            double b = BB[i * n + i];
            if (std::fabs(b) < 1e-14) {
                /* BB diagonal zero → infinite eigenvalue. Surface as
                 * +Inf so downstream isstable / pole tests behave
                 * predictably. */
                ere[i] = std::numeric_limits<double>::infinity();
            } else {
                ere[i] = AA[i * n + i] / b;
            }
            eim[i] = 0.0;
            i += 1;
        }
    }

    /* Sort by ascending real part, tie-break on imaginary. Mirrors
     * matlab_eig's order so test snapshots agree. */
    for (int64_t a = 0; a < n; ++a) {
        for (int64_t b = a + 1; b < n; ++b) {
            bool swap = (ere[b] < ere[a]) ||
                        (ere[b] == ere[a] && eim[b] < eim[a]);
            if (swap) {
                std::swap(ere[a], ere[b]);
                std::swap(eim[a], eim[b]);
            }
        }
    }

    if (complex_pairs == 0) {
        matlab_mat *E = mat_alloc(n, 1);
        for (int64_t k = 0; k < n; ++k) E->data[k] = ere[k];
        return E;
    }
    matlab_mat_c *Ec = mat_c_alloc(n, 1);
    for (int64_t k = 0; k < n; ++k) {
        Ec->re[k] = ere[k];
        Ec->im[k] = eim[k];
    }
    return (matlab_mat *)Ec;
}

/*-------------------------------------------------------------------------
 * Lyapunov / Stein equation solvers.
 *
 *   lyap(A, Q):    A X + X A' + Q = 0     (continuous Lyapunov)
 *   dlyap(A, Q):   A X A' - X + Q = 0     (discrete / Stein equation)
 *
 * Implementation: vectorize and LU-solve. Vectorising the matrix
 * equation in row-major form gives an n^2 * n^2 dense system that
 * the existing `lu_decompose` + `lu_solve_column` helpers handle
 * straight away. O(n^6) cost — fine for the small plants typical of
 * the Control System Toolbox surface (n typically 2-10). For large
 * plants the proper approach is Bartels-Stewart back-substitution
 * on the Schur form (uses matlab_schur from Tier-1.2 follow-on);
 * documented as a follow-on optimisation here.
 *
 * Vectorisation derivation (row-major vec):
 *   vec(A X)  = (A o I) vec(X)    [Kronecker A o I_n]
 *   vec(X A') = (I o A) vec(X)
 *   vec(A X A') = (A o A) vec(X)
 *
 * Continuous: (A o I + I o A) vec(X) = -vec(Q).
 * Discrete:   (A o A - I_{n^2}) vec(X) = -vec(Q).
 *
 * Tier 1.4 of the Control System Toolbox roadmap. See
 * docs/control_toolbox_roadmap.md §2.4. Gates `gram` (controllability /
 * observability gramians as Lyapunov solutions), the H2 system norm,
 * and the balanced realisation that underlies model reduction.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_lyap(matlab_mat *A, matlab_mat *Q) {
    if (!A || !Q || A->rows != A->cols ||
        Q->rows != A->rows || Q->cols != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0) return mat_alloc(0, 0);
    int64_t n2 = n * n;

    /* Build M = A o I + I o A (row-major Kronecker). */
    std::vector<double> M(n2 * n2, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < n; ++k) {
            double a_ik = A->data[i * n + k];
            for (int64_t j = 0; j < n; ++j)
                M[(i * n + j) * n2 + (k * n + j)] += a_ik;
        }
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            for (int64_t k = 0; k < n; ++k)
                M[(i * n + j) * n2 + (i * n + k)] += A->data[j * n + k];

    /* RHS = -vec(Q). */
    std::vector<double> rhs(n2);
    for (int64_t i = 0; i < n2; ++i) rhs[i] = -Q->data[i];

    std::vector<int64_t> piv(n2);
    int sgn;
    if (lu_decompose(M.data(), n2, piv.data(), &sgn) != 0)
        return mat_alloc(0, 0);

    std::vector<double> x(n2);
    lu_solve_column(M.data(), n2, piv.data(), rhs.data(), x.data());

    matlab_mat *X = mat_alloc(n, n);
    for (int64_t i = 0; i < n2; ++i) X->data[i] = x[i];
    return X;
}

/* 3-argument Sylvester equation: A·X + X·B + C = 0  (note the convention:
 * MATLAB's `lyap(A, B, C)` solves the equation with the +C sign — same
 * as MATLAB Toolbox documentation). A is n×n, B is m×m, C and X are n×m.
 *
 * Vectorisation (row-major):
 *   ((A o I_m) + (I_n o B^T)) · vec(X) = -vec(C)
 *
 * v1 implementation: dense LU on the (n·m)² Kronecker matrix. Same
 * O(N³) cost shape as the 2-arg lyap; the proper Bartels-Stewart on
 * the Schur forms of A and B is the large-plant follow-on.
 *
 * Tier 1.4 follow-on of CST roadmap §2.4. */
matlab_mat *matlab_sylvester(matlab_mat *A, matlab_mat *B, matlab_mat *C) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t m = B->rows;
    if (A->cols != n || B->cols != m) return mat_alloc(0, 0);
    if (C->rows != n || C->cols != m) return mat_alloc(0, 0);
    if (n == 0 || m == 0) return mat_alloc(0, 0);
    int64_t N = n * m;

    /* Build M[(i*m+j), (k*m+l)] = A[i,k]·δ_{j,l} + δ_{i,k}·B[l,j]. */
    std::vector<double> M(N * N, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            for (int64_t k = 0; k < n; ++k)
                M[(i * m + j) * N + (k * m + j)] += A->data[i * n + k];
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            for (int64_t l = 0; l < m; ++l)
                M[(i * m + j) * N + (i * m + l)] += B->data[l * m + j];

    std::vector<double> rhs(N);
    for (int64_t i = 0; i < N; ++i) rhs[i] = -C->data[i];

    std::vector<int64_t> piv(N);
    int sgn;
    if (lu_decompose(M.data(), N, piv.data(), &sgn) != 0)
        return mat_alloc(0, 0);

    std::vector<double> x(N);
    lu_solve_column(M.data(), N, piv.data(), rhs.data(), x.data());

    matlab_mat *X = mat_alloc(n, m);
    for (int64_t i = 0; i < N; ++i) X->data[i] = x[i];
    return X;
}

matlab_mat *matlab_dlyap(matlab_mat *A, matlab_mat *Q) {
    if (!A || !Q || A->rows != A->cols ||
        Q->rows != A->rows || Q->cols != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0) return mat_alloc(0, 0);
    int64_t n2 = n * n;

    /* Build M = A o A - I_{n^2}. Row-major Kronecker A o A:
     *   M[i n + j, k n + l] = A[i, k] * A[j, l]. */
    std::vector<double> M(n2 * n2, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            for (int64_t k = 0; k < n; ++k)
                for (int64_t l = 0; l < n; ++l)
                    M[(i * n + j) * n2 + (k * n + l)] =
                        A->data[i * n + k] * A->data[j * n + l];
    for (int64_t i = 0; i < n2; ++i) M[i * n2 + i] -= 1.0;

    std::vector<double> rhs(n2);
    for (int64_t i = 0; i < n2; ++i) rhs[i] = -Q->data[i];

    std::vector<int64_t> piv(n2);
    int sgn;
    if (lu_decompose(M.data(), n2, piv.data(), &sgn) != 0)
        return mat_alloc(0, 0);

    std::vector<double> x(n2);
    lu_solve_column(M.data(), n2, piv.data(), rhs.data(), x.data());

    matlab_mat *X = mat_alloc(n, n);
    for (int64_t i = 0; i < n2; ++i) X->data[i] = x[i];
    return X;
}

/*-------------------------------------------------------------------------
 * Continuous algebraic Riccati equation - X = care(A, B, Q, R).
 *
 * Solves  A'X + X A - X B R^{-1} B' X + Q = 0  for the unique
 * stabilising solution (X = X' >= 0; A - B R^{-1} B' X is Hurwitz).
 *
 * Algorithm: matrix sign function via Newton iteration on the
 * Hamiltonian matrix H = [[A, -B R^{-1} B']; [-Q, -A']]. After
 * convergence, sign(H) has eigenvalues +-1; P = (I - sign(H))/2
 * projects onto the stable invariant subspace, and X = P_bot * inv(P_top)
 * recovers the Riccati solution.
 *
 * Reference: Roberts (1980), "Linear model reduction and solution of
 * the algebraic Riccati equation by use of the sign function",
 * International J. of Control, 32(4):677-687. Newton iteration
 * S_{k+1} = (S_k + S_k^{-1}) / 2 converges quadratically when H has
 * no eigenvalue on the imaginary axis (i.e. for stabilisable +
 * detectable LQR setups). Each iteration is one inv() + add.
 *
 * Tier 1.5 of the Control System Toolbox roadmap. See
 * docs/control_toolbox_roadmap.md §2.5. Gates `lqr`, `kalman`, `lqg`,
 * and the H_inf system norm. The discrete-time variant `dare` is a
 * follow-on (needs the Cayley transform CARE<->DARE bridge or QZ).
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_care(matlab_mat *A, matlab_mat *B,
                        matlab_mat *Q, matlab_mat *R) {
    if (!A || !B || !Q || !R) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (A->cols != n || B->rows != n || Q->rows != n || Q->cols != n)
        return mat_alloc(0, 0);
    int64_t m = B->cols;
    if (R->rows != m || R->cols != m) return mat_alloc(0, 0);
    if (n == 0) return mat_alloc(0, 0);

    /* Compute B * R^{-1} * B'  (n x n). */
    matlab_mat *Rinv  = matlab_inv(R);
    if (!Rinv || Rinv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Bt    = matlab_transpose(B);
    matlab_mat *BR    = matlab_matmul_mm(B, Rinv);
    matlab_mat *BRiBt = matlab_matmul_mm(BR, Bt);

    /* Hamiltonian H = [[A, -BRiBt], [-Q, -A']] (2n x 2n). */
    int64_t n2 = 2 * n;
    std::vector<double> S(n2 * n2, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            S[i * n2 + j]            =  A->data[i * n + j];
            S[i * n2 + (n + j)]      = -BRiBt->data[i * n + j];
            S[(n + i) * n2 + j]      = -Q->data[i * n + j];
            S[(n + i) * n2 + (n + j)] = -A->data[j * n + i]; /* -A^T */
        }

    /* Newton iteration:  S_{k+1} = (S_k + S_k^{-1}) / 2. */
    std::vector<double> LU(n2 * n2), Sinv(n2 * n2);
    std::vector<int64_t> piv(n2);
    std::vector<double> rhs_col(n2), x_col(n2);
    int sgn;
    const int max_iters = 60;
    const double tol = 1e-12;
    bool converged = false;
    for (int iter = 0; iter < max_iters; ++iter) {
        for (int64_t i = 0; i < n2 * n2; ++i) LU[i] = S[i];
        if (lu_decompose(LU.data(), n2, piv.data(), &sgn) != 0) {
            /* Singular S - bail with empty result. Intermediate
             * matrices are arena-allocated (see Phase-4 RAII policy);
             * no explicit free needed. */
            return mat_alloc(0, 0);
        }
        for (int64_t c = 0; c < n2; ++c) {
            for (int64_t r = 0; r < n2; ++r) rhs_col[r] = (r == c) ? 1.0 : 0.0;
            lu_solve_column(LU.data(), n2, piv.data(),
                            rhs_col.data(), x_col.data());
            for (int64_t r = 0; r < n2; ++r) Sinv[r * n2 + c] = x_col[r];
        }
        double diff_fro = 0.0, S_fro = 0.0;
        for (int64_t i = 0; i < n2 * n2; ++i) {
            double Snew = 0.5 * (S[i] + Sinv[i]);
            double d = Snew - S[i];
            diff_fro += d * d;
            S_fro    += Snew * Snew;
            S[i] = Snew;
        }
        if (S_fro > 0.0 && diff_fro <= tol * tol * S_fro) {
            converged = true;
            break;
        }
    }
    if (!converged) return mat_alloc(0, 0);
    (void)Rinv; (void)Bt; (void)BR; (void)BRiBt; /* arena-managed */

    /* P = (I - S) / 2 projects onto the stable invariant subspace.
     * Take the first n columns: U_top = P[0..n-1, 0..n-1],
     * U_bot = P[n..2n-1, 0..n-1]. For a generic Hamiltonian without
     * imaginary-axis eigenvalues these are linearly independent and
     * U_top is invertible. */
    matlab_mat *Utop = mat_alloc(n, n);
    matlab_mat *Ubot = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double Iij = (i == j) ? 1.0 : 0.0;
            Utop->data[i * n + j] = 0.5 * (Iij - S[i * n2 + j]);
            Ubot->data[i * n + j] = -0.5 * S[(n + i) * n2 + j];
        }

    matlab_mat *Utop_inv = matlab_inv(Utop);
    if (!Utop_inv || Utop_inv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *X = matlab_matmul_mm(Ubot, Utop_inv);

    /* Symmetrise: X should be exactly symmetric; round-trip rounding
     * leaves a tiny asymmetric residue. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j) {
            double s = 0.5 * (X->data[i * n + j] + X->data[j * n + i]);
            X->data[i * n + j] = s;
            X->data[j * n + i] = s;
        }

    return X;
}

/* Forward decls — matlab_dare and a few elementwise helpers are
 * defined later in this TU. matlab_add_mm takes (void*, void*) for
 * complex/real polymorphism (see the BINARY_MM macro definition
 * further down). */
extern "C" matlab_mat *matlab_dare(matlab_mat *Ad, matlab_mat *Bd,
                                    matlab_mat *Q, matlab_mat *R);
extern "C" matlab_mat *matlab_neg_m(matlab_mat *A);
extern "C" matlab_mat *matlab_add_mm(void *A, void *B);

/*-------------------------------------------------------------------------
 * Numerically-robust algebraic Riccati entries — icare / idare.
 *
 *   MATLAB introduced `icare` and `idare` as the recommended successors
 *   to `care` / `dare` (since R2019a). The numerical advantage shows up
 *   on ill-conditioned pencils where the matrix-sign Newton iteration
 *   that backs `care` can stall: `icare` uses an extended-pencil
 *   structure-preserving generalised Schur algorithm
 *   (Mehrmann-Voss), `idare` uses the equivalent symplectic-pencil
 *   form. Both return the same `X` on well-conditioned inputs.
 *
 *   v1 implementation: alias to matlab_care / matlab_dare. The two
 *   names diverge in numerics on pencils that have eigenvalues very
 *   close to the imaginary axis (continuous) or unit circle (discrete);
 *   the Mehrmann-Voss extended pencil avoids the matrix-sign squaring
 *   step that loses 1 bit per iteration there. Shipping the structure-
 *   preserving QZ on the symplectic pencil is the proper follow-on
 *   (gated on the singular-B QZ path, which is the same generalised-
 *   Schur primitive that backs `zero(sys)`). For the small CST-roadmap
 *   plants (n = 2..10) the numerical gap is noise; the rename gives
 *   user code on the modern API surface a working entry today.
 *
 * Tier 1.5 follow-on of CST roadmap §2.5. */
matlab_mat *matlab_icare(matlab_mat *A, matlab_mat *B,
                         matlab_mat *Q, matlab_mat *R) {
    return matlab_care(A, B, Q, R);
}

matlab_mat *matlab_idare(matlab_mat *Ad, matlab_mat *Bd,
                         matlab_mat *Q, matlab_mat *R) {
    return matlab_dare(Ad, Bd, Q, R);
}

/*-------------------------------------------------------------------------
 * 5-arg algebraic Riccati with state-input cross-term — care/dare(A, B,
 * Q, R, S).
 *
 *   The cost functional J = ∫(x'Qx + 2x'Su + u'Ru) dt admits a
 *   reduction to the standard 4-arg form via the change of basis
 *     A_hat = A − B·R⁻¹·S'
 *     Q_hat = Q − S·R⁻¹·S'
 *   which preserves the stabilising solution (the cross-term is
 *   absorbed into the drift matrix and the state weighting).
 *   Discrete analogue:
 *     Ad_hat = Ad − Bd·R⁻¹·S'
 *     Qd_hat = Q  − S·R⁻¹·S'
 *   (same algebra; the dare path is Schur-stable when Ad − Bd·K is
 *   inside the unit disk).
 *
 *   Returns 0×0 when R is singular (S·R⁻¹ undefined) or when the
 *   reduced problem has no stabilising solution. The 6-arg
 *   `care(A, B, Q, R, S, E)` descriptor form (with generalised E·X·E'
 *   shape) reduces to the standard form when E = I and is the
 *   follow-on for E ≠ I (gated on the generalised-Riccati QZ path).
 *
 * Tier 1.5 follow-on of CST roadmap §2.5. */
matlab_mat *matlab_care_5(matlab_mat *A, matlab_mat *B, matlab_mat *Q,
                           matlab_mat *R, matlab_mat *S) {
    if (!A || !B || !Q || !R || !S) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t m = B->cols;
    if (S->rows != n || S->cols != m) return mat_alloc(0, 0);
    matlab_mat *Rinv = matlab_inv(R);
    if (!Rinv || Rinv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *St    = matlab_transpose(S);
    matlab_mat *RinvSt = matlab_matmul_mm(Rinv, St);     /* m × n */
    matlab_mat *BRinvSt = matlab_matmul_mm(B, RinvSt);    /* n × n */
    matlab_mat *negBRinvSt = matlab_neg_m(BRinvSt);
    matlab_mat *Ahat  = matlab_add_mm(A, negBRinvSt);
    matlab_mat *SRinvSt = matlab_matmul_mm(S, RinvSt);    /* n × n */
    matlab_mat *negSRinvSt = matlab_neg_m(SRinvSt);
    matlab_mat *Qhat  = matlab_add_mm(Q, negSRinvSt);
    return matlab_care(Ahat, B, Qhat, R);
}

matlab_mat *matlab_dare_5(matlab_mat *Ad, matlab_mat *Bd, matlab_mat *Q,
                           matlab_mat *R, matlab_mat *S) {
    if (!Ad || !Bd || !Q || !R || !S) return mat_alloc(0, 0);
    int64_t n = Ad->rows;
    int64_t m = Bd->cols;
    if (S->rows != n || S->cols != m) return mat_alloc(0, 0);
    matlab_mat *Rinv = matlab_inv(R);
    if (!Rinv || Rinv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *St    = matlab_transpose(S);
    matlab_mat *RinvSt = matlab_matmul_mm(Rinv, St);
    matlab_mat *BRinvSt = matlab_matmul_mm(Bd, RinvSt);
    matlab_mat *negBRinvSt = matlab_neg_m(BRinvSt);
    matlab_mat *Ahat = matlab_add_mm(Ad, negBRinvSt);
    matlab_mat *SRinvSt = matlab_matmul_mm(S, RinvSt);
    matlab_mat *negSRinvSt = matlab_neg_m(SRinvSt);
    matlab_mat *Qhat = matlab_add_mm(Q, negSRinvSt);
    return matlab_dare(Ahat, Bd, Qhat, R);
}

/*-------------------------------------------------------------------------
 * 5-arg LQR with state-input cross term `lqr(A, B, Q, R, N)` solves
 *   minimise  ∫ x'Q·x + 2·x'·N·u + u'·R·u  dt
 * Optimal feedback K = R⁻¹·(N' + B'·X) where X is the stabilising
 * solution of the 5-arg care. Same algebra for the discrete variant
 * with K = (R + B'·X·B)⁻¹·(N' + B'·X·A) and the X from dare_5.
 *
 * Tier 3.1 of CST roadmap §4.1 — the cross-term form is the natural
 * companion to the 5-arg care/dare already shipped, and lets users
 * write `K = lqr(A, B, Q, R, N)` for non-orthogonal output weights.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_lqr_5(matlab_mat *A, matlab_mat *B, matlab_mat *Q,
                         matlab_mat *R, matlab_mat *N) {
    if (!A || !B || !Q || !R || !N) return mat_alloc(0, 0);
    matlab_mat *X = matlab_care_5(A, B, Q, R, N);
    if (!X || X->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Rinv = matlab_inv(R);
    if (!Rinv || Rinv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Nt = matlab_transpose(N);
    matlab_mat *Bt = matlab_transpose(B);
    matlab_mat *BtX = matlab_matmul_mm(Bt, X);
    matlab_mat *NtPlusBtX = matlab_add_mm(Nt, BtX);
    return matlab_matmul_mm(Rinv, NtPlusBtX);
}

/*-------------------------------------------------------------------------
 * Output-weighted LQR — `K = lqry(sys, Q, R)`.
 *
 * Cost on outputs instead of states:
 *   J = ∫ y'·Q·y + u'·R·u  dt
 *     = ∫ (Cx + Du)'·Q·(Cx + Du) + u'·R·u  dt
 *     = ∫ x'·C'QC·x  +  2·x'·C'QD·u  +  u'·(R + D'QD)·u  dt
 * i.e. lqr_5(A, B, C'QC, R+D'QD, C'QD). For strictly-proper
 * plants (D = 0) this collapses to lqr(A, B, C'·Q·C, R).
 *
 * Tier 3.1 of CST roadmap §4.1. The model-object form
 * `K = lqry(sys, Q, R)` routes here via Lowering.cpp's
 * class-pinned-first-arg dispatch.
 *-------------------------------------------------------------------------*/
/* Forward decls — matlab_lqr lives later in this TU. */
extern "C" matlab_mat *matlab_lqr(matlab_mat *A, matlab_mat *B,
                                   matlab_mat *Q, matlab_mat *R);

matlab_mat *matlab_lqry_ss(matlab_mat *A, matlab_mat *B,
                            matlab_mat *C, matlab_mat *D,
                            matlab_mat *Q, matlab_mat *R) {
    if (!A || !B || !C || !D || !Q || !R) return mat_alloc(0, 0);
    matlab_mat *Ct  = matlab_transpose(C);
    matlab_mat *CtQ = matlab_matmul_mm(Ct, Q);
    matlab_mat *Qx  = matlab_matmul_mm(CtQ, C);
    /* Detect non-zero D — if all zeros, take the strictly-proper
     * branch (lqr 4-arg). Else fall back to lqr_5 with the cross
     * term N = C'·Q·D and effective R + D'·Q·D. */
    bool D_zero = true;
    for (int64_t k = 0; k < D->rows * D->cols; ++k) {
        if (D->data[k] != 0.0) { D_zero = false; break; }
    }
    if (D_zero) return matlab_lqr(A, B, Qx, R);
    matlab_mat *Dt   = matlab_transpose(D);
    matlab_mat *DtQ  = matlab_matmul_mm(Dt, Q);
    matlab_mat *DtQD = matlab_matmul_mm(DtQ, D);
    matlab_mat *Reff = matlab_add_mm(R, DtQD);
    matlab_mat *N    = matlab_matmul_mm(CtQ, D);   /* C'·Q·D */
    return matlab_lqr_5(A, B, Qx, Reff, N);
}

matlab_mat *matlab_dlqr_5(matlab_mat *Ad, matlab_mat *Bd, matlab_mat *Q,
                          matlab_mat *R, matlab_mat *N) {
    if (!Ad || !Bd || !Q || !R || !N) return mat_alloc(0, 0);
    matlab_mat *X = matlab_dare_5(Ad, Bd, Q, R, N);
    if (!X || X->rows == 0) return mat_alloc(0, 0);
    /* K = (R + B'·X·B)⁻¹·(N' + B'·X·A). */
    matlab_mat *Bt = matlab_transpose(Bd);
    matlab_mat *BtX = matlab_matmul_mm(Bt, X);
    matlab_mat *BtXB = matlab_matmul_mm(BtX, Bd);
    matlab_mat *RplusBtXB = matlab_add_mm(R, BtXB);
    matlab_mat *RplusBtXBinv = matlab_inv(RplusBtXB);
    if (!RplusBtXBinv || RplusBtXBinv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *BtXA = matlab_matmul_mm(BtX, Ad);
    matlab_mat *Nt = matlab_transpose(N);
    matlab_mat *NtPlusBtXA = matlab_add_mm(Nt, BtXA);
    return matlab_matmul_mm(RplusBtXBinv, NtPlusBtXA);
}

/*-------------------------------------------------------------------------
 * Continuous-to-discrete state-space conversion (zero-order hold).
 *
 *   [Ad, Bd] = c2d(A, B, Ts)
 *
 * For xdot = A x + B u with ZOH on u (held constant over each sample
 * interval), the discrete-time recurrence is
 *      x[k+1] = Ad x[k] + Bd u[k]
 * where
 *      Ad = expm(A * Ts)
 *      Bd = integral_0^Ts expm(A * tau) B dtau
 *
 * Augmented-matrix trick (van Loan): build  M = [[A, B]; [0, 0]] of
 * size (n+m) x (n+m) and compute  expm(M * Ts) = [[Ad, Bd]; [0, I]].
 * The top-left n*n block is Ad; the top-right n*m block is Bd. One
 * expm call gives both — much cleaner than computing the integral
 * directly. See Tier-2.2 of the CST roadmap.
 *
 * Two public entries (eig_V/eig_D precedent for multi-return splitting):
 *   matlab_c2d_Ad(A, B, Ts)  -> Ad
 *   matlab_c2d_Bd(A, B, Ts)  -> Bd
 * The MATLAB shape  [Ad, Bd] = c2d(A, B, Ts)  routes through the
 * existing 2-return dispatcher in LowerTensorOps.cpp.
 *-------------------------------------------------------------------------*/
static matlab_mat *c2d_aug_expm_(matlab_mat *A, matlab_mat *B, double Ts) {
    if (!A || !B || A->rows != A->cols ||
        B->rows != A->rows) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t m = B->cols;
    int64_t N = n + m;
    /* Build M = [[A * Ts, B * Ts]; [0, 0]]. Note: scaling by Ts here
     * means the resulting expm is exp(M * Ts) = exp([A B; 0 0] * Ts),
     * which the Van Loan identity gives as [[Ad, Bd]; [0, I_m]]. */
    matlab_mat *Maug = mat_alloc(N, N);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            Maug->data[i * N + j] = A->data[i * n + j] * Ts;
        for (int64_t j = 0; j < m; ++j)
            Maug->data[i * N + (n + j)] = B->data[i * m + j] * Ts;
    }
    /* Bottom rows already zero from mat_alloc's calloc. */
    return matlab_expm(Maug);
}

matlab_mat *matlab_c2d_Ad(matlab_mat *A, matlab_mat *B, double Ts) {
    matlab_mat *EM = c2d_aug_expm_(A, B, Ts);
    if (!EM || EM->rows == 0) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t N = EM->rows;
    matlab_mat *Ad = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            Ad->data[i * n + j] = EM->data[i * N + j];
    return Ad;
}

matlab_mat *matlab_c2d_Bd(matlab_mat *A, matlab_mat *B, double Ts) {
    matlab_mat *EM = c2d_aug_expm_(A, B, Ts);
    if (!EM || EM->rows == 0) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t m = B->cols;
    int64_t N = EM->rows;
    matlab_mat *Bd = mat_alloc(n, m);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            Bd->data[i * m + j] = EM->data[i * N + (n + j)];
    return Bd;
}

/*-------------------------------------------------------------------------
 * Controllability / observability gramians as Lyapunov solutions.
 *
 *   Wc = gram_c(A, B)  solves  A Wc + Wc A' + B B' = 0   ->  lyap(A, B B').
 *   Wo = gram_o(A, C)  solves  A' Wo + Wo A + C' C = 0   ->  lyap(A', C' C).
 *
 * Used by the H2 system norm  ||G||_2 = sqrt(trace(B' Wo B)) = sqrt(trace(C Wc C'))
 * and by balanced realisation. Tier 3.4 of the CST roadmap.
 *
 * The matlab_llvm matrix-arg API is functional (not model-object). Once
 * `ss` constructors land (Tier 2.1), the model-object form `gram(sys,'c')`
 * can be written in MATLAB-side as a one-liner over these helpers.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_gram_c(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(0, 0);
    matlab_mat *Bt   = matlab_transpose(B);
    matlab_mat *BBt  = matlab_matmul_mm(B, Bt);
    return matlab_lyap(A, BBt);
}

matlab_mat *matlab_gram_o(matlab_mat *A, matlab_mat *C) {
    if (!A || !C) return mat_alloc(0, 0);
    matlab_mat *At  = matlab_transpose(A);
    matlab_mat *Ct  = matlab_transpose(C);
    matlab_mat *CtC = matlab_matmul_mm(Ct, C);
    return matlab_lyap(At, CtC);
}

/*-------------------------------------------------------------------------
 * Cholesky factor of the controllability gramian — R = lyapchol(A, B).
 *
 *   lyapchol returns an upper-triangular R such that R' R = Wc, where
 *   Wc solves A·Wc + Wc·A' + B·B' = 0 (the controllability Lyapunov
 *   equation). It's the numerically robust input to balanced-truncation
 *   model reduction: SVD of R·R' = Wc gives the Hankel singular values
 *   without ever forming Wc explicitly, dodging the squaring-of-condition-
 *   number that hits a chol(Wc) round trip.
 *
 *   This v1 entry is the round-trip (compute Wc via lyap, then chol),
 *   which is fine for the small plants typical of the practical CST
 *   surface (n = 2..10) and the SPD inputs the gramian path produces.
 *   The square-root Lyapunov solver of Hammarling 1982 (which avoids
 *   forming Wc) is the proper large-plant path; deferred until Bartels-
 *   Stewart on Schur form lands.
 *
 * Tier 1.4 follow-on of the Control System Toolbox roadmap; see
 * docs/control_toolbox_roadmap.md §2.4. Gates the balanced-realisation
 * tail of Tier 4 model reduction.
 *-------------------------------------------------------------------------*/
/* Forward decl — matlab_chol is defined later in this TU. */
extern "C" matlab_mat *matlab_chol(matlab_mat *A);

matlab_mat *matlab_lyapchol(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(0, 0);
    matlab_mat *Wc = matlab_gram_c(A, B);
    if (!Wc || Wc->rows == 0) return mat_alloc(0, 0);
    return matlab_chol(Wc);
}

/*-------------------------------------------------------------------------
 * State-space unit-step response.
 *
 *   y = step_ss(A, B, C, D, dt, N)  returns the N*p output trajectory
 *   under unit-step input u = ones(m, 1) on all inputs simultaneously.
 *   Each row is one time sample; each column is one output channel.
 *
 * Method: ZOH discretise via c2d_aug_expm at sample interval dt, then run
 * the discrete recurrence  x[k+1] = Ad x[k] + Bd u,  y[k] = C x[k] + D u
 * starting from relaxed initial state x[0] = 0.
 *
 * Tier 2.3 of the CST roadmap. The plot-the-output form `step(sys)` is
 * deferred (no native plotting); programs render via fprintf / disp.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_step_ss(matlab_mat *A, matlab_mat *B,
                           matlab_mat *C, matlab_mat *D,
                           double dt, double N_in) {
    if (!A || !B || !C || !D || dt <= 0.0)
        return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t m = B->cols;
    int64_t p = C->rows;
    int64_t N = (int64_t)N_in;
    if (N <= 0) return mat_alloc(0, 0);
    if (A->cols != n || B->rows != n || C->cols != n ||
        D->rows != p || D->cols != m) return mat_alloc(0, 0);

    matlab_mat *Ad = matlab_c2d_Ad(A, B, dt);
    matlab_mat *Bd = matlab_c2d_Bd(A, B, dt);
    matlab_mat *u  = matlab_ones((double)m, 1);  /* unit step on each input */

    /* Output buffer y is N x p. Internal state x carried across iterations. */
    matlab_mat *y = mat_alloc(N, p);
    std::vector<double> x(n, 0.0);
    std::vector<double> xnew(n, 0.0);

    for (int64_t k = 0; k < N; ++k) {
        /* y[k, :] = C x + D u (each row is one time sample). */
        for (int64_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int64_t i = 0; i < n; ++i) sum += C->data[j * n + i] * x[i];
            for (int64_t i = 0; i < m; ++i) sum += D->data[j * m + i] * u->data[i];
            y->data[k * p + j] = sum;
        }
        /* x[k+1] = Ad x + Bd u. */
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += Ad->data[i * n + j] * x[j];
            for (int64_t j = 0; j < m; ++j) s += Bd->data[i * m + j] * u->data[j];
            xnew[i] = s;
        }
        for (int64_t i = 0; i < n; ++i) x[i] = xnew[i];
    }
    return y;
}

/*-------------------------------------------------------------------------
 * State-space impulse response.
 *
 *   y = impulse_ss(A, B, C, D, dt, N) returns the N*p output
 *   trajectory under a Dirac input at t = 0. Method: ZOH discretise
 *   to get Ad = expm(A·dt), then iterate
 *     x[0] = B    (the impulse pushes the state directly to B)
 *     y[k] = C · x[k]    for k = 0, 1, …, N-1
 *     x[k+1] = Ad · x[k]
 *   For strictly proper plants (D = 0) this is the canonical MATLAB
 *   shape. The Dirac-delta contribution D·δ(t) at t = 0 only shows up
 *   in proper plants and is dropped here (discretised away) — the
 *   sampled response captures the C·expm(A·t)·B continuous-time
 *   piece, which is the practically useful part for plotting / time-
 *   domain analysis.
 *
 * SISO and MIMO-1-input shapes both work; for MIMO with m inputs we
 * concatenate per-input impulse responses column-wise (deferred —
 * v1 takes the first input column of B only).
 *
 * Tier 2.3 of the CST roadmap.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_impulse_ss(matlab_mat *A, matlab_mat *B,
                              matlab_mat *C, matlab_mat *D,
                              double dt, double N_in) {
    if (!A || !B || !C || !D || dt <= 0.0) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t p = C->rows;
    int64_t N = (int64_t)N_in;
    if (N <= 0 || A->cols != n || B->rows != n ||
        C->cols != n || D->cols != B->cols ||
        D->rows != p) return mat_alloc(0, 0);

    matlab_mat *Ad = matlab_c2d_Ad(A, B, dt);
    matlab_mat *y  = mat_alloc(N, p);
    std::vector<double> x(n, 0.0);
    /* SISO / first-input branch: x[0] = B[:, 0]. */
    for (int64_t i = 0; i < n; ++i) x[i] = B->data[i * B->cols];
    std::vector<double> xnew(n, 0.0);

    for (int64_t k = 0; k < N; ++k) {
        for (int64_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < n; ++i) s += C->data[j * n + i] * x[i];
            y->data[k * p + j] = s;
        }
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += Ad->data[i * n + j] * x[j];
            xnew[i] = s;
        }
        for (int64_t i = 0; i < n; ++i) x[i] = xnew[i];
    }
    return y;
}

/*-------------------------------------------------------------------------
 * State-space initial-condition response.
 *
 *   y = initial_ss(A, B, C, D, x0, dt, N) returns the N*p output
 *   trajectory under zero input from initial state x0. The free
 *   response y(t) = C · expm(A·t) · x0. Discretised iteration:
 *     x[0] = x0
 *     y[k] = C · x[k]
 *     x[k+1] = Ad · x[k]    Ad = expm(A·dt)
 *   D does not contribute (u ≡ 0). B is kept in the signature so the
 *   dispatch matches the same 6-argument `(A, B, C, D, x0, dt)` shape
 *   the model-object short form `initial(sys, x0, t)` uses (B routes
 *   through c2d to mint Ad consistently with step_ss / impulse_ss).
 *
 * Tier 2.3 of the CST roadmap.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_initial_ss(matlab_mat *A, matlab_mat *B,
                              matlab_mat *C, matlab_mat *D,
                              matlab_mat *x0, double dt, double N_in) {
    (void)D;
    if (!A || !B || !C || !x0 || dt <= 0.0) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t p = C->rows;
    int64_t N = (int64_t)N_in;
    if (N <= 0 || A->cols != n || C->cols != n ||
        x0->rows * x0->cols != n) return mat_alloc(0, 0);

    matlab_mat *Ad = matlab_c2d_Ad(A, B, dt);
    matlab_mat *y  = mat_alloc(N, p);
    std::vector<double> x(n, 0.0);
    for (int64_t i = 0; i < n; ++i) x[i] = x0->data[i];
    std::vector<double> xnew(n, 0.0);

    for (int64_t k = 0; k < N; ++k) {
        for (int64_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < n; ++i) s += C->data[j * n + i] * x[i];
            y->data[k * p + j] = s;
        }
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += Ad->data[i * n + j] * x[j];
            xnew[i] = s;
        }
        for (int64_t i = 0; i < n; ++i) x[i] = xnew[i];
    }
    return y;
}

/*-------------------------------------------------------------------------
 * Transfer-function frequency response: bode_tf(b, a, w).
 *
 *   H(s) = b(s) / a(s) = (b[0] s^N + b[1] s^(N-1) + ... + b[N])
 *                       / (a[0] s^M + a[1] s^(M-1) + ... + a[M])
 *
 * Polynomial-coefficient form following the MATLAB convention
 * (highest-power first). For each frequency w[k] evaluate b(jw) and
 * a(jw) via complex Horner, then H = b(jw) / a(jw); return magnitude
 * (linear) or phase (degrees).
 *
 * Tier-2.4 follow-on. Bridges to SPT users who work in (b, a) form
 * for analog filters. Same eig_V/eig_D 2-return precedent as bode_ss.
 *-------------------------------------------------------------------------*/
static void bode_tf_at_freq_(matlab_mat *b, matlab_mat *a, double w,
                             double *Hr, double *Hi) {
    int64_t Nb = b->rows * b->cols;
    int64_t Na = a->rows * a->cols;
    /* Horner with s = jw: at each step, result = result * (jw) + p[k].
     * (br + j bi) * (jw) = -bi w + j br w. */
    double br = 0.0, bi = 0.0;
    for (int64_t k = 0; k < Nb; ++k) {
        double nbr = -bi * w + b->data[k];
        double nbi =  br * w;
        br = nbr; bi = nbi;
    }
    double ar = 0.0, ai = 0.0;
    for (int64_t k = 0; k < Na; ++k) {
        double nar = -ai * w + a->data[k];
        double nai =  ar * w;
        ar = nar; ai = nai;
    }
    /* H = (br + j bi) / (ar + j ai)
     *   = ((br + j bi)(ar - j ai)) / (ar^2 + ai^2). */
    double d = ar * ar + ai * ai;
    if (d > 1e-300) {
        *Hr = (br * ar + bi * ai) / d;
        *Hi = (bi * ar - br * ai) / d;
    } else { *Hr = 0.0; *Hi = 0.0; }
}

matlab_mat *matlab_bode_tf_mag(matlab_mat *b, matlab_mat *a, matlab_mat *w) {
    if (!b || !a || !w) return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat *mag = mat_alloc(Nf, 1);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_tf_at_freq_(b, a, w->data[k], &Hr, &Hi);
        mag->data[k] = sqrt(Hr * Hr + Hi * Hi);
    }
    return mag;
}

matlab_mat *matlab_bode_tf_phase(matlab_mat *b, matlab_mat *a, matlab_mat *w) {
    if (!b || !a || !w) return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat *phase = mat_alloc(Nf, 1);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_tf_at_freq_(b, a, w->data[k], &Hr, &Hi);
        phase->data[k] = atan2(Hi, Hr) * 180.0 / M_PI;
    }
    return phase;
}

/*-------------------------------------------------------------------------
 * Generalised input simulation - y = lsim_ss(A, B, C, D, u, dt).
 *
 * Same shape as step_ss but the input `u` is an N*m matrix (each row
 * is one sample of the m-input vector). ZOH between samples; relaxed
 * initial state x[0] = 0.
 *
 * y is N*p (each row is the output at the corresponding sample).
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_lsim_ss(matlab_mat *A, matlab_mat *B,
                           matlab_mat *C, matlab_mat *D,
                           matlab_mat *u, double dt) {
    if (!A || !B || !C || !D || !u || dt <= 0.0)
        return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t m = B->cols;
    int64_t p = C->rows;
    int64_t N = u->rows;
    if (A->cols != n || B->rows != n || C->cols != n ||
        D->rows != p || D->cols != m || u->cols != m)
        return mat_alloc(0, 0);
    if (N <= 0) return mat_alloc(0, 0);

    matlab_mat *Ad = matlab_c2d_Ad(A, B, dt);
    matlab_mat *Bd = matlab_c2d_Bd(A, B, dt);
    matlab_mat *y  = mat_alloc(N, p);
    std::vector<double> x(n, 0.0), xnew(n, 0.0);
    for (int64_t k = 0; k < N; ++k) {
        for (int64_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < n; ++i) s += C->data[j * n + i] * x[i];
            for (int64_t i = 0; i < m; ++i)
                s += D->data[j * m + i] * u->data[k * m + i];
            y->data[k * p + j] = s;
        }
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += Ad->data[i * n + j] * x[j];
            for (int64_t j = 0; j < m; ++j)
                s += Bd->data[i * m + j] * u->data[k * m + j];
            xnew[i] = s;
        }
        for (int64_t i = 0; i < n; ++i) x[i] = xnew[i];
    }
    return y;
}

/* Forward declarations - matlab_bode_ss_mag/phase are defined below. */
matlab_mat *matlab_bode_ss_mag  (matlab_mat *A, matlab_mat *B,
                                 matlab_mat *C, matlab_mat *D, matlab_mat *w);
matlab_mat *matlab_bode_ss_phase(matlab_mat *A, matlab_mat *B,
                                 matlab_mat *C, matlab_mat *D, matlab_mat *w);

/*-------------------------------------------------------------------------
 * Stability margins for SISO open-loop L(s) = C (sI - A)^{-1} B + D.
 *
 *   Gm = gain_margin(A, B, C, D, w)
 *   Pm = phase_margin(A, B, C, D, w)
 *
 * Both scan a user-provided frequency grid `w` (typically logspaced).
 * Linear interpolation between adjacent samples locates the crossover.
 *
 * Gain margin:   the smallest 1/|L(jw)| where phase(L) crosses -180
 *                degrees. Returns +Inf if phase never reaches -180 on
 *                the grid (system has infinite gain margin -- the
 *                first-order lowpass case).
 *
 * Phase margin:  180 + phase(L(jw)) at the gain crossover (|L| = 1).
 *                Returns +Inf if |L| never crosses 1 on the grid (the
 *                low-DC-gain case where L < 1 everywhere).
 *
 * Tier 2.4 follow-on. Sits cleanly on bode_ss.
 *-------------------------------------------------------------------------*/
double matlab_gain_margin(matlab_mat *A, matlab_mat *B,
                          matlab_mat *C, matlab_mat *D,
                          matlab_mat *w) {
    if (!A || !B || !C || !D || !w) return INFINITY;
    int64_t Nf = w->rows * w->cols;
    if (Nf < 2) return INFINITY;
    matlab_mat *phase = matlab_bode_ss_phase(A, B, C, D, w);
    matlab_mat *mag   = matlab_bode_ss_mag  (A, B, C, D, w);
    /* Find the first w[k] where phase crosses -180 from above. */
    for (int64_t k = 1; k < Nf; ++k) {
        double p1 = phase->data[k - 1];
        double p2 = phase->data[k];
        if (p1 > -180.0 && p2 <= -180.0) {
            double frac = (p1 + 180.0) / (p1 - p2);
            double m1 = mag->data[k - 1], m2 = mag->data[k];
            double mc = m1 + frac * (m2 - m1);
            if (mc > 1e-300) return 1.0 / mc;
            return INFINITY;
        }
    }
    return INFINITY;
}

double matlab_phase_margin(matlab_mat *A, matlab_mat *B,
                           matlab_mat *C, matlab_mat *D,
                           matlab_mat *w) {
    if (!A || !B || !C || !D || !w) return INFINITY;
    int64_t Nf = w->rows * w->cols;
    if (Nf < 2) return INFINITY;
    matlab_mat *phase = matlab_bode_ss_phase(A, B, C, D, w);
    matlab_mat *mag   = matlab_bode_ss_mag  (A, B, C, D, w);
    /* Find the first w[k] where |L| crosses 1 from above. */
    for (int64_t k = 1; k < Nf; ++k) {
        double m1 = mag->data[k - 1], m2 = mag->data[k];
        if (m1 > 1.0 && m2 <= 1.0) {
            double frac = (m1 - 1.0) / (m1 - m2);
            double p1 = phase->data[k - 1], p2 = phase->data[k];
            double pc = p1 + frac * (p2 - p1);
            return 180.0 + pc;
        }
    }
    return INFINITY;
}

/*-------------------------------------------------------------------------
 * State-space frequency response (SISO).
 *
 *   [mag, phase] = bode_ss(A, B, C, D, w)
 *
 * For each frequency w[k], evaluates  H(jw) = C (jw I - A)^{-1} B + D
 * and returns linear magnitude (not dB) and phase (in degrees).
 *
 * Complex linear solve via the real block decomposition:
 *    M_complex = jw I - A = -A + j (w I)
 *   [[-A, -w I];  [w I, -A]]  [Xr; Xi]  =  [B; 0]
 * The result X_complex = Xr + j Xi has real and imaginary blocks.
 *
 * This avoids a complex LU - we reuse the existing real `lu_decompose`
 * + `lu_solve_column` helpers on a 2n x 2n system. Cost is 4x the
 * single-frequency case versus a true complex LU but for typical
 * control plants (n = 2..10) it's fast enough and dependency-light.
 *
 * SISO only: B must be n*1 (single input), C must be 1*n (single
 * output), D must be 1*1. MIMO is a follow-on (build the full complex
 * H matrix per freq and stack).
 *
 * Tier 2.4 of the CST roadmap (gating margin, allmargin, getPeakGain,
 * sigma 1-output, dcgain, bandwidth — all of which need freqresp).
 *-------------------------------------------------------------------------*/
static int bode_ss_at_freq_(matlab_mat *A, matlab_mat *B,
                            matlab_mat *C, matlab_mat *D,
                            double w, double *Hr_out, double *Hi_out) {
    int64_t n = A->rows;
    int64_t N = 2 * n;
    std::vector<double> M(N * N, 0.0);
    /* Top-left and bottom-right: -A. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double a = A->data[i * n + j];
            M[i * N + j]               = -a;
            M[(n + i) * N + (n + j)]   = -a;
        }
    /* Top-right -wI, bottom-left +wI. */
    for (int64_t i = 0; i < n; ++i) {
        M[i * N + (n + i)]   = -w;
        M[(n + i) * N + i]   =  w;
    }
    std::vector<double> rhs(N, 0.0);
    for (int64_t i = 0; i < n; ++i) rhs[i] = B->data[i];  /* B is n x 1 */

    std::vector<int64_t> piv(N);
    int sgn;
    if (lu_decompose(M.data(), N, piv.data(), &sgn) != 0) {
        *Hr_out = 0.0; *Hi_out = 0.0;
        return -1;
    }
    std::vector<double> X(N);
    lu_solve_column(M.data(), N, piv.data(), rhs.data(), X.data());

    /* H = C * (Xr + j Xi) + D. C is 1 x n, D is 1 x 1. */
    double Hr = 0.0, Hi = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        Hr += C->data[i] * X[i];
        Hi += C->data[i] * X[n + i];
    }
    Hr += D->data[0];
    *Hr_out = Hr;
    *Hi_out = Hi;
    return 0;
}

matlab_mat *matlab_bode_ss_mag(matlab_mat *A, matlab_mat *B,
                                matlab_mat *C, matlab_mat *D,
                                matlab_mat *w) {
    if (!A || !B || !C || !D || !w) return mat_alloc(0, 0);
    if (A->rows != A->cols || B->cols != 1 ||
        C->rows != 1 || D->rows != 1 || D->cols != 1)
        return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat *mag = mat_alloc(Nf, 1);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_ss_at_freq_(A, B, C, D, w->data[k], &Hr, &Hi);
        mag->data[k] = sqrt(Hr * Hr + Hi * Hi);
    }
    return mag;
}

matlab_mat *matlab_bode_ss_phase(matlab_mat *A, matlab_mat *B,
                                  matlab_mat *C, matlab_mat *D,
                                  matlab_mat *w) {
    if (!A || !B || !C || !D || !w) return mat_alloc(0, 0);
    if (A->rows != A->cols || B->cols != 1 ||
        C->rows != 1 || D->rows != 1 || D->cols != 1)
        return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat *phase = mat_alloc(Nf, 1);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_ss_at_freq_(A, B, C, D, w->data[k], &Hr, &Hi);
        phase->data[k] = atan2(Hi, Hr) * 180.0 / M_PI;
    }
    return phase;
}

/*-------------------------------------------------------------------------
 * Raw complex frequency response — `H = freqresp(sys, w)`.
 *
 *   freqresp_ss(A, B, C, D, w) → matlab_mat_c with N rows × 1 col
 *   freqresp_tf(b, a, w)       → matlab_mat_c with N rows × 1 col
 *
 * Returns the complex transfer-function evaluation at each frequency
 * — the underlying quantity that bode / nyquist / nichols all
 * sample. Backs the model-object short form `freqresp(sys, w)`.
 *
 * Tier 2.4 of the CST roadmap.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_freqresp_ss(matlab_mat *A, matlab_mat *B,
                                matlab_mat *C, matlab_mat *D,
                                matlab_mat *w) {
    if (!A || !B || !C || !D || !w) return mat_alloc(0, 0);
    if (A->rows != A->cols || B->cols != 1 ||
        C->rows != 1 || D->rows != 1 || D->cols != 1)
        return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat_c *H = mat_c_alloc(Nf, 1);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_ss_at_freq_(A, B, C, D, w->data[k], &Hr, &Hi);
        H->re[k] = Hr;
        H->im[k] = Hi;
    }
    return (matlab_mat *)H;
}

matlab_mat *matlab_freqresp_tf(matlab_mat *b, matlab_mat *a, matlab_mat *w) {
    if (!b || !a || !w) return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat_c *H = mat_c_alloc(Nf, 1);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_tf_at_freq_(b, a, w->data[k], &Hr, &Hi);
        H->re[k] = Hr;
        H->im[k] = Hi;
    }
    return (matlab_mat *)H;
}

/*-------------------------------------------------------------------------
 * Nyquist plot data — `[re, im] = nyquist(sys, w)`.
 *
 *   nyquist_ss(A, B, C, D, w) → matlab_mat N×2 with columns [re, im]
 *   nyquist_tf(b, a, w)       → matlab_mat N×2 with columns [re, im]
 *
 * Two real columns rather than a complex vector — easier for the
 * common "plot real vs imaginary" downstream use, and dodges the
 * complex-emit-lane formatting drift. Users who want complex values
 * call `freqresp(sys, w)` instead.
 *
 * Tier 2.4 of the CST roadmap.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_nyquist_ss(matlab_mat *A, matlab_mat *B,
                               matlab_mat *C, matlab_mat *D,
                               matlab_mat *w) {
    if (!A || !B || !C || !D || !w) return mat_alloc(0, 0);
    if (A->rows != A->cols || B->cols != 1 ||
        C->rows != 1 || D->rows != 1 || D->cols != 1)
        return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat *RI = mat_alloc(Nf, 2);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_ss_at_freq_(A, B, C, D, w->data[k], &Hr, &Hi);
        RI->data[k * 2 + 0] = Hr;
        RI->data[k * 2 + 1] = Hi;
    }
    return RI;
}

matlab_mat *matlab_nyquist_tf(matlab_mat *b, matlab_mat *a, matlab_mat *w) {
    if (!b || !a || !w) return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat *RI = mat_alloc(Nf, 2);
    for (int64_t k = 0; k < Nf; ++k) {
        double Hr, Hi;
        bode_tf_at_freq_(b, a, w->data[k], &Hr, &Hi);
        RI->data[k * 2 + 0] = Hr;
        RI->data[k * 2 + 1] = Hi;
    }
    return RI;
}

/*-------------------------------------------------------------------------
 * `allmargin(sys)` — gathers gain / phase margins + their crossover
 * frequencies into a single 1×4 row [Gm, Pm, Wcg, Wcp]. MATLAB's
 * struct return is a follow-on (needs the `Inf` / `NaN` field
 * encoding); the row return is the bandwidth_ss / dcgain_ss shape.
 *
 * Gm    = gain_margin(A, B, C, D, w)
 * Pm    = phase_margin(A, B, C, D, w)
 * Wcg   = the ω where phase crosses −180° (gain crossover freq for
 *         the gain margin); Inf if no crossing.
 * Wcp   = the ω where |H| crosses 1 (phase crossover freq for the
 *         phase margin); Inf if no crossing.
 *
 * Tier 2.4 of the CST roadmap.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_allmargin_ss(matlab_mat *A, matlab_mat *B,
                                 matlab_mat *C, matlab_mat *D,
                                 matlab_mat *w) {
    if (!A || !B || !C || !D || !w) return mat_alloc(0, 0);
    int64_t Nf = w->rows * w->cols;
    matlab_mat *out = mat_alloc(1, 4);
    out->data[0] = matlab_gain_margin(A, B, C, D, w);
    out->data[1] = matlab_phase_margin(A, B, C, D, w);
    /* Recompute the crossover frequencies inline. */
    double wcg = INFINITY, wcp = INFINITY;
    if (Nf >= 2) {
        matlab_mat *phase = matlab_bode_ss_phase(A, B, C, D, w);
        matlab_mat *mag   = matlab_bode_ss_mag  (A, B, C, D, w);
        for (int64_t k = 1; k < Nf; ++k) {
            double p1 = phase->data[k - 1], p2 = phase->data[k];
            if (wcg == INFINITY && p1 > -180.0 && p2 <= -180.0) {
                double frac = (p1 + 180.0) / (p1 - p2);
                wcg = w->data[k - 1] +
                      frac * (w->data[k] - w->data[k - 1]);
            }
            double m1 = mag->data[k - 1], m2 = mag->data[k];
            if (wcp == INFINITY && m1 > 1.0 && m2 <= 1.0) {
                double frac = (m1 - 1.0) / (m1 - m2);
                wcp = w->data[k - 1] +
                      frac * (w->data[k] - w->data[k - 1]);
            }
        }
    }
    out->data[2] = wcg;
    out->data[3] = wcp;
    return out;
}

/*-------------------------------------------------------------------------
 * Discrete algebraic Riccati equation.
 *
 *   X = dare(Ad, Bd, Q, R)  solves
 *       Ad' X Ad - X - Ad' X Bd (R + Bd' X Bd)^{-1} Bd' X Ad + Q = 0
 *   for the unique stabilising solution.
 *
 * Algorithm: Newton-Kleinman iteration from K_0 = 0 (so X_0 starts as the
 * dlyap solution of (Ad', Q), the open-loop output covariance). At each
 * step compute  K_k = (R + Bd' X_k Bd)^{-1} Bd' X_k Ad,
 *               A_cl = Ad - Bd K_k,
 *               X_{k+1} = dlyap(A_cl', Q + K_k' R K_k).
 * Newton iterations preserve the closed-loop stability property
 * (Hewer 1971), so once K_0 stabilises, the iteration converges
 * quadratically to the unique stabilising solution.
 *
 * Limitation: K_0 = 0 stabilises only when Ad is already Schur-stable
 * (eigenvalues inside the unit disk). For unstable Ad the user must
 * pre-stabilise (e.g. via continuous-time lqr on the pre-discretised
 * plant). Returns empty 0x0 if the iteration diverges.
 *
 * Tier 1.5 follow-on. See docs/control_toolbox_roadmap.md §2.5. The
 * direct symplectic-pencil approach via QZ is the textbook large-scale
 * algorithm; deferred until QZ is shipped.
 *-------------------------------------------------------------------------*/
/* Forward declarations: matlab_add_mm is defined via the BINARY_MM
 * macro (void* signature for complex/real polymorphism); matlab_neg_m
 * is defined via UNARY_M further down. */
matlab_mat *matlab_add_mm(void *A, void *B);
matlab_mat *matlab_neg_m(matlab_mat *A);

matlab_mat *matlab_dare(matlab_mat *Ad, matlab_mat *Bd,
                        matlab_mat *Q, matlab_mat *R) {
    if (!Ad || !Bd || !Q || !R) return mat_alloc(0, 0);
    int64_t n = Ad->rows;
    int64_t m = Bd->cols;
    if (Ad->cols != n || Bd->rows != n || Q->rows != n || Q->cols != n ||
        R->rows != m || R->cols != m) return mat_alloc(0, 0);
    if (n == 0) return mat_alloc(0, 0);

    /* X_0 is the open-loop output covariance under Q: dlyap(Ad', Q). */
    matlab_mat *Adt = matlab_transpose(Ad);
    matlab_mat *X   = matlab_dlyap(Adt, Q);
    if (!X || X->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Bdt = matlab_transpose(Bd);

    const int max_iter = 60;
    const double tol = 1e-12;
    matlab_mat *Xprev = mat_alloc(n, n);
    for (int iter = 0; iter < max_iter; ++iter) {
        /* K = inv(R + Bd' X Bd) * (Bd' X Ad) */
        matlab_mat *XB    = matlab_matmul_mm(X, Bd);
        matlab_mat *BtXB  = matlab_matmul_mm(Bdt, XB);
        matlab_mat *S     = matlab_add_mm(R, BtXB);
        matlab_mat *Sinv  = matlab_inv(S);
        if (!Sinv || Sinv->rows == 0) return mat_alloc(0, 0);
        matlab_mat *XAd   = matlab_matmul_mm(X, Ad);
        matlab_mat *BtXAd = matlab_matmul_mm(Bdt, XAd);
        matlab_mat *K     = matlab_matmul_mm(Sinv, BtXAd);
        /* Acl = Ad - Bd K */
        matlab_mat *BdK   = matlab_matmul_mm(Bd, K);
        matlab_mat *negBK = matlab_neg_m(BdK);
        matlab_mat *Acl   = matlab_add_mm(Ad, negBK);
        /* Q_aug = Q + K' R K */
        matlab_mat *Kt    = matlab_transpose(K);
        matlab_mat *RK    = matlab_matmul_mm(R, K);
        matlab_mat *KtRK  = matlab_matmul_mm(Kt, RK);
        matlab_mat *Qaug  = matlab_add_mm(Q, KtRK);
        /* X_{k+1} = dlyap(Acl', Q_aug). */
        matlab_mat *Aclt  = matlab_transpose(Acl);
        matlab_mat *Xnew  = matlab_dlyap(Aclt, Qaug);
        if (!Xnew || Xnew->rows == 0) return mat_alloc(0, 0);
        /* Convergence: ||Xnew - X||_F / ||Xnew||_F. */
        double diff = 0.0, xn = 0.0;
        for (int64_t i = 0; i < n * n; ++i) {
            double d = Xnew->data[i] - X->data[i];
            diff += d * d;
            xn   += Xnew->data[i] * Xnew->data[i];
            Xprev->data[i] = X->data[i];
        }
        X = Xnew;
        if (xn > 0.0 && diff <= tol * tol * xn) break;
    }
    /* Symmetrise. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j) {
            double s = 0.5 * (X->data[i * n + j] + X->data[j * n + i]);
            X->data[i * n + j] = s;
            X->data[j * n + i] = s;
        }
    return X;
}

/* Discrete LQR — wrapper. K = (R + B' X B)^{-1} B' X A. */
matlab_mat *matlab_dlqr(matlab_mat *Ad, matlab_mat *Bd,
                        matlab_mat *Q, matlab_mat *R) {
    matlab_mat *X = matlab_dare(Ad, Bd, Q, R);
    if (!X || X->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Bdt   = matlab_transpose(Bd);
    matlab_mat *XBd   = matlab_matmul_mm(X, Bd);
    matlab_mat *BtXB  = matlab_matmul_mm(Bdt, XBd);
    matlab_mat *S     = matlab_add_mm(R, BtXB);
    matlab_mat *Sinv  = matlab_inv(S);
    if (!Sinv || Sinv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *XAd   = matlab_matmul_mm(X, Ad);
    matlab_mat *BtXAd = matlab_matmul_mm(Bdt, XAd);
    matlab_mat *K     = matlab_matmul_mm(Sinv, BtXAd);
    return K;
}

/*-------------------------------------------------------------------------
 * Linear-quadratic regulator (continuous LTI).
 *
 *   K = lqr(A, B, Q, R)  computes the optimal state-feedback gain
 *   minimising the cost  J = integral_0^infty (x' Q x + u' R u) dt
 *   for  xdot = A x + B u. The gain is  K = R^{-1} B' X  where X is
 *   the unique stabilising solution of the algebraic Riccati equation
 *      A' X + X A - X B R^{-1} B' X + Q = 0
 *   (provided by matlab_care; see Tier 1.5).
 *
 *   The closed-loop dynamics  Acl = A - B K  are Hurwitz; the closed-
 *   loop poles are eig(Acl). Returned size: K is m x n where m = B->cols
 *   (number of inputs).
 *
 * Tier 2.4 entry point in the Control System Toolbox roadmap. The
 * 3-return MATLAB shape `[K, S, e] = lqr(A, B, Q, R)` is a follow-on;
 * S = X (the Riccati solution) is exactly what care returns and e is
 * eig(A - B*K), so users can recover them today by calling care + eig
 * directly. Same applies to lqi/lqry which pre-augment A/Q/R.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_lqr(matlab_mat *A, matlab_mat *B,
                       matlab_mat *Q, matlab_mat *R) {
    if (!A || !B || !Q || !R) return mat_alloc(0, 0);
    /* Solve the Riccati for X. */
    matlab_mat *X = matlab_care(A, B, Q, R);
    if (!X || X->rows == 0) return mat_alloc(0, 0);
    /* K = R^{-1} B' X. */
    matlab_mat *Bt    = matlab_transpose(B);
    matlab_mat *BtX   = matlab_matmul_mm(Bt, X);   /* (m x n) */
    matlab_mat *Rinv  = matlab_inv(R);
    if (!Rinv || Rinv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *K     = matlab_matmul_mm(Rinv, BtX); /* (m x n) */
    return K;
}

/*-------------------------------------------------------------------------
 * Closed-loop poles for the 3-return [K, S, e] = lqr(A, B, Q, R) shape.
 *   e = eig(A - B*K)
 * with K computed via matlab_lqr internally. Polymorphic real/complex
 * (eig returns matlab_mat_c when the closed-loop spectrum has imaginary
 * parts). Routes the third result of the multi-return splitter; the
 * existing matlab_lqr / matlab_care entries cover K and S.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_lqr_e(matlab_mat *A, matlab_mat *B,
                         matlab_mat *Q, matlab_mat *R) {
    matlab_mat *K = matlab_lqr(A, B, Q, R);
    if (!K || K->rows == 0) return mat_alloc(0, 0);
    matlab_mat *BK   = matlab_matmul_mm(B, K);
    matlab_mat *nBK  = matlab_neg_m(BK);
    matlab_mat *Acl  = matlab_add_mm(A, nBK);
    return matlab_eig(Acl);
}

/* Discrete companion: e = eig(Ad - Bd*K) where K = dlqr(...). */
matlab_mat *matlab_dlqr_e(matlab_mat *Ad, matlab_mat *Bd,
                          matlab_mat *Q, matlab_mat *R) {
    matlab_mat *K = matlab_dlqr(Ad, Bd, Q, R);
    if (!K || K->rows == 0) return mat_alloc(0, 0);
    matlab_mat *BK   = matlab_matmul_mm(Bd, K);
    matlab_mat *nBK  = matlab_neg_m(BK);
    matlab_mat *Acl  = matlab_add_mm(Ad, nBK);
    return matlab_eig(Acl);
}

/*-------------------------------------------------------------------------
 * Balancing similarity transformation (continuous LTI).
 *
 *   T = balreal_T(A, B, C)  returns an  n x n  similarity transform
 *   such that the realization (A_b, B_b, C_b) = (T^{-1} A T, T^{-1} B,
 *   C T) is internally balanced — i.e. its controllability and
 *   observability gramians are equal and diagonal, with diagonal
 *   entries equal to the Hankel singular values (sorted descending).
 *
 *   Algorithm (Laub 1980, eigendecomposition variant — no Cholesky):
 *     Wc = gram_c(A, B), Wo = gram_o(A, C)
 *     Wc symmetric PSD → eig_sym(Wc) gives  Wc = V_c D_c V_c'
 *     Symmetric square root  X = V_c sqrt(D_c) V_c'  (X² = Wc)
 *     M = X' Wo X = X Wo X (X symmetric); also sym PSD.
 *     M = U S² U' (sym eig); then T = X U S^{-1/2}.
 *   After this T, Wc_new = Wo_new = S = diag(HSVs) (descending).
 *
 *   Requires A Hurwitz so the gramians are bounded; returns 0x0 if
 *   the gramian solves fail.
 *
 *   Tier-4 of CST roadmap (model reduction). The full
 *   `[Ab, Bb, Cb, hsv] = balreal(A, B, C)` 4-return shape is a
 *   follow-on; users assemble the balanced realization today via
 *      T = balreal_T(A, B, C); Ti = inv(T);
 *      Ab = Ti * A * T; Bb = Ti * B; Cb = C * T;
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_balreal_T(matlab_mat *A, matlab_mat *B, matlab_mat *C) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0 || A->cols != n) return mat_alloc(0, 0);

    matlab_mat *Wc = matlab_gram_c(A, B);
    if (!Wc || Wc->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Wo = matlab_gram_o(A, C);
    if (!Wo || Wo->rows == 0) return mat_alloc(0, 0);

    /* Symmetric square root  X = V_c sqrt(D_c) V_c'. */
    matlab_mat *Vc = matlab_eig_V(Wc);
    matlab_mat *Dc = matlab_eig_D(Wc);
    matlab_mat *Vct = matlab_transpose(Vc);
    /* sqrt(D_c) acting on each column of V_c — easiest via element-wise
     * scaling: VD[i, j] = V_c[i, j] * sqrt(D_c[j, j]). */
    matlab_mat *VD = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double d = Dc->data[j * n + j];
            double s = d > 0.0 ? sqrt(d) : 0.0;
            VD->data[i * n + j] = Vc->data[i * n + j] * s;
        }
    }
    matlab_mat *X = matlab_matmul_mm(VD, Vct);  /* X = V_c sqrt(D_c) V_c' */

    /* M = X Wo X (X symmetric). */
    matlab_mat *XWo = matlab_matmul_mm(X, Wo);
    matlab_mat *M   = matlab_matmul_mm(XWo, X);

    /* Symmetric eig: M = U S² U'. */
    matlab_mat *U  = matlab_eig_V(M);
    matlab_mat *S2 = matlab_eig_D(M);
    /* Extract sigma_j = sqrt(S²_jj). Note eig_D returns ascending — we
     * want descending (largest HSV first), so build a permutation that
     * reverses the column order. */
    std::vector<double> sigma(n);
    for (int64_t j = 0; j < n; ++j) {
        double s = S2->data[j * n + j];
        sigma[j] = s > 0.0 ? sqrt(s) : 0.0;
    }
    /* Reorder U columns and sigma entries to descending sigma. */
    std::vector<int64_t> perm(n);
    for (int64_t j = 0; j < n; ++j) perm[j] = n - 1 - j;
    matlab_mat *Uord = mat_alloc(n, n);
    std::vector<double> sigma_ord(n);
    for (int64_t j = 0; j < n; ++j) {
        sigma_ord[j] = sigma[perm[j]];
        for (int64_t i = 0; i < n; ++i)
            Uord->data[i * n + j] = U->data[i * n + perm[j]];
    }

    /* T = X * U_ord * diag(sigma_ord^{-1/2}). Apply the column scale
     * after the matmul. */
    matlab_mat *XU = matlab_matmul_mm(X, Uord);
    matlab_mat *T  = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = sigma_ord[j];
            double sc = s > 0.0 ? 1.0 / sqrt(s) : 0.0;
            T->data[i * n + j] = XU->data[i * n + j] * sc;
        }
    }
    return T;
}

/*-------------------------------------------------------------------------
 * Balanced truncation — k-state model reduction.
 *
 *   balred_A(A, B, C, k)  returns the k×k upper-left block of the
 *   balanced realization. balred_B / balred_C return the corresponding
 *   first-k rows / columns of the balanced B / C.
 *
 *   Algorithm: build the full balanced (A_b, B_b, C_b) via balreal_T,
 *   then keep only the first k rows / columns / both. The dropped
 *   states correspond to the smallest Hankel singular values; the
 *   H∞ error bound is  ||G − G_k||_∞ ≤ 2 · sum(HSV[k+1:n]).
 *
 *   Each entry rebuilds the full balanced realization internally —
 *   for typical CST plants (n = 2..10) this is fine; if the redundancy
 *   matters in practice we can stash the balanced realization in a
 *   thread-local cache later. Tier-4 of the CST roadmap.
 *
 *   The MATLAB-faithful  [Ar, Br, Cr] = balred(A, B, C, k)  3-return
 *   shape is a follow-on (3-return splitter mirroring the c2d / bode
 *   precedent).
 *-------------------------------------------------------------------------*/
static matlab_mat *balred_full_(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                                matlab_mat **out_Bb, matlab_mat **out_Cb) {
    matlab_mat *T = matlab_balreal_T(A, B, C);
    if (!T || T->rows == 0) return NULL;
    matlab_mat *Ti = matlab_inv(T);
    if (!Ti || Ti->rows == 0) return NULL;
    matlab_mat *TiA = matlab_matmul_mm(Ti, A);
    matlab_mat *Ab  = matlab_matmul_mm(TiA, T);
    if (out_Bb) *out_Bb = matlab_matmul_mm(Ti, B);
    if (out_Cb) *out_Cb = matlab_matmul_mm(C, T);
    return Ab;
}

matlab_mat *matlab_balred_A(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                            double kd) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t k = (int64_t)kd;
    if (k <= 0 || k > n) return mat_alloc(0, 0);
    matlab_mat *Ab = balred_full_(A, B, C, NULL, NULL);
    if (!Ab) return mat_alloc(0, 0);
    matlab_mat *Ar = mat_alloc(k, k);
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = 0; j < k; ++j)
            Ar->data[i * k + j] = Ab->data[i * n + j];
    return Ar;
}

matlab_mat *matlab_balred_B(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                            double kd) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t k = (int64_t)kd;
    if (k <= 0 || k > n) return mat_alloc(0, 0);
    matlab_mat *Bb = NULL;
    (void)balred_full_(A, B, C, &Bb, NULL);
    if (!Bb) return mat_alloc(0, 0);
    int64_t m = Bb->cols;
    matlab_mat *Br = mat_alloc(k, m);
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = 0; j < m; ++j)
            Br->data[i * m + j] = Bb->data[i * m + j];
    return Br;
}

matlab_mat *matlab_balred_C(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                            double kd) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t k = (int64_t)kd;
    if (k <= 0 || k > n) return mat_alloc(0, 0);
    matlab_mat *Cb = NULL;
    (void)balred_full_(A, B, C, NULL, &Cb);
    if (!Cb) return mat_alloc(0, 0);
    int64_t p = Cb->rows;
    matlab_mat *Cr = mat_alloc(p, k);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < k; ++j)
            Cr->data[i * k + j] = Cb->data[i * n + j];
    return Cr;
}

/*-------------------------------------------------------------------------
 * Padé approximation of the time-delay element e^{-τs}.
 *
 *   [num, den] = pade(τ, n)
 *
 * Symmetric (diagonal) [n/n] Padé:
 *   R_{n,n}(-τs) = P_n(-τs) / P_n(τs)
 *   P_n(x) = sum_{j=0..n}  (2n−j)!·n!  /  ((2n)!·j!·(n−j)!)  · x^j
 *
 * Coefficient recurrence on the magnitude c_j = (2n−j)!·n! ·
 * τ^j / ((2n)!·j!·(n−j)!):
 *   c_0 = 1
 *   c_j = c_{j-1} · τ · (n−j+1) / (j · (2n−j+1))
 *
 * The numerator is sum_{j=0..n} (−1)^j · c_j · s^j (alternating
 * signs from the −τs substitution); the denominator is
 * sum_{j=0..n} c_j · s^j. Both returned as 1×(n+1) row vectors in
 * descending power, matching MATLAB's `[num, den] = pade(τ, n)`
 * convention.
 *
 * Tier 4.3 of the CST roadmap.
 *-------------------------------------------------------------------------*/
static void pade_coeffs_(double tau, int n, std::vector<double> &c) {
    c.assign(n + 1, 0.0);
    c[0] = 1.0;
    for (int j = 1; j <= n; ++j) {
        c[j] = c[j - 1] * tau * (n - j + 1) /
               ((double)j * (2 * n - j + 1));
    }
}

matlab_mat *matlab_pade_num(double tau, double n_d) {
    int n = (int)n_d;
    if (n < 0) n = 0;
    std::vector<double> c;
    pade_coeffs_(tau, n, c);
    matlab_mat *num = mat_alloc(1, n + 1);
    /* Descending power: position i carries the s^{n-i} coefficient.
     * Numerator picks up (-1)^j alternating signs from -τs. */
    for (int i = 0; i <= n; ++i) {
        int j = n - i;
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        num->data[i] = sign * c[j];
    }
    return num;
}

matlab_mat *matlab_pade_den(double tau, double n_d) {
    int n = (int)n_d;
    if (n < 0) n = 0;
    std::vector<double> c;
    pade_coeffs_(tau, n, c);
    matlab_mat *den = mat_alloc(1, n + 1);
    for (int i = 0; i <= n; ++i) {
        int j = n - i;
        den->data[i] = c[j];
    }
    return den;
}

/* Forward decls — matlab_roots / matlab_poly live in runtime_complex.cpp. */
extern "C" matlab_mat_c *matlab_roots(matlab_mat *p);
extern "C" matlab_mat   *matlab_poly(void *r);

/*-------------------------------------------------------------------------
 * Minimal realisation for the transfer-function form.
 *
 *   [num_r, den_r] = minreal(num, den, tol)
 *
 * Cancels matching pole-zero pairs within `tol` (Euclidean distance
 * in the complex plane). Each (z_i, p_j) pair with |z_i − p_j| ≤
 * tol drops both roots. The reduced polynomial is rebuilt via
 * `matlab_poly` on the surviving roots and rescaled by the original
 * leading coefficient (so the steady-state gain is preserved when
 * no DC pole/zero gets cancelled).
 *
 * The ss-form `minreal(sys)` would go via the controllability /
 * observability staircase decomposition (ctrbf / obsvf, both
 * follow-ons); this tf-form covers the practical pole-zero
 * cancellation surface for scalar transfer functions.
 *
 * Tier 4.1 of the CST roadmap §5.1.
 *-------------------------------------------------------------------------*/
static void minreal_cancel_roots_(matlab_mat_c *zeros, matlab_mat_c *poles,
                                    double tol,
                                    std::vector<double> &zre_out,
                                    std::vector<double> &zim_out,
                                    std::vector<double> &pre_out,
                                    std::vector<double> &pim_out) {
    int64_t nz = zeros ? zeros->rows * zeros->cols : 0;
    int64_t np = poles ? poles->rows * poles->cols : 0;
    std::vector<bool> z_alive(nz, true), p_alive(np, true);
    for (int64_t i = 0; i < nz; ++i) {
        if (!z_alive[i]) continue;
        for (int64_t j = 0; j < np; ++j) {
            if (!p_alive[j]) continue;
            double dr = zeros->re[i] - poles->re[j];
            double di = zeros->im[i] - poles->im[j];
            if (dr * dr + di * di <= tol * tol) {
                z_alive[i] = false;
                p_alive[j] = false;
                break;
            }
        }
    }
    zre_out.clear(); zim_out.clear();
    pre_out.clear(); pim_out.clear();
    for (int64_t i = 0; i < nz; ++i)
        if (z_alive[i]) {
            zre_out.push_back(zeros->re[i]);
            zim_out.push_back(zeros->im[i]);
        }
    for (int64_t j = 0; j < np; ++j)
        if (p_alive[j]) {
            pre_out.push_back(poles->re[j]);
            pim_out.push_back(poles->im[j]);
        }
}

static matlab_mat *polynomial_from_roots_(const std::vector<double> &re,
                                            const std::vector<double> &im,
                                            double lead) {
    int64_t n = (int64_t)re.size();
    matlab_mat_c *R = mat_c_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) { R->re[i] = re[i]; R->im[i] = im[i]; }
    matlab_mat *p = matlab_poly((void *)R);
    if (!p || p->rows * p->cols == 0) return p;
    /* Scale by lead so the result is `lead · monic(s)`. */
    for (int64_t i = 0; i < p->rows * p->cols; ++i) p->data[i] *= lead;
    return p;
}

matlab_mat *matlab_minreal_tf_num(matlab_mat *num, matlab_mat *den,
                                   double tol) {
    if (!num || !den) return mat_alloc(0, 0);
    matlab_mat_c *zeros = matlab_roots(num);
    matlab_mat_c *poles = matlab_roots(den);
    std::vector<double> zre, zim, pre, pim;
    minreal_cancel_roots_(zeros, poles, tol, zre, zim, pre, pim);
    double lead = num->data[0];
    return polynomial_from_roots_(zre, zim, lead);
}

matlab_mat *matlab_minreal_tf_den(matlab_mat *num, matlab_mat *den,
                                   double tol) {
    if (!num || !den) return mat_alloc(0, 0);
    matlab_mat_c *zeros = matlab_roots(num);
    matlab_mat_c *poles = matlab_roots(den);
    std::vector<double> zre, zim, pre, pim;
    minreal_cancel_roots_(zeros, poles, tol, zre, zim, pre, pim);
    double lead = den->data[0];
    return polynomial_from_roots_(pre, pim, lead);
}

/*-------------------------------------------------------------------------
 * Structural minimal realisation `sminreal(A, B, C)`.
 *
 * Drops states that are not both structurally reachable from at
 * least one input (B column) and structurally observable from at
 * least one output (C row). Pure boolean-graph analysis — no
 * numerics, no cancellation tolerance — so it's faster and more
 * predictable than the ctrbf/obsvf staircase minreal. Returns
 * three matrices on the surviving state indices.
 *
 * Algorithm:
 *   1. Reachable set R from B: start with {i : B[i, *] ≠ 0}, then
 *      iterate: i ∈ R ∧ A[j, i] ≠ 0 ⇒ j ∈ R until no change.
 *   2. Observable set O from C: start with {i : C[*, i] ≠ 0}, then
 *      iterate: i ∈ O ∧ A[i, j] ≠ 0 ⇒ j ∈ O until no change.
 *   3. Keep set K = R ∩ O. Build A_s / B_s / C_s on K.
 *
 * Tier 4.1 of CST roadmap §5.1.
 *-------------------------------------------------------------------------*/
static void sminreal_keep_(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                             std::vector<int64_t> &keep_out) {
    int64_t n = A->rows;
    int64_t m = B->cols;
    int64_t p = C->rows;
    std::vector<bool> reach(n, false), obs(n, false);
    /* Reachable from B. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < m; ++k)
            if (B->data[i * m + k] != 0.0) { reach[i] = true; break; }
    bool changed = true;
    while (changed) {
        changed = false;
        for (int64_t j = 0; j < n; ++j) {
            if (reach[j]) continue;
            for (int64_t i = 0; i < n; ++i) {
                if (reach[i] && A->data[j * n + i] != 0.0) {
                    reach[j] = true; changed = true; break;
                }
            }
        }
    }
    /* Observable from C. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < p; ++k)
            if (C->data[k * n + i] != 0.0) { obs[i] = true; break; }
    changed = true;
    while (changed) {
        changed = false;
        for (int64_t i = 0; i < n; ++i) {
            if (obs[i]) continue;
            for (int64_t j = 0; j < n; ++j) {
                if (obs[j] && A->data[i * n + j] != 0.0) {
                    obs[i] = true; changed = true; break;
                }
            }
        }
    }
    keep_out.clear();
    for (int64_t i = 0; i < n; ++i)
        if (reach[i] && obs[i]) keep_out.push_back(i);
}

matlab_mat *matlab_sminreal_A(matlab_mat *A, matlab_mat *B, matlab_mat *C) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (A->cols != n) return mat_alloc(0, 0);
    std::vector<int64_t> keep;
    sminreal_keep_(A, B, C, keep);
    int64_t nk = keep.size();
    matlab_mat *As = mat_alloc(nk, nk);
    for (int64_t i = 0; i < nk; ++i)
        for (int64_t j = 0; j < nk; ++j)
            As->data[i * nk + j] = A->data[keep[i] * n + keep[j]];
    return As;
}

matlab_mat *matlab_sminreal_B(matlab_mat *A, matlab_mat *B, matlab_mat *C) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t m = B->cols;
    std::vector<int64_t> keep;
    sminreal_keep_(A, B, C, keep);
    int64_t nk = keep.size();
    matlab_mat *Bs = mat_alloc(nk, m);
    for (int64_t i = 0; i < nk; ++i)
        for (int64_t j = 0; j < m; ++j)
            Bs->data[i * m + j] = B->data[keep[i] * m + j];
    return Bs;
}

matlab_mat *matlab_sminreal_C(matlab_mat *A, matlab_mat *B, matlab_mat *C) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t p = C->rows;
    std::vector<int64_t> keep;
    sminreal_keep_(A, B, C, keep);
    int64_t nk = keep.size();
    matlab_mat *Cs = mat_alloc(p, nk);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < nk; ++j)
            Cs->data[i * nk + j] = C->data[i * n + keep[j]];
    return Cs;
}

/*-------------------------------------------------------------------------
 * Modal residualisation / truncation `modred(A, B, C, elim, method)`.
 *
 * Drops a subset of states (specified by the `elim` vector of
 * 1-indexed state indices). Two methods:
 *   - Truncate: just drop those rows/columns of A, B, C.
 *   - MatchDC: apply the Schur-complement formula so the reduced
 *     system has the same DC gain as the full one.
 *       Reorder x = [x_keep; x_elim].
 *       A = [A11 A12; A21 A22],  B = [B1; B2],  C = [C1 C2].
 *       A_r = A11 − A12 · A22⁻¹ · A21
 *       B_r = B1  − A12 · A22⁻¹ · B2
 *       C_r = C1  − C2  · A22⁻¹ · A21
 *
 * Method is encoded as f64: 0 = Truncate, 1 = MatchDC. The
 * Lowering.cpp dispatch reads the user's `'Truncate'` /
 * `'MatchDC'` string literal and picks the right method-id at
 * call site.
 *
 * Tier 4.1 of CST roadmap §5.1.
 *-------------------------------------------------------------------------*/
static void modred_partition_(matlab_mat *elim, int64_t n,
                                std::vector<int64_t> &keep_idx,
                                std::vector<int64_t> &elim_idx) {
    std::vector<bool> drop(n, false);
    int64_t nel = elim ? elim->rows * elim->cols : 0;
    for (int64_t i = 0; i < nel; ++i) {
        int64_t idx = (int64_t)elim->data[i] - 1;  /* MATLAB 1-indexed */
        if (idx >= 0 && idx < n) drop[idx] = true;
    }
    keep_idx.clear(); elim_idx.clear();
    for (int64_t i = 0; i < n; ++i) {
        if (drop[i]) elim_idx.push_back(i);
        else         keep_idx.push_back(i);
    }
}

static matlab_mat *submat_(matlab_mat *M, const std::vector<int64_t> &rows,
                             const std::vector<int64_t> &cols) {
    int64_t mr = rows.size(), mc = cols.size();
    int64_t orig_cols = M->cols;
    matlab_mat *S = mat_alloc(mr, mc);
    for (int64_t i = 0; i < mr; ++i)
        for (int64_t j = 0; j < mc; ++j)
            S->data[i * mc + j] = M->data[rows[i] * orig_cols + cols[j]];
    return S;
}

matlab_mat *matlab_modred_A(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                             matlab_mat *elim, double method_id) {
    if (!A || !B || !C || !elim) return mat_alloc(0, 0);
    int64_t n = A->rows;
    std::vector<int64_t> keep, drop;
    modred_partition_(elim, n, keep, drop);
    int64_t nk = keep.size();
    matlab_mat *A11 = submat_(A, keep, keep);
    if (method_id == 0.0 || drop.empty()) return A11;
    matlab_mat *A12 = submat_(A, keep, drop);
    matlab_mat *A21 = submat_(A, drop, keep);
    matlab_mat *A22 = submat_(A, drop, drop);
    matlab_mat *A22inv = matlab_inv(A22);
    if (!A22inv || A22inv->rows == 0) return A11;  /* fallback */
    matlab_mat *Tmp1 = matlab_matmul_mm(A12, A22inv);
    matlab_mat *Tmp2 = matlab_matmul_mm(Tmp1, A21);
    matlab_mat *Neg  = matlab_neg_m(Tmp2);
    (void)nk;
    return matlab_add_mm(A11, Neg);
}

matlab_mat *matlab_modred_B(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                             matlab_mat *elim, double method_id) {
    if (!A || !B || !C || !elim) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t m = B->cols;
    std::vector<int64_t> keep, drop;
    modred_partition_(elim, n, keep, drop);
    std::vector<int64_t> all_cols(m);
    for (int64_t j = 0; j < m; ++j) all_cols[j] = j;
    matlab_mat *B1 = submat_(B, keep, all_cols);
    if (method_id == 0.0 || drop.empty()) return B1;
    matlab_mat *A12 = submat_(A, keep, drop);
    matlab_mat *A22 = submat_(A, drop, drop);
    matlab_mat *B2  = submat_(B, drop, all_cols);
    matlab_mat *A22inv = matlab_inv(A22);
    if (!A22inv || A22inv->rows == 0) return B1;
    matlab_mat *Tmp1 = matlab_matmul_mm(A12, A22inv);
    matlab_mat *Tmp2 = matlab_matmul_mm(Tmp1, B2);
    matlab_mat *Neg  = matlab_neg_m(Tmp2);
    return matlab_add_mm(B1, Neg);
}

matlab_mat *matlab_modred_C(matlab_mat *A, matlab_mat *B, matlab_mat *C,
                             matlab_mat *elim, double method_id) {
    if (!A || !B || !C || !elim) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t p = C->rows;
    std::vector<int64_t> keep, drop;
    modred_partition_(elim, n, keep, drop);
    std::vector<int64_t> all_rows(p);
    for (int64_t i = 0; i < p; ++i) all_rows[i] = i;
    matlab_mat *C1 = submat_(C, all_rows, keep);
    if (method_id == 0.0 || drop.empty()) return C1;
    matlab_mat *A21 = submat_(A, drop, keep);
    matlab_mat *A22 = submat_(A, drop, drop);
    matlab_mat *C2  = submat_(C, all_rows, drop);
    matlab_mat *A22inv = matlab_inv(A22);
    if (!A22inv || A22inv->rows == 0) return C1;
    matlab_mat *Tmp1 = matlab_matmul_mm(C2, A22inv);
    matlab_mat *Tmp2 = matlab_matmul_mm(Tmp1, A21);
    matlab_mat *Neg  = matlab_neg_m(Tmp2);
    return matlab_add_mm(C1, Neg);
}

/*-------------------------------------------------------------------------
 * Thiran fractional-delay all-pass FIR — `b = thiran(D, n)`.
 *
 * Standard Thiran formula for a length-(n+1) all-pass approximation
 * of a fractional delay D (in samples):
 *   a_k = (−1)^k · C(n, k) · ∏_{i=0..n}  (D − n + i) / (D − n + k + i)
 * The denominator polynomial is sum a_k z^{−k}; the numerator is the
 * reversed (mirror-symmetric) coefficient sequence — all-pass shape.
 *
 * Returns the (n+1)-element coefficient row vector. The numerator
 * `b` reverses the denominator. Backs the matrix-arg
 * `thiran(D, n)` builtin. Tier 4.3 of CST roadmap §5.3.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_thiran_a(double D, double n_d) {
    int n = (int)n_d;
    if (n < 0) n = 0;
    matlab_mat *a = mat_alloc(1, n + 1);
    /* Binomial coefficient via Pascal's recurrence (small n only). */
    std::vector<double> binom(n + 1, 0.0);
    binom[0] = 1.0;
    for (int k = 1; k <= n; ++k)
        binom[k] = binom[k - 1] * (n - k + 1) / (double)k;
    for (int k = 0; k <= n; ++k) {
        double prod = 1.0;
        for (int i = 0; i <= n; ++i) {
            double num = D - (double)n + (double)i;
            double den = D - (double)n + (double)k + (double)i;
            if (den != 0.0) prod *= num / den;
        }
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        a->data[k] = sign * binom[k] * prod;
    }
    return a;
}

matlab_mat *matlab_thiran_b(double D, double n_d) {
    int n = (int)n_d;
    if (n < 0) n = 0;
    matlab_mat *a = matlab_thiran_a(D, n_d);
    /* All-pass: b is the mirror image of a. */
    matlab_mat *b = mat_alloc(1, n + 1);
    for (int k = 0; k <= n; ++k) b->data[k] = a->data[n - k];
    return b;
}

/* Forward decl: matlab_isstable is defined just below. */
double matlab_isstable(matlab_mat *A);

/* Forward decls (real_c / imag_c live in runtime_complex.cpp). */
matlab_mat *matlab_real_c(void *A);
matlab_mat *matlab_imag_c(void *A);

/*-------------------------------------------------------------------------
 * Continuous-to-discrete Tustin (bilinear) discretisation.
 *
 *   [Ad, Bd] = c2d_tustin(A, B, Ts)
 *
 * Substitutes s = (2/Ts)·(z − 1)/(z + 1) into the continuous-time
 * state-space, no expm needed:
 *      α  = Ts/2
 *      M  = I − α A
 *      Ad = M⁻¹ · (I + α A)
 *      Bd = Ts · M⁻¹ · B
 *
 * Shipped as two single-return entries (matlab_c2d_tustin_Ad /
 * matlab_c2d_tustin_Bd) mirroring the eig_V / eig_D and existing
 * matlab_c2d_Ad / matlab_c2d_Bd precedent. The MATLAB-faithful
 * `[Ad, Bd] = c2d(A, B, Ts, 'tustin')` form is a follow-on (string-arg
 * dispatch). Tier-2.2 of the CST roadmap.
 *
 * Note: this v1 returns just (Ad, Bd) — same shape as the ZOH c2d
 * we already ship. The exact transfer-function preservation
 * `H_d(z) = H_c((2/Ts)(z−1)/(z+1))` holds without any C/D
 * adjustment when the user keeps the same C/D matrices.
 *-------------------------------------------------------------------------*/
static matlab_mat *c2d_tustin_M_inv_(matlab_mat *A, double Ts) {
    int64_t n = A->rows;
    double alpha = Ts / 2.0;
    matlab_mat *M = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double v = -alpha * A->data[i * n + j];
            if (i == j) v += 1.0;
            M->data[i * n + j] = v;
        }
    return matlab_inv(M);
}

matlab_mat *matlab_c2d_tustin_Ad(matlab_mat *A, matlab_mat *B, double Ts) {
    if (!A) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0 || A->cols != n) return mat_alloc(0, 0);
    (void)B;   /* Ad doesn't actually need B; arg kept for API symmetry. */
    double alpha = Ts / 2.0;
    matlab_mat *Minv = c2d_tustin_M_inv_(A, Ts);
    if (!Minv || Minv->rows == 0) return mat_alloc(0, 0);
    /* P = I + α A. */
    matlab_mat *P = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double v = alpha * A->data[i * n + j];
            if (i == j) v += 1.0;
            P->data[i * n + j] = v;
        }
    return matlab_matmul_mm(Minv, P);
}

matlab_mat *matlab_c2d_tustin_Bd(matlab_mat *A, matlab_mat *B, double Ts) {
    if (!A || !B) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0 || A->cols != n || B->rows != n) return mat_alloc(0, 0);
    matlab_mat *Minv = c2d_tustin_M_inv_(A, Ts);
    if (!Minv || Minv->rows == 0) return mat_alloc(0, 0);
    /* Bd = Ts · M⁻¹ · B. */
    matlab_mat *MinvB = matlab_matmul_mm(Minv, B);
    int64_t m = B->cols;
    matlab_mat *Bd = mat_alloc(n, m);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            Bd->data[i * m + j] = Ts * MinvB->data[i * m + j];
    return Bd;
}

/*-------------------------------------------------------------------------
 * Closed-loop assembly for negative feedback (strictly proper).
 *
 *   [Acl, Bcl, Ccl] = feedback_ss(A1, B1, C1, A2, B2, C2)
 *
 * Builds the closed-loop state-space realisation for negative feedback
 * `T = sys1 / (1 + sys2·sys1)`, both plants assumed strictly proper
 * (D1 = D2 = 0). Block layout:
 *
 *   Acl = [A1,    -B1·C2;
 *          B2·C1,  A2     ]
 *   Bcl = [B1; 0]
 *   Ccl = [C1, 0]
 *
 * For the static-gain feedback case (sys2 has zero states), users get
 * better economy from `Acl = A1 - B1·K·C1` directly.
 *
 * Tier-2 of CST roadmap (System interconnection).  The MATLAB-faithful
 * `feedback(sys1, sys2)` model-object form awaits §3.1.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_feedback_ss_A(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                                 matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    if (!A1 || !B1 || !C1 || !A2 || !B2 || !C2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    if (A1->cols != n1 || A2->cols != n2) return mat_alloc(0, 0);
    int64_t n = n1 + n2;
    /* B1 · C2 → n1 × n2. */
    matlab_mat *B1C2 = matlab_matmul_mm(B1, C2);
    /* B2 · C1 → n2 × n1. */
    matlab_mat *B2C1 = matlab_matmul_mm(B2, C1);
    matlab_mat *Acl = mat_alloc(n, n);
    /* Top-left: A1 (n1 × n1). */
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Acl->data[i * n + j] = A1->data[i * n1 + j];
    /* Top-right: -B1 · C2 (n1 × n2). */
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < n2; ++j)
            Acl->data[i * n + (n1 + j)] = -B1C2->data[i * n2 + j];
    /* Bottom-left: B2 · C1 (n2 × n1). */
    for (int64_t i = 0; i < n2; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Acl->data[(n1 + i) * n + j] = B2C1->data[i * n1 + j];
    /* Bottom-right: A2 (n2 × n2). */
    for (int64_t i = 0; i < n2; ++i)
        for (int64_t j = 0; j < n2; ++j)
            Acl->data[(n1 + i) * n + (n1 + j)] = A2->data[i * n2 + j];
    return Acl;
}

matlab_mat *matlab_feedback_ss_B(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                                 matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)C1; (void)B2; (void)C2;
    if (!A1 || !B1 || !A2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t m = B1->cols;
    matlab_mat *Bcl = mat_alloc(n1 + n2, m);
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < m; ++j)
            Bcl->data[i * m + j] = B1->data[i * m + j];
    /* Bottom block already zero from mat_alloc. */
    return Bcl;
}

matlab_mat *matlab_feedback_ss_C(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                                 matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)B1; (void)B2; (void)C2;
    if (!A1 || !C1 || !A2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t p = C1->rows;
    matlab_mat *Ccl = mat_alloc(p, n1 + n2);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Ccl->data[i * (n1 + n2) + j] = C1->data[i * n1 + j];
    return Ccl;
}

/*-------------------------------------------------------------------------
 * Block-diagonal append — sys = blkdiag(sys1, sys2). MIMO assembly
 * with disjoint input/output channels.
 *   Acl = blkdiag(A1, A2)
 *   Bcl = blkdiag(B1, B2)
 *   Ccl = blkdiag(C1, C2)
 * (Same A as parallel; B and C are block-diagonal instead of stacked.)
 *-------------------------------------------------------------------------*/
/* Forward decl: parallel_ss_A is defined below. */
matlab_mat *matlab_parallel_ss_A(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                                 matlab_mat *A2, matlab_mat *B2, matlab_mat *C2);

matlab_mat *matlab_append_ss_A(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                               matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    /* Same as parallel A — block diagonal of the state matrices. */
    return matlab_parallel_ss_A(A1, B1, C1, A2, B2, C2);
}

matlab_mat *matlab_append_ss_B(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                               matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)C1; (void)C2;
    if (!A1 || !B1 || !A2 || !B2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t m1 = B1->cols, m2 = B2->cols;
    matlab_mat *Bcl = mat_alloc(n1 + n2, m1 + m2);
    int64_t M = m1 + m2;
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < m1; ++j)
            Bcl->data[i * M + j] = B1->data[i * m1 + j];
    for (int64_t i = 0; i < n2; ++i)
        for (int64_t j = 0; j < m2; ++j)
            Bcl->data[(n1 + i) * M + (m1 + j)] = B2->data[i * m2 + j];
    return Bcl;
}

matlab_mat *matlab_append_ss_C(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                               matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)B1; (void)B2;
    if (!A1 || !C1 || !A2 || !C2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t p1 = C1->rows, p2 = C2->rows;
    matlab_mat *Ccl = mat_alloc(p1 + p2, n1 + n2);
    int64_t N = n1 + n2;
    for (int64_t i = 0; i < p1; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Ccl->data[i * N + j] = C1->data[i * n1 + j];
    for (int64_t i = 0; i < p2; ++i)
        for (int64_t j = 0; j < n2; ++j)
            Ccl->data[(p1 + i) * N + (n1 + j)] = C2->data[i * n2 + j];
    return Ccl;
}

/*-------------------------------------------------------------------------
 * Series cascade — sys = sys2 * sys1, strictly proper.
 *   Acl = [A1,    0;
 *          B2·C1, A2]
 *   Bcl = [B1; 0]
 *   Ccl = [0, C2]
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_series_ss_A(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                               matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)B1; (void)C2;
    if (!A1 || !C1 || !A2 || !B2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    if (A1->cols != n1 || A2->cols != n2) return mat_alloc(0, 0);
    int64_t n = n1 + n2;
    matlab_mat *B2C1 = matlab_matmul_mm(B2, C1);
    matlab_mat *Acl = mat_alloc(n, n);
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Acl->data[i * n + j] = A1->data[i * n1 + j];
    for (int64_t i = 0; i < n2; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Acl->data[(n1 + i) * n + j] = B2C1->data[i * n1 + j];
    for (int64_t i = 0; i < n2; ++i)
        for (int64_t j = 0; j < n2; ++j)
            Acl->data[(n1 + i) * n + (n1 + j)] = A2->data[i * n2 + j];
    return Acl;
}

matlab_mat *matlab_series_ss_B(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                               matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)C1; (void)B2; (void)C2;
    if (!A1 || !B1 || !A2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t m = B1->cols;
    matlab_mat *Bcl = mat_alloc(n1 + n2, m);
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < m; ++j)
            Bcl->data[i * m + j] = B1->data[i * m + j];
    return Bcl;
}

matlab_mat *matlab_series_ss_C(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                               matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)B1; (void)C1; (void)B2;
    if (!A1 || !A2 || !C2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t p = C2->rows;
    matlab_mat *Ccl = mat_alloc(p, n1 + n2);
    /* Left block (n1 cols) zero by mat_alloc. Right block: C2 (p × n2). */
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < n2; ++j)
            Ccl->data[i * (n1 + n2) + (n1 + j)] = C2->data[i * n2 + j];
    return Ccl;
}

/*-------------------------------------------------------------------------
 * Parallel sum — sys = sys1 + sys2, strictly proper.
 *   Acl = blkdiag(A1, A2)
 *   Bcl = [B1; B2]
 *   Ccl = [C1, C2]
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_parallel_ss_A(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                                 matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)B1; (void)C1; (void)B2; (void)C2;
    if (!A1 || !A2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    if (A1->cols != n1 || A2->cols != n2) return mat_alloc(0, 0);
    int64_t n = n1 + n2;
    matlab_mat *Acl = mat_alloc(n, n);
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Acl->data[i * n + j] = A1->data[i * n1 + j];
    for (int64_t i = 0; i < n2; ++i)
        for (int64_t j = 0; j < n2; ++j)
            Acl->data[(n1 + i) * n + (n1 + j)] = A2->data[i * n2 + j];
    return Acl;
}

matlab_mat *matlab_parallel_ss_B(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                                 matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)C1; (void)C2;
    if (!A1 || !B1 || !A2 || !B2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t m = B1->cols;
    if (B2->cols != m) return mat_alloc(0, 0);
    matlab_mat *Bcl = mat_alloc(n1 + n2, m);
    for (int64_t i = 0; i < n1; ++i)
        for (int64_t j = 0; j < m; ++j)
            Bcl->data[i * m + j] = B1->data[i * m + j];
    for (int64_t i = 0; i < n2; ++i)
        for (int64_t j = 0; j < m; ++j)
            Bcl->data[(n1 + i) * m + j] = B2->data[i * m + j];
    return Bcl;
}

matlab_mat *matlab_parallel_ss_C(matlab_mat *A1, matlab_mat *B1, matlab_mat *C1,
                                 matlab_mat *A2, matlab_mat *B2, matlab_mat *C2) {
    (void)B1; (void)B2;
    if (!A1 || !C1 || !A2 || !C2) return mat_alloc(0, 0);
    int64_t n1 = A1->rows, n2 = A2->rows;
    int64_t p = C1->rows;
    if (C2->rows != p) return mat_alloc(0, 0);
    matlab_mat *Ccl = mat_alloc(p, n1 + n2);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < n1; ++j)
            Ccl->data[i * (n1 + n2) + j] = C1->data[i * n1 + j];
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < n2; ++j)
            Ccl->data[i * (n1 + n2) + (n1 + j)] = C2->data[i * n2 + j];
    return Ccl;
}

/*-------------------------------------------------------------------------
 * Peak gain over a frequency sweep (rough H∞ approximation).
 *
 *   getPeakGain_ss(A, B, C, D) = max_{w ∈ grid} |H(jw)|
 * where the grid is 1e-3 → 1e6 rad/s, 200 log-spaced points. SISO only.
 * Captures resonant peaks for typical 2nd-order plants; misses sharp
 * resonances between grid points (within ~5% for ζ ≥ 0.05). The exact
 * H∞ norm requires Boyd-Balakrishnan-Kabamba γ-bisection on
 * Hamiltonian eigenvalues (separate slice).
 *-------------------------------------------------------------------------*/
double matlab_getPeakGain_ss(matlab_mat *A, matlab_mat *B,
                             matlab_mat *C, matlab_mat *D) {
    if (!A || !B || !C || !D) return INFINITY;
    int64_t n = A->rows;
    if (n == 0 || A->cols != n || B->rows != n || C->cols != n)
        return 0.0;
    const int Npts = 200;
    const double log_lo = -3, log_hi = 6;
    double peak = 0.0;
    /* Include w = 0 if A is invertible (DC gain). */
    matlab_mat *Ainv = matlab_inv(A);
    if (Ainv && Ainv->rows > 0) {
        matlab_mat *AinvB = matlab_matmul_mm(Ainv, B);
        matlab_mat *CAinvB = matlab_matmul_mm(C, AinvB);
        double dc = D->data[0] - CAinvB->data[0];
        double absdc = dc < 0 ? -dc : dc;
        if (absdc > peak) peak = absdc;
    }
    for (int i = 0; i < Npts; ++i) {
        double w = pow(10.0, log_lo + (double)i / (Npts - 1) * (log_hi - log_lo));
        double Hr = 0, Hi = 0;
        if (bode_ss_at_freq_(A, B, C, D, w, &Hr, &Hi) != 0) continue;
        double mag = sqrt(Hr * Hr + Hi * Hi);
        if (mag > peak) peak = mag;
    }
    return peak;
}

/*-------------------------------------------------------------------------
 * SISO −3 dB bandwidth.
 *
 *   bandwidth_ss(A, B, C, D) returns the lowest frequency w where
 *   |H(jw)| crosses |H(j0)| / sqrt(2) from above. Scans a log-spaced
 *   grid 1e-3 → 1e6 rad/s, linearly interpolates between adjacent
 *   grid points for accuracy. Returns +Inf if no crossover (e.g.
 *   all-pass or unstable plants where DC gain isn't bounded).
 *
 * Forward decls for the helpers used. */
static int bode_ss_at_freq_(matlab_mat *A, matlab_mat *B,
                            matlab_mat *C, matlab_mat *D,
                            double w, double *Hr_out, double *Hi_out);

double matlab_bandwidth_ss(matlab_mat *A, matlab_mat *B,
                           matlab_mat *C, matlab_mat *D) {
    if (!A || !B || !C || !D) return INFINITY;
    int64_t n = A->rows;
    if (n == 0 || A->cols != n || B->rows != n || C->cols != n)
        return INFINITY;
    /* DC gain magnitude. Use the matrix-side dcgain_ss formula (D − CA⁻¹B). */
    matlab_mat *Ainv = matlab_inv(A);
    if (!Ainv || Ainv->rows == 0) return INFINITY;
    matlab_mat *AinvB = matlab_matmul_mm(Ainv, B);
    matlab_mat *CAinvB = matlab_matmul_mm(C, AinvB);
    /* SISO assumption: take (1, 1) entry. */
    double G0 = D->data[0] - CAinvB->data[0];
    double absG0 = G0 < 0 ? -G0 : G0;
    if (absG0 <= 0.0) return INFINITY;   /* zero DC gain → bandwidth undefined */
    double target = absG0 / sqrt(2.0);
    /* Log-spaced grid from 1e-3 to 1e6 rad/s, 200 points (~10 per decade). */
    const int Npts = 200;
    const double w_lo = 1e-3, w_hi = 1e6;
    const double log_lo = log10(w_lo), log_hi = log10(w_hi);
    double prev_w = w_lo, prev_mag = absG0;
    for (int i = 0; i < Npts; ++i) {
        double frac = (double)i / (Npts - 1);
        double w = pow(10.0, log_lo + frac * (log_hi - log_lo));
        double Hr = 0, Hi = 0;
        if (bode_ss_at_freq_(A, B, C, D, w, &Hr, &Hi) != 0) continue;
        double mag = sqrt(Hr * Hr + Hi * Hi);
        if (mag < target && prev_mag >= target && i > 0) {
            /* Linear interpolation in log(w) for accuracy. */
            double t = (prev_mag - target) / (prev_mag - mag);
            double lw = log10(prev_w) + t * (log10(w) - log10(prev_w));
            return pow(10.0, lw);
        }
        prev_w = w; prev_mag = mag;
    }
    return INFINITY;
}

/*-------------------------------------------------------------------------
 * Discrete-to-continuous Tustin (bilinear) reverse mapping.
 *
 *   [A, B] = d2c_tustin(Ad, Bd, Ts)
 *
 * Inverts the Tustin discretisation of c2d_tustin by substituting
 * z = (1 + αs)/(1 − αs), α = Ts/2:
 *      A = (2/Ts)·(Ad − I)·(I + Ad)⁻¹
 *      B = (2/Ts)·(I + Ad)⁻¹·Bd
 *
 * Requires (I + Ad) to be invertible — fails for plants with an Ad
 * eigenvalue at z = -1 (impulse response with sample-period oscillation).
 * Returns 0×0 if singular.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_d2c_tustin_A(matlab_mat *Ad, matlab_mat *Bd, double Ts) {
    if (!Ad) return mat_alloc(0, 0);
    int64_t n = Ad->rows;
    if (n == 0 || Ad->cols != n || Ts <= 0) return mat_alloc(0, 0);
    (void)Bd;
    /* I + Ad. */
    matlab_mat *IpAd = mat_alloc(n, n);
    matlab_mat *AdmI = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double v = Ad->data[i * n + j];
            IpAd->data[i * n + j] = v + (i == j ? 1.0 : 0.0);
            AdmI->data[i * n + j] = v - (i == j ? 1.0 : 0.0);
        }
    matlab_mat *Inv = matlab_inv(IpAd);
    if (!Inv || Inv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Prod = matlab_matmul_mm(AdmI, Inv);
    matlab_mat *A = mat_alloc(n, n);
    double scale = 2.0 / Ts;
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            A->data[i * n + j] = scale * Prod->data[i * n + j];
    return A;
}

matlab_mat *matlab_d2c_tustin_B(matlab_mat *Ad, matlab_mat *Bd, double Ts) {
    if (!Ad || !Bd) return mat_alloc(0, 0);
    int64_t n = Ad->rows;
    if (n == 0 || Ad->cols != n || Bd->rows != n || Ts <= 0)
        return mat_alloc(0, 0);
    matlab_mat *IpAd = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double v = Ad->data[i * n + j];
            IpAd->data[i * n + j] = v + (i == j ? 1.0 : 0.0);
        }
    matlab_mat *Inv = matlab_inv(IpAd);
    if (!Inv || Inv->rows == 0) return mat_alloc(0, 0);
    matlab_mat *InvBd = matlab_matmul_mm(Inv, Bd);
    int64_t m = Bd->cols;
    matlab_mat *B = mat_alloc(n, m);
    double scale = 2.0 / Ts;
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            B->data[i * m + j] = scale * InvBd->data[i * m + j];
    return B;
}

/*-------------------------------------------------------------------------
 * Discrete-time stability test: isstable_d(A) returns 1.0 if every
 * eigenvalue of A is strictly inside the unit disk (|λ| < 1, Schur-
 * stable), else 0.0. Marginal eigenvalues on |λ| = 1 fail (per MATLAB
 * convention).
 *-------------------------------------------------------------------------*/
double matlab_isstable_d(matlab_mat *A) {
    if (!A || A->rows == 0 || A->cols != A->rows) return 0.0;
    matlab_mat *e  = matlab_eig(A);
    matlab_mat *Re = matlab_real_c(e);
    matlab_mat *Im = matlab_imag_c(e);
    int64_t n = Re->rows * Re->cols;
    for (int64_t i = 0; i < n; ++i) {
        double re = Re->data[i], im = Im->data[i];
        double mag2 = re * re + im * im;
        if (mag2 >= 1.0) return 0.0;   /* on or outside unit circle → fail */
    }
    return 1.0;
}

/*-------------------------------------------------------------------------
 * Discrete-time H₂ system norm.
 *
 *   norm_h2_d(A, B, C, D) = sqrt( trace(D · D') + trace(C · Wc · C') )
 * where Wc = dlyap(A, B · B'). The trace(D·D') term is the impulse-
 * response k=0 contribution; the gramian term is the rest.
 *
 * Returns +Inf if A is not Schur-stable (gramian unbounded). Unlike
 * the continuous case, D ≠ 0 is fine for discrete H₂.
 *-------------------------------------------------------------------------*/
double matlab_norm_h2_d(matlab_mat *A, matlab_mat *B,
                        matlab_mat *C, matlab_mat *D) {
    if (!A || !B || !C || !D) return INFINITY;
    int64_t n = A->rows;
    if (n == 0 || A->cols != n || B->rows != n || C->cols != n)
        return INFINITY;
    int64_t m = B->cols;
    int64_t p = C->rows;
    if (D->rows != p || D->cols != m) return INFINITY;
    if (matlab_isstable_d(A) == 0.0) return INFINITY;
    /* Wc = dlyap(A, B B'). */
    matlab_mat *Bt = matlab_transpose(B);
    matlab_mat *BBt = matlab_matmul_mm(B, Bt);
    matlab_mat *Wc = matlab_dlyap(A, BBt);
    if (!Wc || Wc->rows == 0) return INFINITY;
    /* C Wc C' trace. */
    matlab_mat *Ct = matlab_transpose(C);
    matlab_mat *WCt = matlab_matmul_mm(Wc, Ct);
    matlab_mat *CWCt = matlab_matmul_mm(C, WCt);
    double tr_gram = 0.0;
    for (int64_t i = 0; i < p; ++i) tr_gram += CWCt->data[i * p + i];
    /* D D' trace. */
    matlab_mat *Dt = matlab_transpose(D);
    matlab_mat *DDt = matlab_matmul_mm(D, Dt);
    double tr_D = 0.0;
    for (int64_t i = 0; i < p; ++i) tr_D += DDt->data[i * p + i];
    double tr = tr_gram + tr_D;
    return tr > 0.0 ? sqrt(tr) : 0.0;
}

/*-------------------------------------------------------------------------
 * Continuous-time Kalman filter — steady-state gain.
 *
 *   L = kalman_L(A, G, C, Qn, Rn)  for the plant
 *      xdot = A x + G w,  y = C x + v
 *      cov(w) = Qn,       cov(v) = Rn
 *   solves the dual ARE  A·P + P·A' − P·C'·Rn⁻¹·C·P + G·Qn·G' = 0
 *   for the unique stabilising P, then returns L = P · C' · Rn⁻¹.
 *
 *   The estimator dynamics  xdot_hat = (A − L·C) x_hat + L y  are
 *   Hurwitz; eig(A − L·C) are the estimator poles.
 *
 *   Implementation exploits the LQR/Kalman duality: the LQR gain on
 *   the dual system (A', C', G·Qn·G', Rn) is K_dual = Rn⁻¹ C P, so
 *   the Kalman gain is L = (K_dual)' = P C' Rn⁻¹. We just transpose
 *   `lqr(A', C', GQG', Rn)`.
 *
 *   The MATLAB-faithful 4-return shape `[kest, L, P] = kalman(sys, Qn,
 *   Rn)` (estimator state-space + gain + Riccati) is a follow-on once
 *   we have model objects.  Tier 4.2 of the CST roadmap.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_kalman_L(matlab_mat *A, matlab_mat *G, matlab_mat *C,
                            matlab_mat *Qn, matlab_mat *Rn) {
    if (!A || !G || !C || !Qn || !Rn) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0 || A->cols != n) return mat_alloc(0, 0);
    if (G->rows != n) return mat_alloc(0, 0);
    int64_t q = G->cols;
    int64_t p = C->rows;
    if (C->cols != n) return mat_alloc(0, 0);
    if (Qn->rows != q || Qn->cols != q) return mat_alloc(0, 0);
    if (Rn->rows != p || Rn->cols != p) return mat_alloc(0, 0);
    /* GQG' = effective process-noise covariance projected onto state space. */
    matlab_mat *Gt = matlab_transpose(G);
    matlab_mat *GQ = matlab_matmul_mm(G, Qn);
    matlab_mat *GQGt = matlab_matmul_mm(GQ, Gt);
    /* K_dual = lqr(A', C', GQG', Rn). */
    matlab_mat *At = matlab_transpose(A);
    matlab_mat *Ct = matlab_transpose(C);
    matlab_mat *Kdual = matlab_lqr(At, Ct, GQGt, Rn);
    if (!Kdual || Kdual->rows == 0) return mat_alloc(0, 0);
    /* L = K_dual'. */
    return matlab_transpose(Kdual);
}

/*-------------------------------------------------------------------------
 * Discrete-time Kalman filter — steady-state gain.
 *
 *   L = kalmd_L(Ad, G, C, Qn, Rn)  for the discrete plant
 *      x[k+1] = Ad x[k] + G w[k],  y[k] = C x[k] + v[k]
 *   solves the discrete dual ARE for the steady-state covariance P
 *   and returns L = P·C'·(C·P·C' + Rn)⁻¹ (the standard discrete
 *   Kalman gain). Implementation: L' = dlqr(Ad', C', G·Qn·G', Rn).
 *
 *   Limitation inherited from `dare`: requires Ad Schur-stable so
 *   the Newton-Kleinman seeding works.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_kalmd_L(matlab_mat *Ad, matlab_mat *G, matlab_mat *C,
                           matlab_mat *Qn, matlab_mat *Rn) {
    if (!Ad || !G || !C || !Qn || !Rn) return mat_alloc(0, 0);
    int64_t n = Ad->rows;
    if (n == 0 || Ad->cols != n) return mat_alloc(0, 0);
    int64_t q = G->cols;
    int64_t p = C->rows;
    if (G->rows != n || C->cols != n) return mat_alloc(0, 0);
    if (Qn->rows != q || Qn->cols != q) return mat_alloc(0, 0);
    if (Rn->rows != p || Rn->cols != p) return mat_alloc(0, 0);
    matlab_mat *Gt = matlab_transpose(G);
    matlab_mat *GQ = matlab_matmul_mm(G, Qn);
    matlab_mat *GQGt = matlab_matmul_mm(GQ, Gt);
    matlab_mat *At = matlab_transpose(Ad);
    matlab_mat *Ct = matlab_transpose(C);
    matlab_mat *Kdual = matlab_dlqr(At, Ct, GQGt, Rn);
    if (!Kdual || Kdual->rows == 0) return mat_alloc(0, 0);
    return matlab_transpose(Kdual);
}

/*-------------------------------------------------------------------------
 * Steady-state Kalman covariance — the Riccati solution.
 *
 *   P = kalman_P(A, G, C, Qn, Rn) solves the dual continuous ARE:
 *      A·P + P·A' − P·C'·Rn⁻¹·C·P + G·Qn·G' = 0
 *   which is the same as `care(A', C', G·Qn·G', Rn)`.
 *
 * Routes the second result of the multi-return [L, P] = kalman(...) /
 * [L, P] = kalmd(...) splitter; existing matlab_kalman_L / kalmd_L
 * cover the gain.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_kalman_P(matlab_mat *A, matlab_mat *G, matlab_mat *C,
                            matlab_mat *Qn, matlab_mat *Rn) {
    if (!A || !G || !C || !Qn || !Rn) return mat_alloc(0, 0);
    matlab_mat *Gt = matlab_transpose(G);
    matlab_mat *GQ = matlab_matmul_mm(G, Qn);
    matlab_mat *GQGt = matlab_matmul_mm(GQ, Gt);
    matlab_mat *At = matlab_transpose(A);
    matlab_mat *Ct = matlab_transpose(C);
    return matlab_care(At, Ct, GQGt, Rn);
}

matlab_mat *matlab_kalmd_P(matlab_mat *Ad, matlab_mat *G, matlab_mat *C,
                           matlab_mat *Qn, matlab_mat *Rn) {
    if (!Ad || !G || !C || !Qn || !Rn) return mat_alloc(0, 0);
    matlab_mat *Gt = matlab_transpose(G);
    matlab_mat *GQ = matlab_matmul_mm(G, Qn);
    matlab_mat *GQGt = matlab_matmul_mm(GQ, Gt);
    matlab_mat *At = matlab_transpose(Ad);
    matlab_mat *Ct = matlab_transpose(C);
    return matlab_dare(At, Ct, GQGt, Rn);
}

/*-------------------------------------------------------------------------
 * Step-response metrics.
 *
 *   stepinfo(y, t) returns a 1 × 5 row vector
 *     [RiseTime, SettlingTime, Overshoot, Peak, PeakTime]
 *
 * Definitions (MATLAB convention):
 *   - Final = y(end) (steady-state value, assumes the system has settled)
 *   - Peak = max |y|, PeakTime = t at that index
 *   - Overshoot = (Peak - |Final|) / |Final| * 100  (percent; 0 if Final==0)
 *   - RiseTime = t(first index where |y| ≥ 0.9·|Final|) − t(first index where |y| ≥ 0.1·|Final|)
 *   - SettlingTime = t(last index where |y - Final| > 0.02·|Final|), 0 if always within band
 *
 * The 6+ extra fields MATLAB's stepinfo struct exposes
 * (`SettlingMin`/`SettlingMax`/`Undershoot`/`TransientTime`) are
 * follow-ons; the five shipped here cover the common workflow.
 *
 * Tier-3 of CST roadmap. Pure post-processing — sits on top of any
 * step-response producer (`step_ss`, future model-object `step`).
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_stepinfo(matlab_mat *y, matlab_mat *t) {
    if (!y || !t) return mat_alloc(0, 0);
    int64_t n = y->rows * y->cols;
    if (n == 0 || t->rows * t->cols != n) return mat_alloc(0, 0);
    /* Final value = last sample. */
    double Final = y->data[n - 1];
    double absFinal = Final < 0 ? -Final : Final;
    /* Peak |y| and its time. */
    double Peak = 0.0;
    int64_t peakIdx = 0;
    for (int64_t i = 0; i < n; ++i) {
        double v = y->data[i] < 0 ? -y->data[i] : y->data[i];
        if (v > Peak) { Peak = v; peakIdx = i; }
    }
    double PeakTime = t->data[peakIdx];
    /* Overshoot (percent). */
    double Over = 0.0;
    if (absFinal > 0.0) Over = (Peak - absFinal) / absFinal * 100.0;
    if (Over < 0.0) Over = 0.0;   /* clip — overshoot is non-negative */
    /* Rise time: first 10% crossing → first 90% crossing.
     * Use signed Final to handle negative steady state correctly. */
    double t10 = 0.0, t90 = 0.0;
    int64_t i10 = -1, i90 = -1;
    double thresh10 = 0.1 * Final;
    double thresh90 = 0.9 * Final;
    for (int64_t i = 0; i < n; ++i) {
        double v = y->data[i];
        if (i10 < 0 && ((Final >= 0 && v >= thresh10) || (Final < 0 && v <= thresh10)))
            i10 = i;
        if (i90 < 0 && ((Final >= 0 && v >= thresh90) || (Final < 0 && v <= thresh90))) {
            i90 = i; break;
        }
    }
    if (i10 >= 0) t10 = t->data[i10];
    if (i90 >= 0) t90 = t->data[i90];
    double Rise = (i10 >= 0 && i90 >= 0) ? (t90 - t10) : 0.0;
    /* Settling time: last index where |y-Final| > 0.02 * |Final|. */
    double band = 0.02 * absFinal;
    int64_t settleIdx = -1;
    for (int64_t i = n - 1; i >= 0; --i) {
        double dev = y->data[i] - Final;
        if (dev < 0) dev = -dev;
        if (dev > band) { settleIdx = i; break; }
    }
    double Settle = settleIdx >= 0 ? t->data[settleIdx] : 0.0;
    /* Pack into a 1 × 5 row vector. */
    matlab_mat *out = mat_alloc(1, 5);
    out->data[0] = Rise;
    out->data[1] = Settle;
    out->data[2] = Over;
    out->data[3] = Peak;
    out->data[4] = PeakTime;
    return out;
}

/*-------------------------------------------------------------------------
 * State-space DC gain (continuous LTI).
 *
 *   dcgain_ss(A, B, C, D) = lim_{s→0} G(s) = D − C · A⁻¹ · B
 *
 * Returns a  p × m  matrix (one entry per output / input pair). For
 * SISO plants the result is 1 × 1.
 *
 * If A is singular (e.g. an integrator pole at the origin), the DC
 * gain is unbounded — matlab_inv signals this by returning a 0 × 0
 * matrix, which we propagate. Users should check `numel(out) > 0`.
 *
 * Tier-3 of CST roadmap. The MATLAB-faithful `dcgain(sys)` and
 * `dcgain(num, den)` forms are follow-ons (need model objects /
 * polynomial form respectively); the matrix-arg form is the canonical
 * positional API.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_dcgain_ss(matlab_mat *A, matlab_mat *B,
                             matlab_mat *C, matlab_mat *D) {
    if (!A || !B || !C || !D) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (n == 0 || A->cols != n || B->rows != n || C->cols != n)
        return mat_alloc(0, 0);
    int64_t m = B->cols;
    int64_t p = C->rows;
    if (D->rows != p || D->cols != m) return mat_alloc(0, 0);
    matlab_mat *Ainv = matlab_inv(A);
    if (!Ainv || Ainv->rows == 0) return mat_alloc(0, 0);  /* A singular */
    matlab_mat *AinvB = matlab_matmul_mm(Ainv, B);    /* n × m */
    matlab_mat *CAinvB = matlab_matmul_mm(C, AinvB);  /* p × m */
    matlab_mat *out = mat_alloc(p, m);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < m; ++j)
            out->data[i * m + j] = D->data[i * m + j] - CAinvB->data[i * m + j];
    return out;
}

/*-------------------------------------------------------------------------
 * H₂ system norm (continuous LTI, strictly proper).
 *
 *   norm_h2(A, B, C) = sqrt(trace(C · Wc · C'))
 *                    = sqrt(trace(B' · Wo · B))
 * where  Wc = gram_c(A, B) = lyap(A, B B'),
 *        Wo = gram_o(A, C) = lyap(A', C' C).
 *
 * The two formulas are equal — they're the integral of the impulse
 * response squared, which is the same quantity reached from either
 * gramian. We use the C·Wc·C' form (one Lyapunov solve plus a small
 * trace).
 *
 * Returns +Inf if A is not Hurwitz (gramian unbounded). Strictly-
 * proper assumption: when D ≠ 0, the H₂ norm is +Inf for continuous
 * systems (impulse response has a Dirac, integral is infinite). The
 * shipped form ignores D since it's typically 0 in CST workflows; a
 * `norm_h2_d` discrete variant is a follow-on.
 *
 * Tier-3 of CST roadmap. Sits cleanly on Tier-1.4 lyap.
 *-------------------------------------------------------------------------*/
double matlab_norm_h2(matlab_mat *A, matlab_mat *B, matlab_mat *C) {
    if (!A || !B || !C) return INFINITY;
    int64_t n = A->rows;
    if (n == 0 || A->cols != n || B->rows != n || C->cols != n)
        return INFINITY;
    /* Stability check: A must be Hurwitz, otherwise the gramian
     * doesn't exist (Lyapunov solve is ill-conditioned and the H₂
     * norm is unbounded). */
    if (matlab_isstable(A) == 0.0) return INFINITY;
    matlab_mat *Wc = matlab_gram_c(A, B);
    if (!Wc || Wc->rows == 0) return INFINITY;
    matlab_mat *Ct = matlab_transpose(C);
    matlab_mat *WCt = matlab_matmul_mm(Wc, Ct);     /* n x p */
    matlab_mat *CWCt = matlab_matmul_mm(C, WCt);    /* p x p */
    int64_t p = CWCt->rows;
    double tr = 0.0;
    for (int64_t i = 0; i < p; ++i) tr += CWCt->data[i * p + i];
    return tr > 0.0 ? sqrt(tr) : 0.0;
}

/*-------------------------------------------------------------------------
 * Stability test (continuous): isstable(A) returns 1.0 if every
 * eigenvalue of A has strictly negative real part (Hurwitz), else 0.0.
 * Polymorphic over real/complex eig output.
 *-------------------------------------------------------------------------*/
/* Forward declarations: real/imag part extractors live in
 * runtime_complex.cpp; the matlab_runtime.cpp translation unit doesn't
 * include matlab_runtime.h, so declare them locally. */
matlab_mat *matlab_real_c(void *A);
matlab_mat *matlab_imag_c(void *A);

double matlab_isstable(matlab_mat *A) {
    if (!A || A->rows == 0 || A->cols != A->rows) return 0.0;
    matlab_mat *e = matlab_eig(A);
    matlab_mat *Re = matlab_real_c(e);   /* zeros if e is real */
    int64_t n = Re->rows * Re->cols;
    for (int64_t i = 0; i < n; ++i) {
        if (Re->data[i] >= 0.0) return 0.0;
    }
    return 1.0;
}

/*-------------------------------------------------------------------------
 * Damping ratios + natural frequencies (continuous).
 *
 *   damp(A) returns an n x 2 matrix where row k is [wn_k, zeta_k] for
 *   eigenvalue lambda_k of A:
 *     wn   = |lambda|
 *     zeta = -real(lambda) / |lambda|   (so zeta = 1 for purely real
 *                                        Hurwitz poles, 0 for purely imaginary).
 *
 *   MATLAB's full `damp(sys)` returns four columns [pole, damping,
 *   freq, time-const]; here we ship the canonical two-column form
 *   that's what most workflows actually use. The full shape is a
 *   follow-on once we have model objects + multi-return splitters
 *   for the 4-tuple.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_damp(matlab_mat *A) {
    if (!A || A->rows == 0 || A->cols != A->rows) return mat_alloc(0, 0);
    matlab_mat *e = matlab_eig(A);
    matlab_mat *Re = matlab_real_c(e);
    matlab_mat *Im = matlab_imag_c(e);
    int64_t n = Re->rows * Re->cols;
    matlab_mat *out = mat_alloc(n, 2);
    for (int64_t i = 0; i < n; ++i) {
        double re = Re->data[i], im = Im->data[i];
        double wn = sqrt(re * re + im * im);
        double zeta = wn > 0.0 ? -re / wn : 0.0;
        out->data[i * 2 + 0] = wn;
        out->data[i * 2 + 1] = zeta;
    }
    return out;
}

/*-------------------------------------------------------------------------
 * Hankel singular values.
 *
 *   hsvd(A, B, C) returns sqrt(eig(Wc * Wo)) sorted descending, where
 *      Wc = gram_c(A, B) = lyap(A, B B')
 *      Wo = gram_o(A, C) = lyap(A', C' C)
 *
 *   The Hankel singular values are similarity-invariant — they are
 *   intrinsic input-output invariants of the system. Small HSVs
 *   indicate states that contribute little to the I/O map and are the
 *   diagnostic for balanced model reduction (`balred`/`balreal`).
 *
 *   Continuous LTI; discrete uses dlyap. Requires A Hurwitz so the
 *   gramians are bounded. Returns 0x0 if the gramians fail.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_hsvd(matlab_mat *A, matlab_mat *B, matlab_mat *C) {
    if (!A || !B || !C) return mat_alloc(0, 0);
    matlab_mat *Wc = matlab_gram_c(A, B);
    if (!Wc || Wc->rows == 0) return mat_alloc(0, 0);
    matlab_mat *Wo = matlab_gram_o(A, C);
    if (!Wo || Wo->rows == 0) return mat_alloc(0, 0);
    matlab_mat *WW = matlab_matmul_mm(Wc, Wo);  /* n x n */
    matlab_mat *e  = matlab_eig(WW);
    matlab_mat *Re = matlab_real_c(e);
    int64_t n = Re->rows * Re->cols;
    matlab_mat *out = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) {
        double v = Re->data[i];
        out->data[i] = v > 0.0 ? sqrt(v) : 0.0;
    }
    /* Sort descending — MATLAB convention (largest HSV first). */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j)
            if (out->data[i] < out->data[j]) {
                double t = out->data[i]; out->data[i] = out->data[j]; out->data[j] = t;
            }
    return out;
}

/*-------------------------------------------------------------------------
 * Controllability matrix.  Co = ctrb(A, B) = [B, A B, A^2 B, ..., A^{n-1} B].
 * The pair (A, B) is controllable iff rank(Co) = n. This is the
 * structural-rank companion to the energy-based gramian gram_c.
 * Returns an  n x (n*m)  matrix.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_ctrb(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (A->cols != n || B->rows != n) return mat_alloc(0, 0);
    int64_t m = B->cols;
    if (n == 0 || m == 0) return mat_alloc(0, 0);
    matlab_mat *Co = mat_alloc(n, n * m);
    /* Block 0 = B. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            Co->data[i * (n * m) + j] = B->data[i * m + j];
    /* Block k = A^k B = A * (block k-1). */
    matlab_mat *prev = B;
    for (int64_t k = 1; k < n; ++k) {
        matlab_mat *next = matlab_matmul_mm(A, prev);  /* n x m */
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < m; ++j)
                Co->data[i * (n * m) + (k * m + j)] = next->data[i * m + j];
        prev = next;
    }
    return Co;
}

/*-------------------------------------------------------------------------
 * Observability matrix.  Ob = obsv(A, C) = [C; C A; C A^2; ...; C A^{n-1}].
 * The pair (A, C) is observable iff rank(Ob) = n. Structural-rank
 * companion to the energy-based gramian gram_o.
 * Returns a  (p*n) x n  matrix.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_obsv(matlab_mat *A, matlab_mat *C) {
    if (!A || !C) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (A->cols != n || C->cols != n) return mat_alloc(0, 0);
    int64_t p = C->rows;
    if (n == 0 || p == 0) return mat_alloc(0, 0);
    matlab_mat *Ob = mat_alloc(p * n, n);
    /* Row-block 0 = C. */
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < n; ++j)
            Ob->data[i * n + j] = C->data[i * n + j];
    /* Block k = C A^k = (block k-1) * A. */
    matlab_mat *prev = C;
    for (int64_t k = 1; k < n; ++k) {
        matlab_mat *next = matlab_matmul_mm(prev, A);  /* p x n */
        for (int64_t i = 0; i < p; ++i)
            for (int64_t j = 0; j < n; ++j)
                Ob->data[(k * p + i) * n + j] = next->data[i * n + j];
        prev = next;
    }
    return Ob;
}

/*-------------------------------------------------------------------------
 * Pole placement (SISO) — Ackermann's formula.
 *
 *   K = place(A, B, P)  for SISO single-input plant places the closed-
 *   loop poles  eig(A - B K)  at the locations P (a length-n vector,
 *   real or complex). The Ackermann formula for SISO is
 *      K = [0 0 ... 0 1] * inv(ctrb(A, B)) * alpha(A)
 *   where  alpha(s) = prod_i (s - p_i) = s^n + a_{n-1} s^{n-1} + ... + a_0
 *   is the desired closed-loop characteristic polynomial.
 *
 *   alpha(A) is built by Horner on A: M = I; M = M*A + a_k I  (descending
 *   in coefficient index), with the leading 1 starting M off as I.
 *
 *   For complex P with conjugate pairs the resulting alpha has real
 *   coefficients, so K comes back real. We accept either real or
 *   matlab_mat_c P input by reading real/imag halves.
 *
 *   Multi-input place uses the Kautsky-Nichols-Van Dooren algorithm
 *   (extra degrees of freedom for orthogonal-eigenvector conditioning);
 *   deferred. SISO Ackermann is widely-used pedagogically and matches
 *   MATLAB's `acker(A, B, P)`.
 *-------------------------------------------------------------------------*/
matlab_mat *matlab_place(matlab_mat *A, matlab_mat *B, void *P_in) {
    if (!A || !B || !P_in) return mat_alloc(0, 0);
    int64_t n = A->rows;
    if (A->cols != n || B->rows != n || B->cols != 1) return mat_alloc(0, 0);
    /* Read P into real + imag arrays of length n. */
    int64_t pn = 0;
    std::vector<double> pr(n, 0.0), pi(n, 0.0);
    if (mat_is_complex(P_in)) {
        matlab_mat_c *Pc = (matlab_mat_c *)P_in;
        pn = Pc->rows * Pc->cols;
        if (pn != n) return mat_alloc(0, 0);
        for (int64_t i = 0; i < n; ++i) { pr[i] = Pc->re[i]; pi[i] = Pc->im[i]; }
    } else {
        matlab_mat *Pr = (matlab_mat *)P_in;
        pn = Pr->rows * Pr->cols;
        if (pn != n) return mat_alloc(0, 0);
        for (int64_t i = 0; i < n; ++i) { pr[i] = Pr->data[i]; pi[i] = 0.0; }
    }
    /* Build alpha(s) = prod (s - p_i) by complex multiplication. coef[k]
     * is the s^k coefficient (real part — imag must collapse to zero for
     * a valid set of conjugate-paired roots). Length n+1. */
    std::vector<double> ar(n + 1, 0.0), ai(n + 1, 0.0);
    ar[0] = 1.0;  /* polynomial = 1 (constant) initially */
    int64_t deg = 0;
    for (int64_t k = 0; k < n; ++k) {
        /* Multiply current polynomial by (s - p_k). */
        std::vector<double> nr(deg + 2, 0.0), ni(deg + 2, 0.0);
        /* nr/ni[j+1] += ar/ai[j]  (the s * poly part) */
        for (int64_t j = 0; j <= deg; ++j) {
            nr[j + 1] += ar[j];
            ni[j + 1] += ai[j];
            /* Subtract p_k * coef: (re + i*im) * (pr + i*pi). */
            double cr = ar[j], ci = ai[j];
            double mr = cr * pr[k] - ci * pi[k];
            double mi = cr * pi[k] + ci * pr[k];
            nr[j] -= mr;
            ni[j] -= mi;
        }
        for (int64_t j = 0; j <= deg + 1; ++j) { ar[j] = nr[j]; ai[j] = ni[j]; }
        deg += 1;
    }
    /* alpha(A) via Horner on A.  M starts at coef of s^n times I (= 1*I);
     * for k = n-1 down to 0:  M = M*A + ar[k] * I. */
    matlab_mat *M = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i) M->data[i * n + i] = 1.0;
    for (int64_t k = n - 1; k >= 0; --k) {
        matlab_mat *MA = matlab_matmul_mm(M, A);
        matlab_mat *N  = mat_alloc(n, n);
        double a_k = ar[k];
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                double v = MA->data[i * n + j];
                if (i == j) v += a_k;
                N->data[i * n + j] = v;
            }
        }
        M = N;
    }
    /* Build ctrb(A, B), invert it. */
    matlab_mat *Co = matlab_ctrb(A, B);  /* n x n for SISO */
    if (!Co || Co->rows != n || Co->cols != n) return mat_alloc(0, 0);
    matlab_mat *Coinv = matlab_inv(Co);
    if (!Coinv || Coinv->rows == 0) return mat_alloc(0, 0);
    /* K = e_n^T * Coinv * M  where  e_n^T = [0 ... 0 1]. So K is the
     * last row of  Coinv * M  (1 x n). */
    matlab_mat *CinvM = matlab_matmul_mm(Coinv, M);  /* n x n */
    matlab_mat *K = mat_alloc(1, n);
    for (int64_t j = 0; j < n; ++j)
        K->data[j] = CinvM->data[(n - 1) * n + j];
    return K;
}

/*---------- Element-wise arithmetic --------------------------------------*/

/* Forward declarations: the binary macros below check for a complex
 * operand at the top and delegate to the _cc variants; definitions are
 * in the complex section further down. */
matlab_mat_c *matlab_complex_scalar(double re, double im);
matlab_mat_c *matlab_mat_c_from_real(matlab_mat *A);
matlab_mat_c *matlab_add_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_sub_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_emul_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_ediv_cc(matlab_mat_c *A, matlab_mat_c *B);

static matlab_mat_c *to_mat_c(void *p) {
    if (!p) return NULL;
    if (mat_is_complex(p)) return (matlab_mat_c *)p;
    return matlab_mat_c_from_real((matlab_mat *)p);
}

/* Polymorphic matrix binary ops. When either operand is complex
 * (magic-tagged), both are promoted to matlab_mat_c* and the _cc
 * variant runs; otherwise the real fast path takes the original
 * (A, B) signature. The returned ptr is still matlab_mat* to the
 * caller — but if the actual payload is complex, the ptr points at
 * a matlab_mat_c with the magic-tag preserved, so downstream
 * polymorphic consumers keep routing correctly.
 *
 * `epow` is only defined for real inputs at runtime; it keeps the
 * old macro. */
#define BINARY_MM(name, op) \
    matlab_mat *matlab_##name##_mm(void *Ap, void *Bp) { \
        if (mat_is_complex(Ap) || mat_is_complex(Bp)) \
            return (matlab_mat *)matlab_##name##_cc(to_mat_c(Ap), to_mat_c(Bp)); \
        matlab_mat *A = (matlab_mat *)Ap; \
        matlab_mat *B = (matlab_mat *)Bp; \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) C->data[k] = (op); \
        return C; \
    }

#define BINARY_MS(name, op) \
    matlab_mat *matlab_##name##_ms(void *Ap, double s) { \
        if (mat_is_complex(Ap)) \
            return (matlab_mat *)matlab_##name##_cc( \
                (matlab_mat_c *)Ap, matlab_complex_scalar(s, 0)); \
        matlab_mat *A = (matlab_mat *)Ap; \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) C->data[k] = (op); \
        return C; \
    }

#define BINARY_SM(name, op) \
    matlab_mat *matlab_##name##_sm(double s, void *Ap) { \
        if (mat_is_complex(Ap)) \
            return (matlab_mat *)matlab_##name##_cc( \
                matlab_complex_scalar(s, 0), (matlab_mat_c *)Ap); \
        matlab_mat *A = (matlab_mat *)Ap; \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) C->data[k] = (op); \
        return C; \
    }

BINARY_MM(add,  A->data[k] + B->data[k])
BINARY_MM(sub,  A->data[k] - B->data[k])
BINARY_MM(emul, A->data[k] * B->data[k])
BINARY_MM(ediv, A->data[k] / B->data[k])
/* Expanded manually (no complex dispatch): matlab_epow_cc isn't
 * provided — complex pow is rarer than the other ops and its real-only
 * path keeps the ABI stable. */
matlab_mat *matlab_epow_mm(matlab_mat *A, matlab_mat *B) {
    int64_t m = A->rows, n = A->cols;
    matlab_mat *C = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k)
        C->data[k] = pow(A->data[k], B->data[k]);
    return C;
}

BINARY_MS(add,  A->data[k] + s)
BINARY_MS(sub,  A->data[k] - s)
BINARY_MS(emul, A->data[k] * s)
BINARY_MS(ediv, A->data[k] / s)
matlab_mat *matlab_epow_ms(matlab_mat *A, double s) {
    int64_t m = A->rows, n = A->cols;
    matlab_mat *C = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) C->data[k] = pow(A->data[k], s);
    return C;
}

BINARY_SM(add,  s + A->data[k])
BINARY_SM(sub,  s - A->data[k])
BINARY_SM(emul, s * A->data[k])
BINARY_SM(ediv, s / A->data[k])
matlab_mat *matlab_epow_sm(double s, matlab_mat *A) {
    int64_t m = A->rows, n = A->cols;
    matlab_mat *C = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) C->data[k] = pow(s, A->data[k]);
    return C;
}

/* Element-wise comparisons, returning 0.0/1.0 matrices so they feed
 * cleanly into logical indexing (A(A > 0), etc.). */
#define CMP_MM(name, op) \
    matlab_mat *matlab_##name##_mm(matlab_mat *A, matlab_mat *B) { \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) \
            C->data[k] = (A->data[k] op B->data[k]) ? 1.0 : 0.0; \
        return C; \
    }
#define CMP_MS(name, op) \
    matlab_mat *matlab_##name##_ms(matlab_mat *A, double s) { \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) \
            C->data[k] = (A->data[k] op s) ? 1.0 : 0.0; \
        return C; \
    }
#define CMP_SM(name, op) \
    matlab_mat *matlab_##name##_sm(double s, matlab_mat *A) { \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) \
            C->data[k] = (s op A->data[k]) ? 1.0 : 0.0; \
        return C; \
    }

CMP_MM(gt, >)  CMP_MS(gt, >)  CMP_SM(gt, >)
CMP_MM(ge, >=) CMP_MS(ge, >=) CMP_SM(ge, >=)
CMP_MM(lt, <)  CMP_MS(lt, <)  CMP_SM(lt, <)
CMP_MM(le, <=) CMP_MS(le, <=) CMP_SM(le, <=)
CMP_MM(eq, ==) CMP_MS(eq, ==) CMP_SM(eq, ==)
CMP_MM(ne, !=) CMP_MS(ne, !=) CMP_SM(ne, !=)

#undef CMP_MM
#undef CMP_MS
#undef CMP_SM

#undef BINARY_MM
#undef BINARY_MS
#undef BINARY_SM

/*---------- Element-wise unary -------------------------------------------*/

#define UNARY_M(name, expr) \
    matlab_mat *matlab_##name##_m(matlab_mat *A) { \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) { \
            double x = A->data[k]; C->data[k] = (expr); \
        } \
        return C; \
    }

UNARY_M(neg,  -x)
UNARY_M(exp,  exp(x))
UNARY_M(log,  log(x))
UNARY_M(sin,  sin(x))
UNARY_M(cos,  cos(x))
UNARY_M(tan,  tan(x))
/* Degree-argument trigonometry (sind/cosd/tand and the inverse forms).
 * deg->rad = pi/180; rad->deg = 180/pi. */
UNARY_M(sind,  sin(x * 0.017453292519943295))
UNARY_M(cosd,  cos(x * 0.017453292519943295))
UNARY_M(tand,  tan(x * 0.017453292519943295))
UNARY_M(asind, asin(x) * 57.29577951308232)
UNARY_M(acosd, acos(x) * 57.29577951308232)
UNARY_M(atand, atan(x) * 57.29577951308232)
UNARY_M(asin, asin(x))
UNARY_M(acos, acos(x))
UNARY_M(atan, atan(x))
UNARY_M(sinh, sinh(x))
UNARY_M(cosh, cosh(x))
UNARY_M(tanh, tanh(x))
UNARY_M(log2, log2(x))
UNARY_M(log10, log10(x))
UNARY_M(sqrt, sqrt(x))
UNARY_M(abs,  fabs(x))
UNARY_M(sign, (x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)))
UNARY_M(floor, floor(x))
UNARY_M(ceil,  ceil(x))
UNARY_M(round, round(x))   /* MATLAB rounds ties away from zero; C's round() matches. */
UNARY_M(fix,   trunc(x))   /* MATLAB fix = truncate toward zero. */

#undef UNARY_M

/* Scalar versions for when the operand is a plain f64 (needed when the
 * frontend couldn't statically prove the operand was scalar and the scalar
 * arith lowering didn't fire). */
double matlab_exp_s(double x)  { return exp(x);  }
double matlab_log_s(double x)  { return log(x);  }
double matlab_sin_s(double x)  { return sin(x);  }
double matlab_cos_s(double x)  { return cos(x);  }
double matlab_tan_s(double x)  { return tan(x);  }
double matlab_asin_s(double x) { return asin(x); }
double matlab_acos_s(double x) { return acos(x); }
double matlab_atan_s(double x) { return atan(x); }
double matlab_atan2_s(double y, double x) { return atan2(y, x); }
double matlab_sinh_s(double x) { return sinh(x); }
double matlab_cosh_s(double x) { return cosh(x); }
double matlab_tanh_s(double x) { return tanh(x); }
double matlab_log2_s(double x) { return log2(x); }
double matlab_log10_s(double x){ return log10(x); }
double matlab_sqrt_s(double x) { return sqrt(x); }
double matlab_abs_s(double x)  { return fabs(x); }
double matlab_sign_s(double x) {
    return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}
double matlab_floor_s(double x) { return floor(x); }
double matlab_ceil_s(double x)  { return ceil(x); }
double matlab_round_s(double x) { return round(x); }
double matlab_fix_s(double x)   { return trunc(x); }

/* Degree-argument trigonometry, scalar forms.  deg->rad = pi/180;
 * rad->deg = 180/pi. */
double matlab_sind_s(double x)  { return sin(x * 0.017453292519943295); }
double matlab_cosd_s(double x)  { return cos(x * 0.017453292519943295); }
double matlab_tand_s(double x)  { return tan(x * 0.017453292519943295); }
double matlab_asind_s(double x) { return asin(x) * 57.29577951308232; }
double matlab_acosd_s(double x) { return acos(x) * 57.29577951308232; }
double matlab_atand_s(double x) { return atan(x) * 57.29577951308232; }
double matlab_atan2d_s(double y, double x) {
    return atan2(y, x) * 57.29577951308232;
}

/* MATLAB `mod(a,b)`: result has same sign as b (or 0). `rem(a,b)`: sign
 * of a. C's fmod uses sign-of-a, so fmod == rem; derive mod from that. */
double matlab_rem_s(double a, double b) {
    if (b == 0.0) return a;            /* MATLAB: rem(a,0) == a */
    return fmod(a, b);
}
double matlab_mod_s(double a, double b) {
    if (b == 0.0) return a;            /* MATLAB: mod(a,0) == a */
    double r = fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    return r;
}

/* linspace(a, b, n): n points evenly spaced from a to b inclusive.
 * n < 2 returns just [b] per MATLAB. Returns a 1xn row matrix. */
matlab_mat *matlab_linspace(double a, double b, double nd) {
    int64_t n = (int64_t)nd;
    if (n < 1) n = 1;
    matlab_mat *C = mat_alloc(1, n);
    if (n == 1) { C->data[0] = b; return C; }
    double step = (b - a) / (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) C->data[i] = a + step * (double)i;
    C->data[n - 1] = b;                /* exact endpoint */
    return C;
}

/* logspace(a, b, n): n points logarithmically spaced from 10^a to
 * 10^b. Equivalent to 10 .^ linspace(a, b, n). Returns a 1×n row.
 * Used heavily for frequency-response grids in bode / nyquist /
 * allmargin workflows. */
matlab_mat *matlab_logspace(double a, double b, double nd) {
    int64_t n = (int64_t)nd;
    if (n < 1) n = 1;
    matlab_mat *C = mat_alloc(1, n);
    if (n == 1) { C->data[0] = std::pow(10.0, b); return C; }
    double step = (b - a) / (double)(n - 1);
    for (int64_t i = 0; i < n; ++i)
        C->data[i] = std::pow(10.0, a + step * (double)i);
    C->data[n - 1] = std::pow(10.0, b);
    return C;
}

/* atan2 on matrices: elementwise y vs x, both matrices must be same size. */
matlab_mat *matlab_atan2_m(matlab_mat *Y, matlab_mat *X) {
    if (!Y || !X) return mat_alloc(0, 0);
    int64_t m = Y->rows, n = Y->cols;
    if (X->rows != m || X->cols != n) return mat_alloc(0, 0);
    matlab_mat *C = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k)
        C->data[k] = atan2(Y->data[k], X->data[k]);
    return C;
}

/*---------- Indexing -----------------------------------------------------
 * A(i, j) scalar load: 1-based indexing like MATLAB. Out-of-range returns
 * 0 (silently) — a proper implementation would abort or raise an error.
 *-------------------------------------------------------------------------*/

double matlab_subscript2_s(matlab_mat *A, double i, double j) {
    int64_t ri = (int64_t)i - 1, cj = (int64_t)j - 1;
    /* Complex-aware companion to matlab_subscript1_s — see comment
     * there. Reads the real part of an indexed complex element. */
    if (mat_is_complex(A)) {
        matlab_mat_c *C = (matlab_mat_c *)A;
        if (ri < 0 || ri >= C->rows || cj < 0 || cj >= C->cols) return 0.0;
        return C->re[ri * C->cols + cj];
    }
    if (ri < 0 || ri >= A->rows || cj < 0 || cj >= A->cols) return 0.0;
    return A->data[ri * A->cols + cj];
}

double matlab_subscript1_s(matlab_mat *A, double i) {
    int64_t idx = (int64_t)i - 1;
    /* Complex-aware: matlab_mat_c shares the ptr lane.  Detect via
     * the magic word at offset 0 and read from the real-part buffer
     * (matches MATLAB's "real() then subscript" idiom and is the
     * scalar lane's only sensible interpretation when the caller
     * stored the result into an f64 slot). */
    if (mat_is_complex(A)) {
        matlab_mat_c *C = (matlab_mat_c *)A;
        int64_t total = C->rows * C->cols;
        if (idx < 0 || idx >= total) return 0.0;
        return C->re[idx];
    }
    int64_t total = A->rows * A->cols;
    if (idx < 0 || idx >= total) return 0.0;
    return A->data[idx];
}

/*---------- I/O ----------------------------------------------------------*/

/* ---------------------------------------------------------------------- */
/* Try / catch via an error flag.
 *
 * Without stack unwinding support (setjmp/longjmp or LLVM invoke)
 * we can't catch runtime faults. What we CAN catch cleanly is an
 * explicit error() call: matlab_set_error sets a process-global flag,
 * and the try-body's lowering wraps subsequent statements in an
 * scf.if(!flag) guard. After the try-body, the catch-body runs if the
 * flag is set, clearing it first.
 *
 * Single-threaded: parfor bodies don't currently participate in
 * try/catch. If they ever do, this needs thread-local storage.
 */
/* Non-static: shared with runtime_debug.cpp via runtime_internal.h. */
int32_t matlab_error_flag = 0;

/* Error message storage: a heap-copy of the most recent error() string.
 * `matlab_set_error_msg` trims to 1023 bytes and null-terminates;
 * `matlab_err_disp_message` routes to the I/O runtime so catch blocks can
 * do `disp(ME.message)` and get the raw text without needing a new
 * char-matrix descriptor. */
/* Non-static: shared with runtime_debug.cpp via runtime_internal.h. */
char    matlab_error_msg[1024] = {0};
int64_t matlab_error_msg_len   = 0;

/* Forward declarations for the debug-frame snapshot below — the dbg
 * state struct is defined later in this file but matlab_set_error_msg
 * needs to peek at it to capture a backtrace at error time. */
struct matlab_dbg_frame;
/* Defined in runtime_debug.cpp (Phase-2 split). */
void matlab_err_snapshot_frames(void);
/* Defined in runtime_debug.cpp (Phase-2 split). */
void matlab_err_emit_traceback_to_stderr(void);

void matlab_set_error(void) {
    matlab_error_flag = 1;
    matlab_err_snapshot_frames();
    matlab_err_emit_traceback_to_stderr();
}
int32_t matlab_check_error(void) { return matlab_error_flag; }
void matlab_clear_error(void) {
    /* Only clear the flag — the message stays available for the catch
     * body to read (e.g. via ME.message). A subsequent error() call
     * will overwrite the message via matlab_set_error_msg. */
    matlab_error_flag = 0;
}

void matlab_set_error_msg(const char *msg, int64_t len) {
    matlab_error_flag = 1;
    int64_t n = len;
    if (n < 0) n = 0;
    if (n > 1023) n = 1023;
    if (msg && n > 0) memcpy(matlab_error_msg, msg, (size_t)n);
    matlab_error_msg[n] = '\0';
    matlab_error_msg_len = n;
    matlab_err_snapshot_frames();
    matlab_err_emit_traceback_to_stderr();
}

void matlab_disp_str(const char *s, int64_t n); /* forward decl */

void matlab_err_disp_message(void) {
    if (matlab_error_msg_len > 0)
        matlab_disp_str(matlab_error_msg, matlab_error_msg_len);
    else {
        static const char empty[] = "";
        matlab_disp_str(empty, 0);
    }
}

/* ---------------------------------------------------------------------- */
/* Struct storage — s.field = v with f64 and matlab_mat* field values.
 *
 * matlab_struct holds a parallel table of field name / value / kind
 * entries. Lookup is linear scan: name counts in MATLAB structs are
 * small (tens at most), and a hash table would complicate the
 * transpile-friendly C. Fields are looked up case-sensitively. A fresh
 * struct starts empty; set-field appends if the name is new, or
 * overwrites in place if it already exists.
 *
 * Kind tag:
 *   0 = f64 (value held in the double slot)
 *   1 = matlab_mat* (pointer held in the ptr slot)
 *   2 = matlab_struct* (nested struct)
 * Getting a missing field as f64 returns 0.0; getting as a ptr
 * returns a fresh empty matrix so downstream code doesn't crash on
 * null. */
#define MATLAB_STRUCT_CAP_INIT 4

/* matlab_struct_s layout lives in runtime_internal.h (Phase-2 split). */

matlab_struct *matlab_struct_new(void) {
    matlab_struct *s = (matlab_struct *)calloc(1, sizeof(*s));
    s->capacity = MATLAB_STRUCT_CAP_INIT;
    s->names    = (char **)calloc((size_t)s->capacity, sizeof(char *));
    s->kinds    = (int32_t *)calloc((size_t)s->capacity, sizeof(int32_t));
    s->f64_vals = (double *)calloc((size_t)s->capacity, sizeof(double));
    s->ptr_vals = (void **)calloc((size_t)s->capacity, sizeof(void *));
    return s;
}

int32_t struct_find_field(matlab_struct *s, const char *name, int32_t len) {
    for (int32_t i = 0; i < s->nfields; ++i) {
        if ((int32_t)strlen(s->names[i]) == len &&
            memcmp(s->names[i], name, (size_t)len) == 0) {
            return i;
        }
    }
    return -1;
}

static void struct_grow_if_needed(matlab_struct *s) {
    if (s->nfields < s->capacity) return;
    int32_t NewCap = s->capacity * 2;
    s->names    = (char **)realloc(s->names,    (size_t)NewCap * sizeof(char *));
    s->kinds    = (int32_t *)realloc(s->kinds,  (size_t)NewCap * sizeof(int32_t));
    s->f64_vals = (double *)realloc(s->f64_vals,(size_t)NewCap * sizeof(double));
    s->ptr_vals = (void **)realloc(s->ptr_vals, (size_t)NewCap * sizeof(void *));
    for (int32_t i = s->capacity; i < NewCap; ++i) {
        s->names[i] = NULL;
        s->kinds[i] = 0;
        s->f64_vals[i] = 0.0;
        s->ptr_vals[i] = NULL;
    }
    s->capacity = NewCap;
}

int32_t struct_reserve(matlab_struct *s, const char *name, int32_t len) {
    int32_t idx = struct_find_field(s, name, len);
    if (idx >= 0) return idx;
    struct_grow_if_needed(s);
    idx = s->nfields++;
    char *copy = (char *)malloc((size_t)len + 1);
    memcpy(copy, name, (size_t)len);
    copy[len] = '\0';
    s->names[idx] = copy;
    s->kinds[idx] = 0;
    s->f64_vals[idx] = 0.0;
    s->ptr_vals[idx] = NULL;
    return idx;
}

void matlab_struct_set_f64(matlab_struct *s, const char *name, int64_t len, double v) {
    if (!s) return;
    int32_t idx = struct_reserve(s, name, (int32_t)len);
    s->kinds[idx] = 0;
    s->f64_vals[idx] = v;
    s->ptr_vals[idx] = NULL;
}

void matlab_struct_set_mat(matlab_struct *s, const char *name, int64_t len, matlab_mat *m) {
    if (!s) return;
    int32_t idx = struct_reserve(s, name, (int32_t)len);
    s->kinds[idx] = 1;
    s->f64_vals[idx] = 0.0;
    s->ptr_vals[idx] = m;
}

/* String-typed property storage.  Stored with kind=3 to match the
 * workspace-side string encoding (see matlab_struct_get_mat's kind=3
 * pass-through).  Used by classdef kwarg-ctor sugar when a property
 * value is a string literal (e.g. `txsite('Name','X', ...)`). */
void matlab_struct_set_string(matlab_struct *s, const char *name, int64_t len, void *str) {
    if (!s) return;
    int32_t idx = struct_reserve(s, name, (int32_t)len);
    s->kinds[idx] = 3;
    s->f64_vals[idx] = 0.0;
    s->ptr_vals[idx] = str;
}

void *matlab_struct_get_string(matlab_struct *s, const char *name, int64_t len) {
    if (!s) return NULL;
    int32_t idx = struct_find_field(s, name, (int32_t)len);
    if (idx < 0) return NULL;
    if (s->kinds[idx] != 3) return NULL;
    return s->ptr_vals[idx];
}

double matlab_struct_get_f64(matlab_struct *s, const char *name, int64_t len) {
    if (!s) return 0.0;
    int32_t idx = struct_find_field(s, name, (int32_t)len);
    if (idx < 0) return 0.0;
    if (s->kinds[idx] == 0) return s->f64_vals[idx];
    /* If the field holds a 1x1 matrix, unbox to scalar. */
    if (s->kinds[idx] == 1 && s->ptr_vals[idx]) {
        matlab_mat *m = (matlab_mat *)s->ptr_vals[idx];
        if (m->rows == 1 && m->cols == 1) return m->data[0];
    }
    return 0.0;
}

matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name, int64_t len) {
    if (!s) return mat_alloc(0, 0);
    int32_t idx = struct_find_field(s, name, (int32_t)len);
    if (idx < 0) return mat_alloc(0, 0);
    if (s->kinds[idx] == 1 && s->ptr_vals[idx])
        return (matlab_mat *)s->ptr_vals[idx];
    /* kind=2 is a class instance pointer. The caller is the lowered
     * code reading a script-level class-bound variable, and Sema has
     * already typed it as a matlab_obj* — it's only routed through
     * the _get_mat entry because the workspace path is uniformly
     * ptr-typed. Pass the pointer through verbatim so dot-property
     * access and method dispatch see the obj they expect. The
     * historical mat_alloc(0, 0) fallback was harmless under the
     * old kind=1 storage but actively wrong now that obj instances
     * carry kind=2. */
    if (s->kinds[idx] == 2 && s->ptr_vals[idx])
        return (matlab_mat *)s->ptr_vals[idx];
    /* kind=3 is a matlab_string* (script-level "..." binding). The
     * caller is the lowered code reading a script-scope variable
     * through the uniformly ptr-typed _get_mat entry; pass the
     * pointer through verbatim so the string-aware sites
     * downstream (matlab_string_disp, etc.) see the descriptor.
     * Without this, the historical mat_alloc(0, 0) fallback would
     * make `t` (where t is a string) silently print as nothing
     * after we route assignments through matlab_ws_set_string. */
    if (s->kinds[idx] == 3 && s->ptr_vals[idx])
        return (matlab_mat *)s->ptr_vals[idx];
    /* Phase 1.1.F: typed-int matrices (kind=4 / 5 = matlab_mat_u8 *,
     * matlab_mat_i32 *). Same pass-through shape as kind=2/3 — the
     * uniformly-ptr-typed _get_mat entry returns the descriptor pointer
     * verbatim; downstream sites that special-case typed ints
     * (matlab_disp_mat, the binop dispatch) consult the intlane registry
     * to recover the lane. Without this branch the kind=4/5 lookup
     * fell through to `mat_alloc(0, 0)`, which is why bare-name display
     * of an int32/uint8 matrix in the REPL silently printed nothing. */
    if ((s->kinds[idx] == 4 || s->kinds[idx] == 5) && s->ptr_vals[idx])
        return (matlab_mat *)s->ptr_vals[idx];
    /* Phase 5.x — table (6), categorical (9), datetime (10), duration (11)
     * are pointer-shaped values whose workspace storage uses ptr_vals[]
     * the same way kind=2/3 do. The script-scope read path lowers every
     * non-string / non-sym name through matlab_ws_get_mat, but Sema
     * knows the binding's actual type and lowers downstream calls
     * (matlab_table_height, matlab_table_width, matlab_categorical_*, …)
     * accordingly. Returning the raw pointer here keeps the layout
     * faithful so those typed callees read the right fields; the old
     * mat_alloc(0,0) fallback turned every table operation into garbage
     * (height(T) came back as 0 from the all-zero shape, width(T) read
     * past the end of the empty mat into allocator-leak bytes — the
     * `-1.50101e+09` the user saw). */
    if ((s->kinds[idx] == 6  ||
         s->kinds[idx] == 9  ||
         s->kinds[idx] == 10 ||
         s->kinds[idx] == 11 ||
         /* kind=12 is matlab_struct* — plain field-holder, layout-
          * compatible with matlab_obj*.  Pass the pointer through so
          * cross-REPL-turn field accesses see the descriptor. */
         s->kinds[idx] == 12) && s->ptr_vals[idx])
        return (matlab_mat *)s->ptr_vals[idx];
    /* Box a scalar field into a 1x1 matrix. */
    if (s->kinds[idx] == 0) {
        matlab_mat *m = mat_alloc(1, 1);
        m->data[0] = s->f64_vals[idx];
        return m;
    }
    return mat_alloc(0, 0);
}

double matlab_struct_has_field(matlab_struct *s, const char *name, int64_t len) {
    if (!s) return 0.0;
    return struct_find_field(s, name, (int32_t)len) >= 0 ? 1.0 : 0.0;
}

/* ---------------------------------------------------------------------- */
/* Integer type casts. Runtime is still f64 internally, but int32(x),
 * uint8(x), logical(x), etc. truncate and saturate the way MATLAB's
 * typed lattice demands so downstream arithmetic sees the right value.
 * The result stays f64 (our sole numeric dtype), which keeps disp,
 * fprintf and the arithmetic runtime working unchanged. */
static double sat(double x, double lo, double hi) {
    double t = trunc(x);
    if (t < lo) return lo;
    if (t > hi) return hi;
    return t;
}

double matlab_int8_s(double x)   { return sat(x, -128.0,        127.0); }
double matlab_int16_s(double x)  { return sat(x, -32768.0,      32767.0); }
double matlab_int32_s(double x)  { return sat(x, -2147483648.0, 2147483647.0); }
double matlab_int64_s(double x)  { return sat(x, -9.2233720368547758e18,
                                                  9.2233720368547758e18); }
double matlab_uint8_s(double x)  { return sat(x, 0.0, 255.0); }
double matlab_uint16_s(double x) { return sat(x, 0.0, 65535.0); }
double matlab_uint32_s(double x) { return sat(x, 0.0, 4294967295.0); }
double matlab_uint64_s(double x) { return sat(x, 0.0, 1.8446744073709552e19); }
double matlab_double_s(double x) { return x; }
double matlab_single_s(double x) { return (double)(float)x; }
double matlab_logical_s(double x) { return x != 0.0 ? 1.0 : 0.0; }

/* ---------------------------------------------------------------------- */
/* Fixed-Point Designer (fi) helpers — see docs/emit_fixed_point.md §6.2.
 *
 * The stored integer is the only state; the FixedSpec (WL/FL/signedness/
 * overflow/rounding) is passed by argument at every call so the runtime
 * stays stateless and the lowering compiler can fold these calls when
 * it has constant arguments. Native int64/uint64 are the widest stored
 * lanes; sub-native widths (WL=12 etc) are masked + saturated by the
 * caller per assignment.
 *
 * Conventions:
 *   - signed-shift code paths assume two's-complement arithmetic-right-
 *     shift, which gcc/clang/MSVC all implement on every target we
 *     compile against. We don't pretend portability beyond that.
 *   - Nearest rounds halves toward +Inf (`round-half-up`), matching the
 *     MATLAB `Nearest` mode.
 *   - Floor truncates toward -Inf, matching MATLAB `Floor`. */

int64_t matlab_fi_sat_s64(int64_t x, uint8_t WL) {
    if (WL == 0) return 0;
    if (WL >= 64) return x;
    int64_t hi = ((int64_t)1 << (WL - 1)) - 1;
    int64_t lo = -((int64_t)1 << (WL - 1));
    if (x > hi) return hi;
    if (x < lo) return lo;
    return x;
}

uint64_t matlab_fi_sat_u64(uint64_t x, uint8_t WL) {
    if (WL == 0) return 0;
    if (WL >= 64) return x;
    uint64_t hi = ((uint64_t)1 << WL) - 1u;
    if (x > hi) return hi;
    return x;
}

int64_t matlab_fi_round_floor_s(int64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return x < 0 ? -1 : 0;
    return x >> shift; /* arithmetic on gcc/clang/MSVC */
}

int64_t matlab_fi_round_nearest_s(int64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return 0;
    /* round-half-up: add half-LSB, then arithmetic shift. */
    int64_t half = (int64_t)1 << (shift - 1);
    return (x + half) >> shift;
}

uint64_t matlab_fi_round_floor_u(uint64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return 0;
    return x >> shift;
}

uint64_t matlab_fi_round_nearest_u(uint64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return 0;
    uint64_t half = (uint64_t)1 << (shift - 1);
    return (x + half) >> shift;
}

/* Zero rounding (truncate toward zero). For non-negative values this
 * matches Floor; for negative values, add `2^shift - 1` before the
 * arithmetic right shift so the truncation lands on the "smaller in
 * magnitude" integer rather than the more-negative one. */
int64_t matlab_fi_round_zero_s(int64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return 0;
    if (x >= 0) return x >> shift;
    int64_t bias = ((int64_t)1 << shift) - 1;
    return (x + bias) >> shift;
}
uint64_t matlab_fi_round_zero_u(uint64_t x, uint8_t shift) {
    /* Unsigned: zero == floor. */
    return matlab_fi_round_floor_u(x, shift);
}

/* Ceiling rounding (toward +infinity). Add `2^shift - 1` then arithmetic
 * shift — the bias is just enough to push any non-zero remainder up. */
int64_t matlab_fi_round_ceiling_s(int64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return x > 0 ? 1 : 0;
    int64_t bias = ((int64_t)1 << shift) - 1;
    return (x + bias) >> shift;
}
uint64_t matlab_fi_round_ceiling_u(uint64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return x > 0 ? 1 : 0;
    uint64_t bias = ((uint64_t)1 << shift) - 1;
    return (x + bias) >> shift;
}

/* Convergent rounding (banker's, round-half-to-even). Halves round to
 * the nearest even integer rather than always up — eliminates the small
 * positive bias that round-half-up introduces in long DSP chains.
 *
 * Formula: shifted = (x + half - 1 + ((x >> shift) & 1)) >> shift
 *   - For non-half cases this matches Nearest (the +1 in the lsb is
 *     dominated by the existing fractional bits).
 *   - For exact halves the +(parity of pre-shift LSB) tie-breaks to
 *     even: round up if odd, round down if even. */
int64_t matlab_fi_round_convergent_s(int64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return 0;
    int64_t half = (int64_t)1 << (shift - 1);
    int64_t lsb = (x >> shift) & 1;
    return (x + half - 1 + lsb) >> shift;
}
uint64_t matlab_fi_round_convergent_u(uint64_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 64) return 0;
    uint64_t half = (uint64_t)1 << (shift - 1);
    uint64_t lsb = (x >> shift) & 1;
    return (x + half - 1 + lsb) >> shift;
}

/* Convert a real-world double to the stored integer for a fi (signed,WL,FL).
 * Applies the rounding mode to the fractional part, then the overflow mode
 * to the integer-magnitude clip. Phase 1 ships Floor + Nearest; the rest
 * set the error flag so callers can detect unsupported modes. */
int64_t matlab_fi_quantize_s(double v, uint8_t WL, int8_t FL,
                             uint8_t overflow, uint8_t rounding) {
    /* Scale to the fixed-point domain. FL may exceed 53 (mantissa bits),
     * in which case the input double can't represent the full range
     * losslessly — we accept that and document the limitation. */
    double scaled = ldexp(v, FL);
    int64_t stored;
    switch (rounding) {
    case 0: stored = (int64_t)floor(scaled); break;       /* Floor */
    case 1: stored = (int64_t)floor(scaled + 0.5); break; /* Nearest */
    case 2: stored = (int64_t)trunc(scaled); break;       /* Zero */
    case 3: {                                              /* Convergent */
        double r = round(scaled);
        /* round(0.5) returns 1.0 in C99; we need round-half-to-even. */
        double frac = scaled - floor(scaled);
        if (frac == 0.5) {
            int64_t lo = (int64_t)floor(scaled);
            stored = (lo % 2 == 0) ? lo : lo + 1;
        } else {
            stored = (int64_t)r;
        }
        break;
    }
    case 4: stored = (int64_t)ceil(scaled); break;        /* Ceiling */
    default:
        matlab_set_error();
        return 0;
    }
    if (overflow == 1) return matlab_fi_sat_s64(stored, WL);
    /* Wrap: mask to WL bits then sign-extend. */
    if (WL == 0) return 0;
    if (WL >= 64) return stored;
    uint64_t mask = ((uint64_t)1 << WL) - 1u;
    uint64_t bits = ((uint64_t)stored) & mask;
    /* Sign-extend from bit (WL-1). */
    if (bits & ((uint64_t)1 << (WL - 1))) bits |= ~mask;
    return (int64_t)bits;
}

uint64_t matlab_fi_quantize_u(double v, uint8_t WL, int8_t FL,
                              uint8_t overflow, uint8_t rounding) {
    double scaled = ldexp(v, FL);
    if (scaled < 0.0) scaled = 0.0;
    uint64_t stored;
    switch (rounding) {
    case 0: stored = (uint64_t)floor(scaled); break;       /* Floor */
    case 1: stored = (uint64_t)floor(scaled + 0.5); break; /* Nearest */
    case 2: stored = (uint64_t)trunc(scaled); break;       /* Zero (== Floor for unsigned) */
    case 3: {                                               /* Convergent */
        double frac = scaled - floor(scaled);
        if (frac == 0.5) {
            uint64_t lo = (uint64_t)floor(scaled);
            stored = (lo % 2 == 0) ? lo : lo + 1;
        } else {
            stored = (uint64_t)round(scaled);
        }
        break;
    }
    case 4: stored = (uint64_t)ceil(scaled); break;         /* Ceiling */
    default:
        matlab_set_error();
        return 0;
    }
    if (overflow == 1) return matlab_fi_sat_u64(stored, WL);
    if (WL == 0) return 0;
    if (WL >= 64) return stored;
    uint64_t mask = ((uint64_t)1 << WL) - 1u;
    return stored & mask;
}

void matlab_fi_disp_s(int64_t stored, uint8_t WL, int8_t FL) {
    (void)WL;
    /* Render the real-world value: stored * 2^-FL. */
    double v = ldexp((double)stored, -FL);
    matlab_disp_f64(v);
}

void matlab_fi_disp_u(uint64_t stored, uint8_t WL, int8_t FL) {
    (void)WL;
    double v = ldexp((double)stored, -FL);
    matlab_disp_f64(v);
}

/* bin / hex / dec — see matlab_runtime.h §fi.
 * The output strings match MATLAB's fi formatting:
 *   bin: WL bits, MSB first, leading zeros preserved.
 *   hex: ceil(WL/4) hex digits, zero-padded.
 *   dec: signed/unsigned decimal of the stored integer, no padding. */
/* matlab_string_from_literal is defined later in this TU; declare it here
 * with the matching matlab_string* signature so the fi helpers can use
 * it without a circular include. The matlab_string struct itself is
 * declared lazily — we only need a forward struct tag. */
struct matlab_string_s;
extern struct matlab_string_s *matlab_string_from_literal(const char *src,
                                                          int64_t len);

static void *fi_format_string(const char *fmt, ...) {
    char buf[80];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    if ((size_t)n > sizeof buf - 1) n = (int)(sizeof buf - 1);
    return matlab_string_from_literal(buf, (int64_t)n);
}

void *matlab_fi_bin_s(int64_t stored, uint8_t WL) {
    if (WL == 0) return matlab_string_from_literal("", 0);
    if (WL > 64) WL = 64;
    char buf[65];
    uint64_t mask = (WL >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << WL) - 1u);
    uint64_t bits = (uint64_t)stored & mask;
    for (int i = 0; i < WL; ++i)
        buf[WL - 1 - i] = (bits & ((uint64_t)1 << i)) ? '1' : '0';
    buf[WL] = '\0';
    return matlab_string_from_literal(buf, WL);
}

void *matlab_fi_bin_u(uint64_t stored, uint8_t WL) {
    return matlab_fi_bin_s((int64_t)stored, WL);
}

void *matlab_fi_hex_s(int64_t stored, uint8_t WL) {
    if (WL == 0) return matlab_string_from_literal("", 0);
    if (WL > 64) WL = 64;
    int digits = (WL + 3) / 4;
    uint64_t mask = (WL >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << WL) - 1u);
    uint64_t bits = (uint64_t)stored & mask;
    char buf[24];
    int n = snprintf(buf, sizeof buf, "%0*llx", digits, (unsigned long long)bits);
    if (n < 0) n = 0;
    return matlab_string_from_literal(buf, (int64_t)n);
}

void *matlab_fi_hex_u(uint64_t stored, uint8_t WL) {
    return matlab_fi_hex_s((int64_t)stored, WL);
}

void *matlab_fi_dec_s(int64_t stored, uint8_t WL) {
    (void)WL;
    return fi_format_string("%lld", (long long)stored);
}

void *matlab_fi_dec_u(uint64_t stored, uint8_t WL) {
    (void)WL;
    return fi_format_string("%llu", (unsigned long long)stored);
}

/* ---------------------------------------------------------------------- */
/* Typed integer matrix descriptors for `fi` arrays — see plan §6.3.
 *
 * Same layout as matlab_mat but with int64_t / uint64_t element data.
 * No magic field — the fi codepath knows it's working with a typed
 * descriptor from Sema/lowering, so polymorphic dispatch isn't needed
 * (and disp is wired through the fi-aware lowering hook).
 *
 * Phase 3 ships 64-bit lanes only. Tighter lanes (i8/i16/i32) become
 * relevant when the compiler proves a lane fits — Phase 5+. */

typedef struct matlab_mat_i64 {
    int64_t *data;
    int64_t  rows;
    int64_t  cols;
} matlab_mat_i64;

typedef struct matlab_mat_u64 {
    uint64_t *data;
    int64_t   rows;
    int64_t   cols;
} matlab_mat_u64;

static matlab_mat_i64 *mat_i64_alloc(int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    matlab_mat_i64 *A = (matlab_mat_i64 *)calloc(1, sizeof(*A));
    A->rows = m; A->cols = n;
    A->data = (int64_t *)calloc((size_t)(m * n + 1), sizeof(int64_t));
    return A;
}

static matlab_mat_u64 *mat_u64_alloc(int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    matlab_mat_u64 *A = (matlab_mat_u64 *)calloc(1, sizeof(*A));
    A->rows = m; A->cols = n;
    A->data = (uint64_t *)calloc((size_t)(m * n + 1), sizeof(uint64_t));
    return A;
}

matlab_mat_i64 *matlab_mat_i64_zeros(double rows, double cols) {
    return mat_i64_alloc((int64_t)rows, (int64_t)cols);
}
matlab_mat_u64 *matlab_mat_u64_zeros(double rows, double cols) {
    return mat_u64_alloc((int64_t)rows, (int64_t)cols);
}

matlab_mat_i64 *matlab_mat_i64_from_buf(const int64_t *buf,
                                         double rows, double cols) {
    int64_t m = (int64_t)rows, n = (int64_t)cols;
    matlab_mat_i64 *A = mat_i64_alloc(m, n);
    if (buf && m * n > 0)
        memcpy(A->data, buf, (size_t)(m * n) * sizeof(int64_t));
    return A;
}
matlab_mat_u64 *matlab_mat_u64_from_buf(const uint64_t *buf,
                                         double rows, double cols) {
    int64_t m = (int64_t)rows, n = (int64_t)cols;
    matlab_mat_u64 *A = mat_u64_alloc(m, n);
    if (buf && m * n > 0)
        memcpy(A->data, buf, (size_t)(m * n) * sizeof(uint64_t));
    return A;
}
matlab_mat_i64 *matlab_mat_i64_from_scalar(int64_t v) {
    matlab_mat_i64 *A = mat_i64_alloc(1, 1);
    A->data[0] = v;
    return A;
}
matlab_mat_u64 *matlab_mat_u64_from_scalar(uint64_t v) {
    matlab_mat_u64 *A = mat_u64_alloc(1, 1);
    A->data[0] = v;
    return A;
}

double matlab_mat_i64_length(matlab_mat_i64 *A) {
    if (!A) return 0.0;
    return (double)(A->rows > A->cols ? A->rows : A->cols);
}
double matlab_mat_i64_numel(matlab_mat_i64 *A) {
    if (!A) return 0.0;
    return (double)(A->rows * A->cols);
}
double matlab_mat_i64_size_dim(matlab_mat_i64 *A, double dim) {
    if (!A) return 0.0;
    int d = (int)dim;
    if (d == 1) return (double)A->rows;
    if (d == 2) return (double)A->cols;
    return 1.0;
}
int64_t matlab_mat_i64_rows(matlab_mat_i64 *A) { return A ? A->rows : 0; }
int64_t matlab_mat_i64_cols(matlab_mat_i64 *A) { return A ? A->cols : 0; }

double matlab_mat_u64_length(matlab_mat_u64 *A) {
    if (!A) return 0.0;
    return (double)(A->rows > A->cols ? A->rows : A->cols);
}
double matlab_mat_u64_numel(matlab_mat_u64 *A) {
    if (!A) return 0.0;
    return (double)(A->rows * A->cols);
}
double matlab_mat_u64_size_dim(matlab_mat_u64 *A, double dim) {
    if (!A) return 0.0;
    int d = (int)dim;
    if (d == 1) return (double)A->rows;
    if (d == 2) return (double)A->cols;
    return 1.0;
}

/* Linear-index helper. MATLAB indices are 1-based; row vectors and
 * column vectors collapse to a single dimension. */
static int64_t mat_lin_idx(int64_t rows, int64_t cols, double i) {
    int64_t k = (int64_t)i - 1;
    if (k < 0) k = 0;
    int64_t total = rows * cols;
    if (k >= total) k = total - 1;
    return k;
}

int64_t matlab_mat_i64_subscript1_s(matlab_mat_i64 *A, double i) {
    if (!A || A->rows * A->cols == 0) return 0;
    return A->data[mat_lin_idx(A->rows, A->cols, i)];
}
int64_t matlab_mat_i64_subscript2_s(matlab_mat_i64 *A, double i, double j) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t r = (int64_t)i - 1, c = (int64_t)j - 1;
    if (r < 0) r = 0; if (r >= A->rows) r = A->rows - 1;
    if (c < 0) c = 0; if (c >= A->cols) c = A->cols - 1;
    return A->data[r * A->cols + c];
}
uint64_t matlab_mat_u64_subscript1_s(matlab_mat_u64 *A, double i) {
    if (!A || A->rows * A->cols == 0) return 0;
    return A->data[mat_lin_idx(A->rows, A->cols, i)];
}
uint64_t matlab_mat_u64_subscript2_s(matlab_mat_u64 *A, double i, double j) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t r = (int64_t)i - 1, c = (int64_t)j - 1;
    if (r < 0) r = 0; if (r >= A->rows) r = A->rows - 1;
    if (c < 0) c = 0; if (c >= A->cols) c = A->cols - 1;
    return A->data[r * A->cols + c];
}

/* slice1 — gather elements along a 1-D index vector (which is itself
 * a matlab_mat of doubles, e.g. produced by `1:end-1`). */
matlab_mat_i64 *matlab_mat_i64_slice1(matlab_mat_i64 *A, matlab_mat *idx) {
    int64_t n = idx ? idx->rows * idx->cols : 0;
    /* Result is a row vector when A is a row vector or scalar; otherwise
     * a column. For Phase 3 (FIR shape) row is the common case. */
    int64_t rr = (A && A->rows == 1) ? 1 : (n > 0 ? n : 0);
    int64_t cc = (A && A->rows == 1) ? n : 1;
    matlab_mat_i64 *R = mat_i64_alloc(rr, cc);
    for (int64_t k = 0; k < n; ++k) {
        double idxv = idx->data[k];
        R->data[k] = matlab_mat_i64_subscript1_s(A, idxv);
    }
    return R;
}
matlab_mat_u64 *matlab_mat_u64_slice1(matlab_mat_u64 *A, matlab_mat *idx) {
    int64_t n = idx ? idx->rows * idx->cols : 0;
    int64_t rr = (A && A->rows == 1) ? 1 : (n > 0 ? n : 0);
    int64_t cc = (A && A->rows == 1) ? n : 1;
    matlab_mat_u64 *R = mat_u64_alloc(rr, cc);
    for (int64_t k = 0; k < n; ++k) {
        double idxv = idx->data[k];
        R->data[k] = matlab_mat_u64_subscript1_s(A, idxv);
    }
    return R;
}

void matlab_mat_i64_set1_s(matlab_mat_i64 *A, double i, int64_t v) {
    if (!A || A->rows * A->cols == 0) return;
    A->data[mat_lin_idx(A->rows, A->cols, i)] = v;
}
void matlab_mat_u64_set1_s(matlab_mat_u64 *A, double i, uint64_t v) {
    if (!A || A->rows * A->cols == 0) return;
    A->data[mat_lin_idx(A->rows, A->cols, i)] = v;
}

void matlab_mat_i64_fill(matlab_mat_i64 *A, int64_t v) {
    if (!A) return;
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) A->data[k] = v;
}
void matlab_mat_u64_fill(matlab_mat_u64 *A, uint64_t v) {
    if (!A) return;
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) A->data[k] = v;
}

/* Row-vector concat: `[A, B]` along the column axis when both are row
 * vectors. The FIR shift register `[x, delay_line(1:end-1)]` is the
 * gating use case. */
matlab_mat_i64 *matlab_mat_i64_concat_row(matlab_mat_i64 *A,
                                           matlab_mat_i64 *B) {
    int64_t na = A ? A->rows * A->cols : 0;
    int64_t nb = B ? B->rows * B->cols : 0;
    matlab_mat_i64 *R = mat_i64_alloc(1, na + nb);
    if (A && A->data) memcpy(R->data, A->data, (size_t)na * sizeof(int64_t));
    if (B && B->data) memcpy(R->data + na, B->data,
                              (size_t)nb * sizeof(int64_t));
    return R;
}
matlab_mat_u64 *matlab_mat_u64_concat_row(matlab_mat_u64 *A,
                                           matlab_mat_u64 *B) {
    int64_t na = A ? A->rows * A->cols : 0;
    int64_t nb = B ? B->rows * B->cols : 0;
    matlab_mat_u64 *R = mat_u64_alloc(1, na + nb);
    if (A && A->data) memcpy(R->data, A->data, (size_t)na * sizeof(uint64_t));
    if (B && B->data) memcpy(R->data + na, B->data,
                              (size_t)nb * sizeof(uint64_t));
    return R;
}

int64_t matlab_mat_i64_sum(matlab_mat_i64 *A) {
    if (!A) return 0;
    int64_t n = A->rows * A->cols, acc = 0;
    for (int64_t k = 0; k < n; ++k) acc += A->data[k];
    return acc;
}
uint64_t matlab_mat_u64_sum(matlab_mat_u64 *A) {
    if (!A) return 0;
    int64_t n = A->rows * A->cols;
    uint64_t acc = 0;
    for (int64_t k = 0; k < n; ++k) acc += A->data[k];
    return acc;
}

void matlab_mat_i64_disp(matlab_mat_i64 *A, uint8_t WL, int8_t FL) {
    (void)WL;
    if (!A) { matlab_disp_str("(null)", 6); return; }
    int64_t n = A->rows * A->cols;
    pthread_mutex_lock(&matlab_io_mutex);
    /* Render as a row of real-world values (stored * 2^-FL), one line
     * per row. The format roughly matches matlab_disp_mat_f64. */
    for (int64_t r = 0; r < A->rows; ++r) {
        for (int64_t c = 0; c < A->cols; ++c) {
            double v = ldexp((double)A->data[r * A->cols + c], -FL);
            printf("   %7g", v);
        }
        putchar('\n');
    }
    if (n == 0) putchar('\n');
    pthread_mutex_unlock(&matlab_io_mutex);
}
void matlab_mat_u64_disp(matlab_mat_u64 *A, uint8_t WL, int8_t FL) {
    (void)WL;
    if (!A) { matlab_disp_str("(null)", 6); return; }
    int64_t n = A->rows * A->cols;
    pthread_mutex_lock(&matlab_io_mutex);
    for (int64_t r = 0; r < A->rows; ++r) {
        for (int64_t c = 0; c < A->cols; ++c) {
            double v = ldexp((double)A->data[r * A->cols + c], -FL);
            printf("   %7g", v);
        }
        putchar('\n');
    }
    if (n == 0) putchar('\n');
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* ---------------------------------------------------------------------- */
/* Native integer matrix descriptors (Phase 1.1, Option B).
 *
 * Two narrow lanes — uint8 (image data, byte buffers) and int32 (default
 * MATLAB integer for non-double arithmetic). Same row-major layout as
 * matlab_mat / matlab_mat_i64. Saturating semantics live at the cast
 * + arithmetic boundary (Phase 1.1.B); the constructors / indexers /
 * set / slice / disp here are pure storage primitives.
 *
 * Future narrow lanes (i8, i16, u16, u32) follow the same template and
 * land mechanically once this pair is plumbed through lowering. */

typedef struct matlab_mat_u8 {
    uint8_t *data;
    int64_t  rows;
    int64_t  cols;
} matlab_mat_u8;

typedef struct matlab_mat_i32 {
    int32_t *data;
    int64_t  rows;
    int64_t  cols;
} matlab_mat_i32;

/* Phase 1.1.F: typed-int descriptor pointer registry. The matlab_mat_u8
 * and matlab_mat_i32 layouts have data at offset 0 (no magic word, unlike
 * matlab_mat_c) — adding a magic field would break the existing fast path
 * that the binop loops rely on. Instead, every typed-int alloc registers
 * its pointer here so the polymorphic matlab_disp_mat can detect typed-
 * int descriptors arriving through the f64 disp path (REPL, DAP) and
 * reroute them to matlab_mat_u8_disp / matlab_mat_i32_disp. The same
 * registry is also queried by matlab_dbg_ws_kind so the DAP variable
 * inspector labels typed-int bindings as "MxN int32" rather than
 * "MxN double". Mirrors the matlab_string_registry pattern. */
static struct {
    pthread_mutex_t mu;
    void   **ptrs;
    uint8_t *kinds;   /* 0 = u8, 1 = i32 */
    int      count;
    int      cap;
} matlab_intlane_registry = { PTHREAD_MUTEX_INITIALIZER, NULL, NULL, 0, 0 };

static void mat_intlane_registry_add(void *p, uint8_t kind) {
    if (!p) return;
    pthread_mutex_lock(&matlab_intlane_registry.mu);
    if (matlab_intlane_registry.count == matlab_intlane_registry.cap) {
        int ncap = matlab_intlane_registry.cap ? matlab_intlane_registry.cap * 2 : 16;
        void   **np = (void **)realloc(matlab_intlane_registry.ptrs,
                                       (size_t)ncap * sizeof(void *));
        uint8_t *nk = (uint8_t *)realloc(matlab_intlane_registry.kinds,
                                         (size_t)ncap * sizeof(uint8_t));
        if (np && nk) {
            matlab_intlane_registry.ptrs  = np;
            matlab_intlane_registry.kinds = nk;
            matlab_intlane_registry.cap   = ncap;
        }
    }
    if (matlab_intlane_registry.count < matlab_intlane_registry.cap) {
        int i = matlab_intlane_registry.count++;
        matlab_intlane_registry.ptrs[i]  = p;
        matlab_intlane_registry.kinds[i] = kind;
    }
    pthread_mutex_unlock(&matlab_intlane_registry.mu);
}

/* Public ABI: returns -1 if p is not a registered typed-int pointer,
 * 0 for u8, 1 for i32. Used by matlab_disp_mat (in this TU) and by
 * matlab_dbg_ws_kind in runtime_debug.cpp. */
extern "C" int matlab_mat_intlane_kind(const void *p) {
    if (!p) return -1;
    int kind = -1;
    pthread_mutex_lock(&matlab_intlane_registry.mu);
    for (int i = 0; i < matlab_intlane_registry.count; ++i) {
        if (matlab_intlane_registry.ptrs[i] == p) {
            kind = matlab_intlane_registry.kinds[i];
            break;
        }
    }
    pthread_mutex_unlock(&matlab_intlane_registry.mu);
    return kind;
}

static matlab_mat_u8 *mat_u8_alloc(int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    matlab_mat_u8 *A = (matlab_mat_u8 *)calloc(1, sizeof(*A));
    A->rows = m; A->cols = n;
    A->data = (uint8_t *)calloc((size_t)(m * n + 1), sizeof(uint8_t));
    mat_intlane_registry_add(A, /*kind=*/0);
    return A;
}

static matlab_mat_i32 *mat_i32_alloc(int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    matlab_mat_i32 *A = (matlab_mat_i32 *)calloc(1, sizeof(*A));
    A->rows = m; A->cols = n;
    A->data = (int32_t *)calloc((size_t)(m * n + 1), sizeof(int32_t));
    mat_intlane_registry_add(A, /*kind=*/1);
    return A;
}

/* Constructors. */
matlab_mat_u8 *matlab_mat_u8_zeros(double rows, double cols) {
    return mat_u8_alloc((int64_t)rows, (int64_t)cols);
}
matlab_mat_i32 *matlab_mat_i32_zeros(double rows, double cols) {
    return mat_i32_alloc((int64_t)rows, (int64_t)cols);
}

matlab_mat_u8 *matlab_mat_u8_ones(double rows, double cols) {
    matlab_mat_u8 *A = mat_u8_alloc((int64_t)rows, (int64_t)cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) A->data[k] = 1;
    return A;
}
matlab_mat_i32 *matlab_mat_i32_ones(double rows, double cols) {
    matlab_mat_i32 *A = mat_i32_alloc((int64_t)rows, (int64_t)cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) A->data[k] = 1;
    return A;
}

matlab_mat_u8 *matlab_mat_u8_eye(double rows, double cols) {
    matlab_mat_u8 *A = mat_u8_alloc((int64_t)rows, (int64_t)cols);
    int64_t d = A->rows < A->cols ? A->rows : A->cols;
    for (int64_t k = 0; k < d; ++k) A->data[k * A->cols + k] = 1;
    return A;
}
matlab_mat_i32 *matlab_mat_i32_eye(double rows, double cols) {
    matlab_mat_i32 *A = mat_i32_alloc((int64_t)rows, (int64_t)cols);
    int64_t d = A->rows < A->cols ? A->rows : A->cols;
    for (int64_t k = 0; k < d; ++k) A->data[k * A->cols + k] = 1;
    return A;
}

matlab_mat_u8 *matlab_mat_u8_from_buf(const uint8_t *buf,
                                      double rows, double cols) {
    int64_t m = (int64_t)rows, n = (int64_t)cols;
    matlab_mat_u8 *A = mat_u8_alloc(m, n);
    if (buf && m * n > 0)
        memcpy(A->data, buf, (size_t)(m * n) * sizeof(uint8_t));
    return A;
}
matlab_mat_i32 *matlab_mat_i32_from_buf(const int32_t *buf,
                                        double rows, double cols) {
    int64_t m = (int64_t)rows, n = (int64_t)cols;
    matlab_mat_i32 *A = mat_i32_alloc(m, n);
    if (buf && m * n > 0)
        memcpy(A->data, buf, (size_t)(m * n) * sizeof(int32_t));
    return A;
}

matlab_mat_u8 *matlab_mat_u8_from_scalar(uint8_t v) {
    matlab_mat_u8 *A = mat_u8_alloc(1, 1);
    A->data[0] = v;
    return A;
}
matlab_mat_i32 *matlab_mat_i32_from_scalar(int32_t v) {
    matlab_mat_i32 *A = mat_i32_alloc(1, 1);
    A->data[0] = v;
    return A;
}

/* Shape / predicates. */
double matlab_mat_u8_length(matlab_mat_u8 *A) {
    if (!A) return 0.0;
    return (double)(A->rows > A->cols ? A->rows : A->cols);
}
double matlab_mat_u8_numel(matlab_mat_u8 *A) {
    if (!A) return 0.0;
    return (double)(A->rows * A->cols);
}
double matlab_mat_u8_size_dim(matlab_mat_u8 *A, double dim) {
    if (!A) return 0.0;
    int d = (int)dim;
    if (d == 1) return (double)A->rows;
    if (d == 2) return (double)A->cols;
    return 1.0;
}
int64_t matlab_mat_u8_rows(matlab_mat_u8 *A) { return A ? A->rows : 0; }
int64_t matlab_mat_u8_cols(matlab_mat_u8 *A) { return A ? A->cols : 0; }

double matlab_mat_i32_length(matlab_mat_i32 *A) {
    if (!A) return 0.0;
    return (double)(A->rows > A->cols ? A->rows : A->cols);
}
double matlab_mat_i32_numel(matlab_mat_i32 *A) {
    if (!A) return 0.0;
    return (double)(A->rows * A->cols);
}
double matlab_mat_i32_size_dim(matlab_mat_i32 *A, double dim) {
    if (!A) return 0.0;
    int d = (int)dim;
    if (d == 1) return (double)A->rows;
    if (d == 2) return (double)A->cols;
    return 1.0;
}
int64_t matlab_mat_i32_rows(matlab_mat_i32 *A) { return A ? A->rows : 0; }
int64_t matlab_mat_i32_cols(matlab_mat_i32 *A) { return A ? A->cols : 0; }

/* Indexing. Both subscript1 (linear) and subscript2 (row,col) follow
 * MATLAB's 1-based convention; out-of-range indices clamp to the
 * boundary, matching the matlab_mat / matlab_mat_i64 idiom. */
uint8_t matlab_mat_u8_subscript1_s(matlab_mat_u8 *A, double i) {
    if (!A || A->rows * A->cols == 0) return 0;
    return A->data[mat_lin_idx(A->rows, A->cols, i)];
}
uint8_t matlab_mat_u8_subscript2_s(matlab_mat_u8 *A, double i, double j) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t r = (int64_t)i - 1, c = (int64_t)j - 1;
    if (r < 0) r = 0; if (r >= A->rows) r = A->rows - 1;
    if (c < 0) c = 0; if (c >= A->cols) c = A->cols - 1;
    return A->data[r * A->cols + c];
}
int32_t matlab_mat_i32_subscript1_s(matlab_mat_i32 *A, double i) {
    if (!A || A->rows * A->cols == 0) return 0;
    return A->data[mat_lin_idx(A->rows, A->cols, i)];
}
int32_t matlab_mat_i32_subscript2_s(matlab_mat_i32 *A, double i, double j) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t r = (int64_t)i - 1, c = (int64_t)j - 1;
    if (r < 0) r = 0; if (r >= A->rows) r = A->rows - 1;
    if (c < 0) c = 0; if (c >= A->cols) c = A->cols - 1;
    return A->data[r * A->cols + c];
}

/* Stores — caller is expected to pre-saturate via the cast helpers
 * (Phase 1.1.B). Stores here are raw assignment, no clamp. */
void matlab_mat_u8_set1_s(matlab_mat_u8 *A, double i, uint8_t v) {
    if (!A || A->rows * A->cols == 0) return;
    A->data[mat_lin_idx(A->rows, A->cols, i)] = v;
}
void matlab_mat_u8_set2_s(matlab_mat_u8 *A, double i, double j, uint8_t v) {
    if (!A || A->rows * A->cols == 0) return;
    int64_t r = (int64_t)i - 1, c = (int64_t)j - 1;
    if (r < 0) r = 0; if (r >= A->rows) r = A->rows - 1;
    if (c < 0) c = 0; if (c >= A->cols) c = A->cols - 1;
    A->data[r * A->cols + c] = v;
}
void matlab_mat_i32_set1_s(matlab_mat_i32 *A, double i, int32_t v) {
    if (!A || A->rows * A->cols == 0) return;
    A->data[mat_lin_idx(A->rows, A->cols, i)] = v;
}
void matlab_mat_i32_set2_s(matlab_mat_i32 *A, double i, double j, int32_t v) {
    if (!A || A->rows * A->cols == 0) return;
    int64_t r = (int64_t)i - 1, c = (int64_t)j - 1;
    if (r < 0) r = 0; if (r >= A->rows) r = A->rows - 1;
    if (c < 0) c = 0; if (c >= A->cols) c = A->cols - 1;
    A->data[r * A->cols + c] = v;
}

/* slice1 — 1-D gather along an index vector (matlab_mat of doubles). */
matlab_mat_u8 *matlab_mat_u8_slice1(matlab_mat_u8 *A, matlab_mat *idx) {
    int64_t n = idx ? idx->rows * idx->cols : 0;
    int64_t rr = (A && A->rows == 1) ? 1 : (n > 0 ? n : 0);
    int64_t cc = (A && A->rows == 1) ? n : 1;
    matlab_mat_u8 *R = mat_u8_alloc(rr, cc);
    for (int64_t k = 0; k < n; ++k)
        R->data[k] = matlab_mat_u8_subscript1_s(A, idx->data[k]);
    return R;
}
matlab_mat_i32 *matlab_mat_i32_slice1(matlab_mat_i32 *A, matlab_mat *idx) {
    int64_t n = idx ? idx->rows * idx->cols : 0;
    int64_t rr = (A && A->rows == 1) ? 1 : (n > 0 ? n : 0);
    int64_t cc = (A && A->rows == 1) ? n : 1;
    matlab_mat_i32 *R = mat_i32_alloc(rr, cc);
    for (int64_t k = 0; k < n; ++k)
        R->data[k] = matlab_mat_i32_subscript1_s(A, idx->data[k]);
    return R;
}

/* slice2 — 2-D gather along (rows, cols) index vectors. */
matlab_mat_u8 *matlab_mat_u8_slice2(matlab_mat_u8 *A,
                                    matlab_mat *rows, matlab_mat *cols) {
    int64_t nr = rows ? rows->rows * rows->cols : 0;
    int64_t nc = cols ? cols->rows * cols->cols : 0;
    matlab_mat_u8 *R = mat_u8_alloc(nr, nc);
    for (int64_t i = 0; i < nr; ++i) {
        double ri = rows->data[i];
        for (int64_t j = 0; j < nc; ++j) {
            double cj = cols->data[j];
            R->data[i * nc + j] = matlab_mat_u8_subscript2_s(A, ri, cj);
        }
    }
    return R;
}
matlab_mat_i32 *matlab_mat_i32_slice2(matlab_mat_i32 *A,
                                      matlab_mat *rows, matlab_mat *cols) {
    int64_t nr = rows ? rows->rows * rows->cols : 0;
    int64_t nc = cols ? cols->rows * cols->cols : 0;
    matlab_mat_i32 *R = mat_i32_alloc(nr, nc);
    for (int64_t i = 0; i < nr; ++i) {
        double ri = rows->data[i];
        for (int64_t j = 0; j < nc; ++j) {
            double cj = cols->data[j];
            R->data[i * nc + j] = matlab_mat_i32_subscript2_s(A, ri, cj);
        }
    }
    return R;
}

/* fill — every element to a constant. */
void matlab_mat_u8_fill(matlab_mat_u8 *A, uint8_t v) {
    if (!A) return;
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) A->data[k] = v;
}
void matlab_mat_i32_fill(matlab_mat_i32 *A, int32_t v) {
    if (!A) return;
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) A->data[k] = v;
}

/* Row concat — `[A, B]` along columns when both are row vectors.
 * Col concat — `[A; B]` along rows when both have matching cols. */
matlab_mat_u8 *matlab_mat_u8_concat_row(matlab_mat_u8 *A, matlab_mat_u8 *B) {
    int64_t na = A ? A->rows * A->cols : 0;
    int64_t nb = B ? B->rows * B->cols : 0;
    matlab_mat_u8 *R = mat_u8_alloc(1, na + nb);
    if (A && A->data) memcpy(R->data,      A->data, (size_t)na);
    if (B && B->data) memcpy(R->data + na, B->data, (size_t)nb);
    return R;
}
matlab_mat_u8 *matlab_mat_u8_concat_col(matlab_mat_u8 *A, matlab_mat_u8 *B) {
    int64_t ar = A ? A->rows : 0, ac = A ? A->cols : 0;
    int64_t br = B ? B->rows : 0, bc = B ? B->cols : 0;
    int64_t cc = ac > bc ? ac : bc;
    matlab_mat_u8 *R = mat_u8_alloc(ar + br, cc);
    if (A && A->data)
        for (int64_t r = 0; r < ar; ++r)
            memcpy(R->data + r * cc, A->data + r * ac, (size_t)ac);
    if (B && B->data)
        for (int64_t r = 0; r < br; ++r)
            memcpy(R->data + (ar + r) * cc, B->data + r * bc, (size_t)bc);
    return R;
}
matlab_mat_i32 *matlab_mat_i32_concat_row(matlab_mat_i32 *A,
                                          matlab_mat_i32 *B) {
    int64_t na = A ? A->rows * A->cols : 0;
    int64_t nb = B ? B->rows * B->cols : 0;
    matlab_mat_i32 *R = mat_i32_alloc(1, na + nb);
    if (A && A->data)
        memcpy(R->data,      A->data, (size_t)na * sizeof(int32_t));
    if (B && B->data)
        memcpy(R->data + na, B->data, (size_t)nb * sizeof(int32_t));
    return R;
}
matlab_mat_i32 *matlab_mat_i32_concat_col(matlab_mat_i32 *A,
                                          matlab_mat_i32 *B) {
    int64_t ar = A ? A->rows : 0, ac = A ? A->cols : 0;
    int64_t br = B ? B->rows : 0, bc = B ? B->cols : 0;
    int64_t cc = ac > bc ? ac : bc;
    matlab_mat_i32 *R = mat_i32_alloc(ar + br, cc);
    if (A && A->data)
        for (int64_t r = 0; r < ar; ++r)
            memcpy(R->data + r * cc,
                   A->data + r * ac, (size_t)ac * sizeof(int32_t));
    if (B && B->data)
        for (int64_t r = 0; r < br; ++r)
            memcpy(R->data + (ar + r) * cc,
                   B->data + r * bc, (size_t)bc * sizeof(int32_t));
    return R;
}

/* disp — integer formatting (no decimal point, MATLAB native-int style).
 * Width matches matlab_disp_mat_f64 column padding so mixed displays
 * line up reasonably. */
void matlab_mat_u8_disp(matlab_mat_u8 *A) {
    if (!A) { matlab_disp_str("(null)", 6); return; }
    int64_t n = A->rows * A->cols;
    pthread_mutex_lock(&matlab_io_mutex);
    for (int64_t r = 0; r < A->rows; ++r) {
        for (int64_t c = 0; c < A->cols; ++c)
            printf("   %4u", (unsigned)A->data[r * A->cols + c]);
        putchar('\n');
    }
    if (n == 0) putchar('\n');
    pthread_mutex_unlock(&matlab_io_mutex);
}
void matlab_mat_i32_disp(matlab_mat_i32 *A) {
    if (!A) { matlab_disp_str("(null)", 6); return; }
    int64_t n = A->rows * A->cols;
    pthread_mutex_lock(&matlab_io_mutex);
    for (int64_t r = 0; r < A->rows; ++r) {
        for (int64_t c = 0; c < A->cols; ++c)
            printf("   %11d", (int)A->data[r * A->cols + c]);
        putchar('\n');
    }
    if (n == 0) putchar('\n');
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* ---------------------------------------------------------------------- */
/* Phase 1.1.B — saturating arithmetic, comparisons, casts.
 *
 * MATLAB integer arithmetic saturates by default (intmath default = 'on');
 * division rounds half-away-from-zero, matching MATLAB's native-int cast
 * convention. NaN inputs to a cast produce 0 (MATLAB rule).
 *
 * Comparisons return matlab_mat (double 0/1) — the same logical-encoding
 * that the rest of the runtime uses; downstream `if`/`while` already know
 * how to consume it. */

/* Saturation helpers — kept inline so the binop loops stay tight. */
static inline uint8_t sat_d_to_u8(double v) {
    if (v != v)        return 0;          /* NaN */
    if (v <= 0.0)      return 0;
    if (v >= 255.0)    return 255;
    /* Round half-away-from-zero (MATLAB native-int cast rule). */
    return (uint8_t)(v + 0.5);
}
static inline int32_t sat_d_to_i32(double v) {
    if (v != v) return 0;
    if (v <= -2147483648.0) return INT32_MIN;
    if (v >=  2147483647.0) return INT32_MAX;
    return (int32_t)(v >= 0 ? v + 0.5 : v - 0.5);
}
static inline uint8_t sat_i32_to_u8(int32_t v) {
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

/* Saturating arith primitives. */
static inline uint8_t sat_add_u8(uint8_t a, uint8_t b) {
    int r = (int)a + (int)b;
    return r > 255 ? 255 : (uint8_t)r;
}
static inline uint8_t sat_sub_u8(uint8_t a, uint8_t b) {
    return a > b ? (uint8_t)(a - b) : 0;
}
static inline uint8_t sat_mul_u8(uint8_t a, uint8_t b) {
    int r = (int)a * (int)b;
    return r > 255 ? 255 : (uint8_t)r;
}
static inline int32_t sat_add_i32(int32_t a, int32_t b) {
    int64_t r = (int64_t)a + (int64_t)b;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}
static inline int32_t sat_sub_i32(int32_t a, int32_t b) {
    int64_t r = (int64_t)a - (int64_t)b;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}
static inline int32_t sat_mul_i32(int32_t a, int32_t b) {
    int64_t r = (int64_t)a * (int64_t)b;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}

/* Integer division with MATLAB's round-half-away-from-zero semantics.
 * Divide-by-zero saturates to ±max (or 0 when numerator is zero), matching
 * `int32(1)/int32(0)` → INT32_MAX in MATLAB. */
static inline uint8_t round_div_u8(uint8_t a, uint8_t b) {
    if (b == 0) return a == 0 ? 0 : 255;
    unsigned q = (unsigned)a / (unsigned)b;
    unsigned r = (unsigned)a % (unsigned)b;
    if (r * 2u >= (unsigned)b) q += 1u;
    return q > 255u ? 255u : (uint8_t)q;
}
static inline int32_t round_div_i32(int32_t a, int32_t b) {
    if (b == 0) return a == 0 ? 0 : (a > 0 ? INT32_MAX : INT32_MIN);
    int64_t aa = a, bb = b;
    int sign = ((aa < 0) ^ (bb < 0)) ? -1 : 1;
    int64_t abs_a = aa < 0 ? -aa : aa;
    int64_t abs_b = bb < 0 ? -bb : bb;
    int64_t q = abs_a / abs_b;
    int64_t r = abs_a % abs_b;
    if (r * 2 >= abs_b) q += 1;
    int64_t out = sign * q;
    if (out > INT32_MAX) return INT32_MAX;
    if (out < INT32_MIN) return INT32_MIN;
    return (int32_t)out;
}

/* ===== Casts (matrix forms) ===== */

matlab_mat_u8 *matlab_mat_u8_from_double(matlab_mat *A) {
    if (!A) return mat_u8_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat_u8 *R = mat_u8_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = sat_d_to_u8(A->data[k]);
    return R;
}
matlab_mat_i32 *matlab_mat_i32_from_double(matlab_mat *A) {
    if (!A) return mat_i32_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat_i32 *R = mat_i32_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = sat_d_to_i32(A->data[k]);
    return R;
}
matlab_mat *matlab_mat_u8_to_double(matlab_mat_u8 *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = (double)A->data[k];
    return R;
}
matlab_mat *matlab_mat_i32_to_double(matlab_mat_i32 *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = (double)A->data[k];
    return R;
}
matlab_mat_u8 *matlab_mat_u8_from_i32(matlab_mat_i32 *A) {
    if (!A) return mat_u8_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat_u8 *R = mat_u8_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = sat_i32_to_u8(A->data[k]);
    return R;
}
matlab_mat_i32 *matlab_mat_i32_from_u8(matlab_mat_u8 *A) {
    if (!A) return mat_i32_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat_i32 *R = mat_i32_alloc(m, n);
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = (int32_t)A->data[k];
    return R;
}

/* Scalar saturating casts — public wrappers around the static helpers
 * above. Used at MLIR lowering time when a typed-int matrix is mixed
 * with a double scalar in a binop (`A + 2.5`); the lowering coerces the
 * f64 here before calling the typed _ms / _sm runtime entry. */
extern "C" int32_t matlab_d_to_i32_sat(double v) { return sat_d_to_i32(v); }
extern "C" uint8_t matlab_d_to_u8_sat (double v) { return sat_d_to_u8(v); }

/* ===== Element-wise arithmetic ===== */

/* Macro generates _mm / _ms / _sm trio for one (lane, op) pair. The lane
 * type T and the op functor OP are concatenated into the entry-point
 * symbol name so the public ABI is matlab_mat_<lane>_<op>_(mm|ms|sm). */
#define DEF_INT_BINOP(LANE, T, ALLOC, OP, OPNAME)                              \
extern "C" matlab_mat_##LANE *matlab_mat_##LANE##_##OPNAME##_mm(               \
        matlab_mat_##LANE *A, matlab_mat_##LANE *B) {                          \
    if (!A || !B || A->rows != B->rows || A->cols != B->cols)                  \
        return ALLOC(0, 0);                                                    \
    int64_t m = A->rows, n = A->cols;                                          \
    matlab_mat_##LANE *R = ALLOC(m, n);                                        \
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = OP(A->data[k], B->data[k]); \
    return R;                                                                  \
}                                                                              \
extern "C" matlab_mat_##LANE *matlab_mat_##LANE##_##OPNAME##_ms(               \
        matlab_mat_##LANE *A, T s) {                                           \
    if (!A) return ALLOC(0, 0);                                                \
    int64_t m = A->rows, n = A->cols;                                          \
    matlab_mat_##LANE *R = ALLOC(m, n);                                        \
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = OP(A->data[k], s);        \
    return R;                                                                  \
}                                                                              \
extern "C" matlab_mat_##LANE *matlab_mat_##LANE##_##OPNAME##_sm(               \
        T s, matlab_mat_##LANE *A) {                                           \
    if (!A) return ALLOC(0, 0);                                                \
    int64_t m = A->rows, n = A->cols;                                          \
    matlab_mat_##LANE *R = ALLOC(m, n);                                        \
    for (int64_t k = 0; k < m * n; ++k) R->data[k] = OP(s, A->data[k]);        \
    return R;                                                                  \
}

DEF_INT_BINOP(u8,  uint8_t, mat_u8_alloc,  sat_add_u8,    add)
DEF_INT_BINOP(u8,  uint8_t, mat_u8_alloc,  sat_sub_u8,    sub)
DEF_INT_BINOP(u8,  uint8_t, mat_u8_alloc,  sat_mul_u8,    emul)
DEF_INT_BINOP(u8,  uint8_t, mat_u8_alloc,  round_div_u8,  ediv)

DEF_INT_BINOP(i32, int32_t, mat_i32_alloc, sat_add_i32,   add)
DEF_INT_BINOP(i32, int32_t, mat_i32_alloc, sat_sub_i32,   sub)
DEF_INT_BINOP(i32, int32_t, mat_i32_alloc, sat_mul_i32,   emul)
DEF_INT_BINOP(i32, int32_t, mat_i32_alloc, round_div_i32, ediv)

#undef DEF_INT_BINOP

/* ===== Comparisons (return matlab_mat with 0/1 doubles) ===== */

#define DEF_INT_CMP(LANE, T, OP, OPNAME)                                       \
extern "C" matlab_mat *matlab_mat_##LANE##_##OPNAME##_mm(                      \
        matlab_mat_##LANE *A, matlab_mat_##LANE *B) {                          \
    if (!A || !B || A->rows != B->rows || A->cols != B->cols)                  \
        return mat_alloc(0, 0);                                                \
    int64_t m = A->rows, n = A->cols;                                          \
    matlab_mat *R = mat_alloc(m, n);                                           \
    for (int64_t k = 0; k < m * n; ++k)                                        \
        R->data[k] = (A->data[k] OP B->data[k]) ? 1.0 : 0.0;                   \
    return R;                                                                  \
}                                                                              \
extern "C" matlab_mat *matlab_mat_##LANE##_##OPNAME##_ms(                      \
        matlab_mat_##LANE *A, T s) {                                           \
    if (!A) return mat_alloc(0, 0);                                            \
    int64_t m = A->rows, n = A->cols;                                          \
    matlab_mat *R = mat_alloc(m, n);                                           \
    for (int64_t k = 0; k < m * n; ++k)                                        \
        R->data[k] = (A->data[k] OP s) ? 1.0 : 0.0;                            \
    return R;                                                                  \
}                                                                              \
extern "C" matlab_mat *matlab_mat_##LANE##_##OPNAME##_sm(                      \
        T s, matlab_mat_##LANE *A) {                                           \
    if (!A) return mat_alloc(0, 0);                                            \
    int64_t m = A->rows, n = A->cols;                                          \
    matlab_mat *R = mat_alloc(m, n);                                           \
    for (int64_t k = 0; k < m * n; ++k)                                        \
        R->data[k] = (s OP A->data[k]) ? 1.0 : 0.0;                            \
    return R;                                                                  \
}

DEF_INT_CMP(u8,  uint8_t, >,  gt)
DEF_INT_CMP(u8,  uint8_t, >=, ge)
DEF_INT_CMP(u8,  uint8_t, <,  lt)
DEF_INT_CMP(u8,  uint8_t, <=, le)
DEF_INT_CMP(u8,  uint8_t, ==, eq)
DEF_INT_CMP(u8,  uint8_t, !=, ne)

DEF_INT_CMP(i32, int32_t, >,  gt)
DEF_INT_CMP(i32, int32_t, >=, ge)
DEF_INT_CMP(i32, int32_t, <,  lt)
DEF_INT_CMP(i32, int32_t, <=, le)
DEF_INT_CMP(i32, int32_t, ==, eq)
DEF_INT_CMP(i32, int32_t, !=, ne)

#undef DEF_INT_CMP

/* ===== Reductions =====
 * sum/min/max return same-type scalar (1x1); mean rounds the result back
 * to the lane type with saturation, matching MATLAB. For Phase 1.1.B we
 * implement the vector / matrix-as-vector form (single scalar out); the
 * column-wise variant lands when needed. */

uint8_t matlab_mat_u8_sum(matlab_mat_u8 *A) {
    if (!A) return 0;
    int64_t n = A->rows * A->cols;
    int64_t acc = 0;
    for (int64_t k = 0; k < n; ++k) {
        acc += A->data[k];
        if (acc > 255) { acc = 255; }   /* saturate during accumulation */
    }
    return (uint8_t)acc;
}
int32_t matlab_mat_i32_sum(matlab_mat_i32 *A) {
    if (!A) return 0;
    int64_t n = A->rows * A->cols;
    int64_t acc = 0;
    for (int64_t k = 0; k < n; ++k) {
        acc += A->data[k];
        if (acc > INT32_MAX) acc = INT32_MAX;
        if (acc < INT32_MIN) acc = INT32_MIN;
    }
    return (int32_t)acc;
}
uint8_t matlab_mat_u8_mean(matlab_mat_u8 *A) {
    if (!A) return 0;
    int64_t n = A->rows * A->cols;
    if (n == 0) return 0;
    double acc = 0.0;
    for (int64_t k = 0; k < n; ++k) acc += (double)A->data[k];
    return sat_d_to_u8(acc / (double)n);
}
int32_t matlab_mat_i32_mean(matlab_mat_i32 *A) {
    if (!A) return 0;
    int64_t n = A->rows * A->cols;
    if (n == 0) return 0;
    double acc = 0.0;
    for (int64_t k = 0; k < n; ++k) acc += (double)A->data[k];
    return sat_d_to_i32(acc / (double)n);
}
uint8_t matlab_mat_u8_min(matlab_mat_u8 *A) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t n = A->rows * A->cols;
    uint8_t m = A->data[0];
    for (int64_t k = 1; k < n; ++k) if (A->data[k] < m) m = A->data[k];
    return m;
}
uint8_t matlab_mat_u8_max(matlab_mat_u8 *A) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t n = A->rows * A->cols;
    uint8_t m = A->data[0];
    for (int64_t k = 1; k < n; ++k) if (A->data[k] > m) m = A->data[k];
    return m;
}
int32_t matlab_mat_i32_min(matlab_mat_i32 *A) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t n = A->rows * A->cols;
    int32_t m = A->data[0];
    for (int64_t k = 1; k < n; ++k) if (A->data[k] < m) m = A->data[k];
    return m;
}
int32_t matlab_mat_i32_max(matlab_mat_i32 *A) {
    if (!A || A->rows * A->cols == 0) return 0;
    int64_t n = A->rows * A->cols;
    int32_t m = A->data[0];
    for (int64_t k = 1; k < n; ++k) if (A->data[k] > m) m = A->data[k];
    return m;
}

/* ---------------------------------------------------------------------- */
/* Minimal 3-D arrays.
 *
 * A separate matlab_mat3 descriptor {data, rows, cols, depth} so
 * existing 2-D paths keep working unchanged. Data is laid out
 * slice-major (depth varies slowest, cols fastest) so rows+cols
 * stride within a slice like ordinary 2-D, and consecutive slices
 * live contiguously.
 *
 * Only the trio that common 3-D code actually needs is wired for v1:
 * zeros(m, n, p) / ones(m, n, p) constructors, scalar read/write
 * A(i, j, k), size(A, 3). Reductions, slicing, disp and arithmetic
 * are still 2-D-only; calling them on a 3-D array gives undefined
 * results and is documented as a follow-up. */
/* matlab_mat3 layout lives in runtime_internal.h (Phase-2 split). */

matlab_mat3 *mat3_alloc(int64_t m, int64_t n, int64_t p) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    if (p < 0) p = 0;
    matlab_mat3 *A = (matlab_mat3 *)calloc(1, sizeof(*A));
    A->magic = MATLAB_MAT3_MAGIC;
    A->rows = m; A->cols = n; A->depth = p;
    A->data = (double *)calloc((size_t)(m * n * p), sizeof(double));
    return A;
}

matlab_mat3 *matlab_zeros3(double m, double n, double p) {
    return mat3_alloc((int64_t)m, (int64_t)n, (int64_t)p);
}

matlab_mat3 *matlab_ones3(double m, double n, double p) {
    matlab_mat3 *A = mat3_alloc((int64_t)m, (int64_t)n, (int64_t)p);
    int64_t total = A->rows * A->cols * A->depth;
    for (int64_t i = 0; i < total; ++i) A->data[i] = 1.0;
    return A;
}

static int64_t mat3_offset(matlab_mat3 *A, int64_t i, int64_t j, int64_t k) {
    /* Slice-major layout: slice k occupies indices [k*rows*cols, (k+1)*rows*cols),
     * within which row-major rows*cols applies. */
    return k * A->rows * A->cols + i * A->cols + j;
}

double matlab_subscript3_s(matlab_mat3 *A, double i1, double j1, double k1) {
    if (!A) return 0.0;
    int64_t i = (int64_t)i1 - 1;
    int64_t j = (int64_t)j1 - 1;
    int64_t k = (int64_t)k1 - 1;
    if (i < 0 || i >= A->rows) return 0.0;
    if (j < 0 || j >= A->cols) return 0.0;
    if (k < 0 || k >= A->depth) return 0.0;
    return A->data[mat3_offset(A, i, j, k)];
}

void matlab_subscript3_store(matlab_mat3 *A, double i1, double j1,
                              double k1, double v) {
    if (!A) return;
    int64_t i = (int64_t)i1 - 1;
    int64_t j = (int64_t)j1 - 1;
    int64_t k = (int64_t)k1 - 1;
    if (i < 0 || i >= A->rows) return;
    if (j < 0 || j >= A->cols) return;
    if (k < 0 || k >= A->depth) return;
    A->data[mat3_offset(A, i, j, k)] = v;
}

double matlab_size3_dim(matlab_mat3 *A, double d) {
    if (!A) return 0.0;
    int64_t dim = (int64_t)d;
    if (dim == 1) return (double)A->rows;
    if (dim == 2) return (double)A->cols;
    if (dim == 3) return (double)A->depth;
    return 1.0;
}

double matlab_numel3(matlab_mat3 *A) {
    if (!A) return 0.0;
    return (double)(A->rows * A->cols * A->depth);
}

double matlab_ndims3(matlab_mat3 *A) {
    if (!A) return 0.0;
    return A->depth > 1 ? 3.0 : 2.0;
}

/* ---------------------------------------------------------------------- */
/* Minimum classdef support.
 *
 * A matlab_obj is the generic user-defined-class descriptor. Its layout
 * is deliberately ABI-compatible with matlab_struct — every field of
 * matlab_struct appears at the same offset, followed by a class_id tag
 * at the tail. This means matlab_struct_get/set routines work
 * *unchanged* when called with a matlab_obj* — handy because not every
 * method parameter that happens to carry a class instance can be
 * proven at compile time to be an obj, and we'd rather have the
 * reasonable-default path than crash on mis-dispatch. The dedicated
 * matlab_obj_* entries additionally expose the class_id.
 *
 * Methods are emitted as ordinary free functions with a name-mangled
 * form (see lowerer): `ClassName__method`. The first parameter is
 * always the object pointer. There is no virtual-dispatch table in v1
 * because inheritance and overrides are resolved statically at each
 * call site from the pinned class recorded in Sema.
 *
 * All objects are handle-shaped (reference semantics) — MATLAB value
 * classes copy-on-modify, which would require a deeper change to our
 * f64-plus-pointer data model, so they are deferred. */
/* matlab_obj_s layout lives in runtime_internal.h (Phase-2 split). */

/* Live-object registry. matlab_obj has no magic byte at offset 0
 * (its prefix MUST stay matlab_struct-compatible — see comment on
 * struct matlab_obj_s above), so matlab_disp_mat / similar polymorphic
 * entries can't discriminate an obj from a matlab_mat by reading the
 * pointer alone. Every constructor call registers the new pointer
 * here; matlab_obj_is_known() is consulted before treating an
 * incoming `void *` as a matrix. Registration is append-only — the
 * runtime is short-lived and leaks already-freed obj allocations
 * elsewhere, so a stable pointer set is fine. */
static struct {
    pthread_mutex_t mu;
    void **ptrs;
    int count;
    int cap;
} matlab_obj_registry = { PTHREAD_MUTEX_INITIALIZER, NULL, 0, 0 };

static void matlab_obj_registry_add(void *p) {
    if (!p) return;
    pthread_mutex_lock(&matlab_obj_registry.mu);
    if (matlab_obj_registry.count == matlab_obj_registry.cap) {
        int ncap = matlab_obj_registry.cap ? matlab_obj_registry.cap * 2 : 16;
        void **nptrs = (void **)realloc(matlab_obj_registry.ptrs,
                                        (size_t)ncap * sizeof(void *));
        if (nptrs) {
            matlab_obj_registry.ptrs = nptrs;
            matlab_obj_registry.cap = ncap;
        }
    }
    if (matlab_obj_registry.count < matlab_obj_registry.cap) {
        matlab_obj_registry.ptrs[matlab_obj_registry.count++] = p;
    }
    pthread_mutex_unlock(&matlab_obj_registry.mu);
}

int matlab_obj_is_known(const void *p) {
    if (!p) return 0;
    int found = 0;
    pthread_mutex_lock(&matlab_obj_registry.mu);
    for (int i = 0; i < matlab_obj_registry.count; ++i) {
        if (matlab_obj_registry.ptrs[i] == p) { found = 1; break; }
    }
    pthread_mutex_unlock(&matlab_obj_registry.mu);
    return found;
}

matlab_obj *matlab_obj_new(int32_t class_id) {
    matlab_obj *o = (matlab_obj *)calloc(1, sizeof(*o));
    o->capacity = MATLAB_STRUCT_CAP_INIT;
    o->names    = (char **)calloc((size_t)o->capacity, sizeof(char *));
    o->kinds    = (int32_t *)calloc((size_t)o->capacity, sizeof(int32_t));
    o->f64_vals = (double *)calloc((size_t)o->capacity, sizeof(double));
    o->ptr_vals = (void **)calloc((size_t)o->capacity, sizeof(void *));
    o->class_id = class_id;
    matlab_obj_registry_add(o);
    return o;
}

double matlab_obj_class_id(matlab_obj *o) {
    return o ? (double)o->class_id : 0.0;
}

/* Forward decl matching the matlab_string layout — defined later in
 * the same TU. Phase 5.2 / Phase 4 reach into the layout for fast key
 * compare without needing the public accessors. */
struct matlab_string_s_fwd_ {
    char *data;
    int64_t len;
};

/* matlab_cell layout is also defined later; phase 5.2 categorical
 * accesses ptr_vals directly to read the per-element string pointers. */
struct matlab_cell_s_fwd_ {
    int32_t n, cap, rows, cols;
    int32_t *kinds;
    double  *f64_vals;
    void   **ptr_vals;
};


/* ====================================================================== */
/* Phase 5.3 — table.
 *
 * A MATLAB table is a record of named columns where each column is a
 * homogeneous matlab_mat (column vector for v1). Columns can have
 * different element kinds in principle; the v1 lowering produces an
 * f64 column per scalar / vector argument. Auto-named columns get
 * "Var1", "Var2", ...
 *
 * The descriptor:
 *   nvars   # of columns
 *   nrows   row count (taken from the first column at construction)
 *   names   per-column variable name (strdup'd)
 *   data    per-column matlab_mat * (column vector)
 *
 * Surfaces:
 *   T = table(col1, col2, ...)
 *   T.<name>             column read  (returns matlab_mat *)
 *   T.<name> = vec       column write
 *   height(T) / width(T) / numel(T) / size(T, dim)
 *   disp(T)              MATLAB-style table display
 * ====================================================================== */

/* matlab_table_disp renders datetime cells inline, so it needs the
 * full matlab_datetime layout and the epoch-to-civil helper. The
 * struct is small (one double) — defining it here once and forward-
 * declaring the helper avoids reordering the rest of the TU. The
 * canonical definition further down was promoted to this header
 * to keep ODR happy. */
struct matlab_datetime_s { double seconds; };
typedef struct matlab_datetime_s matlab_datetime;
struct matlab_duration_s { double seconds; };
typedef struct matlab_duration_s matlab_duration;
static void epoch_to_civil(double secs, int *y, int *m, int *d,
                            int *hh, int *mm, double *ss);

/* Column kinds. v1 stored only matlab_mat * columns; readtable
 * landed string + datetime columns, so each slot now carries an
 * explicit kind tag. The data pointer interpretation is:
 *   MATLAB_TABLE_KIND_NUMERIC  → matlab_mat *           (column vector)
 *   MATLAB_TABLE_KIND_STRING   → matlab_string ** array (nrows entries)
 *   MATLAB_TABLE_KIND_DATETIME → matlab_datetime ** array (nrows entries) */
enum {
    MATLAB_TABLE_KIND_NUMERIC  = 0,
    MATLAB_TABLE_KIND_STRING   = 1,
    MATLAB_TABLE_KIND_DATETIME = 2,
};

struct matlab_table_s {
    int32_t  nvars;
    int32_t  cap;
    int32_t  nrows;
    char   **names;
    void   **data;     /* matlab_mat * for numeric; pointer-array for string/datetime */
    int8_t  *kinds;    /* one of MATLAB_TABLE_KIND_* per column */
};
typedef struct matlab_table_s matlab_table;

static void table_grow(matlab_table *t, int32_t need) {
    if (t->cap >= need) return;
    int32_t nc = t->cap ? t->cap * 2 : 4;
    while (nc < need) nc *= 2;
    t->names = (char **)realloc(t->names, (size_t)nc * sizeof(char *));
    t->data  = (void **)realloc(t->data,  (size_t)nc * sizeof(void *));
    t->kinds = (int8_t *)realloc(t->kinds, (size_t)nc * sizeof(int8_t));
    for (int32_t i = t->cap; i < nc; ++i) {
        t->names[i] = NULL;
        t->data[i]  = NULL;
        t->kinds[i] = MATLAB_TABLE_KIND_NUMERIC;
    }
    t->cap = nc;
}

extern "C" matlab_table *matlab_table_new(void) {
    return (matlab_table *)calloc(1, sizeof(matlab_table));
}

/* Find the index of a named column; -1 on miss. */
static int32_t table_find(matlab_table *t, const char *name, int64_t len) {
    if (!t) return -1;
    for (int32_t i = 0; i < t->nvars; ++i) {
        if (!t->names[i]) continue;
        if ((int64_t)strlen(t->names[i]) == len &&
            memcmp(t->names[i], name, (size_t)len) == 0) return i;
    }
    return -1;
}

/* Add or replace a column. The runtime takes ownership of the
 * matlab_mat *; the caller must not free it. Slot is tagged as
 * NUMERIC; string / datetime columns go through
 * matlab_table_add_column_kind. */
extern "C" void matlab_table_add_column(matlab_table *t,
                                         const char *name, int64_t namelen,
                                         matlab_mat *col) {
    if (!t) return;
    int32_t i = table_find(t, name, namelen);
    if (i < 0) {
        if (t->nvars == t->cap) table_grow(t, t->nvars + 1);
        i = t->nvars++;
        t->names[i] = (char *)malloc((size_t)namelen + 1);
        memcpy(t->names[i], name, (size_t)namelen);
        t->names[i][namelen] = '\0';
        if (col && t->nrows == 0) {
            int64_t r = col->rows * col->cols;
            t->nrows = (int32_t)r;
        }
    }
    t->data[i]  = col;
    t->kinds[i] = MATLAB_TABLE_KIND_NUMERIC;
}

/* Add or replace a column with an explicit kind. For STRING /
 * DATETIME, `col` is a pointer-array of length `nrows` whose
 * element type matches the kind. The table assumes ownership of
 * the array (and its element pointers) and frees nothing — the
 * pointers come from the runtime registries that already track
 * them. `nrows` is taken from the first column added; subsequent
 * columns are silently truncated/extended only conceptually
 * (disp walks min(nrows, col_len) when applicable). */
extern "C" void matlab_table_add_column_kind(matlab_table *t,
                                              const char *name,
                                              int64_t namelen,
                                              void *col, int32_t kind,
                                              int64_t nrows_hint) {
    if (!t) return;
    int32_t i = table_find(t, name, namelen);
    if (i < 0) {
        if (t->nvars == t->cap) table_grow(t, t->nvars + 1);
        i = t->nvars++;
        t->names[i] = (char *)malloc((size_t)namelen + 1);
        memcpy(t->names[i], name, (size_t)namelen);
        t->names[i][namelen] = '\0';
        if (t->nrows == 0 && nrows_hint > 0)
            t->nrows = (int32_t)nrows_hint;
    }
    t->data[i]  = col;
    t->kinds[i] = (int8_t)kind;
}

/* Numeric column read; returns an empty matrix on miss or if the
 * column has a non-numeric kind. Callers reading non-numeric
 * columns should consult matlab_table_get_kind first. */
extern "C" matlab_mat *matlab_table_get_column(matlab_table *t,
                                                const char *name,
                                                int64_t namelen) {
    int32_t i = table_find(t, name, namelen);
    if (i < 0 || !t->data[i]) return mat_alloc(0, 0);
    if (t->kinds && t->kinds[i] != MATLAB_TABLE_KIND_NUMERIC)
        return mat_alloc(0, 0);
    return (matlab_mat *)t->data[i];
}

/* Returns the column kind (0=numeric, 1=string, 2=datetime) or
 * -1 on miss. Used by future column-read paths that need to
 * dispatch on element type. */
extern "C" double matlab_table_get_kind(matlab_table *t,
                                         const char *name,
                                         int64_t namelen) {
    int32_t i = table_find(t, name, namelen);
    if (i < 0) return -1.0;
    return t->kinds ? (double)t->kinds[i] : 0.0;
}

extern "C" double matlab_table_height(matlab_table *t) {
    return t ? (double)t->nrows : 0.0;
}
extern "C" double matlab_table_width(matlab_table *t) {
    return t ? (double)t->nvars : 0.0;
}
extern "C" double matlab_table_numel(matlab_table *t) {
    return t ? (double)(t->nrows * t->nvars) : 0.0;
}
extern "C" double matlab_table_size_dim(matlab_table *t, double dim) {
    if (!t) return 0.0;
    int d = (int)dim;
    if (d == 1) return (double)t->nrows;
    if (d == 2) return (double)t->nvars;
    return 1.0;
}

/* Iterate columns by index — used by the DAP `variables` drill-in
 * so it can walk a table's columns without learning the matlab_table_s
 * layout. Out-of-range idx returns NULL / -1. The pointer returned by
 * matlab_table_column_data is the raw `data[i]` slot whose
 * interpretation depends on the kind:
 *   NUMERIC  -> matlab_mat *
 *   STRING   -> matlab_string ** (nrows entries)
 *   DATETIME -> matlab_datetime ** (nrows entries) */
extern "C" const char *matlab_table_column_name(matlab_table *t,
                                                 int32_t idx,
                                                 int64_t *out_len) {
    if (out_len) *out_len = 0;
    if (!t || idx < 0 || idx >= t->nvars) return NULL;
    const char *n = t->names[idx];
    if (out_len && n) *out_len = (int64_t)strlen(n);
    return n;
}
extern "C" void *matlab_table_column_data(matlab_table *t, int32_t idx) {
    if (!t || idx < 0 || idx >= t->nvars) return NULL;
    return t->data[idx];
}
extern "C" int32_t matlab_table_column_kind_idx(matlab_table *t, int32_t idx) {
    if (!t || idx < 0 || idx >= t->nvars) return -1;
    return t->kinds ? (int32_t)t->kinds[idx] : 0;
}

extern "C" void matlab_table_disp(matlab_table *t) {
    if (!t) { matlab_disp_str("(empty table)", 13); return; }
    pthread_mutex_lock(&matlab_io_mutex);
    /* Header row: column names with two-space separator. Use a fixed
     * column width so the body lines up. */
    const int W = 12;
    /* Print header. */
    for (int32_t i = 0; i < t->nvars; ++i)
        printf("    %*s", W, t->names[i] ? t->names[i] : "");
    putchar('\n');
    /* Underline. */
    for (int32_t i = 0; i < t->nvars; ++i) {
        printf("    ");
        for (int j = 0; j < W; ++j) putchar('_');
    }
    putchar('\n');
    /* Body — each row, one element per column. Dispatch on the
     * column kind so string/datetime columns render their text
     * form instead of being misread as a matlab_mat. */
    static const char *months[] = {"Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"};
    for (int32_t r = 0; r < t->nrows; ++r) {
        for (int32_t c = 0; c < t->nvars; ++c) {
            int kind = t->kinds ? (int)t->kinds[c] : 0;
            if (kind == MATLAB_TABLE_KIND_STRING) {
                matlab_string_s_fwd_ **arr =
                    (matlab_string_s_fwd_ **)t->data[c];
                matlab_string_s_fwd_ *s = arr ? arr[r] : NULL;
                if (s && s->data) {
                    /* Right-justify within width W, truncate if longer. */
                    int len = (int)s->len;
                    if (len > W) len = W;
                    int pad = W - len;
                    printf("    ");
                    for (int p = 0; p < pad; ++p) putchar(' ');
                    fwrite(s->data, 1, (size_t)len, stdout);
                } else {
                    printf("    %*s", W, "");
                }
            } else if (kind == MATLAB_TABLE_KIND_DATETIME) {
                matlab_datetime **arr = (matlab_datetime **)t->data[c];
                matlab_datetime *d = arr ? arr[r] : NULL;
                if (d) {
                    int y, m, dd, hh, mm; double ss;
                    epoch_to_civil(d->seconds, &y, &m, &dd, &hh, &mm, &ss);
                    int mi = (m - 1) % 12; if (mi < 0) mi += 12;
                    char buf[32];
                    int n = snprintf(buf, sizeof buf,
                                      "%02d-%s-%04d", dd, months[mi], y);
                    printf("    %*.*s", W, n, buf);
                } else {
                    printf("    %*s", W, "");
                }
            } else {
                matlab_mat *col = (matlab_mat *)t->data[c];
                if (col && r < col->rows * col->cols) {
                    double v = col->data[r];
                    if (v == (double)(int64_t)v && fabs(v) < 1e15)
                        printf("    %*lld", W, (long long)v);
                    else
                        printf("    %*.*g", W, 6, v);
                } else {
                    printf("    %*s", W, "");
                }
            }
        }
        putchar('\n');
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* ====================================================================== */
/* readtable / readmatrix — CSV / delimited-text readers.
 *
 * Both functions accept a path (matlab_string *) and stream the
 * file with the standard C I/O so they share semantics with
 * matlab_fopen. The delimiter is auto-detected from the first
 * non-empty line by tallying ',', '\t', ';', '|' (in that order
 * of preference on ties) — covers .csv, .tsv, and the most
 * common ; / | dialects without any user input.
 *
 * Header detection: if every cell in the first row parses as a
 * finite number, no header is assumed (auto-named Var1..VarN);
 * otherwise the first row supplies the column names. This
 * matches MATLAB's default heuristic for files where the column
 * labels are textual and the data is numeric.
 *
 * Per-column type inference (readtable only):
 *   1. all cells parse as numeric           → NUMERIC column
 *   2. all cells match a datetime pattern   → DATETIME column
 *   3. otherwise                            → STRING column
 *
 * v1 limitations (intentional, follow-ups in the roadmap):
 *   - no quote-aware tokenizer: a literal delimiter inside a
 *     '"..."' field is split. CSV with embedded delimiters needs
 *     a separate parser pass.
 *   - no DateLocale / decimal-separator options: '.' decimal,
 *     ASCII whitespace only.
 *   - readmatrix returns NaN for cells that fail strtod (matches
 *     MATLAB's default behaviour for mixed-text input).
 * ====================================================================== */

/* Forward decls for runtime helpers used below — they live further
 * down in the TU. Linkage matches the original definitions:
 * matlab_datetime_* are extern "C", matlab_string_from_literal is
 * not (matches its public-header decl). */
struct matlab_string_s *matlab_string_from_literal(const char *src, int64_t n);
extern "C" matlab_datetime *matlab_datetime_ymd(double y, double m, double d);
extern "C" matlab_datetime *matlab_datetime_ymdhms(double y, double m, double d,
                                                    double h, double mn,
                                                    double s);

/* Trim leading/trailing ASCII whitespace in-place by adjusting
 * the (start, len) view; the underlying buffer is untouched. */
static void csv_trim(const char *s, int64_t len,
                      const char **out, int64_t *outlen) {
    int64_t a = 0, b = len;
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' ||
                     s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' ||
                     s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    *out = s + a; *outlen = b - a;
}

/* Strict numeric parse: returns true and stores the value iff the
 * entire trimmed token is consumed by strtod. Empty tokens count
 * as "not numeric" so they bias header detection toward "yes,
 * this is a header". */
static bool csv_parse_double(const char *s, int64_t n, double *out) {
    if (n <= 0) return false;
    char buf[64];
    if (n >= (int64_t)sizeof buf) return false;
    memcpy(buf, s, (size_t)n);
    buf[n] = '\0';
    char *end = NULL;
    double v = strtod(buf, &end);
    if (!end || end == buf) return false;
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return false;
    *out = v;
    return true;
}

/* Datetime parser. Recognises (in order):
 *   YYYY-MM-DD                   (10 chars)
 *   YYYY/MM/DD                   (10 chars)
 *   YYYY-MM-DD[ T]HH:MM[:SS]     (16 / 19 chars)
 *   YYYY/MM/DD[ T]HH:MM[:SS]
 *   DD-Mon-YYYY                  (e.g. 01-Jan-2024)
 * Returns true on match and writes the y/m/d/hh/mm/ss components.
 * Year 0 and seconds == 0 are unset components. */
static const char *csv_month_lookup[] = {
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec"
};

static bool csv_parse_datetime(const char *s, int64_t n,
                                int *y, int *m, int *d,
                                int *hh, int *mm, double *ss) {
    *y = *m = *d = *hh = *mm = 0; *ss = 0.0;
    if (n < 8) return false;
    /* DD-Mon-YYYY (length 11). */
    if (n == 11 && s[2] == '-' && s[6] == '-') {
        int dd = (s[0] - '0') * 10 + (s[1] - '0');
        if (s[0] < '0' || s[0] > '9' || s[1] < '0' || s[1] > '9')
            return false;
        char mon[4] = {s[3], s[4], s[5], 0};
        int mi = -1;
        for (int i = 0; i < 12; ++i) {
            if (mon[0] == csv_month_lookup[i][0] &&
                mon[1] == csv_month_lookup[i][1] &&
                mon[2] == csv_month_lookup[i][2]) { mi = i; break; }
        }
        if (mi < 0) return false;
        int yy = 0;
        for (int i = 7; i < 11; ++i) {
            if (s[i] < '0' || s[i] > '9') return false;
            yy = yy * 10 + (s[i] - '0');
        }
        *d = dd; *m = mi + 1; *y = yy;
        return true;
    }
    /* YYYY[-/]MM[-/]DD ... — head 10 chars must be ISO-shaped. */
    if (n < 10) return false;
    char d1 = s[4], d2 = s[7];
    if (!((d1 == '-' && d2 == '-') || (d1 == '/' && d2 == '/')))
        return false;
    int yy = 0;
    for (int i = 0; i < 4; ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
        yy = yy * 10 + (s[i] - '0');
    }
    int mo = (s[5] - '0') * 10 + (s[6] - '0');
    int dd = (s[8] - '0') * 10 + (s[9] - '0');
    if (s[5] < '0' || s[5] > '9' || s[6] < '0' || s[6] > '9' ||
        s[8] < '0' || s[8] > '9' || s[9] < '0' || s[9] > '9')
        return false;
    if (mo < 1 || mo > 12 || dd < 1 || dd > 31) return false;
    *y = yy; *m = mo; *d = dd;
    if (n == 10) return true;
    /* Optional time tail: [ T]HH:MM(:SS)?  */
    if (s[10] != ' ' && s[10] != 'T') return false;
    if (n < 16) return false;
    int H = (s[11] - '0') * 10 + (s[12] - '0');
    int M = (s[14] - '0') * 10 + (s[15] - '0');
    if (s[13] != ':') return false;
    if (s[11] < '0' || s[11] > '9' || s[12] < '0' || s[12] > '9' ||
        s[14] < '0' || s[14] > '9' || s[15] < '0' || s[15] > '9')
        return false;
    *hh = H; *mm = M;
    if (n == 16) return true;
    if (n == 19 && s[16] == ':') {
        int S = (s[17] - '0') * 10 + (s[18] - '0');
        if (s[17] < '0' || s[17] > '9' || s[18] < '0' || s[18] > '9')
            return false;
        *ss = (double)S;
        return true;
    }
    return false;
}

/* Pick the delimiter from the first non-empty line by counting
 * candidates. Returns ',' on a hard tie / no candidates so a
 * one-column CSV still degrades gracefully. */
static char csv_detect_delim(const char *buf, int64_t len) {
    int64_t i = 0;
    while (i < len && (buf[i] == ' ' || buf[i] == '\r' || buf[i] == '\n'))
        ++i;
    int64_t e = i;
    while (e < len && buf[e] != '\n') ++e;
    int counts[4] = {0, 0, 0, 0}; /* , \t ; | */
    for (int64_t k = i; k < e; ++k) {
        switch (buf[k]) {
            case ',':  counts[0]++; break;
            case '\t': counts[1]++; break;
            case ';':  counts[2]++; break;
            case '|':  counts[3]++; break;
        }
    }
    int best = 0;
    for (int k = 1; k < 4; ++k) if (counts[k] > counts[best]) best = k;
    if (counts[best] == 0) return ',';
    static const char delims[4] = {',', '\t', ';', '|'};
    return delims[best];
}

/* Split a buffer into a 2-D table of (row, col) → (start, len).
 * Returns the malloc'd contiguous arrays of starts/lens — the
 * caller frees both plus row_offs. row_offs[r] gives the index
 * in starts/lens where row r begins; row_offs[nrows] is the
 * total cell count. ncols_out is the column count of the longest
 * row. Shorter rows are padded with empty cells. */
static void csv_tokenize(const char *buf, int64_t len, char delim,
                          int64_t **row_offs_out, int64_t *nrows_out,
                          int64_t *ncols_out,
                          const char ***starts_out, int64_t **lens_out) {
    int64_t cap_cells = 64, ncells = 0;
    const char **starts = (const char **)malloc((size_t)cap_cells * sizeof(*starts));
    int64_t *lens = (int64_t *)malloc((size_t)cap_cells * sizeof(*lens));
    int64_t cap_rows = 16, nrows = 0;
    int64_t *row_offs = (int64_t *)malloc((size_t)(cap_rows + 1) *
                                            sizeof(*row_offs));
    row_offs[0] = 0;
    int64_t i = 0;
    while (i < len) {
        /* Skip blank line. */
        int64_t line_start = i;
        while (i < len && buf[i] != '\n') ++i;
        int64_t line_end = i;
        if (i < len) ++i;
        /* Trim CR. */
        if (line_end > line_start && buf[line_end - 1] == '\r') --line_end;
        if (line_end == line_start) continue;
        /* Tokenize this line. */
        int64_t k = line_start;
        while (k <= line_end) {
            int64_t cs = k;
            while (k < line_end && buf[k] != delim) ++k;
            const char *cell; int64_t clen;
            csv_trim(buf + cs, k - cs, &cell, &clen);
            if (ncells == cap_cells) {
                cap_cells *= 2;
                starts = (const char **)realloc(starts,
                                                 (size_t)cap_cells * sizeof(*starts));
                lens   = (int64_t *)realloc(lens,
                                             (size_t)cap_cells * sizeof(*lens));
            }
            starts[ncells] = cell;
            lens[ncells]   = clen;
            ncells++;
            if (k >= line_end) break;
            ++k; /* step past delim */
            if (k == line_end) {
                /* Trailing delim → empty trailing cell. */
                if (ncells == cap_cells) {
                    cap_cells *= 2;
                    starts = (const char **)realloc(starts,
                                                     (size_t)cap_cells * sizeof(*starts));
                    lens   = (int64_t *)realloc(lens,
                                                 (size_t)cap_cells * sizeof(*lens));
                }
                starts[ncells] = buf + k;
                lens[ncells]   = 0;
                ncells++;
                break;
            }
        }
        if (nrows + 1 > cap_rows) {
            cap_rows *= 2;
            row_offs = (int64_t *)realloc(row_offs,
                                           (size_t)(cap_rows + 1) *
                                               sizeof(*row_offs));
        }
        nrows++;
        row_offs[nrows] = ncells;
    }
    /* Compute longest row. */
    int64_t ncols = 0;
    for (int64_t r = 0; r < nrows; ++r) {
        int64_t w = row_offs[r + 1] - row_offs[r];
        if (w > ncols) ncols = w;
    }
    *row_offs_out = row_offs;
    *nrows_out = nrows;
    *ncols_out = ncols;
    *starts_out = starts;
    *lens_out = lens;
}

/* Read an entire file into a heap buffer. Returns NULL on miss
 * and writes the size to *len_out. */
static char *csv_slurp(const char *path, int64_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0) sz = 0;
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t got = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[got] = '\0';
    *len_out = (int64_t)got;
    return buf;
}

/* Strip a UTF-8 BOM from the start of buf if present. Adjusts
 * (buf, len) — the original allocation root is unchanged. */
static void csv_strip_bom(const char **buf, int64_t *len) {
    if (*len >= 3 && (unsigned char)(*buf)[0] == 0xEF &&
        (unsigned char)(*buf)[1] == 0xBB && (unsigned char)(*buf)[2] == 0xBF) {
        *buf += 3; *len -= 3;
    }
}

/* Note: the public header types `path` as matlab_string *, but
 * matlab_string_s isn't fully defined this early in the TU. The
 * fwd-declared layout-equivalent matlab_string_s_fwd_ has the
 * same {data, len} fields; the linker symbol name is what
 * matters for cross-module calls. */
extern "C" matlab_table *matlab_readtable(matlab_string_s_fwd_ *path) {
    if (!path || !path->data) {
        static const char msg[] = "readtable: empty path";
        matlab_set_error_msg(msg, (int64_t)(sizeof msg - 1));
        return matlab_table_new();
    }
    /* fopen needs a NUL-terminated path; matlab_string already
     * stores one (we guarantee it elsewhere) but copy defensively. */
    char *p = (char *)malloc((size_t)path->len + 1);
    memcpy(p, path->data, (size_t)path->len);
    p[path->len] = '\0';
    int64_t len = 0;
    char *raw = csv_slurp(p, &len);
    if (!raw) {
        /* Don't silently return an empty table on open failure — that
         * masks relative-path mistakes (e.g. cwd not set under the DAP
         * launch path). Surface a clear error including the bad path
         * via both the error flag (caught by try/catch and by the DAP
         * channel) and a direct stderr write (visible in the standalone
         * binary path, where the traceback emitter is gated off). */
        char msg[1024];
        int n = snprintf(msg, sizeof msg,
                         "readtable: cannot open file '%s'", p);
        free(p);
        if (n < 0) n = 0;
        if (n > (int)sizeof msg - 1) n = (int)sizeof msg - 1;
        fprintf(stderr, "error: %.*s\n", (int)n, msg);
        matlab_set_error_msg(msg, (int64_t)n);
        return matlab_table_new();
    }
    free(p);
    const char *buf = raw;
    csv_strip_bom(&buf, &len);
    char delim = csv_detect_delim(buf, len);
    int64_t *row_offs = NULL; int64_t nrows = 0, ncols = 0;
    const char **starts = NULL; int64_t *lens = NULL;
    csv_tokenize(buf, len, delim, &row_offs, &nrows, &ncols,
                  &starts, &lens);
    matlab_table *t = matlab_table_new();
    if (nrows == 0 || ncols == 0) {
        free(raw); free(row_offs); free(starts); free(lens);
        return t;
    }
    /* Header detection: if any cell in row 0 fails numeric
     * parse, treat row 0 as a header. */
    bool has_header = false;
    int64_t r0_cells = row_offs[1] - row_offs[0];
    for (int64_t c = 0; c < r0_cells; ++c) {
        int64_t ix = row_offs[0] + c;
        double v;
        if (!csv_parse_double(starts[ix], lens[ix], &v)) {
            has_header = true; break;
        }
    }
    int64_t header_row = has_header ? 0 : -1;
    int64_t data_start = has_header ? 1 : 0;
    int64_t data_rows = nrows - data_start;

    /* Build columns. For each column index c in [0, ncols):
     *   - resolve a name (header cell or "VarN")
     *   - probe every data cell to decide kind
     *   - allocate the appropriate column storage and populate */
    char namebuf[64];
    for (int64_t c = 0; c < ncols; ++c) {
        const char *name_s = NULL; int64_t name_n = 0;
        if (header_row == 0 && c < (row_offs[1] - row_offs[0])) {
            int64_t ix = row_offs[0] + c;
            name_s = starts[ix]; name_n = lens[ix];
        }
        if (name_n == 0) {
            int n = snprintf(namebuf, sizeof namebuf, "Var%lld",
                             (long long)(c + 1));
            name_s = namebuf; name_n = n;
        }
        /* Probe column type. all_num → numeric; else if all dates
         * (and not all numeric) → datetime; else string. Empty
         * cells block numeric/datetime so an empty column lands
         * as STRING — least lossy. */
        bool all_num = true, all_date = true;
        int64_t nonempty = 0;
        for (int64_t r = 0; r < data_rows; ++r) {
            int64_t row = data_start + r;
            int64_t row_w = row_offs[row + 1] - row_offs[row];
            const char *cs = NULL; int64_t cl = 0;
            if (c < row_w) {
                int64_t ix = row_offs[row] + c;
                cs = starts[ix]; cl = lens[ix];
            }
            if (cl == 0) { all_num = false; all_date = false; continue; }
            ++nonempty;
            double v;
            if (!csv_parse_double(cs, cl, &v)) all_num = false;
            int yy, mm, dd, hh, mi; double ss;
            if (!csv_parse_datetime(cs, cl, &yy, &mm, &dd, &hh, &mi, &ss))
                all_date = false;
            if (!all_num && !all_date) break;
        }
        if (nonempty == 0) { all_num = false; all_date = false; }
        int kind = all_num ? MATLAB_TABLE_KIND_NUMERIC :
                    (all_date ? MATLAB_TABLE_KIND_DATETIME :
                                MATLAB_TABLE_KIND_STRING);

        if (kind == MATLAB_TABLE_KIND_NUMERIC) {
            matlab_mat *col = mat_alloc(data_rows, 1);
            for (int64_t r = 0; r < data_rows; ++r) {
                int64_t row = data_start + r;
                int64_t row_w = row_offs[row + 1] - row_offs[row];
                double v = std::nan("");
                if (c < row_w) {
                    int64_t ix = row_offs[row] + c;
                    csv_parse_double(starts[ix], lens[ix], &v);
                }
                col->data[r] = v;
            }
            matlab_table_add_column_kind(t, name_s, name_n, col,
                                          MATLAB_TABLE_KIND_NUMERIC,
                                          data_rows);
        } else if (kind == MATLAB_TABLE_KIND_DATETIME) {
            matlab_datetime **col =
                (matlab_datetime **)calloc((size_t)data_rows,
                                            sizeof(*col));
            for (int64_t r = 0; r < data_rows; ++r) {
                int64_t row = data_start + r;
                int64_t row_w = row_offs[row + 1] - row_offs[row];
                if (c < row_w) {
                    int64_t ix = row_offs[row] + c;
                    int yy, mm, dd, hh, mi; double ss;
                    if (csv_parse_datetime(starts[ix], lens[ix],
                                            &yy, &mm, &dd, &hh, &mi, &ss)) {
                        col[r] = (hh || mi || ss != 0.0)
                            ? matlab_datetime_ymdhms(yy, mm, dd,
                                                       hh, mi, ss)
                            : matlab_datetime_ymd(yy, mm, dd);
                    }
                }
            }
            matlab_table_add_column_kind(t, name_s, name_n, col,
                                          MATLAB_TABLE_KIND_DATETIME,
                                          data_rows);
        } else {
            matlab_string_s_fwd_ **col =
                (matlab_string_s_fwd_ **)calloc((size_t)data_rows,
                                                  sizeof(*col));
            for (int64_t r = 0; r < data_rows; ++r) {
                int64_t row = data_start + r;
                int64_t row_w = row_offs[row + 1] - row_offs[row];
                const char *cs = NULL; int64_t cl = 0;
                if (c < row_w) {
                    int64_t ix = row_offs[row] + c;
                    cs = starts[ix]; cl = lens[ix];
                }
                col[r] = (matlab_string_s_fwd_ *)
                    matlab_string_from_literal(cs ? cs : "", cl);
            }
            matlab_table_add_column_kind(t, name_s, name_n, col,
                                          MATLAB_TABLE_KIND_STRING,
                                          data_rows);
        }
    }
    free(raw); free(row_offs); free(starts); free(lens);
    return t;
}

extern "C" matlab_mat *matlab_readmatrix(matlab_string_s_fwd_ *path) {
    if (!path || !path->data) {
        static const char msg[] = "readmatrix: empty path";
        matlab_set_error_msg(msg, (int64_t)(sizeof msg - 1));
        return mat_alloc(0, 0);
    }
    char *p = (char *)malloc((size_t)path->len + 1);
    memcpy(p, path->data, (size_t)path->len);
    p[path->len] = '\0';
    int64_t len = 0;
    char *raw = csv_slurp(p, &len);
    if (!raw) {
        char msg[1024];
        int n = snprintf(msg, sizeof msg,
                         "readmatrix: cannot open file '%s'", p);
        free(p);
        if (n < 0) n = 0;
        if (n > (int)sizeof msg - 1) n = (int)sizeof msg - 1;
        fprintf(stderr, "error: %.*s\n", (int)n, msg);
        matlab_set_error_msg(msg, (int64_t)n);
        return mat_alloc(0, 0);
    }
    free(p);
    const char *buf = raw;
    csv_strip_bom(&buf, &len);
    char delim = csv_detect_delim(buf, len);
    int64_t *row_offs = NULL; int64_t nrows = 0, ncols = 0;
    const char **starts = NULL; int64_t *lens = NULL;
    csv_tokenize(buf, len, delim, &row_offs, &nrows, &ncols,
                  &starts, &lens);
    if (nrows == 0 || ncols == 0) {
        free(raw); free(row_offs); free(starts); free(lens);
        return mat_alloc(0, 0);
    }
    /* Header detection: same heuristic as readtable. */
    bool has_header = false;
    int64_t r0_cells = row_offs[1] - row_offs[0];
    for (int64_t c = 0; c < r0_cells; ++c) {
        int64_t ix = row_offs[0] + c;
        double v;
        if (!csv_parse_double(starts[ix], lens[ix], &v)) {
            has_header = true; break;
        }
    }
    int64_t data_start = has_header ? 1 : 0;
    int64_t data_rows = nrows - data_start;
    /* This runtime stores matrices row-major (matches the
     * matlab_mat_from_buf / matlab_disp_mat_f64 convention). */
    matlab_mat *M = mat_alloc(data_rows, ncols);
    for (int64_t r = 0; r < data_rows; ++r) {
        int64_t row = data_start + r;
        int64_t row_w = row_offs[row + 1] - row_offs[row];
        for (int64_t c = 0; c < ncols; ++c) {
            double v = std::nan("");
            if (c < row_w) {
                int64_t ix = row_offs[row] + c;
                csv_parse_double(starts[ix], lens[ix], &v);
            }
            M->data[r * ncols + c] = v;
        }
    }
    free(raw); free(row_offs); free(starts); free(lens);
    return M;
}

/* ====================================================================== */
/* Phase 5.2 — categorical arrays.
 *
 * matlab_categorical wraps a 1-D vector of int32 category codes (1-based,
 * 0 = <undefined>) and a separate vector of category-name pointers
 * (matlab_string *). Categories are deduplicated on construction and
 * stored in alphabetical order — matches MATLAB's default behaviour
 * for categorical(strvec) without an explicit valueset.
 *
 * Indices and category names share the same descriptor; copies on
 * assignment are shallow (the lowering may add a clone helper if
 * value-semantics need is later). Display: each element prints on
 * its own line with the category name, "<undefined>" for code=0.
 * ====================================================================== */

struct matlab_categorical_s {
    int32_t   n;          /* number of elements */
    int32_t   ncat;       /* number of categories */
    int32_t   cap;
    int32_t   ccap;
    int32_t  *codes;      /* per-element category code (1-based) */
    void    **cats;       /* per-category matlab_string * */
};
typedef struct matlab_categorical_s matlab_categorical;

static int categorical_find_cat(matlab_categorical *c,
                                 const char *s, int64_t len) {
    for (int i = 0; i < c->ncat; ++i) {
        auto *ks = (struct matlab_string_s_fwd_ *)c->cats[i];
        if (!ks) continue;
        if (ks->len == len && ks->data &&
            memcmp(ks->data, s, (size_t)len) == 0) return i;
    }
    return -1;
}

static int categorical_add_cat(matlab_categorical *c, void *str) {
    auto *ks = (struct matlab_string_s_fwd_ *)str;
    int idx = categorical_find_cat(c, ks ? ks->data : "", ks ? ks->len : 0);
    if (idx >= 0) return idx;
    if (c->ncat == c->ccap) {
        int nc = c->ccap ? c->ccap * 2 : 4;
        c->cats = (void **)realloc(c->cats, (size_t)nc * sizeof(void *));
        c->ccap = nc;
    }
    c->cats[c->ncat] = str;
    return c->ncat++;
}

/* Build a categorical from a cell of matlab_string * pointers (the
 * lowering builds the cell up front so the variable-arity path is
 * representable as a single call). */
extern "C" matlab_categorical *matlab_categorical_from_cell(
        struct matlab_cell_s_fwd_ *cell, double n_strs);

extern "C" matlab_categorical *matlab_categorical_from_strs(
        void **strs, int64_t n_strs) {
    matlab_categorical *c = (matlab_categorical *)calloc(1, sizeof(*c));
    c->n = (int32_t)n_strs;
    c->cap = c->n > 0 ? c->n : 1;
    c->codes = (int32_t *)calloc((size_t)c->cap, sizeof(int32_t));
    /* Insert each unique string. We then sort categories alphabetically
     * and remap codes — matches MATLAB's default sort. */
    for (int32_t i = 0; i < c->n; ++i) {
        int32_t code = (int32_t)categorical_add_cat(c, strs[i]) + 1;
        c->codes[i] = code;
    }
    /* Sort categories + remap codes. */
    int32_t *order = (int32_t *)calloc((size_t)c->ncat, sizeof(int32_t));
    for (int32_t i = 0; i < c->ncat; ++i) order[i] = i;
    /* Insertion sort on c->cats by string compare (small N). */
    for (int32_t i = 1; i < c->ncat; ++i) {
        for (int32_t j = i; j > 0; --j) {
            auto *aa = (struct matlab_string_s_fwd_ *)c->cats[order[j]];
            auto *bb = (struct matlab_string_s_fwd_ *)c->cats[order[j-1]];
            int cmp = strcmp(aa ? aa->data : "", bb ? bb->data : "");
            if (cmp < 0) {
                int32_t t = order[j]; order[j] = order[j-1]; order[j-1] = t;
            } else break;
        }
    }
    /* Build the inverse map: rank[old] = new (0-based). */
    int32_t *rank = (int32_t *)calloc((size_t)c->ncat, sizeof(int32_t));
    void   **newcats = (void **)calloc((size_t)c->ncat, sizeof(void *));
    for (int32_t i = 0; i < c->ncat; ++i) {
        rank[order[i]] = i;
        newcats[i] = c->cats[order[i]];
    }
    free(c->cats); c->cats = newcats;
    for (int32_t i = 0; i < c->n; ++i) {
        if (c->codes[i] >= 1)
            c->codes[i] = rank[c->codes[i] - 1] + 1;
    }
    free(rank);
    free(order);
    return c;
}

extern "C" matlab_categorical *matlab_categorical_from_cell(
        struct matlab_cell_s_fwd_ *cell, double n_strs) {
    int n = (int)n_strs;
    if (n < 0) n = 0;
    void **strs = (void **)calloc((size_t)(n > 0 ? n : 1), sizeof(void *));
    for (int i = 0; i < n; ++i) {
        /* Reach into the cell directly — we own it (it was built
         * inline by the lowering for this call site). */
        strs[i] = cell ? cell->ptr_vals[i] : NULL;
    }
    matlab_categorical *r = matlab_categorical_from_strs(strs, n);
    free(strs);
    return r;
}

extern "C" double matlab_categorical_length(matlab_categorical *c) {
    return c ? (double)c->n : 0.0;
}

extern "C" double matlab_categorical_numcats(matlab_categorical *c) {
    return c ? (double)c->ncat : 0.0;
}

extern "C" double matlab_categorical_iscategory(
        matlab_categorical *c, void *key) {
    if (!c || !key) return 0.0;
    auto *ks = (struct matlab_string_s_fwd_ *)key;
    return categorical_find_cat(c, ks ? ks->data : "",
                                  ks ? ks->len : 0) >= 0 ? 1.0 : 0.0;
}

extern "C" struct matlab_cell_s_fwd_ *matlab_categorical_categories(
        matlab_categorical *c) {
    /* Build a small matlab_cell-shaped record by hand using the
     * forward-declared layout — the cell helpers themselves live
     * later in this TU and aren't visible at this point. The
     * downstream cell accessors only read ptr_vals[k] / kinds[k]
     * via the same layout, so this is ABI-correct. */
    struct matlab_cell_s_fwd_ *cell =
        (struct matlab_cell_s_fwd_ *)calloc(1, sizeof(*cell));
    int32_t n = c ? c->ncat : 0;
    cell->n = n;
    cell->cap = n > 0 ? n : 1;
    cell->rows = 1;
    cell->cols = n;
    cell->kinds    = (int32_t *)calloc((size_t)cell->cap, sizeof(int32_t));
    cell->f64_vals = (double  *)calloc((size_t)cell->cap, sizeof(double));
    cell->ptr_vals = (void   **)calloc((size_t)cell->cap, sizeof(void *));
    for (int32_t i = 0; i < n; ++i) {
        cell->kinds[i] = 1;            /* matrix-pointer kind */
        cell->ptr_vals[i] = c->cats[i];
    }
    return cell;
}

extern "C" void matlab_categorical_disp(matlab_categorical *c) {
    if (!c) { matlab_disp_str("(empty categorical)", 19); return; }
    pthread_mutex_lock(&matlab_io_mutex);
    if (c->n == 0) {
        printf("     [0x0 categorical]\n");
    } else {
        for (int32_t i = 0; i < c->n; ++i) {
            const char *name = "<undefined>";
            int64_t len = 11;
            if (c->codes[i] >= 1 && c->codes[i] <= c->ncat) {
                auto *ks = (struct matlab_string_s_fwd_ *)c->cats[c->codes[i] - 1];
                if (ks && ks->data) { name = ks->data; len = ks->len; }
            }
            printf("     %.*s\n", (int)len, name);
        }
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* Compare two categoricals element-wise; returns a matlab_mat with
 * 0/1 logical values. The compare uses category index after a
 * cross-walk: an element is equal iff both sides have the same
 * category name (so categoricals built with disjoint label spaces
 * still compare correctly). */
extern "C" matlab_mat *matlab_categorical_eq(
        matlab_categorical *a, matlab_categorical *b) {
    int n = a && b ? (a->n < b->n ? a->n : b->n) : 0;
    matlab_mat *r = mat_alloc(1, n);
    for (int i = 0; i < n; ++i) {
        if (a->codes[i] == 0 || b->codes[i] == 0) {
            r->data[i] = 0.0; continue;
        }
        auto *as = (struct matlab_string_s_fwd_ *)a->cats[a->codes[i] - 1];
        auto *bs = (struct matlab_string_s_fwd_ *)b->cats[b->codes[i] - 1];
        bool eq = as && bs && as->len == bs->len &&
                  memcmp(as->data, bs->data, (size_t)as->len) == 0;
        r->data[i] = eq ? 1.0 : 0.0;
    }
    return r;
}

/* ====================================================================== */
/* Phase 5.1 — datetime / duration.
 *
 * matlab_datetime stores a single Unix-epoch second count as f64.
 * matlab_duration is a relative second count as f64. The descriptors
 * are heap-allocated so the lowering can pass ptr-typed values around;
 * the runtime exposes constructors, display, and arithmetic.
 *
 * Display: datetime renders as "DD-Mon-YYYY HH:MM:SS" (MATLAB's
 * default); duration as "X seconds" (smart-unit picking is a follow-up).
 * ====================================================================== */

#include <time.h>

/* matlab_datetime_s / matlab_duration_s structs were promoted up
 * to the table-disp section so the renderer can read ->seconds.
 * Re-declaring them here would be ODR violation. */

extern "C" matlab_datetime *matlab_datetime_now(void) {
    matlab_datetime *d = (matlab_datetime *)calloc(1, sizeof(*d));
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    d->seconds = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    return d;
}

/* Compute Unix-epoch seconds from civil date (UTC). Uses Howard Hinnant's
 * date algorithm, which avoids OS time library limits and locale quirks. */
static double civil_to_epoch(int y, int m, int d, int hh, int mm, double ss) {
    /* Normalise so that March is month 1. */
    int ny = m <= 2 ? y - 1 : y;
    int nm = m + (m <= 2 ? 9 : -3);
    long era = (ny >= 0 ? ny : ny - 399) / 400;
    unsigned yoe = (unsigned)(ny - era * 400);
    unsigned doy = (153u * (unsigned)nm + 2u) / 5u + (unsigned)d - 1u;
    unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    long days = era * 146097L + (long)doe - 719468L;
    return (double)days * 86400.0 +
           (double)hh * 3600.0 + (double)mm * 60.0 + ss;
}

extern "C" matlab_datetime *matlab_datetime_ymd(double y, double m, double d) {
    matlab_datetime *t = (matlab_datetime *)calloc(1, sizeof(*t));
    t->seconds = civil_to_epoch((int)y, (int)m, (int)d, 0, 0, 0.0);
    return t;
}

extern "C" matlab_datetime *matlab_datetime_ymdhms(
        double y, double m, double d, double h, double mn, double s) {
    matlab_datetime *t = (matlab_datetime *)calloc(1, sizeof(*t));
    t->seconds = civil_to_epoch((int)y, (int)m, (int)d,
                                 (int)h, (int)mn, s);
    return t;
}

static void epoch_to_civil(double secs,
                            int *y, int *m, int *d,
                            int *hh, int *mm, double *ss) {
    long total = (long)secs;
    double frac = secs - (double)total;
    long days = total >= 0 ? total / 86400 : -((-total + 86399) / 86400);
    long sod = total - days * 86400;
    *hh = (int)(sod / 3600);
    *mm = (int)((sod / 60) % 60);
    *ss = (double)(sod % 60) + frac;
    /* Inverse of civil_to_epoch using Howard Hinnant's algorithm. */
    long z = days + 719468L;
    long era = (z >= 0 ? z : z - 146096) / 146097;
    unsigned doe = (unsigned)(z - era * 146097);
    unsigned yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
    int ny = (int)yoe + (int)(era * 400);
    unsigned doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
    unsigned mp = (5u * doy + 2u) / 153u;
    *d = (int)(doy - (153u * mp + 2u) / 5u + 1u);
    *m = (int)mp + (mp < 10u ? 3 : -9);
    *y = ny + (*m <= 2 ? 1 : 0);
}

extern "C" void matlab_datetime_disp(matlab_datetime *t) {
    if (!t) { matlab_disp_str("(empty datetime)", 16); return; }
    int y, m, d, hh, mm; double ss;
    epoch_to_civil(t->seconds, &y, &m, &d, &hh, &mm, &ss);
    static const char *months[] = {"Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"};
    char buf[64];
    int mi = (m - 1) % 12; if (mi < 0) mi += 12;
    int isec = (int)ss;
    int n = snprintf(buf, sizeof buf, "%02d-%s-%04d %02d:%02d:%02d",
                     d, months[mi], y, hh, mm, isec);
    pthread_mutex_lock(&matlab_io_mutex);
    fwrite(buf, 1, (size_t)n, stdout); putchar('\n');
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* Internal helpers shared between the datetime / duration entries. */
static matlab_duration *dur_make(double s) {
    matlab_duration *d = (matlab_duration *)calloc(1, sizeof(*d));
    d->seconds = s;
    return d;
}

extern "C" matlab_duration *matlab_duration_seconds(double n) { return dur_make(n); }
extern "C" matlab_duration *matlab_duration_minutes(double n) { return dur_make(n * 60.0); }
extern "C" matlab_duration *matlab_duration_hours  (double n) { return dur_make(n * 3600.0); }
extern "C" matlab_duration *matlab_duration_days   (double n) { return dur_make(n * 86400.0); }
extern "C" matlab_duration *matlab_duration_years  (double n) { return dur_make(n * 365.25 * 86400.0); }

extern "C" double matlab_duration_to_seconds(matlab_duration *d) {
    return d ? d->seconds : 0.0;
}
extern "C" double matlab_duration_to_minutes(matlab_duration *d) {
    return d ? d->seconds / 60.0 : 0.0;
}
extern "C" double matlab_duration_to_hours  (matlab_duration *d) {
    return d ? d->seconds / 3600.0 : 0.0;
}
extern "C" double matlab_duration_to_days   (matlab_duration *d) {
    return d ? d->seconds / 86400.0 : 0.0;
}

extern "C" void matlab_duration_disp(matlab_duration *d) {
    if (!d) { matlab_disp_str("(empty duration)", 16); return; }
    double s = d->seconds;
    char buf[64];
    int n;
    /* Smart-unit pick: hours / minutes / seconds. */
    if (fabs(s) >= 86400.0)
        n = snprintf(buf, sizeof buf, "%.4f days", s / 86400.0);
    else if (fabs(s) >= 3600.0)
        n = snprintf(buf, sizeof buf, "%.4f hr", s / 3600.0);
    else if (fabs(s) >= 60.0)
        n = snprintf(buf, sizeof buf, "%.4f min", s / 60.0);
    else
        n = snprintf(buf, sizeof buf, "%.6f sec", s);
    pthread_mutex_lock(&matlab_io_mutex);
    fwrite(buf, 1, (size_t)n, stdout); putchar('\n');
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* Arithmetic. All return fresh descriptors. */
extern "C" matlab_duration *matlab_datetime_sub_datetime(
        matlab_datetime *a, matlab_datetime *b) {
    return dur_make((a ? a->seconds : 0.0) - (b ? b->seconds : 0.0));
}
extern "C" matlab_datetime *matlab_datetime_add_duration(
        matlab_datetime *a, matlab_duration *d) {
    matlab_datetime *r = (matlab_datetime *)calloc(1, sizeof(*r));
    r->seconds = (a ? a->seconds : 0.0) + (d ? d->seconds : 0.0);
    return r;
}
extern "C" matlab_datetime *matlab_datetime_sub_duration(
        matlab_datetime *a, matlab_duration *d) {
    matlab_datetime *r = (matlab_datetime *)calloc(1, sizeof(*r));
    r->seconds = (a ? a->seconds : 0.0) - (d ? d->seconds : 0.0);
    return r;
}
extern "C" matlab_duration *matlab_duration_add(
        matlab_duration *a, matlab_duration *b) {
    return dur_make((a ? a->seconds : 0.0) + (b ? b->seconds : 0.0));
}
extern "C" matlab_duration *matlab_duration_sub(
        matlab_duration *a, matlab_duration *b) {
    return dur_make((a ? a->seconds : 0.0) - (b ? b->seconds : 0.0));
}

/* ====================================================================== */
/* Phase 4 — containers.Map / dictionary.
 *
 * A simple key/value map. Keys are either f64 scalars or strings
 * (matlab_string *). Values are either f64 scalars or matrix pointers
 * (matlab_mat *). Internally a flat parallel-array structure with O(N)
 * lookup — fine for the test corpus and the typical small dictionaries
 * MATLAB programs build.
 *
 * MATLAB exposes two surface APIs (containers.Map predates dictionary)
 * but the lowering / runtime treats them identically. Constructors:
 *   containers.Map()
 *   containers.Map(KeyType, ValueType)            -- keys/values typed
 *   dictionary()
 *   dictionary(k1, v1, k2, v2, ...)               -- inline init
 *
 * Indexing: m(k) read / m(k) = v write.
 * ====================================================================== */

struct matlab_dict_s {
    int32_t n;
    int32_t cap;
    /* Per-slot key kind: 0 = f64, 1 = matlab_string *. */
    int32_t *key_kinds;
    double  *key_f64;
    void   **key_str;
    /* Per-slot value kind: 0 = f64, 1 = matlab_mat *. */
    int32_t *val_kinds;
    double  *val_f64;
    void   **val_ptr;
};
typedef struct matlab_dict_s matlab_dict;

static void dict_grow(matlab_dict *d, int32_t need) {
    if (d->cap >= need) return;
    int32_t newcap = d->cap ? d->cap : 4;
    while (newcap < need) newcap *= 2;
    d->key_kinds = (int32_t *)realloc(d->key_kinds, (size_t)newcap * sizeof(int32_t));
    d->key_f64   = (double  *)realloc(d->key_f64,   (size_t)newcap * sizeof(double));
    d->key_str   = (void   **)realloc(d->key_str,   (size_t)newcap * sizeof(void *));
    d->val_kinds = (int32_t *)realloc(d->val_kinds, (size_t)newcap * sizeof(int32_t));
    d->val_f64   = (double  *)realloc(d->val_f64,   (size_t)newcap * sizeof(double));
    d->val_ptr   = (void   **)realloc(d->val_ptr,   (size_t)newcap * sizeof(void *));
    for (int32_t i = d->cap; i < newcap; ++i) {
        d->key_kinds[i] = 0; d->key_f64[i] = 0.0; d->key_str[i] = NULL;
        d->val_kinds[i] = 0; d->val_f64[i] = 0.0; d->val_ptr[i] = NULL;
    }
    d->cap = newcap;
}

extern "C" matlab_dict *matlab_dict_new(void) {
    return (matlab_dict *)calloc(1, sizeof(matlab_dict));
}

/* The matlab_string_s_fwd_ helper layout was already defined further
 * up (it's used by Phase 5.2 categorical too). */
static int32_t dict_find_str(matlab_dict *d, const char *s, int64_t len) {
    if (!d) return -1;
    for (int32_t i = 0; i < d->n; ++i) {
        if (d->key_kinds[i] != 1) continue;
        auto *ks = (struct matlab_string_s_fwd_ *)d->key_str[i];
        if (!ks) continue;
        if (ks->len == len && ks->data &&
            memcmp(ks->data, s, (size_t)len) == 0) return i;
    }
    return -1;
}

static int32_t dict_find_f64(matlab_dict *d, double k) {
    if (!d) return -1;
    for (int32_t i = 0; i < d->n; ++i) {
        if (d->key_kinds[i] == 0 && d->key_f64[i] == k) return i;
    }
    return -1;
}

static int32_t dict_reserve_str(matlab_dict *d, void *key) {
    auto *ks = (struct matlab_string_s_fwd_ *)key;
    int64_t kl = ks ? ks->len : 0;
    const char *kd = ks ? ks->data : "";
    int32_t idx = dict_find_str(d, kd, kl);
    if (idx >= 0) return idx;
    dict_grow(d, d->n + 1);
    idx = d->n++;
    d->key_kinds[idx] = 1;
    d->key_str[idx] = key;
    return idx;
}

static int32_t dict_reserve_f64(matlab_dict *d, double k) {
    int32_t idx = dict_find_f64(d, k);
    if (idx >= 0) return idx;
    dict_grow(d, d->n + 1);
    idx = d->n++;
    d->key_kinds[idx] = 0;
    d->key_f64[idx] = k;
    return idx;
}

extern "C" void matlab_dict_set_str_f64(matlab_dict *d, void *key, double v) {
    if (!d) return;
    int32_t i = dict_reserve_str(d, key);
    d->val_kinds[i] = 0; d->val_f64[i] = v; d->val_ptr[i] = NULL;
}

extern "C" void matlab_dict_set_str_mat(matlab_dict *d, void *key, matlab_mat *m) {
    if (!d) return;
    int32_t i = dict_reserve_str(d, key);
    d->val_kinds[i] = 1; d->val_f64[i] = 0.0; d->val_ptr[i] = m;
}

extern "C" void matlab_dict_set_num_f64(matlab_dict *d, double k, double v) {
    if (!d) return;
    int32_t i = dict_reserve_f64(d, k);
    d->val_kinds[i] = 0; d->val_f64[i] = v; d->val_ptr[i] = NULL;
}

extern "C" void matlab_dict_set_num_mat(matlab_dict *d, double k, matlab_mat *m) {
    if (!d) return;
    int32_t i = dict_reserve_f64(d, k);
    d->val_kinds[i] = 1; d->val_f64[i] = 0.0; d->val_ptr[i] = m;
}

extern "C" double matlab_dict_get_str_f64(matlab_dict *d, void *key) {
    if (!d) return 0.0;
    auto *ks = (struct matlab_string_s_fwd_ *)key;
    int32_t i = dict_find_str(d, ks ? ks->data : "", ks ? ks->len : 0);
    if (i < 0) return 0.0;
    if (d->val_kinds[i] == 0) return d->val_f64[i];
    if (d->val_kinds[i] == 1 && d->val_ptr[i]) {
        matlab_mat *m = (matlab_mat *)d->val_ptr[i];
        if (m->rows == 1 && m->cols == 1) return m->data[0];
    }
    return 0.0;
}

extern "C" matlab_mat *matlab_dict_get_str_mat(matlab_dict *d, void *key) {
    if (!d) return mat_alloc(0, 0);
    auto *ks = (struct matlab_string_s_fwd_ *)key;
    int32_t i = dict_find_str(d, ks ? ks->data : "", ks ? ks->len : 0);
    if (i < 0) return mat_alloc(0, 0);
    if (d->val_kinds[i] == 1 && d->val_ptr[i])
        return (matlab_mat *)d->val_ptr[i];
    if (d->val_kinds[i] == 0) {
        matlab_mat *m = mat_alloc(1, 1);
        m->data[0] = d->val_f64[i];
        return m;
    }
    return mat_alloc(0, 0);
}

extern "C" double matlab_dict_get_num_f64(matlab_dict *d, double k) {
    if (!d) return 0.0;
    int32_t i = dict_find_f64(d, k);
    if (i < 0) return 0.0;
    if (d->val_kinds[i] == 0) return d->val_f64[i];
    if (d->val_kinds[i] == 1 && d->val_ptr[i]) {
        matlab_mat *m = (matlab_mat *)d->val_ptr[i];
        if (m->rows == 1 && m->cols == 1) return m->data[0];
    }
    return 0.0;
}

extern "C" matlab_mat *matlab_dict_get_num_mat(matlab_dict *d, double k) {
    if (!d) return mat_alloc(0, 0);
    int32_t i = dict_find_f64(d, k);
    if (i < 0) return mat_alloc(0, 0);
    if (d->val_kinds[i] == 1 && d->val_ptr[i])
        return (matlab_mat *)d->val_ptr[i];
    if (d->val_kinds[i] == 0) {
        matlab_mat *m = mat_alloc(1, 1);
        m->data[0] = d->val_f64[i];
        return m;
    }
    return mat_alloc(0, 0);
}

extern "C" double matlab_dict_has_str(matlab_dict *d, void *key) {
    if (!d) return 0.0;
    auto *ks = (struct matlab_string_s_fwd_ *)key;
    return dict_find_str(d, ks ? ks->data : "", ks ? ks->len : 0) >= 0 ? 1.0 : 0.0;
}

extern "C" double matlab_dict_has_num(matlab_dict *d, double k) {
    if (!d) return 0.0;
    return dict_find_f64(d, k) >= 0 ? 1.0 : 0.0;
}

extern "C" double matlab_dict_length(matlab_dict *d) {
    return d ? (double)d->n : 0.0;
}

extern "C" double matlab_dict_remove_str(matlab_dict *d, void *key) {
    if (!d) return 0.0;
    auto *ks = (struct matlab_string_s_fwd_ *)key;
    int32_t i = dict_find_str(d, ks ? ks->data : "", ks ? ks->len : 0);
    if (i < 0) return 0.0;
    /* Shift down. */
    for (int32_t k = i; k < d->n - 1; ++k) {
        d->key_kinds[k] = d->key_kinds[k+1];
        d->key_f64[k]   = d->key_f64[k+1];
        d->key_str[k]   = d->key_str[k+1];
        d->val_kinds[k] = d->val_kinds[k+1];
        d->val_f64[k]   = d->val_f64[k+1];
        d->val_ptr[k]   = d->val_ptr[k+1];
    }
    d->n--;
    return 1.0;
}

extern "C" double matlab_dict_remove_num(matlab_dict *d, double k) {
    if (!d) return 0.0;
    int32_t i = dict_find_f64(d, k);
    if (i < 0) return 0.0;
    for (int32_t kk = i; kk < d->n - 1; ++kk) {
        d->key_kinds[kk] = d->key_kinds[kk+1];
        d->key_f64[kk]   = d->key_f64[kk+1];
        d->key_str[kk]   = d->key_str[kk+1];
        d->val_kinds[kk] = d->val_kinds[kk+1];
        d->val_f64[kk]   = d->val_f64[kk+1];
        d->val_ptr[kk]   = d->val_ptr[kk+1];
    }
    d->n--;
    return 1.0;
}

/* Phase 3 — value-class copy semantics. matlab_obj_clone produces a
 * fresh matlab_obj that owns independent name / kinds / ptr arrays
 * but shares property *values* (matrix-pointer fields are not deep-
 * copied; the obj's own class_id is preserved). MATLAB's value-class
 * rule is "copy-on-assign + mutations are local to the holder";
 * since each property write goes through matlab_obj_set_*, which
 * overwrites the slot in place, the shallow-copy here is sufficient
 * for the read/write field semantics. Nested obj-fields are NOT
 * cloned — those are handle-style references and would need a
 * recursive clone if Phase 4 wants stricter value semantics. */
extern "C" matlab_obj *matlab_obj_clone(matlab_obj *o) {
    if (!o) return matlab_obj_new(0);
    matlab_obj *c = matlab_obj_new(o->class_id);
    /* Reserve enough capacity. */
    while (c->capacity < o->nfields) {
        int32_t newcap = c->capacity ? c->capacity * 2 : MATLAB_STRUCT_CAP_INIT;
        c->names    = (char **)realloc(c->names,    (size_t)newcap * sizeof(char *));
        c->kinds    = (int32_t *)realloc(c->kinds,  (size_t)newcap * sizeof(int32_t));
        c->f64_vals = (double *)realloc(c->f64_vals,(size_t)newcap * sizeof(double));
        c->ptr_vals = (void **)realloc(c->ptr_vals, (size_t)newcap * sizeof(void *));
        for (int32_t i = c->capacity; i < newcap; ++i) {
            c->names[i] = NULL; c->kinds[i] = 0;
            c->f64_vals[i] = 0.0; c->ptr_vals[i] = NULL;
        }
        c->capacity = newcap;
    }
    for (int32_t i = 0; i < o->nfields; ++i) {
        c->names[i]    = strdup(o->names[i] ? o->names[i] : "");
        c->kinds[i]    = o->kinds[i];
        c->f64_vals[i] = o->f64_vals[i];
        c->ptr_vals[i] = o->ptr_vals[i];
    }
    c->nfields = o->nfields;
    return c;
}

/* Each accessor just forwards to the matlab_struct_* variant, because
 * the layout is identical through the struct prefix. Keeping these
 * as distinct symbols lets the frontend pick the name that reflects
 * the programmer's intent (property vs. struct field). */
void matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v) {
    matlab_struct_set_f64((matlab_struct *)o, name, len, v);
}

void matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m) {
    matlab_struct_set_mat((matlab_struct *)o, name, len, m);
}

/* String-typed property setter / getter — used by classdef kwarg
 * sugar when a property value is a `matlab_string *` (string literal
 * or sprintf result).  Layout-compatible with the workspace-side
 * kind=3 storage so DAP property inspection renders it correctly. */
void matlab_obj_set_string(matlab_obj *o, const char *name, int64_t len, void *str) {
    matlab_struct_set_string((matlab_struct *)o, name, len, str);
}

void *matlab_obj_get_string(matlab_obj *o, const char *name, int64_t len) {
    return matlab_struct_get_string((matlab_struct *)o, name, len);
}

/* Runtime-dispatched display for a class-instance property.  Lowering
 * routes `disp(obj.Field)` to this entry when `obj` is class-pinned
 * but the property's static type can't be inferred (e.g. a kwarg-
 * stored value).  The helper looks at the stored `kinds[idx]` and
 * picks the matching disp variant:
 *   kind=0 → matlab_disp_f64 (scalar)
 *   kind=1 → matlab_disp_mat (matrix)
 *   kind=3 → matlab_string_disp (string)
 *   kind=4/5 → typed-int matrix (also routes through matlab_disp_mat)
 *   anything else / missing field → noop
 *
 * Falls back to a blank line on missing-field to mirror MATLAB's
 * silent-on-undefined-field semantics in disp contexts. */
struct matlab_string_s;
extern "C" void matlab_string_disp(struct matlab_string_s *s);
void matlab_disp_mat(void *m);  /* defined later in this TU */
extern "C" const char *matlab_dbg_class_name(int32_t class_id, int64_t *len_out);
void matlab_obj_disp_field(matlab_obj *o, const char *name, int64_t len) {
    matlab_struct *s = (matlab_struct *)o;
    if (!s) { printf("\n"); return; }
    int32_t idx = struct_find_field(s, name, (int32_t)len);
    if (idx < 0) { printf("\n"); return; }
    uint8_t k = s->kinds[idx];
    if (k == 0) {
        matlab_disp_f64(s->f64_vals[idx]);
    } else if (k == 3 && s->ptr_vals[idx]) {
        matlab_string_disp((struct matlab_string_s *)s->ptr_vals[idx]);
    } else if ((k == 1 || k == 4 || k == 5) && s->ptr_vals[idx]) {
        matlab_disp_mat(s->ptr_vals[idx]);
    } else if (k == 2 && s->ptr_vals[idx]) {
        /* Class instance — print as `1x1 ClassName` summary. */
        matlab_obj *child = (matlab_obj *)s->ptr_vals[idx];
        int64_t cnLen = 0;
        const char *cn = matlab_dbg_class_name(child->class_id, &cnLen);
        if (cn) {
            printf("  1x1 %.*s\n", (int)cnLen, cn);
        } else {
            printf("  1x1 <class %d>\n", (int)child->class_id);
        }
    } else {
        printf("\n");
    }
}

double matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len) {
    return matlab_struct_get_f64((matlab_struct *)o, name, len);
}

matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len) {
    return matlab_struct_get_mat((matlab_struct *)o, name, len);
}

/* ---------------------------------------------------------------------- */
/* §3.1 — disp(tf) formatted s-domain rendering.
 *
 * The Lowering.cpp `disp(obj)` class-method dispatch routes a tf-pinned
 * operand through this helper instead of the generic matrix disp path.
 * Output mirrors MATLAB's centred-fraction layout:
 *
 *           s + 2
 *     ----------------
 *     s^2 + 3 s + 5
 *
 *     Continuous-time transfer function.
 *
 * Coefficients are taken from the matlab_obj's Numerator and
 * Denominator properties (matlab_mat *), highest power first. The
 * `var` arg is 's' for continuous-time (today's only path; `tf('z')`
 * is sugared to the same coefficients with sample-time carry-through
 * a follow-on). */
static void matlab_tf_poly_to_str(matlab_mat *coeffs, char var,
                                   std::string &out) {
    if (!coeffs || !coeffs->data) { out = "0"; return; }
    int64_t n = coeffs->rows * coeffs->cols;
    bool first = true;
    char buf[64];
    for (int64_t i = 0; i < n; ++i) {
        double c = coeffs->data[i];
        int deg = (int)(n - 1 - i);
        if (c == 0.0 && (deg != 0 || !first)) continue;
        if (first) {
            if (c < 0) out += '-';
        } else {
            out += (c < 0) ? " - " : " + ";
        }
        first = false;
        double ac = std::fabs(c);
        if (deg == 0) {
            std::snprintf(buf, sizeof(buf), "%g", ac);
            out += buf;
        } else {
            if (ac != 1.0) {
                std::snprintf(buf, sizeof(buf), "%g ", ac);
                out += buf;
            }
            if (deg == 1) {
                out += var;
            } else {
                std::snprintf(buf, sizeof(buf), "%c^%d", var, deg);
                out += buf;
            }
        }
    }
    if (out.empty()) out = "0";
}

extern "C" void matlab_tf_disp(matlab_obj *o) {
    matlab_mat *num = matlab_obj_get_mat(o, "Numerator", 9);
    matlab_mat *den = matlab_obj_get_mat(o, "Denominator", 11);
    std::string num_str, den_str;
    matlab_tf_poly_to_str(num, 's', num_str);
    matlab_tf_poly_to_str(den, 's', den_str);
    size_t width = std::max(num_str.size(), den_str.size());
    std::string bar(width + 2, '-');
    size_t pad_n = (bar.size() - num_str.size()) / 2;
    size_t pad_d = (bar.size() - den_str.size()) / 2;
    std::printf("\n");
    std::printf("%*s%s\n", (int)pad_n, "", num_str.c_str());
    std::printf("  %s\n", bar.c_str());
    std::printf("%*s%s\n", (int)pad_d, "", den_str.c_str());
    std::printf("\n  Continuous-time transfer function.\n\n");
}

/* ---------------------------------------------------------------------- */
/* Real string type ("..." literals, distinct from '...' char arrays).
 *
 * matlab_string is a tiny {data, len} descriptor with a heap-copied
 * payload. The frontend emits matlab_string_from_literal(global, N) for
 * a "..." literal, and `+` between two strings lowers to
 * matlab_string_concat(a, b). disp of a string pointer routes to
 * matlab_string_disp via the frontend's StringBindings tracking.
 *
 * Lifetime is leaked per-program; that's consistent with the rest of
 * the runtime and fine for the short-lived programs the compiler
 * targets today.
 */
struct matlab_string_s {
    char *data;
    int64_t len;
};
typedef struct matlab_string_s matlab_string;

/* Pointer registry — same shape as matlab_obj_registry above. matlab_mat
 * and matlab_string both start with `<heap-pointer> + int64_t fields`,
 * so polymorphic entries (matlab_disp_mat) can't tell them apart by
 * peeking the descriptor alone. The REPL's `disp(t)` path lowers to
 * matlab_disp_mat(matlab_ws_get_mat(...)) regardless of the binding's
 * actual kind because Sema can't see the persisted workspace state
 * across compilations; matlab_disp_mat consults this registry to
 * detect string descriptors and route to matlab_string_disp instead
 * of dereferencing the descriptor's bytes as a numeric matrix. */
static struct {
    pthread_mutex_t mu;
    void **ptrs;
    int count;
    int cap;
} matlab_string_registry = { PTHREAD_MUTEX_INITIALIZER, NULL, 0, 0 };

static void matlab_string_registry_add(void *p) {
    if (!p) return;
    pthread_mutex_lock(&matlab_string_registry.mu);
    if (matlab_string_registry.count == matlab_string_registry.cap) {
        int ncap = matlab_string_registry.cap ? matlab_string_registry.cap * 2 : 16;
        void **nptrs = (void **)realloc(matlab_string_registry.ptrs,
                                        (size_t)ncap * sizeof(void *));
        if (nptrs) {
            matlab_string_registry.ptrs = nptrs;
            matlab_string_registry.cap = ncap;
        }
    }
    if (matlab_string_registry.count < matlab_string_registry.cap) {
        matlab_string_registry.ptrs[matlab_string_registry.count++] = p;
    }
    pthread_mutex_unlock(&matlab_string_registry.mu);
}

int matlab_string_is_known(const void *p) {
    if (!p) return 0;
    int found = 0;
    pthread_mutex_lock(&matlab_string_registry.mu);
    for (int i = 0; i < matlab_string_registry.count; ++i) {
        if (matlab_string_registry.ptrs[i] == p) { found = 1; break; }
    }
    pthread_mutex_unlock(&matlab_string_registry.mu);
    return found;
}

matlab_string *matlab_string_from_literal(const char *src, int64_t len) {
    matlab_string *s = (matlab_string *)calloc(1, sizeof(*s));
    s->len = len < 0 ? 0 : len;
    s->data = (char *)malloc((size_t)s->len + 1);
    if (src && s->len > 0) memcpy(s->data, src, (size_t)s->len);
    s->data[s->len] = '\0';
    matlab_string_registry_add(s);
    return s;
}

matlab_string *matlab_string_concat(matlab_string *a, matlab_string *b) {
    int64_t la = a ? a->len : 0;
    int64_t lb = b ? b->len : 0;
    matlab_string *s = (matlab_string *)calloc(1, sizeof(*s));
    s->len = la + lb;
    s->data = (char *)malloc((size_t)s->len + 1);
    if (a && la > 0) memcpy(s->data, a->data, (size_t)la);
    if (b && lb > 0) memcpy(s->data + la, b->data, (size_t)lb);
    s->data[s->len] = '\0';
    matlab_string_registry_add(s);
    return s;
}

void matlab_string_disp(matlab_string *s) {
    if (!s) return;
    matlab_disp_str(s->data, s->len);
}

double matlab_string_len(matlab_string *s) {
    if (!s) return 0.0;
    return (double)s->len;
}

double matlab_isstring(matlab_string *s) { return s ? 1.0 : 0.0; }

/* Returns a fresh 1x2 row vector [1 1] — the size of a string scalar.
 * Used by the lowering's length/numel/size fold for string bindings:
 * MATLAB treats `"Test"` as a 1x1 string array (one element whose value
 * is the text), so size(s) is `[1 1]`. The fold has to return a
 * matlab_mat* (because callers feed the result into the generic
 * `disp` / arith path); allocating it here keeps the lowering free
 * of inline matrix-construction. */
matlab_mat *matlab_string_size_scalar(void) {
    matlab_mat *m = mat_alloc(1, 2);
    m->data[0] = 1.0;
    m->data[1] = 1.0;
    return m;
}

/* Opaque accessors for the runtime_debug TU (and the DAP/REPL frontend
 * via tools/matlabc/main.cpp). The matlab_string_s layout is private
 * to this TU; callers go through these helpers so the descriptor's
 * fields can move without breaking the workspace inspector. NULL-safe
 * — a missing string reads as a zero-length empty literal. */
const char *matlab_string_get_data(void *s, int64_t *len_out) {
    matlab_string *ms = (matlab_string *)s;
    if (!ms) {
        if (len_out) *len_out = 0;
        return "";
    }
    if (len_out) *len_out = ms->len;
    return ms->data ? ms->data : "";
}

int64_t matlab_string_get_len(void *s) {
    matlab_string *ms = (matlab_string *)s;
    return ms ? ms->len : 0;
}

/* sprintf(fmt, v) -> matlab_string. Only the one-f64 form is wired
 * for now, matching the other fprintf family variants. The expand-
 * escapes helper processes MATLAB-style backslash escapes. */
matlab_string *matlab_sprintf_str(matlab_string *fmt) {
    if (!fmt) return matlab_string_from_literal("", 0);
    char buf[2048];
    int64_t n = expand_escapes(buf, fmt->data, fmt->len);
    if (n >= (int64_t)sizeof buf) n = (int64_t)sizeof buf - 1;
    return matlab_string_from_literal(buf, n);
}

matlab_string *matlab_sprintf_f64(matlab_string *fmt, double v) {
    if (!fmt) return matlab_string_from_literal("", 0);
    char expanded[1024];
    int64_t en = expand_escapes(expanded, fmt->data, fmt->len);
    if (en < (int64_t)sizeof expanded) expanded[en] = '\0';
    else expanded[sizeof expanded - 1] = '\0';
    char out[2048];
    int n = snprintf(out, sizeof out, expanded, v);
    if (n < 0) n = 0;
    if (n >= (int)sizeof out) n = (int)sizeof out - 1;
    return matlab_string_from_literal(out, (int64_t)n);
}

matlab_string *matlab_num2str(double v) {
    char buf[64];
    int n = snprintf(buf, sizeof buf, "%g", v);
    if (n < 0) n = 0;
    return matlab_string_from_literal(buf, (int64_t)n);
}

double matlab_str2double(matlab_string *s) {
    if (!s || !s->data) return 0.0 / 0.0; /* NaN */
    char *end = NULL;
    double v = strtod(s->data, &end);
    if (end == s->data) return 0.0 / 0.0;
    return v;
}

static matlab_string *map_chars(matlab_string *s, int (*f)(int)) {
    if (!s) return matlab_string_from_literal("", 0);
    matlab_string *r = matlab_string_from_literal(s->data, s->len);
    for (int64_t i = 0; i < r->len; ++i)
        r->data[i] = (char)f((unsigned char)r->data[i]);
    return r;
}

static int to_upper_i(int c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }
static int to_lower_i(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

matlab_string *matlab_upper(matlab_string *s) { return map_chars(s, to_upper_i); }
matlab_string *matlab_lower(matlab_string *s) { return map_chars(s, to_lower_i); }

double matlab_startsWith(matlab_string *s, matlab_string *pre) {
    if (!s || !pre) return 0.0;
    if (pre->len > s->len) return 0.0;
    return memcmp(s->data, pre->data, (size_t)pre->len) == 0 ? 1.0 : 0.0;
}

double matlab_endsWith(matlab_string *s, matlab_string *suf) {
    if (!s || !suf) return 0.0;
    if (suf->len > s->len) return 0.0;
    return memcmp(s->data + (s->len - suf->len),
                  suf->data, (size_t)suf->len) == 0 ? 1.0 : 0.0;
}

double matlab_contains(matlab_string *s, matlab_string *needle) {
    if (!s || !needle) return 0.0;
    if (needle->len == 0) return 1.0;
    if (needle->len > s->len) return 0.0;
    return strstr(s->data, needle->data) != NULL ? 1.0 : 0.0;
}

matlab_string *matlab_strtrim(matlab_string *s) {
    if (!s) return matlab_string_from_literal("", 0);
    int64_t lo = 0, hi = s->len;
    while (lo < hi && (unsigned char)s->data[lo] <= ' ') ++lo;
    while (hi > lo && (unsigned char)s->data[hi - 1] <= ' ') --hi;
    return matlab_string_from_literal(s->data + lo, hi - lo);
}

/* strrep(s, old, new): every non-overlapping occurrence of `old` in
 * `s` replaced with `new`. Returns a fresh heap string. */
matlab_string *matlab_strrep(matlab_string *s, matlab_string *old, matlab_string *nw) {
    if (!s) return matlab_string_from_literal("", 0);
    if (!old || old->len == 0) return matlab_string_from_literal(s->data, s->len);
    int64_t new_len = nw ? nw->len : 0;
    /* First pass: count occurrences to size the output buffer. */
    int64_t count = 0;
    const char *p = s->data;
    const char *end = s->data + s->len;
    while (p + old->len <= end) {
        if (memcmp(p, old->data, (size_t)old->len) == 0) {
            ++count;
            p += old->len;
        } else {
            ++p;
        }
    }
    int64_t out_len = s->len + count * (new_len - old->len);
    char *out = (char *)malloc((size_t)out_len + 1);
    char *w = out;
    p = s->data;
    while (p + old->len <= end) {
        if (memcmp(p, old->data, (size_t)old->len) == 0) {
            if (new_len > 0) { memcpy(w, nw->data, (size_t)new_len); w += new_len; }
            p += old->len;
        } else {
            *w++ = *p++;
        }
    }
    while (p < end) *w++ = *p++;
    *w = '\0';
    matlab_string *r = matlab_string_from_literal(out, out_len);
    free(out);
    return r;
}

matlab_string *matlab_strcat(matlab_string *a, matlab_string *b) {
    return matlab_string_concat(a, b);
}

/* sub2ind([m n], i, j): column-major 1-based linear index into an
 * m-by-n matrix. MATLAB is column-major in its linear indexing model
 * even though our underlying storage is row-major — we follow
 * MATLAB here so user-facing semantics line up. */
double matlab_sub2ind(matlab_mat *shape, double di, double dj) {
    if (!shape) return 0.0;
    int64_t total = shape->rows * shape->cols;
    if (total < 2) return 0.0;
    int64_t m = (int64_t)shape->data[0];
    int64_t i = (int64_t)di;   /* 1-based */
    int64_t j = (int64_t)dj;   /* 1-based */
    return (double)((j - 1) * m + i);
}

/* ind2sub([m n], idx): return [i j] as a 1x2 row. Column-major like
 * sub2ind. */
matlab_mat *matlab_ind2sub(matlab_mat *shape, double idx) {
    matlab_mat *R = mat_alloc(1, 2);
    if (!shape) return R;
    int64_t total = shape->rows * shape->cols;
    if (total < 2) return R;
    int64_t m = (int64_t)shape->data[0];
    int64_t k = (int64_t)idx - 1;   /* 0-based */
    if (m <= 0) return R;
    int64_t i = (k % m) + 1;
    int64_t j = (k / m) + 1;
    R->data[0] = (double)i;
    R->data[1] = (double)j;
    return R;
}

/* assert(cond) / assert(cond, msg). Uses the existing error flag so
 * a failed assertion is catchable by try/catch. Two forms: f64-only
 * cond, and cond + matlab_string message. */
void matlab_assert(double cond) {
    if (cond != 0.0) return;
    matlab_set_error_msg("assertion failed", 16);
}

void matlab_assert_msg(double cond, matlab_string *msg) {
    if (cond != 0.0) return;
    if (msg && msg->len > 0)
        matlab_set_error_msg(msg->data, msg->len);
    else
        matlab_set_error_msg("assertion failed", 16);
}

/* -------- Linear algebra tail --------------------------------------------
 *
 * norm / trace / kron / chol / pinv.
 * Pure-C implementations, no BLAS/LAPACK. They're correct to the
 * tolerance a double can naturally reach but aren't tuned for speed
 * or numeric stability — matching the rest of the runtime.
 *--------------------------------------------------------------------------*/

/* norm(A): matrix Frobenius norm (sqrt(sum of squares)). For a
 * vector, this coincides with the 2-norm. */
double matlab_norm(matlab_mat *A) {
    if (!A) return 0.0;
    int64_t total = A->rows * A->cols;
    double acc = 0.0;
    for (int64_t k = 0; k < total; ++k) {
        double x = A->data[k];
        acc += x * x;
    }
    return sqrt(acc);
}

/* trace(A): sum of diagonal. Defined for square matrices; for
 * non-square we sum min(rows, cols) leading-diagonal entries. */
double matlab_trace(matlab_mat *A) {
    if (!A) return 0.0;
    int64_t n = A->rows < A->cols ? A->rows : A->cols;
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) acc += A->data[i * A->cols + i];
    return acc;
}

/* kron(A, B): Kronecker product. Result is (Am*Bm) x (An*Bn) and
 * the (i*Bm+p, j*Bn+q) entry is A[i,j] * B[p,q]. */
/* Phase-5: 4-deep loop collapses to a shape_op lambda. R[r, c] indexes
 * back into A[r/bm, c/bn] * B[r%bm, c%bn] — the canonical Kronecker
 * product in 2-D. */
matlab_mat *matlab_kron(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(0, 0);
    int64_t am = A->rows, an = A->cols;
    int64_t bm = B->rows, bn = B->cols;
    return matlab::runtime::shape_op(am * bm, an * bn,
        [&](int64_t r, int64_t c) {
            return A->data[(r / bm) * an + (c / bn)] *
                   B->data[(r % bm) * bn + (c % bn)];
        }).release();
}

/* conv(u, v) — full 1-D convolution. MATLAB treats u and v as vectors
 * regardless of shape. The output orientation follows MATLAB: if either
 * input is a column vector, the result is a column; otherwise (both row
 * or scalar), the result is a row. Empty input yields an empty matrix. */
matlab_mat *matlab_conv(matlab_mat *u, matlab_mat *v) {
    if (!u || !v) return mat_alloc(0, 0);
    int64_t nu = u->rows * u->cols;
    int64_t nv = v->rows * v->cols;
    if (nu == 0 || nv == 0) return mat_alloc(0, 0);
    int64_t nw = nu + nv - 1;
    int u_is_col = (u->cols == 1 && u->rows > 1);
    int v_is_col = (v->cols == 1 && v->rows > 1);
    int as_col = u_is_col || v_is_col;
    matlab_mat *W = as_col ? mat_alloc(nw, 1) : mat_alloc(1, nw);
    for (int64_t k = 0; k < nw; ++k) {
        double s = 0.0;
        int64_t jlo = k - (nv - 1); if (jlo < 0) jlo = 0;
        int64_t jhi = k;            if (jhi > nu - 1) jhi = nu - 1;
        for (int64_t j = jlo; j <= jhi; ++j)
            s += u->data[j] * v->data[k - j];
        W->data[k] = s;
    }
    return W;
}

/* conv2(A, B) — full 2-D convolution. C[i,j] = sum_{p,q} A[p,q] * B[i-p,j-q]
 * for valid (p,q). Result size is (m1+m2-1) x (n1+n2-1). Returns 0x0 for
 * empty inputs. */
matlab_mat *matlab_conv2(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(0, 0);
    int64_t am = A->rows, an = A->cols;
    int64_t bm = B->rows, bn = B->cols;
    if (am == 0 || an == 0 || bm == 0 || bn == 0) return mat_alloc(0, 0);
    int64_t cm = am + bm - 1, cn = an + bn - 1;
    matlab_mat *C = mat_alloc(cm, cn);
    for (int64_t p = 0; p < am; ++p) {
        for (int64_t q = 0; q < an; ++q) {
            double a = A->data[p * an + q];
            if (a == 0.0) continue;
            for (int64_t r = 0; r < bm; ++r) {
                double *crow = C->data + (p + r) * cn + q;
                const double *brow = B->data + r * bn;
                for (int64_t s = 0; s < bn; ++s)
                    crow[s] += a * brow[s];
            }
        }
    }
    return C;
}

/* filter(b, a, x) — direct-form II transposed.
 *   a(1)*y[n] = sum_k b[k]*x[n-k] - sum_k a[k+1]*y[n-k-1]
 *
 * b and a are flattened to vectors; their order in MATLAB is [b0 b1 ... bN]
 * and [a0 a1 ... aM]. a(1) (i.e. a->data[0]) must be non-zero — we return
 * 0x0 otherwise. b/a are normalized by a0 once, then the loop is the
 * canonical DF-II-T form (one state vector w of length max(N,M)). x can
 * be a vector (the result mirrors x's orientation) or a matrix (filtered
 * column-wise). */
matlab_mat *matlab_filter(matlab_mat *b, matlab_mat *a, matlab_mat *x) {
    if (!b || !a || !x) return mat_alloc(0, 0);
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    if (nb == 0 || na == 0 || a->data[0] == 0.0) return mat_alloc(0, 0);
    double a0 = a->data[0];
    int64_t L = nb > na ? nb : na;
    /* Phase-4 RAII: bn / an / w go from manual calloc/free trio to
     * value-initialised std::vector — zero-fill is implicit. */
    std::vector<double> bn(L), an(L), w(L);
    for (int64_t k = 0; k < nb; ++k) bn[k] = b->data[k] / a0;
    for (int64_t k = 0; k < na; ++k) an[k] = a->data[k] / a0;

    int64_t xm = x->rows, xn = x->cols;
    /* Treat a vector input as a single column for processing; the result
     * is reshaped back to the input's orientation at the end. */
    int x_is_vec = (xm == 1 || xn == 1);
    int64_t cols = x_is_vec ? 1 : xn;
    int64_t rows = x_is_vec ? (xm * xn) : xm;
    matlab::runtime::MatPtr Y = matlab::runtime::make_mat(rows, cols);
    for (int64_t c = 0; c < cols; ++c) {
        for (int64_t i = 0; i < L; ++i) w[i] = 0.0;
        for (int64_t n = 0; n < rows; ++n) {
            double xn_val = x_is_vec ? x->data[n] : x->data[n * xn + c];
            double yn = bn[0] * xn_val + w[0];
            /* Shift the state register and inject the cross-coupled terms. */
            for (int64_t i = 1; i < L; ++i)
                w[i - 1] = bn[i] * xn_val - an[i] * yn + w[i];
            w[L - 1] = 0.0;
            if (x_is_vec) Y->data[n] = yn;
            else          Y->data[n * cols + c] = yn;
        }
    }
    /* Reshape back to original vector orientation. */
    if (x_is_vec) { Y->rows = xm; Y->cols = xn; }
    return Y.release();
}

/* any/all share the colwise-reduce shape but with logical update rules.
 * The result is a 1x1 matrix on a vector input, or a 1xN row of bools
 * (stored as 0.0 / 1.0 doubles) on a matrix input. */
matlab_mat *matlab_any(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m <= 1 || n == 1) {
        int64_t total = m * n;
        double r = 0.0;
        for (int64_t k = 0; k < total; ++k)
            if (A->data[k] != 0.0) { r = 1.0; break; }
        matlab_mat *R = mat_alloc(1, 1);
        R->data[0] = r;
        return R;
    }
    matlab_mat *R = mat_alloc(1, n);
    for (int64_t j = 0; j < n; ++j) {
        double r = 0.0;
        for (int64_t i = 0; i < m; ++i)
            if (A->data[i * n + j] != 0.0) { r = 1.0; break; }
        R->data[j] = r;
    }
    return R;
}

matlab_mat *matlab_all(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m <= 1 || n == 1) {
        int64_t total = m * n;
        double r = 1.0;
        for (int64_t k = 0; k < total; ++k)
            if (A->data[k] == 0.0) { r = 0.0; break; }
        matlab_mat *R = mat_alloc(1, 1);
        R->data[0] = total > 0 ? r : 1.0;  /* all([]) is true in MATLAB */
        return R;
    }
    matlab_mat *R = mat_alloc(1, n);
    for (int64_t j = 0; j < n; ++j) {
        double r = 1.0;
        for (int64_t i = 0; i < m; ++i)
            if (A->data[i * n + j] == 0.0) { r = 0.0; break; }
        R->data[j] = r;
    }
    return R;
}

matlab_mat *matlab_tril(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j <= i && j < n; ++j)
            R->data[i * n + j] = A->data[i * n + j];
    return R;
}

matlab_mat *matlab_triu(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = i; j < n; ++j)
            R->data[i * n + j] = A->data[i * n + j];
    return R;
}

/* var(A): sample variance (N-1 denominator). std(A) = sqrt(var(A)). For
 * a vector, returns a 1x1; for a matrix, returns a 1xN row. */
matlab_mat *matlab_var(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m <= 1 || n == 1) {
        int64_t total = m * n;
        matlab_mat *R = mat_alloc(1, 1);
        if (total < 2) { R->data[0] = 0.0; return R; }
        double mean = 0.0;
        for (int64_t k = 0; k < total; ++k) mean += A->data[k];
        mean /= (double)total;
        double s = 0.0;
        for (int64_t k = 0; k < total; ++k) {
            double d = A->data[k] - mean;
            s += d * d;
        }
        R->data[0] = s / (double)(total - 1);
        return R;
    }
    matlab_mat *R = mat_alloc(1, n);
    for (int64_t j = 0; j < n; ++j) {
        double mean = 0.0;
        for (int64_t i = 0; i < m; ++i) mean += A->data[i * n + j];
        mean /= (double)m;
        double s = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            double d = A->data[i * n + j] - mean;
            s += d * d;
        }
        R->data[j] = (m > 1) ? s / (double)(m - 1) : 0.0;
    }
    return R;
}

matlab_mat *matlab_std(matlab_mat *A) {
    matlab_mat *V = matlab_var(A);
    int64_t total = V->rows * V->cols;
    for (int64_t k = 0; k < total; ++k) V->data[k] = sqrt(V->data[k]);
    return V;
}

/* Median by sort-and-pick on a scratch buffer. n*log(n) per column —
 * fine for the ~thousands-of-elements scripts the runtime targets. */
static int dbl_cmp(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}
static double median_of(double *buf, int64_t n) {
    if (n == 0) return 0.0;
    qsort(buf, (size_t)n, sizeof(double), dbl_cmp);
    if (n & 1) return buf[n / 2];
    return 0.5 * (buf[n / 2 - 1] + buf[n / 2]);
}
matlab_mat *matlab_median(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m <= 1 || n == 1) {
        int64_t total = m * n;
        matlab::runtime::MatPtr R = matlab::runtime::make_mat(1, 1);
        if (total == 0) return R.release();
        std::vector<double> buf(A->data, A->data + total);
        R->data[0] = median_of(buf.data(), total);
        return R.release();
    }
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(1, n);
    std::vector<double> buf(m);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) buf[i] = A->data[i * n + j];
        R->data[j] = median_of(buf.data(), m);
    }
    return R.release();
}

/* diff(A): first-order discrete differences. Vectors → vector of
 * length n-1 with the same orientation; matrices → diff down each
 * column, result is (m-1)xN. Empty if length < 2. */
matlab_mat *matlab_diff(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m <= 1 || n == 1) {
        int64_t total = m * n;
        if (total < 2) return mat_alloc(0, 0);
        int is_col = (n == 1 && m > 1);
        matlab_mat *R = is_col ? mat_alloc(total - 1, 1)
                               : mat_alloc(1, total - 1);
        for (int64_t k = 0; k < total - 1; ++k)
            R->data[k] = A->data[k + 1] - A->data[k];
        return R;
    }
    if (m < 2) return mat_alloc(0, n);
    matlab_mat *R = mat_alloc(m - 1, n);
    for (int64_t i = 0; i < m - 1; ++i)
        for (int64_t j = 0; j < n; ++j)
            R->data[i * n + j] = A->data[(i + 1) * n + j] - A->data[i * n + j];
    return R;
}

/* meshgrid(x, y): X(i,j) = x(j), Y(i,j) = y(i). Each output is
 * length(y) x length(x). ndgrid is the transpose convention:
 * X(i,j) = x(i), Y(i,j) = y(j); each output is length(x) x length(y).
 * The lowering pass splits [X,Y] = meshgrid(...) into two calls so
 * each runtime entry returns one matrix. */
static int64_t numel_(matlab_mat *v) { return v ? v->rows * v->cols : 0; }
matlab_mat *matlab_meshgrid_X(matlab_mat *x, matlab_mat *y) {
    int64_t nx = numel_(x), ny = numel_(y ? y : x);
    if (nx == 0) return mat_alloc(0, 0);
    if (!y) y = x;
    matlab_mat *X = mat_alloc(ny, nx);
    for (int64_t i = 0; i < ny; ++i)
        for (int64_t j = 0; j < nx; ++j)
            X->data[i * nx + j] = x->data[j];
    return X;
}
matlab_mat *matlab_meshgrid_Y(matlab_mat *x, matlab_mat *y) {
    int64_t nx = numel_(x);
    matlab_mat *src_y = y ? y : x;
    int64_t ny = numel_(src_y);
    if (nx == 0 || ny == 0) return mat_alloc(0, 0);
    matlab_mat *Y = mat_alloc(ny, nx);
    for (int64_t i = 0; i < ny; ++i)
        for (int64_t j = 0; j < nx; ++j)
            Y->data[i * nx + j] = src_y->data[i];
    return Y;
}
matlab_mat *matlab_ndgrid_X(matlab_mat *x, matlab_mat *y) {
    int64_t nx = numel_(x);
    matlab_mat *src_y = y ? y : x;
    int64_t ny = numel_(src_y);
    if (nx == 0) return mat_alloc(0, 0);
    matlab_mat *X = mat_alloc(nx, ny);
    for (int64_t i = 0; i < nx; ++i)
        for (int64_t j = 0; j < ny; ++j)
            X->data[i * ny + j] = x->data[i];
    return X;
}
matlab_mat *matlab_ndgrid_Y(matlab_mat *x, matlab_mat *y) {
    int64_t nx = numel_(x);
    matlab_mat *src_y = y ? y : x;
    int64_t ny = numel_(src_y);
    if (nx == 0 || ny == 0) return mat_alloc(0, 0);
    matlab_mat *Y = mat_alloc(nx, ny);
    for (int64_t i = 0; i < nx; ++i)
        for (int64_t j = 0; j < ny; ++j)
            Y->data[i * ny + j] = src_y->data[j];
    return Y;
}

/*=========================================================================
 * Tier-2 builtins: xcorr, polyval, polyfit, roots, interp1, trapz,
 * cumtrapz, gradient, hamming, hann, blackman.
 *=========================================================================*/

/* xcorr(u, v) — full cross-correlation as a row vector of length 2L-1
 * with L = max(numel(u), numel(v)) and lag-zero at index L (1-based).
 *
 * Definition: r[k] = sum_n u[n+k] * v[n], k in {-(L-1)..(L-1)}.
 * Equivalent to conv(u, fliplr(v)) of full shape, after promoting both
 * to length L by zero-padding the shorter one. The shorter side is
 * padded so the output index 0 corresponds to the most negative lag,
 * matching MATLAB's lag-axis convention. */
matlab_mat *matlab_xcorr(matlab_mat *u, matlab_mat *v) {
    if (!u || !v) return mat_alloc(0, 0);
    int64_t nu = u->rows * u->cols;
    int64_t nv = v->rows * v->cols;
    if (nu == 0 || nv == 0) return mat_alloc(0, 0);
    int64_t L = nu > nv ? nu : nv;
    int64_t out_n = 2 * L - 1;
    matlab_mat *R = mat_alloc(1, out_n);
    /* k = lag, ranging over [-(L-1), L-1]. r[k+L-1] is the output cell. */
    for (int64_t k = -(L - 1); k <= L - 1; ++k) {
        double s = 0.0;
        /* Sum over n where both u[n+k] and v[n] are in range. The vectors
         * are treated as length-L sequences (the shorter one padded with
         * zeros above its actual length). */
        int64_t n_lo = k > 0 ? 0 : -k;
        int64_t n_hi_u = (nu - 1) - k;          /* n+k must be < nu */
        int64_t n_hi_v = nv - 1;                /* n   must be < nv */
        int64_t n_hi = n_hi_u < n_hi_v ? n_hi_u : n_hi_v;
        for (int64_t n = n_lo; n <= n_hi; ++n)
            s += u->data[n + k] * v->data[n];
        R->data[k + L - 1] = s;
    }
    return R;
}

/* polyval(p, x) — Horner's method, applied elementwise. p is a vector
 * of coefficients with p[0] the highest power. Output mirrors x's
 * shape. Empty p or empty x returns 0×0. */
matlab_mat *matlab_polyval(matlab_mat *p, matlab_mat *x) {
    if (!p || !x) return mat_alloc(0, 0);
    int64_t np = p->rows * p->cols;
    int64_t nx = x->rows * x->cols;
    if (np == 0 || nx == 0) return mat_alloc(0, 0);
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    for (int64_t i = 0; i < nx; ++i) {
        double acc = p->data[0];
        for (int64_t k = 1; k < np; ++k)
            acc = acc * x->data[i] + p->data[k];
        Y->data[i] = acc;
    }
    return Y;
}

/* polyfit(x, y, n) — least-squares polynomial fit of degree n via
 * normal equations on the Vandermonde matrix. Returns a row vector
 * of length n+1, in MATLAB's highest-power-first order.
 *
 * Solves (V'V) p = V'y where V[i, k] = x[i]^(n-k). Direct LU on the
 * (n+1)×(n+1) normal-equation matrix is fine for the degrees this
 * runtime targets (typically n <= 8). */
matlab_mat *matlab_polyfit(matlab_mat *x, matlab_mat *y, double n_d) {
    if (!x || !y) return mat_alloc(0, 0);
    int64_t m = x->rows * x->cols;
    if (m == 0 || (y->rows * y->cols) != m) return mat_alloc(0, 0);
    int64_t n = (int64_t)n_d;
    if (n < 0) n = 0;
    int64_t k = n + 1;             /* number of coefficients */
    /* Phase-4 RAII: the three scratch buffers (V, A, b) become
     * std::vectors, dropping six manual frees and the singular-path
     * leak that previously freed-then-returned-fresh-mat_alloc. */
    std::vector<double> V(m * k);
    for (int64_t i = 0; i < m; ++i) {
        double xv = x->data[i];
        double pw = 1.0;
        for (int64_t j = k - 1; j >= 0; --j) {
            V[i * k + j] = pw;
            pw *= xv;
        }
    }
    /* Form A = V'V (k x k) and b = V'y (k). */
    std::vector<double> A(k * k, 0.0);
    std::vector<double> b(k, 0.0);
    for (int64_t r = 0; r < k; ++r) {
        for (int64_t c = 0; c < k; ++c) {
            double s = 0.0;
            for (int64_t i = 0; i < m; ++i) s += V[i * k + r] * V[i * k + c];
            A[r * k + c] = s;
        }
        double s = 0.0;
        for (int64_t i = 0; i < m; ++i) s += V[i * k + r] * y->data[i];
        b[r] = s;
    }
    /* Gaussian elimination with partial pivoting (k <= ~10 in practice). */
    for (int64_t i = 0; i < k; ++i) {
        int64_t pivot = i;
        double best = fabs(A[i * k + i]);
        for (int64_t r = i + 1; r < k; ++r) {
            double v = fabs(A[r * k + i]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-300) {
            /* Singular — return zeros. RAII frees the scratch on the
             * way out. */
            return mat_alloc(1, k);
        }
        if (pivot != i) {
            for (int64_t c = 0; c < k; ++c) {
                double t = A[i * k + c]; A[i * k + c] = A[pivot * k + c];
                A[pivot * k + c] = t;
            }
            double t = b[i]; b[i] = b[pivot]; b[pivot] = t;
        }
        for (int64_t r = i + 1; r < k; ++r) {
            double f = A[r * k + i] / A[i * k + i];
            for (int64_t c = i; c < k; ++c) A[r * k + c] -= f * A[i * k + c];
            b[r] -= f * b[i];
        }
    }
    matlab::runtime::MatPtr P = matlab::runtime::make_mat(1, k);
    for (int64_t i = k - 1; i >= 0; --i) {
        double s = b[i];
        for (int64_t c = i + 1; c < k; ++c) s -= A[i * k + c] * P->data[c];
        P->data[i] = s / A[i * k + i];
    }
    return P.release();
}

/* roots(p): defined after mat_c_alloc — see "Tier-2 roots" block below. */

/* polyder(p) — derivative of the polynomial whose coefficients (highest
 * power first) are p. Returns a row vector of length max(np-1, 1). For a
 * scalar input (constant), returns [0]. */
matlab_mat *matlab_polyder(matlab_mat *p) {
    if (!p) return mat_alloc(0, 0);
    int64_t np = p->rows * p->cols;
    if (np == 0) return mat_alloc(0, 0);
    if (np == 1) {
        matlab_mat *D = mat_alloc(1, 1);
        D->data[0] = 0.0;
        return D;
    }
    matlab_mat *D = mat_alloc(1, np - 1);
    /* p[0] is x^(np-1); derivative coefficient is (np-1)*p[0], etc. */
    for (int64_t i = 0; i < np - 1; ++i) {
        double power = (double)(np - 1 - i);
        D->data[i] = power * p->data[i];
    }
    return D;
}

/* polyint(p, k) — antiderivative of p, with constant-of-integration k
 * appended as the new x^0 term. Returns a row vector of length np+1.
 * polyint(p) sets k = 0. */
static matlab_mat *polyint_impl(matlab_mat *p, double k) {
    if (!p) return mat_alloc(0, 0);
    int64_t np = p->rows * p->cols;
    if (np == 0) return mat_alloc(0, 0);
    matlab_mat *I = mat_alloc(1, np + 1);
    /* p[0] is x^(np-1); integral coefficient is p[0] / np for x^np, etc. */
    for (int64_t i = 0; i < np; ++i) {
        double newpow = (double)(np - i);
        I->data[i] = p->data[i] / newpow;
    }
    I->data[np] = k;
    return I;
}
matlab_mat *matlab_polyint(matlab_mat *p)             { return polyint_impl(p, 0.0); }
matlab_mat *matlab_polyint_k(matlab_mat *p, double k) { return polyint_impl(p, k); }

/* interp1(x, y, xi) — 1-D linear interpolation. x must be sorted
 * ascending. Out-of-range xi values produce NaN (MATLAB default). */
matlab_mat *matlab_interp1(matlab_mat *x, matlab_mat *y, matlab_mat *xi) {
    if (!x || !y || !xi) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    int64_t ny = y->rows * y->cols;
    int64_t m = xi->rows * xi->cols;
    if (n == 0 || n != ny || m == 0) return mat_alloc(0, 0);
    matlab_mat *Yi = mat_alloc(xi->rows, xi->cols);
    double xmin = x->data[0], xmax = x->data[n - 1];
    for (int64_t i = 0; i < m; ++i) {
        double q = xi->data[i];
        if (q < xmin || q > xmax) {
            Yi->data[i] = NAN;
            continue;
        }
        /* Binary search for the bracket. */
        int64_t lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int64_t mid = (lo + hi) / 2;
            if (x->data[mid] <= q) lo = mid;
            else hi = mid;
        }
        double x0 = x->data[lo], x1 = x->data[hi];
        double y0 = y->data[lo], y1 = y->data[hi];
        if (x1 == x0) Yi->data[i] = y0;
        else          Yi->data[i] = y0 + (y1 - y0) * (q - x0) / (x1 - x0);
    }
    return Yi;
}

/* trapz(y) — assumes unit spacing. trapz(x, y) — uses x. For a
 * vector input the result is a 1×1; for a matrix it's a 1×N row
 * (one integral per column). */
static double trapz_unit(const double *v, int64_t n) {
    if (n < 2) return 0.0;
    double s = 0.5 * (v[0] + v[n - 1]);
    for (int64_t i = 1; i < n - 1; ++i) s += v[i];
    return s;
}
static double trapz_xy_(const double *x, const double *y, int64_t n) {
    if (n < 2) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < n - 1; ++i)
        s += 0.5 * (x[i + 1] - x[i]) * (y[i] + y[i + 1]);
    return s;
}
matlab_mat *matlab_trapz(matlab_mat *y) {
    if (!y) return mat_alloc(0, 0);
    int64_t m = y->rows, n = y->cols;
    if (m <= 1 || n == 1) {
        int64_t total = m * n;
        matlab::runtime::MatPtr R = matlab::runtime::make_mat(1, 1);
        R->data[0] = trapz_unit(y->data, total);
        return R.release();
    }
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(1, n);
    std::vector<double> col(m);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) col[i] = y->data[i * n + j];
        R->data[j] = trapz_unit(col.data(), m);
    }
    return R.release();
}
matlab_mat *matlab_trapz_xy(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return mat_alloc(0, 0);
    int64_t nx = x->rows * x->cols;
    int64_t ym = y->rows, yn = y->cols;
    if (ym <= 1 || yn == 1) {
        int64_t total = ym * yn;
        if (total != nx) return mat_alloc(0, 0);
        matlab::runtime::MatPtr R = matlab::runtime::make_mat(1, 1);
        R->data[0] = trapz_xy_(x->data, y->data, total);
        return R.release();
    }
    if (nx != ym) return mat_alloc(0, 0);
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(1, yn);
    std::vector<double> col(ym);
    for (int64_t j = 0; j < yn; ++j) {
        for (int64_t i = 0; i < ym; ++i) col[i] = y->data[i * yn + j];
        R->data[j] = trapz_xy_(x->data, col.data(), ym);
    }
    return R.release();
}

/* cumtrapz(y) — running trapezoidal integral with leading zero,
 * unit spacing. Same shape as input. */
matlab_mat *matlab_cumtrapz(matlab_mat *y) {
    if (!y) return mat_alloc(0, 0);
    int64_t m = y->rows, n = y->cols;
    if (m <= 1 || n == 1) {
        int64_t total = m * n;
        matlab_mat *R = mat_alloc(y->rows, y->cols);
        if (total == 0) return R;
        R->data[0] = 0.0;
        for (int64_t i = 1; i < total; ++i)
            R->data[i] = R->data[i - 1] + 0.5 * (y->data[i - 1] + y->data[i]);
        return R;
    }
    matlab_mat *R = mat_alloc(m, n);
    for (int64_t j = 0; j < n; ++j) {
        R->data[0 * n + j] = 0.0;
        for (int64_t i = 1; i < m; ++i)
            R->data[i * n + j] = R->data[(i - 1) * n + j] +
                0.5 * (y->data[(i - 1) * n + j] + y->data[i * n + j]);
    }
    return R;
}

/* gradient(f) — central differences in the interior, one-sided at
 * the endpoints. Same shape as the input. For matrices, takes the
 * gradient down each column (matching MATLAB's single-output form). */
static void gradient_1d(const double *v, double *g, int64_t n) {
    if (n == 0) return;
    if (n == 1) { g[0] = 0.0; return; }
    g[0]     = v[1] - v[0];
    g[n - 1] = v[n - 1] - v[n - 2];
    for (int64_t i = 1; i < n - 1; ++i)
        g[i] = 0.5 * (v[i + 1] - v[i - 1]);
}
matlab_mat *matlab_gradient(matlab_mat *f) {
    if (!f) return mat_alloc(0, 0);
    int64_t m = f->rows, n = f->cols;
    matlab::runtime::MatPtr G = matlab::runtime::make_mat(m, n);
    if (m == 0 || n == 0) return G.release();
    if (m <= 1 || n == 1) {
        gradient_1d(f->data, G->data, m * n);
        return G.release();
    }
    std::vector<double> col(m), out(m);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) col[i] = f->data[i * n + j];
        gradient_1d(col.data(), out.data(), m);
        for (int64_t i = 0; i < m; ++i) G->data[i * n + j] = out[i];
    }
    return G.release();
}

/* DSP windows. All return a column vector of length n. The MATLAB
 * reference uses the symmetric (non-periodic) form. */
matlab_mat *matlab_hamming(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i)
        W->data[i] = 0.54 - 0.46 * cos(2.0 * M_PI * (double)i / denom);
    return W;
}
matlab_mat *matlab_hann(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i)
        W->data[i] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / denom);
    return W;
}
matlab_mat *matlab_blackman(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double a = 2.0 * M_PI * (double)i / denom;
        W->data[i] = 0.42 - 0.5 * cos(a) + 0.08 * cos(2.0 * a);
    }
    return W;
}

/* Helper: parameterized cosine-sum window with up to 5 cosine terms.
 * w[i] = a0 - a1 cos(x) + a2 cos(2x) - a3 cos(3x) + a4 cos(4x),
 * x = 2*pi*i / (n-1). Used by nuttall, blackman-harris, flattop. */
static matlab_mat *cos_sum_window(int64_t n, const double a[5]) {
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double x = 2.0 * M_PI * (double)i / denom;
        W->data[i] = a[0]
                   - a[1] * cos(x)
                   + a[2] * cos(2.0 * x)
                   - a[3] * cos(3.0 * x)
                   + a[4] * cos(4.0 * x);
    }
    return W;
}

matlab_mat *matlab_rectwin(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) W->data[i] = 1.0;
    return W;
}

/* triang(n): symmetric triangular window. MATLAB's `triang` differs from
 * `bartlett` in that triang's endpoints are non-zero. */
matlab_mat *matlab_triang(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double half = (double)(n + 1) / 2.0;
    if (n % 2 == 1) {
        /* odd n */
        for (int64_t i = 0; i < n; ++i) {
            double k = (double)(i + 1);
            double v = (k <= half) ? (2.0 * k / (double)(n + 1))
                                   : (2.0 * ((double)(n + 1) - k) / (double)(n + 1));
            W->data[i] = v;
        }
    } else {
        /* even n */
        for (int64_t i = 0; i < n; ++i) {
            double k = (double)(i + 1);
            double v = (k <= (double)n / 2.0)
                          ? ((2.0 * k - 1.0) / (double)n)
                          : ((2.0 * ((double)n - k) + 1.0) / (double)n);
            W->data[i] = v;
        }
    }
    return W;
}

/* bartlett(n): triangular with zero endpoints. */
matlab_mat *matlab_bartlett(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double k = (double)i;
        W->data[i] = (k <= denom / 2.0) ? (2.0 * k / denom)
                                        : (2.0 * (denom - k) / denom);
    }
    return W;
}

/* barthannwin(n): modified Bartlett-Hann window. */
matlab_mat *matlab_barthannwin(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double t = (double)i / denom - 0.5;
        W->data[i] = 0.62 - 0.48 * fabs(t) + 0.38 * cos(2.0 * M_PI * t);
    }
    return W;
}

/* bohmanwin(n). */
matlab_mat *matlab_bohmanwin(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double x = fabs(2.0 * (double)i / denom - 1.0);
        W->data[i] = (1.0 - x) * cos(M_PI * x) + sin(M_PI * x) / M_PI;
    }
    /* MATLAB forces the endpoints to 0 to remove FP noise. */
    W->data[0] = 0.0;
    W->data[n - 1] = 0.0;
    return W;
}

/* parzenwin(n): de la Vallée Poussin window. */
matlab_mat *matlab_parzenwin(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double N = (double)n;
    for (int64_t i = 0; i < n; ++i) {
        double k = (double)i - (N - 1.0) / 2.0;   /* centred index */
        double a = fabs(k);
        double v;
        if (a <= N / 4.0) {
            double r = a / (N / 2.0);
            v = 1.0 - 6.0 * r * r + 6.0 * r * r * r;
        } else {
            double r = a / (N / 2.0);
            double t = 1.0 - r;
            v = 2.0 * t * t * t;
        }
        W->data[i] = v;
    }
    return W;
}

matlab_mat *matlab_nuttallwin(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    /* Nuttall's continuous-first-derivative coefficients. */
    const double a[5] = { 0.3635819, 0.4891775, 0.1365995, 0.0106411, 0.0 };
    return cos_sum_window(n, a);
}

matlab_mat *matlab_blackmanharris(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    const double a[5] = { 0.35875, 0.48829, 0.14128, 0.01168, 0.0 };
    return cos_sum_window(n, a);
}

matlab_mat *matlab_flattopwin(double n_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    /* MATLAB flattopwin coefficients (symmetric). */
    const double a[5] = { 0.21557895, 0.41663158, 0.277263158,
                          0.083578947, 0.006947368 };
    return cos_sum_window(n, a);
}

/* Modified Bessel I_0 via the standard series — converges fast for the
 * range relevant to Kaiser windows (|x| up to ~beta * pi). */
static double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double y = x * x / 4.0;
    for (int k = 1; k < 60; ++k) {
        term *= y / ((double)k * (double)k);
        sum += term;
        if (term < 1e-16 * sum) break;
    }
    return sum;
}

matlab_mat *matlab_kaiser(double n_d, double beta) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double denom = (double)(n - 1);
    double Ib = bessel_i0(beta);
    for (int64_t i = 0; i < n; ++i) {
        double r = 2.0 * (double)i / denom - 1.0;       /* in [-1, 1] */
        double arg = beta * sqrt(1.0 - r * r);
        W->data[i] = bessel_i0(arg) / Ib;
    }
    return W;
}

matlab_mat *matlab_tukeywin(double n_d, double r) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    if (r <= 0.0) {
        for (int64_t i = 0; i < n; ++i) W->data[i] = 1.0;
        return W;
    }
    if (r >= 1.0) {
        /* r = 1 -> Hann window. */
        double denom = (double)(n - 1);
        for (int64_t i = 0; i < n; ++i)
            W->data[i] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / denom);
        return W;
    }
    double denom = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double x = (double)i / denom;
        double v;
        if (x < r / 2.0) {
            v = 0.5 * (1.0 + cos(2.0 * M_PI / r * (x - r / 2.0)));
        } else if (x <= 1.0 - r / 2.0) {
            v = 1.0;
        } else {
            v = 0.5 * (1.0 + cos(2.0 * M_PI / r * (x - 1.0 + r / 2.0)));
        }
        W->data[i] = v;
    }
    return W;
}

matlab_mat *matlab_gausswin(double n_d, double alpha) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double half = (double)(n - 1) / 2.0;
    for (int64_t i = 0; i < n; ++i) {
        double t = ((double)i - half) / half;
        W->data[i] = exp(-0.5 * (alpha * t) * (alpha * t));
    }
    return W;
}

/* Chebyshev (Dolph-Chebyshev) window. r is the desired sidelobe
 * attenuation in dB. Implementation: evaluate the closed-form
 * frequency-domain response on the N-point grid and inverse-FFT
 * the result, then normalise. We piggyback on the runtime FFT —
 * but since we only have radix-2 / Bluestein for complex inputs,
 * we synthesise the spectrum as real and take a direct DFT
 * (O(N^2)) for portability. N is small in practice (window
 * lengths rarely exceed a few thousand) so this is acceptable. */
matlab_mat *matlab_chebwin(double n_d, double r) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double atten = pow(10.0, r / 20.0);  /* linear sidelobe ratio */
    double beta  = cosh(acosh(atten) / (double)(n - 1));
    /* Spectral samples W[k] = T_{N-1}( beta * cos(pi*k/N) ) / atten,
     * with T_m(x) = cos(m*acos(x)) when |x|<=1, cosh(m*acosh(x)) when |x|>1. */
    std::vector<double> spec(n);
    int N = (int)n;
    int M = N - 1;
    for (int k = 0; k < N; ++k) {
        double x = beta * cos(M_PI * (double)k / (double)N);
        double Tm;
        if (x > 1.0) Tm = cosh((double)M * acosh(x));
        else if (x < -1.0) Tm = ((M & 1) ? -1.0 : 1.0) * cosh((double)M * acosh(-x));
        else Tm = cos((double)M * acos(x));
        /* Apply alternating sign (frequency-shift) so the window is
         * centred — matches MATLAB's even/odd-N convention. */
        spec[k] = ((k & 1) ? -1.0 : 1.0) * Tm / atten;
    }
    /* Inverse real DFT via direct sum (O(N^2)). */
    for (int64_t i = 0; i < n; ++i) {
        double sum = spec[0];
        for (int k = 1; k < N; ++k)
            sum += 2.0 * spec[k] * cos(2.0 * M_PI * (double)k *
                                       ((double)i - (double)(N - 1) / 2.0)
                                       / (double)N);
        W->data[i] = sum;
    }
    /* MATLAB normalises so max(W) == 1. */
    double mx = W->data[0];
    for (int64_t i = 1; i < n; ++i) if (W->data[i] > mx) mx = W->data[i];
    if (mx > 0.0)
        for (int64_t i = 0; i < n; ++i) W->data[i] /= mx;
    return W;
}

/* Taylor window. nbar is the number of nearly-constant-level sidelobes;
 * sll is the desired sidelobe level in dB (negative number, e.g. -30). */
matlab_mat *matlab_taylorwin(double n_d, double nbar_d, double sll_d) {
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    int nbar = (int)nbar_d;
    if (nbar < 1) nbar = 4;     /* MATLAB default */
    double sll = sll_d != 0.0 ? sll_d : -30.0;  /* MATLAB default */
    matlab_mat *W = mat_alloc(n, 1);
    if (n == 1) { W->data[0] = 1.0; return W; }
    double R  = pow(10.0, -sll / 20.0);   /* linear */
    double A  = acosh(R) / M_PI;
    double s2 = (double)(nbar * nbar) / (A * A + ((double)nbar - 0.5) *
                                                  ((double)nbar - 0.5));
    /* Compute Taylor coefficients F_m for m = 1..nbar-1. */
    std::vector<double> F((size_t)nbar, 0.0);
    for (int m = 1; m < nbar; ++m) {
        double num = 1.0, den = 1.0;
        for (int i = 1; i < nbar; ++i) {
            double t1 = 1.0 - (double)(m * m) /
                              (s2 * (A * A + ((double)i - 0.5) *
                                              ((double)i - 0.5)));
            num *= t1;
            if (i != m) {
                double t2 = 1.0 - (double)(m * m) / (double)(i * i);
                den *= t2;
            }
        }
        double sign = (m & 1) ? -1.0 : 1.0;
        F[(size_t)m] = sign * 0.5 * num / den;
    }
    /* Sample the Taylor window: w[k] = 1 + 2*sum_{m=1..nbar-1} F_m
     *                                          * cos(2*pi*m*(k-(N-1)/2)/N). */
    for (int64_t k = 0; k < n; ++k) {
        double sum = 1.0;
        double centred = (double)k - (double)(n - 1) / 2.0;
        for (int m = 1; m < nbar; ++m)
            sum += 2.0 * F[(size_t)m] *
                   cos(2.0 * M_PI * (double)m * centred / (double)n);
        W->data[k] = sum;
    }
    /* Normalise to unit peak. */
    double mx = W->data[0];
    for (int64_t i = 1; i < n; ++i) if (W->data[i] > mx) mx = W->data[i];
    if (mx > 0.0)
        for (int64_t i = 0; i < n; ++i) W->data[i] /= mx;
    return W;
}

/*===========================================================================
 * FIR design (Tier-1 SPT §2.2) — lowpass scope.
 *
 *   b = fir1(n, Wn)         windowed-sinc lowpass FIR (default Hamming)
 *   B = sgolay(k, f)        Savitzky-Golay projection matrix
 *   y = sgolayfilt(x, k, f) Savitzky-Golay smoothing filter
 *
 * fir1 returns a length-(n+1) row vector of impulse-response taps,
 * normalized for unit DC gain. sgolay/sgolayfilt use the standard
 * polynomial-fit approach: V is the (f × (k+1)) Vandermonde matrix
 * of centred indices, B = V (V'V)^-1 V' is the (f × f) projection
 * matrix onto the polynomial-fit space; sgolayfilt applies B's middle
 * row in steady state and the corresponding boundary rows at the edges.
 */
matlab_mat *matlab_fir1(double n_d, double Wn) {
    int n = (int)n_d;
    if (n < 0) n = 0;
    if (Wn <= 0.0) Wn = 1e-12;
    if (Wn >= 1.0) Wn = 1.0 - 1e-12;
    int L = n + 1;                          /* tap count */
    matlab_mat *B = mat_alloc(1, L);
    /* Ideal lowpass impulse response: h_d[k] = Wn * sinc(Wn * (k - n/2)).
     * Use MATLAB's normalized sinc(x) = sin(π·x)/(π·x), with the limit
     * sinc(0) = 1. */
    double centre = (double)n / 2.0;
    for (int k = 0; k < L; ++k) {
        double m = (double)k - centre;
        if (m == 0.0) {
            B->data[k] = Wn;
        } else {
            double arg = M_PI * Wn * m;
            B->data[k] = Wn * sin(arg) / arg;
        }
    }
    /* Default window: Hamming. Multiply elementwise. */
    if (L > 1) {
        double denom = (double)(L - 1);
        for (int k = 0; k < L; ++k) {
            double w = 0.54 - 0.46 * cos(2.0 * M_PI * (double)k / denom);
            B->data[k] *= w;
        }
    }
    /* Normalize for unit DC gain: sum(b) = 1. */
    double s = 0.0;
    for (int k = 0; k < L; ++k) s += B->data[k];
    if (s != 0.0)
        for (int k = 0; k < L; ++k) B->data[k] /= s;
    return B;
}

/* Solve a small linear system A x = b in place via Gaussian elimination
 * with partial pivoting. A is (n × n) row-major, b is (n) — both
 * mutated. Returns false if singular within tolerance. */
static bool sgolay_lu_solve_(double *A, double *b, int n) {
    for (int i = 0; i < n; ++i) {
        int piv = i;
        double best = fabs(A[i * n + i]);
        for (int r = i + 1; r < n; ++r) {
            double v = fabs(A[r * n + i]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return false;
        if (piv != i) {
            for (int c = 0; c < n; ++c) {
                double t = A[i * n + c];
                A[i * n + c] = A[piv * n + c];
                A[piv * n + c] = t;
            }
            double t = b[i]; b[i] = b[piv]; b[piv] = t;
        }
        for (int r = i + 1; r < n; ++r) {
            double f = A[r * n + i] / A[i * n + i];
            for (int c = i; c < n; ++c) A[r * n + c] -= f * A[i * n + c];
            b[r] -= f * b[i];
        }
    }
    /* Back-substitute. */
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int c = i + 1; c < n; ++c) s -= A[i * n + c] * b[c];
        b[i] = s / A[i * n + i];
    }
    return true;
}

/* Compute the (f × f) Savitzky-Golay projection matrix B such that
 * B applied to a length-f window of input gives the polynomial-fit
 * smoothed value. Caller-supplied buffer must be (f × f) row-major. */
static void compute_sgolay_matrix_(int k, int f, double *B) {
    int K = k + 1;
    /* Vandermonde V (f × K), centred at t = -(f-1)/2 .. (f-1)/2. */
    std::vector<double> V((size_t)f * (size_t)K);
    for (int i = 0; i < f; ++i) {
        double t = (double)i - (double)(f - 1) / 2.0;
        double pw = 1.0;
        for (int j = 0; j < K; ++j) {
            V[(size_t)i * (size_t)K + (size_t)j] = pw;
            pw *= t;
        }
    }
    /* G = V'V (K × K). */
    std::vector<double> G((size_t)K * (size_t)K, 0.0);
    for (int a = 0; a < K; ++a)
        for (int b = 0; b < K; ++b) {
            double s = 0.0;
            for (int i = 0; i < f; ++i)
                s += V[(size_t)i * K + a] * V[(size_t)i * K + b];
            G[(size_t)a * K + b] = s;
        }
    /* Solve G X = V' column-by-column to get X = (V'V)^-1 V'
     * (K × f). Then B = V * X (f × f). */
    std::vector<double> X((size_t)K * (size_t)f);
    std::vector<double> Gtmp((size_t)K * (size_t)K);
    std::vector<double> rhs((size_t)K);
    for (int j = 0; j < f; ++j) {
        memcpy(Gtmp.data(), G.data(), (size_t)K * K * sizeof(double));
        for (int a = 0; a < K; ++a) rhs[a] = V[(size_t)j * K + a];
        sgolay_lu_solve_(Gtmp.data(), rhs.data(), K);
        for (int a = 0; a < K; ++a) X[(size_t)a * f + j] = rhs[a];
    }
    /* B = V (f × K) * X (K × f) -> (f × f). */
    for (int i = 0; i < f; ++i)
        for (int j = 0; j < f; ++j) {
            double s = 0.0;
            for (int a = 0; a < K; ++a)
                s += V[(size_t)i * K + a] * X[(size_t)a * f + j];
            B[(size_t)i * f + j] = s;
        }
}

matlab_mat *matlab_sgolay(double k_d, double f_d) {
    int k = (int)k_d, f = (int)f_d;
    if (f < 1) f = 1;
    if (k < 0) k = 0;
    if (k >= f) k = f - 1;
    /* Frame length must be odd. */
    if ((f & 1) == 0) f++;
    matlab_mat *B = mat_alloc(f, f);
    compute_sgolay_matrix_(k, f, B->data);
    return B;
}

matlab_mat *matlab_sgolayfilt(matlab_mat *x, double k_d, double f_d) {
    if (!x) return mat_alloc(0, 0);
    int k = (int)k_d, f = (int)f_d;
    if (f < 1) f = 1;
    if (k < 0) k = 0;
    if (k >= f) k = f - 1;
    if ((f & 1) == 0) f++;
    int64_t N = x->rows * x->cols;
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    if (N < f) {
        /* Frame longer than data: just copy through. */
        memcpy(Y->data, x->data, (size_t)N * sizeof(double));
        return Y;
    }
    std::vector<double> Bm((size_t)f * (size_t)f);
    compute_sgolay_matrix_(k, f, Bm.data());
    int half = (f - 1) / 2;
    /* Edges: rows 0..half-1 of B applied to x[0..f-1].
     * Steady state: middle row applied to a sliding f-window. */
    for (int i = 0; i < half; ++i) {
        double s = 0.0;
        for (int j = 0; j < f; ++j) s += Bm[(size_t)i * f + j] * x->data[j];
        Y->data[i] = s;
    }
    for (int64_t i = half; i < N - half; ++i) {
        double s = 0.0;
        for (int j = 0; j < f; ++j)
            s += Bm[(size_t)half * f + j] * x->data[i - half + j];
        Y->data[i] = s;
    }
    /* Right-edge rows: half+1..f-1 applied to x[N-f..N-1]. */
    for (int i = 0; i < half; ++i) {
        int row = half + 1 + i;
        double s = 0.0;
        for (int j = 0; j < f; ++j)
            s += Bm[(size_t)row * f + j] * x->data[N - f + j];
        Y->data[N - half + i] = s;
    }
    return Y;
}

/* Forward decl — filter_flat_ is defined later in this file (§2.5 helpers). */
static void filter_flat_(const double *b, int64_t nb,
                         const double *a, int64_t na,
                         const double *x, int64_t nx,
                         double *y);

/*===========================================================================
 * Tier-3 SPT §4.4 alignment helpers — xcov / finddelay / dtw.
 *
 *   c = xcov(x, y)         mean-removed cross-correlation
 *   d = finddelay(x, y)    integer lag d s.t. y[n] ≈ x[n − d]
 *   D = dtw(x, y)          dynamic-time-warping distance (scalar)
 *
 * alignsignals (multi-return) and gccphat are deferred to follow-on.
 * xcorr scaling-option strings ('biased'/'unbiased'/'normalized'/...)
 * also deferred — needs string-flag dispatch.
 */
matlab_mat *matlab_xcov(matlab_mat *x, matlab_mat *y);   /* fwd decl */
matlab_mat *matlab_xcorr(matlab_mat *u, matlab_mat *v);  /* shipped earlier */

matlab_mat *matlab_xcov(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return mat_alloc(0, 0);
    int64_t Nx = x->rows * x->cols;
    int64_t Ny = y->rows * y->cols;
    if (Nx == 0 || Ny == 0) return mat_alloc(0, 0);
    /* Compute means and a copy of x/y with mean subtracted. */
    double mx = 0, my = 0;
    for (int64_t i = 0; i < Nx; ++i) mx += x->data[i];
    for (int64_t i = 0; i < Ny; ++i) my += y->data[i];
    mx /= (double)Nx; my /= (double)Ny;
    matlab_mat *xm = mat_alloc(x->rows, x->cols);
    matlab_mat *ym = mat_alloc(y->rows, y->cols);
    for (int64_t i = 0; i < Nx; ++i) xm->data[i] = x->data[i] - mx;
    for (int64_t i = 0; i < Ny; ++i) ym->data[i] = y->data[i] - my;
    matlab_mat *C = matlab_xcorr(xm, ym);
    free(xm->data); free(xm);
    free(ym->data); free(ym);
    return C;
}

double matlab_finddelay_s(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return 0.0;
    int64_t Nx = x->rows * x->cols;
    int64_t Ny = y->rows * y->cols;
    if (Nx == 0 || Ny == 0) return 0.0;
    matlab_mat *C = matlab_xcorr(x, y);
    int64_t Nc = C->rows * C->cols;
    int64_t imax = 0;
    double  vmax = fabs(C->data[0]);
    for (int64_t i = 1; i < Nc; ++i) {
        double v = fabs(C->data[i]);
        if (v > vmax) { vmax = v; imax = i; }
    }
    free(C->data); free(C);
    /* Lag = imax − (N − 1) where N is the larger of (Nx, Ny). */
    int64_t N = Nx > Ny ? Nx : Ny;
    return (double)(imax - (N - 1));
}

double matlab_dtw_s(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return 0.0;
    int64_t Nx = x->rows * x->cols;
    int64_t Ny = y->rows * y->cols;
    if (Nx == 0 || Ny == 0) return 0.0;
    /* DP grid: D[i][j] = |x[i] − y[j]| + min(D[i−1][j], D[i][j−1], D[i−1][j−1]). */
    std::vector<double> D((size_t)(Nx * Ny), 0.0);
    auto IDX = [&](int64_t i, int64_t j) { return (size_t)(i * Ny + j); };
    D[IDX(0, 0)] = fabs(x->data[0] - y->data[0]);
    for (int64_t j = 1; j < Ny; ++j)
        D[IDX(0, j)] = D[IDX(0, j - 1)] + fabs(x->data[0] - y->data[j]);
    for (int64_t i = 1; i < Nx; ++i)
        D[IDX(i, 0)] = D[IDX(i - 1, 0)] + fabs(x->data[i] - y->data[0]);
    for (int64_t i = 1; i < Nx; ++i) {
        for (int64_t j = 1; j < Ny; ++j) {
            double a = D[IDX(i - 1, j)];
            double b = D[IDX(i, j - 1)];
            double c = D[IDX(i - 1, j - 1)];
            double m = a < b ? a : b;
            if (c < m) m = c;
            D[IDX(i, j)] = m + fabs(x->data[i] - y->data[j]);
        }
    }
    return D[IDX(Nx - 1, Ny - 1)];
}

/*===========================================================================
 * Tier-3 SPT §4.2 waveform generators — chirp / sawtooth / square / pulses.
 *
 *   y = chirp(t, f0, t1, f1)  linear-method chirp (cosine)
 *   y = sawtooth(t, w)        sawtooth wave of period 2π, width w (0..1)
 *   y = square(t, duty)       square wave, duty in percent (0..100)
 *   y = gauspuls(t, fc, bw)   Gaussian-modulated sinusoidal pulse
 *   y = rectpuls(t, w)        rectangular pulse of width w (centred at 0)
 *   y = tripuls(t, w)         triangular pulse of width w (centred at 0)
 *   y = sinc(x)               sin(π·x) / (π·x), sinc(0) = 1
 *
 * Output is same-shape as t/x. Default-arg shorthands (e.g. sawtooth(t)
 * with implicit w=1) are deferred to a follow-on slice.
 */
matlab_mat *matlab_chirp(matlab_mat *t, double f0, double t1, double f1) {
    if (!t) return mat_alloc(0, 0);
    int64_t N = t->rows * t->cols;
    matlab_mat *Y = mat_alloc(t->rows, t->cols);
    if (t1 <= 0.0) t1 = 1.0;
    double k = (f1 - f0) / t1;
    for (int64_t i = 0; i < N; ++i) {
        double tau = t->data[i];
        double phi = 2.0 * M_PI * (f0 * tau + 0.5 * k * tau * tau);
        Y->data[i] = cos(phi);
    }
    return Y;
}

matlab_mat *matlab_sawtooth(matlab_mat *t, double w) {
    if (!t) return mat_alloc(0, 0);
    int64_t N = t->rows * t->cols;
    matlab_mat *Y = mat_alloc(t->rows, t->cols);
    if (w < 0.0) w = 0.0;
    if (w > 1.0) w = 1.0;
    for (int64_t i = 0; i < N; ++i) {
        /* Map t to [0, 2π) modulo period. */
        double tau = t->data[i] / (2.0 * M_PI);
        tau -= floor(tau);
        if (tau < w) {
            Y->data[i] = (w > 0.0) ? (-1.0 + 2.0 * tau / w) : 0.0;
        } else {
            Y->data[i] = (w < 1.0) ? (1.0 - 2.0 * (tau - w) / (1.0 - w)) : 0.0;
        }
    }
    return Y;
}

matlab_mat *matlab_square(matlab_mat *t, double duty) {
    if (!t) return mat_alloc(0, 0);
    int64_t N = t->rows * t->cols;
    matlab_mat *Y = mat_alloc(t->rows, t->cols);
    double dfrac = duty / 100.0;
    if (dfrac < 0.0) dfrac = 0.0;
    if (dfrac > 1.0) dfrac = 1.0;
    for (int64_t i = 0; i < N; ++i) {
        double tau = t->data[i] / (2.0 * M_PI);
        tau -= floor(tau);
        Y->data[i] = (tau < dfrac) ? 1.0 : -1.0;
    }
    return Y;
}

matlab_mat *matlab_gauspuls(matlab_mat *t, double fc, double bw) {
    if (!t) return mat_alloc(0, 0);
    int64_t N = t->rows * t->cols;
    matlab_mat *Y = mat_alloc(t->rows, t->cols);
    /* Standard MATLAB gauspuls: alpha set so the spectrum has -6dB
     * fractional bandwidth bw. alpha = (π·fc·bw)² / (4·log(2)). */
    double a = (M_PI * fc * bw);
    a = (a * a) / (4.0 * log(2.0));
    for (int64_t i = 0; i < N; ++i) {
        double tau = t->data[i];
        Y->data[i] = exp(-a * tau * tau) * cos(2.0 * M_PI * fc * tau);
    }
    return Y;
}

matlab_mat *matlab_rectpuls(matlab_mat *t, double w) {
    if (!t) return mat_alloc(0, 0);
    int64_t N = t->rows * t->cols;
    matlab_mat *Y = mat_alloc(t->rows, t->cols);
    double half = w * 0.5;
    for (int64_t i = 0; i < N; ++i) {
        double a = fabs(t->data[i]);
        Y->data[i] = (a < half) ? 1.0 : (a == half ? 0.5 : 0.0);
    }
    return Y;
}

matlab_mat *matlab_tripuls(matlab_mat *t, double w) {
    if (!t) return mat_alloc(0, 0);
    int64_t N = t->rows * t->cols;
    matlab_mat *Y = mat_alloc(t->rows, t->cols);
    double half = w * 0.5;
    for (int64_t i = 0; i < N; ++i) {
        double a = fabs(t->data[i]);
        Y->data[i] = (a < half) ? (1.0 - a / half) : 0.0;
    }
    return Y;
}

matlab_mat *matlab_sinc(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    for (int64_t i = 0; i < N; ++i) {
        double v = x->data[i];
        if (v == 0.0) Y->data[i] = 1.0;
        else { double a = M_PI * v; Y->data[i] = sin(a) / a; }
    }
    return Y;
}

/*===========================================================================
 * Tier-3 SPT §4.1 real multirate — upfirdn / decimate / interp / resample.
 *
 *   y = upfirdn(x, h, p, q)   upsample-by-p → FIR-filter-with-h → downsample-by-q
 *   y = decimate(x, r)        lowpass + downsample-by-r (FIR default)
 *   y = interp(x, r)          upsample-by-r + lowpass (unit-gain interpolation)
 *   y = resample(x, p, q)     polyphase resampling (combined direct algo)
 *
 * decimate / interp / resample build a default lowpass FIR via fir1
 * (Hamming-windowed sinc). Output lengths match MATLAB convention:
 *   decimate: ceil(N / r)      take every r-th of filtered signal
 *   interp:   N * r            r-1 zero-stuffing + lowpass × r gain
 *   resample: ceil(N · p / q)  upsample × p, lowpass, downsample × q
 *
 * The toy `upsample` / `downsample` stubs (zero-stuff / decimate without
 * anti-aliasing) remain for backwards-compat — these are the proper
 * anti-aliased versions. Polyphase decomposition (`polyphase(b, m)`)
 * is a follow-on.
 */
/* upfirdn — supports both real and complex `x`.  The pulse-shape
 * filter `h` is always real (no complex-filter case in MATLAB's
 * usage).  When `x` is complex, the I and Q channels filter
 * independently against the same `h` taps, producing a complex
 * output of matching dimensions.
 *
 * Detect input kind via the layout magic so the same MATLAB-level
 * `upfirdn` entry routes either way without the caller having to
 * pick a separate runtime symbol.  Mirrors how `awgn` / `scatterplot`
 * etc. dispatch on `mat_is_complex(x)` internally. */
void *matlab_upfirdn(void *x_any, matlab_mat *h,
                     double p_d, double q_d) {
    if (!x_any || !h) return mat_alloc(0, 0);
    int p = (int)p_d, q = (int)q_d;
    if (p < 1) p = 1;
    if (q < 1) q = 1;
    int64_t Nh = h->rows * h->cols;
    if (mat_is_complex(x_any)) {
        const matlab_mat_c *xc = (const matlab_mat_c *)x_any;
        int64_t Nx = xc->rows * xc->cols;
        if (Nx == 0 || Nh == 0) return mat_c_alloc(1, 0);
        int64_t N_filtered = Nx * p + Nh - 1;
        int64_t Ny = (N_filtered + q - 1) / q;
        matlab_mat_c *Y = (xc->cols == 1 && xc->rows > 1)
                            ? mat_c_alloc(Ny, 1)
                            : mat_c_alloc(1, Ny);
        for (int64_t m = 0; m < Ny; ++m) {
            double sre = 0.0, sim = 0.0;
            int64_t k = m * q;
            for (int64_t n = 0; n < Nx; ++n) {
                int64_t hi = k - n * p;
                if (hi >= 0 && hi < Nh) {
                    double tap = h->data[hi];
                    sre += xc->re[n] * tap;
                    sim += xc->im[n] * tap;
                }
            }
            Y->re[m] = sre;
            Y->im[m] = sim;
        }
        return Y;
    }
    matlab_mat *x = (matlab_mat *)x_any;
    int64_t Nx = x->rows * x->cols;
    if (Nx == 0 || Nh == 0) return mat_alloc(1, 0);
    /* Output length: full convolution Nx*p + Nh - 1, then ceil-div by q. */
    int64_t N_filtered = Nx * p + Nh - 1;
    int64_t Ny = (N_filtered + q - 1) / q;
    /* Preserve column-shape if input was column. */
    matlab_mat *Y = (x->cols == 1 && x->rows > 1) ? mat_alloc(Ny, 1)
                                                   : mat_alloc(1, Ny);
    for (int64_t m = 0; m < Ny; ++m) {
        double sum = 0.0;
        int64_t k = m * q;
        for (int64_t n = 0; n < Nx; ++n) {
            int64_t hi = k - n * p;
            if (hi >= 0 && hi < Nh) sum += x->data[n] * h->data[hi];
        }
        Y->data[m] = sum;
    }
    return Y;
}

matlab_mat *matlab_decimate(matlab_mat *x, double r_d) {
    if (!x) return mat_alloc(0, 0);
    int r = (int)r_d;
    if (r < 1) r = 1;
    int64_t Nx = x->rows * x->cols;
    int64_t Ny = (Nx + r - 1) / r;
    matlab_mat *Y = (x->cols == 1 && x->rows > 1) ? mat_alloc(Ny, 1)
                                                   : mat_alloc(1, Ny);
    if (r == 1) {
        memcpy(Y->data, x->data, (size_t)Nx * sizeof(double));
        return Y;
    }
    if (Nx == 0) return Y;
    /* Default: 30-tap Hamming-windowed lowpass at 0.8/r (safety margin
     * below the new Nyquist of 1/r). */
    matlab_mat *b = matlab_fir1(30.0, 0.8 / (double)r);
    int64_t Nb = b->rows * b->cols;
    std::vector<double> bn((size_t)Nb), an(1, 1.0);
    for (int64_t i = 0; i < Nb; ++i) bn[(size_t)i] = b->data[i];
    free(b->data); free(b);
    /* Apply causal filter via the existing direct-form-II-T helper. */
    std::vector<double> y_filt((size_t)Nx);
    filter_flat_(bn.data(), Nb, an.data(), 1, x->data, Nx, y_filt.data());
    /* Take every r-th sample starting from index 0. */
    for (int64_t i = 0; i < Ny; ++i) Y->data[i] = y_filt[(size_t)(i * r)];
    return Y;
}

matlab_mat *matlab_interp(matlab_mat *x, double r_d) {
    if (!x) return mat_alloc(0, 0);
    int r = (int)r_d;
    if (r < 1) r = 1;
    int64_t Nx = x->rows * x->cols;
    int64_t Ny = Nx * r;
    matlab_mat *Y = (x->cols == 1 && x->rows > 1) ? mat_alloc(Ny, 1)
                                                   : mat_alloc(1, Ny);
    if (r == 1) {
        memcpy(Y->data, x->data, (size_t)Nx * sizeof(double));
        return Y;
    }
    if (Nx == 0) return Y;
    /* Zero-stuff to length Nx·r. */
    std::vector<double> y_up((size_t)Ny, 0.0);
    for (int64_t i = 0; i < Nx; ++i) y_up[(size_t)(i * r)] = x->data[i];
    /* MATLAB's interp default is a length-(2·4·r+1) Hamming-windowed
     * lowpass at Wn = 1/r, scaled by r for unit-gain interpolation. */
    int filt_order = 8 * r;
    matlab_mat *b = matlab_fir1((double)filt_order, 1.0 / (double)r);
    int64_t Nb = b->rows * b->cols;
    std::vector<double> bn((size_t)Nb), an(1, 1.0);
    for (int64_t i = 0; i < Nb; ++i) bn[(size_t)i] = (double)r * b->data[i];
    free(b->data); free(b);
    filter_flat_(bn.data(), Nb, an.data(), 1, y_up.data(), Ny, Y->data);
    return Y;
}

matlab_mat *matlab_resample(matlab_mat *x, double p_d, double q_d) {
    if (!x) return mat_alloc(0, 0);
    int p = (int)p_d, q = (int)q_d;
    if (p < 1) p = 1;
    if (q < 1) q = 1;
    int64_t Nx = x->rows * x->cols;
    int64_t Ny = (Nx * p + q - 1) / q;
    matlab_mat *Y = (x->cols == 1 && x->rows > 1) ? mat_alloc(Ny, 1)
                                                   : mat_alloc(1, Ny);
    if (p == 1 && q == 1) {
        memcpy(Y->data, x->data, (size_t)Nx * sizeof(double));
        return Y;
    }
    if (Nx == 0) return Y;
    /* Anti-alias filter at the lower of the two Nyquist limits. */
    double Wn = (p >= q) ? (1.0 / (double)p) : (1.0 / (double)q);
    int M = (p > q) ? p : q;
    int filt_order = 8 * M;
    matlab_mat *b = matlab_fir1((double)filt_order, Wn);
    int64_t Nb = b->rows * b->cols;
    std::vector<double> hn((size_t)Nb);
    for (int64_t i = 0; i < Nb; ++i) hn[(size_t)i] = (double)p * b->data[i];
    free(b->data); free(b);
    /* Direct upfirdn-style algorithm: y[m] = sum_n x[n] · h[m·q − n·p]. */
    for (int64_t m = 0; m < Ny; ++m) {
        double sum = 0.0;
        int64_t k = m * q;
        for (int64_t n = 0; n < Nx; ++n) {
            int64_t hi = k - n * p;
            if (hi >= 0 && hi < Nb) sum += hn[(size_t)hi] * x->data[n];
        }
        Y->data[m] = sum;
    }
    return Y;
}

/*===========================================================================
 * Tier-3 SPT §4.3 pulse measurements — findpeaks + scalar reductions.
 *
 *   pks      = findpeaks(x)        local maxima (1-return)
 *   [p, lc]  = findpeaks(x)        peaks + locations (2-return)
 *   rms(x), peak2peak(x), peak2rms(x), rssq(x)   scalar reductions.
 *
 * findpeaks: a sample x[i] (1-based MATLAB index) is a peak iff
 * 1 < i < N AND x[i-1] < x[i] AND x[i] > x[i+1]. Endpoints and
 * plateaus are excluded — matches MATLAB's strict-monotonic
 * definition. Output column lengths = number of peaks found.
 *
 * MinPeakHeight, MinPeakDistance, MinPeakProminence, Threshold,
 * SortStr — name-value pairs deferred to a follow-on slice.
 */
matlab_mat *matlab_findpeaks_pks(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N < 3) return mat_alloc(0, 1);
    std::vector<double> pks;
    for (int64_t i = 1; i < N - 1; ++i)
        if (x->data[i - 1] < x->data[i] && x->data[i] > x->data[i + 1])
            pks.push_back(x->data[i]);
    int64_t M = (int64_t)pks.size();
    matlab_mat *P = mat_alloc(M, M > 0 ? 1 : 0);
    for (int64_t i = 0; i < M; ++i) P->data[i] = pks[i];
    return P;
}

matlab_mat *matlab_findpeaks_locs(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N < 3) return mat_alloc(0, 1);
    std::vector<double> locs;
    for (int64_t i = 1; i < N - 1; ++i)
        if (x->data[i - 1] < x->data[i] && x->data[i] > x->data[i + 1])
            locs.push_back((double)(i + 1));    /* MATLAB 1-based */
    int64_t M = (int64_t)locs.size();
    matlab_mat *L = mat_alloc(M, M > 0 ? 1 : 0);
    for (int64_t i = 0; i < M; ++i) L->data[i] = locs[i];
    return L;
}

double matlab_rms_s(matlab_mat *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N == 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < N; ++i) s += x->data[i] * x->data[i];
    return sqrt(s / (double)N);
}

double matlab_peak2peak_s(matlab_mat *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N == 0) return 0.0;
    double mn = x->data[0], mx = x->data[0];
    for (int64_t i = 1; i < N; ++i) {
        if (x->data[i] < mn) mn = x->data[i];
        if (x->data[i] > mx) mx = x->data[i];
    }
    return mx - mn;
}

double matlab_peak2rms_s(matlab_mat *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N == 0) return 0.0;
    double s = 0.0, peak = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        s += x->data[i] * x->data[i];
        double a = fabs(x->data[i]);
        if (a > peak) peak = a;
    }
    double rms = sqrt(s / (double)N);
    return rms > 0 ? peak / rms : 0.0;
}

double matlab_rssq_s(matlab_mat *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    double s = 0.0;
    for (int64_t i = 0; i < N; ++i) s += x->data[i] * x->data[i];
    return sqrt(s);
}

/*===========================================================================
 * Tier-3 SPT §4.3 — pulse statistics (risetime/falltime/dutycycle/midcross).
 *
 *   t = midcross(x)      mid-reference (50%) crossing sample indices
 *                        (1-based, with sub-sample linear interp).
 *   r = risetime(x)      mean 10%→90% rise time across rising
 *                        transitions, in samples.
 *   f = falltime(x)      mean 90%→10% fall time across falling
 *                        transitions, in samples.
 *   d = dutycycle(x)     fraction of period above the 50% level,
 *                        averaged across full periods.
 *
 * State levels auto-detected as `min(x)` and `max(x)` (simple
 * estimator — histogram-based statelevels is a follow-on). Default
 * reference percentages are 10%, 50%, 90%. Returns scalar averages;
 * per-transition vector outputs are deferred.
 */

/* Linear-interpolate the sample index where the signal crosses
 * `level` between samples i-1 and i. Caller must verify a crossing
 * occurred in that interval. Returns 1-based sample index (fractional). */
static double sub_sample_cross_(const double *x, int64_t i, double level) {
    double a = x[i - 1], b = x[i];
    if (b == a) return (double)i;
    double t = (level - a) / (b - a);
    return (double)i + t;       /* 1-based: i is the after-cross sample */
}

matlab_mat *matlab_midcross(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N < 2) return mat_alloc(0, 1);
    double mn = x->data[0], mx = x->data[0];
    for (int64_t i = 1; i < N; ++i) {
        if (x->data[i] < mn) mn = x->data[i];
        if (x->data[i] > mx) mx = x->data[i];
    }
    double mid = mn + 0.5 * (mx - mn);
    std::vector<double> crosses;
    for (int64_t i = 1; i < N; ++i) {
        double a = x->data[i - 1], b = x->data[i];
        if ((a <= mid && b > mid) || (a >= mid && b < mid))
            crosses.push_back(sub_sample_cross_(x->data, i, mid));
    }
    int64_t M = (int64_t)crosses.size();
    matlab_mat *T = mat_alloc(M, M > 0 ? 1 : 0);
    for (int64_t i = 0; i < M; ++i) T->data[i] = crosses[i];
    return T;
}

/* Compute average sample-distance between low_pct% and high_pct%
 * crossings, measured during transitions in the requested direction
 * (rising = +1, falling = -1). Returns 0 if no transitions found. */
static double mean_transit_(matlab_mat *x, double low_pct, double high_pct,
                            int direction) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N < 2) return 0.0;
    double mn = x->data[0], mx = x->data[0];
    for (int64_t i = 1; i < N; ++i) {
        if (x->data[i] < mn) mn = x->data[i];
        if (x->data[i] > mx) mx = x->data[i];
    }
    double rng = mx - mn;
    /* MATLAB's sense: low_pct < high_pct for risetime, low_pct >
     * high_pct for falltime. Internal `lo`/`hi` are reordered by
     * direction so we always look for low → high in the iteration. */
    double a_pct, b_pct;
    if (direction > 0) { a_pct = low_pct; b_pct = high_pct; }
    else               { a_pct = high_pct; b_pct = low_pct; }
    double a_lvl = mn + a_pct * rng;
    double b_lvl = mn + b_pct * rng;
    /* Find each transition crossing first a_lvl then b_lvl in the
     * requested direction. The two branches are independent `if`s
     * (not `if`/`else if`) so that an abrupt one-sample transition
     * which crosses BOTH levels in a single step can finalize the
     * transit in the same iteration: the a_lvl branch fires first
     * and sets state=1, then the b_lvl branch's `state==1` guard
     * passes and finalizes immediately. Without that, the b_lvl
     * crossing was missed and the next transit's b_lvl was paired
     * with the previous a_lvl, inflating the result by ~one period. */
    double total = 0.0;
    int    count = 0;
    int    state = 0;          /* 0=before a_lvl, 1=passed a_lvl */
    double a_time = 0.0;
    for (int64_t i = 1; i < N; ++i) {
        double prev = x->data[i - 1], cur = x->data[i];
        if (direction > 0) {
            if (state == 0 && prev <= a_lvl && cur > a_lvl) {
                a_time = sub_sample_cross_(x->data, i, a_lvl);
                state = 1;
            }
            if (state == 1 && prev <= b_lvl && cur > b_lvl) {
                double b_time = sub_sample_cross_(x->data, i, b_lvl);
                total += (b_time - a_time);
                count++;
                state = 0;
            }
        } else {
            if (state == 0 && prev >= a_lvl && cur < a_lvl) {
                a_time = sub_sample_cross_(x->data, i, a_lvl);
                state = 1;
            }
            if (state == 1 && prev >= b_lvl && cur < b_lvl) {
                double b_time = sub_sample_cross_(x->data, i, b_lvl);
                total += (b_time - a_time);
                count++;
                state = 0;
            }
        }
    }
    return count > 0 ? total / (double)count : 0.0;
}

double matlab_risetime_s(matlab_mat *x) {
    return mean_transit_(x, 0.1, 0.9, +1);
}

double matlab_falltime_s(matlab_mat *x) {
    return mean_transit_(x, 0.1, 0.9, -1);
}

double matlab_dutycycle_s(matlab_mat *x) {
    matlab_mat *m = matlab_midcross(x);
    int64_t M = m->rows * m->cols;
    if (M < 2) { free(m->data); free(m); return 0.0; }
    /* Pair up midcrosses into rising/falling halves. We need the
     * direction at each crossing — re-derive from the data. */
    int64_t N = x->rows * x->cols;
    double mn = x->data[0], mx = x->data[0];
    for (int64_t i = 1; i < N; ++i) {
        if (x->data[i] < mn) mn = x->data[i];
        if (x->data[i] > mx) mx = x->data[i];
    }
    double mid = mn + 0.5 * (mx - mn);
    std::vector<int> dirs((size_t)M, 0);
    int j = 0;
    for (int64_t i = 1; i < N && j < (int)M; ++i) {
        double a = x->data[i - 1], b = x->data[i];
        if ((a <= mid && b > mid)) { dirs[(size_t)j++] = +1; }
        else if ((a >= mid && b < mid)) { dirs[(size_t)j++] = -1; }
    }
    /* Sum of (next-rising − rising) midcross widths and divide by
     * sum of full periods. */
    double on = 0.0, period = 0.0;
    for (int64_t i = 0; i + 2 < M; ++i) {
        if (dirs[(size_t)i] == +1 && dirs[(size_t)(i + 1)] == -1
            && dirs[(size_t)(i + 2)] == +1) {
            on     += m->data[i + 1] - m->data[i];
            period += m->data[i + 2] - m->data[i];
        }
    }
    free(m->data); free(m);
    return period > 0.0 ? on / period : 0.0;
}

/*===========================================================================
 * Tier-3 SPT §4.3 tail — statelevels / slewrate / pulseperiod / pulsewidth /
 * overshoot / undershoot / settlingtime.
 *
 * statelevels uses the histogram-based estimator MATLAB ships: the
 * signal range is split into NBINS uniform bins, the histogram is
 * separated at its midpoint, and the highest-count bin in each half
 * gives the corresponding state level (bin centre). Falls back to
 * straight min/max when the histogram is too sparse.
 *
 * The remaining functions all sit on top of statelevels + the existing
 * `mean_transit_` / `matlab_midcross` scaffolding.
 */
static void state_levels_(matlab_mat *x, double *lo, double *hi) {
    int64_t N = x ? x->rows * x->cols : 0;
    if (N == 0) { *lo = 0.0; *hi = 0.0; return; }
    double mn = x->data[0], mx = x->data[0];
    for (int64_t i = 1; i < N; ++i) {
        if (x->data[i] < mn) mn = x->data[i];
        if (x->data[i] > mx) mx = x->data[i];
    }
    if (mx <= mn) { *lo = mn; *hi = mx; return; }
    constexpr int NBINS = 100;
    int counts[NBINS] = {0};
    double rng = mx - mn;
    for (int64_t i = 0; i < N; ++i) {
        int b = (int)((x->data[i] - mn) / rng * (double)NBINS);
        if (b < 0) b = 0;
        if (b >= NBINS) b = NBINS - 1;
        counts[b]++;
    }
    int half = NBINS / 2;
    int lo_b = 0, hi_b = NBINS - 1;
    int lo_c = -1, hi_c = -1;
    for (int b = 0; b < half; ++b)
        if (counts[b] > lo_c) { lo_c = counts[b]; lo_b = b; }
    for (int b = half; b < NBINS; ++b)
        if (counts[b] > hi_c) { hi_c = counts[b]; hi_b = b; }
    *lo = mn + (lo_b + 0.5) * rng / (double)NBINS;
    *hi = mn + (hi_b + 0.5) * rng / (double)NBINS;
}

matlab_mat *matlab_statelevels(matlab_mat *x) {
    matlab_mat *L = mat_alloc(2, 1);
    state_levels_(x, &L->data[0], &L->data[1]);
    return L;
}

/* slewrate: (high - low) / mean_risetime. With unit sample spacing the
 * units come out as signal-units per sample. MATLAB returns one value
 * per transition; we return the mean rising slewrate as a scalar
 * (matches the risetime/falltime/dutycycle convention already shipped). */
double matlab_slewrate_s(matlab_mat *x) {
    if (!x || x->rows * x->cols < 2) return 0.0;
    double lo, hi;
    state_levels_(x, &lo, &hi);
    double rt = mean_transit_(x, 0.1, 0.9, +1);
    if (rt <= 0.0 || hi <= lo) return 0.0;
    return (0.8 * (hi - lo)) / rt;          /* 10–90 % rise → 0.8·range */
}

/* Mean distance between consecutive rising midcrosses. */
double matlab_pulseperiod_s(matlab_mat *x) {
    matlab_mat *m = matlab_midcross(x);
    int64_t M = m ? m->rows * m->cols : 0;
    if (!x || M < 2) { if (m) { free(m->data); free(m); } return 0.0; }
    /* Re-derive direction at each crossing. */
    int64_t N = x->rows * x->cols;
    double mn = x->data[0], mx = x->data[0];
    for (int64_t i = 1; i < N; ++i) {
        if (x->data[i] < mn) mn = x->data[i];
        if (x->data[i] > mx) mx = x->data[i];
    }
    double mid = mn + 0.5 * (mx - mn);
    std::vector<double> rising;
    rising.reserve((size_t)M);
    int j = 0;
    for (int64_t i = 1; i < N && j < (int)M; ++i) {
        double a = x->data[i - 1], b = x->data[i];
        if (a <= mid && b > mid) rising.push_back(m->data[j++]);
        else if (a >= mid && b < mid) j++;
    }
    free(m->data); free(m);
    if (rising.size() < 2) return 0.0;
    double sum = 0.0;
    for (size_t i = 1; i < rising.size(); ++i)
        sum += rising[i] - rising[i - 1];
    return sum / (double)(rising.size() - 1);
}

/* Mean distance from each rising midcross to the next falling midcross. */
double matlab_pulsewidth_s(matlab_mat *x) {
    matlab_mat *m = matlab_midcross(x);
    int64_t M = m ? m->rows * m->cols : 0;
    if (!x || M < 2) { if (m) { free(m->data); free(m); } return 0.0; }
    int64_t N = x->rows * x->cols;
    double mn = x->data[0], mx = x->data[0];
    for (int64_t i = 1; i < N; ++i) {
        if (x->data[i] < mn) mn = x->data[i];
        if (x->data[i] > mx) mx = x->data[i];
    }
    double mid = mn + 0.5 * (mx - mn);
    std::vector<int> dirs((size_t)M, 0);
    int j = 0;
    for (int64_t i = 1; i < N && j < (int)M; ++i) {
        double a = x->data[i - 1], b = x->data[i];
        if (a <= mid && b > mid)      dirs[(size_t)j++] = +1;
        else if (a >= mid && b < mid) dirs[(size_t)j++] = -1;
    }
    double sum = 0.0;
    int    cnt = 0;
    for (int64_t i = 0; i + 1 < M; ++i) {
        if (dirs[(size_t)i] == +1 && dirs[(size_t)(i + 1)] == -1) {
            sum += m->data[i + 1] - m->data[i];
            cnt++;
        }
    }
    free(m->data); free(m);
    return cnt > 0 ? sum / (double)cnt : 0.0;
}

/* Mean overshoot above the high state level on rising transitions,
 * expressed as percent of (high - low). Looks within one period after
 * each rising midcross; returns 0 if no overshoot detected. */
double matlab_overshoot_s(matlab_mat *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N < 2) return 0.0;
    double lo, hi;
    state_levels_(x, &lo, &hi);
    if (hi <= lo) return 0.0;
    double rng = hi - lo;
    int    cnt = 0;
    double total_pct = 0.0;
    int    above = 0;       /* tracks "have we crossed into the high state since last reset" */
    double max_after = lo;
    for (int64_t i = 0; i < N; ++i) {
        double v = x->data[i];
        if (!above && v >= hi) {
            above = 1;
            max_after = v;
        } else if (above) {
            if (v > max_after) max_after = v;
            if (v < lo + 0.5 * rng) {
                /* Edge has fully fallen back below midpoint —
                 * record overshoot and reset. */
                if (max_after > hi)
                    total_pct += 100.0 * (max_after - hi) / rng;
                cnt++;
                above = 0;
                max_after = lo;
            }
        }
    }
    if (above && max_after > hi) {
        total_pct += 100.0 * (max_after - hi) / rng;
        cnt++;
    }
    return cnt > 0 ? total_pct / (double)cnt : 0.0;
}

/* Mean undershoot below the low state level on falling transitions. */
double matlab_undershoot_s(matlab_mat *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N < 2) return 0.0;
    double lo, hi;
    state_levels_(x, &lo, &hi);
    if (hi <= lo) return 0.0;
    double rng = hi - lo;
    int    cnt = 0;
    double total_pct = 0.0;
    int    below = 0;
    double min_after = hi;
    for (int64_t i = 0; i < N; ++i) {
        double v = x->data[i];
        if (!below && v <= lo) {
            below = 1;
            min_after = v;
        } else if (below) {
            if (v < min_after) min_after = v;
            if (v > lo + 0.5 * rng) {
                if (min_after < lo)
                    total_pct += 100.0 * (lo - min_after) / rng;
                cnt++;
                below = 0;
                min_after = hi;
            }
        }
    }
    if (below && min_after < lo) {
        total_pct += 100.0 * (lo - min_after) / rng;
        cnt++;
    }
    return cnt > 0 ? total_pct / (double)cnt : 0.0;
}

/* Mean settling time: from each rising midcross, the number of samples
 * until x stays within `d` (fractional, e.g. 0.02 = ±2 %) of the high
 * state level for the rest of the pulse. d defaults to 0.02 if non-positive. */
double matlab_settlingtime_s(matlab_mat *x, double d) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N < 2) return 0.0;
    if (!(d > 0.0)) d = 0.02;
    double lo, hi;
    state_levels_(x, &lo, &hi);
    if (hi <= lo) return 0.0;
    double rng = hi - lo;
    double tol = d * rng;
    double mid = lo + 0.5 * rng;
    double total = 0.0;
    int    cnt   = 0;
    for (int64_t i = 1; i < N; ++i) {
        double a = x->data[i - 1], b = x->data[i];
        if (a <= mid && b > mid) {
            double t_mid = sub_sample_cross_(x->data, i, mid);
            /* Walk forward from i until x stays within ±tol of hi
             * for the remainder of the rising pulse (i.e. until it
             * falls back below mid). */
            int64_t last_violation = i;
            int64_t k = i;
            while (k < N && x->data[k] >= mid) {
                if (fabs(x->data[k] - hi) > tol) last_violation = k;
                k++;
            }
            if (last_violation + 1 < N) {
                total += (double)(last_violation + 1) - t_mid;
                cnt++;
            }
            i = k;
        }
    }
    return cnt > 0 ? total / (double)cnt : 0.0;
}

/*===========================================================================
 * Tier-3 SPT §4.3 — envelope / hampel / medfilt1.
 *
 *   y = medfilt1(x, n)   1-D median filter (length-n sliding window,
 *                        zero-padded edges, n must be odd — coerced).
 *   y = hampel(x, k)     outlier replace via running-median + MAD test
 *                        (window length 2k+1, threshold 3·1.4826·MAD).
 *   y = envelope(x)      upper envelope via linear interpolation between
 *                        local maxima. Same shape as input.
 */
matlab_mat *matlab_medfilt1(matlab_mat *x, double n_d) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int n = (int)n_d;
    if (n < 1) n = 1;
    if ((n & 1) == 0) n++;       /* coerce to odd */
    int half = (n - 1) / 2;
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    if (N == 0) return Y;
    std::vector<double> buf((size_t)n);
    for (int64_t i = 0; i < N; ++i) {
        for (int j = 0; j < n; ++j) {
            int64_t k = i - half + j;
            buf[(size_t)j] = (k >= 0 && k < N) ? x->data[k] : 0.0;
        }
        std::sort(buf.begin(), buf.end());
        Y->data[i] = buf[(size_t)half];
    }
    return Y;
}

matlab_mat *matlab_hampel(matlab_mat *x, double k_d) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int k = (int)k_d;
    if (k < 1) k = 1;
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    if (N == 0) return Y;
    int n = 2 * k + 1;
    std::vector<double> win((size_t)n);
    for (int64_t i = 0; i < N; ++i) {
        int len = 0;
        for (int j = -k; j <= k; ++j) {
            int64_t idx = i + j;
            if (idx >= 0 && idx < N) win[(size_t)len++] = x->data[idx];
        }
        std::vector<double> w(win.begin(), win.begin() + len);
        std::sort(w.begin(), w.end());
        double med = w[(size_t)(len / 2)];
        if (len > 1 && (len % 2 == 0))
            med = 0.5 * (w[(size_t)(len / 2 - 1)] + w[(size_t)(len / 2)]);
        std::vector<double> dev((size_t)len);
        for (int j = 0; j < len; ++j) dev[(size_t)j] = fabs(w[(size_t)j] - med);
        std::sort(dev.begin(), dev.end());
        double mad = dev[(size_t)(len / 2)];
        if (len > 1 && (len % 2 == 0))
            mad = 0.5 * (dev[(size_t)(len / 2 - 1)] + dev[(size_t)(len / 2)]);
        double sigma = 1.4826 * mad;
        Y->data[i] = (fabs(x->data[i] - med) > 3.0 * sigma) ? med : x->data[i];
    }
    return Y;
}

matlab_mat *matlab_envelope(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    if (N == 0) return Y;
    if (N < 3) {
        for (int64_t i = 0; i < N; ++i) Y->data[i] = fabs(x->data[i]);
        return Y;
    }
    /* Find local maxima of x. Linear-interpolate between them; clamp
     * the endpoints to the closest interior maximum value. */
    std::vector<int64_t> idx;
    std::vector<double>  val;
    for (int64_t i = 1; i < N - 1; ++i)
        if (x->data[i - 1] < x->data[i] && x->data[i] > x->data[i + 1]) {
            idx.push_back(i); val.push_back(x->data[i]);
        }
    if (idx.empty()) {
        /* No interior peak: fall back to global max held flat. */
        double mx = x->data[0];
        for (int64_t i = 1; i < N; ++i) if (x->data[i] > mx) mx = x->data[i];
        for (int64_t i = 0; i < N; ++i) Y->data[i] = mx;
        return Y;
    }
    /* Left tail: hold first peak's value. */
    for (int64_t i = 0; i <= idx[0]; ++i) Y->data[i] = val[0];
    /* Interior interpolation. */
    for (size_t s = 0; s + 1 < idx.size(); ++s) {
        int64_t a = idx[s], b = idx[s + 1];
        double  va = val[s], vb = val[s + 1];
        for (int64_t i = a + 1; i <= b; ++i) {
            double t = (double)(i - a) / (double)(b - a);
            Y->data[i] = va + t * (vb - va);
        }
    }
    /* Right tail: hold last peak's value. */
    for (int64_t i = idx.back() + 1; i < N; ++i) Y->data[i] = val.back();
    return Y;
}

/*===========================================================================
 * Tier-2 SPT §3.2 linear prediction — Levinson + LPC + AR estimators.
 *
 *   a = levinson(r, p)   AR coefficients from autocorrelation via the
 *                        Levinson-Durbin recursion.
 *   a = lpc(x, p)        biased-autocorr LPC: r = autocorr(x); levinson(r, p).
 *   a = aryule(x, p)     same shape as lpc but uses Yule-Walker biased
 *                        autocorrelation; output is mathematically equal
 *                        to lpc(x, p) for our scope (single-output form).
 *   a = arburg(x, p)     Burg's method — minimises forward + backward
 *                        prediction errors recursively.
 *
 * All return a (1 × (p+1)) row vector. The reflection coefficients
 * and final prediction-error variance (3-return forms) are deferred.
 */
matlab_mat *matlab_levinson(matlab_mat *r, double p_d) {
    if (!r) return mat_alloc(0, 0);
    int p = (int)p_d;
    if (p < 1) p = 1;
    int64_t nr = r->rows * r->cols;
    if (nr < p + 1) p = (int)nr - 1;
    if (p < 0) return mat_alloc(0, 0);
    /* Levinson-Durbin: solves the p×p Toeplitz system.
     * a[0] = 1; recursion through orders 1..p. */
    std::vector<double> a((size_t)(p + 1), 0.0);
    std::vector<double> aprev((size_t)(p + 1), 0.0);
    a[0] = 1.0;
    double E = r->data[0];
    if (E == 0.0) {
        matlab_mat *A = mat_alloc(1, p + 1);
        A->data[0] = 1.0;
        return A;
    }
    for (int m = 1; m <= p; ++m) {
        double k = -r->data[m];
        for (int j = 1; j < m; ++j) k -= a[j] * r->data[m - j];
        k /= E;
        aprev = a;
        for (int j = 1; j < m; ++j) a[j] = aprev[j] + k * aprev[m - j];
        a[m] = k;
        E *= (1.0 - k * k);
        if (E <= 0.0) break;        /* numerical / non-PSD r */
    }
    matlab_mat *A = mat_alloc(1, p + 1);
    for (int i = 0; i <= p; ++i) A->data[i] = a[i];
    return A;
}

/* Biased autocorrelation of x at lags 0..p. Returns a (p+1)-length
 * vector. Used internally by lpc / aryule. */
static void biased_autocorr_(const double *x, int64_t N, int p,
                             std::vector<double> &r) {
    r.assign((size_t)(p + 1), 0.0);
    for (int k = 0; k <= p; ++k) {
        double s = 0.0;
        for (int64_t n = 0; n < N - k; ++n) s += x[n] * x[n + k];
        r[(size_t)k] = s / (double)N;
    }
}

matlab_mat *matlab_lpc(matlab_mat *x, double p_d) {
    if (!x) return mat_alloc(0, 0);
    int p = (int)p_d;
    if (p < 1) p = 1;
    int64_t N = x->rows * x->cols;
    if (N < (int64_t)p + 1) {
        matlab_mat *A = mat_alloc(1, p + 1);
        A->data[0] = 1.0;
        return A;
    }
    std::vector<double> r;
    biased_autocorr_(x->data, N, p, r);
    /* Build a temporary matlab_mat for r and call levinson. */
    matlab_mat rm = { /*data*/ r.data(), /*rows*/ 1, /*cols*/ (int64_t)r.size() };
    return matlab_levinson(&rm, (double)p);
}

matlab_mat *matlab_aryule(matlab_mat *x, double p_d) {
    /* Single-output form; same as lpc for our scope. */
    return matlab_lpc(x, p_d);
}

matlab_mat *matlab_arburg(matlab_mat *x, double p_d) {
    if (!x) return mat_alloc(0, 0);
    int p = (int)p_d;
    if (p < 1) p = 1;
    int64_t N = x->rows * x->cols;
    if (N < (int64_t)p + 1) {
        matlab_mat *A = mat_alloc(1, p + 1);
        A->data[0] = 1.0;
        return A;
    }
    /* Burg recursion: f and b are forward/backward prediction errors;
     * a is the AR coefficient vector (with a[0] = 1). */
    std::vector<double> f((size_t)N), b((size_t)N);
    for (int64_t i = 0; i < N; ++i) { f[(size_t)i] = b[(size_t)i] = x->data[i]; }
    std::vector<double> a((size_t)(p + 1), 0.0);
    a[0] = 1.0;
    std::vector<double> aprev = a;
    for (int m = 1; m <= p; ++m) {
        /* Reflection coefficient k = -2 sum f·b / (sum f² + sum b²),
         * over the valid index range. */
        double num = 0.0, den = 0.0;
        for (int64_t i = m; i < N; ++i) {
            num += f[(size_t)i] * b[(size_t)(i - 1)];
            den += f[(size_t)i] * f[(size_t)i] +
                   b[(size_t)(i - 1)] * b[(size_t)(i - 1)];
        }
        double k = (den != 0.0) ? (-2.0 * num / den) : 0.0;
        /* Update AR coefficients. */
        aprev = a;
        for (int j = 1; j < m; ++j) a[j] = aprev[j] + k * aprev[m - j];
        a[m] = k;
        /* Update forward / backward errors in place. */
        std::vector<double> fnew = f, bnew = b;
        for (int64_t i = m; i < N; ++i) {
            fnew[(size_t)i] = f[(size_t)i] + k * b[(size_t)(i - 1)];
            bnew[(size_t)i] = b[(size_t)(i - 1)] + k * f[(size_t)i];
        }
        f = fnew; b = bnew;
    }
    matlab_mat *A = mat_alloc(1, p + 1);
    for (int i = 0; i <= p; ++i) A->data[i] = a[i];
    return A;
}

/* pyulear(x, p, N) / pburg(x, p, N) — AR-based PSD estimators.
 *
 * Algorithm: design an AR(p) model via Yule-Walker (pyulear) or Burg
 * (pburg), then evaluate |1 / A(e^{jω})|² · σ² on an N-point grid.
 * The error variance σ² is approximated by r[0]·prod(1 − k_i²) for
 * Yule-Walker; for Burg we use r[0] directly (acceptable for the
 * single-output PSD shape).
 *
 * Returns (M × 1) where M = N (full grid, normalised fs = 1).
 */
static matlab_mat *ar_psd_(matlab_mat *a_coefs, double sigma2, int Ng) {
    if (!a_coefs || Ng <= 0) return mat_alloc(0, 0);
    int64_t na = a_coefs->rows * a_coefs->cols;
    matlab_mat *P = mat_alloc(Ng, 1);
    for (int k = 0; k < Ng; ++k) {
        double w = M_PI * (double)k / (double)Ng;
        double re = 0, im = 0;
        for (int64_t i = 0; i < na; ++i) {
            double a_ = -w * (double)i;
            re += a_coefs->data[i] * cos(a_);
            im += a_coefs->data[i] * sin(a_);
        }
        double mag2 = re * re + im * im;
        P->data[k] = (mag2 > 0) ? sigma2 / mag2 : 0.0;
    }
    return P;
}

matlab_mat *matlab_pyulear(matlab_mat *x, double p_d, double N_d) {
    if (!x) return mat_alloc(0, 0);
    matlab_mat *a = matlab_aryule(x, p_d);
    /* sigma² estimate: biased autocorr at lag 0. */
    double s = 0;
    int64_t Nx = x->rows * x->cols;
    for (int64_t i = 0; i < Nx; ++i) s += x->data[i] * x->data[i];
    double sigma2 = (Nx > 0) ? s / (double)Nx : 1.0;
    matlab_mat *P = ar_psd_(a, sigma2, (int)N_d);
    free(a->data); free(a);
    return P;
}

matlab_mat *matlab_pburg(matlab_mat *x, double p_d, double N_d) {
    if (!x) return mat_alloc(0, 0);
    matlab_mat *a = matlab_arburg(x, p_d);
    double s = 0;
    int64_t Nx = x->rows * x->cols;
    for (int64_t i = 0; i < Nx; ++i) s += x->data[i] * x->data[i];
    double sigma2 = (Nx > 0) ? s / (double)Nx : 1.0;
    matlab_mat *P = ar_psd_(a, sigma2, (int)N_d);
    free(a->data); free(a);
    return P;
}

/*===========================================================================
 * Tier-2 SPT §3.4 transforms — DCT-II / DCT-III / Walsh-Hadamard.
 *
 *   y = dct(x)      MATLAB-orthonormal DCT-II (forward DCT).
 *   y = idct(X)     MATLAB-orthonormal DCT-III (inverse DCT-II).
 *   y = fwht(x)     Walsh-Hadamard transform, natural / Hadamard
 *                   ordering, divided by N (matches MATLAB default).
 *
 * dct/idct use direct O(N²) sums — fine at the lengths SPT users
 * typically apply (≤ a few thousand). The N-point FFT trick is a
 * future optimization.
 */
matlab_mat *matlab_dct(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N == 0) return mat_alloc(x->rows, x->cols);
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    double s0 = sqrt(1.0 / (double)N);
    double s1 = sqrt(2.0 / (double)N);
    for (int64_t k = 0; k < N; ++k) {
        double sum = 0.0;
        for (int64_t n = 0; n < N; ++n)
            sum += x->data[n] *
                   cos(M_PI * (double)(2 * n + 1) * (double)k / (2.0 * (double)N));
        Y->data[k] = (k == 0 ? s0 : s1) * sum;
    }
    return Y;
}

matlab_mat *matlab_idct(matlab_mat *X) {
    if (!X) return mat_alloc(0, 0);
    int64_t N = X->rows * X->cols;
    if (N == 0) return mat_alloc(X->rows, X->cols);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    double s0 = sqrt(1.0 / (double)N);
    double s1 = sqrt(2.0 / (double)N);
    for (int64_t n = 0; n < N; ++n) {
        double sum = X->data[0] * s0;
        for (int64_t k = 1; k < N; ++k)
            sum += X->data[k] * s1 *
                   cos(M_PI * (double)(2 * n + 1) * (double)k / (2.0 * (double)N));
        Y->data[n] = sum;
    }
    return Y;
}

/* Walsh-Hadamard via butterfly. Length must be power-of-2; otherwise
 * we round up by zero-padding. Output is divided by N (MATLAB
 * default normalization, matching the doc's "WH = HW * w / N"). */
matlab_mat *matlab_fwht(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t Nin = x->rows * x->cols;
    if (Nin == 0) return mat_alloc(x->rows, x->cols);
    /* Round up to power of 2. */
    int64_t N = 1;
    while (N < Nin) N <<= 1;
    matlab_mat *Y = mat_alloc(x->rows == 1 ? 1 : N,
                              x->rows == 1 ? N : 1);
    /* Copy + zero-pad. */
    std::vector<double> buf((size_t)N, 0.0);
    for (int64_t i = 0; i < Nin; ++i) buf[i] = x->data[i];
    /* In-place butterfly. */
    for (int64_t half = 1; half < N; half <<= 1) {
        for (int64_t i = 0; i < N; i += 2 * half) {
            for (int64_t j = 0; j < half; ++j) {
                double a = buf[i + j];
                double b = buf[i + j + half];
                buf[i + j]        = a + b;
                buf[i + j + half] = a - b;
            }
        }
    }
    /* Normalize by N (MATLAB default). */
    for (int64_t i = 0; i < N; ++i) Y->data[i] = buf[i] / (double)N;
    return Y;
}

/*===========================================================================
 * Close-the-loop helpers (Tier-1 SPT §2.5).
 *
 *   y = filtfilt(b, a, x)   forward-backward zero-phase IIR filtering
 *   y = sosfilt(sos, x)     cascade of second-order sections
 *   h = impz(b, a, N)       impulse response of an IIR
 *   s = stepz(b, a, N)      step response of an IIR
 *   gd = grpdelay(b, a, N)  group delay τ(ω) = −d/dω arg(H(e^{jω}))
 */

/* Internal direct-form-II transposed filter on a flat double buffer.
 * b and a are normalized so a[0] = 1; caller must pre-normalize. */
static void filter_flat_(const double *b, int64_t nb,
                         const double *a, int64_t na,
                         const double *x, int64_t nx,
                         double *y) {
    int64_t L = nb > na ? nb : na;
    std::vector<double> w((size_t)L, 0.0);
    for (int64_t n = 0; n < nx; ++n) {
        double yn = (nb > 0 ? b[0] * x[n] : 0.0) + w[0];
        /* Shift the delay line: w[i] = b[i+1]*x[n] − a[i+1]*yn + w[i+1]. */
        for (int64_t i = 0; i < L - 1; ++i) {
            double bi = (i + 1 < nb) ? b[i + 1] : 0.0;
            double ai = (i + 1 < na) ? a[i + 1] : 0.0;
            w[i] = bi * x[n] - ai * yn + w[i + 1];
        }
        if (L > 0) {
            double bi = (L < nb) ? b[L]     : 0.0;
            double ai = (L < na) ? a[L]     : 0.0;
            w[L - 1] = bi * x[n] - ai * yn;
        }
        y[n] = yn;
    }
}

/* Compute the unit-step steady-state initial-condition vector for a
 * direct-form II transposed IIR filter with normalised (b, a, a[0]=1).
 * Solves (I - A) zi = B where:
 *   A_ij = -a[i+1] if j == 0
 *          1 if j == i+1
 *          0 otherwise   (companion-form state transition)
 *   B_i  = b[i+1] - a[i+1] * b[0]
 * This is the canonical scipy.signal.lfilter_zi formulation. The
 * returned vector has length N = max(nb, na) - 1. Multiply by the
 * boundary input value (x[0] at the front, x[end] at the back) to
 * use as a filter IC.
 */
static std::vector<double> filter_steady_state_ic_(
    const std::vector<double> &bn, const std::vector<double> &an) {
    int64_t L = (int64_t)(bn.size() > an.size() ? bn.size() : an.size());
    int N = (int)(L - 1);
    if (N <= 0) return {};
    /* Pad bn / an out to length L so indices i = 1..N are well defined. */
    std::vector<double> b((size_t)L, 0.0), a((size_t)L, 0.0);
    for (size_t i = 0; i < bn.size(); ++i) b[i] = bn[i];
    for (size_t i = 0; i < an.size(); ++i) a[i] = an[i];
    /* Build (I - A) as an N×N matrix in row-major order. */
    std::vector<double> M((size_t)(N * N), 0.0);
    std::vector<double> rhs((size_t)N);
    for (int i = 0; i < N; ++i) {
        /* Row i of (I - A): identity minus A_ij. */
        for (int j = 0; j < N; ++j) {
            double Aij = 0.0;
            if (j == 0)     Aij = -a[i + 1];
            if (j == i + 1) Aij = 1.0;
            M[(size_t)(i * N + j)] = (i == j ? 1.0 : 0.0) - Aij;
        }
        rhs[(size_t)i] = b[i + 1] - a[i + 1] * b[0];
    }
    /* Gaussian elimination with partial pivoting. */
    for (int k = 0; k < N; ++k) {
        int piv = k;
        double pv = fabs(M[(size_t)(k * N + k)]);
        for (int r = k + 1; r < N; ++r) {
            double v = fabs(M[(size_t)(r * N + k)]);
            if (v > pv) { pv = v; piv = r; }
        }
        if (pv < 1e-300) return std::vector<double>((size_t)N, 0.0);
        if (piv != k) {
            for (int j = 0; j < N; ++j)
                std::swap(M[(size_t)(k * N + j)], M[(size_t)(piv * N + j)]);
            std::swap(rhs[(size_t)k], rhs[(size_t)piv]);
        }
        for (int r = k + 1; r < N; ++r) {
            double f = M[(size_t)(r * N + k)] / M[(size_t)(k * N + k)];
            for (int j = k; j < N; ++j)
                M[(size_t)(r * N + j)] -= f * M[(size_t)(k * N + j)];
            rhs[(size_t)r] -= f * rhs[(size_t)k];
        }
    }
    std::vector<double> zi((size_t)N);
    for (int i = N - 1; i >= 0; --i) {
        double s = rhs[(size_t)i];
        for (int j = i + 1; j < N; ++j)
            s -= M[(size_t)(i * N + j)] * zi[(size_t)j];
        zi[(size_t)i] = s / M[(size_t)(i * N + i)];
    }
    return zi;
}

/* Direct-form II transposed filter with explicit initial-condition
 * vector. Used by filtfilt's Gustafsson-IC path. */
static void filter_flat_zi_(const double *b, int64_t nb,
                            const double *a, int64_t na,
                            const double *zi, int64_t nz,
                            const double *x, int64_t nx, double *y) {
    /* z[] is the DF-II-T state, length max(nb, na) - 1. */
    int64_t L = nb > na ? nb : na;
    int64_t Nz = L - 1;
    if (Nz < 0) Nz = 0;
    std::vector<double> z((size_t)Nz, 0.0);
    for (int64_t i = 0; i < Nz && i < nz; ++i) z[(size_t)i] = zi[i];
    for (int64_t n = 0; n < nx; ++n) {
        double xn = x[n];
        double yn = (nb > 0 ? b[0] : 0.0) * xn + (Nz > 0 ? z[0] : 0.0);
        for (int64_t i = 0; i + 1 < Nz; ++i) {
            double bi = (i + 1 < nb) ? b[i + 1] : 0.0;
            double ai = (i + 1 < na) ? a[i + 1] : 0.0;
            z[(size_t)i] = bi * xn + z[(size_t)(i + 1)] - ai * yn;
        }
        if (Nz > 0) {
            double bi = (Nz < nb) ? b[Nz] : 0.0;
            double ai = (Nz < na) ? a[Nz] : 0.0;
            z[(size_t)(Nz - 1)] = bi * xn - ai * yn;
        }
        y[n] = yn;
    }
}

/* filtfilt(b, a, x) — forward-backward filter for zero-phase response.
 *
 * Reflection-pads x by 3·(L-1) samples on each side (odd reflection,
 * the same scheme MATLAB / scipy use), forward-filters from a steady-
 * state IC (lfilter_zi multiplied by the boundary input value),
 * reverses, forward-filters again from the matching IC, reverses, and
 * trims the padding. This matches scipy.signal.filtfilt with its
 * default method='pad' / padtype='odd' — the steady-state IC removes
 * the transient that the previous zero-IC implementation produced for
 * DC-like inputs (constant signals are now preserved exactly). The
 * stricter 1996 Gustafsson method (scipy's method='gust') solves an
 * explicit edge-elimination system instead of padding; that's a
 * separate follow-on. */
matlab_mat *matlab_filtfilt(matlab_mat *b, matlab_mat *a, matlab_mat *x) {
    if (!b || !a || !x) return mat_alloc(0, 0);
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    int64_t nx = x->rows * x->cols;
    if (na == 0 || a->data[0] == 0.0 || nx == 0) return mat_alloc(0, 0);
    /* Normalize for a[0] = 1. */
    std::vector<double> bn(nb), an(na);
    double a0 = a->data[0];
    for (int64_t i = 0; i < nb; ++i) bn[i] = b->data[i] / a0;
    for (int64_t i = 0; i < na; ++i) an[i] = a->data[i] / a0;
    int64_t L = nb > na ? nb : na;
    int64_t pad = 3 * (L - 1);
    if (pad < 0) pad = 0;
    if (pad > nx - 1) pad = nx - 1;
    /* Reflect-pad. */
    int64_t total = nx + 2 * pad;
    std::vector<double> xp((size_t)total);
    for (int64_t i = 0; i < pad; ++i)
        xp[i] = 2.0 * x->data[0] - x->data[pad - i];
    for (int64_t i = 0; i < nx; ++i) xp[pad + i] = x->data[i];
    for (int64_t i = 0; i < pad; ++i)
        xp[pad + nx + i] = 2.0 * x->data[nx - 1] - x->data[nx - 2 - i];
    /* Compute lfilter_zi (unit-step IC) once and scale by xp[0] /
     * xp[end] for the two passes. */
    std::vector<double> zi_unit = filter_steady_state_ic_(bn, an);
    int64_t Nz = (int64_t)zi_unit.size();
    std::vector<double> zi_fwd((size_t)Nz);
    for (int64_t i = 0; i < Nz; ++i) zi_fwd[(size_t)i] = zi_unit[(size_t)i] * xp[0];
    /* Forward pass. */
    std::vector<double> y1((size_t)total);
    filter_flat_zi_(bn.data(), nb, an.data(), na,
                    zi_fwd.data(), Nz, xp.data(), total, y1.data());
    /* Reverse, filter again starting from the matching IC at the new
     * boundary value (last sample of y1). */
    std::vector<double> rev((size_t)total);
    for (int64_t i = 0; i < total; ++i) rev[i] = y1[total - 1 - i];
    std::vector<double> zi_bwd((size_t)Nz);
    for (int64_t i = 0; i < Nz; ++i) zi_bwd[(size_t)i] = zi_unit[(size_t)i] * rev[0];
    std::vector<double> y2((size_t)total);
    filter_flat_zi_(bn.data(), nb, an.data(), na,
                    zi_bwd.data(), Nz, rev.data(), total, y2.data());
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    for (int64_t i = 0; i < nx; ++i) Y->data[i] = y2[total - 1 - (pad + i)];
    return Y;
}

/* sosfilt(sos, x) — apply a cascade of second-order sections. sos is
 * an L × 6 matrix [b0 b1 b2 a0 a1 a2] per row. Each section is filtered
 * in series with its predecessor's output. Each row's a0 is implicitly
 * normalized to 1 by the section coefficient layout. */
matlab_mat *matlab_sosfilt(matlab_mat *sos, matlab_mat *x) {
    if (!sos || !x) return mat_alloc(0, 0);
    int64_t L = sos->rows;
    int64_t W = sos->cols;
    int64_t nx = x->rows * x->cols;
    if (W != 6 || L == 0 || nx == 0) {
        /* Degenerate: copy x through. */
        matlab_mat *Y = mat_alloc(x->rows, x->cols);
        for (int64_t i = 0; i < nx; ++i) Y->data[i] = x->data[i];
        return Y;
    }
    std::vector<double> buf((size_t)nx);
    for (int64_t i = 0; i < nx; ++i) buf[i] = x->data[i];
    std::vector<double> next((size_t)nx);
    for (int64_t s = 0; s < L; ++s) {
        const double *r = sos->data + s * 6;
        double bsec[3] = { r[0], r[1], r[2] };
        double asec[3] = { r[3], r[4], r[5] };
        if (asec[0] == 0.0) continue;
        for (int i = 0; i < 3; ++i) bsec[i] /= asec[0];
        for (int i = 0; i < 3; ++i) asec[i] /= asec[0];
        filter_flat_(bsec, 3, asec, 3, buf.data(), nx, next.data());
        for (int64_t i = 0; i < nx; ++i) buf[i] = next[i];
    }
    matlab_mat *Y = mat_alloc(x->rows, x->cols);
    for (int64_t i = 0; i < nx; ++i) Y->data[i] = buf[i];
    return Y;
}

/* impz(b, a, N) — impulse response. Drive the filter with [1 0 0 ...]. */
matlab_mat *matlab_impz(matlab_mat *b, matlab_mat *a, double N_d) {
    if (!b || !a) return mat_alloc(0, 0);
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    int64_t N = (int64_t)N_d;
    if (N <= 0 || na == 0 || a->data[0] == 0.0) return mat_alloc(0, 0);
    std::vector<double> bn(nb), an(na);
    double a0 = a->data[0];
    for (int64_t i = 0; i < nb; ++i) bn[i] = b->data[i] / a0;
    for (int64_t i = 0; i < na; ++i) an[i] = a->data[i] / a0;
    std::vector<double> imp((size_t)N, 0.0);
    imp[0] = 1.0;
    matlab_mat *H = mat_alloc(N, 1);
    filter_flat_(bn.data(), nb, an.data(), na, imp.data(), N, H->data);
    return H;
}

/* stepz(b, a, N) — step response. Drive with all-ones. */
matlab_mat *matlab_stepz(matlab_mat *b, matlab_mat *a, double N_d) {
    if (!b || !a) return mat_alloc(0, 0);
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    int64_t N = (int64_t)N_d;
    if (N <= 0 || na == 0 || a->data[0] == 0.0) return mat_alloc(0, 0);
    std::vector<double> bn(nb), an(na);
    double a0 = a->data[0];
    for (int64_t i = 0; i < nb; ++i) bn[i] = b->data[i] / a0;
    for (int64_t i = 0; i < na; ++i) an[i] = a->data[i] / a0;
    std::vector<double> step((size_t)N, 1.0);
    matlab_mat *S = mat_alloc(N, 1);
    filter_flat_(bn.data(), nb, an.data(), na, step.data(), N, S->data);
    return S;
}

/* grpdelay(b, a, N) — group delay via finite difference on the
 * unwrapped phase of H(e^{jω}). Returns an N-point real column at
 * the same N frequency points freqz uses ([0, π) equally spaced).
 * Algorithm: compute H at N points and at N points slightly offset;
 * τ ≈ −Δarg / Δω, with arg unwrapped via cumulative atan2 unwrapping. */
matlab_mat *matlab_grpdelay(matlab_mat *b, matlab_mat *a, double N_d) {
    if (!b || !a) return mat_alloc(0, 0);
    int N = (int)N_d;
    if (N <= 1) return mat_alloc(0, 0);
    int64_t nb = b->rows * b->cols;
    int64_t na = a->rows * a->cols;
    if (na == 0 || a->data[0] == 0.0) return mat_alloc(0, 0);
    std::vector<double> bn(nb), an(na);
    double a0 = a->data[0];
    for (int64_t i = 0; i < nb; ++i) bn[i] = b->data[i] / a0;
    for (int64_t i = 0; i < na; ++i) an[i] = a->data[i] / a0;
    /* Evaluate H at the N freqz frequency grid and at a tiny offset
     * to compute the local derivative of phase. */
    matlab_mat *G = mat_alloc(N, 1);
    double dw = (M_PI / (double)N) * 1e-4;        /* small offset */
    for (int k = 0; k < N; ++k) {
        double w0 = M_PI * (double)k / (double)N;
        double w1 = w0 + dw;
        auto evalArg = [&](double w) {
            double nr = 0, ni = 0;
            for (int64_t i = 0; i < nb; ++i) {
                double a_ = -w * (double)i;
                nr += bn[i] * cos(a_);
                ni += bn[i] * sin(a_);
            }
            double dr = 0, di = 0;
            for (int64_t i = 0; i < na; ++i) {
                double a_ = -w * (double)i;
                dr += an[i] * cos(a_);
                di += an[i] * sin(a_);
            }
            double denom = dr * dr + di * di;
            double hr = (nr * dr + ni * di) / denom;
            double hi = (ni * dr - nr * di) / denom;
            return atan2(hi, hr);
        };
        double arg0 = evalArg(w0);
        double arg1 = evalArg(w1);
        /* Unwrap one step: keep the difference in (-π, π]. */
        double d = arg1 - arg0;
        while (d >  M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        G->data[k] = -d / dw;
    }
    return G;
}

/* chol(A): upper-triangular Cholesky factor R such that R'*R = A,
 * for a symmetric positive-definite A. Returns a zero matrix if A
 * is not SPD (i.e. a negative diagonal appears). */
matlab_mat *matlab_chol(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    /* Phase-4 RAII: MatPtr ensures the descriptor is freed if any
     * intermediate path throws (none today, but defensive). */
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(n, n);
    /* Upper-triangular factor, row-major. */
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            double s = A->data[i * n + j];
            for (int64_t k = 0; k < i; ++k)
                s -= R->data[k * n + i] * R->data[k * n + j];
            if (i == j) {
                if (s <= 0.0) {
                    /* Not SPD — Phase-6 helper sets the runtime error
                     * message + flag and returns a fresh empty matrix
                     * (R is dropped via MatPtr's destructor). */
                    return matlab::runtime::fail_with_msg(
                        "chol: matrix is not positive definite", 38);
                }
                R->data[i * n + j] = sqrt(s);
            } else {
                R->data[i * n + j] = s / R->data[i * n + i];
            }
        }
    }
    return R.release();
}

/* pinv(A): Moore-Penrose pseudoinverse via the normal-equation route
 * appropriate for the matrix's shape:
 *   - square & invertible: pinv(A) = inv(A).
 *   - tall   (m > n): pinv(A) = (A' A)^-1 A'.
 *   - wide   (m < n): pinv(A) = A' (A A')^-1.
 * Numerically fine for well-conditioned inputs; real MATLAB uses SVD
 * for rank-deficient cases, which we don't have. */
matlab_mat *matlab_pinv(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m == n) return matlab_inv(A);
    matlab_mat *AT = matlab_transpose(A);
    if (m > n) {
        /* pinv = (A' A)^-1 A' */
        matlab_mat *ATA = matlab_matmul_mm(AT, A);
        matlab_mat *inv_ATA = matlab_inv(ATA);
        matlab_mat *R = matlab_matmul_mm(inv_ATA, AT);
        return R;
    }
    /* m < n: pinv = A' (A A')^-1 */
    matlab_mat *AAT = matlab_matmul_mm(A, AT);
    matlab_mat *inv_AAT = matlab_inv(AAT);
    matlab_mat *R = matlab_matmul_mm(AT, inv_AAT);
    return R;
}

/* LU with partial pivoting.
 *
 * Decomposes A (n x n) into P*A = L*U where L is unit lower-triangular
 * and U is upper-triangular. The two-output variants return L and U
 * separately (P is implicit — users who need it can't currently get
 * it, but the factors returned are for the permuted matrix).
 */
static void lu_factor(matlab_mat *A, matlab_mat *L, matlab_mat *U,
                      int64_t *piv) {
    int64_t n = A->rows;
    /* Copy A -> U initially; L starts as identity. */
    for (int64_t k = 0; k < n * n; ++k) U->data[k] = A->data[k];
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            L->data[i * n + j] = (i == j) ? 1.0 : 0.0;
        piv[i] = i;
    }
    for (int64_t k = 0; k < n; ++k) {
        /* Partial pivot: find the largest |U[k..n-1, k]| */
        int64_t p = k;
        double best = fabs(U->data[k * n + k]);
        for (int64_t i = k + 1; i < n; ++i) {
            double v = fabs(U->data[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        if (p != k) {
            /* swap rows p, k in U */
            for (int64_t j = 0; j < n; ++j) {
                double t = U->data[k * n + j];
                U->data[k * n + j] = U->data[p * n + j];
                U->data[p * n + j] = t;
            }
            /* swap rows p, k in L's computed columns (j < k) */
            for (int64_t j = 0; j < k; ++j) {
                double t = L->data[k * n + j];
                L->data[k * n + j] = L->data[p * n + j];
                L->data[p * n + j] = t;
            }
            int64_t t = piv[k]; piv[k] = piv[p]; piv[p] = t;
        }
        double diag = U->data[k * n + k];
        if (diag == 0.0) continue; /* singular pivot; leave as-is */
        for (int64_t i = k + 1; i < n; ++i) {
            double factor = U->data[i * n + k] / diag;
            L->data[i * n + k] = factor;
            for (int64_t j = k; j < n; ++j)
                U->data[i * n + j] -= factor * U->data[k * n + j];
        }
    }
}

matlab_mat *matlab_lu_L(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    /* Phase-4 RAII: U + piv are scratch; L is the result. The previous
     * code freed U with two manual free()s and a malloc'd piv array. */
    matlab::runtime::MatPtr L = matlab::runtime::make_mat(n, n);
    matlab::runtime::MatPtr U = matlab::runtime::make_mat(n, n);
    std::vector<int64_t> piv(n);
    lu_factor(A, L.get(), U.get(), piv.data());
    return L.release();
    /* U + piv freed by RAII as the function returns. */
}

matlab_mat *matlab_lu_U(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    matlab::runtime::MatPtr L = matlab::runtime::make_mat(n, n);
    matlab::runtime::MatPtr U = matlab::runtime::make_mat(n, n);
    std::vector<int64_t> piv(n);
    lu_factor(A, L.get(), U.get(), piv.data());
    return U.release();
}

/* QR via classical Gram-Schmidt (with re-orthogonalisation pass for
 * decent numeric behaviour). A is m x n with m >= n; Q is m x n,
 * R is n x n. Rank-deficient columns get zero columns in Q. */
static void qr_factor(matlab_mat *A, matlab_mat *Q, matlab_mat *R) {
    int64_t m = A->rows, n = A->cols;
    /* Copy columns of A into Q's storage then orthogonalise. */
    for (int64_t k = 0; k < m * n; ++k) Q->data[k] = A->data[k];
    for (int64_t k = 0; k < n * n; ++k) R->data[k] = 0.0;

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < j; ++i) {
            double dot = 0.0;
            for (int64_t r = 0; r < m; ++r)
                dot += Q->data[r * n + i] * Q->data[r * n + j];
            R->data[i * n + j] = dot;
            for (int64_t r = 0; r < m; ++r)
                Q->data[r * n + j] -= dot * Q->data[r * n + i];
        }
        double nrm = 0.0;
        for (int64_t r = 0; r < m; ++r) {
            double v = Q->data[r * n + j];
            nrm += v * v;
        }
        nrm = sqrt(nrm);
        R->data[j * n + j] = nrm;
        if (nrm > 0.0) {
            for (int64_t r = 0; r < m; ++r) Q->data[r * n + j] /= nrm;
        }
    }
}

matlab_mat *matlab_qr_Q(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m < n) return mat_alloc(0, 0);
    /* Phase-4 RAII — R is scratch, Q is the result. */
    matlab::runtime::MatPtr Q = matlab::runtime::make_mat(m, n);
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(n, n);
    qr_factor(A, Q.get(), R.get());
    return Q.release();
}

matlab_mat *matlab_qr_R(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m < n) return mat_alloc(0, 0);
    matlab::runtime::MatPtr Q = matlab::runtime::make_mat(m, n);
    matlab::runtime::MatPtr R = matlab::runtime::make_mat(n, n);
    qr_factor(A, Q.get(), R.get());
    return R.release();
}

/*=========================================================================
 * Tier-3 builtins: rank, cond, null, orth, imfilter, padarray, interp2,
 * upsample, downsample.
 *
 * These build on the existing SVD / EIG / QR / conv2 primitives — none
 * of them implement new core numeric kernels, so the failure modes of
 * the underlying routines (Jacobi eig only handles symmetric inputs,
 * Gram-Schmidt QR is unpivoted, SVD returns σ-values only) propagate
 * directly. See docs/runtime.md "Tier 3" for the implications.
 *=========================================================================*/

double matlab_rank(matlab_mat *A) {
    if (!A) return 0.0;
    int64_t m = A->rows, n = A->cols;
    if (m == 0 || n == 0) return 0.0;
    matlab_mat *S = matlab_svd(A);
    int64_t k = S->rows * S->cols;
    if (k == 0) return 0.0;
    double smax = S->data[0];
    double tol = (double)(m > n ? m : n) * smax * 2.220446049250313e-16;
    int64_t r = 0;
    for (int64_t i = 0; i < k; ++i) if (S->data[i] > tol) r++;
    return (double)r;
}

double matlab_cond(matlab_mat *A) {
    if (!A) return 0.0;
    matlab_mat *S = matlab_svd(A);
    int64_t k = S->rows * S->cols;
    if (k == 0) return 0.0;
    double smax = S->data[0];
    double smin = S->data[k - 1];
    if (smin == 0.0) return INFINITY;
    return smax / smin;
}

/* null(A): orthonormal basis for ker(A). Symmetric eig of A'*A —
 * eigenvectors with eigenvalue ≈ 0 form the null-space basis. Tolerance
 * is max-eig * n * eps (matches MATLAB's default rtol). */
matlab_mat *matlab_null(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m == 0 || n == 0) return mat_alloc(n, n);
    matlab_mat *AT = matlab_transpose(A);
    matlab_mat *ATA = matlab_matmul_mm(AT, A);   /* n x n, symmetric */
    matlab_mat *V = matlab_eig_V(ATA);            /* n x n */
    matlab_mat *D = matlab_eig_D(ATA);            /* n x n diag */
    double lmax = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = D->data[i * n + i];
        if (d > lmax) lmax = d;
    }
    double tol = lmax * (double)n * 2.220446049250313e-16;
    int64_t cnt = 0;
    for (int64_t i = 0; i < n; ++i)
        if (D->data[i * n + i] <= tol) cnt++;
    matlab_mat *N = mat_alloc(n, cnt);
    int64_t col = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (D->data[i * n + i] > tol) continue;
        for (int64_t r = 0; r < n; ++r)
            N->data[r * cnt + col] = V->data[r * n + i];
        col++;
    }
    return N;
}

/* orth(A): orthonormal basis for col(A). For m >= n, QR + rank
 * truncation (assumes the leading columns are linearly independent
 * — true for typical full-rank inputs but a known limitation for
 * rank-deficient matrices with non-leading dependent columns). For
 * m < n, eig of A*A' is reliable. */
matlab_mat *matlab_orth(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m == 0 || n == 0) return mat_alloc(m, 0);
    int64_t r = (int64_t)matlab_rank(A);
    if (r == 0) return mat_alloc(m, 0);
    if (m >= n) {
        matlab_mat *Q = matlab_qr_Q(A);
        if (r == n) return Q;
        matlab_mat *Qr = mat_alloc(m, r);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < r; ++j)
                Qr->data[i * r + j] = Q->data[i * n + j];
        return Qr;
    }
    /* m < n: eig of A*A' (m x m, symmetric). */
    matlab_mat *AT = matlab_transpose(A);
    matlab_mat *AAT = matlab_matmul_mm(A, AT);
    matlab_mat *V = matlab_eig_V(AAT);
    matlab_mat *D = matlab_eig_D(AAT);
    double lmax = 0.0;
    for (int64_t i = 0; i < m; ++i) {
        double d = D->data[i * m + i];
        if (d > lmax) lmax = d;
    }
    double tol = lmax * (double)m * 2.220446049250313e-16;
    matlab_mat *Q = mat_alloc(m, r);
    int64_t col = 0;
    for (int64_t i = 0; i < m; ++i) {
        if (D->data[i * m + i] <= tol) continue;
        if (col >= r) break;
        for (int64_t row = 0; row < m; ++row)
            Q->data[row * r + col] = V->data[row * m + i];
        col++;
    }
    return Q;
}

/* imfilter(A, h): conv2(A, h) cropped to A's size. Centre-aligned —
 * the kernel's centre is floor(size(h)/2). */
matlab_mat *matlab_imfilter(matlab_mat *A, matlab_mat *h) {
    if (!A || !h) return mat_alloc(0, 0);
    int64_t am = A->rows, an = A->cols;
    int64_t bm = h->rows, bn = h->cols;
    if (am == 0 || an == 0 || bm == 0 || bn == 0) return mat_alloc(0, 0);
    matlab_mat *full = matlab_conv2(A, h);
    int64_t cn = an + bn - 1;
    int64_t off_r = (bm - 1) / 2;
    int64_t off_c = (bn - 1) / 2;
    matlab_mat *R = mat_alloc(am, an);
    for (int64_t i = 0; i < am; ++i)
        for (int64_t j = 0; j < an; ++j)
            R->data[i * an + j] = full->data[(i + off_r) * cn + (j + off_c)];
    return R;
}

/* padarray(A, padsize): zero-pad. padsize is [pre_rows pre_cols] or a
 * scalar applied to both. Symmetric (same padding before / after). */
matlab_mat *matlab_padarray(matlab_mat *A, matlab_mat *padsize) {
    if (!A || !padsize) return mat_alloc(0, 0);
    int64_t ps_n = padsize->rows * padsize->cols;
    int64_t pad_r, pad_c;
    if (ps_n >= 2)      { pad_r = (int64_t)padsize->data[0];
                           pad_c = (int64_t)padsize->data[1]; }
    else if (ps_n == 1) { pad_r = pad_c = (int64_t)padsize->data[0]; }
    else                  return mat_alloc(0, 0);
    if (pad_r < 0) pad_r = 0;
    if (pad_c < 0) pad_c = 0;
    int64_t am = A->rows, an = A->cols;
    int64_t out_m = am + 2 * pad_r;
    int64_t out_n = an + 2 * pad_c;
    matlab_mat *R = mat_alloc(out_m, out_n);
    for (int64_t i = 0; i < am; ++i)
        for (int64_t j = 0; j < an; ++j)
            R->data[(i + pad_r) * out_n + (j + pad_c)] = A->data[i * an + j];
    return R;
}

/* interp2(X, Y, V, Xq, Yq): bilinear interpolation. X is a sorted 1xN
 * row, Y a sorted Mx1 column, V is MxN. Out-of-range queries → NaN. */
matlab_mat *matlab_interp2(matlab_mat *X, matlab_mat *Y, matlab_mat *V,
                           matlab_mat *Xq, matlab_mat *Yq) {
    if (!X || !Y || !V || !Xq || !Yq) return mat_alloc(0, 0);
    int64_t nx = X->rows * X->cols;
    int64_t ny = Y->rows * Y->cols;
    if (nx == 0 || ny == 0 || V->rows != ny || V->cols != nx)
        return mat_alloc(0, 0);
    int64_t m = Xq->rows * Xq->cols;
    if (m != Yq->rows * Yq->cols) return mat_alloc(0, 0);
    matlab_mat *R = mat_alloc(Xq->rows, Xq->cols);
    double xmin = X->data[0], xmax = X->data[nx - 1];
    double ymin = Y->data[0], ymax = Y->data[ny - 1];
    for (int64_t i = 0; i < m; ++i) {
        double xq = Xq->data[i], yq = Yq->data[i];
        if (xq < xmin || xq > xmax || yq < ymin || yq > ymax) {
            R->data[i] = NAN; continue;
        }
        int64_t xlo = 0, xhi = nx - 1;
        while (xhi - xlo > 1) {
            int64_t mid = (xlo + xhi) / 2;
            if (X->data[mid] <= xq) xlo = mid; else xhi = mid;
        }
        int64_t ylo = 0, yhi = ny - 1;
        while (yhi - ylo > 1) {
            int64_t mid = (ylo + yhi) / 2;
            if (Y->data[mid] <= yq) ylo = mid; else yhi = mid;
        }
        double x0 = X->data[xlo], x1 = X->data[xhi];
        double y0 = Y->data[ylo], y1 = Y->data[yhi];
        double tx = (x1 == x0) ? 0.0 : (xq - x0) / (x1 - x0);
        double ty = (y1 == y0) ? 0.0 : (yq - y0) / (y1 - y0);
        double v00 = V->data[ylo * nx + xlo];
        double v01 = V->data[ylo * nx + xhi];
        double v10 = V->data[yhi * nx + xlo];
        double v11 = V->data[yhi * nx + xhi];
        double v_top    = v00 * (1.0 - tx) + v01 * tx;
        double v_bottom = v10 * (1.0 - tx) + v11 * tx;
        R->data[i] = v_top * (1.0 - ty) + v_bottom * ty;
    }
    return R;
}

/* upsample / downsample. Output orientation mirrors the input; works
 * on 1-D vectors only (matrix inputs are flattened). */
matlab_mat *matlab_upsample(matlab_mat *x, double n_d) {
    if (!x) return mat_alloc(0, 0);
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    int64_t L = x->rows * x->cols;
    int64_t outL = L * n;
    int is_col = (x->cols == 1 && x->rows > 1);
    matlab_mat *R = is_col ? mat_alloc(outL, 1) : mat_alloc(1, outL);
    for (int64_t i = 0; i < L; ++i) R->data[i * n] = x->data[i];
    return R;
}

matlab_mat *matlab_downsample(matlab_mat *x, double n_d) {
    if (!x) return mat_alloc(0, 0);
    int64_t n = (int64_t)n_d;
    if (n < 1) n = 1;
    int64_t L = x->rows * x->cols;
    int64_t outL = (L + n - 1) / n;
    int is_col = (x->cols == 1 && x->rows > 1);
    matlab_mat *R = is_col ? mat_alloc(outL, 1) : mat_alloc(1, outL);
    for (int64_t i = 0; i < outL; ++i) R->data[i] = x->data[i * n];
    return R;
}

/* ============================================================================
 * REPL workspace + DAP hook infrastructure are in runtime/runtime_debug.cpp
 * (Phase 2 of docs/port_runtime_2_cpp.md). Originally lived inline here;
 * extracted to its own TU so matlab_runtime.cpp stays focused on the
 * numerical core. The two TUs share private layouts via runtime_internal.h.
 * ==========================================================================*/

/* rmfield(s, 'name'): remove a field in place and return the same ptr.
 * MATLAB's rmfield conceptually returns a new struct, but mutating
 * in place + returning the same pointer matches the common
 * `s = rmfield(s, 'x')` idiom. If the field doesn't exist we leave
 * the struct untouched. */
matlab_struct *matlab_struct_rmfield(matlab_struct *s, const char *name,
                                      int64_t len) {
    if (!s) return s;
    int32_t idx = struct_find_field(s, name, (int32_t)len);
    if (idx < 0) return s;
    /* Free the heap-copied name and shift the remaining entries left. */
    free(s->names[idx]);
    for (int32_t i = idx; i < s->nfields - 1; ++i) {
        s->names[i]    = s->names[i + 1];
        s->kinds[i]    = s->kinds[i + 1];
        s->f64_vals[i] = s->f64_vals[i + 1];
        s->ptr_vals[i] = s->ptr_vals[i + 1];
    }
    --s->nfields;
    return s;
}

/* ---------------------------------------------------------------------- */
/* Cell arrays — 1-D tagged containers.
 *
 * Each slot is tagged with a kind (0 = f64, 1 = matlab_mat*). Index is
 * 1-based to match MATLAB. Out-of-range get returns 0.0 (f64) or an
 * empty matrix (mat). Autogrows on set past end.
 */
/* Phase 1.3: 2-D cells. The legacy n-only descriptor is preserved as
 * the 1-D / "row vector" shape (rows=1, cols=n). 2-D cells track rows
 * and cols explicitly; element layout is row-major so the 1-D
 * accessors keep working over the linear backing arrays. */
struct matlab_cell_s {
    int32_t n;        /* total element count (rows * cols) */
    int32_t cap;
    int32_t rows;     /* Phase 1.3 */
    int32_t cols;     /* Phase 1.3 */
    int32_t *kinds;
    double *f64_vals;
    void **ptr_vals;
};
typedef struct matlab_cell_s matlab_cell;

static void cell_grow_to(matlab_cell *c, int32_t need) {
    if (c->cap >= need) return;
    int32_t NewCap = c->cap ? c->cap : 4;
    while (NewCap < need) NewCap *= 2;
    c->kinds    = (int32_t *)realloc(c->kinds,    (size_t)NewCap * sizeof(int32_t));
    c->f64_vals = (double *)realloc(c->f64_vals,  (size_t)NewCap * sizeof(double));
    c->ptr_vals = (void **)realloc(c->ptr_vals,   (size_t)NewCap * sizeof(void *));
    for (int32_t i = c->cap; i < NewCap; ++i) {
        c->kinds[i] = 0;
        c->f64_vals[i] = 0.0;
        c->ptr_vals[i] = NULL;
    }
    c->cap = NewCap;
}

matlab_cell *matlab_cell_new(double n) {
    matlab_cell *c = (matlab_cell *)calloc(1, sizeof(*c));
    int32_t cap0 = n > 0 ? (int32_t)n : 4;
    cell_grow_to(c, cap0);
    /* 1-D: a row vector (1 x n). The 1-D accessors set c->n directly,
     * which is treated as the column count when rows==1. */
    c->rows = 1;
    c->cols = (int32_t)(n > 0 ? n : 0);
    return c;
}

/* Phase 1.3: 2-D cell construction. Allocates a rows*cols backing,
 * sets the shape fields, and pre-populates n = rows*cols so the
 * existing 1-D accessors (cell_numel, cell_get_*) work over the
 * row-major linear layout. */
matlab_cell *matlab_cell_new_2d(double rows, double cols) {
    matlab_cell *c = (matlab_cell *)calloc(1, sizeof(*c));
    int32_t r = rows > 0 ? (int32_t)rows : 0;
    int32_t k = cols > 0 ? (int32_t)cols : 0;
    int32_t need = r * k;
    cell_grow_to(c, need > 0 ? need : 1);
    c->rows = r;
    c->cols = k;
    c->n = need;
    return c;
}

void matlab_cell_set_f64(matlab_cell *c, double i1, double v) {
    if (!c) return;
    int32_t i = (int32_t)i1 - 1;
    if (i < 0) return;
    if (i >= c->cap) cell_grow_to(c, i + 1);
    if (i >= c->n) c->n = i + 1;
    c->kinds[i] = 0;
    c->f64_vals[i] = v;
    c->ptr_vals[i] = NULL;
}

void matlab_cell_set_mat(matlab_cell *c, double i1, matlab_mat *m) {
    if (!c) return;
    int32_t i = (int32_t)i1 - 1;
    if (i < 0) return;
    if (i >= c->cap) cell_grow_to(c, i + 1);
    if (i >= c->n) c->n = i + 1;
    c->kinds[i] = 1;
    c->f64_vals[i] = 0.0;
    c->ptr_vals[i] = m;
}

double matlab_cell_get_f64(matlab_cell *c, double i1) {
    if (!c) return 0.0;
    int32_t i = (int32_t)i1 - 1;
    if (i < 0 || i >= c->n) return 0.0;
    if (c->kinds[i] == 0) return c->f64_vals[i];
    /* If the slot holds a 1x1 matrix, unbox to scalar. */
    if (c->kinds[i] == 1 && c->ptr_vals[i]) {
        matlab_mat *m = (matlab_mat *)c->ptr_vals[i];
        if (m->rows == 1 && m->cols == 1) return m->data[0];
    }
    return 0.0;
}

matlab_mat *matlab_cell_get_mat(matlab_cell *c, double i1) {
    if (!c) return mat_alloc(0, 0);
    int32_t i = (int32_t)i1 - 1;
    if (i < 0 || i >= c->n) return mat_alloc(0, 0);
    if (c->kinds[i] == 1 && c->ptr_vals[i])
        return (matlab_mat *)c->ptr_vals[i];
    if (c->kinds[i] == 0) {
        matlab_mat *m = mat_alloc(1, 1);
        m->data[0] = c->f64_vals[i];
        return m;
    }
    return mat_alloc(0, 0);
}

double matlab_cell_numel(matlab_cell *c) {
    if (!c) return 0.0;
    return (double)c->n;
}

double matlab_iscell(matlab_cell *c) {
    return c ? 1.0 : 0.0;
}

/* ===== Phase 1.3 — 2-D cell accessors and shape ===== */

double matlab_cell_rows(matlab_cell *c) {
    if (!c) return 0.0;
    /* Legacy 1-D cells default rows=1 / cols=n in matlab_cell_new; an
     * old-school cell that was constructed via direct n-grow still has
     * rows=0 and cols=0, in which case we fall back to (1, n). */
    return c->rows > 0 ? (double)c->rows : (c->n > 0 ? 1.0 : 0.0);
}

double matlab_cell_cols(matlab_cell *c) {
    if (!c) return 0.0;
    return c->cols > 0 ? (double)c->cols : (double)c->n;
}

double matlab_cell_size_dim(matlab_cell *c, double dim) {
    if (!c) return 0.0;
    int d = (int)dim;
    if (d == 1) return matlab_cell_rows(c);
    if (d == 2) return matlab_cell_cols(c);
    return 1.0;
}

static int32_t cell_lin_2d(matlab_cell *c, double r1, double k1) {
    /* Row-major: idx = (r-1)*cols + (k-1). */
    int32_t r = (int32_t)r1 - 1;
    int32_t k = (int32_t)k1 - 1;
    int32_t cols = c->cols > 0 ? c->cols : c->n;
    if (r < 0 || k < 0 || cols <= 0) return -1;
    return r * cols + k;
}

void matlab_cell_set_f64_2d(matlab_cell *c, double r1, double k1, double v) {
    if (!c) return;
    int32_t i = cell_lin_2d(c, r1, k1);
    if (i < 0) return;
    if (i >= c->cap) cell_grow_to(c, i + 1);
    if (i >= c->n) c->n = i + 1;
    c->kinds[i] = 0;
    c->f64_vals[i] = v;
    c->ptr_vals[i] = NULL;
}

void matlab_cell_set_mat_2d(matlab_cell *c, double r1, double k1, matlab_mat *m) {
    if (!c) return;
    int32_t i = cell_lin_2d(c, r1, k1);
    if (i < 0) return;
    if (i >= c->cap) cell_grow_to(c, i + 1);
    if (i >= c->n) c->n = i + 1;
    c->kinds[i] = 1;
    c->f64_vals[i] = 0.0;
    c->ptr_vals[i] = m;
}

double matlab_cell_get_f64_2d(matlab_cell *c, double r1, double k1) {
    if (!c) return 0.0;
    int32_t i = cell_lin_2d(c, r1, k1);
    if (i < 0 || i >= c->n) return 0.0;
    if (c->kinds[i] == 0) return c->f64_vals[i];
    if (c->kinds[i] == 1 && c->ptr_vals[i]) {
        matlab_mat *m = (matlab_mat *)c->ptr_vals[i];
        if (m->rows == 1 && m->cols == 1) return m->data[0];
    }
    return 0.0;
}

matlab_mat *matlab_cell_get_mat_2d(matlab_cell *c, double r1, double k1) {
    if (!c) return mat_alloc(0, 0);
    int32_t i = cell_lin_2d(c, r1, k1);
    if (i < 0 || i >= c->n) return mat_alloc(0, 0);
    if (c->kinds[i] == 1 && c->ptr_vals[i])
        return (matlab_mat *)c->ptr_vals[i];
    if (c->kinds[i] == 0) {
        matlab_mat *m = mat_alloc(1, 1);
        m->data[0] = c->f64_vals[i];
        return m;
    }
    return mat_alloc(0, 0);
}

/* Cell concat: [a, b] (horizontal) requires matching row counts;
 * [a; b] (vertical) requires matching col counts. The result is a
 * fresh cell that reuses the source cells' element pointers (cells
 * own descriptors via the original allocs, not via copies). */
matlab_cell *matlab_cell_concat_row(matlab_cell *a, matlab_cell *b) {
    if (!a) return b;
    if (!b) return a;
    int32_t ar = a->rows > 0 ? a->rows : 1;
    int32_t br = b->rows > 0 ? b->rows : 1;
    int32_t ac = a->cols > 0 ? a->cols : a->n;
    int32_t bc = b->cols > 0 ? b->cols : b->n;
    if (ar != br) return matlab_cell_new(0);
    int32_t nc = ac + bc;
    matlab_cell *c = matlab_cell_new_2d((double)ar, (double)nc);
    for (int32_t r = 0; r < ar; ++r) {
        for (int32_t kk = 0; kk < ac; ++kk) {
            int32_t src = r * ac + kk;
            int32_t dst = r * nc + kk;
            c->kinds[dst]    = a->kinds[src];
            c->f64_vals[dst] = a->f64_vals[src];
            c->ptr_vals[dst] = a->ptr_vals[src];
        }
        for (int32_t kk = 0; kk < bc; ++kk) {
            int32_t src = r * bc + kk;
            int32_t dst = r * nc + ac + kk;
            c->kinds[dst]    = b->kinds[src];
            c->f64_vals[dst] = b->f64_vals[src];
            c->ptr_vals[dst] = b->ptr_vals[src];
        }
    }
    return c;
}

matlab_cell *matlab_cell_concat_col(matlab_cell *a, matlab_cell *b) {
    if (!a) return b;
    if (!b) return a;
    int32_t ar = a->rows > 0 ? a->rows : 1;
    int32_t br = b->rows > 0 ? b->rows : 1;
    int32_t ac = a->cols > 0 ? a->cols : a->n;
    int32_t bc = b->cols > 0 ? b->cols : b->n;
    if (ac != bc) return matlab_cell_new(0);
    int32_t nr = ar + br;
    matlab_cell *c = matlab_cell_new_2d((double)nr, (double)ac);
    /* Top: copy a row-major into rows [0, ar). */
    for (int32_t i = 0; i < ar * ac; ++i) {
        c->kinds[i]    = a->kinds[i];
        c->f64_vals[i] = a->f64_vals[i];
        c->ptr_vals[i] = a->ptr_vals[i];
    }
    /* Bottom: copy b row-major into rows [ar, ar+br). */
    for (int32_t i = 0; i < br * bc; ++i) {
        int32_t dst = ar * ac + i;
        c->kinds[dst]    = b->kinds[i];
        c->f64_vals[dst] = b->f64_vals[i];
        c->ptr_vals[dst] = b->ptr_vals[i];
    }
    return c;
}

/* ---------------------------------------------------------------------- */

/* Get-or-create a nested child struct at s.name. Returns the child
 * struct pointer, creating an empty one and stashing it in the parent
 * if the field doesn't exist yet. Used for s.a.b = v to resolve the
 * intermediate s.a level. */
matlab_struct *matlab_struct_get_child_struct(matlab_struct *s,
                                               const char *name, int64_t len) {
    if (!s) return matlab_struct_new();
    int32_t idx = struct_find_field(s, name, (int32_t)len);
    if (idx >= 0 && s->kinds[idx] == 2 && s->ptr_vals[idx])
        return (matlab_struct *)s->ptr_vals[idx];
    matlab_struct *child = matlab_struct_new();
    idx = struct_reserve(s, name, (int32_t)len);
    s->kinds[idx] = 2;
    s->ptr_vals[idx] = child;
    return child;
}

/* ====================================================================== */
/* Phase 2 — Struct arrays.
 *
 * `s(i).x = v` and `s(i).x` read. A matlab_struct_arr is a 1-D vector of
 * matlab_struct* pointers; element 1..N each hold their own field set.
 * Auto-grows on OOB write (MATLAB fills the gap with empty structs);
 * reads OOB return an empty struct so the field-get path silently
 * returns 0 / NULL rather than dereferencing garbage.
 *
 * The DAP / scope inspector treats struct_arr as a separate "kind" via
 * matlab_dbg_ws_kind == 6 (handled by the Lowering side: workspace
 * stores/loads use matlab_ws_set_struct_arr / _get_struct_arr).
 * ====================================================================== */

struct matlab_struct_arr_s {
    int32_t n;
    int32_t cap;
    matlab_struct **elems;
};
typedef struct matlab_struct_arr_s matlab_struct_arr;

static void struct_arr_grow_to(matlab_struct_arr *a, int32_t need) {
    if (a->cap >= need) return;
    int32_t NewCap = a->cap ? a->cap : 4;
    while (NewCap < need) NewCap *= 2;
    a->elems = (matlab_struct **)realloc(a->elems,
                                          (size_t)NewCap * sizeof(matlab_struct *));
    for (int32_t i = a->cap; i < NewCap; ++i) a->elems[i] = NULL;
    a->cap = NewCap;
}

extern "C" matlab_struct_arr *matlab_struct_arr_new(void) {
    return (matlab_struct_arr *)calloc(1, sizeof(matlab_struct_arr));
}

/* Auto-grows to index i (1-based) and returns the element struct
 * pointer, creating empty structs at any newly-reachable indices. */
extern "C" matlab_struct *matlab_struct_arr_get_or_create(
        matlab_struct_arr *a, double i1) {
    if (!a) return matlab_struct_new();
    int32_t i = (int32_t)i1 - 1;
    if (i < 0) return matlab_struct_new();
    if (i >= a->cap) struct_arr_grow_to(a, i + 1);
    /* Fill any gap (and the slot itself) with empty structs so reads
     * via matlab_struct_arr_get on intermediate indices return a real
     * struct rather than NULL. */
    for (int32_t k = a->n; k <= i; ++k) {
        if (!a->elems[k]) a->elems[k] = matlab_struct_new();
    }
    if (i >= a->n) a->n = i + 1;
    if (!a->elems[i]) a->elems[i] = matlab_struct_new();
    return a->elems[i];
}

/* Read-only access; OOB returns a fresh empty struct (so the
 * downstream matlab_struct_get_* call returns 0 / NULL cleanly
 * instead of segfaulting). The empty struct is leaked here, which
 * is fine for the test corpus and matches the leak shape of
 * matlab_struct_get_mat / mat_alloc(0,0) on missing fields. */
extern "C" matlab_struct *matlab_struct_arr_get(matlab_struct_arr *a, double i1) {
    if (!a) return matlab_struct_new();
    int32_t i = (int32_t)i1 - 1;
    if (i < 0 || i >= a->n) return matlab_struct_new();
    if (!a->elems[i]) return matlab_struct_new();
    return a->elems[i];
}

extern "C" double matlab_struct_arr_length(matlab_struct_arr *a) {
    if (!a) return 0.0;
    return (double)a->n;
}

extern "C" double matlab_struct_arr_numel(matlab_struct_arr *a) {
    return matlab_struct_arr_length(a);
}

extern "C" double matlab_struct_arr_size_dim(matlab_struct_arr *a, double dim) {
    /* MATLAB struct arrays default to a row vector (1 x N). */
    if (!a) return 0.0;
    int d = (int)dim;
    if (d == 1) return a->n > 0 ? 1.0 : 0.0;
    if (d == 2) return (double)a->n;
    return 1.0;
}

/* ---------------------------------------------------------------------- */
/* Global / persistent storage.
 *
 * The compiler assigns a unique integer ID per global or persistent name
 * (persistent names are namespaced by the declaring function). Each ID
 * indexes a flat scalar table. matlab_global_get_f64 reads the current
 * value; matlab_global_set_f64 writes it. Unset slots read as 0.0.
 *
 * Capacity is fixed at compile time — 128 slots cover any plausible
 * hand-written MATLAB program in the test suite. Bumping it just means
 * enlarging the array; no dynamic growth because the IDs are handed out
 * in compile order and never freed.
 *
 * No mutex: single-threaded reads/writes. parfor bodies don't currently
 * access globals (their slots are captured by value via the reduction
 * dispatcher). If that ever changes we'll need one.
 */
#define MATLAB_GLOBAL_TABLE_SIZE 128
static double matlab_global_table[MATLAB_GLOBAL_TABLE_SIZE];
/* Parallel pointer table — used by `persistent` storage of typed values
 * like matlab_mat_i64* (fi arrays). Independent of the f64 table; an ID
 * may legitimately use one or the other depending on the binding's
 * declared type. NULL signifies "unset" so the caller (lowered isempty
 * check) initialises on the first read. See plan §12. */
static void *matlab_global_ptr_table[MATLAB_GLOBAL_TABLE_SIZE];

double matlab_global_get_f64(int32_t id) {
    if (id < 0 || id >= MATLAB_GLOBAL_TABLE_SIZE) return 0.0;
    return matlab_global_table[id];
}

void matlab_global_set_f64(int32_t id, double v) {
    if (id < 0 || id >= MATLAB_GLOBAL_TABLE_SIZE) return;
    matlab_global_table[id] = v;
}

void *matlab_persistent_get_ptr(int32_t id) {
    if (id < 0 || id >= MATLAB_GLOBAL_TABLE_SIZE) return NULL;
    return matlab_global_ptr_table[id];
}

void matlab_persistent_set_ptr(int32_t id, void *p) {
    if (id < 0 || id >= MATLAB_GLOBAL_TABLE_SIZE) return;
    matlab_global_ptr_table[id] = p;
}

double matlab_persistent_isempty(int32_t id) {
    if (id < 0 || id >= MATLAB_GLOBAL_TABLE_SIZE) return 1.0;
    return matlab_global_ptr_table[id] == NULL ? 1.0 : 0.0;
}

/* Class-instance disp. Prints `ClassName with properties:` followed by
 * one line per field. Reads class_id and field metadata directly off
 * the matlab_obj layout — caller must have already confirmed via
 * matlab_obj_is_known() that the pointer really is an obj. */

/* Forward decl — defined in runtime_debug.cpp (Phase-2 split). */
const char *matlab_dbg_class_name(int32_t class_id, int64_t *len_out);

void matlab_disp_obj(matlab_obj *o) {
    if (!o) return;
    int64_t cnLen = 0;
    const char *cn = matlab_dbg_class_name(o->class_id, &cnLen);
    pthread_mutex_lock(&matlab_io_mutex);
    if (cn && cnLen > 0)
        printf("  %.*s with properties:\n\n", (int)cnLen, cn);
    else
        printf("  <class %d> with properties:\n\n", o->class_id);
    for (int i = 0; i < o->nfields; ++i) {
        const char *name = o->names[i] ? o->names[i] : "?";
        printf("    %s: ", name);
        if (o->kinds[i] == 0) {
            printf("%g\n", o->f64_vals[i]);
        } else if (o->kinds[i] == 1 || o->kinds[i] == 2) {
            void *p = o->ptr_vals[i];
            if (!p) { printf("[]\n"); continue; }
            if (matlab_obj_is_known(p)) {
                int32_t ccid = ((matlab_obj *)p)->class_id;
                int64_t ccnLen = 0;
                const char *ccn = matlab_dbg_class_name(ccid, &ccnLen);
                if (ccn && ccnLen > 0)
                    printf("[1x1 %.*s]\n", (int)ccnLen, ccn);
                else
                    printf("[1x1 <class %d>]\n", ccid);
            } else if (mat_is_complex(p) || mat_is_3d(p)) {
                printf("[matrix]\n");
            } else {
                matlab_mat *m = (matlab_mat *)p;
                if (m->rows == 1 && m->cols == 1 && m->data)
                    printf("%g\n", m->data[0]);
                else
                    printf("[%lldx%lld double]\n",
                           (long long)m->rows, (long long)m->cols);
            }
        } else {
            printf("?\n");
        }
    }
    printf("\n");
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* Matrix disp. Special-cases 1×1 to print scalar-style and 1×N to print
 * on one line (matching MATLAB's default disp formatting). Polymorphic:
 * accepts either a real matlab_mat* or a complex matlab_mat_c* — the
 * magic-tag check on the real path keeps the fast-path branch-free
 * for normal use (first-field read that stays in cache).
 *
 * Also defends against a class-instance pointer arriving here: the
 * REPL JIT lowers `disp(<name>)` as `matlab_disp_mat(matlab_ws_get_mat(...))`
 * regardless of whether <name> is bound to a matrix or a class instance,
 * because its fresh Sema can't see the workspace's kind tags. The
 * registry check below routes obj inputs through matlab_disp_obj so
 * the cast-and-deref-as-matrix below doesn't read garbage and SEGV. */
void matlab_disp_mat(void *Aptr) {
    if (!Aptr) return;
    if (matlab_obj_is_known(Aptr)) {
        matlab_disp_obj((matlab_obj *)Aptr);
        return;
    }
    /* String descriptor arriving through the matrix-disp path. The
     * REPL JIT lowers `disp(t)` / bare `t` as matlab_disp_mat(
     * matlab_ws_get_mat(...)) for any pointer-typed binding the
     * fresh-Sema can't classify; route registered string descriptors
     * to matlab_string_disp so the user sees the text instead of a
     * matrix-cast of the descriptor bytes (which used to render as
     * `4 x <heap-garbage>` doubles). */
    if (matlab_string_is_known(Aptr)) {
        matlab_string_disp((matlab_string *)Aptr);
        return;
    }
    /* Typed-int descriptors (Phase 1.1.F). The REPL / DAP path arrives
     * here with the typed-int pointer because the workspace stores
     * everything as matlab_ws_set_mat. The intlane registry tells us
     * the actual lane so we can route to the right disp formatter. */
    {
        int kind = matlab_mat_intlane_kind(Aptr);
        if (kind == 0) { matlab_mat_u8_disp ((matlab_mat_u8  *)Aptr); return; }
        if (kind == 1) { matlab_mat_i32_disp((matlab_mat_i32 *)Aptr); return; }
    }
    if (mat_is_complex(Aptr)) {
        matlab_disp_mat_c((matlab_mat_c *)Aptr);
        return;
    }
    matlab_mat *A = (matlab_mat *)Aptr;
    /* Matches MATLAB: disp of an empty matrix prints nothing. */
    if (A->rows == 0 || A->cols == 0) return;
    if (A->rows == 1 && A->cols == 1) {
        pthread_mutex_lock(&matlab_io_mutex);
        printf("%g\n", A->data[0]);
        pthread_mutex_unlock(&matlab_io_mutex);
        return;
    }
    matlab_disp_mat_f64(A->data, A->rows, A->cols);
}

/* ============================================================================
 * Complex matrix descriptor + FFT family are in runtime/runtime_complex.cpp
 * (Phase 2.5 of docs/port_runtime_2_cpp.md). Originally lived inline here;
 * extracted so matlab_runtime.cpp keeps its focus on the real-side core.
 * The two TUs share the matlab_mat_c layout, MATLAB_MAT_C_MAGIC, and
 * mat_c_alloc via runtime_internal.h.
 * ==========================================================================*/

/* ---------------------------------------------------------------------- */
/* Minimal file I/O.
 *
 * MATLAB exposes file I/O via integer file identifiers (0 = stdin,
 * 1 = stdout, 2 = stderr by convention; 3+ are user-opened files).
 * We keep a small fixed-size table mapping id -> FILE* and return the
 * id as a double to match how other scalars flow through the runtime.
 *
 * Only the common cases are supported in v1:
 *   fid = fopen(path, mode);        % path, mode are string literals
 *   fprintf(fid, fmt);              % write literal
 *   fprintf(fid, fmt, v);           % write one f64
 *   s = fgetl(fid);                 % read one line (no trailing NL)
 *   matlab_feof(fid)                % 1 at EOF else 0
 *   fclose(fid);                    % 0 on success, -1 on failure
 */
#define MATLAB_FILE_TABLE_SIZE 64
static FILE *matlab_file_table[MATLAB_FILE_TABLE_SIZE];
static pthread_mutex_t matlab_file_mutex = PTHREAD_MUTEX_INITIALIZER;
static int matlab_file_table_initialised = 0;

static void matlab_file_table_init(void) {
    if (matlab_file_table_initialised) return;
    matlab_file_table_initialised = 1;
    matlab_file_table[0] = stdin;
    matlab_file_table[1] = stdout;
    matlab_file_table[2] = stderr;
}

double matlab_fopen(matlab_string *path, matlab_string *mode) {
    if (!path || !mode) return -1.0;
    pthread_mutex_lock(&matlab_file_mutex);
    matlab_file_table_init();
    FILE *f = fopen(path->data, mode->data);
    if (!f) {
        pthread_mutex_unlock(&matlab_file_mutex);
        return -1.0;
    }
    int slot = -1;
    for (int i = 3; i < MATLAB_FILE_TABLE_SIZE; ++i) {
        if (!matlab_file_table[i]) { slot = i; break; }
    }
    if (slot < 0) { fclose(f); pthread_mutex_unlock(&matlab_file_mutex); return -1.0; }
    matlab_file_table[slot] = f;
    pthread_mutex_unlock(&matlab_file_mutex);
    return (double)slot;
}

double matlab_fclose(double fd) {
    int i = (int)fd;
    if (i < 3 || i >= MATLAB_FILE_TABLE_SIZE) return -1.0;
    pthread_mutex_lock(&matlab_file_mutex);
    FILE *f = matlab_file_table[i];
    matlab_file_table[i] = NULL;
    pthread_mutex_unlock(&matlab_file_mutex);
    if (!f) return -1.0;
    return fclose(f) == 0 ? 0.0 : -1.0;
}

static FILE *matlab_file_lookup(double fd) {
    int i = (int)fd;
    if (i < 0 || i >= MATLAB_FILE_TABLE_SIZE) return NULL;
    matlab_file_table_init();
    return matlab_file_table[i];
}

void matlab_fprintf_file_str(double fd, matlab_string *fmt) {
    FILE *f = matlab_file_lookup(fd);
    if (!f || !fmt) return;
    char buf[4096];
    int64_t len = expand_escapes(buf, fmt->data, (int64_t)fmt->len);
    if (len < (int64_t)sizeof buf) buf[len] = '\0';
    else buf[sizeof buf - 1] = '\0';
    pthread_mutex_lock(&matlab_io_mutex);
    fputs(buf, f);
    pthread_mutex_unlock(&matlab_io_mutex);
}

void matlab_fprintf_file_f64(double fd, matlab_string *fmt, double v) {
    FILE *f = matlab_file_lookup(fd);
    if (!f || !fmt) return;
    char buf[4096];
    int64_t len = expand_escapes(buf, fmt->data, (int64_t)fmt->len);
    if (len < (int64_t)sizeof buf) buf[len] = '\0';
    else buf[sizeof buf - 1] = '\0';
    pthread_mutex_lock(&matlab_io_mutex);
    fprintf(f, buf, v);
    pthread_mutex_unlock(&matlab_io_mutex);
}

matlab_string *matlab_fgetl(double fd) {
    FILE *f = matlab_file_lookup(fd);
    if (!f) return matlab_string_from_literal("", 0);
    char buf[4096];
    if (!fgets(buf, sizeof buf, f))
        return matlab_string_from_literal("", 0);
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') { buf[len - 1] = '\0'; len--; }
    if (len > 0 && buf[len - 1] == '\r') { buf[len - 1] = '\0'; len--; }
    return matlab_string_from_literal(buf, (int64_t)len);
}

double matlab_feof(double fd) {
    FILE *f = matlab_file_lookup(fd);
    if (!f) return 1.0;
    return feof(f) ? 1.0 : 0.0;
}

/* Binary file I/O.
 *
 * matlab_fread(fd, n) reads up to n doubles (8 bytes each) from the
 * file and returns them as an n-by-1 column matrix. A short read
 * (fewer than n doubles available) shrinks the result — MATLAB
 * behaves the same way.
 *
 * matlab_fwrite_mat(fd, A) writes every element of the matrix A in
 * row-major / column-major (we use row-major internally — the same
 * layout matlab_fread produces). */
matlab_mat *matlab_fread(double fd, double n) {
    FILE *f = matlab_file_lookup(fd);
    int64_t want = (int64_t)n;
    if (want < 0) want = 0;
    matlab_mat *A = mat_alloc(want, 1);
    if (!f || want == 0) { A->rows = 0; return A; }
    size_t got = fread(A->data, sizeof(double), (size_t)want, f);
    A->rows = (int64_t)got;
    return A;
}

double matlab_fwrite_mat(double fd, matlab_mat *A) {
    FILE *f = matlab_file_lookup(fd);
    if (!f || !A) return 0.0;
    size_t n = (size_t)(A->rows * A->cols);
    pthread_mutex_lock(&matlab_io_mutex);
    size_t wrote = fwrite(A->data, sizeof(double), n, f);
    pthread_mutex_unlock(&matlab_io_mutex);
    return (double)wrote;
}

double matlab_fwrite_f64(double fd, double v) {
    FILE *f = matlab_file_lookup(fd);
    if (!f) return 0.0;
    pthread_mutex_lock(&matlab_io_mutex);
    size_t wrote = fwrite(&v, sizeof(double), 1, f);
    pthread_mutex_unlock(&matlab_io_mutex);
    return (double)wrote;
}

/* save / load — custom binary format. NOT MATLAB .mat compatible.
 *
 * Layout: 4-byte magic "MLB1", int64 rows, int64 cols, rows*cols
 * doubles. One matrix per file; no variable names, no structs, no
 * cells. Purpose is to let scripts round-trip numeric data across
 * runs without relying on the MathWorks-specific MAT-File v5
 * format, which would need a dedicated parser.
 *
 * API diverges from MATLAB: save(path, A) takes the *value* rather
 * than a variable-name string, because the compiler doesn't retain
 * variable-name metadata at runtime. Likewise A = load(path)
 * returns the matrix directly, not a struct with fieldname-per-var. */
static const char MATLAB_SAVE_MAGIC[4] = {'M', 'L', 'B', '1'};

double matlab_save_mat(matlab_string *path, matlab_mat *A) {
    if (!path || !A) return -1.0;
    FILE *f = fopen(path->data, "wb");
    if (!f) return -1.0;
    pthread_mutex_lock(&matlab_io_mutex);
    fwrite(MATLAB_SAVE_MAGIC, 1, 4, f);
    fwrite(&A->rows, sizeof(int64_t), 1, f);
    fwrite(&A->cols, sizeof(int64_t), 1, f);
    size_t n = (size_t)(A->rows * A->cols);
    if (n > 0) fwrite(A->data, sizeof(double), n, f);
    pthread_mutex_unlock(&matlab_io_mutex);
    fclose(f);
    return 0.0;
}

matlab_mat *matlab_load_mat(matlab_string *path) {
    if (!path) return mat_alloc(0, 0);
    FILE *f = fopen(path->data, "rb");
    if (!f) return mat_alloc(0, 0);
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, MATLAB_SAVE_MAGIC, 4) != 0) {
        fclose(f);
        return mat_alloc(0, 0);
    }
    int64_t rows = 0, cols = 0;
    if (fread(&rows, sizeof(int64_t), 1, f) != 1 ||
        fread(&cols, sizeof(int64_t), 1, f) != 1) {
        fclose(f);
        return mat_alloc(0, 0);
    }
    matlab_mat *A = mat_alloc(rows, cols);
    size_t n = (size_t)(rows * cols);
    if (n > 0) fread(A->data, sizeof(double), n, f);
    fclose(f);
    return A;
}

/*=========================================================================
 * Initial-value ODE solvers.
 *
 * ode45  — Dormand–Prince 5(4), seven-stage FSAL embedded RK pair.
 * ode23  — Bogacki–Shampine 3(2), four-stage FSAL embedded RK pair.
 *
 * Phase 1 supports scalar y only: f has signature `double(double, double)`.
 * Output grid is the set of accepted step endpoints (no dense interpolation
 * yet). Tolerances and step-control are MATLAB's defaults: rtol = 1e-3,
 * atol = 1e-6, fac = 0.9, fac in [0.2, 5.0], max 100 000 steps.
 *
 * The lowering splits `[t,y] = ode45(...)` into back-to-back
 * matlab_ode45_t / matlab_ode45_y calls sharing the same operands. We
 * memoise the last solve in a thread-local slot so the second call hits
 * the cache instead of re-integrating. */
typedef double (*ode_rhs_t)(double, double);

struct ode_cache_slot {
    void *fp;
    matlab_mat *tspan;
    double y0;
    double rtol;
    double atol;
    double max_step;
    double init_step;
    int refine;
    int print_stats;    /* part of the key — different value re-solves */
    int kind;          /* 45 or 23 — solvers don't share grids */
    matlab_mat *t;
    matlab_mat *y;
    int n_acc;          /* solver stats — read by matlab_ode*_stats     */
    int n_rej;
    int n_fev;
    int valid;
};

#if defined(__GNUC__) || defined(__clang__)
__thread struct ode_cache_slot ode_cache_;
#else
struct ode_cache_slot ode_cache_;
#endif

/* Append (tv, yv) to a growing pair of buffers. Doubles capacity as
 * needed; caller frees both buffers via matlab_mat ownership. */
static void ode_push(double **T, double **Y, int64_t *n, int64_t *cap,
                     double tv, double yv) {
    if (*n == *cap) {
        *cap = (*cap) * 2;
        *T = (double *)realloc(*T, (size_t)(*cap) * sizeof(double));
        *Y = (double *)realloc(*Y, (size_t)(*cap) * sizeof(double));
    }
    (*T)[*n] = tv;
    (*Y)[*n] = yv;
    ++(*n);
}

/* Wrap two heap buffers of length n each into freshly-allocated column
 * matlab_mat descriptors, copying so the caller can free the buffers. */
static void ode_buffers_to_mats(double *T, double *Y, int64_t n,
                                matlab_mat **out_t, matlab_mat **out_y) {
    matlab_mat *Tm = mat_alloc(n, 1);
    matlab_mat *Ym = mat_alloc(n, 1);
    if (n > 0) {
        memcpy(Tm->data, T, (size_t)n * sizeof(double));
        memcpy(Ym->data, Y, (size_t)n * sizeof(double));
    }
    *out_t = Tm;
    *out_y = Ym;
}

/* Cubic Hermite interpolation between (t_n, y_n) and (t_n+1, y_n+1)
 * with derivatives k_n=f(t_n,y_n) and k_n1=f(t_n+1,y_n+1).
 * θ ∈ [0, 1]; returns y at t_n + θ*h. 3rd-order accurate, sufficient for
 * smooth-looking plot output between RK45 step endpoints. */
static inline double ode_hermite(double y, double y1, double k, double k1,
                                  double h, double th) {
    double th2 = th * th;
    double th3 = th2 * th;
    return (2.0*th3 - 3.0*th2 + 1.0) * y
         + (-2.0*th3 + 3.0*th2)     * y1
         + h * (th3 - 2.0*th2 + th) * k
         + h * (th3 - th2)          * k1;
}

static void rk_solve_dp45(ode_rhs_t f,
                           const double *targets, int64_t n_targets,
                           double y0,
                           double rtol, double atol,
                           double max_step, double init_step, int refine,
                           double **T, double **Y, int64_t *N,
                           int *out_n_acc, int *out_n_rej, int *out_n_fev) {
    const int max_steps = 100000;
    int n_acc = 0, n_rej = 0, n_fev = 0;
    /* refine = 1 → only the step endpoint is emitted; refine = N → N-1
     * Hermite-interpolated interior samples plus the endpoint. MATLAB's
     * ode45 default is 4, ode23's default is 1.
     *
     * When n_targets > 2 we ignore Refine and emit at exactly the
     * supplied target times — matches MATLAB's behaviour for
     * `tspan = [t0 t1 t2 ... tN]`. The integrator still chooses its
     * own adaptive step; targets are filled in by Hermite from the
     * accepted-step bracket. */
    if (refine < 1) refine = 1;
    if (n_targets < 2) { *T = NULL; *Y = NULL; *N = 0; return; }
    double t0 = targets[0];
    double tf = targets[n_targets - 1];
    int user_grid = (n_targets > 2);

    int64_t cap = user_grid ? n_targets : 256;
    double *Tb = (double *)malloc((size_t)cap * sizeof(double));
    double *Yb = (double *)malloc((size_t)cap * sizeof(double));
    int64_t n = 0;

    double t = t0, y = y0;
    ode_push(&Tb, &Yb, &n, &cap, t, y);
    /* In user-grid mode the seed at targets[0] is already emitted; the
     * next target to fill is index 1. */
    int64_t next_tgt = 1;

    /* Initial step: small fraction of the interval, signed so backward
     * integration also terminates. Zero span just emits the seed point.
     * `init_step` overrides the heuristic when > 0; we still negate it
     * for backward integration so the user passes a magnitude. */
    double span = tf - t0;
    double h = (init_step > 0.0) ? (span >= 0.0 ? init_step : -init_step)
                                 : span * 0.01;
    if (h == 0.0 || span == 0.0) {
        *T = Tb; *Y = Yb; *N = n;
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    int forward = h > 0;
    /* Cap the initial step at MaxStep too — otherwise a user-provided
     * InitialStep > MaxStep would be silently honoured for one step. */
    if (max_step > 0.0) {
        if (h >  max_step) h =  max_step;
        if (h < -max_step) h = -max_step;
    }

    double k1 = f(t, y);
    ++n_fev;
    int steps = 0;
    while ((forward ? t < tf : t > tf) && steps < max_steps) {
        ++steps;
        if (forward ? (t + h > tf) : (t + h < tf)) h = tf - t;

        double k2 = f(t + h * (1.0/5.0),
                      y + h * (k1 * (1.0/5.0)));
        double k3 = f(t + h * (3.0/10.0),
                      y + h * (k1 * (3.0/40.0) + k2 * (9.0/40.0)));
        double k4 = f(t + h * (4.0/5.0),
                      y + h * (k1 * (44.0/45.0) - k2 * (56.0/15.0)
                              + k3 * (32.0/9.0)));
        double k5 = f(t + h * (8.0/9.0),
                      y + h * (k1 * (19372.0/6561.0)
                              - k2 * (25360.0/2187.0)
                              + k3 * (64448.0/6561.0)
                              - k4 * (212.0/729.0)));
        double k6 = f(t + h,
                      y + h * (k1 * (9017.0/3168.0)
                              - k2 * (355.0/33.0)
                              + k3 * (46732.0/5247.0)
                              + k4 * (49.0/176.0)
                              - k5 * (5103.0/18656.0)));
        double y5 = y + h * (k1 * (35.0/384.0)
                            + k3 * (500.0/1113.0)
                            + k4 * (125.0/192.0)
                            - k5 * (2187.0/6784.0)
                            + k6 * (11.0/84.0));
        double k7 = f(t + h, y5);
        n_fev += 6;       /* k2..k7 */

        /* Embedded 4th-order error estimate: e = h * sum((b5 - b4) * k_i). */
        double err = h * (k1 * (71.0/57600.0)
                         - k3 * (71.0/16695.0)
                         + k4 * (71.0/1920.0)
                         - k5 * (17253.0/339200.0)
                         + k6 * (22.0/525.0)
                         - k7 * (1.0/40.0));
        double scale = atol + rtol * fmax(fabs(y), fabs(y5));
        double normerr = (scale > 0) ? fabs(err) / scale : 0.0;

        if (normerr <= 1.0) {
            ++n_acc;
            if (user_grid) {
                /* Emit at every target time that fell inside this step.
                 * The bracket is [t, t+h] for forward, [t+h, t] for
                 * backward; the targets array is monotonic in the
                 * integration direction. The final target is the step
                 * endpoint by construction (we clamped h above), so we
                 * use y5 directly to dodge round-off. */
                while (next_tgt < n_targets) {
                    double tt = targets[next_tgt];
                    int in_range = forward ? (tt <= t + h) : (tt >= t + h);
                    if (!in_range) break;
                    double th = (h == 0.0) ? 0.0 : (tt - t) / h;
                    double yi = (next_tgt == n_targets - 1)
                        ? y5
                        : ode_hermite(y, y5, k1, k7, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, tt, yi);
                    ++next_tgt;
                }
            } else {
                /* Emit refine-1 interior samples then the step endpoint
                 * via cubic Hermite. y/k1 are pre-step values; (y5, k7)
                 * are the step-end values used as the right-hand Hermite
                 * anchor. */
                for (int j = 1; j <= refine; ++j) {
                    double th = (double)j / (double)refine;
                    double ti = t + h * th;
                    double yi = (j == refine)
                        ? y5
                        : ode_hermite(y, y5, k1, k7, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, ti, yi);
                }
            }
            t += h;
            y  = y5;
            k1 = k7;                                     /* FSAL */
            if (user_grid && next_tgt >= n_targets) break;
        } else {
            ++n_rej;
        }

        double fac = (normerr == 0.0) ? 5.0
                                      : 0.9 * pow(normerr, -1.0/5.0);
        if (fac < 0.2) fac = 0.2;
        if (fac > 5.0) fac = 5.0;
        h *= fac;
        if (max_step > 0.0) {
            if (h >  max_step) h =  max_step;
            if (h < -max_step) h = -max_step;
        }
    }

    *T = Tb; *Y = Yb; *N = n;
    if (out_n_acc) *out_n_acc = n_acc;
    if (out_n_rej) *out_n_rej = n_rej;
    if (out_n_fev) *out_n_fev = n_fev;
}

static void rk_solve_bs23(ode_rhs_t f,
                           const double *targets, int64_t n_targets,
                           double y0,
                           double rtol, double atol,
                           double max_step, double init_step, int refine,
                           double **T, double **Y, int64_t *N,
                           int *out_n_acc, int *out_n_rej, int *out_n_fev) {
    const int max_steps = 100000;
    int n_acc = 0, n_rej = 0, n_fev = 0;
    if (refine < 1) refine = 1;
    if (n_targets < 2) { *T = NULL; *Y = NULL; *N = 0; return; }
    double t0 = targets[0];
    double tf = targets[n_targets - 1];
    int user_grid = (n_targets > 2);

    int64_t cap = user_grid ? n_targets : 256;
    double *Tb = (double *)malloc((size_t)cap * sizeof(double));
    double *Yb = (double *)malloc((size_t)cap * sizeof(double));
    int64_t n = 0;

    double t = t0, y = y0;
    ode_push(&Tb, &Yb, &n, &cap, t, y);
    int64_t next_tgt = 1;

    double span = tf - t0;
    double h = (init_step > 0.0) ? (span >= 0.0 ? init_step : -init_step)
                                 : span * 0.01;
    if (h == 0.0 || span == 0.0) {
        *T = Tb; *Y = Yb; *N = n;
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    int forward = h > 0;
    if (max_step > 0.0) {
        if (h >  max_step) h =  max_step;
        if (h < -max_step) h = -max_step;
    }

    double k1 = f(t, y);
    ++n_fev;
    int steps = 0;
    while ((forward ? t < tf : t > tf) && steps < max_steps) {
        ++steps;
        if (forward ? (t + h > tf) : (t + h < tf)) h = tf - t;

        double k2 = f(t + h * 0.5,
                      y + h * (k1 * 0.5));
        double k3 = f(t + h * 0.75,
                      y + h * (k2 * 0.75));
        double y3 = y + h * (k1 * (2.0/9.0)
                           + k2 * (1.0/3.0)
                           + k3 * (4.0/9.0));
        double k4 = f(t + h, y3);
        n_fev += 3;       /* k2, k3, k4 */

        /* Error: 3rd-order minus 2nd-order combo. b3 - b2 is
         *   [-5/72, 1/12, 1/9, -1/8] over (k1, k2, k3, k4). */
        double err = h * (k1 * (-5.0/72.0)
                         + k2 * (1.0/12.0)
                         + k3 * (1.0/9.0)
                         - k4 * (1.0/8.0));
        double scale = atol + rtol * fmax(fabs(y), fabs(y3));
        double normerr = (scale > 0) ? fabs(err) / scale : 0.0;

        if (normerr <= 1.0) {
            ++n_acc;
            if (user_grid) {
                while (next_tgt < n_targets) {
                    double tt = targets[next_tgt];
                    int in_range = forward ? (tt <= t + h) : (tt >= t + h);
                    if (!in_range) break;
                    double th = (h == 0.0) ? 0.0 : (tt - t) / h;
                    double yi = (next_tgt == n_targets - 1)
                        ? y3
                        : ode_hermite(y, y3, k1, k4, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, tt, yi);
                    ++next_tgt;
                }
            } else {
                for (int j = 1; j <= refine; ++j) {
                    double th = (double)j / (double)refine;
                    double ti = t + h * th;
                    double yi = (j == refine)
                        ? y3
                        : ode_hermite(y, y3, k1, k4, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, ti, yi);
                }
            }
            t += h;
            y  = y3;
            k1 = k4;                                     /* FSAL */
            if (user_grid && next_tgt >= n_targets) break;
        } else {
            ++n_rej;
        }

        double fac = (normerr == 0.0) ? 5.0
                                      : 0.9 * pow(normerr, -1.0/3.0);
        if (fac < 0.2) fac = 0.2;
        if (fac > 5.0) fac = 5.0;
        h *= fac;
        if (max_step > 0.0) {
            if (h >  max_step) h =  max_step;
            if (h < -max_step) h = -max_step;
        }
    }

    *T = Tb; *Y = Yb; *N = n;
    if (out_n_acc) *out_n_acc = n_acc;
    if (out_n_rej) *out_n_rej = n_rej;
    if (out_n_fev) *out_n_fev = n_fev;
}

/* Cache-aware dispatch: integrate once per (kind, fp, tspan, y0, opts)
 * tuple and stash the result so the paired _t / _y call returns the
 * other half without re-running the solver. `print_stats` triggers a
 * MATLAB-style summary at the end of integration; only fires on the
 * actual-solve path so the paired _t/_y calls don't print twice. */
static void rosen_solve_23s(ode_rhs_t f,
                             const double *targets, int64_t n_targets,
                             double y0,
                             double rtol, double atol,
                             double max_step, double init_step, int refine,
                             double **T, double **Y, int64_t *N,
                             int *out_n_acc, int *out_n_rej, int *out_n_fev);

static void ode_compute(int kind, ode_rhs_t f, matlab_mat *tspan, double y0,
                        double rtol, double atol,
                        double max_step, double init_step, int refine,
                        int print_stats) {
    if (ode_cache_.valid &&
        ode_cache_.fp == (void *)f &&
        ode_cache_.tspan == tspan &&
        ode_cache_.y0 == y0 &&
        ode_cache_.rtol == rtol &&
        ode_cache_.atol == atol &&
        ode_cache_.max_step == max_step &&
        ode_cache_.init_step == init_step &&
        ode_cache_.refine == refine &&
        ode_cache_.print_stats == print_stats &&
        ode_cache_.kind == kind) {
        return;
    }
    /* New solve: drop any previously-cached matrices. The compiler
     * generally consumes both _t and _y of the prior call before issuing
     * a new pair, so freeing here is safe. */
    if (ode_cache_.valid) {
        if (ode_cache_.t) {
            free(ode_cache_.t->data); free(ode_cache_.t);
        }
        if (ode_cache_.y) {
            free(ode_cache_.y->data); free(ode_cache_.y);
        }
        ode_cache_.t = NULL;
        ode_cache_.y = NULL;
        ode_cache_.valid = 0;
    }

    int64_t n_tgt = tspan ? tspan->rows * tspan->cols : 0;
    if (!f || n_tgt < 2 || !tspan->data) {
        ode_cache_.t = mat_alloc(0, 1);
        ode_cache_.y = mat_alloc(0, 1);
        ode_cache_.n_acc = 0;
        ode_cache_.n_rej = 0;
        ode_cache_.n_fev = 0;
    } else {
        double *Tb = NULL, *Yb = NULL;
        int64_t n = 0;
        int n_acc = 0, n_rej = 0, n_fev = 0;
        if (kind == 45) rk_solve_dp45(f, tspan->data, n_tgt, y0,
                                       rtol, atol, max_step, init_step,
                                       refine, &Tb, &Yb, &n,
                                       &n_acc, &n_rej, &n_fev);
        else if (kind == 235)
                        rosen_solve_23s(f, tspan->data, n_tgt, y0,
                                        rtol, atol, max_step, init_step,
                                        refine, &Tb, &Yb, &n,
                                        &n_acc, &n_rej, &n_fev);
        else            rk_solve_bs23(f, tspan->data, n_tgt, y0,
                                       rtol, atol, max_step, init_step,
                                       refine, &Tb, &Yb, &n,
                                       &n_acc, &n_rej, &n_fev);
        ode_buffers_to_mats(Tb, Yb, n, &ode_cache_.t, &ode_cache_.y);
        free(Tb); free(Yb);
        ode_cache_.n_acc = n_acc;
        ode_cache_.n_rej = n_rej;
        ode_cache_.n_fev = n_fev;
        if (print_stats) {
            fprintf(stdout, "%d successful steps\n", n_acc);
            fprintf(stdout, "%d failed attempts\n",  n_rej);
            fprintf(stdout, "%d function evaluations\n", n_fev);
            fflush(stdout);
        }
    }
    ode_cache_.fp        = (void *)f;
    ode_cache_.tspan     = tspan;
    ode_cache_.y0        = y0;
    ode_cache_.rtol      = rtol;
    ode_cache_.atol      = atol;
    ode_cache_.max_step  = max_step;
    ode_cache_.init_step = init_step;
    ode_cache_.refine     = refine;
    ode_cache_.print_stats = print_stats;
    ode_cache_.kind       = kind;
    ode_cache_.valid      = 1;
}

/* Pull RelTol / AbsTol / MaxStep / InitialStep / Refine / Stats from an
 * options struct, with MATLAB defaults (1e-3 / 1e-6) for the tolerances,
 * 0 (= use built-in heuristics) for the step bounds, and a kind-specific
 * Refine default supplied by the caller (4 for ode45, 1 for ode23).
 * Stats is treated as a numeric flag — non-zero turns on the
 * end-of-integration summary. (MATLAB accepts the string 'on' here; the
 * frontend's struct-set lowering doesn't yet wire string values into
 * matlab_struct_set_f64 cleanly, so we accept opts.Stats = 1 instead.) */
static void ode_opts_resolve(matlab_struct *opts,
                              double *rtol, double *atol,
                              double *max_step, double *init_step,
                              int *refine, int default_refine,
                              int *print_stats) {
    *rtol = 1e-3; *atol = 1e-6;
    *max_step = 0.0; *init_step = 0.0;
    *refine = default_refine;
    *print_stats = 0;
    if (!opts) return;
    if (matlab_struct_has_field(opts, "RelTol", 6) != 0.0)
        *rtol = matlab_struct_get_f64(opts, "RelTol", 6);
    if (matlab_struct_has_field(opts, "AbsTol", 6) != 0.0)
        *atol = matlab_struct_get_f64(opts, "AbsTol", 6);
    if (matlab_struct_has_field(opts, "MaxStep", 7) != 0.0)
        *max_step = matlab_struct_get_f64(opts, "MaxStep", 7);
    if (matlab_struct_has_field(opts, "InitialStep", 11) != 0.0)
        *init_step = matlab_struct_get_f64(opts, "InitialStep", 11);
    if (matlab_struct_has_field(opts, "Refine", 6) != 0.0) {
        int r = (int)matlab_struct_get_f64(opts, "Refine", 6);
        if (r >= 1) *refine = r;
    }
    if (matlab_struct_has_field(opts, "Stats", 5) != 0.0) {
        double s = matlab_struct_get_f64(opts, "Stats", 5);
        *print_stats = (s != 0.0);
    }
}

/* Clone a column matrix so the caller owns the returned pointer
 * independently of the cache slot. */
static matlab_mat *mat_clone_col(matlab_mat *src) {
    if (!src) return mat_alloc(0, 1);
    matlab_mat *out = mat_alloc(src->rows, src->cols);
    int64_t n = src->rows * src->cols;
    if (n > 0) memcpy(out->data, src->data, (size_t)n * sizeof(double));
    return out;
}

matlab_mat *matlab_ode45_t(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(45, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 4, 0);
    return mat_clone_col(ode_cache_.t);
}

matlab_mat *matlab_ode45_y(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(45, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 4, 0);
    return mat_clone_col(ode_cache_.y);
}

matlab_mat *matlab_ode23_t(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(23, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_col(ode_cache_.t);
}

matlab_mat *matlab_ode23_y(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(23, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_col(ode_cache_.y);
}

/* 4-arg form: ode45(@f, tspan, y0, opts). `opts` is a struct with
 * optional RelTol / AbsTol / MaxStep / InitialStep fields. Other
 * MATLAB fields (Refine, OutputFcn, …) are silently ignored —
 * defaults remain MATLAB-compatible. */
matlab_mat *matlab_ode45_t_opts(ode_rhs_t f, matlab_mat *tspan, double y0,
                                 matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, /*default*/ 4, &ps);
    ode_compute(45, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_col(ode_cache_.t);
}

matlab_mat *matlab_ode45_y_opts(ode_rhs_t f, matlab_mat *tspan, double y0,
                                 matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 4, &ps);
    ode_compute(45, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_col(ode_cache_.y);
}

matlab_mat *matlab_ode23_t_opts(ode_rhs_t f, matlab_mat *tspan, double y0,
                                 matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, /*default*/ 1, &ps);
    ode_compute(23, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_col(ode_cache_.t);
}

matlab_mat *matlab_ode23_y_opts(ode_rhs_t f, matlab_mat *tspan, double y0,
                                 matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 1, &ps);
    ode_compute(23, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_col(ode_cache_.y);
}

/* =====================================================================
 * Vector-y solvers.
 *
 * Same Dormand-Prince / Bogacki-Shampine adaptive integration as the
 * scalar path, but operating on D-component vectors. The user's RHS
 * has signature `matlab_mat *(*f)(double t, matlab_mat *y)` — accepts
 * a Dx1 column matrix, returns a fresh Dx1 column with dy/dt.
 *
 * Output `Y` is laid out row-major as N rows × D cols (the MATLAB
 * convention: Y(i, :) is the state at t(i)).
 * ===================================================================== */
typedef matlab_mat *(*ode_rhs_v_t)(double, matlab_mat *);

struct ode_v_cache_slot {
    void *fp;
    matlab_mat *tspan;
    matlab_mat *y0;
    double rtol;
    double atol;
    double max_step;
    double init_step;
    int refine;
    int print_stats;
    int kind;
    int64_t D;          /* state dimension */
    matlab_mat *t;      /* Nx1 time grid */
    matlab_mat *y;      /* NxD state matrix */
    int n_acc;
    int n_rej;
    int n_fev;
    int valid;
};

#if defined(__GNUC__) || defined(__clang__)
__thread struct ode_v_cache_slot ode_v_cache_;
#else
struct ode_v_cache_slot ode_v_cache_;
#endif

static void mat_free_(matlab_mat *m) {
    if (!m) return;
    free(m->data);
    free(m);
}

/* Call f with a fresh-looking (Dx1 column) matlab_mat copying y, get a
 * matlab_mat result, copy its first D entries into out, and free the
 * result (it was allocated fresh by the user code). The yt scratch is
 * reused across stages to avoid per-call descriptor allocation. */
static void ode_v_call(ode_rhs_v_t f, double t, const double *y, int64_t D,
                        matlab_mat *yt, double *out) {
    memcpy(yt->data, y, (size_t)D * sizeof(double));
    matlab_mat *dy = f(t, yt);
    if (dy && dy->data) {
        int64_t nd = dy->rows * dy->cols;
        if (nd > D) nd = D;
        memcpy(out, dy->data, (size_t)nd * sizeof(double));
        if (nd < D) memset(out + nd, 0, (size_t)(D - nd) * sizeof(double));
    } else {
        memset(out, 0, (size_t)D * sizeof(double));
    }
    mat_free_(dy);
}

/* Cubic Hermite per-component. y0/y1 are vectors at the bracket
 * endpoints; k0/k1 are the corresponding derivatives. */
static void ode_v_hermite(const double *y0, const double *y1,
                           const double *k0, const double *k1,
                           double h, double th, int64_t D, double *out) {
    double th2 = th * th;
    double th3 = th2 * th;
    double a = 2.0*th3 - 3.0*th2 + 1.0;
    double b = -2.0*th3 + 3.0*th2;
    double c = h * (th3 - 2.0*th2 + th);
    double d = h * (th3 - th2);
    for (int64_t j = 0; j < D; ++j)
        out[j] = a*y0[j] + b*y1[j] + c*k0[j] + d*k1[j];
}

/* Push a (t, y[0..D]) row to the growing Tb / Yb buffers. Yb is
 * row-major NxD. */
static void ode_v_push(double **Tb, double **Yb, int64_t *n, int64_t *cap,
                        int64_t D, double tv, const double *yv) {
    if (*n == *cap) {
        *cap = (*cap) * 2;
        *Tb = (double *)realloc(*Tb, (size_t)(*cap) * sizeof(double));
        *Yb = (double *)realloc(*Yb, (size_t)(*cap) * (size_t)D * sizeof(double));
    }
    (*Tb)[*n] = tv;
    memcpy(*Yb + (*n) * D, yv, (size_t)D * sizeof(double));
    ++(*n);
}

/* Vector Dormand-Prince 5(4). Mirror of rk_solve_dp45 with vector
 * arithmetic. The scratch buffers k1..k7, y_new, err, and y are all
 * length D. The yt mat is a single Dx1 descriptor reused per f-call. */
static void rk_solve_dp45_v(ode_rhs_v_t f,
                              const double *targets, int64_t n_targets,
                              const double *y0, int64_t D,
                              double rtol, double atol,
                              double max_step, double init_step, int refine,
                              double **T, double **Y, int64_t *N,
                              int *out_n_acc, int *out_n_rej, int *out_n_fev) {
    const int max_steps = 100000;
    int n_acc = 0, n_rej = 0, n_fev = 0;
    if (refine < 1) refine = 1;
    if (n_targets < 2 || D <= 0) {
        *T = NULL; *Y = NULL; *N = 0;
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    double t0 = targets[0], tf = targets[n_targets - 1];
    int user_grid = (n_targets > 2);

    int64_t cap = user_grid ? n_targets : 256;
    double *Tb = (double *)malloc((size_t)cap * sizeof(double));
    double *Yb = (double *)malloc((size_t)cap * (size_t)D * sizeof(double));
    int64_t n = 0;

    double *y    = (double *)malloc((size_t)D * sizeof(double));
    double *y_new = (double *)malloc((size_t)D * sizeof(double));
    double *k1 = (double *)malloc((size_t)D * sizeof(double));
    double *k2 = (double *)malloc((size_t)D * sizeof(double));
    double *k3 = (double *)malloc((size_t)D * sizeof(double));
    double *k4 = (double *)malloc((size_t)D * sizeof(double));
    double *k5 = (double *)malloc((size_t)D * sizeof(double));
    double *k6 = (double *)malloc((size_t)D * sizeof(double));
    double *k7 = (double *)malloc((size_t)D * sizeof(double));
    double *stg = (double *)malloc((size_t)D * sizeof(double));
    double *err = (double *)malloc((size_t)D * sizeof(double));

    matlab_mat *yt = mat_alloc(D, 1);    /* reusable input scratch */

    memcpy(y, y0, (size_t)D * sizeof(double));
    double t = t0;
    ode_v_push(&Tb, &Yb, &n, &cap, D, t, y);
    int64_t next_tgt = 1;

    double span = tf - t0;
    double h = (init_step > 0.0) ? (span >= 0.0 ? init_step : -init_step)
                                 : span * 0.01;
    if (h == 0.0 || span == 0.0) {
        *T = Tb; *Y = Yb; *N = n;
        free(y); free(y_new); free(k1); free(k2); free(k3);
        free(k4); free(k5); free(k6); free(k7); free(stg); free(err);
        mat_free_(yt);
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    int forward = h > 0;
    if (max_step > 0.0) {
        if (h >  max_step) h =  max_step;
        if (h < -max_step) h = -max_step;
    }

    ode_v_call(f, t, y, D, yt, k1);
    ++n_fev;
    int steps = 0;
    while ((forward ? t < tf : t > tf) && steps < max_steps) {
        ++steps;
        if (forward ? (t + h > tf) : (t + h < tf)) h = tf - t;

        for (int64_t j = 0; j < D; ++j) stg[j] = y[j] + h * k1[j] * (1.0/5.0);
        ode_v_call(f, t + h * (1.0/5.0), stg, D, yt, k2);

        for (int64_t j = 0; j < D; ++j)
            stg[j] = y[j] + h * (k1[j] * (3.0/40.0) + k2[j] * (9.0/40.0));
        ode_v_call(f, t + h * (3.0/10.0), stg, D, yt, k3);

        for (int64_t j = 0; j < D; ++j)
            stg[j] = y[j] + h * (k1[j] * (44.0/45.0) - k2[j] * (56.0/15.0)
                                 + k3[j] * (32.0/9.0));
        ode_v_call(f, t + h * (4.0/5.0), stg, D, yt, k4);

        for (int64_t j = 0; j < D; ++j)
            stg[j] = y[j] + h * (k1[j] * (19372.0/6561.0)
                                 - k2[j] * (25360.0/2187.0)
                                 + k3[j] * (64448.0/6561.0)
                                 - k4[j] * (212.0/729.0));
        ode_v_call(f, t + h * (8.0/9.0), stg, D, yt, k5);

        for (int64_t j = 0; j < D; ++j)
            stg[j] = y[j] + h * (k1[j] * (9017.0/3168.0)
                                 - k2[j] * (355.0/33.0)
                                 + k3[j] * (46732.0/5247.0)
                                 + k4[j] * (49.0/176.0)
                                 - k5[j] * (5103.0/18656.0));
        ode_v_call(f, t + h, stg, D, yt, k6);

        for (int64_t j = 0; j < D; ++j)
            y_new[j] = y[j] + h * (k1[j] * (35.0/384.0)
                                  + k3[j] * (500.0/1113.0)
                                  + k4[j] * (125.0/192.0)
                                  - k5[j] * (2187.0/6784.0)
                                  + k6[j] * (11.0/84.0));
        ode_v_call(f, t + h, y_new, D, yt, k7);
        n_fev += 6;

        /* Per-component error & componentwise scale. Inf-norm. */
        double normerr = 0.0;
        for (int64_t j = 0; j < D; ++j) {
            err[j] = h * (k1[j] * (71.0/57600.0)
                         - k3[j] * (71.0/16695.0)
                         + k4[j] * (71.0/1920.0)
                         - k5[j] * (17253.0/339200.0)
                         + k6[j] * (22.0/525.0)
                         - k7[j] * (1.0/40.0));
            double scale = atol + rtol * fmax(fabs(y[j]), fabs(y_new[j]));
            double e = (scale > 0) ? fabs(err[j]) / scale : 0.0;
            if (e > normerr) normerr = e;
        }

        if (normerr <= 1.0) {
            ++n_acc;
            if (user_grid) {
                while (next_tgt < n_targets) {
                    double tt = targets[next_tgt];
                    int in_range = forward ? (tt <= t + h) : (tt >= t + h);
                    if (!in_range) break;
                    double th = (h == 0.0) ? 0.0 : (tt - t) / h;
                    if (next_tgt == n_targets - 1) {
                        ode_v_push(&Tb, &Yb, &n, &cap, D, tt, y_new);
                    } else {
                        double *interp = stg;  /* reuse stg as scratch */
                        ode_v_hermite(y, y_new, k1, k7, h, th, D, interp);
                        ode_v_push(&Tb, &Yb, &n, &cap, D, tt, interp);
                    }
                    ++next_tgt;
                }
            } else {
                for (int j = 1; j <= refine; ++j) {
                    double th = (double)j / (double)refine;
                    double ti = t + h * th;
                    if (j == refine) {
                        ode_v_push(&Tb, &Yb, &n, &cap, D, ti, y_new);
                    } else {
                        double *interp = stg;
                        ode_v_hermite(y, y_new, k1, k7, h, th, D, interp);
                        ode_v_push(&Tb, &Yb, &n, &cap, D, ti, interp);
                    }
                }
            }
            t += h;
            memcpy(y,  y_new, (size_t)D * sizeof(double));
            memcpy(k1, k7,    (size_t)D * sizeof(double));        /* FSAL */
            if (user_grid && next_tgt >= n_targets) break;
        } else {
            ++n_rej;
        }

        double fac = (normerr == 0.0) ? 5.0
                                      : 0.9 * pow(normerr, -1.0/5.0);
        if (fac < 0.2) fac = 0.2;
        if (fac > 5.0) fac = 5.0;
        h *= fac;
        if (max_step > 0.0) {
            if (h >  max_step) h =  max_step;
            if (h < -max_step) h = -max_step;
        }
    }

    *T = Tb; *Y = Yb; *N = n;
    if (out_n_acc) *out_n_acc = n_acc;
    if (out_n_rej) *out_n_rej = n_rej;
    if (out_n_fev) *out_n_fev = n_fev;
    free(y); free(y_new); free(k1); free(k2); free(k3);
    free(k4); free(k5); free(k6); free(k7); free(stg); free(err);
    mat_free_(yt);
}

/* Vector Bogacki-Shampine 3(2). Same shape as the scalar version but
 * working on D-component vectors. */
static void rk_solve_bs23_v(ode_rhs_v_t f,
                              const double *targets, int64_t n_targets,
                              const double *y0, int64_t D,
                              double rtol, double atol,
                              double max_step, double init_step, int refine,
                              double **T, double **Y, int64_t *N,
                              int *out_n_acc, int *out_n_rej, int *out_n_fev) {
    const int max_steps = 100000;
    int n_acc = 0, n_rej = 0, n_fev = 0;
    if (refine < 1) refine = 1;
    if (n_targets < 2 || D <= 0) {
        *T = NULL; *Y = NULL; *N = 0;
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    double t0 = targets[0], tf = targets[n_targets - 1];
    int user_grid = (n_targets > 2);

    int64_t cap = user_grid ? n_targets : 256;
    double *Tb = (double *)malloc((size_t)cap * sizeof(double));
    double *Yb = (double *)malloc((size_t)cap * (size_t)D * sizeof(double));
    int64_t n = 0;

    double *y    = (double *)malloc((size_t)D * sizeof(double));
    double *y_new = (double *)malloc((size_t)D * sizeof(double));
    double *k1 = (double *)malloc((size_t)D * sizeof(double));
    double *k2 = (double *)malloc((size_t)D * sizeof(double));
    double *k3 = (double *)malloc((size_t)D * sizeof(double));
    double *k4 = (double *)malloc((size_t)D * sizeof(double));
    double *stg = (double *)malloc((size_t)D * sizeof(double));
    double *err = (double *)malloc((size_t)D * sizeof(double));
    matlab_mat *yt = mat_alloc(D, 1);

    memcpy(y, y0, (size_t)D * sizeof(double));
    double t = t0;
    ode_v_push(&Tb, &Yb, &n, &cap, D, t, y);
    int64_t next_tgt = 1;

    double span = tf - t0;
    double h = (init_step > 0.0) ? (span >= 0.0 ? init_step : -init_step)
                                 : span * 0.01;
    if (h == 0.0 || span == 0.0) {
        *T = Tb; *Y = Yb; *N = n;
        free(y); free(y_new); free(k1); free(k2); free(k3);
        free(k4); free(stg); free(err);
        mat_free_(yt);
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    int forward = h > 0;
    if (max_step > 0.0) {
        if (h >  max_step) h =  max_step;
        if (h < -max_step) h = -max_step;
    }

    ode_v_call(f, t, y, D, yt, k1);
    ++n_fev;
    int steps = 0;
    while ((forward ? t < tf : t > tf) && steps < max_steps) {
        ++steps;
        if (forward ? (t + h > tf) : (t + h < tf)) h = tf - t;

        for (int64_t j = 0; j < D; ++j) stg[j] = y[j] + h * k1[j] * 0.5;
        ode_v_call(f, t + h * 0.5, stg, D, yt, k2);

        for (int64_t j = 0; j < D; ++j) stg[j] = y[j] + h * k2[j] * 0.75;
        ode_v_call(f, t + h * 0.75, stg, D, yt, k3);

        for (int64_t j = 0; j < D; ++j)
            y_new[j] = y[j] + h * (k1[j] * (2.0/9.0)
                                  + k2[j] * (1.0/3.0)
                                  + k3[j] * (4.0/9.0));
        ode_v_call(f, t + h, y_new, D, yt, k4);
        n_fev += 3;

        double normerr = 0.0;
        for (int64_t j = 0; j < D; ++j) {
            err[j] = h * (k1[j] * (-5.0/72.0)
                         + k2[j] * (1.0/12.0)
                         + k3[j] * (1.0/9.0)
                         - k4[j] * (1.0/8.0));
            double scale = atol + rtol * fmax(fabs(y[j]), fabs(y_new[j]));
            double e = (scale > 0) ? fabs(err[j]) / scale : 0.0;
            if (e > normerr) normerr = e;
        }

        if (normerr <= 1.0) {
            ++n_acc;
            if (user_grid) {
                while (next_tgt < n_targets) {
                    double tt = targets[next_tgt];
                    int in_range = forward ? (tt <= t + h) : (tt >= t + h);
                    if (!in_range) break;
                    double th = (h == 0.0) ? 0.0 : (tt - t) / h;
                    if (next_tgt == n_targets - 1) {
                        ode_v_push(&Tb, &Yb, &n, &cap, D, tt, y_new);
                    } else {
                        double *interp = stg;
                        ode_v_hermite(y, y_new, k1, k4, h, th, D, interp);
                        ode_v_push(&Tb, &Yb, &n, &cap, D, tt, interp);
                    }
                    ++next_tgt;
                }
            } else {
                for (int j = 1; j <= refine; ++j) {
                    double th = (double)j / (double)refine;
                    double ti = t + h * th;
                    if (j == refine) {
                        ode_v_push(&Tb, &Yb, &n, &cap, D, ti, y_new);
                    } else {
                        double *interp = stg;
                        ode_v_hermite(y, y_new, k1, k4, h, th, D, interp);
                        ode_v_push(&Tb, &Yb, &n, &cap, D, ti, interp);
                    }
                }
            }
            t += h;
            memcpy(y,  y_new, (size_t)D * sizeof(double));
            memcpy(k1, k4,    (size_t)D * sizeof(double));
            if (user_grid && next_tgt >= n_targets) break;
        } else {
            ++n_rej;
        }

        double fac = (normerr == 0.0) ? 5.0
                                      : 0.9 * pow(normerr, -1.0/3.0);
        if (fac < 0.2) fac = 0.2;
        if (fac > 5.0) fac = 5.0;
        h *= fac;
        if (max_step > 0.0) {
            if (h >  max_step) h =  max_step;
            if (h < -max_step) h = -max_step;
        }
    }

    *T = Tb; *Y = Yb; *N = n;
    if (out_n_acc) *out_n_acc = n_acc;
    if (out_n_rej) *out_n_rej = n_rej;
    if (out_n_fev) *out_n_fev = n_fev;
    free(y); free(y_new); free(k1); free(k2); free(k3);
    free(k4); free(stg); free(err);
    mat_free_(yt);
}

static void rosen_solve_23s_v(ode_rhs_v_t f,
                               const double *targets, int64_t n_targets,
                               const double *y0, int64_t D,
                               double rtol, double atol,
                               double max_step, double init_step, int refine,
                               double **T, double **Y, int64_t *N,
                               int *out_n_acc, int *out_n_rej, int *out_n_fev);

/* Vector cache-aware dispatch. Mirrors ode_compute. The cache is a
 * separate slot from the scalar one (different ABI); both can be live
 * simultaneously since the user picks a path by y0 type. */
static void ode_v_compute(int kind, ode_rhs_v_t f, matlab_mat *tspan,
                           matlab_mat *y0,
                           double rtol, double atol,
                           double max_step, double init_step, int refine,
                           int print_stats) {
    if (ode_v_cache_.valid &&
        ode_v_cache_.fp == (void *)f &&
        ode_v_cache_.tspan == tspan &&
        ode_v_cache_.y0 == y0 &&
        ode_v_cache_.rtol == rtol &&
        ode_v_cache_.atol == atol &&
        ode_v_cache_.max_step == max_step &&
        ode_v_cache_.init_step == init_step &&
        ode_v_cache_.refine == refine &&
        ode_v_cache_.print_stats == print_stats &&
        ode_v_cache_.kind == kind) {
        return;
    }
    if (ode_v_cache_.valid) {
        if (ode_v_cache_.t) { free(ode_v_cache_.t->data); free(ode_v_cache_.t); }
        if (ode_v_cache_.y) { free(ode_v_cache_.y->data); free(ode_v_cache_.y); }
        ode_v_cache_.t = NULL;
        ode_v_cache_.y = NULL;
        ode_v_cache_.valid = 0;
    }

    int64_t n_tgt = tspan ? tspan->rows * tspan->cols : 0;
    int64_t D = y0 ? y0->rows * y0->cols : 0;
    if (!f || n_tgt < 2 || D <= 0) {
        ode_v_cache_.t = mat_alloc(0, 1);
        ode_v_cache_.y = mat_alloc(0, D > 0 ? D : 1);
        ode_v_cache_.D = D;
        ode_v_cache_.n_acc = 0;
        ode_v_cache_.n_rej = 0;
        ode_v_cache_.n_fev = 0;
    } else {
        double *Tb = NULL, *Yb = NULL;
        int64_t n = 0;
        int n_acc = 0, n_rej = 0, n_fev = 0;
        if (kind == 45) rk_solve_dp45_v(f, tspan->data, n_tgt,
                                          y0->data, D,
                                          rtol, atol, max_step, init_step,
                                          refine, &Tb, &Yb, &n,
                                          &n_acc, &n_rej, &n_fev);
        else if (kind == 235)
                        rosen_solve_23s_v(f, tspan->data, n_tgt,
                                          y0->data, D,
                                          rtol, atol, max_step, init_step,
                                          refine, &Tb, &Yb, &n,
                                          &n_acc, &n_rej, &n_fev);
        else            rk_solve_bs23_v(f, tspan->data, n_tgt,
                                          y0->data, D,
                                          rtol, atol, max_step, init_step,
                                          refine, &Tb, &Yb, &n,
                                          &n_acc, &n_rej, &n_fev);
        /* Wrap into matlab_mat descriptors. */
        matlab_mat *Tm = mat_alloc(n, 1);
        matlab_mat *Ym = mat_alloc(n, D);
        if (n > 0) {
            memcpy(Tm->data, Tb, (size_t)n * sizeof(double));
            memcpy(Ym->data, Yb, (size_t)n * (size_t)D * sizeof(double));
        }
        ode_v_cache_.t = Tm;
        ode_v_cache_.y = Ym;
        ode_v_cache_.D = D;
        ode_v_cache_.n_acc = n_acc;
        ode_v_cache_.n_rej = n_rej;
        ode_v_cache_.n_fev = n_fev;
        free(Tb); free(Yb);
        if (print_stats) {
            fprintf(stdout, "%d successful steps\n", n_acc);
            fprintf(stdout, "%d failed attempts\n",  n_rej);
            fprintf(stdout, "%d function evaluations\n", n_fev);
            fflush(stdout);
        }
    }
    ode_v_cache_.fp         = (void *)f;
    ode_v_cache_.tspan      = tspan;
    ode_v_cache_.y0         = y0;
    ode_v_cache_.rtol       = rtol;
    ode_v_cache_.atol       = atol;
    ode_v_cache_.max_step   = max_step;
    ode_v_cache_.init_step  = init_step;
    ode_v_cache_.refine     = refine;
    ode_v_cache_.print_stats = print_stats;
    ode_v_cache_.kind       = kind;
    ode_v_cache_.valid      = 1;
}

static matlab_mat *mat_clone_(matlab_mat *src) {
    if (!src) return mat_alloc(0, 1);
    matlab_mat *out = mat_alloc(src->rows, src->cols);
    int64_t n = src->rows * src->cols;
    if (n > 0) memcpy(out->data, src->data, (size_t)n * sizeof(double));
    return out;
}

matlab_mat *matlab_ode45_v_t(ode_rhs_v_t f, matlab_mat *tspan, matlab_mat *y0) {
    ode_v_compute(45, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 4, 0);
    return mat_clone_(ode_v_cache_.t);
}
matlab_mat *matlab_ode45_v_y(ode_rhs_v_t f, matlab_mat *tspan, matlab_mat *y0) {
    ode_v_compute(45, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 4, 0);
    return mat_clone_(ode_v_cache_.y);
}
matlab_mat *matlab_ode23_v_t(ode_rhs_v_t f, matlab_mat *tspan, matlab_mat *y0) {
    ode_v_compute(23, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_(ode_v_cache_.t);
}
matlab_mat *matlab_ode23_v_y(ode_rhs_v_t f, matlab_mat *tspan, matlab_mat *y0) {
    ode_v_compute(23, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_(ode_v_cache_.y);
}

matlab_mat *matlab_ode45_v_t_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                    matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 4, &ps);
    ode_v_compute(45, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_(ode_v_cache_.t);
}
matlab_mat *matlab_ode45_v_y_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                    matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 4, &ps);
    ode_v_compute(45, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_(ode_v_cache_.y);
}
matlab_mat *matlab_ode23_v_t_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                    matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 1, &ps);
    ode_v_compute(23, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_(ode_v_cache_.t);
}
matlab_mat *matlab_ode23_v_y_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                    matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 1, &ps);
    ode_v_compute(23, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_(ode_v_cache_.y);
}

static matlab_struct *ode_v_stats_struct_from_cache(void) {
    matlab_struct *s = matlab_struct_new();
    matlab_struct_set_f64(s, "nsteps",  6, (double)ode_v_cache_.n_acc);
    matlab_struct_set_f64(s, "nfailed", 7, (double)ode_v_cache_.n_rej);
    matlab_struct_set_f64(s, "nfevals", 7, (double)ode_v_cache_.n_fev);
    return s;
}

matlab_struct *matlab_ode45_v_stats(ode_rhs_v_t f, matlab_mat *tspan,
                                      matlab_mat *y0) {
    ode_v_compute(45, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 4, 0);
    return ode_v_stats_struct_from_cache();
}
matlab_struct *matlab_ode45_v_stats_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                           matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 4, &ps);
    ode_v_compute(45, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return ode_v_stats_struct_from_cache();
}
matlab_struct *matlab_ode23_v_stats(ode_rhs_v_t f, matlab_mat *tspan,
                                      matlab_mat *y0) {
    ode_v_compute(23, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return ode_v_stats_struct_from_cache();
}
matlab_struct *matlab_ode23_v_stats_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                           matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 1, &ps);
    ode_v_compute(23, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return ode_v_stats_struct_from_cache();
}

/* Vector ode23s public entries. */
matlab_mat *matlab_ode23s_v_t(ode_rhs_v_t f, matlab_mat *tspan, matlab_mat *y0) {
    ode_v_compute(235, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_(ode_v_cache_.t);
}
matlab_mat *matlab_ode23s_v_y(ode_rhs_v_t f, matlab_mat *tspan, matlab_mat *y0) {
    ode_v_compute(235, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_(ode_v_cache_.y);
}
matlab_mat *matlab_ode23s_v_t_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                     matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 1, &ps);
    ode_v_compute(235, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_(ode_v_cache_.t);
}
matlab_mat *matlab_ode23s_v_y_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                     matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 1, &ps);
    ode_v_compute(235, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_(ode_v_cache_.y);
}
matlab_struct *matlab_ode23s_v_stats(ode_rhs_v_t f, matlab_mat *tspan,
                                      matlab_mat *y0) {
    ode_v_compute(235, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return ode_v_stats_struct_from_cache();
}
matlab_struct *matlab_ode23s_v_stats_opts(ode_rhs_v_t f, matlab_mat *tspan,
                                            matlab_mat *y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, 1, &ps);
    ode_v_compute(235, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return ode_v_stats_struct_from_cache();
}

/* Build a fresh stats struct from the cache slot. Field names match
 * MATLAB's `[t,y,sol] = ode45(...)` solver-stats fields. */
static matlab_struct *ode_stats_struct_from_cache(void) {
    matlab_struct *s = matlab_struct_new();
    matlab_struct_set_f64(s, "nsteps",  6, (double)ode_cache_.n_acc);
    matlab_struct_set_f64(s, "nfailed", 7, (double)ode_cache_.n_rej);
    matlab_struct_set_f64(s, "nfevals", 7, (double)ode_cache_.n_fev);
    return s;
}

/* 3-return form: `[t, y, stats] = ode45(@f, tspan, y0[, opts])`. The
 * lowering splits the site into matlab_ode45_t / _y / _stats. The
 * cache memoises (n_acc, n_rej, n_fev) on solve so the third call
 * just packages them into a struct. */
matlab_struct *matlab_ode45_stats(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(45, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 4, 0);
    return ode_stats_struct_from_cache();
}

matlab_struct *matlab_ode45_stats_opts(ode_rhs_t f, matlab_mat *tspan,
                                        double y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, /*default*/ 4, &ps);
    ode_compute(45, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return ode_stats_struct_from_cache();
}

matlab_struct *matlab_ode23_stats(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(23, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return ode_stats_struct_from_cache();
}

matlab_struct *matlab_ode23_stats_opts(ode_rhs_t f, matlab_mat *tspan,
                                        double y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, /*default*/ 1, &ps);
    ode_compute(23, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return ode_stats_struct_from_cache();
}

/* ode23s public entries (scalar y). Refine default = 1 matches MATLAB;
 * users can request denser output via `opts.Refine = N`. */
matlab_mat *matlab_ode23s_t(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(235, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_col(ode_cache_.t);
}

matlab_mat *matlab_ode23s_y(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(235, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return mat_clone_col(ode_cache_.y);
}

matlab_mat *matlab_ode23s_t_opts(ode_rhs_t f, matlab_mat *tspan, double y0,
                                  matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, /*default*/ 1, &ps);
    ode_compute(235, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_col(ode_cache_.t);
}

matlab_mat *matlab_ode23s_y_opts(ode_rhs_t f, matlab_mat *tspan, double y0,
                                  matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, /*default*/ 1, &ps);
    ode_compute(235, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return mat_clone_col(ode_cache_.y);
}

matlab_struct *matlab_ode23s_stats(ode_rhs_t f, matlab_mat *tspan, double y0) {
    ode_compute(235, f, tspan, y0, 1e-3, 1e-6, 0.0, 0.0, 1, 0);
    return ode_stats_struct_from_cache();
}

matlab_struct *matlab_ode23s_stats_opts(ode_rhs_t f, matlab_mat *tspan,
                                         double y0, matlab_struct *opts) {
    double rtol, atol, mxs, ins; int rfn, ps;
    ode_opts_resolve(opts, &rtol, &atol, &mxs, &ins, &rfn, /*default*/ 1, &ps);
    ode_compute(235, f, tspan, y0, rtol, atol, mxs, ins, rfn, ps);
    return ode_stats_struct_from_cache();
}

/* =====================================================================
 * ode23s — Rosenbrock 2(3) stiff solver (Shampine).
 *
 * Uses one Jacobian per accepted step (numerical via central finite-
 * difference) and three "linear" stages — for scalar y the linear
 * system reduces to a division by W = 1 - h*d*J. Where ode45/ode23
 * blow up on stiff systems (eigenvalues of J with large negative real
 * parts force tiny explicit steps), the Rosenbrock W-method stays
 * stable because the implicit factor (I - h*d*J) absorbs the stiff
 * modes.
 *
 * MATLAB's ode23s uses these same coefficients; output should match
 * to within accept tolerance. Refine defaults to 1 (matches MATLAB).
 * ===================================================================== */
static void rosen_solve_23s(ode_rhs_t f,
                             const double *targets, int64_t n_targets,
                             double y0,
                             double rtol, double atol,
                             double max_step, double init_step, int refine,
                             double **T, double **Y, int64_t *N,
                             int *out_n_acc, int *out_n_rej, int *out_n_fev) {
    const int max_steps = 100000;
    int n_acc = 0, n_rej = 0, n_fev = 0;
    if (refine < 1) refine = 1;
    if (n_targets < 2) {
        *T = NULL; *Y = NULL; *N = 0;
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    double t0 = targets[0], tf = targets[n_targets - 1];
    int user_grid = (n_targets > 2);

    int64_t cap = user_grid ? n_targets : 256;
    double *Tb = (double *)malloc((size_t)cap * sizeof(double));
    double *Yb = (double *)malloc((size_t)cap * sizeof(double));
    int64_t n = 0;

    double t = t0, y = y0;
    ode_push(&Tb, &Yb, &n, &cap, t, y);
    int64_t next_tgt = 1;

    double span = tf - t0;
    double h = (init_step > 0.0) ? (span >= 0.0 ? init_step : -init_step)
                                 : span * 0.01;
    if (h == 0.0 || span == 0.0) {
        *T = Tb; *Y = Yb; *N = n;
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    int forward = h > 0;
    if (max_step > 0.0) {
        if (h >  max_step) h =  max_step;
        if (h < -max_step) h = -max_step;
    }

    /* Rosenbrock-Shampine 2(3) coefficients. d = 1/(2+√2), e32 = 6+√2. */
    const double SQRT2 = 1.41421356237309504880;
    const double d_   = 1.0 / (2.0 + SQRT2);
    const double e32  = 6.0 + SQRT2;
    /* sqrt(eps) for the FD Jacobian step size. */
    const double SQRT_EPS = 1.490116119384765625e-8;

    int steps = 0;
    while ((forward ? t < tf : t > tf) && steps < max_steps) {
        ++steps;
        if (forward ? (t + h > tf) : (t + h < tf)) h = tf - t;

        double F0 = f(t, y);                                /* +1 fev */
        /* Central-FD Jacobian: J = (f(t,y+δ) - f(t,y-δ)) / (2δ). */
        double eps_step = SQRT_EPS * fmax(fabs(y), 1.0);
        double Jp = f(t, y + eps_step);
        double Jm = f(t, y - eps_step);                     /* +2 fevs */
        n_fev += 3;
        double J = (Jp - Jm) / (2.0 * eps_step);

        double W = 1.0 - h * d_ * J;
        if (W == 0.0) W = 1e-30;

        /* Stage 1: W*k1 = F0  → scalar division. */
        double k1 = F0 / W;

        /* Stage 2: F1 = f(t+h/2, y+h*k1/2);  W*k2 = F1 - k1; k2 += k1. */
        double F1 = f(t + 0.5*h, y + 0.5*h*k1);
        ++n_fev;
        double k2 = (F1 - k1) / W + k1;

        /* Provisional solution (2nd order). */
        double y_new = y + h * k2;

        /* Stage 3 (for the embedded error estimate). */
        double F2 = f(t + h, y_new);
        ++n_fev;
        double k3 = (F2 - e32 * (k2 - F1) - 2.0 * (k1 - F0)) / W;

        double err   = (h / 6.0) * (k1 - 2.0 * k2 + k3);
        double scale = atol + rtol * fmax(fabs(y), fabs(y_new));
        double normerr = (scale > 0) ? fabs(err) / scale : 0.0;

        if (normerr <= 1.0) {
            ++n_acc;
            /* Cubic-Hermite dense output using F0 (slope at t) and F2
             * (slope at t+h). Same Hermite formula as the explicit
             * solvers — works for any RK-style method as long as we
             * have endpoint slopes. */
            if (user_grid) {
                while (next_tgt < n_targets) {
                    double tt = targets[next_tgt];
                    int in_range = forward ? (tt <= t + h) : (tt >= t + h);
                    if (!in_range) break;
                    double th = (h == 0.0) ? 0.0 : (tt - t) / h;
                    double yi = (next_tgt == n_targets - 1)
                        ? y_new
                        : ode_hermite(y, y_new, F0, F2, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, tt, yi);
                    ++next_tgt;
                }
            } else {
                for (int j = 1; j <= refine; ++j) {
                    double th = (double)j / (double)refine;
                    double ti = t + h * th;
                    double yi = (j == refine)
                        ? y_new
                        : ode_hermite(y, y_new, F0, F2, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, ti, yi);
                }
            }
            t += h;
            y  = y_new;
            if (user_grid && next_tgt >= n_targets) break;
        } else {
            ++n_rej;
        }

        /* Order p = 2 for Rosenbrock23 → exponent -1/(p+1) = -1/3. */
        double fac = (normerr == 0.0) ? 5.0
                                      : 0.9 * pow(normerr, -1.0/3.0);
        if (fac < 0.2) fac = 0.2;
        if (fac > 5.0) fac = 5.0;
        h *= fac;
        if (max_step > 0.0) {
            if (h >  max_step) h =  max_step;
            if (h < -max_step) h = -max_step;
        }
    }

    *T = Tb; *Y = Yb; *N = n;
    if (out_n_acc) *out_n_acc = n_acc;
    if (out_n_rej) *out_n_rej = n_rej;
    if (out_n_fev) *out_n_fev = n_fev;
}

/* ---- Vector ode23s: Rosenbrock 2(3) for systems ----------------------
 * Same Shampine pair; instead of dividing by W = 1 - h*d*J we factor
 * the DxD matrix W = I - h*d*J once per step (LU + partial pivot) and
 * back-solve three times — one per Rosenbrock stage. The Jacobian is
 * built column-by-column via central FD: 2D extra f-evals per accepted
 * step on top of the three stage evals. This is the same cost MATLAB's
 * ode23s pays. */

/* Doolittle LU with partial pivoting in-place on a DxD row-major buffer.
 * perm[i] is the row that ends up at position i. Returns 0 on success,
 * 1 on singular. */
static int lu_factor_pp(double *A, int *perm, int D) {
    for (int i = 0; i < D; ++i) perm[i] = i;
    for (int k = 0; k < D; ++k) {
        /* Pivot: find row r >= k with largest |A[r][k]|. */
        int piv = k;
        double maxv = fabs(A[k * D + k]);
        for (int r = k + 1; r < D; ++r) {
            double v = fabs(A[r * D + k]);
            if (v > maxv) { maxv = v; piv = r; }
        }
        if (maxv < 1e-300) return 1;       /* effectively singular */
        if (piv != k) {
            for (int c = 0; c < D; ++c) {
                double t = A[k * D + c]; A[k * D + c] = A[piv * D + c]; A[piv * D + c] = t;
            }
            int tp = perm[k]; perm[k] = perm[piv]; perm[piv] = tp;
        }
        double diag = A[k * D + k];
        for (int r = k + 1; r < D; ++r) {
            double m = A[r * D + k] / diag;
            A[r * D + k] = m;
            for (int c = k + 1; c < D; ++c)
                A[r * D + c] -= m * A[k * D + c];
        }
    }
    return 0;
}

/* Solve LU * x = b given the factorization in A and the row permutation.
 * b is permuted in place; x receives the result. */
static void lu_solve(const double *A, const int *perm, const double *b,
                      double *x, int D) {
    /* Apply permutation: y_perm[i] = b[perm[i]]. */
    for (int i = 0; i < D; ++i) x[i] = b[perm[i]];
    /* Forward substitute L (unit diagonal). */
    for (int i = 1; i < D; ++i) {
        double s = x[i];
        for (int j = 0; j < i; ++j) s -= A[i * D + j] * x[j];
        x[i] = s;
    }
    /* Back substitute U. */
    for (int i = D - 1; i >= 0; --i) {
        double s = x[i];
        for (int j = i + 1; j < D; ++j) s -= A[i * D + j] * x[j];
        x[i] = s / A[i * D + i];
    }
}

static void rosen_solve_23s_v(ode_rhs_v_t f,
                               const double *targets, int64_t n_targets,
                               const double *y0, int64_t D,
                               double rtol, double atol,
                               double max_step, double init_step, int refine,
                               double **T, double **Y, int64_t *N,
                               int *out_n_acc, int *out_n_rej, int *out_n_fev) {
    const int max_steps = 100000;
    int n_acc = 0, n_rej = 0, n_fev = 0;
    if (refine < 1) refine = 1;
    if (n_targets < 2 || D <= 0) {
        *T = NULL; *Y = NULL; *N = 0;
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    double t0 = targets[0], tf = targets[n_targets - 1];
    int user_grid = (n_targets > 2);

    int64_t cap = user_grid ? n_targets : 256;
    double *Tb = (double *)malloc((size_t)cap * sizeof(double));
    double *Yb = (double *)malloc((size_t)cap * (size_t)D * sizeof(double));
    int64_t n = 0;

    /* Working buffers. */
    double *y    = (double *)malloc((size_t)D * sizeof(double));
    double *y_new = (double *)malloc((size_t)D * sizeof(double));
    double *F0 = (double *)malloc((size_t)D * sizeof(double));
    double *F1 = (double *)malloc((size_t)D * sizeof(double));
    double *F2 = (double *)malloc((size_t)D * sizeof(double));
    double *Fp = (double *)malloc((size_t)D * sizeof(double));
    double *Fm = (double *)malloc((size_t)D * sizeof(double));
    double *k1 = (double *)malloc((size_t)D * sizeof(double));
    double *k2 = (double *)malloc((size_t)D * sizeof(double));
    double *k3 = (double *)malloc((size_t)D * sizeof(double));
    double *stg = (double *)malloc((size_t)D * sizeof(double));
    double *rhs = (double *)malloc((size_t)D * sizeof(double));
    double *err = (double *)malloc((size_t)D * sizeof(double));
    double *W   = (double *)malloc((size_t)D * (size_t)D * sizeof(double));
    int    *perm = (int *)   malloc((size_t)D * sizeof(int));
    matlab_mat *yt = mat_alloc(D, 1);

    memcpy(y, y0, (size_t)D * sizeof(double));
    double t = t0;
    ode_v_push(&Tb, &Yb, &n, &cap, D, t, y);
    int64_t next_tgt = 1;

    double span = tf - t0;
    double h = (init_step > 0.0) ? (span >= 0.0 ? init_step : -init_step)
                                 : span * 0.01;
    if (h == 0.0 || span == 0.0) {
        *T = Tb; *Y = Yb; *N = n;
        free(y); free(y_new); free(F0); free(F1); free(F2);
        free(Fp); free(Fm); free(k1); free(k2); free(k3);
        free(stg); free(rhs); free(err); free(W); free(perm);
        mat_free_(yt);
        if (out_n_acc) *out_n_acc = 0;
        if (out_n_rej) *out_n_rej = 0;
        if (out_n_fev) *out_n_fev = 0;
        return;
    }
    int forward = h > 0;
    if (max_step > 0.0) {
        if (h >  max_step) h =  max_step;
        if (h < -max_step) h = -max_step;
    }

    const double SQRT2 = 1.41421356237309504880;
    const double d_   = 1.0 / (2.0 + SQRT2);
    const double e32  = 6.0 + SQRT2;
    const double SQRT_EPS = 1.490116119384765625e-8;

    int steps = 0;
    while ((forward ? t < tf : t > tf) && steps < max_steps) {
        ++steps;
        if (forward ? (t + h > tf) : (t + h < tf)) h = tf - t;

        /* F0 = f(t, y). */
        ode_v_call(f, t, y, D, yt, F0);
        ++n_fev;

        /* Build Jacobian column-by-column via central FD. */
        for (int64_t j = 0; j < D; ++j) {
            double yj = y[j];
            double dj = SQRT_EPS * fmax(fabs(yj), 1.0);
            memcpy(stg, y, (size_t)D * sizeof(double));
            stg[j] = yj + dj;
            ode_v_call(f, t, stg, D, yt, Fp);
            stg[j] = yj - dj;
            ode_v_call(f, t, stg, D, yt, Fm);
            n_fev += 2;
            double inv2dj = 1.0 / (2.0 * dj);
            for (int64_t i = 0; i < D; ++i)
                W[i * D + j] = -h * d_ * (Fp[i] - Fm[i]) * inv2dj;
        }
        for (int64_t i = 0; i < D; ++i) W[i * D + i] += 1.0;

        /* Factor W = I - h*d*J once. */
        if (lu_factor_pp(W, perm, (int)D) != 0) {
            /* Singular: shrink step and reject. */
            ++n_rej;
            h *= 0.5;
            if (forward ? (h <= 0) : (h >= 0)) break;
            continue;
        }

        /* Stage 1: W * k1 = F0. */
        lu_solve(W, perm, F0, k1, (int)D);

        /* Stage 2: F1 = f(t + h/2, y + h*k1/2);  W * k2 = F1 - k1; k2 += k1. */
        for (int64_t i = 0; i < D; ++i) stg[i] = y[i] + 0.5 * h * k1[i];
        ode_v_call(f, t + 0.5 * h, stg, D, yt, F1);
        ++n_fev;
        for (int64_t i = 0; i < D; ++i) rhs[i] = F1[i] - k1[i];
        lu_solve(W, perm, rhs, k2, (int)D);
        for (int64_t i = 0; i < D; ++i) k2[i] += k1[i];

        /* Provisional solution. */
        for (int64_t i = 0; i < D; ++i) y_new[i] = y[i] + h * k2[i];

        /* Stage 3 (for error). */
        ode_v_call(f, t + h, y_new, D, yt, F2);
        ++n_fev;
        for (int64_t i = 0; i < D; ++i)
            rhs[i] = F2[i] - e32 * (k2[i] - F1[i]) - 2.0 * (k1[i] - F0[i]);
        lu_solve(W, perm, rhs, k3, (int)D);

        /* Error: (h/6) * (k1 - 2*k2 + k3). Inf-norm of err / scale. */
        double normerr = 0.0;
        for (int64_t i = 0; i < D; ++i) {
            err[i] = (h / 6.0) * (k1[i] - 2.0 * k2[i] + k3[i]);
            double scale = atol + rtol * fmax(fabs(y[i]), fabs(y_new[i]));
            double e = (scale > 0) ? fabs(err[i]) / scale : 0.0;
            if (e > normerr) normerr = e;
        }

        if (normerr <= 1.0) {
            ++n_acc;
            if (user_grid) {
                while (next_tgt < n_targets) {
                    double tt = targets[next_tgt];
                    int in_range = forward ? (tt <= t + h) : (tt >= t + h);
                    if (!in_range) break;
                    double th = (h == 0.0) ? 0.0 : (tt - t) / h;
                    if (next_tgt == n_targets - 1) {
                        ode_v_push(&Tb, &Yb, &n, &cap, D, tt, y_new);
                    } else {
                        ode_v_hermite(y, y_new, F0, F2, h, th, D, stg);
                        ode_v_push(&Tb, &Yb, &n, &cap, D, tt, stg);
                    }
                    ++next_tgt;
                }
            } else {
                for (int j = 1; j <= refine; ++j) {
                    double th = (double)j / (double)refine;
                    double ti = t + h * th;
                    if (j == refine) {
                        ode_v_push(&Tb, &Yb, &n, &cap, D, ti, y_new);
                    } else {
                        ode_v_hermite(y, y_new, F0, F2, h, th, D, stg);
                        ode_v_push(&Tb, &Yb, &n, &cap, D, ti, stg);
                    }
                }
            }
            t += h;
            memcpy(y, y_new, (size_t)D * sizeof(double));
            if (user_grid && next_tgt >= n_targets) break;
        } else {
            ++n_rej;
        }

        double fac = (normerr == 0.0) ? 5.0
                                      : 0.9 * pow(normerr, -1.0/3.0);
        if (fac < 0.2) fac = 0.2;
        if (fac > 5.0) fac = 5.0;
        h *= fac;
        if (max_step > 0.0) {
            if (h >  max_step) h =  max_step;
            if (h < -max_step) h = -max_step;
        }
    }

    *T = Tb; *Y = Yb; *N = n;
    if (out_n_acc) *out_n_acc = n_acc;
    if (out_n_rej) *out_n_rej = n_rej;
    if (out_n_fev) *out_n_fev = n_fev;
    free(y); free(y_new); free(F0); free(F1); free(F2);
    free(Fp); free(Fm); free(k1); free(k2); free(k3);
    free(stg); free(rhs); free(err); free(W); free(perm);
    mat_free_(yt);
}

/* =====================================================================
 * pdepe — 1-D parabolic-elliptic PDE solver via method-of-lines.
 *
 * MATLAB call shape:
 *   sol = pdepe(m, @pdefun, @icfun, @bcfun, xmesh, tspan)
 *
 * v1 scope: m = 0 (Cartesian), scalar PDE, Dirichlet boundary
 * conditions only (ql = qr = 0). Internally discretises space on the
 * supplied non-uniform xmesh via finite differences, eliminates the
 * boundary points from the state vector (Dirichlet values come from
 * solving the BC for u), and hands the interior ODE system to ode23s
 * for stiff time integration. The returned sol matrix is N_t × N_x —
 * sol(i, j) = u(t_i, x_j).
 *
 * Function-pointer ABIs follow the anon-function shapes:
 *   pdefn:  matlab_mat *(*)(double x, double t, double u, double dudx)
 *           returning [c; f; s] as a 3×1 column.
 *   icfn:   double (*)(double x)
 *   bcfn:   matlab_mat *(*)(double xl, double ul, double xr, double ur, double t)
 *           returning [pl; ql; pr; qr] as a 4×1 column. ql == qr == 0
 *           required (Dirichlet); other forms not yet supported.
 *
 * Stiffness handling — using ode23s under the hood means the 1-D heat
 * equation, advection-diffusion, and similar parabolic problems work
 * correctly even with fine spatial grids (where ode45 would collapse
 * to micro-steps).
 * ===================================================================== */

typedef matlab_mat *(*pdepe_pdefn_t)(double, double, double, double);
typedef double      (*pdepe_icfn_t) (double);
typedef matlab_mat *(*pdepe_bcfn_t) (double, double, double, double, double);

struct pdepe_ctx {
    pdepe_pdefn_t pdefn;
    pdepe_bcfn_t  bcfn;
    const double *xmesh;
    int64_t Nx;
    int m;              /* 0 = Cartesian, 1 = cylindrical, 2 = spherical */
    int err_flag;       /* 0 ok; 1 BC eval failed                       */
};

/* x^m for the supported symmetry settings. Avoids pow() overhead for the
 * three common values. */
static inline double pdepe_xpow(double x, int m) {
    if (m == 0) return 1.0;
    if (m == 1) return x;
    if (m == 2) return x * x;
    return pow(x, (double)m);
}
#if defined(__GNUC__) || defined(__clang__)
__thread struct pdepe_ctx pdepe_ctx_;
#else
struct pdepe_ctx pdepe_ctx_;
#endif

/* Evaluate the user's bcfun at the current boundary u-values and
 * return the four BC scalars. Returns 1 on shape failure. */
static int pdepe_eval_bc(double t, double ul, double ur,
                         double *pl, double *ql,
                         double *pr, double *qr) {
    double xl = pdepe_ctx_.xmesh[0];
    double xr = pdepe_ctx_.xmesh[pdepe_ctx_.Nx - 1];
    matlab_mat *r = pdepe_ctx_.bcfn(xl, ul, xr, ur, t);
    if (!r || r->rows * r->cols < 4) {
        if (r) mat_free_(r);
        return 1;
    }
    *pl = r->data[0];
    *ql = r->data[1];
    *pr = r->data[2];
    *qr = r->data[3];
    mat_free_(r);
    return 0;
}

/* Full-state RHS handed to ode23s_v. State vector U has dimension Nx
 * (every mesh point is part of the state). Each F call:
 *   - Evaluates bcfn at the current u[0], u[Nx-1].
 *   - For Dirichlet (ql == 0): snaps u to g(t) = u - pl (linear form),
 *     and forces F[boundary] = 0 so the integrator doesn't drift.
 *   - For Neumann/Robin (ql ≠ 0): computes f at the boundary as
 *     f = -pl/ql and uses it in the boundary-cell discretisation.
 *
 * The Cartesian (m = 0) form treats every cell with width dx_i; the
 * boundary cell width is dx_first / 2 (and dx_last / 2). For m ≠ 0 we
 * weight fluxes by x^m at midpoints and divide by x_i^m at nodes
 * (Skeel-Berzins integration). */
static matlab_mat *pdepe_rhs(double t, matlab_mat *Ufull) {
    int64_t Nx = pdepe_ctx_.Nx;
    if (!Ufull || Ufull->rows * Ufull->cols != Nx) return mat_alloc(Nx, 1);

    double *u = (double *)malloc((size_t)Nx * sizeof(double));
    memcpy(u, Ufull->data, (size_t)Nx * sizeof(double));

    /* Evaluate BC at current boundary values. */
    double pl, ql, pr, qr;
    if (pdepe_eval_bc(t, u[0], u[Nx - 1], &pl, &ql, &pr, &qr) != 0) {
        pdepe_ctx_.err_flag = 1;
        free(u);
        return mat_alloc(Nx, 1);
    }
    int dirichlet_left  = (ql == 0.0);
    int dirichlet_right = (qr == 0.0);

    /* Snap Dirichlet boundary values: for the standard linear form
     *   pl = ul - g(t),
     * we have g(t) = ul_current - pl_current. For nonlinear forms
     * this is a 1-step Newton-like correction. */
    if (dirichlet_left)  u[0]      = u[0]      - pl;
    if (dirichlet_right) u[Nx - 1] = u[Nx - 1] - pr;

    /* Boundary fluxes for Neumann/Robin: f_bdy = -pl/ql. */
    double f_left_bdy  = dirichlet_left  ? 0.0 : -pl / ql;
    double f_right_bdy = dirichlet_right ? 0.0 : -pr / qr;

    /* Compute interior fluxes f_{i+1/2} for i = 0 .. Nx-2. */
    double *flx = (double *)malloc((size_t)(Nx - 1) * sizeof(double));
    for (int64_t i = 0; i < Nx - 1; ++i) {
        double xL = pdepe_ctx_.xmesh[i];
        double xR = pdepe_ctx_.xmesh[i + 1];
        double dx = xR - xL;
        if (dx == 0.0) dx = 1e-30;
        double xm   = 0.5 * (xL + xR);
        double um   = 0.5 * (u[i] + u[i + 1]);
        double dudx = (u[i + 1] - u[i]) / dx;
        matlab_mat *r = pdepe_ctx_.pdefn(xm, t, um, dudx);
        flx[i] = (r && r->rows * r->cols >= 2) ? r->data[1] : 0.0;
        if (r) mat_free_(r);
    }

    matlab_mat *out = mat_alloc(Nx, 1);

    /* For m > 0 (cylindrical / spherical), weight fluxes by x^m at
     * midpoints and divide the divergence by x_i^m at nodes. m = 0 is
     * a no-op (xpow ≡ 1). */
    int mm = pdepe_ctx_.m;
    /* Pre-multiply midpoint fluxes by x_{i+1/2}^m. */
    if (mm != 0) {
        for (int64_t i = 0; i < Nx - 1; ++i) {
            double xm = 0.5 * (pdepe_ctx_.xmesh[i] + pdepe_ctx_.xmesh[i + 1]);
            flx[i] *= pdepe_xpow(xm, mm);
        }
    }

    /* Left boundary node 0. */
    if (dirichlet_left) {
        out->data[0] = 0.0;
    } else {
        double xi = pdepe_ctx_.xmesh[0];
        double ui = u[0];
        double dudx = (u[1] - u[0]) /
                      (pdepe_ctx_.xmesh[1] - pdepe_ctx_.xmesh[0]);
        matlab_mat *r = pdepe_ctx_.pdefn(xi, t, ui, dudx);
        double c = (r && r->rows * r->cols >= 1) ? r->data[0] : 1.0;
        double s = (r && r->rows * r->cols >= 3) ? r->data[2] : 0.0;
        if (r) mat_free_(r);
        if (c == 0.0) c = 1e-30;
        double cell_w = 0.5 * (pdepe_ctx_.xmesh[1] - pdepe_ctx_.xmesh[0]);
        double xpow_l = pdepe_xpow(xi, mm);
        double f_l_bdy_w = (mm != 0) ? f_left_bdy * xpow_l : f_left_bdy;
        double inv_xpow = (xpow_l == 0.0) ? 0.0 : (1.0 / xpow_l);
        out->data[0] = (((flx[0] - f_l_bdy_w) / cell_w) * inv_xpow + s) / c;
    }

    /* Interior nodes i = 1 .. Nx-2. */
    for (int64_t i = 1; i < Nx - 1; ++i) {
        double xi = pdepe_ctx_.xmesh[i];
        double ui = u[i];
        double dudx = (u[i + 1] - u[i - 1]) /
                      (pdepe_ctx_.xmesh[i + 1] - pdepe_ctx_.xmesh[i - 1]);
        matlab_mat *r = pdepe_ctx_.pdefn(xi, t, ui, dudx);
        double c = (r && r->rows * r->cols >= 1) ? r->data[0] : 1.0;
        double s = (r && r->rows * r->cols >= 3) ? r->data[2] : 0.0;
        if (r) mat_free_(r);
        if (c == 0.0) c = 1e-30;
        double dx_avg = 0.5 * (pdepe_ctx_.xmesh[i + 1] - pdepe_ctx_.xmesh[i - 1]);
        double dflux  = flx[i] - flx[i - 1];
        double xpow_i = pdepe_xpow(xi, mm);
        double inv_xpow = (xpow_i == 0.0) ? 0.0 : (1.0 / xpow_i);
        out->data[i] = ((dflux / dx_avg) * inv_xpow + s) / c;
    }

    /* Right boundary node Nx-1. */
    if (dirichlet_right) {
        out->data[Nx - 1] = 0.0;
    } else {
        double xi = pdepe_ctx_.xmesh[Nx - 1];
        double ui = u[Nx - 1];
        double dudx = (u[Nx - 1] - u[Nx - 2]) /
                      (pdepe_ctx_.xmesh[Nx - 1] - pdepe_ctx_.xmesh[Nx - 2]);
        matlab_mat *r = pdepe_ctx_.pdefn(xi, t, ui, dudx);
        double c = (r && r->rows * r->cols >= 1) ? r->data[0] : 1.0;
        double s = (r && r->rows * r->cols >= 3) ? r->data[2] : 0.0;
        if (r) mat_free_(r);
        if (c == 0.0) c = 1e-30;
        double cell_w = 0.5 * (pdepe_ctx_.xmesh[Nx - 1] - pdepe_ctx_.xmesh[Nx - 2]);
        double xpow_r = pdepe_xpow(xi, mm);
        double f_r_bdy_w = (mm != 0) ? f_right_bdy * xpow_r : f_right_bdy;
        double inv_xpow = (xpow_r == 0.0) ? 0.0 : (1.0 / xpow_r);
        out->data[Nx - 1] = (((f_r_bdy_w - flx[Nx - 2]) / cell_w) * inv_xpow + s) / c;
    }

    free(u);
    free(flx);
    return out;
}

matlab_mat *matlab_pdepe(double m, void *pdefn_p, void *icfn_p, void *bcfn_p,
                          matlab_mat *xmesh, matlab_mat *tspan) {
    if (!xmesh || !tspan || !pdefn_p || !icfn_p || !bcfn_p)
        return mat_alloc(0, 0);
    int64_t Nx = xmesh->rows * xmesh->cols;
    int64_t Nt = tspan->rows * tspan->cols;
    if (Nx < 3 || Nt < 2) return mat_alloc(0, 0);
    int mi = (int)m;
    if (mi < 0 || mi > 2 || (double)mi != m) return mat_alloc(0, 0);
    /* For m > 0 the discretization divides by x^m at each node — the
     * mesh must be strictly positive. (MATLAB allows xmesh[0] = 0 with
     * a special axis-of-symmetry treatment; deferred to follow-up.) */
    if (mi != 0 && xmesh->data[0] <= 0.0) return mat_alloc(0, 0);

    pdepe_pdefn_t pdefn = (pdepe_pdefn_t)pdefn_p;
    pdepe_icfn_t  icfn  = (pdepe_icfn_t)icfn_p;
    pdepe_bcfn_t  bcfn  = (pdepe_bcfn_t)bcfn_p;

    pdepe_ctx_.pdefn = pdefn;
    pdepe_ctx_.bcfn  = bcfn;
    pdepe_ctx_.xmesh = xmesh->data;
    pdepe_ctx_.Nx    = Nx;
    pdepe_ctx_.m     = mi;
    pdepe_ctx_.err_flag = 0;
    /* Invalidate the ode23s_v cache: successive pdepe calls share the
     * same _pdepe_rhs fn pointer and may share the same y0 values, so
     * the cache key would otherwise hit stale solutions when only the
     * pdepe context (m, bcfn, …) changed. */
    if (ode_v_cache_.valid) {
        if (ode_v_cache_.t) { free(ode_v_cache_.t->data); free(ode_v_cache_.t); }
        if (ode_v_cache_.y) { free(ode_v_cache_.y->data); free(ode_v_cache_.y); }
        ode_v_cache_.t = NULL;
        ode_v_cache_.y = NULL;
        ode_v_cache_.valid = 0;
    }

    /* Initial state covers ALL mesh points (boundaries included). */
    matlab_mat *u0 = mat_alloc(Nx, 1);
    for (int64_t i = 0; i < Nx; ++i) u0->data[i] = icfn(xmesh->data[i]);

    /* Integrate via ode23s_v (handles stiff parabolic problems). */
    matlab_mat *T = matlab_ode23s_v_t(pdepe_rhs, tspan, u0);
    matlab_mat *U = matlab_ode23s_v_y(pdepe_rhs, tspan, u0);
    int64_t Nt_out = T->rows;

    /* Re-snap Dirichlet boundary values at each output time so any
     * minor drift inside the integrator doesn't appear in `sol`. */
    matlab_mat *sol = mat_alloc(Nt_out, Nx);
    if (Nt_out > 0) {
        memcpy(sol->data, U->data,
               (size_t)Nt_out * (size_t)Nx * sizeof(double));
        for (int64_t k = 0; k < Nt_out; ++k) {
            double t = T->data[k];
            double pl, ql, pr, qr;
            double ul = sol->data[k * Nx + 0];
            double ur = sol->data[k * Nx + (Nx - 1)];
            if (pdepe_eval_bc(t, ul, ur, &pl, &ql, &pr, &qr) != 0) continue;
            if (ql == 0.0) sol->data[k * Nx + 0]      = ul - pl;
            if (qr == 0.0) sol->data[k * Nx + (Nx - 1)] = ur - pr;
        }
    }
    return sol;
}

/* =====================================================================
 * ode_events — IVP solver with event detection.
 *
 * Compromise API: `[t, y, te, ye, ie] = ode_events(@f, tspan, y0, @evt)`.
 * Non-MATLAB syntax (MATLAB's canonical form is
 * `ode45(@f, tspan, y0, odeset('Events', @evt))`); the explicit @evt
 * argument avoids the function-handle-in-struct ABI question.
 *
 * v1 scope:
 *   - Scalar y only (vector y is the natural follow-up).
 *   - Single event, returned as a 3×1 column [value; isterminal; direction].
 *   - direction = 0  → fire on any sign change (default for most users)
 *   - direction = +1 → fire only on rising crossings
 *   - direction = -1 → fire only on falling crossings
 *   - Cubic-Hermite dense output between accepted RK45 steps.
 *   - Bisection root-finder (50 iterations, |v| < 1e-12 stop).
 *
 * The 5-result dispatch in LowerTensorOps wires this to runtime entries
 * matlab_ode_events_{t,y,te,ye,ie} sharing a thread-local cache.
 * ===================================================================== */
typedef matlab_mat *(*ode_evt_t)(double t, double y);

struct ode_events_cache_slot {
    void *fp;
    void *evt;
    matlab_mat *tspan;
    double y0;
    matlab_mat *t;
    matlab_mat *y;
    matlab_mat *te;
    matlab_mat *ye;
    matlab_mat *ie;
    int valid;
};

#if defined(__GNUC__) || defined(__clang__)
__thread struct ode_events_cache_slot ode_events_cache_;
#else
struct ode_events_cache_slot ode_events_cache_;
#endif

static int ode_evt_eval(ode_evt_t evt, double t, double y,
                         double *value, int *isterminal, int *direction) {
    matlab_mat *r = evt(t, y);
    if (!r || r->rows * r->cols < 1) {
        if (r) mat_free_(r);
        *value = 0.0; *isterminal = 0; *direction = 0;
        return 1;
    }
    int64_t nd = r->rows * r->cols;
    *value      = r->data[0];
    *isterminal = (nd >= 2) ? (int)r->data[1] : 0;
    *direction  = (nd >= 3) ? (int)r->data[2] : 0;
    mat_free_(r);
    return 0;
}

static double ode_evt_bisect(ode_evt_t evt, double t, double h,
                              double y, double y_new, double k1, double k7,
                              double v0, double v1) {
    (void)v1;
    double lo = 0.0, hi = 1.0;
    double vlo = v0;
    for (int it = 0; it < 50; ++it) {
        double mid = 0.5 * (lo + hi);
        double y_mid = ode_hermite(y, y_new, k1, k7, h, mid);
        double v; int term, dir;
        if (ode_evt_eval(evt, t + mid * h, y_mid, &v, &term, &dir) != 0) return mid;
        if (fabs(v) < 1e-12 || (hi - lo) < 1e-15) return mid;
        if ((vlo < 0.0 && v > 0.0) || (vlo > 0.0 && v < 0.0)) {
            hi = mid;
        } else {
            lo = mid; vlo = v;
        }
    }
    return 0.5 * (lo + hi);
}

static void rk_solve_dp45_events(ode_rhs_t f, ode_evt_t evt,
                                  const double *targets, int64_t n_targets,
                                  double y0,
                                  double rtol, double atol,
                                  double max_step, double init_step,
                                  int refine,
                                  double **T, double **Y, int64_t *N,
                                  double **TE, double **YE, int **IE,
                                  int64_t *NE) {
    const int max_steps = 100000;
    if (refine < 1) refine = 1;
    if (n_targets < 2) {
        *T = NULL; *Y = NULL; *N = 0;
        *TE = NULL; *YE = NULL; *IE = NULL; *NE = 0;
        return;
    }
    double t0 = targets[0], tf = targets[n_targets - 1];
    int user_grid = (n_targets > 2);
    int64_t cap = user_grid ? n_targets : 256;
    double *Tb = (double *)malloc((size_t)cap * sizeof(double));
    double *Yb = (double *)malloc((size_t)cap * sizeof(double));
    int64_t n = 0;
    int64_t ev_cap = 16;
    double *TEb = (double *)malloc((size_t)ev_cap * sizeof(double));
    double *YEb = (double *)malloc((size_t)ev_cap * sizeof(double));
    int    *IEb = (int *)   malloc((size_t)ev_cap * sizeof(int));
    int64_t ne = 0;

    double t = t0, y = y0;
    ode_push(&Tb, &Yb, &n, &cap, t, y);
    int64_t next_tgt = 1;

    double span = tf - t0;
    double h = (init_step > 0.0) ? (span >= 0.0 ? init_step : -init_step)
                                 : span * 0.01;
    if (h == 0.0 || span == 0.0) {
        *T = Tb; *Y = Yb; *N = n;
        *TE = TEb; *YE = YEb; *IE = IEb; *NE = ne;
        return;
    }
    int forward = h > 0;
    if (max_step > 0.0) {
        if (h >  max_step) h =  max_step;
        if (h < -max_step) h = -max_step;
    }

    double k1 = f(t, y);
    double v_prev; int term_prev, dir_prev;
    ode_evt_eval(evt, t, y, &v_prev, &term_prev, &dir_prev);

    int steps = 0;
    int halted = 0;
    while ((forward ? t < tf : t > tf) && steps < max_steps && !halted) {
        ++steps;
        if (forward ? (t + h > tf) : (t + h < tf)) h = tf - t;

        double k2 = f(t + h * (1.0/5.0),
                      y + h * (k1 * (1.0/5.0)));
        double k3 = f(t + h * (3.0/10.0),
                      y + h * (k1 * (3.0/40.0) + k2 * (9.0/40.0)));
        double k4 = f(t + h * (4.0/5.0),
                      y + h * (k1 * (44.0/45.0) - k2 * (56.0/15.0)
                              + k3 * (32.0/9.0)));
        double k5 = f(t + h * (8.0/9.0),
                      y + h * (k1 * (19372.0/6561.0)
                              - k2 * (25360.0/2187.0)
                              + k3 * (64448.0/6561.0)
                              - k4 * (212.0/729.0)));
        double k6 = f(t + h,
                      y + h * (k1 * (9017.0/3168.0)
                              - k2 * (355.0/33.0)
                              + k3 * (46732.0/5247.0)
                              + k4 * (49.0/176.0)
                              - k5 * (5103.0/18656.0)));
        double y5 = y + h * (k1 * (35.0/384.0)
                            + k3 * (500.0/1113.0)
                            + k4 * (125.0/192.0)
                            - k5 * (2187.0/6784.0)
                            + k6 * (11.0/84.0));
        double k7 = f(t + h, y5);
        double err = h * (k1 * (71.0/57600.0)
                         - k3 * (71.0/16695.0)
                         + k4 * (71.0/1920.0)
                         - k5 * (17253.0/339200.0)
                         + k6 * (22.0/525.0)
                         - k7 * (1.0/40.0));
        double scale = atol + rtol * fmax(fabs(y), fabs(y5));
        double normerr = (scale > 0) ? fabs(err) / scale : 0.0;

        if (normerr <= 1.0) {
            double v_new; int term_new, dir_setting;
            ode_evt_eval(evt, t + h, y5, &v_new, &term_new, &dir_setting);
            int crossed = 0;
            if (v_prev * v_new < 0.0) {
                int rising = (v_new > v_prev);
                if (dir_setting == 0) crossed = 1;
                else if (dir_setting > 0 && rising) crossed = 1;
                else if (dir_setting < 0 && !rising) crossed = 1;
            }
            if (crossed) {
                double th_star = ode_evt_bisect(evt, t, h, y, y5, k1, k7,
                                                 v_prev, v_new);
                double te = t + th_star * h;
                double ye = ode_hermite(y, y5, k1, k7, h, th_star);
                if (ne == ev_cap) {
                    ev_cap *= 2;
                    TEb = (double *)realloc(TEb, (size_t)ev_cap * sizeof(double));
                    YEb = (double *)realloc(YEb, (size_t)ev_cap * sizeof(double));
                    IEb = (int *)   realloc(IEb, (size_t)ev_cap * sizeof(int));
                }
                TEb[ne] = te;
                YEb[ne] = ye;
                IEb[ne] = 1;
                ++ne;
                if (term_new) {
                    ode_push(&Tb, &Yb, &n, &cap, te, ye);
                    halted = 1;
                    break;
                }
            }
            v_prev = v_new;

            if (user_grid) {
                while (next_tgt < n_targets) {
                    double tt = targets[next_tgt];
                    int in_range = forward ? (tt <= t + h) : (tt >= t + h);
                    if (!in_range) break;
                    double th = (h == 0.0) ? 0.0 : (tt - t) / h;
                    double yi = (next_tgt == n_targets - 1)
                        ? y5
                        : ode_hermite(y, y5, k1, k7, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, tt, yi);
                    ++next_tgt;
                }
            } else {
                for (int j = 1; j <= refine; ++j) {
                    double th = (double)j / (double)refine;
                    double ti = t + h * th;
                    double yi = (j == refine)
                        ? y5
                        : ode_hermite(y, y5, k1, k7, h, th);
                    ode_push(&Tb, &Yb, &n, &cap, ti, yi);
                }
            }
            t += h;
            y  = y5;
            k1 = k7;
            if (user_grid && next_tgt >= n_targets) break;
        }

        double fac = (normerr == 0.0) ? 5.0
                                      : 0.9 * pow(normerr, -1.0/5.0);
        if (fac < 0.2) fac = 0.2;
        if (fac > 5.0) fac = 5.0;
        h *= fac;
        if (max_step > 0.0) {
            if (h >  max_step) h =  max_step;
            if (h < -max_step) h = -max_step;
        }
    }

    *T = Tb; *Y = Yb; *N = n;
    *TE = TEb; *YE = YEb; *IE = IEb; *NE = ne;
}

static void ode_events_compute(ode_rhs_t f, ode_evt_t evt,
                                matlab_mat *tspan, double y0) {
    if (ode_events_cache_.valid &&
        ode_events_cache_.fp == (void *)f &&
        ode_events_cache_.evt == (void *)evt &&
        ode_events_cache_.tspan == tspan &&
        ode_events_cache_.y0 == y0) {
        return;
    }
    if (ode_events_cache_.valid) {
        if (ode_events_cache_.t)  { free(ode_events_cache_.t->data);  free(ode_events_cache_.t);  }
        if (ode_events_cache_.y)  { free(ode_events_cache_.y->data);  free(ode_events_cache_.y);  }
        if (ode_events_cache_.te) { free(ode_events_cache_.te->data); free(ode_events_cache_.te); }
        if (ode_events_cache_.ye) { free(ode_events_cache_.ye->data); free(ode_events_cache_.ye); }
        if (ode_events_cache_.ie) { free(ode_events_cache_.ie->data); free(ode_events_cache_.ie); }
        ode_events_cache_.t = ode_events_cache_.y = NULL;
        ode_events_cache_.te = ode_events_cache_.ye = ode_events_cache_.ie = NULL;
        ode_events_cache_.valid = 0;
    }
    int64_t n_tgt = tspan ? tspan->rows * tspan->cols : 0;
    if (!f || !evt || n_tgt < 2 || !tspan->data) {
        ode_events_cache_.t  = mat_alloc(0, 1);
        ode_events_cache_.y  = mat_alloc(0, 1);
        ode_events_cache_.te = mat_alloc(0, 1);
        ode_events_cache_.ye = mat_alloc(0, 1);
        ode_events_cache_.ie = mat_alloc(0, 1);
    } else {
        double *Tb = NULL, *Yb = NULL;
        double *TEb = NULL, *YEb = NULL;
        int    *IEb = NULL;
        int64_t n = 0, ne = 0;
        rk_solve_dp45_events(f, evt, tspan->data, n_tgt, y0,
                              1e-3, 1e-6, 0.0, 0.0, 4,
                              &Tb, &Yb, &n, &TEb, &YEb, &IEb, &ne);
        ode_buffers_to_mats(Tb, Yb, n, &ode_events_cache_.t,
                             &ode_events_cache_.y);
        free(Tb); free(Yb);
        matlab_mat *Te = mat_alloc(ne, 1);
        matlab_mat *Ye = mat_alloc(ne, 1);
        matlab_mat *Ie = mat_alloc(ne, 1);
        if (ne > 0) {
            memcpy(Te->data, TEb, (size_t)ne * sizeof(double));
            memcpy(Ye->data, YEb, (size_t)ne * sizeof(double));
            for (int64_t k = 0; k < ne; ++k) Ie->data[k] = (double)IEb[k];
        }
        ode_events_cache_.te = Te;
        ode_events_cache_.ye = Ye;
        ode_events_cache_.ie = Ie;
        free(TEb); free(YEb); free(IEb);
    }
    ode_events_cache_.fp    = (void *)f;
    ode_events_cache_.evt   = (void *)evt;
    ode_events_cache_.tspan = tspan;
    ode_events_cache_.y0    = y0;
    ode_events_cache_.valid = 1;
}

static matlab_mat *mat_clone_col_e(matlab_mat *src) {
    if (!src) return mat_alloc(0, 1);
    matlab_mat *out = mat_alloc(src->rows, src->cols);
    int64_t n = src->rows * src->cols;
    if (n > 0) memcpy(out->data, src->data, (size_t)n * sizeof(double));
    return out;
}

matlab_mat *matlab_ode_events_t(ode_rhs_t f, matlab_mat *tspan,
                                 double y0, void *evt_p) {
    ode_events_compute(f, (ode_evt_t)evt_p, tspan, y0);
    return mat_clone_col_e(ode_events_cache_.t);
}
matlab_mat *matlab_ode_events_y(ode_rhs_t f, matlab_mat *tspan,
                                 double y0, void *evt_p) {
    ode_events_compute(f, (ode_evt_t)evt_p, tspan, y0);
    return mat_clone_col_e(ode_events_cache_.y);
}
matlab_mat *matlab_ode_events_te(ode_rhs_t f, matlab_mat *tspan,
                                  double y0, void *evt_p) {
    ode_events_compute(f, (ode_evt_t)evt_p, tspan, y0);
    return mat_clone_col_e(ode_events_cache_.te);
}
matlab_mat *matlab_ode_events_ye(ode_rhs_t f, matlab_mat *tspan,
                                  double y0, void *evt_p) {
    ode_events_compute(f, (ode_evt_t)evt_p, tspan, y0);
    return mat_clone_col_e(ode_events_cache_.ye);
}
matlab_mat *matlab_ode_events_ie(ode_rhs_t f, matlab_mat *tspan,
                                  double y0, void *evt_p) {
    ode_events_compute(f, (ode_evt_t)evt_p, tspan, y0);
    return mat_clone_col_e(ode_events_cache_.ie);
}

} /* extern "C" */
