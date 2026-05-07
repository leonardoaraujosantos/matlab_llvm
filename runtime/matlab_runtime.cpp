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
 * Jacobi eigenvalue iteration for symmetric matrices.
 *
 * Returns a column vector of eigenvalues in ascending order. If the input
 * isn't symmetric, we work on H = (A + Aᵀ)/2, which returns correct
 * eigenvalues for any symmetric input and a reasonable approximation for
 * slightly-non-symmetric inputs. For genuinely non-symmetric matrices
 * (e.g. with complex eigenvalues), this is garbage — a future extension
 * would add QR iteration for the general case.
 *
 * Algorithm: repeatedly find the largest off-diagonal element (or sweep
 * over all pairs) and apply a Jacobi rotation R that zeros it in the
 * 2×2 principal submatrix indexed by (p, q). After convergence, the
 * diagonal of H holds the eigenvalues.
 */
matlab_mat *matlab_eig(matlab_mat *A_in) {
    if (!A_in || A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
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

/* matlab_eig_V(A): eigenvector matrix (columns = eigenvectors), ordered
 * so the i-th column corresponds to the i-th ascending eigenvalue. */
matlab_mat *matlab_eig_V(matlab_mat *A_in) {
    if (!A_in || A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
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

/* matlab_eig_D(A): diagonal matrix of eigenvalues (ascending). */
matlab_mat *matlab_eig_D(matlab_mat *A_in) {
    if (!A_in || A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
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
matlab_mat *matlab_size(matlab_mat *A) {
    matlab_mat *R = mat_alloc(1, 2);
    R->data[0] = (double)A->rows;
    R->data[1] = (double)A->cols;
    return R;
}

/* size(A, dim). dim is 1-based; 1=rows, 2=cols; any other dim returns 1. */
double matlab_size_dim(matlab_mat *A, double dim) {
    int64_t d = (int64_t)dim;
    if (d == 1) return (double)A->rows;
    if (d == 2) return (double)A->cols;
    return 1.0;
}

double matlab_length(matlab_mat *A) {
    if (A->rows == 0 || A->cols == 0) return 0.0;
    return (double)(A->rows > A->cols ? A->rows : A->cols);
}

double matlab_numel(matlab_mat *A)  { return (double)(A->rows * A->cols); }
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
    if (ri < 0 || ri >= A->rows || cj < 0 || cj >= A->cols) return 0.0;
    return A->data[ri * A->cols + cj];
}

double matlab_subscript1_s(matlab_mat *A, double i) {
    int64_t idx = (int64_t)i - 1;
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

struct matlab_table_s {
    int32_t  nvars;
    int32_t  cap;
    int32_t  nrows;
    char   **names;
    void   **data;     /* per-column matlab_mat * */
};
typedef struct matlab_table_s matlab_table;

static void table_grow(matlab_table *t, int32_t need) {
    if (t->cap >= need) return;
    int32_t nc = t->cap ? t->cap * 2 : 4;
    while (nc < need) nc *= 2;
    t->names = (char **)realloc(t->names, (size_t)nc * sizeof(char *));
    t->data  = (void **)realloc(t->data,  (size_t)nc * sizeof(void *));
    for (int32_t i = t->cap; i < nc; ++i) {
        t->names[i] = NULL;
        t->data[i]  = NULL;
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
 * matlab_mat *; the caller must not free it. */
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
    t->data[i] = col;
}

extern "C" matlab_mat *matlab_table_get_column(matlab_table *t,
                                                const char *name,
                                                int64_t namelen) {
    int32_t i = table_find(t, name, namelen);
    if (i < 0 || !t->data[i]) return mat_alloc(0, 0);
    return (matlab_mat *)t->data[i];
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
    /* Body — each row, one element per column. */
    for (int32_t r = 0; r < t->nrows; ++r) {
        for (int32_t c = 0; c < t->nvars; ++c) {
            matlab_mat *col = (matlab_mat *)t->data[c];
            if (col && r < col->rows * col->cols) {
                double v = col->data[r];
                /* Round-trip integer-valued doubles as %d for cleanliness. */
                if (v == (double)(int64_t)v && fabs(v) < 1e15)
                    printf("    %*lld", W, (long long)v);
                else
                    printf("    %*.*g", W, 6, v);
            } else {
                printf("    %*s", W, "");
            }
        }
        putchar('\n');
    }
    pthread_mutex_unlock(&matlab_io_mutex);
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

struct matlab_datetime_s { double seconds; };
typedef struct matlab_datetime_s matlab_datetime;

struct matlab_duration_s { double seconds; };
typedef struct matlab_duration_s matlab_duration;

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

double matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len) {
    return matlab_struct_get_f64((matlab_struct *)o, name, len);
}

matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len) {
    return matlab_struct_get_mat((matlab_struct *)o, name, len);
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
