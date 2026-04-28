/* Tiny MATLAB-runtime shim. Linked with programs produced by matlabc's
 * -emit-llvm pipeline.
 *
 * All functions use a leading `matlab_` prefix to avoid collision with libc
 * and to make the calling convention explicit to the compiler frontend.
 */

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  /* for write(2), used by matlab_err_emit_traceback_to_stderr */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* A single global mutex serializes all stdout I/O so parfor bodies that call
 * disp/fprintf don't interleave mid-line. This is a tiny concession to
 * predictability; real MATLAB uses per-worker stdout aggregation. */
static pthread_mutex_t matlab_io_mutex = PTHREAD_MUTEX_INITIALIZER;

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

typedef struct matlab_mat {
    double *data;      /* row-major, rows*cols doubles */
    int64_t rows;
    int64_t cols;
} matlab_mat;

/* Forward-declared here (layout + body further down in the complex
 * section) so matlab_disp_mat and the other polymorphic entries can
 * discriminate real vs. complex descriptors via the magic marker.
 *
 * matlab_mat3 carries its own magic for the same reason — once the
 * DAP server starts walking matrices for the variables panel, it
 * needs a way to tell 2-D and 3-D apart from a kind=1 ptr without
 * a separate kind id. The magics are at offset 0 of each tagged
 * descriptor; matlab_mat is untagged (its first 8 bytes are a heap
 * data pointer whose low 32 bits won't collide in practice). */
#define MATLAB_MAT_C_MAGIC 0xC0FFEE01u
#define MATLAB_MAT3_MAGIC  0xC0FFEE03u

typedef struct matlab_mat_c matlab_mat_c;
void matlab_disp_mat_c(matlab_mat_c *A);

static int mat_is_complex(const void *p) {
    if (!p) return 0;
    return *(const uint32_t *)p == MATLAB_MAT_C_MAGIC;
}
static int mat_is_3d(const void *p) {
    if (!p) return 0;
    return *(const uint32_t *)p == MATLAB_MAT3_MAGIC;
}

static matlab_mat *mat_alloc(int64_t m, int64_t n) {
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
    matlab_mat *A = mat_alloc(rm, cn);
    for (int64_t k = 0; k < rm * cn; ++k) A->data[k] = 1.0;
    return A;
}

matlab_mat *matlab_eye(double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    matlab_mat *A = mat_alloc(rm, cn);
    int64_t d = rm < cn ? rm : cn;
    for (int64_t i = 0; i < d; ++i) A->data[i * cn + i] = 1.0;
    return A;
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

/* inv(A): Gauss-Jordan via LU, solving A*X = I column by column. */
matlab_mat *matlab_inv(matlab_mat *A) {
    if (A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    double *LU = (double *)malloc((size_t)(n * n) * sizeof(double));
    memcpy(LU, A->data, (size_t)(n * n) * sizeof(double));
    int64_t *piv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int sign;
    if (lu_decompose(LU, n, piv, &sign) != 0) {
        free(LU); free(piv);
        return mat_alloc(0, 0);
    }
    matlab_mat *X = mat_alloc(n, n);
    double *rhs = (double *)malloc((size_t)n * sizeof(double));
    double *col = (double *)malloc((size_t)n * sizeof(double));
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t i = 0; i < n; ++i) rhs[i] = (i == c) ? 1.0 : 0.0;
        lu_solve_column(LU, n, piv, rhs, col);
        for (int64_t i = 0; i < n; ++i) X->data[i * n + c] = col[i];
    }
    free(rhs); free(col); free(piv); free(LU);
    return X;
}

/* A \ B: solve A*X = B (MATLAB left divide). B may have multiple columns. */
matlab_mat *matlab_mldivide_mm(matlab_mat *A, matlab_mat *B) {
    if (A->rows != A->cols || A->rows != B->rows) return mat_alloc(0, 0);
    int64_t n = A->rows;
    int64_t k = B->cols;
    double *LU = (double *)malloc((size_t)(n * n) * sizeof(double));
    memcpy(LU, A->data, (size_t)(n * n) * sizeof(double));
    int64_t *piv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int sign;
    if (lu_decompose(LU, n, piv, &sign) != 0) {
        free(LU); free(piv);
        return mat_alloc(0, 0);
    }
    matlab_mat *X = mat_alloc(n, k);
    double *rhs = (double *)malloc((size_t)n * sizeof(double));
    double *col = (double *)malloc((size_t)n * sizeof(double));
    for (int64_t c = 0; c < k; ++c) {
        for (int64_t i = 0; i < n; ++i) rhs[i] = B->data[i * k + c];
        lu_solve_column(LU, n, piv, rhs, col);
        for (int64_t i = 0; i < n; ++i) X->data[i * k + c] = col[i];
    }
    free(rhs); free(col); free(piv); free(LU);
    return X;
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
    /* `U` (m×n) starts as a copy of A. */
    double *U = (double *)malloc((size_t)(m * n) * sizeof(double));
    memcpy(U, A->data, (size_t)(m * n) * sizeof(double));

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
    double *sv = (double *)malloc((size_t)n * sizeof(double));
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
    matlab_mat *S = mat_alloc(k, 1);
    for (int64_t i = 0; i < k; ++i) S->data[i] = sv[i];
    free(sv);
    free(U);
    (void)T;  /* T is kept alive by the arena-leak policy */
    return S;
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
    if (A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
    double *H = (double *)malloc((size_t)(n * n) * sizeof(double));
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

    matlab_mat *E = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) E->data[i] = H[i * n + i];
    /* Insertion sort, ascending. */
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            if (E->data[j] < E->data[i]) {
                double t = E->data[i]; E->data[i] = E->data[j]; E->data[j] = t;
            }
        }
    }
    free(H);
    return E;
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
    double *H = (double *)malloc((size_t)(n * n) * sizeof(double));
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
    free(H);
}

/* matlab_eig_V(A): eigenvector matrix (columns = eigenvectors), ordered
 * so the i-th column corresponds to the i-th ascending eigenvalue. */
matlab_mat *matlab_eig_V(matlab_mat *A_in) {
    if (A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
    double *eigvals = (double *)malloc((size_t)n * sizeof(double));
    matlab_mat *V = mat_alloc(n, n);
    jacobi_sym(A_in, eigvals, V->data);
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
    free(eigvals);
    return V;
}

/* matlab_eig_D(A): diagonal matrix of eigenvalues (ascending). */
matlab_mat *matlab_eig_D(matlab_mat *A_in) {
    if (A_in->rows != A_in->cols) return mat_alloc(0, 0);
    int64_t n = A_in->rows;
    double *eigvals = (double *)malloc((size_t)n * sizeof(double));
    double *Vtmp = (double *)malloc((size_t)(n * n) * sizeof(double));
    jacobi_sym(A_in, eigvals, Vtmp);
    /* Ascending sort of eigvals. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j)
            if (eigvals[j] < eigvals[i]) {
                double t = eigvals[i]; eigvals[i] = eigvals[j]; eigvals[j] = t;
            }
    matlab_mat *D = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) D->data[i] = 0.0;
    for (int64_t i = 0; i < n; ++i) D->data[i * n + i] = eigvals[i];
    free(eigvals);
    free(Vtmp);
    return D;
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

matlab_mat *matlab_transpose(matlab_mat *A) {
    matlab_mat *B = mat_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i)
        for (int64_t j = 0; j < A->cols; ++j)
            B->data[j * A->rows + i] = A->data[i * A->cols + j];
    return B;
}

/* diag(A): if A is a row or column vector, build an n×n matrix with A on
 * the main diagonal. Otherwise extract the main diagonal as a column. */
matlab_mat *matlab_diag(matlab_mat *A) {
    if (A->rows == 1 || A->cols == 1) {
        int64_t n = A->rows * A->cols;
        matlab_mat *D = mat_alloc(n, n);
        for (int64_t i = 0; i < n; ++i) D->data[i * n + i] = A->data[i];
        return D;
    }
    int64_t d = A->rows < A->cols ? A->rows : A->cols;
    matlab_mat *V = mat_alloc(d, 1);
    for (int64_t i = 0; i < d; ++i) V->data[i] = A->data[i * A->cols + i];
    return V;
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

matlab_mat *matlab_repmat(matlab_mat *A, double m, double n) {
    int64_t tm = (int64_t)m, tn = (int64_t)n;
    int64_t nr = A->rows * tm, nc = A->cols * tn;
    matlab_mat *B = mat_alloc(nr, nc);
    for (int64_t bi = 0; bi < tm; ++bi)
        for (int64_t bj = 0; bj < tn; ++bj)
            for (int64_t i = 0; i < A->rows; ++i)
                for (int64_t j = 0; j < A->cols; ++j) {
                    int64_t r = bi * A->rows + i;
                    int64_t c = bj * A->cols + j;
                    B->data[r * nc + c] = A->data[i * A->cols + j];
                }
    return B;
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
    double *tmp = cap > 0 ? (double *)malloc((size_t)cap * sizeof(double)) : NULL;
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
        qsort(tmp, (size_t)u, sizeof(double), cmp_double_asc);
        int64_t uu = 0;
        for (int64_t k = 0; k < u; ++k)
            if (uu == 0 || tmp[uu - 1] != tmp[k]) tmp[uu++] = tmp[k];
        u = uu;
    }
    matlab_mat *R = mat_alloc(u, 1);
    if (u > 0) memcpy(R->data, tmp, (size_t)u * sizeof(double));
    if (tmp) free(tmp);
    return R;
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

matlab_mat *matlab_horzcat(matlab_mat *A, matlab_mat *B) {
    if (!A) return B;
    if (!B) return A;
    if (A->rows != B->rows) return mat_alloc(0, 0);
    int64_t m = A->rows, na = A->cols, nb = B->cols;
    matlab_mat *R = mat_alloc(m, na + nb);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < na; ++j)
            R->data[i * (na + nb) + j] = A->data[i * na + j];
        for (int64_t j = 0; j < nb; ++j)
            R->data[i * (na + nb) + na + j] = B->data[i * nb + j];
    }
    return R;
}

matlab_mat *matlab_vertcat(matlab_mat *A, matlab_mat *B) {
    if (!A) return B;
    if (!B) return A;
    if (A->cols != B->cols) return mat_alloc(0, 0);
    int64_t n = A->cols, ma = A->rows, mb = B->rows;
    matlab_mat *R = mat_alloc(ma + mb, n);
    memcpy(R->data, A->data, (size_t)(ma * n) * sizeof(double));
    memcpy(R->data + ma * n, B->data, (size_t)(mb * n) * sizeof(double));
    return R;
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
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        memcpy(R->data, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
        return R;
    }
    return matlab_transpose(A);
}

/* squeeze(A) is a no-op for 2-D matrices — MATLAB's squeeze only
 * collapses singleton dims in higher-rank arrays, which we don't
 * model. Keeps the name available as a syntactic identity. */
matlab_mat *matlab_squeeze(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    memcpy(R->data, A->data, (size_t)(m * n) * sizeof(double));
    return R;
}

matlab_mat *matlab_fliplr(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            R->data[i * n + j] = A->data[i * n + (n - 1 - j)];
    return R;
}

matlab_mat *matlab_flipud(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(m, n);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            R->data[i * n + j] = A->data[(m - 1 - i) * n + j];
    return R;
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
 * cols-by-rows. Element at (i, j) in the result is taken from
 * (j, cols-1-i) in the input. */
matlab_mat *matlab_rot90(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    matlab_mat *R = mat_alloc(n, m);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            R->data[i * m + j] = A->data[j * n + (n - 1 - i)];
    return R;
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
static int32_t matlab_error_flag = 0;

/* Error message storage: a heap-copy of the most recent error() string.
 * `matlab_set_error_msg` trims to 1023 bytes and null-terminates;
 * `matlab_err_disp_message` routes to the I/O runtime so catch blocks can
 * do `disp(ME.message)` and get the raw text without needing a new
 * char-matrix descriptor. */
static char matlab_error_msg[1024] = {0};
static int64_t matlab_error_msg_len = 0;

/* Forward declarations for the debug-frame snapshot below — the dbg
 * state struct is defined later in this file but matlab_set_error_msg
 * needs to peek at it to capture a backtrace at error time. */
struct matlab_dbg_frame;
static void matlab_err_snapshot_frames(void);
static void matlab_err_emit_traceback_to_stderr(void);

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

struct matlab_struct_s {
    int32_t nfields;
    int32_t capacity;
    char **names;
    int32_t *kinds;
    double *f64_vals;
    void **ptr_vals;
};
typedef struct matlab_struct_s matlab_struct;

matlab_struct *matlab_struct_new(void) {
    matlab_struct *s = (matlab_struct *)calloc(1, sizeof(*s));
    s->capacity = MATLAB_STRUCT_CAP_INIT;
    s->names    = (char **)calloc((size_t)s->capacity, sizeof(char *));
    s->kinds    = (int32_t *)calloc((size_t)s->capacity, sizeof(int32_t));
    s->f64_vals = (double *)calloc((size_t)s->capacity, sizeof(double));
    s->ptr_vals = (void **)calloc((size_t)s->capacity, sizeof(void *));
    return s;
}

static int32_t struct_find_field(matlab_struct *s, const char *name, int32_t len) {
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

static int32_t struct_reserve(matlab_struct *s, const char *name, int32_t len) {
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
typedef struct matlab_mat3 {
    /* Magic at offset 0 lets mat_is_3d() discriminate this descriptor
     * from a plain matlab_mat* without a separate kind id; see the
     * comment near MATLAB_MAT3_MAGIC. _pad keeps `data` 8-byte
     * aligned. */
    uint32_t magic;
    uint32_t _pad;
    double *data;
    int64_t rows, cols, depth;
} matlab_mat3;

static matlab_mat3 *mat3_alloc(int64_t m, int64_t n, int64_t p) {
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
struct matlab_obj_s {
    /* matlab_struct fields — MUST MATCH matlab_struct_s exactly. */
    int32_t nfields;
    int32_t capacity;
    char **names;
    int32_t *kinds;
    double *f64_vals;
    void **ptr_vals;
    /* Class tag — appended so the struct-compatible prefix stays
     * well-defined. */
    int32_t class_id;
};
typedef struct matlab_obj_s matlab_obj;

matlab_obj *matlab_obj_new(int32_t class_id) {
    matlab_obj *o = (matlab_obj *)calloc(1, sizeof(*o));
    o->capacity = MATLAB_STRUCT_CAP_INIT;
    o->names    = (char **)calloc((size_t)o->capacity, sizeof(char *));
    o->kinds    = (int32_t *)calloc((size_t)o->capacity, sizeof(int32_t));
    o->f64_vals = (double *)calloc((size_t)o->capacity, sizeof(double));
    o->ptr_vals = (void **)calloc((size_t)o->capacity, sizeof(void *));
    o->class_id = class_id;
    return o;
}

double matlab_obj_class_id(matlab_obj *o) {
    return o ? (double)o->class_id : 0.0;
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

matlab_string *matlab_string_from_literal(const char *src, int64_t len) {
    matlab_string *s = (matlab_string *)calloc(1, sizeof(*s));
    s->len = len < 0 ? 0 : len;
    s->data = (char *)malloc((size_t)s->len + 1);
    if (src && s->len > 0) memcpy(s->data, src, (size_t)s->len);
    s->data[s->len] = '\0';
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
matlab_mat *matlab_kron(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(0, 0);
    int64_t am = A->rows, an = A->cols;
    int64_t bm = B->rows, bn = B->cols;
    matlab_mat *R = mat_alloc(am * bm, an * bn);
    for (int64_t i = 0; i < am; ++i)
        for (int64_t p = 0; p < bm; ++p)
            for (int64_t j = 0; j < an; ++j)
                for (int64_t q = 0; q < bn; ++q) {
                    double av = A->data[i * an + j];
                    double bv = B->data[p * bn + q];
                    R->data[(i * bm + p) * (an * bn) + (j * bn + q)] = av * bv;
                }
    return R;
}

/* chol(A): upper-triangular Cholesky factor R such that R'*R = A,
 * for a symmetric positive-definite A. Returns a zero matrix if A
 * is not SPD (i.e. a negative diagonal appears). */
matlab_mat *matlab_chol(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    matlab_mat *R = mat_alloc(n, n);
    /* Upper-triangular factor, row-major. */
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            double s = A->data[i * n + j];
            for (int64_t k = 0; k < i; ++k)
                s -= R->data[k * n + i] * R->data[k * n + j];
            if (i == j) {
                if (s <= 0.0) {
                    /* Not SPD — zero out and bail out. */
                    for (int64_t k = 0; k < n * n; ++k) R->data[k] = 0.0;
                    matlab_set_error_msg("chol: matrix is not positive definite", 38);
                    return R;
                }
                R->data[i * n + j] = sqrt(s);
            } else {
                R->data[i * n + j] = s / R->data[i * n + i];
            }
        }
    }
    return R;
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
    matlab_mat *L = mat_alloc(n, n);
    matlab_mat *U = mat_alloc(n, n);
    int64_t *piv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    lu_factor(A, L, U, piv);
    free(piv);
    free(U->data); free(U);
    return L;
}

matlab_mat *matlab_lu_U(matlab_mat *A) {
    if (!A || A->rows != A->cols) return mat_alloc(0, 0);
    int64_t n = A->rows;
    matlab_mat *L = mat_alloc(n, n);
    matlab_mat *U = mat_alloc(n, n);
    int64_t *piv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    lu_factor(A, L, U, piv);
    free(piv);
    free(L->data); free(L);
    return U;
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
    matlab_mat *Q = mat_alloc(m, n);
    matlab_mat *R = mat_alloc(n, n);
    qr_factor(A, Q, R);
    free(R->data); free(R);
    return Q;
}

matlab_mat *matlab_qr_R(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    int64_t m = A->rows, n = A->cols;
    if (m < n) return mat_alloc(0, 0);
    matlab_mat *Q = mat_alloc(m, n);
    matlab_mat *R = mat_alloc(n, n);
    qr_factor(A, Q, R);
    free(Q->data); free(Q);
    return R;
}

/* -------- REPL workspace -------------------------------------------------
 *
 * A single matlab_struct* holds every variable the user has assigned
 * across REPL inputs. Each JIT-compiled input uses matlab_ws_get_* /
 * matlab_ws_set_* in place of local slots so state persists across
 * invocations. Field names are the user-visible variable names.
 *
 * The runtime lazily allocates the workspace on first touch, which
 * means the normal AOT path (matlabc -emit-llvm / -emit-c / ...)
 * never pays for it — these symbols only get linked into a program
 * when the compiler emits references to them, which today only the
 * REPL mode does.
 *--------------------------------------------------------------------------*/

static matlab_struct *matlab_ws = NULL;

static void matlab_ws_init_if_needed(void) {
    if (!matlab_ws) matlab_ws = matlab_struct_new();
}

/* Forward-declared up here so the matlab_ws_get_* / matlab_ws_set_*
 * call sites below compile. The bodies live alongside the rest of
 * the dbg machinery further down where matlab_dbg state is in
 * scope. */
static void matlab_ws_check_watch(const char *name, int64_t len);
static void matlab_ws_check_read_watch(const char *name, int64_t len);

double matlab_ws_get_f64(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    double v = matlab_struct_get_f64(matlab_ws, name, len);
    /* Read watchpoint check. Fast path: when n_wp is 0 (the
     * common case — no read-watches active) the entire body is a
     * single mutex-free load + compare and the JIT's REPL-mode
     * read sites pay no measurable cost. The full check fires
     * only once a read-watch is armed. */
    matlab_ws_check_read_watch(name, len);
    return v;
}

/* Forward decls for the undo helpers — defined alongside the
 * watch helpers in the matlab_dbg section. matlab_ws_push_undo
 * takes its own dbg-mutex acquisition since matlab_dbg itself
 * isn't visible up here (defined later). */
static void matlab_ws_push_undo(const char *name, int64_t len,
                                 int kind_being_written);

void matlab_ws_set_f64(const char *name, int64_t len, double v) {
    matlab_ws_init_if_needed();
    matlab_ws_push_undo(name, len, /*kind=*/0);
    matlab_struct_set_f64(matlab_ws, name, len, v);
    matlab_ws_check_watch(name, len);
}

matlab_mat *matlab_ws_get_mat(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_mat *m = matlab_struct_get_mat(matlab_ws, name, len);
    matlab_ws_check_read_watch(name, len);
    return m;
}

void matlab_ws_set_mat(const char *name, int64_t len, matlab_mat *m) {
    matlab_ws_init_if_needed();
    matlab_ws_push_undo(name, len, /*kind=*/1);
    matlab_struct_set_mat(matlab_ws, name, len, m);
    matlab_ws_check_watch(name, len);
}

/* Class-instance assignment to the script-level workspace. Stores
 * the obj pointer with kind=2 so matlab_dbg_ws_kind reports it as
 * an object — the DAP formatter then routes through the obj path
 * (`1x1 ClassName`, expandable into properties) instead of treating
 * the pointer as a matlab_mat * and reading garbage. */
void matlab_ws_set_obj(const char *name, int64_t len, matlab_obj *o) {
    matlab_ws_init_if_needed();
    matlab_ws_push_undo(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 2;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = o;
    matlab_ws_check_watch(name, len);
}

double matlab_ws_has(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    return matlab_struct_has_field(matlab_ws, name, len);
}

/* For the REPL's `whos` / `clear` style commands. */
void matlab_ws_clear(void) {
    /* Cheapest correct clear: allocate a fresh struct and let the old
     * one leak. Leak is bounded by the number of clear() calls in a
     * session, which is negligible for human-paced use. */
    matlab_ws = matlab_struct_new();
}

/* Forward declaration — the definition is later in the file, but
 * matlab_ws_clear_one needs the symbol. */
matlab_struct *matlab_struct_rmfield(matlab_struct *s, const char *name,
                                      int64_t len);

/* Remove a single variable from the workspace. Silent no-op if the
 * name isn't present. Matches MATLAB's `clear name` form. */
void matlab_ws_clear_one(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_struct_rmfield(matlab_ws, name, len);
}

/* `who` prints just the variable names, one per line. `whos` adds
 * shape/class columns. Both read the workspace struct. */
void matlab_ws_who(void) {
    matlab_ws_init_if_needed();
    pthread_mutex_lock(&matlab_io_mutex);
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        printf("%s\n", matlab_ws->names[i]);
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

void matlab_ws_whos(void) {
    matlab_ws_init_if_needed();
    pthread_mutex_lock(&matlab_io_mutex);
    printf("  %-16s %-16s %-8s\n", "Name", "Size", "Class");
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        const char *name = matlab_ws->names[i];
        if (matlab_ws->kinds[i] == 0) {
            printf("  %-16s %-16s %-8s\n", name, "1x1", "double");
        } else if (matlab_ws->kinds[i] == 1) {
            matlab_mat *m = (matlab_mat *)matlab_ws->ptr_vals[i];
            char shape[32];
            if (m) snprintf(shape, sizeof shape, "%lldx%lld",
                            (long long)m->rows, (long long)m->cols);
            else    snprintf(shape, sizeof shape, "-");
            printf("  %-16s %-16s %-8s\n", name, shape, "double");
        } else {
            printf("  %-16s %-16s %-8s\n", name, "?", "?");
        }
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* dbg(x) / dbg(x, "label") — source-located debug print to stderr.
 * The frontend passes the source file + line (derived from the call
 * site's Location) and the variable name (empty if the argument
 * isn't a bare NameExpr). Value is either an f64 scalar or a
 * matlab_mat* depending on which overload the lowerer selected. */
void matlab_dbg_f64(const char *file, int64_t file_len,
                    int32_t line,
                    const char *label, int64_t label_len,
                    double v) {
    /* The file / label strings come from LLVM globals that are NOT
     * null-terminated, so use the explicit length in the format. */
    int fl = (int)(file_len > 0 ? file_len : 0);
    int ll = (int)(label_len > 0 ? label_len : 0);
    const char *flt = file ? file : "<repl>";
    if (!file) fl = (int)strlen(flt);
    pthread_mutex_lock(&matlab_io_mutex);
    fprintf(stderr, "%.*s:%d: %.*s = %g\n",
            fl, flt, line,
            ll > 0 ? ll : (int)strlen("<expr>"),
            ll > 0 ? label : "<expr>", v);
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* -------- Full DAP hook infrastructure ------------------------------------
 *
 * Injected into the JIT'd code by matlabc -g / matlabc -dap. The hook is
 * called at each top-level statement boundary with (file_id, line) where
 * file_id is the SourceManager's FileID cast to i32. The DAP server
 * (in matlabc's -dap mode) shares this state via locks + condvar and
 * drives the debuggee through setBreakpoints / continue / next commands.
 *
 * Breakpoints are stored as a linear array keyed by (file_id, line). A
 * small capped array is fine since human-set breakpoints don't scale
 * past a few dozen.
 *
 * Frames are tracked by matlab_dbg_enter_frame / _leave_frame so the
 * DAP server can return a multi-entry stackTrace. When -g is on, every
 * emitted user-function body calls enter on entry and leave before
 * each return.
 */
#define MATLAB_DBG_MAX_BREAKPOINTS 256
#define MATLAB_DBG_MAX_FRAMES 128
/* Per-frame Locals: bounded at lowering time by how many distinct
 * named slots a single user function can carry. 64 is well above what
 * any of our examples currently produce; bump if needed. */
#define MATLAB_DBG_MAX_LOCALS 64

/* Per-frame mini-workspace entry. Mirrors the shape of the script-
 * level matlab_ws struct but is keyed by frame index so the DAP
 * server can pick the right slice for the user's selected frame.
 * `kind` follows the same convention as matlab_dbg_ws_kind:
 *   0 = f64 scalar
 *   1 = matlab_mat * (numeric matrix descriptor)
 *   2 = matlab_obj * (user classdef instance — `ptr` is the obj,
 *       its class_id field doubles as the registry key for class
 *       names). The matrix / object pointers are borrowed from the
 *       JIT's slot — the slot is alive for the lifetime of the
 *       frame, which is exactly when the DAP server reads from us. */
struct matlab_dbg_local {
    char *name;       /* heap-copied, null-terminated */
    int64_t name_len;
    int kind;         /* 0 = f64, 1 = matrix ptr, 2 = obj ptr */
    double f64;
    void *ptr;
};

struct matlab_dbg_frame_locals {
    int n;
    struct matlab_dbg_local entries[MATLAB_DBG_MAX_LOCALS];
};

enum matlab_dbg_action {
    MATLAB_DBG_RUN       = 0,   /* no pause (no breakpoints hit) */
    MATLAB_DBG_CONTINUE  = 1,   /* resume from a pause */
    MATLAB_DBG_STEP_OVER = 2,   /* break at next statement at <= target depth */
    MATLAB_DBG_STEP_IN   = 3,   /* break at the very next statement */
    MATLAB_DBG_STEP_OUT  = 4,   /* break at next statement at <  target depth */
    MATLAB_DBG_STOP      = 5,   /* terminate the program */
};

struct matlab_dbg_frame {
    int32_t file_id;
    int32_t line;
    const char *fn_name;
};

/* Record kinds for the reverse-stepping undo log. matlab_dbg_state
 * holds a fixed-size ring buffer of these; matlab_dbg_step_back
 * walks them in reverse to revert variable writes.
 *
 *   0 = statement boundary {file_id, line, thread_slot}
 *   1 = ws_set_f64 {name, prev_kind, prev_f64, prev_existed}
 *   2 = ws_set_mat / ws_set_obj {name, prev_kind, prev_ptr,
 *       prev_existed}
 *   3 = frame_set_* {thread_slot, frame_idx, name, prev_kind,
 *       prev_f64, prev_ptr, prev_existed}
 *   4 = irreversible-op marker (disp / fprintf etc.) — stepBack
 *       refuses to walk past one of these. */
struct matlab_dbg_undo_rec {
    int8_t kind;
    int8_t prev_kind;
    int8_t prev_existed;
    int8_t _pad;
    int32_t file_id;
    int32_t line;
    int32_t frame_idx;
    int32_t thread_slot;
    char *name;        /* heap-owned for kinds 1/2/3 */
    int64_t name_len;
    double prev_f64;
    void *prev_ptr;
};

struct matlab_dbg_state {
    int enabled;
    int stop_on_entry;
    pthread_mutex_t mu;
    pthread_cond_t cv_client;   /* debugger thread waits on this when paused */
    pthread_cond_t cv_server;   /* server waits on this when requesting pause */

    /* Last-hit pause point, published after the hook blocks. */
    int paused;
    int32_t cur_file_id;
    int32_t cur_line;
    /* Index into bp_* of the breakpoint that triggered the current
     * pause. -1 when the pause came from stepping rather than a bp.
     * The DAP server reads cond_text[cur_bp_idx] / log_text[...] to
     * decide whether to evaluate before notifying the IDE. */
    int cur_bp_idx;

    /* What to do when resumed. */
    enum matlab_dbg_action action;
    int32_t step_target_depth;

    /* Exception-breakpoint filter: when set, the hook pauses on the
     * first statement after matlab_set_error fires. Toggled by the
     * DAP server's `setExceptionBreakpoints` handler in response to
     * the IDE's "Pause on Errors" UI. */
    int pause_on_error;

    /* Set non-zero when the current pause was triggered by a
     * `keyboard` builtin call (not a step / bp / error). The DAP
     * server reads this in monitorMain to surface a stop reason of
     * "entry" so the IDE renders the keyboard glyph rather than a
     * generic step/pause. Cleared by the next resume. */
    int paused_from_keyboard;

    /* Data breakpoints (write watchpoints). The DAP server adds an
     * entry via matlab_dbg_add_watchpoint; the runtime's set_*
     * functions check the table after every workspace / frame-local
     * write and trip a pause if the name matches.
     *
     * Scope encoding:
     *   0 = "any" (matches script ws *or* any frame)
     *   1 = script workspace only (matlab_ws_set_*)
     *   2 = innermost frame only (matlab_dbg_frame_set_*)
     * v1 ships scope=0 since the DAP IDE picks the watch from the
     * Variables panel and the user expects "stop when this name
     * gets reassigned anywhere"; tighter scoping can layer on later
     * via the dataBreakpointInfo `accessType` argument.
     *
     * `last_writer_idx` is set by the set_* sites when a watchpoint
     * trips, mirroring how cur_bp_idx works for line breakpoints —
     * the DAP server reads it to surface the originating watch's
     * id in the stopped event's hitBreakpointIds. */
    int n_wp;
    char *wp_name[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t wp_name_len[MATLAB_DBG_MAX_BREAKPOINTS];
    int32_t wp_scope[MATLAB_DBG_MAX_BREAKPOINTS];
    int32_t wp_id[MATLAB_DBG_MAX_BREAKPOINTS];   /* DAP-assigned id */
    /* Access kind: 0 = write only (default; matches the original
     * watch-on-set behaviour), 1 = read only, 2 = read+write.
     * The check helpers below filter by this so a read-only watch
     * doesn't trip on a regular `matlab_ws_set_*`. */
    int8_t  wp_access[MATLAB_DBG_MAX_BREAKPOINTS];
    int last_wp_idx;   /* index of the watchpoint that tripped, or -1 */
    int paused_from_watch;

    /* Breakpoints (file_id, line) — linear scan. cond_text and
     * log_text are heap-owned (NULL when absent). cond_disabled flips
     * to 1 once the DAP server reports a condition syntax error so
     * subsequent hits don't keep retrying it.
     *
     * Hit-count gating: hit_count counts every time the hook reaches
     * this bp's line (incremented unconditionally on a match);
     * hit_op + hit_target encode the user's `hitCondition` (e.g.
     * `>= 100` is op=GE, target=100). The hook compares count vs.
     * target with op — only triggers a pause when the test passes.
     * op=0 means no hit-count gate (default; the bp pauses every
     * time the line runs). */
    int n_bp;
    int32_t bp_file[MATLAB_DBG_MAX_BREAKPOINTS];
    int32_t bp_line[MATLAB_DBG_MAX_BREAKPOINTS];
    char *cond_text[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t cond_len[MATLAB_DBG_MAX_BREAKPOINTS];
    char *log_text[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t log_len[MATLAB_DBG_MAX_BREAKPOINTS];
    int cond_disabled[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t hit_count[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t hit_target[MATLAB_DBG_MAX_BREAKPOINTS];
    /* hit_op encoding (0 = none, no gate):
     *   1 = ==   (stop on the Nth hit only)
     *   2 = >=   (stop on hit N and every hit after — most common)
     *   3 = >    (stop after N hits)
     *   4 = %    (stop every Nth hit, e.g. `%5` for every 5th iter)
     * Anything else is treated as no gate. */
    int hit_op[MATLAB_DBG_MAX_BREAKPOINTS];

    /* Frame stack. The shared (cross-thread) frames[] array stays here
     * for the legacy single-threaded DAP path, but per-thread frame
     * chains (see thread_frames[] below) now own the source of
     * truth in multi-threaded sessions.
     *
     * On first hook fire from a given thread, n_frames is copied into
     * the thread's per-thread slot from this template. Subsequent
     * enter_frame / leave_frame / frame_set_* mutate the thread's
     * own slot. The shared frames[] is updated in lockstep with the
     * paused-thread's chain so DAP inspectors that read frames[]
     * directly (the legacy code paths) keep working — the
     * paused-thread's view is what's exposed. */
    int n_frames;
    struct matlab_dbg_frame frames[MATLAB_DBG_MAX_FRAMES];
    /* Per-frame Locals. Index aligns with `frames[]`: frame 0 is the
     * script's mini-ws (parallel to matlab_ws but populated by the
     * lowering's mirror calls — covers loop induction variables and
     * other slot-stored vars that don't go through matlab_ws_set_*).
     * Frames 1..n-1 are user-function frames. Cleared on enter, freed
     * on leave. */
    struct matlab_dbg_frame_locals frame_locals[MATLAB_DBG_MAX_FRAMES];

    /* Per-thread frame chain. `thread_keys[i]` is the pthread_t;
     * `thread_n_frames[i]` / `thread_frames[i][]` /
     * `thread_frame_locals[i][]` is that thread's own call-stack
     * state. The thread registry (thread_keys / thread_ids /
     * n_threads) above is the index. The hook reads/writes the
     * calling thread's slot; the DAP inspector functions
     * (frame_count / frame_at / frame_local_*) take an implicit
     * thread idx via paused_thread_idx so the pause is reported
     * against the right call stack.
     *
     * Capacity-32 matches the registry; per-thread MAX_FRAMES is
     * the same as the shared frames[]. The memory cost is bounded
     * (~32 * 32 * sizeof(matlab_dbg_frame)) so we just inline. */
    int                          thread_n_frames[32];
    struct matlab_dbg_frame      thread_frames[32][MATLAB_DBG_MAX_FRAMES];
    struct matlab_dbg_frame_locals thread_frame_locals[32][MATLAB_DBG_MAX_FRAMES];
    int32_t                      thread_step_target_depth[32];

    /* Reverse-stepping undo log. Members declared inline; the
     * matlab_dbg_undo_rec struct itself is at file scope (above
     * matlab_dbg_state). */
    int n_undo;
    int undo_head;     /* next slot to write (ring buffer head) */
    int undo_full;     /* set once we've wrapped — gates how far we can rewind */
    struct matlab_dbg_undo_rec undo_log[4096];
    /* Recording flag — clear during the rewind itself so
     * apply-undo's reverse-set doesn't push a meta-record. */
    int recording_undo;

    /* File-id <-> name table. Populated by matlab_dbg_register_file. */
    int n_files;
    const char *file_names[256];
    int64_t file_name_lens[256];

    /* Thread registry. Populated lazily on first hook entry from
     * each pthread that runs JIT'd code (the main worker plus any
     * parfor-spawned workers). The DAP server's `threads` request
     * enumerates this list; `stopped` events name the originating
     * thread by id.
     *
     * Identity is the pthread_t value; the `id` we hand to the DAP
     * client is a sequential integer (1 = main worker; 2..N =
     * parfor workers in spawn order). The mapping is one-shot per
     * thread and persists for the rest of the session — even if a
     * thread is joined, we keep its slot so any earlier `stopped`
     * event id stays valid in the IDE's UI history.
     *
     * Limitation (v1): the frame stack itself (`frames[]` /
     * `frame_locals[]` / `n_frames` above) is shared across all
     * threads. A parfor body that hits a bp will surface the right
     * thread id in the stopped event, but `stackTrace(threadId)`
     * returns whatever the last-modifying thread put on the global
     * stack — which can be the queried thread or a sibling. Per-
     * thread frame stacks are the follow-up; documented in
     * docs/debug.md. */
    int n_threads;
    pthread_t thread_keys[32];
    int32_t  thread_ids[32];   /* sequential, 1-based; matches DAP threadId */
    /* Index into thread_keys/_ids of the thread that hit the
     * current pause, or -1 when no pause is active. Set by the
     * hook when should_pause flips on; cleared on resume. The
     * DAP server reads it via matlab_dbg_paused_thread_id() to
     * surface the originating thread on `stopped` events. */
    int paused_thread_idx;

    /* Class-id -> class-name table. Populated by
     * matlab_dbg_register_class at the top of the script body when -g
     * is on (one entry per classdef in the translation unit). The DAP
     * server uses this to surface a class instance as
     * `1x1 ClassName` in the LOCALS panel and in the watch box.
     * 64 is far above what any realistic program touches; a linear
     * scan is cheap given how rarely these are read.
     *
     * `class_names[i]` is heap-copied on register and never freed —
     * the registration is once-per-program and the strings are tiny. */
    int n_classes;
    int32_t class_ids[64];
    char *class_names[64];
    int64_t class_name_lens[64];
};

static struct matlab_dbg_state matlab_dbg = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv_client = PTHREAD_COND_INITIALIZER,
    .cv_server = PTHREAD_COND_INITIALIZER,
    .action = MATLAB_DBG_RUN,
};

/* Forward decls for the per-thread chain helpers — definitions
 * live further down alongside enter_frame. Multiple call sites
 * up here (err_snapshot_frames, watch_trip, keyboard_hook) need
 * to consult the per-thread chain before its definition appears,
 * so they're declared at the top of the matlab_dbg section.
 * Same shape for the undo-log helpers used by enable() and the
 * matlab_ws_set_* / matlab_dbg_frame_set_* call sites further
 * up the file. */
static int matlab_dbg_thread_slot_locked(void);
static int matlab_dbg_thread_init_chain_locked(void);
static void matlab_dbg_undo_clear_locked(void);

/* Forward decl: defined alongside matlab_dbg_enter_frame below but
 * called from matlab_dbg_enable to clear any frame-locals state left
 * over from a prior launch. */
static void matlab_dbg_free_frame_locals(int frame_idx);

/* --- error() backtrace snapshot --------------------------------------
 *
 * matlab_set_error / matlab_set_error_msg snapshot the current frame
 * stack here BEFORE any unwind pops the runtime frames. Without the
 * snapshot, by the time the script returns to the DAP server (or a
 * `disp(ME.message)` runs in a catch body) the leave_frame calls
 * fired on each function return have erased the call site that
 * threw, leaving us with nothing useful to print.
 *
 * The snapshot is intentionally a value-copy (file_id, line, name
 * pointer). The fn_name pointers stored in matlab_dbg.frames[].fn_name
 * are runtime-owned (either string literals from the JIT'd const
 * globals, or "<script>" itself), so copying the pointer is safe —
 * they outlive the snapshot.
 *
 * `matlab_err_emit_traceback_to_stderr` prints the snapshot to stderr
 * with the format:
 *
 *   error: <msg>
 *     at <fn> (<file>:<line>)
 *     at <fn> (<file>:<line>)
 *
 * Gated on matlab_dbg.enabled so that non-debug binaries (the
 * production -emit-c / -emit-cpp / -emit-llvm path with no -dap)
 * keep their existing silent semantics — only DAP / `-g` runs see
 * the diagnostic. */
static int matlab_err_n_frames = 0;
static struct matlab_dbg_frame matlab_err_frames[MATLAB_DBG_MAX_FRAMES];

/* Forward decl — defined later in the file. The error snapshot
 * path and the hook both need it, but its body sits next to the
 * other thread-chain helpers far below. */
static int matlab_dbg_thread_init_chain_locked(void);

static void matlab_err_snapshot_frames(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Free any names retained from a previous error snapshot before
     * stamping new ones in — otherwise repeated error() calls leak. */
    for (int i = 0; i < matlab_err_n_frames; ++i) {
        free((char *)matlab_err_frames[i].fn_name);
        matlab_err_frames[i].fn_name = NULL;
    }
    /* Snapshot the calling thread's per-thread chain (post-refactor
     * source of truth for frame state). The shared frames[] is now
     * a paused-thread snapshot, refreshed only on hook pause; an
     * error fired between pauses sees stale data there. The
     * per-thread chain is always up-to-date because every
     * enter_frame / leave_frame / hook fire from this thread
     * touched it directly. */
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    if (n > MATLAB_DBG_MAX_FRAMES) n = MATLAB_DBG_MAX_FRAMES;
    for (int i = 0; i < n; ++i) {
        matlab_err_frames[i].file_id =
            matlab_dbg.thread_frames[slot][i].file_id;
        matlab_err_frames[i].line =
            matlab_dbg.thread_frames[slot][i].line;
        const char *src = matlab_dbg.thread_frames[slot][i].fn_name;
        if (src) {
            size_t L = strlen(src);
            char *copy = (char *)malloc(L + 1);
            if (copy) { memcpy(copy, src, L); copy[L] = '\0'; }
            matlab_err_frames[i].fn_name = copy;
        } else {
            matlab_err_frames[i].fn_name = NULL;
        }
    }
    matlab_err_n_frames = n;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Resolve a file_id back to its registered name. Mirrors
 * matlab_dbg_file_name but is callable without holding the dbg mutex
 * (the caller already takes care of synchronization). Returns
 * "<unknown>" when the file_id is out of range. */
static const char *matlab_err_file_name_locked(int32_t file_id, int64_t *len_out) {
    int max = (int)(sizeof matlab_dbg.file_names /
                    sizeof matlab_dbg.file_names[0]);
    if (file_id >= 1 && file_id <= max) {
        const char *name = matlab_dbg.file_names[file_id - 1];
        if (name) {
            if (len_out) *len_out = matlab_dbg.file_name_lens[file_id - 1];
            return name;
        }
    }
    static const char unknown[] = "<unknown>";
    if (len_out) *len_out = (int64_t)(sizeof unknown - 1);
    return unknown;
}

static void matlab_err_emit_traceback_to_stderr(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int debug_on = matlab_dbg.enabled;
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (!debug_on) return;

    /* Build the whole traceback into a fixed-size buffer and emit it
     * via a single write(2) call. We can't use fprintf here because
     * libc's <stdio.h> file lock can deadlock if the worker thread
     * happens to hold a recursive_mutex inside LLVM's ExecutionEngine
     * at the point error() fires — observed during DAP shutdown when
     * stderr-bound fprintf races with the engine's own diagnostic
     * stream. write(2) bypasses all of that. */
    char buf[2048];
    size_t off = 0;
    #define APP_LIT(s) do { \
        size_t l = sizeof(s) - 1; \
        if (off + l > sizeof buf) l = sizeof buf - off; \
        memcpy(buf + off, s, l); \
        off += l; \
    } while (0)
    #define APP(fmt, ...) do { \
        if (off < sizeof buf) { \
            int n = snprintf(buf + off, sizeof buf - off, fmt, __VA_ARGS__); \
            if (n > 0) off += (size_t)n > sizeof buf - off \
                              ? sizeof buf - off : (size_t)n; \
        } \
    } while (0)

    APP_LIT("error: ");
    if (matlab_error_msg_len > 0) {
        size_t mlen = (size_t)matlab_error_msg_len;
        if (off + mlen > sizeof buf) mlen = sizeof buf - off;
        memcpy(buf + off, matlab_error_msg, mlen);
        off += mlen;
    }
    APP_LIT("\n");

    pthread_mutex_lock(&matlab_dbg.mu);
    for (int idx = matlab_err_n_frames - 1; idx >= 0; --idx) {
        const struct matlab_dbg_frame *f = &matlab_err_frames[idx];
        const char *fn = f->fn_name ? f->fn_name : "<frame>";
        int64_t fnLen = 0;
        const char *file = matlab_err_file_name_locked(f->file_id, &fnLen);
        APP("  at %s (%.*s:%d)\n", fn, (int)fnLen, file, f->line);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    #undef APP

    (void)!write(STDERR_FILENO, buf, off);
}

/* Public read-only accessors so the DAP server (or a future REPL UI)
 * can render the same backtrace as a structured response. */
int matlab_err_traceback_count(void) {
    int n;
    pthread_mutex_lock(&matlab_dbg.mu);
    n = matlab_err_n_frames;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

int matlab_err_traceback_at(int i, int32_t *file_id, int32_t *line,
                             const char **fn_name) {
    int ok = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    /* i = 0 = innermost; mirrors matlab_dbg_frame_at's API shape. */
    int idx = matlab_err_n_frames - 1 - i;
    if (idx >= 0 && idx < matlab_err_n_frames) {
        if (file_id) *file_id = matlab_err_frames[idx].file_id;
        if (line)    *line    = matlab_err_frames[idx].line;
        if (fn_name) *fn_name = matlab_err_frames[idx].fn_name;
        ok = 1;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* DAP `setExceptionBreakpoints` plumbing — toggle the pause-on-error
 * filter the hook checks above. Held under matlab_dbg.mu so a flip
 * mid-eval doesn't race the hook's read of the same field. */
void matlab_dbg_set_pause_on_error(int on) {
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.pause_on_error = on ? 1 : 0;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP `exceptionInfo` reader — surfaces the message captured by
 * matlab_set_error_msg before the unwind. Returns NULL/0 when no
 * error has fired this session. The buffer is owned by the runtime
 * (static char[1024], null-terminated); the caller must not free it. */
const char *matlab_dbg_last_error_msg(int64_t *len_out) {
    if (len_out) *len_out = matlab_error_msg_len;
    return matlab_error_msg_len > 0 ? matlab_error_msg : NULL;
}

/* Lowered call site for a `keyboard` builtin in user code. Sets
 * paused=1 and blocks on the same condvar a real breakpoint uses,
 * so the DAP server's monitor thread wakes and emits a `stopped`
 * event. The `paused_from_keyboard` flag tells monitorMain to
 * surface stop reason="entry" instead of "step" — the IDE then
 * renders the keyboard / pause-on-source glyph rather than a
 * generic step icon.
 *
 * No-op when matlab_dbg.enabled == 0 (release builds without -g):
 * a `keyboard` call simply returns immediately. The latest source
 * location is left as whatever the most recent matlab_dbg_hook
 * recorded — already what the user wants for the call site
 * because the hook fires at the same statement. */
/* Forward decl — defined alongside the thread enumeration helpers
 * further down. The keyboard / watch trip code below needs it. */

void matlab_dbg_keyboard_hook(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (!matlab_dbg.enabled) {
        pthread_mutex_unlock(&matlab_dbg.mu);
        return;
    }
    int thr_idx = matlab_dbg_thread_init_chain_locked();
    /* Copy the innermost frame's (file_id, line) into the cur_*
     * fields so the DAP `stopped` event reports the keyboard call
     * site, then snapshot the calling thread's per-thread chain
     * into the shared frames[] view so DAP inspectors that read
     * frames[]/frame_locals[] directly see the caller's stack
     * (not whatever the last paused thread left). */
    int n_thr = matlab_dbg.thread_n_frames[thr_idx];
    if (n_thr > 0) {
        matlab_dbg.cur_file_id =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].file_id;
        matlab_dbg.cur_line =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].line;
    }
    int snap_n = n_thr > MATLAB_DBG_MAX_FRAMES ? MATLAB_DBG_MAX_FRAMES : n_thr;
    matlab_dbg.n_frames = snap_n;
    for (int i = 0; i < snap_n; ++i) {
        matlab_dbg.frames[i] = matlab_dbg.thread_frames[thr_idx][i];
        matlab_dbg.frame_locals[i] =
            matlab_dbg.thread_frame_locals[thr_idx][i];
    }
    matlab_dbg.cur_bp_idx = -1;
    matlab_dbg.paused = 1;
    matlab_dbg.paused_from_keyboard = 1;
    matlab_dbg.paused_thread_idx = thr_idx;
    pthread_cond_broadcast(&matlab_dbg.cv_server);
    while (matlab_dbg.paused) {
        pthread_cond_wait(&matlab_dbg.cv_client, &matlab_dbg.mu);
    }
    matlab_dbg.paused_from_keyboard = 0;
    matlab_dbg.paused_thread_idx = -1;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP-side reader: was the most recent pause triggered by a
 * keyboard() call? monitorMain checks this before mapping
 * (BpIdx == -1) to reason="step", switching to "entry" instead
 * when this flag is set. */
int matlab_dbg_was_paused_from_keyboard(void) {
    int v;
    pthread_mutex_lock(&matlab_dbg.mu);
    v = matlab_dbg.paused_from_keyboard;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return v;
}

/* --- Data breakpoints (write watchpoints) --------------------------- */

/* Add a watchpoint by name with caller-assigned id (the DAP server
 * encodes its dataId from the name's hash so subsequent setBreakpoints
 * round-trips reuse the same id). scope is 0 (any) / 1 (script-ws) /
 * 2 (innermost-frame). Returns 1 on success, 0 on table-full or
 * duplicate. The runtime owns the heap-copy of `name`. */
/* Forward decl — _ex body follows immediately, but the back-compat
 * shim above forwards into it. */
int matlab_dbg_add_watchpoint_ex(const char *name, int64_t name_len,
                                  int32_t scope, int32_t id,
                                  int32_t access);

int matlab_dbg_add_watchpoint(const char *name, int64_t name_len,
                               int32_t scope, int32_t id) {
    /* Backward-compat shim — defaults to write-only (the original
     * accessType v1 supported). New callers should use the _ex
     * variant below. */
    return matlab_dbg_add_watchpoint_ex(name, name_len, scope, id,
                                         /*access=*/0);
}

int matlab_dbg_add_watchpoint_ex(const char *name, int64_t name_len,
                                  int32_t scope, int32_t id,
                                  int32_t access) {
    if (!name || name_len <= 0) return 0;
    if (access < 0 || access > 2) access = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    /* De-dup: if a watch with the same id already exists, refresh
     * its scope+access rather than appending a duplicate row. */
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        if (matlab_dbg.wp_id[i] == id) {
            matlab_dbg.wp_scope[i] = scope;
            matlab_dbg.wp_access[i] = (int8_t)access;
            pthread_mutex_unlock(&matlab_dbg.mu);
            return 1;
        }
    }
    int ok = matlab_dbg.n_wp < MATLAB_DBG_MAX_BREAKPOINTS;
    if (ok) {
        int i = matlab_dbg.n_wp;
        matlab_dbg.wp_name[i] = (char *)malloc((size_t)name_len + 1);
        memcpy(matlab_dbg.wp_name[i], name, (size_t)name_len);
        matlab_dbg.wp_name[i][name_len] = '\0';
        matlab_dbg.wp_name_len[i] = name_len;
        matlab_dbg.wp_scope[i] = scope;
        matlab_dbg.wp_id[i] = id;
        matlab_dbg.wp_access[i] = (int8_t)access;
        matlab_dbg.n_wp++;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Wipe the entire watchpoint table. The DAP `setDataBreakpoints`
 * request passes a fresh full list each time (same semantics as
 * setBreakpoints), so the cleanest implementation is clear-then-add.
 * Keeps the per-call code in the DAP handler simple. */
void matlab_dbg_clear_watchpoints(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        free(matlab_dbg.wp_name[i]);
        matlab_dbg.wp_name[i] = NULL;
        matlab_dbg.wp_name_len[i] = 0;
        matlab_dbg.wp_scope[i] = 0;
        matlab_dbg.wp_id[i] = 0;
        matlab_dbg.wp_access[i] = 0;
    }
    matlab_dbg.n_wp = 0;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP-side reader for the stopped-event handler: returns the id of
 * the most recent tripped watchpoint, or 0 if no watch has tripped
 * since the last resume. Cleared on resume by the worker's hook. */
int32_t matlab_dbg_last_watchpoint_id(void) {
    int32_t id = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.last_wp_idx >= 0 &&
        matlab_dbg.last_wp_idx < matlab_dbg.n_wp)
        id = matlab_dbg.wp_id[matlab_dbg.last_wp_idx];
    pthread_mutex_unlock(&matlab_dbg.mu);
    return id;
}

/* "Was the most recent pause caused by a tripped watchpoint?" — same
 * shape as matlab_dbg_was_paused_from_keyboard. The monitor checks
 * this when mapping BpIdx==-1 to a stop reason; "data breakpoint" is
 * the DAP standard reason for watchpoint hits. */
int matlab_dbg_was_paused_from_watch(void) {
    int v;
    pthread_mutex_lock(&matlab_dbg.mu);
    v = matlab_dbg.paused_from_watch;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return v;
}

/* Internal: scan the watchpoint table for a name match. scope_hint
 * is the call site's scope (1 = script-ws, 2 = frame-set); a watch
 * with scope=0 (any) matches both. Returns the index of the matching
 * watch, or -1 on miss. CALLER MUST HOLD matlab_dbg.mu — this is
 * called from inside the set_* lock-region. */
static int matlab_dbg_watch_check(const char *name, int64_t name_len,
                                   int32_t scope_hint) {
    /* Write-path: skip read-only watches. access==0 (write) and
     * access==2 (readWrite) qualify; access==1 (read-only) does
     * not. */
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        if (matlab_dbg.wp_name_len[i] != name_len) continue;
        int32_t s = matlab_dbg.wp_scope[i];
        if (s != 0 && s != scope_hint) continue;
        if (matlab_dbg.wp_access[i] == 1) continue;  /* read-only */
        if (memcmp(matlab_dbg.wp_name[i], name, (size_t)name_len) == 0)
            return i;
    }
    return -1;
}

/* Read-path counterpart. Called from matlab_ws_get_*; only matches
 * watches whose access kind includes "read" (1 or 2). Same scope-
 * filter shape as the write check. CALLER MUST HOLD matlab_dbg.mu. */
static int matlab_dbg_watch_check_read(const char *name, int64_t name_len,
                                        int32_t scope_hint) {
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        if (matlab_dbg.wp_name_len[i] != name_len) continue;
        int32_t s = matlab_dbg.wp_scope[i];
        if (s != 0 && s != scope_hint) continue;
        int8_t a = matlab_dbg.wp_access[i];
        if (a != 1 && a != 2) continue;  /* write-only — skip */
        if (memcmp(matlab_dbg.wp_name[i], name, (size_t)name_len) == 0)
            return i;
    }
    return -1;
}

/* Internal: trip a watchpoint. Sets the paused-from-watch flag plus
 * cur_* fields and blocks on the same condvar a real bp uses, so
 * the DAP monitor wakes and emits a `stopped` event. Same pattern
 * as matlab_dbg_keyboard_hook. CALLER MUST HOLD matlab_dbg.mu. */
/* --- Reverse stepping (undo log) ---------------------------------- */

/* Append a record to the ring buffer, evicting the oldest if full.
 * For kinds 1/2/3 the heap-owned `name` of the evicted record is
 * freed. CALLER MUST HOLD matlab_dbg.mu. */
static struct matlab_dbg_undo_rec *matlab_dbg_undo_alloc_locked(void) {
    int slot = matlab_dbg.undo_head;
    struct matlab_dbg_undo_rec *r = &matlab_dbg.undo_log[slot];
    /* Evict the previous tenant's heap allocation. */
    if (matlab_dbg.undo_full && r->name) {
        free(r->name);
        r->name = NULL;
    }
    matlab_dbg.undo_head = (slot + 1) % 4096;
    if (matlab_dbg.undo_head == 0) matlab_dbg.undo_full = 1;
    if (!matlab_dbg.undo_full) matlab_dbg.n_undo = matlab_dbg.undo_head;
    else matlab_dbg.n_undo = 4096;
    /* Reset to a clean record. */
    memset(r, 0, sizeof *r);
    return r;
}

/* Clear the entire undo log — called on enable() so re-launches
 * start fresh, and after a successful rewind so a subsequent
 * forward-step's writes don't conflate with stale undo records. */
static void matlab_dbg_undo_clear_locked(void) {
    int n = matlab_dbg.undo_full ? 4096 : matlab_dbg.undo_head;
    for (int i = 0; i < n; ++i) {
        free(matlab_dbg.undo_log[i].name);
        matlab_dbg.undo_log[i].name = NULL;
    }
    matlab_dbg.undo_head = 0;
    matlab_dbg.undo_full = 0;
    matlab_dbg.n_undo = 0;
}

/* Stamp a statement-boundary record. The hook calls this on every
 * fire so stepBack knows where each statement began. */
static void matlab_dbg_undo_record_stmt_locked(int32_t file_id,
                                                int32_t line,
                                                int32_t thread_slot) {
    if (!matlab_dbg.recording_undo) return;
    struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
    r->kind = 0;
    r->file_id = file_id;
    r->line = line;
    r->thread_slot = thread_slot;
}

/* Stamp an irreversible-op marker. The set_error path and the
 * disp/fprintf JIT entries call this so a stepBack that reaches
 * the marker stops with a clear message instead of silently
 * walking past a printed line. */
void matlab_dbg_undo_record_irreversible(const char *reason) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.recording_undo) {
        struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
        r->kind = 4;
        if (reason) {
            int64_t L = (int64_t)strlen(reason);
            r->name = (char *)malloc((size_t)L + 1);
            if (r->name) {
                memcpy(r->name, reason, (size_t)L + 1);
                r->name_len = L;
            }
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Capture current value of `name` from matlab_ws before it gets
 * overwritten. Returns prev_kind (-1 if missing), prev_f64,
 * prev_ptr. Used by the undo path on every ws_set_*. CALLER MUST
 * HOLD matlab_dbg.mu — matlab_struct accesses are themselves
 * lock-free, but we want the snapshot atomic w.r.t. the upcoming
 * write. */
static void matlab_ws_capture_prior(const char *name, int64_t len,
                                     int8_t *out_kind, int8_t *out_existed,
                                     double *out_f64, void **out_ptr) {
    *out_kind = -1; *out_existed = 0; *out_f64 = 0.0; *out_ptr = NULL;
    if (!matlab_ws) return;
    /* Walk the matlab_struct in-place — no public accessor returns
     * the kind alongside the value efficiently, so reach into the
     * struct directly. */
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        int nl = (int)strlen(matlab_ws->names[i]);
        if (nl == (int)len &&
            memcmp(matlab_ws->names[i], name, (size_t)len) == 0) {
            *out_existed = 1;
            *out_kind = (int8_t)matlab_ws->kinds[i];
            *out_f64 = matlab_ws->f64_vals[i];
            *out_ptr = matlab_ws->ptr_vals[i];
            return;
        }
    }
}

/* Push a ws_set undo record. Takes the dbg mutex itself — called
 * from the matlab_ws_set_* sites which can't see matlab_dbg.mu
 * directly (the static variable is defined further down in this
 * TU). The fast path is a single n_undo / recording_undo check
 * inside the lock; if recording is off (no DAP session), the
 * function returns immediately. */
static void matlab_ws_push_undo(const char *name, int64_t len,
                                 int kind_being_written) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (!matlab_dbg.recording_undo) {
        pthread_mutex_unlock(&matlab_dbg.mu);
        return;
    }
    int8_t prev_kind, prev_existed;
    double prev_f64;
    void *prev_ptr;
    matlab_ws_capture_prior(name, len, &prev_kind, &prev_existed,
                             &prev_f64, &prev_ptr);
    struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
    /* Kind 1 for f64 writes, 2 for mat/obj writes — the rewind
     * path uses this to pick the right matlab_ws_set_* on undo. */
    r->kind = (kind_being_written == 0) ? 1 : 2;
    r->prev_kind = prev_kind;
    r->prev_existed = prev_existed;
    r->prev_f64 = prev_f64;
    r->prev_ptr = prev_ptr;
    r->name = (char *)malloc((size_t)len + 1);
    if (r->name) {
        memcpy(r->name, name, (size_t)len);
        r->name[len] = '\0';
        r->name_len = len;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Frame-local undo: capture prior entry from the named frame
 * (innermost of the calling thread). Same shape as the ws helper
 * but operates on thread_frame_locals. CALLER HOLDS matlab_dbg.mu. */
static void matlab_dbg_frame_push_undo_locked(int thread_slot,
                                                int frame_idx,
                                                const char *name,
                                                int64_t len) {
    if (!matlab_dbg.recording_undo) return;
    struct matlab_dbg_frame_locals *fl =
        &matlab_dbg.thread_frame_locals[thread_slot][frame_idx];
    int8_t prev_kind = -1, prev_existed = 0;
    double prev_f64 = 0.0;
    void *prev_ptr = NULL;
    for (int i = 0; i < fl->n; ++i) {
        if (fl->entries[i].name_len == len &&
            memcmp(fl->entries[i].name, name, (size_t)len) == 0) {
            prev_existed = 1;
            prev_kind = (int8_t)fl->entries[i].kind;
            prev_f64 = fl->entries[i].f64;
            prev_ptr = fl->entries[i].ptr;
            break;
        }
    }
    struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
    r->kind = 3;
    r->thread_slot = thread_slot;
    r->frame_idx = frame_idx;
    r->prev_kind = prev_kind;
    r->prev_existed = prev_existed;
    r->prev_f64 = prev_f64;
    r->prev_ptr = prev_ptr;
    r->name = (char *)malloc((size_t)len + 1);
    if (r->name) {
        memcpy(r->name, name, (size_t)len);
        r->name[len] = '\0';
        r->name_len = len;
    }
}

/* Rewind one statement: pop undo records starting from undo_head
 * back until a kind=0 (statement boundary) is found, applying each
 * write in reverse order. Stops at an irreversible-op marker
 * (kind=4) without rewinding past it. Returns the line number to
 * resume at, or 0 if the log is exhausted / the next record is an
 * irreversible op.
 *
 * After rewinding, the next forward-step will re-execute the
 * statement we just rolled back. The undo log itself is NOT
 * cleared — the records we just popped are gone (head moved
 * back), but anything older stays in case the user wants to
 * stepBack again. */
int matlab_dbg_step_back(int32_t *out_file_id, int32_t *out_line,
                         char *out_msg, int64_t msg_cap) {
    if (out_file_id) *out_file_id = 0;
    if (out_line) *out_line = 0;
    if (out_msg && msg_cap > 0) out_msg[0] = '\0';
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.n_undo == 0) {
        if (out_msg && msg_cap > 0)
            snprintf(out_msg, (size_t)msg_cap, "undo log is empty");
        pthread_mutex_unlock(&matlab_dbg.mu);
        return 0;
    }
    /* Algorithm:
     *   1. Drop the head boundary record (the current "now" marker
     *      from the most recent hook fire — represents the line the
     *      worker is paused at).
     *   2. Walk back, applying each non-boundary record's revert.
     *   3. Stop at the next boundary; that's the new "now". The
     *      boundary itself stays in the log (head = boundary_idx +
     *      1) so subsequent stepBacks see it as their head and drop
     *      it on entry.
     *
     * `head` points one PAST the last written entry, so the actual
     * top is at head-1. Wrap negative-mod the same way the ring
     * buffer wraps positive. */
    int idx = matlab_dbg.undo_head;
    int wall = matlab_dbg.undo_full ? 4096 : matlab_dbg.undo_head;
    int popped = 0;
    /* Disable recording so any matlab_struct_* writes we issue here
     * to roll values back don't push fresh undo records. */
    matlab_dbg.recording_undo = 0;
    int hit_boundary = 0;
    int hit_irreversible = 0;
    int32_t boundary_file_id = 0, boundary_line = 0;
    int boundary_idx = -1;

    /* Step 1: drop the current "now" boundary. The hook stamps a
     * boundary on every fire, so head-1 IS that stamp when paused
     * at a fresh statement (the typical case). If the top isn't a
     * boundary — e.g. the bp fired mid-statement after some
     * writes — we keep walking; the writes get reverted normally
     * and the boundary we hit is the previous statement's. */
    {
        int peek = (idx == 0) ? 4096 - 1 : idx - 1;
        struct matlab_dbg_undo_rec *r = &matlab_dbg.undo_log[peek];
        if (r->kind == 0) {
            idx = peek;
            ++popped;
        }
    }

    while (popped < wall) {
        idx = (idx == 0) ? 4096 - 1 : idx - 1;
        struct matlab_dbg_undo_rec *r = &matlab_dbg.undo_log[idx];
        ++popped;
        if (r->kind == 4) {
            /* Irreversible op marker — stop here. The user has
             * to live with the prior printed output. Don't pop
             * the marker; leave it so a second stepBack also
             * stops here, not behind it. */
            hit_irreversible = 1;
            if (out_msg && msg_cap > 0) {
                if (r->name)
                    snprintf(out_msg, (size_t)msg_cap,
                             "can't reverse past: %s", r->name);
                else
                    snprintf(out_msg, (size_t)msg_cap,
                             "can't reverse past an irreversible operation");
            }
            ++idx;
            if (idx >= 4096) idx = 0;
            --popped;
            break;
        }
        if (r->kind == 0) {
            /* Statement boundary — this is the new "now". Keep
             * it in the log so the next stepBack drops it (per
             * step 1). */
            hit_boundary = 1;
            boundary_idx = idx;
            boundary_file_id = r->file_id;
            boundary_line = r->line;
            /* Advance idx past the boundary so head ends up just
             * after it; the boundary record stays in place. */
            ++idx;
            if (idx >= 4096) idx = 0;
            --popped;  /* don't count the kept boundary as popped */
            break;
        }
        /* Apply the undo: revert the write described by this record. */
        if (r->kind == 1 || r->kind == 2) {
            /* matlab_ws revert. If the variable existed before,
             * restore the previous (kind, value/ptr); if it
             * didn't, remove the binding entirely via
             * matlab_struct_rmfield so the rewound state matches
             * the pre-write workspace exactly (no stale "x = 0"
             * shadow). */
            if (r->prev_existed) {
                if (r->prev_kind == 0) {
                    matlab_struct_set_f64(matlab_ws, r->name, r->name_len,
                                           r->prev_f64);
                } else if (r->prev_kind == 1) {
                    matlab_struct_set_mat(matlab_ws, r->name, r->name_len,
                                           (matlab_mat *)r->prev_ptr);
                } else if (r->prev_kind == 2) {
                    int32_t i = struct_reserve(matlab_ws, r->name,
                                                (int32_t)r->name_len);
                    matlab_ws->kinds[i] = 2;
                    matlab_ws->f64_vals[i] = 0.0;
                    matlab_ws->ptr_vals[i] = r->prev_ptr;
                }
            } else {
                /* Variable didn't exist before — remove the
                 * binding so `who` / `whos` / DAP variable
                 * inspection see the pre-write state. */
                matlab_struct_rmfield(matlab_ws, r->name, r->name_len);
            }
        } else if (r->kind == 3) {
            /* frame_local revert. Walk the entries[] of the
             * stamped frame and reset the named entry. If the
             * variable didn't exist pre-write, drop it from the
             * table (last-entry-swap) so subsequent reads miss. */
            int t = r->thread_slot;
            int f = r->frame_idx;
            if (t >= 0 && t < 32 && f >= 0 && f < MATLAB_DBG_MAX_FRAMES) {
                struct matlab_dbg_frame_locals *fl =
                    &matlab_dbg.thread_frame_locals[t][f];
                int found = -1;
                for (int i = 0; i < fl->n; ++i) {
                    if (fl->entries[i].name_len == r->name_len &&
                        memcmp(fl->entries[i].name, r->name,
                                (size_t)r->name_len) == 0) {
                        found = i; break;
                    }
                }
                if (r->prev_existed) {
                    if (found >= 0) {
                        fl->entries[found].kind = r->prev_kind;
                        fl->entries[found].f64 = r->prev_f64;
                        fl->entries[found].ptr = r->prev_ptr;
                    }
                } else if (found >= 0) {
                    /* Variable didn't exist before — drop it. */
                    free(fl->entries[found].name);
                    fl->entries[found] = fl->entries[fl->n - 1];
                    fl->n--;
                }
            }
        }
        /* The popped record's name copy is freed when its slot is
         * later overwritten by undo_alloc; no free here. */
    }
    /* Move head back by `popped` (the boundary we hit stays in the
     * log because we decremented popped before breaking). */
    (void)boundary_idx;  /* used only for the assertion in builds with -DDBG */
    matlab_dbg.undo_head = idx;
    matlab_dbg.n_undo -= popped;
    if (matlab_dbg.n_undo < 0) matlab_dbg.n_undo = 0;
    /* If we walked the whole buffer and never hit a boundary or
     * irreversible op, the rewind is best-effort — the IDE got
     * its values rolled back but no line to resume at. Treat as
     * "nothing more to rewind". */
    matlab_dbg.recording_undo = 1;
    /* Refresh shared frames[] from the paused thread so DAP
     * inspectors see the rewound view. */
    int p = matlab_dbg.paused_thread_idx;
    if (p >= 0 && p < 32) {
        int n = matlab_dbg.thread_n_frames[p];
        if (n > MATLAB_DBG_MAX_FRAMES) n = MATLAB_DBG_MAX_FRAMES;
        matlab_dbg.n_frames = n;
        for (int i = 0; i < n; ++i) {
            matlab_dbg.frames[i] = matlab_dbg.thread_frames[p][i];
            matlab_dbg.frame_locals[i] =
                matlab_dbg.thread_frame_locals[p][i];
        }
    }
    if (hit_boundary) {
        if (out_file_id) *out_file_id = boundary_file_id;
        if (out_line) *out_line = boundary_line;
        matlab_dbg.cur_file_id = boundary_file_id;
        matlab_dbg.cur_line = boundary_line;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return hit_irreversible ? -1 : (hit_boundary ? 1 : 0);
}

static void matlab_dbg_watch_trip(int wp_idx) {
    int thr_idx = matlab_dbg_thread_init_chain_locked();
    int n_thr = matlab_dbg.thread_n_frames[thr_idx];
    if (n_thr > 0) {
        matlab_dbg.cur_file_id =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].file_id;
        matlab_dbg.cur_line =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].line;
    }
    /* Snapshot calling thread's chain into shared frames[] for
     * DAP inspectors. Same trick as matlab_dbg_hook on pause. */
    int snap_n = n_thr > MATLAB_DBG_MAX_FRAMES ? MATLAB_DBG_MAX_FRAMES : n_thr;
    matlab_dbg.n_frames = snap_n;
    for (int i = 0; i < snap_n; ++i) {
        matlab_dbg.frames[i] = matlab_dbg.thread_frames[thr_idx][i];
        matlab_dbg.frame_locals[i] =
            matlab_dbg.thread_frame_locals[thr_idx][i];
    }
    matlab_dbg.cur_bp_idx = -1;
    matlab_dbg.last_wp_idx = wp_idx;
    matlab_dbg.paused = 1;
    matlab_dbg.paused_from_watch = 1;
    matlab_dbg.paused_thread_idx = thr_idx;
    pthread_cond_broadcast(&matlab_dbg.cv_server);
    while (matlab_dbg.paused) {
        pthread_cond_wait(&matlab_dbg.cv_client, &matlab_dbg.mu);
    }
    matlab_dbg.paused_from_watch = 0;
    matlab_dbg.last_wp_idx = -1;
    matlab_dbg.paused_thread_idx = -1;
}

/* Lazy thread registration. Called from the hook on every entry —
 * fast path is a constant-time scan of the (small) thread_keys
 * table. New thread → append + assign sequential id. CALLER MUST
 * HOLD matlab_dbg.mu (the hook already does). Returns the slot
 * index in the threads table; the DAP-facing thread id is
 * thread_ids[idx]. */
static int matlab_dbg_thread_slot_locked(void) {
    pthread_t self = pthread_self();
    for (int i = 0; i < matlab_dbg.n_threads; ++i) {
        if (pthread_equal(matlab_dbg.thread_keys[i], self)) return i;
    }
    if (matlab_dbg.n_threads >= 32) {
        /* Table full — reuse slot 0 (main worker). Means the
         * 33rd parfor worker borrows the main worker's id. Better
         * than refusing to track and breaking the hook entirely. */
        return 0;
    }
    int idx = matlab_dbg.n_threads++;
    matlab_dbg.thread_keys[idx] = self;
    /* Sequential id starting at 1; thread 1 is the main worker
     * registered on its first hook fire. Matches the DAP server's
     * pre-existing assumption that threadId 1 is "main". */
    matlab_dbg.thread_ids[idx] = idx + 1;
    return idx;
}

/* DAP-side enumeration: total registered threads. */
int matlab_dbg_thread_count(void) {
    int n;
    pthread_mutex_lock(&matlab_dbg.mu);
    n = matlab_dbg.n_threads;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

/* DAP-side: thread id at index. Returns 0 on out-of-range. */
int32_t matlab_dbg_thread_id_at(int idx) {
    int32_t id = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (idx >= 0 && idx < matlab_dbg.n_threads)
        id = matlab_dbg.thread_ids[idx];
    pthread_mutex_unlock(&matlab_dbg.mu);
    return id;
}

/* DAP-side: id of the thread that triggered the current pause, or
 * 0 if no pause is active. */
int32_t matlab_dbg_paused_thread_id(void) {
    int32_t id = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    int idx = matlab_dbg.paused_thread_idx;
    if (idx >= 0 && idx < matlab_dbg.n_threads)
        id = matlab_dbg.thread_ids[idx];
    pthread_mutex_unlock(&matlab_dbg.mu);
    return id;
}

/* Body of the matlab_ws_set_* watchpoint helper — forward-declared
 * up by the matlab_ws_set_* sites where matlab_dbg state isn't yet
 * in scope. The write has already landed when this fires, so the
 * IDE inspecting the variable on pause sees the new value (matches
 * gdb's "old/new" model where the new value is visible at the stop). */
static void matlab_ws_check_watch(const char *name, int64_t len) {
    if (!name || len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int idx = matlab_dbg_watch_check(name, len, /*scope_hint=*/1);
        if (idx >= 0) matlab_dbg_watch_trip(idx);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Read-side counterpart. Fast path: when n_wp is 0 (no watches at
 * all) the n_wp check fails outside the lock, so we bail without
 * paying mutex cost. The full check happens only when read-watches
 * are armed.
 *
 * Note: scope_hint is hardcoded to 1 (script-ws) because the only
 * read-path call sites are matlab_ws_get_*. Frame-local reads in
 * user code go through stack slots and never call into this API,
 * so they aren't visible to read-watches. The DAP `setDataBreakpoints`
 * handler advertises this limitation in its accessTypes. */
static void matlab_ws_check_read_watch(const char *name, int64_t len) {
    if (!name || len <= 0) return;
    /* Lock-free fast path. matlab_dbg.n_wp is an `int` and the
     * worst case of a torn read is at most a one-statement delay
     * before the watch fires — preferable to taking the global
     * mutex on every JIT-emitted ws_get_* call. */
    if (matlab_dbg.n_wp == 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int idx = matlab_dbg_watch_check_read(name, len, /*scope_hint=*/1);
        if (idx >= 0) matlab_dbg_watch_trip(idx);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Called from the server thread to enable the hook and set the
 * stop-on-entry mode before the worker starts. */
void matlab_dbg_enable(int stop_on_entry) {
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.enabled = 1;
    matlab_dbg.stop_on_entry = stop_on_entry ? 1 : 0;
    matlab_dbg.action = stop_on_entry ? MATLAB_DBG_STEP_IN : MATLAB_DBG_RUN;
    matlab_dbg.n_frames = 1;
    matlab_dbg.frames[0].file_id = 0;
    matlab_dbg.frames[0].line = 0;
    matlab_dbg.frames[0].fn_name = "<script>";
    matlab_dbg.last_wp_idx = -1;
    matlab_dbg.paused_thread_idx = -1;
    /* Turn on undo recording for reverse-stepping. The hook stamps
     * a stmt-boundary record on every fire; ws_set_* / frame_set_*
     * push prev-value records before each write. Clearing the log
     * here ensures a re-launch starts with an empty undo history. */
    matlab_dbg_undo_clear_locked();
    matlab_dbg.recording_undo = 1;
    /* Reset the thread registry on every launch so DAP threadIds
     * start fresh — a re-launch otherwise carries stale entries
     * from the prior session into the IDE's threads pane. Per-
     * thread frame chains and Locals are cleared in lockstep. */
    matlab_dbg.n_threads = 0;
    /* Clear any stale Locals captured during a previous launch
     * (both the legacy shared frame_locals[] and every per-thread
     * slot). dbg state is process-static and DAP can re-launch. */
    for (int i = 0; i < MATLAB_DBG_MAX_FRAMES; ++i)
        matlab_dbg_free_frame_locals(i);
    for (int t = 0; t < 32; ++t) {
        matlab_dbg.thread_n_frames[t] = 0;
        matlab_dbg.thread_step_target_depth[t] = 0;
        for (int i = 0; i < MATLAB_DBG_MAX_FRAMES; ++i) {
            struct matlab_dbg_frame_locals *fl =
                &matlab_dbg.thread_frame_locals[t][i];
            for (int e = 0; e < fl->n; ++e) free(fl->entries[e].name);
            fl->n = 0;
            /* Also free any heap-owned fn_name on stale frames so a
             * re-launch doesn't leak the prior session's strings. */
            char *owned =
                (char *)matlab_dbg.thread_frames[t][i].fn_name;
            if (owned && i > 0) {
                /* Frame 0 is the literal "<script>" — never freed. */
                free(owned);
            }
            matlab_dbg.thread_frames[t][i].fn_name = NULL;
            matlab_dbg.thread_frames[t][i].file_id = 0;
            matlab_dbg.thread_frames[t][i].line = 0;
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Register (file_id -> filename) so the DAP server can resolve
 * breakpoints by file path. Called once per source file before the
 * debuggee starts. file_id is 1-based; we store 0-based. */
void matlab_dbg_register_file(int32_t file_id,
                               const char *name, int64_t name_len) {
    if (file_id <= 0 || file_id > (int32_t)(sizeof matlab_dbg.file_names /
                                              sizeof matlab_dbg.file_names[0]))
        return;
    /* Copy the name so we own it. */
    char *copy = (char *)malloc((size_t)name_len + 1);
    memcpy(copy, name, (size_t)name_len);
    copy[name_len] = '\0';
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.file_names[file_id - 1] = copy;
    matlab_dbg.file_name_lens[file_id - 1] = name_len;
    if (file_id > matlab_dbg.n_files) matlab_dbg.n_files = file_id;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Register (class_id -> class-name) so the DAP server can render a
 * class instance as `1x1 ClassName` instead of falling back to the
 * matrix shape (which read garbage off the obj struct). Called once
 * per classdef from the lowered script entry when -g is on. The
 * registration is idempotent — re-registering the same class_id
 * overwrites the existing entry, which keeps the path safe under
 * repeated launches in long-lived DAP sessions. The string is heap-
 * copied here and freed at process exit (i.e. never — small and
 * bounded by the number of distinct classdefs in the program). */
void matlab_dbg_register_class(int32_t class_id,
                                const char *name, int64_t name_len) {
    if (class_id <= 0 || !name || name_len <= 0) return;
    char *copy = (char *)malloc((size_t)name_len + 1);
    if (!copy) return;
    memcpy(copy, name, (size_t)name_len);
    copy[name_len] = '\0';
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = -1;
    for (int i = 0; i < matlab_dbg.n_classes; ++i) {
        if (matlab_dbg.class_ids[i] == class_id) { slot = i; break; }
    }
    if (slot < 0) {
        int cap = (int)(sizeof matlab_dbg.class_ids /
                        sizeof matlab_dbg.class_ids[0]);
        if (matlab_dbg.n_classes < cap) {
            slot = matlab_dbg.n_classes++;
            matlab_dbg.class_ids[slot] = class_id;
            matlab_dbg.class_names[slot] = NULL;
        }
    }
    if (slot >= 0) {
        free(matlab_dbg.class_names[slot]);
        matlab_dbg.class_names[slot] = copy;
        matlab_dbg.class_name_lens[slot] = name_len;
    } else {
        free(copy);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Look up the class name registered for a given class_id. Returns
 * NULL if the class hasn't been registered (DebugMode off, or a
 * built-in struct slipped through with kind=2 — defensive). */
const char *matlab_dbg_class_name(int32_t class_id, int64_t *len_out) {
    const char *name = NULL;
    int64_t len = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    for (int i = 0; i < matlab_dbg.n_classes; ++i) {
        if (matlab_dbg.class_ids[i] == class_id) {
            name = matlab_dbg.class_names[i];
            len  = matlab_dbg.class_name_lens[i];
            break;
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (len_out) *len_out = name ? len : 0;
    return name;
}

/* Property introspection on a matlab_obj. Used by the DAP server to
 * expand a class-instance row into one child per property. The obj
 * pointer is borrowed from the per-frame Locals table; reading
 * fields is lock-free (the mutating paths run on the debuggee
 * thread, which is paused while the server is reading). */
int matlab_dbg_obj_field_count(void *obj) {
    if (!obj) return 0;
    return ((matlab_obj *)obj)->nfields;
}

const char *matlab_dbg_obj_field_name(void *obj, int i, int64_t *len_out) {
    if (!obj || i < 0) { if (len_out) *len_out = 0; return NULL; }
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) { if (len_out) *len_out = 0; return NULL; }
    const char *n = o->names[i];
    if (len_out) *len_out = n ? (int64_t)strlen(n) : 0;
    return n;
}

int matlab_dbg_obj_field_kind(void *obj, int i) {
    if (!obj || i < 0) return -1;
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) return -1;
    return o->kinds[i];
}

double matlab_dbg_obj_field_f64(void *obj, int i) {
    if (!obj || i < 0) return 0.0;
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) return 0.0;
    return o->f64_vals[i];
}

void *matlab_dbg_obj_field_ptr(void *obj, int i) {
    if (!obj || i < 0) return NULL;
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) return NULL;
    return o->ptr_vals[i];
}

int32_t matlab_dbg_obj_class_id_of(void *obj) {
    return obj ? ((matlab_obj *)obj)->class_id : 0;
}

/* Look up a registered filename by file_id. Returns NULL if unknown.
 * The returned pointer is valid for the lifetime of the process —
 * we own the heap copy made by matlab_dbg_register_file. Used by the
 * DAP server to resolve a paused frame's file_id back to a path so
 * stackTrace responses can reference the correct source. */
const char *matlab_dbg_file_name(int32_t file_id, int64_t *len_out) {
    const char *name = NULL;
    int64_t len = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (file_id >= 1 &&
        file_id <= (int32_t)(sizeof matlab_dbg.file_names /
                              sizeof matlab_dbg.file_names[0])) {
        name = matlab_dbg.file_names[file_id - 1];
        len = matlab_dbg.file_name_lens[file_id - 1];
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (len_out) *len_out = name ? len : 0;
    return name;
}

/* Called from the server thread. Returns the previous breakpoint
 * count for that file so the server can clear-and-reset atomically.
 * Simple: we wipe every breakpoint for that file then re-add. The
 * cond_text / log_text heap copies are freed before compaction so a
 * setBreakpoints replay doesn't leak. */
void matlab_dbg_clear_breakpoints_in_file(int32_t file_id) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int w = 0;
    for (int i = 0; i < matlab_dbg.n_bp; ++i) {
        if (matlab_dbg.bp_file[i] == file_id) {
            free(matlab_dbg.cond_text[i]);
            free(matlab_dbg.log_text[i]);
            continue;
        }
        matlab_dbg.bp_file[w] = matlab_dbg.bp_file[i];
        matlab_dbg.bp_line[w] = matlab_dbg.bp_line[i];
        matlab_dbg.cond_text[w] = matlab_dbg.cond_text[i];
        matlab_dbg.cond_len[w]  = matlab_dbg.cond_len[i];
        matlab_dbg.log_text[w]  = matlab_dbg.log_text[i];
        matlab_dbg.log_len[w]   = matlab_dbg.log_len[i];
        matlab_dbg.cond_disabled[w] = matlab_dbg.cond_disabled[i];
        matlab_dbg.hit_count[w]  = matlab_dbg.hit_count[i];
        matlab_dbg.hit_target[w] = matlab_dbg.hit_target[i];
        matlab_dbg.hit_op[w]     = matlab_dbg.hit_op[i];
        ++w;
    }
    /* Zero out the slots we evicted so subsequent _ex inserts don't
     * inherit a stale pointer the compaction loop just moved away.
     * Hit-count fields reset to 0 so a re-set bp counts from
     * scratch; otherwise repeated `setBreakpoints` round-trips
     * during a debug session would silently inherit prior counts. */
    for (int i = w; i < matlab_dbg.n_bp; ++i) {
        matlab_dbg.cond_text[i] = NULL; matlab_dbg.cond_len[i] = 0;
        matlab_dbg.log_text[i]  = NULL; matlab_dbg.log_len[i]  = 0;
        matlab_dbg.cond_disabled[i] = 0;
        matlab_dbg.hit_count[i] = 0;
        matlab_dbg.hit_target[i] = 0;
        matlab_dbg.hit_op[i] = 0;
    }
    matlab_dbg.n_bp = w;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

int matlab_dbg_add_breakpoint(int32_t file_id, int32_t line) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int ok = matlab_dbg.n_bp < MATLAB_DBG_MAX_BREAKPOINTS;
    if (ok) {
        int i = matlab_dbg.n_bp;
        matlab_dbg.bp_file[i] = file_id;
        matlab_dbg.bp_line[i] = line;
        matlab_dbg.cond_text[i] = NULL; matlab_dbg.cond_len[i] = 0;
        matlab_dbg.log_text[i]  = NULL; matlab_dbg.log_len[i]  = 0;
        matlab_dbg.cond_disabled[i] = 0;
        matlab_dbg.n_bp++;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Conditional / log-point-aware insert with optional hit-count
 * gate. Either text pointer may be NULL (with matching len = 0)
 * to mean "no condition" / "no log". hit_op == 0 disables the
 * hit-count gate. The runtime owns the heap copy so the server
 * can release its own buffers immediately after returning. */
int matlab_dbg_add_breakpoint_ex2(int32_t file_id, int32_t line,
                                   const char *cond, int64_t cond_len,
                                   const char *log,  int64_t log_len,
                                   int hit_op, int64_t hit_target) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int ok = matlab_dbg.n_bp < MATLAB_DBG_MAX_BREAKPOINTS;
    if (ok) {
        int i = matlab_dbg.n_bp;
        matlab_dbg.bp_file[i] = file_id;
        matlab_dbg.bp_line[i] = line;
        matlab_dbg.cond_text[i] = NULL; matlab_dbg.cond_len[i] = 0;
        matlab_dbg.log_text[i]  = NULL; matlab_dbg.log_len[i]  = 0;
        matlab_dbg.cond_disabled[i] = 0;
        matlab_dbg.hit_count[i] = 0;
        matlab_dbg.hit_target[i] = hit_target;
        matlab_dbg.hit_op[i] = hit_op;
        if (cond && cond_len > 0) {
            matlab_dbg.cond_text[i] = (char *)malloc((size_t)cond_len + 1);
            memcpy(matlab_dbg.cond_text[i], cond, (size_t)cond_len);
            matlab_dbg.cond_text[i][cond_len] = '\0';
            matlab_dbg.cond_len[i] = cond_len;
        }
        if (log && log_len > 0) {
            matlab_dbg.log_text[i] = (char *)malloc((size_t)log_len + 1);
            memcpy(matlab_dbg.log_text[i], log, (size_t)log_len);
            matlab_dbg.log_text[i][log_len] = '\0';
            matlab_dbg.log_len[i] = log_len;
        }
        matlab_dbg.n_bp++;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Backward-compat wrapper for the v1 _ex API (no hit-count gate). */
int matlab_dbg_add_breakpoint_ex(int32_t file_id, int32_t line,
                                  const char *cond, int64_t cond_len,
                                  const char *log,  int64_t log_len) {
    return matlab_dbg_add_breakpoint_ex2(file_id, line, cond, cond_len,
                                          log, log_len, 0, 0);
}

/* Snapshot the cond / log text for a given bp index. Caller-supplied
 * pointers receive runtime-owned strings that stay valid until the
 * next clear_breakpoints_in_file call. The disabled out-param is
 * non-zero when the condition was previously rejected (eval failed)
 * — callers should treat the bp as condition-less but still suppress
 * the pause to match VS Code's "broken condition is silent" UX.
 * Returns 0 on out-of-range. */
int matlab_dbg_breakpoint_meta(int idx, const char **cond, int64_t *cond_len,
                                const char **log, int64_t *log_len,
                                int *disabled) {
    int ok = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (idx >= 0 && idx < matlab_dbg.n_bp) {
        if (cond)     *cond     = matlab_dbg.cond_text[idx];
        if (cond_len) *cond_len = matlab_dbg.cond_len[idx];
        if (log)      *log      = matlab_dbg.log_text[idx];
        if (log_len)  *log_len  = matlab_dbg.log_len[idx];
        if (disabled) *disabled = matlab_dbg.cond_disabled[idx];
        ok = 1;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

void matlab_dbg_disable_condition(int idx) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (idx >= 0 && idx < matlab_dbg.n_bp)
        matlab_dbg.cond_disabled[idx] = 1;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

int matlab_dbg_get_pause_bp(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int idx = matlab_dbg.cur_bp_idx;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return idx;
}

/* Called from the server thread after handling a stopped event.
 * Sets the next action and wakes the worker. */
void matlab_dbg_resume(int action) {
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.action = (enum matlab_dbg_action)action;
    /* Step targets are per-thread: a step in worker A must use
     * worker A's depth, not whatever the legacy shared
     * n_frames last got snapshotted to. We seed every thread's
     * target depth from the currently-paused thread's depth so
     * the resume kicks the right thread to the right place. */
    int paused = matlab_dbg.paused_thread_idx;
    if (paused >= 0 && paused < 32) {
        int n = matlab_dbg.thread_n_frames[paused];
        if (action == MATLAB_DBG_STEP_OVER)
            matlab_dbg.thread_step_target_depth[paused] = n;
        else if (action == MATLAB_DBG_STEP_OUT)
            matlab_dbg.thread_step_target_depth[paused] = n - 1;
    }
    /* Legacy single-thread fallback: keep updating the shared
     * step_target_depth so any unconverted single-threaded
     * stepping path still reads a sane value. */
    if (action == MATLAB_DBG_STEP_OVER)
        matlab_dbg.step_target_depth = matlab_dbg.n_frames;
    else if (action == MATLAB_DBG_STEP_OUT)
        matlab_dbg.step_target_depth = matlab_dbg.n_frames - 1;
    matlab_dbg.paused = 0;
    pthread_cond_broadcast(&matlab_dbg.cv_client);
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Called from the server thread to read the current pause point. */
void matlab_dbg_get_pause(int32_t *file_id, int32_t *line) {
    pthread_mutex_lock(&matlab_dbg.mu);
    *file_id = matlab_dbg.cur_file_id;
    *line = matlab_dbg.cur_line;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Frame counts are published so the server can draw a stackTrace. */
int matlab_dbg_frame_count(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int n = matlab_dbg.n_frames;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

/* Snapshot frame i (0-based, 0 = innermost) into caller-supplied outs.
 * Returns 1 on success. fn_name's storage is runtime-owned. */
int matlab_dbg_frame_at(int i, int32_t *file_id, int32_t *line,
                         const char **fn_name) {
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Frames are stored with index 0 = outermost. Convert. */
    int idx = matlab_dbg.n_frames - 1 - i;
    int ok = idx >= 0 && idx < matlab_dbg.n_frames;
    if (ok) {
        *file_id = matlab_dbg.frames[idx].file_id;
        *line    = matlab_dbg.frames[idx].line;
        *fn_name = matlab_dbg.frames[idx].fn_name;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Workspace snapshot — the server asks for these on every `variables`
 * request. Output uses a small array populated by the caller; the
 * server copies fields out while holding no runtime lock. The struct
 * for a variable's value is returned as its stored f64 or matrix
 * pointer; the server formats for display. */
int matlab_dbg_ws_count(void) {
    matlab_ws_init_if_needed();
    return matlab_ws ? matlab_ws->nfields : 0;
}

const char *matlab_dbg_ws_name(int i, int64_t *len_out) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) {
        *len_out = 0;
        return "";
    }
    const char *n = matlab_ws->names[i];
    *len_out = (int64_t)strlen(n);
    return n;
}

int matlab_dbg_ws_kind(int i) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) return -1;
    return matlab_ws->kinds[i];
}

double matlab_dbg_ws_f64(int i) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) return 0.0;
    return matlab_ws->f64_vals[i];
}

void *matlab_dbg_ws_ptr(int i) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) return NULL;
    return matlab_ws->ptr_vals[i];
}

/* Shape accessors used by the DAP `variables` formatter. Thin wrappers
 * around the opaque matlab_mat struct — the DAP server doesn't have
 * access to the internal layout. */
int64_t matlab_dbg_mat_rows(matlab_mat *m) { return m ? m->rows : 0; }
int64_t matlab_dbg_mat_cols(matlab_mat *m) { return m ? m->cols : 0; }

/* Element accessor for the DAP matrix-expansion path. Out-of-range
 * indices return 0.0 so a malformed children request can't read past
 * the data buffer. Indices are 1-based to match how the DAP server
 * presents cells (`(1,1)`, `(1,2)`, ...) — we subtract one before
 * indexing the row-major buffer. Complex / 3-D / typed-int matrices
 * have their own accessors below; this one returns 0.0 if asked
 * about a tagged descriptor. */
double matlab_dbg_mat_get(matlab_mat *m, int64_t i, int64_t j) {
    if (!m || !m->data) return 0.0;
    if (i < 1 || j < 1) return 0.0;
    if (mat_is_complex(m) || mat_is_3d(m)) return 0.0;
    if (i > m->rows || j > m->cols) return 0.0;
    /* Row-major: data[(i-1) * cols + (j-1)]. */
    return m->data[(i - 1) * m->cols + (j - 1)];
}

/* Discriminators + per-kind accessors used by the DAP `variables`
 * expander to drill into complex and 3-D matrices.
 *
 * The DAP server stores any kind=1 ws/frame value as a `void *`
 * because matlab_mat / matlab_mat_c / matlab_mat3 share that LLVM
 * type but have different layouts. Each helper below begins by
 * confirming the magic before accessing layout-specific fields, so
 * passing a plain matlab_mat into `matlab_dbg_mat_c_re()` is a
 * defensive zero rather than a wild read. */
int32_t matlab_dbg_mat_kind(const void *p) {
    if (!p) return 0;
    if (mat_is_complex(p)) return 2;   /* matlab_mat_c */
    if (mat_is_3d(p))      return 3;   /* matlab_mat3   */
    return 1;                          /* plain matlab_mat */
}
/* matlab_mat_c accessors are defined alongside its struct body
 * further down in the complex section — that section needs to be
 * in scope to access ->re / ->im / ->rows / ->cols. The discriminator
 * above is layout-agnostic (reads only the magic at offset 0) so
 * it lives here. */
int64_t matlab_dbg_mat_c_rows(const matlab_mat_c *m);
int64_t matlab_dbg_mat_c_cols(const matlab_mat_c *m);
double matlab_dbg_mat_c_re(const matlab_mat_c *m, int64_t i, int64_t j);
double matlab_dbg_mat_c_im(const matlab_mat_c *m, int64_t i, int64_t j);
int64_t matlab_dbg_mat3_rows(const matlab_mat3 *m) {
    if (!m || !mat_is_3d(m)) return 0;
    return m->rows;
}
int64_t matlab_dbg_mat3_cols(const matlab_mat3 *m) {
    if (!m || !mat_is_3d(m)) return 0;
    return m->cols;
}
int64_t matlab_dbg_mat3_depth(const matlab_mat3 *m) {
    if (!m || !mat_is_3d(m)) return 0;
    return m->depth;
}
/* Memory-inspection accessors. The DAP `readMemory` / `writeMemory`
 * requests use a `memoryReference` (per spec, a hex string) plus an
 * offset to identify what to read. We hand out memory refs only for
 * matrix data buffers — everything else is opaque or scalar — and
 * the readMemory handler decodes the hex back to a pointer to walk
 * the cells as raw bytes. Returning the buffer pointer + total byte
 * size lets the DAP server bound the read so a 100MB readMemory
 * request can't walk past the buffer. */
void *matlab_dbg_mat_data_ptr(void *Mraw) {
    if (!Mraw) return NULL;
    int32_t kind = matlab_dbg_mat_kind(Mraw);
    if (kind == 1) return ((matlab_mat *)Mraw)->data;
    if (kind == 3) return ((matlab_mat3 *)Mraw)->data;
    /* Complex matrices have two parallel buffers (re/im); a single
     * pointer can't cover both. Refuse for now — the IDE's memory
     * view would only see the real component, which would be
     * misleading. */
    return NULL;
}
int64_t matlab_dbg_mat_data_bytes(void *Mraw) {
    if (!Mraw) return 0;
    int32_t kind = matlab_dbg_mat_kind(Mraw);
    if (kind == 1) {
        matlab_mat *m = (matlab_mat *)Mraw;
        return m->rows * m->cols * (int64_t)sizeof(double);
    }
    if (kind == 3) {
        matlab_mat3 *m = (matlab_mat3 *)Mraw;
        return m->rows * m->cols * m->depth * (int64_t)sizeof(double);
    }
    return 0;
}

double matlab_dbg_mat3_get(const matlab_mat3 *m,
                           int64_t i, int64_t j, int64_t k) {
    if (!m || !mat_is_3d(m) || !m->data) return 0.0;
    if (i < 1 || j < 1 || k < 1) return 0.0;
    if (i > m->rows || j > m->cols || k > m->depth) return 0.0;
    /* Slice-major: matches mat3_offset above. */
    return m->data[(k - 1) * m->rows * m->cols + (i - 1) * m->cols + (j - 1)];
}

/* The injected hook. Called from JIT'd code at each statement entry
 * when compiled with -g. Takes (file_id, line) as raw ints so the
 * emitted call is cheap — just two arith.constant ops feeding a
 * known runtime symbol. */
void matlab_dbg_hook(int32_t file_id, int32_t line) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (!matlab_dbg.enabled) {
        pthread_mutex_unlock(&matlab_dbg.mu);
        return;
    }
    /* Lazy-register the calling thread on first hook entry so the
     * DAP server can enumerate it via `threads`. Also seeds the
     * thread's per-thread frame chain with a `<script>` entry on
     * first touch so frame[0].fn_name reads correctly. */
    int thr_idx = matlab_dbg_thread_init_chain_locked();
    int *thr_n = &matlab_dbg.thread_n_frames[thr_idx];
    /* Update the innermost frame's line in the calling thread's
     * own chain. Concurrent parfor workers each touch their own
     * slot, so no cross-thread corruption. */
    if (*thr_n > 0) {
        matlab_dbg.thread_frames[thr_idx][*thr_n - 1].file_id = file_id;
        matlab_dbg.thread_frames[thr_idx][*thr_n - 1].line = line;
    }
    /* Statement-boundary record for reverse stepping. The undo
     * log gets one of these per hook fire; stepBack walks back
     * until it finds the previous boundary. Cheap (no allocation
     * — kind=0 records just stamp ints). */
    matlab_dbg_undo_record_stmt_locked(file_id, line, thr_idx);

    int should_pause = 0;
    int matched_bp = -1;
    /* Stepping: decide based on action + the calling thread's own
     * depth (step targets are per-thread; a step in worker A
     * shouldn't fire when worker B reaches its target depth). */
    switch (matlab_dbg.action) {
    case MATLAB_DBG_STEP_IN:
        should_pause = 1;
        break;
    case MATLAB_DBG_STEP_OVER:
        if (*thr_n <= matlab_dbg.thread_step_target_depth[thr_idx])
            should_pause = 1;
        break;
    case MATLAB_DBG_STEP_OUT:
        if (*thr_n <= matlab_dbg.thread_step_target_depth[thr_idx])
            should_pause = 1;
        break;
    case MATLAB_DBG_STOP:
        pthread_mutex_unlock(&matlab_dbg.mu);
        pthread_exit(NULL);
        return;
    default:
        break;
    }
    /* Breakpoint check (regardless of step action). Records the
     * matched index so the DAP server can read the breakpoint's
     * condition / log strings without re-walking the table.
     *
     * Hit-count gate: when hit_op is set, increment hit_count and
     * compare to hit_target with the encoded operator. A hit_op
     * of 0 (no gate) goes straight to should_pause = 1, matching
     * the prior behaviour. The gate runs BEFORE the conditional /
     * log eval so a `hitCondition: ">= 100"` skips the JIT cost
     * for the first 99 hits — important for tight loops. */
    for (int i = 0; i < matlab_dbg.n_bp; ++i) {
        if (matlab_dbg.bp_file[i] == file_id &&
            matlab_dbg.bp_line[i] == line) {
            matched_bp = i;
            int op = matlab_dbg.hit_op[i];
            if (op != 0) {
                int64_t c = ++matlab_dbg.hit_count[i];
                int64_t t = matlab_dbg.hit_target[i];
                int gate = 0;
                switch (op) {
                case 1: gate = (c == t); break;
                case 2: gate = (c >= t); break;
                case 3: gate = (c >  t); break;
                case 4: gate = (t > 0 && c % t == 0); break;
                default: gate = 1; break;
                }
                if (!gate) break;
            }
            should_pause = 1;
            break;
        }
    }
    /* Exception-breakpoint filter: pause if the error flag is set
     * AND the DAP client has enabled the `error` filter. Reads the
     * error flag directly to avoid recursing through the public API
     * while we already hold matlab_dbg.mu. */
    if (matlab_dbg.pause_on_error && matlab_error_flag) {
        should_pause = 1;
    }
    if (should_pause) {
        matlab_dbg.cur_file_id = file_id;
        matlab_dbg.cur_line = line;
        matlab_dbg.cur_bp_idx = matched_bp;
        matlab_dbg.paused = 1;
        matlab_dbg.paused_thread_idx = thr_idx;
        /* Snapshot the calling thread's frame chain into the
         * shared frames[] / frame_locals[] arrays so DAP
         * inspectors that still read those directly see the
         * paused thread's stack. The legacy single-threaded
         * accessors (matlab_dbg_frame_count / _frame_at /
         * _frame_local_*) are unmodified — they read from
         * frames[]/frame_locals[] which is now a snapshot view.
         *
         * Names are kept as-is (the per-thread chain owns them),
         * so the snapshot is a shallow copy. The shared array's
         * matlab_dbg_free_frame_locals path is no longer called
         * during normal lifecycle — ownership stays with the
         * per-thread arrays. */
        int n = matlab_dbg.thread_n_frames[thr_idx];
        if (n > MATLAB_DBG_MAX_FRAMES) n = MATLAB_DBG_MAX_FRAMES;
        matlab_dbg.n_frames = n;
        for (int i = 0; i < n; ++i) {
            matlab_dbg.frames[i] = matlab_dbg.thread_frames[thr_idx][i];
            matlab_dbg.frame_locals[i] =
                matlab_dbg.thread_frame_locals[thr_idx][i];
        }
        /* Signal the server that we're paused; wait for resume. */
        pthread_cond_broadcast(&matlab_dbg.cv_server);
        while (matlab_dbg.paused) {
            pthread_cond_wait(&matlab_dbg.cv_client, &matlab_dbg.mu);
        }
        matlab_dbg.paused_thread_idx = -1;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Frame-tracking hooks (used when -g is on and we instrument user
 * function entry/exit). The name pointer the JIT hands us is into a
 * read-only global that is NOT null-terminated — the global is sized
 * exactly to the bytes of the function name, with no trailing 0. Any
 * caller that subsequently uses %s on `fn_name` would read past the
 * global into whatever happens to be next in the constant pool, which
 * is exactly what tripped up the DAP `stackTrace` response and the
 * error()-backtrace printer.
 *
 * Heap-copy the name on enter and free it on leave. This keeps every
 * downstream consumer (DAP server, traceback printer, future eval)
 * able to treat fn_name as a plain C string. The cost is a tiny
 * malloc/free per call when -g is on, which is the path that's
 * already paying the per-statement hook overhead. */
/* Free the locals stored at frame_idx — used both on leave_frame
 * and as a defensive reset on enter_frame in case a previous run
 * left stale entries (shouldn't happen with balanced enter/leave but
 * cheap insurance for recursive functions reusing the same depth). */
static void matlab_dbg_free_frame_locals(int frame_idx) {
    /* Caller must hold matlab_dbg.mu. */
    if (frame_idx < 0 || frame_idx >= MATLAB_DBG_MAX_FRAMES) return;
    struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
    for (int i = 0; i < fl->n; ++i) {
        free(fl->entries[i].name);
        fl->entries[i].name = NULL;
    }
    fl->n = 0;
}

/* Resolve the calling thread's per-thread frame chain, lazily
 * seeding it with a `<script>` entry on first touch so DAP
 * inspectors always have a frame[0] to read. CALLER MUST HOLD
 * matlab_dbg.mu. */
static int matlab_dbg_thread_init_chain_locked(void) {
    int slot = matlab_dbg_thread_slot_locked();
    if (matlab_dbg.thread_n_frames[slot] == 0) {
        matlab_dbg.thread_n_frames[slot] = 1;
        matlab_dbg.thread_frames[slot][0].file_id = 0;
        matlab_dbg.thread_frames[slot][0].line = 0;
        matlab_dbg.thread_frames[slot][0].fn_name = "<script>";
    }
    return slot;
}

void matlab_dbg_enter_frame(const char *fn_name, int64_t name_len) {
    if (name_len < 0) name_len = 0;
    char *owned = (char *)malloc((size_t)name_len + 1);
    if (owned) {
        if (name_len > 0) memcpy(owned, fn_name, (size_t)name_len);
        owned[name_len] = '\0';
    }
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Per-thread chain push. Each pthread (main worker, parfor
     * workers) maintains its own call-stack so concurrent
     * parfor-body enters don't trample each other. The shared
     * frames[] is refreshed only when this thread pauses (in
     * the hook) so DAP inspectors that still read frames[]
     * directly see the paused thread's stack. */
    int slot = matlab_dbg_thread_init_chain_locked();
    int *pn = &matlab_dbg.thread_n_frames[slot];
    if (*pn < MATLAB_DBG_MAX_FRAMES) {
        struct matlab_dbg_frame_locals *fl =
            &matlab_dbg.thread_frame_locals[slot][*pn];
        for (int i = 0; i < fl->n; ++i) free(fl->entries[i].name);
        fl->n = 0;
        matlab_dbg.thread_frames[slot][*pn].fn_name = owned;
        matlab_dbg.thread_frames[slot][*pn].file_id = 0;
        matlab_dbg.thread_frames[slot][*pn].line = 0;
        (*pn)++;
    } else {
        free(owned);  /* table full; drop the name we copied */
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

void matlab_dbg_leave_frame(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = matlab_dbg_thread_init_chain_locked();
    int *pn = &matlab_dbg.thread_n_frames[slot];
    if (*pn > 1) {
        (*pn)--;
        char *owned = (char *)matlab_dbg.thread_frames[slot][*pn].fn_name;
        matlab_dbg.thread_frames[slot][*pn].fn_name = NULL;
        free(owned);
        struct matlab_dbg_frame_locals *fl =
            &matlab_dbg.thread_frame_locals[slot][*pn];
        for (int i = 0; i < fl->n; ++i) free(fl->entries[i].name);
        fl->n = 0;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Mirror entry points called from the lowering after every store to
 * a named slot when DebugMode is on. Records the variable's current
 * value into the innermost frame's mini-workspace so the DAP server
 * can render Locals for any frame in the stack — not just the
 * script-level workspace.
 *
 * The implementation is deliberately the simple linear-scan one:
 * MATLAB programs' per-function variable counts are tiny (a handful)
 * and stores are cheap; a hash table would be heavier for no gain.
 * Names are heap-copied on first set (subsequent updates reuse the
 * existing entry). The matrix pointer is stored as borrowed — the
 * matrix struct itself is owned by the JIT's slot or workspace and
 * survives at least until matlab_dbg_leave_frame fires. */
/* Generic find-or-alloc operating on a caller-supplied
 * frame_locals slot. Lets the per-thread frame_set_* path target
 * its own thread's slot without going through the shared
 * matlab_dbg.frame_locals[]. */
static int matlab_dbg_frame_local_find_or_alloc_in(
    struct matlab_dbg_frame_locals *fl,
    const char *name, int64_t name_len) {
    if (!fl) return -1;
    for (int i = 0; i < fl->n; ++i) {
        if (fl->entries[i].name_len == name_len &&
            memcmp(fl->entries[i].name, name, (size_t)name_len) == 0)
            return i;
    }
    if (fl->n >= MATLAB_DBG_MAX_LOCALS) return -1;
    char *copy = (char *)malloc((size_t)name_len + 1);
    if (!copy) return -1;
    memcpy(copy, name, (size_t)name_len);
    copy[name_len] = '\0';
    int idx = fl->n++;
    fl->entries[idx].name = copy;
    fl->entries[idx].name_len = name_len;
    fl->entries[idx].kind = 0;
    fl->entries[idx].f64 = 0.0;
    fl->entries[idx].ptr = NULL;
    return idx;
}

static int matlab_dbg_frame_local_find_or_alloc(int frame_idx,
                                                 const char *name,
                                                 int64_t name_len) {
    /* Caller holds the dbg mutex. Returns an index in entries[] or
     * -1 if the table is full / frame_idx is out of range. */
    if (frame_idx < 0 || frame_idx >= MATLAB_DBG_MAX_FRAMES) return -1;
    struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
    for (int i = 0; i < fl->n; ++i) {
        if (fl->entries[i].name_len == name_len &&
            memcmp(fl->entries[i].name, name, (size_t)name_len) == 0)
            return i;
    }
    if (fl->n >= MATLAB_DBG_MAX_LOCALS) return -1;
    char *copy = (char *)malloc((size_t)name_len + 1);
    if (!copy) return -1;
    memcpy(copy, name, (size_t)name_len);
    copy[name_len] = '\0';
    int idx = fl->n++;
    fl->entries[idx].name = copy;
    fl->entries[idx].name_len = name_len;
    fl->entries[idx].kind = 0;
    fl->entries[idx].f64 = 0.0;
    fl->entries[idx].ptr = NULL;
    return idx;
}

/* Resolve the calling thread's innermost-frame frame_locals slot,
 * lazily seeding the chain if this is the thread's first touch.
 * Returns NULL if the chain is empty (n == 0) — caller drops the
 * write silently in that case. */
static struct matlab_dbg_frame_locals *
matlab_dbg_thread_innermost_locals_locked(void) {
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    if (n <= 0 || n > MATLAB_DBG_MAX_FRAMES) return NULL;
    return &matlab_dbg.thread_frame_locals[slot][n - 1];
}

void matlab_dbg_frame_set_f64(const char *name, int64_t name_len, double v) {
    if (!name || name_len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Write into the calling thread's innermost-frame slot. Per-
     * thread storage means concurrent parfor workers' Locals don't
     * trample each other's frames. The shared frame_locals[] is
     * refreshed by the hook on pause via the snapshot-to-shared
     * copy, so DAP inspectors see the paused thread's view. */
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    if (n > 0)
        matlab_dbg_frame_push_undo_locked(slot, n - 1, name, name_len);
    struct matlab_dbg_frame_locals *fl =
        matlab_dbg_thread_innermost_locals_locked();
    if (fl) {
        int idx = matlab_dbg_frame_local_find_or_alloc_in(fl, name, name_len);
        if (idx >= 0) {
            fl->entries[idx].kind = 0;
            fl->entries[idx].f64 = v;
            fl->entries[idx].ptr = NULL;
        }
    }
    /* Watchpoint check on frame-local writes. scope_hint=2 (frame).
     * Already inside the dbg mutex, so we call _watch_check /
     * _trip directly without re-locking. */
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int wp = matlab_dbg_watch_check(name, name_len, /*scope_hint=*/2);
        if (wp >= 0) matlab_dbg_watch_trip(wp);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

void matlab_dbg_frame_set_mat(const char *name, int64_t name_len, void *mat) {
    if (!name || name_len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    if (n > 0)
        matlab_dbg_frame_push_undo_locked(slot, n - 1, name, name_len);
    struct matlab_dbg_frame_locals *fl =
        matlab_dbg_thread_innermost_locals_locked();
    if (fl) {
        int idx = matlab_dbg_frame_local_find_or_alloc_in(fl, name, name_len);
        if (idx >= 0) {
            fl->entries[idx].kind = 1;
            fl->entries[idx].ptr = mat;
            fl->entries[idx].f64 = 0.0;
        }
    }
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int wp = matlab_dbg_watch_check(name, name_len, /*scope_hint=*/2);
        if (wp >= 0) matlab_dbg_watch_trip(wp);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Class-instance variant. `obj` is a matlab_obj* whose class_id tag
 * is the registry key for the class name (see
 * matlab_dbg_register_class). Same lifetime contract as set_mat —
 * the obj is borrowed from the JIT's slot. */
void matlab_dbg_frame_set_obj(const char *name, int64_t name_len, void *obj) {
    if (!name || name_len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    if (n > 0)
        matlab_dbg_frame_push_undo_locked(slot, n - 1, name, name_len);
    struct matlab_dbg_frame_locals *fl =
        matlab_dbg_thread_innermost_locals_locked();
    if (fl) {
        int idx = matlab_dbg_frame_local_find_or_alloc_in(fl, name, name_len);
        if (idx >= 0) {
            fl->entries[idx].kind = 2;
            fl->entries[idx].ptr = obj;
            fl->entries[idx].f64 = 0.0;
        }
    }
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int wp = matlab_dbg_watch_check(name, name_len, /*scope_hint=*/2);
        if (wp >= 0) matlab_dbg_watch_trip(wp);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP read-side: enumerate Locals for a given frame index. Frame
 * indexing here matches matlab_dbg.frames[] (0 = outermost / script,
 * n_frames-1 = innermost). The DAP server adapts this to its own
 * top-of-stack-first frame ordering. */
int matlab_dbg_frame_locals_count(int frame_idx) {
    int n = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames)
        n = matlab_dbg.frame_locals[frame_idx].n;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

const char *matlab_dbg_frame_local_name(int frame_idx, int i,
                                         int64_t *len_out) {
    const char *p = NULL;
    int64_t L = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) {
            p = fl->entries[i].name;
            L = fl->entries[i].name_len;
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (len_out) *len_out = L;
    return p;
}

int matlab_dbg_frame_local_kind(int frame_idx, int i) {
    int k = -1;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) k = fl->entries[i].kind;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return k;
}

double matlab_dbg_frame_local_f64(int frame_idx, int i) {
    double v = 0.0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) v = fl->entries[i].f64;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return v;
}

void *matlab_dbg_frame_local_ptr(int frame_idx, int i) {
    void *p = NULL;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) p = fl->entries[i].ptr;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return p;
}

/* Blocks until the worker is paused or has exited. Used by the
 * server to know when it can handle client requests safely. */
void matlab_dbg_wait_for_pause(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    while (!matlab_dbg.paused)
        pthread_cond_wait(&matlab_dbg.cv_server, &matlab_dbg.mu);
    pthread_mutex_unlock(&matlab_dbg.mu);
}

int matlab_dbg_is_paused(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int p = matlab_dbg.paused;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return p;
}

void matlab_dbg_mat(const char *file, int64_t file_len,
                    int32_t line,
                    const char *label, int64_t label_len,
                    matlab_mat *m) {
    int fl = (int)(file_len > 0 ? file_len : 0);
    int ll = (int)(label_len > 0 ? label_len : 0);
    const char *flt = file ? file : "<repl>";
    if (!file) fl = (int)strlen(flt);
    pthread_mutex_lock(&matlab_io_mutex);
    if (!m) {
        fprintf(stderr, "%.*s:%d: %.*s = <null>\n",
                fl, flt, line,
                ll > 0 ? ll : (int)strlen("<expr>"),
                ll > 0 ? label : "<expr>");
        pthread_mutex_unlock(&matlab_io_mutex);
        return;
    }
    fprintf(stderr, "%.*s:%d: %.*s = [%lldx%lld]\n",
            fl, flt, line,
            ll > 0 ? ll : (int)strlen("<expr>"),
            ll > 0 ? label : "<expr>",
            (long long)m->rows, (long long)m->cols);
    /* Also print the matrix content (up to 8 rows / 8 cols) so
     * small matrices are readable inline. */
    int64_t maxr = m->rows > 8 ? 8 : m->rows;
    int64_t maxc = m->cols > 8 ? 8 : m->cols;
    for (int64_t i = 0; i < maxr; ++i) {
        fprintf(stderr, "  ");
        for (int64_t j = 0; j < maxc; ++j) {
            fprintf(stderr, " %10g", m->data[i * m->cols + j]);
        }
        if (m->cols > 8) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }
    if (m->rows > 8) fprintf(stderr, "  ...\n");
    pthread_mutex_unlock(&matlab_io_mutex);
}

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
struct matlab_cell_s {
    int32_t n;
    int32_t cap;
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

/* Matrix disp. Special-cases 1×1 to print scalar-style and 1×N to print
 * on one line (matching MATLAB's default disp formatting). Polymorphic:
 * accepts either a real matlab_mat* or a complex matlab_mat_c* — the
 * magic-tag check on the real path keeps the fast-path branch-free
 * for normal use (first-field read that stays in cache). */
void matlab_disp_mat(void *Aptr) {
    if (!Aptr) return;
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

/*===========================================================================
 * Complex numbers.
 *
 * Representation mirrors the real matrix: a heap descriptor with separate
 * real / imaginary f64 planes (row-major, rows*cols entries each). Separate
 * planes (rather than interleaved pairs) keep the existing SIMD-friendly
 * contiguous-loop shape on the real-only fast path and let us share scalar
 * math kernels between real and complex matrices.
 *
 * Scalars are 1x1 matrices — same trick the real runtime uses so the
 * compiler only has to plumb one MLIR type (`!llvm.ptr`). `matlab.const_complex`
 * from the frontend lowers to `matlab_complex_scalar(re, im)` which
 * allocates a 1x1 matlab_mat_c.
 *
 * Interop: a `matlab_mat_c` never mixes with `matlab_mat` at runtime. The
 * Lowerer decides, per binop, whether to route to the real or complex
 * variant based on Sema's propagated Dtype. When the operands disagree
 * (e.g. `real + 2i`), the Lowerer promotes the real operand via
 * matlab_mat_c_from_real before the complex op.
 *===========================================================================*/

/* matlab_mat_c layout. MATLAB_MAT_C_MAGIC + mat_is_complex() are
 * forward-declared near the top of the runtime so the polymorphic
 * real-side entries (matlab_disp_mat, etc.) can discriminate the
 * layout without pulling in the full complex runtime upfront. */
struct matlab_mat_c {
    uint32_t magic;    /* MATLAB_MAT_C_MAGIC */
    uint32_t _pad;     /* keep re/im 8-byte aligned */
    double *re;        /* row-major, rows*cols doubles */
    double *im;        /* row-major, rows*cols doubles */
    int64_t rows;
    int64_t cols;
};

static matlab_mat_c *mat_c_alloc(int64_t m, int64_t n) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    matlab_mat_c *A = (matlab_mat_c *)calloc(1, sizeof(matlab_mat_c));
    A->magic = MATLAB_MAT_C_MAGIC;
    A->rows = m; A->cols = n;
    A->re = (double *)calloc((size_t)(m * n + 1), sizeof(double));
    A->im = (double *)calloc((size_t)(m * n + 1), sizeof(double));
    return A;
}

/* DAP-side accessors for complex matrices. Forward-declared in the
 * matlab_dbg section above; defined here where the matlab_mat_c
 * layout is fully visible. Each defends against being called with
 * a non-complex descriptor by re-checking the magic byte. */
int64_t matlab_dbg_mat_c_rows(const matlab_mat_c *m) {
    if (!m || !mat_is_complex(m)) return 0;
    return m->rows;
}
int64_t matlab_dbg_mat_c_cols(const matlab_mat_c *m) {
    if (!m || !mat_is_complex(m)) return 0;
    return m->cols;
}
double matlab_dbg_mat_c_re(const matlab_mat_c *m, int64_t i, int64_t j) {
    if (!m || !mat_is_complex(m) || !m->re) return 0.0;
    if (i < 1 || j < 1) return 0.0;
    if (i > m->rows || j > m->cols) return 0.0;
    return m->re[(i - 1) * m->cols + (j - 1)];
}
double matlab_dbg_mat_c_im(const matlab_mat_c *m, int64_t i, int64_t j) {
    if (!m || !mat_is_complex(m) || !m->im) return 0.0;
    if (i < 1 || j < 1) return 0.0;
    if (i > m->rows || j > m->cols) return 0.0;
    return m->im[(i - 1) * m->cols + (j - 1)];
}

/* Constructors ----------------------------------------------------------*/

matlab_mat_c *matlab_complex_scalar(double re, double im) {
    matlab_mat_c *A = mat_c_alloc(1, 1);
    A->re[0] = re; A->im[0] = im;
    return A;
}

/* Promote a real matrix to complex (zero imag). Used at binop sites where
 * one operand is complex and the other is real. Allocates a fresh
 * descriptor; the caller's real matrix is unchanged. */
matlab_mat_c *matlab_mat_c_from_real(matlab_mat *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
    return C;
}

matlab_mat_c *matlab_mat_c_from_buf(const double *re, const double *im,
                                     double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    matlab_mat_c *A = mat_c_alloc(rm, cn);
    memcpy(A->re, re, (size_t)(rm * cn) * sizeof(double));
    if (im) memcpy(A->im, im, (size_t)(rm * cn) * sizeof(double));
    return A;
}

/* Unary ------------------------------------------------------------------*/

/* conj / real / imag / angle / abs are polymorphic: they accept either a
 * real matlab_mat* or a complex matlab_mat_c*. The mat_is_complex() check
 * distinguishes the two at the ABI boundary. On the real path, imag()
 * returns zeros, conj() is an identity copy, etc. */
matlab_mat_c *matlab_conj_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
        memcpy(C->re, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
        return C;  /* imag is already zeroed by calloc */
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) {
        C->re[k] =  A->re[k];
        C->im[k] = -A->im[k];
    }
    return C;
}

matlab_mat_c *matlab_neg_c(matlab_mat_c *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) {
        C->re[k] = -A->re[k];
        C->im[k] = -A->im[k];
    }
    return C;
}

matlab_mat *matlab_real_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        /* real(A) on a real A is just A (copy to avoid aliasing). */
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        memcpy(R->data, A->data, (size_t)(A->rows * A->cols) * sizeof(double));
        return R;
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    memcpy(R->data, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    return R;
}

matlab_mat *matlab_imag_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        matlab_mat *A = (matlab_mat *)Aptr;
        return mat_alloc(A->rows, A->cols);  /* zeros */
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    memcpy(R->data, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    return R;
}

matlab_mat *matlab_angle_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        /* angle(a) for real a: 0 if a >= 0, pi if a < 0. */
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        int64_t n = A->rows * A->cols;
        for (int64_t k = 0; k < n; ++k)
            R->data[k] = A->data[k] < 0.0 ? M_PI : 0.0;
        return R;
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k) R->data[k] = atan2(A->im[k], A->re[k]);
    return R;
}

matlab_mat *matlab_abs_c(void *Aptr) {
    if (!Aptr) return mat_alloc(0, 0);
    if (!mat_is_complex(Aptr)) {
        matlab_mat *A = (matlab_mat *)Aptr;
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        int64_t n = A->rows * A->cols;
        for (int64_t k = 0; k < n; ++k) R->data[k] = fabs(A->data[k]);
        return R;
    }
    matlab_mat_c *A = (matlab_mat_c *)Aptr;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    int64_t n = A->rows * A->cols;
    for (int64_t k = 0; k < n; ++k)
        R->data[k] = hypot(A->re[k], A->im[k]);
    return R;
}

/* Element-wise binary ---------------------------------------------------*/

/* Broadcast: 1x1 matches any shape. Returns the shape to use. 0 on
 * incompatible shapes. */
static int mat_c_bcast(matlab_mat_c *A, matlab_mat_c *B,
                        int64_t *m_out, int64_t *n_out) {
    if (!A || !B) return 0;
    if (A->rows == B->rows && A->cols == B->cols) {
        *m_out = A->rows; *n_out = A->cols; return 1;
    }
    if (A->rows == 1 && A->cols == 1) {
        *m_out = B->rows; *n_out = B->cols; return 1;
    }
    if (B->rows == 1 && B->cols == 1) {
        *m_out = A->rows; *n_out = A->cols; return 1;
    }
    return 0;
}

#define MAT_C_BINARY(NAME, OP_RE, OP_IM) \
    matlab_mat_c *matlab_##NAME##_cc(matlab_mat_c *A, matlab_mat_c *B) { \
        int64_t m, n; \
        if (!mat_c_bcast(A, B, &m, &n)) return mat_c_alloc(0, 0); \
        matlab_mat_c *C = mat_c_alloc(m, n); \
        int a_scalar = A->rows == 1 && A->cols == 1; \
        int b_scalar = B->rows == 1 && B->cols == 1; \
        for (int64_t k = 0; k < m * n; ++k) { \
            double ar = a_scalar ? A->re[0] : A->re[k]; \
            double ai = a_scalar ? A->im[0] : A->im[k]; \
            double br = b_scalar ? B->re[0] : B->re[k]; \
            double bi = b_scalar ? B->im[0] : B->im[k]; \
            C->re[k] = (OP_RE); \
            C->im[k] = (OP_IM); \
            (void)ar; (void)ai; (void)br; (void)bi; \
        } \
        return C; \
    }

MAT_C_BINARY(add, ar + br, ai + bi)
MAT_C_BINARY(sub, ar - br, ai - bi)
MAT_C_BINARY(emul, ar*br - ai*bi, ar*bi + ai*br)
/* Element-wise divide: (a + bi) / (c + di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2) */
matlab_mat_c *matlab_ediv_cc(matlab_mat_c *A, matlab_mat_c *B) {
    int64_t m, n;
    if (!mat_c_bcast(A, B, &m, &n)) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(m, n);
    int a_scalar = A->rows == 1 && A->cols == 1;
    int b_scalar = B->rows == 1 && B->cols == 1;
    for (int64_t k = 0; k < m * n; ++k) {
        double ar = a_scalar ? A->re[0] : A->re[k];
        double ai = a_scalar ? A->im[0] : A->im[k];
        double br = b_scalar ? B->re[0] : B->re[k];
        double bi = b_scalar ? B->im[0] : B->im[k];
        double denom = br*br + bi*bi;
        C->re[k] = (ar*br + ai*bi) / denom;
        C->im[k] = (ai*br - ar*bi) / denom;
    }
    return C;
}

#undef MAT_C_BINARY

/* Matrix multiply: C(i,j) = sum_k A(i,k) * B(k,j). Naive O(m*n*p). */
matlab_mat_c *matlab_matmul_cc(matlab_mat_c *A, matlab_mat_c *B) {
    if (!A || !B || A->cols != B->rows) return mat_c_alloc(0, 0);
    int64_t m = A->rows, p = A->cols, n = B->cols;
    matlab_mat_c *C = mat_c_alloc(m, n);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double sr = 0, si = 0;
            for (int64_t k = 0; k < p; ++k) {
                double ar = A->re[i*p + k], ai = A->im[i*p + k];
                double br = B->re[k*n + j], bi = B->im[k*n + j];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            C->re[i*n + j] = sr;
            C->im[i*n + j] = si;
        }
    }
    return C;
}

/* Transpose variants: `.'` keeps entries as-is; `'` (ctranspose) also
 * conjugates. Both swap dims. */
matlab_mat_c *matlab_transpose_c(matlab_mat_c *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i)
        for (int64_t j = 0; j < A->cols; ++j) {
            C->re[j*A->rows + i] = A->re[i*A->cols + j];
            C->im[j*A->rows + i] = A->im[i*A->cols + j];
        }
    return C;
}

matlab_mat_c *matlab_ctranspose_c(matlab_mat_c *A) {
    if (!A) return mat_c_alloc(0, 0);
    matlab_mat_c *C = mat_c_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i)
        for (int64_t j = 0; j < A->cols; ++j) {
            C->re[j*A->rows + i] =  A->re[i*A->cols + j];
            C->im[j*A->rows + i] = -A->im[i*A->cols + j];
        }
    return C;
}

/* Display ---------------------------------------------------------------*/

static void disp_complex_scalar(double re, double im) {
    /* MATLAB-ish: "re + imi" or "re - imi"; drop parts that are exactly 0
     * except when both are zero (print "0"). */
    if (im == 0.0 && re == 0.0) { printf("0\n"); return; }
    if (im == 0.0) { printf("%g\n", re); return; }
    if (re == 0.0) {
        printf("%gi\n", im);
        return;
    }
    if (im < 0.0) printf("%g - %gi\n", re, -im);
    else          printf("%g + %gi\n", re, im);
}

void matlab_disp_mat_c(matlab_mat_c *A) {
    if (!A) return;
    if (A->rows == 0 || A->cols == 0) return;
    pthread_mutex_lock(&matlab_io_mutex);
    if (A->rows == 1 && A->cols == 1) {
        disp_complex_scalar(A->re[0], A->im[0]);
        pthread_mutex_unlock(&matlab_io_mutex);
        return;
    }
    for (int64_t i = 0; i < A->rows; ++i) {
        for (int64_t j = 0; j < A->cols; ++j) {
            double re = A->re[i*A->cols + j];
            double im = A->im[i*A->cols + j];
            if (j) printf("   ");
            if (im >= 0.0) printf("%9.4g + %.4gi", re, im);
            else           printf("%9.4g - %.4gi", re, -im);
        }
        printf("\n");
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* Shape queries / rows / cols (DAP formatter reads these via the real
 * runtime; duplicate for complex). */
int64_t matlab_mat_c_rows(matlab_mat_c *A) { return A ? A->rows : 0; }
int64_t matlab_mat_c_cols(matlab_mat_c *A) { return A ? A->cols : 0; }

/*===========================================================================
 * FFT — pure-C Cooley-Tukey.
 *
 * Two code paths:
 *   - Power-of-two N: standard iterative radix-2 DIT with bit reversal.
 *     O(N log N), exact result (modulo rounding) for all power-of-2 N.
 *   - General N: Bluestein's algorithm — expresses DFT(x) as a convolution
 *     (which is itself radix-2 FFT'd at the next power of 2 >= 2N-1). O(N
 *     log N) asymptotically; a few-× slower than direct radix-2 but no
 *     dependency on N's factorization.
 *
 * Both operate on a matlab_mat_c. Input is either (1, N) or (N, 1) for a
 * 1-D vector; matrix inputs get fft applied along the first non-singleton
 * dim (MATLAB convention — we only ship fft-along-rows-or-single-column
 * here, matching typical use).
 *===========================================================================*/

static int is_power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* In-place radix-2 DIT FFT on the size-N arrays re[], im[]. inverse=1
 * applies the conjugate twiddle (caller is responsible for the 1/N scale
 * that MATLAB's ifft applies). Pre: N is a power of 2 and N >= 1. */
static void fft_radix2_inplace(double *re, double *im, int64_t N, int inverse) {
    if (N < 2) return;
    /* Bit-reversal permutation. */
    int64_t j = 0;
    for (int64_t i = 1; i < N; ++i) {
        int64_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    /* Iterative butterflies. */
    double sign = inverse ? 1.0 : -1.0;
    for (int64_t len = 2; len <= N; len <<= 1) {
        double ang = sign * 2.0 * M_PI / (double)len;
        double wlen_r = cos(ang), wlen_i = sin(ang);
        for (int64_t i = 0; i < N; i += len) {
            double w_r = 1.0, w_i = 0.0;
            int64_t half = len >> 1;
            for (int64_t k = 0; k < half; ++k) {
                double u_r = re[i + k];
                double u_i = im[i + k];
                double v_r = re[i + k + half] * w_r - im[i + k + half] * w_i;
                double v_i = re[i + k + half] * w_i + im[i + k + half] * w_r;
                re[i + k] = u_r + v_r;
                im[i + k] = u_i + v_i;
                re[i + k + half] = u_r - v_r;
                im[i + k + half] = u_i - v_i;
                double nw_r = w_r * wlen_r - w_i * wlen_i;
                double nw_i = w_r * wlen_i + w_i * wlen_r;
                w_r = nw_r; w_i = nw_i;
            }
        }
    }
}

/* Bluestein's algorithm for arbitrary N. Builds chirp a[n] = x[n] * w^(-n^2/2)
 * and convolves with b[n] = w^(n^2/2), then multiplies by a[n]^-1 again.
 * We use a power-of-2 FFT of size M >= 2N-1 for the convolution. inverse=1
 * applies the conjugate twiddle (caller handles the 1/N normalization). */
static void fft_bluestein(double *re, double *im, int64_t N, int inverse) {
    if (N < 2) return;
    /* M = next power of 2 >= 2N - 1. */
    int64_t M = 1;
    while (M < 2 * N - 1) M <<= 1;

    double *chirp_r = (double *)calloc((size_t)N, sizeof(double));
    double *chirp_i = (double *)calloc((size_t)N, sizeof(double));
    double *a_r = (double *)calloc((size_t)M, sizeof(double));
    double *a_i = (double *)calloc((size_t)M, sizeof(double));
    double *b_r = (double *)calloc((size_t)M, sizeof(double));
    double *b_i = (double *)calloc((size_t)M, sizeof(double));

    double sign = inverse ? 1.0 : -1.0;
    /* chirp[n] = exp(sign * i * pi * n^2 / N). Precompute for n in [0, N).
     * Use (n*n) mod (2N) to keep the argument small for large N. */
    for (int64_t n = 0; n < N; ++n) {
        /* ang = sign * pi * (n^2 mod (2N)) / N */
        int64_t nn = (n * n) % (2 * N);
        double ang = sign * M_PI * (double)nn / (double)N;
        chirp_r[n] = cos(ang);
        chirp_i[n] = sin(ang);
    }
    /* a[n] = x[n] * conj(chirp[n]) */
    for (int64_t n = 0; n < N; ++n) {
        a_r[n] = re[n] * chirp_r[n] + im[n] * chirp_i[n];
        a_i[n] = im[n] * chirp_r[n] - re[n] * chirp_i[n];
    }
    /* b[n] = chirp[n] for 0 <= n < N, and b[M-n] = chirp[n] for 1 <= n < N.
     * Zero elsewhere. This is the "symmetric" Bluestein kernel shape. */
    b_r[0] = chirp_r[0]; b_i[0] = chirp_i[0];
    for (int64_t n = 1; n < N; ++n) {
        b_r[n] = chirp_r[n]; b_i[n] = chirp_i[n];
        b_r[M - n] = chirp_r[n]; b_i[M - n] = chirp_i[n];
    }
    /* Convolution via FFT(a) * FFT(b), then IFFT. */
    fft_radix2_inplace(a_r, a_i, M, 0);
    fft_radix2_inplace(b_r, b_i, M, 0);
    for (int64_t k = 0; k < M; ++k) {
        double pr = a_r[k] * b_r[k] - a_i[k] * b_i[k];
        double pi = a_r[k] * b_i[k] + a_i[k] * b_r[k];
        a_r[k] = pr; a_i[k] = pi;
    }
    fft_radix2_inplace(a_r, a_i, M, 1);
    /* Scale the inverse-FFT result by 1/M and multiply by conj(chirp). */
    for (int64_t n = 0; n < N; ++n) {
        double yr = a_r[n] / (double)M;
        double yi = a_i[n] / (double)M;
        re[n] = yr * chirp_r[n] + yi * chirp_i[n];
        im[n] = yi * chirp_r[n] - yr * chirp_i[n];
    }
    free(chirp_r); free(chirp_i);
    free(a_r); free(a_i);
    free(b_r); free(b_i);
}

/* Apply 1-D FFT to each column of the caller's matrix in place. */
static void fft_columns_inplace(double *re, double *im,
                                 int64_t rows, int64_t cols, int inverse) {
    double *col_r = (double *)malloc((size_t)rows * sizeof(double));
    double *col_i = (double *)malloc((size_t)rows * sizeof(double));
    for (int64_t c = 0; c < cols; ++c) {
        for (int64_t r = 0; r < rows; ++r) {
            col_r[r] = re[r * cols + c];
            col_i[r] = im[r * cols + c];
        }
        if (is_power_of_two(rows))
            fft_radix2_inplace(col_r, col_i, rows, inverse);
        else
            fft_bluestein(col_r, col_i, rows, inverse);
        for (int64_t r = 0; r < rows; ++r) {
            re[r * cols + c] = col_r[r];
            im[r * cols + c] = col_i[r];
        }
    }
    free(col_r); free(col_i);
}

/* Apply 1-D FFT to each row. */
static void fft_rows_inplace(double *re, double *im,
                              int64_t rows, int64_t cols, int inverse) {
    for (int64_t r = 0; r < rows; ++r) {
        if (is_power_of_two(cols))
            fft_radix2_inplace(re + r * cols, im + r * cols, cols, inverse);
        else
            fft_bluestein(re + r * cols, im + r * cols, cols, inverse);
    }
}

/* Public fft / ifft entries. Take an opaque ptr — either matlab_mat*
 * (real) or matlab_mat_c* (complex). mat_is_complex() discriminates
 * by the magic marker at byte 0 and the real path is auto-promoted
 * via matlab_mat_c_from_real. MATLAB's dim rule: vectors are FFT'd
 * along their non-singleton dim; matrices along columns (dim 1). */
matlab_mat_c *matlab_fft_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    if (A->rows == 1) {
        fft_rows_inplace(C->re, C->im, C->rows, C->cols, 0);
    } else {
        fft_columns_inplace(C->re, C->im, C->rows, C->cols, 0);
    }
    return C;
}

matlab_mat_c *matlab_ifft_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    int64_t n = (A->rows == 1) ? A->cols : A->rows;
    if (A->rows == 1) {
        fft_rows_inplace(C->re, C->im, C->rows, C->cols, 1);
    } else {
        fft_columns_inplace(C->re, C->im, C->rows, C->cols, 1);
    }
    /* MATLAB ifft applies the 1/N scale. */
    double inv = 1.0 / (double)n;
    int64_t total = C->rows * C->cols;
    for (int64_t k = 0; k < total; ++k) {
        C->re[k] *= inv;
        C->im[k] *= inv;
    }
    return C;
}

/* 2-D variants: FFT along rows then columns (or vice versa — order
 * doesn't matter for a separable transform). */
matlab_mat_c *matlab_fft2_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    fft_rows_inplace(C->re, C->im, C->rows, C->cols, 0);
    fft_columns_inplace(C->re, C->im, C->rows, C->cols, 0);
    return C;
}

matlab_mat_c *matlab_ifft2_c(void *Aptr) {
    if (!Aptr) return mat_c_alloc(0, 0);
    matlab_mat_c *A = mat_is_complex(Aptr)
        ? (matlab_mat_c *)Aptr
        : matlab_mat_c_from_real((matlab_mat *)Aptr);
    matlab_mat_c *C = mat_c_alloc(A->rows, A->cols);
    memcpy(C->re, A->re, (size_t)(A->rows * A->cols) * sizeof(double));
    memcpy(C->im, A->im, (size_t)(A->rows * A->cols) * sizeof(double));
    fft_rows_inplace(C->re, C->im, C->rows, C->cols, 1);
    fft_columns_inplace(C->re, C->im, C->rows, C->cols, 1);
    double inv = 1.0 / (double)(C->rows * C->cols);
    int64_t total = C->rows * C->cols;
    for (int64_t k = 0; k < total; ++k) {
        C->re[k] *= inv;
        C->im[k] *= inv;
    }
    return C;
}

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
