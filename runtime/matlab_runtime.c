/* Tiny MATLAB-runtime shim. Linked with programs produced by matlabc's
 * -emit-llvm pipeline.
 *
 * All functions use a leading `matlab_` prefix to avoid collision with libc
 * and to make the calling convention explicit to the compiler frontend.
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define BINARY_MM(name, op) \
    matlab_mat *matlab_##name##_mm(matlab_mat *A, matlab_mat *B) { \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) C->data[k] = (op); \
        return C; \
    }

#define BINARY_MS(name, op) \
    matlab_mat *matlab_##name##_ms(matlab_mat *A, double s) { \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) C->data[k] = (op); \
        return C; \
    }

#define BINARY_SM(name, op) \
    matlab_mat *matlab_##name##_sm(double s, matlab_mat *A) { \
        int64_t m = A->rows, n = A->cols; \
        matlab_mat *C = mat_alloc(m, n); \
        for (int64_t k = 0; k < m * n; ++k) C->data[k] = (op); \
        return C; \
    }

BINARY_MM(add,  A->data[k] + B->data[k])
BINARY_MM(sub,  A->data[k] - B->data[k])
BINARY_MM(emul, A->data[k] * B->data[k])
BINARY_MM(ediv, A->data[k] / B->data[k])
BINARY_MM(epow, pow(A->data[k], B->data[k]))

BINARY_MS(add,  A->data[k] + s)
BINARY_MS(sub,  A->data[k] - s)
BINARY_MS(emul, A->data[k] * s)
BINARY_MS(ediv, A->data[k] / s)
BINARY_MS(epow, pow(A->data[k], s))

BINARY_SM(add,  s + A->data[k])
BINARY_SM(sub,  s - A->data[k])
BINARY_SM(emul, s * A->data[k])
BINARY_SM(ediv, s / A->data[k])
BINARY_SM(epow, pow(s, A->data[k]))

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
UNARY_M(sqrt, sqrt(x))
UNARY_M(abs,  fabs(x))

#undef UNARY_M

/* Scalar versions for when the operand is a plain f64 (needed when the
 * frontend couldn't statically prove the operand was scalar and the scalar
 * arith lowering didn't fire). */
double matlab_exp_s(double x)  { return exp(x);  }
double matlab_log_s(double x)  { return log(x);  }
double matlab_sin_s(double x)  { return sin(x);  }
double matlab_cos_s(double x)  { return cos(x);  }
double matlab_tan_s(double x)  { return tan(x);  }
double matlab_sqrt_s(double x) { return sqrt(x); }
double matlab_abs_s(double x)  { return fabs(x); }

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

/* Matrix disp. Special-cases 1×1 to print scalar-style and 1×N to print
 * on one line (matching MATLAB's default disp formatting). */
void matlab_disp_mat(matlab_mat *A) {
    if (!A) return;
    if (A->rows == 1 && A->cols == 1) {
        pthread_mutex_lock(&matlab_io_mutex);
        printf("%g\n", A->data[0]);
        pthread_mutex_unlock(&matlab_io_mutex);
        return;
    }
    matlab_disp_mat_f64(A->data, A->rows, A->cols);
}
