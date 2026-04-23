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

void matlab_set_error(void) { matlab_error_flag = 1; }
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
    double *data;
    int64_t rows, cols, depth;
} matlab_mat3;

static matlab_mat3 *mat3_alloc(int64_t m, int64_t n, int64_t p) {
    if (m < 0) m = 0;
    if (n < 0) n = 0;
    if (p < 0) p = 0;
    matlab_mat3 *A = (matlab_mat3 *)calloc(1, sizeof(*A));
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

double matlab_global_get_f64(int32_t id) {
    if (id < 0 || id >= MATLAB_GLOBAL_TABLE_SIZE) return 0.0;
    return matlab_global_table[id];
}

void matlab_global_set_f64(int32_t id, double v) {
    if (id < 0 || id >= MATLAB_GLOBAL_TABLE_SIZE) return;
    matlab_global_table[id] = v;
}

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
