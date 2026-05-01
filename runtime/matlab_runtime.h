#ifndef MATLAB_RUNTIME_H
#define MATLAB_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types — the layout lives in matlab_runtime.c; generated code only
// ever passes pointers to these.
typedef struct matlab_mat      matlab_mat;
typedef struct matlab_mat_c    matlab_mat_c;
typedef struct matlab_struct_s matlab_struct;
typedef struct matlab_cell_s   matlab_cell;

// Parfor callback.
typedef void (*matlab_parfor_body_t)(double iv, void *state);

// I/O.
void matlab_disp_str(const char *s, int64_t n);
void matlab_disp_f64(double v);
void matlab_disp_vec_f64(const double *data, int64_t n);
void matlab_disp_mat_f64(const double *data, int64_t m, int64_t n);
void matlab_disp_mat(void *A);  /* polymorphic: matlab_mat* or matlab_mat_c* */
void matlab_fprintf_str(const char *fmt, int64_t n);
void matlab_fprintf_f64(const char *fmt, int64_t n, double v);
void matlab_fprintf_f64_2(const char *fmt, int64_t n, double a, double b);
void matlab_fprintf_f64_3(const char *fmt, int64_t n,
                          double a, double b, double c);
void matlab_fprintf_f64_4(const char *fmt, int64_t n,
                          double a, double b, double c, double d);
double matlab_input_num(const char *prompt, int64_t plen);

/* Timing & sleep.
 *   matlab_pause(s)        — sleep s seconds. s<=0 / NaN returns immediately.
 *   matlab_pause_keypress() — block until any byte arrives on stdin (matches
 *                             MATLAB's no-arg `pause`). No-op if stdin is
 *                             not a tty so non-interactive runs don't hang.
 *   matlab_tic()            — record the monotonic-clock start for the
 *                             default tic/toc slot (per-thread).
 *   matlab_toc()            — return seconds elapsed since the last tic
 *                             on this thread (0.0 if tic never called).
 *   matlab_toc_print()      — same elapsed read, printed as MATLAB does:
 *                             "Elapsed time is X.YYY seconds."
 */
void   matlab_pause(double seconds);
void   matlab_pause_keypress(void);
void   matlab_tic(void);
double matlab_toc(void);
void   matlab_toc_print(void);

// Parallel / reductions.
void matlab_parfor_dispatch(double start, double step, double end,
                            matlab_parfor_body_t body, void *state);
void matlab_reduce_add_f64(double *ptr, double delta);

// Matrix constructors.
matlab_mat *matlab_mat_from_buf(const double *buf, double m, double n);
matlab_mat *matlab_mat_from_scalar(double x);
// `if M` / `while M` truth test: 1 iff M is non-empty AND all elems non-zero.
int8_t      matlab_mat_truth(matlab_mat *m);
matlab_mat *matlab_empty_mat(void);
matlab_mat *matlab_zeros(double m, double n);
matlab_mat *matlab_ones(double m, double n);
matlab_mat *matlab_eye(double m, double n);
matlab_mat *matlab_magic(double nd);
matlab_mat *matlab_rand(double m, double n);
matlab_mat *matlab_randn(double m, double n);
matlab_mat *matlab_range(double start, double step, double end);
matlab_mat *matlab_repmat(matlab_mat *A, double m, double n);

// Linear algebra.
matlab_mat *matlab_matmul_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_inv(matlab_mat *A);
matlab_mat *matlab_mldivide_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_mrdivide_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_svd(matlab_mat *A_in);
matlab_mat *matlab_eig(matlab_mat *A_in);
matlab_mat *matlab_eig_V(matlab_mat *A_in);
matlab_mat *matlab_eig_D(matlab_mat *A_in);
double      matlab_det(matlab_mat *A);
matlab_mat *matlab_transpose(matlab_mat *A);
matlab_mat *matlab_diag(matlab_mat *A);
matlab_mat *matlab_reshape(matlab_mat *A, double m, double n);
matlab_mat *matlab_matpow(matlab_mat *A, double n);

/* Convolution (full shape — MATLAB default). conv(u,v) treats u and v as
 * vectors and returns a vector of length numel(u)+numel(v)-1, oriented as
 * a row if both inputs are rows (or scalar), otherwise as a column.
 * conv2(A,B) returns an (m1+m2-1)x(n1+n2-1) matrix. */
matlab_mat *matlab_conv(matlab_mat *u, matlab_mat *v);
matlab_mat *matlab_conv2(matlab_mat *A, matlab_mat *B);

/* IIR/FIR filter — y = filter(b, a, x). Implements the difference equation
 *   a(1)*y[n] = sum_k b[k]*x[n-k] - sum_k a[k+1]*y[n-k-1]
 * via direct-form II transposed. b and a are vectors; x can be a vector
 * (filtered as a vector) or a matrix (filtered column-wise). a(1) must be
 * non-zero (returns 0x0 otherwise). */
matlab_mat *matlab_filter(matlab_mat *b, matlab_mat *a, matlab_mat *x);

/* Logical reductions. MATLAB rule (matches sum/mean/min/max):
 *   - vector → 1x1 logical (0 or 1, stored as a double).
 *   - matrix → 1xN row, one bool per column. */
matlab_mat *matlab_any(matlab_mat *A);
matlab_mat *matlab_all(matlab_mat *A);

/* Triangular extraction: tril(A) zeroes everything strictly above the
 * main diagonal; triu(A) zeroes everything strictly below. */
matlab_mat *matlab_tril(matlab_mat *A);
matlab_mat *matlab_triu(matlab_mat *A);

/* Spectral shift — moves the zero-frequency bin to the centre of the
 * spectrum (fftshift) or back (ifftshift). Polymorphic on real and
 * complex inputs. For 2-D inputs with both rows>1 and cols>1, swaps
 * quadrants; for vectors, swaps halves. */
matlab_mat_c *matlab_fftshift_c(void *A);
matlab_mat_c *matlab_ifftshift_c(void *A);

/* Dispersion + median. std/var follow the same shape rule as mean
 * (vector→scalar, matrix→1xN row). N-1 normalisation (sample variance),
 * to match MATLAB's default. median uses linear-time quickselect on a
 * scratch copy. */
matlab_mat *matlab_std(matlab_mat *A);
matlab_mat *matlab_var(matlab_mat *A);
matlab_mat *matlab_median(matlab_mat *A);

/* First-order discrete differences. On a vector, returns a vector of
 * length n-1 with v(i+1) - v(i). On a matrix, differences are taken
 * down each column. */
matlab_mat *matlab_diff(matlab_mat *A);

/* Coordinate matrices. meshgrid uses image (xy) ordering — X varies
 * along columns, Y along rows. ndgrid uses array (ij) ordering —
 * X varies along rows, Y along columns. Both accept either two
 * vectors or one (used for both axes). The compiler splits a
 * `[X,Y] = meshgrid(...)` site into two single-output runtime calls. */
matlab_mat *matlab_meshgrid_X(matlab_mat *x, matlab_mat *y);
matlab_mat *matlab_meshgrid_Y(matlab_mat *x, matlab_mat *y);
matlab_mat *matlab_ndgrid_X(matlab_mat *x, matlab_mat *y);
matlab_mat *matlab_ndgrid_Y(matlab_mat *x, matlab_mat *y);

/* Cross-correlation. xcorr(u, v) treats both as vectors and returns
 * a vector of length 2*max(numel(u), numel(v)) - 1, with lag-zero at
 * index max(N,M) (1-based: max(N,M)). For real inputs it equals
 * conv(u, fliplr(v)) shifted to the standard lag origin. */
matlab_mat *matlab_xcorr(matlab_mat *u, matlab_mat *v);

/* Polynomial helpers. MATLAB stores coefficients highest-power-first:
 *   p = [a_n, a_(n-1), ..., a_1, a_0]
 *   polyval(p, x) -> a_n*x^n + ... + a_0, evaluated elementwise on x.
 *   polyfit(x, y, n) -> least-squares fit of degree n (returns a
 *     row vector of length n+1 in the same coefficient order).
 *   roots(p) -> column vector with the n roots of p (complex layout). */
matlab_mat   *matlab_polyval(matlab_mat *p, matlab_mat *x);
matlab_mat   *matlab_polyfit(matlab_mat *x, matlab_mat *y, double n);
matlab_mat_c *matlab_roots(matlab_mat *p);

/* 1-D linear interpolation. interp1(x, y, xi) requires x to be sorted
 * and the same length as y. xi can be any shape; the output mirrors
 * xi's shape. Out-of-range xi values produce NaN (MATLAB default). */
matlab_mat *matlab_interp1(matlab_mat *x, matlab_mat *y, matlab_mat *xi);

/* Trapezoidal integration / differentiation. trapz(y) assumes unit
 * spacing; trapz(x, y) uses x. cumtrapz(y) returns a running integral
 * of the same length as y, leading 0. gradient(f) is central-difference
 * in the interior, one-sided at the endpoints. All three follow the
 * vector-vs-matrix shape rule (matrix → column-wise, result is 1xN
 * for trapz, same-shape for cumtrapz / gradient). */
matlab_mat *matlab_trapz(matlab_mat *y);
matlab_mat *matlab_trapz_xy(matlab_mat *x, matlab_mat *y);
matlab_mat *matlab_cumtrapz(matlab_mat *y);
matlab_mat *matlab_gradient(matlab_mat *f);

/* DSP windows. n must be >= 1; returns a column vector of length n. */
matlab_mat *matlab_hamming(double n);
matlab_mat *matlab_hann(double n);
matlab_mat *matlab_blackman(double n);

/*--- Tier 3: SVD-derived linalg + image-processing wrappers + 2-D interp ---*/

/* rank(A): count of singular values larger than max(m,n)*sigma_max*eps.
 * cond(A): sigma_max / sigma_min (Inf if rank-deficient). */
double matlab_rank(matlab_mat *A);
double matlab_cond(matlab_mat *A);

/* null(A): orthonormal basis for ker(A). Computed via eigendecomposition
 * of A'*A — eigenvectors with eigenvalue ≈ 0 form the null-space basis.
 * orth(A): orthonormal basis for col(A). For m >= n uses QR + rank
 * truncation; for m < n falls back to eig of A*A'. */
matlab_mat *matlab_null(matlab_mat *A);
matlab_mat *matlab_orth(matlab_mat *A);

/* imfilter(A, h): 2-D filtering with 'same' output size — applies h
 * to A then crops the conv2 'full' result by floor(size(h)/2) on
 * each side. Boundary handling is implicit zero (same as conv2).
 * padarray(A, padsize): zero-pad an image by [pre_rows pre_cols]
 * (or scalar applied to both dims); returns
 * (rows + 2*pad_r) x (cols + 2*pad_c). */
matlab_mat *matlab_imfilter(matlab_mat *A, matlab_mat *h);
matlab_mat *matlab_padarray(matlab_mat *A, matlab_mat *padsize);

/* interp2(X, Y, V, Xq, Yq): bilinear interpolation. X is a sorted
 * 1xN row, Y is a sorted Mx1 column, V is MxN. Xq / Yq must have the
 * same shape; output mirrors that shape. Out-of-range queries → NaN. */
matlab_mat *matlab_interp2(matlab_mat *X, matlab_mat *Y, matlab_mat *V,
                           matlab_mat *Xq, matlab_mat *Yq);

/* upsample(x, n): insert n-1 zeros between samples (length L*n).
 * downsample(x, n): take every n-th sample starting at index 1. */
matlab_mat *matlab_upsample(matlab_mat *x, double n);
matlab_mat *matlab_downsample(matlab_mat *x, double n);

// Element-wise binary ops (matrix/matrix, matrix/scalar, scalar/matrix).
matlab_mat *matlab_add_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_sub_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_emul_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ediv_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_epow_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_add_ms(matlab_mat *A, double s);
matlab_mat *matlab_sub_ms(matlab_mat *A, double s);
matlab_mat *matlab_emul_ms(matlab_mat *A, double s);
matlab_mat *matlab_ediv_ms(matlab_mat *A, double s);
matlab_mat *matlab_epow_ms(matlab_mat *A, double s);
matlab_mat *matlab_add_sm(double s, matlab_mat *A);
matlab_mat *matlab_sub_sm(double s, matlab_mat *A);
matlab_mat *matlab_emul_sm(double s, matlab_mat *A);
matlab_mat *matlab_ediv_sm(double s, matlab_mat *A);
matlab_mat *matlab_epow_sm(double s, matlab_mat *A);

// Element-wise comparisons (return 0/1 matrices).
matlab_mat *matlab_gt_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ge_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_lt_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_le_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_eq_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ne_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_gt_ms(matlab_mat *A, double s);
matlab_mat *matlab_ge_ms(matlab_mat *A, double s);
matlab_mat *matlab_lt_ms(matlab_mat *A, double s);
matlab_mat *matlab_le_ms(matlab_mat *A, double s);
matlab_mat *matlab_eq_ms(matlab_mat *A, double s);
matlab_mat *matlab_ne_ms(matlab_mat *A, double s);
matlab_mat *matlab_gt_sm(double s, matlab_mat *A);
matlab_mat *matlab_ge_sm(double s, matlab_mat *A);
matlab_mat *matlab_lt_sm(double s, matlab_mat *A);
matlab_mat *matlab_le_sm(double s, matlab_mat *A);
matlab_mat *matlab_eq_sm(double s, matlab_mat *A);
matlab_mat *matlab_ne_sm(double s, matlab_mat *A);

// Element-wise unary ops on matrices.
matlab_mat *matlab_neg_m(matlab_mat *A);
matlab_mat *matlab_exp_m(matlab_mat *A);
matlab_mat *matlab_log_m(matlab_mat *A);
matlab_mat *matlab_sin_m(matlab_mat *A);
matlab_mat *matlab_cos_m(matlab_mat *A);
matlab_mat *matlab_tan_m(matlab_mat *A);
matlab_mat *matlab_sqrt_m(matlab_mat *A);
matlab_mat *matlab_abs_m(matlab_mat *A);

// Column-wise / scalar reductions.
matlab_mat *matlab_sum(matlab_mat *A);
matlab_mat *matlab_prod(matlab_mat *A);
matlab_mat *matlab_mean(matlab_mat *A);
matlab_mat *matlab_min(matlab_mat *A);
matlab_mat *matlab_max(matlab_mat *A);
matlab_mat *matlab_min_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_max_mm(matlab_mat *A, matlab_mat *B);

// Shape / predicates.
matlab_mat *matlab_size(matlab_mat *A);
double matlab_size_dim(matlab_mat *A, double dim);
double matlab_length(matlab_mat *A);
double matlab_numel(matlab_mat *A);
double matlab_ndims(matlab_mat *A);
double matlab_end_of_dim(matlab_mat *A, double dim);
double matlab_isempty(matlab_mat *A);
double matlab_isequal(matlab_mat *A, matlab_mat *B);

// Subscripting.
double      matlab_subscript1_s(matlab_mat *A, double i);
double      matlab_subscript2_s(matlab_mat *A, double i, double j);
matlab_mat *matlab_slice1(matlab_mat *A, matlab_mat *idx);
matlab_mat *matlab_slice2(matlab_mat *A, matlab_mat *rows, matlab_mat *cols);
void matlab_slice_store1(matlab_mat *A, matlab_mat *idx, matlab_mat *V);
void matlab_slice_store1_scalar(matlab_mat *A, matlab_mat *idx, double v);
void matlab_slice_store2(matlab_mat *A, matlab_mat *rows, matlab_mat *cols,
                         matlab_mat *V);
void matlab_slice_store2_scalar(matlab_mat *A, matlab_mat *rows,
                                matlab_mat *cols, double v);
matlab_mat *matlab_find(matlab_mat *A);
matlab_mat *matlab_erase_rows(matlab_mat *A, matlab_mat *rows);
matlab_mat *matlab_erase_cols(matlab_mat *A, matlab_mat *cols);

// Scalar math builtins.
double matlab_exp_s(double x);
double matlab_log_s(double x);
double matlab_sin_s(double x);
double matlab_cos_s(double x);
double matlab_tan_s(double x);
double matlab_sqrt_s(double x);
double matlab_abs_s(double x);

// Fixed-Point Designer (fi) — see docs/emit_fixed_point.md §6.2.
// Overflow modes: 0 = Wrap, 1 = Saturate.
// Rounding modes: 0 = Floor, 1 = Nearest, 2 = Zero, 3 = Convergent, 4 = Ceiling.
// Phase 1 ships Floor + Nearest; the others trip matlab_set_error and return 0.
int64_t  matlab_fi_sat_s64(int64_t x, uint8_t WL);
uint64_t matlab_fi_sat_u64(uint64_t x, uint8_t WL);
int64_t  matlab_fi_round_floor_s(int64_t x, uint8_t shift);
int64_t  matlab_fi_round_nearest_s(int64_t x, uint8_t shift);
uint64_t matlab_fi_round_floor_u(uint64_t x, uint8_t shift);
uint64_t matlab_fi_round_nearest_u(uint64_t x, uint8_t shift);
// Phase 5 rounding modes.
int64_t  matlab_fi_round_zero_s(int64_t x, uint8_t shift);
uint64_t matlab_fi_round_zero_u(uint64_t x, uint8_t shift);
int64_t  matlab_fi_round_ceiling_s(int64_t x, uint8_t shift);
uint64_t matlab_fi_round_ceiling_u(uint64_t x, uint8_t shift);
int64_t  matlab_fi_round_convergent_s(int64_t x, uint8_t shift);
uint64_t matlab_fi_round_convergent_u(uint64_t x, uint8_t shift);
int64_t  matlab_fi_quantize_s(double v, uint8_t WL, int8_t FL,
                              uint8_t overflow, uint8_t rounding);
uint64_t matlab_fi_quantize_u(double v, uint8_t WL, int8_t FL,
                              uint8_t overflow, uint8_t rounding);
void     matlab_fi_disp_s(int64_t  stored, uint8_t WL, int8_t FL);
void     matlab_fi_disp_u(uint64_t stored, uint8_t WL, int8_t FL);

// Typed integer matrix descriptors for `fi` arrays — see plan §6.3 and
// docs/emit_fixed_point.md. Phase 3 ships 64-bit lanes only; tighter
// lanes (i32/i16/i8) come later. Same row-major layout as matlab_mat.
typedef struct matlab_mat_i64 matlab_mat_i64;
typedef struct matlab_mat_u64 matlab_mat_u64;

// Constructors.
matlab_mat_i64 *matlab_mat_i64_zeros(double rows, double cols);
matlab_mat_i64 *matlab_mat_i64_from_buf(const int64_t *buf, double rows, double cols);
matlab_mat_i64 *matlab_mat_i64_from_scalar(int64_t v);
matlab_mat_u64 *matlab_mat_u64_zeros(double rows, double cols);
matlab_mat_u64 *matlab_mat_u64_from_buf(const uint64_t *buf, double rows, double cols);
matlab_mat_u64 *matlab_mat_u64_from_scalar(uint64_t v);

// Shape / predicates.
double  matlab_mat_i64_length(matlab_mat_i64 *A);
double  matlab_mat_i64_numel (matlab_mat_i64 *A);
double  matlab_mat_i64_size_dim(matlab_mat_i64 *A, double dim);
int64_t matlab_mat_i64_rows  (matlab_mat_i64 *A);
int64_t matlab_mat_i64_cols  (matlab_mat_i64 *A);
double  matlab_mat_u64_length(matlab_mat_u64 *A);
double  matlab_mat_u64_numel (matlab_mat_u64 *A);
double  matlab_mat_u64_size_dim(matlab_mat_u64 *A, double dim);

// Indexing.
int64_t  matlab_mat_i64_subscript1_s(matlab_mat_i64 *A, double i);
int64_t  matlab_mat_i64_subscript2_s(matlab_mat_i64 *A, double i, double j);
uint64_t matlab_mat_u64_subscript1_s(matlab_mat_u64 *A, double i);
uint64_t matlab_mat_u64_subscript2_s(matlab_mat_u64 *A, double i, double j);
matlab_mat_i64 *matlab_mat_i64_slice1(matlab_mat_i64 *A, matlab_mat *idx);
matlab_mat_u64 *matlab_mat_u64_slice1(matlab_mat_u64 *A, matlab_mat *idx);

// In-place scalar store (used by `A(i) = v` and persistent updates).
void matlab_mat_i64_set1_s(matlab_mat_i64 *A, double i, int64_t  v);
void matlab_mat_u64_set1_s(matlab_mat_u64 *A, double i, uint64_t v);

// Fill every element with a constant stored value (used by fi(ones(...))).
void matlab_mat_i64_fill(matlab_mat_i64 *A, int64_t  v);
void matlab_mat_u64_fill(matlab_mat_u64 *A, uint64_t v);

// Concat (1-D, vector style — `[x, A(1:end-1)]`).
matlab_mat_i64 *matlab_mat_i64_concat_row(matlab_mat_i64 *A, matlab_mat_i64 *B);
matlab_mat_u64 *matlab_mat_u64_concat_row(matlab_mat_u64 *A, matlab_mat_u64 *B);

// Reductions (return scalar stored int).
int64_t  matlab_mat_i64_sum(matlab_mat_i64 *A);
uint64_t matlab_mat_u64_sum(matlab_mat_u64 *A);

// disp — render every element via the fi disp helper (real-world double).
void matlab_mat_i64_disp(matlab_mat_i64 *A, uint8_t WL, int8_t FL);
void matlab_mat_u64_disp(matlab_mat_u64 *A, uint8_t WL, int8_t FL);

// Native integer matrix descriptors (Phase 1.1, Option B).
// Storage is row-major. Saturation lives at the cast/arith boundary
// (Phase 1.1.B); these primitives are pure storage ops.
typedef struct matlab_mat_u8  matlab_mat_u8;
typedef struct matlab_mat_i32 matlab_mat_i32;

// Constructors.
matlab_mat_u8  *matlab_mat_u8_zeros (double rows, double cols);
matlab_mat_u8  *matlab_mat_u8_ones  (double rows, double cols);
matlab_mat_u8  *matlab_mat_u8_eye   (double rows, double cols);
matlab_mat_u8  *matlab_mat_u8_from_buf   (const uint8_t *buf, double r, double c);
matlab_mat_u8  *matlab_mat_u8_from_scalar(uint8_t v);
matlab_mat_i32 *matlab_mat_i32_zeros(double rows, double cols);
matlab_mat_i32 *matlab_mat_i32_ones (double rows, double cols);
matlab_mat_i32 *matlab_mat_i32_eye  (double rows, double cols);
matlab_mat_i32 *matlab_mat_i32_from_buf   (const int32_t *buf, double r, double c);
matlab_mat_i32 *matlab_mat_i32_from_scalar(int32_t v);

// Shape / predicates.
double  matlab_mat_u8_length  (matlab_mat_u8 *A);
double  matlab_mat_u8_numel   (matlab_mat_u8 *A);
double  matlab_mat_u8_size_dim(matlab_mat_u8 *A, double dim);
int64_t matlab_mat_u8_rows    (matlab_mat_u8 *A);
int64_t matlab_mat_u8_cols    (matlab_mat_u8 *A);
double  matlab_mat_i32_length  (matlab_mat_i32 *A);
double  matlab_mat_i32_numel   (matlab_mat_i32 *A);
double  matlab_mat_i32_size_dim(matlab_mat_i32 *A, double dim);
int64_t matlab_mat_i32_rows    (matlab_mat_i32 *A);
int64_t matlab_mat_i32_cols    (matlab_mat_i32 *A);

// Indexing (read).
uint8_t matlab_mat_u8_subscript1_s (matlab_mat_u8 *A, double i);
uint8_t matlab_mat_u8_subscript2_s (matlab_mat_u8 *A, double i, double j);
int32_t matlab_mat_i32_subscript1_s(matlab_mat_i32 *A, double i);
int32_t matlab_mat_i32_subscript2_s(matlab_mat_i32 *A, double i, double j);

// Indexing (write — caller pre-saturates).
void matlab_mat_u8_set1_s (matlab_mat_u8 *A, double i, uint8_t v);
void matlab_mat_u8_set2_s (matlab_mat_u8 *A, double i, double j, uint8_t v);
void matlab_mat_i32_set1_s(matlab_mat_i32 *A, double i, int32_t v);
void matlab_mat_i32_set2_s(matlab_mat_i32 *A, double i, double j, int32_t v);

// Slicing.
matlab_mat_u8  *matlab_mat_u8_slice1 (matlab_mat_u8 *A, matlab_mat *idx);
matlab_mat_u8  *matlab_mat_u8_slice2 (matlab_mat_u8 *A, matlab_mat *rows, matlab_mat *cols);
matlab_mat_i32 *matlab_mat_i32_slice1(matlab_mat_i32 *A, matlab_mat *idx);
matlab_mat_i32 *matlab_mat_i32_slice2(matlab_mat_i32 *A, matlab_mat *rows, matlab_mat *cols);

// Fill + concat.
void matlab_mat_u8_fill (matlab_mat_u8 *A, uint8_t v);
void matlab_mat_i32_fill(matlab_mat_i32 *A, int32_t v);
matlab_mat_u8  *matlab_mat_u8_concat_row (matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat_u8  *matlab_mat_u8_concat_col (matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat_i32 *matlab_mat_i32_concat_row(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat_i32 *matlab_mat_i32_concat_col(matlab_mat_i32 *A, matlab_mat_i32 *B);

// disp — native integer formatting (no decimal).
void matlab_mat_u8_disp (matlab_mat_u8 *A);
void matlab_mat_i32_disp(matlab_mat_i32 *A);

// Casts (matrix forms). Saturating where the destination is narrower
// or signedness changes; widening to double is exact.
matlab_mat_u8  *matlab_mat_u8_from_double (matlab_mat *A);
matlab_mat_i32 *matlab_mat_i32_from_double(matlab_mat *A);
matlab_mat     *matlab_mat_u8_to_double   (matlab_mat_u8 *A);
matlab_mat     *matlab_mat_i32_to_double  (matlab_mat_i32 *A);
matlab_mat_u8  *matlab_mat_u8_from_i32    (matlab_mat_i32 *A);
matlab_mat_i32 *matlab_mat_i32_from_u8    (matlab_mat_u8 *A);

// Scalar saturating casts — used when a typed-int matrix is mixed with a
// double scalar in a binop (`A + 2.5`); the lowering coerces the scalar
// here before calling the typed _ms / _sm runtime entry.
int32_t matlab_d_to_i32_sat(double v);
uint8_t matlab_d_to_u8_sat (double v);

// Typed-int descriptor pointer registry (Phase 1.1.F). Returns -1 if p is
// not a registered typed-int matrix pointer, 0 for matlab_mat_u8 *, 1 for
// matlab_mat_i32 *. The polymorphic matlab_disp_mat / matlab_dbg_ws_kind
// consult this so REPL / DAP display picks the right lane.
int matlab_mat_intlane_kind(const void *p);

// Element-wise arithmetic with MATLAB saturation semantics.
matlab_mat_u8  *matlab_mat_u8_add_mm (matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat_u8  *matlab_mat_u8_add_ms (matlab_mat_u8 *A, uint8_t s);
matlab_mat_u8  *matlab_mat_u8_add_sm (uint8_t s, matlab_mat_u8 *A);
matlab_mat_u8  *matlab_mat_u8_sub_mm (matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat_u8  *matlab_mat_u8_sub_ms (matlab_mat_u8 *A, uint8_t s);
matlab_mat_u8  *matlab_mat_u8_sub_sm (uint8_t s, matlab_mat_u8 *A);
matlab_mat_u8  *matlab_mat_u8_emul_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat_u8  *matlab_mat_u8_emul_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat_u8  *matlab_mat_u8_emul_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat_u8  *matlab_mat_u8_ediv_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat_u8  *matlab_mat_u8_ediv_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat_u8  *matlab_mat_u8_ediv_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat_i32 *matlab_mat_i32_add_mm (matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat_i32 *matlab_mat_i32_add_ms (matlab_mat_i32 *A, int32_t s);
matlab_mat_i32 *matlab_mat_i32_add_sm (int32_t s, matlab_mat_i32 *A);
matlab_mat_i32 *matlab_mat_i32_sub_mm (matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat_i32 *matlab_mat_i32_sub_ms (matlab_mat_i32 *A, int32_t s);
matlab_mat_i32 *matlab_mat_i32_sub_sm (int32_t s, matlab_mat_i32 *A);
matlab_mat_i32 *matlab_mat_i32_emul_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat_i32 *matlab_mat_i32_emul_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat_i32 *matlab_mat_i32_emul_sm(int32_t s, matlab_mat_i32 *A);
matlab_mat_i32 *matlab_mat_i32_ediv_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat_i32 *matlab_mat_i32_ediv_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat_i32 *matlab_mat_i32_ediv_sm(int32_t s, matlab_mat_i32 *A);

// Element-wise comparisons (return matlab_mat with 0/1 doubles).
matlab_mat *matlab_mat_u8_gt_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat *matlab_mat_u8_gt_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat *matlab_mat_u8_gt_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat *matlab_mat_u8_ge_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat *matlab_mat_u8_ge_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat *matlab_mat_u8_ge_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat *matlab_mat_u8_lt_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat *matlab_mat_u8_lt_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat *matlab_mat_u8_lt_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat *matlab_mat_u8_le_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat *matlab_mat_u8_le_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat *matlab_mat_u8_le_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat *matlab_mat_u8_eq_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat *matlab_mat_u8_eq_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat *matlab_mat_u8_eq_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat *matlab_mat_u8_ne_mm(matlab_mat_u8 *A, matlab_mat_u8 *B);
matlab_mat *matlab_mat_u8_ne_ms(matlab_mat_u8 *A, uint8_t s);
matlab_mat *matlab_mat_u8_ne_sm(uint8_t s, matlab_mat_u8 *A);
matlab_mat *matlab_mat_i32_gt_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat *matlab_mat_i32_gt_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat *matlab_mat_i32_gt_sm(int32_t s, matlab_mat_i32 *A);
matlab_mat *matlab_mat_i32_ge_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat *matlab_mat_i32_ge_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat *matlab_mat_i32_ge_sm(int32_t s, matlab_mat_i32 *A);
matlab_mat *matlab_mat_i32_lt_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat *matlab_mat_i32_lt_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat *matlab_mat_i32_lt_sm(int32_t s, matlab_mat_i32 *A);
matlab_mat *matlab_mat_i32_le_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat *matlab_mat_i32_le_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat *matlab_mat_i32_le_sm(int32_t s, matlab_mat_i32 *A);
matlab_mat *matlab_mat_i32_eq_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat *matlab_mat_i32_eq_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat *matlab_mat_i32_eq_sm(int32_t s, matlab_mat_i32 *A);
matlab_mat *matlab_mat_i32_ne_mm(matlab_mat_i32 *A, matlab_mat_i32 *B);
matlab_mat *matlab_mat_i32_ne_ms(matlab_mat_i32 *A, int32_t s);
matlab_mat *matlab_mat_i32_ne_sm(int32_t s, matlab_mat_i32 *A);

// Reductions returning same-type scalars.
uint8_t matlab_mat_u8_sum (matlab_mat_u8 *A);
uint8_t matlab_mat_u8_mean(matlab_mat_u8 *A);
uint8_t matlab_mat_u8_min (matlab_mat_u8 *A);
uint8_t matlab_mat_u8_max (matlab_mat_u8 *A);
int32_t matlab_mat_i32_sum (matlab_mat_i32 *A);
int32_t matlab_mat_i32_mean(matlab_mat_i32 *A);
int32_t matlab_mat_i32_min (matlab_mat_i32 *A);
int32_t matlab_mat_i32_max (matlab_mat_i32 *A);

// bin(n) / hex(n) / dec(n) — render the stored integer as a matlab_string.
// Each helper allocates a heap-owned descriptor; the caller passes it on
// to disp/strlen/etc. through the regular string-binding path.
void *matlab_fi_bin_s(int64_t  stored, uint8_t WL);
void *matlab_fi_bin_u(uint64_t stored, uint8_t WL);
void *matlab_fi_hex_s(int64_t  stored, uint8_t WL);
void *matlab_fi_hex_u(uint64_t stored, uint8_t WL);
void *matlab_fi_dec_s(int64_t  stored, uint8_t WL);
void *matlab_fi_dec_u(uint64_t stored, uint8_t WL);

// Try/catch error flag.
void    matlab_set_error(void);
int32_t matlab_check_error(void);
void    matlab_clear_error(void);

// Structs.
matlab_struct *matlab_struct_new(void);
void matlab_struct_set_f64(matlab_struct *s, const char *name, int64_t len,
                           double v);
void matlab_struct_set_mat(matlab_struct *s, const char *name, int64_t len,
                           matlab_mat *m);
double matlab_struct_get_f64(matlab_struct *s, const char *name, int64_t len);
matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name,
                                  int64_t len);
double matlab_struct_has_field(matlab_struct *s, const char *name, int64_t len);
matlab_struct *matlab_struct_get_child_struct(matlab_struct *s,
                                              const char *name, int64_t len);

// Cells.
matlab_cell *matlab_cell_new(double n);
void matlab_cell_set_f64(matlab_cell *c, double i1, double v);
void matlab_cell_set_mat(matlab_cell *c, double i1, matlab_mat *m);
double matlab_cell_get_f64(matlab_cell *c, double i1);
matlab_mat *matlab_cell_get_mat(matlab_cell *c, double i1);
double matlab_cell_numel(matlab_cell *c);
double matlab_iscell(matlab_cell *c);

/* Phase 1.3 — 2-D cells. Row-major layout; the legacy 1-D accessors
 * keep working on cells of any shape (linear index across the
 * row-major buffer). Bracket cell-concat builds a fresh cell that
 * borrows the source elements' pointers. */
matlab_cell *matlab_cell_new_2d(double rows, double cols);
double  matlab_cell_rows(matlab_cell *c);
double  matlab_cell_cols(matlab_cell *c);
double  matlab_cell_size_dim(matlab_cell *c, double dim);
void    matlab_cell_set_f64_2d(matlab_cell *c, double r1, double k1, double v);
void    matlab_cell_set_mat_2d(matlab_cell *c, double r1, double k1, matlab_mat *m);
double  matlab_cell_get_f64_2d(matlab_cell *c, double r1, double k1);
matlab_mat *matlab_cell_get_mat_2d(matlab_cell *c, double r1, double k1);
matlab_cell *matlab_cell_concat_row(matlab_cell *a, matlab_cell *b);
matlab_cell *matlab_cell_concat_col(matlab_cell *a, matlab_cell *b);

// Global / persistent.
double matlab_global_get_f64(int32_t id);
void   matlab_global_set_f64(int32_t id, double v);
// Typed (pointer) persistent slots — used by fi-array persistents and
// any future heap-backed persistent types. See plan §12.
void  *matlab_persistent_get_ptr(int32_t id);
void   matlab_persistent_set_ptr(int32_t id, void *p);
double matlab_persistent_isempty(int32_t id);

// Complex numbers.
matlab_mat_c *matlab_complex_scalar(double re, double im);
matlab_mat_c *matlab_mat_c_from_real(matlab_mat *A);
matlab_mat_c *matlab_mat_c_from_buf(const double *re, const double *im,
                                     double m, double n);
/* Polymorphic: accept real matlab_mat* or complex matlab_mat_c*. */
matlab_mat_c *matlab_conj_c(void *A);
matlab_mat_c *matlab_neg_c(matlab_mat_c *A);
matlab_mat   *matlab_real_c(void *A);
matlab_mat   *matlab_imag_c(void *A);
matlab_mat   *matlab_angle_c(void *A);
matlab_mat   *matlab_abs_c(void *A);
matlab_mat_c *matlab_add_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_sub_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_emul_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_ediv_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_matmul_cc(matlab_mat_c *A, matlab_mat_c *B);
matlab_mat_c *matlab_transpose_c(matlab_mat_c *A);
matlab_mat_c *matlab_ctranspose_c(matlab_mat_c *A);
void          matlab_disp_mat_c(matlab_mat_c *A);

// FFT — pure-C Cooley-Tukey (radix-2 + Bluestein for general N). All
// accept either a real matlab_mat* or a complex matlab_mat_c*.
matlab_mat_c *matlab_fft_c(void *A);
matlab_mat_c *matlab_ifft_c(void *A);
matlab_mat_c *matlab_fft2_c(void *A);
matlab_mat_c *matlab_ifft2_c(void *A);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MATLAB_RUNTIME_H
