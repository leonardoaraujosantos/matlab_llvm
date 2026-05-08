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

/* Initial-value ODE solvers — adaptive single-step methods on scalar y.
 * Both ode45 (Dormand–Prince 5(4)) and ode23 (Bogacki–Shampine 3(2))
 * follow the MATLAB call shape:
 *   [t, y] = ode45(@f, [t0 tf], y0)        % scalar y0 only (Phase 1)
 * `f` is a function handle with signature `double f(double t, double y)`.
 * `tspan` is a row vector with at least the two integration endpoints;
 * `y0` is a scalar initial condition. Output `t` and `y` are column
 * vectors of the same length.
 *
 * If tspan has more than two elements (`[t0 t1 t2 ... tN]`) the
 * integrator emits y at *exactly* those times via cubic-Hermite dense
 * output, matching MATLAB. The Refine option is ignored in this mode.
 *
 * Defaults match MATLAB: rtol = 1e-3, atol = 1e-6, max-steps = 100000.
 * Vector y is a planned follow-up (would change the handle ABI).
 *
 * Each [t,y] = ode45(...) site is split by the lowering pass into two
 * single-output runtime calls (matlab_ode45_t / matlab_ode45_y). The
 * implementation memoises the most-recent (handle, tspan, y0) call so
 * the second leg returns the cached y without re-integrating. */
typedef double (*matlab_ode_rhs)(double t, double y);
matlab_mat *matlab_ode45_t(matlab_ode_rhs f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode45_y(matlab_ode_rhs f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23_t(matlab_ode_rhs f, matlab_mat *tspan, double y0);
matlab_mat *matlab_ode23_y(matlab_ode_rhs f, matlab_mat *tspan, double y0);

/* 4-arg form: `[t,y] = ode45(@f, tspan, y0, opts)`. `opts` is a struct
 * built MATLAB-style (`opts.RelTol = 1e-6; opts.AbsTol = 1e-9;`); a
 * subset of MATLAB's odeset is honoured: RelTol, AbsTol, MaxStep,
 * InitialStep, Refine. Missing fields fall back to the 3-arg defaults
 * (1e-3 / 1e-6) and the built-in step heuristics. Default Refine is
 * 4 for ode45, 1 for ode23 (matching MATLAB). */
matlab_mat *matlab_ode45_t_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);
matlab_mat *matlab_ode45_y_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);
matlab_mat *matlab_ode23_t_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);
matlab_mat *matlab_ode23_y_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                 double y0, matlab_struct *opts);

/* 3-return form: `[t, y, stats] = ode45(@f, tspan, y0[, opts])`.
 * `stats` is a freshly-allocated struct with fields nsteps / nfailed /
 * nfevals (matching MATLAB's solver-stats output). The companion
 * matlab_ode*_stats_opts variants take an opts struct as the 4th arg. */
matlab_struct *matlab_ode45_stats(matlab_ode_rhs f, matlab_mat *tspan,
                                    double y0);
matlab_struct *matlab_ode45_stats_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                         double y0, matlab_struct *opts);
matlab_struct *matlab_ode23_stats(matlab_ode_rhs f, matlab_mat *tspan,
                                    double y0);
matlab_struct *matlab_ode23_stats_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                         double y0, matlab_struct *opts);

/* ode23s — Rosenbrock 2(3) stiff solver (Shampine). Same call shapes as
 * ode45 / ode23 but appropriate for problems with widely-separated time
 * constants. Uses one numerical-FD Jacobian per accepted step plus three
 * linear solves; the implicit factor (I - h*d*J) absorbs stiff modes
 * that would force tiny explicit steps. Scalar and vector forms. */
matlab_mat    *matlab_ode23s_t(matlab_ode_rhs f, matlab_mat *tspan, double y0);
matlab_mat    *matlab_ode23s_y(matlab_ode_rhs f, matlab_mat *tspan, double y0);
matlab_mat    *matlab_ode23s_t_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                     double y0, matlab_struct *opts);
matlab_mat    *matlab_ode23s_y_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                     double y0, matlab_struct *opts);
matlab_struct *matlab_ode23s_stats(matlab_ode_rhs f, matlab_mat *tspan,
                                    double y0);
matlab_struct *matlab_ode23s_stats_opts(matlab_ode_rhs f, matlab_mat *tspan,
                                         double y0, matlab_struct *opts);

/* Vector-y solvers — system of ODEs. Same Dormand-Prince / Bogacki-
 * Shampine pair as the scalar path; user RHS takes a Dx1 column matrix
 * and returns a Dx1 column with dy/dt. Output `y` is N rows × D cols
 * (MATLAB convention: y(i, :) is the state at t(i)). */
typedef matlab_mat *(*matlab_ode_rhs_v)(double t, matlab_mat *y);

matlab_mat    *matlab_ode23s_v_t(matlab_ode_rhs_v f, matlab_mat *tspan,
                                  matlab_mat *y0);
matlab_mat    *matlab_ode23s_v_y(matlab_ode_rhs_v f, matlab_mat *tspan,
                                  matlab_mat *y0);
matlab_mat    *matlab_ode23s_v_t_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                       matlab_mat *y0, matlab_struct *opts);
matlab_mat    *matlab_ode23s_v_y_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                       matlab_mat *y0, matlab_struct *opts);
matlab_struct *matlab_ode23s_v_stats(matlab_ode_rhs_v f, matlab_mat *tspan,
                                      matlab_mat *y0);
matlab_struct *matlab_ode23s_v_stats_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                            matlab_mat *y0, matlab_struct *opts);

/* ode_events — IVP solver with event detection. v1: scalar y, single
 * event. The event function returns a 3×1 column [value; isterminal;
 * direction]. The 5-result form is split by the lowering pass into
 * five paired runtime calls (matlab_ode_events_{t,y,te,ye,ie}) sharing
 * a thread-local cache. */
matlab_mat *matlab_ode_events_t (matlab_ode_rhs f, matlab_mat *tspan,
                                  double y0, void *evt);
matlab_mat *matlab_ode_events_y (matlab_ode_rhs f, matlab_mat *tspan,
                                  double y0, void *evt);
matlab_mat *matlab_ode_events_te(matlab_ode_rhs f, matlab_mat *tspan,
                                  double y0, void *evt);
matlab_mat *matlab_ode_events_ye(matlab_ode_rhs f, matlab_mat *tspan,
                                  double y0, void *evt);
matlab_mat *matlab_ode_events_ie(matlab_ode_rhs f, matlab_mat *tspan,
                                  double y0, void *evt);

/* pdepe — 1-D parabolic-elliptic PDE solver via method-of-lines.
 *   sol = pdepe(m, @pdefun, @icfun, @bcfun, xmesh, tspan)
 *
 * v1 scope: m = 0 (Cartesian), scalar PDE, Dirichlet BCs (ql = qr = 0).
 * Spatial discretisation runs on the user-supplied (possibly non-
 * uniform) xmesh; the resulting interior ODE system is integrated by
 * ode23s_v, so stiff parabolic problems work without manual tuning.
 *
 * Function-pointer ABIs (the anon-function shapes our outliner emits):
 *   pdefn:  matlab_mat *(*)(double x, double t, double u, double dudx)
 *           returning [c; f; s] as a 3×1 column.
 *   icfn:   double (*)(double x)
 *   bcfn:   matlab_mat *(*)(double xl, double ul, double xr, double ur, double t)
 *           returning [pl; ql; pr; qr]; ql == qr == 0 required.
 *
 * Output sol is N_t × N_x with sol(i, j) = u(t_i, x_j). For unsupported
 * cases (m ≠ 0, or Neumann/Robin BCs) returns a 0×0 matrix. */
matlab_mat *matlab_pdepe(double m, void *pdefn, void *icfn, void *bcfn,
                          matlab_mat *xmesh, matlab_mat *tspan);

matlab_mat *matlab_ode45_v_t(matlab_ode_rhs_v f, matlab_mat *tspan,
                              matlab_mat *y0);
matlab_mat *matlab_ode45_v_y(matlab_ode_rhs_v f, matlab_mat *tspan,
                              matlab_mat *y0);
matlab_mat *matlab_ode23_v_t(matlab_ode_rhs_v f, matlab_mat *tspan,
                              matlab_mat *y0);
matlab_mat *matlab_ode23_v_y(matlab_ode_rhs_v f, matlab_mat *tspan,
                              matlab_mat *y0);
matlab_mat *matlab_ode45_v_t_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                   matlab_mat *y0, matlab_struct *opts);
matlab_mat *matlab_ode45_v_y_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                   matlab_mat *y0, matlab_struct *opts);
matlab_mat *matlab_ode23_v_t_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                   matlab_mat *y0, matlab_struct *opts);
matlab_mat *matlab_ode23_v_y_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                   matlab_mat *y0, matlab_struct *opts);
matlab_struct *matlab_ode45_v_stats(matlab_ode_rhs_v f, matlab_mat *tspan,
                                     matlab_mat *y0);
matlab_struct *matlab_ode45_v_stats_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                           matlab_mat *y0, matlab_struct *opts);
matlab_struct *matlab_ode23_v_stats(matlab_ode_rhs_v f, matlab_mat *tspan,
                                     matlab_mat *y0);
matlab_struct *matlab_ode23_v_stats_opts(matlab_ode_rhs_v f, matlab_mat *tspan,
                                           matlab_mat *y0, matlab_struct *opts);

/* polyder(p), polyint(p[, k]) — derivative / antiderivative of a
 * polynomial whose coefficients are p (highest-power-first). Both
 * return a row vector. polyint without a constant treats k = 0. */
matlab_mat *matlab_polyder(matlab_mat *p);
matlab_mat *matlab_polyint(matlab_mat *p);
matlab_mat *matlab_polyint_k(matlab_mat *p, double k);

/* poly(r) — coefficients of the monic polynomial with roots r. Accepts
 * either a real or complex vector of roots; the returned coefficients
 * are real with any residual imaginary part dropped. Output is a
 * 1 × (n+1) row vector. */
matlab_mat *matlab_poly(void *r);

/* IIR filter design (Tier-1 SPT §2.1) — lowpass scope.
 *
 *   [b, a] = butter(n, Wn)         digital Butterworth lowpass
 *   [b, a] = cheby1(n, Rp, Wn)     digital Chebyshev I lowpass
 *   H      = freqz(b, a, N)        complex frequency response (Nx1)
 *   [H, w] = freqz(b, a, N)        + frequency-axis vector
 *
 * Multi-return is split into independent runtime entries
 * matlab_<filt>_b / _a (eig precedent). Wn is normalized in [0, 1]
 * with 1 = Nyquist. cheby1 takes the passband ripple Rp in dB.
 */
matlab_mat   *matlab_butter_b(double n, double Wn);
matlab_mat   *matlab_butter_a(double n, double Wn);
matlab_mat   *matlab_cheby1_b(double n, double Rp, double Wn);
matlab_mat   *matlab_cheby1_a(double n, double Rp, double Wn);
matlab_mat   *matlab_cheby2_b(double n, double Rs, double Wn);
matlab_mat   *matlab_cheby2_a(double n, double Rs, double Wn);
/* Band variants — high/bandpass/stop. Bandpass / bandstop take W1 / W2
 * as separate doubles (the LowerTensorOps dispatch unpacks the matrix
 * Wn = [W1 W2] before the call). */
matlab_mat   *matlab_butter_hp_b(double n, double Wn);
matlab_mat   *matlab_butter_hp_a(double n, double Wn);
matlab_mat   *matlab_butter_bp_b(double n, double W1, double W2);
matlab_mat   *matlab_butter_bp_a(double n, double W1, double W2);
matlab_mat   *matlab_butter_bs_b(double n, double W1, double W2);
matlab_mat   *matlab_butter_bs_a(double n, double W1, double W2);
matlab_mat   *matlab_cheby1_hp_b(double n, double Rp, double Wn);
matlab_mat   *matlab_cheby1_hp_a(double n, double Rp, double Wn);
matlab_mat   *matlab_cheby1_bp_b(double n, double Rp, double W1, double W2);
matlab_mat   *matlab_cheby1_bp_a(double n, double Rp, double W1, double W2);
matlab_mat   *matlab_cheby1_bs_b(double n, double Rp, double W1, double W2);
matlab_mat   *matlab_cheby1_bs_a(double n, double Rp, double W1, double W2);
matlab_mat   *matlab_cheby2_hp_b(double n, double Rs, double Wn);
matlab_mat   *matlab_cheby2_hp_a(double n, double Rs, double Wn);
matlab_mat   *matlab_cheby2_bp_b(double n, double Rs, double W1, double W2);
matlab_mat   *matlab_cheby2_bp_a(double n, double Rs, double W1, double W2);
matlab_mat   *matlab_cheby2_bs_b(double n, double Rs, double W1, double W2);
matlab_mat   *matlab_cheby2_bs_a(double n, double Rs, double W1, double W2);
/* §2.1 follow-on — standalone analog↔digital + form conversions. */
matlab_mat   *matlab_bilinear_b(matlab_mat *b, matlab_mat *a, double fs);
matlab_mat   *matlab_bilinear_a(matlab_mat *b, matlab_mat *a, double fs);
matlab_mat_c *matlab_freqs(matlab_mat *b, matlab_mat *a, matlab_mat *w);
matlab_mat_c *matlab_tf2zp_z(matlab_mat *b, matlab_mat *a);
matlab_mat_c *matlab_tf2zp_p(matlab_mat *b, matlab_mat *a);
double        matlab_tf2zp_k(matlab_mat *b, matlab_mat *a);
matlab_mat   *matlab_zp2tf_b(matlab_mat_c *z, matlab_mat_c *p, double k);
matlab_mat   *matlab_zp2tf_a(matlab_mat_c *z, matlab_mat_c *p, double k);
double        matlab_cheb2ord_n(double Wp, double Ws, double Rp, double Rs);
double        matlab_cheb2ord_Wn(double Wp, double Ws, double Rp, double Rs);
matlab_mat   *matlab_besself_b(double n, double Wo);
matlab_mat   *matlab_besself_a(double n, double Wo);
matlab_mat   *matlab_tf2sos(matlab_mat *b, matlab_mat *a);
matlab_mat   *matlab_sos2tf_b(matlab_mat *sos);
matlab_mat   *matlab_sos2tf_a(matlab_mat *sos);
matlab_mat_c *matlab_freqz(matlab_mat *b, matlab_mat *a, double N);
matlab_mat_c *matlab_freqz_h(matlab_mat *b, matlab_mat *a, double N);
matlab_mat   *matlab_freqz_w(matlab_mat *b, matlab_mat *a, double N);

/* Order-selection helpers — return [n, Wn] via paired *_n / *_Wn
 * entries (each computes the full result internally). Lowpass scope. */
double matlab_buttord_n(double Wp, double Ws, double Rp, double Rs);
double matlab_buttord_Wn(double Wp, double Ws, double Rp, double Rs);
double matlab_cheb1ord_n(double Wp, double Ws, double Rp, double Rs);
double matlab_cheb1ord_Wn(double Wp, double Ws, double Rp, double Rs);

/* Tier-2 §3.1 nonparametric spectral estimation. Single-output form,
 * default fs = 1 (normalised). The 2-return [Pxx, f] form is a
 * follow-on. */
matlab_mat   *matlab_periodogram(matlab_mat *x);
matlab_mat   *matlab_pwelch(matlab_mat *x, matlab_mat *win, double noverlap);
/* spectrogram (Tier-2 §3.3) — single-output (M × K) magnitude-squared
 * STFT per (freq, frame). Default fs = 1 (normalised). */
matlab_mat   *matlab_spectrogram(matlab_mat *x, matlab_mat *win, double noverlap);

/* Tier-3 §4.4 alignment helpers — xcov / finddelay / dtw.
 * alignsignals (multi-return) is a follow-on. */
matlab_mat *matlab_xcov(matlab_mat *x, matlab_mat *y);
double      matlab_finddelay_s(matlab_mat *x, matlab_mat *y);
double      matlab_dtw_s(matlab_mat *x, matlab_mat *y);

/* Tier-3 §4.2 waveform generators — chirp / sawtooth / square / pulses /
 * sinc. All take a time-vector argument and return same-shape signal. */
matlab_mat *matlab_chirp(matlab_mat *t, double f0, double t1, double f1);
matlab_mat *matlab_sawtooth(matlab_mat *t, double w);
matlab_mat *matlab_square(matlab_mat *t, double duty);
matlab_mat *matlab_gauspuls(matlab_mat *t, double fc, double bw);
matlab_mat *matlab_rectpuls(matlab_mat *t, double w);
matlab_mat *matlab_tripuls(matlab_mat *t, double w);
matlab_mat *matlab_sinc(matlab_mat *x);

/* Tier-3 §4.1 real multirate — proper anti-aliased versions
 * complementing the toy upsample / downsample stubs. */
matlab_mat *matlab_upfirdn(matlab_mat *x, matlab_mat *h, double p, double q);
matlab_mat *matlab_decimate(matlab_mat *x, double r);
matlab_mat *matlab_interp(matlab_mat *x, double r);
matlab_mat *matlab_resample(matlab_mat *x, double p, double q);

/* Tier-3 §4.3 pulse measurements — findpeaks + scalar reductions. */
matlab_mat *matlab_findpeaks_pks(matlab_mat *x);
matlab_mat *matlab_findpeaks_locs(matlab_mat *x);
double      matlab_rms_s(matlab_mat *x);
double      matlab_peak2peak_s(matlab_mat *x);
double      matlab_peak2rms_s(matlab_mat *x);
double      matlab_rssq_s(matlab_mat *x);
matlab_mat *matlab_medfilt1(matlab_mat *x, double n);
matlab_mat *matlab_hampel(matlab_mat *x, double k);
matlab_mat *matlab_envelope(matlab_mat *x);
matlab_mat *matlab_midcross(matlab_mat *x);
double      matlab_risetime_s(matlab_mat *x);
double      matlab_falltime_s(matlab_mat *x);
double      matlab_dutycycle_s(matlab_mat *x);
/* Tier-3 §4.3 tail — pulse-statistics follow-on. */
matlab_mat *matlab_statelevels(matlab_mat *x);
double      matlab_slewrate_s(matlab_mat *x);
double      matlab_pulseperiod_s(matlab_mat *x);
double      matlab_pulsewidth_s(matlab_mat *x);
double      matlab_overshoot_s(matlab_mat *x);
double      matlab_undershoot_s(matlab_mat *x);
double      matlab_settlingtime_s(matlab_mat *x, double d);

/* Tier-2 §3.2 linear prediction. */
matlab_mat   *matlab_levinson(matlab_mat *r, double p);
matlab_mat   *matlab_lpc(matlab_mat *x, double p);
matlab_mat   *matlab_aryule(matlab_mat *x, double p);
matlab_mat   *matlab_arburg(matlab_mat *x, double p);
matlab_mat   *matlab_pyulear(matlab_mat *x, double p, double N);
matlab_mat   *matlab_pburg(matlab_mat *x, double p, double N);

/* Tier-2 §3.1 cross-spectral helpers (Welch-based). */
matlab_mat_c *matlab_cpsd(matlab_mat *x, matlab_mat *y, matlab_mat *win, double noverlap);
matlab_mat   *matlab_mscohere(matlab_mat *x, matlab_mat *y, matlab_mat *win, double noverlap);
matlab_mat_c *matlab_tfestimate(matlab_mat *x, matlab_mat *y, matlab_mat *win, double noverlap);

/* Tier-2 §3.4 transforms — DCT-II / DCT-III / Walsh-Hadamard /
 * Hilbert (analytic signal) / Goertzel (single-bin DFT). */
matlab_mat   *matlab_dct(matlab_mat *x);
matlab_mat   *matlab_idct(matlab_mat *X);
matlab_mat   *matlab_fwht(matlab_mat *x);
matlab_mat_c *matlab_hilbert(matlab_mat *x);
matlab_mat_c *matlab_goertzel(matlab_mat *x, double k);

/* Close-the-loop helpers (Tier-1 §2.5).
 *   filtfilt(b, a, x)  — forward-backward zero-phase IIR filtering
 *   sosfilt(sos, x)    — cascade of biquad second-order sections
 *   impz(b, a, N)      — impulse response (Nx1)
 *   stepz(b, a, N)     — step response (Nx1)
 *   grpdelay(b, a, N)  — group delay τ(ω) via finite-difference phase
 */
matlab_mat *matlab_filtfilt(matlab_mat *b, matlab_mat *a, matlab_mat *x);
matlab_mat *matlab_sosfilt(matlab_mat *sos, matlab_mat *x);
matlab_mat *matlab_impz(matlab_mat *b, matlab_mat *a, double N);
matlab_mat *matlab_stepz(matlab_mat *b, matlab_mat *a, double N);
matlab_mat *matlab_grpdelay(matlab_mat *b, matlab_mat *a, double N);

/* FIR design (Tier-1 §2.2) — lowpass scope.
 *
 *   b = fir1(n, Wn)          windowed-sinc lowpass FIR (default Hamming
 *                            window). Returns 1×(n+1) impulse response.
 *   B = sgolay(k, f)         Savitzky-Golay (f × f) projection matrix.
 *                            f must be odd; coerced if even.
 *   y = sgolayfilt(x, k, f)  Apply Savitzky-Golay smoothing to x.
 */
matlab_mat *matlab_fir1(double n, double Wn);
matlab_mat *matlab_sgolay(double k, double f);
matlab_mat *matlab_sgolayfilt(matlab_mat *x, double k, double f);

/* [r, p, k] = residue(b, a) — partial-fraction expansion of B(s)/A(s).
 * Distinct-pole scope (Tier-1): repeated poles produce numerically
 * degraded residues. r and p are complex column vectors of length
 * deg(a); k is a real row vector with the polynomial direct term
 * (empty if deg(b) < deg(a)). Each MATLAB output slot binds to its
 * own runtime entry — mirrors the [V, D] = eig(A) precedent. */
matlab_mat_c *matlab_residue_r(matlab_mat *b, matlab_mat *a);
matlab_mat_c *matlab_residue_p(matlab_mat *b, matlab_mat *a);
matlab_mat   *matlab_residue_k(matlab_mat *b, matlab_mat *a);

/* DSP windows. n must be >= 1; returns a column vector of length n.
 * All use the symmetric (non-periodic) form, matching MATLAB's default.
 * Two-arg windows (kaiser, tukeywin, gausswin, chebwin) take their
 * shape parameter as the second double; taylorwin takes (n, nbar, sll). */
matlab_mat *matlab_hamming(double n);
matlab_mat *matlab_hann(double n);
matlab_mat *matlab_blackman(double n);
matlab_mat *matlab_rectwin(double n);
matlab_mat *matlab_triang(double n);
matlab_mat *matlab_bartlett(double n);
matlab_mat *matlab_barthannwin(double n);
matlab_mat *matlab_bohmanwin(double n);
matlab_mat *matlab_parzenwin(double n);
matlab_mat *matlab_nuttallwin(double n);
matlab_mat *matlab_blackmanharris(double n);
matlab_mat *matlab_flattopwin(double n);
matlab_mat *matlab_kaiser(double n, double beta);
matlab_mat *matlab_tukeywin(double n, double r);
matlab_mat *matlab_gausswin(double n, double alpha);
matlab_mat *matlab_chebwin(double n, double r);
matlab_mat *matlab_taylorwin(double n, double nbar, double sll);

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
/* Phase 5.3 — table. A record of named columns where each column is
 * a matlab_mat * (column vector for v1). Constructors and column
 * accessors below; display and shape introspection follow. */
typedef struct matlab_table_s matlab_table;
matlab_table *matlab_table_new(void);
void          matlab_table_add_column(matlab_table *t, const char *name,
                                       int64_t namelen, matlab_mat *col);
matlab_mat   *matlab_table_get_column(matlab_table *t, const char *name,
                                       int64_t namelen);
double        matlab_table_height(matlab_table *t);
double        matlab_table_width(matlab_table *t);
double        matlab_table_numel(matlab_table *t);
double        matlab_table_size_dim(matlab_table *t, double dim);
void          matlab_table_disp(matlab_table *t);

/* Phase 5.2 — categorical. 1-D vector of category indices with a
 * deduplicated, alphabetically-sorted category-name table. */
typedef struct matlab_categorical_s matlab_categorical;
matlab_categorical *matlab_categorical_from_strs(void **strs, int64_t n);
matlab_categorical *matlab_categorical_from_cell(matlab_cell *cell, double n);
double              matlab_categorical_length(matlab_categorical *c);
double              matlab_categorical_numcats(matlab_categorical *c);
double              matlab_categorical_iscategory(matlab_categorical *c, void *key);
/* Returns a matlab_cell *-compatible pointer holding the category
 * names (matlab_string * per slot). The opaque void * keeps the
 * declaration order tolerant of the runtime split. */
void               *matlab_categorical_categories(matlab_categorical *c);
void                matlab_categorical_disp(matlab_categorical *c);
matlab_mat         *matlab_categorical_eq(matlab_categorical *a, matlab_categorical *b);

/* Phase 5.1 — datetime / duration. Both descriptors wrap a double:
 * datetime carries seconds-since-Unix-epoch, duration is a relative
 * span. Display uses MATLAB's default formats; arithmetic forms
 * (datetime - datetime, datetime ± duration, duration ± duration)
 * land via dedicated entries below. */
typedef struct matlab_datetime_s matlab_datetime;
typedef struct matlab_duration_s matlab_duration;
matlab_datetime *matlab_datetime_now(void);
matlab_datetime *matlab_datetime_ymd(double y, double m, double d);
matlab_datetime *matlab_datetime_ymdhms(double y, double m, double d,
                                         double h, double mn, double s);
void             matlab_datetime_disp(matlab_datetime *t);
matlab_duration *matlab_duration_seconds(double n);
matlab_duration *matlab_duration_minutes(double n);
matlab_duration *matlab_duration_hours  (double n);
matlab_duration *matlab_duration_days   (double n);
matlab_duration *matlab_duration_years  (double n);
double           matlab_duration_to_seconds(matlab_duration *d);
double           matlab_duration_to_minutes(matlab_duration *d);
double           matlab_duration_to_hours  (matlab_duration *d);
double           matlab_duration_to_days   (matlab_duration *d);
void             matlab_duration_disp(matlab_duration *d);
matlab_duration *matlab_datetime_sub_datetime(matlab_datetime *a, matlab_datetime *b);
matlab_datetime *matlab_datetime_add_duration(matlab_datetime *a, matlab_duration *d);
matlab_datetime *matlab_datetime_sub_duration(matlab_datetime *a, matlab_duration *d);
matlab_duration *matlab_duration_add(matlab_duration *a, matlab_duration *b);
matlab_duration *matlab_duration_sub(matlab_duration *a, matlab_duration *b);

/* Phase 4 — containers.Map / dictionary. A flat key/value table with
 * mixed key types (f64 or matlab_string *) and value types (f64 or
 * matlab_mat *). v1 backs both `containers.Map` and `dictionary` with
 * the same descriptor. */
typedef struct matlab_dict_s matlab_dict;
matlab_dict *matlab_dict_new(void);
void         matlab_dict_set_str_f64(matlab_dict *d, void *key, double v);
void         matlab_dict_set_str_mat(matlab_dict *d, void *key, matlab_mat *m);
void         matlab_dict_set_num_f64(matlab_dict *d, double k, double v);
void         matlab_dict_set_num_mat(matlab_dict *d, double k, matlab_mat *m);
double       matlab_dict_get_str_f64(matlab_dict *d, void *key);
matlab_mat  *matlab_dict_get_str_mat(matlab_dict *d, void *key);
double       matlab_dict_get_num_f64(matlab_dict *d, double k);
matlab_mat  *matlab_dict_get_num_mat(matlab_dict *d, double k);
double       matlab_dict_has_str(matlab_dict *d, void *key);
double       matlab_dict_has_num(matlab_dict *d, double k);
double       matlab_dict_length(matlab_dict *d);
double       matlab_dict_remove_str(matlab_dict *d, void *key);
double       matlab_dict_remove_num(matlab_dict *d, double k);

/* Phase 2 — struct arrays (`s(i).x`). matlab_struct_arr* holds a
 * vector of matlab_struct* elements; the 1-based indexing path
 * auto-grows on write and returns empty structs on OOB read. */
typedef struct matlab_struct_arr_s matlab_struct_arr;
matlab_struct_arr *matlab_struct_arr_new(void);
matlab_struct    *matlab_struct_arr_get_or_create(matlab_struct_arr *a,
                                                  double i1);
matlab_struct    *matlab_struct_arr_get(matlab_struct_arr *a, double i1);
double            matlab_struct_arr_length(matlab_struct_arr *a);
double            matlab_struct_arr_numel(matlab_struct_arr *a);
double            matlab_struct_arr_size_dim(matlab_struct_arr *a, double dim);

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
