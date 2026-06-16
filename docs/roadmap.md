# Roadmap

Forward-looking work tracker for `matlab_llvm`. Organized by effort
horizon and dependency chain, not strict priority — what gets done next
depends on which items unblock real users.

For shipped work, see [`feature_status.md`](feature_status.md). For
detailed history of each backend, see the per-backend `emit_*.md`
docs.

---

## Conventions

- **Effort** is calendar time at one focused implementation session
  per stage (the existing Phase 5.6.x cadence). A "week" means
  ~5 sessions, not 40 hours.
- **Status** legend:
  - 🔵 not started
  - 🟡 in progress / partial
  - 🟢 done (kept here for context until rolled into `feature_status`)
- **Scope dependency** notes flag items that must land first.

---

## Recently shipped (data-container + multi-return arc)

A focused arc of nine phases closed since the last roadmap refresh.
Authoritative status is in [`feature_status.md`](feature_status.md);
roadmap-side summary:

| Phase | What | Gating test |
|---|---|---|
| 1.1 | Typed `int32` / `uint8` matrix runtime — saturating arith, comparisons, casts, REPL+DAP display, Python+TS parity | `int_matrix_binops.m`, `int_image_filter.m`, `int_pixel_math.m` |
| 1.2 | `varargout` (pure + mixed forms) and plain user-fn multi-return (was broken — both LHS got the same value) | `varargout_basic.m` |
| 1.3 | 2-D cell literals + `C{r,k}` indexing + `[A,B]` / `[A;B]` cell concat | `cell_2d.m` |
| 2 | Struct arrays (`s(i).x`) with auto-grow, `length`/`numel`/`size` | `struct_arr_basic.m` |
| 3 | OOP value-class copy-on-assign for non-`< handle` classes | `value_class_copy.m` |
| 4 | `containers.Map` / `dictionary` — string + numeric keys, f64 + matrix values | `dict_basic.m` |
| 5.1 | scalar `datetime` / `duration` with constructors, display, arithmetic | `datetime_basic.m` |
| 5.2 | 1-D `categorical` from string array — disp / length / iscategory / categories | `categorical_basic.m` |
| 5.3 | `table` — auto-named + `'VariableNames'`, dot column access, dynamic add, `height`/`width`/`disp` | `table_basic.m` |
| 6   | **Symbolic Math Toolbox** via [SymPP](https://github.com/leonardoaraujosantos/SymPP) — `syms`, `diff`, `int`, `simplify`, `expand`, `factor`, `subs`, `solve`, `vpa`, `taylor`, `limit`, `dsolve`, `pdsolve`, `laplace`/`fourier`/`ztrans` (+ inverses), `assume`, sym arithmetic dispatch on `+ - * / ^ ==`, sym-typed elementary functions, REPL JIT + DAP variable inspector, opt-in via `-DMATLAB_LLVM_WITH_SYM=ON` | `test/RunSym/sym_phase_a.m`, `sym_phase_b.m` |
| 6.1 | **Symbolic matrices + multi-eq + IVP + numeric solve** — new `matlab_symmat` opaque type (kind=8) with cross-TU REPL persistence + DAP rendering; `sym_matrix`, `sym_eye`, `sym_zeros`, `sym_det`, `sym_inv`, `sym_transpose`, `sym_trace`, `sym_rank`, `sym_linsolve`, `sym_dsolve_system`; fixed-arity multi-eq solvers `sym_solve_2x2` / `sym_solve_3x3` returning a symmat (one row per joint solution); `nsolve`, `vpasolve`, `dsolve_ivp` (1-cond), `apply_ivp` (1-cond), `checkodesol` | `test/RunSym/sym_phase_b1.m` |
| 6.2 | **Phase 6.2 ergonomics** — standard `[a 1; 2 b]` matrix literal syntax detects sym entries (no more explicit `sym_matrix(...)` constructor); variadic `sym_solve_sys([eq...], [var...])` for systems of any size via LLVM stack-array lowering; multi-condition `dsolve_ivp` / `apply_ivp` taking parallel sym vectors; `simplify` auto-chains `refine()` so assumptions propagate (`simplify(sqrt(y*y))` → `y` after `assume(y,'positive')`); `sym('pi')` / `sym('exp1')` resolve to SymPP singletons; LLVM ptr added to `RefineSlotTypes::isScalarPrim` so sym slots get type-promoted and Mem2Reg'd | `test/RunSym/sym_phase_b2.m` |
| 7   | **Initial-value ODE solvers** — `ode45` (Dormand–Prince 5(4)) and `ode23` (Bogacki–Shampine 3(2)) with adaptive FSAL step + cubic-Hermite dense output; scalar and **vector `y`** (system of ODEs via anon-handle retyping pre-pass in `LowerAnonCalls`); forward / backward / user-time-grid `tspan`; full odeset surface — `RelTol`, `AbsTol`, `MaxStep`, `InitialStep`, `Refine`, `Stats`; 2-return `[t, y]`, 3-return `[t, y, stats]`. C++ / Python / TypeScript runtimes bit-identical. See [`docs/ode.md`](ode.md). | `test/Run/math_ode45_*.m`, `test/Runtime/test_ode.c` |
| 7.1 | **Stiff solver — `ode23s`** (Rosenbrock 2(3), Shampine). One numerical-FD Jacobian per accepted step + LU-factored `(I − h·d·J)` shared across three linear back-solves; scalar y → division, vector y → in-place LU with partial pivoting in the runtime. Robertson kinetics solves in ~9 steps where `ode45` diverges. Mirrored in Python (`numpy.linalg.solve`) and TypeScript (custom LU). | `test/Run/math_ode23s_basic.m`, `math_ode23s_robertson.m` |
| 7.2 | **Numerical PDE — `pdepe`** (1-D parabolic-elliptic, method-of-lines). MATLAB-compatible call shape `pdepe(m, @pdefun, @icfun, @bcfun, xmesh, tspan)`. Coverage: Cartesian / cylindrical / spherical (`m = 0, 1, 2`); Dirichlet, Neumann, Robin BCs; non-uniform mesh; scalar PDE. Spatial finite-difference discretisation; resulting full-state ODE system handed to `ode23s_v` for stiff time integration. Heat-equation gating `u_t = u_xx` on a 21-point mesh recovers the analytic `exp(-π²t)·sin(πx)` to ~1e-3. Cylindrical Laplacian on annulus recovers the log-profile steady state to ~2e-5. Bit-identical across C++ / Python / TS. | `test/Run/math_pdepe_heat.m`, `math_pdepe_neumann.m`, `math_pdepe_radial.m` |
| 7.3 | **Event detection — `ode_events`.** 5-return builtin `[t, y, te, ye, ie] = ode_events(@f, tspan, y0, @evt)`. The event function returns a 3×1 column `[value; isterminal; direction]`; the integrator brackets each accepted DP45 step on `value`, then bisects within the bracket to localize the crossing. `direction` filters rising / falling / either; `isterminal = 1` halts integration at the event. Wired through Resolver + LowerTensorOps as a 5-result builtin (4 operands: rhs handle, tspan, y0, evt handle). Mirrored across C++ / Python / TS — ball-drop event reproduces to ~2e-13 across all three. Non-MATLAB call shape: routing through `opts.Events` is gated on the function-handle-in-struct ABI (same blocker as `OutputFcn` / `Mass`). | `test/Run/math_ode_events_ball.m` |

---

## Recently shipped (Signal Processing Toolbox arc, 2026-05-06 → 2026-05-07)

A second focused arc closing the practical SPT user surface across
Tier-1 (design / apply / inspect), Tier-2 (spectral / parametric /
time-frequency), and Tier-3 (multirate / measurements / alignment).
Drafted from the R2026a Signal Processing Toolbox User's Guide
(2048 pages, 27 chapters). Per-toolbox plan in
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md);
authoritative status in [`feature_status.md`](feature_status.md).

| Phase | What | Gating test |
|---|---|---|
| SPT 1 (§2.3) | **Windows tail** — 14 new windows (`rectwin`, `triang`, `bartlett`, `barthannwin`, `bohmanwin`, `parzenwin`, `nuttallwin`, `blackmanharris`, `flattopwin`, `kaiser`, `tukeywin`, `gausswin`, `chebwin`, `taylorwin`) plus retrofit of pre-existing `hamming`/`hann`/`blackman` into Python / TS for parity. Symmetric (non-periodic) form throughout. | `test/Run/sig_windows.m`, `test/Runtime/test_signal.c` |
| SPT 2 (§2.4) | **Polynomial helpers** — `roots` (Durand-Kerner / Weierstrass simultaneous iteration; the long-standing eig dependency the roadmap had flagged turned out to be moot — DK was already shipped), `poly` (repeated convolution by `[1, −r_i]`), `polyder`, `polyint`, `polyint(p, k)`, plus `[r, p, k] = residue(b, a)` distinct-pole expansion via cover-up rule (multi-return shape mirrors `[V, D] = eig`). | `test/Run/sig_poly.m`, `sig_residue.m` |
| SPT 3 (§2.1) | **IIR lowpass design** — `[b, a] = butter(n, Wn)`, `[b, a] = cheby1(n, Rp, Wn)`, `[b, a] = cheby2(n, Rs, Wn)` via bilinear-transform from analog prototypes (cheby2 needs the generalised `lowpass_from_analog_pz_` helper for j-axis zeros). `freqz(b, a, N)` + `[H, w] = freqz(...)`. Order helpers `[n, Wn] = buttord(...)` / `cheb1ord(...)`. Lowpass scope only — band variants attempted as a separate slice but discarded (peak normalisation off, deferred). | `test/Run/sig_iir.m`, `sig_iir_more.m` |
| SPT 4 (§2.2) | **FIR design** — `fir1(n, Wn)` windowed-sinc with default Hamming, `sgolay(k, f)` returning the projection matrix `B = V (V'V)⁻¹ V'`, `sgolayfilt(x, k, f)` applying it with proper boundary-row handling. | `test/Run/sig_fir.m` |
| SPT 5 (§2.5) | **Close-the-loop filter helpers** — `filtfilt(b, a, x)` forward-backward zero-phase (reflection padding + zero ICs; Gustafsson IC trick deferred), `sosfilt(sos, x)` cascade biquad, `impz`/`stepz`/`grpdelay` response inspection. All five share the internal `filter_flat_` direct-form-II-transposed helper. | `test/Run/sig_filt.m` |
| SPT 6 (§3.4) | **Transforms tail** — `dct`/`idct` (orthonormal, direct O(N²)), `fwht` (in-place butterfly, Hadamard ordering, divided by N), `hilbert` (FFT zero-negative-half + IFFT, returns `matlab_mat_c`), `goertzel` (single-bin DFT, 1×1 complex). | `test/Run/sig_xform.m` |
| SPT 7 (§3.1) | **Nonparametric spectral** — `periodogram(x)`, `pwelch(x, win, noverlap)`, `cpsd(x, y, win, noverlap)`, `mscohere(x, y, win, noverlap)`, `tfestimate(x, y, win, noverlap)`. Single-output, default `fs = 1`. cpsd / tfestimate return `matlab_mat_c`; the TS lane returns magnitude only (no native complex shape — same precedent as `roots` / `fft_c`). | `test/Run/sig_psd.m`, `sig_xspec.m` |
| SPT 8 (§3.2) | **Linear prediction + parametric PSD** — `levinson(r, p)` Levinson-Durbin recursion, `lpc(x, p)` (biased-autocorr + Levinson), `aryule(x, p)`, `arburg(x, p)` Burg forward+backward minimization, plus AR-based PSD `pyulear(x, p, N)` and `pburg(x, p, N)`. | `test/Run/sig_lp.m`, `sig_xspec.m` |
| SPT 9 (§3.3) | **Time-frequency** — `S = spectrogram(x, win, noverlap)` single-output `\|STFT\|²` per (freq, frame). 2-/3-return forms and `stft`/`istft` deferred. | `test/Run/sig_spec.m` |
| SPT 10 (§4.3) | **Pulse measurements core** — `findpeaks` (1-return + multi-return `[pks, locs]`, strict-monotonic), scalar reductions `rms`/`peak2peak`/`peak2rms`/`rssq`, signal cleanup `medfilt1`/`hampel`/`envelope`, pulse statistics `midcross`/`risetime`/`falltime`/`dutycycle` with auto-detected min/max state levels and 10/50/90% reference percentages. MinPeak* options + slewrate/pulseperiod/pulsewidth/overshoot/undershoot/settlingtime deferred. | `test/Run/sig_peaks.m`, `sig_pulse.m`, `sig_stat.m` |
| SPT 11 (§4.1) | **Real multirate** — `upfirdn(x, h, p, q)` (direct algorithm, no zero-stuffed buffer), `decimate(x, r)`, `interp(x, r)`, `resample(x, p, q)`. Replaces the toy `upsample`/`downsample` stubs. Default lowpass via `fir1`. Output lengths match MATLAB: `decimate` ⌈N/r⌉, `interp` N·r, `resample` ⌈N·p/q⌉. | `test/Run/sig_resample.m` |
| SPT 12 (§4.2) | **Waveform generators** — `chirp(t, f0, t1, f1)` linear method, `sawtooth(t, w)`, `square(t, duty)`, `gauspuls(t, fc, bw)`, `rectpuls(t, w)`, `tripuls(t, w)`, `sinc(x)`. | `test/Run/sig_wfmalign.m` |
| SPT 13 (§4.4) | **Alignment helpers** — `xcov(x, y)` mean-removed cross-correlation, `finddelay(x, y)` argmax of `\|xcorr\|`, `dtw(x, y)` dynamic time warping (scalar distance). `alignsignals` multi-return + `gccphat` deferred. | `test/Run/sig_wfmalign.m` |

Cumulative test deltas vs. `e812c3f` (pre-SPT baseline): **+17 run-tests** on the LLVM/C/C++/-strict lanes (172 → 189), **+17** on emit-python (161 → 178 + 11 skip), **+14** on emit-typescript (152 → 166, with 3 new skips for complex-valued tests where TS NDArray drops the imaginary part). All 42 ctest lanes regression-clean. ~80 new SPT runtime entries wired end-to-end.

---

## Recently shipped (Control System Toolbox arc, 2026-05-08 → 2026-05-11)

A third focused arc closing the practical CST user surface across
Tier-1 (numeric prerequisites), Tier-2 (SISO design loop), Tier-3
(state-space design + analysis), Tier-4 (model reduction + MIMO
interconnect + time-delay tail), plus the §3.1 model-object
classdefs (`tf` / `ss` / `zpk` / `pid` / `frd` with operator
overloads and the full short-form surface). Drafted from the
R2026a Control System Toolbox User's Guide (1982 pages, ~24
chapters). Per-toolbox plan in
[`control_toolbox_roadmap.md`](control_toolbox_roadmap.md);
authoritative status in [`feature_status.md`](feature_status.md).

| Phase | What | Gating test |
|---|---|---|
| CST 1.1 | **Non-symmetric `eig`** — 1-return polymorphic real/complex via Hessenberg reduction + Francis double-shift QR with deflation. Symmetric path retained on the Jacobi fast lane. Python lane reimplements Francis QR rather than deferring to scipy (macOS Anaconda numpy/scipy ABI mismatch). | `test/Run/linalg_eig_nonsym.m` |
| CST 1.2 | **`hess` + `schur`** — Hessenberg reduction (in-place Householder reflections) and real Schur decomposition (1-return T and 2-return `[U, T]` via the eig_V/eig_D precedent). Same numerical core as non-sym eig with the orthogonal accumulator threaded through. | `test/Run/linalg_hess.m`, `linalg_schur.m` |
| CST 1.3 | **`expm`** — scaling-and-squaring with [13/13] Padé (Higham 2005). Bit-identical across the 5 emit lanes; gates Tier-2 (`c2d` ZOH, `lsim`, `initial`, gramian time-domain forms). | `test/Run/linalg_expm.m`, `test/Runtime/test_linalg.c` |
| CST 1.4 | **`lyap` + `dlyap`** — continuous Lyapunov + discrete (Stein) via vec + dense LU on the n²·n² Kronecker matrix. O(n^6) cost; Bartels-Stewart on Schur form is the large-plant follow-on. Gates `gram_c`/`gram_o`, the H₂ norm, balanced realisation. | `test/Run/linalg_lyap.m` |
| CST 1.5 | **`care` + `dare`** — algebraic Riccati. `care` via matrix-sign Newton on the Hamiltonian (Roberts 1980); `dare` via Newton-Kleinman seeded from `dlyap(Ad', Q)` (Hewer 1971; requires Schur-stable Ad). 1-return + 3-return `[X, K, L]` shapes (X = Riccati, K = LQ gain, L = closed-loop spectrum) via splitter that fans out to `matlab_care/dare` + `matlab_lqr/dlqr` + `matlab_lqr_e/dlqr_e`. | `test/Run/linalg_care.m`, `ctrl_care_3ret.m` |
| CST 2.2 | **`c2d` (ZOH) + `c2d_tustin` + `d2c_tustin`** — Van Loan augmented-matrix `expm` for ZOH, closed-form bilinear for Tustin / inverse Tustin. All three return 2-tuple `[Ad, Bd]` / `[A, B]` via dedicated splitters. | `test/Run/ctrl_c2d.m`, `ctrl_c2d_tustin.m`, `ctrl_d2c_tustin.m` |
| CST 2.4 | **Frequency response** — SISO `bode_ss` (real 2n×2n decomposition of the complex linear solve), `bode_tf` (complex Horner on `(b, a)`), `gain_margin`/`phase_margin`, `bandwidth_ss`, `getPeakGain_ss` (rough H∞ approximation via log-spaced grid). 2-return splitters for `[mag, phase]`. | `test/Run/ctrl_bode.m`, `ctrl_bode_tf.m`, `ctrl_lsim_margin.m`, `ctrl_bandwidth.m`, `ctrl_pole_peak.m` |
| CST 2.3 | **Time-domain simulation** — `step_ss(A, B, C, D, dt, N)` (ZOH discretise + recurrence), `lsim_ss` (generalised input simulation), `stepinfo(y, t)` (1×5 row `[Rise, Settle, Over, Peak, PeakTime]`). | `test/Run/ctrl_step_gram.m`, `ctrl_lsim_margin.m`, `ctrl_stepinfo.m` |
| CST 4.1 | **`lqr` + `dlqr`** — 1-return form via `care`/`dare`. Plus 3-return `[K, S, e]` shape that returns gain + Riccati solution + closed-loop poles. | `test/Run/ctrl_lqr.m`, `ctrl_dlqr.m`, `ctrl_lqr_3ret.m` |
| CST 4.2 | **Kalman / Kalmd** — `kalman_L` / `kalmd_L` via LQR/Kalman duality (`L = (lqr(A', C', G·Qn·G', Rn))'`). Plus 2-return `[L, P] = kalman(...)` shape returning gain + dual-care covariance. | `test/Run/ctrl_kalman.m`, `ctrl_kalman_2ret.m` |
| CST 4.3 | **`place`** — SISO pole placement via Ackermann's formula `K = [0…01]·ctrb⁻¹·α(A)`; α expanded by complex polynomial multiplication (collapses to real for conjugate-paired roots). Multi-input Kautsky-Nichols-Van Dooren is a follow-on. | `test/Run/ctrl_place.m` |
| CST 4.4 | **Controllability / observability** — structural-rank `ctrb`/`obsv` block matrices; energy-based `gram_c`/`gram_o` via lyap. | `test/Run/ctrl_step_gram.m`, `ctrl_place.m` |
| CST 4.5 / 4.6 | **System metrics** — `norm_h2` (continuous, `sqrt(trace(C·Wc·C'))`), `norm_h2_d` (discrete, with D-feedthrough term), `dcgain_ss`, `isstable` / `isstable_d`, `damp`, `hsvd` (Hankel singular values). | `test/Run/ctrl_norm_h2.m`, `ctrl_norm_h2_d.m`, `ctrl_charac.m`, `ctrl_dcgain.m` |
| CST 5.1 | **Model reduction** — `balreal_T(A, B, C)` (similarity transform via Laub 1980 eigendecomposition variant — sym-eig + lyap, no Cholesky); `balred_A` / `balred_B` / `balred_C` plus 3-return splitter `[Ar, Br, Cr] = balred(A, B, C, k)` for k-state truncation. H∞ error bound `2·sum(HSV[k+1:n])`. | `test/Run/ctrl_balreal.m`, `ctrl_balred.m` |
| CST 2.6 (interconnection, matrix-arg) | **`feedback_ss` + `series_ss` + `parallel_ss` + `append_ss`** — closed-loop assembly for two strictly-proper plants. All four are 3-return splitters `[Acl, Bcl, Ccl] = name(A1, B1, C1, A2, B2, C2)`. Generalised splitter (one block recognises all four function names). | `test/Run/ctrl_feedback.m`, `ctrl_interconnect.m` |
| CST bug fix | **meshgrid / ndgrid multi-return type inference** — `[xx, yy] = meshgrid(...)` typed both outputs as Any so downstream `exp(xx)` fell through to scalar Double, triggering an `arith.mulf(f64, !llvm.ptr)` LLVM lowering crash. Fixed in `lib/Sema/TypeInference.cpp` (per-LHS Array(Double, matrix) type) and `lib/MLIR/Lowering.cpp` (multi-return MLIR result-type table). Plus added `.skip-emit-c` / `.skip-emit-cpp` support to `run_tests_emitc.sh` mirroring the existing python/ts skip convention. | `test/Run/lang_multiret_meshgrid.m` |
| CST §3.1 prep | **CST stdlib prelude wiring** in `tools/matlabc/main.cpp` — auto-prepends `runtime/cst_classdefs.m` (when present, located via the same `<bin>/../runtime/...` walk as `findRuntimePy`) so model-object classdefs like `tf` / `ss` / can land as a stdlib without per-test boilerplate. Plus a real bugfix to `lib/MLIR/Lowering.cpp:3539` field-store dispatch — tensor-typed RHS now routes to `matlab_obj_set_mat` / `matlab_struct_set_mat` (was always `_f64`). | (infra; no behavioural test) |
| CST 1 leftovers (closure) | **`logm`** (Schur-Parlett recurrence), **`lyapchol`**, 3-arg Sylvester `lyap(A, B, C)`, **`qz`** (4-return; B-invertible path), 2-return `[V, D] = eig` for non-sym A (real-only), 5-arg cross-term `care(A,B,Q,R,S)`/`dare`, `icare`/`idare` aliases, **`[H, P] = hess`**, **generalised `eig(A, B)`** via QZ + 2×2-block quadratic. | `test/Run/linalg_logm.m`, `linalg_lyapchol.m`, `linalg_sylvester.m`, `linalg_qz.m`, `linalg_icare_idare.m`, `linalg_eig_gen.m` |
| CST §3.1 (model objects, full slice) | **`tf` classdef** with constructor + property reads + tf-vs-tf operator overloads (`+ − ∗ / −`) + scalar mixing (`G + 2`, `5 ∗ G`, …) + the `s = tf([1 0], 1)` Laplace-variable composition idiom + **`tf('s')` / `tf('z')` char-literal sugar** (intercepted at the constructor-call lowering and rewritten to `tf([1 0], 1)` — char literals don't survive the constructor body, so the rewrite happens at the call site) + **`disp(tf)` formatted s-domain rendering** (centred-fraction layout via the runtime helper `matlab_tf_disp`, dispatched through Lowering.cpp's existing `disp(obj)` class-method route). **`ss` / `zpk` / `pid` / `frd` classdefs** with constructors + property storage + operator overloads (ss block-diagonal A assembly for `+ − ∗ −`; zpk root concatenation + gain product for `∗ / −`; pid coefficient-wise; frd element-wise on `ResponseData`). Auto-prepended per-class preludes (`cst_class_<name>.m`) so a tf-only program doesn't pay the unused-classdef cost. Heavy compiler-side enablement: class-method monomorphisation fix (Sema didn't refine signatures on matrix-typed class properties), sibling-clone retargeting for `matlab.call` ↔ `func.call` conversion in `LowerUserCalls`, none-typed operand relaxation, `pinnedOfRhs` recursion through `BinaryOp` / `UnaryOp` / `CallOrIndex`, binary-op scalar-boxing wrapper restricted to CST classes, and Resolver-side `PinnedClass` propagation through extended-binary-operator method overloads. | `test/Run/ctrl_tf_basic.m`, `ctrl_tf_disp.m`, `ctrl_model_objects.m`, `ctrl_zpk_ops.m`, `ctrl_ss_ops.m`, `ctrl_pid_ops.m`, `ctrl_frd_ops.m` |
| CST model-object short forms (value-returning) | **`pole(sys)`**, **`step(sys [, dt, N])`**, **`bode(sys, w)`**, **`dcgain(sys)`**, **`bandwidth(sys)`**, **`lsim(sys, u, dt)`** for ss / tf (where applicable). Class-pinned-first-arg dispatch in `Lowering.cpp::CallOrIndex` unpacks the relevant properties via `matlab_obj_get_mat` and routes to the matching matrix-arg primitive. | `test/Run/ctrl_sys_short.m` |
| CST Tier-2 leftovers | **`impulse(sys [, dt, N])`** + **`initial(sys, x0 [, dt, N])`** (free-response builtins via ZOH discretisation + recurrence). **`freqresp(sys, w)`** (raw complex H(jω) as `matlab_mat_c`), **`nyquist(sys, w)`** (N×2 real `[re, im]` columns), **`allmargin(sys, w)`** (1×4 `[Gm, Pm, Wcg, Wcp]`), **`damp(sys)` / `isstable(sys)`** model-object short forms. **`logspace(a, b, n)`** runtime (the standard frequency-grid builder for bode / nyquist / allmargin). | `test/Run/ctrl_tier2_response.m` |
| CST Tier-3 leftovers | **5-arg cross-term `lqr(A, B, Q, R, N)` / `dlqr`** via `care_5` / `dare_5` + the matching gain-extraction algebra. **`lqry(sys, Q, R)`** output-weighted LQR (strictly-proper branch collapses to `lqr(A, B, C'·Q·C, R)`; the D ≠ 0 path uses `lqr_5` with cross term `N = C'·Q·D` and effective `R_eff = R + D'·Q·D`). **`acker(A, B, p)`** alias of `place`. Model-object short forms `ctrb(sys)` / `obsv(sys)` / `gram(sys, 'c'\|'o')` / `norm(sys)` / `norm(sys, 2)`. | `test/Run/ctrl_tier3_design.m` |
| CST Tier-4 leftovers (slice 1) | **Padé time-delay approximation `[num, den] = pade(τ, n)`** (closed-form [n/n] symmetric Padé recurrence). **`[num_r, den_r] = minreal(num, den, tol)`** tf-form pole-zero cancellation via roots-then-poly with leading-coefficient preservation. Model-object short forms `hsvd(sys)` / `balreal_T(sys)`. | `test/Run/ctrl_tier4_reduce.m` |
| CST Tier-4 leftovers (slice 2 + Sema-pin) | **Sema-pin architectural piece** — `pinnedOfRhs` now propagates the class pin through a known set of class-returning builtin short forms; when the callee is a Builtin name in {`c2d`, `c2d_tustin`, `d2c_tustin`, `feedback`, `series`, `parallel`, `append`, `blkdiag`, `sminreal`, `modred`} AND the first argument is class-pinned to `ss`, the assignment LHS slot inherits the pin. Mirrors the existing constructor and unary-op-on-class pin paths. Unlocks: **`c2d(sys, Ts)`** (returns fresh `ss(Ad, Bd, sys.C, sys.D)`), **`feedback(sys1, sys2)`** / **`series(sys1, sys2)`** / **`parallel(sys1, sys2)`** (route to `matlab_<name>_ss_{A,B,C}` triple + ss-constructor wrap), **`append(sys1, sys2)`** / **`blkdiag(sys1, sys2)`** (block-diagonal MIMO assembly via `matlab_append_ss_{A,B,C}`). Plus **`sminreal(sys)`** (structural minimality via boolean-graph reach/observability analysis) and **`modred(sys, elim, 'Truncate'\|'MatchDC')`** (modal residualisation with optional Schur-complement DC matching). **Thiran fractional-delay all-pass FIR `[b, a] = thiran(D, n)`**. | `test/Run/ctrl_tier4_assemble.m`, `ctrl_tier4_close.m` |

Cumulative test deltas vs. pre-CST baseline: **+59 run-tests** on the LLVM lane (189 → 248), with the same +59 on emit-c (where applicable). Class-method tests carry `.skip-emit-{c,cpp,python,typescript}` markers because the EmitC pass currently emits classdef property layouts as all-`double` (matrix-typed properties don't fit the struct layout, and class instances flow through the emit pipeline by-value while runtime helpers expect `void *`). Python lane: +35 with `.stdout-python` overrides for numpy bracket repr + ~25 model-object skips. TS lane: +28 with similar skips. ~80 new CST runtime entries wired end-to-end across the slice (15 in Tier-1 closure, ~30 in §3.1 model objects, ~15 in Tier-2/3 short forms, ~20 in Tier-4 close).

**Practical workflow now end-to-end usable through model objects**: design (`lqr` / `lqry` / `place` / `acker` / `kalman_L`) → discretise (`c2d(sys, Ts)`) → close the loop (`feedback / series / parallel / append / blkdiag`) → reduce (`sminreal / modred / balred / minreal`) → simulate (`step / impulse / initial / lsim`) → analyse (`bode / freqresp / nyquist / margin / allmargin / pole / damp / norm / hsvd / dcgain / bandwidth`) → time-delay approx (`pade / thiran`). Both matrix-arg primitives and model-object short forms ship.

**Still 🔵 in the CST surface** (none gates the practical workflow; each is a separate slice):
- ss-form `minreal(sys)` — needs `ctrbf` / `obsvf` controllability/observability staircase decompositions
- Multi-return on model-object call sites: `[Ar, Br, Cr, hsv] = balred(sys, k)` (Lowering.cpp multi-return dispatch on class-pinned operands)
- Graph-style MIMO assembly: `connect(blocks, inputs, outputs)`, `sumblk('e = r - y', size)`, `lft(sys1, sys2, nu, ny)`
- H∞ norm: `norm(sys, Inf)`, `hinfnorm` (Boyd-Balakrishnan-Kabamba bisection on Hamiltonian eigenvalues)
- §4.6 advanced stability/sensitivity: `stabsep` (needs ordered Schur), `freqsep`, `loopsens`, `gangoffour`
- Internal-delay representation on `ss` / `tf` classdefs + `bode` / `step` / `freqresp` consuming the delay properties; `absorbDelay`, `delayss`
- Long-tail Tier-1 numerical optimisations: Moler-Stewart QZ for singular-B, Bartels-Stewart Schur-form `lyap` / `dlyap` / `sylvester`, Hammarling `lyapchol`, 6-arg descriptor `care`, `logm` 2×2 Schur blocks, 2-return `[V, D] = eig` complex eigenvectors
- emit-c / emit-cpp / emit-python / emit-typescript lane parity for the §3.1 model-object tests (EmitC currently models classdef properties as all-`double` and passes class instances by value rather than via `void *`; the §3.1 tests carry `.skip-emit-*` markers)

---

Open follow-ups carried forward (still on the roadmap):

- **Phase 5.4 — `timetable`.** Builds on `table` + `datetime` row index.
- **Narrower / wider int lanes** — i8 / i16 / i64 / u16 / u32 / u64 matrix descriptors against the same template as Phase 1.1.
- **Full method-dispatch value semantics** for OOP — needs test-corpus migration to either rebind or `< handle`-annotate the existing class fixtures.
- **Heterogeneous table columns** — string / categorical / datetime columns alongside numeric.
- **Phase 6.3 — Symbolic Math Toolbox Tier-5 (MATLAB-API polish)**. Full per-tier plan now lives at [`symbolic_toolbox_roadmap.md`](symbolic_toolbox_roadmap.md); roadmap-side summary: `matlabFunction(f, vars)` wrapping the SymPP-emitted Octave source into a callable function handle (§6.1, biggest ergonomics win); AppliedFunction lifting pass `diff(y(x), x)` → SymPP `(y, yp, x)` form (§6.2); cell-array array-arg lowering for `rsolve` / `groebner` / `pythagorean_triples` / `linear_diophantine` — runtime entries exist (§6.3); substitution + simplification tail `subs` cell-form / `combine` / `rewrite` / `collect` / `horner` / `numden` / `partfrac` (§6.4); extended assumption properties `even` / `odd` / `prime` / `algebraic` / `complex` — gated on SymPP-side mask extension (§6.5). Tier-6 is `-emit-python` via SymPy (high value) + `-emit-typescript` via mathjs (low priority).
- **Phase 7.4 — Higher-order stiff solver.** `ode15s` (variable-order BDF + Newton iteration on top of the shipped FD-Jacobian / LU infrastructure from 7.1). Other stiff variants (`ode23t`, `ode23tb`, `ode15i` for DAEs) drop in cheaply once BDF lands. Will speed up `pdepe` on tight tolerances and very-stiff parabolic problems.
- **Phase 7.5 — `pdepe` extensions.** Multi-component systems (`npde > 1`); axis-of-symmetry handling for `xmesh(1) = 0` with `m > 0`; plumbing `odeset` (including `Events`) through to the time integrator. Then 2-D parabolic via `pdepe`-style on a tensor-product mesh.
- **Phase 7.6 — Events through `odeset`.** Bracket+bisect event detection ships today as the dedicated `ode_events` builtin (Phase 7.3); promoting it onto `opts.Events = @evt` for `ode45` / `ode23` / `ode23s` is gated on the function-handle-in-struct ABI work that also unblocks `OutputFcn` (live progress callback) and `Mass` (mass-matrix DAEs).
- **Vector `y` via *named* user functions** — currently anon-only; the LowerUserCalls signature-refinement gate rejects `tensor<Nxf64>` ↔ `tensor<Nx1xf64>` shape mismatches and needs widening.
- **Phase 8 — SV `state_display` regression** (commit `3622f10`): the const-fold pass eliminates assignments to output-port slots when the value comes from a persistent register read. A Phase-6.2 attempt to fix it via `LowerScalarSlots` cast insertion + EmitSV `arith.fptosi` rendering surfaced a cocotb timing divergence between the Python and SV references for `fir_asic_pipelined`; needs lockstep pipeline-equivalence work.
- **SPT §2.1 follow-on — IIR family completion (tail).** The big slice
  shipped: band variants (HP/BP/BS) of `butter`/`cheby1`/`cheby2` (the
  bandpass peak-normalisation bug from the previous attempt was the
  prewarp-vs-bilinear T-convention mismatch — `bilinear_pole_` now
  uses `(2+s)/(2-s)` so all four filter types reproduce scipy / MATLAB
  exactly); `besself` (analog Bessel-Thomson, MATLAB's norm='phase');
  standalone `bilinear` / `freqs`; `cheb2ord`; `tf2zp` / `zp2tf` /
  `tf2sos` / `sos2tf` form conversions. Still open: `ellip` +
  `ellipord` (Jacobi elliptic), the analog prototype builtins as
  standalone 3-return entries (`buttap` / `cheb1ap` / `cheb2ap` /
  `ellipap` / `besselap`), and the remaining state-space / zp-to-sos
  conversions (`tf2ss` / `ss2tf` / `zp2sos`).
- **SPT §2.2 follow-on — richer FIR design.** `fir2` (frequency sampling), `firls` (least-squares), `firpm` (Parks-McClellan / Remez exchange), `firrcos` (raised-cosine), `kaiserord`.
- **SPT §2.5 follow-on — strict Gustafsson `filtfilt`.** The
  lfilter_zi-based steady-state IC path that scipy uses by default
  (method='pad') shipped (constant signals now preserved exactly).
  The strict 1996 Gustafsson method (scipy's method='gust') uses an
  explicit edge-elimination linear system instead of padding; that's
  still open. Plus `phasez` / `zerophase` real-valued response helpers.
- **SPT Tier-2 follow-on — multitaper + STFT + subspace methods.** `dpss` (Slepian sequences via tridiagonal eig), `pmtm`, `stft`/`istft` (with COLA inversion for `istft`), `pspectrum`, `instfreq`, `instbw`, `czt` (chirp Z-transform via Bluestein), `cceps`/`rceps`/`icceps`, `pcov`/`pmcov`, `pmusic`/`peig`/`rootmusic`/`rooteig`, `prony`/`stmcb`.
- **SPT §4.1/§4.2/§4.4 follow-ons.** Polyphase decomposition (`polyphase`); chirp non-linear methods (quadratic / log / hyperbolic), `pulstran`, `diric`, `gmonopuls`, `vco`; `alignsignals` (multi-return), `gccphat`, `xcorr` scaling-option strings (`'biased'`/`'unbiased'`/`'normalized'`/`'coeff'`).
- **SPT §4.3 follow-on.** `findpeaks` name-value options (`MinPeakHeight`/`MinPeakDistance`/`MinPeakProminence`/`Threshold`/`SortStr`) — gated on Sema's name-value-arg parsing. The pulse-statistics tail (`statelevels`, `slewrate`, `pulseperiod`, `pulsewidth`, `overshoot`, `undershoot`, `settlingtime`) shipped in the §4.3 closure slice.
- **SPT Tier-4 — `digitalFilter` system object.** `designfilt`-style entry returning a filter handle that `filter`/`filtfilt`/`freqz` can polymorphically accept. Needs a new descriptor type alongside `matlab_dict` / `matlab_symmat`. Plus `dsp.SOSFilter` / `dsp.FIRFilter` HDL system objects for hooking the SystemVerilog backend onto the filter design path.
- **SPT Tier-4 — wavelets.** `cwt`, `dwt`/`idwt`, `wavedec`/`waverec`, `wvd` (Wigner-Ville), `fsst`/`ifsst` (Fourier synchrosqueezed). 2–3 weeks each, stand-alone algorithmic work.
- ~~**Sema follow-on — user functions shadow builtins.**~~ ✅ shipped in `5125af0` — `Scope::declare` now allows `Builtin → Function` and `Builtin → Class` promotion, so `function y = sin(x)` / `classdef filter` / etc. correctly override the same-name builtin in the current TU. The `square` user-fn fixture renamed to `squarem` in the SPT §4.2 slice (commit `39111c5`) can stay as-is — the rename is harmless and the fix works regardless.

---

## Recently shipped (Model Predictive Control Toolbox arc, 2026-05-19 → 2026-05-20)

A focused arc closing the practical MPC user surface across all six
tiers, on top of the already-shipped Control System and Optimization
toolboxes. Runtime in [`runtime/toolbox/mpc/`](../runtime/toolbox/mpc/);
per-toolbox plan in [`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md);
authoritative status in [`feature_status.md`](feature_status.md).

| Phase | What | Gating tests |
|---|---|---|
| MPC T1 | **Linear MPC core** — `mpc` / `mpcstate` classdefs, `mpcmove`, `sim`, hand-coded KWIK active-set QP (Schmid-Biegler-Bemporad), prediction-matrix builder. Class-form `c2d` / `kalman` integration (`ss` gains a `Ts` property + 5-arg ctor). | `mpc_t1_*` (5) |
| MPC T2 | **Constraints + disturbances** — output + mixed input/output constraints, ECR soft slack, output-disturbance integrator, MV blocking, run-time bound overrides via `mpcmoveopt`. | `mpc_t2_*` (5) |
| MPC T3 | **Adaptive / TV / gain-scheduled / LPV** + the mflow `MpcMove` block deploying through emit-c/cpp/python/SV + cocotb SIL. | `mpc_t3_*` (4) |
| MPC T4 | **Explicit MPC** via offline grid tessellation (zero run-time QP) + standalone `mpcActiveSetSolver` + finite-control-set MPC. | `mpc_t4_*` (3) |
| MPC T5 | **Nonlinear MPC** — `nlmpc` / `nlmpcmove` over the shipped `fmincon` with an RK4 prediction rollout and an anonymous-handle StateFcn (SISO `pendulum` + MIMO `twin_rotor`). | `mpc_t5_*` (2) |
| MPC T6 | **Carve-down sweep** — continuous-plant auto-c2d, rate bounds, MV-tracking (`Wu`/`u_target`, gradient + Hessian), `setEstimator`/`getEstimator`/`review`, `mpcsimopt`, reference previewing, RK4 in `nlmpcmove`. | `mpc_t6_*` (6) |

**25 MPC gating tests green.** Headlines `examples/mpc/{dc_servo_mpc,
paper_machine, pendulum_nlmpc, twin_rotor_nlmpc}.m` (SISO/MIMO × linear/
nonlinear) + `examples/quadrotor/` (Symbolic Math Toolbox derives the
6-DOF EOM; cascade MPC-position / PID-attitude flight controller with
plots).

> The Optimization, PDE, Symbolic Math, and Fixed-Point Designer
> toolboxes also shipped in earlier arcs not separately logged here;
> their status lives in [`feature_status.md`](feature_status.md) and the
> per-toolbox roadmap docs.

---

## Recently shipped (REPL + cross-turn + toolbox-polish arc, 2026-06-14 → 2026-06-16)

An eight-issue arc closed via one PR each (#297–#304), every PR CI-green with a
regression test. Roadmap-side summary; authoritative status in
[`feature_status.md`](feature_status.md):

| Issue | What | Gating test |
|---|---|---|
| #290 | REPL multi-line block input (`for`/`if`/`while`/`switch`/`try`/`function`, `...` continuation) — verified already-working, locked | `test/Repl/run_tests.sh` (multiline_*) |
| #291 | REPL **persistent history** (`$MATLABC_HISTFILE` / `~/.matlabc_history`) + **tab completion** (session fns + builtins/keywords) | `test/Repl/run_pty_tests.py` (`repl-pty-tests`) |
| #292 | Cells — `c{i} = matrix/string` element assignment (kind-tracked read-back) + `cell(m,n)` preallocation | `regress_cell_element_assign.m` |
| #293 | Control model objects — `tfdata`/`ssdata`/`c2d`/`disp(tf)` verified + locked | `regress_control_model_objects.m` |
| #294 | Fixed-point `fi * <scalar>` — widened the product work-integer (`Ls.WL+Rs.WL`) to fix overflow → 0 | `regress_fi_scalar_mul.m` |
| #296 | Deep Learning — `trainingOptions(solver,…)` + `trainnet(X,T,net,loss,opts)` MATLAB API on the dlnetwork trainer (adam/sgdm/rmsprop) | `regress_dl_trainnet.m` |
| #289 | Char arrays — cross-turn REPL `c == 'l'` / `c + 1` operate on codes (new kind=18 + `Binding::IsChar`; disp/concat unchanged); follows #265 | `test/Repl/run_tests.sh` (xturn_char_*) |
| #295 | GPU `-emit-{cuda,metal,opencl} -o <dir>` output-dir flag — completes the AOT bundle API (bundle built + ran on RTX 5060) | `run_gpu_emit_tests.sh` (`gpu-emit-tests`) |

---

## Near-term (~1 month)

### 1. HDL Verification with CocoTB 🔵

Wire generated SystemVerilog modules to a Python testbench harness
using [CocoTB](https://www.cocotb.org/). Each `examples/hdl/*.m`
module gets a paired `<name>_tb.py` that drives clk/rst, walks
through a handful of input vectors, and asserts the output matches
the MATLAB reference (run via the existing C/C++/Python emission of
the same source).

**Why it matters.** Today the SV pipeline is verified by
Verilator's lint pass (76 fixtures lint-clean) + Yosys generic
synth on the non-FSM subset. Lint catches
syntax / signedness / width issues but doesn't prove the RTL
*behaves* the same as the MATLAB source. CocoTB closes that gap:
the same MATLAB program is the golden reference and the
implementation under test.

**Scope.**
- New `test/EmitSVCocoTB/` directory mirroring `test/EmitSV/`.
- Helper script `just verify-cocotb <name>.m` that:
  1. Emits SV via `-emit-systemverilog`.
  2. Emits Python via `-emit-python` (the reference model).
  3. Runs CocoTB with a small driver that feeds the SV
     simulation and the Python model the same vectors per cycle.
  4. Diffs outputs cycle-by-cycle.
- CI lane `cocotb-tests` (gated on CocoTB + Verilator + Icarus
  presence; skip-if-missing rather than required).
- Per-module `*_tb.py` for the 8 `examples/hdl/` modules
  (`alu_16bit`, `mux_4to_1_16bit`, `counter_0_to_10`,
  `mealy_fsm`, `moore_fsm`, `vector_processor`,
  `sequential_processor`, `fir_asic_pipelined`).

**Out of scope (for v1).**
- UVM-style coverage / functional checking.
- Multi-clock testbenches.
- Proving timing closure.

**Dependencies.** None — both ends of the bridge already work.

**Effort.** ~1 week (harness + 8 testbenches + CI lane).

---

### 2. Tier 2: persistent fi-arrays in software backends 🔵

The remaining 3 of 8 `examples/hdl/` modules
(`fir_asic_pipelined`, `sequential_processor`, `vector_processor`)
use **persistent fi-arrays** —
`persistent buf; buf = fi(zeros(1, N), ...)` — which the SV path
lowers to N parallel registers via Stage F, but the C/C++/Python/TS
backends don't model.

**Why it matters.** Streaming / windowed signal-processing in pure
software is a real use case (FIR filters in C, sliding windows,
buffered DSP), not just an HDL idiom. Tier 2 unblocks all 8
HDL examples for software emission and makes the existing fi-array
support useful end-to-end.

**Scope.**
- C: `static T name[N] = {<init>};` at function entry; reads /
  writes through `name[k]`.
- C++: same.
- Python: `<fn>.<name> = [<init>] * N` at module scope; reads /
  writes through `<fn>.<name>[k]`.
- TS: `let <fn>_<name>: number[] = [<init>] * N;`.
- Recognize the `matlab_persistent_get_ptr → subscript1_s` and
  `matlab_persistent_set_ptr → array-of-stores` chains; suppress
  the runtime-call form and emit array indexing.

**Dependencies.** Tier 1 (shipped) recognizes the canonical
isempty pattern; Tier 2 extends the same recognition to the
array-typed persistent ABI.

**Effort.** ~3 days per backend × 4 backends = ~2 weeks.

---

### 3. SV codegen polish 🔵

The 8 HDL examples lint clean, but a few cosmetic / quality
issues remain that don't block synthesis but read awkwardly:

- **Storage-class literals on register width casts.**
  `count_reg <= 4'(8'sd0)` could just be `count_reg <= 4'sd0`.
  The wrap-cast is redundant when the source is a constant.
- **Saturate constant rendering.** Things like `64'sd68719476735`
  (= 2³⁶−1) read more naturally as `36'sh7FFF_FFFFF`. Cosmetic
  but DSP code is full of these.
- **`v0_1`, `v1_1`, ... synthetic intermediate names.** The
  saturate-clamp temps in `vector_processor` / `sequential_processor`
  / `fir_asic_pipelined` use compiler-generated names. Could
  derive semantic names from the surrounding context (e.g.
  `acc_clamped_1`, `prod_extended`) — much more readable RTL.
- **Comment-block placement on persistent declarations.** The
  source `% Estágio 0: Entradas` next to a `persistent delay_line`
  has no SV-side anchor right now (the declarations live in the
  prelude, not always_comb). Should attach to the prelude
  declaration block.

**Scope.** Each is independent; can be ordered by user impact or
done together.

**Effort.** ~2 days total.

---

### 4. SV codegen: pragma path for `-emit-c` / `-emit-cpp` 🔵

Today `% hdl: port(name, fi, signed, W, F)` pragmas are SV-only —
applying them to the C/C++/Python/TS pipelines would let function-
only `.m` files (no typed driver) compile to software too.

**Why it matters.** Asked-for already during the C/C++ audit
(`alu_16bit.m` standalone fails with `unsupported op: matlab.alloc`
because no driver pins types). Reusing the pragma machinery is
the smallest fix.

**Scope.** Lift the `IsSVPath` gate on `runApplyPortTypePragmas`
in `tools/matlabc/main.cpp`; verify nothing else gates on the
SV-only assumption.

**Effort.** ~30 min + regression check.

---

### 5. SPT follow-ons (highest-leverage gaps) 🔵

The SPT arc closed the practical "design / apply / inspect / measure"
surface; the highest-leverage open items, in priority order:

- **§2.1 IIR family completion (tail).** The bulk of §2.1 shipped:
  band variants HP/BP/BS for `butter`/`cheby1`/`cheby2`, `besself`,
  standalone `bilinear`/`freqs`, `cheb2ord`, `tf2zp`/`zp2tf` /
  `tf2sos`/`sos2tf` form conversions. Open: `ellip` + `ellipord`
  (Jacobi elliptic functions), the analog prototype builtins
  (`buttap`/`cheb1ap`/etc.) as standalone 3-return entries, and
  `tf2ss` / `ss2tf` / `zp2sos`. **Effort:** ~2 sessions.
- **§2.5 strict Gustafsson `filtfilt`.** The lfilter_zi-based
  steady-state IC path (scipy's pad-method default) shipped — constant
  signals now preserved exactly. The strict 1996 Gustafsson method
  (scipy's method='gust') uses an explicit edge-elimination linear
  system instead of padding; that and `phasez` / `zerophase` are
  still open. **Effort:** ~3 sessions.
- **§4.3 follow-on — `findpeaks` name-value options.** `MinPeakHeight`,
  `MinPeakDistance`, `MinPeakProminence`, `Threshold`, `SortStr`. The
  pulse-statistics tail (`slewrate`, `pulseperiod`, `pulsewidth`,
  `overshoot`, `undershoot`, `settlingtime`, `statelevels`) shares
  the same edge-detection scaffolding. **Effort:** ~3 sessions.
- **§3 multitaper + STFT.** `dpss` + `pmtm` for high-resolution
  spectral estimation; `stft` / `istft` (with COLA inversion) to close
  the time-frequency tail alongside the existing `spectrogram`.
  **Effort:** ~1 week.
These are independent and can land in any order; pick by what
unblocks real workflows.

The Sema "user-functions shadow builtins" follow-on that previously
appeared here landed in `5125af0` and is no longer open.

---

### 6. Runtime: arena allocator + leak audit 🟡

The C runtime currently uses `malloc`/`free` per matrix +
ref-counting on some paths. Two pain points:

- **Allocator pressure** in tight loops (e.g. `for i = 1:1000;
  A = A + B; end` allocates a fresh result matrix each iter).
- **No leak tracking surface.** Programs that genuinely leak
  (held refs in REPL workspace) are invisible until ASAN.

**Scope.**
- Per-call arena reset for the AOT-compiled paths.
- `MATLAB_RT_TRACE=1` env-var prints `alloc / free / leak`
  summary at exit.
- Optional: bump-allocator with explicit reset in JIT-mode for
  long REPL sessions.

**Effort.** ~1 week.

---

## Mid-term (~1–3 months)

### 7. Block language (visual nodes → AST → MLIR) 🟢

**v1 shipped.** The MatForge IDE now saves `.mflow` JSON files
that `matlabc` and `matlab-lsp` both consume. The implementation
chose graph → AST (rather than the originally planned graph →
MLIR direct), which got every existing backend — LLVM / C / C++ /
Python / TS / SV / fixed-point / hardware-report — for free, plus
a free `-emit-matlab` round-trip via the existing `formatAST`.

Five phases shipped:
- **1.** JSON loader + schema validation, byte-precise diagnostics
  (`-dump-flow`).
- **2.** Linear chain → AST: `variable`, `expression`, `display`,
  `input`, `assignment`, `constant`, `function_call`,
  `matrix_literal`.
- **3.** Structured control flow: `if` / `for` / `while` /
  `break` / `continue` / `return`, arbitrary nesting; refuses
  irreducible CFGs.
- **4.** Sub-flows lifted to top-level `Function`s;
  `function_definition` and `subflow_call` blocks.
- **4b.** `custom` blocks with three provenance modes: inline
  `source` / sibling `path` / `library_id` (resolved via
  `--block-path` + `MATFORGE_BLOCK_PATH`); function-insertion
  dedup; arity validation.
- **5.** Cross-backend round-trip lane (`.mflow` ≡
  round-tripped `.m` across C / C++ / Python / TS); `matlab-lsp`
  accepts `.mflow` URIs.

8 examples under `examples/mflow/` and 4 ctest lanes.
See [`flowchart_frontend.md`](flowchart_frontend.md) and the
shipped row in [`feature_status.md`](feature_status.md).

**Open follow-ups (v2 territory, not blocking).**
- Richer block library: `Delay (z⁻¹)`, `FIR`, `IIR (DF-II)`,
  `FSM (state diagram)`, `Counter`, `Accumulator` as primitive
  block kinds rather than custom blocks. Each becomes a small
  Phase-2/3-style render rule.
- Round-trip text ↔ blocks editing (currently one-way).
- 2-D / image-pipeline blocks — overlaps with item #7.

---

### 8. Improve HDL codegen: 2-D fi matrices + RAM inference 🔵

The biggest remaining SV scope gap. Today the pipeline supports
1-D fi arrays (shipped via Stage E + Stage F); 2-D matrices are
needed for image-processing pipelines and matrix-multiply HDL.

**Scope.**
- 2-D fi storage: `logic signed [W-1:0] mem [R][C]` declaration
  + 2-D subscript reads / writes.
- RAM inference for large 2-D persistents:
  `persistent buf; buf = fi(zeros(1, 1024), ...)` should infer
  a synth-tool-recognized SRAM block (`always_ff @(posedge clk)
  if (we) mem[addr] <= din;`) instead of 1024 parallel registers.
- Shape recognition: differentiate "small N for shift register
  → N parallel regs (Stage F today)" from "large N for
  data buffer → RAM block".

**Effort.** ~2 weeks.

---

### 9. SystemVerilog → MATLAB (reverse direction) 🔵

Take legacy synthesizable SV (or simple sequential RTL with
clocked persistent state) and lift it into MATLAB source for
verification, simulation, or porting.

**Why it matters.**
- HDL teams often have SV reference implementations and want
  to iterate on the algorithm in MATLAB (faster, with NumPy
  / matplotlib).
- Lets a designer take an existing IP block, lift it to MATLAB,
  modify it, and re-emit to SV via the existing forward path —
  closing the loop.

**Scope (v1).**
- Lex + parse a synthesizable SV subset:
  - `always_ff` (single clock + sync/async reset).
  - `always_comb` (combinational logic).
  - `unique case` and `if/else` chains.
  - `logic [N-1:0]` and `logic signed [N-1:0]` register declarations.
  - One-hot and binary-encoded `typedef enum` FSMs.
- Lift to a typed MATLAB AST:
  - SV register → `persistent` MATLAB var with
    `if isempty(_); _ = init; end` reset pattern (the same
    idiom Tier 1 recognizes).
  - `unique case (state)` → `switch state`.
  - Sized integer literals → `fi(_, signed, W, F)`.
- Output: pretty-printed MATLAB source via the existing
  formatter.

**Out of scope (for v1).**
- Verilog (only SystemVerilog).
- Multi-clock / CDC handling.
- Generate blocks / parameterized modules.
- Behavioral SV beyond the synthesizable subset.

**Dependencies.** None new — uses the existing AST + formatter.

**Effort.** ~3 weeks.

---

### 10. REPL: line editing + history + JIT cache 🟡

Most of the ergonomics shipped (#290 / #291):

- ✅ **Line editing** — history navigation (↑/↓), Ctrl-A/E motion,
  Ctrl-U/K kill, Ctrl-L clear, in `readLineRaw`.
- ✅ **Multi-line input** — `for`/`if`/`while`/`switch`/`try`/`function`
  blocks accumulate until block + bracket depth return to 0 (#290).
- ✅ **Persistent history** — `$MATLABC_HISTFILE` (or `~/.matlabc_history`
  for a TTY session); piped/CI runs don't persist (#291).
- ✅ **Tab completion** — session functions + curated builtins/keywords (#291).
- 🔵 **Persistent JIT cache** keyed by hashed source so repeated
  function definitions don't re-JIT cold. *(the remaining item)*

**Effort.** ~0.5 week (JIT cache wiring only).

---

### 11. Improve HDL codegen: pipelining + retiming 🔵

Beyond the v1 stage-F register split, the pipeline doesn't
automatically rebalance critical paths. For DSP designs that need
to hit a target frequency, this matters.

**Scope.**
- `% hdl: target_freq(N_MHZ)` pragma.
- Compute critical-path estimate per always_comb block (op count
  × per-op latency table).
- Insert pipeline registers when the path exceeds budget.
- Already-shipped scaffolding: `-sv-input-pipeline=N` / `-sv-output-
  pipeline=N` for fixed-stage pipelining.

**Out of scope.** Sophisticated retiming (moving registers
across logic). Just insertion at safe boundaries.

**Effort.** ~2 weeks.

---

## Long-term / exploratory

### 12. MATLAB graphics / `plot` ✅ (initial)

Headless Cairo-backed plot runtime under `runtime/plot/` plus a
matlabc codegen pass (`lib/MLIR/Passes/LowerPlot.cpp`) lowering
MATLAB calls to runtime symbols. Builds on macOS / Linux / iOS
from one codebase; no subprocess, no display server.

**Shipped surface (≈50 % of MATLAB plot calls in the wild)**:
plot / plot3 / scatter / bar / stem / stairs / area / errorbar /
histogram / imshow / imagesc / pcolor / contour / contourf / quiver /
mesh / surf, plus title / xlabel / ylabel / zlabel / text / legend
(varargs) / colorbar / colormap (gray/parula/jet/viridis/hot/cool) /
grid / hold / axis (string + numeric) / box / xlim / ylim / view /
xline / yline / xticks / yticks / xticklabels / yticklabels / yyaxis /
loglog / semilogx / semilogy / subplot / saveas / print / figure / gcf /
close. Property/value pairs (`'LineWidth'`, `'Color'`, `'LineStyle'`,
`'Marker'`, `'MarkerSize'`, `'DisplayName'`) honoured. Auto colour
cycle (MATLAB R2014b+ palette). Output: PNG / SVG / PDF in memory or
on disk.

**Status & roadmap**: see [`plotting.md`](plotting.md). Open follow-ons
include `legend({'a','b'})` cell form (needs `LowerTensorOps` work),
`polarplot`, `tiledlayout`, TeX/LaTeX in labels, lighting / shading
options, `boxplot` / `violinplot` / `pie`, the volume-rendering 3D
family (`isosurface` / `slice` / `streamline`), `fplot` / `fmesh` /
`fsurf` (need symbolic backend), animation, DPI-controlled output.

Build with `-DMATLAB_LLVM_WITH_PLOT=ON` (requires Cairo via pkg-config).

---

### 13. `.mat` file save / load 🔵

Already documented in [`docs/save_load_compat.md`](save_load_compat.md).
Goal: read MATLAB v7.3 (HDF5-based) `.mat` files into the runtime
workspace and vice versa. Not a full MATLAB compatibility matrix;
just the common cases (`save('out.mat', '-v7.3')` followed by
`load('out.mat')` in another session).

**Effort.** ~2 weeks.

---

### 14. Toolbox stubs for symbolic / optimization ✅ (superseded — shipped first-class)

**Superseded.** This entry originally proposed thin stubs routing to
`sympy` / `scipy.optimize`. Both shipped instead as **first-class,
hand-coded, MathWorks-free surfaces**: Symbolic Math via SymPP (Tiers
1 → 4, see [`symbolic_toolbox_roadmap.md`](symbolic_toolbox_roadmap.md))
and Optimization Toolbox (Tiers 1 → 5, see
[`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)) — no external
runtime dependency, full `-emit-*` participation. Sixteen toolbox
surfaces ship today; the next candidates are in §16 below.

---

### 15. Partial Differential Equation Toolbox ✅

Per-toolbox roadmap at
[`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md).  Closes the 2-D
and 3-D FEM workflow on top of the existing `pdepe` 1-D MOL surface:
`createpde` / `femodel` → geometry (DG / `multicuboid` / STL) → tet
mesher → linear-elasticity / thermal / EM assembly → sparse solve →
post-processing (`VonMisesStress`, `interpolateStress`,
`pdeplot3D`).

The **headline gating example** is
[`examples/pde/wind_stress_3d.m`](../examples/pde/wind_stress_3d.m) —
a 3-D model under 250 km/h aerodynamic wind pressure with a von
Mises stress map on the deformed shape.  Closing that example
end-to-end (compile → JIT → execute → REPL inspect → PNG out)
closes PDE-Tier-2.

**Critical-path prerequisites** (each is its own sub-project; see
§10 of the per-toolbox doc):
- Sparse matrices (`matlab_mat_sparse`, sparse `\`, Krylov suite) —
  unblocks every tier (1 wk).
- 2-D + 3-D mesher (Delaunay-of-Bowyer-Watson in-tree; optional
  `MATLAB_LLVM_WITH_TETGEN=ON` for production meshes) (2 wk).
- STL importer (~2 sessions, lets users bring their own 3-D model).
- Unstructured-mesh plotting (`trisurf2d` / `trisurf3d` painters in
  `runtime/plot/`) (1 wk).

**Effort.** ~8 wk to the wind-stress demo (6 wk infrastructure +
2 wk PDE-specific assembly).  Tiers 3+ (transient / modal / thermal
/ EM / nonlinear / ROM) add ~6 wk on top.

**Status (2026-05-13).** **Eleven shipped arcs** close the full
Tier-1 → Tier-4 surface plus all the polish items.  Sparse CSR
infra (`matlab_mat_sparse`) with PCG, MINRES, and ILU(0)-preconditioned
GMRES.  Lanczos shift-invert with mode-shape retention.  Modal
superposition + Rayleigh damping.  T10 quadratic tetrahedra with
super-convergent Gauss-point stress recovery + per-node von Mises.
STL + GLB importers (surface + voxelize-AABB volumetric).  Full
`femodel` classdef façade + MATLAB-faithful legacy aliases
(`solvepde`, `solvepdeeig`, `specifyCoefficients`,
`applyBoundaryCondition`, `pdegplot`, `pdemesh`, `pdeplot`,
`pdeplot3D`).  AnalysisType dispatch covering structuralStatic /
Transient / Modal / Frequency (real and damped-complex via
2N×2N real-bordered) / TransientModal / StaticNL / StaticTL /
thermalSteadyState (+ Picard `k(T)`) / thermalTransient /
electrostatic / magnetostatic / dcConduction /
harmonicElectromagnetic.  Thermal-stress coupling
(`cellLoad(Temperature=…)`).  Modal-truncation **and** full
Craig-Bampton ROMs (`reduce`, `reconstructSolution`,
`pde_reduce_craig_bampton`).  Geometry primitives
(`multicuboid` / `multicylinder` / `multisphere`) +
`refineMesh` / `refineMeshBey` (Bey 8-subdivision) / `adaptmesh`.
N-component coupled scalar PDEs (`pde_solve_multi_n`).

**33 PDE end-to-end tests green** on the LLVM lane.  Cross-
toolbox regression spot-checks (signal / control / ODE / comm /
RF): clean.

**Remaining (mostly polish):**
- Deep Total-Lagrangian element kernel (full Green-Lagrange B_NL
  + geometric K_σ for true large-rotation problems).
- Hanging-node red-green propagation for partial Bey refinement.
- Real Delaunay / TetGen mesher (today's volumetric meshing
  uses voxelize-AABB + Kuhn 6-tet — adequate but mesh-quality
  ceiling shows up on T10 bending).
- 3-D Gouraud shading on the unstructured mesh painter
  (per-triangle flat today).

**Explicit carve-outs** (per project memory): PINN / GNN / FNO,
Battery P2D, STEP import, PDE Modeler 2-D app, full 3-D Nédélec
edge-element vector EM.

---

### 16. Future toolboxes — drafted compatibility roadmaps 🔵

After the **nineteen shipped toolbox surfaces** (see
[`feature_status.md`](feature_status.md) and the README's
"Shipped Toolboxes" table — Curve Fitting, Wavelet, DSP System + DSP HDL,
and GPU Coder were drafted here and have since shipped), two of the
original drafts remain — **full tiered compatibility roadmaps**, same
format as the shipped ones: per-tier function tables, reusable-infra
map, headline tracer-bullet, Compile/Execute · Debug/REPL · examples ·
tests breakdown, carve-outs, effort summary. They are ordered by
**gain** (demand × reuse of the already-shipped substrate — so cost is
low and payoff high):

| # | Toolbox | Roadmap | Tiers · effort | Why it's cheap | Headline |
|---|---|---|---|---|---|
| 1 | **Curve Fitting** ✅ | [`curve_fitting_toolbox_roadmap.md`](curve_fitting_toolbox_roadmap.md) | 6 · shipped | rode shipped `polyfit`/`polyval`/`interp1` + Optim `lsqcurvefit`/`lsqnonlin` | `census_fit.m` (`fit`→`gof`→forecast→`plot`) |
| 2 | **Wavelet** ✅ | [`wavelet_toolbox_roadmap.md`](wavelet_toolbox_roadmap.md) | 6 · shipped | extended Signal — `conv`/`fft`/`dct`/`fwht`/`upfirdn` + Image `psnr` + Stats `fitcsvm` | `denoise_signal.m` (`wavedec`→`wthresh`→`waverec`) |
| 3 | **DSP System + DSP HDL** ✅ | [`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md) | 8 · shipped | Signal filters + `fi` + emit-SV/cocotb lane; **T1 SO model unblocked Comm/RF SO tiers** | `dsphdl_fir_stream.m` (fixed-point FIR → synthesizable SV + cocotb SIL) |
| 4 | **Sensor Fusion and Tracking** | [`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md) | 6 · ~12.5 wk | **EKF/UKF cores already shipped** (Ident T5); ODE + linalg + PRNG | `imu_gps_fusion.m` (IMU+GPS → `insfilterMARG`) |
| 5 | **Robotics System** | [`robotics_toolbox_roadmap.md`](robotics_toolbox_roadmap.md) | 6 · ~13 wk | **IK is `lsqnonlin`/`fminunc`** (Optim shipped); FK/Jacobian = linalg; URDF meshes reuse the PDE STL importer | `ik_path_trace.m` (`loadrobot`→`inverseKinematics`) |
| 6 | **Verilog-A analog simulator** (tractable-Simscape path) | [`dae_solver_roadmap.md`](dae_solver_roadmap.md) | 10 · ~14 wk min cut (T1–6) | DAE core extends shipped `ode23s` (LU/FD-Jacobian); MNA assembly reuses **sparse triplets + GMRES-ILU**; `laplace_*` reuses **`tf2ss`/`lsim`**; AC reuses **complex matrix ops**; HB/Monte-Carlo reuse **GPU GEMM** | `diode_rectifier.m` / `amp_ac_noise.m` (import `.va` → MNA → transient/AC/noise) |

**Cross-cutting sequencing:**

- **The System-Object lowering fix** — the documented blocker (a
  tensor-typed RHS routed through `_set_f64` after monomorphization fails
  the verifier; CST roadmap §12 / Comm roadmap §15) — is **DSP Tier-1**,
  and it is the same fix that gates Comm Tier-3+, RF-Tier-1+, Antenna
  classdef tiers, and the stateful filters/trackers of Sensor Fusion.
  Landing it **once** (via DSP) unblocks all of them — the
  highest-leverage single item across the whole queue.
- **The `quaternion` + coordinate-transform foundation** is **shared** by
  Sensor Fusion Tier-1 and Robotics Tier-1 — build it once across the
  two; whichever ships first, the other reuses it.
- **Robotics is largely shippable *without* the SO fix** — its objects
  (`rigidBodyTree`, `inverseKinematics`, `occupancyMap`, collision
  geometry) are built-once-then-queried, not stateful System Objects;
  only 3 objects (`jointSpaceMotionModel`/`stateEstimatorPF`/`rateControl`)
  touch it — so it carries the least sequencing risk.

**Recommended order**: **Curve Fitting first** (near-free; ~4 wk to the
90% `fit`/`gof`/`confint` workflow). Then either **DSP Tier-1** (to land
the SO fix and unblock the SO-gated tiers project-wide) or the
**Sensor-Fusion + Robotics spatial-math stack** (shared `quaternion`
foundation; directly serves the `examples/quadrotor/` flight-control
work and the shipped MPC/CST loops). **Wavelet** slots in any time as a
self-contained Signal extension.

These are **plans, not commitments** — each tier is independently
shippable and demoable, and the per-toolbox docs are the source of truth.

### 17. Complete toolbox audit — ranked backlog (2026-05-24) 🔵

Full sweep against the official MathWorks R2026a toolbox catalog
(~130 product names). Cross-referenced against `runtime/toolbox/` and
the README badge ("19 shipped"). Effort buckets: **XS** <1w · **S** 1-3w
· **M** 3-6w · **L** 6-10w · **XL** 10+w (one focused implementation
session ≈ 1/5 of a week, per the project's existing cadence).

#### Already shipped (matched against the official list)

Algorithm/numerics: **DSP System** · **DSP HDL** · **Wavelet** ·
**Curve Fitting** · **Optimization** · **Global Optimization** ·
**Statistics and Machine Learning** · **Image Processing** ·
**Signal Processing** · **Symbolic Math** · **System Identification**
· **Control System** · **Model Predictive Control** · **Partial
Differential Equation**. Comms/RF: **Communications** · **RF** ·
**Antenna**. Codegen/HW: **MATLAB Coder** (we *are* matlabc) ·
**Embedded Coder** (mflowLink) · **HDL Coder** (SV emit) ·
**HDL Verifier** (cocotb SIL) · **GPU Coder** (T1+T2.A + CUDA/Metal/OpenCL
emit lanes) · **Stateflow** (backend). Partials: **Parallel Computing**
(`parfor` outliner) · **Fixed-Point Designer** (`fi` lane). Propagation
folded into the Comm roadmap as priority §3.

#### Out of scope (decided, not shipping)

| Category | Reason |
|---|---|
| **Simulink** + all Simulink-* (`Coder`, `Compiler`, `Compiler SDK`, `Coverage`, `Design Verifier`, `Test`, `Report Generator`, `Real-Time`, `Desktop Real-Time`, `3D Animation`, `Check`, `Code Inspector`, `Control Design`, `Design Optimization`, `FMU Builder`, `Fault Analyzer`, `PLC Coder`, `Copilot`) | Graphical block-diagram UX — replaced by `.mflow` text-DSL |
| **All Blocksets** (`AUTOSAR`, `C2000`, `DDS`, `Motor Control`, `Powertrain`, `Raspberry Pi`, `STM32`, `Vehicle Dynamics`, `SoC`, `RF Blockset`, `Aerospace Blockset`, `Audio Blockset`, `Mixed-Signal Blockset`) | Same — Simulink-bound |
| **Simscape** + `Battery` / `Driveline` / `Electrical` / `Fluids` / `Multibody` / `Multibody Link` | Full Simscape (acausal DSL + Pantelides DAE + Simulink integration) stays out. **But the tractable subset — acausal physical/circuit simulation via a DAE core + MNA, driven by imported Verilog-A — is drafted in [`dae_solver_roadmap.md`](dae_solver_roadmap.md)** (no Simscape DSL, no Simulink, no Pantelides for the index-1 device networks in scope). See §16 row 6. |
| **SimBiology**, **SimEvents** | Simulink-bound |
| **All Polyspace** (`Access`, `as You Code`, `Bug Finder`, `Code Prover`, `Copilot`, `Products for Ada`, `Test`) | Static-analysis suite — different problem class |
| **Compliance kits**: DO Qualification (DO-178), IEC Certification (ISO 26262 / IEC 61508) | Documentation/process products |
| **Services**: MATLAB Online Courses, Grader, Mobile, Copilot, Parallel Server, Production Server, Web App Server, Compiler, Compiler SDK, Report Generator, Test, Drive | Hosted services |
| **Host bindings**: Database, Datafeed, Data Acquisition, Instrument Control, Image Acquisition, Industrial Communication, Vehicle Network, OPC, Spreadsheet Link, ThingSpeak, Cloud Integrations | OS/HW driver wrappers; no algorithmic content |
| **RoadRunner**, **RoadRunner Scenario** | 3D scene authoring tools |
| **Requirements Toolbox**, **System Composer** | Architecture/req modeling UX |
| **Installation and Licensing** | n/a |

#### Missing — ranked by value × effort

##### Top recommendations (high value, leverages shipped stack)

| # | Toolbox | Effort | Why now |
|---|---|---|---|
| 1 | **Sensor Fusion and Tracking** | M (3-4w) | EKF/UKF already in Ident T5; add `trackingEKF/UKF/IMM`, `phd`, `insfilter`, track managers. Unlocks the **quadrotor IMU/GPS fusion demo**. Roadmap drafted ([`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md)). |
| 2 | **Aerospace Toolbox** | S (2-3w) | Pure numerics: quaternion / DCM utils, ECEF/NED/ECI frames, standard atmosphere, gravity (`gravityWGS84`), World Magnetic Model. Pairs directly with quadrotor + sensor fusion. |
| 3 | **Phased Array System** | M (4w) | RF + Antenna + Propagation all shipped — add array geometry (`phased.ULA/URA`), beamformers (delay-and-sum / MVDR / Capon), DOA (MUSIC/ESPRIT), `phased.Radar`, radar equation. Closes a complete RF research stack. |
| 4 | **Computer Vision** | M (4-6w) | Image Processing shipped. Add feature detectors (Harris/FAST/ORB), RANSAC, geometric registration, stereo, camera calibration, optical flow. Bridges to UAV/AV demos. |
| 5 | **Robust Control** | M (4-5w) | Control System shipped. Add `hinfsyn` / `musyn` / `uncertain` / LMI solver. Modest effort on top of `lqr` / `care` / SDP. |
| 6 | **Navigation Toolbox** | M (5-6w) | Builds on Sensor Fusion. `plannerRRT/HybridAStar`, occupancy maps, `monteCarloLocalization`, INS/GPS integration. AV/drone capstone. |

##### Tier 2 (high value, larger lift)

| # | Toolbox | Effort | Notes |
|---|---|---|---|
| ~~7~~ | **Deep Learning** | ✅ **SHIPPED 2026-05-28** | All 6 tiers — `dlarray` + reverse-mode autodiff tape, `dlnetwork` carrier + `trainnet` driver + custom-loop API (`adamupdate`/`sgdmupdate`/`rmspropupdate`), full activation/loss/conv2d/conv1d/BN/LN/GN/IN/RMSNorm/dropout surface, LSTM/GRU/BiLSTM/LSTMP + functional MHA + `embed`, residual/transfer-learning/GAN/VAE/Siamese/Neural-ODE patterns, gradCAM (MLP + image-domain)/occlusion/LIME/tsne, classification metrics, dlquantizer calibrate/validate, magnitude pruning, programmatic experiment-sweep harness. **DL HDL H1–H5** ships in the same toolbox: INT8 quant → fi-typed SystemVerilog → cocotb bit-accuracy (`% hdl: precise_fi`) → LSTM-on-FPGA → minimal ONNX inference-graph importer (~56 ops). 39 `dl_*.m` tests + 44 examples. See [`feature_status.md`](feature_status.md) + [`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md). |
| ~~8~~ | **Reinforcement Learning** | ✅ **SHIPPED 2026-05-31** (PR #83) | All 6 tiers + a beyond-list GRPO bonus. Tabular `rlQAgent`/`rlSARSAAgent` (grid-world/MDP) → DQN → REINFORCE (`rlPGAgent`) → DDPG → **TD3 / PPO / SAC** → **`rlGRPOAgent`** (DeepSeek, critic-free, with a Countdown verifier env). Eight deep agents, all riding the shipped `dlnetwork`/`dlgradient` autodiff tape with zero duplication — the keystone dividend. Surfaced + fixed a general dlnet gemm-transpose memory leak (~20 GB → ~810 MB for DDPG, helps all DL training). 10 gating tests + 10 examples. See [`feature_status.md`](feature_status.md) + [`reinforcement_learning_toolbox_roadmap.md`](reinforcement_learning_toolbox_roadmap.md). Carved: TRPO, SAC auto-temperature, `rlFunctionEnv` classdef, training-monitor/MBPO/deploy infra. |
| 9 | **Robotics System** | L (6-8w) | `rigidBodyTree`, fwd/inv kinematics, motion planning, collisions. Massive overlap with the quadrotor work. Roadmap drafted ([`robotics_toolbox_roadmap.md`](robotics_toolbox_roadmap.md)). |
| 10 | **UAV Toolbox** | M (4-5w) **after #1+#6+#9** | Mostly integration — drone flight stack on top of the three above. |
| 11 | **ROS Toolbox** | M (3-4w) | Plain ROS1/ROS2 marshalling (publish/subscribe + msg types). Self-contained protocol work. |

##### Tier 3 (focused / niche / scientific)

| # | Toolbox | Effort | Notes |
|---|---|---|---|
| 12 | **Lidar Toolbox** | M (4w) | Point clouds, ICP, voxel grids, normals. Builds on Image Proc + Stats. |
| 13 | **Mapping Toolbox** | S (2-3w) | Geo transforms, projections (Mercator/UTM), `geotiff` reader. Light. |
| 14 | **Audio Toolbox** | S (3w) | DSP toolbox wrap + MFCC, voice features, audio effects. |
| 15 | **Predictive Maintenance** | S (2-3w) | Orchestrates Stats/ML + Signal + Ident — mostly shipped pieces. |
| 16 | **Fuzzy Logic** | S (2w) | Mamdani/Sugeno inference. Niche. |
| 17 | **Bioinformatics** | S (3-4w) | Smith-Waterman / Needleman-Wunsch, phylogenetic trees, BWT/FM-index. Self-contained numerics. |
| ~~18~~ | **Econometrics** | ✅ **SHIPPED 2026-05-26** | All 6 tiers — `arima`/`garch`/`egarch`/`gjr`/`varm`/`ssm`/`dssm`/`bayeslm`/`dtmc` + full test surface (`adftest`/`autocorr`/`egcitest`/…). See [`feature_status.md`](feature_status.md) + [`econometrics_toolbox_roadmap.md`](econometrics_toolbox_roadmap.md). Carve-downs: `regARIMA`, `msVAR`, `vecm` model object. |
| 19 | **Financial** + **Financial Instruments** | S+S (3w+3w) | Black-Scholes, MC pricing, yield curves, portfolio optim. |
| 20 | **Risk Management** | XS (1w) | VaR/CVaR/scenarios. Trivial atop Stats. |
| 21 | **Text Analytics** | S (2w) light / L heavy w/ embeddings | Tokenization/BoW/TF-IDF/LDA. Heavy if doing transformer embeddings. |
| 22 | **Medical Imaging** | M (3-4w) | DICOM I/O + 3-D volume registration + segmentation atop Image Processing. |
| 23 | **Model-Based Calibration** | S (2-3w) | DOE + GP modeling — Stats + Curve Fitting overlap. |

##### Tier 4 (wireless/RF family — only if pursuing comms research)

| # | Toolbox | Effort | Notes |
|---|---|---|---|
| 24 | **Radar Toolbox** | M (4w) **with Phased Array** | Range-Doppler, CFAR, SAR. Natural pair with #3. |
| 25 | **5G Toolbox** | XL (10w+) | OFDM, LDPC, polar codes, channel models, PUSCH/PUCCH. Massive surface. |
| 26 | **WLAN Toolbox** | L (6-8w) | 802.11 a/b/g/n/ac/ax PHY. |
| 27 | **LTE Toolbox** | L (~8w) | Legacy of 5G — bulk overlap. |
| 28 | **Satellite Communications** | M-L (6w) | DVB-S2X, orbit propagation (ties Aerospace). |
| 29 | **Bluetooth** | M (4w) | BT PHY / LE. Smaller surface than 802.11. |
| 30 | **Wireless HDL** | M (4-5w) | Pairs with DSP HDL — HDL emit lane for wireless functions. |
| 31 | **Wireless Network / Testbench** | L+ | Network simulation / HIL — out of pure-software scope mostly. |

##### Tier 5 (HDL + verification specialists, very narrow)

| # | Toolbox | Notes |
|---|---|---|
| 32 | **Vision HDL** | DL-CV pipelines targeting FPGAs. Build after CV. |
| ~~33~~ | **Deep Learning HDL** | ✅ **SHIPPED 2026-05-28** (H1–H5) — bundled with the Deep Learning Toolbox (row 7). INT8 weight quant (`dlquantize`/`dlqscale`), fi-typed SV emission of a quantized MLP (`dlhdl_quant_mlp.m`, EmitSV regression sweep), cocotb bit-accuracy via the new `% hdl: precise_fi` opt-in pragma, LSTM-on-FPGA combinational + sequential cells (`dlhdl_lstm_step.m` closes the recurrent loop inside the DUT via `persistent` + `always_ff`), minimal ONNX inference-graph importer (~56 ops). cocotb DL-HDL sweep 44/44.  Simulation surface only — silicon (bitstream/board/LIBIIO/vendor-synthesis) explicitly carved. |
| 34 | **Mixed-Signal / SerDes / Signal Integrity** | RF Toolbox extension for IC-level signaling. |
| 35 | **RF PCB Toolbox** | PCB EM solver. Heavy. |
| 36 | **Automated Driving** | Pairs with #4 (CV) + #6 (Navigation) + #1 (Sensor Fusion). |

#### Recommendation: next slice

If you want **maximum demo value per week**:

> **Aerospace (S) → Sensor Fusion (M) → Computer Vision (M) → Navigation (M)** ≈ 14-18 weeks. Completes a self-contained "autonomous drone" research stack on top of the shipped MPC + GPU + Symbolic + Quadrotor demos.

If you want **maximum strategic value** for compiler reach (now that Deep Learning has shipped):

> **Reinforcement Learning** — ✅ **SHIPPED** (PR #83). It rode directly on the shipped `dlnetwork`/`dlgradient` autodiff tape exactly as predicted ("trivial after DL"): all six tiers (tabular → DQN → REINFORCE → DDPG → TD3/PPO/SAC) plus a beyond-list **GRPO** agent with a verifier env. Pair the now-shipped DL+RL with **Computer Vision (M)** and **Text Analytics (S–L)** to convert them into the broader AI demo surface.

If you want **quick wins**:

> **Risk Management (XS) → Mapping (S) → Audio (S) → Aerospace (S) → Predictive Maintenance (S)** ≈ 11-13 weeks for 5 toolboxes, each independently demoable, all leveraging shipped components.

These are **plans, not commitments** — each tier is independently
shippable and demoable, and the per-toolbox docs are the source of truth.

---

## What's intentionally NOT on the roadmap

- **Full MATLAB language compatibility.** Pursuing this leads to
  toolbox dependencies and `.mat` file format edge cases that
  defeat the project's "self-contained, MathWorks-free" design
  goal.
- **GUI primitives** (`uicontrol`, `app designer` apps). The
  graphics roadmap entry above is rendering-only, no interaction.
- **Live Editor / `.mlx`** notebook format.
- **MEX file compatibility.** The runtime ABI is stable inside
  this project; cross-compatibility with MathWorks's MEX C
  interface is a separate engineering effort that brings little
  benefit since users on this stack want a MathWorks-free path.
- **Code obfuscation / encryption** (MathWorks `.p` files).

---

## Cross-cutting quality work

These don't fit a single roadmap slot but get folded into other
work as it lands:

- **Sema diagnostic robustness — clean rejection of unknown builtins.**
  Currently when a script calls an unshipped function name (e.g.
  `peaks`, `magma`, `surfl`), Sema accepts the call, MIR/MLIR builds
  a `matlab.call_builtin` op, and the LLVM translator chokes at the
  end of the pipeline with `missing LLVMTranslationDialectInterface
  registration for dialect for op: matlab.call_builtin`. The error
  is confusing because the *real* problem (an undefined function
  name) gets buried under translator-internal jargon, and downstream
  ops (`matlab.subscript`, `matlab.const_char`) cascade with the
  same message. **Fix**: reject unknown names at `Resolver.cpp` time
  with a clean `error: undefined function 'NAME'` and stop processing
  the rest of the script. Should cap the error cascade at one
  diagnostic per failing name instead of generating one per
  downstream use. **Effort**: ~2 sessions. **Status**: 🔵 in-flight
  alongside the plotting Tier-3 slice (peaks / surfc / extra colormaps).
- **Name-value-on-unknown-handle bail.** Related: when
  `set(handle, 'Name', value, ...)` or `title(text, 'Name', value)`
  is called on a handle returned by an unshipped function, the
  multi-pair name-value lowering leaves `matlab.const_char` ops in
  the IR that don't translate. **Fix**: if the handle is `none`-typed
  (because the producer is unknown), short-circuit the name-value
  pairs into a no-op so they don't leak into LLVM IR. **Effort**:
  ~3 sessions.


- **Test corpus growth.** Original ≥150 run-tests + 50 SV goldens target met during the data-container arc; current corpus is **248 run-tests** + **77 SV goldens**. Next milestone: ≥300 run-tests as the open SPT follow-ons (richer FIR, multitaper / STFT, strict Gustafsson `filtfilt`) and CST follow-ons (graph-style MIMO assembly via `connect`/`sumblk`/`lft`, H∞ norm, ss-form `minreal`) exercise more of the surface, and growing the SV-cocotb cycle-by-cycle lane (item #1) to cover all 39 HDL examples rather than just 8.
- **Formatter idempotency** verified by a fixed-point CI lane
  (parse → format → parse → format → identical).
- **Doc-up-to-dateness check** as a CI step (parse `feature_status.md`,
  verify every claimed `✅` has at least one test).
- **Performance benchmarks** baseline-tracked across releases —
  matrix-multiply / FIR / FFT / parfor reduction at a few sizes,
  recorded per commit.

---

## Update cadence

This file is updated at the end of each multi-week implementation
arc — most recently after the **Control System Toolbox arc**
(CST 1.1–5.1 numerical stack + SISO design loop + state-space
design + analysis + model reduction + interconnection plus §3.1
model objects and the model-object short-form surface — see
"Recently shipped" above), prior to that the **Signal Processing
Toolbox arc**
(SPT 1–13: windows, polynomial helpers, IIR/FIR design, close-the-loop,
transforms, spectral, LP, time-frequency, pulse measurements, multirate,
waveform generators, alignment), prior to that the data-container +
multi-return arc (Phases 1.1 / 1.2 / 1.3 / 2 / 3 / 4 / 5.1 / 5.2 / 5.3),
prior to that the SystemVerilog Phase 5.6 closure and the multi-backend
persistent + isempty Tier 1.

Items get demoted from this roadmap to `feature_status.md` /
the relevant `emit_*.md` once shipped. Items get retired (no
demote) when the design has been superseded by a different
approach.
