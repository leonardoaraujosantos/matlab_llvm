# System Identification Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** System-Identification-Toolbox programs.

Source: *System Identification Toolbox User's Guide* (R2026a, 1878
pages, 25 chapters: Choosing Your Approach · Data Import & Processing ·
Transform Data · Linear Model Identification · Process Models ·
Input-Output Polynomial Models · State-Space Models · Transfer Function
Models · Frequency-Response Models · Impulse-Response Models · Nonlinear
Black-Box · Hammerstein-Wiener · ODE/Grey-Box · Neural State-Space ·
Reduced Order Modeling · Time Series · Recursive ID · Online Estimation ·
Model Analysis · Preferences · Control Design Applications · Blocks ·
App · UI Help · Diagnostics & Prognostics).

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/ident/data_driven_mpc.m`](../examples/ident/data_driven_mpc.m):
*estimate a state-space plant from measured input-output data with
`ssest`, validate the fit with `compare` (NRMSE %), convert the
identified `idss` to a plain `ss`, then drop that `ss` straight into the
already-shipped `mpc(plant,p,m)` controller and run a closed-loop sim*.
This couples the System-ID core to **two** previously-shipped toolboxes
(Control System + MPC) and tells the canonical *data-driven control*
story; achieving it end-to-end is what closes **Ident-Tier-3** below.

Companion docs: [`feature_status.md`](feature_status.md),
[`roadmap.md`](roadmap.md),
[`control_toolbox_roadmap.md`](control_toolbox_roadmap.md) (identified
models convert to / from `ss`/`tf`/`zpk`/`frd`),
[`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md) (PEM is a
nonlinear least-squares problem solved with `lsqnonlin`/`fminunc`/
`fmincon`), [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md)
(spectral analysis + AR estimation + FFT reused wholesale),
[`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md) (headline drops the
identified plant into `mpc(...)`).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the smallest end-to-end LLVM-lane loop: the `iddata` container +
  `arx`/`ar` linear-least-squares estimation + `sim`/`predict`/`compare`
  validation + model→`tf`/`ss` conversion so the existing CST plots
  (`bode`/`step`/`pzmap`) light up. **Tier-2** is the PEM core
  (`armax`/`oe`/`bj`/`iv4` nonlinear polynomial models + residual
  analysis + order selection). **Tier-3** closes the headline demo
  (state-space `n4sid`/`ssest`, `tfest`, `procest`, `era` + the
  `idss`/`idtf`/`idproc` model classes + `idmodel`→`ss`/`tf`
  conversion). **Tier-4** is frequency-domain / spectral (`spa`/`etfe`/
  `spafdr`→`idfrd`), `impulseest` correlation analysis, linear grey-box
  (`greyest`/`idgrey`), and `forecast`. **Tier-5** is nonlinear
  black-box (`nlarx`/`nlhw`), nonlinear grey-box (`nlgreyest`), and the
  recursive/online stack (`recursive*` + `extendedKalmanFilter`/
  `unscentedKalmanFilter`/`particleFilter`). **Tier-6** is the
  carve-down polish sweep (mirrors the MPC-roadmap pattern).
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: T1 ~1.5 wk,
  T2 ~2 wk, T3 ~2.5 wk, T4 ~2 wk, T5 ~3 wk, T6 ~1 wk (~12 wk full).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started. **ALL
  six tiers shipped 2026-05-20** (`runtime/toolbox/ident/`).  Tier-3
  closes the `data_driven_mpc` headline; Tier-4 adds spectral / impulse /
  grey-box; Tier-5 adds EKF/UKF state estimation, recursive RLS, and
  nonlinear grey-box; Tier-6 adds regularized ARX + parameter
  introspection (`arxOptions`, `getcov`, `getpvec`, `setpvec`).
  Carve-downs (nonlinear-black-box `nlarx`/`nlhw`, `particleFilter`,
  estimation `Report` struct, MIMO) documented per tier.
- **Compile/Execute path** (identical pattern across rows): Sema
  registers each estimator/method name in
  [`lib/Sema/Resolver.cpp::registerBuiltins()`](../lib/Sema/Resolver.cpp);
  per-builtin shape/dtype rules go in
  [`lib/Sema/TypeInference.cpp`](../lib/Sema/TypeInference.cpp);
  `matlab.call_builtin @name(...)` is rewritten to
  `llvm.call @matlab_ident_*(...)` inside
  [`LowerTensorOps.cpp`](../lib/MLIR/Passes/LowerTensorOps.cpp) (split
  into a dedicated `LowerIdent.cpp` once Ident entries exceed ~10 rows —
  same precedent as PDE / Comm / Optim); runtime entries live in a new
  [`runtime/toolbox/ident/runtime_ident.cpp`](../runtime/toolbox/ident/runtime_ident.cpp)
  with the model classes in
  [`runtime/toolbox/ident/ident_classdefs.m`](../runtime/toolbox/ident/ident_classdefs.m),
  mirroring `runtime/toolbox/{optim,mpc}/`.
- **Class auto-prelude**: extend the
  [`tools/matlabc/main.cpp`](../tools/matlabc/main.cpp) prelude table
  (`findCstPrelude`/`userMentionsCstClasses` pattern) so that mentioning
  `iddata` / `idpoly` / `idss` / `idtf` / `idproc` / `idfrd` / `idgrey`
  / `idnlarx` / `idnlhw` / `idnlgrey` / `nlarx` / `arx` / `ssest` / …
  auto-prepends `ident_classdefs.m`. Add the model classes to the
  `IsCstClass`-style allowlist in Sema so property reads get
  matrix-typed.
- **Debug / REPL**: every new descriptor type (`iddata`, `idfrd`, and
  the `idpoly`/`idss`/`idtf`/`idproc`/`idgrey`/`idnlarx`/`idnlhw`/
  `idnlgrey` model objects, plus the estimation-`Report` struct) needs a
  renderer in
  [`runtime/runtime_debug.cpp`](../runtime/runtime_debug.cpp)
  (`matlab_ws_set_*` family) and a DAP child-walker — same pattern as
  `tf`/`ss` (`matlab_tf_disp`) in CST and `OptimizationProblem` in Optim.
  The `idpoly` renderer reuses `matlab_tf_disp`'s polynomial-fraction
  pretty-printer; `iddata` renders as a channel/sample summary table.
- **No external solver dependencies**: matching the project's hand-coded
  precedent (PDE's Lanczos, Control's Schur/Lyap/Riccati, Optim's LM/SQP,
  MPC's KWIK QP), System-ID is hand-coded too — **no N4SID library, no
  CT/ML autodiff, no external NLP**. The numeric fuel (SVD, QR,
  `mldivide` least-squares, `lsqnonlin`/`fminunc`/`fmincon`, `fft`,
  `pwelch`/`cpsd`/`tfestimate`, `levinson`/`aryule`/`arburg`, `ode45`/
  `ode23s`) already ships.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Ident code yet)

The following primitives already exist and **do not need to be re-built**
for System-ID. This toolbox sits on an unusually deep shipped base — most
of the numeric work is already done; the new code is mostly *model
classes + the regressor/predictor machinery that glues data to solvers*.

| Group | Surface (already shipped) | Location | How System-ID uses it |
|---|---|---|---|
| LTI model classes | `ss` (A/B/C/D/Ts), `tf`, `zpk`, `pid`, `frd` classdefs + operator overloads | `runtime/toolbox/control/cst_class_*.m` | Identified models **convert to** these; `idfrd` reuses `frd`. |
| LTI analysis | `bode`/`step`/`impulse`/`initial`/`lsim`/`freqresp`/`nyquist`/`pzmap`/`dcgain`/`stepinfo` | `runtime/matlab_runtime.{h,cpp}` (`matlab_*_ss`/`_tf`) | Validation plots for identified models — convert then call. |
| Discretization | `c2d`/`d2c` (ZOH + Tustin) | `matlab_c2d_*`, `matlab_d2c_*` | Continuous↔discrete identified models; CT estimation. |
| Dense linear algebra | `mldivide` (incl. **overdetermined least-squares**), `qr`, `svd` (+`_U`/`_S`/`_V`), `lu`, `chol`, `eig`, `pinv` | `runtime/matlab_runtime.cpp` | **ARX = QR least-squares**; **n4sid = block-Hankel SVD**; ERA = Hankel SVD. |
| Nonlinear solvers | `lsqnonlin` (Levenberg-Marquardt), `lsqcurvefit`, `fminunc` (BFGS+FD), `fmincon` (aug-Lagrangian), `lsqlin`, `lsqnonneg` | `runtime/toolbox/optim/runtime_optim.cpp` | **PEM core** — `armax`/`oe`/`bj`/`ssest`/`tfest`/`procest`/`greyest`/`nlarx` all minimize prediction error via these. |
| Function-handle ABI | `void *fn_p` cast to typed fn-ptr inside runtime (proven by `lsqnonlin`, `nlmpc` StateFcn) + `LowerAnonCalls` retyping | `runtime_optim.cpp`, `LowerAnonCalls.cpp` | Grey-box ODE structure functions + nonlinear-ARX custom regressors. |
| ODE solvers | `ode45`, `ode23`, `ode23s` (function + vector forms) | `runtime/matlab_runtime.cpp` | `idnlgrey`/`idgrey` continuous-time state rollout during PEM. |
| FFT | `fft`/`ifft`/`fft2`/`ifft2` (complex descriptor) | `matlab_fft_c` etc. | Time↔frequency data transform on `iddata`; frequency-domain estimation. |
| Spectral analysis | `pwelch`, `cpsd`, `mscohere`, `tfestimate`, `periodogram`, `spectrogram` | `matlab_runtime.h` / `runtime_complex.cpp` | **`spa`/`spafdr`/`etfe`** spectral models lean directly on these. |
| AR / linear prediction | `levinson`, `lpc`, `aryule`, `arburg`, `pyulear`, `pburg` | `matlab_runtime.h` | **`ar` time-series estimation** is nearly free (Yule-Walker / Burg). |
| Filtering / resampling | `filter`, `filtfilt`, `sosfilt`, `resample`, `decimate`, `interp`, `xcov`, `finddelay` | `matlab_runtime.h` | Data preprocessing (`idfilt`/`detrend`/`resample`), `impulseest` (`xcov`), `delayest` (`finddelay`). |
| Kalman (steady-state) | `kalman`/`kalmd` gain `L`, error covariance `P` | `matlab_kalman_*`, `matlab_kalmd_*` | Innovations-form `idss` `K` matrix; **NOT** the dynamic recursive loop (that is new — see Tier-5). |
| Classdef plumbing | `matlab_obj_new`/`matlab_obj_set_*`/`matlab_obj_get_mat`, kwarg-ctor sugar (`Name`,value), class-pinned method dispatch, REPL persistence (`matlab_ws_set_obj`) | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | Host for every `id*` model class + `iddata`/`idfrd`. |
| Sema/Lowering | `registerBuiltins()` array; `matlab.call_builtin`→`llvm.call` table | `lib/Sema/Resolver.cpp`, `LowerTensorOps.cpp` | Add Ident names; per-builtin shape rules in `TypeInference.cpp`. |

**Net assessment**: the *numerics* are ~80% present. The genuinely new
work is (a) the **`iddata` data container** and the **`id*` model
classes**, (b) the **regressor-matrix / one-step-predictor machinery**
that turns data + a model structure into a least-squares or PEM problem,
(c) **subspace identification** (n4sid block-Hankel SVD), and (d) the
**recursive/adaptive + EKF/UKF/PF dynamic filtering loop** (none of which
exists today).

---

## 2. Tier-1 — Smallest end-to-end Ident loop (`iddata` + `arx`/`ar` + validate) ✅ shipped 2026-05-20

Goal: load/construct time-domain data, estimate a linear ARX (or AR
time-series) model by least-squares (the MathWorks ARX algorithm —
"least-squares estimation using QR-factorization for overdetermined
linear equations"), then `sim`/`predict`/`compare` it and reuse the CST
machinery. Shipped in
[`runtime/toolbox/ident/runtime_ident.cpp`](../runtime/toolbox/ident/runtime_ident.cpp)
+ [`ident_classdefs.m`](../runtime/toolbox/ident/ident_classdefs.m).

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 1.1 | `iddata(y,u,Ts)` class | ✅ | OutputData/InputData (`matrix`-annotated) + Ts/Tstart; `iddata(y,[],Ts)` for a time series. SISO. | `ident_classdefs.m` |
| 1.2 | `arx(data,[na nb nk])` | ✅ | Regressor Φ from lagged y,u; θ via **normal equations** `ΦᵀΦθ=Φᵀy` (the shared `mldivide` only factors square systems — QR-on-Φ is a Tier-2 numerical follow-on). Returns `idpoly`. | `matlab_ident_arx` |
| 1.3 | `ar(data,na)` | ✅ | Time-series AR — reuses shipped `aryule` (Yule-Walker / Levinson-Durbin) | `matlab_ident_ar` |
| 1.4 | `idpoly` class | ✅ | A/B/C/D/F (`matrix`-annotated) + Ts + NoiseVariance + nk + Np/Ns. Zero-arg ctor (arx/ar populate in place). | `ident_classdefs.m` |
| 1.5 | `sim(model,u)` | ✅ | Deterministic output via `matlab_filter(B,A,u)` | `matlab_ident_sim` |
| 1.6 | `predict(model,data,K)` | ✅ | 1-step ARX predictor `ŷ=(1−A)y+Bu`; K≥1e6 (`Inf`) ≡ sim. Arbitrary-K is Tier-2. | `matlab_ident_predict` |
| 1.7 | `compare(data,model)` | ✅ | **NRMSE fit %** (sim when input present, else 1-step predict). Returns the scalar fit. Multi-return `[y,fit,x0]` is Tier-6. | `matlab_ident_compare` |
| 1.8 | model→`ss`/`tf` | ✅ | `ss(idpoly)` controllable-canonical (carries discrete Ts) + `tf(idpoly)` B/A extraction → CST `pole`/`step`/`bode`. Dispatched in the constructor path (`ss`/`tf` resolve as classes). | `matlab_ident_poly2ss_{A,B,C,D}` |
| 1.9 | quality metrics | ✅ | `fpe = V·(N+d)/(N−d)`, `aic = N·log V + 2d`, `goodnessOfFit` (NRMSE cost). `aicc`/`bic`/MSE/MAE variants → Tier-6. | `matlab_ident_fpe` / `_aic` / `_goodness` |

**Headline-within-tier (shipped)**:
[`examples/ident/arx_lab_process.m`](../examples/ident/arx_lab_process.m)
— the *"Estimating Simple Models from Real Laboratory Process Data"*
workflow (UG §4).  600-sample second-order lab record with measured
input + disturbance → `arx [2 2 1]` (recovers `A=[1,-1.5,0.7]`,
`B=[0,1.0,0.5]`) → `compare` 96.95 % fit → `ss(model)` poles
`0.75 ± 0.37i`.

**Gating tests** (LLVM lane, `.skip-emit-*` markers): `ident_t1_arx.m`
(exact ARX recovery + Np/Ns/V), `ident_t1_ar.m` (Yule-Walker AR),
`ident_t1_validate.m` (sim/predict/compare/goodnessOfFit, 100 % fit),
`ident_t1_convert.m` (ss/tf conversion + `pole`).

**Compile/Execute wiring (as built)**: `arx`/`ar`/`compare`/`predict`/
`fpe`/`aic`/`goodnessOfFit` registered in `Resolver.cpp`; `arx`/`ar` →
`idpoly` pin in `pinnedOfRhs`; class-pinned-first-arg dispatch in
`Lowering.cpp` keyed on `iddata`/`idpoly` (coexists with the
identically-named MPC `sim` + CST `ss`/`tf` routes); `ss(idpoly)` /
`tf(idpoly)` intercepted in the constructor-call path; loose-match
`matlab_ident_*` entries in `LowerTensorOps.cpp` `pde_table`; prelude
auto-inject of `ident_classdefs.m` in `tools/matlabc/main.cpp` (REPL +
AOT tables, `ident` added to both `kToolboxDirs`).

**Tier-1 carve-downs** (deferred): MIMO `iddata`/`arx`; `iddata` subref
`z(1:100)`/`size`/`get`/`set`; `idpoly` REPL pretty-printer (reusing
`matlab_tf_disp`); direct `idpoly(A,B,…)` construction; `polydata`/
`tfdata` extractors; QR-conditioned (vs normal-equations) LS; arbitrary-K
predictor; `compare` multi-return + plotting.

---

## 3. Tier-2 — PEM core: nonlinear polynomial models + validation ✅ shipped 2026-05-20

Goal: the iterative prediction-error estimators. Every linear polynomial
model `A·y = (B/F)·u + (C/D)·e` shares **one** general predictor — the
prediction error is `e = (D/C)·(A·y − (B/F)·u)` (`compute_pe` in the
runtime; ARX/ARMAX/OE/BJ are all special cases). armax/oe/bj minimise
`‖e(θ)‖²` over the polynomial coefficients with the shipped **`lsqnonlin`
(Levenberg-Marquardt)**, feeding `y`/`u`/orders to a file-static residual
callback through a `thread_local` context (the proven `nlmpc` pattern),
initialised from an ARX fit.

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 2.1 | `armax(data,[na nb nc nk])` | ✅ | A·y = B·u + C·e; PEM over [a,b,c], ARX-seeded | `matlab_ident_armax` |
| 2.2 | `oe(data,[nb nf nk])` | ✅ | Output-error y = (B/F)·u + e; PEM over [b,f] | `matlab_ident_oe` |
| 2.3 | `bj(data,[nb nc nd nf nk])` | ✅ | Box-Jenkins y = (B/F)·u + (C/D)·e; PEM over [b,c,d,f] | `matlab_ident_bj` |
| 2.4 | `iv4(data,[na nb nk])` | ✅ | Instrumental-variable ARX (noise-colour robust): instruments = ARX-simulated x̂, solve (ZᵀΦ)θ=ZᵀY. Full 4-stage prefilter → follow-on. | `matlab_ident_iv4` |
| 2.6 | `resid(model,data)` | ✅ | Whiteness diagnostic: returns `[maxAutoCorr; maxCrossCorr]` (normalised residual autocorr lags 1..M + residual/input cross-corr). The plot + 99 % bands are a carve-down. | `matlab_ident_resid` |
| 2.7 | `pe(model,data)` | ✅ | Prediction-error vector (the general `compute_pe` residual) | `matlab_ident_pe` |
| 2.8 | `delayest(data)` | ✅ | Input-delay estimate: ARX[2 2 nk] loss-minimising scan over nk | `matlab_ident_delayest` |

**Also (Tier-1 generalised by Tier-2)**: `sim` is now `B/(A·F)·u` (correct
for OE/BJ, not just `B/A`); `predict` 1-step is now `ŷ = y − pe` (correct
for every structure).

**Headline (shipped)**:
[`examples/ident/armax_refine.m`](../examples/ident/armax_refine.m) — the
*"Estimate Models Using armax"* refinement story (UG §6).  1000-sample
ARMAX record (C=[1 0.7]); a baseline `arx` leaves **coloured residuals
(max autocorr 0.46)**, then `armax` recovers `C2=0.71` and **whitens them
(0.10)**.

**Gating tests** (LLVM lane): `ident_t2_armax.m` (A/B/C recovery + fit),
`ident_t2_oe.m` (B/F recovery), `ident_t2_bj.m` (B/F recovery + fit),
`ident_t2_resid.m` (pe length + whiteness stats), `ident_t2_iv4.m` (IV
de-biasing + `delayest`).

**Compile/Execute wiring (as built)**: `armax`/`oe`/`bj`/`iv4` →
`idpoly` pin in `pinnedOfRhs`, alloc-then-populate dispatch (same shape
as `arx`); `pe`/`resid` return matrices, `delayest` returns f64; all in
the `iddata`/`idpoly`-keyed `Lowering.cpp` block + loose-match
`pde_table` entries.

**Tier-2 carve-downs** (deferred): `pem(data,template)` generic refine;
`struc`/`arxstruct`/`selstruc` order-grid search (needs matrix-of-orders
plumbing); `*Options` objects (Focus / SearchMethod / MaxIterations);
MIMO; QR-conditioned LS; OE/BJ stability enforcement during the LM
search (today relies on the ARX-seeded start staying near a stable
optimum); ARIMA/integrator noise.

---

## 4. Tier-3 — State-space + transfer-function estimation (closes the headline) ✅ shipped 2026-05-20

Goal: the workhorse black-box estimators and the cross-toolbox payoff.
Subspace identification is the one genuinely new numeric block; it is
realized here via **Ho-Kalman / ERA on estimated Markov parameters**: a
high-order FIR least-squares fit gives the impulse response `g(0..2s)`,
whose block-Hankel is factored by SVD (taken through the symmetric Gram
`HᵀH`, since the shipped `matlab_svd` returns only singular values) to
extract `A`/`B`/`C`/`D`. Any valid realization is similarity-equivalent,
which is all the downstream `ss`/`mpc` consumers need.

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 3.1 | `idss` class | ✅ | A/B/C/D/K/x0 + Ts + NoiseVariance/Np/Ns; zero-arg ctor, matrix-annotated props | `ident_classdefs.m` |
| 3.2 | `n4sid(data,nx)` | ✅ | Subspace via FIR-Markov + ERA Hankel-SVD (Gram eig); SISO. Projection-based N4SID + auto-order + CVA/MOESP weights → follow-on. | `matlab_ident_n4sid` |
| 3.3 | `ssest(data,nx)` | ✅ | Same subspace core (**headline estimator**); PEM refinement of the seed → follow-on. | `matlab_ident_ssest` |
| 3.6 | `tfest(data,np,nz)` | ✅ | Discrete TF = OE structure, so maps to `oe(data,[nz+1,np,0])`; returned in idpoly form (B=num, F=den). idtf Num/Den-named class → fidelity follow-on. | `matlab_ident_tfest` |
| 3.8 | `idmodel`→`ss`/`tf` | ✅ | `ss(idss)` copies A/B/C/D/Ts → CST `ss` (constructor-path interception); `ss(idpoly)`/`tf(idpoly)` from Tier-1. | `matlab_ident_ss_{A,B,C,D}` |
| — | `sim`/`compare` for idss | ✅ | State-space `x⁺=Ax+Bu, y=Cx+Du` sim + NRMSE; routed by the model's class. | `matlab_ident_sim_ss` / `_compare_ss` |

**🎯 Headline tracer-bullet (shipped, closes Tier-3)**:
[`examples/ident/data_driven_mpc.m`](../examples/ident/data_driven_mpc.m)
— `z = iddata(y,u,0.1); sys = ssest(z,2); compare(z,sys); P = ss(sys);
ctrl = mpc(P,10,3); sim(ctrl,30,1.0)`. 600 I/O samples → order-2 subspace
estimate (poles `0.75 ± 0.37i` recovered exactly) → **96.8 % fit → `ss`
→ shipped `mpc` → closed-loop step tracking `r=1.0` to `1.000`**. Couples
System ID → Control System → MPC with no first-principles model.

**Gating tests** (LLVM lane): `ident_t3_ssest.m` (nx/Ts/fit + trace/det
pole invariants), `ident_t3_tfest.m` (denominator recovery + fit),
`ident_t3_data_driven.m` (the full chain incl. ident+mpc prelude
coexistence).

**Compile/Execute wiring (as built)**: `n4sid`/`ssest` → `idss` pin +
alloc-then-populate (scalar `nx`); `tfest` → `idpoly` pin (OE path);
`sim`/`compare` route to `_ss` variants when the model is `idss`
(compare keys on the arg-1 class); `ss(idss)` intercepted in the
constructor-call path alongside `ss(idpoly)`.

**Tier-3 carve-downs** (deferred): projection-based N4SID + auto-order
(SVD-gap) + CVA/SSARX/MOESP weights; PEM refinement of the subspace seed;
innovations gain `K` (currently 0); `ssregest`; standalone `era(H,nx)`;
`idtf`/`idproc` named classes + `procest` (continuous process models —
need a continuous-time + constrained-`fmincon` fit); `zpk(idsys)`;
`findstates`; MIMO.

---

## 5. Tier-4 — Frequency-domain / spectral + impulse-response + linear grey-box ✅ shipped 2026-05-20

Goal: the non-parametric estimators (nearly free given the shipped FFT),
the `idfrd` frequency-response object, and user-structured linear
grey-box.

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 5.1 | `idfrd` class | ✅ | Frequency-response data — Frequency (rad/s) + ResponseMag + ResponsePhase (real cols, sidestepping complex obj storage) + Ts | `ident_classdefs.m` |
| 5.2 | `etfe(data)` | ✅ | Empirical TF estimate `fft(y)./fft(u)` via the shipped complex `matlab_fft_c`; one-sided. | `matlab_ident_etfe` |
| 5.3 | `spa(data)` | ✅ | Frequency-smoothed cross/auto spectra `Φyu/Φuu` (Blackman-Tukey-style moving average). `spafdr` + lag-window variants → follow-on. | `matlab_ident_spa` |
| 5.4 | `impulseest(data,N)` | ✅ | First N Markov params via the shared FIR least-squares; returned as an idpoly FIR (A=1, B=g). `cra`/regularized FIR → follow-on. | `matlab_ident_impulseest` |
| 5.6 | `idgrey` + `greyest` | ✅ | **Linear grey-box (headline)**: user `structfn(par)` returns the packed continuous `[A B; C D]` (matlab_mat ABI), greyest ZOH-discretizes + PEM-fits `par` via `lsqnonlin`. | `matlab_ident_greyest` |
| 5.7 | `forecast(model,data,K)` | ✅ | K-step time-series forecast — deterministic A/B recursion forward (future e=0, future u=0). Confidence bands + ARMAX C/D → follow-on. | `matlab_ident_forecast` |

**🎯 Headline (shipped)**:
[`examples/ident/greybox_msd.m`](../examples/ident/greybox_msd.m) — recover
the physical constants of a mass-spring-damper from data.  `structfn =
@(p) [0 1 0; -p(1) -p(2) 1; 1 0 0]` maps `par=[k/m,c/m]` to the packed
realization; `greyest(z, [3;1], structfn, 2)` recovers **k/m = 4.0000,
c/m = 1.1960 (ω_n = 2.0 rad/s), 99.92 % fit**.  Uses the nlmpc-style
function-handle ABI + ZOH `c2d` + `lsqnonlin`.

**Gating tests** (LLVM lane): `ident_t4_greybox.m` (param recovery),
`ident_t4_impulse.m` (impulseest Markov params + forecast vs AR
recursion), `ident_t4_spectral.m` (etfe/spa DC gain = static gain 7.5,
Nf = N/2+1).

**Compile/Execute wiring (as built)**: `greyest` handle arg retyped in
`LowerAnonCalls` (operand 2 user / operand 3 runtime, mirroring
`nlmpcmove`); `greyest`→`idgrey`, `etfe`/`spa`→`idfrd`, `impulseest`→
`idpoly` pins; `idgrey` reuses the idss state-space `sim`/`compare`/`ss`
routes; `etfe`/`spa` use the shipped `matlab_fft_c` (re/im).

**Tier-4 carve-downs** (deferred): `spafdr` + lag-window spectral;
`cra`/regularized-FIR `impulseest`; frequency-domain estimation
(`arx`/`tfest`/`ssest` on `idfrd`); ARMAX `forecast` + confidence bands;
continuous-time grey-box returning `[A,B,C,D,K,x0]` as separate outputs;
data preprocessing (`detrend`/`idfilt`/`resample`/`merge`/`misdata`);
`getpvec`/`setpvec`/`getcov`; MIMO.

---

## 6. Tier-5 — Nonlinear grey-box + recursive/online + state estimation ✅ shipped 2026-05-20

Goal: the heavy slice. The nonlinear grey-box reuses the function-handle
ABI + `lsqnonlin`; the EKF/UKF stack is **the only fully-new numeric
subsystem** — the dynamic Kalman filtering loop (the shipped CST
`kalman` is steady-state-gain only).

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 6.3 | `idnlgrey` + `nlgreyest` | ✅ | Nonlinear grey-box: user ODE rhs `StateFcn(z=[x;u;par]) → dx/dt`; Euler rollout + `lsqnonlin` param fit. | `matlab_ident_nlgreyest` |
| 6.4 | `recursiveARX` / `recursiveLS` | ✅ | Forgetting-factor RLS `θ=θ+K(y−φᵀθ)`, `K=Pφ/(λ+φᵀPφ)`, `P=(P−KφᵀP)/λ`. recursiveARX buffers the I/O to build φ; tracks time-varying dynamics. `recursiveAR`/Kalman/gradient variants → follow-on. | `matlab_ident_rarx_*` / `_rls_*` |
| 6.6 | `extendedKalmanFilter` | ✅ | **EKF**: `predict(obj,@StateFcn)` + `correct(obj,@MeasFcn,y)`; FD Jacobian; mutable State/StateCovariance. Scalar measurement. | `matlab_ident_ekf_{predict,correct}` |
| 6.7 | `unscentedKalmanFilter` | ✅ | **UKF**: Julier scaled-unscented-transform sigma points (Cholesky spread) predict/correct. | `matlab_ident_ukf_{predict,correct}` |

**🎯 Headlines (shipped)**:
[`examples/ident/ukf_state_estimation.m`](../examples/ident/ukf_state_estimation.m)
— reconstruct a pendulum's full state (incl. the **never-measured
velocity**, `−0.708` vs true `−0.705`) from noisy angle-only data with
the UKF/EKF (UG Ch.18 *Nonlinear State Estimation Using UKF*); and
[`examples/ident/recursive_arx_tracking.m`](../examples/ident/recursive_arx_tracking.m)
— a `recursiveARX` follows a plant pole that jumps `0.50 → 0.85`
sample-by-sample (UG *Online ARX Parameter Estimation for Tracking
Time-Varying System Dynamics*).

**Gating tests** (LLVM lane): `ident_t5_ekf.m` (EKF/UKF pendulum state
recovery), `ident_t5_recursive.m` (recursiveARX jump-tracking +
recursiveLS), `ident_t5_nlgrey.m` (nonlinear grey-box param recovery).

**Compile/Execute wiring (as built)**: EKF/UKF + `nlgreyest` handles
retyped in `LowerAnonCalls` (predict/correct op 1, nlgreyest op 3);
`extendedKalmanFilter`/`unscentedKalmanFilter`/`recursiveLS`/
`recursiveARX` constructed via constructor-path alloc-then-populate
(zero-arg shell + runtime init — bare-param matrix assignment in a ctor
body is brittle); `predict`/`correct`/`step` dispatched by the filter
class; mutable objects updated in place (the `mpcstate` pattern).

**Tier-5 carve-downs** (deferred): **nonlinear black-box** `idnlarx`/
`nlarx` + `idnlhw`/`nlhw` (the mapping-object surface: `idSigmoidNetwork`/
`idWaveletNetwork`/`idTreePartition`); `particleFilter`; `recursiveARMAX`/
`recursiveOE`/`recursiveBJ` recursive-PEM; `recursiveAR` + Kalman/gradient
recursive variants; analytic Jacobians / multi-output measurement for
EKF/UKF; `ode45`/`ode23s` (vs Euler) rollout in `nlgreyest`; MIMO.

---

## 7. Tier-6 — Carve-down sweep / polish ✅ shipped 2026-05-20

A focused pass over items deferred from earlier tiers — mirrors the
MPC-roadmap Tier-6 pattern.  This tier closes the toolbox arc: every
named tier is now shipped.

| # | Surface | Status | Algorithm / notes | Runtime entry |
|---|---|:-:|---|---|
| 7.1 | `arxOptions` + 3-arg `arx(data,orders,opt)` | ✅ | Ridge LS: `θ = (ΦᵀΦ + λI)⁻¹·Φᵀy`.  `opt.Regularization` is the scalar λ.  idpoly gains a `Lambda` field carrying λ. | `matlab_ident_arx_reg` |
| 7.2 | `getcov(model)` | ✅ | Parameter covariance `V · (ΦᵀΦ + λI)⁻¹` from the regularized Gram arx now caches on the idpoly (`Gram` field). 0×0 for non-arx models (PEM Fisher-information variant → follow-on). | `matlab_ident_getcov` |
| 7.3 | `getpvec(model)` / `setpvec(model, θ)` | ✅ | Reconstruct / write the packed parameter vector `θ = [a₁..a_na, b_{nk+1}..b_{nk+nb}]`.  Preserves nk and monic A(1)=1. | `matlab_ident_getpvec` / `_setpvec` |

**Headline (shipped)**:
[`examples/ident/arx_regularization.m`](../examples/ident/arx_regularization.m)
— the *"Regularized Estimates of Model Parameters"* workflow (UG §1).
On a 60-sample noisy ARX record, plain `arx` returns `(a=-0.516, b=1.053)`;
ridged `arx(..., opt)` with `λ=1` returns `(a=-0.510, b=1.035)` — closer
to truth `(-0.5, 1.0)`.  Demonstrates the `arxOptions` carrier and the
`getpvec`/`setpvec`/`getcov` introspection trio.

**Gating tests** (LLVM lane): `ident_t6_arx_reg.m` (arxOptions plumbing,
Lambda round-trip), `ident_t6_introspect.m` (getpvec / setpvec /
getcov shapes).

**Compile/Execute wiring (as built)**: `arxOptions` registered as a
classdef + pinned in `pinnedOfRhs`; 3-arg arx dispatch reads
`opt.Regularization` via `matlab_obj_get_f64` and routes to
`matlab_ident_arx_reg`; `getcov`/`getpvec`/`setpvec` class-pinned-first-
arg dispatch on `idpoly`; `arx_ls` extended to accept λ and optionally
return the Gram (cached on the idpoly as `Gram`); the existing 5-arg
`arx_ls` callers (`armax`/`oe`/`bj` PEM seeds, `iv4`, `delayest`) route
through a thin wrapper with λ = 0.

**Open follow-ons (carve-down of the carve-down sweep)**:
- Estimation `Report` struct (Fit.FitPercent / .LossFcn / .MSE /
  .FPE / .AIC, Parameters.Free, SearchInfo.Iterations) populated and
  REPL-rendered.
- `[model, x0]` / `[y, fit, x0]` multi-return forms across estimators.
- Uncertainty bands on `bode` / `step` (`showConfidence`); confidence
  intervals on `forecast`.
- Other `*Options` carriers (`armaxOptions` / `oeOptions` / `bjOptions`
  / `ssestOptions`) — same shape as `arxOptions`.
- Fisher-information covariance for PEM-fit models (`getcov` on
  `armax`/`oe`/`bj`/`ssest`).
- `merge` of models + multi-experiment `iddata`; ARIMA / seasonal
  (`D`-integrator) idpoly; `translatecov`; `sim` with noise.

---

## 8. Carve-outs (explicitly out of scope)

Matching the established roadmap discipline (GUI / Simulink / external-DL
deps are always carved):

- **System Identification App** + **Time Series Modeler app** + **System
  Identification UI Help** (Chapters 23–24) — interactive GUIs; the
  command-line API is the whole surface here.
- **Simulink blocks** (Chapter 22: online-estimation blocks,
  simulate-identified-model block, Recursive Estimator block). A future
  `mflow` RecursiveEstimator/IdModel block could host these through the
  Embedded-Coder lanes (cf. MPC's `MpcMove` block), but it is **not** in
  this roadmap.
- **Neural State-Space** (`idNeuralStateSpace`, Chapter 14) + **LSTM /
  cascade-correlation / `narxnet`** nonlinear models — Deep Learning
  Toolbox dependency.
- **Machine-learning NLARX mappings** (Gaussian-process, regression-tree,
  `idGaussianProcess`) — Statistics & ML Toolbox dependency.
- **Reduced Order Modeling** chapter (Chapter 15) — LPV/neural-SS ROM,
  depends on the carved neural-SS + LPV machinery.
- **C-MEX grey-box files** (`idnlgrey` C-MEX form) — MEX FFI; only the
  **MATLAB-file** grey-box form (6.3) ships.
- **Diagnostics & Prognostics** (Chapter 25) — Predictive Maintenance
  Toolbox overlap.
- **App-only data import** (drag-drop, session `.sid` files), App
  preferences (Chapter 20).

---

## 8b. Open follow-ons (carve-down index)

All six tiers are shipped, but each closed tier left a small set of
deferred items so the core could ship sooner.  This section consolidates
every "next slice" + the §8 explicit carve-outs into one scannable
backlog, ordered roughly by user-visible payoff.  Pick from the top when
re-opening the toolbox.

### Highest-value remaining work

| # | Item | Tier | Effort | Why it matters |
|---|---|---|---|---|
| F1 | **`nlarx` / `idnlarx`** — nonlinear ARX with the mapping-object surface (`idSigmoidNetwork`, `idWaveletNetwork`, `idTreePartition`); custom-regressor handles; PEM via `lsqnonlin`. | T5 | ~3 sess | The single biggest functional gap.  Headline candidate: nonlinear two-tank / engine-torque identification. |
| F2 | **`nlhw` / `idnlhw`** — Hammerstein-Wiener (input NL → LTI → output NL); piecewise-linear / sigmoid / saturation / deadzone estimators. | T5 | ~2 sess | Pairs with F1 — same `lsqnonlin` infrastructure. |
| F3 | **Estimation `Report` struct** — `Report.Fit.FitPercent` / `.LossFcn` / `.MSE` / `.FPE` / `.AIC` / `Report.Parameters.Free` / `Report.SearchInfo.Iterations`, populated by every estimator and REPL-rendered. | T6 | ~1 sess | MathWorks-standard model-introspection API; trivially scaffoldable on the cached `NoiseVariance` / `Np` / `Ns`. |
| F4 | **Multi-return forms** — `[model, x0] = ssest(...)` / `[y, fit, x0] = compare(...)` / `[θ, P] = arx(...)` etc. across estimators. | T6 | ~1 sess | Closes a frequent UG paste-and-run failure mode. |
| F5 | **Other `*Options` carriers** — `armaxOptions` / `oeOptions` / `bjOptions` / `ssestOptions` (Focus, InitialCondition, EnforceStability, OutputWeight, Regularization Lambda/R/Nominal).  Mirrors the shipped `arxOptions`. | T6 | ~2 sess | Unblocks production-grade PEM tuning. |
| F6 | **`particleFilter`** — sequential Monte-Carlo state estimation (StateTransitionFcn + MeasurementLikelihoodFcn handles, systematic-resampling). | T5 | ~2 sess | Completes the EKF/UKF/PF trio.  Same function-handle ABI; only the resampling is new. |
| F7 | **Fisher-information `getcov` for PEM-fit models** — currently `getcov` returns the asymptotic LS covariance `V·(ΦᵀΦ+λI)⁻¹` only for `arx`; PEM (`armax`/`oe`/`bj`/`ssest`/`greyest`) needs `J(θ̂)ᵀ·J(θ̂)` from the LM residual Jacobian. | T6 | ~1 sess | Closes the introspection-on-every-estimator promise. |
| F8 | **MIMO `iddata`** — multi-output / multi-input matrices through `arx` / `ssest` / `compare` / sim / predict.  Today's surface is SISO; the regressor / state-space cores generalise readily. | T1–T5 | ~3 sess | The single biggest scope gap — most industrial datasets are multi-output. |
| F9 | **Frequency-domain estimation on `idfrd`** — `arx` / `tfest` / `ssest` accepting an `idfrd` (not just an `iddata`); least-squares in the frequency domain. | T4 | ~1 sess | Lets the shipped `etfe` / `spa` feed the parametric estimators. |

### Numerical / algorithmic follow-ons

| Item | Tier | Notes |
|---|---|---|
| QR-conditioned LS for `arx` / `iv4` (vs the shipped Gram + normal-equations) | T1 | Better conditioning on near-singular regressors.  Needs a `matlab_mldivide` overdetermined path. |
| Projection-based N4SID with auto-order (SVD-gap) + CVA / SSARX / MOESP weights | T3 | Today's subspace uses Ho-Kalman/ERA on FIR-estimated Markov params — works but isn't the canonical N4SID. |
| PEM refinement of the subspace seed in `ssest` | T3 | A second pass over the n4sid output via `lsqnonlin` on the canonical state-space parameterisation. |
| `ssregest` — regularized high-order ARX → ss → balred | T3 | Improves accuracy on short / noisy data. |
| Standalone `era(H, nx)` — expose the Hankel-SVD realization as a public function on impulse-response data | T3 | Already implemented internally; just needs a public entry point. |
| `idtf` / `idproc` named classes + `procest` continuous process models (P1 / P2 / P1D / …) | T3 | Today `tfest` returns an `idpoly` (OE form).  `procest` needs bounded-`fmincon` over the parameterised structures. |
| `findstates(model, data)` initial-state estimation | T3 | Single `mldivide` once the model is fit; nice to have. |
| `cra` / regularized FIR variants of `impulseest` | T4 | The current `impulseest` is plain FIR LS. |
| `spafdr` + lag-window spectral variants of `spa` | T4 | Different smoothing kernels on the cross/auto spectra. |
| Continuous-time grey-box returning `[A, B, C, D, K, x0]` as separate outputs | T4 | Today's `structfn` returns the packed `[A B; C D]`; matches the function-handle ABI but limits expressiveness. |
| ARMAX `forecast` + confidence bands; `forecast` on `idss` | T4 | Today's `forecast` is the deterministic AR/ARX recursion. |
| Recursive PEM — `recursiveARMAX` / `recursiveOE` / `recursiveBJ` (approximate gradient ψ via filtered regressors) | T5 | Builds on the shipped RLS core. |
| `recursiveAR` + Kalman / normalized-gradient / unnormalized-gradient recursive variants | T5 | Sliding-window finite-history forms too. |
| Analytic Jacobians for EKF / UKF (vs the current FD) + multi-output measurement | T5 | FD works but is slower / less accurate; analytic needs a small user-handle convention extension. |
| `ode45` / `ode23s` rollout in `nlgreyest` (vs the current Euler) | T5 | Reuses the shipped ODE solvers; better accuracy on stiff models. |
| `showConfidence` uncertainty bands on `bode` / `step` | T6 | Needs the Fisher-info `getcov` from F7 first. |
| `merge` of models + multi-experiment `iddata` end-to-end | T6 | Independent — useful when several batches share the same plant. |
| ARIMA / seasonal `idpoly` forms (integrating-noise via a `D` polynomial) | T6 | A small extension to `compute_pe`. |
| `translatecov`, `sim(model, u, simOptions('AddNoise', 1))` | T6 | Cosmetic, last-mile polish. |

### Explicit carve-outs (deliberately out of scope — see §8)

These need external dependencies or live in adjacent toolboxes; pulling
them in would change the project's scope.

- System Identification App + Time Series Modeler app + UI Help (interactive GUIs; the command-line API is the whole surface).
- Simulink blocks (Chapter 22) — could become an `mflow` `RecursiveEstimator` / `IdModel` block through the Embedded Coder lanes (cf. MPC's `MpcMove`), but not in this roadmap.
- Neural State-Space (`idNeuralStateSpace`, Chapter 14) + LSTM / cascade-correlation / `narxnet` — Deep Learning Toolbox dependency.
- ML-NLARX mappings (Gaussian-process, regression-tree, `idGaussianProcess`) — Statistics & ML Toolbox dependency.
- Reduced Order Modeling chapter (Chapter 15) — depends on the carved neural-SS / LPV machinery.
- C-MEX grey-box (`idnlgrey` C-MEX form) — MEX FFI dependency; the MATLAB-file form ships.
- Diagnostics & Prognostics (Chapter 25) — Predictive Maintenance Toolbox overlap.
- App-only data import (drag-drop, session `.sid` files), App preferences (Chapter 20).

---

## 9. Dependency summary

```
Tier-1 (iddata + arx/ar + compare)      ── needs: mldivide LS, aryule/arburg, CST tf/ss, bode/step
   └─ Tier-2 (armax/oe/bj/iv4 + resid)  ── needs: lsqnonlin (Optim), xcov
        └─ Tier-3 (n4sid/ssest/tfest/procest)  ── needs: svd, lsqnonlin/fmincon  ◀── HEADLINE: ss(idsys)→mpc()
             └─ Tier-4 (spa/etfe/impulseest/greyest/forecast)  ── needs: fft, pwelch/cpsd/tfestimate, fn-handle ABI, ode45
                  └─ Tier-5 (nlarx/nlhw/nlgreyest + recursive* + EKF/UKF/PF)  ── needs: fmincon, ode23s, NEW dynamic-filter loop
                       └─ Tier-6 (Report/covariance/options/regularization polish)
```

**Critical new build (not reusable from elsewhere)**: (1) `iddata`
container + all `id*` model classes, (2) regressor-matrix + one-step
predictor machinery, (3) n4sid block-Hankel SVD subspace core, (4)
dynamic recursive + EKF/UKF/PF filtering loop (Tier-5). Everything else
is glue around already-shipped numerics.
