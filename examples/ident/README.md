# System Identification Toolbox Examples

Programs that exercise the System Identification Toolbox runtime in
matlab_llvm.  See
[`docs/ident_toolbox_roadmap.md`](../../docs/ident_toolbox_roadmap.md)
for the full tiered roadmap.

## Tier-1 (shipped)

The smallest end-to-end loop: the `iddata` time-domain container, linear
**ARX** (`arx`, QR/normal-equations least-squares) and **AR** (`ar`,
Yule-Walker) estimation, simulation / prediction / validation (`sim`,
`predict`, `compare` NRMSE fit %, `goodnessOfFit`), the `fpe` / `aic`
quality metrics, and the `idpoly` → `ss` / `tf` conversion that lets the
already-shipped Control System Toolbox analysis (`pole`, `step`, `bode`)
operate on identified models.

| Example | User's Guide | Notes |
|---|---|---|
| [`arx_lab_process.m`](arx_lab_process.m) | §4 *Estimating Simple Models from Real Laboratory Process Data* | **Tier-1 headline.**  Synthesises a 600-sample second-order lab record with a measured input + small disturbance, estimates an ARX `[2 2 1]` model, validates it (`compare` → 96.95 % fit), and converts it to a discrete `ss` to read its z-plane poles (`0.75 ± 0.37i`) via the shipped CST `pole`. |

### Surface covered

- **`iddata(y, u, Ts)`** — time-domain I/O container (`iddata(y, [], Ts)`
  for a time series).
- **`arx(data, [na nb nk])`** — ARX least-squares; returns an `idpoly`
  with `A` / `B` / `nk` / `Ts` / `NoiseVariance` / `Np` / `Ns`.
- **`ar(data, na)`** — AR time-series estimation (Yule-Walker).
- **`sim(model, u)`** / **`predict(model, data, K)`** — deterministic
  simulation and the K-step (Tier-1: 1-step) predictor.
- **`compare(data, model)`** — NRMSE fit percentage.
- **`goodnessOfFit(yhat, y)`** — NRMSE cost; **`fpe(model)`** /
  **`aic(model)`** — final-prediction-error / Akaike metrics.
- **`ss(model)`** / **`tf(model)`** — controllable-canonical (Ts-tagged)
  and coefficient-extraction conversions into Control System Toolbox
  model objects.

### Tier-1 limitations (carve-downs)

SISO only; direct `idpoly(A, B, …)` construction is deferred (arx / ar /
armax / oe / bj are the model factories); the `idss` / `idtf` / `idproc`
model classes are Tier-3.  See the roadmap for the full plan.

## Tier-2 (shipped)

The prediction-error-minimisation core: the iterative polynomial
estimators `armax` (A·y=B·u+C·e), `oe` (y=B/F·u+e) and `bj`
(y=B/F·u+C/D·e) — all minimising the general prediction error
`e=(D/C)·(A·y−B/F·u)` with the shipped Optimization Toolbox `lsqnonlin`
— plus instrumental-variables `iv4`, residual validation (`pe`, `resid`),
and `delayest`.

| Example | User's Guide | Notes |
|---|---|---|
| [`armax_refine.m`](armax_refine.m) | §6 *Estimate Models Using armax* | **Tier-2 headline.**  Fits a coloured-noise ARMAX record first with `arx` (residual autocorr 0.46 — coloured) then with `armax` (recovers `C2=0.71`, residual autocorr 0.10 — whitened), showing the noise-model refinement and `resid` whiteness drop. |

### Surface covered

- **`armax` / `oe` / `bj`** — PEM estimators (ARX-seeded LM), populate
  `idpoly` A/B/C/D/F.
- **`iv4`** — instrumental-variable ARX, robust to noise colour.
- **`pe(model, data)`** — prediction-error vector;
  **`resid(model, data)`** — `[maxAutoCorr; maxCrossCorr]` whiteness
  diagnostic.
- **`delayest(data)`** — input transport-delay estimate.
- `sim` / `predict` are now structure-general (`B/(A·F)` simulation;
  `ŷ = y − pe` one-step prediction).

### Tier-2 limitations (carve-downs)

`pem` generic refine; `struc` / `arxstruct` / `selstruc` order-grid
search; `*Options` objects; MIMO; OE/BJ stability enforcement during the
search.

## Tier-3 (shipped)

State-space and transfer-function estimation, and the cross-toolbox
payoff.  `n4sid` / `ssest` identify a state-space `idss` from data via
Ho-Kalman/ERA realization (FIR-estimated Markov parameters → block-Hankel
SVD); `tfest` estimates a transfer function (OE form).  The identified
model converts to a Control System Toolbox `ss` and feeds the shipped
Model Predictive Control designer.

| Example | User's Guide | Notes |
|---|---|---|
| [`data_driven_mpc.m`](data_driven_mpc.m) | Ch.21 *Using Identified Models for Control Design* | **Tier-3 headline.**  600 I/O samples → `ssest` order-2 (recovers poles `0.75±0.37i`) → `compare` 96.8 % → `ss(idsys)` → `mpc` → closed-loop step tracking `r=1.0` to `1.000`.  System ID → CST → MPC, no first-principles model. |

### Surface covered

- **`n4sid(data, nx)` / `ssest(data, nx)`** — subspace state-space
  estimation → `idss` (A/B/C/D/K/x0/Ts).
- **`tfest(data, np, nz)`** — transfer-function estimation (idpoly OE
  form: B = numerator, F = denominator).
- **`ss(idss)`** — convert the identified model into a CST `ss` (carries
  discrete Ts) for analysis and control design.
- `sim` / `compare` operate on `idss` via state-space simulation.

### Tier-3 limitations (carve-downs)

Projection-based N4SID + auto-order + CVA/MOESP weights; PEM refinement
of the subspace seed; innovations gain `K` (0 today); `ssregest`;
standalone `era`; `idtf`/`idproc` named classes + `procest`; `findstates`;
MIMO.

## Tier-4 (shipped)

Frequency-domain / spectral and impulse-response non-parametric
estimators, and user-structured linear grey-box.  `etfe`/`spa` produce
an `idfrd` frequency response; `impulseest` an FIR impulse response;
`greyest` fits the physical parameters of a structure function;
`forecast` extends a time series.

| Example | User's Guide | Notes |
|---|---|---|
| [`greybox_msd.m`](greybox_msd.m) | Ch.13 *ODE Parameter Estimation (Grey-Box Modeling)* | **Tier-4 headline.**  Recover the physical constants `[k/m, c/m]` of a mass-spring-damper from data: `structfn = @(p)[0 1 0; -p(1) -p(2) 1; 1 0 0]`, `greyest(z,[3;1],structfn,2)` → `k/m=4.0000`, `c/m=1.1960` (ω_n=2.0), 99.92 % fit. Uses the function-handle ABI + ZOH `c2d` + `lsqnonlin`. |

### Surface covered

- **`greyest(data, par0, @structfn, nx)`** — linear grey-box parameter
  estimation → `idgrey` (`.Parameters` + realized A/B/C/D/Ts).
- **`etfe(data)` / `spa(data)`** — non-parametric frequency response →
  `idfrd` (Frequency / ResponseMag / ResponsePhase).
- **`impulseest(data, N)`** — FIR impulse response → `idpoly`.
- **`forecast(model, data, K)`** — K-step time-series forecast.

### Tier-4 limitations (carve-downs)

`spafdr` + lag-window spectral; `cra`/regularized-FIR `impulseest`;
frequency-domain estimation on `idfrd`; ARMAX `forecast` + confidence
bands; continuous-time grey-box with `[A,B,C,D,K,x0]` separate outputs;
data preprocessing (`detrend`/`idfilt`/`resample`/`merge`/`misdata`);
MIMO.

## Tier-5 (shipped)

The heavy slice: nonlinear state estimation (the project's first dynamic
Kalman filtering loop), online recursive estimation, and nonlinear
grey-box.

| Example | User's Guide | Notes |
|---|---|---|
| [`ukf_state_estimation.m`](ukf_state_estimation.m) | Ch.18 *Nonlinear State Estimation Using UKF* | **Tier-5 headline.**  Reconstruct a pendulum's `[angle; rate]` from noisy angle-only measurements with `unscentedKalmanFilter`/`extendedKalmanFilter` — recovers the never-measured velocity (`−0.708` vs true `−0.705`). |
| [`recursive_arx_tracking.m`](recursive_arx_tracking.m) | *Online ARX Parameter Estimation for Tracking Time-Varying Dynamics* | **Tier-5 headline.**  `recursiveARX` with a forgetting factor tracks a plant pole that jumps `0.50 → 0.85` sample-by-sample. |

### Surface covered

- **`extendedKalmanFilter` / `unscentedKalmanFilter`** — `predict(obj,
  @StateFcn)` + `correct(obj, @MeasFcn, y)` dynamic nonlinear state
  estimation (FD Jacobian / sigma points; scalar measurement).
- **`recursiveARX([na nb nk])` / `recursiveLS(np)`** — forgetting-factor
  RLS online estimation; `step(obj, …)` updates in place.
- **`nlgreyest(data, par0, @StateFcn, nx)`** — nonlinear grey-box
  parameter estimation (ODE-rhs handle + Euler rollout + `lsqnonlin`).

### Tier-5 limitations (carve-downs)

Nonlinear black-box `nlarx`/`nlhw` (mapping objects); `particleFilter`;
`recursiveARMAX`/`recursiveOE`/`recursiveBJ`; `recursiveAR` + Kalman/
gradient recursive variants; analytic Jacobians / multi-output
measurement for EKF/UKF; `ode45`-based `nlgreyest` rollout; MIMO.

## Tier-6 (shipped)

The carve-down sweep — regularized ARX via `arxOptions`, and parameter
introspection (`getcov` / `getpvec` / `setpvec`).

| Example | User's Guide | Notes |
|---|---|---|
| [`arx_regularization.m`](arx_regularization.m) | §1 *Regularized Estimates of Model Parameters* | **Tier-6 headline.**  On a 60-sample noisy record, plain `arx` returns `(a=-0.516, b=1.053)`; ridged `arx(...,opt)` with `λ=1` returns `(a=-0.510, b=1.035)` — closer to truth `(-0.5, 1.0)`.  Also demos `getpvec` / `setpvec` / `getcov`. |

### Surface covered

- **`arxOptions()`** — option carrier; `.Regularization = λ` enables
  ridge LS `θ = (ΦᵀΦ + λI)⁻¹·Φᵀy`.
- **`arx(data, orders, opt)`** — 3-arg regularized form (returns
  `idpoly` with `Lambda` field carrying λ).
- **`getcov(model)`** — parameter covariance `V · (ΦᵀΦ + λI)⁻¹` from
  the regularized Gram arx caches on the idpoly.
- **`getpvec(model)` / `setpvec(model, θ)`** — packed parameter-vector
  introspection (preserves nk + monic A(1)=1).

### Tier-6 limitations (carve-downs)

Estimation `Report` struct, multi-return forms across estimators,
uncertainty bands on `bode` / `step`, other `*Options` carriers
(`armaxOptions` / `oeOptions` / etc.), Fisher-information covariance for
PEM-fit models, `merge` of models, ARIMA / seasonal forms,
`translatecov`.
