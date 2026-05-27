# Econometrics Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Econometrics-Toolbox programs.

Source: *Econometrics Toolbox User's Guide* (R2026a, 12 chapters:
Getting Started · Data Preprocessing · Model Selection · Econometric
Modeler · Time Series Regression Models · Bayesian Linear Regression ·
Conditional Mean Models · Conditional Variance Models · Multivariate
Time Series Models · Structural Change Models · State-Space Models ·
Functions). **4,726 pages** — our deepest-paged plan yet, but the page
count is misleading: the toolbox is overwhelmingly *time-series model
objects + estimation + diagnostics*, and the numeric machinery for that
is **already shipped** in three sibling toolboxes.

This is the **highest-synergy next toolbox** — it's the direct
econometric sibling of the just-shipped Financial Toolbox, and it
reuses more shipped code than anything we've planned:

- **System Identification** already ships `armax` / `oe` / `bj` /
  `arx` — i.e. **ARMA estimation via prediction-error minimisation over
  `lsqnonlin`** — which is ~70 % of `arima` estimation; plus `n4sid` /
  `ssest` / `idss` (state-space) and the `extendedKalmanFilter` /
  `unscentedKalmanFilter` **Kalman filter/smoother loop**, which `ssm` /
  `dssm` reuse directly.
- **Statistics and Machine Learning** ships the hand-coded t / F / χ²
  CDFs (`sgammp` / `sbetai`), `regress` / `fitlm`, `mvnrnd`, RNG, the
  HMM `hmmviterbi` / `hmmtrain` (Baum-Welch) — which the **Markov-chain
  (`dtmc`)** and **Markov-switching (`msVAR`)** models reuse.
- **Financial** (just shipped) gives the **`timetable`** data container,
  the **ECM** missing-data estimator, the performance metrics, and the
  **alloc-then-populate classdef carrier + runtime discriminant**
  pattern (`RiskKind`) that every model object here mirrors.
- **Optimization** (`fminunc` / `fmincon` / `lsqnonlin`) fronts every
  maximum-likelihood estimation; **LAPACK** (`chol` / `svd` / `qr` /
  `eig` / `mldivide`) fronts the linear algebra.

**No external dependency** (no statsmodels / no R) — every estimator is
hand-coded over the shipped numeric base, exactly as Financial,
Stats, and Ident were.

The headline tracer-bullet (the gating example for the whole roadmap)
is [`examples/econ/arima_cpi_forecast.m`](../examples/econ/arima_cpi_forecast.m):
*the canonical Box-Jenkins demo — load a macro series (CPI), check
stationarity (`adftest`), difference it, inspect `autocorr`/`parcorr`,
fit `Mdl = arima(p,D,q)` with `estimate`, run residual diagnostics
(`lbqtest`, `archtest`), and `forecast` h steps ahead with confidence
bands*. This exercises the test → model-object → `estimate` → `infer`
→ `forecast` arc end-to-end; achieving it closes **Econ-Tier-2**. The
companion `examples/econ/garch_volatility.m` (GARCH on FX returns) is
the **Econ-Tier-3** tracer-bullet, and `examples/econ/var_macro.m`
(VAR of CPI + unemployment with impulse responses) is the
**Econ-Tier-4** tracer-bullet.

Companion docs:
[`system_identification_toolbox_roadmap.md`](system_identification_toolbox_roadmap.md)
(`arima`/`regARIMA` estimation reuses the shipped `armax`/`oe`/`bj`
PEM machinery; `ssm`/`dssm` reuse the `idss` + Kalman loop),
[`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md)
(hypothesis-test CDFs, `mvnrnd`, HMM Baum-Welch for the switching
models), [`financial_toolbox_roadmap.md`](financial_toolbox_roadmap.md)
(the `timetable` container, ECM, and the model-object classdef
pattern), [`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)
(every ML estimate rides `fminunc` / `lsqnonlin`),
[`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order.
  **Tier-1** is *data prep + the test surface* (Ch 2 + Ch 3):
  differencing / log / `price2ret`, the HP + seasonal filters,
  `autocorr` / `parcorr`, the diagnostic tests (`lbqtest`, `archtest`,
  `aicbic`, `lmtest`/`waldtest`/`lratiotest`), and the unit-root tests
  (`adftest`, `pptest`, `kpsstest`, `lmctest`, `vratiotest`). These are
  the gating dependencies for every model that follows. **Tier-2** is
  the **`arima` conditional-mean family** (Ch 7) — `arima`/AR/MA/ARMA/
  SARIMA/ARIMAX, `estimate`/`infer`/`forecast`/`simulate`/`filter` over
  the shipped Ident PEM machinery — **the headline**. **Tier-3** is the
  **`garch`/`egarch`/`gjr` conditional-variance family** (Ch 8) over a
  variance recursion + `fminunc` MLE. **Tier-4** is the **multivariate
  lane** (Ch 9): `varm` (VAR) estimate/forecast/`irf`, `vecm` (VEC), and
  the cointegration tests (`egcitest` Engle-Granger, `jcitest`/`jcontest`
  Johansen). **Tier-5** is **state-space + regression-with-ARIMA-errors**
  (Ch 11 + Ch 5): `ssm`/`dssm` Kalman filter/smooth/estimate/forecast
  (reusing the Ident Kalman loop) and the `regARIMA` object. **Tier-6**
  is **Bayesian + structural-change** (Ch 6 + Ch 10): `bayeslm`
  (conjugate + MCMC linear regression), `dtmc` (Markov chains),
  Markov-switching (`msVAR`) + threshold-switching over the shipped HMM
  Baum-Welch, plus the Time-Series-Regression I–X example series.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1
  ~2 wk · T2 ~2.5 wk · T3 ~2 wk · T4 ~2.5 wk · T5 ~2.5 wk · T6 ~3 wk
  (~14.5 wk full)** — but T2's headline rides so heavily on the shipped
  Ident `armax` that its *net-new* cost is closer to 1.5 wk. T1–T3 alone
  close the 80 % everyday econometrics workflow (test stationarity, fit
  ARIMA, model volatility); T4 closes the macro/VAR workflow.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Everything below is 🔵 not started** — there is no `arima` / `garch`
  / `varm` / `vecm` / `ssm` / `adftest` / `autocorr` / `egcitest` /
  `bayeslm` / `dtmc` in the runtime today. The deep shipped base
  (Ident `armax`/`ssest`/Kalman, Stats test-CDFs/`mvnrnd`/HMM,
  Financial `timetable`/ECM/classdef-carrier, Optim `fminunc`/
  `lsqnonlin`, LAPACK) makes this **mostly composition, not net-new
  numerics**.

---

## 1. Already-shipped pieces this roadmap leans on

Counter to the 4,726-page first impression, **<20 % of the roadmap is
net-new mathematics**. The dominant donors:

| Need                                       | Shipped in                              |
|--------------------------------------------|-----------------------------------------|
| ARMA estimation (PEM over `lsqnonlin`)     | **Ident** `armax`/`oe`/`bj`/`arx`       |
| State-space + Kalman filter/smoother       | **Ident** `idss`/`n4sid`/`ssest` + EKF/UKF |
| Recursive estimation (RLS)                 | **Ident** `recursiveARX`/`recursiveLS`  |
| t / F / χ² CDFs (test p-values)            | **Stats** `sgammp` / `sbetai`           |
| Linear regression / OLS / CIs              | **Stats** `regress` / `fitlm`           |
| Multivariate normal sampling               | **Stats** `mvnrnd`                      |
| HMM Viterbi + Baum-Welch                   | **Stats** `hmmviterbi` / `hmmtrain`     |
| `timetable` time-series container          | **Financial** (Phase 5.4)               |
| ECM missing-data estimator                 | **Financial** `ecmnmle` / `ecmncov`     |
| Model-object classdef carrier + discriminant | **Financial** `Portfolio`/`RiskKind`  |
| Nonlinear MLE / constrained optimisation   | **Optim** `fminunc` / `fmincon`         |
| Cholesky / SVD / QR / `eig` / `mldivide`   | LAPACK lane (Phases 1–3)                |
| RNG with seed (`rng` / `randn`)            | **Stats** T1                            |
| `fft` / `filter`                           | **Signal** / **DSP**                    |
| Plot (line / fill / area)                  | Cairo plot backend                      |

The genuinely **new** numerics are confined to:

- **§T1** *Unit-root + stationarity tests* (`adftest`/`pptest`/`kpsstest`
  /`lmctest`/`vratiotest`) — regression + a small baked-in table of
  Dickey-Fuller / KPSS critical values. ~250 LOC.
- **§T1** *HP filter* (`hpfilter`) — a sparse penalised-least-squares
  smoother (banded system). ~60 LOC.
- **§T3** *GARCH/EGARCH/GJR likelihood* — the conditional-variance
  recursion + Gaussian/t log-likelihood, maximised by `fminunc`.
  ~300 LOC across the three variants.
- **§T4** *Cointegration tests* — Engle-Granger (`egcitest`: residual
  ADF) + Johansen (`jcitest`/`jcontest`: a reduced-rank eigenvalue
  problem on `eig`). ~250 LOC.
- **§T4** *VAR/VEC estimation* — multivariate LS / reduced-rank
  regression. ~200 LOC.
- **§T6** *Bayesian linear regression* (`bayeslm`) — conjugate
  Normal-Inverse-Gamma posterior + a Gibbs/MCMC sampler for the
  non-conjugate priors. ~300 LOC.

Everything else (ARIMA estimation, state-space, Markov chains,
Markov-switching, the diagnostic tests, ACF/PACF) composes from the
shipped donors above.

---

## 2. Tier-1 — Data Preprocessing + Model-Selection Tests (~2 wk)

**Goal.** Close Ch 2 + Ch 3. After Tier-1, a script that loads a macro
series, transforms it, tests it for stationarity / autocorrelation /
ARCH effects, and selects a lag order is end-to-end compilable — the
prerequisite for every model that follows.

### 2.1 Data transformations (Ch 2)
- Differencing: `diff` (shipped) + the lag-operator forms; seasonal +
  nonseasonal differencing.
- `price2ret` / `ret2price` (log + simple returns — mirrors the shipped
  Financial `tick2ret`/`ret2tick`).
- Detrending + deseasonalisation: moving-average trend, stable seasonal
  filter, `S(n,m)` seasonal filter.
- **`hpfilter`** (Hodrick-Prescott, one- and two-sided) — banded
  penalised LS. *New numerical contribution.*

### 2.2 ACF / PACF (Ch 3)
- `autocorr` (sample ACF + bounds), `parcorr` (PACF via Durbin-Levinson
  — `levinson` is shipped in Signal), `crosscorr`.

### 2.3 Diagnostic + comparison tests (Ch 3)
- `lbqtest` (Ljung-Box Q on χ² CDF), `archtest` (Engle ARCH LM test),
  `aicbic` (information criteria), `lmtest` / `waldtest` / `lratiotest`
  (the LR / LM / Wald model-comparison trio — closed-form on the shipped
  χ² CDF).
- `hac` (Newey-West HAC covariance), `fgls` (feasible GLS).

### 2.4 Unit-root + stationarity tests (Ch 3)
- `adftest` (Augmented Dickey-Fuller), `pptest` (Phillips-Perron),
  `kpsstest` (KPSS), `lmctest` (Leybourne-McCabe), `vratiotest`
  (variance-ratio). Each is a regression + a baked-in critical-value
  table. *New numerical contribution* (§1).

### 2.5 Headline demo
`examples/econ/stationarity_workflow.m` — load a trending series,
`adftest` (fail to reject unit root) → difference → `adftest` (reject)
→ `autocorr`/`parcorr` → `lbqtest`. Closes Tier-1.

---

## 3. Tier-2 — Conditional Mean Models: the `arima` family (~2.5 wk) — **HEADLINE**

Closes Ch 7. The most-used object in the toolbox.

### 3.1 `arima` classdef
Carries `AR`/`MA`/`SAR`/`SMA` polynomials, `D`/`Seasonality`, `Constant`,
`Variance`, `Distribution` (Gaussian / t), `Beta` (ARIMAX regressors).
Shorthand `arima(p,D,q)` + longhand name-value + dot-notation
modification. Mirror the **alloc-then-populate + class-pinned dispatch**
carrier proven by Financial `Portfolio` and Ident `idpoly`.

### 3.2 Estimation + inference
- `estimate(Mdl, y)` — ML estimation. **Rides the shipped Ident
  `armax`/`oe`/`bj` PEM machinery**: ARIMA(p,D,q) = difference `y` `D`
  times, then ARMA(p,q) estimation by prediction-error minimisation over
  `lsqnonlin` with the shipped `compute_pe` residual. Seasonal lags +
  ARIMAX regressors extend the regressor block.
- `infer(Mdl, y)` — residuals + loglikelihood (the Ident `pe`/`resid`
  path).
- `summarize(Mdl)` — coefficient table + standard errors + fit stats
  (`aicbic`).

### 3.3 Forecast / simulate / filter
- `forecast(Mdl, h, y)` — MMSE multi-step forecast + variance bands.
- `simulate(Mdl, n)` — Monte Carlo sample paths (reuses shipped RNG).
- `filter(Mdl, e)` — filter innovations through the model.

### 3.4 Headline demo
`examples/econ/arima_cpi_forecast.m` — Box-Jenkins on CPI:
test → difference → identify → `estimate` → diagnostics → `forecast`.
**Gating example for the whole Econometrics roadmap.**

---

## 4. Tier-3 — Conditional Variance Models: GARCH family (~2 wk)

Closes Ch 8.

### 4.1 `garch` / `egarch` / `gjr` classdefs
Carry the constant + ARCH + GARCH (+ leverage for E/GJR) coefficient
cells, `Offset`, `Distribution`. Shorthand `garch(p,q)` + name-value.

### 4.2 Estimation + inference
- `estimate(Mdl, y)` — Gaussian/t MLE of the variance recursion via
  `fminunc` (constrained to stationarity + positivity). *New numerical
  contribution* (§1). EGARCH uses the log-variance recursion; GJR adds
  the asymmetric-shock indicator.
- `infer(Mdl, y)` — conditional variances + loglikelihood.

### 4.3 Forecast / simulate
- `forecast(Mdl, h, y)` — multi-step conditional-variance forecast.
- `simulate(Mdl, n)` — paths with conditional heteroscedasticity.

### 4.4 Headline demo
`examples/econ/garch_volatility.m` — fit `garch(1,1)` to FX returns,
infer the conditional-variance series, forecast volatility. Closes
Tier-3.

---

## 5. Tier-4 — Multivariate Time Series + Cointegration (~2.5 wk)

Closes Ch 9.

### 5.1 `varm` (Vector Autoregression)
Classdef carrying `AR` cell (k×k per lag), `Constant`, `Trend`, `Beta`
(VARX), `Covariance`. `estimate(Mdl, Y)` via multivariate LS;
`forecast`, `simulate`, **`irf`** (orthogonalised + generalised impulse
responses via `chol` of the residual covariance), `fevd` (variance
decomposition).

### 5.2 `vecm` (Vector Error-Correction) + cointegration
- `vecm` classdef (cointegration rank `r`, adjustment `α`, cointegrating
  `β`).
- **`egcitest`** (Engle-Granger: OLS residual ADF), **`jcitest`** /
  **`jcontest`** (Johansen trace + max-eigenvalue, a reduced-rank
  eigenvalue problem on the shipped `eig`). *New numerical contribution*
  (§1). `jcitest` → `vecm` parameter estimation.

### 5.3 Headline demo
`examples/econ/var_macro.m` — VAR of CPI + unemployment, lag selection
via `aicbic`, `estimate`, `forecast`, impulse responses. Closes Tier-4.

---

## 6. Tier-5 — State-Space + Regression with ARIMA Errors (~2.5 wk)

Closes Ch 11 + Ch 5.

### 6.1 `ssm` / `dssm` (state-space)
Classdef carrying the `A`/`B`/`C`/`D` system matrices (time-invariant +
the implicit/parameter-mapping forms). **Reuses the shipped Ident Kalman
loop** for: `filter` (Kalman filter), `smooth` (RTS smoother),
`estimate` (ML over `fminunc` with the Kalman-filter loglikelihood),
`forecast`, `simulate`. `dssm` is the diffuse-prior variant.

### 6.2 `regARIMA` (regression with ARIMA errors)
Classdef carrying a regression `Beta` + an ARIMA error model.
`estimate`/`infer`/`forecast`/`simulate`. Reuses the Tier-2 ARIMA
machinery with a regression mean.

### 6.3 Headline demo
`examples/econ/ssm_kalman.m` — a local-level / local-linear-trend
state-space model, `estimate` via Kalman MLE, `smooth` the latent state.
Closes Tier-5.

---

## 7. Tier-6 — Bayesian + Structural Change (~3 wk)

Closes Ch 6 + Ch 10.

### 7.1 `bayeslm` (Bayesian linear regression)
Conjugate (`diffuseblm` / `conjugateblm`) closed-form Normal-Inverse-
Gamma posterior + a Gibbs sampler for the semiconjugate / lasso / SSVS
priors. `estimate`/`forecast`/`simulate`. *New numerical contribution*
(§1) — reuses `mvnrnd` + the shipped RNG.

### 7.2 `dtmc` (discrete-time Markov chains)
Classdef over a transition matrix (reuses the shipped Financial
`transprob`). `asymptotics` (stationary distribution via `eig`),
`redistribute` (state evolution), `simulate`, `lazy`/`classify`.

### 7.3 Markov-switching + threshold-switching
- `msVAR` (Markov-switching dynamic regression) — reuses the shipped
  Stats **HMM Baum-Welch** (`hmmtrain`) for the regime-inference EM, with
  per-regime VAR/regression estimation in the M-step.
- Threshold-switching dynamic regression (`tsVAR`-style).

### 7.4 Time Series Regression I–X (Ch 5)
The canonical example series (collinearity, influential observations,
spurious regression, predictor selection, HAC/GLS) — built on the
shipped `regress`/`fitlm` + the Tier-1 `hac`/`fgls`.

---

## 8. Carve-outs (deliberately out of scope)

| Area | Reason |
|------|--------|
| **Econometric Modeler app** (Ch 4) | App-Designer GUI; matches the project-wide GUI carve-out. The programmatic API (Tiers 1–6) covers the same modelling surface. |
| **DSGE models** (Ch 11 §11.203) | linearised dynamic-stochastic-general-equilibrium solving — a research sub-toolbox. |
| **Bayesian / non-Gaussian state-space** (Ch 11 Bayesian SSM, particle SSM) | needs a particle filter + non-Gaussian SSM engine beyond the linear-Gaussian Kalman core. |
| **HMC / NUTS samplers** (Ch 6 §6.18) | the gradient-based MCMC samplers; `bayeslm` ships the conjugate + Gibbs path. |
| **Diebold-Li / yield-curve + carbon-emission worked examples** | depend on Financial + external data; trace as follow-on demos. |
| **`varm`/`vecm` with full exogenous + trend regressor matrices, structural VAR (SVAR) identification** | ship the reduced-form VAR/VEC first; SVAR identification is a follow-on. |

---

## 9. Execution order

1. **T1 §2.1–2.2 data transforms + ACF/PACF** (~3 sessions)
2. **T1 §2.3–2.4 diagnostic + unit-root tests** (~5 sessions) — the
   new critical-value tables
3. **T1 §2.5 headline `stationarity_workflow.m`** — *first green*
4. **T2 §3.1–3.2 `arima` classdef + `estimate` over Ident PEM**
   (~6 sessions) — the headline tier
5. **T2 §3.3–3.4 forecast/simulate + `arima_cpi_forecast.m` →
   PR / merge → README badge bump**
6. **T3 GARCH family** (~8 sessions)
7. **T4 `varm` + cointegration + `vecm`** (~10 sessions)
8. **T5 `ssm`/`dssm` (reuse Ident Kalman) + `regARIMA`** (~10 sessions)
9. **T6 `bayeslm` + `dtmc` + `msVAR` + TSReg examples** (~12 sessions)

Total: ~14.5 wk; **T1–T2 (the 60 % everyday workflow) is ~5 wk** and
unlocks the iconic Box-Jenkins demo.

---

## 10. Layout in the repo

```
runtime/toolbox/econ/
├── runtime_econ_tests.cpp        (T1 — unit-root + diagnostic tests, ACF/PACF, HP filter)
├── runtime_econ_arima.cpp        (T2 — conditional-mean estimation/forecast)
├── runtime_econ_garch.cpp        (T3 — conditional-variance estimation/forecast)
├── runtime_econ_var.cpp          (T4 — VAR/VEC + cointegration)
├── runtime_econ_ssm.cpp          (T5 — state-space Kalman + regARIMA)
├── runtime_econ_bayes.cpp        (T6 — bayeslm)
├── runtime_econ_switch.cpp       (T6 — dtmc + msVAR + threshold)
└── econ_classdefs.m              (arima/garch/egarch/gjr/varm/vecm/
                                    ssm/dssm/regARIMA/bayeslm/dtmc/msVAR)

examples/econ/
├── stationarity_workflow.m       (T1 headline)
├── arima_cpi_forecast.m          (T2 headline — overall)
├── garch_volatility.m            (T3 headline)
├── var_macro.m                   (T4 headline)
├── ssm_kalman.m                  (T5 headline)
└── bayeslm_regression.m          (T6 headline)
```

Mirrors the `runtime/toolbox/{finance,ident,stats,…}/` layout.

---

## 11. Known gaps and risks

- **ARIMA estimation reuses Ident `armax`, but the ABIs differ.** Ident
  returns an `idpoly`; `arima` is its own classdef. The plan reuses the
  *numeric core* (`compute_pe` + `lsqnonlin`), not the object — wire a
  thin adapter that packs the estimated polynomials into an `arima`
  carrier. Verify the differencing pre-step composes cleanly.
- **`timetable` input.** `arima`/`varm`/`estimate` accept `timetable`
  data in MATLAB; the shipped Financial timetable lane covers the
  container, but the estimators must accept both a raw matrix and a
  timetable (extract the variable columns). Reuse the
  `matlab_timetable_get_column` path.
- **Critical-value tables.** `adftest`/`kpsstest`/`jcitest` interpolate
  p-values from response-surface tables (MacKinnon / Osterwald-Lenum).
  Bake the standard tables in; document the interpolation as an
  approximation versus the MathWorks simulated surfaces.
- **GARCH MLE conditioning.** The variance recursion's likelihood is
  flat near the stationarity boundary; `fminunc` needs a sensible
  parameter transform (log/logit) + presample-variance back-cast — the
  same care the Ident PEM seeds needed.
- **Multi-return shapes.** `[h,p,stat,cvalue] = adftest(...)` and
  `[Mdl,EstParamCov,logL,info] = estimate(...)` are multi-output. Reuse
  the shipped multi-return splitter machinery (Stats `anova1`,
  Ident `compare`).

---

## 12. Why this toolbox now

The user shipped the **Financial Toolbox** (Tiers 1–7) immediately
before this. Econometrics is its direct sibling: the same quant /
economist / risk-analyst persona (the User's Guide §1.1 "Expected
Users" overlaps Financial's verbatim), the same `timetable` data
container, and — critically — the **System Identification Toolbox
already shipped the hard part** (ARMA estimation via PEM, state-space
realisation, the Kalman loop). This is the cheapest large toolbox left
to ship per net-new line of code: a "repackage finished estimators
behind the econometric model-object API + add the test surface"
roadmap, not an "invent the numerics" one. The genuinely new work
(unit-root tests, GARCH likelihood, cointegration, Bayesian regression)
is ~1.3 kLOC across a 14.5-wk plan.
