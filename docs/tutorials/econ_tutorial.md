# Econometrics Toolbox — Tutorial

The Econometrics Toolbox runtime brings the classic time-series modelling surface to the matlab_llvm compiler: stationarity tests, the Box-Jenkins `arima` family, GARCH volatility models, vector autoregressions, state-space/Kalman models, and Bayesian regression. Workflows compile to native code with no external LAPACK or stats dependency. Each example below is also a per-tier gating test.

## Supported features

- **Tier-1 — preprocessing + tests**: `diff`, `autocorr`, `parcorr`, `adftest`, `kpsstest`, `lbqtest`, `archtest`.
- **Tier-2 — conditional mean (`arima`)**: `arima(p,d,q)`, `estimate`, `infer`, `forecast`; estimated-model fields `Est.AR`, `Est.MA`, `Est.Variance`.
- **Tier-3 — conditional variance (GARCH)**: `garch(p,q)` (plus EGARCH/GJR per roadmap), `estimate`, `infer`, `forecast`; fields `Est.GARCH`, `Est.ARCH`.
- **Tier-4 — multivariate / cointegration**: `varm(numseries, lags)`, `estimate`, `forecast`, `irf` (impulse responses); fields `Est.NumSeries`, `Est.P`, `Est.AR` (cell/matrix coefficients) — plus the Johansen cointegration tests.
- **Tier-5 — state-space**: `ssm(A,B,C,D)`, `estimate`, `smooth` (RTS smoother), `forecast`; fields `Est.B`, `Est.D`.
- **Tier-6 — Bayesian + Markov chains**: `bayeslm(numpredictors)`, `estimate`, `forecast`; posterior fields `Post.Beta`, `Post.Sigma2`; `dtmc(P)` discrete-time Markov chain with `asymptotics`.

## Build & run

```bash
build/matlabc -emit-llvm examples/econ/arima_cpi_forecast.m > /tmp/arima_cpi_forecast.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/arima_cpi_forecast.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/arima_cpi_forecast
/tmp/arima_cpi_forecast
```

## Worked examples

### Box-Jenkins ARIMA forecast  (`examples/econ/arima_cpi_forecast.m`)

The headline tracer-bullet: the full Box-Jenkins workflow on a synthetic CPI-like price index — test for a unit root, difference, identify orders from ACF/PACF, estimate, run residual diagnostics, then forecast 12 months ahead.

```matlab
fprintf('Level ADF reject-unit-root: %.0f\n', adftest(cpi));   % 0 (nonstationary)
dcpi = diff(cpi);
fprintf('Diff  ADF reject-unit-root: %.0f\n', adftest(dcpi));  % 1 (stationary)

acf  = autocorr(dcpi, 6);
pacf = parcorr(dcpi, 6);

Mdl = arima(1, 1, 0);
Est = estimate(Mdl, cpi);
fprintf('Estimated AR(1) = %.3f\n', Est.AR(1));
fprintf('Innovation var  = %.4f\n', Est.Variance);

res = infer(Est, cpi);                              % innovations
fprintf('Ljung-Box reject-white: %.0f\n', lbqtest(res, 12));

yF = forecast(Est, 12, cpi);                        % 12-step forecast
```

`adftest` returns 0 on the level (fails to reject the unit root) and 1 on the differenced series. `infer` recovers the innovations for the Ljung-Box check, and the 12-month forecast trends above the last observed value.

### Stationarity workflow  (`examples/econ/stationarity_workflow.m`)

The Tier-1 gate: a random walk with drift, confirmed nonstationary by ADF *and* KPSS (which disagree by construction — that's the point), then differenced to stationarity.

```matlab
h0  = adftest(y);     % 0  — fails to reject unit root
hk0 = kpsstest(y);    % 1  — rejects stationarity
dy  = diff(y);
h1  = adftest(dy);    % 1  — differenced series is stationary
acf  = autocorr(dy, 5);
pacf = parcorr(dy, 5);
hlb  = lbqtest(dy, 10);   % 0  — increments are white-ish
```

ADF and KPSS have opposite null hypotheses, so a difference-stationary series gives `adftest = 0` and `kpsstest = 1` on the level, flipping after one difference.

### GARCH volatility modelling  (`examples/econ/garch_volatility.m`)

Model volatility clustering in an FX-return-like series with a GARCH(1,1).

```matlab
fprintf('ARCH effects present: %.0f\n', archtest(ret, 4));   % 1

Mdl = garch(1, 1);
Est = estimate(Mdl, ret);
fprintf('GARCH coeff (beta):  %.3f\n', Est.GARCH(1));
fprintf('ARCH  coeff (alpha): %.3f\n', Est.ARCH(1));
fprintf('Persistence:         %.3f\n', Est.GARCH(1) + Est.ARCH(1));

hv = infer(Est, ret);                 % conditional variance series
vF = forecast(Est, 20, ret);          % 20-period volatility forecast
```

`archtest` confirms conditional heteroscedasticity before fitting. The β+α persistence is near 1 for a clustering series; `infer` gives the fitted conditional variance and `forecast` projects it forward.

### Macro VAR with impulse responses  (`examples/econ/var_macro.m`)

A bivariate VAR(2) jointly modelling inflation and unemployment, then forecasting and tracing impulse responses.

```matlab
Y   = [infl unemp];
Mdl = varm(2, 2);                     % 2 series, 2 lags
Est = estimate(Mdl, Y);
fprintf('AR1 infl<-infl:  %.3f\n', Est.AR(1,1));

yF = forecast(Est, 8, Y);             % 8-step joint forecast
ir = irf(Est, 12);                    % impulse responses, 12 horizons
fprintf('Impact on infl:  %.3f\n', ir(1,1));
```

`Est.AR(i,j)` indexes the lag-1 coefficient matrix; `forecast` returns an h × numseries matrix; `irf` returns the impulse-response paths.

### State-space Kalman smoothing  (`examples/econ/ssm_kalman.m`)

A local-level model (random-walk state + noisy observation) recovers the latent signal with the Kalman filter and RTS smoother.

```matlab
A = ones(1,1); B = ones(1,1); C = ones(1,1); D = ones(1,1);
Mdl = ssm(A, B, C, D);
Est = estimate(Mdl, y);
fprintf('Estimated process std (B): %.3f\n', Est.B(1));
xs = smooth(Est, y);                  % RTS-smoothed latent level
yF = forecast(Est, 10, y);
```

The script compares the SSE of the raw observations vs. the smoothed estimate against the true level — the smoother's error is markedly lower.

### Bayesian regression + regime chain  (`examples/econ/bayeslm_regression.m`)

Bayesian linear regression with a diffuse prior (posterior mean recovers the OLS estimate), plus a two-regime Markov chain.

```matlab
Mdl  = bayeslm(3);
Post = estimate(Mdl, X, y);
fprintf('Posterior beta1: %.3f\n', Post.Beta(2));   % ~2.5 (true coeff)
yf = forecast(Post, [1 0.8 0.3]);

P  = [0.90 0.10; 0.25 0.75];
mc = dtmc(P);
pis = asymptotics(mc);                % long-run stationary distribution
```

With a diffuse prior the posterior `Post.Beta` recovers the true `1.0 + 2.5·x1 − 0.8·x2` coefficients; `dtmc` + `asymptotics` gives the stationary regime probabilities.

## Limitations & carve-outs

From [`docs/econometrics_toolbox_roadmap.md`](../econometrics_toolbox_roadmap.md):

- All six tier cores are shipped. Tier-4 ships the cointegration *tests* (Johansen) but the `vecm` model object is deferred. EGARCH/GJR are in the GARCH family.
- **Deferred full features**: `regARIMA` (Tier-5 — its `estimate` path) and `msVAR` / threshold-switching models (Tier-6).
- **Out of scope**: the **Econometric Modeler app** (App-Designer GUI), matching the project-wide GUI carve-out. The programmatic API (Tiers 1–6) covers the same modelling surface.
- Examples synthesize their own data with a deterministic LCG/Box-Muller loop because there is no MAT-file fixture decoder on this lane.

## See also

- Roadmap: [`docs/econometrics_toolbox_roadmap.md`](../econometrics_toolbox_roadmap.md)
- Examples: `examples/econ/`
