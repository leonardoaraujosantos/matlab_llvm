# Econometrics Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Econometrics Toolbox in `matlab_llvm` (all 6 tiers): data preprocessing and diagnostics, ARIMA conditional-mean models, GARCH-family conditional-variance models, VAR / cointegration, state-space models, and Bayesian regression / discrete-time Markov chains.

## Requirements

### Requirement: Data preprocessing and diagnostics
The system SHALL provide data-prep transforms and time-series diagnostic tests.

#### Scenario: Run a diagnostic test
- **WHEN** a program calls `price2ret`/`ret2price`, `hpfilter`, `autocorr`/`parcorr`/`crosscorr`, or unit-root/stationarity tests (`adftest`, `pptest`, `kpsstest`, `lbqtest`, `archtest`)
- **THEN** the system SHALL return the transformed series or test statistic/decision (matlab_econ_price2ret, matlab_econ_hpfilter, matlab_econ_autocorr, matlab_econ_adftest, matlab_econ_lbqtest) (doc: docs/econometrics_toolbox_roadmap.md) (src: runtime/toolbox/econ/runtime_econ.cpp, runtime/toolbox/econ/econ_classdefs.m)

### Requirement: ARIMA conditional-mean models
The system SHALL estimate, infer, forecast, and simulate `arima` models.

#### Scenario: Fit and forecast an ARIMA model
- **WHEN** a program builds an `arima` model and calls estimate / infer / forecast / simulate (Hannan-Rissanen)
- **THEN** the system SHALL return the estimated parameters, residuals, forecasts, or simulated paths (matlab_econ_arima_estimate, matlab_econ_arima_infer, matlab_econ_arima_forecast, matlab_econ_arima_simulate) (doc: docs/econometrics_toolbox_roadmap.md) (src: runtime/toolbox/econ/runtime_econ.cpp, runtime/toolbox/econ/econ_classdefs.m)

### Requirement: Conditional-variance (GARCH-family) models
The system SHALL estimate and forecast GARCH/EGARCH/GJR models.

#### Scenario: Fit and forecast a GARCH model
- **WHEN** a program builds a `garch`/`egarch`/`gjr` model and calls estimate / infer / forecast / simulate (Nelder-Mead MLE)
- **THEN** the system SHALL return the estimated parameters and conditional-variance forecasts (matlab_econ_garch_estimate, matlab_econ_garch_infer, matlab_econ_garch_forecast, matlab_econ_garch_simulate) (doc: docs/econometrics_toolbox_roadmap.md) (src: runtime/toolbox/econ/runtime_econ.cpp, runtime/toolbox/econ/econ_classdefs.m)

### Requirement: Multivariate models and cointegration
The system SHALL estimate VAR models and run cointegration tests.

#### Scenario: Fit a VAR and test cointegration
- **WHEN** a program builds a `varm` model, estimates/forecasts/simulates, computes impulse responses, or runs Engle-Granger / Johansen cointegration tests
- **THEN** the system SHALL return the estimated VAR, IRFs, or cointegration test results (matlab_econ_varm_estimate, matlab_econ_varm_forecast, matlab_econ_varm_irf, matlab_econ_egcitest, matlab_econ_jcitest) (doc: docs/econometrics_toolbox_roadmap.md) (src: runtime/toolbox/econ/runtime_econ.cpp, runtime/toolbox/econ/econ_classdefs.m)

### Requirement: State-space models
The system SHALL estimate, filter, smooth, and forecast `ssm`/`dssm` models.

#### Scenario: Filter a state-space model
- **WHEN** a program builds an `ssm`/`dssm` model and calls estimate / filter / smooth / forecast (Kalman)
- **THEN** the system SHALL return the filtered/smoothed states and forecasts (matlab_econ_ssm_estimate, matlab_econ_ssm_filter, matlab_econ_ssm_smooth, matlab_econ_ssm_forecast) (doc: docs/econometrics_toolbox_roadmap.md) (src: runtime/toolbox/econ/runtime_econ.cpp, runtime/toolbox/econ/econ_classdefs.m)

### Requirement: Bayesian regression and Markov chains
The system SHALL estimate Bayesian linear regression and simulate discrete-time Markov chains.

#### Scenario: Estimate Bayesian regression and simulate a DTMC
- **WHEN** a program builds a `bayeslm` and estimates/forecasts, or builds a `dtmc` and simulates / computes asymptotics
- **THEN** the system SHALL return the posterior estimates or the simulated chain / stationary distribution (matlab_econ_bayeslm_estimate, matlab_econ_bayeslm_forecast, matlab_econ_dtmc_simulate, matlab_econ_dtmc_asymptotics) (doc: docs/econometrics_toolbox_roadmap.md) (src: runtime/toolbox/econ/runtime_econ.cpp, runtime/toolbox/econ/econ_classdefs.m)
