# System Identification Toolbox Spec

## Purpose
Documents the shipped subset of the System Identification Toolbox in the matlab_llvm compiler: learning linear and nonlinear dynamic models from measured input-output data via least-squares and prediction-error methods, plus online and grey-box estimation and state filtering, with export to Control System Toolbox `ss`/`tf`. All six tiers are marked shipped (2026-05-20). (doc: docs/ident_toolbox_roadmap.md) (src: runtime/toolbox/ident)

## Requirements

### Requirement: Identification data and model objects
The system SHALL provide data containers and identified-model classes. (src: runtime/toolbox/ident/ident_classdefs.m)

#### Scenario: Construct data and models
- **WHEN** a program constructs `iddata`, or receives an `idpoly`/`idss`/`idfrd`/`idgrey`/`idnlgrey` model from an estimator
- **THEN** the system SHALL hold time-domain data and identified-model parameters (A/B/C/D/F polynomials, state-space matrices, noise variance, fit metrics) for downstream use

### Requirement: Linear least-squares and prediction-error estimation
The system SHALL provide linear estimators and validation utilities. (doc: docs/ident_toolbox_roadmap.md) (src: runtime/toolbox/ident/runtime_ident.cpp)

#### Scenario: Estimate and validate a model
- **WHEN** a program calls `arx`, `ar`, `armax`, `oe`, `bj`, or `iv4` on `iddata` and validates with `predict`, `compare`, `pe`, `resid`, `fpe`, `aic`, or `delayest`
- **THEN** the system SHALL return the identified model and the corresponding fit/diagnostic results

### Requirement: State-space and transfer-function estimation
The system SHALL provide subspace and transfer-function estimation with control-design export. (src: runtime/toolbox/ident/runtime_ident.cpp)

#### Scenario: Subspace identification to control model
- **WHEN** a program calls `n4sid`, `ssest`, or `tfest` and then `ss`/`tf` on the result
- **THEN** the system SHALL return an identified state-space/transfer-function model (via FIR-Markov + ERA Hankel-SVD for n4sid) convertible to a Control System Toolbox object

### Requirement: Frequency-domain and grey-box estimation
The system SHALL provide spectral, impulse-response, and grey-box estimation. (src: runtime/toolbox/ident/runtime_ident.cpp)

#### Scenario: Spectral and grey-box analysis
- **WHEN** a program calls `etfe`, `spa`, `impulseest`, `greyest`, `nlgreyest`, or `forecast`
- **THEN** the system SHALL return the empirical transfer-function/spectral estimate, Markov parameters, fitted (nonlinear) grey-box parameters, or K-step forecast respectively

### Requirement: Online estimation and state filtering
The system SHALL provide recursive estimators and nonlinear state filters. (src: runtime/toolbox/ident/ident_classdefs.m) (src: runtime/toolbox/ident/runtime_ident.cpp)

#### Scenario: Recursive update and filtering loop
- **WHEN** a program uses `recursiveLS`/`recursiveARX` `step` updates or `extendedKalmanFilter`/`unscentedKalmanFilter` `predict`/`correct` loops
- **THEN** the system SHALL update parameters or state estimates and covariance from each new measurement

### Requirement: Regularization and parameter introspection
The system SHALL provide regularized estimation and parameter accessors. (src: runtime/toolbox/ident/runtime_ident.cpp)

#### Scenario: Regularized fit with introspection
- **WHEN** a program calls `arx` with `arxOptions` (ridge regularization) and inspects with `getcov`, `getpvec`, or `setpvec`
- **THEN** the system SHALL apply ridge least-squares and expose the parameter covariance and packed parameter vector
