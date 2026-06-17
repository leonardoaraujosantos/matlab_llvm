# Curve Fitting Toolbox Spec

## Purpose
Documents the shipped subset of the Curve Fitting Toolbox in the matlab_llvm compiler: fitting of parametric polynomial models, nonlinear library models, custom equations, interpolants, smoothing splines, and surface fits, riding the shipped Optimization, polynomial, and interpolation infrastructure. Tiers 1-6 are marked shipped (2026-05-23). (doc: docs/curve_fitting_toolbox_roadmap.md) (src: runtime/toolbox/curvefit)

## Requirements

### Requirement: Polynomial fitting and the cfit object
The system SHALL provide a polynomial fit engine, fitted-curve objects, and goodness-of-fit. (src: runtime/toolbox/curvefit/runtime_curvefit.cpp) (src: runtime/toolbox/curvefit/curvefit_classdefs.m)

#### Scenario: Fit a polynomial and inspect the model
- **WHEN** a program calls `fit(x, y, 'polyN')` and inspects the `cfit` via `feval`/call syntax, `coeffvalues`, `formula`, `numcoeffs`, or requests `[f, gof]`
- **THEN** the system SHALL return a fitted `cfit` object with center-and-scale conditioning and a goodness-of-fit struct (sse, rsquare, dfe, adjrsquare, rmse)

### Requirement: Nonlinear library models and fit options
The system SHALL provide nonlinear library models and a fit-options carrier. (src: runtime/toolbox/curvefit/runtime_curvefit.cpp) (src: runtime/toolbox/curvefit/curvefit_classdefs.m)

#### Scenario: Fit an exponential or Gaussian with options
- **WHEN** a program calls `fit` with `exp1`/`exp2`, `power1`/`power2`, `gauss1`-`gauss8`, `fourier1`-`fourier8`, or `sin1`-`sin8`, optionally passing `fitoptions` (StartPoint, Lower, Upper, Weights, Robust, MaxIter)
- **THEN** the system SHALL return the fitted nonlinear model, honoring weighted and robust (Bisquare/LAR) least-squares when requested

### Requirement: Custom models and postprocessing
The system SHALL provide custom equation fitting and curve postprocessing. (src: runtime/toolbox/curvefit/runtime_curvefit.cpp)

#### Scenario: Fit a custom equation and postprocess
- **WHEN** a program builds a `fittype` from an equation string and calls `fit`, then `confint`, `differentiate`, `integrate`, or `coeffnames`
- **THEN** the system SHALL fit the custom model via Levenberg-Marquardt and return confidence intervals, derivatives, integrals, or parameter names

### Requirement: Interpolation and smoothing
The system SHALL provide interpolant fit types and data smoothing. (src: runtime/toolbox/curvefit/runtime_curvefit.cpp)

#### Scenario: Interpolate or smooth data
- **WHEN** a program calls `fit` with `linearinterp`/`nearestinterp`/`pchipinterp`/`cubicinterp`/`splineinterp`, `smooth` (moving/lowess/loess/rlowess/rloess/sgolay), or `csaps`
- **THEN** the system SHALL return the interpolant fit, smoothed series, or cubic smoothing spline respectively

### Requirement: Surface fitting
The system SHALL provide bivariate polynomial surface fitting via the sfit object. (src: runtime/toolbox/curvefit/runtime_curvefit.cpp) (src: runtime/toolbox/curvefit/curvefit_classdefs.m)

#### Scenario: Fit and evaluate a surface
- **WHEN** a program calls `fit([x y], z, 'polyNM')` and evaluates the `sfit` via `feval`/call syntax or `coeffvalues`, optionally requesting `[sf, gof]`
- **THEN** the system SHALL return a fitted bivariate polynomial surface (poly11-poly55) with goodness-of-fit

### Requirement: Spline fitting and ppform evaluators
The system SHALL provide piecewise-polynomial spline construction and evaluators. (src: runtime/toolbox/curvefit/runtime_curvefit.cpp) (src: runtime/toolbox/curvefit/curvefit_classdefs.m)

#### Scenario: Build and evaluate a spline
- **WHEN** a program calls `spline`, `pchip`, or `ppmak`, then `fnval`, `fnder`, `fnint`, or `fnbrk`
- **THEN** the system SHALL return a `ppform` object and the requested evaluation, derivative, integral, or extracted part
