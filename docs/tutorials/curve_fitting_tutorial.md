# Curve Fitting Toolbox — Tutorial

A hand-coded Curve-Fitting-Toolbox subset over the project's `polyfit`/`polyval` and a hand-coded Levenberg-Marquardt solver — no external dependency. It covers the everyday `fit` → `cfit`/`gof` → `feval`/`f(x)` arc for polynomials, library models (exponential/Gaussian/sine/Fourier), custom equations, smoothing, surface fits (`sfit`), and the ppform spline (`fn*`) layer.

## Supported features

- **Polynomial & goodness-of-fit:** `fit(...,'poly2')`, `cfit` objects with call syntax `f(x)`, `feval`, `[f,gof]` with `gof.rsquare` / `gof.adjrsquare` / `gof.rmse`, `disp(f)`.
- **Library nonlinear models:** `'exp2'` (two-term exponential), `'gauss2'` (two-peak Gaussian), `sinN` / `fourierN`, fitted by self-seeding Levenberg-Marquardt; `coeffvalues`, `differentiate`.
- **Custom equations:** `fittype('a + b*sin(2*pi*x/12) + ...')` parsed and fitted with multistart finite-difference LM; `confint`, `differentiate`, `integrate`.
- **Smoothing & interpolation:** `smooth` (moving average + `'rloess'` robust local regression), interpolant `cfit` (`'splineinterp'`), `csaps` / smoothing spline.
- **Surface fitting:** `fit([x y], z, 'poly55')` → `sfit`, evaluated on a `meshgrid` via `feval`.
- **Spline (ppform) layer:** `spline` (not-a-knot cubic), `pchip`, `fnval`, `fnder`, `fnint`, `fnbrk`, `ppmak`.

## Build & run

```bash
build/matlabc -emit-llvm examples/curvefit/census_fit.m > /tmp/census_fit.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/census_fit.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/census_fit
/tmp/census_fit
```

Swap `census_fit` for any other file under `examples/curvefit/`.

## Worked examples

### US-census quadratic fit + forecast — HEADLINE  (`examples/curvefit/census_fit.m`)

The canonical census demo: fit a quadratic, read goodness-of-fit, then forecast with call syntax / `feval`.

```matlab
cdate = (1790:10:1990)';
pop   = [3.929; 5.308; ... ; 248.710];

% ----- fit a quadratic + read goodness-of-fit -------------------------
[f, gof] = fit(cdate, pop, 'poly2');
disp(f);
fprintf('R-squared = %.4f\n', gof.rsquare);
fprintf('RMSE      = %.4f\n', gof.rmse);

% ----- forecast future censuses ---------------------------------------
yrs = (2000:10:2030)';
pf  = feval(f, yrs);
fprintf('forecast 2030 = %.1f million\n', pf(4));
```

`fit` returns a `cfit` plus a `gof` struct. It centers-and-scales the predictor internally for conditioning (a raw Vandermonde in calendar years is hopeless at degree 2), so `disp(f)` reports the normalization. The quadratic least-squares rides the shipped `polyfit`/`polyval`.

### Two-term exponential decay  (`examples/curvefit/exp_decay_fit.m`)

The UG "Fit Exponential Models" workflow: recover both amplitudes and rate constants from a sum of two decays with the `'exp2'` library model.

```matlab
y = 5.0 * exp(-1.5 * t) + 2.0 * exp(-0.3 * t);
[f, gof] = fit(t, y, 'exp2');
c = coeffvalues(f);
fprintf('term 1: amp=%.3f rate=%.3f\n', c(1), c(2));
fprintf('term 2: amp=%.3f rate=%.3f\n', c(3), c(4));
```

No `StartPoint` is supplied — the fit seeds itself from a log-linear regression, then refines with the hand-coded LM (analytic Jacobian).

### Custom-equation seasonal fit  (`examples/curvefit/enso_fourier.m`)

A 12-month sinusoid described directly as a `fittype` string, fitted then post-processed with `differentiate`.

```matlab
ft = fittype('a + b*sin(2*pi*x/12) + c*cos(2*pi*x/12)');
[f, gof] = fit(month, sst, ft);
cc = coeffvalues(f);

d = differentiate(f, xe);            % rate of change of the fitted signal
fprintf('rate at month 3 = %.3f\n', d(3));
```

The custom equation is parsed and fitted with multistart finite-difference Levenberg-Marquardt.

### Cubic spline interpolation  (`examples/curvefit/spline_interp.m`)

The ppform spline layer: build a not-a-knot cubic, evaluate it and its derivative, and integrate it.

```matlab
pp = spline(x, y);                            % not-a-knot cubic ppform
fprintf('ppform: order=%.0f pieces=%.0f\n', fnbrk(pp,'order'), fnbrk(pp,'pieces'));
yf  = fnval(pp, xf);
dpp = fnder(pp);                              % derivative spline
ipp = fnint(pp);                              % antiderivative
fprintf('integral over [0,10] = %.4f\n', fnval(ipp, 10));
```

`fnbrk` unpacks ppform fields; `fnder`/`fnint` produce new ppform splines evaluated again with `fnval`.

### Other examples (briefly)

- `peaks_gauss.m` — `'gauss2'` two-peak deconvolution, recovering amplitude/centre/width per peak via `coeffvalues`.
- `robust_smooth.m` — `smooth(y,7)` moving average vs `smooth(y,7,'rloess')` robust local regression on data with gross outliers, then a `'splineinterp'` interpolant `cfit`.
- `franke_surface.m` — `fit([xs ys], zs, 'poly55')` surface fit → `sfit`, evaluated on a `meshgrid` with `feval`.

## Limitations & carve-outs

- **Curve Fitter app** (`curveFitter`) and **Spline Tool** (`splinetool`) — the programmatic `fit`/`fittype` API is the whole target; the project is headless.
- **Generate-code-from-app** and **Live Editor tasks** — N/A.
- **Session save/reopen** (`.sfit`).
- **MATLAB-Coder codegen of fits** beyond the existing `-emit-*` lanes.
- **NURBS authoring depth** beyond library shapes, and `predint` / `excludedata` / interpolant surfaces / `tpaps` / B-form / Chebyshev splines — documented Tier-5/6 carve-downs.

## See also

- Roadmap: [`curve_fitting_toolbox_roadmap.md`](../curve_fitting_toolbox_roadmap.md)
- Examples: `examples/curvefit/`
