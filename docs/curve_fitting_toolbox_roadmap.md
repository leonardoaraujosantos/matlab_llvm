# Curve Fitting Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Curve-Fitting-Toolbox programs.

Source: *Curve Fitting Toolbox User's Guide* (R2026a, 13 chapters:
Getting Started · Interactive Fitting · Programmatic Curve and Surface
Fitting · Linear and Nonlinear Regression · Custom Linear and Nonlinear
Regression · Interpolation and Smoothing · Fit Postprocessing · About
Splines · Simple Spline Examples · Types of Splines · Advanced Spline
Examples · Examples · Functions).

This is the **cheapest big-win toolbox left** — "I have `(x, y)`, fit me
a curve" is a universal MATLAB entry point, and almost the entire
numeric base it needs is already shipped. Polynomial fitting rides the
existing `polyfit`/`polyval`/`mldivide`/`qr`; nonlinear library + custom
models ride the shipped Optimization Toolbox (`lsqcurvefit`/`lsqnonlin`
Levenberg-Marquardt + `lsqlin` for bound-constrained linear LS);
interpolant fits ride the existing `interp1`/`interp2`; smoothing reuses
the Signal Processing `sgolayfilt`/`medfilt1`; and the fitted-model
object is the **exact** alloc-then-populate + class-pinned-dispatch
classdef pattern proven by `LinearModel` (Stats), `tf`/`ss` (CST), and
`idpoly`/`idss` (Ident). **No external dependency** (no GSL, no Eigen,
no SciPy parity shim) — every fit core is hand-coded over the shipped
linear-algebra + optimisation kernel.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/curvefit/census_fit.m`](../examples/curvefit/census_fit.m):
*the canonical US-census demo — load decennial population `(cdate, pop)`,
fit it with `fit(cdate, pop, 'poly2')`, read `[f, gof]` (R² / RMSE),
evaluate the model at a future date `f(2030)` to forecast, then
`plot(f, cdate, pop)` against the data*. This exercises the
`fit` → `cfit` object → goodness-of-fit → `feval` → `plot` arc
end-to-end; achieving it closes **CFit-Tier-1**. The companion
`examples/curvefit/enso_fourier.m` (a Fourier + custom-equation fit) is
the **CFit-Tier-2/3** tracer-bullet.

Companion docs: [`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)
(every nonlinear + custom fit leans on the shipped `lsqcurvefit` /
`lsqnonlin` / `lsqlin`), [`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md)
(`LinearModel` shares the coefficient-CI / R² machinery; `fit(...,'poly2')`
and `fitlm` are siblings), [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md)
(`smooth`'s Savitzky-Golay + median branches reuse `sgolayfilt` /
`medfilt1`), [`plotting.md`](plotting.md) (`plot(f, x, y)` /
`plot(sfit)` route through the Cairo backend),
[`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the fit engine + the polynomial library + the `cfit` model object +
  goodness-of-fit (`fit(x,y,'polyN')`, `[f,gof]`, `feval`, `f(x)`,
  `plot(f,x,y)`). **Tier-2** is the nonlinear library models
  (exponential / power / Gaussian / Fourier / sum-of-sines / rational /
  logarithmic / sigmoidal / Weibull) over `lsqcurvefit` + the
  `fitoptions` surface (`StartPoint` / `Lower` / `Upper` / `Robust` /
  `Weights`). **Tier-3** is custom models (`fittype('a*exp(b*x)+c')`
  nonlinear + `fittype({...})` linear basis) and fit postprocessing
  (`differentiate` / `integrate` / `confint` / `predint` / residuals).
  **Tier-4** is interpolation + smoothing (interpolant fit types, the
  `smooth` function, `smoothingspline` / `csaps` / `spaps`). **Tier-5**
  is surface fitting (`fit([x y], z, ...)` → `sfit`, interpolant /
  lowess / polynomial surfaces, `tpaps` thin-plate). **Tier-6** is the
  Spline-Fitting half (ppform / B-form construction + `fn*` evaluators +
  tensor-product / rational / Chebyshev splines) plus carve-down polish
  (Simulink-lookup export, code-gen).
- **Effort** is in the existing Phase 5.6.x cadence (one focused session
  ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~1 wk · T2
  ~1.5 wk · T3 ~1.5 wk · T4 ~1.5 wk · T5 ~1.5 wk · T6 ~2.5 wk (~9.5 wk
  full)**. This is one of the *smallest* single-toolbox roadmaps by
  net-new code because so much rides on shipped solvers — each tier is
  independently shippable and demoable, and Tiers 1–3 alone close the
  90% everyday workflow.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Everything below is 🔵 not started** — clean slate. There is no
  `fit` / `fittype` / `cfit` / `sfit` / `smooth` / `csaps` / `fnval` in
  the runtime today; the deep shipped base (`polyfit`, `interp1`,
  `lsqcurvefit`, classdef plumbing) is what makes it cheap.
- **The model object is a classdef descriptor**: `cfit` (curve) and
  `sfit` (surface) each carry the model-type tag + the fitted coefficient
  vector + (for nonlinear) the Jacobian-at-solution for CIs, and expose a
  `feval` method plus the `f(x)` *call-syntax* override. This is the same
  alloc-then-populate + class-pinned-dispatch pattern as `LinearModel`
  (Stats) / `tf` (CST) / `idpoly` (Ident); auto-prepend
  `curvefit_classdefs.m` via the prelude trigger tables. The `f(x)`
  call-syntax (a class instance invoked like a function) reuses the
  `subsref`/`feval`-on-object route already used for evaluating fitted
  models.
- **`fittype` / `fitoptions` are lightweight carriers**: `fittype`
  returns a model descriptor (library tag *or* a synthesised
  function-handle for custom equations); `fitoptions` returns a struct of
  solver knobs. Both are intercepted at the constructor-call lowering
  (the classdef path), exactly like `optimoptions` (GADS) and
  `arxOptions` (Ident).
- **No external dependencies**: matching the project precedent — linear
  models via the shipped `polyfit` / `mldivide` / `qr`; nonlinear +
  custom via the shipped `lsqcurvefit` / `lsqnonlin`; bound-constrained
  linear via `lsqlin`; interpolants via `interp1` / `interp2`;
  smoothing-spline / `csaps` via a hand-coded tridiagonal natural-cubic
  solve; lowess/loess via local weighted `polyfit`.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Curve-Fitting code yet)

| Group | Surface (already shipped) | Location | How Curve Fitting uses it |
|---|---|---|---|
| Polynomial kernel | `polyfit`, `polyval`, `roots`, `poly`, `polyder`, `polyint` | `lib/Sema/Resolver.cpp` → `matlab_polyfit` / `matlab_polyval` (`runtime/matlab_runtime.cpp`) | `fit(x,y,'polyN')` linear LS (Tier-1); `differentiate`/`integrate` on polynomial fits (Tier-3). |
| Optim solvers | `lsqcurvefit` / `lsqnonlin` (Levenberg-Marquardt), `lsqlin` (bound-constrained linear LS), `fminunc`, `fmincon` | `runtime/toolbox/optim/runtime_optim.cpp` | Every nonlinear library + custom fit (Tier-2/3); `Lower`/`Upper` coefficient bounds via `lsqcurvefit` bounds; constraint points via `fmincon`. |
| Function-handle ABI | `void *fn_p` → `matlab_mat*(*)(...)`, `LowerAnonCalls` retyping | `runtime_optim.cpp`, `lib/MLIR/Passes/LowerAnonCalls.cpp` | The model handle for `fittype('a*exp(b*x)+c')` custom equations + `lsqcurvefit` model evaluation (Tier-2/3). |
| Interpolation | `interp1` (`'linear'`/`'nearest'`/`'pchip'`/`'spline'`/`'cubic'`), `interp2` | `lib/Sema/Resolver.cpp` → `matlab_interp1` | Interpolant fit types (Tier-4) + surface interpolants (Tier-5). |
| Dense linear algebra | `mldivide`, `qr`, `chol`, `svd`, `pinv`, `inv` | `runtime/matlab_runtime.cpp` | OLS / weighted LS normal equations (Tier-1/2); thin-plate + smoothing-spline linear solves (Tier-4/5); coefficient-covariance from the Jacobian (Tier-3). |
| Signal smoothing | `sgolayfilt`, `medfilt1`, `filter`, `conv` | `runtime/matlab_runtime.cpp` | `smooth(y,'sgolay')` / `'movmedian'` / `'moving'` branches (Tier-4). |
| Reductions / stats | `mean`, `std`, `sum`, `sort`, `unique`, `var` | `runtime/matlab_runtime.cpp` | Center-and-scale (`(x-μ)/σ`), gof statistics (SSE/R²/RMSE), robust-fit weight scaling (Tier-1/2). |
| Classdef plumbing | `matlab_obj_new` / `_set_*` / `_get_mat`, kwarg-ctor sugar, class-pinned dispatch, REPL persist, DAP render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The `cfit` / `sfit` / `fittype` / `fitoptions` descriptors + the `f(x)` call-syntax override. |
| Special functions | `erf`/`erfc`, `gammaln`, the internal Student-t inverse `stinv` (Stats), `exp`/`log`/`sin`/`cos` | `matlab_runtime.cpp`, `runtime/toolbox/stats/runtime_stats.cpp` | Library-model evaluation (Gaussian/exp/Fourier); coefficient-CI t-multipliers via the shipped internal `stinv` helper (Tier-3). |
| Plotting | Cairo `plot` / `scatter` / `surf` / `contour` | `runtime/plot/` | `plot(f, x, y)` (curve + data overlay), `plot(sfit)` (surface/contour), residual plots (Tier-1/5/postproc). |
| Tooling | REPL workspace persistence, DAP variable inspector, `disp(obj)` route | `runtime/runtime_debug.cpp`, `lib/MLIR/Lowering.cpp` | `cfit` / `sfit` shown in REPL + debugger; `f` formatted print (`disp(cfit)` → model formula + coefficients). |

**Net assessment**: the *numeric base* (polynomial LS, nonlinear LS,
interpolation, smoothing primitives, classdef plumbing, plotting) is
**already shipped**. The genuinely new code is (a) the **`fit` dispatcher
+ `cfit`/`sfit` objects** (model-type tag → solver routing + `feval`),
(b) the **library-model catalogue** (~11 families × start-point heuristic
+ formula string), (c) the **`fittype`/`fitoptions` carriers** + custom
equation-string → handle synthesis, (d) the **`gof` / `confint` /
`predint` / `differentiate` / `integrate` postprocessing**, and (e) the
**Spline-Fitting `fn*` layer** (ppform/B-form). Each is a self-contained
hand-coded routine over the shipped base — the bulk of the algorithmic
heavy lifting (LM, QR, interpolation) is done.

---

## 2. Tier-1 — Fit engine + polynomial library + `cfit` object + goodness-of-fit 🔵

Goal: the universal "fit me a polynomial and tell me how good it is"
loop. Closes the headline.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `fit(x, y, 'polyN')` | Curve fit dispatcher: parse the model-type string → route polynomial degrees 1–9 to a Vandermonde linear LS. Returns a `cfit`. Center-and-scale on by default for conditioning (store `μ`/`σ` in the object). | `polyfit` / `qr` / `mldivide` |
| 1.2 | `cfit` classdef | Descriptor carrying `{modelTag, coeffs, μ, σ, formula, J}`. Methods: `feval`, `coeffvalues`, `coeffnames`, `formula`, `disp`. Auto-prepended via `curvefit_classdefs.m`. | classdef plumbing |
| 1.3 | `feval(f, xq)` + `f(xq)` call-syntax | Evaluate the fitted model at query points; `f(xq)` (object-as-function) routes through the `subsref`/call-on-object path to `feval`. | call-on-object route |
| 1.4 | `[f, gof] = fit(...)` | Goodness-of-fit struct: `sse`, `rsquare`, `dfe`, `adjrsquare`, `rmse` (computed from residuals + dof). Second return via the existing multi-output splitter. | reductions |
| 1.5 | `[f, gof, output] = fit(...)` | `output` struct: `numobs`, `numparam`, `residuals`, `Jacobian`, `exitflag`, `iterations` (trivially populated for the linear path). | splitter |
| 1.6 | `plot(f, x, y)` | Overlay the fitted curve on the scatter of data (Cairo). `plot(f, x, y, 'residuals')` for the residual plot. | `runtime/plot/` |
| 1.7 | `disp(cfit)` / REPL render | Formatted print: `General model Poly2:` + the formula + the coefficient block with (where available) 95% CIs. DAP variable view shows model tag + coeffs. | `disp(obj)` route, DAP |

**Headline-within-tier**: the census demo —
`f = fit(cdate, pop, 'poly2'); f(2030)` forecast + `[f,gof]` R² + `plot`.

**Compile/Execute wiring**: new `runtime/toolbox/curvefit/runtime_curvefit.cpp`
+ `curvefit_classdefs.m`; register `fit` / `feval` / `coeffvalues` /
`coeffnames` / `formula` in `Resolver.cpp`; `pde_table` loose-match
entries in `LowerTensorOps.cpp` (string model-tag arg → `matlab_string*`,
the same path Image used for `imread('f.png')`); prelude trigger set for
`cfit` / `fittype` / `fitoptions`; intercept `fit(...)` as a
class-returning builtin so `pinnedOfRhs` propagates the `cfit` pin.

---

## 3. Tier-2 — Nonlinear library models + `fitoptions` 🔵

Goal: the named-model catalogue — pick a family by string, get a fitted
nonlinear model with sensible auto start points.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | exponential | `'exp1'` (`a·e^{bx}`), `'exp2'` (`a·e^{bx}+c·e^{dx}`). Start-point heuristic: log-linear regression seed. | `lsqcurvefit` |
| 2.2 | power | `'power1'` (`a·x^b`), `'power2'` (`a·x^b+c`). Log-log seed. | `lsqcurvefit` |
| 2.3 | Gaussian | `'gauss1'`…`'gauss8'` (`Σ aᵢ·e^{−((x−bᵢ)/cᵢ)²}`). Peak-finding seed for centres. | `lsqcurvefit`, `findpeaks` |
| 2.4 | Fourier | `'fourier1'`…`'fourier8'` (`a₀+Σ aₙcos(nωx)+bₙsin(nωx)`). FFT seed for `ω`. | `lsqcurvefit`, `fft` |
| 2.5 | sum of sines | `'sin1'`…`'sin8'` (`Σ aᵢ·sin(bᵢx+cᵢ)`). | `lsqcurvefit` |
| 2.6 | rational | `'rat02'`…`'rat55'` (ratio of polynomials, numerator deg 0–5 / denom 1–5). | `lsqcurvefit` |
| 2.7 | logarithmic / sigmoidal / Weibull | `'log'`-style custom, `'logistic'` 4-param, `'weibull'` (`a·b·x^{b−1}·e^{−a·x^b}`). | `lsqcurvefit` |
| 2.8 | `fitoptions` carrier | `fitoptions('Method','NonlinearLeastSquares','StartPoint',…,'Lower',…,'Upper',…,'Weights',…,'Robust',…)` → struct of solver knobs; `fit(x,y,model,opts)` consumes it. Constructor-intercept path (like `optimoptions`). | classdef plumbing |
| 2.9 | weighted + robust LS | `'Weights'` → weighted residuals into `lsqcurvefit`; `'Robust','Bisquare'`/`'LAR'` → IRLS reweighting loop around the LM core. | `lsqcurvefit`, IRLS |
| 2.10 | bounds | `'Lower'`/`'Upper'` coefficient bounds → `lsqcurvefit(lb, ub)` bound-constrained path. | `lsqcurvefit` bounds |

**Headline-within-tier**: UG "Fit Exponential Models" —
`fit(t, decay, 'exp2')` with auto start points, read the two rate
constants, overlay on data.

**Compile/Execute wiring**: extend the Tier-1 dispatcher with the
nonlinear branch (build the model handle per family + start-point seed,
hand to `lsqcurvefit`); register `fitoptions`; the per-family formula
strings live in `runtime_curvefit.cpp` (read at runtime, like Image's
`fspecial` option strings).

---

## 4. Tier-3 — Custom models + fit postprocessing 🔵

Goal: arbitrary user equations + the post-fit analysis surface (the
"what do I do with `f`" half).

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | custom nonlinear | `fittype('a*exp(-b*x)+c')` — parse the equation string, identify coefficients (non-`x` identifiers) vs the independent var, synthesise a function handle, `fit` via `lsqcurvefit`. | equation-string → handle, `lsqcurvefit` |
| 3.2 | custom linear | `fittype({'1','x','x^2'})` or `fittype('a*sin(x)+b*cos(x)','linear')` — linear-in-coefficients basis → design matrix → linear LS. | `qr`/`mldivide` |
| 3.3 | `differentiate(f, xq)` | Analytic derivative for library models (polynomial via `polyder`, closed-form for exp/gauss/…); finite-difference fallback for custom. | `polyder` |
| 3.4 | `integrate(f, xq, x0)` | Analytic integral for library models (`polyint`); adaptive quadrature fallback. | `polyint`, `trapz` |
| 3.5 | `confint(f, level)` | Coefficient confidence intervals from the Jacobian-at-solution: `Σ = σ²(JᵀJ)⁻¹`, CI = `b ± t·diag(√Σ)`. The t-multiplier reuses the Stats runtime's internal `stinv` (Student-t inverse). | `qr`/`inv`, Stats `stinv` |
| 3.6 | `predint(f, xq, level, kind)` | Prediction / functional bounds (`'observation'` vs `'functional'`); propagate `Σ` through the model gradient at `xq`. | 3.5 + model grad |
| 3.7 | residual analysis | `[f,gof] = fit(...)` residuals + `plot(f,x,y,'residuals')`; standardized residuals. | `runtime/plot/` |
| 3.8 | model utilities | `coeffvalues` / `coeffnames` / `probvalues` / `probnames` / `formula` / `numcoeffs` / `category`. | classdef |

**Headline-within-tier**: the ENSO demo —
`ft = fittype('a + b*sin(2*pi*x/12) + c*cos(2*pi*x/12)')` custom Fourier,
`fit(month, sst, ft)`, then `differentiate` + `confint`.

**Compile/Execute wiring**: the equation-string parser is the one new
front-end-ish piece — it can reuse the anon-function machinery
(`@(x,a,b,c) a*exp(-b*x)+c`) so the body lowers through the existing
`LowerAnonCalls` retyping; register `differentiate` / `integrate` /
`confint` / `predint` as class-pinned-first-arg methods in
`Lowering.cpp::CallOrIndex` (the same dispatch as CST `pole(sys)`).

---

## 5. Tier-4 — Interpolation + smoothing 🔵

Goal: the nonparametric half — interpolant fit types, the `smooth`
function, and smoothing splines.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | interpolant fits | `fit(x,y,'linearinterp'/'nearestinterp'/'pchipinterp'/'cubicinterp'/'splineinterp')` → `cfit` whose `feval` calls the matching `interp1` mode. | `interp1` |
| 4.2 | extrapolation | `'Extrapolation','linear'/'nearest'/'none'` option on interpolant fits; out-of-range query handling. | `interp1` |
| 4.3 | `smooth(y, …)` | `smooth(y)` / `smooth(y,span)` moving average; `'moving'`, `'lowess'`, `'loess'`, `'rlowess'`, `'rloess'`, `'sgolay'`. Local-regression branches via windowed weighted `polyfit`; `'sgolay'` via the shipped `sgolayfilt`. | `sgolayfilt`, `polyfit` |
| 4.4 | smoothing spline | `fit(x,y,'smoothingspline')` + `'SmoothingParam'`; `csaps(x,y,p)` cubic smoothing spline (tridiagonal natural-cubic minimiser of `p·Σwᵢ(yᵢ−s(xᵢ))²+(1−p)∫s''²`). | tridiag solve |
| 4.5 | `spaps` | Smoothest spline within a tolerance (`spaps(x,y,tol)`). | csaps core |
| 4.6 | lowess/loess standalone | The `smooth` local-regression cores also exported as the curve-fit `'lowess'`/`'loess'` model types returning a `cfit`. | 4.3 |

**Headline-within-tier**: UG "Smoothing Data" —
`smooth(noisy, 0.1, 'rloess')` robust local regression vs a moving
average, overlaid.

**Compile/Execute wiring**: `smooth` is a plain matrix-returning builtin
(`Resolver.cpp` + `LowerTensorOps.cpp` `pde_table`, string method arg →
`matlab_string*`); interpolant `cfit` carries the raw `(x,y)` + mode tag
and routes `feval` to `matlab_interp1`; `csaps`/`spaps` get
`runtime_curvefit.cpp` entries.

---

## 6. Tier-5 — Surface fitting (`sfit`) 🔵

Goal: the 2-predictor half — `fit([x y], z, …)` → `sfit`. Needs the 3-D
indexing + `meshgrid` substrate already shipped for Image.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | polynomial surfaces | `fit([x y], z, 'polyNM')` (`poly11`…`poly55`) — bivariate-term design matrix → linear LS → `sfit`. | `qr`/`mldivide` |
| 5.2 | `sfit` classdef | Surface model object: coeffs + bivariate formula + `feval(sf, xq, yq)`. | classdef |
| 5.3 | interpolant surfaces | `fit([x y], z, 'linearinterp'/'cubicinterp'/'nearestinterp'/'biharmonicinterp')` — scattered-data interpolation; biharmonic via the radial-basis solve. | `interp2`, RBF |
| 5.4 | lowess surfaces | `'lowess'`/`'loess'` local-regression surfaces. | 4.3 |
| 5.5 | thin-plate splines | `tpaps(xy, z, p)` thin-plate smoothing spline (radial `r²log r` basis). | dense solve |
| 5.6 | surface plotting | `plot(sf)` / `plot(sf,[x y],z)` → Cairo `surf` + `contour`; `feval` over a `meshgrid`. | `runtime/plot/`, `meshgrid` |
| 5.7 | surface gof | `[sf, gof] = fit([x y], z, …)` SSE/R²/RMSE over the surface. | reductions |

**Headline-within-tier**: UG "Surface Fitting to Franke Data" —
`fit([x y], z, 'poly23')`, `plot(sf,[x y],z)`, read R².

**Compile/Execute wiring**: surface fit needs the `[x y]` two-column
predictor convention + `meshgrid` evaluation (both shipped); `sfit`
mirrors `cfit`; `plot(sf)` reuses the Image-era 3-D / `surf` path.

---

## 7. Tier-6 — Spline Fitting (ppform / B-form) + carve-down polish 🔵

Goal: the Spline-Fitting half of the toolbox (Chapters 8–12) — the
`fn*` function family over the ppform and B-form representations, plus
the remaining polish.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | construction | `spline(x,y)`, `csape(x,y,conds)` (end conditions), `pchip(x,y)`, `spap2`/`spapi` (least-squares / interpolating B-spline), `spaps` (smoothing). | tridiag, `qr` |
| 6.2 | ppform / B-form makers | `ppmak(breaks,coefs)`, `spmak(knots,coefs)`, `fn2fm` (convert between forms), `fnbrk` (extract parts). | descriptor structs |
| 6.3 | `fn*` evaluators | `fnval(f,xq)` (evaluate), `fnder(f)` (differentiate), `fnint(f)` (integrate), `fnplt(f)` (plot), `fnmin`/`fnzeros` (extrema/roots). | `runtime/plot/` |
| 6.4 | tensor-product splines | `csapi`/`spapi` on N-D grids (gridded data), `fnval` over a mesh. | 6.1 |
| 6.5 | rational / NURBS | `rsmak` (rational spline maker — circle/sphere), `rpmak`, rsform evaluation. | 6.2 |
| 6.6 | Chebyshev spline | `chbpnt` + Remez iteration for the equioscillating spline. | Remez loop |
| 6.7 | thin-plate (spline side) | `tpaps` (shared with 5.5), `stmak`/`stcol` scattered-translation form. | 5.5 |
| 6.8 | Simulink-lookup export | `fit` → lookup-table coefficients export (the UG "Export Fit to Simulink Lookup Table" path), reusing the mflowLink lane. | embedded-coder lane |

**Headline-within-tier**: UG "Cubic Spline Interpolation" —
`pp = spline(x,y); fnval(pp, xq)` + `fnplt(fnder(pp))` (plot the
derivative spline).

**Carve-down polish (cross-tier follow-ons)**: `excludedata` /
outlier-exclusion options, `prepareCurveData` / `prepareSurfaceData`
(NaN/orientation cleanup), `cflibhelp` model catalogue listing,
`coeffvalues` on multi-term, `setoptions`/`getoptions` round-trip.

---

## 8. Compile/Execute · Debug/REPL · Examples · Tests (cross-cutting)

The four delivery surfaces the project always closes per toolbox — each
tier ships across **all four** before it counts as done.

### 8.1 Compile / Execute

- **Backends**: LLVM JIT + native + `-emit-c` / `-emit-cpp` are the
  primary lanes (matching Stats/Image — the classdef + function-handle
  ABI is C/C++-shaped). `-emit-python` / `-emit-typescript` parity is a
  per-tier stretch; `-emit-systemverilog` is **not** a target (curve
  fitting is host-side analysis, not synthesizable) — emit a clear
  diagnostic, like sym does.
- **Runtime**: `runtime/toolbox/curvefit/runtime_curvefit.cpp` (fit
  cores, library catalogue, gof, `smooth`, `csaps`, `fn*`) +
  `runtime/toolbox/curvefit/curvefit_classdefs.m` (`cfit` / `sfit` /
  `fittype` / `fitoptions`). Add to the strict no-C-cast list (use
  `static_cast`), mirroring `runtime_images.cpp`.
- **Wiring**: builtin names in `Resolver.cpp`; `pde_table` loose-match +
  string-literal-arg → `matlab_string*` in `LowerTensorOps.cpp`;
  `fit`/`csape`/`spline` registered as class-returning so `pinnedOfRhs`
  propagates the `cfit`/`sfit`/ppform pin; postproc methods
  (`differentiate`/`feval`/`confint`) as class-pinned-first-arg dispatch
  in `Lowering.cpp::CallOrIndex`; prelude trigger set so a `fit`-only
  program doesn't pay the unused-classdef cost.

### 8.2 Debug / REPL

- `cfit` / `sfit` / `fittype` persist across REPL inputs (workspace slot
  with a class tag) and render in the **DAP variable inspector** (model
  tag + coefficient vector), via the shipped `runtime_debug.cpp`
  classdef-render path.
- `disp(f)` formats the MATLAB-faithful model block (`General model
  Poly2:` + formula + `Coefficients (with 95% confidence bounds):`).
- The `f(xq)` call-syntax works in the REPL JIT (object-as-function eval).

### 8.3 Examples (`examples/curvefit/`)

| Example | Closes | Exercises |
|---|---|---|
| `census_fit.m` | **T1 headline** | `fit(...,'poly2')` → `[f,gof]` → `f(2030)` forecast → `plot(f,x,y)` |
| `exp_decay_fit.m` | T2 | `fit(t,y,'exp2')` auto start points + read rate constants |
| `peaks_gauss.m` | T2 | `fit(x,y,'gauss2')` two-peak deconvolution |
| `enso_fourier.m` | **T2/3 tracer** | custom `fittype` Fourier + `differentiate` + `confint` |
| `robust_smooth.m` | T4 | `smooth(...,'rloess')` vs moving average; `'splineinterp'` |
| `franke_surface.m` | T5 | `fit([x y],z,'poly23')` → `sfit` → `plot(sf)` surface/contour |
| `spline_interp.m` | T6 | `spline` / `csaps` / `fnval` / `fnplt(fnder(pp))` |

### 8.4 Tests (`test/Run/`)

Gating tests follow the `curvefit_*.m` convention with a `.stdout`
golden + per-backend `.skip-emit-*` files where a lane is out of scope
(SV always skipped; Python/TS skipped where the classdef path is rough,
matching the Image `image_png_roundtrip` precedent).

| Test | Tier | Asserts |
|---|---|---|
| `curvefit_poly.m` | T1 | `poly2` coeffs + `gof.rsquare` + `f(xq)` to tolerance |
| `curvefit_gof.m` | T1 | `[f,gof,output]` fields populated; SSE/RMSE/dfe |
| `curvefit_exp.m` | T2 | `exp2` recovers known rate constants on synthetic data |
| `curvefit_options.m` | T2 | `fitoptions` `StartPoint`/`Lower`/`Upper` honoured; robust LAR |
| `curvefit_custom.m` | T3 | `fittype('a*exp(-b*x)+c')` custom fit + `confint` bounds |
| `curvefit_diffint.m` | T3 | `differentiate`/`integrate` analytic vs finite-diff |
| `curvefit_smooth.m` | T4 | `smooth` moving/lowess/sgolay branches; `csaps` |
| `curvefit_interp.m` | T4 | interpolant fit types match `interp1` modes |
| `curvefit_surface.m` | T5 | `poly23` surface coeffs + `sfit` `feval` over a grid |
| `curvefit_spline.m` | T6 | `spline`/`csaps` + `fnval`/`fnder`/`fnint` round-trip |

Target: **~10 gating tests** (one per major surface), in line with
Image (10) and Stats (12). Full regression must stay green
(currently 465 run-tests) — the badge bumps to **17 toolboxes** and the
run-tests count grows by the new gating set.

---

## 9. Carve-outs (explicitly out of scope)

Matching the per-toolbox precedent — the GUI / interactive / codegen-UI
surfaces are deferred:

- **Curve Fitter app** (`curveFitter`) and **Spline Tool**
  (`splinetool`) — interactive GUIs; the entire Chapter 2 "Interactive
  Fitting" + the app-driven workflows. The project is headless; the
  programmatic `fit`/`fittype` API is the whole target.
- **Generate Code from the app** (the `Export ▸ Generate Code` button) —
  we *are* the codegen; the app-export path is N/A.
- **Live Editor tasks** for curve fitting.
- **Session save/reopen** (`.sfit` interactive sessions).
- **Code Generation (MATLAB Coder) of fits** beyond the existing
  `-emit-*` lanes — `fiteval`-style standalone C is a stretch follow-on.
- **NURBS authoring depth** beyond `rsmak`/`rpmak` library shapes
  (full rational-spline editing is a Tier-6 stretch).

These are documented follow-ons, not blockers: every numeric and
object-API surface a *script* uses is in Tiers 1–6.

---

## 10. Effort summary

| Tier | Scope | Effort | Net-new code | Status |
|---|---|---|---|---|
| T1 | fit engine + polynomial + `cfit` + gof + plot | ~1 wk | dispatcher + `cfit` + gof | ✅ shipped |
| T2 | nonlinear library models + `fitoptions` | ~1.5 wk | catalogue + start-point seeds + opts carrier | ✅ shipped |
| T3 | custom models + postprocessing | ~1.5 wk | equation parser + diff/int/confint/predint | ✅ shipped (predint carved) |
| T4 | interpolation + smoothing | ~1.5 wk | `smooth` + interpolant `cfit` + `csaps`/`spaps` | ✅ shipped (`spaps` carved) |
| T5 | surface fitting (`sfit`) | ~1.5 wk | bivariate design + `sfit` + `tpaps` | ✅ shipped (poly surfaces; interpolant/lowess/`tpaps` carved) |
| T6 | spline `fn*` layer + polish | ~2.5 wk | ppform/B-form + `fn*` + NURBS/Chebyshev | ✅ shipped (ppform `spline`/`pchip`/`fnval`/`fnder`/`fnint`/`fnbrk`/`ppmak`; B-form/NURBS/Chebyshev/Simulink-export carved) |
| **Total** | | **~9.5 wk** | | **ALL 6 TIERS SHIPPED 2026-05-23 — badge 17** |

**Carve-downs** (documented follow-ons, matching project precedent): rational
`rat_nm` / logistic / Weibull library models · `predint` · custom-linear cell
form `fittype({'1','x','x^2'})` · interpolant/lowess surfaces + `tpaps` ·
`csape` end-conditions / `spap2`/`spapi` B-form / `spaps` / `rsmak` NURBS /
`chbpnt` Chebyshev / `fnmin`/`fnzeros` / tensor-product / Simulink-lookup
export. Two pre-existing Sema gaps surfaced (documented in project memory):
mixing scalar `f(x)` with vector `feval(f,xv)` of one model forces a single
matrix return type; `scalar*transcendental(vector)` inside a sum doesn't lower.

**Recommended slice order**: T1 → T2 → T3 closes the everyday 90%
workflow (~4 wk) and is the highest-ROI cut — at that point
`fit`/`fittype`/`feval`/`gof`/`confint` cover what most users mean by
"curve fitting." T4 (smoothing) and T5 (surfaces) are independent
add-ons; T6 (the Spline-Fitting half) is the long tail and can land
incrementally.
