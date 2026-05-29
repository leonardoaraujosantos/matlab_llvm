# System Identification Toolbox — Tutorial

The System Identification Toolbox runtime builds dynamic models from
measured input/output data: linear ARX/AR/ARMAX/OE/BJ polynomial models,
subspace state-space estimation, frequency-response and grey-box
estimation, and online recursive / nonlinear-Kalman estimation. Identified
models convert into Control System Toolbox objects, so they feed straight
into analysis (`pole`, `step`, `bode`) and into the MPC designer. All six
tiers are shipped. The prediction-error engine reuses the shipped
Optimization Toolbox `lsqnonlin`.

## Supported features

- **Data container**: `iddata(y, u, Ts)` (use `[]` for the input to make a
  time series).
- **Linear black-box (Tier-1/2)**: `arx`, `ar` (Yule-Walker), `armax`,
  `oe`, `bj`, `iv4` (instrumental variables).
- **Validation & metrics**: `sim`, `predict`, `compare` (NRMSE fit %),
  `goodnessOfFit`, `fpe`, `aic`, `pe`, `resid`, `delayest`.
- **State-space & TF (Tier-3)**: `n4sid`, `ssest` (subspace -> `idss`),
  `tfest`.
- **Frequency / impulse / grey-box (Tier-4)**: `etfe`, `spa` (-> `idfrd`),
  `impulseest`, `forecast`, `greyest` (linear grey-box).
- **Online & nonlinear (Tier-5)**: `recursiveARX`, `recursiveLS`,
  `extendedKalmanFilter`, `unscentedKalmanFilter`, `nlgreyest`.
- **Regularization & introspection (Tier-6)**: `arxOptions`
  (`.Regularization`), `getcov`, `getpvec`, `setpvec`.
- **Conversions**: `ss(model)`, `tf(model)` into Control System Toolbox.

## Build & run

```bash
build/matlabc -emit-llvm examples/ident/arx_lab_process.m > /tmp/arx_lab_process.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/arx_lab_process.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/arx_lab_process
/tmp/arx_lab_process
```

## Worked examples

### ARX estimation from lab data (`examples/ident/arx_lab_process.m`)

The Tier-1 end-to-end loop: synthesise a second-order lab record, fit an
ARX `[na nb nk]` model, validate it, and convert it to a discrete `ss` to
read its poles via the shipped Control System Toolbox `pole`.

```matlab
z = iddata(y, u, 0.08);            % I/O record at Ts = 0.08 s
m = arx(z, [2 2 1]);               % na=2, nb=2, nk=1 input delay
fprintf('A(q) = [1, %.3f, %.3f]\n', m.A(2), m.A(3));
fit = compare(z, m);               % NRMSE fit ~ 96.95 %
fprintf('FPE = %.5f   AIC = %.2f\n', fpe(m), aic(m));
sys = ss(m);   p = pole(sys);      % idpoly -> discrete ss -> z-plane poles
```

The `idpoly` returned by `arx` exposes `A` / `B` / `nk` / `Ts` /
`NoiseVariance`.

### ARMAX noise-model refinement (`examples/ident/armax_refine.m`)

When the equation noise is coloured, plain ARX distorts the dynamics to
absorb it. `armax` adds a `C(q)` polynomial that captures the colour
directly, leaving whiter residuals. `resid` returns a
`[maxAutoCorr; maxCrossCorr]` whiteness diagnostic.

```matlab
z  = iddata(y, u, 1);
ma = arx(z, [1 1 1]);     ra = resid(ma, z);   % coloured -> large autocorr
mx = armax(z, [1 1 1 1]); rx = resid(mx, z);   % recovers C2 ~ 0.71, whitened
fprintf('ARX   fit = %.1f %%\n', compare(z, ma));
fprintf('ARMAX fit = %.1f %%\n', compare(z, mx));
fprintf('ARMAX FPE = %.5f   AIC = %.1f\n', fpe(mx), aic(mx));
```

### Data-driven control (`examples/ident/data_driven_mpc.m`)

The cross-toolbox payoff: identify a state-space model with `ssest`
(subspace), convert it to a Control System Toolbox `ss`, and hand it to the
MPC designer — no first-principles model is ever written.

```matlab
z   = iddata(y, u, 0.1);
sys = ssest(z, 2);                 % subspace estimate, order 2
fit = compare(z, sys);             % NRMSE ~ 96.8 %
P    = ss(sys);                    % idss -> CST ss (carries discrete Ts)
ctrl = mpc(P, 10, 3);              % prediction horizon 10, control horizon 3
yc   = sim(ctrl, 30, 1.0);         % closed-loop step to setpoint 1.0
```

### Linear grey-box parameter estimation (`examples/ident/greybox_msd.m`)

`greyest` recovers the *physical constants* of a system with known
structure but unknown parameters. The structure function maps a parameter
vector to a packed continuous realization `M = [A B; C D]`; `greyest`
discretizes (ZOH) at the data `Ts` and minimizes the prediction error with
`lsqnonlin`.

```matlab
% mass-spring-damper: x1' = x2; x2' = -(k/m)x1 - (c/m)x2 + (1/m)F; y = x1
structfn = @(p) [0, 1, 0; -p(1), -p(2), 1; 1, 0, 0];
m = greyest(z, [3.0; 1.0], structfn, 2);   % par0 deliberately off
fprintf('k/m = %.4f (true 4.0)\n', m.Parameters(1));
fprintf('c/m = %.4f (true 1.2)\n', m.Parameters(2));
```

### Recursive ARX tracking (`examples/ident/recursive_arx_tracking.m`)

`recursiveARX` follows time-varying dynamics sample-by-sample with a
forgetting factor and no batch re-fit. The estimator object carries its
mutable `Parameters` + `Covariance`.

```matlab
r = recursiveARX([1 1 1]);
r.ForgettingFactor = 0.96;         % < 1 tracks change; 1 = infinite memory
for k = 2:N
    th = step(r, y(k), u(k));      % A = [1 -a] -> a = -th(1)
end
```

The plant pole jumps `0.50 -> 0.85` mid-experiment and the estimate
follows it.

### Nonlinear Kalman state estimation (`examples/ident/ukf_state_estimation.m`)

The project's first dynamic Kalman loop: reconstruct a pendulum's full
`[angle; rate]` state from noisy angle-only measurements, recovering the
never-measured velocity. `StateFcn` / `MeasFcn` are single-argument
handles; the filter object carries its mutable `State` + `StateCovariance`.

```matlab
StateFcn = @(x) [x(1) + 0.1*x(2); x(2) - 0.1*sin(x(1))];
MeasFcn  = @(x) [x(1)];
ukf = unscentedKalmanFilter([0.0; 0.0], eye(2), 0.0001*eye(2), [0.01]);
for k = 1:N
    predict(ukf, StateFcn);
    xu = correct(ukf, MeasFcn, ym(k));
end
% xu ~ [angle; rate] tracks the truth; cross-check with extendedKalmanFilter
```

### Regularized ARX (`examples/ident/arx_regularization.m`)

On short / noisy data a ridge `lambda` on the Gram matrix shrinks the
estimate toward zero and tightens the parameter covariance, exposed through
the `arxOptions` carrier. Tier-6 also adds `getpvec` / `setpvec` / `getcov`.

```matlab
m  = arx(z, [1 1 1]);              % plain LS baseline
opt = arxOptions();   opt.Regularization = 1.0;
mr = arx(z, [1 1 1], opt);         % ridged -> closer to truth, shrunk variance
c  = getcov(mr);                   % parameter covariance
theta = getpvec(mr);               % packed parameter vector
setpvec(mr, [-0.45; 0.95]);        % write back
```

## Limitations & carve-outs

- SISO only across all estimators (MIMO is carved out of every tier).
- Direct `idpoly(A, B, ...)` construction is deferred — `arx` / `ar` /
  `armax` / `oe` / `bj` are the model factories.
- Tier-3: projection-based N4SID, auto-order, innovations gain `K` (0
  today), `idtf` / `idproc` named classes, `procest`, `findstates`.
- Tier-4: `spafdr`, lag-window spectral, regularized-FIR `impulseest`,
  continuous-time grey-box separate outputs, data preprocessing
  (`detrend` / `resample` / `merge` / `misdata`).
- Tier-5: nonlinear black-box `nlarx` / `nlhw`, `particleFilter`,
  `recursiveARMAX` / `recursiveOE` / `recursiveBJ`, analytic Jacobians and
  multi-output measurement for EKF/UKF, `ode45`-based `nlgreyest` rollout.
- Tier-6: estimation `Report` struct, uncertainty bands on `bode`/`step`,
  other `*Options` carriers, `merge`, ARIMA / seasonal forms.

## See also

- Roadmap: [`../ident_toolbox_roadmap.md`](../ident_toolbox_roadmap.md)
- Examples: `examples/ident/`
