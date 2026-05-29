# Control System Toolbox — Tutorial

`matlab_llvm` ships the numeric core of the Control System Toolbox: the linear-algebra primitives a control engineer needs (`expm`, non-symmetric `eig`, `schur`, `lyap`/`dlyap`, `care`/`dare`) and a functional, matrix-argument API built on top of them — LQR/LQG design, pole placement, Kalman gains, frequency response, stability margins, model interconnection, and balanced model reduction. The state-space `(A, B, C, D)` matrices are the primary working representation, but model objects (`tf`, `ss`) and the functions that consume them (`step(sys)`, `bode(sys)`, `margin`, `stepinfo`, `kalman(sys, …)`) now work too. The one remaining gap is the 3-argument model-object discretiser `c2d(sys, Ts, method)` (the matrix-arg `[Ad, Bd] = c2d(A, B, Ts)` form ships).

## Supported features

- **Tier-1 linalg primitives** — `expm`, `eig` (non-symmetric, complex returns), `schur` / `[U, T] = schur(A)`, `lyap` / `dlyap`, `care`, `dare`, `inv`, `det`.
- **Model objects** — `tf(num, den[, Ts])`, `s = tf('s')`, `ss(A, B, C, D)`, `disp(G)` s-domain rendering, operator overloads (`G + H`, `G * H`, `G / H`), `tfdata` / `ssdata`.
- **Model-object analysis** — `step(sys[, t])`, `bode(sys[, w])`, `margin(sys)`, `bandwidth`, `dcgain`, `stepinfo`, `initial(sys, x0, t)`, `kalman(sys, Q, R)`.
- **LQR / LQG design** — `lqr(A, B, Q, R)` (incl. 3-return `[K, S, e]`), `dlqr(Ad, Bd, Q, R)`; Kalman gains `kalman_L` (continuous) / `kalmd_L` (discrete) via the LQR/Kalman duality.
- **Pole placement** — `place(A, B, P)` (Ackermann), `ctrb` / `obsv` controllability & observability matrices.
- **Frequency response (matrix-arg)** — `bode_ss(A, B, C, D, w)`, `bode_tf(b, a, w)`, `gain_margin` / `phase_margin`.
- **Time response (matrix-arg)** — `step_ss(A, B, C, D, dt, N)`, `lsim_ss(A, B, C, D, u, dt)`; discretisation `[Ad, Bd] = c2d(A, B, Ts)` (Van Loan ZOH).
- **Characterization** — `isstable(A)`, `damp(A)` (`[wn, zeta]` per pole), `hsvd(A, B, C)`, `gram_c` / `gram_o`, `norm_h2(A, B, C)`.
- **Interconnection (matrix-arg)** — `series_ss`, `parallel_ss`, `feedback_ss`, `append_ss` (block-diagonal MIMO).
- **Model reduction** — `balreal_T` (balancing transform), `balred_A` / `balred_B` / `balred_C` (k-state balanced truncation).

> **Still open:** the 3-argument model-object discretiser `c2d(sys, Ts, method)` and `d2c` — `examples/control/c2d_zoh_demo.m` documents this API and fails today with `unsupported call shape for built-in function 'c2d'`. Use the matrix-arg `[Ad, Bd] = c2d(A, B, Ts)` form (Van Loan ZOH) instead. `zpk` model objects are also not shipped.

> **Note:** the model-object examples `tf_basic.m`, `step_response_siso.m`, `bode_first_order.m`, `lqr_double_integrator.m`, and `kalman_tracker.m` carry "NOT YET SHIPPED" header comments that are now stale — they compile and run correctly today (the model-object surface landed after those headers were written).

## Build & run

Compile any shipped (✅) example end-to-end:

```bash
build/matlabc -emit-llvm examples/control/lqr_via_care.m > /tmp/lqr_via_care.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/lqr_via_care.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/lqr_via_care
/tmp/lqr_via_care
```

## Worked examples

### Matrix exponential — the workhorse primitive  (`examples/control/expm_basic.m`)

`expm` underpins `c2d` ZOH discretisation, exact `lsim` stepping, and initial-condition response.

```matlab
% Rotation generator: expm([0 1; -1 0]·theta) = rotation by theta.
A = [0 1; -1 0];
R = expm(A * (pi/2));
disp(R(1, 2));   % sin(pi/2) = 1

% The c2d ZOH augmented-matrix trick: expm([A B; 0 0]·Ts) = [Ad Bd; 0 I].
Ts = 0.1;
M  = [-1 0 1; 0 -2 0.5; 0 0 0];
EM = expm(M * Ts);
disp(EM(1, 1));   % exp(-0.1) = 0.9048...
disp(EM(1, 3));   % Bd(1) = integral of exp(-tau) over [0, 0.1]
```

The augmented `[A B; 0 0]` exponential yields both `Ad` (top-left block) and `Bd` (top-right column) in a single call — exactly how `c2d` ZOH is implemented. `eig_poles_demo.m` and `schur_modal_split.m` cover the spectral companions (`eig` of non-symmetric state matrices, `schur` modal split).

### LQR via the algebraic Riccati equation  (`examples/control/lqr_via_care.m`)

The infinite-horizon LQR pipeline: solve `care`, read off the gain, form the closed loop.

```matlab
A = [0 1; 0 0];
B = [0; 1];
Q = [1 0; 0 1];
R = [1];

X = care(A, B, Q, R);          % closed form [sqrt(3) 1; 1 sqrt(3)]
K = inv(R) * (B' * X);         % gain        [1 sqrt(3)]
Acl = A - B * K;
disp(real(eig(Acl)));          % -sqrt(3)/2 each
disp(imag(eig(Acl)));          % +- 0.5

% Optimal cost from x0 = [1; 0]:  J* = x0' X x0.
x0 = [1; 0];
J = x0' * X * x0;
```

`care(A, B, Q, R)` returns the unique stabilising Riccati solution; `K = R⁻¹B'X` is the state-feedback gain and the closed-loop poles land in the open left-half plane. `lqr_discrete_workflow.m` wraps this in the top-level `lqr(A, B, Q, R)` and `[Ad, Bd] = c2d(A, B, Ts)` and rolls the discrete closed loop forward. The discrete-native pipeline is `dare_dlqr_demo.m` (`dare` + `dlqr` via Newton-Kleinman).

### Pole placement  (`examples/control/place_pole_assignment.m`)

The user-facing alternative to LQR — specify *where* the closed-loop poles go and `place` computes the gain. Demonstrated on the unstable inverted pendulum.

```matlab
g = 9.81; L = 1.0;
A = [0 1; g/L, 0];     % open-loop has a +sqrt(g/L) ≈ +3.13 unstable pole
B = [0; 1];

Co = ctrb(A, B);
disp(det(Co));         % nonzero -> controllable

P = [0 - 2.0; 0 - 2.0];
K = place(A, B, P);
Acl = A - B * K;
disp(real(eig(Acl)));  % must equal P = {-2, -2}

Ob = obsv(A, C);       % observability with C = [1 0]
disp(det(Ob));         % nonzero -> observable
```

`ctrb` / `obsv` build the controllability / observability matrices (full rank ⇒ the pair is controllable / observable); `place(A, B, P)` assigns the closed-loop eigenvalues to `P`. The closed-loop `eig(A − BK)` matches the requested poles.

### Steady-state Kalman filter + LQG separation  (`examples/control/kalman_lqg.m`)

`kalman_L` returns the continuous Kalman gain via the LQR/Kalman duality `L = lqr(A', C', G·Qn·G', Rn)'`, stabilising the estimator `A − LC`.

```matlab
A = [1, 1; 0, 0-2];     % open-loop unstable
G = [1, 0; 0, 1];
C = [1, 0];             % measure first state only
L = kalman_L(A, G, C, [1 0; 0 1], [1]);
Aest = A - L * C;
fprintf('estimator Hurwitz: %d\n', isstable(Aest));

% LQG separation principle: closed-loop spectrum = LQR poles UNION Kalman poles.
Klqr = lqr(A, B, [1 0; 0 1], [1]);
Lkal = kalman_L(A, B, C, [1], [1]);
disp(real(eig(A - B * Klqr)));   % LQR closed-loop poles
disp(real(eig(A - Lkal * C)));   % Kalman estimator poles
```

The estimator is Hurwitz even for an open-loop unstable plant; `kalmd_L` is the discrete analogue. The separation principle is shown numerically: the LQG controller's closed-loop poles are the union of the LQR feedback poles and the Kalman estimator poles.

### Frequency response and stability margins  (`examples/control/loop_shaping_workflow.m`, `bode_demo.m`)

A full SISO loop-shaping workflow on the type-1 servo `L(s) = 4/(s(s+2))`, using the matrix-argument frequency API.

```matlab
A = [0 1; 0, 0-2]; B = [0; 1]; C = [4, 0]; D = [0];

[mag, phase] = bode_ss(A, B, C, D, [0.5; 1.0; 1.5; 2.0]);

w_dense = 0.1 + 0.005 * (0:399)';
Pm = phase_margin(A, B, C, D, w_dense);
Gm = gain_margin (A, B, C, D, w_dense);
fprintf('phase margin Pm = %.4f deg\n', Pm);   % 51.83 deg

[Ad, Bd] = c2d(A, B, 0.1);                      % discretise for digital impl
y_open = lsim_ss(A, B, C, D, ones(30,1), 0.1);  % time-domain step
```

`bode_ss(A, B, C, D, w)` returns `[mag, phase]` of `H(jw)`; `phase_margin` / `gain_margin` evaluate over a dense grid. `bode_tf(b, a, w)` (in `bode_tf_filter.m`) is the transfer-function-coefficient counterpart that bridges SPT-designed `butter`/`cheby1` filters straight into CST frequency analysis — and agrees with `bode_ss` exactly for equivalent representations.

### Balanced model reduction  (`examples/control/balred_demo.m`, `balreal_demo.m`)

Truncate the states with the smallest Hankel singular values, with a guaranteed `H∞` error bound.

```matlab
H = hsvd(A, B, C);                 % last two HSVs ~ 1e-7 -> truncate
disp(2 * (H(3,1) + H(4,1)));       % H∞ error bound for k=2 truncation

Ar = balred_A(A, B, C, 2);
Br = balred_B(A, B, C, 2);
Cr = balred_C(A, B, C, 2);
disp(isstable(Ar));                % stability preserved
disp(hsvd(Ar, Br, Cr));            % matches top-2 HSVs of the original
```

`hsvd` exposes the Hankel singular values (small ones flag reducible states); `balred_A/B/C` return the k-state truncated balanced realization, preserving stability and the dominant HSVs. `balreal_demo.m` shows the underlying `balreal_T` balancing transform that makes `Wc = Wo = diag(HSV)`.

### Model objects: `tf`, `step`, `bode`, `stepinfo`  (`examples/control/tf_basic.m`, `step_response_siso.m`, `bode_first_order.m`)

The model-object surface lets you work in transfer-function form directly.

```matlab
G  = tf(1, [tau 1]);                 % G(s) = 1/(tau*s + 1)
[y, tout] = step(G, t);              % unit-step response
S  = stepinfo(y2, t2);               % S.RiseTime, S.SettlingTime, S.Overshoot

[mag, phase, wout] = bode(G, w);     % frequency response
bw = bandwidth(G);  g0 = dcgain(G);
[Gm, Pm, Wcg, Wcp] = margin(tf(4, [1 2 0]));   % gain / phase margins
```

`tf(num, den)` builds the model (also `s = tf('s')` for `(s+2)/(s^2+3*s+5)`-style algebra and a third `Ts` arg for discrete `tf`); `step` / `bode` / `margin` / `stepinfo` consume it. The second-order underdamped step gives peak 1.163 / overshoot 16.37%, and `bode` of `1/(0.5s+1)` shows the expected −3 dB corner at `w = 2`. `lqr_double_integrator.m` exercises the 3-return `[K, S, e] = lqr(A, B, Q, R)` and `initial(sys, x0, t)`; `kalman_tracker.m` exercises `[kest, L, P] = kalman(sys, Q, R)`.

Other shipped examples: `lyap_gramian.m` (`lyap` / `dlyap` gramians + residual checks), `h2norm_demo.m` / `h2norm_via_gramian.m` (`norm_h2` two ways), `charac_triad.m` (`isstable` / `damp` / `hsvd`), and `interconnect_demo.m` (`series_ss` / `parallel_ss` / `feedback_ss` / `append_ss`).

## Limitations & carve-outs

- **`c2d(sys, Ts, method)` / `d2c` not yet shipped** — the 3-argument model-object discretiser fails with `unsupported call shape`; use the matrix-arg `[Ad, Bd] = c2d(A, B, Ts)` form instead. `zpk` model objects are also unshipped (`tf` / `ss` do work).
- **No interactive apps** — Control System Designer, Linear System Analyzer, Control System Tuner, Model Reducer, PID Tuner, Compensator Editor, Linearizer are not language features.
- **No Simulink linearization** (`linearize`, `slLinearizer`, `slTuner`, batch trim/linearize).
- **No LPV / LTV runtime simulation** (`lpvss`, `ltvss`, gain-scheduling).
- **No sparse second-order models** (`sparss`, `mechss`) — needs a full sparse linear-algebra stack.
- **No tuning / robust synthesis** — `systune` / `looptune` / `hinfstruct` / `genss` / `realp` / `tunable*`, `hinfsyn` / `mixsyn` (Robust Control Toolbox), MPC entries, and System Identification (`n4sid`, `tfest`, `ssest`) are separate products / multi-month efforts, deferred.
- **No native plotting** — functions return numeric data; `bodeoptions` / `pzoptions` / plot tools are out (visualization is delegated to the user).

## See also

- Roadmap / design: [`../control_toolbox_roadmap.md`](../control_toolbox_roadmap.md)
- Examples directory: `examples/control/`
