% ode45 / ode23 — initial-value ODE solvers (scalar y).
%
% Tour of the runtime's ODE surface: 2-element tspan with default
% adaptive step + dense output, MATLAB-style odeset (RelTol/AbsTol,
% MaxStep, Refine, Stats), an explicit user-time grid, and a
% backward-time integration.
%
% Model: dy/dt = -2*y + sin(t), y(0) = 1.
% Analytic solution: y(t) = (4*sin(t) - 2*cos(t))/10 + 1.2*exp(-2*t).

f = @(t,y) -2*y + sin(t);
ya_at_10 = (4*sin(10) - 2*cos(10)) / 10 + 1.2/exp(20);

% --- 1. Default 2-element tspan ----------------------------------------
% Adaptive Dormand-Prince 5(4); rtol = 1e-3, atol = 1e-6 (MATLAB
% defaults). Refine = 4 emits four cubic-Hermite sub-points per
% accepted step, so the output looks smooth without re-running the
% solver at intermediate times.

disp('1. ode45 default — 2-element tspan');
[t, y] = ode45(f, [0 10], 1);
disp('  number of output samples:');
disp(length(t));
disp('  y(end) (analytic ≈ -0.0498):');
disp(y(end));

% --- 2. Tight tolerances + Stats ---------------------------------------
% odeset is built MATLAB-style: a struct with named fields.

disp('2. ode45 with tight tolerances and Stats');
opts.RelTol = 1e-9;
opts.AbsTol = 1e-12;
opts.Stats  = 1;          % numeric flag — see runtime/matlab_runtime.h
[t2, y2] = ode45(f, [0 10], 1, opts);
disp('  y(end) (should match analytic to ~1e-9):');
disp(y2(end));
disp('  |error|:');
disp(abs(y2(end) - ya_at_10));

% --- 3. User-supplied output grid --------------------------------------
% Pass tspan with more than two elements to get y at exactly those
% times. The integrator still chooses its own adaptive step; the user
% grid is filled in via Hermite. Refine is ignored in this mode.

disp('3. ode45 on a user-specified time grid');
grid = [0 1 2 3 4 5 6 7 8 9 10];
[t3, y3] = ode45(f, grid, 1);
disp('  output length matches grid length:');
disp(length(t3));
disp('  y at t = 5:');
disp(y3(6));
disp('  y at t = 10:');
disp(y3(end));

% --- 4. Backward-time integration --------------------------------------
% tspan = [t1 t0] with t1 > t0 means integrate backwards from t1 to t0.
% We integrate forward then back to recover y(0) — round-trip error
% should sit close to the relative tolerance.

disp('4. backward-time round-trip');
g = @(t,y) 0 - y;        % dy/dt = -y → y(t) = exp(-t).
[tf, yf] = ode45(g, [0 5], 1);
[tb, yb] = ode45(g, [5 0], yf(end));
disp('  forward y(5) (analytic ≈ 0.00674):');
disp(yf(end));
disp('  backward y(0) recovered from y(5) (should be ≈ 1):');
disp(yb(end));

% --- 5. ode23 with capped step size ------------------------------------
% Bogacki-Shampine 3(2) — lower-order, fewer function evaluations per
% step but smaller useful step size. Refine defaults to 1 for ode23
% (matches MATLAB) so output equals the accepted-step grid unless the
% user overrides it.

disp('5. ode23 with MaxStep cap');
opts2.MaxStep = 0.05;
[t5, y5] = ode23(f, [0 10], 1, opts2);
disp('  point count (capped step → many points):');
disp(length(t5));
disp('  y(end):');
disp(y5(end));

% --- 6. 3-return form: solver-stats struct -----------------------------
% MATLAB's `[t, y, stats] = ode45(...)` packs the step counts into a
% struct with fields nsteps / nfailed / nfevals.

disp('6. 3-return [t, y, stats]');
[t6, y6, stats] = ode45(f, [0 10], 1);
disp('  nsteps:');
disp(stats.nsteps);
disp('  nfailed:');
disp(stats.nfailed);
disp('  nfevals (~6 * nsteps for RK45):');
disp(stats.nfevals);

% --- 7. Systems of ODEs (vector y) -------------------------------------
% Linear oscillator: dy/dt = [-y(2); y(1)] with y(0) = [1; 0]. Closed
% form is y(t) = [cos(t); sin(t)]. The output Y is N rows × D cols
% (MATLAB convention: Y(i, :) is the state at t(i)).

disp('7. vector y — linear oscillator');
y0vec = [1; 0];
[t7, y7] = ode45(@(t,yy) [(0 - yy(2)); yy(1)], [0 6.283185307179586], y0vec);
disp('  number of samples:');
disp(length(t7));
disp('  initial state y(1, :):');
disp(y7(1, 1));
disp(y7(1, 2));
disp('  final state y(end, :) — should be ≈ [1; 0] (cos/sin of 2π):');
disp(y7(end, 1));
disp(y7(end, 2));

% --- 8. Stiff system: ode23s (Rosenbrock 2(3)) -------------------------
% Robertson reaction kinetics — three-component system with rate
% constants 0.04, 1e4, 3e7 (eight orders of magnitude apart). ode45
% would either take thousands of micro-steps or diverge; ode23s
% absorbs the stiff modes via the implicit (I - h*d*J) factor and
% finishes in tens of steps with mass conservation y1+y2+y3 = 1.

disp('8. ode23s — Robertson stiff system');
fr = @(t,y) [(0 - 0.04*y(1) + 1e4*y(2)*y(3));
             (0.04*y(1) - 1e4*y(2)*y(3) - 3e7*y(2)*y(2));
             (3e7*y(2)*y(2))];
[t8, y8, s8] = ode23s(fr, [0 1], [1; 0; 0]);
disp('  number of accepted steps:');
disp(s8.nsteps);
disp('  conservation y1+y2+y3 at t=1 (should be 1):');
disp(y8(end, 1) + y8(end, 2) + y8(end, 3));

% --- 9. Events: ball drop ----------------------------------------------
% dy/dt = -10 (constant downward velocity), y(0) = 100. Event "y == 0"
% on a falling crossing halts integration. Closed form: t* = 10.
%
% Non-MATLAB call shape: ode_events(@f, tspan, y0, @evt). The event
% function returns a 3x1 column [value; isterminal; direction].

disp('9. ode_events — ball drop with terminal event');
fb  = @(t,y) -10;
evt = @(t,y) [y; 1; -1];
[t9, y9, te, ye, ie] = ode_events(fb, [0 20], 100, evt);
disp('  number of events captured:');
disp(numel(te));
disp('  event time (analytic = 10):');
disp(te(1));
disp('  integration halted at t (= event time):');
disp(t9(end));
