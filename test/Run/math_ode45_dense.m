% Verify ode45's dense output (Refine = 4): the integrator should emit
% ~4 evenly-spaced sub-points per accepted step using cubic Hermite
% interpolation, matching MATLAB's default. Sample points must lie close
% to the analytic solution of dy/dt = -y, y(0) = 1 → y(t) = exp(-t).

f = @(t,y) 0 - y;
[t, y] = ode45(f, [0 5], 1);

% More than 30 samples — accepted-step count alone for this gentle ODE
% is around 10-15; with Refine=4 we expect 40+ output rows.
if length(t) > 30; disp(1); else; disp(0); end

% First / last samples are exact.
disp(t(1));
disp(y(1));
disp(t(end));

% Pick an interior sample (around the middle of the time grid) and check
% it tracks exp(-t) to within the dense-output interpolation error.
% (exp(-x) written as 1/exp(x) — the Python emitter doesn't yet handle
% unary negation.)
mid = round(length(t) / 2);
err_mid = abs(y(mid) - 1/exp(t(mid)));
if err_mid < 0.01; disp(1); else; disp(0); end

% Final-state error vs analytic.
if abs(y(end) - 1/exp(5)) < 0.005; disp(1); else; disp(0); end
