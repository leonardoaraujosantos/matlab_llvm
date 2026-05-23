% Curve Fitting Toolbox Tier-4 — interpolant fit types.
% linearinterp / splineinterp / pchipinterp / nearestinterp build a cfit whose
% feval routes through the interpolation kernel; all pass through the data.
x = (0:8)';
y = [0; 1; 0; -1; 0; 1; 0; -1; 0];
fl = fit(x, y, 'linearinterp');
fs = fit(x, y, 'splineinterp');
fp = fit(x, y, 'pchipinterp');
fn = fit(x, y, 'nearestinterp');
fprintf('linear  at 0.5 = %.4f\n', fl(0.5));   % midpoint of 0,1
fprintf('linear  at 4   = %.4f\n', fl(4));      % knot -> 0
fprintf('spline  at 4   = %.4f\n', fs(4));      % knot -> 0
fprintf('pchip   at 4   = %.4f\n', fp(4));      % knot -> 0
fprintf('nearest at 0.4 = %.4f\n', fn(0.4));    % rounds to x=0 -> 0
fprintf('nearest at 0.6 = %.4f\n', fn(0.6));    % rounds to x=1 -> 1
