% fminbnd — bounded 1-D minimisation (Optimization Toolbox Tier-1).
%
% fminbnd minimises a scalar function on a closed interval using
% Brent's localmin: golden-section search refined by parabolic
% interpolation through the three best points.

% --- 1. a simple parabola, minimum at x = 2 ----------------------
m1 = fminbnd(@(x) (x - 2)*(x - 2) + 1, 0, 5);
fprintf('(x-2)^2 + 1 on [0,5]   minimiser: %.6f\n', m1);

% --- 2. sin(x) on [3, 5] — minimum at 3*pi/2 ~ 4.712389 ----------
m2 = fminbnd(@(x) sin(x), 3, 5);
fprintf('sin(x) on [3,5]        minimiser: %.6f\n', m2);

% --- 3. a quartic, minimum at x = 9/4 = 2.25 ---------------------
m3 = fminbnd(@(x) x*x*x*x - 3*x*x*x, 0, 4);
fprintf('x^4 - 3x^3 on [0,4]    minimiser: %.6f\n', m3);

% --- 4. the objective value at the minimum -----------------------
fval = (m1 - 2)*(m1 - 2) + 1;
fprintf('objective at minimum 1: %.6f\n', fval);
