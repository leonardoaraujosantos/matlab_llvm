% fzero — scalar root finding (Optimization Toolbox Tier-1).
%
% fzero brackets a sign change around the initial guess and then
% drives it to a root with Brent's method (van Wijngaarden-Dekker-
% Brent): a hybrid of inverse-quadratic interpolation, the secant
% rule, and bisection.

% --- 1. cos(x) = x  →  the Dottie number ~ 0.739085 --------------
r1 = fzero(@(x) cos(x) - x, 0.5);
fprintf('cos(x) = x         root: %.6f\n', r1);

% --- 2. a cubic with a real root  x^3 - x - 2 = 0 ~ 1.521380 -----
r2 = fzero(@(x) x*x*x - x - 2, 0);
fprintf('x^3 - x - 2 = 0    root: %.6f\n', r2);

% --- 3. bracket form: fzero(@fn, [a b]) requires f(a)*f(b) <= 0 --
%   sin(x) on [3, 4] isolates the root at pi.
r3 = fzero(@(x) sin(x), [3, 4]);
fprintf('sin(x) = 0 on [3,4] root: %.6f   (pi)\n', r3);

% --- 4. residual check -------------------------------------------
res = cos(r1) - r1;
fprintf('residual at root 1: %.2e\n', abs(res));
