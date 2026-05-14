% fsolve — nonlinear systems of equations (Optimization Toolbox
% Tier-1 scalar / Tier-2 N-D).
%
% fsolve finds x with F(x) = 0.  The scalar form uses Newton's method
% with a finite-difference derivative and a Brent fallback; the N-D
% form uses Levenberg-Marquardt on ||F(x)||^2.  The form is selected
% by the shape of the initial point.

% --- 1. scalar equation: x^2 - 2 = 0  →  sqrt(2) -----------------
s = fsolve(@(x) x*x - 2, 1);
fprintf('scalar  x^2 = 2:       x = %.6f\n', s);

% --- 2. a 2x2 system: unit circle intersect the line x1 = x2 -----
%   F(x) = [x1^2 + x2^2 - 1; x1 - x2] = 0  →  x1 = x2 = 1/sqrt(2).
F = @(x) [x(1)*x(1) + x(2)*x(2) - 1; x(1) - x(2)];
r = fsolve(F, [1; 1]);
fprintf('2x2 circle cap line:  x = [%.6f, %.6f]\n', r(1), r(2));

% --- 3. a 3x3 linear-ish system, root [2; 2; 2] ------------------
G = @(x) [x(1) + x(2) + x(3) - 6; x(1) - x(2); x(2) - x(3)];
g = fsolve(G, [0; 0; 0]);
fprintf('3x3 system:           x = [%.4f, %.4f, %.4f]\n', g(1), g(2), g(3));

% --- 4. residual at the 2x2 solution -----------------------------
res = F(r);
fprintf('2x2 residual norm^2:  %.2e\n', res(1)*res(1) + res(2)*res(2));
