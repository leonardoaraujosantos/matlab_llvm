% Problem-based equation solving (Optimization Toolbox Tier-5).
%
% `eqnproblem` is the equation-solving counterpart of `optimproblem`:
% each `lhs == rhs` expression contributes a residual, and `solve`
% finds x satisfying all of them by Levenberg-Marquardt on the
% residual vector.
%
% Tier-5 scope: scalar optimisation variables; `solve` returns the
% solution as a column vector in variable-creation order.

% --- 1. a linear 2x2 system --------------------------------------
%   x + y == 3,  2*x - y == 0   →   x = 1, y = 2.
x = optimvar();
y = optimvar();
prob = eqnproblem();
prob.Equations.e1 = x + y == 3;
prob.Equations.e2 = 2*x - y == 0;
sol = solve(prob);
fprintf('linear 2x2:    x = %.4f, y = %.4f\n', sol(1), sol(2));

% --- 2. a scalar nonlinear equation ------------------------------
%   x^3 + x == 10   →   x = 2.
a = optimvar();
p2 = eqnproblem();
p2.Equations.cubic = a^3 + a == 10;
s2 = solve(p2);
fprintf('scalar cubic:  x = %.4f\n', s2(1));

% --- 3. a nonlinear 2x2 system -----------------------------------
%   u + v^2 == 1,  u - v == 1   →   u = 1, v = 0.
u = optimvar();
v = optimvar();
p3 = eqnproblem();
p3.Equations.curve = u + v^2 == 1;
p3.Equations.line  = u - v == 1;
s3 = solve(p3);
fprintf('nonlinear 2x2: u = %.4f, v = %.4f\n', s3(1), s3(2));
