% Problem-based equation solving — Optimization Toolbox Tier-5.
% `eqnproblem` builds a system of equations from operator-overloaded
% `lhs == rhs` expressions; `solve` finds x satisfying all of them by
% Levenberg-Marquardt on the residual.  Reuses the Tier-4 expression
% DAG + the Tier-2 `lm_solve` core.  See docs/optim_toolbox_roadmap.md
% §6.
%
% Tier-5 scope: scalar variables (created with no arguments); `solve`
% returns the solution as a column vector in variable-creation order.

% --- 1. linear 2x2 system ----------------------------------------
%   x + y == 3,  2*x - y == 0   →   x = 1, y = 2.
x = optimvar();
y = optimvar();
prob = eqnproblem();
prob.Equations.e1 = x + y == 3;
prob.Equations.e2 = 2*x - y == 0;
sol = solve(prob);
e1 = abs(sol(1) - 1) + abs(sol(2) - 2);
if e1 < 1e-6; disp(1); else; disp(0); end

% --- 2. scalar nonlinear equation --------------------------------
%   x^3 + x == 10   →   x = 2  (8 + 2 = 10).
a = optimvar();
p2 = eqnproblem();
p2.Equations.cubic = a^3 + a == 10;
s2 = solve(p2);
if abs(s2(1) - 2) < 1e-6; disp(1); else; disp(0); end

% --- 3. nonlinear 2x2 system -------------------------------------
%   x + y^2 == 1,  x - y == 1
%   Substituting x = 1 + y gives y + y^2 = 0 → y = 0 (from x0 = 0 the
%   Newton step lands exactly on this root) → x = 1, y = 0.
u = optimvar();
v = optimvar();
p3 = eqnproblem();
p3.Equations.circle = u + v^2 == 1;
p3.Equations.line   = u - v == 1;
s3 = solve(p3);
e3 = abs(s3(1) - 1) + abs(s3(2) - 0);
if e3 < 1e-6; disp(1); else; disp(0); end

% --- 4. residuals are ~0 at the solution -------------------------
r1 = s3(1) + s3(2)*s3(2) - 1;
r2 = s3(1) - s3(2) - 1;
if abs(r1) + abs(r2) < 1e-8; disp(1); else; disp(0); end
