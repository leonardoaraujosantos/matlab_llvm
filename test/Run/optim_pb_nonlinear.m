% Problem-based nonlinear optimisation — Optimization Toolbox Tier-4.
% When the objective or a constraint is not linear or quadratic, the
% problem-based `solve` falls back to evaluating the expression DAG
% numerically inside the augmented-Lagrangian core.  See
% docs/optim_toolbox_roadmap.md §5.

% --- 1. shifted quartic, scalar ----------------------------------
%   minimise (x - 2)^4 + 1  → minimiser x = 2.
x = optimvar();
prob = optimproblem();
prob.Objective = (x - 2)^4 + 1;
sol = solve(prob);
if abs(sol(1) - 2) < 1e-2; disp(1); else; disp(0); end

% --- 2. nonlinear objective + nonlinear constraint ---------------
%   minimise (a - 2)^2 + (b - 2)^2  s.t.  a^2 + b^2 <= 1
%   The unconstrained optimum (2,2) is outside the unit disk; the
%   solution sits on the boundary at a = b = 1/sqrt(2) ~ 0.7071.
a = optimvar();
b = optimvar();
p2 = optimproblem();
p2.Objective = (a - 2)^2 + (b - 2)^2;
p2.Constraints.disk = a^2 + b^2 <= 1;
s2 = solve(p2);
e2 = abs(s2(1) - 0.70710678) + abs(s2(2) - 0.70710678);
if e2 < 2e-2; disp(1); else; disp(0); end

% --- 3. the nonlinear constraint is satisfied --------------------
g = s2(1)*s2(1) + s2(2)*s2(2) - 1;
if g < 1e-2; disp(1); else; disp(0); end

% --- 4. a cubic-objective minimisation with a bound --------------
%   minimise x^3 - 3*x  s.t.  x >= 0  → stationary point x = 1.
c = optimvar();
p3 = optimproblem();
p3.Objective = c^3 - 3*c;
p3.Constraints.pos = c >= 0;
s3 = solve(p3);
if abs(s3(1) - 1) < 1e-2; disp(1); else; disp(0); end
