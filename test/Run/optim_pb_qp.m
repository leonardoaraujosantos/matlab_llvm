% Problem-based quadratic programming — Optimization Toolbox Tier-4.
% A quadratic objective built from operator-overloaded expressions is
% reduced to a DAG and solved through the augmented-Lagrangian core.
% See docs/optim_toolbox_roadmap.md §5.

% --- 1. unconstrained quadratic bowl -----------------------------
%   minimise (x-3)^2 + (y+1)^2  → minimiser x = 3, y = -1.
x = optimvar();
y = optimvar();
prob = optimproblem();
prob.Objective = (x - 3)^2 + (y + 1)^2;
sol = solve(prob);
e1 = abs(sol(1) - 3) + abs(sol(2) + 1);
if e1 < 1e-3; disp(1); else; disp(0); end

% --- 2. constrained quadratic ------------------------------------
%   minimise (x-2)^2 + (y-2)^2  s.t.  x + y <= 2
%   The unconstrained optimum (2,2) violates the line; the closest
%   feasible point is x = y = 1.
a = optimvar();
b = optimvar();
p2 = optimproblem();
p2.Objective = (a - 2)^2 + (b - 2)^2;
p2.Constraints.c1 = a + b <= 2;
s2 = solve(p2);
e2 = abs(s2(1) - 1) + abs(s2(2) - 1);
if e2 < 1e-2; disp(1); else; disp(0); end

% --- 3. the constraint is active at the solution -----------------
if abs(s2(1) + s2(2) - 2) < 1e-2; disp(1); else; disp(0); end

% --- 4. cross term: minimise x^2 + x*y + y^2 - 3*x ---------------
%   Stationary point: 2x + y - 3 = 0, x + 2y = 0  → x = 2, y = -1.
c = optimvar();
d = optimvar();
p3 = optimproblem();
p3.Objective = c*c + c*d + d*d - 3*c;
s3 = solve(p3);
e4 = abs(s3(1) - 2) + abs(s3(2) + 1);
if e4 < 1e-2; disp(1); else; disp(0); end
