% Problem-based linear programming — Optimization Toolbox Tier-4.
% Exercises the full problem-based workflow: optimvar -> operator-
% overloaded expressions -> optimproblem -> prob.Objective /
% prob.Constraints -> solve.  See docs/optim_toolbox_roadmap.md §5.
%
% Tier-4 scope: scalar optimisation variables (created with no
% arguments); `solve` returns the solution as a column vector in
% variable-creation order.
%
%   minimise  -x - 2*y
%   s.t.       x +   y <= 4
%              x + 3*y <= 6
%              x, y >= 0
% Vertex (3,1) gives the objective -5 — the optimum.

x = optimvar();
y = optimvar();
prob = optimproblem();
prob.Objective = -x - 2*y;
prob.Constraints.c1 = x + y <= 4;
prob.Constraints.c2 = x + 3*y <= 6;
prob.Constraints.c3 = x >= 0;
prob.Constraints.c4 = y >= 0;
sol = solve(prob);

% --- 1. the optimum is x = 3, y = 1 ------------------------------
e1 = abs(sol(1) - 3) + abs(sol(2) - 1);
if e1 < 1e-4; disp(1); else; disp(0); end

% --- 2. the objective value at the optimum is -5 -----------------
obj = -sol(1) - 2*sol(2);
if abs(obj + 5) < 1e-4; disp(1); else; disp(0); end

% --- 3. both linear inequality constraints are respected ---------
c1v = sol(1) + sol(2) - 4;
c2v = sol(1) + 3*sol(2) - 6;
if c1v < 1e-4; disp(1); else; disp(0); end
if c2v < 1e-4; disp(1); else; disp(0); end

% --- 4. a maximisation problem (prob.Maximize = 1) ---------------
%   maximise x + y  s.t.  x + y <= 4, x,y >= 0  → any vertex on the
%   line; the solver lands on x + y = 4.
a = optimvar();
b = optimvar();
p2 = optimproblem();
p2.Maximize = 1;
p2.Objective = a + b;
p2.Constraints.c1 = a + b <= 4;
p2.Constraints.c2 = a >= 0;
p2.Constraints.c3 = b >= 0;
s2 = solve(p2);
if abs(s2(1) + s2(2) - 4) < 1e-3; disp(1); else; disp(0); end
