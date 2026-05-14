% Problem-based mixed-integer linear programming — Optimization
% Toolbox Tier-4.  Integer variables are created with `optimintvar`;
% a problem-based program with a linear objective + linear
% constraints + integer variables routes through the branch-and-bound
% MILP solver.  See docs/optim_toolbox_roadmap.md §5.

% --- 1. small integer program ------------------------------------
%   maximise x + y  (minimise -x - y)
%   s.t.     x + 2*y <= 7,  x <= 3,  x, y >= 0,  x, y integer
%   x is capped at 3; then 2*y <= 4 → y <= 2.  Optimum (3, 2).
x = optimintvar();
y = optimintvar();
prob = optimproblem();
prob.Objective = -x - y;
prob.Constraints.c1 = x + 2*y <= 7;
prob.Constraints.c2 = x <= 3;
prob.Constraints.c3 = x >= 0;
prob.Constraints.c4 = y >= 0;
sol = solve(prob);
e1 = abs(sol(1) - 3) + abs(sol(2) - 2);
if e1 < 1e-4; disp(1); else; disp(0); end

% --- 2. the solution is integer-valued ---------------------------
fr = abs(sol(1) - round(sol(1))) + abs(sol(2) - round(sol(2)));
if fr < 1e-6; disp(1); else; disp(0); end

% --- 3. the constraints are respected ----------------------------
if sol(1) + 2*sol(2) <= 7 + 1e-6; disp(1); else; disp(0); end

% --- 4. a 0/1 knapsack-style pick --------------------------------
%   maximise 6*a + 5*b  s.t.  3*a + 4*b <= 6,  0 <= a,b <= 1, integer
%   {a=1,b=0}=6 weight 3; {a=0,b=1}=5 weight 4; {a=1,b=1} weight 7 > 6.
%   Best feasible pick is a = 1, b = 0.
a = optimintvar();
b = optimintvar();
p2 = optimproblem();
p2.Objective = -6*a - 5*b;
p2.Constraints.w  = 3*a + 4*b <= 6;
p2.Constraints.a0 = a >= 0;
p2.Constraints.a1 = a <= 1;
p2.Constraints.b0 = b >= 0;
p2.Constraints.b1 = b <= 1;
s2 = solve(p2);
e4 = abs(s2(1) - 1) + abs(s2(2) - 0);
if e4 < 1e-4; disp(1); else; disp(0); end
