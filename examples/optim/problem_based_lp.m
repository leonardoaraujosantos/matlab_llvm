% Problem-based optimisation (Optimization Toolbox Tier-4).
%
% The problem-based workflow builds the objective and constraints
% from operator-overloaded optimisation variables, collects them in
% an `optimproblem`, and hands the whole problem to `solve`, which
% reduces the expression DAG and dispatches automatically (LP / QP /
% MILP / nonlinear).
%
% Tier-4 scope: scalar optimisation variables created with
% `optimvar()` / `optimintvar()`; `solve` returns the solution as a
% column vector in variable-creation order.

% --- 1. a linear program ----------------------------------------
%   minimise  -x - 2*y
%   s.t.       x + y <= 4,  x + 3*y <= 6,  x, y >= 0
%   Optimum (3, 1) with objective -5.
x = optimvar();
y = optimvar();
prob = optimproblem();
prob.Objective = -x - 2*y;
prob.Constraints.c1 = x + y <= 4;
prob.Constraints.c2 = x + 3*y <= 6;
prob.Constraints.c3 = x >= 0;
prob.Constraints.c4 = y >= 0;
sol = solve(prob);
fprintf('LP solution:  x = %.4f, y = %.4f\n', sol(1), sol(2));

% --- 2. a constrained quadratic ----------------------------------
%   minimise (a-2)^2 + (b-2)^2  s.t.  a + b <= 2  →  a = b = 1.
a = optimvar();
b = optimvar();
qp = optimproblem();
qp.Objective = (a - 2)^2 + (b - 2)^2;
qp.Constraints.line = a + b <= 2;
qsol = solve(qp);
fprintf('QP solution:  a = %.4f, b = %.4f\n', qsol(1), qsol(2));

% --- 3. a mixed-integer program ----------------------------------
%   maximise i + j  s.t.  i + 2*j <= 7, i <= 3, i,j >= 0, integer
%   →  i = 3, j = 2.
i = optimintvar();
j = optimintvar();
mp = optimproblem();
mp.Objective = -i - j;
mp.Constraints.c1 = i + 2*j <= 7;
mp.Constraints.c2 = i <= 3;
mp.Constraints.c3 = i >= 0;
mp.Constraints.c4 = j >= 0;
msol = solve(mp);
fprintf('MILP solution: i = %.0f, j = %.0f\n', msol(1), msol(2));
