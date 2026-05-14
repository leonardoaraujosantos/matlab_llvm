% fgoalattain — Optimization Toolbox Tier-3.  Multiobjective goal
% attainment: minimise gamma s.t. F_i(x) - weight_i*gamma <= goal_i,
% via the same epigraph reformulation as fminimax.  See
% docs/optim_toolbox_roadmap.md §4.
%
%   x = fgoalattain(@fun, x0, goal, weight, A, b, Aeq, beq, lb, ub)

% --- 1. linear objectives with a coupling linear constraint ------
%   F(x) = [x1; x2], goal = [0; 0], weight = [1; 2].
%   Constraint x1 + x2 >= 3, written as [-1 -1] x <= -3.
%   Minimise gamma s.t. x1 <= gamma, x2 <= 2*gamma, x1 + x2 >= 3.
%   Pushing x1 = gamma, x2 = 2*gamma gives 3*gamma >= 3 → gamma = 1,
%   so the attainment solution is x = [1; 2].
fun = @(x) [x(1); x(2)];
goal = [0; 0];
weight = [1; 2];
A = [-1, -1];
b = -3;
x = fgoalattain(fun, [0; 0], goal, weight, A, b);
e1 = abs(x(1) - 1) + abs(x(2) - 2);
if e1 < 1e-2; disp(1); else; disp(0); end

% --- 2. the coupling constraint is active ------------------------
if abs(x(1) + x(2) - 3) < 1e-2; disp(1); else; disp(0); end

% --- 3. both goals are attained with the same factor -------------
%   F_i(x) - weight_i*gamma should equal goal_i; with gamma = 1 that
%   means x1 - 1 ~ 0 and x2 - 2 ~ 0.
a1 = x(1) - 1;
a2 = x(2) - 2;
if abs(a1) + abs(a2) < 1e-2; disp(1); else; disp(0); end

% --- 4. bounded goal attainment (10-arg form) --------------------
%   F(x) = [x1; x2], goal = [1; 1], weight = [1; 1], 0 <= x <= 10.
%   Minimise gamma s.t. x1 <= 1 + gamma, x2 <= 1 + gamma; pushing x
%   to its lower bound 0 gives 0 <= 1 + gamma → gamma = -1, x = [0;0].
y = fgoalattain(@(x) [x(1); x(2)], [3; 3], [1; 1], [1; 1], ...
                [], [], [], [], [0; 0], [10; 10]);
e4 = abs(y(1)) + abs(y(2));
if e4 < 1e-2; disp(1); else; disp(0); end
