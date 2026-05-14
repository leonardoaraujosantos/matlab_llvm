% intlinprog — mixed-integer linear programming (Optimization
% Toolbox Tier-3).
%
% intlinprog solves an LP with some variables restricted to integers,
% via depth-first branch-and-bound over the dense simplex: each node
% solves the LP relaxation under tightened bounds, prunes by the
% incumbent objective, and branches on the most-fractional integer
% variable.
%
%   x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub)

% --- 1. a 0/1 knapsack -------------------------------------------
%   maximise 8*x1 + 11*x2 + 6*x3 + 4*x4   (minimise the negation)
%   s.t.     5*x1 + 7*x2 + 4*x3 + 3*x4 <= 10,   x in {0,1}
%   Best feasible subset is {item2, item4}: value 15, weight 10.
f      = [-8; -11; -6; -4];
intcon = [1; 2; 3; 4];
A      = [5, 7, 4, 3];
b      = 10;
lb     = [0; 0; 0; 0];
ub     = [1; 1; 1; 1];
x = intlinprog(f, intcon, A, b, [], [], lb, ub);
fprintf('knapsack pick:   [%.0f %.0f %.0f %.0f]\n', x(1), x(2), x(3), x(4));
fprintf('  total value:   %.0f\n', -(f(1)*x(1) + f(2)*x(2) + f(3)*x(3) + f(4)*x(4)));
fprintf('  total weight:  %.0f  (cap 10)\n', ...
        5*x(1) + 7*x(2) + 4*x(3) + 3*x(4));

% --- 2. a 3x3 assignment problem ---------------------------------
%   Cost C = [4 1 3; 2 0 5; 3 2 2]; each row + each column sums to 1.
%   The cheapest assignment is 1->2, 2->1, 3->3 with cost 5.
fc = [4; 1; 3; 2; 0; 5; 3; 2; 2];
ic = [1; 2; 3; 4; 5; 6; 7; 8; 9];
Aeq = [1 1 1 0 0 0 0 0 0; 0 0 0 1 1 1 0 0 0; 0 0 0 0 0 0 1 1 1; ...
       1 0 0 1 0 0 1 0 0; 0 1 0 0 1 0 0 1 0; 0 0 1 0 0 1 0 0 1];
beq = [1; 1; 1; 1; 1; 1];
xa = intlinprog(fc, ic, [], [], Aeq, beq, zeros(9, 1), ones(9, 1));
cost = 0;
for k = 1:9
    cost = cost + fc(k) * xa(k);
end
fprintf('assignment cost: %.0f\n', cost);
