% linprog — linear programming (Optimization Toolbox Tier-1).
%
% linprog minimises f'*x subject to A*x <= b, Aeq*x = beq and bounds,
% via a dense 2-phase tableau simplex.  Both the 3-argument form
% linprog(f, A, b) and the full 7-argument form are supported.

% A small "diet"-style problem:
%   minimise  -x1 - 2*x2
%   s.t.       x1 +   x2 <= 4
%              x1 + 3*x2 <= 6
%              x >= 0
% Enumerating the vertices, (3,1) gives the objective -5 — the optimum.
f = [-1; -2];
A = [1, 1; 1, 3];
b = [4; 6];
lb = [0; 0];

% --- 1. 3-argument form (default lower bound 0) ------------------
x = linprog(f, A, b);
fprintf('3-arg linprog:  x = [%.4f, %.4f]\n', x(1), x(2));

% --- 2. full 7-argument form with explicit bounds ----------------
ub = [10; 10];
x2 = linprog(f, A, b, [], [], lb, ub);
fprintf('7-arg linprog:  x = [%.4f, %.4f]\n', x2(1), x2(2));

% --- 3. objective value at the optimum ---------------------------
obj = f(1)*x(1) + f(2)*x(2);
fprintf('optimal objective: %.4f\n', obj);

% --- 4. an equality constraint: x1 + x2 = 3 ----------------------
Aeq = [1, 1];
beq = 3;
x3 = linprog(f, A, b, Aeq, beq, lb, ub);
fprintf('with x1+x2=3:   x = [%.4f, %.4f]\n', x3(1), x3(2));
