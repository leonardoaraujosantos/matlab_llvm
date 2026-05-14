% fminimax / fgoalattain — multiobjective optimisation (Optimization
% Toolbox Tier-3).
%
% fminimax minimises max_i F_i(x); fgoalattain minimises a slack
% gamma such that F_i(x) - weight_i*gamma <= goal_i.  Both use the
% epigraph reformulation: with the augmented variable z = [x; gamma],
% minimise gamma subject to the per-objective inequalities, solved
% through the augmented-Lagrangian core.

% --- 1. fminimax — three paraboloids -----------------------------
%   F = [x1^2 + x2^2; (x1-2)^2 + x2^2; x1^2 + (x2-2)^2]
%   By symmetry the minimax point is x = [1; 1], where all three
%   objectives equal 2.
fun = @(x) [x(1)*x(1) + x(2)*x(2); ...
            (x(1) - 2)*(x(1) - 2) + x(2)*x(2); ...
            x(1)*x(1) + (x(2) - 2)*(x(2) - 2)];
x = fminimax(fun, [0; 0]);
fprintf('fminimax point: x = [%.4f, %.4f]\n', x(1), x(2));
F = fun(x);
fprintf('  objectives:   [%.4f, %.4f, %.4f]  (balanced)\n', F(1), F(2), F(3));

% --- 2. fgoalattain — coupled goal attainment --------------------
%   F(x) = [x1; x2], goal = [0; 0], weight = [1; 2]
%   s.t.  x1 + x2 >= 3   (written [-1 -1] x <= -3)
%   The attainment solution is x = [1; 2] with gamma = 1.
g = fgoalattain(@(v) [v(1); v(2)], [0; 0], [0; 0], [1; 2], [-1, -1], -3);
fprintf('fgoalattain point: x = [%.4f, %.4f]\n', g(1), g(2));
