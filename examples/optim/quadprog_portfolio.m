% quadprog — convex quadratic programming (Optimization Toolbox
% Tier-2).
%
% quadprog minimises 1/2 x'Hx + f'x subject to linear constraints
% and bounds.  matlab_llvm routes it through the augmented-Lagrangian
% core with the analytic quadratic objective.
%
%   x = quadprog(H, f, A, b, Aeq, beq, lb, ub)

% --- 1. minimum-variance portfolio -------------------------------
%   minimise  x' * Sigma * x   (so H = 2*Sigma)
%   s.t.      x1 + x2 == 1,  x >= 0
%   With Sigma = [4 1; 1 2] the analytic optimum is w = [0.25; 0.75].
H   = [8, 2; 2, 4];
f   = [0; 0];
Aeq = [1, 1];
beq = 1;
lb  = [0; 0];
w = quadprog(H, f, [], [], Aeq, beq, lb, []);
fprintf('portfolio weights: [%.4f, %.4f]   (sum = %.4f)\n', ...
        w(1), w(2), w(1) + w(2));

% --- 2. an equality-constrained QP with a verifiable optimum -----
%   min x1^2 - 2x1 + x2^2 - 6x2  s.t.  x1 + x2 == 3
%   Lagrange conditions give x = [0.5; 2.5].
x2 = quadprog([2, 0; 0, 2], [-2; -6], [], [], [1, 1], 3, [], []);
fprintf('equality-constrained QP: x = [%.4f, %.4f]\n', x2(1), x2(2));

% --- 3. a bound-constrained QP (diagonal H decouples) ------------
%   same H, f; bounds 0 <= x <= 1.  Unconstrained min [1; 3] is
%   clamped by the upper bound to [1; 1].
x3 = quadprog([2, 0; 0, 2], [-2; -6], [], [], [], [], [0; 0], [1; 1]);
fprintf('bound-constrained QP:    x = [%.4f, %.4f]\n', x3(1), x3(2));
