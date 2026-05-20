% MPC Tier-4 §5.4 — mpcActiveSetSolver standalone QP solve.
% Solve a hand-built QP and compare against a known optimum.
%   min ½·x'·H·x + f'·x  s.t.  A·x ≤ b
% Use H = I, f = -[1; 1] — unconstrained optimum is [1; 1].
% Add A = [1 1], b = 1 — cuts off the unconstrained optimum;
% constrained optimum is [0.5; 0.5] (the projection of [1; 1] onto
% the line x1 + x2 = 1).

H = [1, 0; 0, 1];
f = [0-1; 0-1];

% Unconstrained — pass empty A/b.
A_empty = zeros(0, 2);
b_empty = zeros(0, 1);
x_unc = mpcActiveSetSolver(H, f, A_empty, b_empty);
fprintf('unconstrained x = [%.4f, %.4f]\n', x_unc(1, 1), x_unc(2, 1));

% Constrained.
A = [1, 1];
b = [1];
x_con = mpcActiveSetSolver(H, f, A, b);
fprintf('constrained  x = [%.4f, %.4f]\n', x_con(1, 1), x_con(2, 1));

% Verify A·x = b at the binding constraint (active set).
ax = A * x_con;
fprintf('A·x at optimum = %.4f (must be 1.0)\n', ax(1, 1));
