% lsqnonneg — non-negative least squares (Optimization Toolbox
% Tier-1).
%
% lsqnonneg minimises ||C*x - d||^2 subject to x >= 0, using the
% Lawson-Hanson active-set algorithm: columns move between the
% passive (free) and active (pinned-at-zero) sets, with each major
% step solving the unconstrained least-squares sub-problem on the
% passive columns.

% --- 1. constraint inactive: NNLS == ordinary least squares ------
%   C'C = [2 1; 1 2], C'd = [5; 6]  →  x = [4/3; 7/3], both positive.
C1 = [1, 0; 0, 1; 1, 1];
d1 = [1; 2; 4];
x1 = lsqnonneg(C1, d1);
fprintf('unconstrained-equivalent: x = [%.4f, %.4f]\n', x1(1), x1(2));

% --- 2. constraint active on one variable ------------------------
%   Same C, d = [-1; 2; 1].  Plain least squares wants x1 = -1 < 0;
%   NNLS pins x1 = 0 and fits x2 alone  →  x2 = 1.5.
d2 = [-1; 2; 1];
x2 = lsqnonneg(C1, d2);
fprintf('one bound active:         x = [%.4f, %.4f]\n', x2(1), x2(2));

% --- 3. residual norm at the constrained solution ---------------
r = C1 * x2 - d2;
fprintf('residual norm^2:          %.4f\n', r(1)*r(1) + r(2)*r(2) + r(3)*r(3));
