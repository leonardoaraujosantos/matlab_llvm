% lsqlin — Optimization Toolbox Tier-2.  Constrained linear least
% squares: minimise 1/2 ||C*x - d||^2 subject to linear constraints,
% routed through the augmented-Lagrangian core.  See
% docs/optim_toolbox_roadmap.md.
%
%   x = lsqlin(C, d, A, b, Aeq, beq, lb, ub)

C = [1, 0; 0, 1; 1, 1];
d = [1; 2; 4];

% --- 1. Bounds non-binding: lsqlin == ordinary least squares ------
%   Normal equations C'C x = C'd give x = [4/3; 7/3], both inside
%   the box [0, 10], so the bounds do not bind.
x1 = lsqlin(C, d, [], [], [], [], [0; 0], [10; 10]);
e1 = abs(x1(1) - 1.333333333333333) + abs(x1(2) - 2.333333333333333);
if e1 < 1e-3; disp(1); else; disp(0); end

% --- 2. Inequality constraint becomes active (4-arg form) ---------
%   minimise 1/2||Cx-d||^2  s.t.  x1 + x2 <= 2.  The unconstrained
%   optimum sums to 11/3 > 2, so the constraint binds; the KKT
%   solution is x = [0.5; 1.5].
x2 = lsqlin(C, d, [1, 1], 2);
e2 = abs(x2(1) - 0.5) + abs(x2(2) - 1.5);
if e2 < 1e-2; disp(1); else; disp(0); end

% --- 3. Constrained solution still satisfies x1 + x2 <= 2 ---------
if x2(1) + x2(2) - 2 < 1e-3; disp(1); else; disp(0); end

% --- 4. Equality constraint (6-arg form) --------------------------
%   minimise 1/2||Cx-d||^2  s.t.  x1 + x2 = 2  → x = [0.5; 1.5].
x3 = lsqlin(C, d, [], [], [1, 1], 2);
e4 = abs(x3(1) - 0.5) + abs(x3(2) - 1.5);
if e4 < 1e-2; disp(1); else; disp(0); end
