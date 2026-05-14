% fminimax — Optimization Toolbox Tier-3.  Minimises max_i F_i(x) via
% the epigraph reformulation (minimise gamma s.t. F_i(x) - gamma <= 0)
% routed through the augmented-Lagrangian core.  See
% docs/optim_toolbox_roadmap.md §4.
%
%   x = fminimax(@fun, x0, A, b, Aeq, beq, lb, ub)

% --- 1. three paraboloids centred at (0,0), (2,0), (0,2) ---------
%   F = [x1^2 + x2^2; (x1-2)^2 + x2^2; x1^2 + (x2-2)^2]
%   By symmetry x1 = x2 = t; balancing the centre paraboloid against
%   the other two gives t = 1.  At [1;1] all three equal 2, so the
%   minimax point is x = [1; 1].
fun = @(x) [x(1)*x(1) + x(2)*x(2); ...
            (x(1) - 2)*(x(1) - 2) + x(2)*x(2); ...
            x(1)*x(1) + (x(2) - 2)*(x(2) - 2)];
x = fminimax(fun, [0; 0]);
e1 = abs(x(1) - 1) + abs(x(2) - 1);
if e1 < 1e-2; disp(1); else; disp(0); end

% --- 2. the three objectives are balanced at the solution --------
F = fun(x);
spread = abs(F(1) - F(2)) + abs(F(2) - F(3));
if spread < 1e-2; disp(1); else; disp(0); end

% --- 3. minimax value is ~2 --------------------------------------
mx = F(1);
if F(2) > mx; mx = F(2); end
if F(3) > mx; mx = F(3); end
if abs(mx - 2) < 1e-2; disp(1); else; disp(0); end

% --- 4. bounded minimax: cap x1 <= 0.5 ---------------------------
%   With x1 <= 0.5 the centre and the (0,2) paraboloid still want
%   x1 small / x2 ~ 1; the solution stays feasible and near [0.5; t].
xb = fminimax(fun, [0; 0], [], [], [], [], [-10; -10], [0.5; 10]);
if xb(1) <= 0.5 + 1e-4; disp(1); else; disp(0); end
