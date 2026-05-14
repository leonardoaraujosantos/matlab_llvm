% fmincon — Optimization Toolbox Tier-2.  General constrained
% nonlinear minimisation via an augmented-Lagrangian method with a
% bound-projected BFGS inner solver.  See docs/optim_toolbox_roadmap.md.
%
%   x = fmincon(@fun, x0, A, b, Aeq, beq, lb, ub, @nonlcon)
%
% Objective and nonlcon handles take a vector argument and must index
% it so Sema types the parameter as an array.  Multi-component checks
% use a summed absolute error against one threshold.

% --- 1. Rosenbrock minimised inside the unit disk -----------------
%   min 100*(x2-x1^2)^2 + (1-x1)^2  s.t.  x1^2 + x2^2 <= 1
%   The unconstrained minimiser (1,1) is outside the disk, so the
%   solution sits on the boundary at the MATLAB-documented point
%   x ~= [0.7864, 0.6177].
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
           (1 - x(1))*(1 - x(1));
disk = @(x) [x(1)*x(1) + x(2)*x(2) - 1];
r = fmincon(ros, [0; 0], [], [], [], [], [], [], disk);
e1 = abs(r(1) - 0.7864) + abs(r(2) - 0.6177);
if e1 < 3e-2; disp(1); else; disp(0); end

% --- 2. Solution is feasible (inside the disk) --------------------
g = r(1)*r(1) + r(2)*r(2) - 1;
if g < 1e-3; disp(1); else; disp(0); end

% --- 3. Bound-constrained quadratic (8-arg form) ------------------
%   min (x1-5)^2 + (x2-5)^2  s.t.  0 <= x <= 3   → solution [3; 3].
q = fmincon(@(x) (x(1) - 5)*(x(1) - 5) + (x(2) - 5)*(x(2) - 5), ...
            [0; 0], [], [], [], [], [0; 0], [3; 3]);
e3 = abs(q(1) - 3) + abs(q(2) - 3);
if e3 < 1e-3; disp(1); else; disp(0); end

% --- 4. Linear-inequality constraint (4-arg form) -----------------
%   min (x1-2)^2 + (x2-2)^2  s.t.  x1 + x2 <= 1.5
%   Closest point on the line to (2,2) is x1 = x2 = 0.75.
p = fmincon(@(x) (x(1) - 2)*(x(1) - 2) + (x(2) - 2)*(x(2) - 2), ...
            [0; 0], [1, 1], 1.5);
e4 = abs(p(1) - 0.75) + abs(p(2) - 0.75);
if e4 < 1e-2; disp(1); else; disp(0); end

% --- 5. Linear equality constraint (6-arg form) -------------------
%   min (x1-2)^2 + (x2-2)^2  s.t.  x1 + x2 = 1   → x1 = x2 = 0.5.
s = fmincon(@(x) (x(1) - 2)*(x(1) - 2) + (x(2) - 2)*(x(2) - 2), ...
            [0; 0], [], [], [1, 1], 1);
e5 = abs(s(1) - 0.5) + abs(s(2) - 0.5);
if e5 < 1e-2; disp(1); else; disp(0); end
