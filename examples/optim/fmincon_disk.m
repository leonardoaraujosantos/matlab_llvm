% fmincon — general constrained nonlinear minimisation (Optimization
% Toolbox Tier-2).
%
% fmincon minimises a nonlinear objective subject to bounds, linear
% (in)equalities, and nonlinear inequalities.  matlab_llvm backs it
% with an augmented-Lagrangian method over a bound-projected BFGS
% inner solver; one implementation serves every MATLAB `Algorithm`
% choice.
%
%   x = fmincon(@fun, x0, A, b, Aeq, beq, lb, ub, @nonlcon)

% --- 1. Rosenbrock minimised inside the unit disk ----------------
%   min 100*(x2-x1^2)^2 + (1-x1)^2  s.t.  x1^2 + x2^2 <= 1
%   The unconstrained minimiser (1,1) is outside the disk, so the
%   solution sits on the boundary at x ~= [0.7864, 0.6177].
ros  = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
            (1 - x(1))*(1 - x(1));
disk = @(x) [x(1)*x(1) + x(2)*x(2) - 1];   % nonlcon: c(x) <= 0
r = fmincon(ros, [0; 0], [], [], [], [], [], [], disk);
fprintf('disk-constrained Rosenbrock: x = [%.4f, %.4f]\n', r(1), r(2));
fprintf('  on the boundary?  ||x|| = %.4f\n', sqrt(r(1)*r(1) + r(2)*r(2)));

% --- 2. a bound-constrained quadratic (8-argument form) ----------
%   min (x1-5)^2 + (x2-5)^2  s.t.  0 <= x <= 3   → solution [3; 3].
q = fmincon(@(x) (x(1) - 5)*(x(1) - 5) + (x(2) - 5)*(x(2) - 5), ...
            [0; 0], [], [], [], [], [0; 0], [3; 3]);
fprintf('bound-constrained quadratic: x = [%.4f, %.4f]\n', q(1), q(2));

% --- 3. a linear-inequality constraint (4-argument form) ---------
%   min (x1-2)^2 + (x2-2)^2  s.t.  x1 + x2 <= 1.5  →  x1 = x2 = 0.75.
p = fmincon(@(x) (x(1) - 2)*(x(1) - 2) + (x(2) - 2)*(x(2) - 2), ...
            [0; 0], [1, 1], 1.5);
fprintf('linear-constrained quadratic: x = [%.4f, %.4f]\n', p(1), p(2));
