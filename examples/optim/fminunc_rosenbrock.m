% fminunc — unconstrained N-D minimisation (Optimization Toolbox
% Tier-1).
%
% fminunc uses BFGS quasi-Newton: it maintains an approximate inverse
% Hessian, takes the search direction p = -H*g, and globalises with a
% backtracking Armijo line search.  The gradient is obtained by
% forward finite differences when none is supplied.
%
% The objective handle takes a column vector and must index it, so
% Sema types the parameter as an array.

% --- 1. the Rosenbrock banana, classic start point [-1.2; 1] -----
%   f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2,  minimiser [1; 1].
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
           (1 - x(1))*(1 - x(1));
r = fminunc(ros, [-1.2; 1]);
fprintf('Rosenbrock minimiser: [%.4f, %.4f]\n', r(1), r(2));

% --- 2. objective value at the solution (recomputed inline) ------
fval = 100*(r(2) - r(1)*r(1))*(r(2) - r(1)*r(1)) + (1 - r(1))*(1 - r(1));
fprintf('objective at minimum: %.2e\n', fval);

% --- 3. an anisotropic quadratic bowl ----------------------------
%   f(x) = 10*(x1-4)^2 + 0.5*(x2+2)^2,  minimiser [4; -2].
quad = @(x) 10*(x(1) - 4)*(x(1) - 4) + 0.5*(x(2) + 2)*(x(2) + 2);
q = fminunc(quad, [0; 0]);
fprintf('quadratic minimiser:  [%.4f, %.4f]\n', q(1), q(2));
