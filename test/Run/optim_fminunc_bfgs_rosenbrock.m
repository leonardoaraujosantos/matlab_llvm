% fminunc — Optimization Toolbox Tier-1.4.  Unconstrained N-D
% minimisation via BFGS quasi-Newton with a finite-difference
% gradient and a backtracking Armijo line search.  See
% docs/optim_toolbox_roadmap.md.
%
%   x = fminunc(@fn, x0)   % x0 a column vector
%
% Like fminsearch, the objective indexes its vector argument so the
% outlined anonymous-function ABI is f64(ptr).  Multi-component
% checks use a summed absolute error against one threshold.

% --- 1. Rosenbrock banana, classic start [-1.2; 1] ---------------
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
           (1 - x(1))*(1 - x(1));
r = fminunc(ros, [-1.2; 1]);
e1 = abs(r(1) - 1) + abs(r(2) - 1);
if e1 < 2e-3; disp(1); else; disp(0); end

% --- 2. Objective at the BFGS solution is ~0 ---------------------
%   Recomputed inline (the LLVM lane does not lower a direct
%   call_indirect on an anon handle that was also passed as an arg).
fval = 100*(r(2) - r(1)*r(1))*(r(2) - r(1)*r(1)) + (1 - r(1))*(1 - r(1));
if fval < 1e-6; disp(1); else; disp(0); end

% --- 3. Anisotropic quadratic, minimiser [4; -2] -----------------
%   f(x) = 10*(x1-4)^2 + 0.5*(x2+2)^2
quad = @(x) 10*(x(1) - 4)*(x(1) - 4) + 0.5*(x(2) + 2)*(x(2) + 2);
q = fminunc(quad, [0; 0]);
e3 = abs(q(1) - 4) + abs(q(2) + 2);
if e3 < 2e-4; disp(1); else; disp(0); end

% --- 4. Already at the minimum: should not move -------------------
s = fminunc(quad, [4; -2]);
e4 = abs(s(1) - 4) + abs(s(2) + 2);
if e4 < 2e-6; disp(1); else; disp(0); end
