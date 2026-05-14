% fminsearch — Optimization Toolbox Tier-1.3.  Derivative-free N-D
% minimisation via the Nelder-Mead downhill simplex.  See
% docs/optim_toolbox_roadmap.md.
%
%   x = fminsearch(@fn, x0)   % x0 a column vector
%
% The objective must index its vector argument so Sema types the
% anonymous-function parameter as an array (outlined ABI: f64(ptr)).
% Multi-component checks use a summed absolute error against one
% threshold (the LLVM lane does not lower `&&` as a value).

% --- 1. Rosenbrock banana from the classic start point -----------
%   f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2,   minimiser [1; 1].
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
           (1 - x(1))*(1 - x(1));
r = fminsearch(ros, [-1.2; 1]);
e1 = abs(r(1) - 1) + abs(r(2) - 1);
if e1 < 2e-3; disp(1); else; disp(0); end

% --- 2. Shifted quadratic bowl in 3-D ----------------------------
%   f(x) = (x1-3)^2 + (x2+1)^2 + (x3-2)^2,   minimiser [3; -1; 2].
bowl = @(x) (x(1) - 3)*(x(1) - 3) + (x(2) + 1)*(x(2) + 1) + ...
            (x(3) - 2)*(x(3) - 2);
q = fminsearch(bowl, [0; 0; 0]);
e2 = abs(q(1) - 3) + abs(q(2) + 1) + abs(q(3) - 2);
if e2 < 3e-3; disp(1); else; disp(0); end

% --- 3. Objective value at the Rosenbrock solution is ~0 ----------
%   Recomputed inline (the LLVM lane does not lower a direct
%   call_indirect on an anon handle that was also passed as an arg).
fval = 100*(r(2) - r(1)*r(1))*(r(2) - r(1)*r(1)) + (1 - r(1))*(1 - r(1));
if fval < 1e-6; disp(1); else; disp(0); end
