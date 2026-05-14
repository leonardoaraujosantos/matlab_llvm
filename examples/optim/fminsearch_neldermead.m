% fminsearch — derivative-free N-D minimisation (Optimization
% Toolbox Tier-1).
%
% fminsearch is the Nelder-Mead downhill simplex: it never evaluates
% a gradient, instead reflecting / expanding / contracting / shrinking
% a simplex of n+1 vertices.  Good for noisy or non-smooth objectives.

% --- 1. the Rosenbrock banana ------------------------------------
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
           (1 - x(1))*(1 - x(1));
r = fminsearch(ros, [-1.2; 1]);
fprintf('Rosenbrock minimiser: [%.4f, %.4f]\n', r(1), r(2));

% --- 2. a 3-D quadratic bowl, minimiser [3; -1; 2] ---------------
bowl = @(x) (x(1) - 3)*(x(1) - 3) + (x(2) + 1)*(x(2) + 1) + ...
            (x(3) - 2)*(x(3) - 2);
q = fminsearch(bowl, [0; 0; 0]);
fprintf('3-D bowl minimiser:   [%.4f, %.4f, %.4f]\n', q(1), q(2), q(3));

% --- 3. objective value at the Rosenbrock solution ---------------
fval = 100*(r(2) - r(1)*r(1))*(r(2) - r(1)*r(1)) + (1 - r(1))*(1 - r(1));
fprintf('objective at minimum: %.2e\n', fval);
