% branin_surrogate.m — Global Optimization Toolbox Tier-4 headline.
%
% Surrogate optimization (User's Guide, Surrogate Optimization chapter).
% surrogateopt is the sample-efficient global solver for *expensive*
% objectives: it fits a radial-basis-function surrogate to the points it
% has evaluated and uses it to choose where to sample next, trading the
% surrogate's predicted value against distance from existing samples
% (exploration).  It finds the global optimum in far fewer evaluations
% than a from-scratch stochastic search would need.
%
%   surrogateopt(fun, lb, ub)   — RBF surrogate + adaptive sampling
%
% Branin's function is the canonical surrogate-/Bayesian-optimization
% benchmark: a smooth-but-multimodal 2-D surface with three equal global
% minima at f* = 0.397887.  surrogateopt recovers it; the RBF coefficient
% solves reuse the shipped `mldivide`, and the adaptive sampling runs over
% the shared seeded PRNG (no external dependency).

% Branin: f = (x2 - b x1^2 + c x1 - r)^2 + s(1-t)cos(x1) + s,
%   b=5.1/(4π²), c=5/π, r=6, s=10, t=1/(8π).  Constants are written
% inline (an anon that *captures* variables and is also passed to a
% solver hits a known compiler gap); x1*x1 sidesteps scalar-matpow.
branin = @(x) (x(2) - 5.1/(4*pi*pi)*x(1)*x(1) + 5/pi*x(1) - 6) ...
              * (x(2) - 5.1/(4*pi*pi)*x(1)*x(1) + 5/pi*x(1) - 6) ...
              + 10*(1 - 1/(8*pi))*cos(x(1)) + 10;

rng(7);
xb = surrogateopt(branin, [-5; 0], [10; 15]);
fprintf('Surrogate optimization of Branin (global f* = 0.3979):\n');
fprintf('  f = %.4f\n', branin(xb));

% ----- Also on the six-hump camelback (non-global local minima) -------
camel = @(x) (4 - 2.1*x(1)*x(1) + (x(1)*x(1)*x(1)*x(1))/3)*x(1)*x(1) ...
             + x(1)*x(2) + (-4 + 4*x(2)*x(2))*x(2)*x(2);
rng(7);
xc = surrogateopt(camel, [-3; -2], [3; 2]);
fprintf('\nSurrogate optimization of the six-hump camelback (global f* = -1.0316):\n');
fprintf('  f = %.4f\n', camel(xc));
