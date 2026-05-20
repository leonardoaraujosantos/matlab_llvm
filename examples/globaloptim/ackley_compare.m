% ackley_compare.m — Compare global solvers on the Ackley function.
%
% Adapted from the MathWorks "Compare global solvers" walkthrough.  The
% Ackley function is the textbook multi-modal trap: a nearly flat outer
% region riddled with dozens of shallow local minima surrounds a single
% sharp global minimum at the origin (f = 0).  A gradient-based local
% solver started anywhere off-centre is captured by the nearest ripple
% and never reaches the origin; the Global Optimization solvers sample
% the whole bounded box and find the global basin without a good start
% point.
%
%   f(x) = -20*exp(-0.2*sqrt(0.5*(x1^2 + x2^2)))
%          - exp(0.5*(cos(2*pi*x1) + cos(2*pi*x2))) + 20 + e
%
% Notes on the adaptation to this compiler:
%   * The objective is an anonymous function (the solver handle ABI), and
%     x1^2 is written x(1)*x(1) — both sidestep current frontend gaps.
%   * `ga` is driven through the canonical full signature
%       ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, options)
%     with an `optimoptions('ga', ...)` carrier (PopulationSize /
%     MaxGenerations).  PlotFcn live plots and the
%     [x, fval, exitflag, output] multi-return are Tier-6 follow-ons; the
%     objective value is reported as fun(x) instead of output.fval.

% ----- Ackley function (n = 2) ----------------------------------------
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)*x(1) + x(2)*x(2)))) ...
              - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) ...
              + 20 + exp(1);

lb = [-5; -5];
ub = [ 5;  5];

fprintf('Ackley global minimum is f = 0 at the origin.\n');
fprintf('  sanity: f(0,0) = %.6f\n\n', ackley([0; 0]));

% ----- Baseline: a local solver from a poor start is trapped ----------
rng(0);
xlocal = fminunc(ackley, [3.0; 3.0]);
fprintf('fminunc      from (3,3) : f = %.6f   (trapped in a ripple)\n\n', ...
        ackley(xlocal));

% ----- Genetic algorithm via the full signature + optimoptions --------
options = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 100);
rng(42);
xga = ga(ackley, 2, [], [], [], [], lb, ub, [], options);
fprintf('ga           (50x100)   : f = %.6f   at (%.4f, %.4f)\n', ...
        ackley(xga), xga(1), xga(2));

% ----- Cross-check the other global solvers ---------------------------
rng(42);
xps = particleswarm(ackley, 2, lb, ub);
fprintf('particleswarm           : f = %.6f\n', ackley(xps));

rng(42);
xpat = patternsearch(ackley, [3.0; 3.0], [], [], [], [], lb, ub);
fprintf('patternsearch from (3,3): f = %.6f\n', ackley(xpat));

rng(42);
xsa = simulannealbnd(ackley, [3.0; 3.0], lb, ub);
fprintf('simulannealbnd from (3,3): f = %.6f\n', ackley(xsa));
