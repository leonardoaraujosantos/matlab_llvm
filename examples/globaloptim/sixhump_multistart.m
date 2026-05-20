% sixhump_multistart.m — Global Optimization Toolbox Tier-2 headline.
%
% The "Find Global or Multiple Local Minima" workflow (User's Guide,
% GlobalSearch & MultiStart chapter).  Camelback's six-hump function has
% six local minima, two of them global (f* = -1.0316).  A single local
% solver lands in whatever basin its start point falls into; the
% multi-start meta-solvers explore many basins and return the global.
%
%   problem = createOptimProblem('fmincon', 'objective', @f, 'x0', x0,
%                                'lb', lb, 'ub', ub);
%   [x, f] = run(MultiStart, problem, k);   % k fmincon restarts
%   [x, f] = run(GlobalSearch, problem);    % scatter-search + fmincon
%
% Both meta-solvers reuse the shipped Optimization Toolbox fmincon as the
% local solver; the objective handle rides from createOptimProblem to run
% through a runtime thread-local context.

% Six-hump camelback (x(1)*x(1) form sidesteps the scalar-matpow-in-anon
% compiler gap): f = (4 - 2.1 x1^2 + x1^4/3) x1^2 + x1 x2 + (-4 + 4 x2^2) x2^2.
camel = @(x) (4 - 2.1*x(1)*x(1) + (x(1)*x(1)*x(1)*x(1))/3)*x(1)*x(1) ...
             + x(1)*x(2) ...
             + (-4 + 4*x(2)*x(2))*x(2)*x(2);
lb = [-3; -2];
ub = [ 3;  2];

% ----- Baseline: a single local solve from a poor start ---------------
rng(0);
xlocal = fminunc(camel, [1.6; -0.6]);
fprintf('Single local solve (fminunc) from (1.6, -0.6):\n');
fprintf('  f = %.4f   (trapped in a shallow local minimum)\n\n', camel(xlocal));

% ----- MultiStart: many fmincon restarts ------------------------------
rng(7);
problem = createOptimProblem('fmincon', 'objective', camel, ...
                             'x0', [2; 1], 'lb', lb, 'ub', ub);
ms = MultiStart();
xms = run(ms, problem, 20);
fprintf('MultiStart (20 restarts):\n');
fprintf('  f = %.4f   (global)\n\n', camel(xms));

% ----- GlobalSearch: scatter-search start points ----------------------
rng(7);
gs = GlobalSearch();
xgs = run(gs, problem);
fprintf('GlobalSearch (scatter-search + fmincon):\n');
fprintf('  f = %.4f   (global)\n', camel(xgs));
