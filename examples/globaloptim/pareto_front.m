% pareto_front.m — Global Optimization Toolbox Tier-5 headline.
%
% Multiobjective optimization (User's Guide, Multiobjective Optimization
% chapter).  When several objectives conflict, there is no single best
% point — there is a *Pareto front* of non-dominated trade-offs, each one
% better in some objective and worse in another.  gamultiobj (an NSGA-II
% genetic algorithm with non-dominated sorting + crowding-distance
% selection) and paretosearch (a non-dominated archive refined by a
% direct-search poll) both return the whole front, not a single compromise.
%
%   [x] = gamultiobj(fun, nvars, A, b, Aeq, beq, lb, ub)
%   [x] = paretosearch(fun, nvars, A, b, Aeq, beq, lb, ub)
%
% The objective returns a *vector* of objective values (the same vector-out
% handle ABI as lsqnonlin).  Here two objectives pull the single design
% variable toward opposite targets:
%
%   f1(x) = (x - 1)^2     (minimised at x = +1)
%   f2(x) = (x + 1)^2     (minimised at x = -1)
%
% so the Pareto-optimal set is the entire interval x in [-1, 1].
% (x*x products sidestep the scalar-matpow-in-anonymous-function gap.)
fun = @(x) [(x(1) - 1)*(x(1) - 1); (x(1) + 1)*(x(1) + 1)];

% ----- gamultiobj: NSGA-II ---------------------------------------------
rng(1);
Xg = gamultiobj(fun, 1, [], [], [], [], -3, 3);
fprintf('gamultiobj (NSGA-II):\n');
fprintf('  Pareto set: %.0f non-dominated trade-off points\n', size(Xg, 1));
fprintf('  spanning x = %.2f (f2-optimal) ... %.2f (f1-optimal)\n\n', ...
        min(Xg), max(Xg));

% ----- paretosearch: archive + direct-search poll ---------------------
rng(1);
Xp = paretosearch(fun, 1, [], [], [], [], -3, 3);
fprintf('paretosearch (direct-search archive):\n');
fprintf('  Pareto set: %.0f non-dominated trade-off points\n', size(Xp, 1));
fprintf('  spanning x = %.2f ... %.2f\n', min(Xp), max(Xp));
fprintf('\nBoth recover the full trade-off curve, not a single compromise.\n');
