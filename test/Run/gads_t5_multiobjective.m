% Global Optimization Tier-5 — multiobjective (gamultiobj / paretosearch).
% Bi-objective f1=(x-1)^2, f2=(x+1)^2: the two objectives conflict, so the
% Pareto-optimal set is the whole interval x in [-1,1] (every point is a
% different trade-off).  Both solvers return a set of non-dominated points
% spanning that interval, with endpoints at x=-1 (f2-optimal) and x=+1
% (f1-optimal).  (x*x avoids the scalar-matpow-in-anon gap.)
rng(1);
fun = @(x) [(x(1)-1)*(x(1)-1); (x(1)+1)*(x(1)+1)];

X = gamultiobj(fun, 1, [], [], [], [], -3, 3);
fprintf('gamultiobj   n  = %.0f\n', size(X, 1));   % 40 (population)
fprintf('gamultiobj   lo = %.2f\n', min(X));        % -1.00
fprintf('gamultiobj   hi = %.2f\n', max(X));        %  1.00

Y = paretosearch(fun, 1, [], [], [], [], -3, 3);
fprintf('paretosearch lo = %.2f\n', min(Y));        % -1.00
fprintf('paretosearch hi = %.2f\n', max(Y));        %  1.00
