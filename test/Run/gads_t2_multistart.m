% Global Optimization Tier-2 — MultiStart over a two-basin function.
% f(x) = (x1^2 - 1)^2 + x2^2 has two global minima at (+/-1, 0), f=0.
% A single fmincon from (2,2) finds one basin; MultiStart's restarts
% guarantee the global is found.  (x*x avoids the scalar-matpow gap.)
rng(5);
f = @(x) (x(1)*x(1) - 1)*(x(1)*x(1) - 1) + x(2)*x(2);
lb = [-3; -3];  ub = [3; 3];
problem = createOptimProblem('fmincon', 'objective', f, ...
                             'x0', [2; 2], 'lb', lb, 'ub', ub);
ms = MultiStart();
x = run(ms, problem, 10);
fprintf('MultiStart f = %.4f\n', f(x));   % 0.0000 (global)
fprintf('nvars = %.0f\n', size(x, 1));    % 2
