% Global Optimization Tier-2 — GlobalSearch on the six-hump camelback.
% The camelback has six local minima and two global minima at
% (+/-0.0898, -/+0.7126) with f* = -1.0316.  GlobalSearch's scatter
% sampling + fmincon refinement finds the global value.
rng(11);
camel = @(x) (4 - 2.1*x(1)*x(1) + (x(1)*x(1)*x(1)*x(1))/3)*x(1)*x(1) ...
             + x(1)*x(2) ...
             + (-4 + 4*x(2)*x(2))*x(2)*x(2);
lb = [-3; -2];  ub = [3; 2];
problem = createOptimProblem('fmincon', 'objective', camel, ...
                             'x0', [2; 1], 'lb', lb, 'ub', ub);
gs = GlobalSearch();
x = run(gs, problem);
fprintf('GlobalSearch f = %.4f\n', camel(x));   % -1.0316 (global)
