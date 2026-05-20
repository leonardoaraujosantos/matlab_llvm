% Global Optimization Tier-1 — particle swarm on a multi-modal function.
% A shifted sum of cosines with global min at (2, -3); particleswarm
% finds the basin and the hybrid polish lands it.
rng(7);
fun = @(x) (x(1)-2)*(x(1)-2) + (x(2)+3)*(x(2)+3) ...
           - cos(3*(x(1)-2)) - cos(3*(x(2)+3)) + 2;
lb = [-10; -10];  ub = [10; 10];
x = particleswarm(fun, 2, lb, ub);
fprintf('pso f = %.4f\n', fun(x));   % 0.0000 (global min)
fprintf('x1 = %.3f\n', x(1));        % 2.000
fprintf('x2 = %.3f\n', x(2));        % -3.000
