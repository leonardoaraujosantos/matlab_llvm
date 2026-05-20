% Global Optimization Tier-1 — simulated annealing from a wrong start.
% A many-minima 2-D objective whose global min is at (1, -2); SA escapes
% local traps from the start point (4, 4) and the hybrid polish lands it.
rng(13);
fun = @(x) (x(1)-1)*(x(1)-1) + (x(2)+2)*(x(2)+2) ...
           + 3*sin(2*x(1)) + 3*sin(2*x(2));
x = simulannealbnd(fun, [4; 4], [-10; -10], [10; 10]);
fprintf('sa  f = %.4f\n', fun(x));
fprintf('x1 = %.3f\n', x(1));
fprintf('x2 = %.3f\n', x(2));
