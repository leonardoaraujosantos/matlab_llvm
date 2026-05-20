% Global Optimization Tier-4 — surrogateopt (RBF surrogate).
% Sample-efficient global solver for expensive objectives.  Branin is the
% canonical surrogate-optimization benchmark (global f* = 0.3979); the
% six-hump camelback (global f* = -1.0316, plus non-global local minima)
% checks it finds the global basin.  (x*x form avoids the matpow-in-anon
% gap.)
rng(7);
camel = @(x) (4 - 2.1*x(1)*x(1) + (x(1)*x(1)*x(1)*x(1))/3)*x(1)*x(1) ...
             + x(1)*x(2) + (-4 + 4*x(2)*x(2))*x(2)*x(2);
xc = surrogateopt(camel, [-3; -2], [3; 2]);
fprintf('camel  f = %.4f\n', camel(xc));   % -1.0316

rng(7);
branin = @(x) (x(2) - 5.1/(4*pi*pi)*x(1)*x(1) + 5/pi*x(1) - 6) ...
              * (x(2) - 5.1/(4*pi*pi)*x(1)*x(1) + 5/pi*x(1) - 6) ...
              + 10*(1 - 1/(8*pi))*cos(x(1)) + 10;
xb = surrogateopt(branin, [-5; 0], [10; 15]);
fprintf('branin f = %.4f\n', branin(xb));   % 0.3979
