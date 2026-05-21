% Statistics Toolbox Tier-6 — Bayesian optimization (GP + EI).
rng(7);
f = @(x) (x(1)-3)*(x(1)-3) + (x(2)+1)*(x(2)+1);
xb = bayesopt(f, [-5;-5], [5;5]);
fprintf('x1   %.0f\n', xb(1));
fprintf('x2   %.0f\n', xb(2));
fprintf('fval %.2f\n', f(xb));
