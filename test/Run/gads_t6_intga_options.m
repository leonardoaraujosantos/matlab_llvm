% Global Optimization Tier-6 — optimoptions + integer-constrained ga.
% optimoptions('ga', ...) carries PopulationSize / MaxGenerations / IntCon
% into the solver.  With IntCon = [1 2] both variables are forced integer,
% so the minimum of (x1-2.3)^2 + (x2+1.7)^2 lands at the nearest feasible
% integer (2, -2): f = 0.09 + 0.09 = 0.18.  Without IntCon the same options
% object drives a continuous solve to the true optimum (2.3, -1.7), f = 0.
% (x*x avoids the scalar-matpow-in-anon gap; the fmincon hybrid is auto-
% skipped for the integer solve.)
rng(5);
f = @(x) (x(1)-2.3)*(x(1)-2.3) + (x(2)+1.7)*(x(2)+1.7);

oi = optimoptions('ga', 'PopulationSize', 60, 'MaxGenerations', 80, 'IntCon', [1 2]);
xi = ga(f, 2, [], [], [], [], [-10;-10], [10;10], oi);
fprintf('int  x1 = %.1f\n', xi(1));     %  2.0
fprintf('int  x2 = %.1f\n', xi(2));     % -2.0
fprintf('int  f  = %.4f\n', f(xi));     %  0.1800

oc = optimoptions('ga', 'PopulationSize', 40, 'MaxGenerations', 60);
xc = ga(f, 2, [-10;-10], [10;10], oc);
fprintf('cont f  = %.4f\n', f(xc));     %  0.0000
