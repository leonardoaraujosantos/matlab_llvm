% rastrigin_ga.m — Global Optimization Toolbox Tier-1 headline.
%
% The canonical "Minimize Rastrigin's Function" walkthrough (User's
% Guide, Genetic Algorithm chapter).  Rastrigin's function is the
% standard multi-modal benchmark: a paraboloid studded with a regular
% lattice of ~30 local minima on [-5.12, 5.12]^2, with a single global
% minimum at the origin (f = 0).  A local solver started anywhere but
% the central basin gets trapped; the genetic algorithm explores
% globally, then the fmincon hybrid step polishes the best individual to
% the exact optimum.
%
%   ga(fun, nvars, A, b, Aeq, beq, lb, ub)  — box-bounded genetic search
%
% The three Tier-1 solvers (ga / particleswarm / simulannealbnd) all run
% over the shared seeded PRNG, so results are reproducible via rng, and
% all reuse the shipped Optimization Toolbox fmincon for the hybrid
% polish — no external solver dependency.

% ----- Rastrigin's function (n = 2) ------------------------------------
% f(x) = 10n + sum_i [ x_i^2 - 10 cos(2*pi*x_i) ].   (x(i)*x(i) form
% sidesteps the scalar-matpow-in-anonymous-function compiler gap.)
rastrigin = @(x) 20 ...
    + (x(1)*x(1) - 10*cos(2*pi*x(1))) ...
    + (x(2)*x(2) - 10*cos(2*pi*x(2)));

lb = [-5.12; -5.12];
ub = [ 5.12;  5.12];

% ----- Baseline: a local solver from a poor start gets trapped --------
rng(0);
xlocal = fminunc(rastrigin, [3.1; 2.9]);
fprintf('Local solver (fminunc) from (3.1, 2.9):\n');
fprintf('  f = %.4f   (trapped in a local minimum)\n\n', rastrigin(xlocal));

% ----- Genetic algorithm finds the global basin -----------------------
rng(42);
xga = ga(rastrigin, 2, [], [], [], [], lb, ub);
fprintf('Genetic algorithm (global search + fmincon hybrid):\n');
fprintf('  f = %.4f   at (%.4f, %.4f)\n\n', rastrigin(xga), xga(1), xga(2));

% ----- Cross-check with particle swarm + simulated annealing ----------
rng(42);
xps = particleswarm(rastrigin, 2, lb, ub);
fprintf('Particle swarm:      f = %.4f\n', rastrigin(xps));

rng(42);
xsa = simulannealbnd(rastrigin, [4; 4], lb, ub);
fprintf('Simulated annealing: f = %.4f\n', rastrigin(xsa));
