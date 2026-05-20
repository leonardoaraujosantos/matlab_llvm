% Global Optimization Tier-6 — full ga signature + optimoptions on Ackley.
% Exercises the canonical 10-arg call form
%   ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, options)
% with an optimoptions carrier (nonlcon = [] — unconstrained).  The Ackley
% function traps a local solver off-centre (fminunc from (3,3) ~6.56) but
% the global solvers find f = 0 at the origin.  (x(1)*x(1) avoids the
% scalar-matpow-in-anon gap.)
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)*x(1) + x(2)*x(2)))) ...
              - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + 20 + exp(1);

rng(0);
xl = fminunc(ackley, [3.0; 3.0]);
fprintf('fminunc f : %.4f\n', ackley(xl));      % 6.5596 (trapped)

opts = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 100);
rng(42);
xg = ga(ackley, 2, [], [], [], [], [-5;-5], [5;5], [], opts);
fprintf('ga f      : %.4f\n', ackley(xg));       % 0.0000

rng(42);
xp = particleswarm(ackley, 2, [-5;-5], [5;5]);
fprintf('pso f     : %.4f\n', ackley(xp));        % 0.0000
