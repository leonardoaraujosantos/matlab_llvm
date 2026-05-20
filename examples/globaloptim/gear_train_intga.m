% Mixed-integer gear-train design  (Global Optimization Toolbox, Tier-6).
% ----------------------------------------------------------------------
% A compound gear train made of four gears with integer tooth counts
% z1..z4 realises the ratio (z1*z2)/(z3*z4).  We want that ratio to
% approximate the target 1/6.931 as closely as possible.  Tooth counts
% are PHYSICAL — they must be integers in a manufacturable range — so this
% is a pure mixed-integer optimization, the defining Tier-6 capability:
%
%   optimoptions('ga', 'IntCon', [1 2 3 4])
%
% forces all four design variables to integer values and `ga` searches the
% discrete design space directly (the fmincon hybrid polish is auto-skipped
% — a continuous refinement is meaningless once the variables are integer).
% This is the classic Sandgren gear-train benchmark; the global optimum is
% an error near 1e-9, far below any rounded-continuous guess.
rng(7);

target = 1 / 6.931;                                 % desired gear ratio

% Squared ratio error.  The target constant is inlined (1/6.931) rather
% than captured — a solver-bound anon must be capture-free — and the
% expression is repeated rather than squared with ^2 to stay on the
% supported scalar-arithmetic path inside an anon.
err = @(z) ((1/6.931) - (z(1)*z(2)) / (z(3)*z(4))) * ...
           ((1/6.931) - (z(1)*z(2)) / (z(3)*z(4)));

lb = [12; 12; 12; 12];                              % min teeth per gear
ub = [60; 60; 60; 60];                              % max teeth per gear

opts = optimoptions('ga', 'PopulationSize', 200, ...
                          'MaxGenerations', 200, ...
                          'IntCon', [1 2 3 4]);

z     = ga(err, 4, [], [], [], [], lb, ub, opts);
ratio = (z(1)*z(2)) / (z(3)*z(4));

fprintf('tooth counts : %.0f  %.0f  %.0f  %.0f\n', z(1), z(2), z(3), z(4));
fprintf('gear ratio   : %.6f   (target %.6f)\n', ratio, target);
fprintf('ratio error  : %.3e\n', err(z));
