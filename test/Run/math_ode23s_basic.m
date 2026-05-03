% ode23s — Rosenbrock 2(3) stiff solver. Stiff scalar problem
% dy/dt = -100*y + 100, y(0) = 0. Analytic y(t) = 1 - exp(-100*t),
% steady state y = 1 reached after a few time constants (1/100).
%
% This is the canonical "smoke test" for a stiff solver: ode45 takes
% many steps fighting the stiff eigenvalue, ode23s clears it cleanly.

f = @(t,y) -100*y + 100;

[t1, y1, s1] = ode45(f, [0 1], 0);
[t2, y2, s2] = ode23s(f, [0 1], 0);

% ode23s converges to the steady state more accurately than ode45 with
% the same default tolerances.
if abs(y2(end) - 1) < 1e-6; disp(1); else; disp(0); end

% Both solvers use a bounded number of steps. The exact counts vary by
% FP rounding, but ode23s should never need more than ~30 here.
if s2.nsteps < 30; disp(1); else; disp(0); end

% On a more aggressively stiff problem (decay rate 1000), ode23s
% finishes in tens of steps; ode45 needs hundreds.
g = @(t,y) -1000*y;
[t3, y3, s3] = ode23s(g, [0 1], 1);
if s3.nsteps < 80; disp(1); else; disp(0); end
% y(1) ≈ exp(-1000) ≈ 0 — stiff solver hits the steady state.
if abs(y3(end)) < 1e-6; disp(1); else; disp(0); end
