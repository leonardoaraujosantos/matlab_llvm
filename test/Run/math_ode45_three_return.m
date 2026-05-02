% 3-return form: [t, y, stats] = ode45(@f, tspan, y0[, opts]). The
% stats struct carries MATLAB-style fields nsteps / nfailed / nfevals.

f = @(t,y) -2*y + sin(t);
[t, y, stats] = ode45(f, [0 10], 1);
disp(length(t));
disp(stats.nsteps);
disp(stats.nfailed);
disp(stats.nfevals);

% 4-arg variant with odeset. Tight RelTol *and* AbsTol drive the
% integrator to many more steps than the default — count is robust.
opts.RelTol = 1e-9;
opts.AbsTol = 1e-12;
[t2, y2, stats2] = ode45(f, [0 10], 1, opts);
disp(stats2.nsteps);
if stats2.nsteps > 5 * stats.nsteps; disp(1); else; disp(0); end

% ode23 stats — its higher per-step error means more accepted steps
% than ode45 at the same tolerance.
[t3, y3, stats3] = ode23(f, [0 10], 1);
if stats3.nsteps > stats.nsteps; disp(1); else; disp(0); end
