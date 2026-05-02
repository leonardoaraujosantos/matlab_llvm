% Verify the 4-arg form `[t,y] = ode45(@f, tspan, y0, opts)` honours
% RelTol / AbsTol from a struct built MATLAB-style. dy/dt = -2y + sin(t)
% with y(0)=1 has analytic y(t) = (4*sin(t) - 2*cos(t))/10 + 1.2*exp(-2t).

f = @(t,y) -2*y + sin(t);

opts.RelTol = 1e-9;
opts.AbsTol = 1e-12;
[t_tight, y_tight] = ode45(f, [0 10], 1, opts);

% Analytic value at t=10. Written without unary negation so the Python
% emitter can render the literal arithmetic.
ya = (4*sin(10) - 2*cos(10)) / 10 + 1.2/exp(20);

% Tight tolerance: error is many orders of magnitude smaller than the
% default (1e-3). We assert better than 1e-7 — well below default.
err_tight = abs(y_tight(end) - ya);
if err_tight < 1e-7; disp(1); else; disp(0); end

% A loose RelTol should *use fewer steps* than the tight setting. The
% exact counts vary by FP details, but tight ≫ loose is robust.
opts2.RelTol = 1e-1;
opts2.AbsTol = 1e-1;
[t_loose, y_loose] = ode45(f, [0 10], 1, opts2);
if length(t_tight) > 4 * length(t_loose); disp(1); else; disp(0); end

% ode23 with options also dispatches to the _opts entries.
[t23, y23] = ode23(f, [0 10], 1, opts);
if abs(y23(end) - ya) < 1e-5; disp(1); else; disp(0); end
