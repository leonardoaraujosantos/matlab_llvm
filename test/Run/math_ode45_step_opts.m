% Verify MaxStep / InitialStep odeset options. The runtime caps the
% adaptive step at MaxStep (forcing more output points) and uses
% InitialStep instead of the 1%-of-span heuristic when supplied.

f = @(t,y) -2*y + sin(t);

% Default — 19 accepted steps × Refine=4 + 1 seed = 77 dense samples.
[t1, y1] = ode45(f, [0 10], 1);

% MaxStep = 0.05 forces small adaptive steps; output point count
% balloons proportionally.
opts.MaxStep = 0.05;
[t2, y2] = ode45(f, [0 10], 1, opts);
if length(t2) > 5 * length(t1); disp(1); else; disp(0); end

% InitialStep = 0.001 — the first dense-output sub-point sits at
% θ=1/4 of the first step, i.e. 0.001 / 4 = 0.00025.
opts2.InitialStep = 0.001;
[t3, y3] = ode45(f, [0 10], 1, opts2);
if abs(t3(2) - 0.00025) < 1e-9; disp(1); else; disp(0); end

% MaxStep also applies on the way down — backward leg should land at 0.
opts3.MaxStep = 0.1;
g = @(t,y) 0 - y;
[t4, y4] = ode45(g, [1 0], exp(0)/exp(1), opts3);
disp(t4(end));
