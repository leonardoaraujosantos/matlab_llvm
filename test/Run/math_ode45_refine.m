% Verify the Refine option in odeset. Refine = 1 emits only the
% accepted-step endpoints (matches MATLAB's ode23 default and disables
% dense output for ode45). Refine = 8 doubles the dense-output count
% relative to the default of 4.

f = @(t,y) -2*y + sin(t);

% Baseline: default Refine = 4.
[t1, y1] = ode45(f, [0 10], 1);

% Refine = 1: just step endpoints. Length should be ~ length(t1)/4.
opts.Refine = 1;
[t2, y2] = ode45(f, [0 10], 1, opts);
if length(t2) * 3 < length(t1); disp(1); else; disp(0); end

% Endpoints (final accepted step) carry the same y(t_f) regardless of
% Refine — only intermediate emits change.
if abs(y1(end) - y2(end)) < 1e-12; disp(1); else; disp(0); end

% Refine = 8: ~2x the points of default Refine=4.
opts2.Refine = 8;
[t3, y3] = ode45(f, [0 10], 1, opts2);
if length(t3) > 1.5 * length(t1); disp(1); else; disp(0); end

% ode23 default Refine = 1 (matches MATLAB). Confirm by counting points.
[t4, y4] = ode23(f, [0 10], 1);
% Number of dense points for ode23 with default Refine=1: equal to
% (number of accepted steps) + 1 seed. With opts3.Refine = 4 it's
% 4 * accepted + 1 seed, so 4× larger.
opts3.Refine = 4;
[t5, y5] = ode23(f, [0 10], 1, opts3);
if length(t5) > 3 * length(t4); disp(1); else; disp(0); end
