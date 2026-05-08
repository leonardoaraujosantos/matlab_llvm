% Tier-1 (Signal Processing Toolbox roadmap §2.5): close-the-loop
% filter helpers — filtfilt, sosfilt, impz, stepz, grpdelay.

[b, a] = butter(3, 0.4);

% impz: first sample of impulse response equals b[0] for DF-II-T.
h = impz(b, a, 8);
disp(h(1));         % b[0]

% stepz: step response of a unit-DC-gain lowpass converges to 1.
s = stepz(b, a, 32);
disp(s(32));        % near 1 — finite-tail transient is sub-1e-3

% filtfilt of a DC signal returns the DC value EXACTLY (no transient
% with the lfilter_zi-based steady-state initial conditions — this is
% scipy's filtfilt method='pad' default, the same convention MATLAB
% uses for filtfilt's pad mode).
x = ones(1, 16) * 5;
y = filtfilt(b, a, x);
disp(y(8));         % 5 exactly

% filtfilt is approximately zero-phase on palindromic inputs; the
% residual at the boundary captures the small mismatch between the
% odd-reflection padding and the steady-state IC scaling. The strict
% Gustafsson 1996 method (scipy's method='gust') eliminates this
% residual entirely; that's a separate follow-on.
xs = [1 2 3 4 5 5 4 3 2 1];
ys = filtfilt(b, a, xs);
disp(ys(1) - ys(10));      % small boundary residual

% grpdelay at DC for a stable causal lowpass is positive and finite.
% Print the raw value rather than a comparison so the C lane's
% disp(true)=-1 vs Python lane's disp(True)=1 boolean rendering
% doesn't diverge across lanes.
gd = grpdelay(b, a, 8);
disp(gd(1));             % > 0 and bounded

% sosfilt with a unit-gain pass-through SOS row.
sos = [1 0 0 1 0 0];
yp = sosfilt(sos, [1 2 3 4 5]);
disp(yp);                % unchanged
