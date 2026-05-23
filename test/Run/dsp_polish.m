% DSP System Toolbox Tier-6 — linalg + polish filter System Objects.
%
% LevinsonSolver wraps the Levinson-Durbin recursion (Toeplitz LS for AR
% prediction); NotchPeakFilter is a tunable second-order notch / peak;
% LowpassFilter and HighpassFilter are design-and-filter SOs that lazily
% build a windowed-sinc FIR on first call and stream against persisted
% state thereafter.
r = [1.0 0.5 0.3 0.15];
solver = dsp_LevinsonSolver();             % flat name works too
a = solver(r);
fprintf('levinson n=%d a1=%.3f\n', numel(a), a(1));

n = 0:599;
tone_notch = sin(0.4 * pi * n);
tone_pass  = sin(0.1 * pi * n);
mixed      = tone_notch + tone_pass;

nf  = dsp.NotchPeakFilter(0.4, 0.08);
nf2 = dsp.NotchPeakFilter(0.4, 0.08);
y_n = nf(tone_notch);
y_p = nf2(tone_pass);
fprintf('notch reject=%.4f passband=%.3f\n', ...
        max(abs(y_n(400:600))), max(abs(y_p(400:600))));

lp = dsp.LowpassFilter('CutoffFrequency', 0.2);
y_l = lp(mixed);
fprintf('lp tail=%.3f (keeps 0.1pi, suppresses 0.4pi)\n', max(abs(y_l(400:600))));

hp = dsp.HighpassFilter('CutoffFrequency', 0.25);
y_h = hp(mixed);
fprintf('hp tail=%.3f (keeps 0.4pi, suppresses 0.1pi)\n', max(abs(y_h(400:600))));
