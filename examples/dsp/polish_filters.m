% polish_filters.m — DSP System Toolbox Tier-6.
%
% Design-and-filter convenience System Objects.  dsp.LowpassFilter and
% dsp.HighpassFilter bundle the windowed-sinc FIR design with the
% streaming filter step — the user spec is just (CutoffFrequency,
% FilterOrder).  dsp.NotchPeakFilter is a tunable second-order biquad
% that can act as either a notch or a peak based on the IsPeak flag.

n = 0:799;
mixed = sin(2 * pi * 0.03 * n) + sin(2 * pi * 0.25 * n);

% Low-pass: keep 0.03, suppress 0.25.
lp = dsp.LowpassFilter('CutoffFrequency', 0.15);
y_lp = lp(mixed);
fprintf('lowpass tail amp = %.3f (low tone, ~1)\n', max(abs(y_lp(600:800))));

% High-pass: keep 0.25, suppress 0.03.
hp = dsp.HighpassFilter('CutoffFrequency', 0.15);
y_hp = hp(mixed);
fprintf('highpass tail amp = %.3f (high tone, ~1)\n', max(abs(y_hp(600:800))));

% Notch at 0.5 (Nyquist-normalised) -> kills the 0.25 cycles/sample tone.
nt = dsp.NotchPeakFilter(0.5, 0.08);
y_nt = nt(mixed);
fprintf('notch tail amp = %.3f (low tone survives)\n', max(abs(y_nt(600:800))));

% LevinsonSolver: AR coefficients from a synthetic autocorrelation.
r = [1.0 0.5 0.3 0.15 0.08];
solv = dsp.LevinsonSolver();
a = solv(r);
fprintf('AR(4) lead taps: %.3f %.3f %.3f\n', a(1), a(2), a(3));
