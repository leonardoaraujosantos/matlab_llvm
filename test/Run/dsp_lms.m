% DSP System Toolbox Tier-3 — adaptive filters (LMS / RLS).
%
% Acoustic noise cancellation: a clean tone is corrupted by a filtered
% version of a noise reference v.  The adaptive System Object learns the
% echo path from v and subtracts it; the error output converges to the
% clean tone.  Exercises the obj(x, d) -> step dispatch with persisted
% Weights + tapped-input state across the single streaming call.
n = 0:499;
s = sin(2 * pi * 0.02 * n);              % clean tone
rng(1);
v = randn(1, 500);                       % noise reference
d = s + filter([1 0.6 -0.3], 1, v);      % observed = tone + correlated noise

lms = dsp.LMSFilter('Length', 8, 'StepSize', 0.02);
e = lms(v, d);
rls = dsp.RLSFilter('Length', 8, 'ForgettingFactor', 0.99);
er = rls(v, d);

before = sqrt(mean((d(300:500) - s(300:500)) .^ 2));
lms_after = sqrt(mean((e(300:500)  - s(300:500)) .^ 2));
rls_after = sqrt(mean((er(300:500) - s(300:500)) .^ 2));

fprintf('noise rms before = %.3f\n', before);
fprintf('LMS residual     = %.3f\n', lms_after);
fprintf('RLS residual     = %.3f\n', rls_after);
% Both adaptive filters reduce the noise by well over 3x.
w = lms.getWeights();
fprintf('LMS lead tap     = %.3f\n', w(1));
