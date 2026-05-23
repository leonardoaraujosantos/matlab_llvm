% lms_anc.m — DSP System Toolbox Tier-3 tracer.
%
% Acoustic Noise Cancellation with a dsp.LMSFilter System Object (the
% canonical adaptive-filter demo).  A clean speech-like tone is buried
% under noise that is a filtered copy of a measurable reference (e.g. a
% microphone near the noise source).  The LMS filter adaptively models the
% acoustic path from the reference to the corrupting noise and subtracts
% it; the error signal is the recovered clean tone.
%
% The adaptive weights persist across the streaming call (the Tier-1 SO
% state model); getWeights reads the converged echo-path estimate.

n = 0:799;
clean = sin(2 * pi * 0.015 * n) + 0.5 * sin(2 * pi * 0.045 * n);

% Noise reference + the unknown acoustic path that corrupts the signal.
rng(7);
ref  = randn(1, 800);
path = [0.8 -0.5 0.3 -0.1];
corrupting = filter(path, 1, ref);

mic = clean + corrupting;             % what the primary microphone hears

% Adapt: NLMS for robust convergence regardless of reference power.
anc = dsp.LMSFilter('Length', 12, 'StepSize', 0.05);
anc.Method = 1;                       % normalized LMS
recovered = anc(ref, mic);

% Convergence metrics over the settled tail.
tail = 500:800;
noise_rms = sqrt(mean(corrupting(tail) .^ 2));
resid_rms = sqrt(mean((recovered(tail) - clean(tail)) .^ 2));
fprintf('noise RMS at mic     = %.4f\n', noise_rms);
fprintf('residual after ANC   = %.4f\n', resid_rms);

% The learned filter approximates the acoustic path.
w = anc.getWeights();
fprintf('estimated path: %.2f %.2f %.2f %.2f\n', w(1), w(2), w(3), w(4));
