% adaptive_eq.m — Adaptive channel equalizer combining T1 + T3 + T5.
%
% End-to-end DSP chain that composes three tiers:
%   T5 source   — dsp.SineWave generates a known clean tone.
%   T1 channel  — dsp.FIRFilter models a multi-tap acoustic / RF echo
%                  that distorts the tone.
%   T3 equalizer — dsp.LMSFilter learns to invert the channel from a
%                  training reference (the clean source).
%
% After convergence, the LMS error output is the residual; getWeights
% returns the learned equalizer taps that approximately invert the
% channel impulse response.

% Source: a 30 Hz tone at 1 kHz sampling (T5 SO).
src = dsp.SineWave('Frequency', 30);
src.SampleRate = 1000;
src.SamplesPerFrame = 400;
clean = src();

% Channel: a 3-tap echo path that introduces intersymbol-like distortion.
ch = dsp.FIRFilter('Numerator', [1 0.6 -0.3]);
distorted = ch(clean);

% Equalizer: LMS adapts the 8-tap inverse channel.
eq = dsp.LMSFilter('Length', 8, 'StepSize', 0.01);
e = eq(distorted, clean);

% Convergence metrics over the settled tail (samples 300..400).
tail_d = sqrt(mean((distorted(300:400) - clean(300:400)) .^ 2));
tail_e = sqrt(mean(e(300:400) .^ 2));

fprintf('clean signal RMS    = %.3f\n', sqrt(mean(clean(300:400) .^ 2)));
fprintf('distortion RMS      = %.3f\n', tail_d);
fprintf('residual error RMS  = %.3f\n', tail_e);

w = eq.getWeights();
fprintf('learned EQ taps     = %.3f %.3f %.3f %.3f\n', ...
        w(1), w(2), w(3), w(4));
