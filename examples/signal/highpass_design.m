% Chebyshev I highpass — design + apply via the 'high' string variant.
%
% Drop the DC trend / low-frequency drift from a signal and keep the
% high-frequency content. cheby1 with 0.5 dB passband ripple gives a
% sharper transition than Butterworth at the same order, at the cost
% of equiripple in the passband.

fs = 1000;
t  = (0:1/fs:1-1/fs);

% Build a baseline-drift + high-frequency-tone composite via two chirps.
slow_drift = chirp(t, 0, 1, 5);     % 0..5 Hz drift
fast_tone  = chirp(t, 200, 1, 300); % 200..300 Hz content
x = slow_drift + fast_tone;

% Highpass at 50 Hz with cheby1 — pass the 'high' string.
[b, a] = cheby1(5, 0.5, 50/(fs/2), 'high');
y = filter(b, a, x);

% The drift component drops out; only the fast tone remains.
fprintf('input  rms (drift+tone): %.4f\n', rms(x));
fprintf('output rms (tone only):  %.4f\n', rms(y));
fprintf('drift-only rms:          %.4f\n', rms(slow_drift));
fprintf('fast-tone-only rms:      %.4f\n', rms(fast_tone));
% The output rms should be close to the fast-tone rms, well below the
% input rms (which mixed both components).
