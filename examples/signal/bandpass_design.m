% Butterworth bandpass — design + apply.
%
% Pass the 30..70 Hz band of a chirp 0..200 Hz at 1 kHz sampling.
% The band-variant dispatch recognises a 2-element [W1 W2] vector
% as "bandpass" automatically.

fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 0, 1, 200);

% Bandpass: edges at 30 Hz and 70 Hz, normalised to Nyquist (fs/2 = 500).
% Pass the normalised pair as a literal 2-element vector — the band-
% variant dispatch in LowerTensorOps only matches when the Wn operand
% is a `tensor<2xf64>` with a `matlab.concat_row` defining op.
[b, a] = butter(4, [0.06 0.14]);
y      = filtfilt(b, a, x);

% Sample around the times when the chirp crosses 50 Hz (centre of
% passband, around t = 0.25 s, sample index 250) and 150 Hz (well
% in the stopband, around t = 0.75 s, sample index 750).
fprintf('passband centre rms: %.4f\n', rms(y(225:275)));
fprintf('stopband centre rms: %.4f\n', rms(y(725:775)));
fprintf('attenuation: %.1f dB\n', ...
    20 * log10(rms(y(225:275)) / rms(y(725:775))));

% [b, a] coefficient counts: 2n+1 each for an order-n bandpass.
fprintf('numerator length: %g\n', length(b));
fprintf('denominator length: %g\n', length(a));
