% Sample-rate conversion via resample / decimate / interp.
%
% Take a 1 kHz-sampled chirp and convert it to 1.5 kHz (upsample by
% 3 then downsample by 2) using the polyphase resampler. The output
% has the same physical time span but more samples.

fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 0, 1, 200);

% resample(x, p, q): output length = ceil(N * p / q).
y = resample(x, 3, 2);
fprintf('input  length: %g\n', length(x));
fprintf('output length: %g\n', length(y));

% Plain decimate: anti-aliased downsample by 4 → length / 4.
yd = decimate(x, 4);
fprintf('decimated length: %g\n', length(yd));

% Plain interp: anti-aliased upsample by 4 → length * 4.
yi = interp(x, 4);
fprintf('interpolated length: %g\n', length(yi));

% RMS sanity: anti-aliased filtering preserves energy in the surviving
% band; decimate-by-4 attenuates content above fs/8 = 125 Hz, so a
% 0..200 Hz chirp loses some energy.
fprintf('input rms:        %.4f\n', rms(x));
fprintf('decimated rms:    %.4f\n', rms(yd));
