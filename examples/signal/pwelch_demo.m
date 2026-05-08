% Power spectral density via Welch's averaged periodogram.
%
% Build a chirp from 50 to 200 Hz at 1 kHz sampling and estimate the
% PSD with pwelch. The result has a peak in the bin range that
% matches the chirp's frequency content.

fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 50, 1, 200);

% pwelch with a 256-point Hamming window and 128-point overlap
% (the canonical 50% overlap segmentation).
nwin = 256;
win  = hamming(nwin);
P    = pwelch(x, win, 128);

% Pxx is a length-129 column (single-sided one-output form, fs = 1
% normalised). Since the chirp covers 50..200 Hz of [0, 500] Hz
% Nyquist, energy lives in bins ~ round(50/500 * 128) = 13 through
% round(200/500 * 128) = 51.
fprintf('PSD length: %g\n', length(P));
fprintf('PSD bin 1   (DC):       %.4e\n', P(1));
fprintf('PSD bin 25  (mid-band): %.4e\n', P(25));
fprintf('PSD bin 100 (above):    %.4e\n', P(100));
% Mid-band P(25) (~95 Hz) should be 1-2 orders of magnitude above
% out-of-band P(100) (~390 Hz).
