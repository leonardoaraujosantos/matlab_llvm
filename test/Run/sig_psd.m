% Tier-2 (Signal Processing Toolbox roadmap §3.1): nonparametric
% spectral estimation — periodogram + Welch's averaged-modified-
% periodogram. Single-output form, default fs = 1.

% Periodogram of an impulse: PSD is flat (single-sided, with mid-bin
% doubling). For length-8 impulse: P = [1/8, 2/8, 2/8, 2/8, 1/8].
imp = [1 0 0 0 0 0 0 0];
P = periodogram(imp);
disp(P);
disp(sum(P));      % 1 (Parseval — total energy of impulse is 1)

% pwelch with a hamming window of length 4, no overlap.
x = [1 2 3 4 5 6 7 8];
w = hamming(4);
P2 = pwelch(x, w, 0);
disp(P2);
disp(size(P2, 1));    % 4/2 + 1 = 3
