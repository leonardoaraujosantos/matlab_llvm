% Tier-2 (Signal Processing Toolbox roadmap §3.3): spectrogram —
% short-time Fourier transform, magnitude squared per (freq, frame).
% Single-output form. stft / istft deferred to follow-on slice.

% Length-16 chirp-like input, hamming window of length 4, 50% overlap.
% Frames: (16-4)/2 + 1 = 7. Frequency bins: 4/2 + 1 = 3.
% Output is a 3 × 7 matrix.
x = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16];
w = hamming(4);
S = spectrogram(x, w, 2);
disp(size(S, 1));        % 3
disp(size(S, 2));        % 7
disp(S(1, 1));           % first frame, DC bin

% Sum of all bins = total energy across frames (Parseval per frame,
% summed across frames).
disp(sum(sum(S)));
