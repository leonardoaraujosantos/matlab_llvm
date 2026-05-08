% Time-frequency analysis via spectrogram.
%
% A linear chirp 0 → 250 Hz at 1 kHz sampling has its instantaneous
% frequency moving across the band. The spectrogram captures this
% as a diagonal stripe of energy in the (time, frequency) plane.

fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 0, 1, 250);

% 128-point Hamming window with 64-point overlap (50%). spectrogram
% returns |STFT|² as an (nfreq × nframe) matrix.
S = spectrogram(x, hamming(128), 64);

fprintf('S rows (frequency bins): %g\n', size(S, 1));
fprintf('S cols (time frames):    %g\n', size(S, 2));

% Energy in the first time frame (early chirp, near 0 Hz) should
% concentrate in low bins. Energy in a late frame should concentrate
% in high bins. Print the peak |STFT|² in three frames as a sanity
% check that the diagonal energy stripe is captured.
firstcol = S(:, 1);
midcol   = S(:, round(size(S, 2) / 2));
lastcol  = S(:, size(S, 2));

disp('first frame peak:');
disp(max(firstcol));
disp('mid   frame peak:');
disp(max(midcol));
disp('last  frame peak:');
disp(max(lastcol));
% All three should be of comparable magnitude (the chirp passes a
% similar amount of energy through the window each frame); the
% *bin* of the peak shifts upward as time progresses.
