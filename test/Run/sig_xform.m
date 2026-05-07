% Tier-2 (Signal Processing Toolbox roadmap §3.4): transforms tail —
% DCT-II / DCT-III / Walsh-Hadamard / Hilbert / Goertzel.

x = [1 2 3 4 5 6 7 8];

% DCT-II: orthonormal forward transform.
y = dct(x);
disp(y);

% DCT-III round-trip recovers the input exactly.
xr = idct(y);
disp(max(abs(x - xr)) < 1e-10);    % -1 (true) on C lane, 1 on Python

% Walsh-Hadamard of a unit impulse: all-equal output (1/N each entry).
imp = [1 0 0 0 0 0 0 0];
disp(fwht(imp));

% Goertzel single-bin DFT magnitude — bin 2 (1-based, second positive
% frequency). Magnitude matches |FFT[1]| for the same input.
z = goertzel(x, 2);
disp(abs(z));    % single positive scalar
