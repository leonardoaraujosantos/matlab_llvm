% Tier-1 (Signal Processing Toolbox roadmap §2.2): FIR design via
% windowed-sinc + Savitzky-Golay smoothing. Lowpass scope; fir2,
% firls, firrcos, firpm, kaiserord deferred to follow-on slices.

% fir1 lowpass, length 11 (order 10), cutoff Wn = 0.4. Default
% window is Hamming.
b = fir1(10, 0.4);
disp(sum(b));            % unit DC gain by normalisation -> 1
disp(b(1) - b(11));      % symmetric impulse response -> 0
disp(b(6));              % centre tap (peak)

% Savitzky-Golay (k=2, f=5) — the canonical 5-point quadratic-fit
% smoothing kernel.
B = sgolay(2, 5);
disp(B(3, :));           % steady-state coefficients
                         %   = [-3/35, 12/35, 17/35, 12/35, -3/35]

% sgolayfilt of a perfect line should return the same line — a
% degree-2 polynomial fit reproduces degree-1 inputs exactly.
x = [0 1 2 3 4 5 6 7 8 9];
y = sgolayfilt(x, 2, 5);
disp(y - x);             % residual ~ 0 throughout

% sgolayfilt of a parabola (degree 2) — also reproduced exactly by
% k = 2 fit.
xp = [0 1 4 9 16 25 36 49 64 81];
yp = sgolayfilt(xp, 2, 5);
disp(yp - xp);           % residual ~ 0
