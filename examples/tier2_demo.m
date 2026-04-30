% Tier-2 builtins demo: xcorr, polyval, polyfit, roots, interp1,
% trapz, cumtrapz, gradient, hamming, hann, blackman.
%
% Each call has a hand-checkable expected result printed alongside.

% --- xcorr ---------------------------------------------------------------
% Full lag axis. With L = max(3, 2) = 3 the output has length 5 and
% lag-zero is at index 3 (1-based). The shorter input is implicitly
% zero-padded so the lags k in {-2, -1, 0, +1, +2} land at indices 1..5.
disp('xcorr([1 2 3], [1 1]) — expect [0 1 3 5 3] (lags -2..+2):');
disp(xcorr([1 2 3], [1 1]));

% Autocorrelation of a unit step is a triangle.
disp('xcorr([1 1 1], [1 1 1]) — expect [1 2 3 2 1]:');
disp(xcorr([1 1 1], [1 1 1]));

% --- polyval / polyfit / roots ------------------------------------------
% Evaluate y = x^2 - 3x + 2 at x = 0..3 — expect [2 0 0 2].
p = [1 -3 2];
disp('polyval([1 -3 2], 0:3) — expect [2 0 0 2]:');
disp(polyval(p, [0 1 2 3]));

% Fit a degree-2 polynomial through five points on y = 2x^2 + 1.
xs = [-2 -1 0 1 2];
ys = 2 * xs .* xs + 1;   % [9 3 1 3 9]
disp('polyfit(x, 2x^2+1, 2) — expect [2 0 1]:');
disp(polyfit(xs, ys, 2));

% Roots of x^2 - 5x + 6 = (x-2)(x-3) — expect 3 and 2 (any order).
disp('roots([1 -5 6]) — expect 2 and 3:');
disp(roots([1 -5 6]));

% Roots of x^2 + 1 — expect ±i.
disp('roots([1 0 1]) — expect ±i:');
disp(roots([1 0 1]));

% --- interp1 ------------------------------------------------------------
% Linear interp between known sample points.
xk = [0 1 2 3 4];
yk = [0 1 4 9 16];
disp('interp1([0..4], y, [0.5 1.5 2.5 3.5]) — expect [0.5 2.5 6.5 12.5]:');
disp(interp1(xk, yk, [0.5 1.5 2.5 3.5]));

% --- trapz / cumtrapz / gradient ----------------------------------------
% trapz with unit spacing of [1 2 3 4 5] — expect 12.
disp('trapz([1 2 3 4 5]) — expect 12:');
disp(trapz([1 2 3 4 5]));

% trapz(x, y) with y = x^2 from x=0..2: integral = 8/3 ~= 2.667. Trapezoid
% rule with these 5 sample points overestimates slightly.
xs = [0 0.5 1 1.5 2];
ys = xs .* xs;
disp('trapz(x, x.^2) on x=0:.5:2 — expect ~2.75 (trapezoidal):');
disp(trapz(xs, ys));

% cumtrapz of [1 1 1 1 1] — expect [0 1 2 3 4].
disp('cumtrapz([1 1 1 1 1]) — expect [0 1 2 3 4]:');
disp(cumtrapz([1 1 1 1 1]));

% gradient of [1 4 9 16 25]: central diffs in interior, one-sided at ends.
% Expected: [3 4 6 8 9].
disp('gradient([1 4 9 16 25]) — expect [3 4 6 8 9]:');
disp(gradient([1 4 9 16 25]));

% --- DSP windows --------------------------------------------------------
disp('hamming(5) (column vector):');
disp(hamming(5));

disp('hann(5):');
disp(hann(5));

disp('blackman(5):');
disp(blackman(5));
