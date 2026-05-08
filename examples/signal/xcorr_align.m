% Cross-correlation and signal alignment.
%
% Take a chirp, create a delayed copy, and recover the delay via
% finddelay (argmax of the cross-correlation magnitude). Also
% demonstrate xcov (mean-removed cross-correlation) and dtw
% (dynamic time-warping distance for non-linear alignment).

fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 50, 1, 200);

% Build y as x delayed by 25 samples (delay-region samples set to 0,
% then copy x[0..N-delay-1] into y[delay..N-1]).
delay = 25;
y = zeros(1, length(x));
for k = 1:length(x) - delay
  y(k + delay) = x(k);
end

% finddelay returns the signed argmax of |xcorr(x, y)|. Sign convention:
% positive when x leads y, negative when y leads x. Since we built y
% as x delayed by 25 samples (y lags x), the result is -25.
d = finddelay(x, y);
fprintf('recovered delay (signed): %g\n', d);
fprintf('expected magnitude:       %g\n', delay);

% xcov (mean-removed cross-correlation) — peak indicates the lag.
c = xcov(x, y);
fprintf('xcov length: %g\n', length(c));
disp('xcov peak:');
disp(max(c));

% Dynamic time warping — scalar distance, similar signals → small.
fprintf('dtw(x, x):  %.3f\n', dtw(x, x));
fprintf('dtw(x, y):  %.3f\n', dtw(x, y));
