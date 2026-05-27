% Econometrics Toolbox — Tier-1 headline.
% The canonical stationarity workflow: take a trending (unit-root) series,
% confirm nonstationarity with an ADF test, difference it, confirm the
% differenced series is stationary, then inspect its autocorrelation.

% --- Synthesize a random walk with drift (difference-stationary) ----------
N = 150;
e = zeros(N, 1);
s = 7919;
for t = 1:N
    for k = 1:5
        s = mod(1103515245 * s + 12345, 2147483648);
    end
    e(t) = s / 2147483648 - 0.5;
end

y = zeros(N, 1);
y(1) = e(1);
for t = 2:N
    y(t) = 0.05 + y(t-1) + e(t);     % drift 0.05 + unit root
end

% --- Test the level series: ADF should FAIL to reject the unit root -------
h0 = adftest(y);
fprintf('Level series ADF reject-unit-root: %.0f\n', h0);     % 0

% KPSS on the level series should REJECT stationarity --------------------
hk0 = kpsstest(y);
fprintf('Level series KPSS reject-stationary: %.0f\n', hk0);  % 1

% --- Difference and re-test: the differenced series IS stationary ---------
dy = diff(y);
h1 = adftest(dy);
fprintf('Differenced ADF reject-unit-root: %.0f\n', h1);      % 1

% --- Inspect the differenced series's autocorrelation structure -----------
acf = autocorr(dy, 5);
pacf = parcorr(dy, 5);
fprintf('ACF lag0 = %.4f\n', acf(1));        % 1.0000
fprintf('PACF lag0 = %.4f\n', pacf(1));      % 1.0000

% --- Ljung-Box on the differenced series (white-ish increments) -----------
hlb = lbqtest(dy, 10);
fprintf('Differenced Ljung-Box reject-white: %.0f\n', hlb);   % 0

fprintf('Stationarity workflow complete.\n');
