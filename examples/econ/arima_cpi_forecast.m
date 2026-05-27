% Econometrics Toolbox — Tier-2 headline (overall roadmap tracer-bullet).
% The canonical Box-Jenkins workflow on a CPI-like price index:
%   test stationarity -> difference -> identify -> estimate -> diagnose ->
%   forecast.

% --- Synthesize a CPI-like index: random walk with drift + AR dynamics ---
N = 240;                                  % 20 years of monthly data
e = zeros(N, 1);
s = 31337;
for t = 1:N
    for k = 1:5
        s = mod(1103515245 * s + 12345, 2147483648);
    end
    e(t) = 0.4 * (s / 2147483648 - 0.5);
end

cpi = zeros(N, 1);
cpi(1) = 100;
g = 0;
for t = 2:N
    g = 0.5 * g + e(t);                   % AR(1) growth shock
    cpi(t) = cpi(t-1) + 0.25 + g;         % drift 0.25/month + dynamics
end

% --- Step 1: the level series is nonstationary (unit root) --------------
fprintf('Level ADF reject-unit-root: %.0f\n', adftest(cpi));   % 0

% --- Step 2: difference once; the growth series is stationary -----------
dcpi = diff(cpi);
fprintf('Diff  ADF reject-unit-root: %.0f\n', adftest(dcpi));  % 1

% --- Step 3: identify orders from ACF/PACF of the differenced series -----
acf = autocorr(dcpi, 6);
pacf = parcorr(dcpi, 6);
fprintf('Diff ACF(1)  = %.3f\n', acf(2));
fprintf('Diff PACF(1) = %.3f\n', pacf(2));

% --- Step 4: fit ARIMA(1,1,0) and inspect the fit -----------------------
Mdl = arima(1, 1, 0);
Est = estimate(Mdl, cpi);
fprintf('Estimated AR(1) = %.3f\n', Est.AR(1));
fprintf('Innovation var  = %.4f\n', Est.Variance);

% --- Step 5: residual diagnostics (Ljung-Box on the innovations) --------
res = infer(Est, cpi);
fprintf('Ljung-Box reject-white: %.0f\n', lbqtest(res, 12));

% --- Step 6: forecast 12 months ahead -----------------------------------
h = 12;
yF = forecast(Est, h, cpi);
fprintf('Forecast horizon: %.0f\n', numel(yF));
fprintf('Last observed CPI:  %.2f\n', cpi(N));
fprintf('CPI forecast (12m): %.2f\n', yF(h));    % > last observed (rising)

fprintf('Box-Jenkins ARIMA forecast complete.\n');
