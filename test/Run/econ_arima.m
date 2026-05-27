% Econometrics Toolbox Tier-2 — arima estimate / forecast / infer / simulate.

% --- Generate a known AR(1): y_t = 5 + 0.6 y_{t-1} + e_t ----------------
N = 300;
e = zeros(N, 1);
s = 24601;
for t = 1:N
    for k = 1:5
        s = mod(1103515245 * s + 12345, 2147483648);
    end
    e(t) = 0.5 * (s / 2147483648 - 0.5);
end
y = zeros(N, 1);
y(1) = 12.5;                        % near the unconditional mean 5/(1-0.6)
for t = 2:N
    y(t) = 5 + 0.6 * y(t-1) + e(t);
end

% --- Fit AR(1) [arima(1,0,0)] and recover the coefficient --------------
Mdl = arima(1, 0, 0);
Est = estimate(Mdl, y);
fprintf('AR1 = %.2f\n', Est.AR(1));            % ~0.60
c = Est.Constant;
fprintf('mean = %.1f\n', c / (1 - Est.AR(1))); % ~5/(1-0.6)=12.5

% --- Residuals (infer) have near-zero mean -----------------------------
res = infer(Est, y);
fprintf('nres = %.0f\n', numel(res));          % 300

% --- Forecast 5 steps; the level should hover near the mean ~12.5 ------
yF = forecast(Est, 5, y);
fprintf('nfc = %.0f\n', numel(yF));            % 5
fprintf('fc5 = %.1f\n', yF(5));                % ~12.x (near the mean)

% --- ARIMA(0,1,1) on a random walk + MA noise: difference works --------
M2 = arima(0, 1, 1);
rw = zeros(N, 1);
rw(1) = e(1);
for t = 2:N
    rw(t) = rw(t-1) + e(t) + 0.3 * e(t-1);
end
E2 = estimate(M2, rw);
fprintf('MA1 = %.3f\n', E2.MA(1));             % estimated MA(1) coefficient
f2 = forecast(E2, 3, rw);
fprintf('nfc2 = %.0f\n', numel(f2));           % 3

% --- simulate a path of a known length ---------------------------------
ysim = simulate(Est, 20);
fprintf('nsim = %.0f\n', numel(ysim));         % 20
