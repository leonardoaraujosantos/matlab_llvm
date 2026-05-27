% Econometrics Toolbox — Tier-3 headline.
% Model the volatility of an FX-return-like series with a GARCH(1,1):
% returns are near-zero-mean but show volatility clustering.

% --- Synthesize returns with volatility clustering (a true GARCH DGP) ---
N = 500;
s = 271828;
z = zeros(N, 1);
for t = 1:N
    for k = 1:5
        s = mod(1103515245 * s + 12345, 2147483648);
    end
    u1 = s / 2147483648;
    for k = 1:5
        s = mod(1103515245 * s + 12345, 2147483648);
    end
    u2 = s / 2147483648;
    if u1 < 1e-12, u1 = 1e-12; end
    z(t) = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
end
h = zeros(N, 1);
ret = zeros(N, 1);
h(1) = 0.1;
ret(1) = sqrt(h(1)) * z(1);
for t = 2:N
    h(t) = 0.05 + 0.88 * h(t-1) + 0.08 * ret(t-1)^2;
    ret(t) = sqrt(h(t)) * z(t);
end

% --- An ARCH test confirms conditional heteroscedasticity --------------
fprintf('ARCH effects present: %.0f\n', archtest(ret, 4));   % 1

% --- Fit a GARCH(1,1) -------------------------------------------------
Mdl = garch(1, 1);
Est = estimate(Mdl, ret);
fprintf('GARCH coeff (beta):  %.3f\n', Est.GARCH(1));
fprintf('ARCH  coeff (alpha): %.3f\n', Est.ARCH(1));
fprintf('Persistence:         %.3f\n', Est.GARCH(1) + Est.ARCH(1));

% --- Inferred conditional variance tracks the volatility ----------------
hv = infer(Est, ret);
fprintf('Conditional var samples: %.0f\n', numel(hv));

% --- Forecast volatility over the next 20 periods -----------------------
vF = forecast(Est, 20, ret);
fprintf('Volatility forecast (h=20): %.4f\n', vF(20));

fprintf('GARCH volatility modeling complete.\n');
