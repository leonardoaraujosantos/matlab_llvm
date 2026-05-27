% Econometrics Toolbox Tier-3 — garch / egarch / gjr estimate / infer /
% forecast / simulate.

% --- Generate a GARCH(1,1) return series with known parameters ----------
%     h_t = 0.02 + 0.85 h_{t-1} + 0.10 e_{t-1}^2,  e_t = sqrt(h_t) z_t
N = 600;
s = 13579;
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
    z(t) = sqrt(-2 * log(u1)) * cos(2 * pi * u2);   % standard normal
end
h = zeros(N, 1);
r = zeros(N, 1);
h(1) = 0.02 / (1 - 0.85 - 0.10);
r(1) = sqrt(h(1)) * z(1);
for t = 2:N
    h(t) = 0.02 + 0.85 * h(t-1) + 0.10 * r(t-1)^2;
    r(t) = sqrt(h(t)) * z(t);
end

% --- Fit garch(1,1) and recover the persistence (alpha + beta ~ 0.95) ---
Mdl = garch(1, 1);
Est = estimate(Mdl, r);
persist = Est.GARCH(1) + Est.ARCH(1);
fprintf('garch_persist = %.2f\n', persist);    % ~0.9x (true 0.95)
fprintf('garch_kappa = %.4f\n', Est.Constant); % small positive

% --- infer the conditional variance series -----------------------------
hv = infer(Est, r);
fprintf('nhv = %.0f\n', numel(hv));            % 600
fprintf('hv_mid = %.4f\n', hv(300));           % positive variance

% --- forecast volatility 10 steps --------------------------------------
vF = forecast(Est, 10, r);
fprintf('nvf = %.0f\n', numel(vF));            % 10
fprintf('vf10 = %.4f\n', vF(10));              % positive variance

% --- gjr(1,1) fits an asymmetric model ---------------------------------
G = gjr(1, 1);
EG = estimate(G, r);
fprintf('gjr_kind = %.0f\n', EG.ModelKind);    % 3

% --- simulate a path -----------------------------------------------------
rsim = simulate(Est, 50);
fprintf('nsim = %.0f\n', numel(rsim));         % 50
