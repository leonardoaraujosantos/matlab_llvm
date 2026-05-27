% Econometrics Toolbox Tier-4 — varm (VAR) + cointegration tests.

N = 800;
% two INDEPENDENT standard-normal shock streams (separate LCG seeds)
s1 = 98765; s2 = 13577;
e1 = zeros(N, 1); e2 = zeros(N, 1);
for t = 1:N
    for k = 1:5, s1 = mod(1103515245 * s1 + 12345, 2147483648); end
    u1 = s1 / 2147483648;
    for k = 1:5, s1 = mod(1103515245 * s1 + 12345, 2147483648); end
    u2 = s1 / 2147483648;
    if u1 < 1e-12, u1 = 1e-12; end
    e1(t) = sqrt(-2*log(u1)) * cos(2*pi*u2);
    for k = 1:5, s2 = mod(1103515245 * s2 + 12345, 2147483648); end
    u3 = s2 / 2147483648;
    for k = 1:5, s2 = mod(1103515245 * s2 + 12345, 2147483648); end
    u4 = s2 / 2147483648;
    if u3 < 1e-12, u3 = 1e-12; end
    e2(t) = sqrt(-2*log(u3)) * cos(2*pi*u4);
end

% --- A bivariate VAR(1): A = [0.5 0.1; 0.2 0.4], c = [0.1; 0.2] ---------
y1 = zeros(N, 1); y2 = zeros(N, 1);
for t = 2:N
    y1(t) = 0.1 + 0.5*y1(t-1) + 0.1*y2(t-1) + e1(t);
    y2(t) = 0.2 + 0.2*y1(t-1) + 0.4*y2(t-1) + e2(t);
end
Y = [y1 y2];

Mdl = varm(2, 1);
Est = estimate(Mdl, Y);
fprintf('A11 = %.2f\n', Est.AR(1,1));          % ~0.50
fprintf('A22 = %.2f\n', Est.AR(2,2));          % ~0.40
fprintf('nseries = %.0f\n', Est.NumSeries);    % 2

yF = forecast(Est, 5, Y);
fprintf('fc_rows = %.0f\n', size(yF,1));        % 5
fprintf('fc_cols = %.0f\n', size(yF,2));        % 2

ir = irf(Est, 10);
fprintf('irf_rows = %.0f\n', size(ir,1));       % 10
fprintf('irf11 = %.3f\n', ir(1,1));             % impact response of series 1

% --- Cointegration: build a cointegrated pair (w2 RW, w1 = w2 + noise) --
w2 = zeros(N, 1); w1 = zeros(N, 1);
for t = 2:N
    w2(t) = w2(t-1) + e2(t);
end
for t = 1:N
    w1(t) = w2(t) + 0.5 * e1(t);     % stationary spread -> cointegrated
end
Wc = [w1 w2];
fprintf('eg_coint = %.0f\n', egcitest(Wc));     % 1 (cointegrated)
fprintf('jci_coint = %.0f\n', jcitest(Wc));     % 1

% --- Non-cointegrated: two independent random walks --------------------
r1 = zeros(N, 1); r2 = zeros(N, 1);
for t = 2:N
    r1(t) = r1(t-1) + e1(t);
    r2(t) = r2(t-1) + e2(t);
end
Wn = [r1 r2];
fprintf('eg_indep = %.0f\n', egcitest(Wn));     % 0 (not cointegrated)
