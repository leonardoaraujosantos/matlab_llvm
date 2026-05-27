% Econometrics Toolbox — Tier-4 headline.
% A bivariate macro VAR: model CPI inflation and the unemployment rate
% jointly, then forecast and trace the impulse responses.

N = 240;
s = 5150;
e1 = zeros(N,1); e2 = zeros(N,1);
for t = 1:N
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    u1 = s / 2147483648;
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    u2 = s / 2147483648;
    if u1 < 1e-12, u1 = 1e-12; end
    e1(t) = 0.3 * sqrt(-2*log(u1)) * cos(2*pi*u2);
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    u3 = s / 2147483648;
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    u4 = s / 2147483648;
    if u3 < 1e-12, u3 = 1e-12; end
    e2(t) = 0.2 * sqrt(-2*log(u3)) * cos(2*pi*u4);
end

% inflation (infl) and unemployment (unemp) with cross-feedback
infl = zeros(N,1); unemp = zeros(N,1);
infl(1) = 2.0; unemp(1) = 5.0;
for t = 2:N
    infl(t)  = 0.5 + 0.6*infl(t-1) - 0.1*unemp(t-1) + e1(t);
    unemp(t) = 2.0 - 0.15*infl(t-1) + 0.7*unemp(t-1) + e2(t);
end
Y = [infl unemp];

% --- Fit a VAR(2) ------------------------------------------------------
Mdl = varm(2, 2);
Est = estimate(Mdl, Y);
fprintf('Series modeled:  %.0f\n', Est.NumSeries);
fprintf('Lag order:       %.0f\n', Est.P);
fprintf('AR1 infl<-infl:  %.3f\n', Est.AR(1,1));
fprintf('AR1 unemp<-unemp: %.3f\n', Est.AR(2,2));

% --- Forecast 8 quarters ahead -----------------------------------------
yF = forecast(Est, 8, Y);
fprintf('Forecast size:   %.0f x %.0f\n', size(yF,1), size(yF,2));
fprintf('Inflation +8:    %.2f\n', yF(8,1));
fprintf('Unemployment +8: %.2f\n', yF(8,2));

% --- Impulse responses to an inflation shock ---------------------------
ir = irf(Est, 12);
fprintf('IRF size:        %.0f x %.0f\n', size(ir,1), size(ir,2));
fprintf('Impact on infl:  %.3f\n', ir(1,1));

fprintf('Macro VAR analysis complete.\n');
