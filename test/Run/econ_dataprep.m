% Econometrics Toolbox Tier-1 — data prep + ACF/PACF smoke test.
P = [100; 102; 101; 105; 107; 106; 110];
r = price2ret(P);              % 6 log returns
fprintf('nret=%.0f\n', numel(r));
fprintf('r1=%.4f\n', r(1));    % log(102/100) = 0.019803

Pb = ret2price(r);             % recovers a normalized price path
fprintf('npr=%.0f\n', numel(Pb));
fprintf('ratio=%.4f\n', Pb(7) / Pb(1));   % 110/100 = 1.1000

y = [1; 2; 3; 2; 1; 2; 3; 2; 1; 2; 3; 2];
acf = autocorr(y, 4);          % length 5, acf(1) = 1
fprintf('acf0=%.4f\n', acf(1));
pacf = parcorr(y, 4);          % length 5, pacf(1) = 1
fprintf('pacf0=%.4f\n', pacf(1));

trend = hpfilter(y, 100);      % smoothed trend, same length
fprintf('ntrend=%.0f\n', numel(trend));
