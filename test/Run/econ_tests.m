% Econometrics Toolbox Tier-1 — diagnostic + unit-root test battery.

% --- Deterministic pseudo-white noise via a linear congruential generator.
N = 200;
e = zeros(N, 1);
s = 12345;
for t = 1:N
    % Advance several LCG steps per sample to break the lattice serial
    % correlation of a bare congruential generator.
    for k = 1:5
        s = mod(1103515245 * s + 12345, 2147483648);
    end
    e(t) = s / 2147483648 - 0.5;     % ~ Uniform(-0.5, 0.5), uncorrelated
end

% --- Stationary AR(1): y_t = 0.2 y_{t-1} + e_t ---
ys = zeros(N, 1);
ys(1) = e(1);
for t = 2:N
    ys(t) = 0.2 * ys(t-1) + e(t);
end

% --- Random walk (unit root): rw_t = rw_{t-1} + e_t ---
rw = zeros(N, 1);
rw(1) = e(1);
for t = 2:N
    rw(t) = rw(t-1) + e(t);
end

% ADF: stationary AR(1) -> reject unit root (h=1); random walk -> h=0.
fprintf('adf_stationary=%.0f\n', adftest(ys));   % 1
fprintf('adf_randomwalk=%.0f\n', adftest(rw));    % 0

% KPSS: random walk -> reject stationarity (h=1); AR(1) stationary -> h=0.
fprintf('kpss_randomwalk=%.0f\n', kpsstest(rw));  % 1
fprintf('kpss_stationary=%.0f\n', kpsstest(ys));  % 0

% Ljung-Box on white noise -> fail to reject (h=0).
fprintf('lbq_white=%.0f\n', lbqtest(e, 10));      % 0

% ARCH test on white noise -> no volatility clustering (h=0).
fprintf('arch_white=%.0f\n', archtest(e, 2));     % 0

% aicbic: -2*logL + 2k.  logL=-50, k=3 -> 106.
fprintf('aic=%.0f\n', aicbic(-50, 3));            % 106

% lratiotest: logLu=-48, logLr=-52, dof=2 -> stat=8, p<0.05 -> h=1.
fprintf('lr=%.0f\n', lratiotest(-48, -52, 2));    % 1

% variance-ratio on random walk -> fail to reject RW (h=0).
fprintf('vr_randomwalk=%.0f\n', vratiotest(rw));  % 0

% HAC covariance of a tiny OLS: X = [1 x], y ~ 2 + 3x.
X = [1 1; 1 2; 1 3; 1 4; 1 5];
yy = [5.1; 8.0; 10.9; 14.1; 17.0];
V = hac(X, yy);
fprintf('hac_dim=%.0fx%.0f\n', size(V,1), size(V,2));   % 2x2
