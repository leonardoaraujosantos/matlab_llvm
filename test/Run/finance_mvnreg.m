% Financial Toolbox Tier-4 §1 — regression with missing data (ECM) +
% CAPM.

% --- Complete data: ecmnmle/ecmncov should match plain mean/cov. ---
X = [1.0 2.0; 2.0 4.1; 3.0 5.9; 4.0 8.2; 5.0 9.8];
mu = ecmnmle(X);
fprintf('mean: %.4f %.4f\n', mu(1), mu(2));   % col means 3.0, 6.0

Cv = ecmncov(X);
fprintf('cov(1,1)=%.4f cov(2,2)=%.4f cov(1,2)=%.4f\n', ...
        Cv(1,1), Cv(2,2), Cv(1,2));

% --- Missing data: introduce a NaN, ECM should impute sensibly. ---
Xm = [1.0 2.0; 2.0 4.1; 3.0 NaN; 4.0 8.2; 5.0 9.8];
mum = ecmnmle(Xm);
fprintf('ECM mean with NaN: %.4f %.4f\n', mum(1), mum(2));
% col-2 mean should stay near 6 (the NaN is imputed from the strong
% linear relationship with col-1).

% --- mvnrmle regression: y = X*beta. ---
% Design X = [ones, predictor]; true beta = [1; 2].
Xd = [1 1; 1 2; 1 3; 1 4; 1 5];
y  = [3.0; 5.0; 7.0; 9.0; 11.0];   % exactly 1 + 2*x
beta = mvnrmle(y, Xd);
fprintf('beta = %.4f %.4f\n', beta(1), beta(2));   % 1.0, 2.0

% --- CAPM: alpha + beta of an asset vs market. ---
% asset = 0.5 + 1.2 * market (with rf = 0).
market = [0.02; -0.01; 0.03; 0.015; -0.005; 0.025];
asset  = 1.2 * market + 0.001;   % beta ~1.2, small positive alpha
ab = capm(asset, market, 0);
fprintf('capm alpha=%.4f beta=%.4f\n', ab(1), ab(2));
