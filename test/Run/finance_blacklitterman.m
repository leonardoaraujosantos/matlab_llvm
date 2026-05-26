% black_litterman.m — Financial Toolbox Tier-7 HEADLINE.
%
% Combine the market-equilibrium prior with investor views via the
% Black-Litterman model, then feed the posterior expected returns
% into a Portfolio mean-variance frontier. Mirrors the User's Guide
% §4.223 workflow.

% 3-asset market: covariance + market-cap weights.
Sigma = [ 0.04  0.01  0.005
          0.01  0.05  0.008
          0.005 0.008 0.03 ];
wmkt  = [0.5; 0.3; 0.2];

% Implied equilibrium returns Pi = delta * Sigma * wmkt.
delta = 2.5;
tau   = 0.025;

% Investor view: asset 1 will outperform asset 3 by 6% (a stronger
% spread than the equilibrium prior implies). Pick row P puts +1 on
% asset 1, -1 on asset 3; Q is the 6% view spread.
P = [1 0 -1];
Q = [0.06];

mu_bl = blacklitterman(Sigma, wmkt, P, Q, tau, delta);
fprintf('BL posterior returns: %.4f %.4f %.4f\n', ...
        mu_bl(1), mu_bl(2), mu_bl(3));

% The view tilts asset 1 up and asset 3 down relative to the prior.
Pi = delta * Sigma * wmkt;
fprintf('equilibrium prior:    %.4f %.4f %.4f\n', Pi(1), Pi(2), Pi(3));
fprintf('view tilt a1: %.4f  a3: %.4f\n', mu_bl(1) - Pi(1), mu_bl(3) - Pi(3));

% Feed the posterior into a Portfolio and get the max-Sharpe weights.
p = Portfolio();
p = setAssetMoments(p, mu_bl, Sigma);
p = setDefaultConstraints(p);
w = estimateMaxSharpeRatio(p);
fprintf('BL max-Sharpe weights: %.4f %.4f %.4f\n', w(1), w(2), w(3));

rm = estimatePortMoments(p, w);
fprintf('BL portfolio risk=%.4f return=%.4f\n', rm(1), rm(2));
