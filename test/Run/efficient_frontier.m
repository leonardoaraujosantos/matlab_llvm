% efficient_frontier.m — Financial Toolbox Tier-3 HEADLINE demo.
%
% Construct a 5-asset Portfolio object, sweep the mean-variance
% frontier, and locate the tangency (max-Sharpe) portfolio. Mirrors
% the canonical Markowitz workflow from the User's Guide §3.5:
%
%   p = Portfolio('AssetMean', m, 'AssetCovar', C)
%   p = setDefaultConstraints(p)
%   pwgt = estimateFrontier(p, 20)
%   [risk, ret] = estimatePortMoments(p, pwgt)
%   plotFrontier(p)
%
% The asset universe is a synthetic 5-stock mix with realistic
% expected returns + a positive-definite covariance.

% --- Asset universe ---
m = [0.10; 0.12; 0.08; 0.07; 0.15];
C = [ 0.04  0.01  0.005  0.001  0.02
      0.01  0.05  0.008  0.002  0.015
      0.005 0.008 0.03   0.003  0.01
      0.001 0.002 0.003  0.02   0.005
      0.02  0.015 0.01   0.005  0.08];

% --- Portfolio object ---
p = Portfolio();
p = setAssetMoments(p, m, C);
p = setDefaultConstraints(p);

% --- 20-point frontier sweep ---
K = 20;
W = estimateFrontier(p, K);
fprintf('frontier: assets=%.0f, points=%.0f\n', size(W,1), size(W,2));

% --- Endpoint moments (min-variance vs max-return) ---
w_lo = W(:, 1);
w_hi = W(:, K);
rm_lo = estimatePortMoments(p, w_lo);
rm_hi = estimatePortMoments(p, w_hi);
fprintf('min-var portfolio: risk=%.4f, return=%.4f\n', rm_lo(1), rm_lo(2));
fprintf('max-ret portfolio: risk=%.4f, return=%.4f\n', rm_hi(1), rm_hi(2));

% --- Tangency / Max-Sharpe ---
w_ms = estimateMaxSharpeRatio(p);
rm_ms = estimatePortMoments(p, w_ms);
sharpe_ratio = rm_ms(2) / rm_ms(1);
fprintf('max-Sharpe portfolio: risk=%.4f, return=%.4f, sharpe=%.4f\n', ...
        rm_ms(1), rm_ms(2), sharpe_ratio);

% --- Frontier-by-return reverse lookup ---
% Find weights that exactly hit return = 0.10 (asset 1's mean).
w_target = estimateFrontierByReturn(p, 0.10);
rm_target = estimatePortMoments(p, w_target);
fprintf('target r=0.10: risk=%.4f, return=%.4f\n', rm_target(1), rm_target(2));

% --- estimateAssetMoments from synthetic returns ---
% 100 monthly returns; mean each column should be near m, and
% sample covariance near C with sampling error.
rng(0);
T = 100;
returns = zeros(T, 5);
for t = 1:T
    for j = 1:5
        returns(t, j) = m(j) + randn(1) * sqrt(C(j, j));
    end
end
moments = estimateAssetMoments(p, returns);
fprintf('estimated mean: %.3f %.3f %.3f %.3f %.3f\n', ...
        moments(1,1), moments(2,1), moments(3,1), moments(4,1), moments(5,1));
