% Financial Toolbox Tier-5 §1 — PortfolioCVaR scenario optimization.

% 8 equally-likely scenarios over 3 assets (rows = scenarios).
S = [ 0.05  0.03  0.08
      0.02 -0.01  0.04
     -0.03  0.06 -0.02
      0.04  0.02  0.06
     -0.06 -0.04 -0.08
      0.03  0.05  0.01
      0.01  0.00  0.03
     -0.02  0.04 -0.05 ];

p = PortfolioCVaR();
p = setScenarios(p, S);
p = setProbabilityLevel(p, 0.75);
p = setDefaultConstraints(p);

% CVaR of an equal-weight portfolio at 75% level.
w = [1/3; 1/3; 1/3];
cv = estimatePortRisk(p, w);
fprintf('CVaR(equal-weight) = %.4f\n', cv);

% VaR at the same level.
vr = estimatePortVaR(p, w);
fprintf('VaR(equal-weight)  = %.4f\n', vr);

% Frontier sweep.
W = estimateFrontier(p, 5);
fprintf('frontier: assets=%.0f, points=%.0f\n', size(W,1), size(W,2));

% Leftmost frontier point targets the minimum return; report its CVaR
% and confirm the weights are budget-feasible (sum to 1).
w_lo = W(:, 1);
cv_lo = estimatePortRisk(p, w_lo);
fprintf('frontier point-1 risk = %.4f\n', cv_lo);
fprintf('weights sum = %.4f\n', w_lo(1) + w_lo(2) + w_lo(3));
