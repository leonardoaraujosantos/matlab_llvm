% Financial Toolbox Tier-5 §2 — PortfolioMAD scenario optimization.
% Mean-absolute-deviation risk on the same 8-scenario / 3-asset set.

S = [ 0.05  0.03  0.08
      0.02 -0.01  0.04
     -0.03  0.06 -0.02
      0.04  0.02  0.06
     -0.06 -0.04 -0.08
      0.03  0.05  0.01
      0.01  0.00  0.03
     -0.02  0.04 -0.05 ];

p = PortfolioMAD();
p = setScenarios(p, S);
p = setDefaultConstraints(p);

% MAD of equal-weight portfolio.
w = [1/3; 1/3; 1/3];
md = estimatePortRisk(p, w);
fprintf('MAD(equal-weight) = %.4f\n', md);

% Frontier sweep.
W = estimateFrontier(p, 5);
fprintf('frontier: assets=%.0f, points=%.0f\n', size(W,1), size(W,2));

% Each frontier portfolio is budget-feasible; report a mid-point MAD.
w_mid = W(:, 3);
md_mid = estimatePortRisk(p, w_mid);
fprintf('mid-frontier MAD = %.4f\n', md_mid);
fprintf('weights sum = %.4f\n', w_mid(1) + w_mid(2) + w_mid(3));
