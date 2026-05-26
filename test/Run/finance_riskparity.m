% Financial Toolbox Tier-7 §3 — risk parity + risk budgeting.

% 3-asset covariance with differing volatilities.
C = [ 0.04  0.01  0.00
      0.01  0.09  0.01
      0.00  0.01  0.16 ];

% Equal-risk-contribution portfolio.
w = riskparity(C);
fprintf('ERC weights: %.4f %.4f %.4f\n', w(1), w(2), w(3));
% Lower-vol assets get larger weights; w1 > w2 > w3.

% Verify the risk contributions are equal (~1/3 each).
rc = riskcontribution(C, w);
fprintf('risk contrib: %.4f %.4f %.4f\n', rc(1), rc(2), rc(3));
fprintf('contrib sum = %.4f\n', rc(1) + rc(2) + rc(3));

% Custom risk budget: put 50% of risk in asset 1, 30% in 2, 20% in 3.
b = [0.5; 0.3; 0.2];
wb = riskbudget(C, b);
fprintf('budget weights: %.4f %.4f %.4f\n', wb(1), wb(2), wb(3));
rcb = riskcontribution(C, wb);
fprintf('budget contrib: %.4f %.4f %.4f\n', rcb(1), rcb(2), rcb(3));
