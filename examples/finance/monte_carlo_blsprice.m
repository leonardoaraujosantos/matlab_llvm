% monte_carlo_blsprice.m — Financial Toolbox Tier-6 HEADLINE.
%
% Price a European call by Monte Carlo simulation of geometric
% Brownian motion under the risk-neutral measure, and compare the
% discounted sample-mean payoff to the closed-form Black-Scholes
% price from Tier-2. This is the canonical "do the two agree?"
% validation of an SDE engine.

rng(0);

% Option + market parameters.
S0    = 100;
K     = 100;
r     = 0.05;
T     = 1.0;
sigma = 0.20;

% Closed-form reference (Tier-2).
ref = blsprice(S0, K, r, T, sigma);
fprintf('Black-Scholes closed form = %.4f\n', ref);

% Risk-neutral GBM: drift = r.
g = gbm(r, sigma, S0);

% Simulate terminal prices with the exact GBM transition.
nTrials = 20000;
P = simBySolution(g, 252, T/252, nTrials);
ST = P(253, :);

% Discounted expected payoff of the call (optpricemc avoids the
% elementwise max(mat, scalar) lowering gap).
mcPrice = optpricemc(ST, K, r, T);
fprintf('Monte Carlo price         = %.4f\n', mcPrice);
fprintf('abs error vs closed form  = %.4f\n', abs(mcPrice - ref));

% Quasi-Monte-Carlo primitive: a Halton sequence has lower
% discrepancy than pseudo-random draws.
H = haltonseq(8, 2);
fprintf('halton(1,:) = %.4f %.4f\n', H(1,1), H(1,2));
fprintf('halton(2,:) = %.4f %.4f\n', H(2,1), H(2,2));
