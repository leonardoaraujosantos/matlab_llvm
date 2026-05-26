% Financial Toolbox Tier-6 §1 — SDE simulation (Euler-Maruyama).

rng(0);

% GBM: 5% drift, 20% vol, start at 100. Simulate 1 year in 252 steps,
% 2000 trials. The terminal-mean should be near S0*exp(mu*T) = 105.13.
g = gbm(0.05, 0.20, 100);
P = simByEuler(g, 252, 1/252, 2000);
fprintf('GBM paths: %.0f x %.0f\n', size(P,1), size(P,2));
fprintf('start row all 100: P(1,1)=%.1f\n', P(1,1));

% Mean terminal value across trials (row 253).
term = P(253, :);
fprintf('GBM terminal mean ~ %.2f (truth 105.13)\n', mean(term));

% HWV mean-reversion: speed=2, level=0.05, sigma=0.01, start 0.03.
% Over time the mean should pull toward the level 0.05.
h = hwv(2.0, 0.05, 0.01, 0.03);
Ph = simByEuler(h, 252, 1/252, 1000);
fprintf('HWV terminal mean ~ %.4f (level 0.05)\n', mean(Ph(253, :)));

% CIR stays non-negative: speed=1.5, level=0.04, sigma=0.05, start 0.03.
c = cir(1.5, 0.04, 0.05, 0.03);
Pc = simByEuler(c, 252, 1/252, 1000);
fprintf('CIR terminal mean ~ %.4f (level 0.04)\n', mean(Pc(253, :)));

% simBySolution exact GBM: terminal mean should also be ~105.13.
Ps = simBySolution(g, 252, 1/252, 2000);
fprintf('GBM exact terminal mean ~ %.2f\n', mean(Ps(253, :)));
