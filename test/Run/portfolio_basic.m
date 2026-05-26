% Financial Toolbox Tier-3 §1 — Portfolio classdef foundation.
% Constructor + setters + simple port-moment evaluation.

p = Portfolio();
% Two-asset universe.
m = [0.05; 0.10];                  % expected returns
C = [0.04, 0.01; 0.01, 0.09];      % covariance matrix
p = setAssetMoments(p, m, C);
p = setDefaultConstraints(p);

% 50/50 portfolio.
w = [0.5; 0.5];
fprintf('estimatePortReturn = %.4f\n', estimatePortReturn(p, w));   % 0.075
fprintf('estimatePortRisk   = %.4f\n', estimatePortRisk(p, w));     % sqrt(0.0375) = 0.1936

% Multi-return form: [risk, return].
rm = estimatePortMoments(p, w);
fprintf('moments = %.4f %.4f\n', rm(1), rm(2));
