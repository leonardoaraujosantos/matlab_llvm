% Financial Toolbox Tier-7 §1 — close the Portfolio method gaps:
% estimateBounds, estimateFrontierByRisk, estimatePortFrontier.

m = [0.10; 0.12; 0.08];
C = [ 0.04  0.01  0.005
      0.01  0.05  0.008
      0.005 0.008 0.03 ];
p = Portfolio();
p = setAssetMoments(p, m, C);
p = setDefaultConstraints(p);

% estimateBounds: [minReturn, maxReturn].
b = estimateBounds(p);
fprintf('return bounds: [%.4f, %.4f]\n', b(1), b(2));

% estimatePortFrontier: Kx2 [risk, return] table.
pts = estimatePortFrontier(p, 5);
fprintf('frontier points: %.0f x %.0f\n', size(pts,1), size(pts,2));
fprintf('point1 risk=%.4f ret=%.4f\n', pts(1,1), pts(1,2));
fprintf('point5 risk=%.4f ret=%.4f\n', pts(5,1), pts(5,2));
% risk should increase from point1 to point5.
fprintf('risk rises = %.4f\n', pts(5,1) - pts(1,1));

% estimateFrontierByRisk: find weights for a target risk between the
% endpoints, then confirm the realised risk matches.
targetSigma = 0.16;
w = estimateFrontierByRisk(p, targetSigma);
realised = estimatePortRisk(p, w);
fprintf('target risk %.4f -> realised %.4f\n', targetSigma, realised);
fprintf('weights sum = %.4f\n', w(1) + w(2) + w(3));
