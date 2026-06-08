% stereo_depth.m — Computer Vision Toolbox Phase-C (Tier-5).
% ----------------------------------------------------------------------
% Estimate depth from a rectified stereo pair: compute the disparity map with
% block matching, then recover metric depth via depth = focal * baseline /
% disparity.  Also triangulate a known correspondence to a 3-D point.

rng(2);
left  = imgaussfilt(rand(48, 64) * 255, 2);
right = imtranslate(left, [-6 0]);           % right view: 6-px disparity

D = disparityBM(left, right, 16);
dCenter = median(median(D(:, 28:36)));
fprintf('median disparity (center): %.0f px\n', round(dCenter));

focal = 700; baseline = 0.12;                % camera parameters (px, metres)
depth = focal * baseline / max(dCenter, 1);
fprintf('estimated depth: %.2f m\n', depth);

% triangulate one correspondence to a 3-D world point.
C1 = [600 0 0; 0 600 0; 0 0 1; 0 0 0];
C2 = [600 0 0; 0 600 0; 0 0 1; -72 0 0];
Xt = [0.3 -0.2 1.5];
a = [Xt 1]*C1; b = [Xt 1]*C2;
W = triangulate([a(1)/a(3) a(2)/a(3)], [b(1)/b(3) b(2)/b(3)], C1, C2);
fprintf('triangulated point: (%.2f, %.2f, %.2f)\n', W(1), W(2), W(3));
