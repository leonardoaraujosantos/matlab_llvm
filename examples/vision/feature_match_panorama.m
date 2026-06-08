% feature_match_panorama.m — Computer Vision Toolbox Phase-A headline.
% ----------------------------------------------------------------------
% The canonical "Create Panorama" / feature-registration workflow: detect
% corner features in two overlapping images, describe them, match them across
% the pair, robustly estimate the geometric transform between them with
% RANSAC, and warp one image onto the other's frame.  All over the shipped
% Image Processing (imgaussfilt / imtranslate / imwarp / affine2d) +
% linear-algebra substrate; no Deep Learning, no OpenCV.

rng(1);
base  = imgaussfilt(rand(80, 80) * 255, 2);      % distinctive textured scene
right = imtranslate(base, [6 3]);                % a second overlapping view

% 1) Detect + describe corner features in each view.
p1 = detectHarrisFeatures(base);
p2 = detectHarrisFeatures(right);
f1 = extractFeatures(base, p1);
f2 = extractFeatures(right, p2);
fprintf('features: view1=%d  view2=%d\n', size(p1,1), size(p2,1));

% 2) Match features across the two views.
idx = matchFeatures(f1, f2);
fprintf('putative matches: %d\n', size(idx,1));
m1 = p1(idx(:,1), :);
m2 = p2(idx(:,2), :);

% 3) Robustly estimate the geometric transform (RANSAC rejects mismatches).
T = estgeotform2d(m1, m2, 'affine');
fprintf('estimated translation: tx=%.1f ty=%.1f (true 6, 3)\n', T(3,1), T(3,2));

% 4) Warp the second view into the first view's frame.
tform = affine2d(T);
warped = imwarp(right, tform);
fprintf('panorama-warped view: %dx%d\n', size(warped,1), size(warped,2));
fprintf('feature-matching registration complete.\n');
