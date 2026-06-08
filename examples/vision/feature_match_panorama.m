% feature_match_panorama.m — Computer Vision Toolbox Phase-A headline.
% ----------------------------------------------------------------------
% The canonical "Create Panorama" / feature-registration workflow on a REAL
% photograph (a brick building facade): detect corner features in two
% overlapping views, describe and match them across the pair, robustly
% estimate the geometric transform with RANSAC, and warp one view onto the
% other's frame.  Results are written out as images:
%   /tmp/cv_features.png  — detected corners overlaid on the facade
%   /tmp/cv_panorama.png  — the second view warped into the first's frame
% All over the shipped Image Processing + linear-algebra substrate; no Deep
% Learning, no OpenCV.

I1 = imread('data/facade.png');          % real facade photo (200x200 gray)
I2 = imtranslate(I1, [12 6]);            % a second, overlapping view

% 1) Detect + describe corner features (keep the strongest for speed/clarity).
p1 = detectHarrisFeatures(I1); p1 = p1(1:80, :);
p2 = detectHarrisFeatures(I2); p2 = p2(1:80, :);
f1 = extractFeatures(I1, p1);
f2 = extractFeatures(I2, p2);
fprintf('features: view1=%d  view2=%d\n', size(p1,1), size(p2,1));

% 2) Match features across the two views.
idx = matchFeatures(f1, f2);
fprintf('putative matches: %d\n', size(idx,1));
m1 = p1(idx(:,1), :);
m2 = p2(idx(:,2), :);

% 3) Robustly estimate the geometric transform (RANSAC rejects mismatches).
T = estgeotform2d(m1, m2, 'affine');
fprintf('estimated translation: tx=%.1f ty=%.1f (true 12, 6)\n', T(3,1), T(3,2));

% 4) Image outputs: corner overlay + warped panorama.
overlay = insertMarker(I1, p1);
imwrite(overlay, '/tmp/cv_features.png');

tform  = affine2d(T);
warped = imwarp(I2, tform);
imwrite(warped, '/tmp/cv_panorama.png');

fprintf('wrote /tmp/cv_features.png (%dx%d) and /tmp/cv_panorama.png (%dx%d)\n', ...
        size(overlay,1), size(overlay,2), size(warped,1), size(warped,2));
fprintf('feature-matching registration complete.\n');
