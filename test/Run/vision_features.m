% Computer Vision Toolbox Tier-1 — feature detection / description / matching.
%   detectHarrisFeatures + extractFeatures + matchFeatures recover a known
%   image shift; matching a feature set against itself is an exact identity;
%   extractHOGFeatures / extractLBPFeatures return fixed-length descriptors.
rng(1);
I  = imgaussfilt(rand(80, 80) * 255, 2);   % distinctive (non-repetitive) texture
I2 = imtranslate(I, [5 0]);                % shift +5 columns

p1 = detectHarrisFeatures(I);
p2 = detectHarrisFeatures(I2);
f1 = extractFeatures(I, p1);
f2 = extractFeatures(I2, p2);
idx = matchFeatures(f1, f2);

% The matched correspondences must recover the +5 column shift.
dx = p2(idx(:,2), 1) - p1(idx(:,1), 1);
fprintf('matched shift dx: %.0f\n', median(dx));            % 5
fprintf('descriptor length: %.0f\n', size(f1, 2));          % 121 (11x11 patch)

% Matching a feature set against itself is an exact identity map.
si = matchFeatures(f1, f1);
fprintf('self-match identity err: %.0f\n', sum(abs(si(:,1) - si(:,2))));   % 0
fprintf('self-match completeness: %.0f\n', size(si,1) - size(p1,1));       % 0

% Shi-Tomasi + FAST also detect corners on the textured image.
fprintf('min-eigen detects: %.0f\n', min(size(detectMinEigenFeatures(I),1), 1));
fprintf('FAST detects: %.0f\n', min(size(detectFASTFeatures(I),1), 1));

% Fixed-length global descriptors.
fprintf('HOG length: %.0f\n', size(extractHOGFeatures(I), 2));
fprintf('LBP length: %.0f\n', size(extractLBPFeatures(I), 2));
