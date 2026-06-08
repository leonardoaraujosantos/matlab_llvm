% stereo_depth.m — Computer Vision Toolbox Phase-C (Tier-5).
% ----------------------------------------------------------------------
% Estimate depth from a rectified stereo pair built from a REAL photograph.
% A closer foreground patch is given a larger horizontal shift than the
% background, so block matching recovers two depth layers; we then convert
% disparity to metric depth via depth = focal * baseline / disparity, and
% triangulate a known correspondence to a 3-D point.  Result image:
%   /tmp/cv_disparity.png — the block-matching disparity map (bright = closer)

left = imread('data/facade.png');       % real left view
bg   = imtranslate(left, [-4 0]);       % background: 4-px disparity (far)
fg   = imtranslate(left, [-10 0]);      % foreground: 10-px disparity (near)
right = bg;
right(60:140, 60:140) = fg(60:140, 60:140);   % a closer object in the centre

D = disparityBM(left, right, 16);
dFg = median(median(D(80:120, 80:120)));       % foreground (centre)
dBg = median(median(D(10:30, 10:30)));         % background (corner)
fprintf('disparity: foreground=%.0f px  background=%.0f px\n', round(dFg), round(dBg));

focal = 700; baseline = 0.12;           % camera parameters (px, metres)
fprintf('depth: foreground=%.2f m  background=%.2f m\n', ...
        focal * baseline / max(dFg,1), focal * baseline / max(dBg,1));

% triangulate one correspondence to a 3-D world point.
C1 = [600 0 0; 0 600 0; 0 0 1; 0 0 0];
C2 = [600 0 0; 0 600 0; 0 0 1; -72 0 0];
Xt = [0.3 -0.2 1.5];
a = [Xt 1]*C1; b = [Xt 1]*C2;
W = triangulate([a(1)/a(3) a(2)/a(3)], [b(1)/b(3) b(2)/b(3)], C1, C2);
fprintf('triangulated point: (%.2f, %.2f, %.2f)\n', W(1), W(2), W(3));

% Image output: the disparity map, scaled to the 16-px search range.
dispImg = D ./ 16 .* 255;
imwrite(dispImg, '/tmp/cv_disparity.png');
fprintf('wrote /tmp/cv_disparity.png (%dx%d)\n', size(dispImg,1), size(dispImg,2));
