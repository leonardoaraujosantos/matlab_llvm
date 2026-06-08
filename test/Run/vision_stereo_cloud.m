% Computer Vision Toolbox Tier-5/6 — camera geometry/stereo + point clouds.
%   triangulate recovers a known 3-D point; disparityBM recovers a known
%   shift; pcwrite/pcread round-trip; pcfitplane recovers a plane normal;
%   pcregistericp recovers a rigid translation; pcdownsample merges voxels.

% --- Tier-5: triangulation ---
C1 = [500 0 0; 0 500 0; 0 0 1; 0 0 0];        % camera 1 (4x3 [X Y Z 1]*C)
C2 = [500 0 0; 0 500 0; 0 0 1; -50 0 0];      % camera 2 (baseline)
X  = [0.2 0.1 2.0];
u1 = [X 1]*C1; p1 = [u1(1)/u1(3) u1(2)/u1(3)];
u2 = [X 1]*C2; p2 = [u2(1)/u2(3) u2(2)/u2(3)];
W = triangulate(p1, p2, C1, C2);
fprintf('triangulated x100: (%.0f, %.0f, %.0f)\n', round(W(1)*100), round(W(2)*100), round(W(3)*100));

% --- Tier-5: stereo block-matching disparity ---
rng(1);
IL = imgaussfilt(rand(40,50)*255, 2);
IR = imtranslate(IL, [-4 0]);                 % right shifted left by 4 -> disparity 4
D = disparityBM(IL, IR, 12);
fprintf('stereo disparity (center): %.0f\n', round(median(median(D(:, 20:30)))));

% --- Tier-6: point cloud I/O + plane fit + ICP + downsample ---
pts = [0 0 0; 1 0 0; 0 1 0; 1 1 0; 0.5 0.5 0; 0.2 0.8 0; 0.9 0.3 0];   % on z=0
pcwrite('/tmp/vision_cloud.ply', pts);
rd = pcread('/tmp/vision_cloud.ply');
fprintf('PLY roundtrip: %.0f pts, err %.2f\n', size(rd,1), sum(sum(abs(rd - pts))));

pl = pcfitplane(pts, 0.05);
fprintf('fitted plane normal: (%.0f %.0f %.0f)\n', abs(pl(1)), abs(pl(2)), abs(pl(3)));

moved = pts; moved(:,1) = moved(:,1) + 0.5;
Tf = pcregistericp(moved, pts);
fprintf('ICP recovered tx: %.1f\n', Tf(1,4));

ds = pcdownsample([0 0 0; 0.01 0 0; 5 5 5], 1.0);
fprintf('voxel downsample 3 -> %.0f\n', size(ds,1));
