% pointcloud_plane_fit.m — Computer Vision Toolbox Phase-C (Tier-6).
% ----------------------------------------------------------------------
% Point-cloud ground-plane extraction: write/read a PLY cloud, voxel-grid
% downsample it, fit the dominant plane with RANSAC (pcfitplane), and align a
% transformed copy back with ICP (pcregistericp).

% a cloud: a planar floor (z~0) plus a few off-plane outlier points.
floorpts = [0 0 0; 2 0 0; 0 2 0; 2 2 0; 1 1 0; 0.5 1.5 0; 1.5 0.5 0; 1 0 0; 0 1 0];
outliers = [1 1 3; 0.5 0.5 2];
cloud = [floorpts; outliers];

pcwrite('/tmp/scene_cloud.ply', cloud);
c = pcread('/tmp/scene_cloud.ply');
fprintf('cloud: %d points read from PLY\n', size(c,1));

c = pcdownsample(c, 1.2);
fprintf('after voxel downsample: %d points\n', size(c,1));

plane = pcfitplane(cloud, 0.1);
fprintf('ground-plane normal: (%.1f, %.1f, %.1f)\n', abs(plane(1)), abs(plane(2)), abs(plane(3)));

% register a translated copy back onto the original.
% register the floor points (RANSAC plane handled outliers above; ICP wants inliers)
shifted = floorpts; shifted(:,1) = shifted(:,1) + 0.1;
T = pcregistericp(shifted, floorpts);
fprintf('ICP alignment: tx=%.1f (recovers the -0.1 inverse)\n', T(1,4));
