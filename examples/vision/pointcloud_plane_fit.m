% pointcloud_plane_fit.m — Computer Vision Toolbox Phase-C (Tier-6).
% ----------------------------------------------------------------------
% From a REAL photograph to a 3-D point cloud and back to geometry: build a
% stereo disparity map of the facade, back-project it into a 3-D cloud with
% reconstructScene, persist it as PLY, voxel-downsample it, then fit the
% dominant plane with RANSAC (pcfitplane).  A fronto-parallel facade
% reconstructs to a plane whose normal points along the viewing axis.
% Result image:
%   /tmp/cv_depth.png — the depth map the cloud was reconstructed from

left = imread('data/facade.png');
bg   = imtranslate(left, [-4 0]);       % background depth layer
fg   = imtranslate(left, [-10 0]);      % nearer foreground layer
right = bg;
right(60:140, 60:140) = fg(60:140, 60:140);
D = disparityBM(left, right, 16);

cloud = reconstructScene(D);            % N x 3 back-projected 3-D points
fprintf('reconstructed 3-D points: %d\n', size(cloud,1));

pcwrite('/tmp/facade_cloud.ply', cloud);
c = pcread('/tmp/facade_cloud.ply');
fprintf('PLY round-trip: %d points\n', size(c,1));

ds = pcdownsample(cloud, 12.0);
fprintf('after voxel downsample: %d points\n', size(ds,1));

plane = pcfitplane(cloud, 6.0);
fprintf('facade plane normal: (%.1f, %.1f, %.1f)\n', ...
        abs(plane(1)), abs(plane(2)), abs(plane(3)));

% Image output: the depth map underpinning the reconstruction.
depthImg = D ./ 16 .* 255;
imwrite(depthImg, '/tmp/cv_depth.png');
fprintf('wrote /tmp/cv_depth.png (%dx%d)\n', size(depthImg,1), size(depthImg,2));
