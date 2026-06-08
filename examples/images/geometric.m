% geometric.m — Image Processing Toolbox Tier-3.
% ----------------------------------------------------------------------
% Resize, rotate, crop and warp an image, then recover an unknown affine
% transform from matched control points with fitgeotform2d.  Resampling
% uses nearest / bilinear / bicubic kernels; warping inverts the 3x3
% transform and bilinearly samples the source.
I = checkerboard(10, 3, 3) * 255;           % 60x60 pattern

% ----- resize / rotate / crop -----------------------------------------
half = imresize(I, 0.5);                     % bicubic by default
fprintf('imresize 0.5     -> %.0fx%.0f\n', size(half, 1), size(half, 2));
rot  = imrotate(I, 30, 'bilinear');          % 'loose' bounding box
fprintf('imrotate 30 deg  -> %.0fx%.0f\n', size(rot, 1), size(rot, 2));
sub  = imcrop(I, [10 10 19 19]);
fprintf('imcrop 20x20     -> %.0fx%.0f\n', size(sub, 1), size(sub, 2));

% ----- affine warp (rotate + scale via an affine2d) -------------------
th = 20 * pi / 180; s = 1.3;
A  = affine2d([s*cos(th) s*sin(th) 0; -s*sin(th) s*cos(th) 0; 0 0 1]);
warped = imwarp(I, A);
fprintf('imwarp (rot+scale) -> %.0fx%.0f\n', size(warped, 1), size(warped, 2));

% ----- recover a transform from control points ------------------------
moving = [0 0; 1 0; 0 1; 1 1];
fixed  = [3 5; 5 5; 3 7; 5 7];               % scale 2x + translate (3,5)
tform  = fitgeotform2d(moving, fixed, 'affine');
fprintf('recovered scale  = %.2f (true 2.00)\n', tform.T(1, 1));
fprintf('recovered tx,ty  = %.2f, %.2f (true 3, 5)\n', tform.T(3, 1), tform.T(3, 2));

% ----- write the resampled / warped result images ---------------------
imwrite(half,   '/tmp/img_geo_resize.png');
imwrite(rot,    '/tmp/img_geo_rotate.png');
imwrite(warped, '/tmp/img_geo_warp.png');
fprintf('wrote /tmp/img_geo_{resize,rotate,warp}.png\n');
