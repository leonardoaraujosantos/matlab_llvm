% A4: RGBA PNG round-trip — imread keeps the alpha channel (depth-4).
% Build a 4-channel image, write it as an RGBA PNG (colour type 6), read
% it back, and confirm the alpha survives.  (Note: real MATLAB returns RGB
% with alpha as a separate output; depth-4 is this project's arbitrary-depth
% choice for keeping alpha — see docs/any_shape_roadmap.md A4.)
R = checkerboard(4,2,2)*200;
G = checkerboard(4,2,2)*120;
B = checkerboard(4,2,2)*60;
A = checkerboard(4,2,2)*255;          % alpha channel
rgba = cat(3, R, G, B, A);
fprintf('built d=%.0f\n', size(rgba,3));
imwrite(rgba, '/tmp/rt_rgba.png');
Q = imread('/tmp/rt_rgba.png');
fprintf('read %.0fx%.0fx%.0f\n', size(Q,1), size(Q,2), size(Q,3));
fprintf('rgb maxdiff %.0f\n', max(max(imabsdiff(Q(:,:,1), R))));
fprintf('alpha maxdiff %.0f\n', max(max(imabsdiff(Q(:,:,4), A))));
fprintf('pix(1,1) %.0f %.0f %.0f\n', Q(1,1,1), Q(1,1,2), Q(1,1,3));
fprintf('pix(1,1) a %.0f\n', Q(1,1,4));
