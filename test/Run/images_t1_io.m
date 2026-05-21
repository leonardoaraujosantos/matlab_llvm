% Image Processing Tier-1 — I/O round-trip + types + arithmetic + histogram.
I = checkerboard(4, 2, 2) * 255;         % 16x16, values {0,255}
imwrite(I, '/tmp/mlimg_t1.pgm');
J = imread('/tmp/mlimg_t1.pgm');
fprintf('rows    %.0f\n', size(J, 1));
fprintf('maxv    %.0f\n', max(max(J)));
fprintf('mean2   %.2f\n', mean2(J));
fprintf('compl   %.0f\n', max(max(imcomplement(J))));
A = imadd(J, 100);
fprintf('addsat  %.0f\n', max(max(A)));
D = imabsdiff(A, J);
fprintf('absdiff %.0f\n', max(max(D)));
h = imhist(J);
fprintf('hist255 %.0f\n', h(256));
g = im2double(J);
fprintf('im2dbl  %.4f\n', max(max(g)));
u = im2uint8(g);
fprintf('im2u8   %.0f\n', max(max(u)));
