% PNG read/write round-trip (lossless), using the hand-coded codec.
I = checkerboard(4, 2, 2) * 255;            % 16x16 grayscale
imwrite(I, '/tmp/rt_gray.png');
J = imread('/tmp/rt_gray.png');
fprintf('gray %.0fx%.0f maxdiff %.0f\n', size(J,1), size(J,2), max(max(imabsdiff(J, I))));
% RGB round-trip via cat(3,...)
R = checkerboard(4,2,2)*200; G = checkerboard(4,2,2)*120; B = checkerboard(4,2,2)*60;
C = cat(3, R, G, B);
imwrite(C, '/tmp/rt_rgb.png');
D = imread('/tmp/rt_rgb.png');
fprintf('rgb %.0fx%.0fx%.0f\n', size(D,1), size(D,2), size(D,3));
fprintf('chan maxdiff %.0f\n', max(max(imabsdiff(D(:,:,2), C(:,:,2)))));
fprintf('pix(1,1) %.0f %.0f %.0f\n', D(1,1,1), D(1,1,2), D(1,1,3));
