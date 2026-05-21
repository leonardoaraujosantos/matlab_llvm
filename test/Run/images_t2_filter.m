% Image Processing Tier-2 — filtering + enhancement.
I = checkerboard(8, 2, 2) * 255;          % 32x32
G = fspecial('gaussian', 5, 1.0);
fprintf('gsum    %.4f\n', sum(sum(G)));
S = fspecial('sobel');
fprintf('sobel11 %.0f\n', S(1, 1));
av = fspecial('average', 3);
fprintf('avg     %.4f\n', av(2, 2));
B = imgaussfilt(I, 2.0);
fprintf('gauss   %.0fx%.0f\n', size(B, 1), size(B, 2));
F = imfilter(I, G);
fprintf('filt    %.0fx%.0f\n', size(F, 1), size(F, 2));
M = medfilt2(I, [3 3]);
fprintf('median  %.0fx%.0f\n', size(M, 1), size(M, 2));
H = histeq(I);
fprintf('histeq  %.0f\n', max(max(H)));
b = imboxfilt(I, 3);
fprintf('box     %.0fx%.0f\n', size(b, 1), size(b, 2));
