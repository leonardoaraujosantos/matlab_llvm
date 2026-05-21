% Image Processing Tier-6 — transforms / quality / ROI / colour / block / deblur.
I = checkerboard(8, 2, 2) * 255;             % 32x32
fprintf('immse_self %.4f\n', immse(I, I));
fprintf('psnr %.2f\n', psnr(I + 10, I));
% colour round-trips reduced through rgb2gray (3-D indexing unsupported)
L = zeros(8, 8); L(2:4, 2:4) = 1; L(5:7, 5:7) = 2;
rgb = label2rgb(L); g1 = rgb2gray(rgb);
fprintf('hsv_err %.3f\n', max(max(abs(g1 - rgb2gray(hsv2rgb(rgb2hsv(rgb)))))));
fprintf('lab_err %.3f\n', max(max(abs(g1 - rgb2gray(lab2rgb(rgb2lab(rgb)))))));
fprintf('hsv_size %.0f\n', size(rgb2hsv(rgb), 3));
% transforms
fprintf('dct2_err %.6f\n', max(max(abs(idct2(dct2(I)) - I))));
S = zeros(20, 20); S(:, 10) = 1;
pk = houghpeaks(hough(S), 1);
fprintf('hough_peak_theta %.0f\n', pk(1, 2));
rd = radon(I, [0 45 90]);
fprintf('radon_size %.0fx%.0f\n', size(rd, 1), size(rd, 2));
% ROI + block
fprintf('poly2mask %.0f\n', sum(sum(poly2mask([2 8 8 2], [2 2 8 8], 10, 10))));
c = im2col(I, [4 4]);
fprintf('im2col %.0fx%.0f\n', size(c, 1), size(c, 2));
% deblur
psf = fspecial('gaussian', 5, 1);
deb = deconvwnr(imfilter(I, psf), psf, 0.01);
fprintf('deconvwnr %.0fx%.0f\n', size(deb, 1), size(deb, 2));
