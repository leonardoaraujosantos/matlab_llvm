% transforms.m — Image Processing Toolbox Tier-6.
% ----------------------------------------------------------------------
% The transform / quality / colour / deblur surface: the 2-D DCT and its
% energy compaction, colour-space round-trips, Hough line detection, image
% quality metrics, and Wiener deconvolution.  Colour conversions operate on
% whole M×N×3 images (pipeline-style — element indexing of 3-D arrays is a
% documented gap), so round-trips are checked by reducing through rgb2gray.
I = checkerboard(10, 3, 3) * 255;            % 60x60

% ----- 2-D DCT: energy compaction + perfect reconstruction ------------
D = dct2(I);
fprintf('dct2 reconstruction error = %.2e\n', max(max(abs(idct2(D) - I))));
fprintf('DC coefficient = %.1f (image energy)\n', D(1, 1));

% ----- colour-space round-trips ---------------------------------------
L = zeros(40, 40); L(5:20, 5:20) = 1; L(22:38, 22:38) = 2;
rgb = label2rgb(L);
g  = rgb2gray(rgb);
fprintf('HSV   round-trip error = %.3f\n', max(max(abs(g - rgb2gray(hsv2rgb(rgb2hsv(rgb)))))));
fprintf('YCbCr round-trip error = %.3f\n', max(max(abs(g - rgb2gray(ycbcr2rgb(rgb2ycbcr(rgb)))))));
fprintf('Lab   round-trip error = %.3f\n', max(max(abs(g - rgb2gray(lab2rgb(rgb2lab(rgb)))))));

% ----- Hough line detection -------------------------------------------
edges = zeros(50, 50);
for k = 1:50
    edges(k, k) = 1;                          % a 45-degree diagonal line
end
H = hough(edges);
pk = houghpeaks(H, 1);
fprintf('strongest line at theta index %.0f (of 180)\n', pk(1, 2));

% ----- quality metrics + Wiener deblur --------------------------------
psf  = fspecial('gaussian', 7, 2);
blur = imfilter(I, psf);
fprintf('blurred PSNR vs original  = %.2f dB\n', psnr(blur, I));
deb  = deconvwnr(blur, psf, 0.005);
fprintf('restored PSNR vs original = %.2f dB\n', psnr(deb, I));
fprintf('SSIM restored vs original = %.3f\n', ssim(deb, I));
