% filtering.m — Image Processing Toolbox Tier-2.
% ----------------------------------------------------------------------
% The "Designing and Implementing Linear Filters" + enhancement workflow:
% Gaussian smoothing with fspecial+imfilter, salt-and-pepper denoising
% with a median filter, unsharp sharpening, and histogram equalisation.
rng(7);
I = checkerboard(12, 2, 2) * 255;           % 48x48 test pattern

% ----- linear smoothing -----------------------------------------------
g = fspecial('gaussian', 7, 1.5);
smooth = imfilter(I, g);
fprintf('gaussian kernel sums to %.4f\n', sum(sum(g)));
fprintf('smoothing reduced std: %.1f -> %.1f\n', std2(I), std2(smooth));

% ----- median denoising of salt & pepper ------------------------------
noisy = imnoise(I, 'salt & pepper', 0.15);
clean = medfilt2(noisy, [3 3]);
fprintf('median filter: noisy std %.1f -> clean std %.1f\n', std2(noisy), std2(clean));

% ----- sharpening + histogram equalisation ----------------------------
sharp = imsharpen(I);
fprintf('sharpen mean = %.1f\n', mean2(sharp));
eq = histeq(I);
fprintf('histeq output range = [%.0f %.0f]\n', min(min(eq)), max(max(eq)));

% ----- write before/after result images -------------------------------
imwrite(noisy,  '/tmp/img_filt_noisy.png');
imwrite(clean,  '/tmp/img_filt_denoised.png');
imwrite(smooth, '/tmp/img_filt_gaussian.png');
imwrite(eq,     '/tmp/img_filt_histeq.png');
fprintf('wrote /tmp/img_filt_{noisy,denoised,gaussian,histeq}.png\n');
