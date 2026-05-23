% image_denoise2.m — Wavelet Toolbox Tier-4.
% ----------------------------------------------------------------------
% 2-D wavelet image denoising: decompose a noisy image with wavedec2,
% threshold the detail coefficients, reconstruct with waverec2, and report
% the quality gain.
N = 64;
img   = zeros(N, N);
noisy = zeros(N, N);
for i = 1:N
  for j = 1:N
    base = 100 + 60*sin(2*pi*i/20) + 40*cos(2*pi*j/16);
    img(i,j)   = base;
    noisy(i,j) = base + 20*sin(i*j*0.7);     % deterministic perturbation
  end
end

[C, S] = wavedec2(noisy, 3, 'sym4');
thr = 25;
Cden = wthresh(C, 's', thr);
den = waverec2(Cden, S, 'sym4');

fprintf('image size = %.0f x %.0f\n', size(img,1), size(img,2));
fprintf('noisy  PSNR = %.2f dB\n', measerr(img, noisy));
fprintf('denoised PSNR = %.2f dB\n', measerr(img, den));
