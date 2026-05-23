% wdenoise_compare.m — Wavelet Toolbox Tier-2.
% ----------------------------------------------------------------------
% Compare threshold-selection rules for wavelet shrinkage and report the
% recovered quality with measerr (PSNR).
xclean = wnoise(4, 11);             % Doppler
n = length(xclean);
t = (0:n-1);
xn = xclean + 1.2*sin(2*pi*t/2.5) + 0.8*cos(2*pi*t/3.5);

xu = wden(xn, 'sqtwolog', 's', 'sln', 5, 'sym4');
xr = wden(xn, 'rigrsure', 's', 'sln', 5, 'sym4');

fprintf('universal  PSNR = %.2f dB\n', measerr(xclean, xu));
fprintf('SURE       PSNR = %.2f dB\n', measerr(xclean, xr));
fprintf('noisy      PSNR = %.2f dB\n', measerr(xclean, xn));
