% denoise_signal.m — Wavelet Toolbox Tier-1/2 headline.
% ----------------------------------------------------------------------
% The canonical Donoho-Johnstone denoising demo: synthesise a noisy
% "heavy sine" test signal with wnoise, decompose it with wavedec, then
% reconstruct after wavelet shrinkage (universal soft threshold at the
% MAD-estimated noise level) with wdenoise.  The SNR improvement is the
% denoising payoff.  No external dependency — wnoise / wavedec / wthresh /
% thselect / wnoisest / waverec are all hand-coded over the shipped conv.
xclean = wnoise(3, 11);              % heavy sine, length 2048
n = length(xclean);

% additive coloured perturbation (deterministic for a reproducible demo)
t = (0:n-1);
noise = 2.0*sin(2*pi*t/3.0) + 1.5*cos(2*pi*t/2.0);
xn = xclean + noise;

% 5-level sym4 decomposition, then automatic shrinkage denoising
[C, L] = wavedec(xn, 5, 'sym4');
sigma  = wnoisest(C, L, 1);
thr    = sigma * thselect(detcoef(C, L, 1), 'sqtwolog');
fprintf('estimated noise sigma = %.4f\n', sigma);
fprintf('universal threshold   = %.4f\n', thr);

xd = wdenoise(xn, 6, 'sym4');

snr_in  = 20*log10(norm(xclean)/norm(xn - xclean));
snr_out = 20*log10(norm(xclean)/norm(xd - xclean));
fprintf('input  SNR = %.2f dB\n', snr_in);
fprintf('output SNR = %.2f dB\n', snr_out);
fprintf('SNR improvement = %.2f dB\n', snr_out - snr_in);
