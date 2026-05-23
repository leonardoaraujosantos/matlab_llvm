% wavelet shrinkage denoising lifts the SNR
xclean = wnoise(3, 11);
n = length(xclean);
t = (0:n-1);
noise = 2.0*sin(2*pi*t/3.0) + 1.5*cos(2*pi*t/2.0);
xn = xclean + noise;
xd = wdenoise(xn, 6, 'sym4');
snr_in  = 20*log10(norm(xclean)/norm(xn - xclean));
snr_out = 20*log10(norm(xclean)/norm(xd - xclean));
fprintf('signal length: %.0f\n', n);
fprintf('SNR gain dB: %.0f\n', round(snr_out - snr_in));
