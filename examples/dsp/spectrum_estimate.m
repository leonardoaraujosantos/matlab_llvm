% spectrum_estimate.m — DSP System Toolbox Tier-5.
%
% dsp.SpectrumEstimator runs a Hann-windowed periodogram with exponential
% averaging.  Drive a two-tone signal at 50 Hz and 180 Hz (1 kHz sampling,
% 256-point FFT, 50% overlap) and confirm the PSD shows two clear peaks.

fs = 1000;
N  = 256;
src1 = dsp.SineWave('Frequency',  50);
src1.SampleRate = fs;
src1.SamplesPerFrame = N;
src2 = dsp.SineWave('Frequency', 180);
src2.SampleRate = fs;
src2.SamplesPerFrame = N;

se = dsp.SpectrumEstimator('FFTLength', N);

% Drive multiple frames so the exponential PSD average settles.
psd = zeros(1, N / 2 + 1);
for k = 1:20
    x = src1() + 0.5 * src2();
    psd = se(x);
end

% Expected peak bins (one-sided): round(f/fs * N).
f1 = round( 50 / fs * N);      % 13
f2 = round(180 / fs * N);      % 46
fprintf('PSD length = %d\n', numel(psd));
fprintf('amplitude at 50 Hz bin (%d)  = %.4f\n', f1, psd(f1 + 1));
fprintf('amplitude at 180 Hz bin (%d) = %.4f\n', f2, psd(f2 + 1));

% Both peaks are well above the noise floor between them.
mid = round((f1 + f2) / 2) + 1;
fprintf('floor mid bin = %.4f (much smaller)\n', psd(mid));
