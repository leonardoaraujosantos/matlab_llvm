% DSP System Toolbox Tier-5 — dsp.SpectrumEstimator + dsp.AsyncBuffer.
%
% Drive a known tone (50 Hz at 1 kHz, 128-point FFT) through the spectrum
% estimator; the peak bin should be ~50/1000*128 = 6.  AsyncBuffer is a
% FIFO that preserves insertion order across write/read calls.
src = dsp.SineWave('Frequency', 50);
src.SampleRate = 1000;
src.SamplesPerFrame = 200;
x = src();

se = dsp.SpectrumEstimator('FFTLength', 128);
for k = 1:10
    psd = se(x);                       % drive long enough to settle
end
[~, ki] = max(psd);
fprintf('FFT peak bin = %d (expected ~6 for 50/1000*128)\n', ki - 1);
fprintf('PSD length   = %d (expected 65 = FFT/2 + 1)\n', numel(psd));

% AsyncBuffer FIFO: push two batches, pull as one batch.
ab = dsp.AsyncBuffer('Capacity', 512);
ab.write([1 2 3 4 5]);
ab.write([6 7 8]);
out = ab.read(6);
fprintf('fifo:  %.0f %.0f %.0f %.0f %.0f %.0f\n', ...
        out(1), out(2), out(3), out(4), out(5), out(6));
