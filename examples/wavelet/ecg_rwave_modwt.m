% ecg_rwave_modwt.m — Wavelet Toolbox Tier-4 tracer-bullet.
% ----------------------------------------------------------------------
% MODWT-based R-wave detection (the UG "R Wave Detection in the ECG"
% pipeline): the MODWT is shift-invariant, so a multiresolution analysis
% isolates the QRS-energy scale, and the R-waves are the peaks of that
% reconstructed detail band.
fs = 250;
N  = 1000;
t  = (0:N-1)/fs;
% synthetic ECG: periodic QRS spikes (~1.25 Hz heart rate) + baseline wander
hr = 1.25;
ecg = 0.1*sin(2*pi*0.3*t);                 % baseline wander
for k = 1:N
  ph = mod(t(k)*hr, 1.0);
  u  = (ph - 0.5) * 40;
  ecg(k) = ecg(k) + exp(-u * u);            % QRS spike
end

w   = modwt(ecg, 'sym4', 5);
mra = modwtmra(w, 'sym4');
% QRS energy concentrates in detail levels 3-4
qrs = mra(3, :) + mra(4, :);
fprintf('mra rows = %.0f\n', size(mra,1));
fprintf('mra reconstructs signal: %.2e\n', max(abs(ecg - sum(mra,1))));

pk = findpeaks(qrs);
fprintf('R-wave candidates found = %.0f\n', length(pk));
