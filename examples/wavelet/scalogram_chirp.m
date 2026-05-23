% scalogram_chirp.m — Wavelet Toolbox Tier-3 tracer-bullet.
% ----------------------------------------------------------------------
% Continuous wavelet transform scalogram of a quadratic chirp.  The CWT is
% an FFT-domain convolution of the signal with scaled analytic Morlet
% wavelets; the magnitude |W(a,b)| shows the swept frequency ridge.
fs = 1000;
t  = (0:1023)/fs;
% chirp sweeping 20 -> 200 Hz
x = sin(2*pi*(20*t + 90*t.^2));

[wt, f] = cwt(x, fs);
mag = abs(wt);
fprintf('scalogram size = %.0f x %.0f\n', size(mag,1), size(mag,2));

% ridge at the start vs end of the record
e_start = mag(:, 64);
e_end   = mag(:, 960);
[~, is] = max(e_start);
[~, ie] = max(e_end);
fprintf('ridge freq at t=0.06s : %.0f Hz\n', round(f(is)));
fprintf('ridge freq at t=0.96s : %.0f Hz\n', round(f(ie)));

xr = icwt(wt);
c = sum(x.*xr) / sqrt(sum(x.^2) * sum(xr.^2));
fprintf('icwt shape correlation = %.3f\n', c);
