% Butterworth lowpass — design + zero-phase apply.
%
% Build a 1-second linear chirp from 10 Hz to 200 Hz at 1 kHz sampling,
% then drop everything above 100 Hz with a 6th-order Butterworth LP.
% filtfilt applies the filter forward then backward to give zero phase,
% with steady-state initial conditions so a constant DC component is
% preserved exactly.

fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 10, 1, 200);

% Wn is normalised to Nyquist (fs/2). 100 / (fs/2) = 0.2.
[b, a] = butter(6, 100/(fs/2));
y      = filtfilt(b, a, x);

% Energy in the second half of the chirp (above 100 Hz) is mostly
% gone after lowpass filtering. Compare RMS of first 200 samples
% (10..50 Hz, still in the passband) vs last 200 samples (160..200 Hz,
% in the stopband).
xa = x(1:200);
xb = x(800:999);
ya = y(1:200);
yb = y(800:999);

fprintf('input  passband rms: %.4f\n',   rms(xa));
fprintf('input  stopband rms: %.4f\n',   rms(xb));
fprintf('output passband rms: %.4f\n',   rms(ya));
fprintf('output stopband rms: %.4f\n',   rms(yb));
fprintf('stopband attenuation: %.1f dB\n', 20 * log10(rms(xb) / rms(yb)));
