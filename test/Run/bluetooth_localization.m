% Bluetooth Toolbox Tier-5/6 — localization + test & measurement.
%   bleAngleEstimate (angle-of-arrival from a uniform linear array snapshot),
%   bluetoothFrequencyOffset (carrier-frequency-offset estimate) and
%   bluetoothFrequencyDeviation (GFSK peak deviation).
k = (0:7)';

% AoA: synthesize the half-wavelength ULA response at a known angle, recover.
phi30 = 2*pi*0.5*sin(30*pi/180)*k;
sv30  = cos(phi30) + 1j*sin(phi30);
fprintf('AoA(30 deg)  estimate: %.1f\n', bleAngleEstimate(sv30, 0.5));

phin15 = 2*pi*0.5*sin(-15*pi/180)*k;
svn15  = cos(phin15) + 1j*sin(phin15);
fprintf('AoA(-15 deg) estimate: %.1f\n', bleAngleEstimate(svn15, 0.5));

% Carrier frequency offset: a pure tone at 0.05 cycles/sample.
n    = (0:199)';
tone = cos(2*pi*0.05*n) + 1j*sin(2*pi*0.05*n);
fprintf('CFO estimate: %.4f\n', bluetoothFrequencyOffset(tone));

% GFSK peak frequency deviation of an LE1M waveform = h/(2*sps) = 0.5/16.
rng(2);
wf = bleWaveformGenerator(round(rand(80,1)), 'LE1M', 8, 17);
fprintf('freq deviation: %.4f\n', bluetoothFrequencyDeviation(wf, 8));
