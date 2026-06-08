% ble_aoa_estimate.m — Bluetooth Toolbox Phase-C (Tier-5/6).
% ----------------------------------------------------------------------
% Bluetooth LE direction finding: a constant-tone-extension (CTE) signal
% arriving at a uniform linear array carries an inter-element phase slope set
% by its angle of arrival.  Sweep a set of true angles, synthesize the
% half-wavelength ULA snapshot for each, and recover the angle with
% bleAngleEstimate.  Then measure the carrier-frequency offset and GFSK peak
% deviation of an LE waveform with the Tier-6 measurement helpers.

numElements = 8;
spacing     = 0.5;                  % half-wavelength array
k           = (0:numElements-1)';

fprintf('Bluetooth LE angle-of-arrival estimation (8-element ULA):\n');
trueAngles = [-45 -20 0 20 45 60];
for i = 1:6
    a   = trueAngles(i);
    phi = 2*pi*spacing*sin(a*pi/180)*k;
    sv  = cos(phi) + 1j*sin(phi);
    fprintf('  true %3.0f deg -> estimate %5.1f deg\n', a, bleAngleEstimate(sv, spacing));
end

% Tier-6 measurements on a generated LE waveform.
rng(1);
wf = bleWaveformGenerator(round(rand(160,1)), 'LE1M', 8, 17);
fprintf('LE1M peak frequency deviation: %.4f cycles/sample (h/(2*sps)=0.03125)\n', ...
        bluetoothFrequencyDeviation(wf, 8));
fprintf('LE1M residual carrier offset:  %.4f cycles/sample (~0, balanced GFSK)\n', ...
        bluetoothFrequencyOffset(wf));
