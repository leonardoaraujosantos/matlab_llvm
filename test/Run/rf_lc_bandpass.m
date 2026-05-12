% 4-element LC bandpass-Tee filter centered around 1 GHz.
%   ω0 = 2π·1e9.  L = 1/(ω0·sqrt(z0)·sqrt(BW_fraction))
%   For a 100-MHz BW around 1 GHz on 50 Ω: pick L = 80 nH, C = 0.317 pF
%   (resonant at 1 GHz when ω0² = 1/LC = 4π²·1e18).
%   1/LC = 1/(80e-9 · 0.317e-12) = 3.94e19 = (2π·1e9)² ≈ 3.95e19 ✓

L1 = 80e-9;
C1 = 0.317e-12;
L2 = 0.317e-9;     % swap for the shunt branch
C2 = 80e-12;

freqs = [3.0e8; 1.0e9; 3.0e9];

% Topology 4 = Bandpass-Tee.  At ω0 = 1 GHz the series-LC branches go
% to Z=0 (pass-through) and the shunt-LC-parallel sees Y=0 (open).
f = rfckt_lcfilter4(4, L1, C1, L2, C2, freqs, 50.0);
disp(f.NumPorts);
disp(f.Topology);
% At 1 GHz: |S21| should be ~1.0 (passband).  Off-center frequencies
% should attenuate.
disp(tsS21(f));

% Topology 6 = Bandstop-Tee — opposite: ω0 = 1 GHz should be REJECTED.
f6 = rfckt_lcfilter4(6, L1, C1, L2, C2, freqs, 50.0);
disp(f6.Topology);
disp(tsS21(f6));
