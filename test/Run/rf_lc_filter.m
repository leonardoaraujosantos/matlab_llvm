% 3-element LC lowpass-Tee filter.
%   L = 7.96 nH, C = 3.18 pF chosen for ~1 GHz cutoff (50 Ω system).
% At passband (100 MHz) S21 ≈ 1; at stopband (3 GHz) S21 rolls off.

L = 7.9577e-9;
Cc = 3.1831e-12;
freqs = [1.0e8; 1.0e9; 3.0e9];

% Topology 0 = Lowpass-Tee (series-L, shunt-C, series-L).
f = rfckt_lcfilter(0, L, Cc, freqs, 50.0);
disp(f.NumPorts);
disp(f.Topology);
disp(tsS21(f));

% Topology 1 = Lowpass-Pi (shunt-C, series-L, shunt-C).
f2 = rfckt_lcfilter(1, Cc, L, freqs, 50.0);
disp(f2.Topology);
disp(tsS21(f2));
