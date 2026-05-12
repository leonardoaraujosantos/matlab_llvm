% ANT-Tier-2 — half-wave dipole pattern.  Directivity ≈ 2.15 dBi
% (1.64 linear) is the textbook reference.  The closed-form
% pattern integral must reproduce this within ~1%.

freq = 1.0e9;
c0   = 2.99792458e8;
lambda = c0 / freq;
L = 0.5 * lambda;
a = 0.001 * lambda;

p = antennaWirePattern(L, a, 21, freq, 181);
disp(round(p.Directivity_dBi * 100));   % 215 (≈ 2.15 dBi)
disp(round(p.Zin_re * 100));            % 7308
disp(round(p.Zin_im * 100));            % 4252
