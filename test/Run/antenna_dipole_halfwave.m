% ANT-Tier-2 — closed-form thin half-wave dipole reference.
%
% At resonance (L = λ/2), the textbook input impedance of a center-fed
% thin dipole is Z_in ≈ 73.13 + j42.55 Ω.  The EMF closed-form must
% reproduce this to a few decimals.
%
% Geometry: L = λ/2, a = 0.001λ, freq = 1 GHz → λ = 0.3 m, L = 0.15 m.

freq = 1.0e9;
c0   = 2.99792458e8;
lambda = c0 / freq;
L = 0.5 * lambda;
a = 0.001 * lambda;

r = antennaWireSolve(L, a, 21, freq);
disp(round(r.Zin_re * 100));        % 7308 (≈ 73.08)
disp(round(r.Zin_im * 100));        % 4252 (≈ 42.52)
disp(round(r.VSWR * 100));          % 218  (≈ 2.18)
disp(round(r.ReturnLoss_dB * 10));  % 86   (≈ 8.6 dB)
