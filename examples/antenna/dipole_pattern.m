% ANT-Tier-2 example — half-wave dipole radiation pattern.
%
% Closed-form sinusoidal-current pattern:
%   F(θ) = (cos(½ kL cosθ) − cos(½ kL)) / sin θ
% Half-wave directivity ≈ 2.15 dBi (1.64 linear).  Pattern is
% broadside (peak at θ = π/2), with deep nulls along the wire axis.

freq = 1.0e9;
c0   = 2.99792458e8;
lambda = c0 / freq;
L = 0.5 * lambda;
a = 0.001 * lambda;

p = antennaWirePattern(L, a, 21, freq, 181);
disp(p.Directivity_dBi);    % ~2.15
disp(p.Zin_re);             % ~73.08
disp(p.Zin_im);             % ~42.52
