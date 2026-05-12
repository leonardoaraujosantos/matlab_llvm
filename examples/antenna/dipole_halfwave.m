% ANT-Tier-2 example — center-fed thin half-wave dipole at 1 GHz.
%
% Closed-form induced-EMF method (Balanis Eq. 8-60) for a sinusoidal
% current distribution.  Returns Z_in, S11 (50 Ω), VSWR, return loss.
% The textbook half-wave reference is Z_in ≈ 73.13 + j42.55 Ω.

freq = 1.0e9;
c0   = 2.99792458e8;
lambda = c0 / freq;
L = 0.5 * lambda;        % 0.15 m
a = 0.001 * lambda;      % thin-wire, a/λ = 0.001

r = antennaWireSolve(L, a, 21, freq);
disp(r.Zin_re);          % ~73.08
disp(r.Zin_im);          % ~42.52
disp(r.VSWR);            % ~2.18 (referenced to 50 Ω)
disp(r.ReturnLoss_dB);   % ~8.6 dB
