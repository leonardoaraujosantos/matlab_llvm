% rfbudget — Friis cascade over a 3-stage RF chain.
%
% Stages: low-noise amplifier (LNA), then a mixer, then an IF amp.
%
%   Stage     Gain (dB)   NF (dB)   IP3 (dBm)
%   ─────────────────────────────────────────
%   LNA           20         1.5      15
%   Mixer         -8         8        10
%   IF amp        25         5         8
%
% Cascaded gain = 20 - 8 + 25 = 37 dB.
% Cascaded NF via Friis (linear):
%   F1 = 10^0.15 = 1.4125,  G1 = 100
%   F2 = 10^0.8  = 6.3096
%   F3 = 10^0.5  = 3.1623,  G2 = 10^-0.8 = 0.1585
%   F_total = F1 + (F2-1)/G1 + (F3-1)/(G1·G2)
%          = 1.4125 + 5.3096/100 + 2.1623/15.849
%          = 1.4125 + 0.0531 + 0.1364
%          = 1.6020
%   NF_total = 10·log10(1.6020) = 2.0466 dB

g = [20.0; -8.0; 25.0];
n = [1.5;  8.0;  5.0];
ip3 = [15.0; 10.0; 8.0];

r = rfbudgetFriis(g, n, ip3, -30.0, 1.0e6);

disp(r.Gain_dB);              % 37
disp(r.OutputPower_dBm);      % -30 + 37 = 7
disp(r.NumStages);            % 3
% NF cascade — match to 0.01 dB tolerance.  Print rounded to 4 digits
% via direct disp; the exact value will print as 2.0466 (matches the
% Friis formula above).
disp(r.NF_dB);
