% Smoke for gammams / gammaml + rfbudgetTable scalar metadata.
%
% Device from rf_stability_unconditional.m:
%   s11=0.3, s12=0.05, s21=2, s22=0.4 (K=3.752, mu1=1.842; stable).
% Hand calc:
%   Δ = 0.12 − 0.10 = 0.02
%   B1 = 1 + 0.09 − 0.16 − 0.0004 = 0.9296
%   C1 = 0.3 − 0.02·0.4 = 0.292
%   γ_MS = (0.9296 − √(0.864 − 0.341))/(0.584) = 0.353 (small-mag root)

s11 = complex(0.3, 0.0);
s12 = complex(0.05, 0.0);
s21 = complex(2.0, 0.0);
s22 = complex(0.4, 0.0);
disp(gammams(s11, s12, s21, s22));
disp(gammaml(s11, s12, s21, s22));

% rfbudgetTable scalar metadata.
g = [20.0; -8.0; 25.0];
n = [1.5;  8.0;  5.0];
ip3 = [15.0; 10.0; 8.0];
b = rfbudgetTable(g, n, ip3, -30.0, 1.0e6);
disp(b.NumStages);        % 3
disp(b.InputPower_dBm);   % -30
disp(b.Bandwidth_Hz);     % 1e6
