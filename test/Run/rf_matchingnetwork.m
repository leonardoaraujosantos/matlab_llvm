% L-section matchingnetwork: match 100 Ω source to 50 Ω load at 1 GHz.
%
% Q = sqrt(R_high/R_low - 1) = sqrt(2 - 1) = 1.
% Topology code 0 = shunt-series (source-side shunt, then series toward load).
% Component values for a low-pass L:
%   X_series = Q · R_low = 50 Ω → L_series = X/(2π·f) ≈ 7.96 nH
%   X_shunt  = R_high / Q = 100 Ω → C_shunt = 1/(2π·f·X) ≈ 1.59 pF

m = matchingnetwork(100.0, 0.0, 50.0, 0.0, 1.0e9);
disp(m.Q);              % 1
disp(m.Topology);       % 0
disp(m.L_series_H);     % 7.9577e-9
disp(m.C_shunt_F);      % 1.5915e-12
