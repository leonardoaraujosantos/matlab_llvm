% T-section + Pi-section matching networks.
%
% T: match 100→50 Ω at 1 GHz with Q=3.  R_virtual = max(100,50)·(9+1) = 1000 Ω.
% Pi: same impedances + Q.  R_virtual = min(100,50)/(9+1) = 5 Ω.

t = matchingnetworkT(100.0, 0.0, 50.0, 0.0, 1.0e9, 3.0);
disp(t.Topology);
disp(t.R_virtual);

p = matchingnetworkPi(100.0, 0.0, 50.0, 0.0, 1.0e9, 3.0);
disp(p.Topology);
disp(p.R_virtual);

% Component values vary with the algorithm's internal Q_a / Q_b split.
% Verify they're all positive (a sane match design).
disp(t.L1_series_H > 0.0);
disp(t.L2_series_H > 0.0);
disp(t.C_shunt_F > 0.0);
disp(p.L_series_H > 0.0);
disp(p.C1_shunt_F > 0.0);
disp(p.C2_shunt_F > 0.0);
