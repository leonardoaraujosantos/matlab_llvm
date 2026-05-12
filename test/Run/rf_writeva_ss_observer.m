% Tier-3 Verilog-A export — writeVerilogASS on a 3rd-order Butterworth
% in observable canonical form (mostly-zero A but with feed-through).
%
% Observable canonical form for H(s) = 1 / (s^3 + 2s^2 + 2s + 1):
%   A = [0, 0, -1; 1, 0, -2; 0, 1, -2]
%   B = [1; 0; 0]
%   C = [0, 0, 1]
%   D = 0.0

A = [0.0,  0.0, -1.0; ...
     1.0,  0.0, -2.0; ...
     0.0,  1.0, -2.0];
B = [1.0; 0.0; 0.0];
Cm = [0.0, 0.0, 1.0];
D = 0.0;
ok = writeVerilogASS(A, B, Cm, D, "/tmp/_rf_writeva_ss_observer.va");
disp(ok);              % 1
