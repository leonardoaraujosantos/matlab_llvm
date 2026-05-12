% Tier-3 Verilog-A export — writeVerilogASS on a 1st-order low-pass.
%
% RC low-pass in state-space form:
%   dx/dt = -1/(R*C) * x + 1/(R*C) * u
%   y     = x
% With R = 1e3 Ω, C = 1e-9 F:  1/(RC) = 1e6.
% A = [-1e6], B = [1e6], C = [1], D = 0.

A = [-1.0e6];
B = [1.0e6];
Cm = [1.0];
D = 0.0;
ok = writeVerilogASS(A, B, Cm, D, "/tmp/_rf_writeva_ss_lp1.va");
disp(ok);              % 1
