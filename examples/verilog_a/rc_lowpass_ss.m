% RC low-pass in continuous state-space form, exported via
% writeVerilogASS.
%
%   dx/dt = -1/(R*C) * x + 1/(R*C) * u
%   y     = x
% A = [-1/RC],  B = [1/RC],  C = [1],  D = 0.

R = 1.0e3;
C = 1.0e-9;
inv_RC = 1.0 / (R*C);

A = [-inv_RC];
B = [inv_RC];
Cm = [1.0];
D = 0.0;

ok = writeVerilogASS(A, B, Cm, D, "rc_lowpass_ss.va");
disp(ok);
