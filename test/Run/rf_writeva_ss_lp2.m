% Tier-3 Verilog-A export — writeVerilogASS on a 2nd-order low-pass
% in controllable canonical form.
%
%   H(s) = w0^2 / (s^2 + 2*z*w0*s + w0^2)
% with w0 = 2*pi*1e6, z = 0.707.  Controllable canonical:
%   A = [0 1; -w0^2 -2*z*w0]
%   B = [0; 1]
%   C = [w0^2 0]
%   D = 0

w0 = 2.0 * 3.141592653589793 * 1.0e6;
z  = 0.7071067811865476;

A = [0.0, 1.0; -w0*w0, -2.0*z*w0];
B = [0.0; 1.0];
Cm = [w0*w0, 0.0];
D = 0.0;
ok = writeVerilogASS(A, B, Cm, D, "/tmp/_rf_writeva_ss_lp2.va");
disp(ok);              % 1
