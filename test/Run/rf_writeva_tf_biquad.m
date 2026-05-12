% Tier-2 Verilog-A export — writeVerilogATF on a 2nd-order resonant
% biquad with Q-controlled damping.
%
%   H(s) = 1 / ( s^2/w0^2 + s/(Q*w0) + 1 )
%
% With w0 = 2*pi*1e6 (1 MHz) and Q = 0.707 (Butterworth-2),
%   num = [1]
%   den = [1/w0^2; 1/(Q*w0); 1]

w0 = 2.0 * 3.141592653589793 * 1.0e6;
Q  = 0.7071067811865476;
num = [1.0];
den = [1.0/(w0*w0); 1.0/(Q*w0); 1.0];
ok  = writeVerilogATF(num, den, "/tmp/_rf_writeva_tf_biquad.va");
disp(ok);              % 1
