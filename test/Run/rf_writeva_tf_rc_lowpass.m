% Tier-2 Verilog-A export — writeVerilogATF on a 1st-order RC lowpass.
%
% RC low-pass: H(s) = 1 / (R*C*s + 1)
% With R = 1e3 Ω, C = 1e-9 F, the cutoff is 1/(2*pi*R*C) ≈ 159 kHz.
%
% MATLAB tf(num, den) convention: descending power of s, so
%   num = [1]
%   den = [R*C, 1]

num = [1.0];
den = [1.0e-6; 1.0];
ok  = writeVerilogATF(num, den, "/tmp/_rf_writeva_tf_rc_lp.va");
disp(ok);              % 1
