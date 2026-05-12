% 1st-order RC low-pass via tf(num, den) and writeVerilogATF.
%
%   H(s) = 1 / (R*C*s + 1)
% With R = 1 kΩ, C = 1 nF the −3 dB cutoff is 1 / (2*pi*R*C) ≈ 159 kHz.
%
% MATLAB tf(num, den) stores coefficients in DESCENDING power of s.
% The emitter reverses to ASCENDING for laplace_nd internally.

R = 1.0e3;
C = 1.0e-9;
num = [1.0];
den = [R*C; 1.0];

ok = writeVerilogATF(num, den, "rc_lowpass.va");
disp(ok);
