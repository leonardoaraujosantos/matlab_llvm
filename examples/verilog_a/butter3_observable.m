% 3rd-order Butterworth low-pass in observable canonical form,
% exported via writeVerilogASS.
%
%   H(s) = 1 / (s^3 + 2 s^2 + 2 s + 1)     (normalized to w0 = 1)
%
% Observable canonical:
%   A = [0  0  -1; 1  0  -2; 0  1  -2]
%   B = [1; 0; 0]
%   C = [0  0  1]
%   D = 0

A = [0.0,  0.0, -1.0; ...
     1.0,  0.0, -2.0; ...
     0.0,  1.0, -2.0];
B = [1.0; 0.0; 0.0];
Cm = [0.0, 0.0, 1.0];
D = 0.0;

ok = writeVerilogASS(A, B, Cm, D, "butter3_observable.va");
disp(ok);
