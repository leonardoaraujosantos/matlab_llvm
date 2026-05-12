% 2nd-order Butterworth low-pass at 1 MHz via tf coefficients and
% writeVerilogATF.
%
%   H(s) = 1 / ( s^2/w0^2 + (sqrt(2)/w0)*s + 1 )
% with w0 = 2*pi*1e6 (1 MHz cutoff).

w0 = 2.0 * 3.141592653589793 * 1.0e6;
Q  = 0.7071067811865476;         % Butterworth-2 Q

num = [1.0];
den = [1.0/(w0*w0); 1.0/(Q*w0); 1.0];

ok = writeVerilogATF(num, den, "biquad_butter.va");
disp(ok);
