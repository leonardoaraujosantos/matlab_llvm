% Controllable canonical 2nd-order biquad in state-space form,
% exported via writeVerilogASS.
%
%   H(s) = w0^2 / (s^2 + 2*z*w0*s + w0^2)
% controllable canonical:
%   A = [0   1; -w0^2  -2*z*w0]
%   B = [0; 1]
%   C = [w0^2  0]
%   D = 0
%
% The emitted .va has two `ddt(x[i])` contributions and one V(out)
% contribution — the simulator's analog kernel integrates the ODE
% directly.

w0 = 2.0 * 3.141592653589793 * 1.0e6;
z  = 0.7071067811865476;          % Butterworth-2 damping ratio

A = [0.0,        1.0; ...
     -w0*w0,    -2.0*z*w0];
B = [0.0; 1.0];
Cm = [w0*w0,    0.0];
D = 0.0;

ok = writeVerilogASS(A, B, Cm, D, "biquad_ss_controllable.va");
disp(ok);
