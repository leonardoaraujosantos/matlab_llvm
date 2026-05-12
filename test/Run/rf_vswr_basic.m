% Smoke test: VSWR from a synthetic 1-frequency 2-port.
%
% Input reflection coefficient = 0.2 + 0j → VSWR = 1.5.
% Output reflection coefficient = 0.0 + 0j → VSWR = 1.0.
% Round-trip via gammaIn with a matched load checks the algebra.

% Build a synthetic 2-port: s11=0.2, s22=0.0, s12=0, s21=1 (real),
% then for a matched load (zl = z0 = 50), gammaIn collapses to s11.
s11 = complex(0.2, 0.0);
s12 = complex(0.0, 0.0);
s21 = complex(1.0, 0.0);
s22 = complex(0.0, 0.0);
gin = gammaIn(s11, s12, s21, s22, 50.0, 50.0);
v = vswr(gin);
disp(v(1));   % (1 + 0.2)/(1 - 0.2) = 1.5

% Round-trip via the matched termination on the output port:
% gammaOut with zs = z0 collapses to s22 = 0 → VSWR = 1.0.
gout = gammaOut(s11, s12, s21, s22, 50.0, 50.0);
v2 = vswr(gout);
disp(v2(1));  % 1.0
