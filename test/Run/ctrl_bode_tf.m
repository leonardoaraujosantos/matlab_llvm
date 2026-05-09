% Tier 2.4 follow-on - bode_tf(b, a, w) frequency response from
% transfer-function polynomial coefficients. Bridges to SPT users who
% work in (b, a) form for filters. Same magnitude / phase output shape
% as bode_ss; complex Horner evaluation.

% --- 1. First-order lowpass H(s) = 1/(s+1) — same closed forms as
%   the bode_ss test (|H(1)| = 1/sqrt(2), phase(1) = -45 deg).
b = [1];
a = [1; 1];
w = [0; 1; 10];
[mag, phase] = bode_tf(b, a, w);
fprintf('|H(0)|   = %.6f, phase = %.4f\n', mag(1, 1), phase(1, 1));    % 1, 0
fprintf('|H(1)|   = %.6f, phase = %.4f\n', mag(2, 1), phase(2, 1));    % 0.7071, -45
fprintf('|H(10)|  = %.6f, phase = %.4f\n', mag(3, 1), phase(3, 1));    % 0.0995, -84.29

% --- 2. Second-order with one zero: H(s) = (s + 2) / (s^2 + 3s + 5).
b2 = [1; 2];
a2 = [1; 3; 5];
[m2, p2] = bode_tf(b2, a2, w);
% At s = 0:  H = 2 / 5 = 0.4
fprintf('\n|H2(0)|  = %.6f, phase = %.4f\n', m2(1, 1), p2(1, 1));     % 0.4, 0
% At s = j:  num = (j + 2), den = (-1 + 3j + 5) = (4 + 3j)
%             H = (2 + j) / (4 + 3j) = (2+j)(4-3j) / (16+9) = (11 - 2j)/25
%             |H| = sqrt(121 + 4)/25 = sqrt(125)/25 = sqrt(5)/5 ~ 0.4472
fprintf('|H2(1)|  = %.6f, phase = %.4f\n', m2(2, 1), p2(2, 1));

% --- 3. Magnitude-only single-return form.
m_only = bode_tf(b, a, w);
fprintf('\nm_only(2) = %.6f\n', m_only(2, 1));    % matches mag(2) above

% --- 4. bode_tf agrees with bode_ss for the same plant.
%   ss(A=-1, B=1, C=1, D=0) is equivalent to H(s) = 1/(s+1).
A = [0-1];
B = [1];
C = [1];
D = [0];
m_ss = bode_ss(A, B, C, D, w);
fprintf('\nbode_ss vs bode_tf at w=1: %.6f vs %.6f\n', m_ss(2, 1), mag(2, 1));
% Should agree exactly.
