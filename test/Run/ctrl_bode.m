% Tier 2.4 (CST roadmap §3.4) — SISO state-space frequency response.
% [mag, phase] = bode_ss(A, B, C, D, w) returns linear magnitude and
% phase in degrees. Uses the real 2n*2n block decomposition of the
% complex linear system  (jw I - A) X = B  so it sits on the existing
% pivoted-LU helper without needing complex linalg.

% --- 1. First-order lowpass H(s) = 1 / (tau*s + 1), tau = 1.
A = [0-1];
B = [1];
C = [1];
D = [0];

% Three checkpoint frequencies.
w = [0.0; 1.0; 10.0];
[mag, phase] = bode_ss(A, B, C, D, w);
fprintf('|H(w=0)|   = %.6f\n', mag(1, 1));      % 1.000000 (DC gain)
fprintf('phase(0)   = %.6f\n', phase(1, 1));    % 0.000000
fprintf('|H(w=1)|   = %.6f\n', mag(2, 1));      % 0.707107 (-3 dB at corner)
fprintf('phase(1)   = %.6f\n', phase(2, 1));    % -45.000000
fprintf('|H(w=10)|  = %.6f\n', mag(3, 1));      % 0.099504 (~ 1/w)
fprintf('phase(10)  = %.6f\n', phase(3, 1));    % -84.289407

% --- 2. Magnitude-only single-return form.
m = bode_ss(A, B, C, D, w);
fprintf('m(2)       = %.6f\n', m(2, 1));        % 0.707107 (matches mag above)

% --- 3. Second-order underdamped: H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2).
%   wn = 1, zeta = 0.5. State-space realisation:
%     A = [0 1; -wn^2  -2*zeta*wn], B = [0; wn^2], C = [1 0], D = 0.
%   Resonance peak near w = wn, |H(wn)| = 1/(2*zeta) = 1.
wn = 1; zeta = 0.5;
A2 = [0 1; 0-wn*wn, 0-2*zeta*wn];
B2 = [0; wn*wn];
C2 = [1, 0];
D2 = [0];
% Magnitude at the natural frequency: theoretical resonant peak occurs
% slightly below wn; at exactly wn,  |H(jwn)| = 1/(2*zeta).
mag_wn = bode_ss(A2, B2, C2, D2, [wn]);
fprintf('|H(wn)|    = %.6f\n', mag_wn(1, 1));   % 1.000000 (= 1/(2*zeta))

% --- 4. DC gain via bode_ss at w = 0 matches -C A^{-1} B.
DC = bode_ss(A2, B2, C2, D2, [0]);
DC_pred = (0 - C2) * inv(A2) * B2;
fprintf('DC gain    = %.6f\n', DC(1, 1));         % 1.000000
fprintf('DC pred    = %.6f\n', DC_pred(1, 1));    % 1.000000
