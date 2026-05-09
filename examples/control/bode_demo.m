% bode_ss frequency response demo — Tier 2.4.
%
% [mag, phase] = bode_ss(A, B, C, D, w) gives the magnitude / phase of
% H(jw) = C (jw I - A)^{-1} B + D for a SISO state-space plant.
% Together with `step_ss`, this is the analysis surface a control
% engineer needs for SISO loop-shaping and stability margin analysis.

% --- 1. Mass-spring-damper plant (lightly damped).
%   xdot = [0 1; -k/m -c/m] x + [0; 1/m] u,  y = [1 0] x.
% wn = sqrt(k/m) = 3,  zeta = c/(2*sqrt(km)) = 0.1.
m = 1.0;  k = 9.0;  c = 0.6;
A = [0 1;  0-k/m,  0-c/m];
B = [0;  1/m];
C = [1, 0];
D = [0];

% Spot-check at four frequencies — the resonance is near w = wn = 3.
w = [0.1; 1.0; 3.0; 10.0];
[mag, phase] = bode_ss(A, B, C, D, w);
fprintf('|H(0.1)|  = %.6f, phase = %.4f deg\n', mag(1, 1), phase(1, 1));
fprintf('|H(1.0)|  = %.6f, phase = %.4f deg\n', mag(2, 1), phase(2, 1));
fprintf('|H(3.0)|  = %.6f, phase = %.4f deg\n', mag(3, 1), phase(3, 1));
fprintf('|H(10.0)| = %.6f, phase = %.4f deg\n', mag(4, 1), phase(4, 1));

% --- 2. DC gain via bode_ss(0) matches the closed-form -C A^{-1} B = 1/k.
DC = bode_ss(A, B, C, D, [0]);
DC_pred = (0 - C) * inv(A) * B;
fprintf('\nDC gain (bode_ss):    %.6f\n', DC(1, 1));
fprintf('DC gain (closed):     %.6f\n', DC_pred(1, 1));
fprintf('1/k:                  %.6f\n', 1/k);

% --- 3. Resonant-peak magnitude — for a lightly-damped 2nd-order plant,
%   peak |H| ~ 1/(k * 2 * zeta * sqrt(1-zeta^2)). For our values
%   (1/k = 1/9 = 0.111, zeta = 0.1):  peak ~ 0.557.
%
% Verify by sampling at the theoretical peak frequency.
wn   = sqrt(k / m);
zeta = c / (2 * sqrt(k * m));
wp   = wn * sqrt(1 - 2 * zeta * zeta);
mpeak = bode_ss(A, B, C, D, [wp]);
fprintf('\nresonant peak: |H(wp)| = %.6f  at  wp = %.4f rad/s\n', ...
        mpeak(1, 1), wp);
fprintf('zeta = %.4f, peak prediction 1/(k*2*zeta*sqrt(1-zeta^2)) = %.6f\n', ...
        zeta, 1 / (k * 2 * zeta * sqrt(1 - zeta * zeta)));
