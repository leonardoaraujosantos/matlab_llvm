% bode(sys) — frequency response of a SISO LTI system.
%
% Tier 2.4 (control_toolbox_roadmap.md §3.4) — NOT YET SHIPPED.
% Internally implemented as polyval(num, j*w) / polyval(den, j*w) for
% tf models, or solving (j*w*I - A) * X = B then C*X + D for ss
% models. Both paths sit on existing primitives — no new linalg work
% beyond complex matrix solve.

% --- 1. First-order lowpass.  G(s) = 1 / (tau*s + 1), tau = 0.5.
%   Corner frequency w_c = 1/tau = 2 rad/s.
%   At w = w_c:    |G| = 1/sqrt(2)  →  -3.01 dB,  phase = -45 deg.
%   At w = 10*w_c: |G| ≈ 0.0995     →  -20.04 dB (high-freq -20 dB/dec).
tau = 0.5;
G   = tf(1, [tau 1]);
w   = logspace(-1, 2, 200);
[mag, phase, wout] = bode(G, w);

% Find the index closest to the corner frequency (w = 2).
[~, idx_c] = min(abs(wout - 2));
disp('|G| at corner (linear, ~0.7071):');
disp(mag(idx_c));
disp('phase at corner (deg, ~-45):');
disp(phase(idx_c));

% --- 2. Bandwidth.  -3 dB frequency.
bw = bandwidth(G);
disp('bandwidth (rad/s, ~2.0):');
disp(bw);

% --- 3. DC gain.
g0 = dcgain(G);
disp('DC gain (~1.0):');
disp(g0);

% --- 4. Open-loop type-1 system — gain and phase margins.
%   L(s) = 4 / (s * (s + 2)).  Should have positive Pm and Inf Gm.
L = tf(4, [1 2 0]);
[Gm, Pm, Wcg, Wcp] = margin(L);
disp('phase margin (deg):');
disp(Pm);
disp('gain crossover (rad/s):');
disp(Wcp);
