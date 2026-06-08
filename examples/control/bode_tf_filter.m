% bode_tf bridges SPT-designed filter coefficients to CST analysis.
% Tier-2.4 follow-on demo.
%
% The Signal Processing Toolbox designers (butter / cheby1 / cheby2)
% return (b, a) polynomial coefficients of the analog or digital
% transfer function. With bode_tf(b, a, w) the same coefficients
% feed straight into a Bode-style frequency-response analysis.

% --- 1. First-order analog lowpass H(s) = 1/(s + 1).
% Closed form: |H(1)| = 1/sqrt(2) (-3 dB at the corner).
b1 = [1];
a1 = [1; 1];
w  = [0.01; 0.1; 1.0; 10.0; 100.0];
[mag1, ph1] = bode_tf(b1, a1, w);
fprintf('first-order lowpass H(s) = 1/(s+1)\n');
fprintf('   w     |H|       phase\n');
fprintf('  0.01  %.6f  %.4f\n', mag1(1, 1), ph1(1, 1));
fprintf('  1.00  %.6f  %.4f\n', mag1(3, 1), ph1(3, 1));   % 0.7071, -45
fprintf('  100   %.6f  %.4f\n', mag1(5, 1), ph1(5, 1));

% --- 2. Notch filter via biquad: H(s) = (s^2 + wn^2) / (s^2 + 2*zeta*wn*s + wn^2).
% wn = 5, zeta = 0.1.
% At s = j*wn, H = 0 (notch).
% At s = 0, H = 1 (DC pass-through).
% At s = j*Inf, H = 1 (high-frequency pass-through).
wn   = 5;
zeta = 0.1;
b2 = [1; 0; wn*wn];
a2 = [1; 2*zeta*wn; wn*wn];
w2 = [0.1; 1; 5; 25; 100];
mn = bode_tf(b2, a2, w2);
fprintf('\nnotch filter (wn = %g, zeta = %g):\n', wn, zeta);
fprintf('  w = %.1f   |H| = %.6f  (DC pass)\n',     w2(1, 1), mn(1, 1));
fprintf('  w = %.1f   |H| = %.6f  (DC pass)\n',     w2(2, 1), mn(2, 1));
fprintf('  w = %.1f   |H| = %.6f  (notch -> 0)\n',  w2(3, 1), mn(3, 1));   % 0
fprintf('  w = %.1f   |H| = %.6f  (HF pass)\n',     w2(4, 1), mn(4, 1));
fprintf('  w = %.1f   |H| = %.6f  (HF pass)\n',     w2(5, 1), mn(5, 1));

% --- 3. Cross-check: bode_tf and bode_ss must agree for equivalent
% representations of the same plant.
%   H(s) = 1/(s+1) <=> ss(A=-1, B=1, C=1, D=0).
A3 = [0-1];
B3 = [1];
C3 = [1];
D3 = [0];
mss = bode_ss(A3, B3, C3, D3, w);
fprintf('\nbode_tf vs bode_ss at w = 1.0:\n');
fprintf('  bode_tf: %.6f\n', mag1(3, 1));
fprintf('  bode_ss: %.6f\n', mss(3, 1));
% Difference must be ~ 0 (numerically identical).

% ----- plot the notch magnitude response -----------------------------
wsw  = logspace(-1, 2, 400)';
mnsw = bode_tf(b2, a2, wsw);
figure; plot(log10(wsw), 20*log10(mnsw + 1e-6), 'b-'); grid on;
xlabel('log_{10} \omega (rad/s)'); ylabel('|H| (dB)');
title('notch filter (wn=5, zeta=0.1)');
saveas(gcf, '/tmp/ctrl_notch.png');
