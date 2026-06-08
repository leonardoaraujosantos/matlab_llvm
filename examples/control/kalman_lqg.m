% Tier 4.2 — steady-state Kalman filter + LQG separation principle.
% kalman_L(A, G, C, Qn, Rn) returns the continuous Kalman gain;
% kalmd_L(...) the discrete analog. Both use the LQR/Kalman duality:
%      L = (lqr(A', C', G·Qn·G', Rn))'.
% The estimator dynamics  xdot_hat = (A − L·C) x_hat + L y  are
% Hurwitz (continuous) / Schur-stable (discrete).

% --- 1. SISO 1×1 closed form. a = -1, G = 1, C = 1, Qn = Rn = 1.
% Dual ARE: -2P - P² + 1 = 0 → P = sqrt(2) - 1 ≈ 0.4142.
% L = P → ≈ 0.4142.
fprintf('1×1 Kalman gain (closed form sqrt(2) - 1 = %.6f):\n', sqrt(2)-1);
disp(kalman_L([0-1], [1], [1], [1], [1]));

% --- 2. Open-loop unstable plant. Kalman estimator stabilises it.
A = [1, 1; 0, 0-2];
G = [1, 0; 0, 1];   % process noise on each state
C = [1, 0];         % measure first state only
Qn = [1, 0; 0, 1];
Rn = [1];

L = kalman_L(A, G, C, Qn, Rn);
fprintf('Kalman gain L (2×1 — observer column):\n');
disp(L);

Aest = A - L * C;
fprintf('estimator Hurwitz: %d\n', isstable(Aest));
fprintf('estimator poles (real parts):\n');
disp(real(eig(Aest)));

% --- 3. Discrete Kalman gain on a Schur-stable plant.
Ad = [0.7, 0.1; 0.0, 0.4];
Ld = kalmd_L(Ad, G, C, Qn, [0.5]);
fprintf('discrete Kalman gain Ld:\n');
disp(Ld);
ed = eig(Ad - Ld * C);
fprintf('|eig(Ad - Ld C)|^2 (must be < 1):\n');
disp(real(ed) .* real(ed) + imag(ed) .* imag(ed));

% --- 4. LQG separation principle. Design LQR (state feedback) and
% Kalman (state estimation) on the same plant; the closed-loop LQG
% controller poles are the *union* of the LQR closed-loop poles and
% the Kalman estimator poles.
B = [0; 1];
Klqr = lqr(A, B, [1, 0; 0, 1], [1]);
% Kalman with B = G (input-channel process noise).
Lkal = kalman_L(A, B, C, [1], [1]);

fprintf('LQR feedback K:\n');
disp(Klqr);
fprintf('Kalman gain L:\n');
disp(Lkal);

fprintf('LQR closed-loop poles (real parts):\n');
disp(real(eig(A - B * Klqr)));
fprintf('Kalman estimator poles (real parts):\n');
disp(real(eig(A - Lkal * C)));
fprintf('LQG closed-loop spectrum = union of the two — separation principle.\n');

% ----- plot the LQG spectrum: LQR poles U Kalman estimator poles -----
elqr = eig(A - B * Klqr); ekal = eig(A - Lkal * C);
re = [real(elqr); real(ekal)]; im = [imag(elqr); imag(ekal)];
figure; scatter(real(elqr), imag(elqr)); hold on; scatter(real(ekal), imag(ekal));
plot([0 0], [min(im)-1, max(im)+1], 'k-');     % stability boundary (Im axis)
grid on; axis([min(re)-1, 0.5, min(im)-1, max(im)+1]);
xlabel('Re'); ylabel('Im'); title('LQG spectrum = LQR U Kalman estimator poles');
legend('LQR', 'Kalman');
saveas(gcf, '/tmp/ctrl_lqg_poles.png');
