% Tier 4.2 (CST roadmap §4.2) — steady-state Kalman filter gain.
%   kalman_L(A, G, C, Qn, Rn) returns L for plant
%      xdot = A x + G w,  y = C x + v,  cov(w)=Qn, cov(v)=Rn.
%   The estimator (A − L·C) is Hurwitz; eig(A − L·C) are estimator poles.
% Implementation exploits LQR/Kalman duality:
%      L = (lqr(A', C', G·Qn·G', Rn))'.

% --- 1. SISO 1×1 closed form. a = -1, G = 1, C = 1, Qn = Rn = 1.
%   Dual ARE: -2P - P² + 1 = 0 → P = sqrt(2) - 1.
%   L = P → ≈ 0.4142.
A = [0-1];
G = [1];
C = [1];
Qn = [1];
Rn = [1];
disp('1×1 Kalman gain L (closed form sqrt(2) - 1):');
disp(kalman_L(A, G, C, Qn, Rn));

% --- 2. Open-loop unstable plant. Estimator must Hurwitz-stabilise.
A2 = [1, 1; 0, 0-2];
G2 = [1, 0; 0, 1];
C2 = [1, 0];
Q2 = [1, 0; 0, 1];
R2 = [1];
L2 = kalman_L(A2, G2, C2, Q2, R2);
disp('Kalman gain L (2×1 — observer column):');
disp(L2);

Aest = A2 - L2 * C2;
disp('isstable(A - L*C) — must be 1:');
disp(isstable(Aest));

disp('estimator poles (must all have negative real part):');
disp(real(eig(Aest)));

% --- 3. Discrete Kalman gain. Schur-stable plant.
Ad = [0.7, 0.1; 0.0, 0.4];
Gd = [1, 0; 0, 1];
Cd = [1, 0];
Qd = [1, 0; 0, 1];
Rd = [0.5];
Ld = kalmd_L(Ad, Gd, Cd, Qd, Rd);
disp('discrete Kalman gain L:');
disp(Ld);

Adest = Ad - Ld * Cd;
ed = eig(Adest);
mag2 = real(ed) .* real(ed) + imag(ed) .* imag(ed);
disp('|eig(Ad - L Cd)|^2 (must all be < 1):');
disp(mag2);

% --- 4. LQG: combine LQR (state feedback) and Kalman (state estimation)
% on the same plant. The separation principle says the closed-loop poles
% of the LQG controller are the union of the LQR closed-loop poles and
% the Kalman estimator poles.
%
% Use plant 2: A2, G2 = B*[1] = [0; 1] (process noise on input channel).
B2 = [0; 1];
G2b = B2;
Qb = [1];   % scalar process-noise variance (input-channel)
Klqr = lqr(A2, B2, [1, 0; 0, 1], [1]);
Lkal = kalman_L(A2, G2b, C2, Qb, R2);
disp('LQR feedback K:');
disp(Klqr);
disp('Kalman gain L:');
disp(Lkal);
disp('LQR closed-loop poles (real parts):');
disp(real(eig(A2 - B2 * Klqr)));
disp('Kalman estimator poles (real parts):');
disp(real(eig(A2 - Lkal * C2)));
