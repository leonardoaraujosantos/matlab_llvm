% Tier 2 (Control System Toolbox roadmap §4.1 — discrete branch).
% K = dlqr(Ad, Bd, Q, R) returns the optimal discrete-time LQ feedback
% gain for  x[k+1] = Ad x[k] + Bd u[k]  minimising the quadratic cost
%   J = sum_{k=0}^inf (x' Q x + u' R u).
% X = dare(Ad, Bd, Q, R) returns the stabilising solution of the
% discrete algebraic Riccati equation; dlqr is the 1-line wrapper
%   K = (R + B' X B)^{-1} B' X A.
% Newton-Kleinman iteration seeds from X_0 = dlyap(Ad', Q); requires
% Schur-stable Ad (the typical case after c2d of a stable plant).

% --- 1. Diagonal Schur-stable plant — easy hand-checkable closed form.
%   Ad = diag(0.5, 0.6), Bd = [1; 1], Q = I, R = 1.
% Each axis decouples; per-axis dare reduces to scalar quadratic
%   X^2 + (R - (a^2 - 1)) X / b^2 - Q*R / b^2 = 0
% solved by the positive root.
Ad = [0.5 0; 0 0.6];
Bd = [1; 1];
Q  = [1 0; 0 1];
R  = [1];
X  = dare(Ad, Bd, Q, R);
% X must be symmetric positive definite — check both diagonal entries
% positive and the off-diagonal symmetric.
disp('dare X (must be SPD, X(1,2) == X(2,1)):');
disp(X);

K  = dlqr(Ad, Bd, Q, R);
fprintf('%.6f\n', K(1, 1));    % positive feedback gain on x1
fprintf('%.6f\n', K(1, 2));    % positive feedback gain on x2

% Closed-loop must be Schur-stable: |eig(Ad - Bd K)| < 1.
Acl = Ad - Bd * K;
e   = eig(Acl);
mag2 = real(e) .* real(e) + imag(e) .* imag(e);
disp('|eig(Acl)|^2 (must be < 1):');
disp(mag2);

% --- 2. DARE residual self-consistency:
%   A' X A - X - A' X B (R + B' X B)^{-1} B' X A + Q = 0.
% A 2x2 plant with a deliberately non-trivial coupling.
Ad2 = [0.7 0.1; 0 0.4];   % eig = {0.7, 0.4}, both inside unit disk
Bd2 = [1; 0.5];
Q2  = [1.5 0.2; 0.2 1];
R2  = [0.5];
X2  = dare(Ad2, Bd2, Q2, R2);
At  = Ad2';
AtXA  = At * X2 * Ad2;
BtXB  = Bd2' * X2 * Bd2;
S     = R2 + BtXB;
AtXB  = At * X2 * Bd2;
BtXA  = Bd2' * X2 * Ad2;
Drop  = AtXB / S * BtXA;
res   = AtXA - X2 - Drop + Q2;
% Print the Riccati X for the second plant — symmetry is the easiest
% byte-stable witness that dare converged (X(1,2) == X(2,1)).
disp('dare X2 (symmetric):');
disp(X2);

% --- 3. dlqr designed on a c2d-discretised double integrator.
% Continuous: A = [0 1; 0 0], B = [0; 1]. After c2d at Ts = 0.1 the
% plant has marginally-stable poles (z = 1, 1) — Newton-Kleinman with
% X_0 = dlyap(...) cannot stabilise that on its own, so the documented
% recipe is to use lqr in continuous time and discretise the gain.
% Here we keep the example to a Schur-stable plant by adding a small
% damping  A = [0 1; -0.5 -0.5]  (poles at -0.25 +- j*0.661).
Ad3 = [0.987 0.097; -0.049 0.940];   % expm([0 1; -0.5 -0.5] * 0.1)
Bd3 = [0.005; 0.097];
Q3  = [1 0; 0 1];
R3  = [1];
K3  = dlqr(Ad3, Bd3, Q3, R3);
disp('dlqr gain K (1x2):');
disp(K3);
Acl3 = Ad3 - Bd3 * K3;
e3   = eig(Acl3);
m3   = real(e3) .* real(e3) + imag(e3) .* imag(e3);
disp('|eig(Acl)|^2 < 1 (closed-loop Schur-stable):');
disp(m3);
