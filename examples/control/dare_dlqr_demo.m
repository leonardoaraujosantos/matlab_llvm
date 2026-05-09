% Tier 2 (CST roadmap §4.1, discrete branch) — dare + dlqr workflow.
% The discrete-time analogue of `care` + `lqr`: Newton-Kleinman
% iteration solves
%   A' X A - X - A' X B (R + B' X B)^{-1} B' X A + Q = 0
% for the unique stabilising X = X' >= 0; `dlqr` reads off
%   K = (R + B' X B)^{-1} B' X A
% and the closed-loop  Ad - Bd K  is Schur-stable (|eig| < 1).
%
% Newton-Kleinman seeds from X_0 = dlyap(Ad', Q), so it requires Ad
% already Schur-stable — the typical case after `c2d` of a damped
% continuous plant.

% --- 1. Diagonal Schur-stable plant. Per-axis dare reduces to the
% scalar quadratic
%   X^2 + (R - (a^2 - 1)) X / b^2 - Q*R / b^2 = 0
% solved by the positive root.
Ad = [0.5 0; 0 0.6];
Bd = [1; 1];
Q  = [1 0; 0 1];
R  = [1];

X = dare(Ad, Bd, Q, R);
disp('Riccati X (must be SPD, X(1,2) == X(2,1)):');
disp(X);

K = dlqr(Ad, Bd, Q, R);
disp('dlqr feedback gain K (1 x 2):');
disp(K);

% Closed loop must be Schur-stable.
Acl = Ad - Bd * K;
m2  = real(eig(Acl)) .* real(eig(Acl)) ...
    + imag(eig(Acl)) .* imag(eig(Acl));
disp('|eig(Acl)|^2 (closed-loop poles inside unit disk):');
disp(m2);

% --- 2. Damped 2-D plant from a c2d-discretised mass-spring-damper.
%   xdot = [0 1; -wn^2 -2*zeta*wn] x + [0; 1] u,  wn = 2, zeta = 0.5.
% At Ts = 0.1 the discretised plant is well inside the unit disk.
wn   = 2; zeta = 0.5;
Ac   = [0 1; 0 - wn*wn, 0 - 2*zeta*wn];
Bc   = [0; 1];
[Ad2, Bd2] = c2d(Ac, Bc, 0.1);

% Heavier penalty on position than velocity.
Q2 = [10 0; 0 1];
R2 = [1];
X2 = dare(Ad2, Bd2, Q2, R2);
K2 = dlqr(Ad2, Bd2, Q2, R2);

disp('mass-spring-damper Riccati X2:');
disp(X2);
disp('dlqr gain K2 (state feedback):');
disp(K2);

% Closed-loop discrete poles.
Acl2 = Ad2 - Bd2 * K2;
e2   = eig(Acl2);
m2b  = real(e2) .* real(e2) + imag(e2) .* imag(e2);
disp('|eig(Acl2)|^2 (must be < 1):');
disp(m2b);

% --- 3. Riccati residual self-consistency. Frobenius norm on a
% deliberately non-trivial 2 x 2 plant. Printing element-wise so the
% residual structure is visible.
Ad3 = [0.7 0.1; 0 0.4];     % eig = {0.7, 0.4}
Bd3 = [1; 0.5];
Q3  = [1.5 0.2; 0.2 1];
R3  = [0.5];
X3  = dare(Ad3, Bd3, Q3, R3);
At3 = Ad3';
res = At3*X3*Ad3 - X3 - (At3*X3*Bd3) / (R3 + Bd3'*X3*Bd3) * (Bd3'*X3*Ad3) + Q3;
disp('discrete Riccati residual (entries ~0 to round-off):');
disp(res);
