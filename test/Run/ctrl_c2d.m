% Tier 2.2 (Control System Toolbox roadmap §3.2) — c2d zero-order hold
% [Ad, Bd] = c2d(A, B, Ts) discretises  xdot = A x + B u  for ZOH on u.
% Uses the Van Loan augmented-matrix expm trick:
%   expm([A*Ts B*Ts; 0 0]) = [Ad Bd; 0 I_m]
% one expm call gives both Ad and Bd. Sits cleanly on Tier-1.3 expm.

% --- 1. Diagonal A — closed-form Ad and Bd.
%   A = [-1 0; 0 -2], B = [1; 0.5], Ts = 0.1.
%   Ad = diag(exp(-0.1), exp(-0.2))
%   Bd[i] = (1 - exp(A[i]*Ts)) / (-A[i]) * B[i]
A = [0-1, 0; 0, 0-2];
B = [1; 0.5];
Ts = 0.1;
[Ad, Bd] = c2d(A, B, Ts);
fprintf('%.6f\n', Ad(1, 1));     % 0.904837 (exp(-0.1))
fprintf('%.6f\n', Ad(2, 2));     % 0.818731 (exp(-0.2))
fprintf('%.6f\n', Ad(1, 2));     % 0.000000
fprintf('%.6f\n', Bd(1, 1));     % 0.095163 ((1 - exp(-0.1)) / 1)
fprintf('%.6f\n', Bd(2, 1));     % 0.045317 ((1 - exp(-0.2)) / 2 * 0.5)

% --- 2. Discretise the double integrator.
%   A = [0 1; 0 0], B = [0; 1], Ts = 0.5.
%   Closed form for ZOH on a double integrator:
%     Ad = [1 Ts; 0 1] = [1 0.5; 0 1]
%     Bd = [Ts^2/2; Ts] = [0.125; 0.5]
A2 = [0 1; 0 0];
B2 = [0; 1];
[Ad2, Bd2] = c2d(A2, B2, 0.5);
fprintf('%.6f\n', Ad2(1, 1));    % 1.000000
fprintf('%.6f\n', Ad2(1, 2));    % 0.500000
fprintf('%.6f\n', Ad2(2, 1));    % 0.000000
fprintf('%.6f\n', Ad2(2, 2));    % 1.000000
fprintf('%.6f\n', Bd2(1, 1));    % 0.125000
fprintf('%.6f\n', Bd2(2, 1));    % 0.500000

% --- 3. Discrete-time stability check after LQR design.
% Continuous LQR places stable closed-loop poles; discretising
% preserves stability for any Ts > 0 (eigenvalues map via z = exp(s*Ts)).
Q = [1 0; 0 1];
R = [1];
K = lqr(A2, B2, Q, R);     % continuous LQR gain
Acl = A2 - B2 * K;          % closed-loop A
[Adcl, Bdcl] = c2d(Acl, B2, 0.1);
% Discrete poles: |eig(Adcl)| < 1.
e = eig(Adcl);
mag2 = real(e) .* real(e) + imag(e) .* imag(e);
disp('discrete poles squared magnitude (must be < 1):');
disp(mag2);
