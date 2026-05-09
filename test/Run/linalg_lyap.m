% Tier 1.4 (Control System Toolbox roadmap §2.4) — Lyapunov solvers.
%   lyap(A, Q):   A X + X A' + Q = 0   (continuous Lyapunov)
%   dlyap(A, Q):  A X A' - X + Q = 0   (discrete / Stein equation)
% Vectorise + dense LU; correct for the small plants typical of the
% CST surface. Gates `gram`, the H2 norm, and balanced realisation.

% --- 1. Continuous Lyapunov, 1x1: a x + x a + q = 0 -> x = -q/(2 a).
%   a = -1, q = 1  ->  x = 0.5.
A1 = [0-1];
Q1 = [1];
X1 = lyap(A1, Q1);
fprintf('%.6f\n', X1(1, 1));     % 0.500000

% --- 2. Continuous Lyapunov, 2x2 diagonal: A = diag(-1, -2), Q = I.
%   X is also diagonal: X[i,i] = -Q[i,i] / (2 A[i,i])  =>  X = diag(0.5, 0.25).
A2 = [0-1, 0; 0, 0-2];
Q2 = [1 0; 0 1];
X2 = lyap(A2, Q2);
fprintf('%.6f\n', X2(1, 1));     % 0.500000
fprintf('%.6f\n', X2(2, 2));     % 0.250000
fprintf('%.6f\n', X2(1, 2));     % 0.000000

% --- 3. Continuous Lyapunov self-consistency. Build a stable 3x3
% asymmetric A, find X via the lyap call, verify the residual
% A X + X A' + Q is zero to round-off.
A3 = [0-2, 1, 0;  0, 0-3, 1;  1, 0, 0-1];
Q3 = [1 0 0; 0 2 0; 0 0 3];
X3 = lyap(A3, Q3);
R = A3 * X3 + X3 * A3' + Q3;
fprintf('%.6f\n', abs(R(1, 1)));     % 0.000000  (Frobenius residual)
fprintf('%.6f\n', abs(R(3, 3)));     % 0.000000

% --- 4. Discrete Lyapunov, 1x1: a^2 x - x + q = 0 -> x = q/(1 - a^2).
%   a = 0.5, q = 1  ->  x = 4/3.
Ad1 = [0.5];
Qd1 = [1];
Xd1 = dlyap(Ad1, Qd1);
fprintf('%.6f\n', Xd1(1, 1));    % 1.333333

% --- 5. Discrete Lyapunov, stable 2x2: A = diag(0.5, 0.6), Q = I.
%   X diagonal: X[i,i] = Q[i,i] / (1 - A[i,i]^2).
%   X = diag(1/0.75, 1/0.64) = diag(1.333..., 1.5625).
Ad2 = [0.5 0; 0 0.6];
Qd2 = [1 0; 0 1];
Xd2 = dlyap(Ad2, Qd2);
fprintf('%.6f\n', Xd2(1, 1));    % 1.333333
fprintf('%.6f\n', Xd2(2, 2));    % 1.562500

% --- 6. Discrete Lyapunov self-consistency.
Ad3 = [0.5 0.1 0; 0 0.6 0.2; 0.1 0 0.4];
Qd3 = [1 0 0; 0 1 0; 0 0 1];
Xd3 = dlyap(Ad3, Qd3);
Rd  = Ad3 * Xd3 * Ad3' - Xd3 + Qd3;
fprintf('%.6f\n', abs(Rd(1, 1)));    % 0.000000
fprintf('%.6f\n', abs(Rd(2, 3)));    % 0.000000
