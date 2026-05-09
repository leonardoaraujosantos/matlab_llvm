% Tier 1.5 (Control System Toolbox roadmap §2.5) — continuous
% algebraic Riccati. X = care(A, B, Q, R) solves
%      A'X + XA - X B R^{-1} B' X + Q = 0
% for the unique stabilising solution. Implemented via matrix sign
% function Newton iteration on the Hamiltonian. Gates lqr / kalman /
% lqg and the H_inf system norm.

% --- 1. CARE 1x1: a = -1, b = 1, q = 1, r = 1.
%   Equation: -2x - x^2 + 1 = 0  ->  x = -1 + sqrt(2) ~ 0.414.
A1 = [0-1];
B1 = [1];
Q1 = [1];
R1 = [1];
X1 = care(A1, B1, Q1, R1);
fprintf('%.6f\n', X1(1, 1));     % 0.414214

% --- 2. CARE for the double integrator: A = [0 1; 0 0], B = [0; 1],
%   Q = I, R = 1. Closed-form X = [sqrt(3) 1; 1 sqrt(3)].
A2 = [0 1; 0 0];
B2 = [0; 1];
Q2 = [1 0; 0 1];
R2 = [1];
X2 = care(A2, B2, Q2, R2);
fprintf('%.6f\n', X2(1, 1));     % 1.732051  (sqrt(3))
fprintf('%.6f\n', X2(1, 2));     % 1.000000
fprintf('%.6f\n', X2(2, 1));     % 1.000000
fprintf('%.6f\n', X2(2, 2));     % 1.732051

% --- 3. The LQR feedback gain K = R^{-1} B' X.  For the double
% integrator with Q = I, R = 1: K = [1 sqrt(3)].
K = inv(R2) * (B2' * X2);
fprintf('%.6f\n', K(1, 1));      % 1.000000
fprintf('%.6f\n', K(1, 2));      % 1.732051

% --- 4. Closed-loop poles eig(A - B*K) — the LQR-stabilised plant.
% For Q = I, R = 1 the symmetric root locus places poles at
% s = -sqrt(3)/2 +- j 0.5 (magnitude 1, damping 0.866).
% (Render real/imag of the whole complex column rather than indexing,
% which avoids the const_char lowering gap on complex matrix indexing.)
Acl = A2 - B2 * K;
e = eig(Acl);
disp(real(e));
disp(imag(e));

% --- 5. CARE residual self-consistency on a stable plant.
%   A' X + X A - X B R^{-1} B' X + Q should be ~0 to round-off.
A3 = [0-1, 0.5; 0, 0-2];
B3 = [1; 0.5];
Q3 = [2, 0.3; 0.3, 1];
R3 = [0.4];
X3 = care(A3, B3, Q3, R3);
Res = A3' * X3 + X3 * A3 - X3 * B3 * (inv(R3) * (B3' * X3)) + Q3;
fprintf('%.6f\n', abs(Res(1, 1)));    % 0.000000
fprintf('%.6f\n', abs(Res(2, 2)));    % 0.000000
fprintf('%.6f\n', abs(Res(1, 2)));    % 0.000000
