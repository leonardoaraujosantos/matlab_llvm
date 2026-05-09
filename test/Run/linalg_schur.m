% Tier 1.2 follow-on (Control System Toolbox roadmap §2.2) — real
% Schur decomposition. T = schur(A) and [U, T] = schur(A) such that
% A = U T U' with U orthogonal and T upper quasi-triangular (1x1 and
% 2x2 diagonal blocks). Same Hessenberg + Francis-QR pipeline as
% non-symmetric eig, with the orthogonal accumulator U threaded
% through both passes. Gates Bartels-Stewart Lyapunov (Tier 1.4) and
% ordered-Schur Riccati (Tier 1.5).

% --- 1. Upper-triangular A is a fixed point — sigma == 0 in every
% column, QR deflates immediately. T = A, U = I.
A = [1 2 3; 0 5 6; 0 0 8];
T = schur(A);
fprintf('%.6f\n', T(1, 1));    % 1.000000
fprintf('%.6f\n', T(1, 2));    % 2.000000
fprintf('%.6f\n', T(2, 2));    % 5.000000
fprintf('%.6f\n', T(3, 3));    % 8.000000
fprintf('%.6f\n', T(2, 1));    % 0.000000
fprintf('%.6f\n', T(3, 1));    % 0.000000

% --- 2. Reconstruction A = U * T * U' for an asymmetric matrix.
%   Verify trace(A) = trace(T) (similarity invariant).
B = [1 2 3; 4 5 6; 7 8 10];
[U, Tb] = schur(B);
% Trace check (preserved exactly by similarity, modulo rounding).
fprintf('%.6f\n', Tb(1, 1) + Tb(2, 2) + Tb(3, 3));  % trace(B) = 16
% Reconstruction: A_back = U * T * U'.  The (1, 1) entry of U*T*U'
% should match A(1, 1) = 1 within 1e-9.
Ut = U';
A_back = U * Tb * Ut;
fprintf('%.6f\n', A_back(1, 1));   % 1.000000

% --- 3. Symmetric input: T is diagonal (eigenvalues), U is the
% orthogonal eigenvector matrix. Skip — the symmetry detection in
% matlab_eig is independent of schur, and schur of a symmetric matrix
% still goes through Francis QR (which converges quickly to the
% diagonal form).
S = [4 1; 1 4];
[Us, Ts] = schur(S);
% Eigenvalues of [4 1; 1 4] are 3 and 5; trace = 8.
fprintf('%.6f\n', Ts(1, 1) + Ts(2, 2));   % 8.000000

% --- 4. det(T) = det(A) (Schur similarity preserves det).
%   For a 2x2 [4 1; 1 4]: det = 16 - 1 = 15.
fprintf('%.6f\n', Ts(1, 1) * Ts(2, 2) - Ts(1, 2) * Ts(2, 1));  % 15.000000
