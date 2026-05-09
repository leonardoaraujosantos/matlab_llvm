% Tier 1.2 (Control System Toolbox roadmap §2.2) — Hessenberg
% reduction. H = hess(A) via Householder reflections. Building block
% for the Francis QR iteration that produces real Schur form (which
% non-symmetric eig, schur, lyap, and care all sit on).

% Upper-triangular A is a fixed point — sigma == 0 in every column, so
% the inner Householder skip applies and H == A.
A = [1 2 3; 0 5 6; 0 0 8];
H = hess(A);
fprintf('%.6f\n', H(1, 1));     % 1.000000
fprintf('%.6f\n', H(1, 2));     % 2.000000
fprintf('%.6f\n', H(2, 2));     % 5.000000
fprintf('%.6f\n', H(3, 3));     % 8.000000
fprintf('%.6f\n', H(2, 1));     % 0.000000
fprintf('%.6f\n', H(3, 1));     % 0.000000

% 3x3 dense — the algorithm should zero H(3, 1) and preserve trace.
B = [1 2 3; 4 5 6; 7 8 10];
HB = hess(B);
fprintf('%.6f\n', HB(3, 1));    % 0.000000  (forced by Householder)
fprintf('%.6f\n', abs(HB(2, 1) + sqrt(65))); % 0.000000  (-sqrt(65) on the subdiagonal)
fprintf('%.6f\n', HB(1, 1) + HB(2, 2) + HB(3, 3));    % 16.000000  (trace preserved)

% 4x4 symmetric — Hessenberg of a symmetric matrix is tridiagonal:
% the upper-right corner is also driven to zero.
S = [4 1 -2 2; 1 2 0 1; -2 0 3 -2; 2 1 -2 -1];
HS = hess(S);
fprintf('%.6f\n', HS(3, 1));    % 0.000000
fprintf('%.6f\n', HS(4, 1));    % 0.000000
fprintf('%.6f\n', HS(4, 2));    % 0.000000
fprintf('%.6f\n', abs(HS(1, 3)));    % 0.000000  (tridiagonal)
fprintf('%.6f\n', abs(HS(1, 4)));    % 0.000000
fprintf('%.6f\n', abs(HS(2, 4)));    % 0.000000
% Trace = 4 + 2 + 3 + (-1) = 8.
fprintf('%.6f\n', HS(1, 1) + HS(2, 2) + HS(3, 3) + HS(4, 4));   % 8.000000
