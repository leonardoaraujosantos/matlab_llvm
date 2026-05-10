% [V, D] = eig(A) — 2-return non-symmetric path (Tier-1.1 follow-on
% of CST roadmap §2.1). v1 path handles the all-real-eigenvalues case
% via Schur back-substitution. Complex eigenvalue pairs return 0×0
% (deferred to a complex-arithmetic follow-on).

% --- 1. Diagonal A. Eigenvalues = a_ii, eigenvectors = canonical basis.
A1 = [3 0 0; 0 5 0; 0 0 7];
[V1, D1] = eig(A1);
% Eigenvalues land on the Schur diagonal in input order: 3, 5, 7.
disp(D1(1,1));         % 3
disp(D1(2,2));         % 5
disp(D1(3,3));         % 7
% Eigenvectors are unit columns; sign-of-column is implementation
% defined, so check |V1(i, i)| = 1 (eigenvector aligned with axis i).
disp(round(abs(V1(1,1))*1e10)/1e10);   % 1
disp(round(abs(V1(2,2))*1e10)/1e10);   % 1
disp(round(abs(V1(3,3))*1e10)/1e10);   % 1
% Off-diagonal of V1 is zero for diagonal A.
disp(round(V1(1,2)*1e10)/1e10);        % 0
disp(round(V1(2,1)*1e10)/1e10);        % 0

% --- 2. Upper-triangular non-symmetric A. Eigenvalues = diag entries.
A2 = [2 1 0; 0 5 1; 0 0 7];
[V2, D2] = eig(A2);
disp(D2(1,1));         % 2
disp(D2(2,2));         % 5
disp(D2(3,3));         % 7
% Verify A · V = V · D  (column by column, only the (1,1) and (2,2)
% products to keep stdout small).
AV = A2 * V2;
VD = V2 * D2;
disp(round((AV(1,1) - VD(1,1))*1e10)/1e10);    % 0
disp(round((AV(2,2) - VD(2,2))*1e10)/1e10);    % 0
disp(round((AV(3,3) - VD(3,3))*1e10)/1e10);    % 0

% --- 3. Non-symmetric general A with distinct real eigenvalues.
% Eigenvalues of [[3 1]; [0 5]] are {3, 5}.
A3 = [3 1; 0 5];
[V3, D3] = eig(A3);
disp(D3(1,1));         % 3
disp(D3(2,2));         % 5
% Reconstruction: A · V = V · D.
AV3 = A3 * V3;
VD3 = V3 * D3;
disp(round((AV3(1,1) - VD3(1,1))*1e10)/1e10);  % 0
disp(round((AV3(2,2) - VD3(2,2))*1e10)/1e10);  % 0

% --- 4. Failure path: rotation matrix has eigenvalues ±i (complex pair).
% v1 returns 0×0 for both V and D.
A4 = [0 1; -1 0];
[V4, D4] = eig(A4);
disp(size(V4, 1));     % 0
disp(size(D4, 1));     % 0
