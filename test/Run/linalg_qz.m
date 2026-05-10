% [AA, BB, Q, Z] = qz(A, B) — generalised Schur decomposition
% (Tier-1.2 follow-on of CST roadmap §2.2). v1 path is layered on
% schur(B^{-1}·A) + qr(B·U) and requires B invertible.

% --- 1. B = I gives the standard Schur of A. AA = T (upper quasi-
% triangular), BB = I, Q = U' (orthogonal), Z = U.
A1 = [4 1 -2 2; 1 2 0 1; -2 0 3 -2; 2 1 -2 -1];
B1 = eye(4);
[AA, BB, Q, Z] = qz(A1, B1);

% AA must be upper quasi-triangular (entries below the first
% sub-diagonal are zero — same shape as schur(A)).
disp(round(AA(3,1)*1e10)/1e10);     % 0
disp(round(AA(4,1)*1e10)/1e10);     % 0
disp(round(AA(4,2)*1e10)/1e10);     % 0

% BB = I when B = I: print rounded diagonal + off-diagonal.
disp(round(BB(1,1)*1e10)/1e10);     % 1
disp(round(BB(2,2)*1e10)/1e10);     % 1
disp(round(BB(1,2)*1e10)/1e10);     % 0

% Reconstruction: Q · A · Z = AA, Q · B · Z = BB. Spot-check the
% (1,1) entries.
QAZ = Q * A1 * Z;
disp(round((QAZ(1,1) - AA(1,1))*1e10)/1e10);   % 0
disp(round((QAZ(2,1) - AA(2,1))*1e10)/1e10);   % 0
QBZ = Q * B1 * Z;
disp(round((QBZ(1,1) - BB(1,1))*1e10)/1e10);   % 0

% --- 2. Diagonal A and diagonal B → eigenvalues = a_ii / b_ii.
A2 = [6 0; 0 8];
B2 = [2 0; 0 4];
[AA2, BB2, Q2, Z2] = qz(A2, B2);
% Generalised eigenvalues: AA2(i,i) / BB2(i,i) = {3, 2}.
% Sort doesn't matter — just confirm the ratios match the set.
e1 = AA2(1,1) / BB2(1,1);
e2 = AA2(2,2) / BB2(2,2);
% Print sorted ascending.
if e1 < e2
    disp(round(e1*1e10)/1e10);
    disp(round(e2*1e10)/1e10);
else
    disp(round(e2*1e10)/1e10);
    disp(round(e1*1e10)/1e10);
end

% --- 3. Failure path — singular B → returns 0×0.
A3 = [1 2; 3 4];
B3 = [1 2; 2 4];     % rank 1
[AA3, BB3, Q3, Z3] = qz(A3, B3);
disp(size(AA3, 1));     % 0
disp(size(BB3, 1));     % 0
disp(size(Q3, 1));      % 0
disp(size(Z3, 1));      % 0
