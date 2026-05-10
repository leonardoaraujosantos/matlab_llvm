% [H, P] = hess(A) — 2-return Hessenberg shape (Tier-1.2 follow-on of
% CST roadmap §2.2). H is upper Hessenberg; P is the orthogonal
% similarity P' A P = H.

% --- 1. 4×4 dense matrix.
A = [4 1 -2 2; 1 2 0 1; -2 0 3 -2; 2 1 -2 -1];
[H, P] = hess(A);

% Subdiagonal cleanup: H must be exactly Hessenberg (entries below
% the first sub-diagonal are zero).
disp(H(3,1));          % 0
disp(H(4,1));          % 0
disp(H(4,2));          % 0

% Orthogonality: P' P should equal I to round-off.
PtP = P' * P;
disp(round(PtP(1,1)*1e10)/1e10);     % 1
disp(round(PtP(1,2)*1e10)/1e10);     % 0
disp(round(PtP(2,2)*1e10)/1e10);     % 1

% Similarity: P' A P should match H to round-off. Spot-check (1,1)
% and the first sub-diagonal.
PtAP = P' * A * P;
disp(round((PtAP(1,1) - H(1,1))*1e10)/1e10);   % 0
disp(round((PtAP(2,1) - H(2,1))*1e10)/1e10);   % 0

% --- 2. 1-return form still works.
H1 = hess(A);
disp(round((H1(1,1) - H(1,1))*1e10)/1e10);     % 0
disp(round((H1(2,1) - H(2,1))*1e10)/1e10);     % 0
