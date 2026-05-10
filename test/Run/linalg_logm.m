% logm — matrix logarithm (Tier-1.3 follow-on of CST roadmap §2.3).
% Schur-then-Parlett-recurrence; valid for inputs whose Schur form is
% upper-triangular with positive distinct real diagonal entries.

% --- 1. Trivial 1×1: logm(x) = log(x) for positive x.
A1 = [4];
L1 = logm(A1);
disp(L1(1,1));         % log(4) ≈ 1.3862944

% --- 2. Diagonal positive 2×2: logm acts element-wise on the diagonal.
A2 = [9 0; 0 16];
L2 = logm(A2);
disp(L2(1,1));         % log(9)  ≈ 2.1972246
disp(L2(1,2));         % 0
disp(L2(2,1));         % 0
disp(L2(2,2));         % log(16) ≈ 2.7725887

% --- 3. Round-trip: logm(expm(A)) ≈ A for an A whose expm is upper-
% triangular with distinct positive eigenvalues. A small upper-triangular
% A with distinct real entries is the easy gating case.
A3 = [0.5 0.2; 0 1.5];
B3 = expm(A3);
L3 = logm(B3);
% Print L3(1,1), L3(2,2), L3(1,2) to 6 decimals — should track A3 to
% close-to-machine precision.
disp(L3(1,1));         % 0.5
disp(L3(1,2));         % 0.2
disp(L3(2,1));         % 0
disp(L3(2,2));         % 1.5

% --- 4. Non-symmetric 2×2 with distinct real eigenvalues. Eigenvalues
% of [[3, 1], [0, 5]] are {3, 5}, both positive distinct.
A4 = [3 1; 0 5];
L4 = logm(A4);
disp(L4(1,1));         % log(3)
disp(L4(2,2));         % log(5)

% --- 5. Failure path: complex eigenvalue pair → returns 0×0.
A5 = [0 1; -1 0];      % rotation, eigvals ±i
L5 = logm(A5);
disp(size(L5,1));      % 0
disp(size(L5,2));      % 0

% --- 6. Failure path: negative eigenvalue → returns 0×0.
A6 = [-2];
L6 = logm(A6);
disp(size(L6,1));      % 0
disp(size(L6,2));      % 0
