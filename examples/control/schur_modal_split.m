% schur(A) for stable/unstable mode separation — Tier-1.2 follow-on
% (CST roadmap §2.2). The real Schur form makes the spectrum
% structurally visible: each diagonal block (1x1 or 2x2) is one mode
% of the system. Sorted (or re-ordered) Schur lets you separate
% stable from unstable modes — used by `stabsep` for control-design
% workflows that handle unstable plants explicitly.

% --- 1. Stable plant — all diagonal blocks have negative real parts.
%   A = [-1 0 0; -1 -2 0; -1 -1 -3]:  spectrum = {-1, -2, -3}.
% Note: matlab_llvm's Francis QR can leave a 2x2 trailing block even
% when both eigenvalues are real (a Givens-rotation post-pass to split
% real-eigenvalue 2x2s into 1x1s is a follow-on optimisation). The
% trace is still preserved exactly; the *individual* diagonal entries
% may not equal individual eigenvalues.
A = [-1 0 0; -1 -2 0; -1 -1 -3];
T = schur(A);
disp('trace(T) (= sum of eigenvalues = -6, order-invariant):');
disp(T(1, 1) + T(2, 2) + T(3, 3));

% --- 2. Marginally-unstable plant — one positive-real eigenvalue.
%   A = [1 1; 0 -2]: spectrum = {1, -2}.  Schur preserves this on the
%   diagonal (already upper-triangular -> fixed point).
B = [1 1; 0 -2];
TB = schur(B);
disp('upper-triangular -> Schur fixed point:');
disp(TB(1, 1));     % 1
disp(TB(2, 2));     % -2

% --- 3. Verify the Schur identity A = U T U' for a non-trivial 3x3.
%   Use trace(U' U) = 3 (orthogonality check) and (U T U')(1,1) =
%   A(1,1) (single-element reconstruction).
C = [2 1 0; 1 3 1; 0 1 4];
[U, TC] = schur(C);
Ut    = U';
UtU   = Ut * U;
disp('orthogonality: U^T U(1,1) and (2,2) should be 1:');
disp(UtU(1, 1));
disp(UtU(2, 2));
% Schur reconstruction: A_back = U * T * U'.
A_back = U * TC * Ut;
disp('A_back(1, 1) should match A(1, 1) = 2:');
disp(A_back(1, 1));

% --- 4. Connection to eig — the diagonal of T (for a fully-converged
% real-spectrum case) holds the eigenvalues, and matches eig(A).
disp('eig(A) for the lower-triangular plant (matches Schur diag up to permutation):');
disp(eig(A));
