% logm of a full (non-triangular) matrix with real eigenvalues. The Francis QR
% leaves the two real eigenvalues as a 2x2 Schur block; matlab_logm now
% standardizes it (Givens triangularize) so log(expm(M)) == M.
A = [0 1; -2 -3];          % real eigenvalues -1, -2
M = expm(A * 0.1);
L = logm(M);
fprintf('%.4f %.4f\n', L(1,1), L(1,2));
fprintf('%.4f %.4f\n', L(2,1), L(2,2));
