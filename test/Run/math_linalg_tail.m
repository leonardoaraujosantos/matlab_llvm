% Linear algebra tail: norm / trace / kron / chol / pinv / qr / lu.
%
% Small matrices with clean results to keep the stdout diff stable.

% norm: Frobenius norm of a matrix == sqrt of sum of squares.
v = [3 4];
disp(norm(v));               % 5

% trace: sum of diagonal.
M1 = [1 2; 3 4];
disp(trace(M1));             % 5

% kron: Kronecker product. 2x2 kron 2x2 -> 4x4.
E = [1 0; 0 1];
B = [1 2; 3 4];
disp(kron(E, B));

% chol: A = R' * R for an SPD matrix. Pick a known SPD so R is tidy.
A = [4 2; 2 3];
R = chol(A);
disp(R);

% pinv on a tall (3x2) full-column-rank matrix.
T = [1 0; 0 1; 1 1];
P = pinv(T);
disp(P * T);                 % I(2)

% qr: Q is orthonormal, R upper-triangular, Q*R == M1.
[Q, RR] = qr(M1);
disp(Q * RR);

% lu: L is unit lower, U is upper. With partial pivoting L*U == P*M1
% rather than M1, so a row-swap can show up at the output — here
% pivoting puts the larger |3| in the first row, so L*U reconstructs
% [3 4; 1 2].
[L, U] = lu(M1);
disp(L * U);
