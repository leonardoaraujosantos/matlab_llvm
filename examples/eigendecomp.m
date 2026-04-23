% Eigendecomposition of a small symmetric tridiagonal matrix.
% Two return shapes are supported:
%     d       = eig(A)   -> eigenvalues as a column vector
%     [V, D]  = eig(A)   -> V is the matrix of eigenvectors (columns),
%                            D is the diagonal matrix of eigenvalues.
A = [4 1 0;
     1 3 1;
     0 1 2];

disp('A =');
disp(A);

disp('eigenvalues of A (single-return eig):');
disp(eig(A));

disp('[V, D] = eig(A) — diagonal of D:');
[V, D] = eig(A);
disp(D);

disp('V * D * V'' (should reconstruct A):');
disp(V * D * V');

disp('det(A) =');
disp(det(A));

disp('inv(A) =');
disp(inv(A));

disp('A * inv(A) (should be the identity):');
disp(A * inv(A));
