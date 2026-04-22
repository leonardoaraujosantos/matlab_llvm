% Eigendecomposition of a small symmetric tridiagonal matrix. The
% current eig() runtime returns just the eigenvalues when called
% without a left-hand-side matrix capture, so we display those.
A = [4 1 0;
     1 3 1;
     0 1 2];

disp('A =');
disp(A);

disp('eigenvalues of A:');
disp(eig(A));

disp('determinant:');
disp(det(A));

disp('inverse:');
disp(inv(A));

disp('A * inv(A) (should be the identity):');
disp(A * inv(A));
