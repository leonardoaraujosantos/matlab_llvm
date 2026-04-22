% Build two matrices and demonstrate matrix multiplication plus
% element-wise arithmetic.
A = [1 2 3;
     4 5 6;
     7 8 10];

B = [1 0 0;
     0 2 0;
     0 0 3];

disp('A =');
disp(A);
disp('B =');
disp(B);

disp('A * B =');
disp(A * B);

disp('A .* B =');
disp(A .* B);

disp('A'' =');
disp(A');
