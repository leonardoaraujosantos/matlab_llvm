% Minimum 3-D array support. zeros(m,n,p) / ones(m,n,p) allocate a
% matlab_mat3 descriptor (slice-major layout); scalar A(i,j,k) reads
% and writes route to matlab_subscript3_s / matlab_subscript3_store.
A = zeros(2, 3, 4);
disp(A(1, 1, 1));          % 0
A(1, 2, 3) = 42;
disp(A(1, 2, 3));          % 42
A(2, 3, 4) = 7;
disp(A(2, 3, 4));          % 7
disp(size(A, 1));          % 2
disp(size(A, 2));          % 3
disp(size(A, 3));          % 4
disp(numel(A));            % 24
disp(ndims(A));            % 3

B = ones(2, 2, 2);
disp(B(1, 1, 1));          % 1
disp(B(2, 2, 2));          % 1
B(1, 1, 2) = 5;
disp(B(1, 1, 2));          % 5
