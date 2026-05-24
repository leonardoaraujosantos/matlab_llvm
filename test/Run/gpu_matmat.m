% T5 gate — gpucoder.matrixMatrixKernel with @plus (standard GEMM).
% Validates the custom-GEMM template against the reference A*B.
A = [1 2; 3 4];
B = [5 6; 7 8];
plusfn = @(a,b) a + b;
C = gpucoder.matrixMatrixKernel(plusfn, A, B);
disp(C);
