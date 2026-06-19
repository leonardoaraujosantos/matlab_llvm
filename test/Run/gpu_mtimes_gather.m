% #333 regression: gpuArray arithmetic must operate on the underlying
% matrix, not a carrier object. Before the fix gpuArray was a classdef
% object, so `Ag * Bg` ran matlab_matmul on the OBJECT pointer and
% returned an empty (0x0) result; gather() then returned empty. (gpu_gemm.m
% passed spuriously because sum(sum((empty).^2)) == 0.) gpuArray is now an
% identity carrier on the CPU-debug lane, so host matrix ops see through it.
A = [1 2; 3 4];
B = [5 6; 7 8];
Ag = gpuArray(A);
Bg = gpuArray(B);

% mtimes: [1 2;3 4]*[5 6;7 8] = [19 22; 43 50]
C = gather(Ag * Bg);
[m, n] = size(C);
fprintf('mtimes size = %d %d\n', m, n);
fprintf('mtimes sum  = %d\n', sum(sum(C)));
fprintf('mtimes c22  = %d\n', C(2, 2));

% plus: elementwise (1+5)+(2+6)+(3+7)+(4+8) = 36
D = gather(Ag + Bg);
fprintf('plus sum    = %d\n', sum(sum(D)));

% gather of a bare gpuArray is identity
G = gather(Ag);
fprintf('gather sum  = %d\n', sum(sum(G)));
