% gpuArray PCT-surface GEMM validation.  Exercises:
%   - gpuArray(X)         — wrap a host matrix
%   - A * B on gpuArrays  — mtimes inherited from matrix lane
%   - gather(g)
% A is identity, B is the row-major counter [1..16]; expect C == B.
A = gpuArray([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]);
B = gpuArray([1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]);
Cgpu = A * B;
C = gather(Cgpu);
diff = C - [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
fprintf('gemm err = %.0f\n', sum(sum(diff .* diff)));
% Pin real values too — `err = 0` alone passes on an empty C (#333), which
% is exactly how the broken gpuArray carrier hid the bug.
[gm, gn] = size(C);
fprintf('gemm size = %d %d\n', gm, gn);
fprintf('gemm C34 = %.0f\n', C(3, 4));
