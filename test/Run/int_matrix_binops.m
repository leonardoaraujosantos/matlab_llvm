% Phase 1.1.D — typed-int matrix binops route through the typed runtime
% entry points (matlab_mat_i32_*/u8_*). Saturating arithmetic on overflow,
% saturating-cast for mixed double scalars, comparisons return logical.
A = int32([10 20 30; 40 50 60]);
B = int32([1 2 3; 4 5 6]);
disp(A + B);
disp(A - B);
disp(A .* B);
disp(A ./ B);

M = int32([2000000000, -2000000000]);
disp(M + M);                 % saturates to {INT32_MAX, INT32_MIN}

disp(A + 1.5);               % 1.5 -> 2 (round half-away-from-zero)
disp(2.5 + A);               % 2.5 -> 3

C = int32([1 5 10]);
disp(C > 4);
disp(C == 5);

U = uint8([10 200 250]);
V = uint8([5 60 100]);
disp(U + V);                 % {15, 255 (sat), 255 (sat)}
disp(U - V);                 % {5, 140, 150}
disp(V > U);
