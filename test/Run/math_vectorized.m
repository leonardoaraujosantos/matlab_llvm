% Vectorization demo: no explicit loops, just whole-matrix ops.
A = ones(3, 3);
B = eye(3, 3);
C = 2 * A + 5 * B;
disp(C);
D = (A + B) .^ 2;
disp(D);
disp(sum(D));
