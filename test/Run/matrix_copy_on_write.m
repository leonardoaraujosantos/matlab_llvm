% Copy-on-write: `B = A` must give B an independent buffer, so a later
% indexed write to B does not mutate A (MATLAB value semantics).
A = [1 2; 3 4];
B = A;
B(1, 1) = 99;
fprintf('A = %.0f %.0f %.0f %.0f\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf('B = %.0f %.0f %.0f %.0f\n', B(1,1), B(1,2), B(2,1), B(2,2));

% Row vector + a second independent copy from the same source.
v = [10 20 30];
w = v;
w(2) = 200;
fprintf('v = %.0f %.0f %.0f\n', v(1), v(2), v(3));
fprintf('w = %.0f %.0f %.0f\n', w(1), w(2), w(3));

% Chained copy: C = A (already modified A is untouched); D = C; D mutated.
C = A;
D = C;
D(2, 2) = 7;
fprintf('C(2,2) = %.0f  D(2,2) = %.0f\n', C(2,2), D(2,2));
