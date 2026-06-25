% 3x3 linear solve inside a loop, result drives an accumulation.
M = [4, 1, 0; 1, 3, 1; 0, 1, 2];
y = [0; 0; 0];
for n = 1:10
    b = [1.0; 2.0; 3.0] + 0.1 * y;
    y = M \ b;
end
fprintf('y=%.6f %.6f %.6f\n', y(1), y(2), y(3));
