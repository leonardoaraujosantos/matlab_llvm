% Nested loops with an inner matrix accumulation (sub-stepping).
X = [0.0; 0.0];
for outer = 1:5
    for inner = 1:4
        X = X + [0.1; 0.2];
    end
    fprintf('outer=%d X=(%.2f, %.2f)\n', outer, X(1), X(2));
end
