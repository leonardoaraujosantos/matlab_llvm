% RK4-style chained update with a nested matrix expression and unary minus.
X = [0.10; 0.0];
for n = 1:30
    k1 = [X(2); -X(1)];
    k2 = [X(2); -X(1) - 0.01];
    k3 = [X(2); -X(1) - 0.02];
    k4 = [X(2); -X(1) - 0.03];
    X = X + (0.02/6) * (k1 + 2*k2 + 2*k3 + k4);
end
fprintf('x1=%.6f x2=%.6f\n', X(1), X(2));
