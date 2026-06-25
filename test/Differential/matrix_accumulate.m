% Loop-carried matrix accumulation: X = X + dt*k (the canonical ODE/Euler step).
X = [0; 0; 0.2; 0];
k = [1; -1; 0.5; -0.5];
for i = 1:20
    X = X + 0.05 * k;
end
disp(X);
