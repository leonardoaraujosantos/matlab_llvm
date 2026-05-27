% Deep Learning Tier-2 — reverse-mode autodiff (dlgradient).
% loss = sum(sigmoid(W*x)); analytic grad = sigmoid'(z_i)*x_j (hand-verified).
W = dlarray([0.5 -0.3; 0.2 0.8]);
x = dlarray([1.0; 2.0]);
loss = sum(sigmoid(W*x));
fprintf('loss = %.4f\n', extractdata(loss));
g = dlgradient(loss, W);
fprintf('dL/dW = [%.4f %.4f; %.4f %.4f]\n', g(1,1), g(1,2), g(2,1), g(2,2));
% gradient w.r.t. x as well
gx = dlgradient(loss, x);
fprintf('dL/dx = [%.4f; %.4f]\n', gx(1), gx(2));
