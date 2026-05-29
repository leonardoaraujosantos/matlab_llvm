% dl_residual_train.m — Deep Learning T5 (architectures).  Train a 4-layer
% MLP that uses skip connections — the dlarray autodiff handles the residual
% `+ x` automatically since `plus` is overloaded.  Compared to a plain
% feed-forward net of the same depth, the residual variant converges faster
% and to a lower loss on the same data, which is the practical pay-off of
% ResNet-style architectures.

D = 6; H = 6; N = 20;

rng(0);
Xd = randn(D, N);
% Target = a non-linear transform of X.
Td = tanh(Xd) + 0.3 * (Xd .* Xd);

% Trainable weights (4 layers).  Hidden width = input width so the skip
% connections are dimension-aligned.
W1 = dlarray(0.3 * randn(H, D)); b1 = dlarray(zeros(H, 1));
W2 = dlarray(0.3 * randn(H, H)); b2 = dlarray(zeros(H, 1));
W3 = dlarray(0.3 * randn(H, H)); b3 = dlarray(zeros(H, 1));
W4 = dlarray(0.3 * randn(D, H)); b4 = dlarray(zeros(D, 1));

X = dlarray(Xd); T = dlarray(Td);

lr = 0.05;
nIter = 200;
initLoss = 0;
for it = 1:nIter
    h1 = relu(W1 * X  + b1);
    % Residual block: relu(W3 * relu(W2*h1 + b2) + b3) + h1
    h2 = relu(W2 * h1 + b2);
    h3 = relu(W3 * h2 + b3) + h1;
    y  = W4 * h3 + b4;

    loss = mse(y, T);
    Lv = extractdata(loss);
    if it == 1; initLoss = Lv(1); end

    g1 = dlgradient(loss, W1); gb1 = dlgradient(loss, b1);
    g2 = dlgradient(loss, W2); gb2 = dlgradient(loss, b2);
    g3 = dlgradient(loss, W3); gb3 = dlgradient(loss, b3);
    g4 = dlgradient(loss, W4); gb4 = dlgradient(loss, b4);

    W1 = dlarray(extractdata(W1) - lr * g1); b1 = dlarray(extractdata(b1) - lr * gb1);
    W2 = dlarray(extractdata(W2) - lr * g2); b2 = dlarray(extractdata(b2) - lr * gb2);
    W3 = dlarray(extractdata(W3) - lr * g3); b3 = dlarray(extractdata(b3) - lr * gb3);
    W4 = dlarray(extractdata(W4) - lr * g4); b4 = dlarray(extractdata(b4) - lr * gb4);
end

% Final loss.
h1 = relu(W1 * X  + b1);
h2 = relu(W2 * h1 + b2);
h3 = relu(W3 * h2 + b3) + h1;
y  = W4 * h3 + b4;
Lf = extractdata(mse(y, T)); finalLoss = Lf(1);

fprintf('initial residual loss rounds to %.0f\n', round(initLoss));
fprintf('final residual loss rounds to %.0f\n', round(finalLoss));
