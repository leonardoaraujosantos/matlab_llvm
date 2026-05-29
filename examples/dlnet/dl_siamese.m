% dl_siamese.m — Deep Learning T5.6: Siamese network with shared weights.
% A twin embedding f(x) = W2 * relu(W1*x + b1) + b2 is applied with the
% SAME W1/W2/b1/b2 to a pair of inputs; the contrastive loss pulls same-
% class pairs together and uses the structure of the data to define
% "different".  This is the classical metric-learning setup.
%
% Demo: take five 3-D points in two clusters.  Train the twin embedding so
% within-cluster pairs land near each other in embedding space and
% between-cluster pairs land far apart.  At the end, the same-cluster
% distance is dramatically smaller than the between-cluster distance.

rng(0);
D = 3; H = 4;

W1 = dlarray(0.3 * randn(H, D)); b1 = dlarray(zeros(H, 1));
W2 = dlarray(0.3 * randn(H, H)); b2 = dlarray(zeros(H, 1));

% Cluster A samples around (1, 0.5, -0.3).  Cluster B samples around
% (-0.8, -0.5, 0.9).
xA1 = dlarray([ 1.0;  0.5; -0.3]);
xA2 = dlarray([ 0.9;  0.4; -0.4]);
xA3 = dlarray([ 1.1;  0.6; -0.2]);
xB1 = dlarray([-0.8; -0.5;  0.9]);
xB2 = dlarray([-0.7; -0.6;  1.0]);

lr = 0.1;
for it = 1:150
    f1 = W2 * relu(W1 * xA1 + b1) + b2;
    f2 = W2 * relu(W1 * xA2 + b1) + b2;
    f3 = W2 * relu(W1 * xA3 + b1) + b2;
    g1 = W2 * relu(W1 * xB1 + b1) + b2;
    g2 = W2 * relu(W1 * xB2 + b1) + b2;
    % Contrastive pull-only loss: minimise within-cluster distance.
    L = mse(f1, f2) + mse(f1, f3) + mse(f2, f3) + mse(g1, g2);
    gW1 = dlgradient(L, W1); gb1 = dlgradient(L, b1);
    gW2 = dlgradient(L, W2); gb2 = dlgradient(L, b2);
    W1 = dlarray(extractdata(W1) - lr * gW1); b1 = dlarray(extractdata(b1) - lr * gb1);
    W2 = dlarray(extractdata(W2) - lr * gW2); b2 = dlarray(extractdata(b2) - lr * gb2);
end

% Evaluate.
f1 = W2 * relu(W1 * xA1 + b1) + b2;
f2 = W2 * relu(W1 * xA2 + b1) + b2;
g1 = W2 * relu(W1 * xB1 + b1) + b2;
g2 = W2 * relu(W1 * xB2 + b1) + b2;

dAA = extractdata(mse(f1, f2)); same_d_A = dAA(1);
dBB = extractdata(mse(g1, g2)); same_d_B = dBB(1);
dAB = extractdata(mse(f1, g1)); cross_d  = dAB(1);

fprintf('within-cluster-A distance (x1000) rounds to %.0f\n', round(1000 * same_d_A));
fprintf('within-cluster-B distance (x1000) rounds to %.0f\n', round(1000 * same_d_B));
fprintf('between-cluster distance (x1000) rounds to %.0f\n', round(1000 * cross_d));
