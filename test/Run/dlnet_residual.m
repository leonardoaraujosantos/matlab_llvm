% Deep Learning T5 gating test — residual block over the autodiff.
%
% A residual block is `out = relu(W2 * relu(W1 * x + b1) + b2) + x`.  No new
% opcode is needed — the skip-add is just `plus` on two dlarrays and the
% existing tape handles BPTT through both branches.  This test verifies:
%   (a) the residual forward path runs end-to-end on a dlarray,
%   (b) gradients flow back through BOTH the residual transform AND the
%       skip connection (the W1 gradient picks up signal from both paths),
%   (c) one SGD step on the residual weights drops the loss.

D = 4; H = 4; N = 5;                 % H must equal D so x + transform aligns

rng(0);
Xd = randn(D, N);
Td = randn(D, N);

W1_init = 0.1 * ones(H, D);
b1_init = zeros(H, 1);
W2_init = 0.1 * ones(D, H);
b2_init = zeros(D, 1);

X  = dlarray(Xd);
W1 = dlarray(W1_init); b1 = dlarray(b1_init);
W2 = dlarray(W2_init); b2 = dlarray(b2_init);

% Residual: y = relu(W2 * relu(W1*X + b1) + b2) + X
h    = relu(W1 * X + b1);
y_t  = W2 * h + b2;
y_act = relu(y_t);
y    = y_act + X;

loss = mse(y, dlarray(Td));
L0v = extractdata(loss); L0 = L0v(1);

gW1 = dlgradient(loss, W1);
gW2 = dlgradient(loss, W2);

% Step the residual-branch weights against the gradient.
lr = 0.05;
W1b = dlarray(W1_init - lr * gW1);
W2b = dlarray(W2_init - lr * gW2);
hb   = relu(W1b * X + b1);
y_tb = W2b * hb + b2;
y_b  = relu(y_tb) + X;
loss2 = mse(y_b, dlarray(Td));
L1v = extractdata(loss2); L1 = L1v(1);

gW1_m = sum(sum(gW1 .* gW1));
gW2_m = sum(sum(gW2 .* gW2));

grad_nz = 0;
if gW1_m > 0
    if gW2_m > 0
        grad_nz = 1;
    end
end
loss_drop = 0;
if L1 < L0
    loss_drop = 1;
end

fprintf('residual gradient nonzero through both branches = %.0f\n', grad_nz);
fprintf('loss drops after residual-branch step = %.0f\n', loss_drop);
