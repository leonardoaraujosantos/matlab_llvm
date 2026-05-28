% Deep Learning T4 gating test — bidirectional LSTM, BPTT through both
% directions via packed [forward; backward] weights.  The output is 2H × T
% with the backward sub-sequence re-aligned to original time order.

D = 2; H = 2; T = 3;

Xd = [ 0.5  0.2 -0.3;
      -0.1  0.4  0.7 ];
% Target shape matches Y -> 2H × T = 4 × 3.
Td = [ 0.0  0.1  0.2;
       0.3  0.4  0.5;
      -0.1 -0.2 -0.3;
       0.6  0.5  0.4 ];

W_init = 0.1 * ones(8*H, D);
R_init = 0.1 * ones(8*H, H);
b_init = zeros(8*H, 1);

X   = dlarray(Xd);
H0f = dlarray(zeros(H, 1)); C0f = dlarray(zeros(H, 1));
H0b = dlarray(zeros(H, 1)); C0b = dlarray(zeros(H, 1));
W = dlarray(W_init);
R = dlarray(R_init);
b = dlarray(b_init);

Y = bilstm(X, H0f, C0f, H0b, C0b, W, R, b);
loss = mse(Y, dlarray(Td));
L0v = extractdata(loss); L0 = L0v(1);

gW = dlgradient(loss, W);
gR = dlgradient(loss, R);
gb = dlgradient(loss, b);

lr = 0.2;
W2 = dlarray(W_init - lr * gW);
R2 = dlarray(R_init - lr * gR);
b2 = dlarray(b_init - lr * gb);
Y2 = bilstm(X, H0f, C0f, H0b, C0b, W2, R2, b2);
loss2 = mse(Y2, dlarray(Td));
L1v = extractdata(loss2); L1 = L1v(1);

Yd = extractdata(Y);
shape_ok = 0;
if size(Yd, 1) == 2*H
    if size(Yd, 2) == T
        shape_ok = 1;
    end
end

gW_m = sum(sum(gW .* gW));
gR_m = sum(sum(gR .* gR));
gb_m = sum(gb .* gb);

% Both directions of the packed weights must receive non-zero gradient.
forward_grad = 0; backward_grad = 0;
for r_idx = 1:(4*H)
    for d = 1:D
        forward_grad = forward_grad + abs(gW(r_idx, d));
    end
end
for r_idx = (4*H + 1):(8*H)
    for d = 1:D
        backward_grad = backward_grad + abs(gW(r_idx, d));
    end
end
both_dirs = 0;
if forward_grad > 0
    if backward_grad > 0
        both_dirs = 1;
    end
end
loss_drop = 0;
if L1 < L0
    loss_drop = 1;
end

fprintf('bilstm shape ok = %.0f\n', shape_ok);
fprintf('both directions get gradient = %.0f\n', both_dirs);
fprintf('loss drops after BPTT step = %.0f\n', loss_drop);
