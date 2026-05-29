% Deep Learning T4 gating test — projected LSTM.  Hidden state runs at full
% size H internally, but the recurrence + the output go through a learned
% projection P (Hp × H, Hp < H).  Verifies (a) output shape uses Hp not H,
% (b) gradient also flows through P, (c) one BPTT step drops the loss.

D = 2; H = 4; Hp = 2; T = 3;

Xd = [ 1.0  0.5 -0.3;
       0.2 -0.1  0.4 ];
Td = [ 0.1  0.2  0.3;
      -0.1  0.0  0.1 ];

W_init = 0.1 * ones(4*H, D);
R_init = 0.1 * ones(4*H, Hp);
P_init = 0.1 * ones(Hp, H);
b_init = zeros(4*H, 1);

X  = dlarray(Xd);
h0 = dlarray(zeros(Hp, 1));   % projected initial hidden
c0 = dlarray(zeros(H, 1));    % full-size initial cell
W  = dlarray(W_init);
R  = dlarray(R_init);
P  = dlarray(P_init);
b  = dlarray(b_init);

Y = lstmp(X, h0, c0, W, R, P, b);
loss = mse(Y, dlarray(Td));
L0v = extractdata(loss); L0 = L0v(1);

gW = dlgradient(loss, W);
gR = dlgradient(loss, R);
gP = dlgradient(loss, P);
gb = dlgradient(loss, b);

lr = 0.3;
W2 = dlarray(W_init - lr * gW);
R2 = dlarray(R_init - lr * gR);
P2 = dlarray(P_init - lr * gP);
b2 = dlarray(b_init - lr * gb);
Y2 = lstmp(X, h0, c0, W2, R2, P2, b2);
loss2 = mse(Y2, dlarray(Td));
L1v = extractdata(loss2); L1 = L1v(1);

Yd = extractdata(Y);
shape_ok = 0;
if size(Yd, 1) == Hp
    if size(Yd, 2) == T
        shape_ok = 1;
    end
end

gP_m = sum(sum(gP .* gP));
gR_m = sum(sum(gR .* gR));
proj_learns = 0;
if gP_m > 0
    if gR_m > 0
        proj_learns = 1;
    end
end
loss_drop = 0;
if L1 < L0
    loss_drop = 1;
end

fprintf('lstmp shape uses Hp = %.0f\n', shape_ok);
fprintf('projection gets gradient = %.0f\n', proj_learns);
fprintf('loss drops after BPTT step = %.0f\n', loss_drop);
