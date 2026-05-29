% dl_rmsprop_prune — exercises rmspropupdate + magnitude pruning helpers.
%
% Trains a tiny 2-feature linear regressor (no bias) via RMSProp until loss
% drops, then prune_mask + mask_sparsity sanity-check.

X = [1 0; 0 1; 1 1; 2 1; 1 2; 2 2];
y = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0];

W = zeros(2, 1);
V_W = zeros(2, 1);
lr = 0.05;
gamma = 0.9;
eps = 1e-8;

dlreset();
Xdl = dlarray(X);
ydl = dlarray(y);
Wdl = dlarray(W);
yhat = Xdl * Wdl;
diff = yhat - ydl;
loss = sum(diff .* diff);
Lv0 = extractdata(loss);
L0 = Lv0(1, 1);

for t = 1:200
    dlreset();
    Xdl = dlarray(X);
    ydl = dlarray(y);
    Wdl = dlarray(W);
    yhat = Xdl * Wdl;
    diff = yhat - ydl;
    loss = sum(diff .* diff);
    gW = dlgradient(loss, Wdl);
    W = rmspropupdate(W, gW, V_W, lr, gamma, eps);
end

dlreset();
Xdl = dlarray(X);
ydl = dlarray(y);
Wdl = dlarray(W);
yhat = Xdl * Wdl;
diff = yhat - ydl;
loss_f = sum(diff .* diff);
Lf = extractdata(loss_f);
L_last = Lf(1, 1);

% Prune the bottom 50% of |W| on a surrogate of known magnitudes.
Wsurr = [3.0; -2.5; 0.05; -0.03];
M = prune_mask(Wsurr, 0.5);
sp = mask_sparsity(M);

fprintf('dl_rmsprop_prune: loss(0)=%.3f loss(200)=%.3f\n', L0, L_last);
fprintf('dl_rmsprop_prune: prune sparsity = %.2f (target 0.50)\n', sp);

ok_loss = L_last < L0 * 0.10;
ok_sp = abs(sp - 0.5) < 0.01;
if ok_loss && ok_sp
    fprintf('dl_rmsprop_prune: PASS\n');
else
    fprintf('dl_rmsprop_prune: FAIL\n');
end
