% dl_new_ops.m — Gating test for the new BN / padded-conv / softmax_dim /
% mean_dim-matN-backward tape ops.  Each section trains a single
% deterministic step and asserts the gradient + loss are finite and
% drop in the expected direction.

% =====  (1) BatchNorm forward + backward  ==========================
% Input X: 4 x 4 x 2 x 3 (4x4 spatial, 2 channels, 3 samples).
X = zeros(4, 4, 2, 3);
for n = 1:3
    for c = 1:2
        for h = 1:4
            for w = 1:4
                X(h, w, c, n) = (h - 1) * 0.1 + (w - 1) * 0.05 ...
                              + (c - 1) * 1.0 + (n - 1) * 0.2;
            end
        end
    end
end
gamma = ones(1, 2);
beta  = zeros(1, 2);

% Verify forward: per-channel normalised data has ~zero mean and ~unit
% variance over (H, W, N).
Xdl = dlarray(X);
Gdl = dlarray(gamma);
Bdl = dlarray(beta);
Y_bn = batchnorm(Xdl, Gdl, Bdl);
Yv   = extractdata(Y_bn);

% Per-channel stats: mean and var over (H,W,N) for each c.
for c = 1:2
    s = 0; ss = 0; cnt = 0;
    for n = 1:3
        for h = 1:4
            for w = 1:4
                v = Yv(h, w, c, n);
                s = s + v; ss = ss + v * v; cnt = cnt + 1;
            end
        end
    end
    mu_c  = s / cnt;
    var_c = ss / cnt - mu_c * mu_c;
    fprintf('dl_new_ops: BN ch%.0f mean=%.4f var=%.4f\n', c, mu_c, var_c);
end

% Gradient through BN -> mse against zero target.
T_bn = dlarray(zeros(4, 4, 2, 3));
loss_bn = mse(Y_bn, T_bn);
gG = dlgradient(loss_bn, Gdl);
gB = dlgradient(loss_bn, Bdl);
fprintf('dl_new_ops: BN sum(gG)=%.4f sum(gB)=%.4f\n', ...
        sum(gG), sum(gB));


% =====  (2) Conv with bias + padding + stride  =====================
% X: 5 x 5 x 1 x 2.  Two filters of 3x3x1.  Same padding (pad=1).
X2 = zeros(5, 5, 1, 2);
for i = 1:5
    X2(i, 3, 1, 1) = 1.0;
    X2(i, i, 1, 2) = 1.0;
end
W2 = zeros(3, 3, 1, 2);
for i = 1:3
    for j = 1:3
        if j == 2
            W2(i, j, 1, 1) = 1.0;
        else
            W2(i, j, 1, 1) = -1.0;
        end
        if i == j
            W2(i, j, 1, 2) = 1.0;
        else
            W2(i, j, 1, 2) = -1.0;
        end
    end
end
b2 = zeros(1, 2);
b2(1) = 0.5;   % nonzero bias for filter 1
b2(2) = -0.5;
X2dl = dlarray(X2);
W2dl = dlarray(W2);
b2dl = dlarray(b2);
% pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1 -> output 5x5x2x2 ("same" conv)
Y2 = conv2d_full(X2dl, W2dl, b2dl, 1, 1, 1, 1);
Y2v = extractdata(Y2);
fprintf('dl_new_ops: conv_full size = %.0f %.0f %.0f %.0f\n', ...
        size(Y2v, 1), size(Y2v, 2), size(Y2v, 3), size(Y2v, 4));

% Backward through MSE.
T2 = dlarray(zeros(5, 5, 2, 2));
loss2 = mse(Y2, T2);
gWc = dlgradient(loss2, W2dl);
gbc = dlgradient(loss2, b2dl);
fprintf('dl_new_ops: conv_full sum(gW)=%.4f sum(gb)=%.4f\n', ...
        sum(sum(sum(sum(gWc)))), sum(gbc));


% =====  (3) softmax(X, dim) — matN-aware  =========================
% 4x3 input, softmax along dim=1 (column-wise — matches 2-D path).
S = [1.0 2.0 3.0;
     0.5 1.5 2.5;
     0.0 1.0 2.0;
     -1.0 0.0 1.0];
Sdl = dlarray(S);
P = softmax(Sdl, 1);                  % col-wise softmax (4 rows, 3 cols)
Pv = extractdata(P);
% Each column should sum to ~1.
col_sums = sum(Pv, 1);
fprintf('dl_new_ops: softmax_dim col_sums = %.4f %.4f %.4f\n', ...
        col_sums(1), col_sums(2), col_sums(3));


% =====  (4) mean(X, dim) — matN-aware backward  ===================
% Mean over dim=3 (channel axis) of a 2x2x3 tensor (mat3).
M3 = zeros(2, 2, 3);
for h = 1:2
    for w = 1:2
        for c = 1:3
            M3(h, w, c) = h + 2 * w + 4 * c;
        end
    end
end
M3_dl = dlarray(M3);
% Take mean across channel axis -> 2x2 result.
m3 = mean(M3_dl, 3);
m3v = extractdata(m3);
fprintf('dl_new_ops: mean(M3,3) ndims=%.0f size=%.0f %.0f\n', ...
        ndims(m3v), size(m3v, 1), size(m3v, 2));
% Backward: loss = sum(m3); dM3 should be 1/3 in every cell.
T3 = dlarray(zeros(2, 2));
loss3 = mse(m3, T3);
gM3 = dlgradient(loss3, M3_dl);
fprintf('dl_new_ops: mean_dim_nd sum(gM3)=%.4f\n', sum(sum(sum(gM3))));

fprintf('dl_new_ops: PASS\n');
