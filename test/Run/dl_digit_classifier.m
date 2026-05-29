% dl_digit_classifier.m — 4-class quadrant classifier trained with Adam
% + dlreset() per iter, using the full Tier-C normalization family.
%
% Demonstrates the new tape-scoping + Adam + larger-than-toy model:
%   dlreset() at the top of each iter -> bounded tape size
%   adamupdate(W, g, m, v, t, lr, b1, b2, eps) -> Adam optimizer step
%   conv2d_full + batchnorm + relu + maxpool2d + reshape + FC + softmax
%
% Trains on 4 distinct 8x8 binary quadrant patterns -> 4 classes.

% 4 quadrant patterns stored directly into a rank-4 X (8x8x1x4) so
% Sema sees the matN type from allocation.
X = zeros(8, 8, 1, 4);
for i = 1:4
    for j = 1:4
        X(i, j, 1, 1) = 1.0;            % sample 1: top-left
        X(i, j + 4, 1, 2) = 1.0;        % sample 2: top-right
        X(i + 4, j, 1, 3) = 1.0;        % sample 3: bottom-left
        X(i + 4, j + 4, 1, 4) = 1.0;    % sample 4: bottom-right
    end
end
T_oh = [1 0 0 0;
        0 1 0 0;
        0 0 1 0;
        0 0 0 1];

% Initial weights — small uniform.
Wconv    = 0.05 * ones(3, 3, 1, 8);
bconv    = zeros(1, 8);
gamma_bn = ones(1, 8);
beta_bn  = zeros(1, 8);
W_fc     = 0.05 * ones(4, 128);

% Adam state.
M_Wc = zeros(3, 3, 1, 8); V_Wc = zeros(3, 3, 1, 8);
M_bc = zeros(1, 8);       V_bc = zeros(1, 8);
M_g  = zeros(1, 8);       V_g  = zeros(1, 8);
M_b  = zeros(1, 8);       V_b  = zeros(1, 8);
M_Wf = zeros(4, 128);     V_Wf = zeros(4, 128);

% ---- Initial forward pass (no training yet) — measure L0.
dlreset();
Xdl      = dlarray(X);
Tdl      = dlarray(T_oh);
Wconv_dl = dlarray(Wconv);
bconv_dl = dlarray(bconv);
gamma_dl = dlarray(gamma_bn);
beta_dl  = dlarray(beta_bn);
W_fc_dl  = dlarray(W_fc);
Y_c    = conv2d_full(Xdl, Wconv_dl, bconv_dl, 1, 1, 1, 1);
Y_p    = maxpool2d(relu(batchnorm(Y_c, gamma_dl, beta_dl)), 2, 2);
Y_f    = reshape(Y_p, 128, 4);
yhat0  = softmax(W_fc_dl * Y_f);
loss0  = crossentropy(yhat0, Tdl);
Lv0    = extractdata(loss0);

% ---- Training loop with Adam + dlreset() each iter.
lr = 0.05;
for t = 1:40
    dlreset();
    Xdl      = dlarray(X);
    Tdl      = dlarray(T_oh);
    Wconv_dl = dlarray(Wconv);
    bconv_dl = dlarray(bconv);
    gamma_dl = dlarray(gamma_bn);
    beta_dl  = dlarray(beta_bn);
    W_fc_dl  = dlarray(W_fc);

    Y_c    = conv2d_full(Xdl, Wconv_dl, bconv_dl, 1, 1, 1, 1);
    Y_bn   = batchnorm(Y_c, gamma_dl, beta_dl);
    Y_r    = relu(Y_bn);
    Y_p    = maxpool2d(Y_r, 2, 2);
    Y_f    = reshape(Y_p, 128, 4);
    logits = W_fc_dl * Y_f;
    yhat   = softmax(logits);
    loss   = crossentropy(yhat, Tdl);

    gWc = dlgradient(loss, Wconv_dl);
    gbc = dlgradient(loss, bconv_dl);
    gG  = dlgradient(loss, gamma_dl);
    gB  = dlgradient(loss, beta_dl);
    gWf = dlgradient(loss, W_fc_dl);

    Wconv    = adamupdate(Wconv,    gWc, M_Wc, V_Wc, t, lr, 0.9, 0.999, 1e-8);
    bconv    = adamupdate(bconv,    gbc, M_bc, V_bc, t, lr, 0.9, 0.999, 1e-8);
    gamma_bn = adamupdate(gamma_bn, gG,  M_g,  V_g,  t, lr, 0.9, 0.999, 1e-8);
    beta_bn  = adamupdate(beta_bn,  gB,  M_b,  V_b,  t, lr, 0.9, 0.999, 1e-8);
    W_fc     = adamupdate(W_fc,     gWf, M_Wf, V_Wf, t, lr, 0.9, 0.999, 1e-8);
end

% ---- Final forward pass — measure L_last + self-recognition trace.
dlreset();
Xdl      = dlarray(X);
Tdl      = dlarray(T_oh);
Wconv_dl = dlarray(Wconv);
bconv_dl = dlarray(bconv);
gamma_dl = dlarray(gamma_bn);
beta_dl  = dlarray(beta_bn);
W_fc_dl  = dlarray(W_fc);
Y_c      = conv2d_full(Xdl, Wconv_dl, bconv_dl, 1, 1, 1, 1);
Y_p      = maxpool2d(relu(batchnorm(Y_c, gamma_dl, beta_dl)), 2, 2);
Y_f      = reshape(Y_p, 128, 4);
yhatF    = softmax(W_fc_dl * Y_f);
lossF    = crossentropy(yhatF, Tdl);
LvF      = extractdata(lossF);

% Sum P .* T inside the dl graph so extractdata yields a 1x1; index
% (1,1) to coerce 1x1 -> f64.
trace_dl  = sum(sum(yhatF .* Tdl));
trace_mat = extractdata(trace_dl);
trace_v   = trace_mat(1, 1);
Lv0_v     = Lv0(1, 1);
LvF_v     = LvF(1, 1);

final_sz = dltape_size(0);

fprintf('dl_digit_classifier: loss(0)=%.3f loss(40)=%.3f\n', Lv0_v, LvF_v);
fprintf('dl_digit_classifier: final tape size = %.0f (bounded by dlreset)\n', final_sz);
fprintf('dl_digit_classifier: P .* T sum = %.3f / 4 (mean self-prob = %.3f)\n', ...
        trace_v, trace_v / 4);

if LvF_v < Lv0_v - 0.3 && trace_v > 2.5
    fprintf('dl_digit_classifier: PASS\n');
else
    fprintf('dl_digit_classifier: FAIL\n');
end
