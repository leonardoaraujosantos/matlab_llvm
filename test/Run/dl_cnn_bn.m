% dl_cnn_bn.m — Deeper CNN with BatchNorm + same-padded conv with bias.
%
% Demonstrates the new Tier-C tape ops in a realistic flow:
%   conv2d_full (pad=1, stride=1, bias) -> batchnorm -> relu -> avgpool
%   -> reshape (flatten) -> FC -> softmax_dim(., 1) -> crossentropy.
%
% Same 3-channel colour-signature task as dl_cnn_classifier.m, but with:
%   - same-padding so the output of conv keeps the spatial dims (6x6),
%     letting us use BN over a meaningful (H*W*N) = 6*6*4 = 144-cell pop
%     per channel.
%   - bias term on the conv.
%   - BN between conv and ReLU.
%   - softmax over the class axis (dim=1) instead of column-wise softmax
%     (semantically identical for this shape — exercises the dim path).

% --- input batch: 6x6x3x4 (same as the simpler CNN demo) ---
X = zeros(6, 6, 3, 4);
for h = 1:6
    for w = 1:6
        X(h, w, 1, 1) = 1.0;          % R bright
        X(h, w, 2, 2) = 1.0;          % G bright
        X(h, w, 3, 3) = 1.0;          % B bright
        X(h, w, 1, 4) = 0.7;          % R + G
        X(h, w, 2, 4) = 0.7;
    end
end
Xdl = dlarray(X);

% --- targets: 3-class one-hot, 4 samples ---
T_oh = [1 0 0 1;
        0 1 0 0;
        0 0 1 0];
Tdl = dlarray(T_oh);

% --- learnable params ---
% Conv: 4 filters of 3x3x3 (biased toward each input channel).
Wconv = zeros(3, 3, 3, 4);
for k = 1:4
    for h = 1:3
        for w = 1:3
            if k <= 3
                Wconv(h, w, k, k) = 0.15;
            else
                Wconv(h, w, 1, 4) = 0.10;
                Wconv(h, w, 2, 4) = 0.10;
            end
        end
    end
end
bconv = zeros(1, 4);   % conv bias, one per filter

% BN params (one per output channel = 4).
gamma_bn = ones(1, 4);
beta_bn  = zeros(1, 4);

% FC layer.  After same-padded conv we keep 6x6x4x4; avgpool 2x2 -> 3x3x4x4.
% Flatten 3*3*4 = 36 features per sample, 4 samples.
W_fc = zeros(3, 36);
for r = 1:3
    for c = 1:36
        if mod(r + c, 3) == 0
            W_fc(r, c) = 0.08;
        else
            W_fc(r, c) = -0.05;
        end
    end
end

Wconv_dl   = dlarray(Wconv);
bconv_dl   = dlarray(bconv);
gamma_dl   = dlarray(gamma_bn);
beta_dl    = dlarray(beta_bn);
W_fc_dl    = dlarray(W_fc);

L0 = extractdata(dlarray(0.0));
L_last = extractdata(dlarray(0.0));
lr_conv = 0.10;
lr_bn   = 0.05;
lr_fc   = 0.15;

for k = 1:60
    Y_c  = conv2d_full(Xdl, Wconv_dl, bconv_dl, 1, 1, 1, 1);   % 6x6x4x4 (same pad)
    Y_bn = batchnorm(Y_c, gamma_dl, beta_dl);                  % 6x6x4x4
    Y_r  = relu(Y_bn);                                          % 6x6x4x4
    Y_p  = avgpool2d(Y_r, 2, 2);                                % 3x3x4x4
    Y_f  = reshape(Y_p, 36, 4);                                 % 36 x 4
    logits = W_fc_dl * Y_f;                                     % 3 x 4
    yhat   = softmax(logits, 1);                                % axis=1 (class)
    loss   = crossentropy(yhat, Tdl);

    Lv = extractdata(loss);
    if k == 1, L0 = Lv; end
    L_last = Lv;

    gWc = dlgradient(loss, Wconv_dl);
    gbc = dlgradient(loss, bconv_dl);
    gG  = dlgradient(loss, gamma_dl);
    gB  = dlgradient(loss, beta_dl);
    gWf = dlgradient(loss, W_fc_dl);

    Wconv_dl  = dlarray(extractdata(Wconv_dl) - lr_conv * gWc);
    bconv_dl  = dlarray(extractdata(bconv_dl) - lr_conv * gbc);
    gamma_dl  = dlarray(extractdata(gamma_dl) - lr_bn   * gG);
    beta_dl   = dlarray(extractdata(beta_dl)  - lr_bn   * gB);
    W_fc_dl   = dlarray(extractdata(W_fc_dl)  - lr_fc   * gWf);
end

fprintf('dl_cnn_bn: loss(0)=%.4f loss(60)=%.4f\n', L0, L_last);

% Accuracy on the training batch.
Y_c  = conv2d_full(Xdl, Wconv_dl, bconv_dl, 1, 1, 1, 1);
Y_p  = avgpool2d(relu(batchnorm(Y_c, gamma_dl, beta_dl)), 2, 2);
Y_f  = reshape(Y_p, 36, 4);
yhat = softmax(W_fc_dl * Y_f, 1);
yp   = extractdata(yhat);
ncorrect = 0;
for n = 1:4
    best = 1; bv = yp(1, n);
    for c = 2:3
        if yp(c, n) > bv
            best = c; bv = yp(c, n);
        end
    end
    target_class = n;
    if n == 4, target_class = 1; end
    if best == target_class, ncorrect = ncorrect + 1; end
end
fprintf('dl_cnn_bn: training accuracy = %.0f/4\n', ncorrect);

if ncorrect >= 3
    fprintf('dl_cnn_bn: PASS\n');
else
    fprintf('dl_cnn_bn: FAIL\n');
end
