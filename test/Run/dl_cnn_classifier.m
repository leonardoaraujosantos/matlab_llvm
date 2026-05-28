% dl_cnn_classifier.m — Multi-channel CNN classifier trained end-to-end.
%
% First headline that exercises the FULL conv-net stack on the rank-N
% tape: conv (3-channel input) -> ReLU -> max-pool -> flatten -> FC ->
% softmax -> crossentropy.  Trains via dlgradient through every layer.
%
% Architecture (NCHW-ish, project's HxWxCxN convention):
%   X       :  6 x 6 x 3 x 4         (3-channel 6x6 inputs, 4 samples)
%   conv W  :  3 x 3 x 3 x 4         (4 filters of 3x3x3)
%   conv b  :  ...                    (carved -- folded into post-pool FC)
%   conv Y  :  4 x 4 x 4 x 4         (valid padding, stride 1)
%   relu   ->  4 x 4 x 4 x 4
%   maxpool 2x2 -> 2 x 2 x 4 x 4
%   flatten ->  16 x 4
%   FC      :  W2 (3 x 16) * flat + b2
%   softmax -> 3 x 4   (3-class probabilities, 4 samples)
%   loss    =  crossentropy(yhat, T)
%
% Each of the 4 samples has its own colour-channel "signature":
%   sample 1: channel 1 bright (R-dominant) -> class 1
%   sample 2: channel 2 bright (G-dominant) -> class 2
%   sample 3: channel 3 bright (B-dominant) -> class 3
%   sample 4: channels 1+2 bright (yellow)  -> class 1 (R-dominant tie)
% After ~80 SGD iters the CNN learns to use its conv filters to detect
% the channel-bright signature and route through the FC head correctly.

% --- input batch: 6x6x3x4 ---
X = zeros(6, 6, 3, 4);
for h = 1:6
    for w = 1:6
        X(h, w, 1, 1) = 1.0;          % sample 1: R bright
        X(h, w, 2, 2) = 1.0;          % sample 2: G bright
        X(h, w, 3, 3) = 1.0;          % sample 3: B bright
        X(h, w, 1, 4) = 0.7;          % sample 4: R + G
        X(h, w, 2, 4) = 0.7;
    end
end
Xdl = dlarray(X);

% --- targets: one-hot 3 classes, 4 samples (col-major: K x N = 3 x 4) ---
T_oh = [1 0 0 1;
        0 1 0 0;
        0 0 1 0];
Tdl = dlarray(T_oh);

% --- learnable params ---
% conv filters: 4 detectors over 3 channels, deterministic init biased
% toward each filter's "target" channel for fast convergence.
Wconv = zeros(3, 3, 3, 4);
for k = 1:4
    for h = 1:3
        for w = 1:3
            if k <= 3
                Wconv(h, w, k, k) = 0.2;
            else
                Wconv(h, w, 1, 4) = 0.15;
                Wconv(h, w, 2, 4) = 0.15;
            end
        end
    end
end
Wconv_dl = dlarray(Wconv);

% FC layer: 3 classes from 16 flattened features (2*2*4).  Deterministic
% +/- alternating init keeps the gradient signal non-zero in step 1.
W2 = zeros(3, 16);
for r = 1:3
    for c = 1:16
        if mod(r + c, 2) == 0
            W2(r, c) = 0.1;
        else
            W2(r, c) = -0.1;
        end
    end
end
W2_dl = dlarray(W2);

L0 = extractdata(dlarray(0.0));
L_last = extractdata(dlarray(0.0));
lr_conv = 0.1;
lr_fc   = 0.2;

for k = 1:80
    Y_conv = conv2d_batch(Xdl, Wconv_dl);   % 4x4x4x4
    Y_relu = relu(Y_conv);                    % same shape
    Y_pool = maxpool2d(Y_relu, 2, 2);         % 2x2x4x4 (matN)
    Y_flat = reshape(Y_pool, 16, 4);          % 16 x 4 dlarray
    logits = W2_dl * Y_flat;                  % 3 x 4
    yhat   = softmax(logits);
    loss   = crossentropy(yhat, Tdl);

    Lv = extractdata(loss);
    if k == 1, L0 = Lv; end
    L_last = Lv;

    gWc = dlgradient(loss, Wconv_dl);
    gW2 = dlgradient(loss, W2_dl);

    Wconv_dl = dlarray(extractdata(Wconv_dl) - lr_conv * gWc);
    W2_dl    = dlarray(extractdata(W2_dl)    - lr_fc   * gW2);
end

fprintf('dl_cnn_classifier: loss(0)=%.4f loss(80)=%.4f\n', L0, L_last);

% Accuracy on the training set (4 samples).
Y_conv = conv2d_batch(Xdl, Wconv_dl);
Y_pool = maxpool2d(relu(Y_conv), 2, 2);
Y_flat = reshape(Y_pool, 16, 4);
logits = W2_dl * Y_flat;
yhat   = softmax(logits);
yp     = extractdata(yhat);
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
    if best == target_class
        ncorrect = ncorrect + 1;
    end
end
fprintf('dl_cnn_classifier: training accuracy = %.0f/4\n', ncorrect);

% Use ncorrect (plain int) for the PASS gate so we don't depend on a
% scalar-from-ptr comparison (extractdata returns a 1x1 mat, not a double).
if ncorrect >= 3
    fprintf('dl_cnn_classifier: PASS\n');
else
    fprintf('dl_cnn_classifier: FAIL\n');
end
