% dl_cnn_train.m — CNN with autodiff through rank-4 conv2d_batch.
%
% End-to-end forward AND backward through a true convolutional layer
% over the Tier C matN descriptor.  Trains a single 3x3x1x2 filter
% bank on a per-cell MSE target so the demo doesn't depend on reshape /
% global-pool on dlarray (those would need separate OP_RESHAPE /
% OP_MEAN_DIM tape nodes on matN -- carved follow-ons).
%
% Architecture: conv2d_batch -> mse against a target rank-4 tensor.
% Input shape:  5x5x1x2  (two single-channel "images")
% Filter shape: 3x3x1x2  (two filters)
% Output shape: 3x3x2x2  (Hout=Wout=3, K=2, N=2)
% Target:       3x3x2x2  (filter k should fire on sample k; off-diagonal zero)

% Two-sample input: pure vertical bar / pure NW-SE diagonal.
X = zeros(5, 5, 1, 2);
for i = 1:5
    X(i, 3, 1, 1) = 1.0;
    X(i, i, 1, 2) = 1.0;
end
Xdl = dlarray(X);

% Per-cell target: filter 1 high on sample 1, filter 2 high on sample 2.
T = zeros(3, 3, 2, 2);
for h = 1:3
    for w = 1:3
        T(h, w, 1, 1) = 1.0;        % filter 1 on sample 1: HIGH
        T(h, w, 2, 2) = 1.0;        % filter 2 on sample 2: HIGH
    end
end
Tdl = dlarray(T);

% Small initial filter bank wrapped in a dlarray.  Updates stay on the
% matrix-pointer lane (extractdata - lr*gW returns a fresh matN ptr,
% wrapped back in dlarray); this sidesteps the tensor-typed slot vs
% ptr-typed gradient mismatch that bites the plain SGD form.
Wdl = dlarray(0.1 * ones(3, 3, 1, 2));

L0 = extractdata(dlarray(0.0));
L_last = extractdata(dlarray(0.0));
for k = 1:30
    Y = conv2d_batch(Xdl, Wdl);      % 3x3x2x2 matN
    loss = mse(Y, Tdl);

    Lv = extractdata(loss);
    if k == 1, L0 = Lv; end
    L_last = Lv;

    gW  = dlgradient(loss, Wdl);
    lr  = 1.0;
    Wdl = dlarray(extractdata(Wdl) - lr * gW);
end

fprintf('dl_cnn_train: loss(0)=%.4f loss(30)=%.4f\n', L0, L_last);
if L_last < L0 - 0.05
    fprintf('dl_cnn_train: PASS (loss dropped through conv backprop)\n');
else
    fprintf('dl_cnn_train: FAIL (loss did not decrease)\n');
end
