% dl_quantize_check.m — Deep Learning HDL Tier-H1: train a small MLP in
% double precision, then quantize every weight to symmetric INT8 and verify
% the resulting (dequantized) inference still classifies the toy task
% correctly.  This closes the software side of the HDL on-ramp: H2 will
% emit the quantized forward as fi-typed SystemVerilog; H3 will cocotb-
% verify the SV bit-accurately against the dequantized double inference
% computed here.
%
% Task: same 3-class linearly-separable toy from `dl_mlp_train.m` (6 points
% in 2-D, 3 classes).  After ~300 SGD iterations the double net is at 100%
% accuracy with loss < 0.02; quantizing the W1/b1/W2/b2 weights to INT8
% must keep accuracy at 100% (the per-tensor scale is the int8 LSB, and the
% network has enough margin to absorb it).

rng(0);
Xd = [1 1.2 4 4.2 2.5 2.3; 1 0.8 1 0.8 4 4.2];
Td = [1 1 0 0 0 0; 0 0 1 1 0 0; 0 0 0 0 1 1];
labels = [1 1 2 2 3 3];

X = dlarray(Xd); T = dlarray(Td);
W1 = dlarray(0.5*randn(8,2)); b1 = dlarray(zeros(8,1));
W2 = dlarray(0.5*randn(3,8)); b2 = dlarray(zeros(3,1));

lr = 0.5;
for it = 1:300
    H = relu(W1*X + b1);
    Y = softmax(W2*H + b2);
    loss = crossentropy(Y, T);
    gW1 = dlgradient(loss, W1); gb1 = dlgradient(loss, b1);
    gW2 = dlgradient(loss, W2); gb2 = dlgradient(loss, b2);
    W1 = dlarray(extractdata(W1) - lr*gW1); b1 = dlarray(extractdata(b1) - lr*gb1);
    W2 = dlarray(extractdata(W2) - lr*gW2); b2 = dlarray(extractdata(b2) - lr*gb2);
end

% --- Double-precision inference (baseline) ---------------------------------
% Plain-matrix bias broadcasting isn't wired, so apply the bias per column
% explicitly.  (Inside the autodiff dlnet handles column-broadcast itself.)
W1d = extractdata(W1); b1d = extractdata(b1);
W2d = extractdata(W2); b2d = extractdata(b2);

Z1d = W1d * Xd;
for j = 1:size(Z1d, 2)
    for i = 1:size(Z1d, 1)
        Z1d(i, j) = Z1d(i, j) + b1d(i);
        if Z1d(i, j) < 0; Z1d(i, j) = 0; end
    end
end
Z2d = W2d * Z1d;
for j = 1:size(Z2d, 2)
    for i = 1:size(Z2d, 1)
        Z2d(i, j) = Z2d(i, j) + b2d(i);
    end
end
Y_d_logits = Z2d;

% --- INT8-quantized inference ----------------------------------------------
W1q = dlquantize(W1d); b1q = dlquantize(b1d);
W2q = dlquantize(W2d); b2q = dlquantize(b2d);

Z1q = W1q * Xd;
for j = 1:size(Z1q, 2)
    for i = 1:size(Z1q, 1)
        Z1q(i, j) = Z1q(i, j) + b1q(i);
        if Z1q(i, j) < 0; Z1q(i, j) = 0; end
    end
end
Z2q = W2q * Z1q;
for j = 1:size(Z2q, 2)
    for i = 1:size(Z2q, 1)
        Z2q(i, j) = Z2q(i, j) + b2q(i);
    end
end
Y_q_logits = Z2q;

% --- Compare accuracy + reconstruction error -------------------------------
correct_d = 0; correct_q = 0;
for j = 1:6
    pd = 1; bd = Y_d_logits(1, j);
    pq = 1; bq = Y_q_logits(1, j);
    for k = 2:3
        if Y_d_logits(k, j) > bd; bd = Y_d_logits(k, j); pd = k; end
        if Y_q_logits(k, j) > bq; bq = Y_q_logits(k, j); pq = k; end
    end
    if pd == labels(j); correct_d = correct_d + 1; end
    if pq == labels(j); correct_q = correct_q + 1; end
end

max_logit_drift = 0;
for i = 1:3
    for j = 1:6
        e = abs(Y_d_logits(i, j) - Y_q_logits(i, j));
        if e > max_logit_drift; max_logit_drift = e; end
    end
end

s1 = dlqscale(W1d); s2 = dlqscale(W2d);
fprintf('double accuracy = %.0f\n', 100 * correct_d / 6);
fprintf('int8 accuracy = %.0f\n', 100 * correct_q / 6);
fprintf('W1 scale (int8 LSB) x 1000 rounds to %.0f\n', round(1000 * s1(1)));
fprintf('W2 scale (int8 LSB) x 1000 rounds to %.0f\n', round(1000 * s2(1)));
fprintf('max logit drift x 100 rounds to %.0f\n', round(100 * max_logit_drift));
