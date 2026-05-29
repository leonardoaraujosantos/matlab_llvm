% Deep Learning T5 gating test — transfer learning pattern over the
% autodiff.  `dlnetwork`/`freezeLayers`/`replaceLayer` are carved object-
% array APIs, but the underlying training pattern works today with a custom
% loop: keep the pretrained feature extractor as plain numeric matrices
% (no gradient flow), and train only the classifier head as a dlarray.
%
% This test verifies that the head receives gradient and trains, while
% the frozen weights' identity (we never wrap them as dlarray and never
% call dlgradient on them) means they are mathematically guaranteed to
% remain unchanged.  The data uses three clear class prototypes so the
% test is deterministic and reaches > 90% accuracy.

D = 4; H = 6; C = 3; N = 9;

% "Pretrained" feature extractor (frozen).
Wenc = [ 0.5 -0.2  0.3  0.1;
         0.1  0.4 -0.5  0.2;
        -0.3  0.2  0.4 -0.1;
         0.6 -0.1  0.0  0.3;
         0.0  0.5  0.1 -0.4;
        -0.2  0.3 -0.4  0.5 ];
benc = [0.1; -0.2; 0.0; 0.05; 0.15; -0.1];

% Three class prototypes (3 samples per class).
Xd = [ 1.5  1.6  1.4  -0.4 -0.5 -0.3   0.1  0.0  0.2;
       0.2  0.3  0.1   1.2  1.3  1.1  -0.5 -0.6 -0.4;
      -0.3 -0.2 -0.4   0.6  0.5  0.7   1.0  1.1  0.9;
       0.5  0.6  0.4  -0.5 -0.4 -0.6   1.2  1.3  1.1 ];
Td = [ 1 1 1 0 0 0 0 0 0;
       0 0 0 1 1 1 0 0 0;
       0 0 0 0 0 0 1 1 1 ];
labels = [1 1 1 2 2 2 3 3 3];

% Frozen forward pass — plain matrices, no autodiff.
Z = zeros(H, N);
for n = 1:N
    for h = 1:H
        s = benc(h);
        for d = 1:D
            s = s + Wenc(h, d) * Xd(d, n);
        end
        if s < 0; s = 0; end
        Z(h, n) = s;
    end
end

% Trainable classifier head only.
W2 = dlarray(0.1 * ones(C, H));
b2 = dlarray(zeros(C, 1));
Zdl = dlarray(Z);
Tdl = dlarray(Td);

lr = 0.3;
initLoss = 0;
for it = 1:120
    logits = W2 * Zdl + b2;
    probs  = softmax(logits);
    loss   = crossentropy(probs, Tdl);
    Lv = extractdata(loss);
    if it == 1; initLoss = Lv(1); end
    gW2 = dlgradient(loss, W2);
    gb2 = dlgradient(loss, b2);
    W2 = dlarray(extractdata(W2) - lr * gW2);
    b2 = dlarray(extractdata(b2) - lr * gb2);
end

logits = W2 * Zdl + b2;
probs  = softmax(logits);
finalL = extractdata(crossentropy(probs, Tdl)); finalLoss = finalL(1);
P  = extractdata(probs);

correct = 0;
for n = 1:N
    pred = 1; bestv = P(1, n);
    for c = 2:C
        if P(c, n) > bestv; bestv = P(c, n); pred = c; end
    end
    if pred == labels(n); correct = correct + 1; end
end

loss_drop = 0;
if finalLoss < initLoss
    loss_drop = 1;
end

fprintf('head loss drops with frozen encoder = %.0f\n', loss_drop);
fprintf('transfer-learning accuracy = %.0f\n', 100 * correct / N);
