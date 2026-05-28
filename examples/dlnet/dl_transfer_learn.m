% dl_transfer_learn.m — Deep Learning T5 (transfer learning).  Adapt a
% pretrained feature extractor to a new downstream task by freezing the
% encoder and training only a fresh classifier head over the autodiff.
%
% In MATLAB's object-array API this is `freezeLayers(net, …)` + `replaceLayer`
% — both wait on `dlnetwork`.  The underlying *training* pattern works
% today: keep the pretrained encoder as plain numeric matrices (no dlarray,
% no gradient flow) and train only the head with `dlgradient`.  Because the
% encoder is never wrapped as a dlarray and never appears as a `var` argument
% to `dlgradient`, it is mathematically guaranteed to remain unchanged across
% training.

% --- Pretrained encoder (plain matrices, frozen) ---------------------------
% This would normally be loaded from a saved network — we synthesise a small
% deterministic encoder that has learned to project D=4 inputs to H=6 useful
% features.
D = 4; H = 6; C = 3; N = 24;

Wenc = [ 0.5 -0.2  0.3  0.1;
         0.1  0.4 -0.5  0.2;
        -0.3  0.2  0.4 -0.1;
         0.6 -0.1  0.0  0.3;
         0.0  0.5  0.1 -0.4;
        -0.2  0.3 -0.4  0.5 ];
benc = [0.1; -0.2; 0.0; 0.05; 0.15; -0.1];

rng(0);
% Three classes, each with a characteristic direction in D-space.
Xd = zeros(D, N);
Td = zeros(C, N);
labels = zeros(1, N);
for n = 1:N
    cls = 1 + mod(n - 1, 3);
    labels(n) = cls;
    Td(cls, n) = 1;
    if cls == 1
        proto = [ 1.5; 0.2; -0.3; 0.5];
    elseif cls == 2
        proto = [-0.4;  1.2;  0.6; -0.5];
    else
        proto = [ 0.1; -0.5;  1.0;  1.2];
    end
    Xd(:, n) = proto + 0.2 * randn(D, 1);
end

% Run X through the frozen encoder, ONCE, outside the autodiff.
Z = zeros(H, N);
for n = 1:N
    for h = 1:H
        s = benc(h);
        for d = 1:D
            s = s + Wenc(h, d) * Xd(d, n);
        end
        if s < 0; s = 0; end       % ReLU
        Z(h, n) = s;
    end
end

% --- New classifier head trained over the autodiff -------------------------
Whead = dlarray(0.3 * randn(C, H));
bhead = dlarray(zeros(C, 1));
Zdl   = dlarray(Z);                % features are constants to the head
Tdl   = dlarray(Td);

lr = 0.3;
nIter = 150;
initLoss = 0;
for it = 1:nIter
    logits = Whead * Zdl + bhead;
    probs  = softmax(logits);
    loss   = crossentropy(probs, Tdl);
    Lv = extractdata(loss);
    if it == 1; initLoss = Lv(1); end
    gW = dlgradient(loss, Whead);
    gb = dlgradient(loss, bhead);
    Whead = dlarray(extractdata(Whead) - lr * gW);
    bhead = dlarray(extractdata(bhead) - lr * gb);
end

% --- Evaluate ----------------------------------------------------------------
logits = Whead * Zdl + bhead;
probs  = softmax(logits);
Lf = extractdata(crossentropy(probs, Tdl)); finalLoss = Lf(1);
P  = extractdata(probs);

correct = 0;
for n = 1:N
    pred = 1; bestv = P(1, n);
    for c = 2:C
        if P(c, n) > bestv; bestv = P(c, n); pred = c; end
    end
    if pred == labels(n); correct = correct + 1; end
end

fprintf('initial head loss rounds to %.0f\n', round(initLoss));
fprintf('final head loss rounds to %.0f\n', round(finalLoss));
fprintf('transfer-learning accuracy = %.0f\n', 100 * correct / N);
