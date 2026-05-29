% dl_gru_sequence.m — Deep Learning T4: a functional GRU trained end-to-end
% on the same first-bit-memory task as `dl_lstm_sequence.m`.  The forward
% pass uses fewer gates than LSTM (reset + update + candidate, no separate
% cell state), so on the same task the GRU should also reach 100% accuracy
% while being a step lighter than the LSTM headline.

D = 1; H = 4; T = 6; N = 8;

Xall = [ 1 0 1 0 1 1;
         1 1 0 1 0 0;
         1 0 0 1 1 0;
         1 1 1 0 0 1;
         0 1 0 1 0 1;
         0 0 1 1 0 0;
         0 1 1 0 1 1;
         0 0 0 1 1 0 ];
Yall = Xall(:, 1).';

rng(0);
Wx = dlarray(0.3 * randn(3*H, D));
Wr = dlarray(0.3 * randn(3*H, H));
bL = dlarray(zeros(3*H, 1));
Wy = dlarray(0.3 * randn(1, H));
by = dlarray(zeros(1, 1));

lr = 0.5;
nIter = 200;
initLoss = 0;
for it = 1:nIter
    sumL = 0;
    gWx_acc = zeros(3*H, D); gWr_acc = zeros(3*H, H); gbL_acc = zeros(3*H, 1);
    gWy_acc = zeros(1, H);   gby_acc = zeros(1, 1);
    for n = 1:N
        Xn = dlarray(Xall(n, :));
        h0 = dlarray(zeros(H, 1));
        target = dlarray(Yall(n));

        Hseq   = gru(Xn, h0, Wx, Wr, bL);
        logits = Wy * Hseq + by;
        p      = sigmoid(mean(logits));
        oneT   = dlarray(1) - target;
        oneP   = dlarray(1) - p;
        loss_n = dlarray(0) - (target * log(p) + oneT * log(oneP));

        Lv = extractdata(loss_n); sumL = sumL + Lv(1);

        gWx_acc = gWx_acc + dlgradient(loss_n, Wx);
        gWr_acc = gWr_acc + dlgradient(loss_n, Wr);
        gbL_acc = gbL_acc + dlgradient(loss_n, bL);
        gWy_acc = gWy_acc + dlgradient(loss_n, Wy);
        gby_acc = gby_acc + dlgradient(loss_n, by);
    end
    if it == 1; initLoss = sumL; end

    Wx = dlarray(extractdata(Wx) - (lr / N) * gWx_acc);
    Wr = dlarray(extractdata(Wr) - (lr / N) * gWr_acc);
    bL = dlarray(extractdata(bL) - (lr / N) * gbL_acc);
    Wy = dlarray(extractdata(Wy) - (lr / N) * gWy_acc);
    by = dlarray(extractdata(by) - (lr / N) * gby_acc);
end

correct = 0;
finalLoss = 0;
for n = 1:N
    Xn = dlarray(Xall(n, :));
    h0 = dlarray(zeros(H, 1));
    Hseq   = gru(Xn, h0, Wx, Wr, bL);
    logits = Wy * Hseq + by;
    p      = sigmoid(mean(logits));
    pv     = extractdata(p);
    pred   = pv(1) > 0.5;
    if pred == (Yall(n) > 0.5); correct = correct + 1; end
    target = dlarray(Yall(n));
    oneT   = dlarray(1) - target;
    oneP   = dlarray(1) - p;
    loss_n = dlarray(0) - (target * log(p) + oneT * log(oneP));
    Lv     = extractdata(loss_n); finalLoss = finalLoss + Lv(1);
end

fprintf('initial loss rounds to %.0f\n', round(initLoss));
fprintf('final loss rounds to %.0f\n', round(finalLoss));
fprintf('memory-task accuracy = %.0f\n', 100 * correct / N);
