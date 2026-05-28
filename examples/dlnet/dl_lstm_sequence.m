% dl_lstm_sequence.m — Deep Learning T4 headline: a functional LSTM trained
% end-to-end on a tiny memorisation task, via the project's reverse-mode
% autodiff (BPTT through a single OP_LSTM tape node).
%
% Task: read a length-6 binary sequence in {0,1}^6 and output a single bit
% indicating whether the *first* input was 1.  A canonical memory test —
% a plain feed-forward net can't see past the current step, but an LSTM
% with its forget gate can remember the first bit across the sequence.
%
% Setup: one LSTM cell with H = 4 hidden units, a 1×H linear head to a logit,
% sigmoid + binary cross-entropy.  Eight fixed sequences cover both labels
% with controlled noise so the test is deterministic without RNG.
%
% BPTT is automatic: the OP_LSTM tape node carries every per-timestep gate
% and state, and `dlgradient` walks them backward in time.

D = 1; H = 4; T = 6; N = 8;

% Each row is a length-6 sequence; label = first bit.  Four 1-leading and
% four 0-leading sequences, with different noise patterns on the tail.
Xall = [ 1 0 1 0 1 1;
         1 1 0 1 0 0;
         1 0 0 1 1 0;
         1 1 1 0 0 1;
         0 1 0 1 0 1;
         0 0 1 1 0 0;
         0 1 1 0 1 1;
         0 0 0 1 1 0 ];
Yall = Xall(:, 1).';     % 1 x N

rng(0);
Wx = dlarray(0.3 * randn(4*H, D));
Wr = dlarray(0.3 * randn(4*H, H));
bL = dlarray(zeros(4*H, 1));
Wy = dlarray(0.3 * randn(1, H));
by = dlarray(zeros(1, 1));

lr = 0.5;
nIter = 200;
initLoss = 0;
for it = 1:nIter
    sumL = 0;
    gWx_acc = zeros(4*H, D); gWr_acc = zeros(4*H, H); gbL_acc = zeros(4*H, 1);
    gWy_acc = zeros(1, H);   gby_acc = zeros(1, 1);
    for n = 1:N
        Xn = dlarray(Xall(n, :));        % 1 x T  (D x T with D = 1)
        h0 = dlarray(zeros(H, 1));
        c0 = dlarray(zeros(H, 1));
        target = dlarray(Yall(n));

        Hseq   = lstm(Xn, h0, c0, Wx, Wr, bL);   % H x T
        logits = Wy * Hseq + by;                 % 1 x T
        p      = sigmoid(mean(logits));          % scalar
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

% Evaluate.
correct = 0;
finalLoss = 0;
for n = 1:N
    Xn = dlarray(Xall(n, :));
    h0 = dlarray(zeros(H, 1));
    c0 = dlarray(zeros(H, 1));
    Hseq   = lstm(Xn, h0, c0, Wx, Wr, bL);
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
