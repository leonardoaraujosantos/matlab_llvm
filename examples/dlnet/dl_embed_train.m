% dl_embed_train.m — Deep Learning T4.4: wordEmbeddingLayer's functional
% core trained end-to-end via the autodiff (gather forward + scatter-add
% backward through OP_EMBED).
%
% Toy task: a 5-word vocabulary where each word has a known 3-dim "meaning"
% target.  We learn an embedding matrix E (3 x 5) by sampling words from a
% sequence and minimising MSE against the targets.  Repeated tokens
% accumulate gradient through the scatter-add — making the per-token
% contribution invariant to repetition, exactly the behaviour the
% wordEmbeddingLayer relies on for NLP training.

D = 3; Vv = 5;

% Targets per token (the embedding the network should learn).
T_true = [ 1.0 -1.0  0.5  0.2 -0.3;
           0.0  0.5 -0.5  0.8  0.1;
          -0.5  0.2  0.3 -0.4  0.6 ];

% Random initial embedding.
rng(0);
E = dlarray(0.1 * randn(D, Vv));

% A training "sentence" — token ids that visit some words multiple times.
seq = [1 2 3 1 4 5 2 3 4 5 1 2];

% Lift the targets corresponding to seq into a (D x length(seq)) matrix.
Ts = zeros(D, length(seq));
for n = 1:length(seq)
    for d = 1:D
        Ts(d, n) = T_true(d, seq(n));
    end
end
Tdl = dlarray(Ts);

lr = 0.5;
nIter = 150;
initLoss = 0;
for it = 1:nIter
    Y = embed(E, seq);
    loss = mse(Y, Tdl);
    Lv = extractdata(loss);
    if it == 1; initLoss = Lv(1); end
    gE = dlgradient(loss, E);
    E = dlarray(extractdata(E) - lr * gE);
end

% Final loss + spot-check: each learned column should be close to T_true.
Y = embed(E, seq);
Lf = extractdata(mse(Y, Tdl)); finalLoss = Lf(1);

learned = extractdata(E);
max_err = 0;
for v = 1:Vv
    for d = 1:D
        e = abs(learned(d, v) - T_true(d, v));
        if e > max_err; max_err = e; end
    end
end

converged = 0;
if max_err < 0.01
    converged = 1;
end

fprintf('initial embedding loss rounds to %.0f\n', round(initLoss));
fprintf('final embedding loss rounds to %.0f\n', round(finalLoss));
fprintf('per-element error under 0.01 = %.0f\n', converged);
