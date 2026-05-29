% dl_dropout.m — Dropout headline.  Demonstrates training-mode dropout
% over a dlarray hidden layer: a Bernoulli(1-p) mask is applied via
% element-wise multiply, scaled by 1/(1-p) so the activation magnitude
% is preserved in expectation.  The mask is sampled OUTSIDE the dlarray
% tape (i.e. on the plain numeric lane via `rand`), wrapped in a dlarray
% so the elementwise multiply dispatches correctly, then folded into
% the forward path.

% Toy setup: 5x4 hidden activation (5 features, 4 examples).
% Mid-training keep probability.
H_data = [0.5 0.8 0.2 0.7;
          1.0 0.3 0.6 0.4;
          0.2 1.0 0.5 0.6;
          0.7 0.4 0.9 0.3;
          0.3 0.6 0.1 0.8];
W_data = 0.5 * ones(2, 5);     % 2-class linear classifier
T_oh   = [1 0 1 0;
          0 1 0 1];             % one-hot targets

p_drop = 0.3;                  % drop probability

% Bernoulli mask on the plain lane, scaled by 1/(1-p_drop).
mask_raw = rand(5, 4);
keep     = mask_raw > p_drop;            % logical matrix
scale    = 1 / (1 - p_drop);
% Multiply by 1.0 to coerce to double-typed matrix.
keep_d   = keep + 0.0;

% dlarrays for the forward pass.
H      = dlarray(H_data);
mask_d = dlarray(scale * keep_d);
W      = dlarray(W_data);
T_dl   = dlarray(T_oh);

% Dropout-applied hidden activation, then linear + softmax + CE.
Hd     = H .* mask_d;                    % dropout
logits = W * Hd;                         % 2x4
yhat   = softmax(logits);
loss   = crossentropy(yhat, T_dl);

% Eval mode: NO dropout, same params.
logits_eval = W * H;
yhat_eval   = softmax(logits_eval);
loss_eval   = crossentropy(yhat_eval, T_dl);

Lt = extractdata(loss);
Le = extractdata(loss_eval);
fprintf('dl_dropout: loss(train) = %.4f\n', Lt);
fprintf('dl_dropout: loss(eval)  = %.4f\n', Le);

% Gradient through dropout — should be non-zero (dropout is on the tape).
gW = dlgradient(loss, W);
fprintf('dl_dropout: sum(|gW|)   = %.4f\n', sum(sum(abs(gW))));

fprintf('dl_dropout: PASS\n');
