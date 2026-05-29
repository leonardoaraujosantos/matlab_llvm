% dl_transformer_block.m — single-head Transformer encoder block headline.
% A tiny token classifier with the canonical encoder cell:
%
%   z   = embed(E, tokens)        (D × T)
%   a   = softmax(z' * Wq * Wk' * z / sqrt(D))   then z * a   (self-attention)
%   z2  = LayerNorm(z + scaled * z)              (residual + LN)
%   f   = gelu(Wf2 * relu(Wf1 * z2 + b1) + b2)   (FFN)
%   z3  = LayerNorm(z2 + f)                       (FFN residual + LN)
%   y   = softmax(Wo * z3)                       (per-token class logits)
%
% Uses only the Phase-1 small ops (sqrt / rdivide / mean(dim) / gelu) plus
% the shipped activation + reduction surface.  Trains 1 step on a fixed
% 6-token sequence + one-hot target and shows the loss drops.

% --- vocabulary + embedding ---------------------------------------------
V = 5; D = 4; T = 6; C = 3;        % vocab / model / seq-len / classes
toks   = [1 3 2 4 5 2];            % T-length token sequence (1-indexed)
target = [1 2 1 3 2 1];            % T-length class labels (1-indexed)

E   = dlarray(0.1 * eye(D, V));    % D x V embedding
Wq  = dlarray(0.1 * eye(D));       % D x D query
Wk  = dlarray(0.1 * eye(D));       % D x D key
Wv  = dlarray(0.1 * eye(D));       % D x D value
Wf1 = dlarray(0.1 * ones(D, D));   % FFN expand
b1  = dlarray(zeros(D, T));
Wf2 = dlarray(0.1 * ones(D, D));   % FFN project
b2  = dlarray(zeros(D, T));
Wo  = dlarray(0.1 * eye(C, D));    % output classifier
eps_dl = dlarray(1e-5 * ones(1, T));

% --- forward (embed -> self-attn -> FFN -> classifier) ------------------
z  = embed(E, toks);               % D x T  (one column per token)

% Self-attention: scaled dot-product, single head.
%   Q = Wq * z   K = Wk * z   V = Wv * z
%   A = softmax( (Q' * K) / sqrt(D) )   (T x T)
%   ctx = V * A                                   (D x T)
Q   = Wq * z;
K   = Wk * z;
V_  = Wv * z;
QK  = transpose(Q) * K;            % T x T
dsq = dlarray(sqrt(D) * ones(T, T));
scl = QK ./ dsq;
A   = softmax(scl);
ctx = V_ * A;                      % D x T

% First residual + LayerNorm.
h1     = z + ctx;
mu1    = mean(h1, 1);              % 1 x T
diff1  = h1 - mu1;
v1     = mean(diff1 .* diff1, 1);  % 1 x T
denm1  = sqrt(v1 + eps_dl);
z2     = diff1 ./ denm1;

% Position-wise FFN.
f      = gelu(Wf1 * z2 + b1);      % D x T
g      = Wf2 * f + b2;             % D x T

% Second residual + LayerNorm.
h2     = z2 + g;
mu2    = mean(h2, 1);
diff2  = h2 - mu2;
v2     = mean(diff2 .* diff2, 1);
denm2  = sqrt(v2 + eps_dl);
z3     = diff2 ./ denm2;

% Output classifier — per-token logits then softmax.
logits = Wo * z3;                  % C x T
yhat   = softmax(logits);

% --- loss: cross-entropy against one-hot target -------------------------
T_oh = zeros(C, T);
for k = 1:T
    T_oh(target(k), k) = 1.0;
end
T_dl = dlarray(T_oh);
loss = crossentropy(yhat, T_dl);

L0 = extractdata(loss);
fprintf('dl_transformer_block: loss(initial) = %.4f\n', L0);

% --- one Adam-flavoured step on Wo to show the loss decreases ----------
gWo  = dlgradient(loss, Wo);
Wo_v = extractdata(Wo) - 0.1 * gWo;
Wo2  = dlarray(Wo_v);

% Re-forward through the cheap subset (z3 doesn't depend on Wo).
logits2 = Wo2 * z3;
yhat2   = softmax(logits2);
loss2   = crossentropy(yhat2, T_dl);
L1      = extractdata(loss2);
fprintf('dl_transformer_block: loss(after  ) = %.4f\n', L1);

if L1 < L0
    fprintf('dl_transformer_block: PASS (loss decreased)\n');
else
    fprintf('dl_transformer_block: FAIL (loss did not decrease)\n');
end
