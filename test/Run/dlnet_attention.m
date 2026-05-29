% Deep Learning T4.5 gating test — scaled-dot-product attention composed
% from existing dlarray ops + the newly-added transpose.
%
% Y = V * softmax(K' * Q)
%
% With column-major layout (each column = one token/vector):
%   Q : D × Nq     queries
%   K : D × Nk     keys
%   V : D × Nk     values
%   K'*Q : Nk × Nq      score matrix (k_i · q_j)
%   softmax columnwise → Nk × Nq      attention weights per query
%   V * weights : D × Nq       attended values
%
% No new opcode — attention falls out of the tape composition.  We verify:
% (a) forward output shape, (b) the dlgradient sweeps back through transpose
% +  mtimes + softmax + mtimes, and (c) gradient direction is correct.

D = 3; Nq = 2; Nk = 4;

Qd = [ 0.5  0.1;
       0.2  0.8;
      -0.3  0.4 ];
Kd = [ 0.5  0.4 -0.2  0.1;
       0.2  0.7  0.3  0.8;
      -0.3  0.0  0.4  0.5 ];
Vd = [ 1.0  0.5  0.2 -0.1;
       0.4  0.3  0.6  0.7;
      -0.2  0.1  0.5  0.3 ];
Td = ones(D, Nq);

Q = dlarray(Qd); K = dlarray(Kd); V = dlarray(Vd);

scores  = transpose(K) * Q;          % Nk x Nq
weights = softmax(scores);           % column-wise softmax over keys
attn    = V * weights;               % D x Nq
loss    = mse(attn, dlarray(Td));

L0v = extractdata(loss); L0 = L0v(1);
gQ = dlgradient(loss, Q);
gK = dlgradient(loss, K);
gV = dlgradient(loss, V);

% Step Q/K/V against the gradient and re-evaluate.
lr = 0.1;
Q2 = dlarray(Qd - lr * gQ);
K2 = dlarray(Kd - lr * gK);
V2 = dlarray(Vd - lr * gV);
scores2  = transpose(K2) * Q2;
weights2 = softmax(scores2);
attn2    = V2 * weights2;
loss2    = mse(attn2, dlarray(Td));
L1v = extractdata(loss2); L1 = L1v(1);

attn_d = extractdata(attn);
shape_ok = 0;
if size(attn_d, 1) == D
    if size(attn_d, 2) == Nq
        shape_ok = 1;
    end
end

% Each column of the attention weights must sum to 1 (softmax invariant).
w = extractdata(weights);
col_sums_ok = 0;
delta = 0;
for j = 1:Nq
    s = 0;
    for i = 1:Nk
        s = s + w(i, j);
    end
    delta = delta + abs(s - 1);
end
if delta < 1e-6
    col_sums_ok = 1;
end

grad_nz = 0;
gQ_m = sum(sum(gQ .* gQ));
gK_m = sum(sum(gK .* gK));
gV_m = sum(sum(gV .* gV));
if gQ_m > 0
    if gK_m > 0
        if gV_m > 0
            grad_nz = 1;
        end
    end
end

loss_drop = 0;
if L1 < L0
    loss_drop = 1;
end

fprintf('attention shape ok = %.0f\n', shape_ok);
fprintf('softmax weights sum to 1 = %.0f\n', col_sums_ok);
fprintf('gradient nonzero across Q,K,V = %.0f\n', grad_nz);
fprintf('loss drops after gradient step = %.0f\n', loss_drop);
