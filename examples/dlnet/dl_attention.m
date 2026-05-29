% dl_attention.m — Deep Learning T4.5: scaled-dot-product attention trained
% end-to-end through the project's reverse-mode autodiff.  The forward pass
% is composed from existing dlarray ops + the newly-added transpose;
% backward is `dlgradient` sweeping through that composition.
%
% Setup: a tiny "associative recall" task.  Given a set of 4 (key, value)
% pairs (D = 3) and a single query that matches exactly one of the keys,
% learn projection matrices Wq, Wk, Wv such that attention(Wq*q, Wk*K, Wv*V)
% reproduces the target value.  This is the smallest demonstrable
% transformer-style operation: the network must learn to focus on the
% correct key via the softmax over scores.

D = 3; Nk = 4;

% Fixed key/value pairs (column-major: each column = one entry).
K_raw = [ 1.0  0.0  0.0  0.5;
          0.0  1.0  0.0 -0.5;
          0.0  0.0  1.0  0.5 ];
V_raw = [ 2.0  3.0  4.0  5.0;
         -1.0 -2.0 -3.0 -4.0;
          0.5  0.6  0.7  0.8 ];
% Query is a perturbed copy of key 2; target = value 2.
q_raw  = [ 0.0; 0.95; 0.05 ];
target = V_raw(:, 2);

% Trainable projection matrices.  Random-perturbed-from-identity init keeps
% the forward well-scaled (softmax doesn't saturate) while still giving the
% optimiser non-trivial work — without the perturbation the constructed
% target lands on the initial output and the loss starts at zero.
rng(0);
init = eye(D) + 0.4 * randn(D, D);
Wq = dlarray(init);
Wk = dlarray(init);
Wv = dlarray(init);

% Lift constants into dlarray once (cheaper than re-wrapping each iter).
qd = dlarray(q_raw);
Kd = dlarray(K_raw);
Vd = dlarray(V_raw);
Td = dlarray(target);

lr = 0.05;
nIter = 80;
initLoss = 0;
for it = 1:nIter
    Q = Wq * qd;      % D x 1
    K = Wk * Kd;      % D x Nk
    V = Wv * Vd;      % D x Nk

    scores  = transpose(K) * Q;     % Nk x 1
    weights = softmax(scores);      % column-wise softmax
    attn    = V * weights;          % D x 1

    loss = mse(attn, Td);

    Lv = extractdata(loss);
    if it == 1; initLoss = Lv(1); end

    gWq = dlgradient(loss, Wq);
    gWk = dlgradient(loss, Wk);
    gWv = dlgradient(loss, Wv);

    Wq = dlarray(extractdata(Wq) - lr * gWq);
    Wk = dlarray(extractdata(Wk) - lr * gWk);
    Wv = dlarray(extractdata(Wv) - lr * gWv);
end

% Final readout.
Q  = Wq * qd;
K  = Wk * Kd;
V  = Wv * Vd;
sc = transpose(K) * Q;
w  = softmax(sc);
y  = V * w;
finalLoss = extractdata(mse(y, Td));
wv = extractdata(w);

fprintf('initial attention loss rounds to %.0f\n', round(initLoss));
fprintf('final attention loss rounds to %.0f\n', round(finalLoss(1)));
% Verify the softmax learned to peak on the matching key (index 2).
peak = 1; bestv = wv(1, 1);
for i = 2:Nk
    if wv(i, 1) > bestv; bestv = wv(i, 1); peak = i; end
end
fprintf('attention peaks on key %.0f\n', peak);
