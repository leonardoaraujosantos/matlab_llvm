% Deep Learning T4.4 gating test — wordEmbeddingLayer's functional core.
%
% Y = embed(E, idx) gathers E(:, idx(n)) into the n-th column of Y; the
% pullback scatter-adds the adjoint back into the touched embedding columns
% (only those columns receive a gradient).
%
% Workflow: define a 3 x 5 embedding matrix and a length-4 token sequence
% that visits some indices more than once.  Verify forward shape, train
% gradient against a target, take one SGD step, confirm the loss drops.

D = 3; Vv = 5; N = 4;

Ed = [ 0.1  0.2 -0.1  0.4  0.5;
       0.3 -0.2  0.1  0.0  0.6;
      -0.4  0.5  0.2  0.3 -0.1 ];
idx = [2 4 2 5];          % token-2 visited twice -> gradient must double up

E = dlarray(Ed);
Y = embed(E, idx);        % D x N
Td = ones(D, N);
T  = dlarray(Td);
loss = mse(Y, T);

L0v = extractdata(loss); L0 = L0v(1);
gE  = dlgradient(loss, E);

% Step against the gradient and re-evaluate.
lr = 0.5;
E2 = dlarray(Ed - lr * gE);
Y2 = embed(E2, idx);
loss2 = mse(Y2, T);
L1v = extractdata(loss2); L1 = L1v(1);

Yd = extractdata(Y);
shape_ok = 0;
if size(Yd, 1) == D
    if size(Yd, 2) == N
        shape_ok = 1;
    end
end

% Untouched columns of E (index 1, 3 in 1-based) must have zero gradient.
% Touched indices: 2, 4, 5.
untouched_zero = 0;
delta = 0;
for d = 1:D
    delta = delta + abs(gE(d, 1)) + abs(gE(d, 3));
end
if delta < 1e-12
    untouched_zero = 1;
end

% Index 2 was visited twice -> its gradient column should be roughly the
% sum of two single-visit contributions (i.e. larger than columns hit once).
g2_mag = 0; g4_mag = 0;
for d = 1:D
    g2_mag = g2_mag + abs(gE(d, 2));
    g4_mag = g4_mag + abs(gE(d, 4));
end
double_count_ok = 0;
if g2_mag > g4_mag
    double_count_ok = 1;
end

loss_drop = 0;
if L1 < L0
    loss_drop = 1;
end

fprintf('embed shape ok = %.0f\n', shape_ok);
fprintf('untouched columns have zero gradient = %.0f\n', untouched_zero);
fprintf('repeated index accumulates gradient = %.0f\n', double_count_ok);
fprintf('loss drops after gradient step = %.0f\n', loss_drop);
