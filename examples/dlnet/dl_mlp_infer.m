% Deep Learning Toolbox Tier-1 — MLP inference with dlarray.
% A 2-layer classifier forward pass written with natural operators on
% dlarray (the UG "custom" / functional form — the object-array layer
% container is a documented carve-down).  No autodiff needed for inference.
W1 = dlarray([0.10 0.20 -0.15 0.05; -0.30 0.40 0.25 0.10; 0.05 -0.20 0.30 0.45]);
b1 = dlarray([0.1; -0.1; 0.05]);
W2 = dlarray([0.3 -0.2 0.5; -0.4 0.6 0.1; 0.2 0.1 -0.3]);
b2 = dlarray([0.0; 0.0; 0.0]);

features = dlarray([5.1; 3.5; 1.4; 0.2]);   % one iris-like sample
scores   = softmax(W2*relu(W1*features + b1) + b2);
p        = extractdata(scores);
fprintf('class scores = %.4f %.4f %.4f\n', p(1), p(2), p(3));
[mx, idx] = max([p(1) p(2) p(3)]);
fprintf('predicted class = %.0f (score %.4f)\n', idx, mx);
