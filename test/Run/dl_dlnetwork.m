% dl_dlnetwork — dlnetwork carrier + trainnet driver end-to-end.
%
% Unlocks the carve-down item "dlnetwork object-array carrier +
% trainnet builtin driver" without needing classdef array literals
% (separate Sema gate).  The carrier stores layer descriptors in a
% thread-local runtime vector keyed by an integer handle; the
% user-facing API is:
%
%   net = dlnetwork();         -- create a fresh net (handle scalar)
%   net = addFC(net, W, b);    -- append fully-connected layer
%   net = addRelu(net);        -- append ReLU
%   net = addSigmoid(net);     -- append sigmoid
%   net = addTanh(net);        -- append tanh
%   net = addSoftmax(net);     -- append softmax (column-wise)
%   Y   = netPredict(net, X);  -- forward
%   loss = trainnet(net, X, Y_target, lr, n_iter);  -- Adam training
%
% trainnet runs forward, computes MSE loss, backprops through every
% layer (FC: dW/db via dY*x^T / sum(dY, dim=2); activations: analytic
% derivatives), and updates FC weights with Adam.

% Tiny binary classification: 4 samples, 3 features -> 2 classes.
X = [1.0  0.5  0.2  0.8;
     0.3  0.9  0.1  0.4;
     0.7  0.2  0.6  0.5];

T = [1 0 1 0;
     0 1 0 1];

% Architecture: 3 -> 4 -> 2 -> softmax.
W1 = 0.1 * ones(4, 3);
b1 = zeros(4, 1);
W2 = 0.1 * ones(2, 4);
b2 = zeros(2, 1);

net = dlnetwork();
net = addFC(net, W1, b1);
net = addRelu(net);
net = addFC(net, W2, b2);
net = addSoftmax(net);

n_layers = netNumLayers(net);
fprintf('dl_dlnetwork: net has %.0f layers\n', n_layers);

% Initial forward (before training).
Y0 = netPredict(net, X);
fprintf('dl_dlnetwork: pre-train Y(:, 1) = [%.3f; %.3f]\n', Y0(1, 1), Y0(2, 1));

% Train.
lr = 0.05;
n_iter = 200;
final_loss = trainnet(net, X, T, lr, n_iter);

% Post-train forward.
Yf = netPredict(net, X);
fprintf('dl_dlnetwork: post-train Y(:, 1) = [%.3f; %.3f]\n', Yf(1, 1), Yf(2, 1));
fprintf('dl_dlnetwork: post-train Y(:, 2) = [%.3f; %.3f]\n', Yf(1, 2), Yf(2, 2));
fprintf('dl_dlnetwork: final loss = %.4f\n', final_loss);

% PASS criteria: 4 layers tracked, post-train Y(:,1) leans toward class 1
% (Yf(1,1) > Yf(2,1)) and Y(:,2) leans toward class 2.
if n_layers == 4 && Yf(1, 1) > Yf(2, 1) && Yf(2, 2) > Yf(1, 2) && final_loss < 0.5
    fprintf('dl_dlnetwork: PASS\n');
else
    fprintf('dl_dlnetwork: FAIL\n');
end
