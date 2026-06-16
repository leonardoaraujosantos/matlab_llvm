% dl_trainnet — the MATLAB training API on the dlnetwork carrier (#296).
%
%   options = trainingOptions(solver, "MaxEpochs", N, "InitialLearnRate", lr)
%   net     = trainnet(X, T, net, lossFcn, options)
%
% trainingOptions builds an options handle (solver + hyperparameters);
% trainnet runs forward/backward and updates the FC weights with the chosen
% solver (adam / sgdm / rmsprop), returning the trained net. Loss is MSE.

% Tiny binary classification: 4 samples, 3 features -> 2 classes.
X = [1.0 0.5 0.2 0.8;
     0.3 0.9 0.1 0.4;
     0.7 0.2 0.6 0.5];
T = [1 0 1 0;
     0 1 0 1];

net = dlnetwork();
net = addFC(net, 0.1*ones(4,3), zeros(4,1));
net = addRelu(net);
net = addFC(net, 0.1*ones(2,4), zeros(2,1));
net = addSoftmax(net);

Y0 = netPredict(net, X);
fprintf('dl_trainnet: pre-train  Y(:,1) = [%.3f; %.3f]\n', Y0(1,1), Y0(2,1));

options = trainingOptions("adam", "MaxEpochs", 200, "InitialLearnRate", 0.05);
net = trainnet(X, T, net, "mse", options);

Yf = netPredict(net, X);
fprintf('dl_trainnet: post-train Y(:,1) = [%.3f; %.3f]\n', Yf(1,1), Yf(2,1));
fprintf('dl_trainnet: post-train Y(:,2) = [%.3f; %.3f]\n', Yf(1,2), Yf(2,2));

if Yf(1,1) > Yf(2,1) && Yf(2,2) > Yf(1,2)
    fprintf('dl_trainnet: PASS\n');
else
    fprintf('dl_trainnet: FAIL\n');
end
