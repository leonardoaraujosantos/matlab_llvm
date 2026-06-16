% regress_dl_trainnet.m — #296: the MATLAB training API
%   options = trainingOptions(solver, "MaxEpochs", N, "InitialLearnRate", lr)
%   net     = trainnet(X, T, net, lossFcn, options)
% wired on top of the dlnetwork carrier + shared trainer. Covers adam / sgdm /
% rmsprop solver selection and that the returned net is trained (the tiny
% binary problem must be classified correctly post-train). The custom
% trainnet(net,X,T,lr,n_iter) form is exercised by examples/dlnet/dl_dlnetwork.m.

X = [1.0 0.5 0.2 0.8;
     0.3 0.9 0.1 0.4;
     0.7 0.2 0.6 0.5];
T = [1 0 1 0;
     0 1 0 1];

solvers = ["adam", "sgdm", "rmsprop"];
for s = 1:3
    net = dlnetwork();
    net = addFC(net, 0.1*ones(4,3), zeros(4,1));
    net = addRelu(net);
    net = addFC(net, 0.1*ones(2,4), zeros(2,1));
    net = addSoftmax(net);
    options = trainingOptions(solvers(s), "MaxEpochs", 300, "InitialLearnRate", 0.05);
    net = trainnet(X, T, net, "mse", options);
    Yf = netPredict(net, X);
    if Yf(1,1) > Yf(2,1) && Yf(2,2) > Yf(1,2)
        fprintf('solver %.0f: trained OK\n', s);
    else
        fprintf('solver %.0f: FAIL\n', s);
    end
end
