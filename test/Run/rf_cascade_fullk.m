% Generalized Redheffer star: arbitrary k inner-connection ports.
% For 2-port self-cascade (k=1), result must match cascadeSparams2 +
% cascadeSparamsNFull (the fixed-k=N/2 variant) exactly.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");

% Use k=1 → exactly equivalent to cascadeSparamsNFull (N=2, k=N/2=1).
res = cascadeSparamsNFullK(data, data, 1);
disp(res.NumPorts);              % 2
disp(tsS11(res));                 % matches the existing T-cascade form
