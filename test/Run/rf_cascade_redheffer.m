% Full Redheffer-star N-port cascade test.  Use the 2-port test_amp
% fixture cascaded with itself.  For 2-port (N=2, k=1), the result
% is also a 2-port.  Compare against the existing cascadeSparams2
% closed-form scalar formula.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
S11 = tsS11(data); S12 = tsS12(data);
S21 = tsS21(data); S22 = tsS22(data);

% Existing 2-port T-parameter cascade.
ref = cascadeSparams2(S11, S12, S21, S22, S11, S12, S21, S22);
disp(tsS11(ref));

% Full Redheffer star cascade on the same data.
res = cascadeSparamsNFull(data, data);
disp(res.NumPorts);              % 2
disp(tsS11(res));                 % Should match the T-parameter form above
