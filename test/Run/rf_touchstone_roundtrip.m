% Round-trip Touchstone reader → writer → reader.  Load test_amp.s2p,
% write it back, re-read.  Final values should be byte-identical
% modulo MA-format printing precision.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
ok = touchstoneWrite("/tmp/_rt.s2p", data);
disp(ok);                       % 1 (success)

d2 = touchstoneRead("/tmp/_rt.s2p");
disp(d2.NumPorts);              % 2
disp(d2.Z0);                    % 50
disp(tsS11(d2));                % 0.2; 0.3 (unchanged)
disp(tsS21(d2));                % 2.0; 1.8
