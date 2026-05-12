% MathWorks-faithful lowercase aliases.  These should produce
% byte-identical results to the long-form names.

data = sparameters("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
disp(data.NumPorts);
S11 = tsS11(data); S12 = tsS12(data);
S21 = tsS21(data); S22 = tsS22(data);

% s2y, s2z, s2h, s2abcd, s2g, s2t and inverses via lowercase names.
y = s2y(S11, S12, S21, S22, 50.0);
disp(tsYij(y, 1, 1));    % matches sparamS2y(...)

z = s2z(S11, S12, S21, S22, 50.0);
disp(tsZij(z, 1, 1));

h = s2h(S11, S12, S21, S22, 50.0);
disp(tsHij(h, 1, 1));

g = s2g(S11, S12, S21, S22, 50.0);
disp(tsGij(g, 1, 1));

abcd = s2abcd(S11, S12, S21, S22, 50.0);
disp(tsAbcdA(abcd));

t = s2t(S11, S12, S21, S22);
disp(tsTij(t, 1, 1));

% rfbudget alias.
g_chain = [20.0; -8.0; 25.0];
n = [1.5; 8.0; 5.0];
ip3 = [15.0; 10.0; 8.0];
b = rfbudget(g_chain, n, ip3, -30.0, 1.0e6);
disp(b.Gain_dB);    % 37
disp(b.NF_dB);      % 2.0468
