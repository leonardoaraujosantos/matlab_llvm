% S → ABCD + S → H closed-form 2-port cross-conversions.
%
% test_amp.s2p at freq 1: s11=0.2, s12=0.1, s21=2.0, s22=0.05, z0=50:
%   ABCD: A=0.335, B=13.25, C=0.0028, D=0.26
%   H:    h11=50.96, h12=0.1923, h21=-3.846, h22=0.01077

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
S11 = tsS11(data); S12 = tsS12(data);
S21 = tsS21(data); S22 = tsS22(data);

abcd = sparamS2abcd(S11, S12, S21, S22, 50.0);
disp(tsAbcdA(abcd));
disp(tsAbcdB(abcd));
disp(tsAbcdC(abcd));
disp(tsAbcdD(abcd));

h = sparamS2h(S11, S12, S21, S22, 50.0);
disp(tsHij(h, 1, 1));
disp(tsHij(h, 1, 2));
disp(tsHij(h, 2, 1));
disp(tsHij(h, 2, 2));

% Round-trip: H → S should reproduce the input S.
h11 = tsHij(h, 1, 1); h12 = tsHij(h, 1, 2);
h21 = tsHij(h, 2, 1); h22 = tsHij(h, 2, 2);
s_back = sparamH2s(h11, h12, h21, h22, 50.0);
disp(tsS11(s_back));
