% S ↔ G and S ↔ T round-trip tests.
%
% S → G → S and S → T → S should reproduce the original S to numerical
% precision for any well-conditioned 2-port.

data = touchstoneRead("fixtures/rf/test_amp.s2p");
S11 = tsS11(data); S12 = tsS12(data);
S21 = tsS21(data); S22 = tsS22(data);

% S → G → S.
g = sparamS2g(S11, S12, S21, S22, 50.0);
g11 = tsGij(g, 1, 1); g12 = tsGij(g, 1, 2);
g21 = tsGij(g, 2, 1); g22 = tsGij(g, 2, 2);
sg_back = sparamG2s(g11, g12, g21, g22, 50.0);
disp(tsS11(sg_back));   % should equal S11

% S → T → S.
t = sparamS2t(S11, S12, S21, S22);
t11 = tsTij(t, 1, 1); t12 = tsTij(t, 1, 2);
t21 = tsTij(t, 2, 1); t22 = tsTij(t, 2, 2);
st_back = sparamT2s(t11, t12, t21, t22);
disp(tsS11(st_back));   % should equal S11
disp(tsS21(st_back));   % should equal S21
