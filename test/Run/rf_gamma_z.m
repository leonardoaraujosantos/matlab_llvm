% gamma2z + z2gamma Smith-chart helpers.  Round-trip Γ → Z → Γ should
% be exact for any reflection coefficient.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
S11 = tsS11(data);
% Γ = 0.2 → Z = 50 · 1.2/0.8 = 75 Ω.   Γ = 0.3 → Z = 50 · 1.3/0.7 ≈ 92.86 Ω.
z = gamma2z(S11, 50.0);
disp(z);
g = z2gamma(z, 50.0);
disp(g);    % Should match S11 (0.2; 0.3)
