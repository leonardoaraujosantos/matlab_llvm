% newref — re-reference S to a new impedance.
% Sanity check: round-trip 50 → 75 → 50 should recover the original S.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
S11_orig = tsS11(data);
disp(S11_orig);                  % [0.2; 0.3]

% Re-reference 50 → 75 Ω.
data75 = newref(data, 75.0);
disp(data75.Z0);                  % 75
disp(tsS11(data75));              % different (S11 changes)

% Re-reference back: 75 → 50 should recover the original.
data50 = newref(data75, 50.0);
disp(data50.Z0);                  % 50
disp(tsS11(data50));              % should match S11_orig
