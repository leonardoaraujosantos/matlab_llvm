% Complex LU regression — the s2y / s2z / s2h / s2g / snp2smp / newref /
% cascadeSparamsN* helpers all go through the internal complex matrix
% inverse.  With the native complex LU swap, these should still produce
% the same numerical results as the 2N × 2N real-equivalent path.

data = touchstoneRead("fixtures/rf/test_amp.s2p");
S11 = tsS11(data); S12 = tsS12(data);
S21 = tsS21(data); S22 = tsS22(data);

% S → Y → S round-trip via the N-port path (uses complex LU twice).
y = sparamS2yN(data);
disp(tsYij(y, 1, 1));            % matches the pre-LU output

% newref round-trip 50 → 75 → 50 (uses complex LU twice per freq).
d75 = newref(data, 75.0);
d50 = newref(d75, 50.0);
disp(tsS11(d50));                 % must match the original (0.2, 0.3)

% snp2smpZ with non-z0 termination (uses complex LU once per freq).
data4 = touchstoneRead("fixtures/rf/diff_pair.s4p");
sub = snp2smpZ(data4, [1; 2], [100.0; 100.0], 2);
disp(tsS11(sub));                 % matches pre-LU output
