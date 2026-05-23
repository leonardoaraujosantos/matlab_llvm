% N-port S→ABCD and S→H (even-N, block-partitioned via Y).
%
% Sanity check on the 2-port test_amp fixture: h = N/2 = 1, so the
% block matrices are 1×1 — equivalent to the scalar 2-port form
% (matches sparamS2abcd / sparamS2h closed-form values).

data = touchstoneRead("fixtures/rf/test_amp.s2p");

ab = sparamS2abcdN(data);
disp(ab.NumPorts);            % 2

hh = sparamS2hN(data);
disp(hh.NumPorts);            % 2
% Field is H_ij (i,j run 1..N).  For N=2: H11/H12/H21/H22.
disp(tsHij(hh, 1, 1));
disp(tsHij(hh, 2, 1));
disp(tsHij(hh, 2, 2));
