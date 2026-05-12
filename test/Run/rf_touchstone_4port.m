% Multi-port Touchstone reader smoke test (s4p).
%
% Loads diff_pair.s4p (2 frequencies, 4 ports, symmetric diff-pair
% structure: S11=S22=S33=S44, S12=S21=S34=S43 (pair coupling), small
% S13/S14/S23/S24 (mode coupling)).  Validates the port-count
% detection, the row-major S layout decoding, and the tsSij()
% generic getter.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/diff_pair.s4p");
disp(data.NumPorts);     % 4
disp(data.Z0);            % 50

% Diagonal: small reflection (matched port).
S11 = tsSij(data, 1, 1);
disp(S11);                % 2x1 column: 0.1 + 0i, 0.15 + 0i

% Strong port-1 to port-2 coupling.
S21 = tsSij(data, 2, 1);
disp(S21);                % 2x1 column: 0.9 + 0i, 0.85 + 0i

% Weak port-1 to port-3 coupling.
S31 = tsSij(data, 3, 1);
disp(S31);                % 2x1 column: 0.05 + 0i, 0.06 + 0i
