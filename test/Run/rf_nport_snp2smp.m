% N-port → m-port extraction.  Pull a 2-port from the 4-port fixture
% by keeping ports [1, 2] (which become S11/S12/S21/S22 of the
% sub-network).  With matched terminations at the dropped ports
% (1.5 Ω = z0), the sub-block IS the original S(:, :, [1 2]).

data = touchstoneRead("fixtures/rf/diff_pair.s4p");
sub = snp2smp(data, [1; 2], 2);
disp(sub.NumPorts);            % 2
disp(sub.Z0);                  % 50
disp(tsS11(sub));              % 0.1 + 0i; 0.15 + 0i (same as full data S11)
disp(tsS21(sub));              % 0.9 + 0i; 0.85 + 0i (same as full data S21)
