% Touchstone v2 reader smoke test.
%
% Reads v2_amp.ts which uses the v2 bracket-keyword surface:
%   [Version] 2.0
%   [Number of Ports] 2
%   [Two-Port Order] 12_21      (row-major s2p layout, no transpose)
%   [Reference] 50 50
%   [Number of Frequencies] 2
%   [Network Data]
%   ... data ...
%   [End]
%
% With row-major 12_21 order, s11/s12/s21/s22 are decoded in their
% NATURAL position (no s2p historical transpose).  Same data values
% as test_amp.s2p but laid out differently — the parser should pick
% them up via the [Two-Port Order] hint.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/v2_amp.ts");
disp(data.NumPorts);     % 2 (from [Number of Ports])
disp(data.Z0);            % 50 (from [Reference])
disp(tsS11(data));        % 0.2 / 0.3 — same as test_amp.s2p
disp(tsS21(data));        % 2.0 / 1.8 — same as test_amp.s2p
