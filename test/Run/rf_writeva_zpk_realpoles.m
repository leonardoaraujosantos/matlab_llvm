% Tier-2 Verilog-A export — writeVerilogAZPK on a real-pole filter.
%
% Two real poles at s = -1e9 and s = -3e9, no zeros, k = 1.
% Expected emitted denominator (ascending power of s):
%   (s + 1e9)(s + 3e9) = s^2 + 4e9 s + 3e18
%   ascending: {3e18, 4e9, 1}

zeros_col = zeros(0, 1);    % no zeros
poles_col = [-1.0e9; -3.0e9];
k = 1.0;
ok = writeVerilogAZPK(zeros_col, poles_col, k, "/tmp/_rf_writeva_zpk_real.va");
disp(ok);              % 1
