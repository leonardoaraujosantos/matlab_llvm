% Smoke test for the RFCktCascade / Parallel / Series / Shunt + RFRational
% classdef skeletons.

ch = RFCktCascade();
disp(ch.Frequency_Hz);   % 1e9 default
disp(ch.Label);          % "cascade"

p = RFCktParallel(2);
disp(p.NumBlocks);       % 2

s = RFCktSeries(50.0, 25.0);
disp(s.Z_re);            % 50
disp(s.Z_im);            % 25

sh = RFCktShunt(0.02, 0.0);
disp(sh.Y_re);           % 0.02

mdl = RFRational();
disp(mdl.D);             % 0
disp(mdl.Order);         % 0
disp(mdl.Delay);         % 0
