% Smoke test for the new RF Toolbox classdef skeletons.
%
% Each of RFYparameters / RFZparameters / RFHparameters / RFGparameters
% / RFAbcdparameters / RFTparameters / RFCktAmplifier / RFCktMixer /
% RFCktPassive instantiates cleanly with sensible defaults and accepts
% the constructor args.

y = RFYparameters();
disp(y.NumPorts);             % 2
disp(y.Impedance);            % 50

z = RFZparameters(3, 75.0);
disp(z.NumPorts);             % 3
disp(z.Impedance);            % 75

h = RFHparameters(2, 50.0);
disp(h.NumPorts);             % 2

g = RFGparameters();
disp(g.Impedance);            % 50 (default)

ab = RFAbcdparameters();
disp(ab.NumPorts);            % 2

t = RFTparameters();
disp(t.NumPorts);             % 2

a = RFCktAmplifier(2.5, 25.0, 30.0);
disp(a.NF_dB);                % 2.5
disp(a.Gain_dB);              % 25
disp(a.IP3_dBm);              % 30
disp(a.Frequency_Hz);         % 1e9 (default)

m = RFCktMixer();
disp(m.NF_dB);                % 8 (default)
disp(m.ConversionGain_dB);    % -8
disp(m.LO_Frequency_Hz);      % 1e9

p = RFCktPassive(1.5);
disp(p.Loss_dB);              % 1.5
