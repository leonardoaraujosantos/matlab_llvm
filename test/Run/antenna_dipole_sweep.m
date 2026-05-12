% ANT-Tier-2 — frequency sweep + RF-Toolbox bridge.  Build an
% antenna S-parameter struct shaped like the RF Toolbox
% `sparameters` return: S11 / Frequencies / Z0 / NumPorts.

L = 0.15;            % half-wave at 1 GHz
a = 0.0003;

freqs = [7e8; 8.5e8; 1.0e9; 1.15e9; 1.3e9];
sp = antennaWireSparameters(L, a, 21, freqs);
disp(sp.NumPorts);   % 1
disp(sp.Z0);         % 50
