% ANT-Tier-2 example — frequency sweep + RF Toolbox bridge.
%
% Build an S11(f) sweep for a 1 GHz dipole, then drop it into
% touchstoneWrite to get a .s1p file consumable by any RF tool.

L = 0.15;            % half-wave at 1 GHz
a = 0.0003;

freqs = [7e8; 8e8; 9e8; 1.0e9; 1.1e9; 1.2e9; 1.3e9];
sp = antennaWireSparameters(L, a, 21, freqs);
disp(sp.NumPorts);   % 1
disp(sp.Z0);         % 50

% Hand off to the RF Toolbox: writes an s1p Touchstone file with
% the swept S11(f) data, ready for Spectre / ngspice / ADS.
touchstoneWrite("dipole_1ghz.s1p", sp);
