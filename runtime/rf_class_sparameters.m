% RFSparameters — RF Toolbox network parameter container.
%
% MATLAB API: `sparameters(filename)` reads a Touchstone .sNp file
% and exposes `Parameters` (P×P×F complex cube), `Frequencies`
% (F×1 column), `NumPorts`, and `Impedance` (reference, default
% 50 Ω).
%
% v1 (this slice) ships only the catalog skeleton: NumPorts +
% Impedance scalar properties, manual-construction form
% `RFSparameters(num_ports, z0)`.  The actual Touchstone parser
% + 3-D complex Parameters cube + Frequencies vector land in a
% follow-on slice (matlab_rf_read_touchstone runtime entry +
% matrix-property storage for classdef instances).

classdef RFSparameters < handle
    properties
        % Number of ports the network has (typically 2 for a
        % standard two-port amplifier / filter / cable).
        NumPorts
        % Reference impedance in ohms.  Default 50 Ω (RF standard).
        Impedance
    end
    methods
        function obj = RFSparameters(num_ports, z0)
            if nargin >= 1
                obj.NumPorts = num_ports;
            else
                obj.NumPorts = 2;
            end
            if nargin >= 2
                obj.Impedance = z0;
            else
                obj.Impedance = 50.0;
            end
        end
    end
end
