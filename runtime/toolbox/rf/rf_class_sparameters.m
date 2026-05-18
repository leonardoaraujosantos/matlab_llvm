% RFSparameters — RF Toolbox network parameter container (2-port).
%
% MathWorks-faithful semantics on a CamelCase class name (the
% lowercase `sparameters` constructor name is reserved for a future
% slice that adds a name-resolution alias).
%
% The 2-port subset of the surface defined in docs/comm_toolbox_roadmap.md
% §9.1.  Holds:
%   NumPorts      — number of ports (2 today)
%   Impedance     — reference impedance, Ω (default 50)
%   S11, S12,
%   S21, S22      — complex column vectors (length NumFreqs)
%   Frequencies   — real column vector, Hz
%
% v1 ships TWO constructor forms:
%   p = RFSparameters()             % skeleton, 2 ports, 50 Ω
%   p = RFSparameters(np, z0)       % skeleton, custom np / z0
%
% Touchstone loading goes through the function-form `touchstoneRead`:
%   data = touchstoneRead('amp.s2p');
%   p = RFSparameters(2, data.Z0);
%   p.S11 = data.S11;  p.S12 = data.S12;
%   p.S21 = data.S21;  p.S22 = data.S22;
%   p.Frequencies = data.Frequencies;
%
% The function-form analyses (gammaIn, gammaOut, vswr, powerGain,
% stabilityK, stabilityMu, s2tf, cascadeSparams2) accept the four S
% column vectors directly, so dropping in the struct fields straight
% out of touchstoneRead skips the classdef boilerplate when the user
% just wants numbers out.

classdef RFSparameters < handle
    properties
        NumPorts
        Impedance
        S11
        S12
        S21
        S22
        Frequencies
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
