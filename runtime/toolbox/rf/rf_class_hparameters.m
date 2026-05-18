% RFHparameters — hybrid h-parameter container (2-port v1).
%
% h11 has units of Ω, h22 of S, h12/h21 are dimensionless.  Mirrors
% the RFSparameters property surface.  Population from sparamS2h is
% a simple per-field assignment from the runtime helper's return.

classdef RFHparameters < handle
    properties
        NumPorts
        Impedance
        H11
        H12
        H21
        H22
        Frequencies
    end
    methods
        function obj = RFHparameters(num_ports, z0)
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
