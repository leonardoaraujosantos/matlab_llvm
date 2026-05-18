% RFGparameters — inverse-hybrid g-parameter container (2-port v1).
%
% Dual of the h-parameters: g11 has units of S, g22 of Ω, g12/g21
% are dimensionless.  Skeleton matches RFHparameters.

classdef RFGparameters < handle
    properties
        NumPorts
        Impedance
        G11
        G12
        G21
        G22
        Frequencies
    end
    methods
        function obj = RFGparameters(num_ports, z0)
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
