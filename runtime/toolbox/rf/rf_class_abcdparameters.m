% RFAbcdparameters — ABCD (transmission) parameter container (2-port v1).
%
% Property names match the standard ABCD convention (A dimensionless,
% B in Ω, C in S, D dimensionless).  Population from sparamS2abcd is
% a direct per-field assignment from the runtime helper's return.

classdef RFAbcdparameters < handle
    properties
        NumPorts
        Impedance
        A
        B
        C
        D
        Frequencies
    end
    methods
        function obj = RFAbcdparameters(num_ports, z0)
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
