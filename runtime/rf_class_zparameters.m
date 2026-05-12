% RFZparameters — impedance-parameter container (2-port v1 skeleton).
%
% Parallels RFSparameters / RFYparameters.  Wraps the per-frequency
% Z-matrix; population from `sparamS2z` / `sparamS2zN` is a one-shot
% assignment from the runtime helper's struct return.

classdef RFZparameters < handle
    properties
        NumPorts
        Impedance
        Z11
        Z12
        Z21
        Z22
        Frequencies
    end
    methods
        function obj = RFZparameters(num_ports, z0)
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
