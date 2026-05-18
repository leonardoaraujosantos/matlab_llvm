% RFYparameters — admittance-parameter container (2-port v1 skeleton).
%
% MathWorks-faithful semantics on a CamelCase class name (the
% lowercase `yparameters` constructor name will be aliased once
% name-resolution aliasing lands).
%
% Holds the per-frequency Y-matrix together with NumPorts /
% Impedance / Frequencies, paralleling RFSparameters.  The function-
% form `sparamS2y` / `sparamS2yN` produce a matlab_struct with the
% Y_ij fields directly; this classdef is the property-holder wrapper
% for users who prefer the MathWorks object syntax.

classdef RFYparameters < handle
    properties
        NumPorts
        Impedance
        Y11
        Y12
        Y21
        Y22
        Frequencies
    end
    methods
        function obj = RFYparameters(num_ports, z0)
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
