% RFTparameters — chain (transmission-scattering) T-parameter container.
%
% T-parameters are the cascading-friendly transform of S-parameters:
% for a 2-port,  T = [[ -det(S)/s21, s11/s21 ], [ -s22/s21, 1/s21 ]].
% Cascading two 2-ports via T-parameters is just T_AB = T_A · T_B; the
% closed-form chain shipped in `cascadeSparams2` already operates in
% this representation internally.

classdef RFTparameters < handle
    properties
        NumPorts
        Impedance
        T11
        T12
        T21
        T22
        Frequencies
    end
    methods
        function obj = RFTparameters(num_ports, z0)
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
