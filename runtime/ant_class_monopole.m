% AntMonopole — Antenna Toolbox catalog classdef (geometry-only).
%
% MATLAB API: `monopole('Height', H, 'Width', W, 'GroundPlaneLength',
% G, 'GroundPlaneWidth', G)`.
%
% Quarter-wave monopole over a (typically infinite) ground plane.
% Conceptually a half-dipole with image-theory mirror; impedance is
% half of the equivalent dipole.  v1 ships geometry only; pattern /
% impedance need ANT-Tier-2 MoM.

classdef AntMonopole < handle
    properties
        % Monopole height above the ground plane (m).  ~λ/4 at
        % resonance.
        Height
        % Strip width (m); see AntDipole comments.
        Width
        % Square ground plane edge length (m).  In real catalog
        % designs a 1-2 λ ground plane gives the canonical 36.5 Ω
        % input impedance; smaller planes shift the impedance higher.
        GroundPlaneLength
    end
    methods
        function obj = AntMonopole(height_m, width_m, gp_size_m)
            if nargin >= 1
                obj.Height = height_m;
            else
                obj.Height = 1.0;
            end
            if nargin >= 2
                obj.Width = width_m;
            else
                obj.Width = 0.05;
            end
            if nargin >= 3
                obj.GroundPlaneLength = gp_size_m;
            else
                obj.GroundPlaneLength = 2.0;
            end
        end
    end
end
