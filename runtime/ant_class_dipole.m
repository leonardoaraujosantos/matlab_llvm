% AntDipole — Antenna Toolbox catalog classdef (geometry-only).
%
% MATLAB API: `dipole('Length', L, 'Width', W, 'FeedOffset', 0)`.
% Flat name used here pending package-syntax support — port from
% MathWorks examples by rewriting `dipole(...)` -> `AntDipole(...)`.
%
% v1 ships geometry storage only (Length / Width / FeedOffset).  The
% `pattern(ant, freq)`, `impedance(ant, freq)`, and `sparameters(ant,
% f)` methods require the ANT-Tier-2 wire-MoM solver — they ship in
% a follow-on slice once Pocklington Z-matrix + Z·I = V solve land.

classdef AntDipole < handle
    properties
        % Total length of the dipole (m).  At resonance this is ~λ/2.
        Length
        % Strip width (m).  For thin-wire approximations the Width is
        % converted to an equivalent radius (radius = Width / 4).
        Width
        % Feed offset from center (m).  0 = center-fed (canonical
        % half-wave dipole); ≠0 shifts the impedance.
        FeedOffset
    end
    methods
        function obj = AntDipole(length_m, width_m, feed_offset)
            if nargin >= 1
                obj.Length = length_m;
            else
                obj.Length = 2.0;
            end
            if nargin >= 2
                obj.Width = width_m;
            else
                obj.Width = 0.05;
            end
            if nargin >= 3
                obj.FeedOffset = feed_offset;
            else
                obj.FeedOffset = 0.0;
            end
        end
    end
end
