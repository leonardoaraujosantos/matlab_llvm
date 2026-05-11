% TxSite — transmitter site classdef.
%
% MATLAB API: `txsite('Name', 'X', 'Latitude', 42.3, 'Longitude',
% -71.35, 'AntennaHeight', 10, 'TransmitterFrequency', 2.5e9,
% 'TransmitterPower', 5)`.  Flat name `TxSite` pending package-syntax
% support; the kwarg sugar (see Lowering.cpp constructor dispatch)
% makes the verbatim MathWorks call shape work.
%
% Properties match the MathWorks API:
%   Name                  string  — site label
%   Latitude              double  — degrees, WGS84
%   Longitude             double  — degrees, WGS84
%   AntennaHeight         double  — m above ground (default 10)
%   TransmitterFrequency  double  — Hz
%   TransmitterPower      double  — W (note: MathWorks uses W, not dBm)
%   AntennaAngle          double  — boresight azimuth, degrees (default 0)
%   SystemLoss            double  — feeder + connector loss, dB (default 0)

classdef TxSite < handle
    properties
        Name string
        Latitude
        Longitude
        AntennaHeight
        TransmitterFrequency
        TransmitterPower
        AntennaAngle
        % `AntennaGain` is the TX antenna gain in dBi.  v1 ships a
        % scalar (defaults to 0 dBi, isotropic).  Once the wire-MoM
        % solver lands (ANT-Tier-2) this will be derived from a
        % directional `Antenna` property + pattern lookup.  Users
        % can still set it directly: `tx.AntennaGain = 8.5;`.
        AntennaGain
        SystemLoss
    end
    methods
        function obj = TxSite()
            % No-arg ctor.  All properties default to 0 / unset; the
            % kwarg sugar at the call site populates them from the
            % `'Key', value` pairs in the user's invocation.
            obj.AntennaHeight = 10.0;
            obj.AntennaAngle = 0.0;
            obj.AntennaGain = 0.0;
            obj.SystemLoss = 0.0;
        end

        function d = link(tx, rx)
            % Great-circle distance between sites, in meters.
            d = haversine(tx.Latitude, tx.Longitude, ...
                          rx.Latitude, rx.Longitude);
        end

        function b = bearing_(tx, rx)
            % Forward bearing tx -> rx, in compass degrees [0, 360).
            % Suffixed underscore to avoid clashing with the global
            % `bearing` function name when called function-style.
            b = bearing(tx.Latitude, tx.Longitude, ...
                        rx.Latitude, rx.Longitude);
        end

        function clear = los(tx, rx)
            % Line-of-sight check between sites (k=4/3 Earth model).
            % Returns 1.0 (clear) or 0.0 (obstructed by Earth bulge).
            clear = propLosSites(tx.Latitude, tx.Longitude, tx.AntennaHeight, ...
                                  rx.Latitude, rx.Longitude, rx.AntennaHeight);
        end

        function coverage(tx, pm)
            % Default 30×30 coverage grid centred on the txsite,
            % spanning a 10 km × 10 km square.  Prints summary stats
            % (max / median / min RX power in dBm) rather than
            % returning the matrix — matches MathWorks `coverage(tx,
            % ...)` side-effect-only signature.
            %
            % v1 uses free-space path loss regardless of pm.Kind —
            % model-selector dispatch lands in a follow-on slice
            % (needs the runtime to consume an integer model code
            % instead of a kind string per cell).
            half_lat = 0.045;            % ~5 km in lat-degrees
            half_lon = half_lat ./ cos(tx.Latitude .* (3.14159265 ./ 180));
            n_cells = 30;
            heightmap = zeros(0, 1);
            grid = coverageGrid(tx.Latitude, tx.Longitude, tx.AntennaHeight, ...
                tx.TransmitterFrequency, tx.TransmitterPower, ...
                0.0, 0.0, heightmap, ...
                tx.Latitude - half_lat, tx.Latitude + half_lat, ...
                tx.Longitude - half_lon, tx.Longitude + half_lon, ...
                n_cells, n_cells, ...
                1.5, 0.0, ...
                0, 80.0, 99.0, 99.0);
            disp(max(grid(:)));
            disp(median(grid(:)));
            disp(min(grid(:)));
        end
    end
end
