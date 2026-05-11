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
        SystemLoss
    end
    methods
        function obj = TxSite()
            % No-arg ctor.  All properties default to 0 / unset; the
            % kwarg sugar at the call site populates them from the
            % `'Key', value` pairs in the user's invocation.
            obj.AntennaHeight = 10.0;
            obj.AntennaAngle = 0.0;
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
    end
end
