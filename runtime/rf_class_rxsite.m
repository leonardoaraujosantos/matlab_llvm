% RxSite — receiver site classdef.
%
% MATLAB API: `rxsite('Name', 'Y', 'Latitude', 42.35, 'Longitude',
% -71.10, 'AntennaHeight', 1.5, 'ReceiverSensitivity', -100)`.
%
% Properties:
%   Name                 string  — site label
%   Latitude             double  — degrees, WGS84
%   Longitude            double  — degrees, WGS84
%   AntennaHeight        double  — m above ground (default 1)
%   AntennaAngle         double  — boresight azimuth, degrees (default 0)
%   ReceiverSensitivity  double  — dBm noise floor (default -100)
%   SystemLoss           double  — feeder + connector loss, dB (default 0)

classdef RxSite < handle
    properties
        Name string
        Latitude
        Longitude
        AntennaHeight
        AntennaAngle
        AntennaGain
        ReceiverSensitivity
        SystemLoss
    end
    methods
        function obj = RxSite()
            obj.AntennaHeight = 1.0;
            obj.AntennaAngle = 0.0;
            obj.AntennaGain = 0.0;
            obj.ReceiverSensitivity = -100.0;
            obj.SystemLoss = 0.0;
        end

        function ss = sigstrength(rx, tx, pm)
            % Received signal strength in dBm.  Inter-procedural
            % class pinning in the Resolver (see lib/Sema/Resolver.cpp
            % propagation pass) carries the call-site `pm` argument's
            % PropagationModel pin into this method body, so the
            % `pathloss(pm, rx, tx)` dispatch routes correctly.
            %
            %   ss = 10*log10(P_W * 1000) + TX_gain - pathloss
            %        + RX_gain - tx.SystemLoss - rx.SystemLoss
            %
            % Antenna gains default to 0 dBi (isotropic).  When
            % tx.Antenna is wired to a catalog antenna with pattern
            % data, the lookup routes through antennaGain (lands
            % alongside ANT-Tier-2 wire-MoM).
            pl = pathloss(pm, rx, tx);
            ss = 10.0 .* log10(tx.TransmitterPower .* 1000.0) ...
                  + tx.AntennaGain - pl + rx.AntennaGain ...
                  - tx.SystemLoss - rx.SystemLoss;
        end
    end
end
