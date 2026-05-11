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
        ReceiverSensitivity
        SystemLoss
    end
    methods
        function obj = RxSite()
            obj.AntennaHeight = 1.0;
            obj.AntennaAngle = 0.0;
            obj.ReceiverSensitivity = -100.0;
            obj.SystemLoss = 0.0;
        end
    end
    %
    % `sigstrength(rx, tx, pm)` ships as a top-level builtin that
    % dispatches to the C runtime `matlab_prop_sigstrength` rather
    % than a MATLAB-side method.  Reason: the compiler doesn't yet
    % propagate class pinning from call-site args into method-body
    % parameters (an inter-procedural Sema piece), so `pm.Kind`
    % inside a hypothetical `sigstrength` method body would read
    % via the wrong dispatch path.  The C path reads the obj
    % properties directly via matlab_obj_get_f64 / _get_string.
end
