% PropagationModel — propagation-model classdef.
%
% MATLAB API: `propagationModel("longley-rice")`,
% `propagationModel("freespace")`, `propagationModel("rain")` etc.
% Flat name `PropagationModel` pending package-syntax support.
%
% Supported kinds (case- + hyphen-insensitive):
%   freespace / fspl              Friis free-space
%   longley-rice / itm            ITM (flat terrain)
%   rain                          ITU-R P.838 rain attenuation
%   gas / atmospheric             ITU-R P.676 oxygen + water vapour
%   fog / cloud                   ITU-R P.840 fog attenuation
%   close-in / ci                 Close-in macrocell
%   hata / cost231 / egli /
%     ecc33 / sui / ericsson9999  empirical macrocell models
%
% `pathloss(pm, rx, tx)` computes the path loss in dB between the
% supplied rxsite and txsite, dispatching on `pm.Kind`.

classdef PropagationModel < handle
    properties
        Kind string
    end
    methods
        function obj = PropagationModel(kind)
            if nargin >= 1
                obj.Kind = kind;
            else
                obj.Kind = "freespace";
            end
        end

        function pl = pathloss(pm, rx, tx)
            % Dispatch to the runtime model selector with the txsite
            % + rxsite geometry + frequency.  The `pm.Kind` string is
            % passed through verbatim — the runtime helper does case-
            % and hyphen-insensitive matching against the supported
            % kind list.
            pl = propPathlossDispatch(pm.Kind, ...
                tx.Latitude, tx.Longitude, tx.AntennaHeight, ...
                rx.Latitude, rx.Longitude, rx.AntennaHeight, ...
                tx.TransmitterFrequency);
        end
    end
end
