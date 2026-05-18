% RFCktSeries — series connection element (single-component series
% impedance in the signal path).  Holds the series impedance (real +
% imaginary parts) and the operating frequency.  For a series Z in a
% z0-referenced 2-port, the S-parameters are:
%   s11 = s22 =  Z / (Z + 2·z0)
%   s12 = s21 = 2·z0 / (Z + 2·z0)
% These are computed on demand in `analyze(s, freqs)` via the function-
% form helpers.

classdef RFCktSeries < handle
    properties
        Z_re
        Z_im
        Frequency_Hz
        Label
    end
    methods
        function obj = RFCktSeries(z_re, z_im)
            if nargin >= 1
                obj.Z_re = z_re;
            else
                obj.Z_re = 0.0;
            end
            if nargin >= 2
                obj.Z_im = z_im;
            else
                obj.Z_im = 0.0;
            end
            obj.Frequency_Hz = 1.0e9;
            obj.Label = "series";
        end

        function s = analyze(obj, freqs)
            s = rfAnalyzeSeries(obj.Z_re, obj.Z_im, freqs, 50.0);
        end
    end
end
