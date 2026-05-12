% RFCktShunt — shunt-to-ground element.  Holds the shunt admittance
% (real + imaginary parts) of a single-component shunt block in a
% z0-referenced 2-port:
%   s11 = s22 = −Y·z0 / (Y·z0 + 2)
%   s12 = s21 = 2 / (Y·z0 + 2)
% These are computed on demand in `analyze(s, freqs)` via the function-
% form helpers.

classdef RFCktShunt < handle
    properties
        Y_re
        Y_im
        Frequency_Hz
        Label
    end
    methods
        function obj = RFCktShunt(y_re, y_im)
            if nargin >= 1
                obj.Y_re = y_re;
            else
                obj.Y_re = 0.0;
            end
            if nargin >= 2
                obj.Y_im = y_im;
            else
                obj.Y_im = 0.0;
            end
            obj.Frequency_Hz = 1.0e9;
            obj.Label = "shunt";
        end

        function s = analyze(obj, freqs)
            s = rfAnalyzeShunt(obj.Y_re, obj.Y_im, freqs, 50.0);
        end
    end
end
