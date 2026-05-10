% Frequency-response data — auto-prepended when matlabc sees `frd(`
% or `frd =` in the user input. See cst_classdefs.m for the
% umbrella comment + the tf classdef.

classdef frd
    properties
        ResponseData
        Frequency
    end
    methods
        function obj = frd(response, freqs)
            % `frd(response, freqs)` — frequency-response data: a
            % complex H(jω) sampled on a frequency grid. v1 stores
            % the data unmodified; downstream consumers (bode plots,
            % nyquist plots) can read the two properties.
            if nargin >= 1, obj.ResponseData = response; end
            if nargin >= 2, obj.Frequency = freqs; end
        end
    end
end
