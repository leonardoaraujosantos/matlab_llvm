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
            % complex H(jω) sampled on a frequency grid.
            %
            % Operator overloads assume two operands share the same
            % Frequency vector — element-wise on ResponseData mirrors
            % the frequency-domain semantics of parallel sum (`+`)
            % and series cascade (`*` is `.*` because two H(jω)
            % cascades multiply pointwise). True grid-mismatch
            % interpolation is a follow-on.
            if nargin >= 1, obj.ResponseData = response; end
            if nargin >= 2, obj.Frequency = freqs; end
        end

        function r = plus(a, b)
            r = frd(a.ResponseData + b.ResponseData, a.Frequency);
        end

        function r = minus(a, b)
            r = frd(a.ResponseData - b.ResponseData, a.Frequency);
        end

        function r = mtimes(a, b)
            % H_ab(jω) = H_a(jω) · H_b(jω) — pointwise, not matrix
            % multiply. Series cascade in the frequency domain.
            r = frd(a.ResponseData .* b.ResponseData, a.Frequency);
        end

        function r = uminus(a)
            r = frd(-a.ResponseData, a.Frequency);
        end
    end
end
