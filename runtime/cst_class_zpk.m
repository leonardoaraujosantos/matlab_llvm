% Zero-pole-gain model — auto-prepended when matlabc sees `zpk(`
% or `zpk =` in the user input. See cst_classdefs.m for the
% umbrella comment + the tf classdef.

classdef zpk
    properties
        Z
        P
        K
    end
    methods
        function obj = zpk(z, p, k)
            % `zpk(z, p, k)` — zero-pole-gain model.
            %   G(s) = K · prod(s - z_i) / prod(s - p_j)
            % v1 stores the three properties; operator overloads
            % (zpk * zpk → concatenate roots and multiply gains;
            % zpk + zpk → conv-and-cross to a tf and back) are
            % follow-ons.
            if nargin >= 1, obj.Z = z; end
            if nargin >= 2, obj.P = p; end
            if nargin >= 3, obj.K = k; end
        end
    end
end
