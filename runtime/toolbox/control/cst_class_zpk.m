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
            % zpk * zpk: concatenate roots and multiply gains.
            % zpk / zpk: invert the divisor (swap Z/P, take 1/K)
            %            and compose. -zpk: negate gain.
            % zpk + zpk requires polynomial expansion of both root
            % vectors to a common-denominator tf and is a follow-on.
            if nargin >= 1, obj.Z = z; end
            if nargin >= 2, obj.P = p; end
            if nargin >= 3, obj.K = k; end
        end

        function r = mtimes(a, b)
            % Series cascade: roots concatenate, gains multiply.
            new_z = vertcat(a.Z, b.Z);
            new_p = vertcat(a.P, b.P);
            new_k = a.K * b.K;
            r = zpk(new_z, new_p, new_k);
        end

        function r = mrdivide(a, b)
            % Right-divide: a / b ≡ a * inv(b). inv(b) swaps zeros
            % and poles and inverts the gain.
            new_z = vertcat(a.Z, b.P);
            new_p = vertcat(a.P, b.Z);
            new_k = a.K / b.K;
            r = zpk(new_z, new_p, new_k);
        end

        function r = uminus(a)
            r = zpk(a.Z, a.P, -a.K);
        end
    end
end
