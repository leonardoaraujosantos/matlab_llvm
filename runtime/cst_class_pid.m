% PID controller model — auto-prepended when matlabc sees `pid(`
% or `pid =` in the user input. See cst_classdefs.m for the
% umbrella comment + the tf classdef.

classdef pid
    properties
        Kp
        Ki
        Kd
        Tf
    end
    methods
        function obj = pid(Kp, Ki, Kd, Tf)
            % `pid(Kp, Ki, Kd, Tf)` — PID controller.
            %   C(s) = Kp + Ki/s + Kd · s / (Tf · s + 1)
            %
            % Operator overloads in v1: pid + pid / pid - pid / -pid
            % combine coefficient-wise. The Laplace-form `tf(pid)`
            % conversion is a follow-on; once shipped, pid * pid and
            % scalar mixing fall out via the tf algebra.
            if nargin >= 1, obj.Kp = Kp; end
            if nargin >= 2, obj.Ki = Ki; end
            if nargin >= 3, obj.Kd = Kd; end
            if nargin >= 4, obj.Tf = Tf; end
        end

        function r = plus(a, b)
            % Coefficient-wise sum. Tf is averaged because it acts as
            % a denominator pole — a sum of two distinct Tfs would
            % require splitting into two parallel D-paths and is not
            % representable in a single pid struct.
            r = pid(a.Kp + b.Kp, a.Ki + b.Ki, a.Kd + b.Kd, ...
                    (a.Tf + b.Tf) / 2);
        end

        function r = minus(a, b)
            r = pid(a.Kp - b.Kp, a.Ki - b.Ki, a.Kd - b.Kd, ...
                    (a.Tf + b.Tf) / 2);
        end

        function r = uminus(a)
            r = pid(-a.Kp, -a.Ki, -a.Kd, a.Tf);
        end
    end
end
