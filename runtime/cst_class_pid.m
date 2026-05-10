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
            % v1 stores the four gains; the Laplace-form expansion +
            % `tf(C)` conversion are follow-ons.
            if nargin >= 1, obj.Kp = Kp; end
            if nargin >= 2, obj.Ki = Ki; end
            if nargin >= 3, obj.Kd = Kd; end
            if nargin >= 4, obj.Tf = Tf; end
        end
    end
end
