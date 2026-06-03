% Class instances carry a concrete object<Class> type (#40), not Any.
% A constructor call, an operator overload on the instance, and a copy
% should all infer `object<Box>` so the monomorphizer can bucket them.
classdef Box
    properties
        v
    end
    methods
        function obj = Box(x)
            obj.v = x;
        end
        function r = plus(a, b)
            r = Box(a.v + b.v);
        end
    end
end

function out = use()
    g = Box(3);        % constructor      -> object<Box>
    h = Box(4);        % constructor      -> object<Box>
    s = g + h;         % operator overload -> object<Box>
    t = s;             % copy              -> object<Box>
    out = t.v;
end
