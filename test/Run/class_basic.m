% Minimum classdef support.
%
% Construct a BasicClass instance, set the Value property via the
% constructor, read it back via property access, and call methods via
% dot-notation (obj.method(args)). The constructor accepts one argument
% and populates the Value property when nargin == 1.

a = BasicClass(3.14);
disp(a.Value);               % 3.14
disp(a.multiplyBy(2));       % 6.28

b = BasicClass(1.5);
b.Value = 2.5;
disp(b.Value);               % 2.5
disp(b.multiplyBy(4));       % 10

c = BasicClass(7);
disp(c.Value);               % 7
disp(c.squared());           % 49

classdef BasicClass
    properties
        Value
    end
    methods
        function obj = BasicClass(val)
            if nargin == 1
                obj.Value = val;
            end
        end
        function r = multiplyBy(obj, n)
            r = obj.Value * n;
        end
        function r = squared(obj)
            r = obj.Value * obj.Value;
        end
    end
end
