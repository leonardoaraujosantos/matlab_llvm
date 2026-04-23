% Inheritance + static methods + `< handle` acceptance.
%
% Shape inherits from Animal: the `describe` method is defined only on
% Animal, but calling it on a Shape instance dispatches through the
% inheritance chain. Shape adds its own `area` method.
%
% Utils defines a static method `square`, called via dot-on-class.

a = Animal(10);
disp(a.Kind);            % 10
disp(a.describe());      % 10

s = Shape(5);
disp(s.Kind);            % 5
disp(s.describe());      % 5 — inherited from Animal
disp(s.area());          % 25

disp(Utils.square(6));   % 36

classdef Animal < handle
    properties
        Kind
    end
    methods
        function obj = Animal(k)
            if nargin == 1
                obj.Kind = k;
            end
        end
        function r = describe(obj)
            r = obj.Kind;
        end
    end
end

classdef Shape < Animal
    methods
        function obj = Shape(k)
            if nargin == 1
                obj.Kind = k;
            end
        end
        function r = area(obj)
            r = obj.Kind * obj.Kind;
        end
    end
end

classdef Utils
    methods (Static)
        function r = square(n)
            r = n * n;
        end
    end
end
