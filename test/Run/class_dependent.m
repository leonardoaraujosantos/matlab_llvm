% Dependent properties.
%
% A Dependent property has no backing storage — its value is computed
% on read by the class's `get.Prop` method. The base Circle class
% stores Radius, and `Area` / `Diameter` are dependent properties
% calculated from it.

c = Circle(3);
disp(c.Radius);              % 3
disp(c.Area);                % 28.2743... (pi * 9)
disp(c.Diameter);            % 6

c2 = Circle(5);
disp(c2.Area);               % 78.5398... (pi * 25)

classdef Circle
    properties
        Radius
    end
    properties (Dependent)
        Area
        Diameter
    end
    methods
        function obj = Circle(r)
            if nargin == 1
                obj.Radius = r;
            end
        end
        function a = get.Area(obj)
            a = 3.14159265358979 * obj.Radius * obj.Radius;
        end
        function d = get.Diameter(obj)
            d = 2 * obj.Radius;
        end
    end
end
