% Operator overloading on a user class. A class that defines methods
% named after the MATLAB operators (plus, minus, mtimes, eq) has
% those dispatches called in place of the built-in arithmetic /
% comparison path when at least one operand is pinned to the class.

a = Vec2(1, 2);
b = Vec2(3, 4);

c = a + b;               % dispatches to Vec2__plus
disp(c.x);               % 4
disp(c.y);               % 6

d = a - b;               % Vec2__minus
disp(d.x);               % -2
disp(d.y);               % -2

s = a * 3;               % Vec2__mtimes
disp(s.x);               % 3
disp(s.y);               % 6

disp(a == b);            % 0 (Vec2__eq)
disp(a == a);            % 1

classdef Vec2
    properties
        x
        y
    end
    methods
        function obj = Vec2(xv, yv)
            if nargin == 2
                obj.x = xv;
                obj.y = yv;
            end
        end
        function r = plus(a, b)
            r = Vec2(a.x + b.x, a.y + b.y);
        end
        function r = minus(a, b)
            r = Vec2(a.x - b.x, a.y - b.y);
        end
        function r = mtimes(a, k)
            r = Vec2(a.x * k, a.y * k);
        end
        function r = eq(a, b)
            r = 0;
            if a.x == b.x
                if a.y == b.y
                    r = 1;
                end
            end
        end
    end
end
