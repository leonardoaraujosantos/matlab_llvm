% Enumeration classes. Each member of the enumeration is a class-
% level constant; ClassName.Member evaluates to the member's 0-based
% index as an f64, so equality comparisons work with plain numeric
% semantics.

c = Color.Green;
disp(c);                 % 1

disp(Color.Red);         % 0
disp(Color.Green);       % 1
disp(Color.Blue);        % 2

if c == Color.Green
    disp(100);           % 100
else
    disp(200);
end

if Color.Red == Color.Blue
    disp(300);
else
    disp(400);           % 400
end

classdef Color
    enumeration
        Red
        Green
        Blue
    end
end
