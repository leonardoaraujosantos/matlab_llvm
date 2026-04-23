% traffic_action(color, is_emergency):
%
% Integer-coded action for a traffic light. Exercises if / elseif /
% else AND switch / case / otherwise in one function.
%
%   is_emergency != 0 overrides everything     -> 9 (go)
%   color out of range (< 1 or > 3)            -> 0 (unknown)
%   1 = red    -> 1 (stop)
%   2 = yellow -> 2 (slow)
%   3 = green  -> 3 (go)

disp('red, normal:');
disp(traffic_action(1, 0));
disp('yellow, normal:');
disp(traffic_action(2, 0));
disp('green, normal:');
disp(traffic_action(3, 0));
disp('red, emergency:');
disp(traffic_action(1, 1));
disp('bogus color:');
disp(traffic_action(7, 0));

function a = traffic_action(color, is_emergency)
    if is_emergency ~= 0
        a = 9;
    elseif color < 1
        a = 0;
    elseif color > 3
        a = 0;
    else
        switch color
            case 1
                a = 1;
            case 2
                a = 2;
            case 3
                a = 3;
            otherwise
                a = 0;
        end
    end
end
