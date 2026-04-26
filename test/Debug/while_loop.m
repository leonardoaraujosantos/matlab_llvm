% while-loop fixture: the while-stmt anchors a hook on the `while`
% keyword; the body's statements get their own hooks.
i = 0;
n = 5;

while i < n

    % a comment that splits the body
    i = i + 1;
end

disp(i);
