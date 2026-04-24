% Else arm contains a call — can't hoist into a ternary.
% Keeps the `double y = 0; if (...) { y = …; } else { y = …; }` shape.
disp(rec(5));

function y = rec(n)
    if n <= 1
        y = 1;
    else
        y = n * rec(n - 1);
    end
end
