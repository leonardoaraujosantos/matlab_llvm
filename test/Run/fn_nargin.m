disp(lone());
disp(add2(5, 7));

function y = lone()
    y = nargin + 100;
end

function y = add2(a, b)
    if nargin == 2
        y = a + b;
    else
        y = a;
    end
end
