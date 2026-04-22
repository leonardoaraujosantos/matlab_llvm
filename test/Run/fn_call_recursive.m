disp(fact(5));

function y = fact(n)
    if n <= 1
        y = 1;
    else
        y = n * fact(n - 1);
    end
end
