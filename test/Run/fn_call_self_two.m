disp(fib(10));
disp(fib(12));

function y = fib(n)
    if n < 2
        y = n;
    else
        y = fib(n - 1) + fib(n - 2);
    end
end
