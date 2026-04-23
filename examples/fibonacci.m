% Classic recursive Fibonacci: fib(n) = fib(n-1) + fib(n-2).
disp('recursive fib(0..9):');
disp(fib(0));
disp(fib(1));
disp(fib(2));
disp(fib(5));
disp(fib(9));
disp(fib(12));

% Iterative form with a while loop for comparison.
disp('iterative fib(0..9):');
n = 10;
a = 0;
b = 1;
i = 0;
while i < n
    disp(a);
    t = a + b;
    a = b;
    b = t;
    i = i + 1;
end

function y = fib(n)
    if n < 2
        y = n;
    else
        y = fib(n - 1) + fib(n - 2);
    end
end
