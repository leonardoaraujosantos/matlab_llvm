% Classic single-recursion example: factorial. True Fibonacci needs
% two self-calls in a single expression (fib(n-1) + fib(n-2)) which
% the current LowerUserCalls pass doesn't handle yet.
disp('fact(1..6):');
disp(fact(1));
disp(fact(2));
disp(fact(3));
disp(fact(4));
disp(fact(5));
disp(fact(6));

function y = fact(n)
    if n <= 1
        y = 1;
    else
        y = n * fact(n - 1);
    end
end
