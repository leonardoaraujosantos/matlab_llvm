% Iterative Fibonacci via a while loop — demonstrates `while` lowering
% and loop-carried scalar state. The recursive form needs
% LowerUserCalls to handle two self-calls in one expression, which
% isn't wired yet (see roadmap).
n = 10;
a = 0;
b = 1;
disp('fib(0..9):');
i = 0;
while i < n
    disp(a);
    t = a + b;
    a = b;
    b = t;
    i = i + 1;
end
