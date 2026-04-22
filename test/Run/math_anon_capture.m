k = 5;
f = @(x) x + k;
disp(f(3));
disp(f(10));

a = 2;
b = 3;
g = @(x) a * x + b;
disp(g(5));
disp(g(10));

% capture-by-value-at-@-time: reassigning k must not alter f.
k = 100;
disp(f(0));
