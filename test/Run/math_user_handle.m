f = @sq;
disp(f(3));
disp(f(7));

g = @add1;
disp(g(10));

function y = sq(x)
    y = x * x;
end

function y = add1(x)
    y = x + 1;
end
