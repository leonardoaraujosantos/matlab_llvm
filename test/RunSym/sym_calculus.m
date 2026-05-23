% Calculus: higher-order + partial diff, definite integral, limit, taylor.
syms x y
f = x^3 + 2*x;
disp(diff(f, x))            % 3*x^2 + 2
disp(diff(f, x, 2))         % 6*x  (2nd derivative)
g = x^2*y + y^3;
disp(diff(g, y))            % x^2 + 3*y^2  (partial wrt y)
disp(int(x^2, x))           % indefinite
disp(int(x, x, 0, 2))       % definite -> 2
disp(limit((1 - cos(x))/x^2, x, 0))   % 1/2
disp(taylor(exp(x), x, 0, 4))
