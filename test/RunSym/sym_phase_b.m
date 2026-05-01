syms x
disp(taylor(sin(x), x, 0, 5))
disp(limit(sin(x)/x, x, 0))
disp(vpa(sym(pi), 32))
syms y yp
disp(dsolve(yp + y, y, yp, x))
syms t s a
disp(laplace(exp(-a*t), t, s))
syms w
disp(fourier(exp(-t*t), t, w))
