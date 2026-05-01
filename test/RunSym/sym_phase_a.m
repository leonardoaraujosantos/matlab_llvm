syms x
syms a b c
f = a*x^2 + b*x + c;
disp(f)
df = diff(f, x);
disp(df)
F = int(x^2, x);
disp(F)
e = expand((x+1)^3);
disp(e)
fc = factor(x^2 - 1, x);
disp(fc)
r = solve(x^2 - 5*x + 6, x);
disp(r)
v = subs(x + 1, x, 2);
disp(v)
n = double(v);
disp(n)
