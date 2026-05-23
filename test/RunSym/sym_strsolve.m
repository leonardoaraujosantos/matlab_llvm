% str2sym parsing + solve + nsolve/vpasolve numeric roots.
syms x
f = str2sym('x^2 - 3*x + 2');
disp(f)
disp(solve(f, x))                       % roots 1, 2
disp(diff(f, x))                        % 2*x - 3
disp(nsolve(cos(x) - x, x, sym(1)))     % ~0.739085
disp(vpasolve(x^2 - sym(2), x, sym(1), 16))   % ~1.41421356
