% Integral transforms + their inverses.
syms t s w
disp(laplace(t, t, s))             % 1/s^2
disp(laplace(sin(w*t), t, s))      % w/(s^2 + w^2)
disp(ilaplace(1/(s^2), s, t))      % t
disp(ilaplace(1/(s + w), s, t))    % exp(-w*t)
syms n z
disp(ztrans(sym(1), n, z))         % z/(z-1)
