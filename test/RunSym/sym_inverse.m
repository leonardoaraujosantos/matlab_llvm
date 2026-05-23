% Inverse transforms — ilaplace / iztrans round-trips.
syms t s
disp(laplace(cos(t), t, s))         % s/(s^2+1)
disp(ilaplace(s/(s^2 + sym(1)), s, t))   % cos(t)
disp(ilaplace(sym(1)/(s^2 + sym(1)), s, t))   % sin(t)
syms n z
disp(iztrans(z/(z - sym(1)), z, n))      % 1
