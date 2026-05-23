% Second-order ODE (auto-classified) + IVP + symbolic identity matrix.
syms x y yp ypp
disp(dsolve(ypp + y, y, yp, ypp, x))     % y'' + y = 0
disp(dsolve(ypp - y, y, yp, ypp, x))     % y'' - y = 0
syms t yt yp2
sol = dsolve(yp2 + yt, yt, yp2, t);      % y' + y = 0
disp(dsolve_ivp(yp2 + yt, yt, yp2, t, sym(0), sym(1)))   % y(0)=1 -> exp(-t)
disp(sym_eye(3))                         % symbolic identity
