tspan = [0 10];
y0 = 1;
f = @(t,y) -2*y + sin(t);

% ode45: Dormand-Prince 5(4). Default rtol = 1e-3.
[t, y] = ode45(f, tspan, y0);
disp(t(1));
disp(t(end));
disp(y(1));

% Exact solution: y(t) = (4*sin(t) - 2*cos(t))/10 + 1.2*exp(-2*t).
ya = (4*sin(10) - 2*cos(10)) / 10 + 1.2*exp(-20);
if abs(y(end) - ya) < 0.01; disp(1); else; disp(0); end

% ode23: Bogacki-Shampine 3(2). Same rtol; a bit less accurate.
[t2, y2] = ode23(f, tspan, y0);
disp(t2(1));
disp(t2(end));
disp(y2(1));
if abs(y2(end) - ya) < 0.05; disp(1); else; disp(0); end
