% Backward-time integration: tspan = [t1 t0] with t1 > t0 means integrate
% from t1 backward to t0. Adaptive RK45 must handle the negative-step
% direction so the loop terminates and the grid endpoint lands exactly
% on t0.
%
% Use dy/dt = -y → y(t) = exp(-t). The reversed problem is stable enough
% for round-trip recovery (forward then back) to within rtol.

f = @(t,y) 0 - y;

% Forward leg: y(0) = 1 → y(1) = e^-1.
[t_fwd, y_fwd] = ode45(f, [0 1], 1);
disp(t_fwd(1));
disp(t_fwd(end));

% Backward leg: integrate from y(1) back to t=0. The grid starts at the
% later time and ends at the earlier one.
[t_back, y_back] = ode45(f, [1 0], y_fwd(end));
disp(t_back(1));
disp(t_back(end));

% Round-trip recovery: y_back(end) should equal y(0) = 1 within rtol.
err = abs(y_back(end) - 1.0);
if err < 0.01; disp(1); else; disp(0); end
