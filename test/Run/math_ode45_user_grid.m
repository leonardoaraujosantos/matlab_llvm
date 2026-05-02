% MATLAB-compatible user grid: when tspan has more than two elements,
% the integrator emits y at exactly those times via cubic-Hermite dense
% output. dy/dt = -y → y(t) = y0 * exp(-t).

f = @(t,y) 0 - y;

% 11 evenly-spaced times in [0, 5]. Output must be exactly 11 rows;
% t must equal the user's grid; y must track exp(-t) closely.
target = [0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5];
[t, y] = ode45(f, target, 1);
disp(length(t));
disp(t(1));
disp(t(6));
disp(t(end));

% Mid-grid sample: y(2.5) ≈ exp(-2.5) ≈ 0.0821 within Hermite tolerance.
err_mid = abs(y(6) - 1/exp(2.5));
if err_mid < 0.01; disp(1); else; disp(0); end

% Last sample exactly tf; value tracks analytic.
err_end = abs(y(end) - 1/exp(5));
if err_end < 0.005; disp(1); else; disp(0); end

% ode23 honours the same user-grid path.
[t2, y2] = ode23(f, target, 1);
disp(length(t2));
disp(t2(end));
err2 = abs(y2(6) - 1/exp(2.5));
if err2 < 0.05; disp(1); else; disp(0); end
