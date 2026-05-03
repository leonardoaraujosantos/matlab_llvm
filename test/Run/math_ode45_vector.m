% Vector y — system of ODEs. Linear oscillator dy/dt = [-y(2); y(1)] with
% y(0) = [1; 0] integrates to y(t) = [cos(t); sin(t)]. At t = 2π the state
% is back at the initial condition.
%
% This exercises the full vector-y stack: anon-function block-arg retyping
% (Sema defaults y to f64; the LowerAnonCalls pre-pass detects ode45 +
% matrix y0 and promotes y to ptr), the LowerTensorOps vector dispatch
% (matlab_ode45_v_*), and the vector RK45 in the runtime.

y0 = [1; 0];
[t, y] = ode45(@(t,yy) [(0 - yy(2)); yy(1)], [0 6.283185307179586], y0);

disp(t(1));
disp(y(1, 1));
disp(y(1, 2));

% Final state: cos(2π) = 1, sin(2π) = 0 within rtol = 1e-3.
if abs(y(end, 1) - 1) < 0.01; disp(1); else; disp(0); end
if abs(y(end, 2) - 0) < 0.01; disp(1); else; disp(0); end

% At t = π/2: cos(π/2) = 0, sin(π/2) = 1.
target = [0 1.5707963267948966 6.283185307179586];
[t2, y2] = ode45(@(t,yy) [(0 - yy(2)); yy(1)], target, y0);
disp(length(t2));
if abs(y2(2, 1) - 0) < 0.01; disp(1); else; disp(0); end
if abs(y2(2, 2) - 1) < 0.01; disp(1); else; disp(0); end

% Higher-dimensional system: y' = A*y for a damped oscillator.
%   [y1', y2', y3'] depend linearly. Just confirm shape + length match.
y0b = [1; 0; 0];
[tb, yb] = ode45(@(t,yy) [(0 - yy(2)); (yy(1) - 0.1*yy(2)); yy(2)], [0 1], y0b);
disp(length(tb));
% y has D = 3 columns (the runtime stores N rows × D cols).
% Check by reading column 3.
disp(yb(1, 3));
