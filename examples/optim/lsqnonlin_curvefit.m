% lsqnonlin / lsqcurvefit — nonlinear least squares (Optimization
% Toolbox Tier-2).
%
% lsqnonlin minimises ||r(x)||^2 where r is a residual vector; it
% uses Levenberg-Marquardt with a finite-difference Jacobian and
% damped normal equations.  lsqcurvefit is the curve-fitting wrapper
% over the residual fun(x, xdata) - ydata.

% Fit  y = a*exp(-b*t)  to noise-free samples generated with a = 2,
% b = 0.5 at t = 0, 1, 2, 3.

% --- 1. lsqnonlin — the residual handle returns r(x) directly ----
resfn = @(x) [x(1)*exp(-x(2)*0) - 2; ...
              x(1)*exp(-x(2)*1) - 1.2130613194; ...
              x(1)*exp(-x(2)*2) - 0.7357588823; ...
              x(1)*exp(-x(2)*3) - 0.4462603202];
r = lsqnonlin(resfn, [1; 1]);
fprintf('lsqnonlin fit:  a = %.4f, b = %.4f\n', r(1), r(2));

% --- 2. lsqcurvefit — the model handle is fun(params, xdata) -----
model = @(x, t) x(1)*exp(-x(2)*t);
xdata = [0; 1; 2; 3];
ydata = [2; 1.2130613194; 0.7357588823; 0.4462603202];
p = lsqcurvefit(model, [1; 1], xdata, ydata);
fprintf('lsqcurvefit fit: a = %.4f, b = %.4f\n', p(1), p(2));

% --- 3. residual sum of squares at the lsqnonlin solution --------
res = resfn(r);
ss = res(1)*res(1) + res(2)*res(2) + res(3)*res(3) + res(4)*res(4);
fprintf('residual sum of squares: %.2e\n', ss);
