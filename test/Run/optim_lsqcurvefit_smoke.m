% lsqcurvefit — Optimization Toolbox Tier-2.  Nonlinear curve fitting
% via Levenberg-Marquardt over the residual fun(x, xdata) - ydata.
% See docs/optim_toolbox_roadmap.md.
%
%   x = lsqcurvefit(@fun, x0, xdata, ydata)
%
% The model handle fun(params, xdata) takes two vector arguments;
% both are retyped to the matrix ABI by the anon pre-pass.

% Model y = a*exp(-b*t); noise-free data generated with a = 2, b = 0.5.
model = @(x, t) x(1)*exp(-x(2)*t);
xdata = [0; 1; 2; 3];
ydata = [2; 1.2130613194; 0.7357588823; 0.4462603202];

% --- 1. Fit recovers the true parameters --------------------------
p = lsqcurvefit(model, [1; 1], xdata, ydata);
e1 = abs(p(1) - 2) + abs(p(2) - 0.5);
if e1 < 1e-4; disp(1); else; disp(0); end

% --- 2. Fitted amplitude is ~2 ------------------------------------
if abs(p(1) - 2) < 1e-4; disp(1); else; disp(0); end

% --- 3. Fitted decay rate is ~0.5 ---------------------------------
if abs(p(2) - 0.5) < 1e-4; disp(1); else; disp(0); end

% --- 4. A second dataset (a = 3, b = 1) ---------------------------
ydata2 = [3; 1.1036383235; 0.4060058497; 0.1493612051];
q = lsqcurvefit(model, [1; 1], xdata, ydata2);
e4 = abs(q(1) - 3) + abs(q(2) - 1);
if e4 < 1e-3; disp(1); else; disp(0); end
