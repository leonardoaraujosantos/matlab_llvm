% lsqnonlin — Optimization Toolbox Tier-2.  Nonlinear least squares
% via Levenberg-Marquardt with a finite-difference Jacobian.  See
% docs/optim_toolbox_roadmap.md.
%
%   x = lsqnonlin(@fun, x0)            % fun returns the residual vector
%   x = lsqnonlin(@fun, x0, lb, ub)    % with bound constraints
%
% Fit y = a*exp(-b*t) to noise-free samples generated with a = 2,
% b = 0.5 at t = 0, 1, 2, 3.  The residual handle returns the vector
% of model-minus-data values; LM should recover [2; 0.5].

resfn = @(x) [x(1)*exp(-x(2)*0) - 2; ...
              x(1)*exp(-x(2)*1) - 1.2130613194; ...
              x(1)*exp(-x(2)*2) - 0.7357588823; ...
              x(1)*exp(-x(2)*3) - 0.4462603202];

% --- 1. Unbounded fit recovers the true parameters ----------------
r = lsqnonlin(resfn, [1; 1]);
e1 = abs(r(1) - 2) + abs(r(2) - 0.5);
if e1 < 1e-4; disp(1); else; disp(0); end

% --- 2. Residual norm at the solution is ~0 -----------------------
res = resfn(r);
ss = res(1)*res(1) + res(2)*res(2) + res(3)*res(3) + res(4)*res(4);
if ss < 1e-8; disp(1); else; disp(0); end

% --- 3. Bounded fit: same problem, generous box -------------------
rb = lsqnonlin(resfn, [1; 1], [0; 0], [10; 10]);
e3 = abs(rb(1) - 2) + abs(rb(2) - 0.5);
if e3 < 1e-4; disp(1); else; disp(0); end

% --- 4. Bound clamps the amplitude --------------------------------
%   Force a <= 1.5: the best feasible amplitude sits on the bound.
rc = lsqnonlin(resfn, [1; 1], [0; 0], [1.5; 10]);
if abs(rc(1) - 1.5) < 1e-3; disp(1); else; disp(0); end
