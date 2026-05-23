% Curve Fitting Toolbox Tier-3 — differentiate + integrate on a fitted model.
% f(x)=x^2 (poly2): the derivative at x is 2x, and the cumulative integral
% from 0 follows x^3/3 (trapezoid over the query grid).
x = (0:6)';
y = x.^2;
g = fit(x, y, 'poly2');

xe = (0:0.5:4)';
d = differentiate(g, xe);
fprintf('df/dx at x=1 = %.4f\n', d(3));   % xe(3)=1 -> 2
fprintf('df/dx at x=3 = %.4f\n', d(7));   % xe(7)=3 -> 6

v = integrate(g, xe);
fprintf('int 0..2 = %.4f\n', v(5));       % xe(5)=2 -> ~8/3
fprintf('int 0..4 = %.4f\n', v(9));       % xe(9)=4 -> ~64/3
