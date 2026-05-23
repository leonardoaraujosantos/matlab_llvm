% Curve Fitting Toolbox Tier-6 — ppform spline layer (spline / fnval / fnder
% / fnint / pchip / ppmak / fnbrk).  A cubic spline reproduces a quadratic
% exactly, so the derivative and integral are analytic-checkable.
x = (0:6)';
y = x.^2;
pp = spline(x, y);
fprintf('spline at 2.5 = %.4f\n', fnval(pp, 2.5));   % 6.25
fprintf('spline at 3   = %.4f\n', fnval(pp, 3));      % knot -> 9
dpp = fnder(pp);
fprintf('deriv at 3 = %.4f\n', fnval(dpp, 3));        % 2x = 6
ipp = fnint(pp);
fprintf('int 0..3 = %.4f\n', fnval(ipp, 3));          % x^3/3 = 9
ph = pchip(x, y);
fprintf('pchip at 5 = %.4f\n', fnval(ph, 5));         % knot -> 25
fprintf('order=%.0f pieces=%.0f\n', fnbrk(pp, 'order'), fnbrk(pp, 'pieces'));

% ppmak round-trip: two quadratic pieces over [0 1 2]
pm = ppmak([0 1 2], [1 0 0; 1 0 1]);
fprintf('ppmak at 0.5 = %.4f\n', fnval(pm, 0.5));     % 1*0.25 = 0.25
fprintf('ppmak at 1.5 = %.4f\n', fnval(pm, 1.5));     % 1*0.25 + 1 = 1.25
