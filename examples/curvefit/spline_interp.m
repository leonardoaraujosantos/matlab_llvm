% spline_interp.m — Curve Fitting Toolbox Tier-6 headline.
% ----------------------------------------------------------------------
% The UG "Cubic Spline Interpolation" workflow on the ppform spline layer:
% build a not-a-knot cubic spline, evaluate it on a fine grid with fnval,
% and plot the derivative spline (fnplt(fnder(pp))).
x = (0:10)';
y = zeros(11, 1);
for k = 1:11
    y(k) = sin(0.6 * (k - 1));               % scalar-loop data-gen
end

pp = spline(x, y);                            % not-a-knot cubic ppform
fprintf('ppform: order=%.0f pieces=%.0f\n', fnbrk(pp, 'order'), fnbrk(pp, 'pieces'));

% evaluate the spline + its derivative on a fine grid
xf = (0:0.1:10)';
yf = fnval(pp, xf);
dpp = fnder(pp);                              % derivative spline
df = fnval(dpp, xf);

% definite integral of the spline over [0, 10]
ipp = fnint(pp);
fprintf('integral over [0,10] = %.4f\n', fnval(ipp, 10));

figure;
plot(x, y, 'o', xf, yf, '-', xf, df, '--'); grid on;
legend('data', 'spline', 'derivative');
xlabel('x'); ylabel('value');
title('cubic spline interpolation + derivative');
saveas(gcf, '/tmp/spline_interp.png');
