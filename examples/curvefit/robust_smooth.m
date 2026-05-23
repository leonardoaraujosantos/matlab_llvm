% robust_smooth.m — Curve Fitting Toolbox Tier-4.
% ----------------------------------------------------------------------
% Noisy data with a couple of gross outliers: compare a plain moving
% average against robust local regression (rloess), then a spline
% interpolant fit.  Robust local regression rejects the spikes that drag
% the moving average off the trend.
x = (1:30)';
y = zeros(30, 1);
for k = 1:30
    y(k) = 0.5 * k + 2.0 * sin(0.4 * k);     % smooth trend (scalar loop)
end
y(10) = y(10) + 12;                          % outliers
y(20) = y(20) - 12;

mv = smooth(y, 7);                            % moving average (span 7)
rl = smooth(y, 7, 'rloess');                  % robust quadratic local regression
fprintf('at the outlier k=10:  data=%.2f  moving=%.2f  rloess=%.2f\n', ...
        y(10), mv(10), rl(10));
fprintf('at a clean point k=15: moving=%.2f  rloess=%.2f\n', mv(15), rl(15));

% an interpolant cfit through the (cleaned) trend
f = fit(x, rl, 'splineinterp');
fprintf('spline interpolant at 15.5 = %.3f\n', f(15.5));

figure;
plot(x, y, 'o', x, mv, '-', x, rl, '-'); grid on;
xlabel('x'); ylabel('y'); title('moving average vs robust loess');
saveas(gcf, '/tmp/robust_smooth.png');
