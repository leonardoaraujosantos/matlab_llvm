% peaks_gauss.m — Curve Fitting Toolbox Tier-2.
% ----------------------------------------------------------------------
% Two-peak Gaussian deconvolution with the 'gauss2' library model:
% recover the amplitude / centre / width of each overlapping peak from the
% summed spectrum.  Start points are seeded automatically (peaks spread
% across the x-range), then refined by the hand-coded Levenberg-Marquardt.
x = (-5:0.2:5)';
t1 = ((x - (-2.0)) / 0.8).^2;          % temps avoid the exp(-(...).^2) trap
t2 = ((x -   2.0)  / 1.0).^2;
y = 3.0 * exp(-t1) + 5.0 * exp(-t2);

[f, gof] = fit(x, y, 'gauss2');
disp(f);
fprintf('R-squared = %.6f\n', gof.rsquare);
c = coeffvalues(f);
fprintf('peak 1: amp=%.3f centre=%.3f width=%.3f\n', c(1), c(2), c(3));
fprintf('peak 2: amp=%.3f centre=%.3f width=%.3f\n', c(4), c(5), c(6));

% overlay the recovered two-peak fit on the data
yq = feval(f, x);
figure;
plot(x, y, 'o', x, yq, '-'); grid on;
xlabel('x'); ylabel('signal');
title('gauss2 — two-peak deconvolution');
saveas(gcf, '/tmp/peaks_gauss.png');
