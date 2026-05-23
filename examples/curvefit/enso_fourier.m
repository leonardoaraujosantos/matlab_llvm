% enso_fourier.m — Curve Fitting Toolbox Tier-2/3 tracer-bullet.
% ----------------------------------------------------------------------
% A custom-equation seasonal fit: a 12-month sinusoid is described directly
% as a fittype string, fitted with the multistart finite-difference
% Levenberg-Marquardt, then post-processed with differentiate (the rate of
% change) and feval (the overlay).  Mirrors the UG ENSO custom-Fourier demo.
month = (1:48)';
sst = zeros(48, 1);
for k = 1:48                               % scalar loop: the seasonal model
    th = 2 * pi * k / 12;                  % (built per-month to keep the data-gen
    sst(k) = 1.5 + 2.0 * sin(th) + 0.8 * cos(th);   % in the scalar lane)
end

ft = fittype('a + b*sin(2*pi*x/12) + c*cos(2*pi*x/12)');
[f, gof] = fit(month, sst, ft);
disp(f);
fprintf('R-squared = %.6f\n', gof.rsquare);
cc = coeffvalues(f);
fprintf('a=%.3f b=%.3f c=%.3f\n', cc(1), cc(2), cc(3));

% rate of change (derivative) of the fitted seasonal signal
xe = (1:48)';
d = differentiate(f, xe);
fprintf('rate at month 3 = %.3f\n', d(3));

% overlay the fitted seasonal curve on the observations
yq = feval(f, xe);
figure;
plot(month, sst, 'o', xe, yq, '-'); grid on;
xlabel('month'); ylabel('SST anomaly');
title('ENSO — custom Fourier fit');
saveas(gcf, '/tmp/enso_fourier.png');
