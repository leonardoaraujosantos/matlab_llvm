% exp_decay_fit.m — Curve Fitting Toolbox Tier-2 headline.
% ----------------------------------------------------------------------
% The UG "Fit Exponential Models" workflow: a signal that is the sum of two
% exponential decays is fitted with the two-term model 'exp2', recovering
% both amplitudes and rate constants.  No StartPoint is supplied — the fit
% seeds itself from a log-linear regression, then refines with the
% hand-coded Levenberg-Marquardt (analytic Jacobian).  No external
% dependency.
t = (0:0.2:8)';
y = 5.0 * exp(-1.5 * t) + 2.0 * exp(-0.3 * t);

[f, gof] = fit(t, y, 'exp2');
disp(f);
fprintf('R-squared = %.6f\n', gof.rsquare);
c = coeffvalues(f);
fprintf('term 1: amp=%.3f rate=%.3f\n', c(1), c(2));
fprintf('term 2: amp=%.3f rate=%.3f\n', c(3), c(4));

% overlay the fitted curve on the data
yq = feval(f, t);
figure;
plot(t, y, 'o', t, yq, '-'); grid on;
xlabel('t'); ylabel('signal');
title('exp2 — two-term exponential decay fit');
saveas(gcf, '/tmp/exp_decay_fit.png');
