% Curve Fitting Toolbox Tier-2 — nonlinear library models.
% exp1 / power1 / gauss1 fitted by the hand-coded Levenberg-Marquardt with
% analytic Jacobians + auto start points (log-linear / log-log / peak seeds).
% Each recovers the synthetic ground-truth parameters to high precision.
x = (0:0.5:5)';
ye = 3.0 * exp(-0.8 * x);
[fe, ge] = fit(x, ye, 'exp1');
ce = coeffvalues(fe);
fprintf('exp1   a=%.3f b=%.3f r2=%.5f\n', ce(1), ce(2), ge.rsquare);

xp = (1:0.5:6)';
yp = 2.5 * xp.^1.7;
[fp, gp] = fit(xp, yp, 'power1');
cp = coeffvalues(fp);
fprintf('power1 a=%.3f b=%.3f r2=%.5f\n', cp(1), cp(2), gp.rsquare);

xg = (-4:0.4:4)';
tg = ((xg - 0.5)/1.2).^2;            % temp avoids the exp(-(...).^2) precedence trap
yg = 5.0 * exp(-tg);
[fg, gg] = fit(xg, yg, 'gauss1');
cg = coeffvalues(fg);
fprintf('gauss1 a=%.3f b=%.3f c=%.3f\n', cg(1), cg(2), cg(3));
fprintf('gauss1 r2=%.5f\n', gg.rsquare);
