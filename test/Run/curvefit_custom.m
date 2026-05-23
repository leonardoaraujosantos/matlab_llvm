% Curve Fitting Toolbox Tier-3 — custom equation fittype + confint + formula.
% A user equation string is parsed (coefficients = non-x identifiers, sorted),
% fitted by a multistart finite-difference Levenberg-Marquardt, and queried.
x = (0:0.25:5)';
t = -1.2 * x;
y = 4.0 * exp(t) + 1.5;          % a*exp(b*x)+c, truth a=4 b=-1.2 c=1.5
ft = fittype('a*exp(b*x) + c');
[f, gof] = fit(x, y, ft);
cc = coeffvalues(f);
fprintf('a=%.3f b=%.3f c=%.3f\n', cc(1), cc(2), cc(3));
fprintf('r2=%.5f ncoef=%.0f\n', gof.rsquare, numcoeffs(f));

% confint on an imperfect linear fit gives a non-degenerate interval.
xl = (1:10)';
yl = [2.1; 4.2; 5.8; 8.3; 9.9; 12.1; 14.2; 15.8; 18.3; 19.9];
g = fit(xl, yl, 'poly1');
ci = confint(g);
fprintf('p1 in [%.4f, %.4f]\n', ci(1,1), ci(2,1));
fprintf('p2 in [%.4f, %.4f]\n', ci(1,2), ci(2,2));
