% Curve Fitting Toolbox Tier-1 — polynomial fit + goodness-of-fit + feval.
% Exercises: fit(x,y,'polyN') dispatcher, the cfit object, the [f,gof]
% multi-return, the f(xq) call-syntax, free-function feval(g,xq), and the
% single-return form.  Polynomial data is recovered exactly (R^2 = 1).
x = (0:10)';
y = 2*x.^2 - 3*x + 1;

[f, gof] = fit(x, y, 'poly2');
fprintf('poly2 rsquare = %.6f\n', gof.rsquare);
fprintf('poly2 f(2)  = %.4f\n', f(2));
fprintf('poly2 f(5)  = %.4f\n', f(5));
fprintf('poly2 f(11) = %.4f\n', f(11));

yl = 4*x + 7;
[g, gof1] = fit(x, yl, 'poly1');
fprintf('poly1 g(3)  = %.4f\n', feval(g, 3));
fprintf('poly1 rmse  = %.6f\n', gof1.rmse);
