% Curve Fitting Toolbox Tier-1 — [f, gof, output] three-way multi-return.
% Asserts the goodness-of-fit struct (sse / rmse / dfe / rsquare) and the
% output struct (numobs / numparam / exitflag + residuals) are populated.
x = (1:8)';
y = 3*x.^2 + 2*x - 5;
[f, gof, output] = fit(x, y, 'poly2');
fprintf('sse      = %.6f\n', gof.sse);
fprintf('rmse     = %.6f\n', gof.rmse);
fprintf('dfe      = %.0f\n', gof.dfe);
fprintf('rsquare  = %.6f\n', gof.rsquare);
fprintf('numobs   = %.0f\n', output.numobs);
fprintf('numparam = %.0f\n', output.numparam);
fprintf('exitflag = %.0f\n', output.exitflag);
