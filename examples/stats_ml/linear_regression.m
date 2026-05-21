% linear_regression.m — Statistics Toolbox: linear regression + assessment.
% ----------------------------------------------------------------------
% Modeled on the regression workflow / "assess … regression performance":
% fit a multiple linear model with fitlm, read the goodness-of-fit
% measures (R^2, adjusted R^2, RMSE) and the coefficient table, then
% predict on new data.
rng(5);
n  = 100;
x1 = unifrnd(0, 10, n, 1);
x2 = unifrnd(0, 5,  n, 1);
y  = 1.5 + 2.0 * x1 - 0.8 * x2 + normrnd(0, 1, n, 1);   % true model

X   = [x1 x2];                          % (assign the horzcat to a variable)
mdl = fitlm(X, y);

% ----- goodness of fit ------------------------------------------------
fprintf('R-squared       = %.4f\n', mdl.Rsquared);
fprintf('Adjusted R^2    = %.4f\n', mdl.RsquaredAdj);
fprintf('RMSE            = %.4f\n', mdl.RMSE);

% ----- estimated coefficients (Estimate / SE / tStat / pValue) --------
fprintf('Intercept       = %.3f  (true 1.5)\n', mdl.Beta(1));
fprintf('Coef x1         = %.3f  (true 2.0)\n', mdl.Beta(2));
fprintf('Coef x2         = %.3f  (true -0.8)\n', mdl.Beta(3));
fprintf('SE(x1)          = %.4f\n', mdl.Coefficients(2, 2));
fprintf('p-value(x1)     = %.6f\n', mdl.Coefficients(2, 4));

% ----- prediction on new data -----------------------------------------
yhat = predict(mdl, [5 2]);
fprintf('predict([5, 2]) = %.3f  (expected ~%.2f)\n', yhat(1), 1.5 + 2.0*5 - 0.8*2);
