% glm_logistic.m — Statistics Toolbox: logistic regression (GLM).
% ----------------------------------------------------------------------
% A generalized linear model with a binomial response and logit link,
% fitted by iteratively reweighted least squares (fitglm).  Predict the
% class-1 probability as a function of the predictor — the shallow-ML
% classification entry point (no Deep Learning dependency).
%
% Hand-built, slightly overlapping dataset (avoids perfect separation,
% which would send the IRLS coefficients to infinity).
x = [1; 2; 2; 3; 3; 4; 4; 5; 5; 6; 6; 7; 7; 8];
y = [0; 0; 1; 0; 1; 0; 1; 1; 0; 1; 1; 1; 0; 1];

mdl = fitglm(x, y);

fprintf('logistic regression (binomial / logit link)\n');
fprintf('  P(y=1 | x=2) = %.3f\n', max(predict(mdl, 2)));
fprintf('  P(y=1 | x=4) = %.3f\n', max(predict(mdl, 4)));
fprintf('  P(y=1 | x=6) = %.3f\n', max(predict(mdl, 6)));
fprintf('  P(y=1 | x=8) = %.3f\n', max(predict(mdl, 8)));
