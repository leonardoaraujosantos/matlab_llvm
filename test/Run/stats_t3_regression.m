% Statistics Toolbox Tier-3 — regression (fitlm / predict / fitglm / regress).
X = [1 4; 2 1; 3 5; 4 2; 5 6; 6 1; 7 3; 8 5];
y = [1; 7; 6; 12; 11; 19; 20; 21];   % y = 2 + 3*x1 - 1*x2
mdl = fitlm(X, y);
fprintf('R2     %.4f\n', mdl.Rsquared);
fprintf('b0     %.4f\n', mdl.Beta(1));
fprintf('b1     %.4f\n', mdl.Beta(2));
fprintf('b2     %.4f\n', mdl.Beta(3));
fprintf('se1    %.4f\n', mdl.Coefficients(2, 2));
yp = predict(mdl, [9 5]);
fprintf('pred   %.4f\n', yp(1));
% regress with explicit intercept column
Xi = [1 1 4; 1 2 1; 1 3 5; 1 4 2; 1 5 6; 1 6 1; 1 7 3; 1 8 5];
b = regress(y, Xi);
fprintf('reg_b1 %.4f\n', b(2));
% ridge
br = ridge(y, X, 0.5);
fprintf('ridge1 %.4f\n', br(1));
% logistic GLM
Xl = [1; 2; 3; 4; 5; 6; 7; 8];
yl = [0; 0; 0; 0; 1; 1; 1; 1];
g = fitglm(Xl, yl);
fprintf('logit  %.4f\n', max(predict(g, 7)));
