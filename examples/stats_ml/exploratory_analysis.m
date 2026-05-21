% exploratory_analysis.m — Statistics Toolbox: exploratory data analysis.
% ----------------------------------------------------------------------
% Modeled on the User's Guide "Exploratory Analysis of Data" workflow:
% summarise a sample with location/spread/shape statistics and quartiles,
% then measure the linear association between two related variables.
rng(7);

x = normrnd(50, 8, 200, 1);             % a 200-sample measurement

% ----- location, spread, shape ----------------------------------------
fprintf('mean      = %.2f\n', mean(x));
fprintf('median    = %.2f\n', median(x));
fprintf('std       = %.2f\n', std(x));
fprintf('iqr       = %.2f\n', iqr(x));
fprintf('range     = %.2f\n', range(x));
fprintf('skewness  = %.3f\n', skewness(x));
fprintf('kurtosis  = %.3f\n', kurtosis(x));

% ----- quartiles (prctile with a vector of percentiles) ---------------
q = prctile(x, [25 50 75]);
fprintf('quartiles = %.2f / %.2f / %.2f\n', q(1), q(2), q(3));

% ----- linear association between two variables -----------------------
y = 0.6 * x + normrnd(0, 4, 200, 1);    % y depends on x plus noise
M = [x y];                              % (assign the horzcat to a variable)
R = corr(M);
C = cov(M);
fprintf('corr(x,y) = %.4f\n', R(1, 2));
fprintf('cov(x,y)  = %.2f\n', C(1, 2));
