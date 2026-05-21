% Statistics Toolbox Tier-1 — descriptive statistics + covariance/correlation.
x = [2 4 4 4 5 5 7 9];
fprintf('median   %.4f\n', median(x));
fprintf('prctile  %.4f\n', prctile(x, 50));
fprintf('iqr      %.4f\n', iqr(x));
fprintf('range    %.4f\n', range(x));
fprintf('mode     %.4f\n', mode(x));
fprintf('skewness %.4f\n', skewness(x));
fprintf('kurtosis %.4f\n', kurtosis(x));
fprintf('geomean  %.4f\n', geomean(x));
fprintf('harmmean %.4f\n', harmmean(x));
X = [1 2; 2 4; 3 6; 4 8];
C = corr(X);
V = cov(X);
fprintf('corr12   %.4f\n', C(1, 2));
fprintf('cov11    %.4f\n', V(1, 1));
