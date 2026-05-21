% fit_normal.m — Statistics and Machine Learning Toolbox, Tier-1 headline.
% ----------------------------------------------------------------------
% "Fit a distribution" — the universal first-hour-with-data workflow.
% Draw a sample from a known Normal, summarise it with descriptive
% statistics, recover its parameters by maximum likelihood with `fitdist`,
% then query the fitted distribution object (cdf / icdf / pdf).
%
% Every piece runs over the shipped, rng-reproducible PRNG and the
% hand-coded distribution cores (normal CDF via libc erf, inverse normal
% via Acklam's rational approximation) — no external dependency.
rng(42);

% A 500-sample "IQ-like" dataset: true mean 100, true sigma 15.
data = normrnd(100, 15, 500, 1);

% ----- Descriptive summary --------------------------------------------
fprintf('mean      = %.2f\n', mean(data));
fprintf('median    = %.2f\n', median(data));
fprintf('std       = %.2f\n', std(data));
fprintf('iqr       = %.2f\n', iqr(data));
fprintf('skewness  = %.3f\n', skewness(data));

% ----- Fit a Normal distribution (maximum likelihood) -----------------
pd = fitdist(data, 'Normal');
fprintf('\nfitdist(Normal): mu = %.2f, sigma = %.2f   (true 100, 15)\n', ...
        pd.mu, pd.sigma);

% ----- Query the fitted model -----------------------------------------
fprintf('P(X <= 115)    = %.4f\n', cdf(pd, 115));
fprintf('95th percentile= %.2f\n', icdf(pd, 0.95));
fprintf('pdf at the mean= %.4f\n', pdf(pd, pd.mu));
