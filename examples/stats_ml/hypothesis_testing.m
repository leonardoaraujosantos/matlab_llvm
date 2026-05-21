% hypothesis_testing.m — Statistics Toolbox: hypothesis testing.
% ----------------------------------------------------------------------
% Modeled on the User's Guide "Hypothesis Testing" workflow: compare a
% control and a treatment group with a parametric two-sample t-test (with
% the full [h, p, ci, stats] decision report), check the equal-variance
% assumption with an F-test, and cross-check with the nonparametric
% Wilcoxon rank-sum test.
rng(3);

control   = normrnd(100, 10, 50, 1);    % baseline
treatment = normrnd(106, 10, 50, 1);    % +6 mean shift

% ----- two-sample t-test (full report) --------------------------------
[h, p, ci, stats] = ttest2(control, treatment);
fprintf('two-sample t-test\n');
fprintf('  reject equal-means H0 : %.0f\n', h);
fprintf('  p-value               : %.4f\n', p);
fprintf('  t-statistic           : %.3f\n', stats.tstat);
fprintf('  degrees of freedom    : %.0f\n', stats.df);
fprintf('  95%% CI on mean diff   : [%.2f, %.2f]\n', ci(1), ci(2));

% ----- equal-variance assumption (F-test) -----------------------------
[hv, pv] = vartest2(control, treatment);
fprintf('equal-variance F-test : h = %.0f, p = %.4f\n', hv, pv);

% ----- nonparametric cross-check (Wilcoxon rank-sum) ------------------
[pr, hr] = ranksum(control, treatment);
fprintf('Wilcoxon rank-sum     : h = %.0f, p = %.5f\n', hr, pr);
