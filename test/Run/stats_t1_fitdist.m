% Statistics Toolbox Tier-1 — distribution objects (makedist / fitdist).
pd = makedist('Normal', 'mu', 5, 'sigma', 2);
fprintf('mk_mu    %.4f\n', pd.mu);
fprintf('mk_sigma %.4f\n', pd.sigma);
fprintf('pd_pdf   %.4f\n', pdf(pd, 5));
fprintf('pd_cdf   %.4f\n', cdf(pd, 7));
fprintf('pd_icdf  %.4f\n', icdf(pd, 0.975));
x = [4.1 5.2 4.8 5.5 4.9 5.1 4.7 5.3 5.0 4.6];
fn = fitdist(x', 'Normal');
fprintf('fit_mu   %.4f\n', fn.mu);
fprintf('fit_sig  %.4f\n', fn.sigma);
fprintf('fit_cdf5 %.4f\n', cdf(fn, 5));
