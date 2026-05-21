% distribution_fitting.m — Statistics Toolbox: curve & distribution fitting.
% ----------------------------------------------------------------------
% Modeled on the User's Guide "Curve Fitting and Distribution Fitting"
% and "Maximum Likelihood Estimation" workflows: fit probability
% distributions to data by maximum likelihood (fitdist) and query them,
% then fit a deterministic curve with polyfit/polyval.
rng(11);

% ----- distribution fitting (maximum likelihood) ----------------------
lifetime = exprnd(1000, 300, 1);        % component lifetimes, true mean 1000
pe = fitdist(lifetime, 'Exponential');
pn = fitdist(lifetime, 'Normal');
fprintf('exponential MLE: mu     = %.1f  (true 1000)\n', pe.mu);
fprintf('normal MLE     : mu=%.1f sigma=%.1f\n', pn.mu, pn.sigma);
fprintf('P(life <= 500) = %.4f\n', cdf(pe, 500));
fprintf('median life    = %.1f\n', icdf(pe, 0.5));

% ----- curve fitting (least-squares polynomial) -----------------------
xx = [0 1 2 3 4 5];
yy = [1.0 2.1 4.9 10.2 16.8 25.1];      % approximately x^2
c  = polyfit(xx, yy, 2);
fprintf('quadratic fit  : %.3f*x^2 + %.3f*x + %.3f\n', c(1), c(2), c(3));
fprintf('fit at x = 6   : %.2f\n', polyval(c, 6));
