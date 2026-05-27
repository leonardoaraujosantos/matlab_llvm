% Econometrics Toolbox — Tier-6 headline.
% Bayesian linear regression with a diffuse prior: the posterior mean of
% the coefficients equals the OLS estimate, recovering a known model from
% noisy data, plus a Markov-chain regime model.

N = 250;
s = 9091;
x1 = zeros(N,1); x2 = zeros(N,1); e = zeros(N,1);
for t = 1:N
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    x1(t) = s/2147483648;
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    x2(t) = s/2147483648;
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    a = s/2147483648;
    for k = 1:5, s = mod(1103515245*s + 12345, 2147483648); end
    b = s/2147483648;
    if a < 1e-12, a = 1e-12; end
    e(t) = 0.15 * sqrt(-2*log(a)) * cos(2*pi*b);
end
% true regression: GDP growth ~ 1.0 + 2.5*investment - 0.8*rate
y = 1.0 + 2.5*x1 - 0.8*x2 + e;
X = [ones(N,1) x1 x2];

% --- Estimate the Bayesian linear model --------------------------------
Mdl = bayeslm(3);
Post = estimate(Mdl, X, y);
fprintf('Posterior intercept: %.3f\n', Post.Beta(1));
fprintf('Posterior beta1:     %.3f\n', Post.Beta(2));
fprintf('Posterior beta2:     %.3f\n', Post.Beta(3));
fprintf('Posterior Sigma2:    %.4f\n', Post.Sigma2);

% --- Forecast at a scenario design point -------------------------------
Xn = [1 0.8 0.3];
yf = forecast(Post, Xn);
fprintf('Scenario forecast:   %.3f\n', yf(1));

% --- A two-regime Markov chain (e.g. expansion / recession) ------------
%     P(stay expansion)=0.9, P(stay recession)=0.75
P = [0.90 0.10; 0.25 0.75];
mc = dtmc(P);
pis = asymptotics(mc);
fprintf('Long-run P(expansion): %.3f\n', pis(1));
fprintf('Long-run P(recession): %.3f\n', pis(2));

fprintf('Bayesian regression + regime analysis complete.\n');
