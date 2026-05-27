% Econometrics Toolbox Tier-6 — bayeslm (Bayesian linear regression) + dtmc.

% --- Bayesian linear regression: recover known coefficients ------------
N = 300;
s = 555;
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
    e(t) = 0.1 * sqrt(-2*log(a)) * cos(2*pi*b);
end
% true model: y = 2 + 3*x1 - 1.5*x2 + noise
y = 2 + 3*x1 - 1.5*x2 + e;
X = [ones(N,1) x1 x2];

Mdl = bayeslm(3);
Post = estimate(Mdl, X, y);
fprintf('beta0 = %.2f\n', Post.Beta(1));    % ~2.00
fprintf('beta1 = %.2f\n', Post.Beta(2));    % ~3.00
fprintf('beta2 = %.2f\n', Post.Beta(3));    % ~-1.50

% Posterior-mean forecast at two new design points.
Xn = [1 0.5 0.5; 1 1.0 0.0];
yf = forecast(Post, Xn);
fprintf('nf = %.0f\n', numel(yf));           % 2
fprintf('yf1 = %.2f\n', yf(1));              % 2 + 1.5 - 0.75 = 2.75

% --- dtmc Markov chain: stationary distribution + simulation -----------
P = [0.7 0.3; 0.4 0.6];
mc = dtmc(P);
pis = asymptotics(mc);
fprintf('pi1 = %.4f\n', pis(1));             % 4/7 = 0.5714
fprintf('pi2 = %.4f\n', pis(2));             % 3/7 = 0.4286

seq = simulate(mc, 100);
fprintf('nseq = %.0f\n', numel(seq));        % 101
fprintf('s0 = %.0f\n', seq(1));              % 1 (starts in state 1)
