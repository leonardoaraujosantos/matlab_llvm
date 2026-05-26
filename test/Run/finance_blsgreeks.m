% Financial Toolbox Tier-2 §2 — Black-Scholes price + Greeks.
% Standard ATM-ish reference: S=100, X=100, r=5%, T=1, sigma=20%.
S = 100; X = 100; r = 0.05; T = 1.0; sigma = 0.20;

% Closed-form Black-Scholes — well-known textbook values.
%   d1 = (ln(S/X) + (r + sigma^2/2)*T) / (sigma*sqrt(T)) = 0.35
%   d2 = d1 - sigma*sqrt(T) = 0.15
%   N(d1) ~ 0.6368, N(d2) ~ 0.5596
%   Price = 100 * 0.6368 - 100 * exp(-0.05) * 0.5596
%         = 63.68 - 95.123 * 0.5596 = 63.68 - 53.232 = 10.45
c = blsprice(S, X, r, T, sigma);
fprintf('blsprice(100,100,5%%,1y,20%%) = %.4f\n', c);

% Delta = N(d1) ~ 0.6368
fprintf('blsdelta = %.4f\n', blsdelta(S, X, r, T, sigma));

% Gamma = N'(d1) / (S * sigma * sqrt(T))
fprintf('blsgamma = %.4f\n', blsgamma(S, X, r, T, sigma));

% Vega = S * N'(d1) * sqrt(T) ~ 37.52
fprintf('blsvega  = %.4f\n', blsvega(S, X, r, T, sigma));

% Rho ~ 53.23
fprintf('blsrho   = %.4f\n', blsrho(S, X, r, T, sigma));

% Theta — negative.
fprintf('blstheta = %.4f\n', blstheta(S, X, r, T, sigma));

% Lambda = delta * S / price
fprintf('blslambda = %.4f\n', blslambda(S, X, r, T, sigma));

% Implied vol: invert from a known price.
% blsprice at sigma=0.2 ~ 10.45.  Recover sigma.
iv = blsimpv(S, X, r, T, c);
fprintf('blsimpv(c) = %.4f\n', iv);

% Off-the-money put-ish: lower volatility recovery on a different price.
iv2 = blsimpv(S, X, r, T, 12.50);
fprintf('blsimpv(c=12.50) = %.4f\n', iv2);
