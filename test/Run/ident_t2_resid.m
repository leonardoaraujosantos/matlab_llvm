% System Identification Tier-2 — pe (prediction errors) + resid
% (whiteness diagnostic).  An ARMAX model fit to data generated from
% that exact structure leaves near-white residuals: pe variance ≈ the
% innovation variance, and resid's [maxAuto; maxCross] stats are small.
N = 800;
e = zeros(N, 1); u = zeros(N, 1); sd = 99173;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648); e(k) = (sd/2147483648 - 0.5) * 0.2;
    sd = mod(sd * 1103515245 + 12345, 2147483648); u(k) = sign(sd/2147483648 - 0.5);
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.5*y(k-1) + 1.0*u(k-1) + e(k) + 0.3*e(k-1);
end
z = iddata(y, u, 1);
m = armax(z, [1 1 1 1]);

pev = pe(m, z);
fprintf('pe length = %.0f\n', size(pev, 1));   % 800

rr = resid(m, z);
fprintf('maxAuto  = %.2f\n', rr(1));   % small (white residuals)
fprintf('maxCross = %.2f\n', rr(2));   % small (uncorrelated w/ input)
