% System Identification Tier-1 — AR time-series estimation (Yule-Walker).
% A deterministic LCG drives a small innovation into an AR(2) process
% y(t) = 0.3 y(t-1) - 0.2 y(t-2) + e(t); ar() recovers the polynomial.
N = 400;
e = zeros(N, 1);
sd = 12345;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648);
    e(k) = (sd / 2147483648 - 0.5) * 0.3;
end
y = zeros(N, 1);
for k = 3:N
    y(k) = 0.3 * y(k-1) - 0.2 * y(k-2) + e(k);
end
z = iddata(y, [], 1);
m = ar(z, 2);
fprintf('A1 = %.3f\n', m.A(1));   % 1.000 (monic)
fprintf('A2 = %.3f\n', m.A(2));   % ~ -0.3
fprintf('A3 = %.3f\n', m.A(3));   % ~  0.2
fprintf('Np = %.0f\n', m.Np);     % 2
